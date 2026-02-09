#!/usr/bin/env python3
"""
AWS Batch Entrypoint - Reduce Phase Theme Aggregation

This script runs inside the Batch container and aggregates firm-level topics
from the map phase into cross-firm themes. It:
1. Loads all firm topic Parquet files from S3 for the specified quarter
2. Runs ThemeAggregator (BERTopic re-clustering on topic summaries)
3. Generates LLM descriptions for themes if LLM is configured
4. Writes two output files per ADR-007:
   - themes.parquet: Theme metadata
   - theme_contributions.parquet: Theme -> firm/topic mappings

Environment Variables (set by Batch job definition + submission):
    QUARTER: Quarter being processed (e.g., "2023Q1")
    S3_BUCKET: Input/output bucket name
    LLM_BASE_URL: Optional vLLM endpoint for theme descriptions
    LLM_MODEL_NAME: Model name for vLLM (default: Qwen/Qwen3-8B)
    EMBEDDING_MODEL_NAME: SentenceTransformer model (default: all-mpnet-base-v2)
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import ClientError
import pandas as pd
from sentence_transformers import SentenceTransformer

# Configure logging before imports that may log
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Import project modules (PYTHONPATH=/app set in Dockerfile)
from cloud.src.theme_aggregator import ThemeAggregator
from cloud.src.topic_models.bertopic_model import BERTopicModel
from cloud.src.llm.xai_client import XAIClient, LLMUnavailableError


def get_env_required(name: str) -> str:
    """Get required environment variable or fail fast."""
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Required environment variable {name} not set")
    return value


def get_env_optional(name: str, default: str) -> str:
    """Get optional environment variable with default."""
    return os.environ.get(name, default)


def load_firm_topics_from_s3(
    s3_client,
    bucket: str,
    quarter: str,
) -> tuple[List[Dict], Dict[str, Dict]]:
    """
    Load all firm topic Parquet files from the map phase output.

    CRITICAL: Must preserve 'summary' field from map phase output.
    ThemeAggregator uses summary (if present) for clustering quality.

    Args:
        s3_client: Boto3 S3 client
        bucket: S3 bucket name
        quarter: Quarter string (e.g., "2023Q1")

    Returns:
        Tuple of:
        - firm_results: List of FirmTopicOutput-like dicts for ThemeAggregator
        - firm_metadata: Dict of firm_id -> metadata for contributions output
    """
    prefix = f"intermediate/firm-topics/quarter={quarter}/"
    logger.info(f"Loading firm topics from s3://{bucket}/{prefix}")

    # List all Parquet files
    paginator = s3_client.get_paginator("list_objects_v2")
    parquet_keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".parquet"):
                parquet_keys.append(obj["Key"])

    logger.info(f"Found {len(parquet_keys)} Parquet files")

    if not parquet_keys:
        raise ValueError(f"No Parquet files found at {prefix}")

    # Download and concatenate all files
    all_rows = []
    for key in parquet_keys:
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            s3_client.download_file(bucket, key, tmp.name)
            df = pd.read_parquet(tmp.name)
            all_rows.append(df)
            os.remove(tmp.name)

    combined_df = pd.concat(all_rows, ignore_index=True)
    logger.info(f"Loaded {len(combined_df)} topic rows from {len(parquet_keys)} files")

    # Group by firm_id to create FirmTopicOutput-like structures
    firm_results = []
    firm_metadata = {}

    for firm_id, group in combined_df.groupby("firm_id"):
        # Extract firm-level metadata from first row
        first_row = group.iloc[0]
        firm_metadata[firm_id] = {
            "firm_name": first_row.get("firm_name", ""),
            "permno": first_row.get("permno"),
            "gvkey": first_row.get("gvkey"),
            "earnings_call_date": first_row.get("earnings_call_date"),
        }

        # Build topics list
        topics = []
        for _, row in group.iterrows():
            topic = {
                "topic_id": row["topic_id"],
                "representation": row.get("representation", ""),
                "summary": row.get("summary", ""),  # CRITICAL: preserve summary
                "keywords": row.get("keywords", []),
                "size": row.get("n_sentences", 0),
                "sentence_ids": row.get("sentence_ids", []),
            }
            topics.append(topic)

        firm_result = {
            "firm_id": firm_id,
            "firm_name": first_row.get("firm_name", ""),
            "n_topics": len(topics),
            "topics": topics,
        }
        firm_results.append(firm_result)

    logger.info(f"Grouped into {len(firm_results)} firms")
    return firm_results, firm_metadata


async def generate_theme_descriptions(
    llm_client: XAIClient,
    themes: List[Dict],
) -> None:
    """
    Generate LLM descriptions for themes (modifies themes in place).
    Raises LLMUnavailableError on failure - no fallback allowed.

    Args:
        llm_client: Shared XAIClient instance (created once per job)
        themes: List of theme dicts from ThemeAggregator

    Raises:
        LLMUnavailableError: If LLM is unavailable after retries
    """
    logger.info(f"Generating LLM descriptions for {len(themes)} themes")

    for i, theme in enumerate(themes):
        # Collect topic summaries for this theme
        topic_summaries = [
            t.get("summary", t.get("representation", ""))
            for t in theme.get("topics", [])
        ]

        # Filter empty summaries
        topic_summaries = [s for s in topic_summaries if s]

        if not topic_summaries:
            raise LLMUnavailableError(
                f"No topic summaries for theme {theme.get('theme_id')}", 0
            )

        # Get theme keywords for context
        theme_keywords = ", ".join(theme.get("keywords", [])[:10])

        # LLMUnavailableError propagates - no fallback
        description = await llm_client.generate_theme_description(
            theme_keywords=theme_keywords,
            topic_summaries=topic_summaries,
            log_prompt=(i == 0),  # Log first prompt for debugging
        )

        if not description:
            raise LLMUnavailableError(
                f"Empty description for theme {theme.get('theme_id')}", 0
            )

        theme["description"] = description


def flatten_themes_to_parquet(
    themes: List[Dict],
    quarter: str,
) -> pd.DataFrame:
    """
    Flatten themes to themes.parquet schema per ADR-007.

    Schema:
    - theme_id: STRING
    - name: STRING (BERTopic representation)
    - description: STRING (LLM-generated)
    - keywords: LIST<STRING>
    - n_topics: INT32
    - n_firms: INT32
    - quarter: STRING
    """
    rows = []
    for theme in themes:
        rows.append({
            "theme_id": theme["theme_id"],
            "name": theme.get("name", ""),
            "description": theme.get("description", ""),
            "keywords": theme.get("keywords", []),
            "n_topics": theme.get("n_topics", 0),
            "n_firms": theme.get("n_firms", 0),
            "quarter": quarter,
        })

    return pd.DataFrame(rows)


def flatten_contributions_to_parquet(
    themes: List[Dict],
    firm_metadata: Dict[str, Dict],
    quarter: str,
) -> pd.DataFrame:
    """
    Flatten theme contributions to theme_contributions.parquet schema per ADR-007.

    Schema:
    - theme_id: STRING (FK to themes)
    - firm_id: STRING (FK to firms)
    - firm_name: STRING (denormalized)
    - permno: INT64 (for event study)
    - gvkey: STRING (for Compustat joins)
    - earnings_call_date: DATE (event date)
    - topic_id: STRING (which topic contributed)
    - n_sentences: INT32 (sentences in contribution)
    - quarter: STRING (partition key)
    """
    rows = []
    for theme in themes:
        theme_id = theme["theme_id"]

        for topic in theme.get("topics", []):
            firm_id = topic["firm_id"]
            meta = firm_metadata.get(firm_id, {})

            rows.append({
                "theme_id": theme_id,
                "firm_id": firm_id,
                "firm_name": meta.get("firm_name", ""),
                "permno": meta.get("permno"),
                "gvkey": meta.get("gvkey"),
                "earnings_call_date": meta.get("earnings_call_date"),
                "topic_id": topic["topic_id"],
                "n_sentences": topic.get("size", 0),
                "quarter": quarter,
            })

    df = pd.DataFrame(rows)

    # Convert date to proper type if present
    if "earnings_call_date" in df.columns and len(df) > 0:
        df["earnings_call_date"] = pd.to_datetime(df["earnings_call_date"]).dt.date

    return df


def write_parquet_to_s3(
    s3_client,
    df: pd.DataFrame,
    bucket: str,
    s3_key: str,
) -> None:
    """Write DataFrame to S3 as Parquet via temp file."""
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        df.to_parquet(tmp.name, index=False, engine="pyarrow")
        s3_client.upload_file(tmp.name, bucket, s3_key)
        os.remove(tmp.name)

    logger.info(f"Wrote {len(df)} rows to s3://{bucket}/{s3_key}")


def main():
    """Main entrypoint for reduce phase Batch job."""
    logger.info("=" * 60)
    logger.info("Financial Topic Modeling - Reduce Phase (Theme Aggregation)")
    logger.info("=" * 60)

    # Parse environment variables
    quarter = get_env_required("QUARTER")
    bucket = get_env_required("S3_BUCKET")
    embedding_model_name = get_env_optional("EMBEDDING_MODEL_NAME", "all-mpnet-base-v2")
    llm_base_url = os.environ.get("LLM_BASE_URL")
    llm_model_name = get_env_optional("LLM_MODEL_NAME", "Qwen/Qwen3-8B")

    logger.info(f"Quarter: {quarter}")
    logger.info(f"S3 Bucket: {bucket}")
    logger.info(f"Embedding model: {embedding_model_name}")
    logger.info(f"LLM configured: {bool(llm_base_url)}")

    # Initialize S3 client
    s3_client = boto3.client("s3")

    # Load firm topics from map phase output
    firm_results, firm_metadata = load_firm_topics_from_s3(s3_client, bucket, quarter)

    if not firm_results:
        logger.error("No firm results found, cannot aggregate themes")
        return 1

    # Check minimum firms requirement
    if len(firm_results) < 2:
        logger.error(f"Only {len(firm_results)} firms found, need at least 2 for theme aggregation")
        return 1

    # Initialize embedding model
    logger.info(f"Loading embedding model: {embedding_model_name}...")
    embedding_model = SentenceTransformer(embedding_model_name, device="cuda")
    logger.info("Embedding model loaded on GPU")

    # Initialize topic model and aggregator
    logger.info("Initializing BERTopicModel and ThemeAggregator...")
    config = {
        "device": "cuda",
        "embedding_model": embedding_model_name,
        "validation": {
            "min_firms": 2,           # Theme must span >= 2 firms
            "max_firm_dominance": 0.4  # No single firm > 40% of topics
        }
    }
    topic_model = BERTopicModel(config, embedding_model=embedding_model)
    aggregator = ThemeAggregator(topic_model, config)
    logger.info("ThemeAggregator initialized")

    # Aggregate firm topics into themes
    logger.info("Running theme aggregation...")
    themes = aggregator.aggregate(firm_results)
    logger.info(f"Discovered {len(themes)} themes")

    # Override theme_id to match ADR-007 format: theme_{quarter}_{seq}
    # ThemeAggregator uses theme_{YYYYMMDD}_{seq} but we need quarter-based IDs
    for i, theme in enumerate(themes):
        theme["theme_id"] = f"theme_{quarter}_{i:03d}"

    if not themes:
        logger.warning("No themes discovered (all filtered by validation)")
        # Still write empty files for downstream compatibility
        themes_df = pd.DataFrame(columns=[
            "theme_id", "name", "description", "keywords", "n_topics", "n_firms", "quarter"
        ])
        contributions_df = pd.DataFrame(columns=[
            "theme_id", "firm_id", "firm_name", "permno", "gvkey",
            "earnings_call_date", "topic_id", "n_sentences", "quarter"
        ])
    else:
        # LLM is REQUIRED for theme descriptions - no fallback allowed
        if not llm_base_url:
            logger.error("LLM_BASE_URL required but not set")
            return 1

        llm_max_concurrent = int(os.environ.get("LLM_MAX_CONCURRENT", "10"))
        logger.info(f"LLM configured: model={llm_model_name}, concurrency={llm_max_concurrent}")

        # CRITICAL: Create loop FIRST, set as current, THEN create client
        # This ensures the Semaphore binds to the correct event loop
        # DO NOT use asyncio.run() - it creates its own loop causing binding issues
        llm_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(llm_loop)

        try:
            # Create client AFTER setting the loop
            llm_client = XAIClient(
                api_key="dummy",  # vLLM doesn't require auth
                config={
                    "model": llm_model_name,
                    "max_concurrent": llm_max_concurrent,
                    "timeout": 60,
                    "max_health_wait": 600,  # 10 min max wait for vLLM recovery
                    "health_poll_interval": 30,  # Poll every 30 seconds
                },
            )

            # Run on the SAME loop - LLMUnavailableError propagates
            llm_loop.run_until_complete(generate_theme_descriptions(llm_client, themes))
        finally:
            llm_loop.close()
            logger.info("Closed LLM event loop")

        # Flatten to DataFrames
        themes_df = flatten_themes_to_parquet(themes, quarter)
        contributions_df = flatten_contributions_to_parquet(themes, firm_metadata, quarter)

    # Write outputs per ADR-007
    themes_key = f"processed/themes/quarter={quarter}/themes.parquet"
    contributions_key = f"processed/themes/quarter={quarter}/theme_contributions.parquet"

    write_parquet_to_s3(s3_client, themes_df, bucket, themes_key)
    write_parquet_to_s3(s3_client, contributions_df, bucket, contributions_key)

    logger.info("=" * 60)
    logger.info(f"Reduce phase complete: {len(themes)} themes")
    logger.info(f"  - themes.parquet: {len(themes_df)} rows")
    logger.info(f"  - theme_contributions.parquet: {len(contributions_df)} rows")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
