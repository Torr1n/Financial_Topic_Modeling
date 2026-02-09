#!/usr/bin/env python3
"""
AWS Batch Entrypoint - Map Phase Firm Processing

This script runs inside the Batch container and processes a batch of firms
assigned via the manifest. It:
1. Loads the manifest from S3 to find assigned firm IDs
2. Resumes from checkpoint if interrupted (Spot instance handling)
3. Fetches transcripts via WRDSConnector (auth via env vars or .pgpass)
4. Processes each firm with BERTopicModel + FirmProcessor
5. Writes Parquet chunks to S3 with checkpoint updates
6. Tracks failures and exits non-zero if any non-skippable errors occur

Environment Variables (set by Batch job definition + submission):
    MANIFEST_S3_KEY: S3 key for the manifest file (JSONL)
    BATCH_ID: This batch's identifier within the manifest
    QUARTER: Quarter being processed (e.g., "2023Q1")
    S3_BUCKET: Output bucket name
    DATA_SOURCE: Data source type ("s3" or "wrds", default: "wrds" for backward compat)
                 - "s3": Read from prefetch location (production, no MFA)
                 - "wrds": Direct WRDS connection (local dev, requires MFA)
    CHECKPOINT_INTERVAL: Firms per checkpoint (default: 50)
    EMBEDDING_MODEL_NAME: SentenceTransformer model (default: all-mpnet-base-v2)
    ALLOW_FAILURES: If "true", tolerates per-firm errors (default: true for production)
    MAX_CONSECUTIVE_FAILURES: Circuit breaker - fail after N consecutive errors (default: 5)
    MAX_FAILURE_RATE: Circuit breaker - fail if rate exceeds this (default: 0.05 = 5%)
    MIN_PROCESSED_FOR_RATE: Min firms before rate check applies (default: 100)
    WRDS_USERNAME: WRDS credentials (optional - WRDSConnector handles fallback)
    WRDS_PASSWORD: WRDS credentials (optional - WRDSConnector handles fallback)
"""

import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

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
import asyncio

from cloud.src.firm_processor import FirmProcessor
from cloud.src.interfaces import DataConnector
from cloud.src.llm.xai_client import XAIClient, LLMUnavailableError
from cloud.src.topic_models.bertopic_model import BERTopicModel


@dataclass
class CircuitBreakerConfig:
    """Configuration for failure circuit breaker."""

    max_consecutive_failures: int = 5
    max_failure_rate: float = 0.05  # 5%
    min_processed_for_rate: int = 100

    @classmethod
    def from_env(cls) -> "CircuitBreakerConfig":
        """Load config from environment variables."""
        return cls(
            max_consecutive_failures=int(get_env_optional("MAX_CONSECUTIVE_FAILURES", "5")),
            max_failure_rate=float(get_env_optional("MAX_FAILURE_RATE", "0.05")),
            min_processed_for_rate=int(get_env_optional("MIN_PROCESSED_FOR_RATE", "100")),
        )


class CircuitBreakerTripped(Exception):
    """Raised when circuit breaker thresholds are exceeded."""

    def __init__(self, reason: str, consecutive: int, total: int, rate: float):
        self.reason = reason
        self.consecutive = consecutive
        self.total = total
        self.rate = rate
        super().__init__(f"Circuit breaker tripped: {reason}")


# Critical error patterns that should fail immediately (systemic issues)
CRITICAL_ERROR_PATTERNS = [
    "WRDS",  # WRDS connection/auth issues
    "CUDA out of memory",  # GPU OOM
    "CUDA error",  # Other CUDA issues
    "OutOfMemoryError",  # Python OOM
    "ConnectionRefused",  # Network issues
    "AuthenticationError",  # Auth failures
    "PermissionDenied",  # IAM/access issues
    "AccessDenied",  # AWS access issues
    "NoCredentialsError",  # Missing AWS creds
    "LLMUnavailableError",  # vLLM unavailable after health-aware retry
]


def is_critical_error(error: Exception) -> bool:
    """
    Check if an error is critical (systemic) and should fail immediately.

    Critical errors indicate infrastructure/auth problems that won't resolve
    by retrying other firms.
    """
    error_str = f"{type(error).__name__}: {str(error)}"
    return any(pattern in error_str for pattern in CRITICAL_ERROR_PATTERNS)


@dataclass
class ProcessingResult:
    """Result of processing a batch of firms."""

    completed: Set[str] = field(default_factory=set)
    skipped: Set[str] = field(default_factory=set)  # No data / no sentences
    failed: List[Dict[str, Any]] = field(default_factory=list)  # Errors with details
    circuit_breaker_tripped: bool = False
    circuit_breaker_reason: Optional[str] = None

    @property
    def has_failures(self) -> bool:
        """True if any non-skippable errors occurred."""
        return len(self.failed) > 0


def get_env_required(name: str) -> str:
    """Get required environment variable or fail fast."""
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Required environment variable {name} not set")
    return value


def get_env_optional(name: str, default: str) -> str:
    """Get optional environment variable with default."""
    return os.environ.get(name, default)


def get_data_connector(data_source: str, quarter: str, bucket: str) -> DataConnector:
    """
    Factory function to create the appropriate data connector.

    Args:
        data_source: "s3" or "wrds"
        quarter: Quarter string (e.g., "2023Q1") - used for S3 connector
        bucket: S3 bucket name - used for S3 connector

    Returns:
        DataConnector instance (S3TranscriptConnector or WRDSConnector)

    Raises:
        ValueError: If data_source is unknown
    """
    if data_source == "s3":
        from cloud.src.connectors.s3_connector import S3TranscriptConnector
        logger.info(f"Using S3TranscriptConnector for quarter {quarter}")
        return S3TranscriptConnector(bucket=bucket, quarter=quarter)
    elif data_source == "wrds":
        from cloud.src.connectors.wrds_connector import WRDSConnector
        logger.info("Using WRDSConnector (direct WRDS connection)")
        return WRDSConnector()
    else:
        raise ValueError(
            f"Unknown DATA_SOURCE: '{data_source}'. Expected 's3' or 'wrds'."
        )


def quarter_to_date_range(quarter: str) -> Tuple[str, str]:
    """
    Convert quarter string to date range.

    Args:
        quarter: Format "2023Q1"

    Returns:
        Tuple of (start_date, end_date) in YYYY-MM-DD format
    """
    year = int(quarter[:4])
    q = int(quarter[5])

    quarter_starts = {
        1: (1, 1),
        2: (4, 1),
        3: (7, 1),
        4: (10, 1),
    }
    quarter_ends = {
        1: (3, 31),
        2: (6, 30),
        3: (9, 30),
        4: (12, 31),
    }

    start_month, start_day = quarter_starts[q]
    end_month, end_day = quarter_ends[q]

    start_date = f"{year}-{start_month:02d}-{start_day:02d}"
    end_date = f"{year}-{end_month:02d}-{end_day:02d}"

    return start_date, end_date


def load_manifest(s3_client, bucket: str, manifest_key: str) -> List[Dict]:
    """
    Load manifest from S3 (JSONL format).

    Each line is a JSON object: {"batch_id": "...", "quarter": "...", "firm_ids": [...]}
    """
    logger.info(f"Loading manifest from s3://{bucket}/{manifest_key}")

    response = s3_client.get_object(Bucket=bucket, Key=manifest_key)
    content = response["Body"].read().decode("utf-8")

    batches = []
    for line in content.strip().split("\n"):
        if line:
            batches.append(json.loads(line))

    logger.info(f"Loaded manifest with {len(batches)} batches")
    return batches


def load_checkpoint(
    s3_client, bucket: str, batch_id: str, quarter: str
) -> Tuple[Set[str], int]:
    """
    Load checkpoint of completed firm IDs.

    Returns:
        Tuple of (completed_firm_ids, last_chunk_id).
        Returns (empty set, 0) if no checkpoint exists.
    """
    checkpoint_key = f"progress/{quarter}/{batch_id}_checkpoint.json"

    try:
        response = s3_client.get_object(Bucket=bucket, Key=checkpoint_key)
        data = json.loads(response["Body"].read().decode("utf-8"))
        completed = set(data.get("completed_firm_ids", []))
        last_chunk_id = data.get("last_chunk_id", 0)
        logger.info(
            f"Loaded checkpoint: {len(completed)} firms complete, last_chunk_id={last_chunk_id}"
        )
        return completed, last_chunk_id
    except ClientError as e:
        # Handle both real boto3 and mock S3 clients
        error_code = e.response.get("Error", {}).get("Code", "") if hasattr(e, "response") else ""
        if error_code == "NoSuchKey" or "NoSuchKey" in str(e):
            logger.info("No checkpoint found, starting fresh")
            return set(), 0
        raise
    except Exception as e:
        # Fallback for mock clients that may raise different exceptions
        if "NoSuchKey" in str(type(e).__name__) or "NoSuchKey" in str(e):
            logger.info("No checkpoint found, starting fresh")
            return set(), 0
        raise


def save_checkpoint(
    s3_client,
    bucket: str,
    batch_id: str,
    quarter: str,
    completed_firm_ids: Set[str],
    chunk_id: int,
) -> None:
    """Save checkpoint of completed firm IDs to S3."""
    checkpoint_key = f"progress/{quarter}/{batch_id}_checkpoint.json"

    data = {
        "batch_id": batch_id,
        "quarter": quarter,
        "completed_firm_ids": list(completed_firm_ids),
        "last_chunk_id": chunk_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    s3_client.put_object(
        Bucket=bucket,
        Key=checkpoint_key,
        Body=json.dumps(data).encode("utf-8"),
        ContentType="application/json",
    )
    logger.info(f"Saved checkpoint: {len(completed_firm_ids)} firms complete, chunk_id={chunk_id}")


def write_failures_manifest(
    s3_client,
    bucket: str,
    batch_id: str,
    quarter: str,
    failures: List[Dict[str, Any]],
) -> str:
    """
    Write failures manifest to S3 for post-mortem analysis.

    Returns:
        S3 key of the failures manifest
    """
    failures_key = f"progress/{quarter}/{batch_id}_failures.json"

    data = {
        "batch_id": batch_id,
        "quarter": quarter,
        "n_failures": len(failures),
        "failures": failures,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    s3_client.put_object(
        Bucket=bucket,
        Key=failures_key,
        Body=json.dumps(data, default=str).encode("utf-8"),
        ContentType="application/json",
    )
    logger.warning(f"Wrote failures manifest: {len(failures)} failures to s3://{bucket}/{failures_key}")
    return failures_key


def flatten_to_parquet_rows(
    firm_output: Dict[str, Any],
    firm_metadata: Dict[str, Any],
    quarter: str,
) -> List[Dict]:
    """
    Convert FirmProcessor output to Parquet rows with required identifiers.

    Args:
        firm_output: Output from FirmProcessor.process()
        firm_metadata: Metadata from FirmTranscriptData.metadata
        quarter: Quarter string (included for downstream tooling consistency)

    Returns:
        List of dicts, one per topic, with all required fields
    """
    rows = []

    firm_id = firm_output["firm_id"]
    for topic in firm_output.get("topics", []):
        # topic_id is composite string per ADR-007: {firm_id}_{local_topic_id}
        local_topic_id = topic["topic_id"]
        composite_topic_id = f"{firm_id}_{local_topic_id}"

        rows.append(
            {
                "firm_id": firm_id,
                "firm_name": firm_output["firm_name"],
                "quarter": quarter,  # Include for downstream tooling
                "permno": firm_metadata.get("permno"),
                "gvkey": firm_metadata.get("gvkey"),
                "earnings_call_date": firm_metadata.get("earnings_call_date"),
                "topic_id": composite_topic_id,
                "local_topic_id": local_topic_id,  # Preserve for BERTopic reference
                "representation": topic["representation"],
                "summary": topic.get("summary", ""),  # LLM-generated summary
                "naming_method": topic.get("naming_method", "llm"),  # Always "llm" - validates no fallback
                "keywords": topic["keywords"],
                "n_sentences": topic["size"],
                "sentence_ids": topic["sentence_ids"],
                "processing_timestamp": firm_output["metadata"]["processing_timestamp"],
            }
        )

    return rows


def write_parquet_chunk(
    s3_client,
    bucket: str,
    quarter: str,
    batch_id: str,
    chunk_id: int,
    rows: List[Dict],
) -> str:
    """
    Write Parquet chunk to S3 via temp file.

    Uses local temp file + boto3 upload to avoid s3fs complexity.

    Returns:
        S3 key of the written file
    """
    output_key = f"intermediate/firm-topics/quarter={quarter}/{batch_id}_part_{chunk_id:04d}.parquet"

    df = pd.DataFrame(rows)

    # Convert date to proper type if present
    if "earnings_call_date" in df.columns:
        df["earnings_call_date"] = pd.to_datetime(df["earnings_call_date"]).dt.date

    # Write to temp file and upload
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        df.to_parquet(tmp.name, index=False, engine="pyarrow")
        s3_client.upload_file(tmp.name, bucket, output_key)
        os.remove(tmp.name)

    logger.info(f"Wrote {len(rows)} rows to s3://{bucket}/{output_key}")
    return output_key


async def generate_topic_summaries(
    llm_client: XAIClient,
    firm_output: Dict[str, Any],
    sentences: List,  # TranscriptSentence objects
    firm_id: str,
) -> Dict[str, Any]:
    """
    Add LLM summaries to each topic. Raises LLMUnavailableError on failure.

    Args:
        llm_client: Shared XAIClient instance (created once per job)
        firm_output: Output dict from FirmProcessor.process()
        sentences: List of TranscriptSentence objects from firm_data
        firm_id: Firm identifier for logging

    Returns:
        firm_output dict with 'summary' and 'naming_method' fields added to each topic

    Raises:
        LLMUnavailableError: If LLM is unavailable after retries
    """
    # Build sentence lookup by ID
    sentence_map = {s.sentence_id: s.cleaned_text for s in sentences}

    # Prepare topics for LLM batch processing
    topics_for_llm = []
    for topic in firm_output.get("topics", []):
        # Get up to 50 sentences for the topic
        topic_sentences = [
            sentence_map.get(sid, "")
            for sid in topic.get("sentence_ids", [])[:50]
        ]
        # Filter out empty strings
        topic_sentences = [s for s in topic_sentences if s]

        topics_for_llm.append({
            "representation": topic["representation"],
            "sentences": topic_sentences,
        })

    if not topics_for_llm:
        return firm_output

    # Generate summaries in batch (parallel within client)
    logger.info(f"Generating LLM summaries for {len(topics_for_llm)} topics (firm {firm_id})")
    summaries = await llm_client.generate_batch_summaries(
        topics_for_llm, log_first_prompt=True
    )

    # Add summaries to topics - NO FALLBACK, LLM required
    for i, topic in enumerate(firm_output.get("topics", [])):
        summary = summaries[i] if i < len(summaries) else None
        if not summary:
            raise LLMUnavailableError(f"Empty summary for topic {i} (firm {firm_id})", 0)
        topic["summary"] = summary
        topic["naming_method"] = "llm"  # Always "llm" - validates no fallback occurred

    return firm_output


def process_firms(
    connector: DataConnector,
    firm_ids: List[str],
    start_date: str,
    end_date: str,
    embedding_model: SentenceTransformer,
    processor: FirmProcessor,
    s3_client,
    bucket: str,
    batch_id: str,
    quarter: str,
    completed: Set[str],
    last_chunk_id: int,
    checkpoint_interval: int,
    circuit_breaker: CircuitBreakerConfig,
    llm_loop: Optional[asyncio.AbstractEventLoop] = None,
    llm_client: Optional[XAIClient] = None,
) -> ProcessingResult:
    """
    Process firms with checkpointing, failure tracking, and circuit breaker.

    Args:
        connector: Data connector (S3TranscriptConnector or WRDSConnector)
        firm_ids: All firm IDs assigned to this batch
        start_date, end_date: Date range for transcript fetch
        embedding_model: Pre-loaded SentenceTransformer
        processor: FirmProcessor instance
        s3_client: Boto3 S3 client
        bucket: S3 bucket name
        batch_id: This batch's ID
        quarter: Quarter string
        completed: Set of already-completed firm IDs (from checkpoint)
        last_chunk_id: Last chunk ID from checkpoint (for restart consistency)
        checkpoint_interval: Firms per checkpoint
        circuit_breaker: Circuit breaker configuration for failure thresholds
        llm_loop: Event loop for LLM calls (required if llm_client is provided)
        llm_client: Shared XAIClient instance (created once per job for shared recovery)

    Returns:
        ProcessingResult with completed, skipped, and failed firms
    """
    result = ProcessingResult(completed=completed.copy())

    # Filter out completed firms
    pending = [f for f in firm_ids if f not in completed]
    logger.info(
        f"Processing {len(pending)} pending firms ({len(completed)} already complete)"
    )

    if not pending:
        logger.info("All firms already processed")
        return result

    buffer: List[Dict] = []
    firms_in_buffer: List[str] = []

    # Use last_chunk_id from checkpoint + 1 to avoid overwriting on restart
    chunk_id = last_chunk_id + 1 if completed else 0

    # Circuit breaker state
    consecutive_failures = 0
    processed_count = 0  # Excludes skipped firms for rate calculation

    try:
        for i, firm_id in enumerate(pending):
            try:
                logger.info(f"Processing firm {firm_id} ({i + 1}/{len(pending)})")

                # Fetch transcript for this firm
                transcript_data = connector.fetch_transcripts(
                    firm_ids=[firm_id],
                    start_date=start_date,
                    end_date=end_date,
                )

                # Check if firm has data (skippable - not an error)
                if firm_id not in transcript_data.firms:
                    logger.warning(f"No transcript data for firm {firm_id}, skipping")
                    result.skipped.add(firm_id)
                    result.completed.add(firm_id)
                    # Skipped firms don't reset consecutive failures (they're not successes)
                    continue

                firm_data = transcript_data.firms[firm_id]

                if not firm_data.sentences:
                    logger.warning(f"No sentences for firm {firm_id}, skipping")
                    result.skipped.add(firm_id)
                    result.completed.add(firm_id)
                    continue

                # Compute embeddings
                texts = [s.cleaned_text for s in firm_data.sentences]
                embeddings = embedding_model.encode(texts, show_progress_bar=False)

                # Process with FirmProcessor
                output, _ = processor.process(firm_data, embeddings=embeddings)

                # Generate LLM summaries if configured (uses shared client and loop)
                # LLMUnavailableError propagates to trigger circuit breaker
                if llm_loop is not None and llm_client is not None:
                    output = llm_loop.run_until_complete(generate_topic_summaries(
                        llm_client, output, firm_data.sentences, firm_id
                    ))

                # Flatten to rows with metadata (include quarter)
                rows = flatten_to_parquet_rows(output, firm_data.metadata, quarter)
                buffer.extend(rows)
                firms_in_buffer.append(firm_id)

                # Success - reset consecutive failure counter
                consecutive_failures = 0
                processed_count += 1

                # Checkpoint every N firms
                if len(firms_in_buffer) >= checkpoint_interval:
                    # Write chunk
                    write_parquet_chunk(
                        s3_client, bucket, quarter, batch_id, chunk_id, buffer
                    )

                    # Update completed set
                    result.completed.update(firms_in_buffer)

                    # Save checkpoint
                    save_checkpoint(
                        s3_client, bucket, batch_id, quarter, result.completed, chunk_id
                    )

                    # Reset buffer
                    buffer = []
                    firms_in_buffer = []
                    chunk_id += 1

            except Exception as e:
                # Track failure with details for post-mortem
                logger.error(f"Error processing firm {firm_id}: {e}", exc_info=True)
                result.failed.append({
                    "firm_id": firm_id,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                })

                consecutive_failures += 1
                processed_count += 1

                # Check for critical errors (fail immediately)
                if is_critical_error(e):
                    result.circuit_breaker_tripped = True
                    result.circuit_breaker_reason = (
                        f"Critical error detected: {type(e).__name__}: {str(e)[:100]}"
                    )
                    logger.error(f"Circuit breaker: {result.circuit_breaker_reason}")
                    break

                # Check consecutive failure threshold
                if consecutive_failures >= circuit_breaker.max_consecutive_failures:
                    result.circuit_breaker_tripped = True
                    result.circuit_breaker_reason = (
                        f"Consecutive failures ({consecutive_failures}) >= "
                        f"threshold ({circuit_breaker.max_consecutive_failures})"
                    )
                    logger.error(f"Circuit breaker: {result.circuit_breaker_reason}")
                    break

                # Check failure rate threshold (only after minimum processed)
                if processed_count >= circuit_breaker.min_processed_for_rate:
                    failure_rate = len(result.failed) / processed_count
                    if failure_rate >= circuit_breaker.max_failure_rate:
                        result.circuit_breaker_tripped = True
                        result.circuit_breaker_reason = (
                            f"Failure rate ({failure_rate:.1%}) >= "
                            f"threshold ({circuit_breaker.max_failure_rate:.1%}) "
                            f"after {processed_count} firms"
                        )
                        logger.error(f"Circuit breaker: {result.circuit_breaker_reason}")
                        break
    finally:
        pass  # Event loop cleanup handled by caller

    # Write remaining buffer (even if circuit breaker tripped - save progress)
    if buffer:
        write_parquet_chunk(s3_client, bucket, quarter, batch_id, chunk_id, buffer)
        result.completed.update(firms_in_buffer)
        save_checkpoint(s3_client, bucket, batch_id, quarter, result.completed, chunk_id)

    # Log summary
    logger.info(
        f"Batch {batch_id} processing {'STOPPED' if result.circuit_breaker_tripped else 'complete'}: "
        f"{len(result.completed)} completed, "
        f"{len(result.skipped)} skipped, "
        f"{len(result.failed)} failed"
    )

    return result


def main():
    """Main entrypoint for Batch job."""
    logger.info("=" * 60)
    logger.info("Financial Topic Modeling - Map Phase")
    logger.info("=" * 60)

    # Parse environment variables
    manifest_key = get_env_required("MANIFEST_S3_KEY")
    batch_id = get_env_required("BATCH_ID")
    quarter = get_env_required("QUARTER")
    bucket = get_env_required("S3_BUCKET")
    data_source = get_env_optional("DATA_SOURCE", "wrds")  # Default wrds for backward compat
    checkpoint_interval = int(get_env_optional("CHECKPOINT_INTERVAL", "50"))
    embedding_model_name = get_env_optional("EMBEDDING_MODEL_NAME", "all-mpnet-base-v2")
    allow_failures = get_env_optional("ALLOW_FAILURES", "true").lower() == "true"  # Default true for production
    circuit_breaker = CircuitBreakerConfig.from_env()

    logger.info(f"Batch ID: {batch_id}")
    logger.info(f"Quarter: {quarter}")
    logger.info(f"S3 Bucket: {bucket}")
    logger.info(f"Data source: {data_source}")
    logger.info(f"Checkpoint interval: {checkpoint_interval} firms")
    logger.info(f"Embedding model: {embedding_model_name}")
    logger.info(f"Allow failures: {allow_failures}")
    logger.info(
        f"Circuit breaker: max_consecutive={circuit_breaker.max_consecutive_failures}, "
        f"max_rate={circuit_breaker.max_failure_rate:.1%} (after {circuit_breaker.min_processed_for_rate} firms)"
    )

    # Check WRDS credentials only if using WRDS data source
    if data_source == "wrds":
        if not os.environ.get("WRDS_USERNAME") or not os.environ.get("WRDS_PASSWORD"):
            logger.warning(
                "WRDS_USERNAME/WRDS_PASSWORD not set - WRDSConnector will use "
                ".pgpass or Secrets Manager fallback"
            )
        else:
            logger.info("WRDS credentials found in environment")
    else:
        logger.info(f"Using {data_source} data source (WRDS credentials not needed)")

    # Initialize S3 client
    s3_client = boto3.client("s3")

    # Load manifest and find our batch
    manifest = load_manifest(s3_client, bucket, manifest_key)
    our_batch = None
    for batch in manifest:
        if batch["batch_id"] == batch_id:
            our_batch = batch
            break

    if our_batch is None:
        raise ValueError(f"Batch {batch_id} not found in manifest")

    firm_ids = our_batch["firm_ids"]
    logger.info(f"Found {len(firm_ids)} firms assigned to this batch")

    # Load checkpoint (returns tuple with last_chunk_id)
    completed, last_chunk_id = load_checkpoint(s3_client, bucket, batch_id, quarter)

    # Convert quarter to date range
    start_date, end_date = quarter_to_date_range(quarter)
    logger.info(f"Date range: {start_date} to {end_date}")

    # Initialize models ONCE per job (critical for efficiency)
    logger.info(f"Loading embedding model: {embedding_model_name}...")
    embedding_model = SentenceTransformer(embedding_model_name, device="cuda")
    logger.info("Embedding model loaded on GPU")

    logger.info("Initializing BERTopicModel...")
    config = {
        "device": "cuda",
        "embedding_model": embedding_model_name,
    }
    topic_model = BERTopicModel(config, embedding_model=embedding_model)
    processor = FirmProcessor(topic_model, config)
    logger.info("Topic model and processor initialized")

    # Configure LLM if available
    # CRITICAL: Create loop FIRST, set as current, THEN create client
    # This ensures the Semaphore binds to the correct event loop
    llm_base_url = os.environ.get("LLM_BASE_URL")
    llm_model_name = os.environ.get("LLM_MODEL_NAME", "Qwen/Qwen3-8B")
    llm_concurrency = int(os.environ.get("LLM_MAX_CONCURRENT", "10"))
    llm_loop = None
    llm_client = None

    if llm_base_url:
        # 1. Create event loop FIRST
        llm_loop = asyncio.new_event_loop()

        # 2. Set as current loop BEFORE creating client
        asyncio.set_event_loop(llm_loop)

        # 3. NOW create client (semaphore binds to current loop)
        llm_client = XAIClient(
            api_key="dummy",  # vLLM doesn't require auth
            config={
                "model": llm_model_name,
                "max_concurrent": llm_concurrency,
                "timeout": 60,  # Longer timeout for batch inference
                "max_health_wait": 600,  # 10 min max wait for vLLM recovery
                "health_poll_interval": 30,  # Poll every 30 seconds
            },
        )
        logger.info(
            f"LLM client initialized: base_url={llm_base_url}, "
            f"model={llm_model_name}, concurrency={llm_concurrency}"
        )
    else:
        # LLM is REQUIRED - no keyword fallback allowed (research methodology)
        # For local testing without LLM, set ALLOW_NO_LLM=true (output will have empty summaries)
        if get_env_optional("ALLOW_NO_LLM", "false").lower() != "true":
            logger.error("LLM_BASE_URL required but not set. Set ALLOW_NO_LLM=true to bypass (dev only).")
            return 1
        logger.warning("LLM_BASE_URL not set and ALLOW_NO_LLM=true - summaries will be empty (dev mode)")

    # Create data connector based on DATA_SOURCE
    logger.info(f"Creating {data_source} data connector...")
    connector = get_data_connector(data_source, quarter, bucket)
    logger.info("Data connector ready")

    try:
        result = process_firms(
            connector=connector,
            firm_ids=firm_ids,
            start_date=start_date,
            end_date=end_date,
            embedding_model=embedding_model,
            processor=processor,
            s3_client=s3_client,
            bucket=bucket,
            batch_id=batch_id,
            quarter=quarter,
            completed=completed,
            last_chunk_id=last_chunk_id,
            checkpoint_interval=checkpoint_interval,
            circuit_breaker=circuit_breaker,
            llm_loop=llm_loop,
            llm_client=llm_client,
        )
    finally:
        connector.close()
        # Close LLM event loop after all processing
        if llm_loop is not None:
            llm_loop.close()
            logger.info("Closed LLM event loop")

    logger.info("=" * 60)

    # Always write failures manifest if there were any failures
    if result.has_failures:
        write_failures_manifest(
            s3_client, bucket, batch_id, quarter, result.failed
        )

    # Circuit breaker tripped = always fail (systemic issue detected)
    if result.circuit_breaker_tripped:
        logger.error(
            f"Map phase FAILED (circuit breaker): {result.circuit_breaker_reason}. "
            f"{len(result.failed)} failures recorded. See failures manifest."
        )
        logger.info("=" * 60)
        return 1

    # Handle regular failures
    if result.has_failures:
        if allow_failures:
            # ALLOW_FAILURES=true: log warning but exit success
            logger.warning(
                f"Map phase completed with {len(result.failed)} failures. "
                f"ALLOW_FAILURES=true, exiting with success. "
                f"See failures manifest for details."
            )
            logger.info(
                f"Summary: {len(result.completed)} completed, "
                f"{len(result.skipped)} skipped, {len(result.failed)} failed"
            )
            logger.info("=" * 60)
            return 0
        else:
            # ALLOW_FAILURES=false: fail job on any errors
            logger.error(
                f"Map phase FAILED: {len(result.failed)} firms had errors. "
                f"See failures manifest for details."
            )
            logger.info("=" * 60)
            return 1  # Non-zero exit to signal Batch job failure

    logger.info(
        f"Map phase complete: {len(result.completed)} firms processed, "
        f"{len(result.skipped)} skipped (no data)"
    )
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
