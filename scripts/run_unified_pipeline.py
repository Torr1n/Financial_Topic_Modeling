#!/usr/bin/env python3
"""
Run the unified pipeline - cloud or local deployment.

Environment Variables:
    DATABASE_URL: PostgreSQL connection string (required)
    XAI_API_KEY: xAI API key for LLM summaries (optional, falls back to keywords)
    CSV_PATH: Path to transcripts CSV (default: auto-detect)
    DEVICE: "cuda" or "cpu" (default: auto-detect)
    TEST_MODE: "mag7" to run with MAG7 firms only (for validation)
    MAX_FIRMS: Maximum number of firms to process (default: all)

Usage:
    # Cloud - full run
    python scripts/run_unified_pipeline.py

    # Cloud - MAG7 validation (replicates local test)
    TEST_MODE=mag7 python scripts/run_unified_pipeline.py

    # Local with custom settings
    DATABASE_URL="postgresql://..." MAX_FIRMS=10 python scripts/run_unified_pipeline.py
"""

import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file if present
from dotenv import load_dotenv
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"Loaded environment from: {env_path}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ftm-pipeline")

# Configuration from environment
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    logger.error("DATABASE_URL environment variable is required")
    logger.error("Example: postgresql://ftm:password@host:5432/ftm")
    sys.exit(1)

# Auto-detect CSV path
CSV_PATH = os.environ.get("CSV_PATH")
if not CSV_PATH:
    # Try common locations
    candidates = [
        project_root / "transcripts_2023-01-01_to_2023-03-31_enriched.csv",
        project_root / "data" / "transcripts_2023-01-01_to_2023-03-31_enriched.csv",
        Path("/home/ubuntu/data/transcripts_2023-01-01_to_2023-03-31_enriched.csv"),
    ]
    for candidate in candidates:
        if candidate.exists():
            CSV_PATH = str(candidate)
            break
    if not CSV_PATH:
        logger.error("CSV file not found. Set CSV_PATH environment variable.")
        sys.exit(1)

logger.info(f"Using CSV: {CSV_PATH}")

# Auto-detect device
DEVICE = os.environ.get("DEVICE")
if not DEVICE:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")

# Optional limit on firms
MAX_FIRMS = os.environ.get("MAX_FIRMS")
if MAX_FIRMS:
    MAX_FIRMS = int(MAX_FIRMS)
    logger.info(f"Processing limited to {MAX_FIRMS} firms")

# Test mode - MAG7 firms for validation
TEST_MODE = os.environ.get("TEST_MODE", "").lower()
MAG7_FIRM_IDS = [
    "21835",      # Microsoft
    "29096",      # Alphabet (Google)
    "27444752",   # Tesla, Inc.
    "18749",      # Amazon.com, Inc.
    "20765463",   # Meta Platforms, Inc.
    "32307",      # NVIDIA Corporation
    "24937",      # Apple Inc
    "25016048",   # Broadcom Inc.
    "33348547",   # Arista Networks Inc
    "19691",      # Cisco Systems, Inc.
    "22247",      # Oracle Corporation
]

if TEST_MODE == "mag7":
    logger.info(f"TEST MODE: Running with MAG7 firms only ({len(MAG7_FIRM_IDS)} firms)")

# Load configuration from YAML file
CONFIG_PATH = os.environ.get("CONFIG_PATH")
if not CONFIG_PATH:
    # Try common locations
    config_candidates = [
        project_root / "cloud" / "config" / "production.yaml",
        project_root / "cloud" / "config" / "local.yaml",
    ]
    for candidate in config_candidates:
        if candidate.exists():
            CONFIG_PATH = str(candidate)
            break

if CONFIG_PATH and Path(CONFIG_PATH).exists():
    import yaml
    with open(CONFIG_PATH) as f:
        CONFIG = yaml.safe_load(f)
    logger.info(f"Loaded config from: {CONFIG_PATH}")
else:
    # Fallback to hardcoded defaults (backward compatibility)
    logger.warning("No config file found, using hardcoded defaults")
    CONFIG = {
        "embedding": {
            "model": "all-mpnet-base-v2",
            "dimension": 768,
            "device": "cuda" if DEVICE == "cuda" else "cpu",
        },
        "firm_topic_model": {
            "umap": {
                "n_neighbors": 15,
                "n_components": 10,
                "min_dist": 0.0,
                "metric": "cosine",
                "random_state": 42,
            },
            "hdbscan": {
                "min_cluster_size": 6,
                "min_samples": 2,
                "metric": "euclidean",
                "cluster_selection_method": "leaf",
            },
        },
        "theme_topic_model": {
            "umap": {
                "n_neighbors": 30,
                "n_components": 15,
                "min_dist": 0.05,
                "metric": "cosine",
                "random_state": 42,
            },
            "hdbscan": {
                "min_cluster_size": 10,
                "min_samples": 3,
                "metric": "euclidean",
                "cluster_selection_method": "eom",
            },
        },
        "validation": {
            "min_firms": 2,
            "max_firm_dominance": 0.4,
        },
    }

# Set EMBEDDING_DIMENSION env var for database models
embedding_dim = CONFIG.get("embedding", {}).get("dimension", 768)
os.environ["EMBEDDING_DIMENSION"] = str(embedding_dim)
logger.info(f"Embedding dimension: {embedding_dim}")


class FilteredDataConnector:
    """
    DataConnector wrapper that can filter or limit firms.

    Supports:
    - Specific firm IDs (for MAG7 test mode)
    - Max firms limit (for gradual scaling)
    """

    def __init__(self, csv_path: str, firm_ids: list = None, max_firms: int = None):
        from cloud.src.connectors.local_csv import LocalCSVConnector
        self._connector = LocalCSVConnector(csv_path)
        self._specific_ids = firm_ids  # If set, only these firms
        self._max_firms = max_firms
        self._firm_ids = None

    def get_available_firm_ids(self) -> list:
        """Return firm IDs based on configuration."""
        if self._firm_ids is None:
            if self._specific_ids:
                # Use specific firm IDs (e.g., MAG7)
                self._firm_ids = self._specific_ids
            else:
                # Get all firms, optionally limited
                all_ids = self._connector.get_available_firm_ids()
                if self._max_firms:
                    self._firm_ids = all_ids[:self._max_firms]
                else:
                    self._firm_ids = all_ids
        return self._firm_ids

    def fetch_transcripts(self, firm_ids, start_date, end_date):
        """Fetch transcripts for specified firms."""
        return self._connector.fetch_transcripts(firm_ids, start_date, end_date)


def check_database():
    """Verify database connection and pgvector."""
    from sqlalchemy import create_engine, text

    logger.info("Checking database connection...")
    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            result.fetchone()
        logger.info("Database connection OK")

        # Check pgvector extension
        with engine.connect() as conn:
            result = conn.execute(text("SELECT extname FROM pg_extension WHERE extname = 'vector'"))
            if not result.fetchone():
                logger.info("Enabling pgvector extension...")
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()
                logger.info("pgvector extension enabled")
            else:
                logger.info("pgvector extension already enabled")

        engine.dispose()
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


def run_pipeline():
    """Run the unified pipeline."""
    from cloud.src.pipeline import UnifiedPipeline

    logger.info("=" * 60)
    logger.info("FINANCIAL TOPIC MODELING PIPELINE")
    logger.info("=" * 60)

    # Check database
    if not check_database():
        return False

    # Create data connector based on mode
    logger.info(f"Loading data from: {CSV_PATH}")

    if TEST_MODE == "mag7":
        # MAG7 test mode - use specific firm IDs
        connector = FilteredDataConnector(CSV_PATH, firm_ids=MAG7_FIRM_IDS)
        logger.info(f"MAG7 test mode: {len(MAG7_FIRM_IDS)} firms")
    else:
        # Normal mode - all firms or limited
        connector = FilteredDataConnector(CSV_PATH, max_firms=MAX_FIRMS)

    firm_ids = connector.get_available_firm_ids()
    logger.info(f"Processing {len(firm_ids)} firms")

    # Create and run pipeline
    logger.info("Creating UnifiedPipeline...")
    logger.info("(Loading embedding model - may take a moment)")

    pipeline = UnifiedPipeline(
        database_url=DATABASE_URL,
        config=CONFIG,
        device=DEVICE,
    )

    logger.info("Running pipeline...")
    pipeline.run(connector)

    return True


def print_summary():
    """Print results summary."""
    from sqlalchemy import create_engine, text

    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)

    engine = create_engine(DATABASE_URL)

    with engine.connect() as conn:
        # Counts
        firms = conn.execute(text("SELECT COUNT(*) FROM firms")).fetchone()[0]
        sentences = conn.execute(text("SELECT COUNT(*) FROM sentences")).fetchone()[0]
        topics = conn.execute(text("SELECT COUNT(*) FROM topics")).fetchone()[0]
        themes = conn.execute(text("SELECT COUNT(*) FROM themes")).fetchone()[0]

        logger.info(f"Firms processed: {firms}")
        logger.info(f"Total sentences: {sentences}")
        logger.info(f"Total topics: {topics}")
        logger.info(f"Themes discovered: {themes}")

        # Top themes
        result = conn.execute(text("""
            SELECT name, n_topics, n_firms
            FROM themes
            ORDER BY n_topics DESC
            LIMIT 10
        """))
        top_themes = result.fetchall()

        if top_themes:
            logger.info("\nTop 10 themes:")
            for theme in top_themes:
                name = theme[0][:60] if theme[0] else "Unknown"
                logger.info(f"  - {name}... ({theme[1]} topics from {theme[2]} firms)")

        # Embedding coverage
        result = conn.execute(text("""
            SELECT
                (SELECT COUNT(*) FROM sentences WHERE embedding IS NOT NULL) as sent_emb,
                (SELECT COUNT(*) FROM sentences) as sent_total,
                (SELECT COUNT(*) FROM topics WHERE embedding IS NOT NULL) as topic_emb,
                (SELECT COUNT(*) FROM topics) as topic_total
        """))
        row = result.fetchone()
        logger.info(f"\nEmbedding coverage:")
        logger.info(f"  Sentences: {row[0]}/{row[1]}")
        logger.info(f"  Topics: {row[2]}/{row[3]}")

    engine.dispose()
    logger.info("=" * 60)


def main():
    """Main entry point."""
    import time
    start_time = time.time()

    try:
        success = run_pipeline()
        if success:
            print_summary()
            elapsed = time.time() - start_time
            logger.info(f"\nPipeline completed in {elapsed/60:.1f} minutes")
            return 0
        else:
            logger.error("\nPipeline failed!")
            return 1
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        return 1
    except Exception as e:
        logger.exception(f"\nPipeline failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
