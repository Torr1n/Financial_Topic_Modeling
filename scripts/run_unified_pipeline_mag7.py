#!/usr/bin/env python3
"""
Run the unified pipeline with MAG7 firms for local testing.

Prerequisites:
    1. Start Postgres: docker-compose up -d
    2. Activate venv: source venv/bin/activate
    3. Create .env file with XAI_API_KEY (optional, for LLM summaries)

Usage:
    python scripts/run_unified_pipeline_mag7.py

This script:
    1. Connects to local Postgres (docker-compose)
    2. Loads MAG7 firms from local CSV
    3. Runs UnifiedPipeline (firm processing + theme aggregation)
    4. Outputs summary to console
    5. Results viewable in DBeaver at: postgresql://ftm:ftm_password@localhost:5432/ftm
"""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
from dotenv import load_dotenv
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"Loaded environment from: {env_path}")
else:
    print(f"No .env file found at: {env_path} (LLM summaries will use fallbacks)")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("mag7-test")

# Test firm IDs (MAG7 + additional tech firms from previous output)
TEST_FIRM_IDS = [
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

# Database connection
DATABASE_URL = "postgresql://ftm:ftm_password@localhost:5432/ftm"

# CSV path
CSV_PATH = project_root / "transcripts_2023-01-01_to_2023-03-31_enriched.csv"

# Config
CONFIG = {
    "embedding_model": "all-mpnet-base-v2",
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
    "validation": {
        "min_firms": 2,
        "max_firm_dominance": 0.4,
    },
}


class FilteredDataConnector:
    """
    Custom DataConnector that filters to specific firms.

    Wraps LocalCSVConnector but limits to specified firm IDs.
    """

    def __init__(self, csv_path: str, firm_ids: list):
        from cloud.src.connectors.local_csv import LocalCSVConnector
        self._connector = LocalCSVConnector(csv_path)
        self._firm_ids = firm_ids

    def get_available_firm_ids(self) -> list:
        """Return only the test firm IDs."""
        return self._firm_ids

    def fetch_transcripts(self, firm_ids, start_date, end_date):
        """Fetch transcripts for specified firms."""
        return self._connector.fetch_transcripts(firm_ids, start_date, end_date)


def check_postgres_connection():
    """Verify Postgres is running and accessible."""
    from sqlalchemy import create_engine, text

    logger.info("Checking Postgres connection...")
    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            result.fetchone()
        logger.info("Postgres connection OK")

        # Check pgvector extension
        with engine.connect() as conn:
            result = conn.execute(text("SELECT extname FROM pg_extension WHERE extname = 'vector'"))
            if not result.fetchone():
                logger.info("Enabling pgvector extension...")
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()
                logger.info("pgvector extension enabled")

        engine.dispose()
        return True
    except Exception as e:
        logger.error(f"Postgres connection failed: {e}")
        logger.error("Make sure docker-compose is running: docker-compose up -d")
        return False


def clear_database():
    """Clear existing data for fresh run."""
    from sqlalchemy import create_engine, text

    logger.info("Clearing existing database data...")
    engine = create_engine(DATABASE_URL)

    with engine.connect() as conn:
        # Drop tables in reverse order of dependencies
        conn.execute(text("DROP TABLE IF EXISTS sentences CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS topics CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS themes CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS firms CASCADE"))
        conn.commit()

    engine.dispose()
    logger.info("Database cleared")


def run_pipeline():
    """Run the unified pipeline with MAG7 firms."""
    from cloud.src.pipeline import UnifiedPipeline

    logger.info("=" * 60)
    logger.info("UNIFIED PIPELINE - MAG7 TEST")
    logger.info("=" * 60)

    # Check CSV exists
    if not CSV_PATH.exists():
        logger.error(f"CSV not found: {CSV_PATH}")
        return False

    # Check Postgres
    if not check_postgres_connection():
        return False

    # Clear database for fresh run
    clear_database()

    # Create data connector
    logger.info(f"Loading data from: {CSV_PATH}")
    logger.info(f"Filtering to test firms: {TEST_FIRM_IDS}")
    connector = FilteredDataConnector(str(CSV_PATH), TEST_FIRM_IDS)

    # Create and run pipeline
    logger.info("Creating UnifiedPipeline...")
    logger.info("(This will load the embedding model - may take a moment)")

    pipeline = UnifiedPipeline(
        database_url=DATABASE_URL,
        config=CONFIG,
        device="cpu",  # Use CPU for local testing (change to "cuda" if GPU available)
    )

    logger.info("Running pipeline...")
    pipeline.run(connector)

    return True


def print_results_summary():
    """Print summary of results from Postgres."""
    from sqlalchemy import create_engine, text

    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)

    engine = create_engine(DATABASE_URL)

    with engine.connect() as conn:
        # Firms summary
        result = conn.execute(text("""
            SELECT company_id, name, processed_at
            FROM firms
            ORDER BY name
        """))
        firms = result.fetchall()

        logger.info(f"\nFirms processed: {len(firms)}")
        for firm in firms:
            logger.info(f"  - {firm[0]}: {firm[1]} (processed: {firm[2]})")

        # Topics summary
        result = conn.execute(text("""
            SELECT f.name, COUNT(t.id) as topic_count
            FROM firms f
            LEFT JOIN topics t ON f.id = t.firm_id
            GROUP BY f.id, f.name
            ORDER BY f.name
        """))
        topic_counts = result.fetchall()

        logger.info("\nTopics per firm:")
        for row in topic_counts:
            logger.info(f"  - {row[0]}: {row[1]} topics")

        # Total sentences
        result = conn.execute(text("SELECT COUNT(*) FROM sentences"))
        sentence_count = result.fetchone()[0]
        logger.info(f"\nTotal sentences: {sentence_count}")

        # Total topics
        result = conn.execute(text("SELECT COUNT(*) FROM topics"))
        topic_count = result.fetchone()[0]
        logger.info(f"Total topics: {topic_count}")

        # Themes summary
        result = conn.execute(text("""
            SELECT name, n_topics, n_firms
            FROM themes
            ORDER BY n_topics DESC
        """))
        themes = result.fetchall()

        logger.info(f"\nThemes discovered: {len(themes)}")
        for theme in themes[:10]:  # Top 10
            logger.info(f"  - {theme[0][:50]}... ({theme[1]} topics from {theme[2]} firms)")

        if len(themes) > 10:
            logger.info(f"  ... and {len(themes) - 10} more themes")

        # Check embeddings
        result = conn.execute(text("""
            SELECT
                COUNT(*) as total,
                COUNT(embedding) as with_embedding
            FROM sentences
        """))
        row = result.fetchone()
        logger.info(f"\nSentence embeddings: {row[1]}/{row[0]} populated")

        result = conn.execute(text("""
            SELECT
                COUNT(*) as total,
                COUNT(embedding) as with_embedding
            FROM topics
        """))
        row = result.fetchone()
        logger.info(f"Topic embeddings: {row[1]}/{row[0]} populated")

    engine.dispose()

    logger.info("\n" + "=" * 60)
    logger.info("View full results in DBeaver:")
    logger.info("  Connection: postgresql://ftm:ftm_password@localhost:5432/ftm")
    logger.info("  Tables: firms, sentences, topics, themes")
    logger.info("=" * 60)


def main():
    """Main entry point."""
    try:
        success = run_pipeline()
        if success:
            print_results_summary()
            logger.info("\nPipeline completed successfully!")
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
