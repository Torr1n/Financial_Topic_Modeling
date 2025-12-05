"""
Map phase entrypoint: Process single firm into topics.

This is the entry point for the map phase container. It:
    1. Loads transcript data for a single firm
    2. Runs topic modeling via FirmProcessor
    3. Saves FirmTopicOutput JSON (locally or to S3)
    4. Writes sentences to DynamoDB (cloud mode only)

Environment Variables:
    Required:
        FIRM_ID           - Firm ID to process (from CSV companyid column)

    Cloud Mode (default):
        S3_INPUT_BUCKET   - Bucket containing transcript CSV
        S3_INPUT_KEY      - Key to transcript CSV file
        S3_OUTPUT_BUCKET  - Bucket for FirmTopicOutput JSON
        S3_OUTPUT_PREFIX  - Key prefix for output (default: "firm-topics/")
        DYNAMODB_TABLE    - Table name for sentence writes

    Local Mode (LOCAL_MODE=true):
        LOCAL_INPUT       - Path to local CSV file
        LOCAL_OUTPUT      - Directory for output JSON
        (DynamoDB writes skipped in local mode)

    Optional:
        CONFIG_PATH       - Path to config YAML (default: /app/config/default.yaml)
        LOG_LEVEL         - Logging level (default: INFO)
"""

import os
import sys
import json
import logging
from pathlib import Path

import yaml


# Setup logging early
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("map-entrypoint")


def load_config() -> dict:
    """
    Load configuration from YAML file.

    Tries CONFIG_PATH env var first, then falls back to defaults.
    """
    config_path = os.environ.get("CONFIG_PATH", "/app/config/default.yaml")

    # Also try relative path for local testing
    if not os.path.exists(config_path):
        local_config = Path(__file__).parent.parent.parent / "config" / "default.yaml"
        if local_config.exists():
            config_path = str(local_config)

    if os.path.exists(config_path):
        logger.info(f"Loading config from {config_path}")
        with open(config_path) as f:
            return yaml.safe_load(f)

    # Fallback to hardcoded defaults
    logger.warning("No config file found, using hardcoded defaults")
    return {
        "embedding_model": "all-mpnet-base-v2",
        "umap": {
            "n_neighbors": 15,
            "n_components": 10,
            "min_dist": 0.0,
            "metric": "cosine",
        },
        "hdbscan": {
            "min_cluster_size": 6,
            "min_samples": 2,
        },
        "validation": {
            "min_firms": 2,
            "max_firm_dominance": 0.4,
        },
    }


def main():
    """
    Main entrypoint for map phase container.

    Exits with code 1 on any error.
    """
    try:
        # 1. Parse environment
        firm_id = os.environ.get("FIRM_ID")
        if not firm_id:
            logger.error("FIRM_ID environment variable is required")
            sys.exit(1)

        local_mode = os.environ.get("LOCAL_MODE", "false").lower() == "true"
        logger.info(f"Processing firm: {firm_id}, local_mode={local_mode}")

        # 2. Load config
        config = load_config()

        # 3. Initialize topic model
        from cloud.src.topic_models.bertopic_model import BERTopicModel

        model = BERTopicModel(config)

        # 4. Initialize processor
        from cloud.src.firm_processor import FirmProcessor

        processor = FirmProcessor(model, config)

        # 5. Load data
        if local_mode:
            local_input = os.environ.get("LOCAL_INPUT")
            if not local_input:
                logger.error("LOCAL_INPUT environment variable required in local mode")
                sys.exit(1)

            from cloud.src.connectors.local_csv import LocalCSVConnector

            connector = LocalCSVConnector(local_input)
        else:
            # Cloud mode - read from S3
            from cloud.src.connectors.s3_connector import S3TranscriptConnector

            connector = S3TranscriptConnector(
                bucket=os.environ["S3_INPUT_BUCKET"],
                key=os.environ["S3_INPUT_KEY"],
            )

        # Fetch data for this firm (no date filter for single-firm processing)
        transcript_data = connector.fetch_transcripts(
            firm_ids=[firm_id],
            start_date="1900-01-01",
            end_date="2100-01-01",
        )

        # Check if firm was found
        if firm_id not in transcript_data.firms:
            logger.error(f"Firm {firm_id} not found in data source")
            logger.info(
                f"Available firm IDs: {connector.get_available_firm_ids()[:10]}"
            )
            sys.exit(1)

        firm_data = transcript_data.firms[firm_id]
        logger.info(f"Loaded {len(firm_data.sentences)} sentences for {firm_id}")

        # 6. Process (returns tuple of output dict and topic_assignments)
        output, topic_assignments = processor.process(firm_data)
        logger.info(f"Discovered {output['n_topics']} topics")

        # 7. Save output
        if local_mode:
            output_dir = os.environ.get("LOCAL_OUTPUT")
            if not output_dir:
                logger.error("LOCAL_OUTPUT environment variable required in local mode")
                sys.exit(1)

            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{firm_id}_topics.json")

            with open(output_path, "w") as f:
                json.dump(output, f, indent=2, default=str)

            logger.info(f"Saved to {output_path}")
        else:
            # Cloud mode - write to S3 and DynamoDB
            from cloud.src.s3_utils import upload_json
            from cloud.src.dynamodb_utils import MapPhaseDynamoDBWriter

            # Write JSON to S3
            s3_key = f"{os.environ.get('S3_OUTPUT_PREFIX', 'firm-topics/')}{firm_id}_topics.json"
            upload_json(os.environ["S3_OUTPUT_BUCKET"], s3_key, output)
            logger.info(f"Uploaded to s3://{os.environ['S3_OUTPUT_BUCKET']}/{s3_key}")

            # Write sentences to DynamoDB
            dynamo_writer = MapPhaseDynamoDBWriter(os.environ["DYNAMODB_TABLE"])
            dynamo_writer.write_firm_sentences(output, firm_data.sentences)
            logger.info("Wrote sentences to DynamoDB")

        logger.info("Map phase completed successfully")

    except KeyError as e:
        logger.error(f"Missing required environment variable: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Map phase failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
