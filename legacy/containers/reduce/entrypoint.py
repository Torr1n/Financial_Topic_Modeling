"""
Reduce phase entrypoint: Aggregate firm topics into cross-firm themes.

This is the entry point for the reduce phase container. It:
    1. Loads firm topic results from S3 or local directory
    2. Filters out firms with no valid topics
    3. Runs theme aggregation via ThemeAggregator
    4. Saves themes (locally or to DynamoDB)

Environment Variables:
    Cloud Mode (default):
        S3_INPUT_BUCKET   - Bucket containing firm topic JSONs
        S3_INPUT_PREFIX   - Key prefix for firm results (default: "firm-topics/")
        DYNAMODB_TABLE    - Table name for theme writes

    Local Mode (LOCAL_MODE=true):
        LOCAL_INPUT       - Directory containing firm topic JSONs
        LOCAL_OUTPUT      - Path for output themes JSON

    Optional:
        CONFIG_PATH       - Path to config YAML
        LOG_LEVEL         - Logging level (default: INFO)
        MIN_FIRMS         - Minimum firms required to proceed (default: 2)
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

import yaml

# Setup logging early
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("reduce-entrypoint")


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


def load_firm_results_from_s3(bucket: str, prefix: str) -> Tuple[List[dict], List[str]]:
    """
    Load all firm topic results from S3.

    Returns:
        (loaded_results, skipped_files) - Successfully loaded results and list of skipped file keys
    """
    import boto3

    s3 = boto3.client("s3")

    loaded = []
    skipped = []

    # List all JSON files
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith("_topics.json"):
                continue

            try:
                response = s3.get_object(Bucket=bucket, Key=key)
                data = json.loads(response["Body"].read().decode("utf-8"))

                # Validate required fields
                if "firm_id" not in data or "topics" not in data:
                    logger.warning(f"Skipping {key}: missing required fields")
                    skipped.append(key)
                    continue

                loaded.append(data)
                logger.debug(f"Loaded {key}: {len(data['topics'])} topics")

            except json.JSONDecodeError as e:
                logger.warning(f"Skipping {key}: invalid JSON - {e}")
                skipped.append(key)
            except Exception as e:
                logger.warning(f"Skipping {key}: {e}")
                skipped.append(key)

    return loaded, skipped


def load_firm_results_local(input_dir: str) -> Tuple[List[dict], List[str]]:
    """Load all firm topic results from local directory."""
    loaded = []
    skipped = []

    for json_file in Path(input_dir).glob("*_topics.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)

            if "firm_id" not in data or "topics" not in data:
                logger.warning(f"Skipping {json_file}: missing required fields")
                skipped.append(str(json_file))
                continue

            loaded.append(data)
            logger.debug(f"Loaded {json_file.name}: {len(data['topics'])} topics")

        except json.JSONDecodeError as e:
            logger.warning(f"Skipping {json_file}: invalid JSON - {e}")
            skipped.append(str(json_file))
        except Exception as e:
            logger.warning(f"Skipping {json_file}: {e}")
            skipped.append(str(json_file))

    return loaded, skipped


def main():
    """
    Main entrypoint for reduce phase container.

    Exits with code 1 on any error.
    """
    try:
        local_mode = os.environ.get("LOCAL_MODE", "false").lower() == "true"
        min_firms = int(os.environ.get("MIN_FIRMS", "2"))

        logger.info(
            f"Reduce phase starting, local_mode={local_mode}, min_firms={min_firms}"
        )

        # 1. Load firm results
        logger.info("Loading firm topic results...")
        if local_mode:
            local_input = os.environ.get("LOCAL_INPUT")
            if not local_input:
                logger.error("LOCAL_INPUT environment variable required in local mode")
                sys.exit(1)
            firm_results, skipped = load_firm_results_local(local_input)
        else:
            firm_results, skipped = load_firm_results_from_s3(
                os.environ["S3_INPUT_BUCKET"],
                os.environ.get("S3_INPUT_PREFIX", "firm-topics/"),
            )

        logger.info(f"Loaded {len(firm_results)} firms, skipped {len(skipped)} files")
        if skipped:
            logger.warning(
                f"Skipped files: {skipped[:10]}{'...' if len(skipped) > 10 else ''}"
            )

        # 2. Check minimum firms requirement
        if len(firm_results) < min_firms:
            logger.error(f"Insufficient firms: {len(firm_results)} < {min_firms}")
            sys.exit(1)

        # 3. Filter out firms with only outliers (no valid topics)
        valid_results = [r for r in firm_results if r.get("n_topics", 0) > 0]
        logger.info(f"{len(valid_results)} firms have valid topics (n_topics > 0)")

        if len(valid_results) < min_firms:
            logger.error(
                f"Insufficient valid firms after filtering: {len(valid_results)} < {min_firms}"
            )
            sys.exit(1)

        # 4. Initialize topic model and aggregator
        from cloud.src.topic_models.bertopic_model import BERTopicModel
        from cloud.src.theme_aggregator import ThemeAggregator

        config = load_config()
        model = BERTopicModel(config)
        aggregator = ThemeAggregator(model, config)

        # 5. Aggregate themes
        logger.info("Aggregating topics into themes...")
        themes = aggregator.aggregate(valid_results)
        logger.info(f"Discovered {len(themes)} themes")

        # 6. Generate theme IDs (already done in ThemeAggregator, but log them)
        for theme in themes:
            logger.debug(
                f"Theme {theme['theme_id']}: {theme['name']} ({theme['n_topics']} topics, {theme['n_firms']} firms)"
            )

        # 7. Save output
        if local_mode:
            output_path = os.environ.get("LOCAL_OUTPUT")
            if not output_path:
                logger.error("LOCAL_OUTPUT environment variable required in local mode")
                sys.exit(1)

            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            with open(output_path, "w") as f:
                json.dump(themes, f, indent=2, default=str)
            logger.info(f"Saved {len(themes)} themes to {output_path}")
        else:
            from cloud.src.dynamodb_utils import ReducePhaseDynamoDBWriter

            writer = ReducePhaseDynamoDBWriter(os.environ["DYNAMODB_TABLE"])
            writer.write_themes(themes)
            logger.info(f"Wrote {len(themes)} themes to DynamoDB")

        # 8. Log summary
        logger.info("=== Reduce Phase Summary ===")
        logger.info(f"  Firms loaded: {len(firm_results)}")
        logger.info(f"  Firms with topics: {len(valid_results)}")
        logger.info(f"  Files skipped: {len(skipped)}")
        logger.info(f"  Themes discovered: {len(themes)}")
        if themes:
            total_topics = sum(t["n_topics"] for t in themes)
            logger.info(f"  Total topics in themes: {total_topics}")
        logger.info("Reduce phase completed successfully")

    except KeyError as e:
        logger.error(f"Missing required environment variable: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Reduce phase failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
