"""
Prefetch Check Lambda - Verify prefetch manifest exists for a quarter.

This Lambda is called by Step Functions to check if prefetch data is available
before starting batch processing.

Input:
    {
        "quarter": "2023Q1",
        "bucket": "ftm-pipeline-xxx"
    }

Output:
    {
        "exists": true,
        "quarter": "2023Q1",
        "n_firms": 1234,
        "manifest_key": "prefetch/transcripts/quarter=2023Q1/manifest.json"
    }

Or if not found:
    {
        "exists": false,
        "quarter": "2023Q1",
        "error": "Prefetch manifest not found"
    }
"""

import json
import logging
import os
from typing import Any, Dict

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def get_manifest_key(quarter: str) -> str:
    """Get S3 key for prefetch manifest."""
    return f"prefetch/transcripts/quarter={quarter}/manifest.json"


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler for prefetch existence check.

    Args:
        event: Lambda event with quarter and bucket
        context: Lambda context (unused)

    Returns:
        Dict with exists flag and manifest details
    """
    logger.info(f"Prefetch check event: {json.dumps(event)}")

    quarter = event.get("quarter")
    bucket = event.get("bucket") or os.environ.get("S3_BUCKET")

    if not quarter:
        return {
            "exists": False,
            "error": "Missing required field: quarter",
        }

    if not bucket:
        return {
            "exists": False,
            "error": "Missing required field: bucket (or S3_BUCKET env var)",
        }

    manifest_key = get_manifest_key(quarter)
    s3_client = boto3.client("s3")

    try:
        # Check if manifest exists (HEAD request)
        s3_client.head_object(Bucket=bucket, Key=manifest_key)

        # Load manifest to get firm count
        response = s3_client.get_object(Bucket=bucket, Key=manifest_key)

        # Handle potential gzip compression
        import gzip
        content_encoding = response.get("ContentEncoding", "")
        body = response["Body"].read()

        if content_encoding == "gzip":
            body = gzip.decompress(body)

        manifest = json.loads(body.decode("utf-8"))
        n_firms = manifest.get("n_firms", 0)

        logger.info(f"Prefetch manifest found for {quarter}: {n_firms} firms")

        return {
            "exists": True,
            "quarter": quarter,
            "bucket": bucket,
            "n_firms": n_firms,
            "manifest_key": manifest_key,
        }

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")

        if error_code in ("404", "NoSuchKey"):
            logger.warning(f"Prefetch manifest not found for {quarter}")
            return {
                "exists": False,
                "quarter": quarter,
                "bucket": bucket,
                "error": f"Prefetch manifest not found for quarter {quarter}",
            }

        logger.error(f"Error checking prefetch manifest: {e}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
