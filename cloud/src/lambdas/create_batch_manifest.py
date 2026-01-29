"""
Create Batch Manifest Lambda - Generate batch manifest and return batch_ids.

This Lambda is called by Step Functions to create a batch manifest JSONL file
and return the list of batch_ids for the Map state to iterate over.

CRITICAL: Returns batch_ids ONLY (strings), not full batches with firm_ids.
This avoids the Step Functions 256KB payload limit. Actual firm_ids are read
from S3 manifest at job runtime.

Input:
    {
        "quarter": "2023Q1",
        "bucket": "ftm-pipeline-xxx",
        "batch_size": 1000  # optional, default 1000
    }

Output:
    {
        "quarter": "2023Q1",
        "manifest_s3_key": "manifests/2023Q1/manifest_20231201_120000.jsonl",
        "batch_ids": ["2023Q1_batch_0000", "2023Q1_batch_0001", ...],
        "n_firms": 1234,
        "n_batches": 2
    }
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

DEFAULT_BATCH_SIZE = 1000


def get_prefetch_manifest_key(quarter: str) -> str:
    """Get S3 key for prefetch manifest."""
    return f"prefetch/transcripts/quarter={quarter}/manifest.json"


def load_prefetch_manifest(s3_client, bucket: str, quarter: str) -> Dict[str, Any]:
    """
    Load prefetch manifest from S3.

    Reads firm_ids from the prefetch manifest (S3), NOT from WRDS.
    This ensures Step Functions is completely decoupled from WRDS/MFA.

    Args:
        s3_client: Boto3 S3 client
        bucket: S3 bucket name
        quarter: Quarter string

    Returns:
        Manifest dict with firm_to_chunk mapping

    Raises:
        ValueError: If manifest not found
    """
    import gzip

    manifest_key = get_prefetch_manifest_key(quarter)

    try:
        response = s3_client.get_object(Bucket=bucket, Key=manifest_key)

        # Handle gzip compression
        content_encoding = response.get("ContentEncoding", "")
        body = response["Body"].read()

        if content_encoding == "gzip":
            body = gzip.decompress(body)

        return json.loads(body.decode("utf-8"))

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code in ("404", "NoSuchKey"):
            raise ValueError(f"Prefetch manifest not found for quarter {quarter}")
        raise


def get_firm_ids_from_manifest(manifest: Dict[str, Any]) -> List[str]:
    """Extract sorted firm_ids from prefetch manifest."""
    firm_to_chunk = manifest.get("firm_to_chunk", {})
    return sorted(firm_to_chunk.keys())


def create_batch_manifest(
    s3_client,
    bucket: str,
    quarter: str,
    firm_ids: List[str],
    batch_size: int,
) -> tuple:
    """
    Create batch manifest JSONL on S3.

    Args:
        s3_client: Boto3 S3 client
        bucket: S3 bucket name
        quarter: Quarter string
        firm_ids: List of firm IDs to process
        batch_size: Firms per batch

    Returns:
        Tuple of (manifest_key, batch_ids)
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    manifest_key = f"manifests/{quarter}/manifest_{timestamp}.jsonl"

    batches = []
    batch_ids = []

    for i in range(0, len(firm_ids), batch_size):
        batch_firms = firm_ids[i:i + batch_size]
        batch_id = f"{quarter}_batch_{len(batches):04d}"

        batches.append({
            "batch_id": batch_id,
            "quarter": quarter,
            "firm_ids": batch_firms,
        })
        batch_ids.append(batch_id)

    # Write JSONL to S3
    content = "\n".join(json.dumps(b) for b in batches)
    s3_client.put_object(
        Bucket=bucket,
        Key=manifest_key,
        Body=content.encode("utf-8"),
        ContentType="application/json",
    )

    logger.info(f"Created batch manifest: {manifest_key} with {len(batch_ids)} batches")

    return manifest_key, batch_ids


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler for batch manifest creation.

    Args:
        event: Lambda event with quarter, bucket, and optional batch_size
        context: Lambda context (unused)

    Returns:
        Dict with manifest_s3_key and batch_ids (strings only)
    """
    logger.info(f"Create batch manifest event: {json.dumps(event)}")

    quarter = event.get("quarter")
    bucket = event.get("bucket") or os.environ.get("S3_BUCKET")
    # Guard against None/missing batch_size - use default if falsy
    batch_size = event.get("batch_size") or DEFAULT_BATCH_SIZE

    if not quarter:
        raise ValueError("Missing required field: quarter")

    if not bucket:
        raise ValueError("Missing required field: bucket (or S3_BUCKET env var)")

    s3_client = boto3.client("s3")

    # Load firm_ids from prefetch manifest (NOT from WRDS)
    logger.info(f"Loading firm_ids from prefetch manifest for {quarter}")
    prefetch_manifest = load_prefetch_manifest(s3_client, bucket, quarter)
    firm_ids = get_firm_ids_from_manifest(prefetch_manifest)

    logger.info(f"Found {len(firm_ids)} firms in prefetch manifest")

    # Create batch manifest
    manifest_key, batch_ids = create_batch_manifest(
        s3_client, bucket, quarter, firm_ids, batch_size
    )

    return {
        "quarter": quarter,
        "bucket": bucket,
        "manifest_s3_key": manifest_key,
        "batch_ids": batch_ids,  # Strings only - avoids payload overflow
        "n_firms": len(firm_ids),
        "n_batches": len(batch_ids),
    }
