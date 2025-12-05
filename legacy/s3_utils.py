"""
S3 utility functions for the Financial Topic Modeling pipeline.

Provides simple helpers for JSON upload/download to S3.
Keeps S3 interactions isolated for easier testing with moto.
"""

import json
import logging
from typing import Any, Dict, List

import boto3

logger = logging.getLogger(__name__)


def upload_json(bucket: str, key: str, data: Dict[str, Any]) -> None:
    """
    Upload a dict as JSON to S3.

    Args:
        bucket: S3 bucket name
        key: S3 object key
        data: Dict to serialize as JSON

    Raises:
        botocore.exceptions.ClientError: On S3 errors
    """
    s3 = boto3.client("s3")

    # Serialize with default=str to handle datetime and other non-serializable types
    json_body = json.dumps(data, default=str, indent=2)

    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json_body,
        ContentType="application/json",
    )

    logger.info(f"Uploaded JSON to s3://{bucket}/{key}")


def download_json(bucket: str, key: str) -> Dict[str, Any]:
    """
    Download and parse JSON from S3.

    Args:
        bucket: S3 bucket name
        key: S3 object key

    Returns:
        Parsed JSON dict

    Raises:
        botocore.exceptions.ClientError: On S3 errors (including NoSuchKey)
    """
    s3 = boto3.client("s3")

    response = s3.get_object(Bucket=bucket, Key=key)
    content = response["Body"].read().decode("utf-8")

    logger.info(f"Downloaded JSON from s3://{bucket}/{key}")

    return json.loads(content)


def list_json_files(
    bucket: str,
    prefix: str,
    suffix: str = "_topics.json",
) -> List[str]:
    """
    List JSON files in S3 with given prefix and suffix.

    Args:
        bucket: S3 bucket name
        prefix: Key prefix to filter by
        suffix: Key suffix to filter by (default: "_topics.json")

    Returns:
        List of matching S3 keys
    """
    s3 = boto3.client("s3")

    keys = []
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(suffix):
                keys.append(key)

    logger.info(f"Found {len(keys)} files matching s3://{bucket}/{prefix}*{suffix}")

    return keys
