"""
Summarize Results Lambda - Count succeeded/failed batches from Map state output.

This Lambda is called by Step Functions after the ProcessBatches Map state
to properly count successes and failures before notification.

Input:
    {
        "quarter": "2023Q1",
        "bucket": "ftm-pipeline-xxx",
        "batch_results": [
            {"job_result": {"Status": "SUCCEEDED", ...}},
            {"status": "FAILED", "error": {...}},
            ...
        ],
        "total_batches": 5,
        "execution_name": "2023Q1-20231201-123456"
    }

Output:
    {
        "quarter": "2023Q1",
        "bucket": "ftm-pipeline-xxx",
        "succeeded": 4,
        "failed": 1,
        "total_batches": 5,
        "execution_name": "2023Q1-20231201-123456",
        "has_failures": true
    }
"""

import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def count_results(batch_results: List[Dict[str, Any]]) -> tuple:
    """
    Count succeeded and failed batches from Map state output.

    Args:
        batch_results: List of batch result objects from Map state

    Returns:
        Tuple of (succeeded_count, failed_count, failure_details)
    """
    succeeded = 0
    failed = 0
    failure_details = []

    for result in batch_results:
        # Check for explicit failure marker (from JobFailed state)
        if result.get("status") == "FAILED":
            failed += 1
            failure_details.append({
                "batch_id": result.get("batch_id"),
                "error": result.get("error"),
            })
        # Check for successful Batch job result
        elif "job_result" in result:
            job_status = result["job_result"].get("Status", "")
            if job_status == "SUCCEEDED":
                succeeded += 1
            else:
                # Job completed but with non-success status
                failed += 1
                failure_details.append({
                    "batch_id": result.get("batch_id"),
                    "job_status": job_status,
                })
        else:
            # Unknown result format - count as success if no error indicators
            if "error" not in result:
                succeeded += 1
            else:
                failed += 1
                failure_details.append({
                    "batch_id": result.get("batch_id", "unknown"),
                    "error": result.get("error"),
                })

    return succeeded, failed, failure_details


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler for summarizing batch results.

    Args:
        event: Lambda event with batch_results from Map state
        context: Lambda context (unused)

    Returns:
        Dict with succeeded/failed counts and metadata
    """
    logger.info(f"Summarize results event: {json.dumps(event)}")

    quarter = event.get("quarter", "Unknown")
    bucket = event.get("bucket", "")
    batch_results = event.get("batch_results", [])
    total_batches = event.get("total_batches", len(batch_results))
    execution_name = event.get("execution_name", "Unknown")

    succeeded, failed, failure_details = count_results(batch_results)

    logger.info(f"Quarter {quarter}: {succeeded} succeeded, {failed} failed out of {total_batches}")

    if failure_details:
        logger.warning(f"Failure details: {json.dumps(failure_details)}")

    return {
        "quarter": quarter,
        "bucket": bucket,
        "succeeded": succeeded,
        "failed": failed,
        "total_batches": total_batches,
        "execution_name": execution_name,
        "has_failures": failed > 0,
        "failure_details": failure_details if failed > 0 else None,
    }
