"""
Notify Completion Lambda - Send SNS notification on quarter completion.

This Lambda is called by Step Functions at the end of quarter processing
to send a notification with the results summary.

Input:
    {
        "quarter": "2023Q1",
        "succeeded": 5,
        "failed": 0,
        "total_batches": 5,
        "execution_name": "2023Q1-20231201-123456"
    }

Output:
    {
        "notified": true,
        "message_id": "xxx-yyy-zzz"
    }
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def format_notification_message(event: Dict[str, Any]) -> str:
    """Format the SNS notification message."""
    quarter = event.get("quarter", "Unknown")
    succeeded = event.get("succeeded", 0)
    failed = event.get("failed", 0)
    total_batches = event.get("total_batches", succeeded + failed)
    execution_name = event.get("execution_name", "Unknown")

    status = "SUCCESS" if failed == 0 else "PARTIAL" if succeeded > 0 else "FAILED"

    message = f"""
Financial Topic Modeling - Quarter Processing Complete

Quarter: {quarter}
Status: {status}
Execution: {execution_name}

Results:
  - Succeeded: {succeeded}/{total_batches} batches
  - Failed: {failed}/{total_batches} batches

Timestamp: {datetime.utcnow().isoformat()}Z

---
This is an automated notification from the FTM Step Functions pipeline.
"""
    return message.strip()


def format_notification_subject(event: Dict[str, Any]) -> str:
    """Format the SNS notification subject line."""
    quarter = event.get("quarter", "Unknown")
    failed = event.get("failed", 0)

    if failed == 0:
        return f"[FTM] Quarter {quarter} processing completed successfully"
    else:
        return f"[FTM] Quarter {quarter} processing completed with {failed} failures"


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler for completion notification.

    Args:
        event: Lambda event with processing results
        context: Lambda context (unused)

    Returns:
        Dict with notification status
    """
    logger.info(f"Notify completion event: {json.dumps(event)}")

    topic_arn = os.environ.get("SNS_TOPIC_ARN")

    if not topic_arn:
        logger.warning("SNS_TOPIC_ARN not set, skipping notification")
        return {
            "notified": False,
            "reason": "SNS_TOPIC_ARN environment variable not set",
        }

    sns_client = boto3.client("sns")

    try:
        message = format_notification_message(event)
        subject = format_notification_subject(event)

        response = sns_client.publish(
            TopicArn=topic_arn,
            Message=message,
            Subject=subject,
        )

        message_id = response.get("MessageId")
        logger.info(f"Notification sent: {message_id}")

        return {
            "notified": True,
            "message_id": message_id,
            "quarter": event.get("quarter"),
        }

    except Exception as e:
        logger.error(f"Failed to send notification: {e}")
        # Don't fail the state machine for notification errors
        return {
            "notified": False,
            "error": str(e),
        }
