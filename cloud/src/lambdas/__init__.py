"""
Lambda functions for Step Functions orchestration.

These Lambda functions support the Step Functions state machine for
multi-quarter batch processing orchestration.

Functions:
- prefetch_check: Verify prefetch manifest exists for a quarter
- create_batch_manifest: Create batch manifest and return batch_ids
- summarize_results: Count succeeded/failed batches from Map state output
- notify_completion: Send SNS notification on quarter completion
"""

from cloud.src.lambdas.prefetch_check import handler as prefetch_check_handler
from cloud.src.lambdas.create_batch_manifest import handler as create_batch_manifest_handler
from cloud.src.lambdas.summarize_results import handler as summarize_results_handler
from cloud.src.lambdas.notify_completion import handler as notify_completion_handler

__all__ = [
    "prefetch_check_handler",
    "create_batch_manifest_handler",
    "summarize_results_handler",
    "notify_completion_handler",
]
