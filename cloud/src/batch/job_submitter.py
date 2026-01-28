"""
AWS Batch Job Submitter

Handles manifest creation and job submission for parallel firm processing.

Usage:
    from cloud.src.batch import BatchJobSubmitter

    submitter = BatchJobSubmitter(
        job_definition="ftm-firm-processor",
        job_queue="ftm-queue-main",
        s3_bucket="ftm-pipeline-xxx"
    )

    # Submit a full quarter
    results = submitter.submit_quarter("2023Q1", batch_size=1000)

    # Wait for completion
    final_status = submitter.wait_for_completion([r.job_id for r in results])
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import boto3

logger = logging.getLogger(__name__)


@dataclass
class JobSubmissionResult:
    """Result of a job submission."""

    job_id: str
    job_name: str
    batch_id: str
    quarter: str
    n_firms: int
    submitted_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


class BatchJobSubmitter:
    """
    Submit firm processing jobs to AWS Batch.

    This class handles:
    - Creating manifests (JSONL in S3) that define which firms each job processes
    - Submitting jobs with the correct environment variables
    - Monitoring job status and waiting for completion

    Attributes:
        job_definition: Name of the Batch job definition
        job_queue: Name of the Batch job queue
        s3_bucket: S3 bucket for manifests and output
        region: AWS region (default: us-east-1)
    """

    def __init__(
        self,
        job_definition: str,
        job_queue: str,
        s3_bucket: str,
        region: str = "us-east-1",
    ):
        """
        Initialize the job submitter.

        Args:
            job_definition: Batch job definition name or ARN
            job_queue: Batch job queue name or ARN
            s3_bucket: S3 bucket for manifests and intermediate output
            region: AWS region
        """
        self.job_definition = job_definition
        self.job_queue = job_queue
        self.s3_bucket = s3_bucket
        self.region = region

        self.batch_client = boto3.client("batch", region_name=region)
        self.s3_client = boto3.client("s3", region_name=region)

        logger.info(
            f"Initialized BatchJobSubmitter: queue={job_queue}, bucket={s3_bucket}"
        )

    def create_manifest(
        self, quarter: str, firm_ids: List[str], batch_size: int = 1000
    ) -> tuple[str, int]:
        """
        Create a manifest file in S3 defining batches of firms.

        The manifest is a JSONL file where each line defines one batch:
        {"batch_id": "batch_000", "quarter": "2023Q1", "firm_ids": ["123", "456", ...]}

        Args:
            quarter: Quarter string (e.g., "2023Q1")
            firm_ids: List of all firm IDs to process
            batch_size: Number of firms per batch (default: 1000)

        Returns:
            Tuple of (manifest_s3_key, number_of_batches)
        """
        # Create batches
        batches = []
        for i in range(0, len(firm_ids), batch_size):
            batch_firms = firm_ids[i : i + batch_size]
            batch_id = f"batch_{len(batches):03d}"
            batches.append(
                {
                    "batch_id": batch_id,
                    "quarter": quarter,
                    "firm_ids": batch_firms,
                }
            )

        # Write JSONL to S3
        manifest_key = f"manifests/quarter={quarter}/batches.jsonl"
        lines = [json.dumps(batch) for batch in batches]
        content = "\n".join(lines)

        self.s3_client.put_object(
            Bucket=self.s3_bucket,
            Key=manifest_key,
            Body=content.encode("utf-8"),
            ContentType="application/x-ndjson",
        )

        logger.info(
            f"Created manifest s3://{self.s3_bucket}/{manifest_key} "
            f"with {len(batches)} batches ({len(firm_ids)} firms total)"
        )

        return manifest_key, len(batches)

    def submit_job(
        self, quarter: str, batch_id: str, manifest_key: str, n_firms: int
    ) -> JobSubmissionResult:
        """
        Submit a single job to AWS Batch.

        Args:
            quarter: Quarter string
            batch_id: Batch identifier within the manifest
            manifest_key: S3 key of the manifest file
            n_firms: Number of firms in this batch (for logging)

        Returns:
            JobSubmissionResult with job ID and metadata
        """
        job_name = f"ftm-{quarter}-{batch_id}"

        response = self.batch_client.submit_job(
            jobName=job_name,
            jobQueue=self.job_queue,
            jobDefinition=self.job_definition,
            containerOverrides={
                "environment": [
                    {"name": "MANIFEST_S3_KEY", "value": manifest_key},
                    {"name": "BATCH_ID", "value": batch_id},
                    {"name": "QUARTER", "value": quarter},
                ]
            },
        )

        job_id = response["jobId"]
        logger.info(f"Submitted job {job_name} (ID: {job_id}) with {n_firms} firms")

        return JobSubmissionResult(
            job_id=job_id,
            job_name=job_name,
            batch_id=batch_id,
            quarter=quarter,
            n_firms=n_firms,
        )

    def submit_quarter(
        self,
        quarter: str,
        firm_ids: Optional[List[str]] = None,
        batch_size: int = 1000,
    ) -> List[JobSubmissionResult]:
        """
        Submit all jobs for a quarter.

        Creates a manifest and submits one job per batch.

        Args:
            quarter: Quarter string (e.g., "2023Q1")
            firm_ids: List of firm IDs. If None, must provide via other means.
            batch_size: Firms per batch (default: 1000)

        Returns:
            List of JobSubmissionResult for all submitted jobs

        Raises:
            ValueError: If firm_ids is None or empty
        """
        if not firm_ids:
            raise ValueError("firm_ids must be provided and non-empty")

        # Create manifest
        manifest_key, n_batches = self.create_manifest(quarter, firm_ids, batch_size)

        # Load manifest to get batch details
        response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=manifest_key)
        content = response["Body"].read().decode("utf-8")
        batches = [json.loads(line) for line in content.strip().split("\n") if line]

        # Submit jobs
        results = []
        for batch in batches:
            result = self.submit_job(
                quarter=quarter,
                batch_id=batch["batch_id"],
                manifest_key=manifest_key,
                n_firms=len(batch["firm_ids"]),
            )
            results.append(result)

        logger.info(
            f"Submitted {len(results)} jobs for {quarter} "
            f"({len(firm_ids)} firms total)"
        )

        return results

    def get_job_status(self, job_ids: List[str]) -> Dict[str, Dict]:
        """
        Get status of multiple jobs.

        Args:
            job_ids: List of job IDs to check

        Returns:
            Dict mapping job_id to status info:
            {
                "job_id": {"status": "RUNNING", "statusReason": "...", ...}
            }
        """
        # Batch API limits to 100 jobs per call
        all_status = {}

        for i in range(0, len(job_ids), 100):
            chunk = job_ids[i : i + 100]
            response = self.batch_client.describe_jobs(jobs=chunk)

            for job in response["jobs"]:
                all_status[job["jobId"]] = {
                    "status": job["status"],
                    "statusReason": job.get("statusReason", ""),
                    "jobName": job["jobName"],
                    "startedAt": job.get("startedAt"),
                    "stoppedAt": job.get("stoppedAt"),
                }

        return all_status

    def wait_for_completion(
        self,
        job_ids: List[str],
        poll_interval: int = 60,
        timeout: Optional[int] = None,
    ) -> Dict[str, str]:
        """
        Wait for jobs to complete.

        Args:
            job_ids: List of job IDs to wait for
            poll_interval: Seconds between status checks (default: 60)
            timeout: Maximum wait time in seconds (default: None = no timeout)

        Returns:
            Dict mapping job_id to final status (SUCCEEDED, FAILED)
        """
        terminal_states = {"SUCCEEDED", "FAILED"}
        pending = set(job_ids)
        final_status = {}
        start_time = time.time()

        logger.info(f"Waiting for {len(job_ids)} jobs to complete...")

        while pending:
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                logger.warning(f"Timeout reached, {len(pending)} jobs still pending")
                for job_id in pending:
                    final_status[job_id] = "TIMEOUT"
                break

            # Get status
            status = self.get_job_status(list(pending))

            # Check for completions
            for job_id, info in status.items():
                if info["status"] in terminal_states:
                    final_status[job_id] = info["status"]
                    pending.discard(job_id)
                    logger.info(
                        f"Job {info['jobName']} ({job_id}): {info['status']}"
                    )

            if pending:
                # Log progress
                running = sum(
                    1 for j in pending if status.get(j, {}).get("status") == "RUNNING"
                )
                logger.info(
                    f"Progress: {len(final_status)}/{len(job_ids)} complete, "
                    f"{running} running, {len(pending) - running} pending"
                )
                time.sleep(poll_interval)

        # Summary
        succeeded = sum(1 for s in final_status.values() if s == "SUCCEEDED")
        failed = sum(1 for s in final_status.values() if s == "FAILED")
        logger.info(
            f"All jobs complete: {succeeded} succeeded, {failed} failed"
        )

        return final_status

    def list_quarter_output(self, quarter: str) -> List[str]:
        """
        List all Parquet files produced for a quarter.

        Args:
            quarter: Quarter string (e.g., "2023Q1")

        Returns:
            List of S3 keys for output Parquet files
        """
        prefix = f"intermediate/firm-topics/quarter={quarter}/"

        paginator = self.s3_client.get_paginator("list_objects_v2")
        keys = []

        for page in paginator.paginate(Bucket=self.s3_bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".parquet"):
                    keys.append(obj["Key"])

        return keys

    def cleanup_progress(self, quarter: str) -> int:
        """
        Delete checkpoint files for a quarter (after successful completion).

        Args:
            quarter: Quarter string

        Returns:
            Number of checkpoint files deleted
        """
        prefix = f"progress/{quarter}/"

        paginator = self.s3_client.get_paginator("list_objects_v2")
        deleted = 0

        for page in paginator.paginate(Bucket=self.s3_bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                self.s3_client.delete_object(Bucket=self.s3_bucket, Key=obj["Key"])
                deleted += 1

        logger.info(f"Deleted {deleted} checkpoint files for {quarter}")
        return deleted
