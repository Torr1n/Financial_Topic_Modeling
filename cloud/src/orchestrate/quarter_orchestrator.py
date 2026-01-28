"""
Quarter Orchestrator - Coordinate prefetch check, batch submission, and monitoring.

This module provides the high-level orchestration for processing a quarter:
1. Check if prefetch data exists (manifest.json required)
2. Get firm_ids from manifest (no WRDS connection needed)
3. Create batch manifest, submit jobs with DATA_SOURCE=s3
4. Monitor job completion
5. Return summary

IMPORTANT: Prefetch is REQUIRED for Batch runs. No silent fallback to WRDS.
This prevents accidental MFA challenges from each Batch container.

Usage:
    orchestrator = QuarterOrchestrator(
        s3_bucket="ftm-pipeline-xxx",
        job_definition="ftm-firm-processor",
        job_queue="ftm-queue-main"
    )

    # Check if prefetch exists
    if not orchestrator.prefetch_exists("2023Q1"):
        orchestrator.run_prefetch("2023Q1")

    # Run batch processing
    result = orchestrator.run_quarter("2023Q1", batch_size=1000)
"""

import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import ClientError

from cloud.src.prefetch.wrds_prefetcher import WRDSPrefetcher

logger = logging.getLogger(__name__)


class OrchestratorError(Exception):
    """Base exception for orchestrator errors."""
    pass


class PrefetchRequiredError(OrchestratorError):
    """Raised when prefetch data is missing and required for batch runs."""
    pass


class QuarterOrchestrator:
    """
    Coordinate prefetch, batch submission, and monitoring for a quarter.

    This is the main entry point for production batch runs.
    """

    def __init__(
        self,
        s3_bucket: str,
        job_definition: str,
        job_queue: str,
        region: str = "us-east-1",
        s3_client=None,
        batch_client=None,
    ):
        """
        Initialize orchestrator.

        Args:
            s3_bucket: S3 bucket for manifests and output
            job_definition: AWS Batch job definition name
            job_queue: AWS Batch job queue name
            region: AWS region
            s3_client: Optional S3 client (for testing)
            batch_client: Optional Batch client (for testing)
        """
        self.s3_bucket = s3_bucket
        self.job_definition = job_definition
        self.job_queue = job_queue
        self.region = region
        self._s3_client = s3_client or boto3.client("s3", region_name=region)
        self._batch_client = batch_client or boto3.client("batch", region_name=region)

    def _get_manifest_key(self, quarter: str) -> str:
        """Get S3 key for prefetch manifest."""
        return f"prefetch/transcripts/quarter={quarter}/manifest.json"

    def _load_manifest(self, quarter: str) -> Dict[str, Any]:
        """
        Load prefetch manifest from S3.

        Args:
            quarter: Quarter string

        Returns:
            Manifest dict

        Raises:
            PrefetchRequiredError: If manifest doesn't exist
        """
        import gzip

        manifest_key = self._get_manifest_key(quarter)

        try:
            response = self._s3_client.get_object(Bucket=self.s3_bucket, Key=manifest_key)

            # Handle gzip compression
            content_encoding = response.get("ContentEncoding", "")
            body = response["Body"].read()

            if content_encoding == "gzip":
                body = gzip.decompress(body)

            return json.loads(body.decode("utf-8"))

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "NoSuchKey":
                raise PrefetchRequiredError(
                    f"Prefetch manifest not found for quarter {quarter}. "
                    f"Run prefetch first with: orchestrator.run_prefetch('{quarter}')"
                )
            raise

    def prefetch_exists(self, quarter: str) -> bool:
        """
        Check if prefetch manifest exists for a quarter.

        Args:
            quarter: Quarter string (e.g., "2023Q1")

        Returns:
            True if manifest.json exists, False otherwise
        """
        manifest_key = self._get_manifest_key(quarter)

        try:
            self._s3_client.head_object(Bucket=self.s3_bucket, Key=manifest_key)
            logger.info(f"Prefetch manifest found: s3://{self.s3_bucket}/{manifest_key}")
            return True
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "404":
                logger.info(f"Prefetch manifest not found for quarter {quarter}")
                return False
            raise

    def get_prefetch_firm_ids(self, quarter: str) -> List[str]:
        """
        Get firm IDs from prefetch manifest.

        This is O(1) - reads from manifest.json, no data scan needed.

        Args:
            quarter: Quarter string

        Returns:
            Sorted list of firm IDs in prefetch data

        Raises:
            PrefetchRequiredError: If manifest doesn't exist
        """
        manifest = self._load_manifest(quarter)
        firm_to_chunk = manifest.get("firm_to_chunk", {})
        return sorted(firm_to_chunk.keys())

    def get_prefetch_summary(self, quarter: str) -> Dict[str, Any]:
        """
        Get summary info from prefetch manifest.

        Args:
            quarter: Quarter string

        Returns:
            Dict with n_firms, n_chunks, created_at

        Raises:
            PrefetchRequiredError: If manifest doesn't exist
        """
        manifest = self._load_manifest(quarter)
        return {
            "quarter": manifest.get("quarter"),
            "n_firms": manifest.get("n_firms"),
            "n_chunks": manifest.get("n_chunks"),
            "created_at": manifest.get("created_at"),
        }

    def run_prefetch(
        self,
        quarter: str,
        firm_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run WRDS prefetch for a quarter.

        This requires MFA approval and should be run from a fixed-IP machine.

        Args:
            quarter: Quarter string (e.g., "2023Q1")
            firm_ids: Optional list of specific firm IDs. If None, fetches all.

        Returns:
            Prefetch result dict with n_firms, n_chunks, manifest_key
        """
        logger.info(f"Starting prefetch for quarter {quarter}")
        logger.warning(
            "WRDS prefetch requires MFA approval. "
            "Run this from a fixed-IP machine to avoid multiple MFA challenges."
        )

        prefetcher = WRDSPrefetcher(bucket=self.s3_bucket, region=self.region)
        return prefetcher.prefetch_quarter(quarter, firm_ids=firm_ids)

    def _create_batch_manifest(
        self,
        quarter: str,
        firm_ids: List[str],
        batch_size: int,
    ) -> str:
        """
        Create batch manifest (JSONL) on S3.

        Args:
            quarter: Quarter string
            firm_ids: All firm IDs to process
            batch_size: Firms per batch job

        Returns:
            S3 key of the manifest file
        """
        manifest_key = f"manifests/{quarter}/manifest_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl"

        # Split into batches
        batches = []
        for i in range(0, len(firm_ids), batch_size):
            batch_firms = firm_ids[i:i + batch_size]
            batch_id = f"{quarter}_batch_{len(batches):04d}"
            batches.append({
                "batch_id": batch_id,
                "quarter": quarter,
                "firm_ids": batch_firms,
            })

        # Write JSONL
        content = "\n".join(json.dumps(b) for b in batches)
        self._s3_client.put_object(
            Bucket=self.s3_bucket,
            Key=manifest_key,
            Body=content.encode("utf-8"),
            ContentType="application/json",
        )

        logger.info(
            f"Created batch manifest: {len(batches)} batches, "
            f"{len(firm_ids)} firms -> s3://{self.s3_bucket}/{manifest_key}"
        )
        return manifest_key

    def _submit_batch_job(
        self,
        batch_id: str,
        quarter: str,
        manifest_key: str,
    ) -> str:
        """
        Submit a single batch job to AWS Batch.

        Args:
            batch_id: Batch identifier
            quarter: Quarter string
            manifest_key: S3 key of batch manifest

        Returns:
            Job ID from AWS Batch
        """
        response = self._batch_client.submit_job(
            jobName=f"ftm-{batch_id}",
            jobQueue=self.job_queue,
            jobDefinition=self.job_definition,
            containerOverrides={
                "environment": [
                    {"name": "MANIFEST_S3_KEY", "value": manifest_key},
                    {"name": "BATCH_ID", "value": batch_id},
                    {"name": "QUARTER", "value": quarter},
                    {"name": "DATA_SOURCE", "value": "s3"},  # ALWAYS s3 for batch runs
                ],
            },
        )

        job_id = response["jobId"]
        logger.info(f"Submitted job {batch_id}: {job_id}")
        return job_id

    def _wait_for_jobs(
        self,
        job_ids: List[str],
        poll_interval: int = 60,
        timeout: int = 7200,
    ) -> Dict[str, str]:
        """
        Wait for batch jobs to complete.

        Args:
            job_ids: List of AWS Batch job IDs
            poll_interval: Seconds between status checks
            timeout: Maximum wait time in seconds

        Returns:
            Dict mapping job_id to final status
        """
        start_time = time.time()
        pending = set(job_ids)
        results = {}

        logger.info(f"Waiting for {len(job_ids)} jobs to complete...")

        while pending and (time.time() - start_time) < timeout:
            # Check status of pending jobs
            response = self._batch_client.describe_jobs(jobs=list(pending))

            for job in response.get("jobs", []):
                job_id = job["jobId"]
                status = job["status"]

                if status in ("SUCCEEDED", "FAILED"):
                    results[job_id] = status
                    pending.remove(job_id)
                    logger.info(f"Job {job['jobName']} {status}")

            if pending:
                logger.info(f"{len(pending)} jobs still running...")
                time.sleep(poll_interval)

        # Handle timeout
        for job_id in pending:
            results[job_id] = "TIMEOUT"
            logger.warning(f"Job {job_id} timed out")

        return results

    def run_quarter(
        self,
        quarter: str,
        batch_size: int = 1000,
        wait: bool = True,
        poll_interval: int = 60,
        timeout: int = 7200,
    ) -> Dict[str, Any]:
        """
        Run full workflow for a quarter.

        1. Check prefetch manifest exists (FAIL if missing - no WRDS fallback)
        2. Get firm_ids from manifest (cheap, no data scan)
        3. Create batch manifest, submit jobs with DATA_SOURCE=s3
        4. Wait for completion (if wait=True)
        5. Return summary

        IMPORTANT: Prefetch is REQUIRED for Batch runs. No silent fallback to WRDS.

        Args:
            quarter: Quarter string (e.g., "2023Q1")
            batch_size: Firms per batch job (default: 1000)
            wait: Whether to wait for job completion (default: True)
            poll_interval: Seconds between status checks (default: 60)
            timeout: Maximum wait time in seconds (default: 7200 = 2 hours)

        Returns:
            Dict with summary: n_jobs, n_firms, job_ids, results (if wait=True)

        Raises:
            PrefetchRequiredError: If prefetch manifest doesn't exist
        """
        logger.info(f"Starting quarter processing: {quarter}")

        # 1. Check prefetch exists (REQUIRED)
        if not self.prefetch_exists(quarter):
            raise PrefetchRequiredError(
                f"Prefetch manifest not found for quarter {quarter}. "
                f"Batch runs REQUIRE prefetch data to avoid MFA issues. "
                f"Run: orchestrator.run_prefetch('{quarter}')"
            )

        # 2. Get firm_ids from manifest
        firm_ids = self.get_prefetch_firm_ids(quarter)
        logger.info(f"Found {len(firm_ids)} firms in prefetch manifest")

        # 3. Create batch manifest
        manifest_key = self._create_batch_manifest(quarter, firm_ids, batch_size)

        # 4. Submit jobs
        n_batches = (len(firm_ids) + batch_size - 1) // batch_size
        job_ids = []

        for i in range(n_batches):
            batch_id = f"{quarter}_batch_{i:04d}"
            job_id = self._submit_batch_job(batch_id, quarter, manifest_key)
            job_ids.append(job_id)

        logger.info(f"Submitted {len(job_ids)} batch jobs")

        result = {
            "quarter": quarter,
            "n_firms": len(firm_ids),
            "n_jobs": len(job_ids),
            "batch_size": batch_size,
            "manifest_key": manifest_key,
            "job_ids": job_ids,
            "submitted_at": datetime.utcnow().isoformat() + "Z",
        }

        # 5. Wait for completion if requested
        if wait:
            logger.info("Waiting for jobs to complete...")
            job_results = self._wait_for_jobs(job_ids, poll_interval, timeout)

            succeeded = sum(1 for s in job_results.values() if s == "SUCCEEDED")
            failed = sum(1 for s in job_results.values() if s == "FAILED")
            timeout_count = sum(1 for s in job_results.values() if s == "TIMEOUT")

            result["completed_at"] = datetime.utcnow().isoformat() + "Z"
            result["job_results"] = job_results
            result["summary"] = {
                "succeeded": succeeded,
                "failed": failed,
                "timeout": timeout_count,
            }

            if failed or timeout_count:
                logger.warning(
                    f"Quarter {quarter} completed with issues: "
                    f"{succeeded} succeeded, {failed} failed, {timeout_count} timeout"
                )
            else:
                logger.info(f"Quarter {quarter} completed successfully: {succeeded} jobs")

        return result


def main():
    """CLI entrypoint for quarter orchestration."""
    import argparse

    parser = argparse.ArgumentParser(description="Orchestrate quarter batch processing")
    parser.add_argument("--quarter", required=True, help="Quarter to process (e.g., 2023Q1)")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--job-definition", required=True, help="Batch job definition name")
    parser.add_argument("--job-queue", required=True, help="Batch job queue name")
    parser.add_argument("--batch-size", type=int, default=1000, help="Firms per batch job")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--prefetch", action="store_true", help="Run prefetch (requires MFA)")
    parser.add_argument("--no-wait", action="store_true", help="Don't wait for job completion")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create orchestrator
    orchestrator = QuarterOrchestrator(
        s3_bucket=args.bucket,
        job_definition=args.job_definition,
        job_queue=args.job_queue,
        region=args.region,
    )

    # Run prefetch if requested
    if args.prefetch:
        result = orchestrator.run_prefetch(args.quarter)
        print(f"\nPrefetch complete:")
        print(f"  Firms: {result['n_firms']}")
        print(f"  Chunks: {result['n_chunks']}")
        return

    # Check prefetch exists
    if not orchestrator.prefetch_exists(args.quarter):
        print(f"\nError: Prefetch data not found for {args.quarter}")
        print(f"Run with --prefetch first to fetch data from WRDS")
        return

    # Run quarter
    result = orchestrator.run_quarter(
        args.quarter,
        batch_size=args.batch_size,
        wait=not args.no_wait,
    )

    print(f"\nQuarter {args.quarter} processing:")
    print(f"  Firms: {result['n_firms']}")
    print(f"  Jobs: {result['n_jobs']}")
    print(f"  Batch size: {result['batch_size']}")

    if "summary" in result:
        print(f"\nResults:")
        print(f"  Succeeded: {result['summary']['succeeded']}")
        print(f"  Failed: {result['summary']['failed']}")
        print(f"  Timeout: {result['summary']['timeout']}")


if __name__ == "__main__":
    main()
