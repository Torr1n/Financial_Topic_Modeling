"""
Unit tests for QuarterOrchestrator.

Tests orchestration logic:
- prefetch_exists() checks for manifest.json
- get_prefetch_firm_ids() reads from manifest
- Job submission with DATA_SOURCE=s3
- Error handling for missing prefetch
"""

import gzip
import json
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from cloud.src.orchestrate.quarter_orchestrator import (
    OrchestratorError,
    PrefetchRequiredError,
    QuarterOrchestrator,
)


def create_mock_manifest(n_firms=100, n_chunks=5):
    """Create a mock manifest for testing."""
    firm_to_chunk = {}
    for i in range(n_firms):
        firm_id = f"firm_{i:04d}"
        chunk_idx = i // (n_firms // n_chunks)
        firm_to_chunk[firm_id] = f"chunk_{chunk_idx:04d}.parquet"

    return {
        "quarter": "2023Q1",
        "created_at": "2026-01-28T12:00:00Z",
        "n_firms": n_firms,
        "n_chunks": n_chunks,
        "chunk_sizes": {f"chunk_{i:04d}.parquet": n_firms // n_chunks for i in range(n_chunks)},
        "firm_to_chunk": firm_to_chunk,
    }


class TestPrefetchExists:
    """Tests for prefetch_exists method."""

    def test_returns_true_when_manifest_exists(self):
        """Should return True when manifest.json exists."""
        mock_s3 = MagicMock()
        mock_s3.head_object.return_value = {}  # Success

        orchestrator = QuarterOrchestrator(
            s3_bucket="test-bucket",
            job_definition="test-job",
            job_queue="test-queue",
            s3_client=mock_s3,
        )

        assert orchestrator.prefetch_exists("2023Q1") is True
        mock_s3.head_object.assert_called_once()

    def test_returns_false_when_manifest_missing(self):
        """Should return False when manifest.json doesn't exist."""
        from botocore.exceptions import ClientError

        mock_s3 = MagicMock()
        mock_s3.head_object.side_effect = ClientError(
            {"Error": {"Code": "404"}}, "HeadObject"
        )

        orchestrator = QuarterOrchestrator(
            s3_bucket="test-bucket",
            job_definition="test-job",
            job_queue="test-queue",
            s3_client=mock_s3,
        )

        assert orchestrator.prefetch_exists("2023Q1") is False


class TestGetPrefetchFirmIds:
    """Tests for get_prefetch_firm_ids method."""

    def test_returns_sorted_firm_ids_from_manifest(self):
        """Should return sorted firm_ids from manifest."""
        manifest = create_mock_manifest(n_firms=50, n_chunks=2)
        compressed = gzip.compress(json.dumps(manifest).encode())

        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {
            "Body": BytesIO(compressed),
            "ContentEncoding": "gzip",
        }

        orchestrator = QuarterOrchestrator(
            s3_bucket="test-bucket",
            job_definition="test-job",
            job_queue="test-queue",
            s3_client=mock_s3,
        )

        firm_ids = orchestrator.get_prefetch_firm_ids("2023Q1")

        assert len(firm_ids) == 50
        assert firm_ids == sorted(firm_ids)
        assert "firm_0000" in firm_ids

    def test_raises_when_manifest_missing(self):
        """Should raise PrefetchRequiredError when manifest missing."""
        from botocore.exceptions import ClientError

        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey"}}, "GetObject"
        )

        orchestrator = QuarterOrchestrator(
            s3_bucket="test-bucket",
            job_definition="test-job",
            job_queue="test-queue",
            s3_client=mock_s3,
        )

        with pytest.raises(PrefetchRequiredError) as exc_info:
            orchestrator.get_prefetch_firm_ids("2023Q1")

        assert "2023Q1" in str(exc_info.value)


class TestGetPrefetchSummary:
    """Tests for get_prefetch_summary method."""

    def test_returns_summary_from_manifest(self):
        """Should return summary info from manifest."""
        manifest = create_mock_manifest(n_firms=100, n_chunks=5)
        compressed = gzip.compress(json.dumps(manifest).encode())

        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {
            "Body": BytesIO(compressed),
            "ContentEncoding": "gzip",
        }

        orchestrator = QuarterOrchestrator(
            s3_bucket="test-bucket",
            job_definition="test-job",
            job_queue="test-queue",
            s3_client=mock_s3,
        )

        summary = orchestrator.get_prefetch_summary("2023Q1")

        assert summary["quarter"] == "2023Q1"
        assert summary["n_firms"] == 100
        assert summary["n_chunks"] == 5


class TestCreateBatchManifest:
    """Tests for _create_batch_manifest method."""

    def test_creates_jsonl_manifest(self):
        """Should create JSONL manifest with batches."""
        uploaded_content = None

        def capture_put(Bucket, Key, Body, **kwargs):
            nonlocal uploaded_content
            uploaded_content = Body

        mock_s3 = MagicMock()
        mock_s3.put_object = capture_put

        orchestrator = QuarterOrchestrator(
            s3_bucket="test-bucket",
            job_definition="test-job",
            job_queue="test-queue",
            s3_client=mock_s3,
        )

        firm_ids = [f"firm_{i:04d}" for i in range(250)]
        manifest_key = orchestrator._create_batch_manifest("2023Q1", firm_ids, batch_size=100)

        assert manifest_key.startswith("manifests/2023Q1/manifest_")
        assert manifest_key.endswith(".jsonl")

        # Parse JSONL
        lines = uploaded_content.decode().strip().split("\n")
        assert len(lines) == 3  # 250 firms / 100 batch_size = 3 batches

        batch_0 = json.loads(lines[0])
        assert batch_0["batch_id"] == "2023Q1_batch_0000"
        assert batch_0["quarter"] == "2023Q1"
        assert len(batch_0["firm_ids"]) == 100

        batch_2 = json.loads(lines[2])
        assert len(batch_2["firm_ids"]) == 50  # Remaining firms


class TestSubmitBatchJob:
    """Tests for _submit_batch_job method."""

    def test_submits_with_data_source_s3(self):
        """Should submit job with DATA_SOURCE=s3."""
        mock_batch = MagicMock()
        mock_batch.submit_job.return_value = {"jobId": "job-12345"}

        orchestrator = QuarterOrchestrator(
            s3_bucket="test-bucket",
            job_definition="test-job-def",
            job_queue="test-queue",
            batch_client=mock_batch,
        )

        job_id = orchestrator._submit_batch_job(
            batch_id="2023Q1_batch_0000",
            quarter="2023Q1",
            manifest_key="manifests/2023Q1/manifest.jsonl",
        )

        assert job_id == "job-12345"

        # Verify submit_job call
        call_kwargs = mock_batch.submit_job.call_args[1]
        assert call_kwargs["jobQueue"] == "test-queue"
        assert call_kwargs["jobDefinition"] == "test-job-def"

        # Verify DATA_SOURCE=s3 in environment
        env_vars = {e["name"]: e["value"] for e in call_kwargs["containerOverrides"]["environment"]}
        assert env_vars["DATA_SOURCE"] == "s3"
        assert env_vars["BATCH_ID"] == "2023Q1_batch_0000"
        assert env_vars["QUARTER"] == "2023Q1"


class TestRunQuarter:
    """Tests for run_quarter method."""

    def test_fails_when_prefetch_missing(self):
        """Should raise PrefetchRequiredError when prefetch missing."""
        from botocore.exceptions import ClientError

        mock_s3 = MagicMock()
        mock_s3.head_object.side_effect = ClientError(
            {"Error": {"Code": "404"}}, "HeadObject"
        )

        orchestrator = QuarterOrchestrator(
            s3_bucket="test-bucket",
            job_definition="test-job",
            job_queue="test-queue",
            s3_client=mock_s3,
        )

        with pytest.raises(PrefetchRequiredError) as exc_info:
            orchestrator.run_quarter("2023Q1")

        assert "REQUIRE prefetch data" in str(exc_info.value)

    def test_submits_jobs_with_correct_count(self):
        """Should submit correct number of batch jobs."""
        manifest = create_mock_manifest(n_firms=250, n_chunks=3)
        compressed = gzip.compress(json.dumps(manifest).encode())

        mock_s3 = MagicMock()
        mock_s3.head_object.return_value = {}  # Prefetch exists
        mock_s3.get_object.return_value = {
            "Body": BytesIO(compressed),
            "ContentEncoding": "gzip",
        }

        mock_batch = MagicMock()
        job_counter = [0]

        def mock_submit(**kwargs):
            job_counter[0] += 1
            return {"jobId": f"job-{job_counter[0]}"}

        mock_batch.submit_job = mock_submit

        orchestrator = QuarterOrchestrator(
            s3_bucket="test-bucket",
            job_definition="test-job",
            job_queue="test-queue",
            s3_client=mock_s3,
            batch_client=mock_batch,
        )

        result = orchestrator.run_quarter("2023Q1", batch_size=100, wait=False)

        # 250 firms / 100 batch_size = 3 jobs
        assert result["n_jobs"] == 3
        assert len(result["job_ids"]) == 3
        assert result["n_firms"] == 250

    def test_returns_summary_after_wait(self):
        """Should return job results when wait=True."""
        manifest = create_mock_manifest(n_firms=50, n_chunks=1)
        compressed = gzip.compress(json.dumps(manifest).encode())

        mock_s3 = MagicMock()
        mock_s3.head_object.return_value = {}
        mock_s3.get_object.return_value = {
            "Body": BytesIO(compressed),
            "ContentEncoding": "gzip",
        }

        mock_batch = MagicMock()
        mock_batch.submit_job.return_value = {"jobId": "job-001"}
        mock_batch.describe_jobs.return_value = {
            "jobs": [{"jobId": "job-001", "jobName": "test", "status": "SUCCEEDED"}]
        }

        orchestrator = QuarterOrchestrator(
            s3_bucket="test-bucket",
            job_definition="test-job",
            job_queue="test-queue",
            s3_client=mock_s3,
            batch_client=mock_batch,
        )

        result = orchestrator.run_quarter("2023Q1", batch_size=100, wait=True)

        assert "summary" in result
        assert result["summary"]["succeeded"] == 1
        assert result["summary"]["failed"] == 0


class TestNoWrdsFallback:
    """Tests that confirm no silent WRDS fallback."""

    def test_run_quarter_never_calls_wrds(self):
        """run_quarter should never instantiate WRDSConnector."""
        manifest = create_mock_manifest(n_firms=10, n_chunks=1)
        compressed = gzip.compress(json.dumps(manifest).encode())

        mock_s3 = MagicMock()
        mock_s3.head_object.return_value = {}
        mock_s3.get_object.return_value = {
            "Body": BytesIO(compressed),
            "ContentEncoding": "gzip",
        }

        mock_batch = MagicMock()
        mock_batch.submit_job.return_value = {"jobId": "job-001"}
        mock_batch.describe_jobs.return_value = {
            "jobs": [{"jobId": "job-001", "jobName": "test", "status": "SUCCEEDED"}]
        }

        with patch("cloud.src.connectors.wrds_connector.WRDSConnector") as mock_wrds:
            orchestrator = QuarterOrchestrator(
                s3_bucket="test-bucket",
                job_definition="test-job",
                job_queue="test-queue",
                s3_client=mock_s3,
                batch_client=mock_batch,
            )

            orchestrator.run_quarter("2023Q1", batch_size=100, wait=True)

            # WRDSConnector should never be instantiated
            mock_wrds.assert_not_called()


class TestRunPrefetch:
    """Tests for run_prefetch method."""

    def test_run_prefetch_calls_wrds_prefetcher(self):
        """run_prefetch should use WRDSPrefetcher."""
        with patch("cloud.src.orchestrate.quarter_orchestrator.WRDSPrefetcher") as mock_prefetcher_cls:
            mock_prefetcher = MagicMock()
            mock_prefetcher.prefetch_quarter.return_value = {
                "n_firms": 100,
                "n_chunks": 5,
                "manifest_key": "prefetch/transcripts/quarter=2023Q1/manifest.json",
            }
            mock_prefetcher_cls.return_value = mock_prefetcher

            orchestrator = QuarterOrchestrator(
                s3_bucket="test-bucket",
                job_definition="test-job",
                job_queue="test-queue",
            )

            result = orchestrator.run_prefetch("2023Q1")

            mock_prefetcher_cls.assert_called_once()
            mock_prefetcher.prefetch_quarter.assert_called_once_with("2023Q1", firm_ids=None)
            assert result["n_firms"] == 100


class TestDescribeJobsChunking:
    """Tests for describe_jobs AWS limit handling."""

    def test_describe_jobs_chunks_large_lists(self):
        """Should chunk describe_jobs calls when >100 jobs."""
        mock_batch = MagicMock()

        # Track all describe_jobs calls
        describe_calls = []

        def mock_describe(jobs):
            describe_calls.append(jobs)
            # Return all as SUCCEEDED immediately
            return {
                "jobs": [
                    {"jobId": jid, "jobName": f"job-{jid}", "status": "SUCCEEDED"}
                    for jid in jobs
                ]
            }

        mock_batch.describe_jobs = mock_describe

        orchestrator = QuarterOrchestrator(
            s3_bucket="test-bucket",
            job_definition="test-job",
            job_queue="test-queue",
            batch_client=mock_batch,
        )

        # Create 250 job IDs
        job_ids = [f"job-{i:04d}" for i in range(250)]

        results = orchestrator._wait_for_jobs(job_ids, poll_interval=0, timeout=10)

        # Should have made 3 describe_jobs calls (100, 100, 50)
        assert len(describe_calls) == 3
        assert len(describe_calls[0]) == 100
        assert len(describe_calls[1]) == 100
        assert len(describe_calls[2]) == 50

        # All jobs should be marked as SUCCEEDED
        assert len(results) == 250
        assert all(status == "SUCCEEDED" for status in results.values())

    def test_describe_jobs_works_with_small_lists(self):
        """Should handle <100 jobs without issues."""
        mock_batch = MagicMock()
        mock_batch.describe_jobs.return_value = {
            "jobs": [
                {"jobId": "job-001", "jobName": "test", "status": "SUCCEEDED"},
                {"jobId": "job-002", "jobName": "test", "status": "SUCCEEDED"},
            ]
        }

        orchestrator = QuarterOrchestrator(
            s3_bucket="test-bucket",
            job_definition="test-job",
            job_queue="test-queue",
            batch_client=mock_batch,
        )

        results = orchestrator._wait_for_jobs(
            ["job-001", "job-002"],
            poll_interval=0,
            timeout=10,
        )

        assert len(results) == 2
        mock_batch.describe_jobs.assert_called_once()
