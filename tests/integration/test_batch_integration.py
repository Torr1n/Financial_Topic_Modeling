"""
AWS Batch Integration Tests

This module contains tests for the Batch job submission infrastructure.

Test Categories:
1. Unit tests (local, no AWS) - Test manifest parsing, checkpoint logic, etc.
2. Integration tests (@pytest.mark.integration) - Real AWS, opt-in

Run unit tests:
    pytest tests/integration/test_batch_integration.py -v -k "not integration"

Run integration tests (requires AWS credentials):
    pytest tests/integration/test_batch_integration.py -v -m integration
"""

import json
import os
import tempfile
from datetime import date
from unittest.mock import MagicMock, patch

from botocore.exceptions import ClientError
import pandas as pd
import pytest


# =============================================================================
# Unit Tests (Local, No AWS)
# =============================================================================


class TestManifestCreation:
    """Test manifest JSONL format and batch splitting."""

    def test_manifest_format_single_batch(self):
        """Test manifest with fewer firms than batch size."""
        from cloud.src.batch.job_submitter import BatchJobSubmitter

        # Create a mock S3 client that captures the upload
        mock_s3 = MagicMock()
        captured_content = {}

        def capture_put(Bucket, Key, Body, ContentType):
            captured_content["body"] = Body
            captured_content["key"] = Key

        mock_s3.put_object.side_effect = capture_put

        # Initialize submitter with mock
        submitter = BatchJobSubmitter(
            job_definition="test-job",
            job_queue="test-queue",
            s3_bucket="test-bucket",
        )
        submitter.s3_client = mock_s3

        # Create manifest
        firm_ids = ["firm_1", "firm_2", "firm_3"]
        manifest_key, n_batches = submitter.create_manifest(
            quarter="2023Q1", firm_ids=firm_ids, batch_size=1000
        )

        # Verify
        assert n_batches == 1
        assert manifest_key == "manifests/quarter=2023Q1/batches.jsonl"

        # Parse captured content
        content = captured_content["body"].decode("utf-8")
        batches = [json.loads(line) for line in content.strip().split("\n")]

        assert len(batches) == 1
        assert batches[0]["batch_id"] == "batch_000"
        assert batches[0]["quarter"] == "2023Q1"
        assert batches[0]["firm_ids"] == firm_ids

    def test_manifest_format_multiple_batches(self):
        """Test manifest splits firms into correct batch sizes."""
        from cloud.src.batch.job_submitter import BatchJobSubmitter

        mock_s3 = MagicMock()
        captured_content = {}

        def capture_put(Bucket, Key, Body, ContentType):
            captured_content["body"] = Body

        mock_s3.put_object.side_effect = capture_put

        submitter = BatchJobSubmitter(
            job_definition="test-job",
            job_queue="test-queue",
            s3_bucket="test-bucket",
        )
        submitter.s3_client = mock_s3

        # 25 firms with batch_size=10 -> 3 batches
        firm_ids = [f"firm_{i}" for i in range(25)]
        manifest_key, n_batches = submitter.create_manifest(
            quarter="2023Q2", firm_ids=firm_ids, batch_size=10
        )

        assert n_batches == 3

        content = captured_content["body"].decode("utf-8")
        batches = [json.loads(line) for line in content.strip().split("\n")]

        assert len(batches) == 3
        assert batches[0]["batch_id"] == "batch_000"
        assert len(batches[0]["firm_ids"]) == 10
        assert batches[1]["batch_id"] == "batch_001"
        assert len(batches[1]["firm_ids"]) == 10
        assert batches[2]["batch_id"] == "batch_002"
        assert len(batches[2]["firm_ids"]) == 5


class TestCheckpointLogic:
    """Test checkpoint save/load round-trip consistency."""

    def test_checkpoint_round_trip(self):
        """Test that checkpoint data survives save/load cycle."""
        from cloud.containers.map.entrypoint import load_checkpoint, save_checkpoint

        # Mock S3 client with in-memory storage
        storage = {}

        def mock_put(Bucket, Key, Body, ContentType):
            storage[Key] = Body

        def mock_get(Bucket, Key):
            if Key not in storage:
                raise Exception("NoSuchKey")
            return {"Body": MagicMock(read=lambda: storage[Key])}

        mock_s3 = MagicMock()
        mock_s3.put_object.side_effect = mock_put
        mock_s3.get_object.side_effect = mock_get
        mock_s3.exceptions = MagicMock()
        mock_s3.exceptions.NoSuchKey = Exception

        # Save checkpoint
        completed = {"firm_1", "firm_2", "firm_3"}
        save_checkpoint(
            s3_client=mock_s3,
            bucket="test-bucket",
            batch_id="batch_000",
            quarter="2023Q1",
            completed_firm_ids=completed,
            chunk_id=2,
        )

        # Verify checkpoint was saved
        checkpoint_key = "progress/2023Q1/batch_000_checkpoint.json"
        assert checkpoint_key in storage

        # Load it back (now returns tuple)
        loaded_completed, loaded_chunk_id = load_checkpoint(
            s3_client=mock_s3,
            bucket="test-bucket",
            batch_id="batch_000",
            quarter="2023Q1",
        )

        assert loaded_completed == completed
        assert loaded_chunk_id == 2

    def test_checkpoint_empty_when_missing(self):
        """Test that missing checkpoint returns empty set and chunk_id 0."""
        from cloud.containers.map.entrypoint import load_checkpoint

        # Mock S3 that raises NoSuchKey
        mock_s3 = MagicMock()

        class NoSuchKeyError(Exception):
            pass

        mock_s3.get_object.side_effect = NoSuchKeyError()
        mock_s3.exceptions.NoSuchKey = NoSuchKeyError

        loaded_completed, loaded_chunk_id = load_checkpoint(
            s3_client=mock_s3,
            bucket="test-bucket",
            batch_id="batch_000",
            quarter="2023Q1",
        )

        assert loaded_completed == set()
        assert loaded_chunk_id == 0

    def test_checkpoint_handles_boto3_client_error(self):
        """Test that ClientError with NoSuchKey is handled correctly."""
        from cloud.containers.map.entrypoint import load_checkpoint

        mock_s3 = MagicMock()

        # Simulate real boto3 ClientError
        error_response = {"Error": {"Code": "NoSuchKey", "Message": "Not found"}}
        mock_s3.get_object.side_effect = ClientError(error_response, "GetObject")

        loaded_completed, loaded_chunk_id = load_checkpoint(
            s3_client=mock_s3,
            bucket="test-bucket",
            batch_id="batch_000",
            quarter="2023Q1",
        )

        assert loaded_completed == set()
        assert loaded_chunk_id == 0

    def test_checkpoint_raises_on_other_errors(self):
        """Test that non-NoSuchKey ClientErrors are re-raised."""
        from cloud.containers.map.entrypoint import load_checkpoint

        mock_s3 = MagicMock()

        # Simulate AccessDenied error
        error_response = {"Error": {"Code": "AccessDenied", "Message": "Forbidden"}}
        mock_s3.get_object.side_effect = ClientError(error_response, "GetObject")

        with pytest.raises(ClientError):
            load_checkpoint(
                s3_client=mock_s3,
                bucket="test-bucket",
                batch_id="batch_000",
                quarter="2023Q1",
            )


class TestCircuitBreaker:
    """Test circuit breaker configuration and critical error detection."""

    def test_circuit_breaker_config_defaults(self):
        """Test default circuit breaker values."""
        from cloud.containers.map.entrypoint import CircuitBreakerConfig

        config = CircuitBreakerConfig()
        assert config.max_consecutive_failures == 5
        assert config.max_failure_rate == 0.05
        assert config.min_processed_for_rate == 100

    def test_circuit_breaker_config_from_env(self, monkeypatch):
        """Test loading circuit breaker config from environment."""
        from cloud.containers.map.entrypoint import CircuitBreakerConfig

        monkeypatch.setenv("MAX_CONSECUTIVE_FAILURES", "10")
        monkeypatch.setenv("MAX_FAILURE_RATE", "0.1")
        monkeypatch.setenv("MIN_PROCESSED_FOR_RATE", "50")

        config = CircuitBreakerConfig.from_env()
        assert config.max_consecutive_failures == 10
        assert config.max_failure_rate == 0.1
        assert config.min_processed_for_rate == 50

    def test_is_critical_error_wrds(self):
        """Test that WRDS errors are detected as critical."""
        from cloud.containers.map.entrypoint import is_critical_error

        # WRDS connection error
        class WRDSConnectionError(Exception):
            pass

        assert is_critical_error(WRDSConnectionError("WRDS connection failed"))

    def test_is_critical_error_cuda(self):
        """Test that CUDA errors are detected as critical."""
        from cloud.containers.map.entrypoint import is_critical_error

        class CudaError(Exception):
            pass

        assert is_critical_error(CudaError("CUDA out of memory"))
        assert is_critical_error(CudaError("CUDA error: device-side assert"))

    def test_is_critical_error_auth(self):
        """Test that auth errors are detected as critical."""
        from cloud.containers.map.entrypoint import is_critical_error

        class AuthError(Exception):
            pass

        assert is_critical_error(AuthError("AccessDenied"))
        assert is_critical_error(AuthError("PermissionDenied"))

    def test_is_not_critical_error(self):
        """Test that normal processing errors are not critical."""
        from cloud.containers.map.entrypoint import is_critical_error

        # Normal processing error (e.g., bad data)
        assert not is_critical_error(ValueError("Invalid data format"))
        assert not is_critical_error(KeyError("missing_key"))
        assert not is_critical_error(IndexError("list index out of range"))


class TestQuarterToDateRange:
    """Test quarter string to date range conversion."""

    def test_q1_dates(self):
        """Test Q1 date range."""
        from cloud.containers.map.entrypoint import quarter_to_date_range

        start, end = quarter_to_date_range("2023Q1")
        assert start == "2023-01-01"
        assert end == "2023-03-31"

    def test_q2_dates(self):
        """Test Q2 date range."""
        from cloud.containers.map.entrypoint import quarter_to_date_range

        start, end = quarter_to_date_range("2023Q2")
        assert start == "2023-04-01"
        assert end == "2023-06-30"

    def test_q3_dates(self):
        """Test Q3 date range."""
        from cloud.containers.map.entrypoint import quarter_to_date_range

        start, end = quarter_to_date_range("2023Q3")
        assert start == "2023-07-01"
        assert end == "2023-09-30"

    def test_q4_dates(self):
        """Test Q4 date range."""
        from cloud.containers.map.entrypoint import quarter_to_date_range

        start, end = quarter_to_date_range("2023Q4")
        assert start == "2023-10-01"
        assert end == "2023-12-31"

    def test_different_year(self):
        """Test with different year."""
        from cloud.containers.map.entrypoint import quarter_to_date_range

        start, end = quarter_to_date_range("2025Q3")
        assert start == "2025-07-01"
        assert end == "2025-09-30"


class TestParquetSchema:
    """Test Parquet output schema matches FirmProcessor output."""

    def test_flatten_to_parquet_rows(self):
        """Test conversion from FirmProcessor output to Parquet rows."""
        from cloud.containers.map.entrypoint import flatten_to_parquet_rows

        firm_output = {
            "firm_id": "123456",
            "firm_name": "Test Corp.",
            "n_topics": 2,
            "topics": [
                {
                    "topic_id": 0,
                    "representation": "ai machine learning",
                    "keywords": ["ai", "machine", "learning"],
                    "size": 10,
                    "sentence_ids": ["123456_T1_0001", "123456_T1_0002"],
                },
                {
                    "topic_id": 1,
                    "representation": "revenue growth",
                    "keywords": ["revenue", "growth"],
                    "size": 8,
                    "sentence_ids": ["123456_T1_0003"],
                },
            ],
            "outlier_sentence_ids": [],
            "metadata": {"processing_timestamp": "2024-01-15T12:00:00Z"},
        }

        firm_metadata = {
            "permno": 12345,
            "gvkey": "001234",
            "earnings_call_date": date(2024, 1, 10),
        }

        rows = flatten_to_parquet_rows(firm_output, firm_metadata, quarter="2024Q1")

        # Should have one row per topic
        assert len(rows) == 2

        # Verify first row
        row0 = rows[0]
        assert row0["firm_id"] == "123456"
        assert row0["firm_name"] == "Test Corp."
        assert row0["quarter"] == "2024Q1"  # New field
        assert row0["permno"] == 12345
        assert row0["gvkey"] == "001234"
        assert row0["earnings_call_date"] == date(2024, 1, 10)
        assert row0["topic_id"] == 0
        assert row0["representation"] == "ai machine learning"
        assert row0["keywords"] == ["ai", "machine", "learning"]
        assert row0["n_sentences"] == 10
        assert row0["sentence_ids"] == ["123456_T1_0001", "123456_T1_0002"]
        assert row0["processing_timestamp"] == "2024-01-15T12:00:00Z"

        # Verify second row
        row1 = rows[1]
        assert row1["topic_id"] == 1
        assert row1["representation"] == "revenue growth"
        assert row1["quarter"] == "2024Q1"

    def test_parquet_write_schema(self):
        """Test that Parquet write produces correct schema."""
        from cloud.containers.map.entrypoint import flatten_to_parquet_rows

        firm_output = {
            "firm_id": "123456",
            "firm_name": "Test Corp.",
            "n_topics": 1,
            "topics": [
                {
                    "topic_id": 0,
                    "representation": "test topic",
                    "keywords": ["test"],
                    "size": 5,
                    "sentence_ids": ["123456_T1_0001"],
                }
            ],
            "outlier_sentence_ids": [],
            "metadata": {"processing_timestamp": "2024-01-15T12:00:00Z"},
        }

        firm_metadata = {
            "permno": 12345,
            "gvkey": "001234",
            "earnings_call_date": date(2024, 1, 10),
        }

        rows = flatten_to_parquet_rows(firm_output, firm_metadata, quarter="2024Q1")
        df = pd.DataFrame(rows)

        # Convert date column
        if "earnings_call_date" in df.columns:
            df["earnings_call_date"] = pd.to_datetime(df["earnings_call_date"]).dt.date

        # Write to temp file and read back
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            df.to_parquet(tmp.name, index=False, engine="pyarrow")

            # Read back and verify schema
            df_read = pd.read_parquet(tmp.name)

            expected_columns = {
                "firm_id",
                "firm_name",
                "quarter",  # New field
                "permno",
                "gvkey",
                "earnings_call_date",
                "topic_id",
                "representation",
                "keywords",
                "n_sentences",
                "sentence_ids",
                "processing_timestamp",
            }

            assert set(df_read.columns) == expected_columns

            # Cleanup
            os.unlink(tmp.name)


class TestJobSubmitter:
    """Test job submitter methods."""

    def test_submit_job_calls_batch_api(self):
        """Test that submit_job makes correct API call."""
        from cloud.src.batch.job_submitter import BatchJobSubmitter

        mock_batch = MagicMock()
        mock_batch.submit_job.return_value = {"jobId": "test-job-id-123"}

        submitter = BatchJobSubmitter(
            job_definition="ftm-firm-processor",
            job_queue="ftm-queue-main",
            s3_bucket="test-bucket",
        )
        submitter.batch_client = mock_batch

        result = submitter.submit_job(
            quarter="2023Q1",
            batch_id="batch_000",
            manifest_key="manifests/quarter=2023Q1/batches.jsonl",
            n_firms=100,
        )

        # Verify API was called correctly
        mock_batch.submit_job.assert_called_once()
        call_kwargs = mock_batch.submit_job.call_args.kwargs

        assert call_kwargs["jobName"] == "ftm-2023Q1-batch_000"
        assert call_kwargs["jobQueue"] == "ftm-queue-main"
        assert call_kwargs["jobDefinition"] == "ftm-firm-processor"

        # Verify environment overrides
        env_vars = call_kwargs["containerOverrides"]["environment"]
        env_dict = {e["name"]: e["value"] for e in env_vars}

        assert env_dict["MANIFEST_S3_KEY"] == "manifests/quarter=2023Q1/batches.jsonl"
        assert env_dict["BATCH_ID"] == "batch_000"
        assert env_dict["QUARTER"] == "2023Q1"

        # Verify result
        assert result.job_id == "test-job-id-123"
        assert result.batch_id == "batch_000"
        assert result.quarter == "2023Q1"
        assert result.n_firms == 100

    def test_get_job_status_handles_pagination(self):
        """Test that get_job_status handles > 100 jobs."""
        from cloud.src.batch.job_submitter import BatchJobSubmitter

        mock_batch = MagicMock()

        # Simulate response
        def describe_jobs(jobs):
            return {
                "jobs": [
                    {"jobId": j, "status": "RUNNING", "jobName": f"job-{j}"}
                    for j in jobs
                ]
            }

        mock_batch.describe_jobs.side_effect = describe_jobs

        submitter = BatchJobSubmitter(
            job_definition="test-job",
            job_queue="test-queue",
            s3_bucket="test-bucket",
        )
        submitter.batch_client = mock_batch

        # Request status for 150 jobs (should make 2 API calls)
        job_ids = [f"job-{i}" for i in range(150)]
        status = submitter.get_job_status(job_ids)

        assert len(status) == 150
        assert mock_batch.describe_jobs.call_count == 2

    def test_submit_quarter_rejects_empty_firms(self):
        """Test that submit_quarter raises error for empty firm list."""
        from cloud.src.batch.job_submitter import BatchJobSubmitter

        submitter = BatchJobSubmitter(
            job_definition="test-job",
            job_queue="test-queue",
            s3_bucket="test-bucket",
        )

        with pytest.raises(ValueError, match="firm_ids must be provided"):
            submitter.submit_quarter("2023Q1", firm_ids=[])

        with pytest.raises(ValueError, match="firm_ids must be provided"):
            submitter.submit_quarter("2023Q1", firm_ids=None)


class TestJobSubmissionResult:
    """Test JobSubmissionResult dataclass."""

    def test_result_fields(self):
        """Test that all required fields are present."""
        from cloud.src.batch.job_submitter import JobSubmissionResult

        result = JobSubmissionResult(
            job_id="test-id",
            job_name="test-name",
            batch_id="batch_000",
            quarter="2023Q1",
            n_firms=100,
        )

        assert result.job_id == "test-id"
        assert result.job_name == "test-name"
        assert result.batch_id == "batch_000"
        assert result.quarter == "2023Q1"
        assert result.n_firms == 100
        assert result.submitted_at is not None  # Auto-generated


# =============================================================================
# Integration Tests (Requires AWS)
# =============================================================================


@pytest.mark.integration
class TestBatchIntegration:
    """
    Integration tests that require AWS credentials and running infrastructure.

    These tests are opt-in and require:
    1. AWS credentials configured (via env vars or ~/.aws/credentials)
    2. Batch infrastructure deployed (via terraform apply)
    3. WRDS credentials in Secrets Manager

    Run with: pytest tests/integration/test_batch_integration.py -v -m integration
    """

    @pytest.fixture
    def batch_config(self):
        """Load Batch configuration from terraform output."""
        import subprocess

        # Get terraform outputs
        result = subprocess.run(
            ["terraform", "output", "-json"],
            cwd="cloud/terraform/batch",
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            pytest.skip("Terraform outputs not available - infrastructure not deployed")

        outputs = json.loads(result.stdout)

        return {
            "job_definition": outputs["job_definition_name"]["value"],
            "job_queue": outputs["job_queue_name"]["value"],
            "s3_bucket": outputs["s3_bucket_name"]["value"],
            "ecr_url": outputs["ecr_repository_url"]["value"],
        }

    def test_submit_job_to_batch(self, batch_config):
        """Test real job submission to AWS Batch."""
        from cloud.src.batch.job_submitter import BatchJobSubmitter

        submitter = BatchJobSubmitter(
            job_definition=batch_config["job_definition"],
            job_queue=batch_config["job_queue"],
            s3_bucket=batch_config["s3_bucket"],
        )

        # Create a small test manifest
        firm_ids = ["374372246"]  # Lamb Weston - known to exist in WRDS
        manifest_key, n_batches = submitter.create_manifest(
            quarter="2023Q1", firm_ids=firm_ids, batch_size=10
        )

        assert n_batches == 1
        assert manifest_key == "manifests/quarter=2023Q1/batches.jsonl"

        # Submit job
        result = submitter.submit_job(
            quarter="2023Q1",
            batch_id="batch_000",
            manifest_key=manifest_key,
            n_firms=len(firm_ids),
        )

        assert result.job_id is not None
        assert result.job_name == "ftm-2023Q1-batch_000"

        # Check job status
        status = submitter.get_job_status([result.job_id])
        assert result.job_id in status
        assert status[result.job_id]["status"] in {
            "SUBMITTED",
            "PENDING",
            "RUNNABLE",
            "STARTING",
            "RUNNING",
        }

    @pytest.mark.slow
    def test_process_10_firms_via_batch(self, batch_config):
        """
        End-to-end test: submit job, wait for completion, verify S3 output.

        This test is marked slow because it waits for a real Batch job to complete.
        """
        from cloud.src.batch.job_submitter import BatchJobSubmitter

        submitter = BatchJobSubmitter(
            job_definition=batch_config["job_definition"],
            job_queue=batch_config["job_queue"],
            s3_bucket=batch_config["s3_bucket"],
        )

        # Real Capital IQ company IDs from WRDS
        firm_ids = [
            "18749",     # Amazon.com, Inc.
            "19691",     # Cisco Systems, Inc.
            "21835",     # Microsoft Corporation
            "24937",     # Apple Inc.
            "29096",     # Alphabet Inc.
            "32307",     # NVIDIA Corporation
            "20765463",  # Meta Platforms, Inc.
            "25016048",  # Broadcom Inc.
            "27444752",  # Tesla, Inc.
            "33348547",  # Arista Networks Inc
        ]

        # Submit quarter
        results = submitter.submit_quarter(
            quarter="2023Q1", firm_ids=firm_ids, batch_size=10
        )

        assert len(results) == 1  # All firms fit in one batch

        # Wait for completion (timeout after 30 minutes)
        job_ids = [r.job_id for r in results]
        final_status = submitter.wait_for_completion(
            job_ids, poll_interval=30, timeout=1800
        )

        # Verify at least one succeeded
        succeeded = sum(1 for s in final_status.values() if s == "SUCCEEDED")
        assert succeeded > 0, f"No jobs succeeded: {final_status}"

        # Verify S3 output
        output_files = submitter.list_quarter_output("2023Q1")
        assert len(output_files) > 0, "No output files found"

        # Read and verify Parquet schema
        import boto3

        s3 = boto3.client("s3")
        response = s3.get_object(
            Bucket=batch_config["s3_bucket"], Key=output_files[0]
        )

        with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
            tmp.write(response["Body"].read())
            tmp.flush()

            df = pd.read_parquet(tmp.name)

            # Verify required columns exist
            required_cols = {
                "firm_id",
                "firm_name",
                "quarter",
                "permno",
                "gvkey",
                "earnings_call_date",
                "topic_id",
                "representation",
                "keywords",
                "n_sentences",
                "sentence_ids",
                "processing_timestamp",
            }

            assert required_cols.issubset(set(df.columns))
            assert len(df) > 0
