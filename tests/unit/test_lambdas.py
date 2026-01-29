"""
Unit tests for Step Functions Lambda helper functions.

Tests cover:
- prefetch_check: Verify prefetch manifest existence
- create_batch_manifest: Create batch manifest and return batch_ids
- notify_completion: Send SNS notifications

All AWS calls are mocked - no real AWS resources needed.
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from botocore.exceptions import ClientError


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_prefetch_manifest():
    """Sample prefetch manifest content."""
    return {
        "quarter": "2023Q1",
        "n_firms": 150,
        "n_chunks": 3,
        "created_at": "2023-12-01T12:00:00Z",
        "firm_to_chunk": {
            f"firm_{i}": f"chunk_{i % 3}" for i in range(150)
        }
    }


@pytest.fixture
def mock_s3_client():
    """Create mock S3 client."""
    return MagicMock()


@pytest.fixture
def mock_sns_client():
    """Create mock SNS client."""
    return MagicMock()


# =============================================================================
# Prefetch Check Tests
# =============================================================================

class TestPrefetchCheck:
    """Tests for prefetch_check Lambda function."""

    def test_returns_exists_true_when_manifest_found(self, sample_prefetch_manifest, mock_s3_client):
        """Should return exists=True when manifest is found."""
        from cloud.src.lambdas.prefetch_check import handler

        # Mock successful S3 response
        mock_s3_client.head_object.return_value = {}
        mock_s3_client.get_object.return_value = {
            "Body": MagicMock(read=lambda: json.dumps(sample_prefetch_manifest).encode()),
            "ContentEncoding": "",
        }

        with patch('boto3.client', return_value=mock_s3_client):
            result = handler(
                {"quarter": "2023Q1", "bucket": "test-bucket"},
                None  # context
            )

        assert result["exists"] is True
        assert result["quarter"] == "2023Q1"
        assert result["n_firms"] == 150
        assert result["manifest_key"] == "prefetch/transcripts/quarter=2023Q1/manifest.json"

    def test_returns_exists_false_when_manifest_not_found(self, mock_s3_client):
        """Should return exists=False when manifest is not found."""
        from cloud.src.lambdas.prefetch_check import handler

        # Mock 404 error
        error_response = {"Error": {"Code": "404"}}
        mock_s3_client.head_object.side_effect = ClientError(error_response, "HeadObject")

        with patch('boto3.client', return_value=mock_s3_client):
            result = handler(
                {"quarter": "2023Q2", "bucket": "test-bucket"},
                None
            )

        assert result["exists"] is False
        assert result["quarter"] == "2023Q2"
        assert "error" in result

    def test_returns_error_when_quarter_missing(self):
        """Should return error when quarter is missing."""
        from cloud.src.lambdas.prefetch_check import handler

        result = handler({"bucket": "test-bucket"}, None)

        assert result["exists"] is False
        assert "quarter" in result.get("error", "").lower()

    def test_returns_error_when_bucket_missing(self):
        """Should return error when bucket is missing and not in env."""
        from cloud.src.lambdas.prefetch_check import handler

        with patch.dict('os.environ', {}, clear=True):
            result = handler({"quarter": "2023Q1"}, None)

        assert result["exists"] is False
        assert "bucket" in result.get("error", "").lower()

    def test_handles_gzip_compressed_manifest(self, sample_prefetch_manifest, mock_s3_client):
        """Should handle gzip-compressed manifest files."""
        import gzip

        from cloud.src.lambdas.prefetch_check import handler

        compressed_content = gzip.compress(json.dumps(sample_prefetch_manifest).encode())

        mock_s3_client.head_object.return_value = {}
        mock_s3_client.get_object.return_value = {
            "Body": MagicMock(read=lambda: compressed_content),
            "ContentEncoding": "gzip",
        }

        with patch('boto3.client', return_value=mock_s3_client):
            result = handler(
                {"quarter": "2023Q1", "bucket": "test-bucket"},
                None
            )

        assert result["exists"] is True
        assert result["n_firms"] == 150


# =============================================================================
# Create Batch Manifest Tests
# =============================================================================

class TestCreateBatchManifest:
    """Tests for create_batch_manifest Lambda function."""

    def test_creates_manifest_with_correct_batch_ids(self, sample_prefetch_manifest, mock_s3_client):
        """Should create manifest and return correct batch_ids."""
        from cloud.src.lambdas.create_batch_manifest import handler

        mock_s3_client.get_object.return_value = {
            "Body": MagicMock(read=lambda: json.dumps(sample_prefetch_manifest).encode()),
            "ContentEncoding": "",
        }
        mock_s3_client.put_object.return_value = {}

        with patch('boto3.client', return_value=mock_s3_client):
            result = handler(
                {"quarter": "2023Q1", "bucket": "test-bucket", "batch_size": 50},
                None
            )

        assert result["quarter"] == "2023Q1"
        assert result["n_firms"] == 150
        assert result["n_batches"] == 3  # 150 firms / 50 per batch
        assert len(result["batch_ids"]) == 3
        assert result["batch_ids"][0] == "2023Q1_batch_0000"
        assert result["batch_ids"][1] == "2023Q1_batch_0001"
        assert result["batch_ids"][2] == "2023Q1_batch_0002"

    def test_batch_ids_are_strings_only(self, sample_prefetch_manifest, mock_s3_client):
        """Batch_ids should be strings only (no nested objects) to avoid payload overflow."""
        from cloud.src.lambdas.create_batch_manifest import handler

        mock_s3_client.get_object.return_value = {
            "Body": MagicMock(read=lambda: json.dumps(sample_prefetch_manifest).encode()),
            "ContentEncoding": "",
        }
        mock_s3_client.put_object.return_value = {}

        with patch('boto3.client', return_value=mock_s3_client):
            result = handler(
                {"quarter": "2023Q1", "bucket": "test-bucket", "batch_size": 100},
                None
            )

        # Critical: batch_ids must be strings, not dicts
        for batch_id in result["batch_ids"]:
            assert isinstance(batch_id, str)
            assert "firm_ids" not in batch_id  # Not serialized JSON

    def test_writes_jsonl_to_s3(self, sample_prefetch_manifest, mock_s3_client):
        """Should write JSONL manifest to S3."""
        from cloud.src.lambdas.create_batch_manifest import handler

        mock_s3_client.get_object.return_value = {
            "Body": MagicMock(read=lambda: json.dumps(sample_prefetch_manifest).encode()),
            "ContentEncoding": "",
        }
        mock_s3_client.put_object.return_value = {}

        with patch('boto3.client', return_value=mock_s3_client):
            result = handler(
                {"quarter": "2023Q1", "bucket": "test-bucket", "batch_size": 100},
                None
            )

        # Verify put_object was called
        mock_s3_client.put_object.assert_called_once()
        call_kwargs = mock_s3_client.put_object.call_args.kwargs

        assert call_kwargs["Bucket"] == "test-bucket"
        assert "manifests/2023Q1/" in call_kwargs["Key"]
        assert call_kwargs["Key"].endswith(".jsonl")

        # Verify content is JSONL
        body = call_kwargs["Body"].decode()
        lines = body.strip().split("\n")
        assert len(lines) == 2  # 150 firms / 100 = 2 batches

        # Each line should be valid JSON
        for line in lines:
            batch = json.loads(line)
            assert "batch_id" in batch
            assert "firm_ids" in batch  # Full firm_ids in JSONL, not in response

    def test_raises_error_when_prefetch_missing(self, mock_s3_client):
        """Should raise error when prefetch manifest doesn't exist."""
        from cloud.src.lambdas.create_batch_manifest import handler

        error_response = {"Error": {"Code": "NoSuchKey"}}
        mock_s3_client.get_object.side_effect = ClientError(error_response, "GetObject")

        with patch('boto3.client', return_value=mock_s3_client):
            with pytest.raises(ValueError, match="Prefetch manifest not found"):
                handler(
                    {"quarter": "2023Q3", "bucket": "test-bucket"},
                    None
                )

    def test_uses_default_batch_size(self, sample_prefetch_manifest, mock_s3_client):
        """Should use default batch_size of 1000 if not specified."""
        from cloud.src.lambdas.create_batch_manifest import handler, DEFAULT_BATCH_SIZE

        mock_s3_client.get_object.return_value = {
            "Body": MagicMock(read=lambda: json.dumps(sample_prefetch_manifest).encode()),
            "ContentEncoding": "",
        }
        mock_s3_client.put_object.return_value = {}

        with patch('boto3.client', return_value=mock_s3_client):
            result = handler(
                {"quarter": "2023Q1", "bucket": "test-bucket"},  # No batch_size
                None
            )

        # 150 firms / 1000 = 1 batch
        assert result["n_batches"] == 1
        assert DEFAULT_BATCH_SIZE == 1000


# =============================================================================
# Summarize Results Tests
# =============================================================================

class TestSummarizeResults:
    """Tests for summarize_results Lambda function."""

    def test_counts_all_succeeded(self):
        """Should count all succeeded when no failures."""
        from cloud.src.lambdas.summarize_results import handler

        batch_results = [
            {"batch_id": "2023Q1_batch_0000", "job_result": {"Status": "SUCCEEDED"}},
            {"batch_id": "2023Q1_batch_0001", "job_result": {"Status": "SUCCEEDED"}},
            {"batch_id": "2023Q1_batch_0002", "job_result": {"Status": "SUCCEEDED"}},
        ]

        result = handler(
            {
                "quarter": "2023Q1",
                "bucket": "test-bucket",
                "batch_results": batch_results,
                "total_batches": 3,
                "execution_name": "test-exec",
            },
            None
        )

        assert result["succeeded"] == 3
        assert result["failed"] == 0
        assert result["has_failures"] is False

    def test_counts_failures_from_job_failed_state(self):
        """Should count failures from JobFailed pass state."""
        from cloud.src.lambdas.summarize_results import handler

        batch_results = [
            {"batch_id": "2023Q1_batch_0000", "job_result": {"Status": "SUCCEEDED"}},
            {"batch_id": "2023Q1_batch_0001", "status": "FAILED", "error": {"Error": "SomeError"}},
            {"batch_id": "2023Q1_batch_0002", "job_result": {"Status": "SUCCEEDED"}},
        ]

        result = handler(
            {
                "quarter": "2023Q1",
                "bucket": "test-bucket",
                "batch_results": batch_results,
                "total_batches": 3,
            },
            None
        )

        assert result["succeeded"] == 2
        assert result["failed"] == 1
        assert result["has_failures"] is True
        assert len(result["failure_details"]) == 1
        assert result["failure_details"][0]["batch_id"] == "2023Q1_batch_0001"

    def test_counts_mixed_job_statuses(self):
        """Should handle various job status values."""
        from cloud.src.lambdas.summarize_results import handler

        batch_results = [
            {"batch_id": "batch_0", "job_result": {"Status": "SUCCEEDED"}},
            {"batch_id": "batch_1", "job_result": {"Status": "FAILED"}},
            {"batch_id": "batch_2", "status": "FAILED", "error": {}},
        ]

        result = handler({"batch_results": batch_results}, None)

        assert result["succeeded"] == 1
        assert result["failed"] == 2

    def test_returns_failure_details(self):
        """Should include failure details when failures exist."""
        from cloud.src.lambdas.summarize_results import handler

        batch_results = [
            {
                "batch_id": "2023Q1_batch_0001",
                "status": "FAILED",
                "error": {"Error": "Batch.JobFailed", "Cause": "Container exited with code 1"}
            },
        ]

        result = handler({"batch_results": batch_results}, None)

        assert result["failure_details"] is not None
        assert result["failure_details"][0]["batch_id"] == "2023Q1_batch_0001"
        assert "error" in result["failure_details"][0]

    def test_handles_empty_batch_results(self):
        """Should handle empty batch results gracefully."""
        from cloud.src.lambdas.summarize_results import handler

        result = handler({"batch_results": [], "quarter": "2023Q1"}, None)

        assert result["succeeded"] == 0
        assert result["failed"] == 0
        assert result["has_failures"] is False


# =============================================================================
# Notify Completion Tests
# =============================================================================

class TestNotifyCompletion:
    """Tests for notify_completion Lambda function."""

    def test_sends_notification_on_success(self, mock_sns_client):
        """Should send SNS notification with success message."""
        from cloud.src.lambdas.notify_completion import handler

        mock_sns_client.publish.return_value = {"MessageId": "msg-123"}

        with patch('boto3.client', return_value=mock_sns_client):
            with patch.dict('os.environ', {"SNS_TOPIC_ARN": "arn:aws:sns:us-east-1:123456789:ftm-notifications"}):
                result = handler(
                    {
                        "quarter": "2023Q1",
                        "succeeded": 5,
                        "failed": 0,
                        "total_batches": 5,
                        "execution_name": "2023Q1-20231201-120000"
                    },
                    None
                )

        assert result["notified"] is True
        assert result["message_id"] == "msg-123"

        # Verify SNS publish was called
        mock_sns_client.publish.assert_called_once()
        call_kwargs = mock_sns_client.publish.call_args.kwargs

        assert "successfully" in call_kwargs["Subject"].lower()
        assert "2023Q1" in call_kwargs["Message"]

    def test_sends_notification_on_failure(self, mock_sns_client):
        """Should send SNS notification with failure message."""
        from cloud.src.lambdas.notify_completion import handler

        mock_sns_client.publish.return_value = {"MessageId": "msg-456"}

        with patch('boto3.client', return_value=mock_sns_client):
            with patch.dict('os.environ', {"SNS_TOPIC_ARN": "arn:aws:sns:us-east-1:123456789:ftm-notifications"}):
                result = handler(
                    {
                        "quarter": "2023Q1",
                        "succeeded": 3,
                        "failed": 2,
                        "total_batches": 5,
                    },
                    None
                )

        call_kwargs = mock_sns_client.publish.call_args.kwargs

        assert "failure" in call_kwargs["Subject"].lower() or "2" in call_kwargs["Subject"]
        assert "PARTIAL" in call_kwargs["Message"] or "failed" in call_kwargs["Message"].lower()

    def test_skips_notification_when_topic_not_set(self):
        """Should skip notification when SNS_TOPIC_ARN is not set."""
        from cloud.src.lambdas.notify_completion import handler

        with patch.dict('os.environ', {}, clear=True):
            result = handler(
                {"quarter": "2023Q1", "succeeded": 5, "failed": 0},
                None
            )

        assert result["notified"] is False
        assert "SNS_TOPIC_ARN" in result.get("reason", "")

    def test_handles_sns_error_gracefully(self, mock_sns_client):
        """Should not fail state machine on SNS error."""
        from cloud.src.lambdas.notify_completion import handler

        mock_sns_client.publish.side_effect = Exception("SNS unavailable")

        with patch('boto3.client', return_value=mock_sns_client):
            with patch.dict('os.environ', {"SNS_TOPIC_ARN": "arn:aws:sns:us-east-1:123456789:ftm-notifications"}):
                result = handler(
                    {"quarter": "2023Q1", "succeeded": 5, "failed": 0},
                    None
                )

        # Should return gracefully, not raise
        assert result["notified"] is False
        assert "error" in result

    def test_message_format_includes_key_fields(self, mock_sns_client):
        """Notification message should include all key fields."""
        from cloud.src.lambdas.notify_completion import handler

        mock_sns_client.publish.return_value = {"MessageId": "msg-789"}

        with patch('boto3.client', return_value=mock_sns_client):
            with patch.dict('os.environ', {"SNS_TOPIC_ARN": "arn:aws:sns:us-east-1:123456789:ftm-notifications"}):
                handler(
                    {
                        "quarter": "2023Q1",
                        "succeeded": 5,
                        "failed": 0,
                        "total_batches": 5,
                        "execution_name": "test-execution-123"
                    },
                    None
                )

        message = mock_sns_client.publish.call_args.kwargs["Message"]

        assert "2023Q1" in message
        assert "5" in message  # batches
        assert "test-execution-123" in message


# =============================================================================
# Integration-Style Unit Tests (State Machine Flow)
# =============================================================================

class TestStateMachineFlow:
    """Tests simulating the Step Functions state machine flow."""

    def test_prefetch_to_manifest_flow(self, sample_prefetch_manifest, mock_s3_client):
        """Test the flow from prefetch check to manifest creation."""
        from cloud.src.lambdas.prefetch_check import handler as prefetch_handler
        from cloud.src.lambdas.create_batch_manifest import handler as manifest_handler

        # Setup mock
        mock_s3_client.head_object.return_value = {}
        mock_s3_client.get_object.return_value = {
            "Body": MagicMock(read=lambda: json.dumps(sample_prefetch_manifest).encode()),
            "ContentEncoding": "",
        }
        mock_s3_client.put_object.return_value = {}

        with patch('boto3.client', return_value=mock_s3_client):
            # Step 1: Check prefetch
            prefetch_result = prefetch_handler(
                {"quarter": "2023Q1", "bucket": "test-bucket"},
                None
            )
            assert prefetch_result["exists"] is True

            # Step 2: Create manifest (only if prefetch exists)
            if prefetch_result["exists"]:
                manifest_result = manifest_handler(
                    {"quarter": "2023Q1", "bucket": "test-bucket", "batch_size": 50},
                    None
                )

                # Verify manifest was created
                assert manifest_result["n_batches"] == 3
                assert len(manifest_result["batch_ids"]) == 3

                # batch_ids can be used in Map state
                for batch_id in manifest_result["batch_ids"]:
                    assert isinstance(batch_id, str)
                    assert batch_id.startswith("2023Q1_batch_")
