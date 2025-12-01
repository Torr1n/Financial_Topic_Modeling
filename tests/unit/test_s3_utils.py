"""
Unit tests for S3 utility functions.

Tests are written BEFORE implementation (TDD).
Uses moto for AWS mocking.
"""

import pytest
import json
import boto3
from moto import mock_aws


class TestUploadJson:
    """Tests for upload_json function."""

    @mock_aws
    def test_upload_json_creates_object(self):
        """upload_json should create S3 object."""
        from cloud.src.s3_utils import upload_json

        # Setup mock S3
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")

        # Upload
        data = {"firm_id": "1001", "n_topics": 5}
        upload_json("test-bucket", "output/test.json", data)

        # Verify
        response = s3.get_object(Bucket="test-bucket", Key="output/test.json")
        content = json.loads(response["Body"].read().decode("utf-8"))
        assert content == data

    @mock_aws
    def test_upload_json_handles_nested_data(self):
        """upload_json should handle nested dicts and lists."""
        from cloud.src.s3_utils import upload_json

        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")

        data = {
            "firm_id": "1001",
            "topics": [
                {"topic_id": 0, "keywords": ["ai", "ml"]},
                {"topic_id": 1, "keywords": ["revenue"]},
            ],
            "metadata": {"nested": {"deep": "value"}},
        }
        upload_json("test-bucket", "test.json", data)

        response = s3.get_object(Bucket="test-bucket", Key="test.json")
        content = json.loads(response["Body"].read().decode("utf-8"))
        assert content == data

    @mock_aws
    def test_upload_json_with_custom_encoder(self):
        """upload_json should handle non-serializable types with default encoder."""
        from cloud.src.s3_utils import upload_json
        from datetime import datetime

        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")

        # Data with datetime (not natively JSON serializable)
        data = {"timestamp": datetime(2023, 1, 15, 10, 30, 0)}
        upload_json("test-bucket", "test.json", data)

        # Should succeed with str conversion
        response = s3.get_object(Bucket="test-bucket", Key="test.json")
        content = json.loads(response["Body"].read().decode("utf-8"))
        assert "2023-01-15" in content["timestamp"]


class TestDownloadJson:
    """Tests for download_json function."""

    @mock_aws
    def test_download_json_retrieves_object(self):
        """download_json should retrieve and parse S3 object."""
        from cloud.src.s3_utils import download_json

        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")

        # Put object
        data = {"firm_id": "1001", "n_topics": 5}
        s3.put_object(
            Bucket="test-bucket",
            Key="test.json",
            Body=json.dumps(data),
            ContentType="application/json",
        )

        # Download
        result = download_json("test-bucket", "test.json")

        assert result == data

    @mock_aws
    def test_download_json_not_found_raises(self):
        """download_json should raise for non-existent key."""
        from cloud.src.s3_utils import download_json
        from botocore.exceptions import ClientError

        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")

        with pytest.raises(ClientError):
            download_json("test-bucket", "nonexistent.json")


class TestListJsonFiles:
    """Tests for list_json_files function."""

    @mock_aws
    def test_list_json_files_returns_keys(self):
        """list_json_files should return matching keys."""
        from cloud.src.s3_utils import list_json_files

        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")

        # Create test files
        for firm in ["AAPL", "MSFT", "GOOG"]:
            s3.put_object(
                Bucket="test-bucket",
                Key=f"firm-topics/{firm}_topics.json",
                Body=json.dumps({"firm_id": firm}),
            )

        # List
        files = list_json_files("test-bucket", "firm-topics/")

        assert len(files) == 3
        assert "firm-topics/AAPL_topics.json" in files
        assert "firm-topics/MSFT_topics.json" in files
        assert "firm-topics/GOOG_topics.json" in files

    @mock_aws
    def test_list_json_files_filters_by_suffix(self):
        """list_json_files should only return _topics.json files."""
        from cloud.src.s3_utils import list_json_files

        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")

        # Create mixed files
        s3.put_object(Bucket="test-bucket", Key="firm-topics/AAPL_topics.json", Body="{}")
        s3.put_object(Bucket="test-bucket", Key="firm-topics/config.yaml", Body="")
        s3.put_object(Bucket="test-bucket", Key="firm-topics/readme.md", Body="")

        files = list_json_files("test-bucket", "firm-topics/", suffix="_topics.json")

        assert len(files) == 1
        assert "firm-topics/AAPL_topics.json" in files

    @mock_aws
    def test_list_json_files_empty_prefix(self):
        """list_json_files should handle empty results."""
        from cloud.src.s3_utils import list_json_files

        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")

        files = list_json_files("test-bucket", "nonexistent/")

        assert files == []
