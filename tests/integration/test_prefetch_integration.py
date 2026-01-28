"""
Integration tests for prefetch pipeline.

These tests require AWS credentials and a real S3 bucket.
They verify end-to-end prefetch flow:
- Prefetch 3 firms to real S3
- Verify Parquet schema matches specification
- Verify manifest.json structure
- S3TranscriptConnector reads prefetch output via manifest
- Batch job with DATA_SOURCE=s3 processes correctly

Skip if no AWS credentials available.
"""

import gzip
import json
import os
from datetime import datetime
from io import BytesIO
from unittest.mock import MagicMock, patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cloud.src.models import FirmTranscriptData, TranscriptData, TranscriptSentence

# Skip all tests if no AWS credentials (use boto3 resolution, not env vars only)
import boto3

_creds = boto3.Session().get_credentials()
pytestmark = pytest.mark.skipif(
    not _creds or not _creds.access_key, reason="AWS credentials not available"
)


# Test bucket (should be set for integration tests)
TEST_BUCKET = os.environ.get("FTM_TEST_BUCKET", "ftm-pipeline-78ea68c8")


class TestPrefetchToS3Integration:
    """Integration tests for WRDS prefetch to S3."""

    @pytest.fixture
    def mock_wrds_connector(self):
        """Create mock WRDS connector with test data."""
        mock = MagicMock()

        def mock_fetch(firm_ids, start_date, end_date):
            firms = {}
            for fid in firm_ids:
                if fid in ["firm_001", "firm_002", "firm_003"]:
                    firms[fid] = FirmTranscriptData(
                        firm_id=fid,
                        firm_name=f"Test Corp {fid}",
                        sentences=[
                            TranscriptSentence(
                                sentence_id=f"{fid}_t1_{i:04d}",
                                raw_text=f"This is raw sentence {i} for {fid}.",
                                cleaned_text=f"raw sentence {i} {fid}",
                                speaker_type="CEO" if i % 2 == 0 else "CFO",
                                position=i,
                            )
                            for i in range(10)  # 10 sentences per firm
                        ],
                        metadata={
                            "permno": 10000 + int(fid.split("_")[1]),
                            "gvkey": f"G{fid.split('_')[1]}",
                            "transcript_id": f"t_{fid}",
                            "earnings_call_date": datetime(2023, 1, 15).date(),
                        },
                    )
            return TranscriptData(firms=firms)

        mock.fetch_transcripts = mock_fetch
        mock.close = MagicMock()
        return mock

    @pytest.mark.integration
    def test_prefetch_to_s3_and_read_back(self, mock_wrds_connector):
        """Prefetch 3 firms to S3 and read back via S3TranscriptConnector."""
        import boto3
        from cloud.src.connectors.s3_connector import S3TranscriptConnector
        from cloud.src.prefetch.wrds_prefetcher import WRDSPrefetcher

        s3_client = boto3.client("s3")
        quarter = "2023Q1"
        test_prefix = f"prefetch/transcripts/quarter={quarter}/test_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # Modify prefetcher to use test prefix
        prefetcher = WRDSPrefetcher(
            bucket=TEST_BUCKET,
            s3_client=s3_client,
            wrds_connector=mock_wrds_connector,
        )

        # Patch the key methods to use test prefix
        original_chunk_key = prefetcher._get_chunk_key
        original_manifest_key = prefetcher._get_manifest_key
        original_checkpoint_key = prefetcher._get_checkpoint_key

        def test_chunk_key(q, chunk_id):
            return f"{test_prefix}/chunk_{chunk_id:04d}.parquet"

        def test_manifest_key(q):
            return f"{test_prefix}/manifest.json"

        def test_checkpoint_key(q):
            return f"{test_prefix}/_checkpoint.json"

        prefetcher._get_chunk_key = test_chunk_key
        prefetcher._get_manifest_key = test_manifest_key
        prefetcher._get_checkpoint_key = test_checkpoint_key
        prefetcher.CHUNK_SIZE = 10  # Small chunks for testing

        try:
            # Run prefetch
            result = prefetcher.prefetch_quarter(
                quarter,
                firm_ids=["firm_001", "firm_002", "firm_003"],
            )

            assert result["n_firms"] == 3
            assert result["status"] == "complete"

            # Verify manifest on S3
            manifest_response = s3_client.get_object(
                Bucket=TEST_BUCKET,
                Key=f"{test_prefix}/manifest.json",
            )

            manifest_body = manifest_response["Body"].read()
            if manifest_response.get("ContentEncoding") == "gzip":
                manifest_body = gzip.decompress(manifest_body)
            manifest = json.loads(manifest_body)

            assert manifest["quarter"] == quarter
            assert manifest["n_firms"] == 3
            assert "firm_to_chunk" in manifest
            assert "firm_001" in manifest["firm_to_chunk"]

            # Verify Parquet schema
            chunk_file = manifest["firm_to_chunk"]["firm_001"]
            chunk_key = f"{test_prefix}/{chunk_file}"
            chunk_response = s3_client.get_object(Bucket=TEST_BUCKET, Key=chunk_key)
            table = pq.read_table(BytesIO(chunk_response["Body"].read()))

            expected_columns = {
                "firm_id",
                "firm_name",
                "permno",
                "gvkey",
                "transcript_id",
                "earnings_call_date",
                "sentence_id",
                "raw_text",
                "cleaned_text",
                "speaker_type",
                "position",
                "quarter",
            }
            assert set(table.schema.names) == expected_columns

            # Verify types
            schema_types = {f.name: f.type for f in table.schema}
            assert schema_types["firm_id"] == pa.string()
            assert schema_types["permno"] == pa.int64()
            assert schema_types["position"] == pa.int32()

            # Now read back via S3TranscriptConnector
            # Create connector that uses same test prefix
            connector = S3TranscriptConnector(
                bucket=TEST_BUCKET,
                quarter=quarter,
                s3_client=s3_client,
            )
            connector._get_manifest_key = lambda: f"{test_prefix}/manifest.json"
            connector._get_chunk_key = lambda f: f"{test_prefix}/{f}"

            # Fetch firms
            data = connector.fetch_transcripts(
                ["firm_001", "firm_002"],
                "2023-01-01",
                "2023-03-31",
            )

            assert "firm_001" in data.firms
            assert "firm_002" in data.firms
            assert len(data.firms["firm_001"].sentences) == 10

            # Verify no re-preprocessing (cleaned_text unchanged)
            sentence = data.firms["firm_001"].sentences[0]
            assert sentence.cleaned_text == "raw sentence 0 firm_001"

        finally:
            # Cleanup test files
            try:
                # List and delete test prefix
                paginator = s3_client.get_paginator("list_objects_v2")
                for page in paginator.paginate(Bucket=TEST_BUCKET, Prefix=test_prefix):
                    for obj in page.get("Contents", []):
                        s3_client.delete_object(Bucket=TEST_BUCKET, Key=obj["Key"])
            except Exception as e:
                print(f"Cleanup failed: {e}")


class TestParquetSchemaCompliance:
    """Tests for Parquet schema compliance."""

    def test_schema_matches_specification(self):
        """Verify PREFETCH_SCHEMA matches ADR specification."""
        from cloud.src.prefetch.wrds_prefetcher import PREFETCH_SCHEMA

        # Expected schema from plan
        expected = {
            "firm_id": pa.string(),
            "firm_name": pa.string(),
            "permno": pa.int64(),
            "gvkey": pa.string(),
            "transcript_id": pa.string(),
            "earnings_call_date": pa.date32(),
            "sentence_id": pa.string(),
            "raw_text": pa.string(),
            "cleaned_text": pa.string(),
            "speaker_type": pa.string(),
            "position": pa.int32(),
            "quarter": pa.string(),
        }

        for field in PREFETCH_SCHEMA:
            assert field.name in expected, f"Unexpected field: {field.name}"
            assert (
                field.type == expected[field.name]
            ), f"Type mismatch for {field.name}: {field.type} != {expected[field.name]}"

        assert len(PREFETCH_SCHEMA) == len(expected)


class TestManifestStructure:
    """Tests for manifest.json structure compliance."""

    def test_manifest_has_required_fields(self):
        """Verify manifest has all required fields."""
        from cloud.src.prefetch.wrds_prefetcher import WRDSPrefetcher
        from io import BytesIO

        # Create prefetcher with mock S3
        storage = {}

        def mock_put(Bucket, Key, Body, **kwargs):
            storage[Key] = Body if isinstance(Body, bytes) else Body.encode()

        mock_s3 = MagicMock()
        mock_s3.put_object = mock_put

        prefetcher = WRDSPrefetcher(bucket="test-bucket", s3_client=mock_s3)

        # Write manifest
        firm_to_chunk = {
            "firm_001": "chunk_0000.parquet",
            "firm_002": "chunk_0000.parquet",
        }
        chunk_sizes = {"chunk_0000.parquet": 2}

        prefetcher._write_manifest("2023Q1", firm_to_chunk, chunk_sizes)

        # Parse manifest
        manifest_key = "prefetch/transcripts/quarter=2023Q1/manifest.json"
        manifest_body = gzip.decompress(storage[manifest_key])
        manifest = json.loads(manifest_body)

        # Check required fields per plan
        required_fields = [
            "quarter",
            "created_at",
            "n_firms",
            "n_chunks",
            "chunk_sizes",
            "firm_to_chunk",
        ]
        for field in required_fields:
            assert field in manifest, f"Missing required field: {field}"

        # Verify structure
        assert manifest["quarter"] == "2023Q1"
        assert isinstance(manifest["n_firms"], int)
        assert isinstance(manifest["n_chunks"], int)
        assert isinstance(manifest["chunk_sizes"], dict)
        assert isinstance(manifest["firm_to_chunk"], dict)


class TestMemoryBoundedReads:
    """Tests for memory-bounded reads."""

    def test_selective_loading_bounds_memory(self):
        """Verify S3TranscriptConnector only reads required chunks."""
        from cloud.src.connectors.s3_connector import S3TranscriptConnector

        # Track which chunks are read
        read_chunks = []

        # Create manifest with 25 chunks
        firm_to_chunk = {}
        for i in range(5000):  # 5000 firms
            firm_id = f"firm_{i:04d}"
            chunk_idx = i // 200
            firm_to_chunk[firm_id] = f"chunk_{chunk_idx:04d}.parquet"

        manifest = {
            "quarter": "2023Q1",
            "n_firms": 5000,
            "n_chunks": 25,
            "chunk_sizes": {f"chunk_{i:04d}.parquet": 200 for i in range(25)},
            "firm_to_chunk": firm_to_chunk,
        }

        def mock_get_object(Bucket, Key):
            if Key.endswith("manifest.json"):
                compressed = gzip.compress(json.dumps(manifest).encode())
                return {"Body": BytesIO(compressed), "ContentEncoding": "gzip"}
            else:
                # Track read
                chunk_file = Key.split("/")[-1]
                read_chunks.append(chunk_file)

                # Return mock data
                chunk_idx = int(chunk_file.split("_")[1].split(".")[0])
                start = chunk_idx * 200
                firms = [f"firm_{i:04d}" for i in range(start, start + 200)]

                rows = []
                for fid in firms:
                    rows.append(
                        {
                            "firm_id": fid,
                            "firm_name": f"Corp {fid}",
                            "permno": 10000,
                            "gvkey": "G1",
                            "transcript_id": "t1",
                            "earnings_call_date": pd.Timestamp("2023-01-15").date(),
                            "sentence_id": f"{fid}_s1",
                            "raw_text": "test",
                            "cleaned_text": "test",
                            "speaker_type": "CEO",
                            "position": 0,
                            "quarter": "2023Q1",
                        }
                    )

                df = pd.DataFrame(rows)
                buf = BytesIO()
                df.to_parquet(buf)
                buf.seek(0)
                return {"Body": buf}

        mock_s3 = MagicMock()
        mock_s3.get_object = mock_get_object

        connector = S3TranscriptConnector(
            bucket="test-bucket",
            quarter="2023Q1",
            s3_client=mock_s3,
        )

        # Request 500 firms from 3 non-adjacent chunks (0, 12, 24)
        firm_ids = (
            [f"firm_{i:04d}" for i in range(0, 100)]  # chunk 0
            + [f"firm_{i:04d}" for i in range(2400, 2500)]  # chunk 12
            + [f"firm_{i:04d}" for i in range(4800, 5000)]  # chunk 24
        )

        connector.fetch_transcripts(firm_ids, "2023-01-01", "2023-03-31")

        # Should read only 3 chunks, not all 25
        assert len(read_chunks) == 3
        assert "chunk_0000.parquet" in read_chunks
        assert "chunk_0012.parquet" in read_chunks
        assert "chunk_0024.parquet" in read_chunks


class TestEndToEndWithMockBatch:
    """End-to-end tests with mocked Batch."""

    def test_batch_job_with_data_source_s3(self):
        """Simulate batch job with DATA_SOURCE=s3."""
        # This test verifies the entrypoint can create S3TranscriptConnector
        # We can't run a real batch job in unit tests

        from cloud.containers.map.entrypoint import get_data_connector

        with patch(
            "cloud.src.connectors.s3_connector.S3TranscriptConnector"
        ) as mock_cls:
            mock_connector = MagicMock()
            mock_cls.return_value = mock_connector

            connector = get_data_connector("s3", "2023Q1", "test-bucket")

            mock_cls.assert_called_once_with(bucket="test-bucket", quarter="2023Q1")
            assert connector == mock_connector
