"""
Unit tests for WRDSPrefetcher.

Tests the prefetch-to-S3 pattern:
- Flattening TranscriptData to rows
- Writing Parquet chunks with correct schema
- Writing manifest with firm_to_chunk mapping
- Checkpoint save/load round-trip
- Resume from checkpoint
"""

import gzip
import json
import tempfile
from datetime import datetime
from io import BytesIO
from unittest.mock import MagicMock, patch

import pandas as pd
import pyarrow.parquet as pq
import pytest

from cloud.src.models import FirmTranscriptData, TranscriptData, TranscriptSentence
from cloud.src.prefetch.wrds_prefetcher import (
    PREFETCH_SCHEMA,
    WRDSPrefetcher,
    quarter_to_date_range,
)


class TestQuarterToDateRange:
    """Tests for quarter_to_date_range function."""

    def test_q1(self):
        """Q1 should be Jan 1 to Mar 31."""
        start, end = quarter_to_date_range("2023Q1")
        assert start == "2023-01-01"
        assert end == "2023-03-31"

    def test_q2(self):
        """Q2 should be Apr 1 to Jun 30."""
        start, end = quarter_to_date_range("2023Q2")
        assert start == "2023-04-01"
        assert end == "2023-06-30"

    def test_q3(self):
        """Q3 should be Jul 1 to Sep 30."""
        start, end = quarter_to_date_range("2023Q3")
        assert start == "2023-07-01"
        assert end == "2023-09-30"

    def test_q4(self):
        """Q4 should be Oct 1 to Dec 31."""
        start, end = quarter_to_date_range("2023Q4")
        assert start == "2023-10-01"
        assert end == "2023-12-31"


class TestFlattenToRows:
    """Tests for _flatten_to_rows method."""

    def test_flatten_single_firm(self):
        """Should flatten a single firm's sentences to rows."""
        prefetcher = WRDSPrefetcher(bucket="test-bucket")

        transcript_data = TranscriptData(
            firms={
                "firm_001": FirmTranscriptData(
                    firm_id="firm_001",
                    firm_name="Test Corp",
                    sentences=[
                        TranscriptSentence(
                            sentence_id="firm_001_t1_0000",
                            raw_text="We had a great quarter.",
                            cleaned_text="great quarter",
                            speaker_type="CEO",
                            position=0,
                        ),
                        TranscriptSentence(
                            sentence_id="firm_001_t1_0001",
                            raw_text="Revenue increased by 20%.",
                            cleaned_text="revenue increase",
                            speaker_type="CFO",
                            position=1,
                        ),
                    ],
                    metadata={
                        "permno": 12345,
                        "gvkey": "001234",
                        "transcript_id": "t1",
                        "earnings_call_date": datetime(2023, 1, 15).date(),
                    },
                )
            }
        )

        rows = prefetcher._flatten_to_rows(transcript_data, "2023Q1")

        assert len(rows) == 2
        assert rows[0]["firm_id"] == "firm_001"
        assert rows[0]["firm_name"] == "Test Corp"
        assert rows[0]["permno"] == 12345
        assert rows[0]["gvkey"] == "001234"
        assert rows[0]["sentence_id"] == "firm_001_t1_0000"
        assert rows[0]["raw_text"] == "We had a great quarter."
        assert rows[0]["cleaned_text"] == "great quarter"
        assert rows[0]["speaker_type"] == "CEO"
        assert rows[0]["position"] == 0
        assert rows[0]["quarter"] == "2023Q1"

    def test_flatten_multiple_firms(self):
        """Should flatten multiple firms."""
        prefetcher = WRDSPrefetcher(bucket="test-bucket")

        transcript_data = TranscriptData(
            firms={
                "firm_001": FirmTranscriptData(
                    firm_id="firm_001",
                    firm_name="Corp A",
                    sentences=[
                        TranscriptSentence(
                            sentence_id="firm_001_t1_0000",
                            raw_text="Sentence 1",
                            cleaned_text="sentence one",
                            speaker_type="CEO",
                            position=0,
                        ),
                    ],
                    metadata={"permno": 11111},
                ),
                "firm_002": FirmTranscriptData(
                    firm_id="firm_002",
                    firm_name="Corp B",
                    sentences=[
                        TranscriptSentence(
                            sentence_id="firm_002_t1_0000",
                            raw_text="Sentence 2",
                            cleaned_text="sentence two",
                            speaker_type="CEO",
                            position=0,
                        ),
                    ],
                    metadata={"permno": 22222},
                ),
            }
        )

        rows = prefetcher._flatten_to_rows(transcript_data, "2023Q1")

        assert len(rows) == 2
        firm_ids = {r["firm_id"] for r in rows}
        assert firm_ids == {"firm_001", "firm_002"}


class TestWriteChunk:
    """Tests for _write_chunk method."""

    def test_write_chunk_creates_parquet(self):
        """Should write valid Parquet with correct schema."""
        mock_s3 = MagicMock()
        uploaded_file = None

        def capture_upload(local_path, bucket, key):
            nonlocal uploaded_file
            with open(local_path, "rb") as f:
                uploaded_file = f.read()

        mock_s3.upload_file = capture_upload

        prefetcher = WRDSPrefetcher(bucket="test-bucket", s3_client=mock_s3)

        rows = [
            {
                "firm_id": "firm_001",
                "firm_name": "Test Corp",
                "permno": 12345,
                "gvkey": "001234",
                "transcript_id": "t1",
                "earnings_call_date": datetime(2023, 1, 15).date(),
                "sentence_id": "firm_001_t1_0000",
                "raw_text": "Test sentence.",
                "cleaned_text": "test sentence",
                "speaker_type": "CEO",
                "position": 0,
                "quarter": "2023Q1",
            }
        ]

        chunk_key = prefetcher._write_chunk(rows, "2023Q1", 0)

        assert chunk_key == "prefetch/transcripts/quarter=2023Q1/chunk_0000.parquet"
        assert uploaded_file is not None

        # Read back and verify
        table = pq.read_table(BytesIO(uploaded_file))
        df = table.to_pandas()
        assert len(df) == 1
        assert df.iloc[0]["firm_id"] == "firm_001"


class TestWriteManifest:
    """Tests for _write_manifest method."""

    def test_write_manifest_gzip_compressed(self):
        """Should write gzip-compressed manifest.json."""
        mock_s3 = MagicMock()
        uploaded_body = None

        def capture_put(Bucket, Key, Body, **kwargs):
            nonlocal uploaded_body
            uploaded_body = Body

        mock_s3.put_object = capture_put

        prefetcher = WRDSPrefetcher(bucket="test-bucket", s3_client=mock_s3)

        firm_to_chunk = {
            "firm_001": "chunk_0000.parquet",
            "firm_002": "chunk_0000.parquet",
            "firm_003": "chunk_0001.parquet",
        }
        chunk_sizes = {
            "chunk_0000.parquet": 2,
            "chunk_0001.parquet": 1,
        }

        manifest_key = prefetcher._write_manifest("2023Q1", firm_to_chunk, chunk_sizes)

        assert manifest_key == "prefetch/transcripts/quarter=2023Q1/manifest.json"

        # Decompress and verify
        decompressed = gzip.decompress(uploaded_body)
        manifest = json.loads(decompressed)

        assert manifest["quarter"] == "2023Q1"
        assert manifest["n_firms"] == 3
        assert manifest["n_chunks"] == 2
        assert manifest["firm_to_chunk"]["firm_001"] == "chunk_0000.parquet"
        assert manifest["chunk_sizes"]["chunk_0000.parquet"] == 2


class TestCheckpoint:
    """Tests for checkpoint save/load."""

    def test_get_checkpoint_no_checkpoint(self):
        """Should return empty state when no checkpoint exists."""
        mock_s3 = MagicMock()
        from botocore.exceptions import ClientError
        mock_s3.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey"}}, "GetObject"
        )

        prefetcher = WRDSPrefetcher(bucket="test-bucket", s3_client=mock_s3)

        completed, last_chunk_id, firm_to_chunk = prefetcher._get_checkpoint("2023Q1")

        assert completed == set()
        assert last_chunk_id == -1
        assert firm_to_chunk == {}

    def test_checkpoint_round_trip(self):
        """Should save and load checkpoint correctly."""
        storage = {}

        def mock_put(Bucket, Key, Body, **kwargs):
            storage[Key] = Body

        def mock_get(Bucket, Key):
            if Key in storage:
                return {"Body": BytesIO(storage[Key])}
            from botocore.exceptions import ClientError
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")

        mock_s3 = MagicMock()
        mock_s3.put_object = mock_put
        mock_s3.get_object = mock_get

        prefetcher = WRDSPrefetcher(bucket="test-bucket", s3_client=mock_s3)

        # Save checkpoint
        completed = {"firm_001", "firm_002"}
        firm_to_chunk = {"firm_001": "chunk_0000.parquet", "firm_002": "chunk_0000.parquet"}
        prefetcher._save_checkpoint("2023Q1", completed, 0, firm_to_chunk)

        # Load checkpoint
        loaded_completed, loaded_chunk_id, loaded_mapping = prefetcher._get_checkpoint("2023Q1")

        assert loaded_completed == completed
        assert loaded_chunk_id == 0
        assert loaded_mapping == firm_to_chunk


class TestPrefetchQuarter:
    """Integration-style tests for prefetch_quarter."""

    def test_prefetch_resumes_from_checkpoint(self):
        """Should skip already-completed firms when resuming."""
        storage = {}

        def mock_put(Bucket, Key, Body, **kwargs):
            storage[Key] = Body if isinstance(Body, bytes) else Body.encode("utf-8")

        def mock_get(Bucket, Key):
            if Key in storage:
                body = storage[Key]
                return {"Body": BytesIO(body if isinstance(body, bytes) else body.encode())}
            from botocore.exceptions import ClientError
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")

        def mock_upload(local_path, bucket, key):
            with open(local_path, "rb") as f:
                storage[key] = f.read()

        def mock_delete(Bucket, Key):
            storage.pop(Key, None)

        mock_s3 = MagicMock()
        mock_s3.put_object = mock_put
        mock_s3.get_object = mock_get
        mock_s3.upload_file = mock_upload
        mock_s3.delete_object = mock_delete

        # Create mock WRDS connector
        mock_connector = MagicMock()

        # First firm is already completed, second and third need processing
        def mock_fetch(firm_ids, start_date, end_date):
            data = {}
            for fid in firm_ids:
                if fid in ["firm_002", "firm_003"]:
                    data[fid] = FirmTranscriptData(
                        firm_id=fid,
                        firm_name=f"Corp {fid}",
                        sentences=[
                            TranscriptSentence(
                                sentence_id=f"{fid}_t1_0000",
                                raw_text="Test",
                                cleaned_text="test",
                                speaker_type="CEO",
                                position=0,
                            )
                        ],
                        metadata={"permno": 12345},
                    )
            return TranscriptData(firms=data)

        mock_connector.fetch_transcripts = mock_fetch

        # Create checkpoint showing firm_001 already done
        checkpoint_data = {
            "quarter": "2023Q1",
            "completed_firm_ids": ["firm_001"],
            "last_chunk_id": 0,
            "firm_to_chunk": {"firm_001": "chunk_0000.parquet"},
        }
        storage["prefetch/transcripts/quarter=2023Q1/_checkpoint.json"] = json.dumps(
            checkpoint_data
        ).encode()

        prefetcher = WRDSPrefetcher(
            bucket="test-bucket",
            s3_client=mock_s3,
            wrds_connector=mock_connector,
        )
        prefetcher.CHUNK_SIZE = 100  # Small chunk for testing

        result = prefetcher.prefetch_quarter(
            "2023Q1",
            firm_ids=["firm_001", "firm_002", "firm_003"],
        )

        # Should include all 3 firms in final result
        assert result["n_firms"] == 3
        assert result["status"] == "complete"

        # Verify manifest was written
        manifest_key = "prefetch/transcripts/quarter=2023Q1/manifest.json"
        assert manifest_key in storage

    def test_prefetch_discovers_firms_when_none_provided(self):
        """When firm_ids=None, should use get_firm_ids_in_range (not load all transcripts)."""
        storage = {}

        def mock_put(Bucket, Key, Body, **kwargs):
            storage[Key] = Body if isinstance(Body, bytes) else Body.encode("utf-8")

        def mock_get(Bucket, Key):
            if Key in storage:
                body = storage[Key]
                return {"Body": BytesIO(body if isinstance(body, bytes) else body.encode())}
            from botocore.exceptions import ClientError
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")

        def mock_upload(local_path, bucket, key):
            with open(local_path, "rb") as f:
                storage[key] = f.read()

        def mock_delete(Bucket, Key):
            storage.pop(Key, None)

        mock_s3 = MagicMock()
        mock_s3.put_object = mock_put
        mock_s3.get_object = mock_get
        mock_s3.upload_file = mock_upload
        mock_s3.delete_object = mock_delete

        mock_connector = MagicMock()

        # get_firm_ids_in_range should be called (lightweight query)
        mock_connector.get_firm_ids_in_range.return_value = ["firm_001", "firm_002"]

        # fetch_transcripts for individual firms
        def mock_fetch(firm_ids, start_date, end_date):
            data = {}
            for fid in firm_ids:
                data[fid] = FirmTranscriptData(
                    firm_id=fid,
                    firm_name=f"Corp {fid}",
                    sentences=[
                        TranscriptSentence(
                            sentence_id=f"{fid}_t1_0000",
                            raw_text="Test",
                            cleaned_text="test",
                            speaker_type="CEO",
                            position=0,
                        )
                    ],
                    metadata={"permno": 12345},
                )
            return TranscriptData(firms=data)

        mock_connector.fetch_transcripts = mock_fetch

        prefetcher = WRDSPrefetcher(
            bucket="test-bucket",
            s3_client=mock_s3,
            wrds_connector=mock_connector,
        )
        prefetcher.CHUNK_SIZE = 100

        # Pass firm_ids=None to trigger discovery
        result = prefetcher.prefetch_quarter("2023Q1", firm_ids=None)

        # Should have called get_firm_ids_in_range
        mock_connector.get_firm_ids_in_range.assert_called_once_with("2023-01-01", "2023-03-31")

        # Should have processed discovered firms
        assert result["n_firms"] == 2

    def test_checkpoint_interval_includes_successful_firms(self):
        """Should checkpoint after CHECKPOINT_INTERVAL firms including successes."""
        checkpoint_saves = []
        storage = {}

        def mock_put(Bucket, Key, Body, **kwargs):
            storage[Key] = Body if isinstance(Body, bytes) else Body.encode("utf-8")
            if "_checkpoint.json" in Key:
                checkpoint_saves.append(json.loads(Body))

        def mock_get(Bucket, Key):
            if Key in storage:
                body = storage[Key]
                return {"Body": BytesIO(body if isinstance(body, bytes) else body.encode())}
            from botocore.exceptions import ClientError
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")

        def mock_upload(local_path, bucket, key):
            with open(local_path, "rb") as f:
                storage[key] = f.read()

        def mock_delete(Bucket, Key):
            storage.pop(Key, None)

        mock_s3 = MagicMock()
        mock_s3.put_object = mock_put
        mock_s3.get_object = mock_get
        mock_s3.upload_file = mock_upload
        mock_s3.delete_object = mock_delete

        mock_connector = MagicMock()

        def mock_fetch(firm_ids, start_date, end_date):
            data = {}
            for fid in firm_ids:
                data[fid] = FirmTranscriptData(
                    firm_id=fid,
                    firm_name=f"Corp {fid}",
                    sentences=[
                        TranscriptSentence(
                            sentence_id=f"{fid}_t1_0000",
                            raw_text="Test",
                            cleaned_text="test",
                            speaker_type="CEO",
                            position=0,
                        )
                    ],
                    metadata={"permno": 12345},
                )
            return TranscriptData(firms=data)

        mock_connector.fetch_transcripts = mock_fetch

        prefetcher = WRDSPrefetcher(
            bucket="test-bucket",
            s3_client=mock_s3,
            wrds_connector=mock_connector,
        )
        prefetcher.CHUNK_SIZE = 500  # Large chunk (won't trigger chunk-based checkpoint)
        prefetcher.CHECKPOINT_INTERVAL = 5  # Small interval for testing

        # Process 12 firms - should trigger checkpoint after firm 5 and 10
        firm_ids = [f"firm_{i:03d}" for i in range(12)]
        prefetcher.prefetch_quarter("2023Q1", firm_ids=firm_ids)

        # Should have checkpoints at intervals (5, 10) plus final chunk write
        # With 12 firms: checkpoint at 5, checkpoint at 10, chunk write at 12
        assert len(checkpoint_saves) >= 2, f"Expected at least 2 checkpoints, got {len(checkpoint_saves)}"


class TestParquetSchema:
    """Tests for Parquet schema compliance."""

    def test_schema_has_required_columns(self):
        """PREFETCH_SCHEMA should have all required columns."""
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

        schema_columns = {field.name for field in PREFETCH_SCHEMA}
        assert schema_columns == expected_columns

    def test_schema_types(self):
        """PREFETCH_SCHEMA should have correct types."""
        import pyarrow as pa

        type_map = {field.name: field.type for field in PREFETCH_SCHEMA}

        assert type_map["firm_id"] == pa.string()
        assert type_map["permno"] == pa.int64()
        assert type_map["position"] == pa.int32()
        assert type_map["earnings_call_date"] == pa.date32()
