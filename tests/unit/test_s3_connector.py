"""
Unit tests for S3TranscriptConnector.

Tests manifest-based selective chunk loading:
- Loading manifest.json correctly
- Determining required chunks from firm_ids
- Selective loading (only required chunks)
- Reconstructing TranscriptData (no re-preprocessing)
- get_available_firm_ids from manifest
"""

import gzip
import json
from io import BytesIO
from unittest.mock import MagicMock

import pandas as pd
import pytest

from cloud.src.connectors.s3_connector import (
    ManifestNotFoundError,
    S3TranscriptConnector,
)
from cloud.src.models import TranscriptData


def create_mock_manifest(n_firms=100, n_chunks=5):
    """Create a mock manifest for testing."""
    firm_to_chunk = {}
    chunk_sizes = {}

    firms_per_chunk = n_firms // n_chunks

    for i in range(n_firms):
        firm_id = f"firm_{i:04d}"
        chunk_idx = i // firms_per_chunk
        chunk_file = f"chunk_{chunk_idx:04d}.parquet"
        firm_to_chunk[firm_id] = chunk_file

        if chunk_file not in chunk_sizes:
            chunk_sizes[chunk_file] = 0
        chunk_sizes[chunk_file] += 1

    return {
        "quarter": "2023Q1",
        "created_at": "2026-01-28T12:00:00Z",
        "n_firms": n_firms,
        "n_chunks": n_chunks,
        "chunk_sizes": chunk_sizes,
        "firm_to_chunk": firm_to_chunk,
    }


def create_mock_chunk_data(firm_ids, chunk_file):
    """Create mock Parquet data for a chunk."""
    rows = []
    for firm_id in firm_ids:
        for i in range(3):  # 3 sentences per firm
            rows.append({
                "firm_id": firm_id,
                "firm_name": f"Corp {firm_id}",
                "permno": 10000 + int(firm_id.split("_")[1]),
                "gvkey": f"G{firm_id.split('_')[1]}",
                "transcript_id": f"t_{firm_id}",
                "earnings_call_date": pd.Timestamp("2023-01-15").date(),
                "sentence_id": f"{firm_id}_t1_{i:04d}",
                "raw_text": f"Raw sentence {i} for {firm_id}",
                "cleaned_text": f"cleaned sentence {i}",
                "speaker_type": "CEO",
                "position": i,
                "quarter": "2023Q1",
            })
    return pd.DataFrame(rows)


class TestLoadManifest:
    """Tests for _load_manifest method."""

    def test_load_gzip_manifest(self):
        """Should decompress and parse gzip manifest."""
        manifest = create_mock_manifest(n_firms=10, n_chunks=2)
        compressed = gzip.compress(json.dumps(manifest).encode())

        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {
            "Body": BytesIO(compressed),
            "ContentEncoding": "gzip",
        }

        connector = S3TranscriptConnector(
            bucket="test-bucket",
            quarter="2023Q1",
            s3_client=mock_s3,
        )

        loaded = connector._load_manifest()

        assert loaded["quarter"] == "2023Q1"
        assert loaded["n_firms"] == 10
        assert loaded["n_chunks"] == 2
        assert "firm_to_chunk" in loaded

    def test_manifest_cached(self):
        """Should cache manifest after first load."""
        manifest = create_mock_manifest(n_firms=5, n_chunks=1)
        compressed = gzip.compress(json.dumps(manifest).encode())

        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {
            "Body": BytesIO(compressed),
            "ContentEncoding": "gzip",
        }

        connector = S3TranscriptConnector(
            bucket="test-bucket",
            quarter="2023Q1",
            s3_client=mock_s3,
        )

        # Load twice
        connector._load_manifest()
        connector._load_manifest()

        # Should only call S3 once
        assert mock_s3.get_object.call_count == 1

    def test_manifest_not_found_raises(self):
        """Should raise ManifestNotFoundError when manifest missing."""
        from botocore.exceptions import ClientError

        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey"}}, "GetObject"
        )

        connector = S3TranscriptConnector(
            bucket="test-bucket",
            quarter="2023Q1",
            s3_client=mock_s3,
        )

        with pytest.raises(ManifestNotFoundError) as exc_info:
            connector._load_manifest()

        assert "2023Q1" in str(exc_info.value)


class TestGetChunksForFirms:
    """Tests for _get_chunks_for_firms method."""

    def test_selects_correct_chunks(self):
        """Should select only chunks containing requested firms."""
        manifest = create_mock_manifest(n_firms=100, n_chunks=5)
        compressed = gzip.compress(json.dumps(manifest).encode())

        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {
            "Body": BytesIO(compressed),
            "ContentEncoding": "gzip",
        }

        connector = S3TranscriptConnector(
            bucket="test-bucket",
            quarter="2023Q1",
            s3_client=mock_s3,
        )

        # Request firms from first and third chunks
        firm_ids = ["firm_0000", "firm_0001", "firm_0040", "firm_0041"]
        chunks = connector._get_chunks_for_firms(firm_ids)

        # Should select chunks 0 and 2 (with 20 firms per chunk)
        assert "chunk_0000.parquet" in chunks
        assert "chunk_0002.parquet" in chunks
        assert len(chunks) == 2

    def test_handles_missing_firms(self):
        """Should handle firms not in manifest gracefully."""
        manifest = create_mock_manifest(n_firms=10, n_chunks=1)
        compressed = gzip.compress(json.dumps(manifest).encode())

        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {
            "Body": BytesIO(compressed),
            "ContentEncoding": "gzip",
        }

        connector = S3TranscriptConnector(
            bucket="test-bucket",
            quarter="2023Q1",
            s3_client=mock_s3,
        )

        # Mix of existing and non-existing firms
        firm_ids = ["firm_0001", "nonexistent_firm"]
        chunks = connector._get_chunks_for_firms(firm_ids)

        # Should still find chunk for existing firm
        assert len(chunks) == 1
        assert "chunk_0000.parquet" in chunks


class TestSelectiveLoading:
    """Tests for selective chunk loading."""

    def test_reads_only_required_chunks(self):
        """Given 1000 firm_ids across 5 chunks, should read only 5 chunks."""
        # Create manifest with 5000 firms across 25 chunks
        manifest = create_mock_manifest(n_firms=5000, n_chunks=25)
        compressed = gzip.compress(json.dumps(manifest).encode())

        read_chunks = []

        def mock_get_object(Bucket, Key):
            if Key.endswith("manifest.json"):
                return {"Body": BytesIO(compressed), "ContentEncoding": "gzip"}
            else:
                # Track which chunks are read
                chunk_file = Key.split("/")[-1]
                read_chunks.append(chunk_file)

                # Return mock parquet data
                # Get firms that should be in this chunk
                chunk_idx = int(chunk_file.split("_")[1].split(".")[0])
                firms_per_chunk = 200
                start_firm = chunk_idx * firms_per_chunk
                chunk_firms = [f"firm_{i:04d}" for i in range(start_firm, start_firm + firms_per_chunk)]

                df = create_mock_chunk_data(chunk_firms, chunk_file)
                buffer = BytesIO()
                df.to_parquet(buffer)
                buffer.seek(0)
                return {"Body": buffer}

        mock_s3 = MagicMock()
        mock_s3.get_object = mock_get_object

        connector = S3TranscriptConnector(
            bucket="test-bucket",
            quarter="2023Q1",
            s3_client=mock_s3,
        )

        # Request 1000 firms spread across 5 chunks
        # Firms 0-199 in chunk 0, 200-399 in chunk 1, etc.
        firm_ids = (
            [f"firm_{i:04d}" for i in range(0, 200)] +      # chunk 0
            [f"firm_{i:04d}" for i in range(400, 600)] +    # chunk 2
            [f"firm_{i:04d}" for i in range(800, 1000)] +   # chunk 4
            [f"firm_{i:04d}" for i in range(1200, 1400)] +  # chunk 6
            [f"firm_{i:04d}" for i in range(1600, 1800)]    # chunk 8
        )

        result = connector.fetch_transcripts(firm_ids, "2023-01-01", "2023-03-31")

        # Should read exactly 5 chunks
        assert len(read_chunks) == 5
        assert set(read_chunks) == {
            "chunk_0000.parquet",
            "chunk_0002.parquet",
            "chunk_0004.parquet",
            "chunk_0006.parquet",
            "chunk_0008.parquet",
        }


class TestBuildTranscriptData:
    """Tests for _build_transcript_data method."""

    def test_reconstructs_transcript_data_structure(self):
        """Should reconstruct TranscriptData matching model structure."""
        manifest = create_mock_manifest(n_firms=10, n_chunks=1)
        compressed = gzip.compress(json.dumps(manifest).encode())

        chunk_df = create_mock_chunk_data(["firm_0001", "firm_0002"], "chunk_0000.parquet")
        parquet_buffer = BytesIO()
        chunk_df.to_parquet(parquet_buffer)
        parquet_buffer.seek(0)

        def mock_get_object(Bucket, Key):
            if Key.endswith("manifest.json"):
                return {"Body": BytesIO(compressed), "ContentEncoding": "gzip"}
            else:
                parquet_buffer.seek(0)
                return {"Body": parquet_buffer}

        mock_s3 = MagicMock()
        mock_s3.get_object = mock_get_object

        connector = S3TranscriptConnector(
            bucket="test-bucket",
            quarter="2023Q1",
            s3_client=mock_s3,
        )

        result = connector.fetch_transcripts(["firm_0001"], "2023-01-01", "2023-03-31")

        # Verify structure
        assert isinstance(result, TranscriptData)
        assert "firm_0001" in result.firms
        assert result.firms["firm_0001"].firm_id == "firm_0001"
        assert len(result.firms["firm_0001"].sentences) == 3

        # Verify sentences
        sentence = result.firms["firm_0001"].sentences[0]
        assert sentence.sentence_id == "firm_0001_t1_0000"
        assert sentence.cleaned_text == "cleaned sentence 0"  # No re-preprocessing
        assert sentence.position == 0

        # Verify metadata
        metadata = result.firms["firm_0001"].metadata
        assert metadata["permno"] == 10001
        assert metadata["quarter"] == "2023Q1"

    def test_no_reprocessing(self):
        """Should use cleaned_text as-is, not re-preprocess."""
        manifest = create_mock_manifest(n_firms=5, n_chunks=1)
        compressed = gzip.compress(json.dumps(manifest).encode())

        # Create data with specific cleaned_text
        df = pd.DataFrame([{
            "firm_id": "firm_0001",
            "firm_name": "Test Corp",
            "permno": 12345,
            "gvkey": "G1234",
            "transcript_id": "t1",
            "earnings_call_date": pd.Timestamp("2023-01-15").date(),
            "sentence_id": "firm_0001_t1_0000",
            "raw_text": "The CEO said revenue increased dramatically.",
            "cleaned_text": "ceo revenue increase dramatically",  # Pre-processed
            "speaker_type": "CEO",
            "position": 0,
            "quarter": "2023Q1",
        }])

        parquet_buffer = BytesIO()
        df.to_parquet(parquet_buffer)
        parquet_buffer.seek(0)

        def mock_get_object(Bucket, Key):
            if Key.endswith("manifest.json"):
                return {"Body": BytesIO(compressed), "ContentEncoding": "gzip"}
            else:
                parquet_buffer.seek(0)
                return {"Body": parquet_buffer}

        mock_s3 = MagicMock()
        mock_s3.get_object = mock_get_object

        connector = S3TranscriptConnector(
            bucket="test-bucket",
            quarter="2023Q1",
            s3_client=mock_s3,
        )

        result = connector.fetch_transcripts(["firm_0001"], "2023-01-01", "2023-03-31")

        # Verify cleaned_text is used as-is
        sentence = result.firms["firm_0001"].sentences[0]
        assert sentence.cleaned_text == "ceo revenue increase dramatically"


class TestGetAvailableFirmIds:
    """Tests for get_available_firm_ids method."""

    def test_returns_from_manifest(self):
        """Should return firm_ids directly from manifest (O(1))."""
        manifest = create_mock_manifest(n_firms=100, n_chunks=5)
        compressed = gzip.compress(json.dumps(manifest).encode())

        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {
            "Body": BytesIO(compressed),
            "ContentEncoding": "gzip",
        }

        connector = S3TranscriptConnector(
            bucket="test-bucket",
            quarter="2023Q1",
            s3_client=mock_s3,
        )

        firm_ids = connector.get_available_firm_ids()

        assert len(firm_ids) == 100
        assert firm_ids == sorted(firm_ids)  # Should be sorted
        assert "firm_0000" in firm_ids
        assert "firm_0099" in firm_ids


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_firm_ids_returns_empty(self):
        """Should return empty TranscriptData for empty firm_ids."""
        connector = S3TranscriptConnector(
            bucket="test-bucket",
            quarter="2023Q1",
        )

        result = connector.fetch_transcripts([], "2023-01-01", "2023-03-31")

        assert isinstance(result, TranscriptData)
        assert len(result.firms) == 0

    def test_all_missing_firms_returns_empty(self):
        """Should return empty TranscriptData when all firms missing from manifest."""
        manifest = create_mock_manifest(n_firms=10, n_chunks=1)
        compressed = gzip.compress(json.dumps(manifest).encode())

        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {
            "Body": BytesIO(compressed),
            "ContentEncoding": "gzip",
        }

        connector = S3TranscriptConnector(
            bucket="test-bucket",
            quarter="2023Q1",
            s3_client=mock_s3,
        )

        result = connector.fetch_transcripts(
            ["nonexistent_1", "nonexistent_2"],
            "2023-01-01",
            "2023-03-31",
        )

        assert isinstance(result, TranscriptData)
        assert len(result.firms) == 0

    def test_date_params_ignored(self):
        """Date parameters should be ignored (quarter already filtered)."""
        manifest = create_mock_manifest(n_firms=5, n_chunks=1)
        compressed = gzip.compress(json.dumps(manifest).encode())

        chunk_df = create_mock_chunk_data(["firm_0001"], "chunk_0000.parquet")
        parquet_buffer = BytesIO()
        chunk_df.to_parquet(parquet_buffer)
        parquet_buffer.seek(0)

        def mock_get_object(Bucket, Key):
            if Key.endswith("manifest.json"):
                return {"Body": BytesIO(compressed), "ContentEncoding": "gzip"}
            else:
                parquet_buffer.seek(0)
                return {"Body": parquet_buffer}

        mock_s3 = MagicMock()
        mock_s3.get_object = mock_get_object

        connector = S3TranscriptConnector(
            bucket="test-bucket",
            quarter="2023Q1",
            s3_client=mock_s3,
        )

        # Use different dates - should be ignored
        result = connector.fetch_transcripts(
            ["firm_0001"],
            "2020-01-01",  # Different date
            "2020-12-31",  # Different date
        )

        # Should still return data (dates ignored)
        assert "firm_0001" in result.firms
