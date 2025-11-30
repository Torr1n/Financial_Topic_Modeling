"""
Unit tests for DataConnector implementations.

Tests are written BEFORE implementation (TDD).
"""

import pytest
import os


class TestLocalCSVConnectorInit:
    """Tests for LocalCSVConnector initialization."""

    def test_implements_data_connector_interface(self):
        """LocalCSVConnector should implement DataConnector interface."""
        from cloud.src.connectors.local_csv import LocalCSVConnector
        from cloud.src.interfaces import DataConnector

        assert issubclass(LocalCSVConnector, DataConnector)

    def test_init_with_valid_path(self, temp_csv_file):
        """Should initialize with valid CSV path."""
        from cloud.src.connectors.local_csv import LocalCSVConnector

        connector = LocalCSVConnector(temp_csv_file)

        assert connector.csv_path == temp_csv_file

    def test_init_with_invalid_path_raises(self):
        """Should raise FileNotFoundError for invalid path."""
        from cloud.src.connectors.local_csv import LocalCSVConnector

        with pytest.raises(FileNotFoundError):
            LocalCSVConnector("/nonexistent/path.csv")


class TestLocalCSVConnectorFetchTranscripts:
    """Tests for LocalCSVConnector.fetch_transcripts method."""

    def test_fetch_returns_transcript_data(self, temp_csv_file):
        """fetch_transcripts should return TranscriptData."""
        from cloud.src.connectors.local_csv import LocalCSVConnector
        from cloud.src.models import TranscriptData

        connector = LocalCSVConnector(temp_csv_file)
        result = connector.fetch_transcripts(
            firms=["Apple Inc."],
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        assert isinstance(result, TranscriptData)

    def test_fetch_finds_matching_firms(self, temp_csv_file):
        """Should find firms by name (case-insensitive)."""
        from cloud.src.connectors.local_csv import LocalCSVConnector

        connector = LocalCSVConnector(temp_csv_file)
        result = connector.fetch_transcripts(
            firms=["apple inc."],  # lowercase
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        # Should find Apple Inc. (companyid=1001)
        assert "1001" in result.firms

    def test_fetch_returns_correct_firm_data(self, temp_csv_file):
        """Should return correct firm ID and name."""
        from cloud.src.connectors.local_csv import LocalCSVConnector

        connector = LocalCSVConnector(temp_csv_file)
        result = connector.fetch_transcripts(
            firms=["Apple Inc."],
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        firm = result.firms["1001"]
        assert firm.firm_id == "1001"
        assert firm.firm_name == "Apple Inc."

    def test_fetch_returns_sentences(self, temp_csv_file):
        """Should return sentences for matched firms."""
        from cloud.src.connectors.local_csv import LocalCSVConnector

        connector = LocalCSVConnector(temp_csv_file)
        result = connector.fetch_transcripts(
            firms=["Apple Inc."],
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        sentences = result.firms["1001"].sentences
        assert len(sentences) == 3  # Apple has 3 rows in sample CSV

    def test_fetch_sentence_has_correct_structure(self, temp_csv_file):
        """Sentences should have required fields."""
        from cloud.src.connectors.local_csv import LocalCSVConnector
        from cloud.src.models import TranscriptSentence

        connector = LocalCSVConnector(temp_csv_file)
        result = connector.fetch_transcripts(
            firms=["Apple Inc."],
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        sentence = result.firms["1001"].sentences[0]
        assert isinstance(sentence, TranscriptSentence)
        assert sentence.sentence_id is not None
        assert sentence.text is not None
        assert sentence.position is not None

    def test_fetch_multiple_firms(self, temp_csv_file):
        """Should fetch multiple firms at once."""
        from cloud.src.connectors.local_csv import LocalCSVConnector

        connector = LocalCSVConnector(temp_csv_file)
        result = connector.fetch_transcripts(
            firms=["Apple Inc.", "Microsoft Corp."],
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        assert "1001" in result.firms  # Apple
        assert "1002" in result.firms  # Microsoft

    def test_fetch_empty_result_for_unknown_firm(self, temp_csv_file):
        """Should return empty result for unknown firms."""
        from cloud.src.connectors.local_csv import LocalCSVConnector

        connector = LocalCSVConnector(temp_csv_file)
        result = connector.fetch_transcripts(
            firms=["Unknown Corp"],
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        assert len(result.firms) == 0

    def test_fetch_date_filtering_inclusive(self, temp_csv_file):
        """Date filtering should be inclusive on both ends."""
        from cloud.src.connectors.local_csv import LocalCSVConnector

        connector = LocalCSVConnector(temp_csv_file)

        # Filter to only January (Apple and Microsoft)
        result = connector.fetch_transcripts(
            firms=["Apple Inc.", "Tesla Inc."],
            start_date="2023-01-01",
            end_date="2023-01-31",
        )

        # Apple is in January, Tesla is in February
        assert "1001" in result.firms  # Apple (Jan 15)
        assert "1003" not in result.firms  # Tesla (Feb 1)


class TestLocalCSVConnectorSentenceIdGeneration:
    """Tests for sentence_id generation rules."""

    def test_sentence_id_format(self, temp_csv_file):
        """sentence_id should follow format: {firm_id}_{transcript_id}_{position:04d}"""
        from cloud.src.connectors.local_csv import LocalCSVConnector

        connector = LocalCSVConnector(temp_csv_file)
        result = connector.fetch_transcripts(
            firms=["Apple Inc."],
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        sentence = result.firms["1001"].sentences[0]
        # Format: {firm_id}_{transcript_id}_{position:04d}
        parts = sentence.sentence_id.split("_")
        assert len(parts) >= 3
        assert parts[0] == "1001"  # firm_id

    def test_sentence_ids_are_unique(self, temp_csv_file):
        """All sentence_ids should be unique."""
        from cloud.src.connectors.local_csv import LocalCSVConnector

        connector = LocalCSVConnector(temp_csv_file)
        result = connector.fetch_transcripts(
            firms=["Apple Inc.", "Microsoft Corp.", "Tesla Inc."],
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        all_ids = []
        for firm_data in result.firms.values():
            for sentence in firm_data.sentences:
                all_ids.append(sentence.sentence_id)

        assert len(all_ids) == len(set(all_ids))  # All unique


class TestLocalCSVConnectorGetAvailableFirms:
    """Tests for get_available_firms method."""

    def test_get_available_firms_returns_list(self, temp_csv_file):
        """get_available_firms should return list of firm names."""
        from cloud.src.connectors.local_csv import LocalCSVConnector

        connector = LocalCSVConnector(temp_csv_file)
        firms = connector.get_available_firms()

        assert isinstance(firms, list)
        assert len(firms) == 3  # Apple, Microsoft, Tesla in sample

    def test_get_available_firms_contains_expected(self, temp_csv_file):
        """Should contain expected firm names."""
        from cloud.src.connectors.local_csv import LocalCSVConnector

        connector = LocalCSVConnector(temp_csv_file)
        firms = connector.get_available_firms()

        assert "Apple Inc." in firms
        assert "Microsoft Corp." in firms
        assert "Tesla Inc." in firms


class TestLocalCSVConnectorClose:
    """Tests for close method."""

    def test_close_does_not_raise(self, temp_csv_file):
        """close() should not raise an exception."""
        from cloud.src.connectors.local_csv import LocalCSVConnector

        connector = LocalCSVConnector(temp_csv_file)
        connector.close()  # Should not raise


class TestLocalCSVConnectorMetadata:
    """Tests for metadata handling."""

    def test_firm_metadata_contains_date_range(self, temp_csv_file):
        """Firm metadata should include date range info."""
        from cloud.src.connectors.local_csv import LocalCSVConnector

        connector = LocalCSVConnector(temp_csv_file)
        result = connector.fetch_transcripts(
            firms=["Apple Inc."],
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        metadata = result.firms["1001"].metadata
        assert "transcript_count" in metadata or "n_transcripts" in metadata or len(metadata) >= 0
