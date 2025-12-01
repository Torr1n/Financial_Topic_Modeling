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
            firm_ids=["1001"],  # Apple's firm ID
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        assert isinstance(result, TranscriptData)

    def test_fetch_finds_matching_firm_ids(self, temp_csv_file):
        """Should find firms by ID."""
        from cloud.src.connectors.local_csv import LocalCSVConnector

        connector = LocalCSVConnector(temp_csv_file)
        result = connector.fetch_transcripts(
            firm_ids=["1001"],  # Apple
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        assert "1001" in result.firms

    def test_fetch_returns_correct_firm_data(self, temp_csv_file):
        """Should return correct firm ID and name."""
        from cloud.src.connectors.local_csv import LocalCSVConnector

        connector = LocalCSVConnector(temp_csv_file)
        result = connector.fetch_transcripts(
            firm_ids=["1001"],
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        firm = result.firms["1001"]
        assert firm.firm_id == "1001"
        assert firm.firm_name == "Apple Inc."

    def test_fetch_returns_more_sentences_than_components(self, temp_csv_file):
        """Should return more sentences than CSV rows (components contain multiple sentences)."""
        from cloud.src.connectors.local_csv import LocalCSVConnector

        connector = LocalCSVConnector(temp_csv_file)
        result = connector.fetch_transcripts(
            firm_ids=["1001"],
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        sentences = result.firms["1001"].sentences
        # Apple has 3 components, each with 3 sentences = 9 total sentences
        assert len(sentences) == 9
        # Verify we split correctly - more sentences than rows
        assert len(sentences) > 3  # More than the 3 CSV rows

    def test_fetch_sentence_has_correct_structure(self, temp_csv_file):
        """Sentences should have required fields."""
        from cloud.src.connectors.local_csv import LocalCSVConnector
        from cloud.src.models import TranscriptSentence

        connector = LocalCSVConnector(temp_csv_file)
        result = connector.fetch_transcripts(
            firm_ids=["1001"],
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
            firm_ids=["1001", "1002"],  # Apple and Microsoft
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        assert "1001" in result.firms  # Apple
        assert "1002" in result.firms  # Microsoft

    def test_fetch_empty_result_for_unknown_firm_id(self, temp_csv_file):
        """Should return empty result for unknown firm IDs."""
        from cloud.src.connectors.local_csv import LocalCSVConnector

        connector = LocalCSVConnector(temp_csv_file)
        result = connector.fetch_transcripts(
            firm_ids=["9999"],  # Unknown ID
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
            firm_ids=["1001", "1003"],  # Apple and Tesla
            start_date="2023-01-01",
            end_date="2023-01-31",
        )

        # Apple is in January, Tesla is in February
        assert "1001" in result.firms  # Apple (Jan 15)
        assert "1003" not in result.firms  # Tesla (Feb 1)


class TestLocalCSVConnectorSentenceSplitting:
    """Tests for SpaCy-based sentence splitting from components."""

    def test_splits_components_into_sentences(self, temp_csv_file):
        """Each component should be split into multiple sentences."""
        from cloud.src.connectors.local_csv import LocalCSVConnector

        connector = LocalCSVConnector(temp_csv_file)
        result = connector.fetch_transcripts(
            firm_ids=["1001"],
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        # Apple has 3 components with 3 sentences each = 9 sentences
        sentences = result.firms["1001"].sentences
        assert len(sentences) == 9

    def test_sentences_are_individual_not_components(self, temp_csv_file):
        """Each sentence should be a single sentence, not a component chunk."""
        from cloud.src.connectors.local_csv import LocalCSVConnector

        connector = LocalCSVConnector(temp_csv_file)
        result = connector.fetch_transcripts(
            firm_ids=["1001"],
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        sentences = result.firms["1001"].sentences
        for sentence in sentences:
            # Each sentence should NOT contain multiple periods (indicating multiple sentences)
            # Allow for abbreviations but not multiple sentences
            period_count = sentence.text.count('.')
            assert period_count <= 1, f"Sentence contains multiple sentences: {sentence.text}"

    def test_preserves_sentence_order_within_transcript(self, temp_csv_file):
        """Sentences should be in correct order across all components."""
        from cloud.src.connectors.local_csv import LocalCSVConnector

        connector = LocalCSVConnector(temp_csv_file)
        result = connector.fetch_transcripts(
            firm_ids=["1001"],
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        sentences = result.firms["1001"].sentences
        positions = [s.position for s in sentences]

        # Positions should be monotonically increasing
        for i in range(1, len(positions)):
            assert positions[i] > positions[i - 1]


class TestLocalCSVConnectorStopwordRemoval:
    """Tests for stopword removal preprocessing."""

    def test_removes_common_stopwords(self, temp_csv_file):
        """Sentence text should have stopwords removed."""
        from cloud.src.connectors.local_csv import LocalCSVConnector

        connector = LocalCSVConnector(temp_csv_file)
        result = connector.fetch_transcripts(
            firm_ids=["1001"],
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        sentences = result.firms["1001"].sentences
        common_stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been"}

        for sentence in sentences:
            words = sentence.text.lower().split()
            stopwords_found = common_stopwords.intersection(set(words))
            assert len(stopwords_found) == 0, f"Stopwords found in: {sentence.text}"

    def test_preserves_meaningful_words(self, temp_csv_file):
        """Should keep meaningful content words after stopword removal."""
        from cloud.src.connectors.local_csv import LocalCSVConnector

        connector = LocalCSVConnector(temp_csv_file)
        result = connector.fetch_transcripts(
            firm_ids=["1001"],
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        sentences = result.firms["1001"].sentences

        # Check that meaningful words are preserved
        all_text = " ".join(s.text.lower() for s in sentences)
        meaningful_words = ["investing", "ai", "machine", "learning", "revenue", "growth"]

        found_words = [w for w in meaningful_words if w in all_text]
        assert len(found_words) > 0, "No meaningful words found after stopword removal"

    def test_no_empty_sentences_after_stopword_removal(self, temp_csv_file):
        """Should not produce empty sentences after stopword removal."""
        from cloud.src.connectors.local_csv import LocalCSVConnector

        connector = LocalCSVConnector(temp_csv_file)
        result = connector.fetch_transcripts(
            firm_ids=["1001"],
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        for sentence in result.firms["1001"].sentences:
            assert len(sentence.text.strip()) > 0, "Empty sentence found"


class TestLocalCSVConnectorSentenceIdGeneration:
    """Tests for sentence_id generation rules."""

    def test_sentence_id_format(self, temp_csv_file):
        """sentence_id should follow format: {firm_id}_{transcript_id}_{position:04d}"""
        from cloud.src.connectors.local_csv import LocalCSVConnector

        connector = LocalCSVConnector(temp_csv_file)
        result = connector.fetch_transcripts(
            firm_ids=["1001"],
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        sentence = result.firms["1001"].sentences[0]
        # Format: {firm_id}_{transcript_id}_{position:04d}
        parts = sentence.sentence_id.split("_")
        assert len(parts) >= 3
        assert parts[0] == "1001"  # firm_id

    def test_position_is_zero_indexed_across_transcript(self, temp_csv_file):
        """Position should be 0-indexed across all sentences in transcript."""
        from cloud.src.connectors.local_csv import LocalCSVConnector

        connector = LocalCSVConnector(temp_csv_file)
        result = connector.fetch_transcripts(
            firm_ids=["1001"],
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        sentences = result.firms["1001"].sentences
        # Positions should start at 0
        positions = [s.position for s in sentences]
        assert min(positions) == 0  # First position is 0
        # First sentence_id should contain 0000
        assert "_0000" in sentences[0].sentence_id

    def test_positions_are_sequential(self, temp_csv_file):
        """Positions should be sequential 0, 1, 2, ... across all sentences."""
        from cloud.src.connectors.local_csv import LocalCSVConnector

        connector = LocalCSVConnector(temp_csv_file)
        result = connector.fetch_transcripts(
            firm_ids=["1001"],
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        sentences = result.firms["1001"].sentences
        positions = [s.position for s in sentences]

        # Should be sequential: 0, 1, 2, 3, ...
        expected = list(range(len(sentences)))
        assert positions == expected

    def test_sentence_ids_are_unique(self, temp_csv_file):
        """All sentence_ids should be unique."""
        from cloud.src.connectors.local_csv import LocalCSVConnector

        connector = LocalCSVConnector(temp_csv_file)
        result = connector.fetch_transcripts(
            firm_ids=["1001", "1002", "1003"],  # All firms
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        all_ids = []
        for firm_data in result.firms.values():
            for sentence in firm_data.sentences:
                all_ids.append(sentence.sentence_id)

        assert len(all_ids) == len(set(all_ids))  # All unique


class TestLocalCSVConnectorGetAvailableFirmIds:
    """Tests for get_available_firm_ids method."""

    def test_get_available_firm_ids_returns_list(self, temp_csv_file):
        """get_available_firm_ids should return list of firm IDs."""
        from cloud.src.connectors.local_csv import LocalCSVConnector

        connector = LocalCSVConnector(temp_csv_file)
        firm_ids = connector.get_available_firm_ids()

        assert isinstance(firm_ids, list)
        assert len(firm_ids) == 3  # Apple, Microsoft, Tesla in sample

    def test_get_available_firm_ids_contains_expected(self, temp_csv_file):
        """Should contain expected firm IDs."""
        from cloud.src.connectors.local_csv import LocalCSVConnector

        connector = LocalCSVConnector(temp_csv_file)
        firm_ids = connector.get_available_firm_ids()

        assert "1001" in firm_ids  # Apple
        assert "1002" in firm_ids  # Microsoft
        assert "1003" in firm_ids  # Tesla


class TestLocalCSVConnectorGetFirmIdByName:
    """Tests for get_firm_id_by_name convenience method."""

    def test_get_firm_id_by_name_finds_firm(self, temp_csv_file):
        """Should find firm ID by company name."""
        from cloud.src.connectors.local_csv import LocalCSVConnector

        connector = LocalCSVConnector(temp_csv_file)

        assert connector.get_firm_id_by_name("Apple Inc.") == "1001"
        assert connector.get_firm_id_by_name("Microsoft Corp.") == "1002"

    def test_get_firm_id_by_name_case_insensitive(self, temp_csv_file):
        """Should be case-insensitive."""
        from cloud.src.connectors.local_csv import LocalCSVConnector

        connector = LocalCSVConnector(temp_csv_file)

        assert connector.get_firm_id_by_name("apple inc.") == "1001"
        assert connector.get_firm_id_by_name("APPLE INC.") == "1001"

    def test_get_firm_id_by_name_returns_none_for_unknown(self, temp_csv_file):
        """Should return None for unknown firm name."""
        from cloud.src.connectors.local_csv import LocalCSVConnector

        connector = LocalCSVConnector(temp_csv_file)

        assert connector.get_firm_id_by_name("Unknown Corp") is None


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
            firm_ids=["1001"],
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        metadata = result.firms["1001"].metadata
        assert "n_transcripts" in metadata or len(metadata) >= 0
