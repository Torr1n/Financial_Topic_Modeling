"""
Unit tests for WRDSConnector.

Tests are written BEFORE implementation (TDD).
Uses mocked WRDS connections to avoid network dependencies.
"""

import pytest
from datetime import date
from unittest.mock import MagicMock, patch, PropertyMock
import pandas as pd


# =============================================================================
# TestWRDSConnectorInit
# =============================================================================


class TestWRDSConnectorInit:
    """Tests for WRDSConnector initialization."""

    def test_implements_data_connector_interface(self):
        """WRDSConnector should implement DataConnector interface."""
        from cloud.src.connectors.wrds_connector import WRDSConnector
        from cloud.src.interfaces import DataConnector

        assert issubclass(WRDSConnector, DataConnector)

    def test_init_without_connection_creates_lazy(self):
        """Should initialize with lazy connection when no connection provided."""
        from cloud.src.connectors.wrds_connector import WRDSConnector

        connector = WRDSConnector()

        # Connection should not be created yet
        assert connector._conn is None
        assert connector._owns_connection is True

    def test_init_with_connection_uses_provided(self):
        """Should use provided connection and not own it."""
        from cloud.src.connectors.wrds_connector import WRDSConnector

        mock_conn = MagicMock()
        connector = WRDSConnector(connection=mock_conn)

        assert connector._conn is mock_conn
        assert connector._owns_connection is False

    def test_context_manager_support(self):
        """Should support context manager protocol."""
        from cloud.src.connectors.wrds_connector import WRDSConnector

        # Test __enter__ returns self
        connector = WRDSConnector()
        assert connector.__enter__() is connector

        # Test __exit__ calls close
        with patch.object(connector, 'close') as mock_close:
            connector.__exit__(None, None, None)
            mock_close.assert_called_once()


# =============================================================================
# TestWRDSConnectorFetchTranscripts
# =============================================================================


class TestWRDSConnectorFetchTranscripts:
    """Tests for WRDSConnector.fetch_transcripts method."""

    def test_fetch_returns_transcript_data(self, mock_wrds_dataframe):
        """fetch_transcripts should return TranscriptData."""
        from cloud.src.connectors.wrds_connector import WRDSConnector
        from cloud.src.models import TranscriptData

        mock_conn = MagicMock()
        mock_conn.raw_sql.return_value = mock_wrds_dataframe

        connector = WRDSConnector(connection=mock_conn)
        result = connector.fetch_transcripts(
            firm_ids=["374372246"],
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        assert isinstance(result, TranscriptData)

    def test_fetch_finds_matching_firm_ids(self, mock_wrds_dataframe):
        """Should find firms by ID."""
        from cloud.src.connectors.wrds_connector import WRDSConnector

        mock_conn = MagicMock()
        mock_conn.raw_sql.return_value = mock_wrds_dataframe

        connector = WRDSConnector(connection=mock_conn)
        result = connector.fetch_transcripts(
            firm_ids=["374372246"],
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        assert "374372246" in result.firms

    def test_fetch_returns_correct_firm_structure(self, mock_wrds_dataframe):
        """Should return correct firm ID and name."""
        from cloud.src.connectors.wrds_connector import WRDSConnector

        mock_conn = MagicMock()
        mock_conn.raw_sql.return_value = mock_wrds_dataframe

        connector = WRDSConnector(connection=mock_conn)
        result = connector.fetch_transcripts(
            firm_ids=["374372246"],
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        firm = result.firms["374372246"]
        assert firm.firm_id == "374372246"
        assert firm.firm_name == "Lamb Weston Holdings, Inc."

    def test_fetch_passes_date_params_to_sql(self, mock_wrds_dataframe):
        """Should pass start_date and end_date to SQL query."""
        from cloud.src.connectors.wrds_connector import WRDSConnector

        mock_conn = MagicMock()
        mock_conn.raw_sql.return_value = mock_wrds_dataframe

        connector = WRDSConnector(connection=mock_conn)
        connector.fetch_transcripts(
            firm_ids=["374372246"],
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        # Verify raw_sql was called with date parameters
        mock_conn.raw_sql.assert_called_once()
        call_args = mock_conn.raw_sql.call_args
        params = call_args[1].get('params') or call_args[1]

        assert params['start_date'] == "2023-01-01"
        assert params['end_date'] == "2023-03-31"

    def test_fetch_empty_result_for_unknown_firms(self):
        """Should return empty result for unknown firm IDs."""
        from cloud.src.connectors.wrds_connector import WRDSConnector

        mock_conn = MagicMock()
        # Return empty DataFrame
        mock_conn.raw_sql.return_value = pd.DataFrame()

        connector = WRDSConnector(connection=mock_conn)
        result = connector.fetch_transcripts(
            firm_ids=["9999999"],
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        assert len(result.firms) == 0

    def test_fetch_multiple_firms(self, mock_wrds_dataframe):
        """Should fetch multiple firms at once."""
        from cloud.src.connectors.wrds_connector import WRDSConnector

        mock_conn = MagicMock()
        mock_conn.raw_sql.return_value = mock_wrds_dataframe

        connector = WRDSConnector(connection=mock_conn)
        result = connector.fetch_transcripts(
            firm_ids=["374372246", "24937"],
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        assert "374372246" in result.firms  # Lamb Weston
        assert "24937" in result.firms  # Apple

    def test_fetch_selects_latest_transcript_per_firm(self, mock_wrds_dataframe_multi_transcript):
        """Should select only the latest transcript per firm."""
        from cloud.src.connectors.wrds_connector import WRDSConnector

        mock_conn = MagicMock()
        mock_conn.raw_sql.return_value = mock_wrds_dataframe_multi_transcript

        connector = WRDSConnector(connection=mock_conn)
        result = connector.fetch_transcripts(
            firm_ids=["24937"],
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        firm = result.firms["24937"]
        # Should only have sentences from the newer transcript (2023-01-20)
        # Newer transcript has 2 components with sentence A and B
        # Check that we have the newer transcript's data
        assert firm.metadata["transcript_id"] == "789012"
        assert firm.metadata["earnings_call_date"] == date(2023, 1, 20)

    def test_fetch_retains_all_components_for_selected_transcript(
        self, mock_wrds_dataframe_multi_transcript
    ):
        """Should retain all components from the selected (latest) transcript."""
        from cloud.src.connectors.wrds_connector import WRDSConnector

        mock_conn = MagicMock()
        mock_conn.raw_sql.return_value = mock_wrds_dataframe_multi_transcript

        connector = WRDSConnector(connection=mock_conn)
        result = connector.fetch_transcripts(
            firm_ids=["24937"],
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        firm = result.firms["24937"]
        # Newer transcript has 2 components, each should produce sentences
        # "New transcript sentence A." and "New transcript sentence B."
        assert len(firm.sentences) >= 2


# =============================================================================
# TestWRDSConnectorPermnoLinking
# =============================================================================


class TestWRDSConnectorPermnoLinking:
    """Tests for PERMNO linking behavior."""

    def test_linked_firms_have_permno_in_metadata(self, mock_wrds_dataframe):
        """Linked firms should have PERMNO in metadata."""
        from cloud.src.connectors.wrds_connector import WRDSConnector

        mock_conn = MagicMock()
        mock_conn.raw_sql.return_value = mock_wrds_dataframe

        connector = WRDSConnector(connection=mock_conn)
        result = connector.fetch_transcripts(
            firm_ids=["374372246"],
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        firm = result.firms["374372246"]
        assert "permno" in firm.metadata
        assert firm.metadata["permno"] == 16431

    def test_linked_firms_have_gvkey_in_metadata(self, mock_wrds_dataframe):
        """Linked firms should have GVKEY in metadata."""
        from cloud.src.connectors.wrds_connector import WRDSConnector

        mock_conn = MagicMock()
        mock_conn.raw_sql.return_value = mock_wrds_dataframe

        connector = WRDSConnector(connection=mock_conn)
        result = connector.fetch_transcripts(
            firm_ids=["374372246"],
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        firm = result.firms["374372246"]
        assert "gvkey" in firm.metadata
        assert firm.metadata["gvkey"] == "123456"

    def test_linked_firms_have_link_date_in_metadata(self, mock_wrds_dataframe):
        """Linked firms should have link_date in metadata."""
        from cloud.src.connectors.wrds_connector import WRDSConnector

        mock_conn = MagicMock()
        mock_conn.raw_sql.return_value = mock_wrds_dataframe

        connector = WRDSConnector(connection=mock_conn)
        result = connector.fetch_transcripts(
            firm_ids=["374372246"],
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        firm = result.firms["374372246"]
        assert "link_date" in firm.metadata
        assert firm.metadata["link_date"] == date(2022, 1, 1)

    def test_unlinked_firms_are_skipped(self, mock_wrds_dataframe_unlinked, caplog):
        """Firms without PERMNO should NOT be included in output."""
        from cloud.src.connectors.wrds_connector import WRDSConnector
        import logging

        mock_conn = MagicMock()
        mock_conn.raw_sql.return_value = mock_wrds_dataframe_unlinked

        connector = WRDSConnector(connection=mock_conn)
        with caplog.at_level(logging.WARNING):
            result = connector.fetch_transcripts(
                firm_ids=["999999", "24937"],
                start_date="2023-01-01",
                end_date="2023-03-31",
            )

        # Unlinked firm should NOT be in output
        assert "999999" not in result.firms
        # Linked firm should be in output
        assert "24937" in result.firms

    def test_skipped_firms_logged(self, mock_wrds_dataframe_unlinked, caplog):
        """Skipped (unlinked) firms should be logged."""
        from cloud.src.connectors.wrds_connector import WRDSConnector
        import logging

        mock_conn = MagicMock()
        mock_conn.raw_sql.return_value = mock_wrds_dataframe_unlinked

        connector = WRDSConnector(connection=mock_conn)
        with caplog.at_level(logging.WARNING):
            connector.fetch_transcripts(
                firm_ids=["999999", "24937"],
                start_date="2023-01-01",
                end_date="2023-03-31",
            )

        # Should log that the firm was skipped
        assert any("999999" in record.message or "Unlinked" in record.message
                   for record in caplog.records)


# =============================================================================
# TestWRDSConnectorSentenceProcessing
# =============================================================================


class TestWRDSConnectorSentenceProcessing:
    """Tests for sentence splitting and preprocessing."""

    def test_sentences_have_correct_structure(self, mock_wrds_dataframe):
        """Sentences should have required fields."""
        from cloud.src.connectors.wrds_connector import WRDSConnector
        from cloud.src.models import TranscriptSentence

        mock_conn = MagicMock()
        mock_conn.raw_sql.return_value = mock_wrds_dataframe

        connector = WRDSConnector(connection=mock_conn)
        result = connector.fetch_transcripts(
            firm_ids=["374372246"],
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        sentence = result.firms["374372246"].sentences[0]
        assert isinstance(sentence, TranscriptSentence)
        assert sentence.sentence_id is not None
        assert sentence.raw_text is not None
        assert sentence.cleaned_text is not None
        assert sentence.position is not None

    def test_sentences_have_raw_text(self, mock_wrds_dataframe):
        """Sentences should preserve raw text."""
        from cloud.src.connectors.wrds_connector import WRDSConnector

        mock_conn = MagicMock()
        mock_conn.raw_sql.return_value = mock_wrds_dataframe

        connector = WRDSConnector(connection=mock_conn)
        result = connector.fetch_transcripts(
            firm_ids=["374372246"],
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        sentences = result.firms["374372246"].sentences
        # At least one sentence should have non-empty raw_text
        raw_texts = [s.raw_text for s in sentences]
        assert any(len(t) > 0 for t in raw_texts)

    def test_sentences_have_cleaned_text(self, mock_wrds_dataframe):
        """Sentences should have cleaned/preprocessed text."""
        from cloud.src.connectors.wrds_connector import WRDSConnector

        mock_conn = MagicMock()
        mock_conn.raw_sql.return_value = mock_wrds_dataframe

        connector = WRDSConnector(connection=mock_conn)
        result = connector.fetch_transcripts(
            firm_ids=["374372246"],
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        sentences = result.firms["374372246"].sentences
        # All sentences should have cleaned_text
        for sentence in sentences:
            assert sentence.cleaned_text is not None
            assert len(sentence.cleaned_text) > 0

    def test_sentences_ordered_by_component_order(self, mock_wrds_dataframe):
        """Sentences should be ordered by componentorder."""
        from cloud.src.connectors.wrds_connector import WRDSConnector

        mock_conn = MagicMock()
        mock_conn.raw_sql.return_value = mock_wrds_dataframe

        connector = WRDSConnector(connection=mock_conn)
        result = connector.fetch_transcripts(
            firm_ids=["374372246"],
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        sentences = result.firms["374372246"].sentences
        positions = [s.position for s in sentences]

        # Positions should be monotonically increasing
        for i in range(1, len(positions)):
            assert positions[i] > positions[i - 1]

    def test_sentence_id_format(self, mock_wrds_dataframe):
        """sentence_id should follow format: {firm_id}_{transcript_id}_{position:04d}"""
        from cloud.src.connectors.wrds_connector import WRDSConnector

        mock_conn = MagicMock()
        mock_conn.raw_sql.return_value = mock_wrds_dataframe

        connector = WRDSConnector(connection=mock_conn)
        result = connector.fetch_transcripts(
            firm_ids=["374372246"],
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        sentence = result.firms["374372246"].sentences[0]
        parts = sentence.sentence_id.split("_")
        assert len(parts) >= 3
        assert parts[0] == "374372246"  # firm_id

    def test_splits_multi_sentence_components(self, mock_wrds_dataframe):
        """Should split components with multiple sentences."""
        from cloud.src.connectors.wrds_connector import WRDSConnector

        mock_conn = MagicMock()
        mock_conn.raw_sql.return_value = mock_wrds_dataframe

        connector = WRDSConnector(connection=mock_conn)
        result = connector.fetch_transcripts(
            firm_ids=["374372246"],
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        sentences = result.firms["374372246"].sentences
        # Component 1 has 2 sentences: "Good morning everyone." and "Welcome to the earnings call."
        # We should have more sentences than components (3 components in mock)
        assert len(sentences) > 1

    def test_filters_operator_speaker_type(self, mock_wrds_dataframe):
        """Should filter out Operator speaker type sentences."""
        from cloud.src.connectors.wrds_connector import WRDSConnector

        mock_conn = MagicMock()
        mock_conn.raw_sql.return_value = mock_wrds_dataframe

        connector = WRDSConnector(connection=mock_conn)
        result = connector.fetch_transcripts(
            firm_ids=["374372246"],
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        sentences = result.firms["374372246"].sentences
        # No sentence should have speaker_type == "Operator"
        for sentence in sentences:
            if sentence.speaker_type:
                assert sentence.speaker_type.lower() != "operator"

    def test_removes_stopwords(self, mock_wrds_dataframe):
        """Should remove common stopwords from cleaned_text."""
        from cloud.src.connectors.wrds_connector import WRDSConnector

        mock_conn = MagicMock()
        mock_conn.raw_sql.return_value = mock_wrds_dataframe

        connector = WRDSConnector(connection=mock_conn)
        result = connector.fetch_transcripts(
            firm_ids=["374372246"],
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        sentences = result.firms["374372246"].sentences
        common_stopwords = {"the", "a", "an", "is", "are", "was", "were"}

        for sentence in sentences:
            words = set(sentence.cleaned_text.lower().split())
            stopwords_found = common_stopwords.intersection(words)
            assert len(stopwords_found) == 0, f"Stopwords found: {stopwords_found}"

    def test_filters_single_word_sentences(self, mock_wrds_dataframe):
        """Should filter out sentences with only 1 word after preprocessing."""
        from cloud.src.connectors.wrds_connector import WRDSConnector

        mock_conn = MagicMock()
        mock_conn.raw_sql.return_value = mock_wrds_dataframe

        connector = WRDSConnector(connection=mock_conn)
        result = connector.fetch_transcripts(
            firm_ids=["374372246"],
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        sentences = result.firms["374372246"].sentences
        for sentence in sentences:
            word_count = len(sentence.cleaned_text.split())
            assert word_count > 1, f"Single-word sentence found: {sentence.cleaned_text}"

    def test_positions_are_sequential(self, mock_wrds_dataframe):
        """Positions should be sequential 0, 1, 2, ... across all sentences."""
        from cloud.src.connectors.wrds_connector import WRDSConnector

        mock_conn = MagicMock()
        mock_conn.raw_sql.return_value = mock_wrds_dataframe

        connector = WRDSConnector(connection=mock_conn)
        result = connector.fetch_transcripts(
            firm_ids=["374372246"],
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        sentences = result.firms["374372246"].sentences
        positions = [s.position for s in sentences]

        # Should be sequential: 0, 1, 2, 3, ...
        expected = list(range(len(sentences)))
        assert positions == expected


# =============================================================================
# TestWRDSConnectorGetAvailableFirmIds
# =============================================================================


class TestWRDSConnectorGetAvailableFirmIds:
    """Tests for get_available_firm_ids method."""

    def test_returns_list_of_strings(self):
        """get_available_firm_ids should return list of firm ID strings."""
        from cloud.src.connectors.wrds_connector import WRDSConnector

        mock_conn = MagicMock()
        mock_conn.raw_sql.return_value = pd.DataFrame({
            "firm_id": ["1001", "1002", "1003"]
        })

        connector = WRDSConnector(connection=mock_conn)
        firm_ids = connector.get_available_firm_ids()

        assert isinstance(firm_ids, list)
        assert all(isinstance(fid, str) for fid in firm_ids)

    def test_returns_sorted_unique_ids(self):
        """Should return sorted, unique firm IDs."""
        from cloud.src.connectors.wrds_connector import WRDSConnector

        mock_conn = MagicMock()
        mock_conn.raw_sql.return_value = pd.DataFrame({
            "firm_id": ["1003", "1001", "1002", "1001"]  # Unsorted, with duplicate
        })

        connector = WRDSConnector(connection=mock_conn)
        firm_ids = connector.get_available_firm_ids()

        assert firm_ids == ["1001", "1002", "1003"]


# =============================================================================
# TestWRDSConnectorErrorHandling
# =============================================================================


class TestWRDSConnectorErrorHandling:
    """Tests for error handling."""

    def test_connection_failure_raises_wrds_connection_error(self):
        """Should raise WRDSConnectionError when connection fails."""
        from cloud.src.connectors.wrds_connector import WRDSConnector, WRDSConnectionError
        import sys

        # Create a mock wrds module that raises on Connection()
        mock_wrds_module = MagicMock()
        mock_wrds_module.Connection.side_effect = Exception("Connection refused")

        # Patch sys.modules so the lazy import gets our mock
        with patch.dict(sys.modules, {'wrds': mock_wrds_module}):
            connector = WRDSConnector()

            with pytest.raises(WRDSConnectionError):
                connector.fetch_transcripts(
                    firm_ids=["1001"],
                    start_date="2023-01-01",
                    end_date="2023-03-31",
                )

    def test_query_failure_raises_wrds_query_error(self):
        """Should raise WRDSQueryError when SQL query fails."""
        from cloud.src.connectors.wrds_connector import WRDSConnector, WRDSQueryError

        mock_conn = MagicMock()
        mock_conn.raw_sql.side_effect = Exception("SQL syntax error")

        connector = WRDSConnector(connection=mock_conn)

        with pytest.raises(WRDSQueryError):
            connector.fetch_transcripts(
                firm_ids=["1001"],
                start_date="2023-01-01",
                end_date="2023-03-31",
            )

    def test_invalid_date_format_raises_value_error(self):
        """Should raise ValueError for invalid date format."""
        from cloud.src.connectors.wrds_connector import WRDSConnector

        mock_conn = MagicMock()
        connector = WRDSConnector(connection=mock_conn)

        with pytest.raises(ValueError):
            connector.fetch_transcripts(
                firm_ids=["1001"],
                start_date="01-01-2023",  # Wrong format
                end_date="2023-03-31",
            )


# =============================================================================
# TestWRDSConnectorClose
# =============================================================================


class TestWRDSConnectorClose:
    """Tests for close method."""

    def test_close_owned_connection(self):
        """Should close connection when owned by this instance."""
        from cloud.src.connectors.wrds_connector import WRDSConnector

        mock_conn = MagicMock()

        # Create connector and manually set connection (simulating lazy init completed)
        connector = WRDSConnector()
        connector._conn = mock_conn
        connector._owns_connection = True

        connector.close()

        mock_conn.close.assert_called_once()

    def test_close_does_not_close_passed_connection(self):
        """Should NOT close connection when passed in (caller owns it)."""
        from cloud.src.connectors.wrds_connector import WRDSConnector

        mock_conn = MagicMock()
        connector = WRDSConnector(connection=mock_conn)

        connector.close()

        # Connection should NOT be closed (caller owns it)
        mock_conn.close.assert_not_called()


# =============================================================================
# TestWRDSConnectorEmptyFirmIds
# =============================================================================


class TestWRDSConnectorEmptyFirmIds:
    """Tests for empty firm_ids behavior."""

    def test_empty_firm_ids_fetches_all(self, mock_wrds_dataframe):
        """Empty firm_ids list should fetch all firms in date range."""
        from cloud.src.connectors.wrds_connector import WRDSConnector

        mock_conn = MagicMock()
        mock_conn.raw_sql.return_value = mock_wrds_dataframe

        connector = WRDSConnector(connection=mock_conn)
        result = connector.fetch_transcripts(
            firm_ids=[],  # Empty list = fetch all
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        # Should have both firms from mock data
        assert "374372246" in result.firms
        assert "24937" in result.firms

    def test_empty_firm_ids_passes_none_to_sql(self, mock_wrds_dataframe):
        """Empty firm_ids should pass None to SQL (for ANY pattern)."""
        from cloud.src.connectors.wrds_connector import WRDSConnector

        mock_conn = MagicMock()
        mock_conn.raw_sql.return_value = mock_wrds_dataframe

        connector = WRDSConnector(connection=mock_conn)
        connector.fetch_transcripts(
            firm_ids=[],
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        # Verify firm_ids parameter is None
        call_args = mock_conn.raw_sql.call_args
        params = call_args[1].get('params') or call_args[1]
        assert params.get('firm_ids') is None
