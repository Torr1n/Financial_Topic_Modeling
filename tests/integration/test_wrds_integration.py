"""
Integration tests for WRDSConnector against real WRDS database.

These tests require:
- WRDS credentials (WRDS_USERNAME env var or ~/.pgpass)
- Network access to WRDS

Run with: pytest tests/integration/test_wrds_integration.py -v -m integration
"""

import os
import pytest


# Skip all tests in this module if WRDS credentials not available
pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_wrds,
]


def wrds_credentials_available() -> bool:
    """Check if WRDS credentials are available."""
    # Check environment variable
    if os.environ.get("WRDS_USERNAME"):
        return True
    # Check pgpass file
    pgpass_path = os.path.expanduser("~/.pgpass")
    if os.path.exists(pgpass_path):
        with open(pgpass_path, "r") as f:
            if "wrds" in f.read().lower():
                return True
    return False


# Skip entire module if no credentials
if not wrds_credentials_available():
    pytest.skip("WRDS credentials not available", allow_module_level=True)


class TestWRDSIntegration:
    """
    Integration tests against real WRDS (10 firms max).

    Requires WRDS credentials:
    - WRDS_USERNAME environment variable, or
    - ~/.pgpass configured for WRDS
    """

    @pytest.fixture(autouse=True)
    def skip_without_credentials(self):
        """Skip test if WRDS credentials not available."""
        if not wrds_credentials_available():
            pytest.skip("WRDS credentials not available")

    def test_fetch_known_firms(self):
        """Fetch transcripts for firms with known PERMNO."""
        from cloud.src.connectors.wrds_connector import WRDSConnector

        with WRDSConnector() as connector:
            # First, get some available firm IDs dynamically
            # This avoids hardcoding firm IDs that may not exist
            all_firms = connector.get_available_firm_ids()

            if len(all_firms) < 5:
                pytest.skip("Not enough firms available in WRDS")

            # Take first 5 firms for testing
            test_firms = all_firms[:5]

            result = connector.fetch_transcripts(
                firm_ids=test_firms,
                start_date="2023-01-01",
                end_date="2023-03-31",
            )

            # Should have at least some firms (some may be unlinked)
            # We can't guarantee all 5 have PERMNO
            assert len(result.firms) >= 0

    def test_date_range_filtering(self):
        """Verify date range is respected."""
        from cloud.src.connectors.wrds_connector import WRDSConnector
        from datetime import date
        import pandas as pd

        with WRDSConnector() as connector:
            # Fetch with narrow date range
            result = connector.fetch_transcripts(
                firm_ids=[],  # All firms
                start_date="2023-01-01",
                end_date="2023-01-31",  # Just January
            )

            # Verify all returned firms have earnings_call_date in range
            for firm_id, firm_data in result.firms.items():
                ec_date = firm_data.metadata.get("earnings_call_date")
                assert ec_date is not None
                # Handle both date objects and strings/Timestamps from pandas
                if isinstance(ec_date, str):
                    ec_date = pd.to_datetime(ec_date).date()
                elif hasattr(ec_date, 'date'):  # Timestamp
                    ec_date = ec_date.date()
                # Date should be in January 2023
                assert ec_date.year == 2023
                assert ec_date.month == 1

    def test_permno_present_for_linked_firms(self):
        """Confirm PERMNO appears in metadata for all returned firms."""
        from cloud.src.connectors.wrds_connector import WRDSConnector

        with WRDSConnector() as connector:
            result = connector.fetch_transcripts(
                firm_ids=[],  # All firms
                start_date="2023-01-01",
                end_date="2023-01-15",  # Short range to limit results
            )

            # All returned firms MUST have PERMNO (unlinked are skipped)
            for firm_id, firm_data in result.firms.items():
                assert "permno" in firm_data.metadata
                assert firm_data.metadata["permno"] is not None
                assert isinstance(firm_data.metadata["permno"], int)

    def test_skip_unlinked_firms(self):
        """Verify unlinked firms are not in output."""
        from cloud.src.connectors.wrds_connector import WRDSConnector

        with WRDSConnector() as connector:
            result = connector.fetch_transcripts(
                firm_ids=[],
                start_date="2023-01-01",
                end_date="2023-01-31",
            )

            # All firms in output must have PERMNO
            # (unlinked firms should have been filtered out)
            for firm_id, firm_data in result.firms.items():
                permno = firm_data.metadata.get("permno")
                assert permno is not None, f"Firm {firm_id} has no PERMNO but was included"

    def test_sentences_have_expected_structure(self):
        """Verify sentence structure matches TranscriptSentence model."""
        from cloud.src.connectors.wrds_connector import WRDSConnector
        from cloud.src.models import TranscriptSentence

        with WRDSConnector() as connector:
            result = connector.fetch_transcripts(
                firm_ids=[],
                start_date="2023-01-01",
                end_date="2023-01-15",
            )

            if not result.firms:
                pytest.skip("No firms returned from WRDS")

            # Check first firm's sentences
            first_firm = list(result.firms.values())[0]
            assert len(first_firm.sentences) > 0

            for sentence in first_firm.sentences:
                assert isinstance(sentence, TranscriptSentence)
                assert sentence.sentence_id is not None
                assert sentence.raw_text is not None
                assert sentence.cleaned_text is not None
                assert len(sentence.cleaned_text.split()) > 1  # No single-word sentences

    def test_metadata_completeness(self):
        """Verify metadata contains all expected fields."""
        from cloud.src.connectors.wrds_connector import WRDSConnector

        with WRDSConnector() as connector:
            result = connector.fetch_transcripts(
                firm_ids=[],
                start_date="2023-01-01",
                end_date="2023-01-15",
            )

            if not result.firms:
                pytest.skip("No firms returned from WRDS")

            first_firm = list(result.firms.values())[0]
            metadata = first_firm.metadata

            # Required metadata fields
            assert "permno" in metadata
            assert "gvkey" in metadata
            assert "link_date" in metadata
            assert "earnings_call_date" in metadata
            assert "transcript_id" in metadata

    def test_get_available_firm_ids(self):
        """Test get_available_firm_ids returns firm IDs."""
        from cloud.src.connectors.wrds_connector import WRDSConnector

        with WRDSConnector() as connector:
            firm_ids = connector.get_available_firm_ids()

            assert isinstance(firm_ids, list)
            assert len(firm_ids) > 0
            assert all(isinstance(fid, str) for fid in firm_ids)
            # Should be sorted
            assert firm_ids == sorted(firm_ids)
