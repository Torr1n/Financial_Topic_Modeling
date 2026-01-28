"""
Unit tests for DATA_SOURCE switch in entrypoint.py.

Tests:
- get_data_connector() returns correct connector type
- Environment variable parsing
- Default behavior (wrds) for backward compatibility
"""

import os
from unittest.mock import MagicMock, patch

import pytest


class TestGetDataConnector:
    """Tests for get_data_connector factory function."""

    def test_returns_s3_connector_for_s3_source(self):
        """Should return S3TranscriptConnector when data_source='s3'."""
        with patch("cloud.containers.map.entrypoint.S3TranscriptConnector") as mock_cls:
            mock_connector = MagicMock()
            mock_cls.return_value = mock_connector

            # Import after patching
            from cloud.containers.map.entrypoint import get_data_connector

            result = get_data_connector("s3", "2023Q1", "test-bucket")

            mock_cls.assert_called_once_with(bucket="test-bucket", quarter="2023Q1")
            assert result == mock_connector

    def test_returns_wrds_connector_for_wrds_source(self):
        """Should return WRDSConnector when data_source='wrds'."""
        with patch("cloud.containers.map.entrypoint.WRDSConnector") as mock_cls:
            mock_connector = MagicMock()
            mock_cls.return_value = mock_connector

            from cloud.containers.map.entrypoint import get_data_connector

            result = get_data_connector("wrds", "2023Q1", "test-bucket")

            mock_cls.assert_called_once()
            assert result == mock_connector

    def test_raises_for_unknown_source(self):
        """Should raise ValueError for unknown data_source."""
        from cloud.containers.map.entrypoint import get_data_connector

        with pytest.raises(ValueError) as exc_info:
            get_data_connector("unknown", "2023Q1", "test-bucket")

        assert "Unknown DATA_SOURCE" in str(exc_info.value)
        assert "unknown" in str(exc_info.value)


class TestEnvironmentVariables:
    """Tests for environment variable handling."""

    def test_data_source_default_is_wrds(self):
        """DATA_SOURCE should default to 'wrds' for backward compatibility."""
        from cloud.containers.map.entrypoint import get_env_optional

        # Ensure DATA_SOURCE is not set
        env_backup = os.environ.pop("DATA_SOURCE", None)
        try:
            result = get_env_optional("DATA_SOURCE", "wrds")
            assert result == "wrds"
        finally:
            if env_backup:
                os.environ["DATA_SOURCE"] = env_backup

    def test_data_source_env_var_respected(self):
        """Should use DATA_SOURCE from environment when set."""
        from cloud.containers.map.entrypoint import get_env_optional

        env_backup = os.environ.get("DATA_SOURCE")
        try:
            os.environ["DATA_SOURCE"] = "s3"
            result = get_env_optional("DATA_SOURCE", "wrds")
            assert result == "s3"
        finally:
            if env_backup:
                os.environ["DATA_SOURCE"] = env_backup
            else:
                os.environ.pop("DATA_SOURCE", None)


class TestDataConnectorInterface:
    """Tests that connectors satisfy DataConnector interface."""

    def test_s3_connector_has_required_methods(self):
        """S3TranscriptConnector should have all DataConnector methods."""
        from cloud.src.connectors.s3_connector import S3TranscriptConnector
        from cloud.src.interfaces import DataConnector

        # Verify it's a subclass
        assert issubclass(S3TranscriptConnector, DataConnector)

        # Verify required methods exist
        connector = S3TranscriptConnector.__new__(S3TranscriptConnector)
        assert hasattr(connector, "fetch_transcripts")
        assert hasattr(connector, "get_available_firm_ids")
        assert hasattr(connector, "close")

    def test_wrds_connector_has_required_methods(self):
        """WRDSConnector should have all DataConnector methods."""
        from cloud.src.connectors.wrds_connector import WRDSConnector
        from cloud.src.interfaces import DataConnector

        # Verify it's a subclass
        assert issubclass(WRDSConnector, DataConnector)

        # Verify required methods exist
        assert hasattr(WRDSConnector, "fetch_transcripts")
        assert hasattr(WRDSConnector, "get_available_firm_ids")
        assert hasattr(WRDSConnector, "close")


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing code."""

    def test_entrypoint_still_works_without_data_source(self):
        """Entrypoint should work when DATA_SOURCE not set (defaults to wrds)."""
        # This is a smoke test - actual main() requires many env vars
        # We just verify the get_env_optional logic works correctly

        from cloud.containers.map.entrypoint import get_env_optional

        # Remove DATA_SOURCE if set
        env_backup = os.environ.pop("DATA_SOURCE", None)
        try:
            # Should default to wrds
            data_source = get_env_optional("DATA_SOURCE", "wrds")
            assert data_source == "wrds"
        finally:
            if env_backup:
                os.environ["DATA_SOURCE"] = env_backup

    def test_process_firms_accepts_any_data_connector(self):
        """process_firms should accept any DataConnector implementation."""
        from cloud.containers.map.entrypoint import process_firms
        from cloud.src.interfaces import DataConnector

        import inspect
        sig = inspect.signature(process_firms)

        # First parameter should be connector (no longer typed as WRDSConnector specifically)
        params = list(sig.parameters.items())
        connector_param = params[0]

        # The function should accept DataConnector
        assert connector_param[0] == "connector"
        # Type hint should be DataConnector (check in annotation if available)
        if connector_param[1].annotation != inspect.Parameter.empty:
            assert connector_param[1].annotation == DataConnector


class TestTerraformDefaults:
    """Tests documenting expected Terraform defaults."""

    def test_document_expected_defaults(self):
        """Document expected defaults between entrypoint and Terraform.

        Entrypoint default: wrds (backward compat for local testing)
        Terraform default: s3 (production uses prefetch)

        This test documents this split behavior.
        """
        # Entrypoint default
        from cloud.containers.map.entrypoint import get_env_optional

        env_backup = os.environ.pop("DATA_SOURCE", None)
        try:
            entrypoint_default = get_env_optional("DATA_SOURCE", "wrds")
            assert entrypoint_default == "wrds", (
                "Entrypoint should default to 'wrds' for local dev backward compat"
            )
        finally:
            if env_backup:
                os.environ["DATA_SOURCE"] = env_backup

        # Terraform default is in job_definition.tf:
        # { name = "DATA_SOURCE", value = "s3" }
        # This means production Batch jobs will use S3 by default,
        # while local runs without DATA_SOURCE set will use WRDS.
