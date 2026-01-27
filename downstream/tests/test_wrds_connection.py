"""
Tests for WRDS connection context manager.

These tests use mocking to avoid requiring actual WRDS credentials.
"""
import sys
import pytest
from unittest.mock import Mock, patch, MagicMock


class TestWRDSConnection:
    """Test suite for WRDSConnection context manager."""

    def test_external_connection_not_closed(self):
        """Context manager should not close an externally provided connection."""
        from src.wrds_connection import WRDSConnection

        # Create a mock external connection
        external_conn = Mock()
        external_conn.close = Mock()

        # Use context manager with external connection
        with WRDSConnection(connection=external_conn) as conn:
            assert conn is external_conn

        # External connection should NOT be closed
        external_conn.close.assert_not_called()

    def test_new_connection_closed_on_exit(self):
        """Context manager should close connections it creates."""
        # Create mock wrds module
        mock_wrds = MagicMock()
        mock_connection = Mock()
        mock_wrds.Connection.return_value = mock_connection

        # Patch the wrds module in sys.modules before importing
        with patch.dict(sys.modules, {'wrds': mock_wrds}):
            from src.wrds_connection import WRDSConnection

            # Use context manager without external connection
            with WRDSConnection() as conn:
                assert conn is mock_connection

            # Connection should be closed since we created it
            mock_connection.close.assert_called_once()

    def test_exception_handling_closes_connection(self):
        """Context manager should close self-created connection even on exception."""
        # Create mock wrds module
        mock_wrds = MagicMock()
        mock_connection = Mock()
        mock_wrds.Connection.return_value = mock_connection

        # Patch the wrds module in sys.modules before importing
        with patch.dict(sys.modules, {'wrds': mock_wrds}):
            from src.wrds_connection import WRDSConnection

            # Use context manager and raise an exception
            with pytest.raises(ValueError):
                with WRDSConnection() as conn:
                    raise ValueError("Test exception")

            # Connection should still be closed
            mock_connection.close.assert_called_once()

    def test_exception_handling_does_not_close_external(self):
        """Context manager should not close external connection even on exception."""
        from src.wrds_connection import WRDSConnection

        # Create a mock external connection
        external_conn = Mock()
        external_conn.close = Mock()

        # Use context manager with external connection and raise exception
        with pytest.raises(ValueError):
            with WRDSConnection(connection=external_conn) as conn:
                raise ValueError("Test exception")

        # External connection should NOT be closed
        external_conn.close.assert_not_called()
