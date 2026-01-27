"""
Tests for consolidated utility functions.

These tests use mocking to avoid requiring actual WRDS credentials or file I/O.
"""
import sys
import logging
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pandas as pd
import pytest


class TestSetupLogging:
    """Test suite for setup_logging utility function."""

    def test_setup_logging_configures_console_handler(self):
        """setup_logging should configure a console handler with correct level."""
        from src.utils import setup_logging

        # Clear any existing handlers
        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers[:]

        try:
            setup_logging(log_level='DEBUG')

            # Check that at least one StreamHandler was configured
            stream_handlers = [h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)]
            assert len(stream_handlers) > 0, "Should have at least one StreamHandler"

            # Check the logging level
            assert root_logger.level == logging.DEBUG

        finally:
            # Restore original handlers
            root_logger.handlers = original_handlers

    def test_setup_logging_with_file_handler(self):
        """setup_logging should configure a file handler when log_file specified."""
        from src.utils import setup_logging

        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers[:]

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                log_file = Path(tmpdir) / 'test.log'
                setup_logging(log_level='INFO', log_file=str(log_file))

                # Check that a FileHandler was configured
                file_handlers = [h for h in root_logger.handlers if isinstance(h, logging.FileHandler)]
                assert len(file_handlers) > 0, "Should have at least one FileHandler"

        finally:
            # Close all handlers and restore original
            for handler in root_logger.handlers:
                handler.close()
            root_logger.handlers = original_handlers


class TestLoadSentimentScores:
    """Test suite for load_sentiment_scores utility function."""

    def test_load_sentiment_scores_valid_csv(self):
        """load_sentiment_scores should load and validate a proper CSV."""
        from src.utils import load_sentiment_scores

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / 'sentiment.csv'
            df = pd.DataFrame({
                'permno': [12345, 67890],
                'edate': ['2023-01-15', '2023-01-16'],
                'sentiment': [0.5, -0.3],
                'theme_id': ['theme_1', 'theme_2'],
                'theme_name': ['Theme One', 'Theme Two']
            })
            df.to_csv(csv_path, index=False)

            result = load_sentiment_scores(str(csv_path))

            assert len(result) == 2
            assert 'permno' in result.columns
            assert 'edate' in result.columns
            assert 'sentiment' in result.columns

    def test_load_sentiment_scores_missing_columns_raises(self):
        """load_sentiment_scores should raise ValueError when required columns missing."""
        from src.utils import load_sentiment_scores

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / 'bad_sentiment.csv'
            df = pd.DataFrame({
                'permno': [12345],
                'edate': ['2023-01-15']
                # Missing 'sentiment' column
            })
            df.to_csv(csv_path, index=False)

            with pytest.raises(ValueError, match="Missing required columns"):
                load_sentiment_scores(str(csv_path))


class TestGroupByTheme:
    """Test suite for group_by_theme utility function."""

    def test_group_by_theme_with_theme_columns(self):
        """group_by_theme should group data correctly when theme columns exist."""
        from src.utils import group_by_theme

        df = pd.DataFrame({
            'permno': [12345, 12345, 67890],
            'edate': ['2023-01-15', '2023-01-15', '2023-01-16'],
            'sentiment': [0.5, 0.3, -0.2],
            'theme_id': ['theme_1', 'theme_2', 'theme_1'],
            'theme_name': ['Theme One', 'Theme Two', 'Theme One']
        })

        result = group_by_theme(df)

        assert 'theme_1' in result
        assert 'theme_2' in result
        assert result['theme_1']['theme_name'] == 'Theme One'
        assert len(result['theme_1']['events']) == 2
        assert len(result['theme_2']['events']) == 1

    def test_group_by_theme_without_theme_columns(self):
        """group_by_theme should create single 'all' group when no theme columns."""
        from src.utils import group_by_theme

        df = pd.DataFrame({
            'permno': [12345, 67890],
            'edate': ['2023-01-15', '2023-01-16'],
            'sentiment': [0.5, -0.2]
        })

        result = group_by_theme(df)

        assert 'all' in result
        assert result['all']['theme_name'] == 'all_themes'
        assert len(result['all']['events']) == 2


class TestRunBatchedEventStudy:
    """Test suite for run_batched_event_study utility function."""

    def test_run_batched_event_study_returns_results(self):
        """run_batched_event_study should return DataFrame from ThematicES."""
        # Mock the ThematicES class
        mock_results = pd.DataFrame({
            'permno': [12345, 67890],
            'edate': ['2023-01-15', '2023-01-16'],
            'car': [0.02, -0.01],
            'sentiment': [0.5, -0.3]
        })

        mock_study = MagicMock()
        mock_study.doAll.return_value = mock_results

        # Create a mock event_study module with the ThematicES class
        mock_event_study_module = MagicMock()
        mock_event_study_module.ThematicES = MagicMock(return_value=mock_study)

        # Patch sys.modules to inject our mock module
        with patch.dict('sys.modules', {'src.event_study': mock_event_study_module}):
            # Re-import to get fresh version that uses the mock
            import importlib
            import src.utils
            importlib.reload(src.utils)

            from src.utils import run_batched_event_study

            events = [
                {'permno': 12345, 'edate': '2023-01-15', 'sentiment': 0.5},
                {'permno': 67890, 'edate': '2023-01-16', 'sentiment': -0.3}
            ]
            mock_conn = Mock()

            result = run_batched_event_study(events, wrds_conn=mock_conn)

            assert result is not None
            assert len(result) == 2
            mock_study.doAll.assert_called_once()

    def test_run_batched_event_study_handles_failure(self):
        """run_batched_event_study should return None on failure."""
        mock_study = MagicMock()
        mock_study.doAll.side_effect = Exception("WRDS query failed")

        mock_conn = Mock()

        # Create a mock event_study module with the ThematicES class
        mock_event_study_module = MagicMock()
        mock_event_study_module.ThematicES = MagicMock(return_value=mock_study)

        # Patch sys.modules to inject our mock module
        with patch.dict('sys.modules', {'src.event_study': mock_event_study_module}):
            # Re-import to get fresh version that uses the mock
            import importlib
            import src.utils
            importlib.reload(src.utils)

            from src.utils import run_batched_event_study

            events = [{'permno': 12345, 'edate': '2023-01-15', 'sentiment': 0.5}]

            result = run_batched_event_study(events, wrds_conn=mock_conn)

            assert result is None
