"""
Unit tests for CheckpointManager.

Tests the checkpoint/resume logic for spot instance interruption handling.
"""

import pytest
from unittest.mock import MagicMock


class TestCheckpointManagerInit:
    """Tests for CheckpointManager initialization."""

    def test_accepts_repository(self):
        """CheckpointManager should accept a DatabaseRepository."""
        from cloud.src.pipeline.checkpoint import CheckpointManager

        mock_repo = MagicMock()
        manager = CheckpointManager(mock_repo)

        assert manager.repo is mock_repo


class TestCheckpointManagerGetUnprocessed:
    """Tests for get_unprocessed_firm_ids method."""

    def test_returns_unprocessed_firms(self):
        """Should return firm IDs that haven't been processed."""
        from cloud.src.pipeline.checkpoint import CheckpointManager

        mock_repo = MagicMock()
        mock_repo.get_processed_firm_ids.return_value = ["1001", "1002"]

        manager = CheckpointManager(mock_repo)

        all_firms = ["1001", "1002", "1003", "1004"]
        unprocessed = manager.get_unprocessed_firm_ids(all_firms)

        assert unprocessed == ["1003", "1004"]

    def test_returns_all_when_none_processed(self):
        """Should return all firms when none are processed."""
        from cloud.src.pipeline.checkpoint import CheckpointManager

        mock_repo = MagicMock()
        mock_repo.get_processed_firm_ids.return_value = []

        manager = CheckpointManager(mock_repo)

        all_firms = ["1001", "1002", "1003"]
        unprocessed = manager.get_unprocessed_firm_ids(all_firms)

        assert unprocessed == ["1001", "1002", "1003"]

    def test_returns_empty_when_all_processed(self):
        """Should return empty list when all firms are processed."""
        from cloud.src.pipeline.checkpoint import CheckpointManager

        mock_repo = MagicMock()
        mock_repo.get_processed_firm_ids.return_value = ["1001", "1002", "1003"]

        manager = CheckpointManager(mock_repo)

        all_firms = ["1001", "1002", "1003"]
        unprocessed = manager.get_unprocessed_firm_ids(all_firms)

        assert unprocessed == []

    def test_preserves_order(self):
        """Unprocessed firms should maintain original order."""
        from cloud.src.pipeline.checkpoint import CheckpointManager

        mock_repo = MagicMock()
        mock_repo.get_processed_firm_ids.return_value = ["1002"]

        manager = CheckpointManager(mock_repo)

        all_firms = ["1001", "1002", "1003", "1004"]
        unprocessed = manager.get_unprocessed_firm_ids(all_firms)

        assert unprocessed == ["1001", "1003", "1004"]


class TestCheckpointManagerCheckpoint:
    """Tests for checkpoint method."""

    def test_marks_firm_processed(self):
        """Checkpoint should mark firm as processed."""
        from cloud.src.pipeline.checkpoint import CheckpointManager

        mock_repo = MagicMock()
        mock_session = MagicMock()

        manager = CheckpointManager(mock_repo)
        manager.checkpoint(123, mock_session)

        mock_repo.mark_firm_processed.assert_called_once_with(123)

    def test_commits_session(self):
        """Checkpoint should commit the session."""
        from cloud.src.pipeline.checkpoint import CheckpointManager

        mock_repo = MagicMock()
        mock_session = MagicMock()

        manager = CheckpointManager(mock_repo)
        manager.checkpoint(123, mock_session)

        mock_session.commit.assert_called_once()

    def test_mark_then_commit_order(self):
        """Mark should be called before commit."""
        from cloud.src.pipeline.checkpoint import CheckpointManager

        mock_repo = MagicMock()
        mock_session = MagicMock()

        call_order = []
        mock_repo.mark_firm_processed.side_effect = lambda x: call_order.append("mark")
        mock_session.commit.side_effect = lambda: call_order.append("commit")

        manager = CheckpointManager(mock_repo)
        manager.checkpoint(123, mock_session)

        assert call_order == ["mark", "commit"]


class TestCheckpointManagerGetResumePoint:
    """Tests for get_resume_point method."""

    def test_returns_first_unprocessed_index(self):
        """Should return index of first unprocessed firm."""
        from cloud.src.pipeline.checkpoint import CheckpointManager

        mock_repo = MagicMock()
        mock_repo.get_processed_firm_ids.return_value = ["1001", "1002"]

        manager = CheckpointManager(mock_repo)

        all_firms = ["1001", "1002", "1003", "1004"]
        resume_point = manager.get_resume_point(all_firms)

        assert resume_point == 2  # Index of "1003"

    def test_returns_zero_when_none_processed(self):
        """Should return 0 when no firms are processed."""
        from cloud.src.pipeline.checkpoint import CheckpointManager

        mock_repo = MagicMock()
        mock_repo.get_processed_firm_ids.return_value = []

        manager = CheckpointManager(mock_repo)

        all_firms = ["1001", "1002", "1003"]
        resume_point = manager.get_resume_point(all_firms)

        assert resume_point == 0

    def test_returns_length_when_all_processed(self):
        """Should return list length when all firms are processed."""
        from cloud.src.pipeline.checkpoint import CheckpointManager

        mock_repo = MagicMock()
        mock_repo.get_processed_firm_ids.return_value = ["1001", "1002", "1003"]

        manager = CheckpointManager(mock_repo)

        all_firms = ["1001", "1002", "1003"]
        resume_point = manager.get_resume_point(all_firms)

        assert resume_point == 3  # len(all_firms)
