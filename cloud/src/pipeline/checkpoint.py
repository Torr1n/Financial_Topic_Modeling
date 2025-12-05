"""
CheckpointManager - Resume logic for spot instance interruption.

This is a thin wrapper around DatabaseRepository methods that provides
a clean interface for checkpoint/resume functionality.

The unified pipeline calls checkpoint() after each firm is processed,
enabling resume from the last successful checkpoint if the spot instance
is interrupted.
"""

import logging
from typing import List

from sqlalchemy.orm import Session

from cloud.src.database.repository import DatabaseRepository

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Thin wrapper for checkpoint/resume logic.

    Provides a clean interface for:
    - Getting unprocessed firm IDs (resume point)
    - Marking firms as processed (checkpoint)
    - Committing checkpoints to database

    Usage:
        manager = CheckpointManager(repo)
        unprocessed = manager.get_unprocessed_firm_ids(all_firm_ids)
        for firm_id in unprocessed:
            process_firm(firm_id)
            manager.checkpoint(firm.id, session)
    """

    def __init__(self, repo: DatabaseRepository):
        """
        Initialize CheckpointManager.

        Args:
            repo: DatabaseRepository instance for database operations
        """
        self.repo = repo

    def get_unprocessed_firm_ids(self, all_firm_ids: List[str]) -> List[str]:
        """
        Get firm IDs that haven't been processed yet.

        Args:
            all_firm_ids: List of all firm IDs from data source

        Returns:
            List of firm IDs that need processing
        """
        processed = set(self.repo.get_processed_firm_ids())
        unprocessed = [f for f in all_firm_ids if f not in processed]

        logger.info(
            f"Checkpoint: {len(processed)} processed, "
            f"{len(unprocessed)} unprocessed"
        )
        return unprocessed

    def checkpoint(self, firm_id: int, session: Session) -> None:
        """
        Mark firm as processed and commit.

        This is the checkpoint - if the process is interrupted after this,
        the firm won't be reprocessed on resume.

        Args:
            firm_id: Database ID of the firm (not company_id string)
            session: SQLAlchemy session to commit
        """
        self.repo.mark_firm_processed(firm_id)
        session.commit()
        logger.debug(f"Checkpoint: Firm {firm_id} marked as processed")

    def get_resume_point(self, all_firm_ids: List[str]) -> int:
        """
        Get the index to resume from.

        Returns:
            Index of first unprocessed firm in all_firm_ids list,
            or len(all_firm_ids) if all are processed.
        """
        processed = set(self.repo.get_processed_firm_ids())

        for i, firm_id in enumerate(all_firm_ids):
            if firm_id not in processed:
                logger.info(f"Resume point: index {i}, firm {firm_id}")
                return i

        logger.info("Resume point: all firms processed")
        return len(all_firm_ids)
