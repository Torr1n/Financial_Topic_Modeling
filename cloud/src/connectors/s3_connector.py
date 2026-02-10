"""
S3TranscriptConnector - S3 data connector for cloud deployment.

Downloads a transcript CSV from S3 to a temp file, then delegates
all parsing to LocalCSVConnector. This avoids duplicating CSV/NLP logic.

Design:
    - Lazy loading: S3 download happens on first data access, not __init__
    - Delegation: All transcript parsing handled by LocalCSVConnector
    - Cleanup: Temp file removed on close() or garbage collection
"""

import logging
import os
import tempfile
from typing import List

import boto3

from cloud.src.connectors.local_csv import LocalCSVConnector
from cloud.src.interfaces import DataConnector
from cloud.src.models import TranscriptData

logger = logging.getLogger(__name__)


class S3TranscriptConnector(DataConnector):
    """
    S3 connector for cloud deployment.

    Downloads a transcript CSV from an S3 bucket to a local temp file,
    then delegates all parsing to LocalCSVConnector.

    Args:
        bucket: S3 bucket name
        key: S3 object key for transcript CSV
        region: AWS region (default: us-east-1)
    """

    def __init__(
        self,
        bucket: str,
        key: str,
        region: str = "us-east-1",
    ):
        """
        Initialize S3 connector. Does NOT download data yet (lazy).

        Args:
            bucket: S3 bucket name
            key: S3 object key for the transcript CSV
            region: AWS region
        """
        self.bucket = bucket
        self.key = key
        self.region = region
        self._local_connector: LocalCSVConnector | None = None
        self._temp_path: str | None = None

    def _ensure_loaded(self) -> None:
        """
        Download CSV from S3 to a temp file and create LocalCSVConnector.

        Only runs on the first call; subsequent calls are no-ops.
        """
        if self._local_connector is not None:
            return

        logger.info(f"Downloading s3://{self.bucket}/{self.key} (region={self.region})")

        # Create temp file for the download
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        self._temp_path = tmp.name
        tmp.close()

        # Download from S3
        s3 = boto3.client("s3", region_name=self.region)
        s3.download_file(self.bucket, self.key, self._temp_path)

        logger.info(f"Downloaded to {self._temp_path}")

        # Delegate all CSV parsing to LocalCSVConnector
        self._local_connector = LocalCSVConnector(self._temp_path)

    def fetch_transcripts(
        self,
        firm_ids: List[str],
        start_date: str,
        end_date: str,
    ) -> TranscriptData:
        """
        Fetch transcripts from S3 data.

        Downloads the CSV on first call, then delegates to LocalCSVConnector.

        Args:
            firm_ids: List of firm IDs to fetch
            start_date: YYYY-MM-DD format (inclusive)
            end_date: YYYY-MM-DD format (inclusive)

        Returns:
            TranscriptData with firms mapped to their sentences
        """
        self._ensure_loaded()
        return self._local_connector.fetch_transcripts(firm_ids, start_date, end_date)

    def get_available_firm_ids(self) -> List[str]:
        """
        List available firm IDs in the S3 data.

        Downloads the CSV on first call, then delegates to LocalCSVConnector.

        Returns:
            Sorted list of unique firm IDs
        """
        self._ensure_loaded()
        return self._local_connector.get_available_firm_ids()

    def close(self) -> None:
        """Clean up temp file and close the local connector."""
        if self._local_connector is not None:
            self._local_connector.close()
            self._local_connector = None

        if self._temp_path is not None and os.path.exists(self._temp_path):
            os.unlink(self._temp_path)
            logger.info(f"Cleaned up temp file: {self._temp_path}")
            self._temp_path = None

    def __del__(self):
        """Ensure temp file cleanup on garbage collection."""
        # Defensive: only clean up the file, don't call close() which may
        # trigger other cleanup on partially-destroyed objects.
        if getattr(self, "_temp_path", None) and os.path.exists(self._temp_path):
            os.unlink(self._temp_path)
