"""
LocalCSVConnector - CSV data connector for local testing.

This is a CLEAN reimplementation. The MVP code at Local_BERTopic_MVP is
for intent reference only - we understand the CSV structure from it but
do not port the over-complicated logic.

Design:
    - Simple pandas-based CSV reading
    - Case-insensitive firm name matching
    - Inclusive date filtering
    - Clean sentence_id generation per operational rules
"""

import logging
from typing import List
from datetime import datetime

import pandas as pd

from cloud.src.interfaces import DataConnector
from cloud.src.models import TranscriptData, FirmTranscriptData, TranscriptSentence

logger = logging.getLogger(__name__)


class LocalCSVConnector(DataConnector):
    """
    Local CSV connector for testing and development.

    Expected CSV columns:
        - companyid: Firm identifier (used as firm_id)
        - companyname: Human-readable firm name
        - transcriptid: Transcript identifier
        - componenttext: The transcript text
        - componentorder: Order within transcript
        - mostimportantdateutc: Transcript date (YYYY-MM-DD)
        - speakertypename: Speaker role (CEO, CFO, etc.) - optional

    Args:
        csv_path: Path to the CSV file
    """

    def __init__(self, csv_path: str):
        """
        Initialize with path to CSV file.

        Args:
            csv_path: Path to transcript CSV file

        Raises:
            FileNotFoundError: If CSV file does not exist
        """
        self.csv_path = csv_path

        # Verify file exists
        try:
            self._df = pd.read_csv(csv_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        logger.info(f"Loaded CSV with {len(self._df)} rows from {csv_path}")

    def fetch_transcripts(
        self,
        firms: List[str],
        start_date: str,
        end_date: str,
    ) -> TranscriptData:
        """
        Fetch transcript sentences for specified firms and date range.

        Args:
            firms: List of company names (case-insensitive matching)
            start_date: YYYY-MM-DD format (inclusive)
            end_date: YYYY-MM-DD format (inclusive)

        Returns:
            TranscriptData with firms mapped to their sentences
        """
        logger.info(f"Fetching transcripts for {len(firms)} firms, {start_date} to {end_date}")

        # Case-insensitive firm name matching
        firms_lower = [f.lower() for f in firms]
        mask = self._df["companyname"].str.lower().isin(firms_lower)

        # Date filtering (inclusive)
        if "mostimportantdateutc" in self._df.columns:
            self._df["_date"] = pd.to_datetime(
                self._df["mostimportantdateutc"], errors="coerce"
            )
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            date_mask = (self._df["_date"] >= start_dt) & (self._df["_date"] <= end_dt)
            mask = mask & date_mask

        filtered_df = self._df[mask].copy()
        logger.info(f"Filtered to {len(filtered_df)} rows")

        # Group by firm and build TranscriptData
        firms_dict = {}

        for firm_id, group in filtered_df.groupby("companyid"):
            firm_id_str = str(firm_id)
            firm_name = group["companyname"].iloc[0]

            # Build sentences for this firm
            sentences = []
            for idx, row in group.iterrows():
                transcript_id = str(row.get("transcriptid", ""))
                position = int(row.get("componentorder", 0))
                text = str(row.get("componenttext", ""))

                # Skip empty text
                if not text or text == "nan":
                    continue

                # Generate sentence_id per operational rules
                # Format: {firm_id}_{transcript_id}_{position:04d}
                sentence_id = f"{firm_id_str}_{transcript_id}_{position:04d}"

                # Get speaker type (optional)
                speaker_type = row.get("speakertypename", None)
                if pd.isna(speaker_type):
                    speaker_type = None

                sentence = TranscriptSentence(
                    sentence_id=sentence_id,
                    text=text,
                    speaker_type=speaker_type,
                    position=position,
                )
                sentences.append(sentence)

            # Sort sentences by position
            sentences.sort(key=lambda s: s.position)

            # Get unique transcript count for metadata
            n_transcripts = group["transcriptid"].nunique()

            firms_dict[firm_id_str] = FirmTranscriptData(
                firm_id=firm_id_str,
                firm_name=firm_name,
                sentences=sentences,
                metadata={
                    "n_transcripts": n_transcripts,
                    "date_range": f"{start_date} to {end_date}",
                },
            )

        logger.info(f"Built TranscriptData for {len(firms_dict)} firms")
        return TranscriptData(firms=firms_dict)

    def get_available_firms(self) -> List[str]:
        """
        List all firms available in the CSV.

        Returns:
            Sorted list of unique company names
        """
        if "companyname" in self._df.columns:
            return sorted(self._df["companyname"].unique().tolist())
        return []

    def close(self) -> None:
        """Clean up resources (no-op for CSV)."""
        logger.debug("LocalCSVConnector closed")
