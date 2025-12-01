"""
LocalCSVConnector - CSV data connector for local testing.

This is a CLEAN reimplementation. The MVP code at Local_BERTopic_MVP is
for intent reference only - we understand the CSV structure from it but
do not port the over-complicated logic.

Design:
    - Simple pandas-based CSV reading
    - Firm ID based filtering (matches FIRM_ID env var in map phase)
    - SpaCy-based sentence splitting (components -> sentences)
    - Stopword removal for cleaner topic modeling
    - Inclusive date filtering
    - Clean sentence_id generation per operational rules
"""

import logging
from typing import List, Optional

import pandas as pd
from spacy.lang.en import English

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
        - componenttext: The transcript text (multiple sentences per component)
        - componentorder: Order within transcript
        - mostimportantdateutc: Transcript date (YYYY-MM-DD)
        - speakertypename: Speaker role (CEO, CFO, etc.) - optional

    Key behaviors:
        - Components are concatenated per transcript, then split into sentences
        - Stopwords are removed from sentence text for cleaner topic modeling
        - Sentence IDs use 0-based positions across all sentences in transcript

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

        # Initialize SpaCy with just the sentencizer (no model download needed)
        self._nlp = English()
        self._nlp.add_pipe("sentencizer")

        # Get SpaCy's built-in stopwords
        self._stopwords = self._nlp.Defaults.stop_words

        # Verify file exists and load
        try:
            self._df = pd.read_csv(csv_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Convert companyid to string for consistent matching
        if "companyid" in self._df.columns:
            self._df["companyid"] = self._df["companyid"].astype(str)

        logger.info(f"Loaded CSV with {len(self._df)} rows from {csv_path}")

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into individual sentences using SpaCy.

        Args:
            text: Text to split (may contain multiple sentences)

        Returns:
            List of sentence strings
        """
        if not text or pd.isna(text):
            return []

        doc = self._nlp(str(text))
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    def _remove_stopwords(self, text: str) -> str:
        """
        Remove stopwords from text.

        Args:
            text: Input text

        Returns:
            Text with stopwords removed
        """
        if not text:
            return ""

        # Tokenize and filter stopwords
        doc = self._nlp(text)
        filtered_tokens = [
            token.text for token in doc
            if token.text.lower() not in self._stopwords
            and not token.is_punct
            and not token.is_space
        ]

        return " ".join(filtered_tokens)

    def fetch_transcripts(
        self,
        firm_ids: List[str],
        start_date: str,
        end_date: str,
    ) -> TranscriptData:
        """
        Fetch transcript sentences for specified firms and date range.

        Components are split into individual sentences and stopwords are removed.

        Args:
            firm_ids: List of firm IDs (companyid values)
            start_date: YYYY-MM-DD format (inclusive)
            end_date: YYYY-MM-DD format (inclusive)

        Returns:
            TranscriptData with firms mapped to their sentences
        """
        logger.info(f"Fetching transcripts for {len(firm_ids)} firm IDs, {start_date} to {end_date}")

        # Work on a copy to avoid mutating the original dataframe
        df = self._df.copy()

        # Convert input firm_ids to strings for matching
        firm_ids_str = [str(fid) for fid in firm_ids]

        # Filter by firm ID
        mask = df["companyid"].isin(firm_ids_str)

        # Date filtering (inclusive)
        if "mostimportantdateutc" in df.columns:
            df["_date"] = pd.to_datetime(df["mostimportantdateutc"], errors="coerce")
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            date_mask = (df["_date"] >= start_dt) & (df["_date"] <= end_dt)
            mask = mask & date_mask

        filtered_df = df[mask].copy()

        # Drop temporary _date column if present
        if "_date" in filtered_df.columns:
            filtered_df = filtered_df.drop(columns=["_date"])
        logger.info(f"Filtered to {len(filtered_df)} rows")

        # Group by firm and build TranscriptData
        firms_dict = {}

        for firm_id, firm_group in filtered_df.groupby("companyid"):
            firm_id_str = str(firm_id)
            firm_name = firm_group["companyname"].iloc[0]

            # Process each transcript for this firm
            sentences = []
            sentence_position = 0  # 0-based position across all sentences

            for transcript_id, transcript_group in firm_group.groupby("transcriptid"):
                # Skip NaN transcript IDs
                if pd.isna(transcript_id):
                    logger.warning(f"Skipping row with NaN transcript_id for firm {firm_id_str}")
                    continue

                transcript_id_str = str(int(transcript_id) if isinstance(transcript_id, float) else transcript_id)

                # Sort components by componentorder
                transcript_group = transcript_group.sort_values("componentorder")

                # Process each component
                for _, row in transcript_group.iterrows():
                    component_text = row.get("componenttext", "")
                    if pd.isna(component_text) or not str(component_text).strip():
                        continue

                    # Get speaker type for this component
                    speaker_type = row.get("speakertypename", None)
                    if pd.isna(speaker_type):
                        speaker_type = None

                    # Split component into sentences
                    component_sentences = self._split_into_sentences(str(component_text))

                    for sent_text in component_sentences:
                        # Remove stopwords
                        cleaned_text = self._remove_stopwords(sent_text)

                        # Skip empty sentences after stopword removal
                        if not cleaned_text.strip():
                            continue

                        # Generate sentence_id: {firm_id}_{transcript_id}_{position:04d}
                        sentence_id = f"{firm_id_str}_{transcript_id_str}_{sentence_position:04d}"

                        sentence = TranscriptSentence(
                            sentence_id=sentence_id,
                            text=cleaned_text,
                            speaker_type=speaker_type,
                            position=sentence_position,
                        )
                        sentences.append(sentence)
                        sentence_position += 1

            # Get unique transcript count for metadata
            n_transcripts = firm_group["transcriptid"].nunique()

            firms_dict[firm_id_str] = FirmTranscriptData(
                firm_id=firm_id_str,
                firm_name=firm_name,
                sentences=sentences,
                metadata={
                    "n_transcripts": n_transcripts,
                    "n_components": len(firm_group),
                    "date_range": f"{start_date} to {end_date}",
                },
            )

        logger.info(f"Built TranscriptData for {len(firms_dict)} firms")
        return TranscriptData(firms=firms_dict)

    def get_available_firm_ids(self) -> List[str]:
        """
        List all firm IDs available in the CSV.

        Returns:
            Sorted list of unique firm IDs (as strings)
        """
        if "companyid" in self._df.columns:
            return sorted(self._df["companyid"].unique().tolist())
        return []

    def get_firm_id_by_name(self, firm_name: str) -> Optional[str]:
        """
        Look up firm ID by company name (case-insensitive).

        This is a convenience method for testing when you know the firm name
        but need the ID for fetch_transcripts.

        Args:
            firm_name: Company name to look up

        Returns:
            Firm ID string, or None if not found
        """
        mask = self._df["companyname"].str.lower() == firm_name.lower()
        matches = self._df[mask]
        if len(matches) > 0:
            return str(matches["companyid"].iloc[0])
        return None

    def close(self) -> None:
        """Clean up resources (no-op for CSV)."""
        logger.debug("LocalCSVConnector closed")
