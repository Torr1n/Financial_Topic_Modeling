"""
S3TranscriptConnector - Read prefetched transcripts from S3.

This connector reads transcript data that was prefetched by WRDSPrefetcher.
It uses the manifest.json for efficient, selective chunk loading.

Key design decisions:
    - Manifest-based selective loading: Only reads chunks containing requested firm_ids
    - NO re-preprocessing: Data is already cleaned by prefetcher (cleaned_text ready for embeddings)
    - Firm_id filter only: Date params ignored (quarter already filtered by prefetch)
    - Memory-bounded: Reads only required chunks, not entire quarter

S3 Input Structure:
    s3://{bucket}/prefetch/transcripts/quarter=2023Q1/
        ├── chunk_0000.parquet
        ├── chunk_0001.parquet
        ├── ...
        └── manifest.json (gzip compressed)
"""

import gzip
import io
import json
import logging
from typing import Any, Dict, List, Optional, Set

import boto3
import pandas as pd
from botocore.exceptions import ClientError

from cloud.src.interfaces import DataConnector
from cloud.src.models import FirmTranscriptData, TranscriptData, TranscriptSentence

logger = logging.getLogger(__name__)


class S3TranscriptConnectorError(Exception):
    """Base exception for S3 connector errors."""
    pass


class ManifestNotFoundError(S3TranscriptConnectorError):
    """Raised when manifest.json is not found for a quarter."""
    pass


class S3TranscriptConnector(DataConnector):
    """
    S3 connector for reading prefetched transcript data.

    Reads from S3 prefetch location using manifest.json for efficient
    selective loading. Only loads chunks containing requested firm_ids.

    Args:
        bucket: S3 bucket name
        quarter: Quarter to read from (e.g., "2023Q1")
        region: AWS region (default: us-east-1)

    Usage:
        connector = S3TranscriptConnector(bucket="ftm-pipeline-xxx", quarter="2023Q1")
        data = connector.fetch_transcripts(firm_ids=["123", "456"], ...)
        connector.close()
    """

    def __init__(
        self,
        bucket: str,
        quarter: str,
        region: str = "us-east-1",
        s3_client=None,
    ):
        """
        Initialize S3 connector.

        Args:
            bucket: S3 bucket name
            quarter: Quarter string (e.g., "2023Q1")
            region: AWS region
            s3_client: Optional S3 client (for testing)
        """
        self.bucket = bucket
        self.quarter = quarter
        self.region = region
        self._s3_client = s3_client or boto3.client("s3", region_name=region)
        self._manifest: Optional[Dict[str, Any]] = None

    def _get_manifest_key(self) -> str:
        """Get S3 key for manifest file."""
        return f"prefetch/transcripts/quarter={self.quarter}/manifest.json"

    def _get_chunk_key(self, chunk_file: str) -> str:
        """Get S3 key for a chunk file."""
        return f"prefetch/transcripts/quarter={self.quarter}/{chunk_file}"

    def _load_manifest(self) -> Dict[str, Any]:
        """
        Load manifest.json from prefetch location.

        Returns:
            Manifest dict with firm_to_chunk mapping

        Raises:
            ManifestNotFoundError: If manifest doesn't exist
        """
        if self._manifest is not None:
            return self._manifest

        manifest_key = self._get_manifest_key()
        logger.info(f"Loading manifest from s3://{self.bucket}/{manifest_key}")

        try:
            response = self._s3_client.get_object(Bucket=self.bucket, Key=manifest_key)

            # Handle gzip compression
            content_encoding = response.get("ContentEncoding", "")
            body = response["Body"].read()

            if content_encoding == "gzip" or manifest_key.endswith(".gz"):
                body = gzip.decompress(body)

            self._manifest = json.loads(body.decode("utf-8"))
            logger.info(
                f"Loaded manifest: {self._manifest['n_firms']} firms, "
                f"{self._manifest['n_chunks']} chunks"
            )
            return self._manifest

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "NoSuchKey":
                raise ManifestNotFoundError(
                    f"Manifest not found for quarter {self.quarter}. "
                    f"Run prefetch first: s3://{self.bucket}/{manifest_key}"
                )
            raise S3TranscriptConnectorError(f"Failed to load manifest: {e}")

    def _get_chunks_for_firms(self, firm_ids: List[str]) -> Set[str]:
        """
        Determine which chunks contain the requested firm_ids.

        Args:
            firm_ids: List of firm IDs to look up

        Returns:
            Set of chunk filenames to read
        """
        manifest = self._load_manifest()
        firm_to_chunk = manifest.get("firm_to_chunk", {})

        chunks = set()
        missing_firms = []

        for firm_id in firm_ids:
            if firm_id in firm_to_chunk:
                chunks.add(firm_to_chunk[firm_id])
            else:
                missing_firms.append(firm_id)

        if missing_firms:
            logger.warning(
                f"{len(missing_firms)} firm(s) not found in manifest: "
                f"{missing_firms[:5]}{'...' if len(missing_firms) > 5 else ''}"
            )

        logger.info(
            f"Selected {len(chunks)} chunks for {len(firm_ids)} firm_ids "
            f"({len(missing_firms)} not found)"
        )
        return chunks

    def _read_chunk(self, chunk_file: str) -> pd.DataFrame:
        """
        Read a single Parquet chunk from S3.

        Args:
            chunk_file: Chunk filename (e.g., "chunk_0000.parquet")

        Returns:
            DataFrame with chunk data
        """
        chunk_key = self._get_chunk_key(chunk_file)
        logger.debug(f"Reading chunk: s3://{self.bucket}/{chunk_key}")

        response = self._s3_client.get_object(Bucket=self.bucket, Key=chunk_key)
        body = response["Body"].read()

        df = pd.read_parquet(io.BytesIO(body))
        return df

    def _build_transcript_data(
        self,
        df: pd.DataFrame,
        firm_ids: List[str],
    ) -> TranscriptData:
        """
        Convert DataFrame rows to TranscriptData structure.

        Note: NO re-preprocessing - data is already cleaned by prefetcher.
        We just reconstruct the TranscriptData structure.

        Args:
            df: Combined DataFrame from all relevant chunks
            firm_ids: List of firm_ids to filter to (chunks may contain other firms)

        Returns:
            TranscriptData with requested firms
        """
        # Filter to requested firms (chunks may contain other firms)
        firm_id_set = set(firm_ids)
        df = df[df["firm_id"].isin(firm_id_set)]

        if df.empty:
            return TranscriptData(firms={})

        firms_dict = {}

        for firm_id, firm_group in df.groupby("firm_id"):
            firm_id_str = str(firm_id)

            # Sort by position to maintain order
            firm_group = firm_group.sort_values("position")

            # Build sentences (no re-preprocessing - use cleaned_text as-is)
            sentences = []
            for _, row in firm_group.iterrows():
                sentence = TranscriptSentence(
                    sentence_id=row["sentence_id"],
                    raw_text=row["raw_text"],
                    cleaned_text=row["cleaned_text"],  # Already preprocessed
                    speaker_type=row.get("speaker_type"),
                    position=int(row["position"]),
                )
                sentences.append(sentence)

            # Get metadata from first row
            first_row = firm_group.iloc[0]
            metadata = {
                "permno": int(first_row["permno"]) if pd.notna(first_row["permno"]) else None,
                "gvkey": str(first_row["gvkey"]) if pd.notna(first_row["gvkey"]) else None,
                "earnings_call_date": first_row.get("earnings_call_date"),
                "transcript_id": first_row.get("transcript_id"),
                "quarter": first_row.get("quarter"),
            }

            firms_dict[firm_id_str] = FirmTranscriptData(
                firm_id=firm_id_str,
                firm_name=first_row["firm_name"],
                sentences=sentences,
                metadata=metadata,
            )

        logger.info(f"Built TranscriptData for {len(firms_dict)} firms")
        return TranscriptData(firms=firms_dict)

    def fetch_transcripts(
        self,
        firm_ids: List[str],
        start_date: str,
        end_date: str,
    ) -> TranscriptData:
        """
        Fetch transcripts from S3 prefetch location.

        SELECTIVE LOADING (firm_id filter only, date args ignored):
        1. Determine which chunks contain requested firm_ids (via manifest)
        2. Read ONLY those chunks (not entire quarter)
        3. Filter rows to exact firm_ids (chunks may contain other firms)
        4. Reconstruct TranscriptData (NO re-preprocessing - data already clean)

        Note: start_date/end_date kept for interface compatibility but ignored.
        Quarter-level filtering already done by prefetch.

        Args:
            firm_ids: List of firm IDs to fetch
            start_date: Ignored (kept for interface compatibility)
            end_date: Ignored (kept for interface compatibility)

        Returns:
            TranscriptData with firms mapped to their sentences
        """
        if not firm_ids:
            logger.warning("Empty firm_ids list, returning empty TranscriptData")
            return TranscriptData(firms={})

        logger.info(f"Fetching {len(firm_ids)} firms from quarter {self.quarter}")

        # Determine which chunks to read
        chunks_to_read = self._get_chunks_for_firms(firm_ids)

        if not chunks_to_read:
            logger.warning("No chunks found for requested firm_ids")
            return TranscriptData(firms={})

        # Read all required chunks
        dfs = []
        for chunk_file in sorted(chunks_to_read):
            df = self._read_chunk(chunk_file)
            dfs.append(df)

        # Combine chunks
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Read {len(combined_df)} total rows from {len(chunks_to_read)} chunks")

        # Build TranscriptData
        return self._build_transcript_data(combined_df, firm_ids)

    def get_available_firm_ids(self) -> List[str]:
        """
        List available firm IDs from manifest.

        This is O(1) - reads from manifest.json, no data scan needed.

        Returns:
            Sorted list of firm IDs available in the prefetch data
        """
        manifest = self._load_manifest()
        firm_to_chunk = manifest.get("firm_to_chunk", {})
        return sorted(firm_to_chunk.keys())

    def get_manifest_summary(self) -> Dict[str, Any]:
        """
        Get summary info from manifest.

        Returns:
            Dict with quarter, n_firms, n_chunks, created_at
        """
        manifest = self._load_manifest()
        return {
            "quarter": manifest.get("quarter"),
            "n_firms": manifest.get("n_firms"),
            "n_chunks": manifest.get("n_chunks"),
            "created_at": manifest.get("created_at"),
        }

    def close(self) -> None:
        """Clean up resources (no-op for S3 connector)."""
        pass
