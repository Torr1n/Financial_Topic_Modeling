"""
WRDS Prefetcher - Fetch transcripts and write to S3 as Parquet + manifest.

This module implements the "prefetch to S3" pattern that works around WRDS MFA:
- Runs from a fixed-IP machine with single MFA approval
- Reuses WRDSConnector for data fetching and NLP preprocessing
- Stores preprocessed data ready for embeddings (cleaned_text)
- Chunks ~100-200 firms per Parquet file for balanced loading
- Writes manifest.json mapping firm_id -> chunk file for efficient reads
- Checkpoints every 100 firms for resumability

S3 Output Structure:
    s3://{bucket}/prefetch/transcripts/quarter=2023Q1/
        ├── chunk_0000.parquet      # Firms 0-199
        ├── chunk_0001.parquet      # Firms 200-399
        ├── ...
        ├── manifest.json           # Firm-to-chunk mapping (gzip compressed)
        └── _checkpoint.json        # Resume state
"""

import gzip
import json
import logging
import os
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.exceptions import ClientError

from cloud.src.connectors.wrds_connector import WRDSConnector
from cloud.src.models import TranscriptData

logger = logging.getLogger(__name__)


# Parquet schema for prefetch data
PREFETCH_SCHEMA = pa.schema([
    ("firm_id", pa.string()),
    ("firm_name", pa.string()),
    ("permno", pa.int64()),
    ("gvkey", pa.string()),
    ("transcript_id", pa.string()),
    ("earnings_call_date", pa.date32()),
    ("sentence_id", pa.string()),
    ("raw_text", pa.string()),
    ("cleaned_text", pa.string()),
    ("speaker_type", pa.string()),
    ("position", pa.int32()),
    ("quarter", pa.string()),
])


def quarter_to_date_range(quarter: str) -> Tuple[str, str]:
    """
    Convert quarter string to date range.

    Args:
        quarter: Format "2023Q1"

    Returns:
        Tuple of (start_date, end_date) in YYYY-MM-DD format
    """
    year = int(quarter[:4])
    q = int(quarter[5])

    quarter_starts = {1: (1, 1), 2: (4, 1), 3: (7, 1), 4: (10, 1)}
    quarter_ends = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}

    start_month, start_day = quarter_starts[q]
    end_month, end_day = quarter_ends[q]

    start_date = f"{year}-{start_month:02d}-{start_day:02d}"
    end_date = f"{year}-{end_month:02d}-{end_day:02d}"

    return start_date, end_date


class WRDSPrefetcher:
    """
    Prefetch transcripts from WRDS and write to S3 as Parquet + manifest.

    Usage:
        prefetcher = WRDSPrefetcher(bucket="ftm-pipeline-xxx")
        result = prefetcher.prefetch_quarter("2023Q1")
        # Or with specific firms:
        result = prefetcher.prefetch_quarter("2023Q1", firm_ids=["123", "456"])
    """

    # Firms per Parquet chunk (balance between small file problem and selective loading)
    CHUNK_SIZE = 200

    # Checkpoint every N firms (for resumability)
    CHECKPOINT_INTERVAL = 100

    def __init__(
        self,
        bucket: str,
        region: str = "us-east-1",
        s3_client=None,
        wrds_connector: Optional[WRDSConnector] = None,
    ):
        """
        Initialize the prefetcher.

        Args:
            bucket: S3 bucket name for output
            region: AWS region
            s3_client: Optional S3 client (for testing)
            wrds_connector: Optional WRDSConnector (for testing)
        """
        self.bucket = bucket
        self.region = region
        self._s3_client = s3_client or boto3.client("s3", region_name=region)
        self._wrds_connector = wrds_connector
        self._owns_connector = wrds_connector is None

    def _get_connector(self) -> WRDSConnector:
        """Get or create WRDS connector."""
        if self._wrds_connector is None:
            self._wrds_connector = WRDSConnector()
        return self._wrds_connector

    def _get_checkpoint_key(self, quarter: str) -> str:
        """Get S3 key for checkpoint file."""
        return f"prefetch/transcripts/quarter={quarter}/_checkpoint.json"

    def _get_manifest_key(self, quarter: str) -> str:
        """Get S3 key for manifest file."""
        return f"prefetch/transcripts/quarter={quarter}/manifest.json"

    def _get_chunk_key(self, quarter: str, chunk_id: int) -> str:
        """Get S3 key for a chunk file."""
        return f"prefetch/transcripts/quarter={quarter}/chunk_{chunk_id:04d}.parquet"

    def _get_checkpoint(self, quarter: str) -> Tuple[Set[str], int, Dict[str, str]]:
        """
        Load checkpoint from S3.

        Returns:
            Tuple of (completed_firm_ids, last_chunk_id, firm_to_chunk_mapping)
            Returns (empty set, -1, empty dict) if no checkpoint exists.
        """
        checkpoint_key = self._get_checkpoint_key(quarter)

        try:
            response = self._s3_client.get_object(Bucket=self.bucket, Key=checkpoint_key)
            data = json.loads(response["Body"].read().decode("utf-8"))
            completed = set(data.get("completed_firm_ids", []))
            last_chunk_id = data.get("last_chunk_id", -1)
            firm_to_chunk = data.get("firm_to_chunk", {})
            logger.info(
                f"Loaded checkpoint: {len(completed)} firms complete, "
                f"last_chunk_id={last_chunk_id}"
            )
            return completed, last_chunk_id, firm_to_chunk
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "NoSuchKey":
                logger.info("No checkpoint found, starting fresh")
                return set(), -1, {}
            raise

    def _save_checkpoint(
        self,
        quarter: str,
        completed: Set[str],
        chunk_id: int,
        firm_to_chunk: Dict[str, str],
    ) -> None:
        """Save checkpoint to S3."""
        checkpoint_key = self._get_checkpoint_key(quarter)

        data = {
            "quarter": quarter,
            "completed_firm_ids": sorted(completed),
            "last_chunk_id": chunk_id,
            "firm_to_chunk": firm_to_chunk,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        self._s3_client.put_object(
            Bucket=self.bucket,
            Key=checkpoint_key,
            Body=json.dumps(data).encode("utf-8"),
            ContentType="application/json",
        )
        logger.info(f"Saved checkpoint: {len(completed)} firms, chunk_id={chunk_id}")

    def _flatten_to_rows(
        self,
        transcript_data: TranscriptData,
        quarter: str,
    ) -> List[Dict[str, Any]]:
        """
        Convert TranscriptData to flat rows for Parquet.

        Args:
            transcript_data: Data from WRDSConnector
            quarter: Quarter string for partition key

        Returns:
            List of dicts, one per sentence
        """
        rows = []

        for firm_id, firm_data in transcript_data.firms.items():
            metadata = firm_data.metadata

            for sentence in firm_data.sentences:
                rows.append({
                    "firm_id": firm_id,
                    "firm_name": firm_data.firm_name,
                    "permno": metadata.get("permno"),
                    "gvkey": metadata.get("gvkey"),
                    "transcript_id": metadata.get("transcript_id"),
                    "earnings_call_date": metadata.get("earnings_call_date"),
                    "sentence_id": sentence.sentence_id,
                    "raw_text": sentence.raw_text,
                    "cleaned_text": sentence.cleaned_text,
                    "speaker_type": sentence.speaker_type,
                    "position": sentence.position,
                    "quarter": quarter,
                })

        return rows

    def _write_chunk(
        self,
        rows: List[Dict[str, Any]],
        quarter: str,
        chunk_id: int,
    ) -> str:
        """
        Write rows to Parquet chunk on S3.

        Args:
            rows: List of sentence dicts
            quarter: Quarter string
            chunk_id: Chunk number

        Returns:
            S3 key of written file
        """
        chunk_key = self._get_chunk_key(quarter, chunk_id)

        # Convert to DataFrame
        df = pd.DataFrame(rows)

        # Convert date column
        if "earnings_call_date" in df.columns:
            df["earnings_call_date"] = pd.to_datetime(df["earnings_call_date"]).dt.date

        # Convert to PyArrow table with explicit schema
        table = pa.Table.from_pandas(df, schema=PREFETCH_SCHEMA, preserve_index=False)

        # Write to temp file and upload
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            pq.write_table(table, tmp.name)
            self._s3_client.upload_file(tmp.name, self.bucket, chunk_key)
            os.remove(tmp.name)

        logger.info(f"Wrote chunk {chunk_id}: {len(rows)} rows to s3://{self.bucket}/{chunk_key}")
        return chunk_key

    def _write_manifest(
        self,
        quarter: str,
        firm_to_chunk: Dict[str, str],
        chunk_sizes: Dict[str, int],
    ) -> str:
        """
        Write manifest.json to S3 (gzip compressed).

        Args:
            quarter: Quarter string
            firm_to_chunk: Mapping of firm_id -> chunk filename
            chunk_sizes: Mapping of chunk filename -> firm count

        Returns:
            S3 key of manifest file
        """
        manifest_key = self._get_manifest_key(quarter)

        manifest = {
            "quarter": quarter,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "n_firms": len(firm_to_chunk),
            "n_chunks": len(chunk_sizes),
            "chunk_sizes": chunk_sizes,
            "firm_to_chunk": firm_to_chunk,
        }

        # Compress with gzip
        manifest_json = json.dumps(manifest).encode("utf-8")
        compressed = gzip.compress(manifest_json)

        self._s3_client.put_object(
            Bucket=self.bucket,
            Key=manifest_key,
            Body=compressed,
            ContentType="application/json",
            ContentEncoding="gzip",
        )

        logger.info(
            f"Wrote manifest: {len(firm_to_chunk)} firms, {len(chunk_sizes)} chunks "
            f"({len(compressed)} bytes compressed) to s3://{self.bucket}/{manifest_key}"
        )
        return manifest_key

    def _delete_checkpoint(self, quarter: str) -> None:
        """Delete checkpoint file after successful completion."""
        checkpoint_key = self._get_checkpoint_key(quarter)
        try:
            self._s3_client.delete_object(Bucket=self.bucket, Key=checkpoint_key)
            logger.info(f"Deleted checkpoint: s3://{self.bucket}/{checkpoint_key}")
        except ClientError:
            pass  # Ignore if doesn't exist

    def prefetch_quarter(
        self,
        quarter: str,
        firm_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Prefetch all transcripts for a quarter to S3.

        Args:
            quarter: Quarter string (e.g., "2023Q1")
            firm_ids: Optional list of specific firm IDs. If None, fetches all firms
                     with transcripts in the date range.

        Returns:
            Dict with summary: n_firms, n_chunks, n_sentences, manifest_key
        """
        logger.info(f"Starting prefetch for quarter {quarter}")

        # Convert quarter to date range
        start_date, end_date = quarter_to_date_range(quarter)
        logger.info(f"Date range: {start_date} to {end_date}")

        # Load checkpoint for resume
        completed, last_chunk_id, firm_to_chunk = self._get_checkpoint(quarter)
        chunk_sizes: Dict[str, int] = {}

        # Reconstruct chunk_sizes from firm_to_chunk if resuming
        if firm_to_chunk:
            for firm_id, chunk_file in firm_to_chunk.items():
                chunk_sizes[chunk_file] = chunk_sizes.get(chunk_file, 0) + 1

        # Get connector
        connector = self._get_connector()

        try:
            # If no firm_ids specified, discover firms via lightweight query
            if firm_ids is None:
                # Use lightweight query to get firm IDs (not full transcripts)
                # This avoids loading entire quarter into memory
                logger.info("Discovering firms with transcripts in date range...")
                firm_ids = connector.get_firm_ids_in_range(start_date, end_date)
                logger.info(f"Found {len(firm_ids)} firms with PERMNO links")

            # Filter out already completed firms
            pending_firms = [f for f in firm_ids if f not in completed]
            logger.info(
                f"Processing {len(pending_firms)} pending firms "
                f"({len(completed)} already complete)"
            )

            if not pending_firms:
                logger.info("All firms already processed, writing manifest...")
                manifest_key = self._write_manifest(quarter, firm_to_chunk, chunk_sizes)
                self._delete_checkpoint(quarter)
                return {
                    "quarter": quarter,
                    "n_firms": len(firm_to_chunk),
                    "n_chunks": len(chunk_sizes),
                    "manifest_key": manifest_key,
                    "status": "complete",
                }

            # Process firms in batches
            chunk_id = last_chunk_id + 1
            buffer: List[Dict[str, Any]] = []
            firms_in_buffer: List[str] = []
            total_sentences = 0
            firms_since_checkpoint = 0

            for i, firm_id in enumerate(pending_firms):
                logger.info(f"Processing firm {firm_id} ({i + 1}/{len(pending_firms)})")

                try:
                    # Fetch transcript for this firm (one at a time to bound memory)
                    firm_data = connector.fetch_transcripts(
                        firm_ids=[firm_id],
                        start_date=start_date,
                        end_date=end_date,
                    )

                    # Skip firms without data
                    if firm_id not in firm_data.firms:
                        logger.warning(f"No transcript data for firm {firm_id}, skipping")
                        completed.add(firm_id)
                        firms_since_checkpoint += 1
                        continue

                    if not firm_data.firms[firm_id].sentences:
                        logger.warning(f"No sentences for firm {firm_id}, skipping")
                        completed.add(firm_id)
                        firms_since_checkpoint += 1
                        continue

                    # Flatten to rows
                    rows = self._flatten_to_rows(firm_data, quarter)
                    buffer.extend(rows)
                    firms_in_buffer.append(firm_id)
                    total_sentences += len(rows)
                    firms_since_checkpoint += 1  # Count successful firms for checkpoint interval

                    # Write chunk when buffer reaches size
                    if len(firms_in_buffer) >= self.CHUNK_SIZE:
                        chunk_file = f"chunk_{chunk_id:04d}.parquet"
                        self._write_chunk(buffer, quarter, chunk_id)

                        # Update mappings
                        for fid in firms_in_buffer:
                            firm_to_chunk[fid] = chunk_file
                        chunk_sizes[chunk_file] = len(firms_in_buffer)
                        completed.update(firms_in_buffer)

                        # Save checkpoint
                        self._save_checkpoint(quarter, completed, chunk_id, firm_to_chunk)

                        # Reset buffer
                        buffer = []
                        firms_in_buffer = []
                        chunk_id += 1
                        firms_since_checkpoint = 0

                    # Checkpoint periodically even if chunk not full
                    elif firms_since_checkpoint >= self.CHECKPOINT_INTERVAL:
                        self._save_checkpoint(quarter, completed, chunk_id - 1, firm_to_chunk)
                        firms_since_checkpoint = 0

                except Exception as e:
                    logger.error(f"Error processing firm {firm_id}: {e}")
                    # Save checkpoint before re-raising
                    if firms_in_buffer:
                        self._save_checkpoint(quarter, completed, chunk_id - 1, firm_to_chunk)
                    raise

            # Write remaining buffer
            if buffer:
                chunk_file = f"chunk_{chunk_id:04d}.parquet"
                self._write_chunk(buffer, quarter, chunk_id)

                for fid in firms_in_buffer:
                    firm_to_chunk[fid] = chunk_file
                chunk_sizes[chunk_file] = len(firms_in_buffer)
                completed.update(firms_in_buffer)

            # Write final manifest
            manifest_key = self._write_manifest(quarter, firm_to_chunk, chunk_sizes)

            # Clean up checkpoint
            self._delete_checkpoint(quarter)

            result = {
                "quarter": quarter,
                "n_firms": len(firm_to_chunk),
                "n_chunks": len(chunk_sizes),
                "n_sentences": total_sentences,
                "manifest_key": manifest_key,
                "status": "complete",
            }
            logger.info(f"Prefetch complete: {result}")
            return result

        finally:
            # Clean up connector if we own it
            if self._owns_connector and self._wrds_connector:
                self._wrds_connector.close()
                self._wrds_connector = None


def main():
    """CLI entrypoint for prefetch."""
    import argparse

    parser = argparse.ArgumentParser(description="Prefetch WRDS transcripts to S3")
    parser.add_argument("--quarter", required=True, help="Quarter to prefetch (e.g., 2023Q1)")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--firm-ids", help="Comma-separated firm IDs (optional)")
    parser.add_argument("--region", default="us-east-1", help="AWS region")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Parse firm IDs if provided
    firm_ids = None
    if args.firm_ids:
        firm_ids = [f.strip() for f in args.firm_ids.split(",")]

    # Run prefetch
    prefetcher = WRDSPrefetcher(bucket=args.bucket, region=args.region)
    result = prefetcher.prefetch_quarter(args.quarter, firm_ids=firm_ids)

    print(f"\nPrefetch complete:")
    print(f"  Quarter: {result['quarter']}")
    print(f"  Firms: {result['n_firms']}")
    print(f"  Chunks: {result['n_chunks']}")
    print(f"  Manifest: s3://{args.bucket}/{result['manifest_key']}")


if __name__ == "__main__":
    main()
