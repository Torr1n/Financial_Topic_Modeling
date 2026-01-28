"""
Prefetch module for WRDS transcript data.

This module handles the "prefetch to S3" pattern that works around WRDS MFA:
- Fetch all transcripts for a quarter from a fixed-IP machine (single MFA approval)
- Store preprocessed data in S3 as Parquet + manifest.json
- Batch jobs then read from S3 without touching WRDS
"""

from cloud.src.prefetch.wrds_prefetcher import WRDSPrefetcher

__all__ = ["WRDSPrefetcher"]
