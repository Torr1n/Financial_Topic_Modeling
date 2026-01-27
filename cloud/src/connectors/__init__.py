"""
Data connectors for the Financial Topic Modeling pipeline.

Available connectors:
    - LocalCSVConnector: For local testing with CSV files
    - S3TranscriptConnector: For cloud deployment (Phase 2)
    - WRDSConnector: For WRDS Capital IQ transcripts with PERMNO linking
"""

from cloud.src.connectors.local_csv import LocalCSVConnector
from cloud.src.connectors.s3_connector import S3TranscriptConnector
from cloud.src.connectors.wrds_connector import WRDSConnector

__all__ = ["LocalCSVConnector", "S3TranscriptConnector", "WRDSConnector"]
