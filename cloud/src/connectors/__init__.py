"""
Data connectors for the Financial Topic Modeling pipeline.

Available connectors:
    - LocalCSVConnector: For local testing with CSV files
    - S3TranscriptConnector: For cloud deployment (Phase 2)
"""

from cloud.src.connectors.local_csv import LocalCSVConnector
from cloud.src.connectors.s3_connector import S3TranscriptConnector

__all__ = ["LocalCSVConnector", "S3TranscriptConnector"]
