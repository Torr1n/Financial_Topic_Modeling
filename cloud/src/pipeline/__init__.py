"""
Pipeline package for unified firm/theme processing.

This package contains:
    - UnifiedPipeline: Main orchestration class
    - CheckpointManager: Resume logic for spot instance interruption

Phase 2 of the architecture pivot replaces the distributed AWS Batch
map-reduce pattern with a single GPU instance that:
    - Loads embedding model ONCE
    - Processes firms sequentially with per-firm checkpoints
    - Stores all results in PostgreSQL (no S3 intermediate)
"""

from cloud.src.pipeline.unified_pipeline import UnifiedPipeline
from cloud.src.pipeline.checkpoint import CheckpointManager

__all__ = ["UnifiedPipeline", "CheckpointManager"]
