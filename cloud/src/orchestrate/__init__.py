"""
Orchestration module for quarter-level batch processing.

This module coordinates the full workflow:
1. Check for prefetch data (manifest.json)
2. Submit batch jobs with DATA_SOURCE=s3
3. Monitor job completion
"""

from cloud.src.orchestrate.quarter_orchestrator import QuarterOrchestrator

__all__ = ["QuarterOrchestrator"]
