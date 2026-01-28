"""
AWS Batch Job Submission Module

This module provides the BatchJobSubmitter class for creating manifests
and submitting firm-level processing jobs to AWS Batch.
"""

from .job_submitter import BatchJobSubmitter, JobSubmissionResult

__all__ = ["BatchJobSubmitter", "JobSubmissionResult"]
