# Financial Topic Modeling - Step Functions Infrastructure
# Sprint 5: Multi-quarter orchestration with visual workflow monitoring

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# -----------------------------------------------------------------------------
# DATA SOURCES - Reference existing resources
# -----------------------------------------------------------------------------

# Get current AWS account ID for IAM policies
data "aws_caller_identity" "current" {}

# Get current region for ARN construction
data "aws_region" "current" {}

# Reference existing S3 bucket from main terraform
data "aws_s3_bucket" "pipeline" {
  bucket = var.s3_bucket_name
}

# Reference existing Batch resources
data "aws_batch_job_queue" "main" {
  name = var.job_queue_name
}

data "aws_batch_job_definition" "firm_processor" {
  name = var.job_definition_name
}

data "aws_batch_job_definition" "theme_aggregator" {
  name = var.reduce_job_definition_name
}
