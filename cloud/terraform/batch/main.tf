# Financial Topic Modeling - AWS Batch Infrastructure
# Sprint 3: Parallel firm-level processing
# Separate root from main terraform to avoid state conflicts

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

  default_tags {
    tags = {
      Owner   = "Torrin"
      Project = "financial-topic-modeling"
    }
  }
}

# -----------------------------------------------------------------------------
# DATA SOURCES - Reference existing resources without modifying them
# -----------------------------------------------------------------------------

# Use default VPC (no bespoke VPC - keeps it simple, avoids NAT gateway costs)
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# Reference existing S3 bucket from main terraform
# Reserved prefixes for batch operations:
#   - manifests/     - batch manifests (JSONL)
#   - progress/      - checkpoint files
#   - intermediate/  - Parquet output from map phase
data "aws_s3_bucket" "pipeline" {
  bucket = var.s3_bucket_name
}

# Reference WRDS secret (created out-of-band, not in terraform state)
# Create manually: aws secretsmanager create-secret --name wrds-credentials ...
# Gated: only looked up when enable_wrds_secrets=true
data "aws_secretsmanager_secret" "wrds" {
  count = var.enable_wrds_secrets ? 1 : 0
  name  = "wrds-credentials"
}

# Get current AWS account ID for IAM policies
data "aws_caller_identity" "current" {}

# Get current region for ARN construction
data "aws_region" "current" {}

# -----------------------------------------------------------------------------
# VLLM INTEGRATION - Read base URL from SSM (written by ECS module)
# -----------------------------------------------------------------------------

# Optional: vLLM base URL from ECS module
# This allows Batch jobs to use self-hosted LLM instead of xAI API
# If not deployed, jobs will fall back to XAI_API_KEY env var / xAI API
data "aws_ssm_parameter" "vllm_base_url" {
  count = var.enable_vllm ? 1 : 0
  name  = "/ftm/vllm/base_url"
}
