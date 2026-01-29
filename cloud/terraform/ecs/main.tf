# Financial Topic Modeling - ECS Infrastructure for vLLM
# Sprint 5: Self-hosted LLM for topic naming
# Separate root from batch terraform to avoid state conflicts

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
# DATA SOURCES - Reference existing resources without modifying them
# -----------------------------------------------------------------------------

# Use default VPC (same as batch module)
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# Get current AWS account ID for IAM policies
data "aws_caller_identity" "current" {}

# Get current region for ARN construction
data "aws_region" "current" {}
