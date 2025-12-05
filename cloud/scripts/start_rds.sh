#!/bin/bash
# Start RDS instance before running pipeline
# RDS takes ~5 minutes to become available after starting

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Get region from Terraform outputs (single source of truth)
cd "$SCRIPT_DIR/../terraform"
AWS_REGION=$(terraform output -raw aws_region 2>/dev/null || echo "us-east-1")
cd "$SCRIPT_DIR"

echo "=== Starting RDS Instance ==="
echo "Region: $AWS_REGION"

aws rds start-db-instance \
    --db-instance-identifier ftm-db \
    --region "$AWS_REGION"

echo "RDS starting... waiting for availability (this takes ~5 minutes)"

aws rds wait db-instance-available \
    --db-instance-identifier ftm-db \
    --region "$AWS_REGION"

echo "=== RDS Available ==="

# Show endpoint
ENDPOINT=$(aws rds describe-db-instances \
    --db-instance-identifier ftm-db \
    --region "$AWS_REGION" \
    --query 'DBInstances[0].Endpoint.Address' \
    --output text)

echo "Endpoint: $ENDPOINT"
echo "You can now run: ./launch_pipeline.sh"
