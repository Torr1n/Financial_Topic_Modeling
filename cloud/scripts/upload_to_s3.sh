#!/bin/bash
# Upload pipeline code and data to S3
# Run this after terraform apply and before launching EC2

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=== Uploading Pipeline to S3 ==="

# Get S3 bucket name from Terraform
cd "$SCRIPT_DIR/../terraform"
S3_BUCKET=$(terraform output -raw s3_bucket_name)
echo "Target bucket: $S3_BUCKET"

# Package code (exclude unnecessary files)
cd "$PROJECT_ROOT"
echo "Packaging code..."
tar -czf /tmp/ftm-pipeline.tar.gz \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='.pytest_cache' \
    --exclude='*.egg-info' \
    --exclude='venv' \
    --exclude='.env' \
    --exclude='terraform.tfvars' \
    --exclude='*.tfstate*' \
    cloud/ scripts/ requirements.txt 2>/dev/null || \
tar -czf /tmp/ftm-pipeline.tar.gz \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='.pytest_cache' \
    --exclude='*.egg-info' \
    --exclude='venv' \
    --exclude='.env' \
    --exclude='terraform.tfvars' \
    --exclude='*.tfstate*' \
    cloud/ scripts/

echo "Uploading code package..."
aws s3 cp /tmp/ftm-pipeline.tar.gz "s3://$S3_BUCKET/ftm-pipeline.tar.gz"

# Upload CSV data if present
CSV_FILE="transcripts_2023-01-01_to_2023-03-31_enriched.csv"
if [ -f "$PROJECT_ROOT/$CSV_FILE" ]; then
    echo "Uploading CSV data ($CSV_FILE)..."
    aws s3 cp "$PROJECT_ROOT/$CSV_FILE" "s3://$S3_BUCKET/data/$CSV_FILE"
else
    echo "Warning: $CSV_FILE not found, skipping data upload"
fi

# Cleanup
rm -f /tmp/ftm-pipeline.tar.gz

echo "=== Upload Complete ==="
echo "Code: s3://$S3_BUCKET/ftm-pipeline.tar.gz"
echo "Data: s3://$S3_BUCKET/data/"
