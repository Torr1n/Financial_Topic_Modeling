#!/bin/bash
# Stop RDS instance to save costs when not in use
# Storage cost while stopped: ~$10/month for 100GB
# Note: RDS auto-starts after 7 days if not manually started

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Get region from Terraform outputs (single source of truth)
cd "$SCRIPT_DIR/../terraform"
AWS_REGION=$(terraform output -raw aws_region 2>/dev/null || echo "us-east-1")
cd "$SCRIPT_DIR"

echo "=== Stopping RDS Instance ==="
echo "Region: $AWS_REGION"

aws rds stop-db-instance \
    --db-instance-identifier ftm-db \
    --region "$AWS_REGION"

echo "RDS stop initiated."
echo ""
echo "Cost while stopped:"
echo "  - Compute: \$0/hour"
echo "  - Storage: ~\$0.10/GB/month = ~\$10/month for 100GB"
echo ""
echo "Note: AWS auto-restarts stopped RDS after 7 days."
echo "Run ./start_rds.sh before your next pipeline run."
