#!/bin/bash
# Launch EC2 spot instance to run the pipeline
# All config read from Terraform outputs (single source of truth)
#
# Prerequisites:
#   1. terraform apply completed
#   2. upload_to_s3.sh completed
#   3. .env file with XAI_API_KEY (optional, for LLM summaries)
#
# Environment Variables (optional):
#   TEST_MODE=mag7    Run with MAG7 firms only (for validation)
#   MAX_FIRMS=100     Limit number of firms to process

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=== Launching FTM Pipeline ==="

# Load XAI_API_KEY from .env (optional - only secret not in Terraform)
XAI_API_KEY=""
if [ -f "$PROJECT_ROOT/.env" ]; then
    source "$PROJECT_ROOT/.env"
    if [ -n "$XAI_API_KEY" ]; then
        echo "XAI_API_KEY loaded from .env"
    fi
fi

if [ -z "$XAI_API_KEY" ]; then
    echo "Warning: XAI_API_KEY not set - LLM summaries will fallback to keywords"
fi

# Get ALL config from Terraform outputs (single source of truth)
echo "Reading configuration from Terraform..."
cd "$SCRIPT_DIR/../terraform"

AWS_REGION=$(terraform output -raw aws_region)
EC2_SG_ID=$(terraform output -raw ec2_security_group_id)
S3_BUCKET=$(terraform output -raw s3_bucket_name)
DB_HOST=$(terraform output -raw db_host)
DB_PASSWORD=$(terraform output -raw db_password)
KEY_PAIR_NAME=$(terraform output -raw key_pair_name)
INSTANCE_PROFILE=$(terraform output -raw instance_profile_name)

cd "$SCRIPT_DIR"

echo "Configuration (from Terraform):"
echo "  Region: $AWS_REGION"
echo "  S3 Bucket: $S3_BUCKET"
echo "  DB Host: $DB_HOST"
echo "  EC2 SG: $EC2_SG_ID"
echo "  Key Pair: $KEY_PAIR_NAME"
echo "  Instance Profile: $INSTANCE_PROFILE"

# Dynamic AMI lookup - try SSM parameter for latest DLAMI PyTorch GPU (Ubuntu22, OSS Nvidia driver), fallback to describe-images, then to override
echo ""
echo "Looking up latest Deep Learning AMI..."
AMI_ID=""

# 1) SSM parameter (managed by AWS DLAMI team) - Ubuntu 22.04, OSS Nvidia driver, PyTorch 2.7
AMI_ID=$(aws ssm get-parameter \
    --name "/aws/service/deeplearning/ami/x86_64/oss-nvidia-driver-gpu-pytorch-2.7-ubuntu-22.04/latest/ami-id" \
    --region "$AWS_REGION" \
    --query 'Parameter.Value' \
    --output text 2>/dev/null || echo "")

# 2) Fallback to describe-images if SSM unavailable
if [ -z "$AMI_ID" ] || [ "$AMI_ID" == "None" ]; then
    AMI_ID=$(aws ec2 describe-images \
        --region "$AWS_REGION" \
        --owners amazon \
        --filters "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch *Ubuntu 22.04*" \
                  "Name=state,Values=available" \
        --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
        --output text 2>/dev/null || echo "")
fi

# 3) Allow manual override if both fail
if [ -n "$AMI_ID_OVERRIDE" ]; then
    AMI_ID="$AMI_ID_OVERRIDE"
fi

if [ -z "$AMI_ID" ] || [ "$AMI_ID" == "None" ]; then
    echo "Error: Could not find Deep Learning AMI in $AWS_REGION. Set AMI_ID_OVERRIDE=<ami-id> to proceed."
    exit 1
fi
echo "  AMI: $AMI_ID"

# Pass optional test parameters
TEST_MODE="${TEST_MODE:-}"
MAX_FIRMS="${MAX_FIRMS:-}"
# Allow instance type override and on-demand toggle
INSTANCE_TYPE="${INSTANCE_TYPE:-g4dn.2xlarge}"
USE_SPOT="${USE_SPOT:-true}"
# Allow root volume size override (GB)
BLOCK_DEVICE_SIZE_GB="${BLOCK_DEVICE_SIZE_GB:-50}"

echo ""
if [ -n "$TEST_MODE" ]; then
    echo "Test mode: $TEST_MODE"
fi
if [ -n "$MAX_FIRMS" ]; then
    echo "Max firms: $MAX_FIRMS"
fi
echo "Instance type: $INSTANCE_TYPE"
echo "Use spot: $USE_SPOT"
echo "Root volume size (GB): $BLOCK_DEVICE_SIZE_GB"

# Substitute variables into setup script
# Only substitute the variables we intend to template to avoid clobbering
# runtime shell variables (e.g., PIP_CMD inside setup_ec2.sh).
export S3_BUCKET DB_HOST DB_PASSWORD XAI_API_KEY TEST_MODE MAX_FIRMS
envsubst '${S3_BUCKET} ${DB_HOST} ${DB_PASSWORD} ${XAI_API_KEY} ${TEST_MODE} ${MAX_FIRMS}' \
    < setup_ec2.sh > /tmp/setup_ec2_rendered.sh

echo ""
echo "Launching $INSTANCE_TYPE instance (spot: $USE_SPOT)..."

RUN_INST_ARGS=(
    --region "$AWS_REGION"
    --image-id "$AMI_ID"
    --instance-type "$INSTANCE_TYPE"
    --key-name "$KEY_PAIR_NAME"
    --security-group-ids "$EC2_SG_ID"
    --iam-instance-profile "Name=$INSTANCE_PROFILE"
    --user-data file:///tmp/setup_ec2_rendered.sh
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$BLOCK_DEVICE_SIZE_GB,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]"
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=ftm-pipeline}]'
    --query 'Instances[0].InstanceId'
    --output text
)

if [ "$USE_SPOT" != "false" ]; then
    RUN_INST_ARGS+=(--instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time"}}')
else
    echo "Launching on-demand instance (USE_SPOT=false)"
fi

INSTANCE_ID=$(aws ec2 run-instances "${RUN_INST_ARGS[@]}")

echo "Instance launched: $INSTANCE_ID"
echo ""

# Wait for instance to be running
echo "Waiting for instance to start..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$AWS_REGION"

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --region "$AWS_REGION" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo ""
echo "=== Instance Running ==="
echo "Instance ID: $INSTANCE_ID"
echo "Public IP: $PUBLIC_IP"
echo "Region: $AWS_REGION"
echo ""
echo "SSH: ssh -i ~/.ssh/${KEY_PAIR_NAME}.pem ubuntu@$PUBLIC_IP"
echo "Logs: sudo tail -f /var/log/ftm-pipeline.log"
echo ""
echo "To terminate: aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $AWS_REGION"

# Cleanup
rm -f /tmp/setup_ec2_rendered.sh
