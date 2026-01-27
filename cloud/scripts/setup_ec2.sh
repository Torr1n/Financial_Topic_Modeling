#!/bin/bash
# EC2 User Data Script for FTM Pipeline
# Variables are substituted by launch_pipeline.sh using envsubst
set -ex

# Log everything for debugging
exec > >(tee /var/log/ftm-pipeline.log) 2>&1

echo "=== FTM Pipeline Setup $(date) ==="
echo "Running as user: $(whoami)"

# Variables substituted by launch_pipeline.sh (from Terraform outputs + .env)
S3_BUCKET="${S3_BUCKET}"
DB_HOST="${DB_HOST}"
DB_PASSWORD="${DB_PASSWORD}"
XAI_API_KEY="${XAI_API_KEY}"
TEST_MODE="${TEST_MODE}"
MAX_FIRMS="${MAX_FIRMS}"

echo "Configuration:"
echo "  S3 Bucket: $S3_BUCKET"
echo "  DB Host: $DB_HOST"
echo "  Test Mode: ${TEST_MODE:-none}"
echo "  Max Firms: ${MAX_FIRMS:-all}"

# Work in ubuntu home directory
cd /home/ubuntu

# Download code from S3
echo "=== Downloading code from S3 ==="
aws s3 cp "s3://${S3_BUCKET}/ftm-pipeline.tar.gz" .
tar -xzf ftm-pipeline.tar.gz

# Download data if available
echo "=== Downloading data from S3 ==="
mkdir -p data
aws s3 cp "s3://${S3_BUCKET}/data/" ./data/ --recursive || echo "No data files in S3, checking local..."

# If data not in S3, check if CSV is in the tar
if [ ! -f "data/transcripts_2023-01-01_to_2023-03-31_enriched.csv" ]; then
    if [ -f "transcripts_2023-01-01_to_2023-03-31_enriched.csv" ]; then
        mv transcripts_2023-01-01_to_2023-03-31_enriched.csv data/
    fi
fi

# The Deep Learning AMI has conda environments, activate pytorch
echo "=== Activating PyTorch environment ==="
# Prefer the DLAMI conda env when available; otherwise fall back to system Python
PIP_CMD="pip"
PYTHON_CMD="python"
if [ -f "/home/ubuntu/anaconda3/bin/activate" ]; then
    # Deep Learning AMI path
    # shellcheck disable=SC1091
    source /home/ubuntu/anaconda3/bin/activate pytorch
else
    echo "Conda env not found; using system Python"
    PIP_CMD="pip3"
    PYTHON_CMD="python3"
    if ! command -v "$PIP_CMD" >/dev/null 2>&1; then
        sudo apt-get update -y
        sudo apt-get install -y python3-pip
    fi
fi

# Install additional dependencies
echo "=== Installing dependencies ==="
# Avoid filling the small root volume on fresh instances
$PIP_CMD install --no-cache-dir -r cloud/requirements.txt
$PYTHON_CMD -m spacy download en_core_web_sm

# Install cuML for GPU-accelerated UMAP/HDBSCAN (10-100x speedup)
echo "=== Installing cuML for GPU acceleration ==="
# cuML brings its own CUDA libraries which can conflict with pre-installed PyTorch.
# Solution: Install cuML first, then reinstall PyTorch to use cuML's CUDA libs.
if $PIP_CMD install --no-cache-dir cuml-cu12 --extra-index-url https://pypi.nvidia.com; then
    echo "cuML installed, reinstalling PyTorch for CUDA compatibility..."
    $PIP_CMD install --no-cache-dir torch torchvision --force-reinstall --index-url https://download.pytorch.org/whl/cu121
else
    echo "cuML install failed, will use CPU fallback"
fi

# Clean up caches to reclaim space
rm -rf ~/.cache/pip || true
sudo apt-get clean || true
sudo rm -rf /var/lib/apt/lists/* || true

# Set environment variables for pipeline
export DATABASE_URL="postgresql://ftm:${DB_PASSWORD}@${DB_HOST}:5432/ftm"
export XAI_API_KEY="${XAI_API_KEY}"

# Set test mode environment variables
if [ -n "$TEST_MODE" ]; then
    export TEST_MODE="$TEST_MODE"
fi
if [ -n "$MAX_FIRMS" ]; then
    export MAX_FIRMS="$MAX_FIRMS"
fi

# Enable pgvector extension (idempotent)
echo "=== Enabling pgvector extension ==="
PGPASSWORD="${DB_PASSWORD}" psql -h "${DB_HOST}" -U ftm -d ftm -c "CREATE EXTENSION IF NOT EXISTS vector;" || echo "pgvector may already be enabled or requires manual setup"

# Run the unified pipeline
echo "=== Starting Pipeline $(date) ==="
echo "GPU available: $($PYTHON_CMD -c 'import torch; print(torch.cuda.is_available())')"
echo "cuML available: $($PYTHON_CMD -c 'try:
    from cuml import __version__; print(f"yes (v{__version__})")
except: print("no (CPU fallback)")')"

# Use the general pipeline runner
$PYTHON_CMD scripts/run_unified_pipeline.py

echo "=== Pipeline Complete $(date) ==="
echo "Total runtime logged above. Check results in database."

# Optional: Self-terminate to save costs
# Uncomment the following line to auto-terminate after pipeline completes
# aws ec2 terminate-instances --instance-ids $(curl -s http://169.254.169.254/latest/meta-data/instance-id) --region $(curl -s http://169.254.169.254/latest/meta-data/placement/region)
