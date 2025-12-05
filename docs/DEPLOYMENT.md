# Cloud Deployment Guide

Step-by-step guide for deploying the Financial Topic Modeling pipeline to AWS.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Infrastructure Setup](#infrastructure-setup)
3. [Code Deployment](#code-deployment)
4. [Running the Pipeline](#running-the-pipeline)
5. [Monitoring & Debugging](#monitoring--debugging)
6. [Verifying Results](#verifying-results)
7. [Cost Management](#cost-management)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### AWS Account Setup

```bash
# 1. Configure AWS CLI
aws configure
# Enter: Access Key, Secret Key, Region (us-east-1), Output format (json)

# 2. Verify configuration
aws sts get-caller-identity
# Should return your account ID
```

### Create EC2 Key Pair

```bash
# Create key pair for SSH access
aws ec2 create-key-pair \
  --key-name ftm-key \
  --query 'KeyMaterial' \
  --output text \
  --region us-east-1 > ~/.ssh/ftm-key.pem

# Set permissions
chmod 400 ~/.ssh/ftm-key.pem

# Verify
aws ec2 describe-key-pairs --key-names ftm-key --region us-east-1
```

### Get Your IP Address

```bash
curl -s https://checkip.amazonaws.com
# Note this IP for terraform.tfvars (format: X.X.X.X/32)
```

### Install Terraform

```bash
# macOS
brew install terraform

# Ubuntu/Debian
sudo apt-get install terraform

# Verify
terraform --version  # Should be 1.0+
```

---

## Infrastructure Setup

### 1. Configure Terraform Variables

```bash
cd cloud/terraform
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars`:

```hcl
aws_region    = "us-east-1"
db_username   = "ftm"
db_password   = "YourSecurePassword123!"  # Change this!
my_ip         = "YOUR.IP.ADDRESS/32"       # From prerequisite step
key_pair_name = "ftm-key"                  # From key pair creation
```

### 2. Deploy Infrastructure

```bash
# Initialize Terraform
terraform init

# Preview changes
terraform plan

# Apply (type 'yes' when prompted)
terraform apply
```

**Expected Output:**

```
Apply complete! Resources: 9 added, 0 changed, 0 destroyed.

Outputs:
aws_region = "us-east-1"
db_endpoint = "ftm-db.xxxxx.us-east-1.rds.amazonaws.com:5432"
db_host = "ftm-db.xxxxx.us-east-1.rds.amazonaws.com"
ec2_security_group_id = "sg-xxxxxxxxx"
instance_profile_name = "ftm-ec2-profile"
s3_bucket_name = "ftm-pipeline-xxxxxxxx"
```

**Resources Created:**

- RDS PostgreSQL instance (`ftm-db`)
- S3 bucket for code/data
- EC2 security group (SSH from your IP)
- RDS security group (PostgreSQL from EC2 only)
- IAM role and instance profile

### 3. Verify in AWS Console

| Service               | Check                               |
| --------------------- | ----------------------------------- |
| RDS                   | `ftm-db` exists, status "Available" |
| S3                    | `ftm-pipeline-xxxxx` bucket exists  |
| EC2 → Security Groups | `ftm-ec2-sg` and `ftm-db-sg` exist  |
| IAM → Roles           | `ftm-ec2-pipeline` exists           |

---

## Code Deployment

### 1. Upload Code to S3

```bash
cd cloud/scripts
chmod +x *.sh

./upload_to_s3.sh
```

**Expected Output:**

```
=== Uploading Pipeline to S3 ===
Target bucket: ftm-pipeline-xxxxxxxx
Packaging code...
Uploading code package...
Uploading CSV data...
=== Upload Complete ===
Code: s3://ftm-pipeline-xxxxxxxx/ftm-pipeline.tar.gz
Data: s3://ftm-pipeline-xxxxxxxx/data/
```

### 2. Configure LLM (Optional)

Create `.env` in project root for LLM summaries:

```bash
cd ../..  # Back to project root
echo "XAI_API_KEY=your-xai-api-key" > .env
```

**Note:** If not set, pipeline uses keyword fallbacks instead of LLM summaries.

---

## Running the Pipeline

### MAG7 Validation Run (Recommended First)

```bash
cd cloud/scripts

# Run with 11 test firms
TEST_MODE=mag7 ./launch_pipeline.sh
```

**Expected Output:**

```
=== Launching FTM Pipeline ===
XAI_API_KEY loaded from .env
Reading configuration from Terraform...
Configuration (from Terraform):
  Region: us-east-1
  S3 Bucket: ftm-pipeline-xxxxxxxx
  DB Host: ftm-db.xxxxx.us-east-1.rds.amazonaws.com
  EC2 SG: sg-xxxxxxxxx
  Key Pair: ftm-key
  Instance Profile: ftm-ec2-profile

Looking up latest Deep Learning AMI...
  AMI: ami-xxxxxxxxx

Test mode: mag7
Instance type: g4dn.2xlarge
Use spot: true

Launching g4dn.2xlarge instance (spot: true)...
Instance launched: i-xxxxxxxxxxxxxxxxx

Waiting for instance to start...

=== Instance Running ===
Instance ID: i-xxxxxxxxxxxxxxxxx
Public IP: X.X.X.X
Region: us-east-1

SSH: ssh -i ~/.ssh/ftm-key.pem ubuntu@X.X.X.X
Logs: sudo tail -f /var/log/ftm-pipeline.log

To terminate: aws ec2 terminate-instances --instance-ids i-xxx --region us-east-1
```

### Full Production Run

```bash
# All firms (3,000+), ~4 hours
./launch_pipeline.sh

# Or limit to first N firms
MAX_FIRMS=100 ./launch_pipeline.sh
```

### Environment Variables

| Variable          | Description              | Default        |
| ----------------- | ------------------------ | -------------- |
| `TEST_MODE`       | `mag7` for 11 test firms | (all firms)    |
| `MAX_FIRMS`       | Limit number of firms    | (unlimited)    |
| `INSTANCE_TYPE`   | EC2 instance type        | `g4dn.2xlarge` |
| `USE_SPOT`        | Use spot pricing         | `true`         |
| `AMI_ID_OVERRIDE` | Manual AMI ID            | (auto-detect)  |

---

## Monitoring & Debugging

### SSH into Instance

```bash
ssh -i ~/.ssh/ftm-key.pem ubuntu@<PUBLIC_IP>
```

### View Pipeline Logs

```bash
# On the EC2 instance
sudo tail -f /var/log/ftm-pipeline.log
```

### Expected Log Sequence

```
=== FTM Pipeline Setup ===
Configuration:
  S3 Bucket: ftm-pipeline-xxxxxxxx
  DB Host: ftm-db.xxxxx.us-east-1.rds.amazonaws.com
  Test Mode: mag7
=== Downloading code from S3 ===
=== Installing dependencies ===
=== Enabling pgvector extension ===
=== Starting Pipeline ===
GPU available: True
Loaded config from: /home/ubuntu/cloud/config/production.yaml
TEST MODE: Running with MAG7 firms only (11 firms)
Initializing UnifiedPipeline
Loading embedding model: all-mpnet-base-v2 (dim=768) on cuda
Using cuML GPU-accelerated UMAP and HDBSCAN
[1/11] Processing firm 21835
Computing embeddings for XXX sentences
Fitting BERTopic on XXX documents
Generating LLM summaries for XX topics
Firm 21835 processed: XX topics
[2/11] Processing firm 29096
...
Starting theme aggregation
Aggregating XXX topics into themes
Generating LLM descriptions for XX themes
Building vector indexes...
=== Pipeline Complete ===
RESULTS SUMMARY
Firms processed: 11
Total sentences: ~5000
Total topics: ~350
Themes discovered: ~19
Pipeline completed in X.X minutes
```

### Check Instance Status

```bash
# From local machine
aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=ftm-pipeline" \
  --query 'Reservations[*].Instances[*].[InstanceId,State.Name,PublicIpAddress]' \
  --output table \
  --region us-east-1
```

---

## Verifying Results

### Query Database (from EC2)

```bash
# On the EC2 instance
PGPASSWORD='YourPassword' psql -h ftm-db.xxxxx.us-east-1.rds.amazonaws.com -U ftm -d ftm
```

```sql
-- Check counts
SELECT 'firms' as table_name, COUNT(*) FROM firms
UNION ALL
SELECT 'sentences', COUNT(*) FROM sentences
UNION ALL
SELECT 'topics', COUNT(*) FROM topics
UNION ALL
SELECT 'themes', COUNT(*) FROM themes;

-- View top themes
SELECT id, name, n_topics, n_firms, LEFT(description, 100) as desc
FROM themes
ORDER BY n_topics DESC
LIMIT 10;

-- Check topic summaries
SELECT id, firm_id, LEFT(representation, 50) as keywords, LEFT(summary, 80) as summary
FROM topics
LIMIT 5;

-- Verify embeddings populated
SELECT
  (SELECT COUNT(*) FROM sentences WHERE embedding IS NOT NULL) as sent_emb,
  (SELECT COUNT(*) FROM topics WHERE embedding IS NOT NULL) as topic_emb,
  (SELECT COUNT(*) FROM themes WHERE embedding IS NOT NULL) as theme_emb;
```

### Query from DBeaver (Local)

**Add your IP to RDS security group:**

```bash
cd cloud/terraform
DB_SG=$(terraform output -raw db_security_group_id)

aws ec2 authorize-security-group-ingress \
  --group-id $DB_SG \
  --protocol tcp \
  --port 5432 \
  --cidr YOUR.IP.ADDRESS/32 \
  --region us-east-1
```

**DBeaver Connection:**

- Host: `ftm-db.xxxxx.us-east-1.rds.amazonaws.com`
- Port: `5432`
- Database: `ftm`
- Username: `ftm`
- Password: (from terraform.tfvars)

---

## Cost Management

### Terminate EC2 After Run

```bash
# Get instance ID
INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=ftm-pipeline" "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text \
  --region us-east-1)

# Terminate
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region us-east-1
```

### Stop RDS (Save Costs)

```bash
cd cloud/scripts
./stop_rds.sh
```

**Cost While Stopped:**

- Compute: $0/hour
- Storage: ~$10/month (100GB)

### Start RDS (Before Next Run)

```bash
./start_rds.sh
# Wait ~5 minutes for availability
```

### Cost Breakdown

| Component         | During Run | After Stop      |
| ----------------- | ---------- | --------------- |
| g4dn.2xlarge spot | ~$0.25/hr  | $0 (terminated) |
| RDS db.t4g.large  | ~$0.08/hr  | $0 (stopped)    |
| RDS storage 100GB | -          | ~$10/month      |
| S3                | negligible | negligible      |
| xAI API           | $1-5/run   | $0              |

**MAG7 Test Run:** ~$1-5 total
**Full Production Run:** ~$5-10 total

### Destroy All Infrastructure

```bash
cd cloud/terraform
terraform destroy
# Type 'yes' to confirm
```

---

## Troubleshooting

### "Permission denied (publickey)" on SSH

```bash
# Check key permissions
chmod 400 ~/.ssh/ftm-key.pem

# Verify key name matches
aws ec2 describe-key-pairs --region us-east-1
```

### "Connection refused" on SSH

- Instance still initializing (wait 2-3 minutes)
- Security group doesn't have your IP

```bash
# Check your current IP
curl -s https://checkip.amazonaws.com

# Update security group if IP changed
aws ec2 authorize-security-group-ingress \
  --group-id <EC2_SG_ID> \
  --protocol tcp \
  --port 22 \
  --cidr NEW.IP.ADDRESS/32 \
  --region us-east-1
```

### "No module named 'cloud'"

```bash
# On EC2, check working directory
pwd  # Should be /home/ubuntu
ls   # Should show cloud/, scripts/, etc.
```

### "pgvector extension not found"

```bash
# Connect to RDS manually
PGPASSWORD='...' psql -h ftm-db.xxx.rds.amazonaws.com -U ftm -d ftm

# Create extension
CREATE EXTENSION IF NOT EXISTS vector;
\q
```

### Spot Instance Terminated

Spot instances can be reclaimed by AWS. Pipeline has checkpoint support:

```bash
# Just re-run - will resume from last completed firm
TEST_MODE=mag7 ./launch_pipeline.sh
```

### Pipeline Hangs on "Installing dependencies"

Disk space issue. Use larger root volume:

```bash
BLOCK_DEVICE_SIZE_GB=100 TEST_MODE=mag7 ./launch_pipeline.sh
```

### "CUDA out of memory"

Use smaller batch size or larger instance:

```bash
INSTANCE_TYPE=g4dn.4xlarge ./launch_pipeline.sh
```

---

## Quick Reference

```bash
# Deploy infrastructure
cd cloud/terraform && terraform apply

# Upload code
cd ../scripts && ./upload_to_s3.sh

# Run MAG7 validation
TEST_MODE=mag7 ./launch_pipeline.sh

# Run full pipeline
./launch_pipeline.sh

# SSH into instance
ssh -i ~/.ssh/ftm-key.pem ubuntu@<IP>

# View logs
sudo tail -f /var/log/ftm-pipeline.log

# Stop RDS (save costs)
./stop_rds.sh

# Start RDS
./start_rds.sh

# Destroy all
cd ../terraform && terraform destroy
```
