# Financial Topic Modeling - Cloud Deployment

Minimal AWS infrastructure for running the FTM pipeline at scale.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Default VPC (us-east-1)                      │
│                                                                      │
│   ┌─────────────────┐              ┌──────────────────────────────┐ │
│   │   RDS Postgres  │◄─────────────│   g4dn.2xlarge (Spot)        │ │
│   │   + pgvector    │   port 5432  │   - Deep Learning AMI        │ │
│   │   db.t4g.large  │              │   - Code from S3             │ │
│   └─────────────────┘              └──────────────────────────────┘ │
│                                                                      │
│   ┌─────────────────┐                                                │
│   │   S3 Bucket     │ <- Pipeline code + CSV data                    │
│   │   ftm-pipeline  │                                                │
│   └─────────────────┘                                                │
└─────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

1. **AWS CLI** configured with credentials (`aws configure`)
2. **Terraform** >= 1.0 installed
3. **EC2 Key Pair** created in us-east-1
4. **Your IP address** for SSH access (find at https://whatismyip.com)

## Quick Start

### 1. Configure Variables

```bash
cd cloud/terraform
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your values
```

### 2. Deploy Infrastructure

```bash
terraform init
terraform plan
terraform apply
```

This creates:
- RDS PostgreSQL 15 with pgvector support
- S3 bucket for code/data
- Security groups (EC2 and RDS)
- IAM role and instance profile

### 3. Upload Code to S3

```bash
cd ../scripts
chmod +x *.sh
./upload_to_s3.sh
```

### 4. Set Up Environment (Optional)

Create `.env` in project root for LLM summaries:
```bash
# .env (project root, not committed to git)
# Only XAI_API_KEY is needed - all other config comes from Terraform outputs
XAI_API_KEY=your-xai-api-key  # Optional - falls back to keywords if not set
```

**Note:** DB password, region, and all infrastructure config are read from Terraform outputs automatically.

### 5. Run Pipeline

**First run (MAG7 validation - recommended):**
```bash
# Run with 11 MAG7 firms to validate cloud setup matches local results
TEST_MODE=mag7 ./launch_pipeline.sh
```

**Full run (all firms):**
```bash
./launch_pipeline.sh
```

This launches a g4dn.2xlarge spot instance (~$0.25/hour) that:
1. Downloads code from S3
2. Installs dependencies
3. Enables pgvector
4. Runs the unified pipeline
5. Writes results to RDS

### 6. Monitor Progress

```bash
# SSH into instance
ssh -i ~/.ssh/your-key.pem ubuntu@<public-ip>

# View logs
sudo tail -f /var/log/ftm-pipeline.log
```

### 7. Stop RDS After Run (Save Costs)

```bash
./stop_rds.sh
```

## Cost Management

| Component | Running | Stopped |
|-----------|---------|---------|
| g4dn.2xlarge spot | ~$0.25/hr | $0 (terminated) |
| RDS db.t4g.large | ~$0.08/hr | $0 (stopped) |
| RDS storage 100GB | - | ~$10/month |
| S3 | negligible | negligible |

**Estimated cost per quarterly run:** ~$1.30 (compute only)

## Commands Reference

```bash
# Start RDS (if stopped)
./start_rds.sh

# Upload code/data to S3
./upload_to_s3.sh

# Launch pipeline - MAG7 validation (11 firms, ~10-15 min)
TEST_MODE=mag7 ./launch_pipeline.sh

# Launch pipeline - full run (all firms, ~2-4 hours)
./launch_pipeline.sh

# Launch pipeline - limited firms (e.g., first 100)
MAX_FIRMS=100 ./launch_pipeline.sh

# Stop RDS (after pipeline)
./stop_rds.sh

# Destroy all infrastructure
cd ../terraform && terraform destroy
```

## Troubleshooting

### "pgvector extension not found"
RDS PostgreSQL 15 supports pgvector, but it may need to be enabled manually:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### "Spot instance terminated"
Spot instances can be reclaimed by AWS. The pipeline has checkpoint/resume:
1. Re-run `./launch_pipeline.sh`
2. Pipeline resumes from last completed firm

### "Permission denied" on scripts
```bash
chmod +x cloud/scripts/*.sh
```

### "Database connection refused"
1. Check RDS is running: `aws rds describe-db-instances --db-instance-identifier ftm-db`
2. Check security group allows your EC2: EC2 must use `ftm-ec2-sg`

## Security Notes

- RDS is publicly accessible but restricted to EC2 security group
- SSH restricted to your IP only
- Secrets passed via user-data (acceptable for dev)
- For production: use AWS Secrets Manager or SSM Parameter Store
