# Disaster Recovery Runbook

## Overview

This runbook provides step-by-step procedures for recovering from common failure scenarios in the Financial Topic Modeling pipeline.

---

## Quick Reference

| Scenario                  | Severity | RTO    | Recovery Action               |
| ------------------------- | -------- | ------ | ----------------------------- |
| Spot instance termination | Low      | 5 min  | Re-run `launch_pipeline.sh`   |
| RDS unavailable           | Medium   | 10 min | Check status, wait or restore |
| Data corruption           | High     | 30 min | Restore from backup or re-run |
| API key invalid           | Low      | 5 min  | Regenerate and update .env    |
| Full disk on EC2          | Medium   | 15 min | Increase volume or clean up   |

---

## Scenario 1: Spot Instance Termination

### Symptoms

- SSH connection drops unexpectedly
- Pipeline logs stop
- AWS Console shows instance as "terminated"

### Root Cause

AWS reclaimed spot instance due to capacity demand.

### Recovery Steps

1. **Verify termination**:

   ```bash
   aws ec2 describe-instances \
     --filters "Name=tag:Name,Values=ftm-pipeline" \
     --query "Reservations[].Instances[].State.Name"
   ```

2. **Check database progress**:

   ```bash
   psql $DATABASE_URL -c "SELECT COUNT(*) FROM firms WHERE processed_at IS NOT NULL"
   ```

3. **Re-run pipeline** (resumes automatically):

   ```bash
   cd cloud/scripts
   ./launch_pipeline.sh
   ```

4. **Monitor new instance**:
   ```bash
   ssh -i ~/.ssh/your-key.pem ubuntu@<new-ip>
   sudo tail -f /var/log/ftm-pipeline.log
   ```

### Prevention

- Use `USE_SPOT=false` for critical runs
- Ensure checkpoint/resume is working before large runs

---

## Scenario 2: RDS Unavailable

### Symptoms

- Pipeline fails with "connection refused" or "timeout"
- Cannot connect via psql or DBeaver

### Root Cause Options

1. RDS instance is stopped (intentional cost saving)
2. Security group misconfiguration
3. RDS instance deleted or failed

### Recovery Steps

**If RDS is stopped**:

```bash
cd cloud/scripts
./start_rds.sh
# Wait ~5 minutes for availability
```

**If security group issue**:

```bash
# Verify EC2 security group allows PostgreSQL
aws ec2 describe-security-groups \
  --group-ids <ec2-sg-id> \
  --query "SecurityGroups[].IpPermissionsEgress"

# Verify RDS security group allows from EC2 SG
aws rds describe-db-instances \
  --db-instance-identifier ftm-db \
  --query "DBInstances[].VpcSecurityGroups"
```

**If RDS deleted**:

```bash
# Re-create infrastructure
cd cloud/terraform
terraform apply
# Re-run pipeline from scratch
```

### Prevention

- Don't delete RDS without backup
- Document before making security group changes

---

## Scenario 3: Data Corruption

### Symptoms

- Query returns unexpected results
- Foreign key violations
- Theme counts don't match topic counts

### Root Cause Options

1. Incomplete transaction during failure
2. Manual database modification
3. Schema migration issue

### Recovery Steps

**Option A: Re-run affected firms**:

```sql
-- Reset specific firm
UPDATE firms SET processed_at = NULL WHERE company_id = 'AAPL';
DELETE FROM topics WHERE firm_id = (SELECT id FROM firms WHERE company_id = 'AAPL');
DELETE FROM sentences WHERE firm_id = (SELECT id FROM firms WHERE company_id = 'AAPL');
```

Then re-run pipeline (will reprocess AAPL).

**Option B: Full reset**:

```sql
-- Nuclear option: clear all data
TRUNCATE firms, sentences, topics, themes RESTART IDENTITY CASCADE;
```

Then re-run pipeline from scratch.

**Option C: Restore from backup** (if enabled):

```bash
# List available snapshots
aws rds describe-db-snapshots \
  --db-instance-identifier ftm-db

# Restore to new instance
aws rds restore-db-instance-from-db-snapshot \
  --db-instance-identifier ftm-db-restored \
  --db-snapshot-identifier <snapshot-id>
```

### Prevention

- Enable automated RDS backups
- Never modify database manually during pipeline run
- Test schema changes on dev database first

---

## Scenario 4: API Key Invalid

### Symptoms

- LLM calls fail with 401/403
- Log shows "Authentication failed" or "Invalid API key"

### Root Cause

- API key expired or revoked
- Key not set in environment
- Wrong key (e.g., test vs. production)

### Recovery Steps

1. **Regenerate API key** at https://x.ai (or provider dashboard)

2. **Update local .env**:

   ```bash
   echo "XAI_API_KEY=new-key-here" >> .env
   ```

3. **Re-upload to S3**:

   ```bash
   ./upload_to_s3.sh
   ```

4. **Re-launch pipeline**:
   ```bash
   ./launch_pipeline.sh
   ```

### Prevention

- Store API key in AWS Secrets Manager (production)
- Set calendar reminder for key rotation

---

## Scenario 5: Full Disk on EC2

### Symptoms

- Pipeline fails with "No space left on device"
- Cannot write logs or checkpoints

### Root Cause

- Root volume too small for data
- Logs filling disk
- Downloaded data larger than expected

### Recovery Steps

1. **Check disk usage** (if instance accessible):

   ```bash
   df -h
   du -sh /home/ubuntu/*
   ```

2. **Clean up** (if possible):

   ```bash
   rm -rf /home/ubuntu/.cache/huggingface/*  # Model cache
   rm -f /home/ubuntu/*.tar.gz  # Downloaded archives
   ```

3. **If instance unrecoverable**, re-launch with larger volume:
   ```bash
   BLOCK_DEVICE_SIZE_GB=100 ./launch_pipeline.sh
   ```

### Prevention

- Default volume size is 50GB (sufficient for MAG7)
- For full runs, use 100GB+
- Monitor disk usage in logs

---

## Scenario 6: CUDA Out of Memory

### Symptoms

- Pipeline fails with "CUDA out of memory"
- GPU utilization spikes to 100% then crashes

### Root Cause

- Batch size too large for GPU memory
- Multiple models loaded simultaneously
- Memory leak in processing loop

### Recovery Steps

1. **Reduce batch size** in config:

   ```yaml
   # cloud/config/production.yaml
   embedding:
     batch_size: 32 # Reduce from 64
   ```

2. **Use smaller embedding model**:

   ```yaml
   embedding:
     model: "all-MiniLM-L6-v2" # Smaller than mpnet
   ```

3. **Re-upload and re-run**:
   ```bash
   ./upload_to_s3.sh
   ./launch_pipeline.sh
   ```

### Prevention

- Test with larger firms (more sentences) before full run
- Monitor GPU memory via `nvidia-smi` during development

---

## Scenario 7: Terraform State Corruption

### Symptoms

- `terraform plan` shows unexpected changes
- Resources exist in AWS but not in state
- "Resource already exists" errors

### Root Cause

- Manual AWS Console changes
- State file deleted or corrupted
- Partial apply interrupted

### Recovery Steps

1. **Import existing resources**:

   ```bash
   terraform import aws_db_instance.main ftm-db
   terraform import aws_s3_bucket.pipeline ftm-pipeline-xxxxx
   ```

2. **Or destroy and recreate** (if no critical data):

   ```bash
   terraform destroy
   terraform apply
   ```

3. **Backup database first** if recreating:
   ```bash
   pg_dump $DATABASE_URL > backup.sql
   ```

### Prevention

- Never modify infrastructure via AWS Console
- Keep `terraform.tfstate` in version control (or use remote state)
- Always run `terraform plan` before `apply`

---

## Emergency Contacts

| Role              | Contact         | Escalation            |
| ----------------- | --------------- | --------------------- |
| Pipeline Owner    | (Your email)    | First responder       |
| AWS Account Admin | (Admin email)   | Billing/access issues |
| Course Instructor | (If applicable) | Academic context      |

---

## Post-Incident Review

After any incident:

1. **Document** what happened (symptoms, root cause, resolution)
2. **Update** this runbook if new scenario encountered
3. **Implement** prevention measures
4. **Test** recovery procedure if not verified
