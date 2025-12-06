# Incident Response

## Overview

This document provides procedures for responding to incidents affecting the Financial Topic Modeling pipeline.

---

## Incident Severity Levels

| Level             | Description                | Response Time | Examples                              |
| ----------------- | -------------------------- | ------------- | ------------------------------------- |
| **P1 - Critical** | Complete system failure    | Immediate     | RDS deleted, all data lost            |
| **P2 - High**     | Major functionality broken | <1 hour       | Pipeline won't start, API key invalid |
| **P3 - Medium**   | Degraded performance       | <4 hours      | Slow processing, partial failures     |
| **P4 - Low**      | Minor issues               | <24 hours     | Log warnings, cosmetic issues         |

---

## Incident Response Flow

```
┌─────────────┐
│   Detect    │
│  (Logs/     │
│   Alerts)   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Assess    │
│  Severity   │
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐
│  Mitigate   │────▶│  Document   │
│  (Stop      │     │  (What      │
│   Bleeding) │     │   happened) │
└──────┬──────┘     └─────────────┘
       │
       ▼
┌─────────────┐
│   Resolve   │
│  (Fix root  │
│   cause)    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Post-      │
│  Mortem     │
└─────────────┘
```

---

## Common Incidents

### Incident: Pipeline Fails to Start

**Symptoms**:

- `launch_pipeline.sh` exits with error
- No EC2 instance created

**Diagnosis**:

```bash
# Check Terraform outputs exist
cd cloud/terraform
terraform output

# Check AWS credentials
aws sts get-caller-identity

# Check key pair exists
aws ec2 describe-key-pairs --key-names your-key-name
```

**Resolution**:

1. Re-run `terraform apply` if outputs missing
2. Configure AWS credentials if identity check fails
3. Create key pair if missing: `aws ec2 create-key-pair --key-name ftm-key`

**Prevention**: Run `terraform plan` before each pipeline run.

---

### Incident: Database Connection Failed

**Symptoms**:

- Log shows "connection refused" or "timeout"
- Pipeline fails at initialization

**Diagnosis**:

```bash
# Check RDS status
aws rds describe-db-instances \
  --db-instance-identifier ftm-db \
  --query "DBInstances[].DBInstanceStatus"

# Check security groups
aws ec2 describe-security-groups --group-ids sg-xxx
```

**Resolution**:

1. If status is "stopped": `./start_rds.sh`
2. If security group issue: Verify EC2 SG allows outbound, RDS SG allows from EC2 SG
3. If credentials wrong: Check `terraform output db_password`

**Prevention**: Always run `./start_rds.sh` before pipeline.

---

### Incident: LLM API Errors

**Symptoms**:

- Topics created without summaries
- Log shows 401, 403, or 429 errors

**Diagnosis**:

```bash
# Test API key
curl https://api.x.ai/v1/models \
  -H "Authorization: Bearer $XAI_API_KEY"
```

**Resolution**:
| Error | Cause | Fix |
|-------|-------|-----|
| 401 | Invalid key | Regenerate at x.ai |
| 403 | Forbidden | Check account status |
| 429 | Rate limit | Reduce max_concurrent |
| 500 | API down | Wait and retry |

**Prevention**: Test API key before large runs.

---

### Incident: Out of Memory

**Symptoms**:

- Process killed by OOM killer
- Log shows "CUDA out of memory"
- Instance becomes unresponsive

**Diagnosis**:

```bash
# Check dmesg for OOM
dmesg | grep -i "out of memory"

# Check GPU memory
nvidia-smi
```

**Resolution**:

1. Reduce batch size in config
2. Use smaller embedding model
3. Use larger instance type

**Prevention**: Monitor memory during test runs.

---

### Incident: Spot Instance Terminated

**Symptoms**:

- SSH connection drops
- Instance status shows "terminated"

**Diagnosis**:

```bash
aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=ftm-pipeline" \
  --query "Reservations[].Instances[].[InstanceId,State.Name,StateReason.Message]"
```

**Resolution**:

1. Verify checkpoint in database
2. Re-run `./launch_pipeline.sh`
3. Pipeline resumes automatically

**Prevention**: This is expected behavior. Ensure checkpoint/resume is working.

---

### Incident: Data Corruption

**Symptoms**:

- Queries return unexpected results
- Foreign key violations
- Inconsistent counts

**Diagnosis**:

```sql
-- Check referential integrity
SELECT t.id, t.firm_id
FROM topics t
LEFT JOIN firms f ON t.firm_id = f.id
WHERE f.id IS NULL;

-- Check orphaned sentences
SELECT COUNT(*) FROM sentences WHERE topic_id IS NULL AND firm_id IS NOT NULL;
```

**Resolution**:

1. Identify affected data
2. Delete corrupted records
3. Re-process affected firms

**Prevention**: Never manually modify database during pipeline run.

---

## Escalation Matrix

| Severity | First Responder | Escalate To       | Timeframe |
| -------- | --------------- | ----------------- | --------- |
| P1       | Project Lead    | Course Instructor | Immediate |
| P2       | Project Lead    | -                 | 1 hour    |
| P3       | Project Lead    | -                 | 4 hours   |
| P4       | Project Lead    | -                 | Next day  |

---

## Communication Template

### Incident Notification

```
Subject: [P{LEVEL}] FTM Pipeline Incident - {Brief Description}

Status: {Investigating | Mitigating | Resolved}
Impact: {Description of impact}
Start Time: {When issue began}
Detection: {How issue was detected}

Current Actions:
- {Action 1}
- {Action 2}

Next Update: {Time of next update}
```

### Post-Mortem Template

```
## Incident Summary
- Date: {Date}
- Duration: {Start} to {End}
- Severity: P{Level}
- Impact: {Description}

## Timeline
- {Time}: {Event}
- {Time}: {Event}

## Root Cause
{Detailed explanation}

## Resolution
{What fixed it}

## Action Items
- [ ] {Preventive measure 1}
- [ ] {Preventive measure 2}

## Lessons Learned
{What we learned}
```

---

## Monitoring Checklist

### Before Each Run

- [ ] RDS is running (`./start_rds.sh`)
- [ ] API key is valid (test with curl)
- [ ] Terraform outputs exist
- [ ] S3 has latest code (`./upload_to_s3.sh`)

### During Run

- [ ] SSH accessible
- [ ] Logs streaming (`tail -f /var/log/ftm-pipeline.log`)
- [ ] GPU utilized (`nvidia-smi`)
- [ ] No error patterns in logs

### After Run

- [ ] All firms processed
- [ ] Themes created
- [ ] RDS stopped (`./stop_rds.sh`)
- [ ] EC2 terminated (automatic for spot)

---

## Recovery Runbook References

| Scenario         | Runbook Section        |
| ---------------- | ---------------------- |
| Spot termination | DR Runbook: Scenario 1 |
| RDS unavailable  | DR Runbook: Scenario 2 |
| Data corruption  | DR Runbook: Scenario 3 |
| API key invalid  | DR Runbook: Scenario 4 |
| Full disk        | DR Runbook: Scenario 5 |
| CUDA OOM         | DR Runbook: Scenario 6 |
| Terraform issues | DR Runbook: Scenario 7 |
