# Backpressure & Kill Switches

## Overview

This document describes mechanisms for controlling resource consumption, preventing runaway costs, and safely stopping pipeline execution.

---

## Backpressure Controls

### 1. LLM Rate Limiting

**Problem**: Uncontrolled concurrent LLM requests can trigger rate limits or accumulate excessive costs.

**Control**: Semaphore-based concurrency limit

```python
# From cloud/src/llm/xai_client.py
class XAIClient:
    def __init__(self, config):
        self.max_concurrent = config.get("max_concurrent", 50)
        self._semaphore = asyncio.Semaphore(self.max_concurrent)

    async def generate_summary(self, topic):
        async with self._semaphore:  # Blocks if 50 requests in flight
            return await self._call_api(topic)
```

**Configuration**:

```yaml
# cloud/config/production.yaml
llm:
  max_concurrent: 50 # Max simultaneous API calls
  timeout: 30 # Seconds per request
  max_retries: 3 # Retry on failure
```

**Behavior**:

- Requests 1-50: Execute immediately
- Requests 51+: Queue until slot available
- Effect: Smooth API usage, avoid 429 errors

### 2. Firm Processing Limit

**Problem**: Full dataset (3,000+ firms) is expensive and time-consuming.

**Control**: `MAX_FIRMS` environment variable

```bash
# Process only first 100 firms
MAX_FIRMS=100 ./launch_pipeline.sh

# Process only MAG7 test firms
TEST_MODE=mag7 ./launch_pipeline.sh
```

**Implementation**:

```python
# From scripts/run_unified_pipeline.py
max_firms = os.environ.get("MAX_FIRMS")
if max_firms:
    firms = firms[:int(max_firms)]
    logger.info(f"Limited to {max_firms} firms")
```

### 3. Database Connection Pool

**Problem**: Excessive database connections can exhaust RDS limits.

**Control**: SQLAlchemy connection pool

```python
# From cloud/src/database/connection.py
engine = create_engine(
    DATABASE_URL,
    pool_size=5,        # Max active connections
    max_overflow=10,    # Burst capacity
    pool_timeout=30,    # Wait for connection
)
```

**Effect**: Pipeline uses max 15 connections regardless of concurrent operations.

---

## Kill Switches

### 1. Manual EC2 Termination

**When**: Pipeline is misbehaving, runaway costs, or emergency stop.

**Method**:

```bash
# Find instance ID
aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=ftm-pipeline" \
  --query "Reservations[].Instances[].InstanceId"

# Terminate
aws ec2 terminate-instances --instance-ids i-0abc123
```

**Effect**:

- Instance terminated immediately
- Pipeline stops mid-execution
- Data up to last checkpoint preserved
- Re-run resumes from checkpoint

### 2. RDS Stop (Cost Kill Switch)

**When**: Pause all costs when not actively running.

**Method**:

```bash
./stop_rds.sh
```

**Effect**:

- RDS compute stops ($0.08/hr → $0)
- Storage charges continue (~$10/mo)
- Database unavailable until started
- Data preserved

### 3. LLM API Key Revocation

**When**: Suspected key compromise or emergency cost stop.

**Method**:

1. Go to https://x.ai/dashboard
2. Revoke current API key
3. Generate new key
4. Update `.env` file

**Effect**:

- All LLM calls fail immediately
- Pipeline continues but topics have no summaries
- No further LLM charges

### 4. Terraform Destroy (Nuclear Option)

**When**: Complete teardown needed.

**Method**:

```bash
cd cloud/terraform
terraform destroy
```

**Effect**:

- ALL resources deleted (RDS, S3, security groups)
- **DATA LOST** unless backed up
- No further charges
- Must re-apply to recreate

**WARNING**: Export data first:

```bash
pg_dump $DATABASE_URL > backup.sql
aws s3 sync s3://ftm-pipeline-xxx ./backup/
```

---

## Automatic Kill Switches

### 1. Spot Instance Termination

**Trigger**: AWS reclaims capacity.

**Effect**: Instance terminates with 2-minute warning.

**Recovery**: Re-run `launch_pipeline.sh` (resumes from checkpoint).

### 2. Pipeline Self-Termination

**Trigger**: Pipeline completes successfully.

**Configuration** (optional, disabled by default):

```bash
# In setup_ec2.sh (uncomment to enable)
# shutdown -h now  # Self-terminate after completion
```

### 3. Cost Budget Alert

**Trigger**: Spending exceeds threshold.

**Configuration**:

```bash
aws budgets create-budget \
  --budget-name "ftm-kill-switch" \
  --budget-limit Amount=100,Unit=USD \
  ...
```

**Effect**: Email notification (manual action required).

---

## Graceful Shutdown

### Signal Handling

```python
# From unified_pipeline.py (recommended implementation)
import signal

def handle_shutdown(signum, frame):
    logger.warning("Shutdown signal received, completing current firm...")
    global shutdown_requested
    shutdown_requested = True

signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)

# In processing loop
for firm in firms:
    if shutdown_requested:
        logger.info("Graceful shutdown after completing firm")
        break
    process_firm(firm)
```

**Effect**:

- Ctrl+C or SIGTERM triggers graceful shutdown
- Current firm completes and commits
- Pipeline exits cleanly
- No data loss

---

## Control Summary

| Control           | Type         | Trigger | Effect                        |
| ----------------- | ------------ | ------- | ----------------------------- |
| Semaphore (50)    | Backpressure | Auto    | Limits concurrent LLM calls   |
| MAX_FIRMS         | Backpressure | Manual  | Limits firms processed        |
| Connection Pool   | Backpressure | Auto    | Limits DB connections         |
| EC2 Terminate     | Kill Switch  | Manual  | Stops pipeline immediately    |
| RDS Stop          | Kill Switch  | Manual  | Stops database costs          |
| API Key Revoke    | Kill Switch  | Manual  | Stops LLM costs               |
| Terraform Destroy | Kill Switch  | Manual  | Deletes all resources         |
| Spot Termination  | Automatic    | AWS     | Stops instance (resume later) |
| Budget Alert      | Warning      | AWS     | Notification to act           |

---

## Emergency Procedures

### Cost Runaway

1. **Immediate**: `aws ec2 terminate-instances --instance-ids <id>`
2. **Within 1 hour**: `./stop_rds.sh`
3. **Review**: Check AWS Cost Explorer for cause
4. **Prevent**: Adjust MAX_FIRMS, review LLM usage

### Rate Limit Hit

1. **Immediate**: Pipeline retries automatically (3x)
2. **If persistent**: Reduce `max_concurrent` in config
3. **If blocked**: Wait for rate limit reset (usually 1 min)
4. **Long-term**: Contact xAI for higher limits

### Out of Memory

1. **Immediate**: Instance may crash or kill process
2. **Recovery**: Re-launch with smaller batch size
3. **Config**: Reduce `batch_size` in production.yaml
4. **Alternative**: Use larger instance (g4dn.4xlarge)
