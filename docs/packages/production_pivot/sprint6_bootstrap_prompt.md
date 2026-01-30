# Sprint 6 Bootstrap: Integration Testing & Production Validation

## Context: You Are Continuing from Sprint 5

The previous instance deployed Sprint 5 infrastructure (vLLM on ECS + Step Functions). **All infrastructure is deployed and running, but has NOT been end-to-end tested.** Your task is to complete integration testing and proceed to production validation.

**Critical reads before doing anything:**
1. `docs/packages/production_pivot/sprint5_instance_summary.md` - What was built and why
2. `docs/deployment/sprint5_deployment_guide.md` - How to operate the infrastructure

---

## Guiding Principles (Non-Negotiable)

These principles come from the project's founding document and must guide every decision:

> "The best engineers write code my mom could read."

1. **Simplicity over complexity** - Boring technology, obvious patterns
2. **Document the "why"** - Code comments explain reasoning, not mechanics
3. **Under-engineer the "how"** - Minimum viable complexity for current needs
4. **Test as you go** - Validate before building on top
5. **No premature abstraction** - Three similar lines > one clever helper

**Anti-patterns to avoid:**
- Over-engineering for hypothetical futures
- Adding features beyond what's requested
- Clever code that requires mental compilation
- Leaving resources running that cost money

---

## Current State (As Left by Previous Instance)

### Deployed (Verify Current Status)

| Component | Status | Evidence |
|-----------|--------|----------|
| vLLM ECS Service | ✅ Deployed | Verify with `describe-services` (may be scaled down) |
| ALB Target | ✅ Deployed | Verify health once vLLM is running |
| Step Functions | ✅ Deployed | State machine exists, not yet executed |
| Lambda Functions | ✅ Deployed | Unit tests pass |
| Batch Integration | ✅ Updated | LLM_BASE_URL in job definition |

### NOT Yet Tested

| Item | Risk Level | Why It Matters |
|------|------------|----------------|
| Step Functions end-to-end | **HIGH** | Core pipeline functionality |
| Batch → vLLM connectivity | **HIGH** | Will fail silently if broken |
| Topic naming quality | Medium | May need prompt tuning |
| Multi-batch throughput | Medium | Could hit resource limits |

### ⚠️ COST ALERT: vLLM May Be Running

vLLM might be scaled up from the previous session. **If it is running, it costs ~$0.37/hour (~$9/day).**

Check if still needed:
```bash
aws ecs describe-services --cluster ftm-vllm-cluster --services ftm-vllm-service \
  --query 'services[0].{desired:desiredCount,running:runningCount}'
```

Scale down when not actively testing:
```bash
aws ecs update-service --cluster ftm-vllm-cluster --service ftm-vllm-service --desired-count 0
aws autoscaling set-desired-capacity --auto-scaling-group-name ftm-vllm-asg --desired-capacity 0
```

---

## Your Immediate Tasks (Priority Order)

### P0: End-to-End Step Functions Test

**Goal:** Verify the entire pipeline works from Step Functions trigger to Batch job completion.

#### Step 1: Check Prefetch Data Exists

```bash
aws lambda invoke --function-name ftm-prefetch-check \
  --payload '{"quarter": "2023Q1", "bucket": "ftm-pipeline-78ea68c8"}' \
  --cli-binary-format raw-in-base64-out /dev/stdout
```

- If `"exists": true` → proceed to Step 2
- If `"exists": false` → you need prefetch data first (coordinate with user)

#### Step 2: Ensure vLLM is Running

```bash
# Check service status
aws ecs describe-services --cluster ftm-vllm-cluster --services ftm-vllm-service \
  --query 'services[0].{desired:desiredCount,running:runningCount}'

# If running=0, scale up
aws ecs update-service --cluster ftm-vllm-cluster --service ftm-vllm-service --desired-count 1
aws autoscaling set-desired-capacity --auto-scaling-group-name ftm-vllm-asg --desired-capacity 1

# Wait for healthy (5-10 minutes for model load)
aws logs tail /ecs/ftm-vllm --since 10m --follow
```

#### Step 3: Start Execution (Small Test)

```bash
aws stepfunctions start-execution \
  --state-machine-arn arn:aws:states:us-east-1:666938415731:stateMachine:ftm-quarter-processor \
  --name "integration-test-$(date +%Y%m%d-%H%M%S)" \
  --input '{"quarter": "2023Q1", "bucket": "ftm-pipeline-78ea68c8", "batch_size": 50}'
```

Use small `batch_size` (50) for quick validation.

#### Step 4: Monitor Execution

```bash
# Get execution ARN from start-execution output, then:
aws stepfunctions describe-execution --execution-arn <ARN> \
  --query '{status:status,error:error,cause:cause}'

# Or watch in console:
# https://us-east-1.console.aws.amazon.com/states/home?region=us-east-1#/statemachines/view/arn:aws:states:us-east-1:666938415731:stateMachine:ftm-quarter-processor
```

#### Step 5: Verify Batch Jobs Hit vLLM

When Batch jobs run, check their logs:
```bash
aws logs tail /aws/batch/ftm --since 30m --follow
```

**What to look for:**
- `LLM_BASE_URL=http://internal-ftm-vllm-alb-...` (using vLLM, not Grok)
- Topic summaries being generated (no connection errors)
- No timeouts connecting to ALB

### P1: Quality Validation

After successful execution:

1. Check output in S3: `s3://ftm-pipeline-78ea68c8/intermediate/firm-topics/quarter=2023Q1/`
2. Review topic summaries - are they meaningful and generalizable?
3. Compare to any existing Grok API baseline if available

### P2: Verify ADR-006

Confirm ADR-006 reflects the actual deployed configuration:
- Host network mode (not awsvpc) - required for HuggingFace downloads
- Target group type `instance` (not `ip`) - required for host mode
- Container-level memory specification (not task-level)
- One task per instance limitation
- No HuggingFace token required (public model)

---

## Key Architecture Decisions (Understand Before Modifying)

### Why Host Network Mode?

**Original plan:** Use `awsvpc` (each task gets own ENI)

**Problem discovered:** Tasks couldn't reach HuggingFace to download model. In `awsvpc` mode with default VPC (no NAT Gateway), task ENIs don't get public IPs.

**Solution:** Use `host` network mode - container shares EC2 instance's network (has public IP).

**Trade-off:** One vLLM task per instance (fixed port 8000). Acceptable for batch workloads.

### Why Manual Scaling (Not Auto-Scaling)?

**Rationale:**
1. Predictability over elasticity for batch workloads
2. Avoid cold starts mid-run (model takes 5-10 min to load)
3. Simpler operational model
4. ADR-006 calls for "keep warm" during runs

**Pattern:** Scale up before processing, scale down after.

### Why batch_ids Only in Map State?

**Constraint:** Step Functions has 256KB payload limit per state.

**Solution:**
- Lambda creates JSONL manifest in S3 with full firm_ids
- Map state receives only batch_ids (strings like `"2023Q1_batch_0000"`)
- Each Batch job reads its firm_ids from S3 manifest at runtime

### Why SSM for Cross-Module Config?

ECS and Batch are separate Terraform roots (different state files). SSM Parameter Store provides decoupling:
- ECS module writes `/ftm/vllm/base_url`
- Batch module reads and injects into job definition

---

## Troubleshooting Guide

### Step Functions Fails at CheckPrefetch

**Error:** `PrefetchRequiredError`

**Cause:** No prefetch data for the quarter

**Fix:** Run prefetch (requires WRDS access with MFA from fixed-IP machine)

### Batch Jobs Can't Connect to vLLM

**Symptom:** Timeout errors in Batch logs

**Debug:**
```bash
# Check ALB security group allows Batch SG
aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=ftm-vllm-alb-sg,ftm-batch-sg" \
  --query 'SecurityGroups[*].{GroupId:GroupId,GroupName:GroupName,Ingress:IpPermissions}'
```

**Likely causes:**
- Security group misconfiguration
- vLLM not running (check ECS service)
- ALB target unhealthy

### vLLM Container Keeps Restarting

**Symptom:** Task count fluctuates, never stabilizes

**Debug:**
```bash
aws logs tail /ecs/ftm-vllm --since 30m
```

**Likely causes:**
- Out of memory (check for CUDA OOM errors)
- Model download failed (check for HuggingFace errors)
- Health check timeout (model still loading)

### Step Functions Stuck in Map State

**Debug:**
```bash
# Check if Batch jobs exist
aws batch list-jobs --job-queue ftm-queue-main --job-status RUNNABLE
aws batch list-jobs --job-queue ftm-queue-main --job-status RUNNING
```

**Likely cause:** Batch compute environment at 0 capacity or no Spot instances available

---

## AWS Resource Reference

| Resource | Name/Value |
|----------|------------|
| S3 Bucket | `ftm-pipeline-78ea68c8` |
| ECS Cluster | `ftm-vllm-cluster` |
| ECS Service | `ftm-vllm-service` |
| ASG | `ftm-vllm-asg` |
| ALB | `ftm-vllm-alb` (internal) |
| Target Group | `ftm-vllm-tg2` |
| State Machine | `ftm-quarter-processor` |
| Batch Queue | `ftm-queue-main` |
| Batch Job Def | `ftm-firm-processor` |
| SSM Parameter | `/ftm/vllm/base_url` |

---

## File Locations

| Purpose | Path |
|---------|------|
| Sprint 5 Summary | `docs/packages/production_pivot/sprint5_instance_summary.md` |
| Deployment Guide | `docs/deployment/sprint5_deployment_guide.md` |
| ECS Terraform | `cloud/terraform/ecs/` |
| Step Functions Terraform | `cloud/terraform/stepfunctions/` |
| Lambda Source | `cloud/src/lambdas/` |
| LLM Client | `cloud/src/llm/xai_client.py` |
| ADR-006 (LLM) | `docs/adr/adr_006_llm_strategy.md` |
| ADR-007 (Storage) | `docs/adr/adr_007_storage_strategy.md` |

---

## Sprint 6 Scope (After Testing Complete)

Once integration testing validates the pipeline:

1. **Multi-quarter validation** - Process 2 quarters (100 firms each)
2. **Theme aggregation** - Run reduce phase on firm topics
3. **Output quality review** - Validate themes are meaningful
4. **Cost analysis** - Actual vs estimated costs
5. **Documentation** - Runbook for operations team

---

## Definition of Done (Integration Testing Phase)

- [ ] Step Functions execution completes successfully
- [ ] Batch jobs connect to vLLM (not Grok API)
- [ ] Topic summaries are generated and stored in S3
- [ ] SummarizeResults correctly reports success/failure counts
- [ ] vLLM scaled down to save costs
- [ ] ADR-006 updated with deployed configuration

---

## Communication Guidelines

When reporting to the user:
- Lead with status (working/broken/unclear)
- Provide specific error messages and log snippets
- Suggest concrete next steps
- Don't over-explain the obvious
- **Always mention cost implications** (is vLLM running?)

---

## Checklist Before Starting

- [ ] Read `sprint5_instance_summary.md`
- [ ] Check vLLM status (running = costing money)
- [ ] Verify prefetch data exists for test quarter
- [ ] Understand host network mode rationale
- [ ] Have AWS CLI configured

**You're ready. Start with P0: End-to-End Step Functions Test.**
