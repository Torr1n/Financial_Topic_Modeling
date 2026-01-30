# Sprint 5 Instance Summary: vLLM + Step Functions

**Session Date**: 2026-01-28
**Context**: Implementation of Sprint 5 plan for self-hosted LLM and workflow orchestration
**Status**: Infrastructure deployed, awaiting end-to-end integration testing

---

## Executive Summary

This session implemented the Sprint 5 plan: deploying vLLM on ECS for topic naming and Step Functions for multi-quarter orchestration. All infrastructure is deployed and operational. The vLLM service was healthy at the end of the session (verify current status). Step Functions state machine is ready for execution. **What remains is end-to-end integration testing with real prefetch data.**

---

## The "What": Work Completed

### Phase 1: ECS/vLLM Infrastructure

**Files Created** (`cloud/terraform/ecs/`):
- `main.tf` - Provider, data sources (default VPC, subnets)
- `variables.tf` - vllm_model, instance_type, scaling parameters
- `iam.tf` - Execution role + task role (2 roles)
- `ecs.tf` - Cluster, task definition, service
- `alb.tf` - Internal ALB, target group, security groups
- `autoscaling.tf` - Launch template, ASG, capacity provider
- `ssm.tf` - Writes ALB DNS to SSM for Batch module consumption
- `outputs.tf` - Scale commands, ARNs, URLs

**Deployed Configuration**:
- Model: `Qwen/Qwen3-8B` via `vllm/vllm-openai:latest`
- Instance: g5.xlarge (24GB A10G GPU), 100% Spot
- Network: **Host mode** (not awsvpc - critical learning)
- Target group: Type `instance` (not `ip`)
- Memory: 14GB hard limit, 12GB soft reservation
- ALB: Internal only, health check on `/health`

### Phase 1.2: xai_client.py Modification

**File Modified**: `cloud/src/llm/xai_client.py`

Added `LLM_BASE_URL` environment variable support with priority chain:
```
config["base_url"] > os.environ["LLM_BASE_URL"] > DEFAULT_BASE_URL (xAI)
```

This is **backward compatible** - without `LLM_BASE_URL` set, defaults to Grok API.

### Phase 1.3-1.5: Tests and Batch Integration

- `tests/unit/test_xai_client.py` - Added `TestClientBaseURL` class (5 tests)
- `tests/integration/test_vllm_integration.py` - New file with `@pytest.mark.vllm` marker
- `cloud/terraform/batch/` - Added `enable_vllm` variable, SSM data source, conditional LLM_BASE_URL injection

### Phase 2: Lambda Functions

**Files Created** (`cloud/src/lambdas/`):
- `__init__.py` - Package exports
- `prefetch_check.py` - Verify S3 manifest exists (O(1) HEAD request)
- `create_batch_manifest.py` - Create JSONL, return batch_ids only (not firm_ids - payload limit)
- `summarize_results.py` - Count succeeded/failed from Map output (added after Codex review)
- `notify_completion.py` - SNS notification (optional)

### Phase 2.2: Step Functions Terraform

**Files Created** (`cloud/terraform/stepfunctions/`):
- `main.tf`, `variables.tf`, `iam.tf`, `lambdas.tf`
- `state_machine.json` - ASL definition with 10 states
- `state_machine.tf` - Resource with templatefile injection
- `outputs.tf` - ARNs, console URL, CLI commands
- `builds/.gitkeep` - For Lambda ZIP artifacts

### Documentation

- `docs/deployment/sprint5_deployment_guide.md` - Complete deployment runbook
- `docs/diagrams/stepfunctions_state_machine.puml` - Technical diagram
- `docs/diagrams/stepfunctions_executive_view.puml` - Stakeholder diagram
- `docs/diagrams/stepfunctions_architecture.puml` - AWS service integration

---

## The "Why": Reasoning Behind Decisions

### Why vLLM Instead of Grok API?

The original pipeline used Grok API for topic naming. This creates two problems:
1. **Rate limits** - At scale (thousands of topics), API throttling becomes a bottleneck
2. **Cost** - Estimated ~$80/quarter with Grok vs ~$2/quarter with self-hosted

vLLM provides an **OpenAI-compatible API**, meaning `xai_client.py` works unchanged - we just point it at a different URL. This is why the `LLM_BASE_URL` abstraction was critical.

### Why ECS on EC2 (Not Fargate or SageMaker)?

| Option | GPU Support | Spot | Cold Start | Complexity |
|--------|-------------|------|------------|------------|
| Fargate | Limited/expensive | No | Medium | Low |
| SageMaker | Yes | Limited | High (5-10 min) | High |
| **ECS on EC2** | **Yes** | **Yes** | **Low** | **Medium** |

ADR-006 specified ECS for better Spot support and lower cold start. SageMaker endpoints have significant spin-up time that would delay batch processing.

### Why Host Network Mode (Critical Learning)?

**Original assumption**: Use `awsvpc` network mode (each task gets own ENI).

**Problem discovered during deployment**: Tasks couldn't reach HuggingFace to download the model. Error: `Connection to huggingface.co timed out`.

**Root cause**: In `awsvpc` mode with default VPC (no NAT Gateway), task ENIs don't get public IPs. `assign_public_ip = true` only works for Fargate, not EC2.

**Solution**: Switch to `host` network mode. The container shares the EC2 instance's network stack directly, including its public IP.

**Trade-off**: One vLLM task per instance (fixed port 8000). Acceptable for our use case - we scale by adding instances, not tasks per instance.

### Why Manual Scaling (Not Auto-Scaling)?

Codex review identified that auto-scaling adds complexity without benefit for batch workloads:

1. **Predictability > Elasticity** - We want vLLM warm and stable during runs, not scaling up/down mid-batch
2. **Avoid cold starts** - Auto-scaling could scale down between batches, causing model reload delays
3. **Simpler operations** - Explicit `scale_up` before runs, `scale_down` after

ADR-006 explicitly calls for "keep warm" strategy during runs. Manual scaling enforces this.

### Why batch_ids Only in Map State?

**Critical constraint**: Step Functions has a 256KB payload limit per state.

If we passed full `firm_ids` arrays (potentially thousands of IDs), we'd overflow this limit. Instead:
- Lambda creates JSONL manifest in S3 with full firm_ids
- Map state receives only batch_ids (strings like `"2023Q1_batch_0000"`)
- Each Batch job reads its firm_ids from S3 manifest at runtime

### Why SSM for Cross-Module Config?

ECS and Batch are **separate Terraform roots** (different state files). They can't reference each other's outputs directly.

SSM Parameter Store provides a decoupling layer:
- ECS module **writes** `/ftm/vllm/base_url` with ALB DNS
- Batch module **reads** this parameter and injects into job definition

This stays correct across redeploys without manual copy-paste.

### Why summarize_results Lambda Was Added

Original state machine had a bug: it always reported success (`failed: 0`) even when jobs failed. The Map state's `JobFailed` catch clause captured errors but didn't aggregate them.

The new `summarize_results` Lambda:
1. Iterates Map output
2. Counts actual succeeded vs failed
3. Returns `has_failures` boolean
4. State machine now fails if `has_failures = true`

---

## The "How": Key Implementation Details

### State Machine Flow

```
CheckPrefetch (Lambda)
    ↓
PrefetchExists? (Choice)
    ├─ No → PrefetchRequired (Fail)
    └─ Yes ↓
CreateBatchManifest (Lambda)
    ↓
ProcessBatches (Map, max 5 concurrent)
    ├─ SubmitBatchJob (batch:submitJob.sync)
    └─ JobFailed (Pass - captures error)
    ↓
SummarizeResults (Lambda)
    ↓
NotifyCompletion (Lambda)
    ↓
CheckForFailures (Choice)
    ├─ has_failures=true → QuarterCompletedWithFailures (Fail)
    └─ has_failures=false → QuarterCompletedSuccessfully (Succeed)
```

### Security Group Chain

```
Batch instances (ftm-batch-sg)
    → ALB (ftm-vllm-alb-sg) - ingress from Batch SG on port 80
        → EC2 instance (ftm-vllm-instance-sg) - ingress from ALB SG on port 8000
```

Note: `ftm-vllm-task-sg` is currently **unused** under host network mode but kept for potential future switch to awsvpc.

### Lambda → S3 Flow

```
prefetch_check:
    HEAD s3://{bucket}/prefetch/transcripts/quarter={Q}/manifest.json

create_batch_manifest:
    GET  s3://{bucket}/prefetch/transcripts/quarter={Q}/manifest.json (firm_ids)
    PUT  s3://{bucket}/manifests/{Q}/manifest_{timestamp}.jsonl
```

---

## Confidence Assessment

### High Confidence

| Item | Reason |
|------|--------|
| ECS/vLLM is running | Logs show `Server: Running on http://0.0.0.0:8000`, target healthy |
| State machine structure | Follows ASL spec, validated by Terraform |
| Lambda functions | Unit tests pass, logic is straightforward |
| xai_client.py backward compatibility | Tested - defaults to xAI without env var |
| SSM parameter approach | Clean decoupling, verified parameter exists |
| Manual scaling | Simple, predictable, matches ADR-006 |

### Medium Confidence

| Item | Concern |
|------|---------|
| Network connectivity Batch → ALB | Security groups look correct, but not tested from actual Batch job |
| Memory sizing (14GB) | Should work for Qwen3-8B, but may need tuning under load |
| Host network mode stability | Works, but less isolation than awsvpc |

### Low Confidence (Needs Testing)

| Item | Why |
|------|-----|
| End-to-end execution | Haven't run Step Functions with real prefetch data |
| vLLM output quality | No comparison of topic names vs Grok API |
| Throughput under load | Haven't tested 5 concurrent batch jobs hitting vLLM |
| Cost estimates | Based on documentation, not actual usage |

---

## What Wasn't Completed

### Deferred to Next Session

1. **End-to-end Step Functions execution** - Requires prefetch data for a quarter
2. **Quality validation** - Compare vLLM topic names vs Grok baseline
3. **Throughput testing** - Run with 5 concurrent batches
4. **ADR-006 verification** - Confirm docs match deployed config (host mode, target_type=instance)

### Intentionally Left as Optional

1. **SNS topic** - `sns_topic_arn` variable exists, not configured
2. **HTTPS on ALB** - Internal traffic, HTTP acceptable
3. **vllm_task security group cleanup** - Kept for future awsvpc switch

### Known Technical Debt

1. **vLLM image uses `latest` tag** - Should pin version for production
2. **No CloudWatch alarms** - Should add for target health, ASG failures
3. **No cost alerts** - Should add budget alerts for GPU spend

---

## Deployment Issues Encountered & Resolutions

| Issue | Error | Resolution |
|-------|-------|------------|
| AWS provider 5.x syntax | `deployment_configuration` block not supported | Changed to `deployment_maximum_percent` (top-level) |
| HuggingFace unreachable | `Connection to huggingface.co timed out` | Changed from `awsvpc` to `host` network mode |
| assign_public_ip invalid | `not supported for this launch type` | Removed (only valid for Fargate) |
| Target group type change | `ResourceInUse` when deleting | Renamed to `ftm-vllm-tg2` with `create_before_destroy` |
| Container memory missing | `At least one of 'memory' or 'memoryReservation' must be specified` | Added `memory` and `memoryReservation` in container definition |

---

## Current AWS Resource State

### ECS Module

| Resource | Name/ARN | Status |
|----------|----------|--------|
| Cluster | ftm-vllm-cluster | Active |
| Service | ftm-vllm-service | Running (1/1) |
| Task Definition | ftm-vllm:N | Active |
| ALB | ftm-vllm-alb | Active |
| Target Group | ftm-vllm-tg2 | Healthy |
| ASG | ftm-vllm-asg | 1 instance running |
| SSM Parameter | /ftm/vllm/base_url | Populated |

### Step Functions Module

| Resource | Name/ARN | Status |
|----------|----------|--------|
| State Machine | ftm-quarter-processor | Active |
| Lambda | ftm-prefetch-check | Active |
| Lambda | ftm-create-batch-manifest | Active |
| Lambda | ftm-summarize-results | Active |
| Lambda | ftm-notify-completion | Active |

### Batch Module

| Resource | Change | Status |
|----------|--------|--------|
| Job Definition | LLM_BASE_URL added | Active (new revision) |

---

## Files Changed Summary

### New Files (32)

```
cloud/terraform/ecs/ (8 files)
cloud/terraform/stepfunctions/ (8 files including builds/.gitkeep)
cloud/src/lambdas/ (5 files)
tests/unit/test_lambdas.py
tests/integration/test_vllm_integration.py
docs/deployment/sprint5_deployment_guide.md
docs/diagrams/stepfunctions_*.puml (3 files)
```

### Modified Files (7)

```
cloud/src/llm/xai_client.py - LLM_BASE_URL support
cloud/terraform/batch/main.tf - SSM data source
cloud/terraform/batch/variables.tf - enable_vllm variable
cloud/terraform/batch/job_definition.tf - Conditional LLM_BASE_URL
cloud/terraform/batch/iam.tf - SSM read permission
tests/unit/test_xai_client.py - TestClientBaseURL class
```

---

## Handoff Checklist for Next Instance

- [ ] Run end-to-end Step Functions execution with prefetch data
- [ ] Validate Batch jobs hit vLLM (check logs for LLM_BASE_URL)
- [ ] Compare topic naming quality vs Grok baseline
- [ ] Run multi-batch throughput test (5 concurrent)
- [ ] Verify ADR-006 reflects deployed configuration details
- [ ] Consider pinning vLLM image version
- [ ] Scale down vLLM when testing complete (cost!)

---

## Key Commands for Next Instance

```bash
# Check vLLM status
aws ecs describe-services --cluster ftm-vllm-cluster --services ftm-vllm-service \
  --query 'services[0].{desired:desiredCount,running:runningCount}'

# Check vLLM logs
aws logs tail /ecs/ftm-vllm --since 30m

# Start Step Functions execution
aws stepfunctions start-execution \
  --state-machine-arn arn:aws:states:us-east-1:666938415731:stateMachine:ftm-quarter-processor \
  --name "test-$(date +%Y%m%d-%H%M%S)" \
  --input '{"quarter": "2023Q1", "bucket": "ftm-pipeline-78ea68c8", "batch_size": 100}'

# Scale down vLLM (IMPORTANT - saves ~$9/day)
aws ecs update-service --cluster ftm-vllm-cluster --service ftm-vllm-service --desired-count 0
aws autoscaling set-desired-capacity --auto-scaling-group-name ftm-vllm-asg --desired-capacity 0
```
