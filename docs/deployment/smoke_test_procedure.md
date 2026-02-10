# Smoke Test Procedure

Sprint 8 smoke test for validating LLM integration (map phase) and reduce phase (cross-firm themes).

## Prerequisites

- AWS credentials configured (`--profile torrin`, account `015705018204`, region `us-west-2`)
- Terraform infrastructure deployed (batch → ecs → batch+vllm → stepfunctions)
- Docker installed locally
- Container image URIs available (from David or self-pushed if ECR push access granted)
- vLLM ECS service exists (can be scaled to 0)

---

## Phase 0: Pin AWS Profile

```bash
# CRITICAL: Pin to research account for entire session
export AWS_PROFILE=torrin
export AWS_REGION=us-west-2
export AWS_DEFAULT_REGION=us-west-2

# Verify identity (must show account 015705018204)
aws sts get-caller-identity
```

---

## Phase 1: Local Validation

```bash
# 1. Run unit tests
pytest tests/integration/test_batch_integration.py -v -m "not integration"

# 2. Validate Terraform
cd cloud/terraform/batch && terraform fmt && terraform validate
cd ../ecs && terraform fmt && terraform validate
cd ../stepfunctions && terraform fmt && terraform validate
cd ../../..
```

---

## Phase 2: Build Images (before Terraform apply)

```bash
# 3. Set image URIs FIRST — Terraform apply requires them
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Option A: David grants ECR push access
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

docker build -t ftm-map -f cloud/containers/map/Dockerfile .
docker tag ftm-map:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/ftm-map:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/ftm-map:latest

docker build -t ftm-reduce -f cloud/containers/reduce/Dockerfile .
docker tag ftm-reduce:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/ftm-reduce:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/ftm-reduce:latest

export MAP_IMAGE_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/ftm-map:latest"
export REDUCE_IMAGE_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/ftm-reduce:latest"

# Option B: David pre-pushes images — use the URIs he provides
# export MAP_IMAGE_URI="015705018204.dkr.ecr.us-west-2.amazonaws.com/ftm-map:latest"
# export REDUCE_IMAGE_URI="015705018204.dkr.ecr.us-west-2.amazonaws.com/ftm-reduce:latest"
```

---

## Phase 3: Infrastructure Deploy

Deploy order: **batch (no vllm) → ecs → batch (with vllm) → stepfunctions**.

Why two-pass batch? Circular dependency:
- ECS needs Batch's security group (`ftm-batch-sg`) → Batch must be deployed first
- Batch with `enable_vllm=true` reads ECS's SSM parameter (`/ftm/vllm/base_url`) → ECS must exist first

```bash
# 4. Apply Batch module FIRST pass (creates SG, compute env, job defs — WITHOUT vllm)
cd cloud/terraform/batch
terraform apply \
  -var="s3_bucket_name=ubc-torrin" \
  -var="enable_vllm=false" \
  -var="use_precreated_roles=true" \
  -var="use_external_images=true" \
  -var="map_image_uri=$MAP_IMAGE_URI" \
  -var="reduce_image_uri=$REDUCE_IMAGE_URI" \
  -var="enable_wrds_secrets=false"
cd ../../..

# 5. Apply ECS module (creates vLLM cluster, ALB, ASG, writes SSM parameter)
cd cloud/terraform/ecs
terraform apply \
  -var="use_precreated_roles=true"
cd ../../..

# 6. Re-apply Batch module SECOND pass (now enable_vllm=true — SSM parameter exists)
cd cloud/terraform/batch
terraform apply \
  -var="s3_bucket_name=ubc-torrin" \
  -var="enable_vllm=true" \
  -var="use_precreated_roles=true" \
  -var="use_external_images=true" \
  -var="map_image_uri=$MAP_IMAGE_URI" \
  -var="reduce_image_uri=$REDUCE_IMAGE_URI" \
  -var="enable_wrds_secrets=false"
cd ../../..

# 7. Apply Step Functions module (wires orchestration — needs batch job defs and queue)
cd cloud/terraform/stepfunctions
terraform apply \
  -var="s3_bucket_name=ubc-torrin" \
  -var="use_precreated_roles=true"
cd ../../..
```

---

## Phase 4: Run Smoke Test

```bash
# 8. Scale up vLLM ASG and ECS service (triggers on-demand base capacity)
aws autoscaling set-desired-capacity --auto-scaling-group-name ftm-vllm-asg --desired-capacity 1
aws ecs update-service --cluster ftm-vllm-cluster --service ftm-vllm-service --desired-count 1

# 9. Wait for vLLM to be healthy (~5 min for model load)
aws ecs wait services-stable --cluster ftm-vllm-cluster --services ftm-vllm-service

# 10. Get state machine ARN
export STATE_MACHINE_ARN=$(cd cloud/terraform/stepfunctions && terraform output -raw state_machine_arn)

# 11. Start pipeline execution
aws stepfunctions start-execution \
  --state-machine-arn $STATE_MACHINE_ARN \
  --input '{"quarter": "2023Q1", "bucket": "ubc-torrin"}' \
  --name "smoke-test-$(date +%Y%m%d-%H%M%S)"
```

---

## Phase 5: Monitor Execution

```bash
# 12. Get execution ARN from output above, then monitor
export EXECUTION_ARN="<paste-execution-arn-from-step-11>"

# Watch execution status (poll every 30s)
watch -n 30 "aws stepfunctions describe-execution --execution-arn $EXECUTION_ARN --query '{status:status,started:startDate}'"

# Or check Step Functions console:
# https://console.aws.amazon.com/states/home?region=us-west-2

# 13. Monitor vLLM logs for LLM calls
aws logs tail /ecs/ftm-vllm --follow --filter-pattern "POST /v1/chat/completions"
```

---

## Phase 6: Evaluate Outputs

```bash
# 14. After execution completes, check map phase output (topic summaries)
aws s3 cp s3://ubc-torrin/intermediate/firm-topics/quarter=2023Q1/ /tmp/map-output/ --recursive
python3 -c "
import pandas as pd
import glob
files = glob.glob('/tmp/map-output/*.parquet')
df = pd.concat([pd.read_parquet(f) for f in files])
print('=== MAP OUTPUT ===')
print(f'Rows: {len(df)}, Columns: {list(df.columns)}')
print(df[['firm_name', 'topic_id', 'representation', 'summary']].head(10))
print(f'Summary populated: {(df[\"summary\"] != \"\").sum()}/{len(df)}')
"

# 15. Check reduce phase output (themes + contributions)
aws s3 ls s3://ubc-torrin/processed/themes/quarter=2023Q1/

aws s3 cp s3://ubc-torrin/processed/themes/quarter=2023Q1/themes.parquet /tmp/
aws s3 cp s3://ubc-torrin/processed/themes/quarter=2023Q1/theme_contributions.parquet /tmp/

python3 -c "
import pandas as pd
themes = pd.read_parquet('/tmp/themes.parquet')
contribs = pd.read_parquet('/tmp/theme_contributions.parquet')
print('=== THEMES ===')
print(f'Count: {len(themes)}')
print(themes[['theme_id', 'name', 'description', 'n_firms', 'n_topics']].head())
print()
print('=== THEME CONTRIBUTIONS ===')
print(f'Count: {len(contribs)}')
print(f'Columns: {list(contribs.columns)}')
print(contribs[['theme_id', 'firm_name', 'permno', 'earnings_call_date']].head())
"
```

---

## Phase 7: Cleanup

```bash
# 16. Scale down vLLM ECS service and ASG (cost savings)
aws ecs update-service --cluster ftm-vllm-cluster --service ftm-vllm-service --desired-count 0
aws autoscaling set-desired-capacity --auto-scaling-group-name ftm-vllm-asg --desired-capacity 0
```

---

## Success Criteria Checklist

- [ ] Map output has `summary` field with natural language (not keywords)
- [ ] Map output has `naming_method` field = `"llm"` for all rows
- [ ] vLLM logs show `/v1/chat/completions` requests
- [ ] `themes.parquet` exists with `theme_id` format `theme_2023Q1_XXX`
- [ ] `theme_contributions.parquet` has columns: `permno`, `gvkey`, `earnings_call_date`
- [ ] Step Functions execution completes with status `SUCCEEDED`

### Verify LLM Naming Quality

```bash
# After map phase completes, verify all rows have valid LLM summaries
python3 -c "
import pandas as pd
import glob
files = glob.glob('/tmp/map-output/*.parquet')
df = pd.concat([pd.read_parquet(f) for f in files])

# Check 1: naming_method must be 'llm' for all rows
naming_methods = df['naming_method'].unique()
print(f'Naming methods found: {naming_methods}')
assert list(naming_methods) == ['llm'], 'ERROR: Found non-LLM naming!'

# Check 2: summary must be non-empty for all rows
empty_summaries = df[df['summary'].isna() | (df['summary'] == '')].shape[0]
print(f'Empty summaries: {empty_summaries}/{len(df)}')
assert empty_summaries == 0, f'ERROR: {empty_summaries} rows have empty summaries!'

# Check 3: summary should differ from representation (not keyword fallback)
keyword_matches = (df['summary'] == df['representation']).sum()
print(f'Summary equals representation: {keyword_matches}/{len(df)}')
if keyword_matches > 0:
    print('WARNING: Some summaries match keywords exactly (may indicate fallback)')

print('SUCCESS: All topics have valid LLM summaries')
"
```

---

## Troubleshooting

### vLLM not responding

```bash
# Check ECS task status
aws ecs list-tasks --cluster ftm-vllm-cluster --service-name ftm-vllm-service
aws ecs describe-tasks --cluster ftm-vllm-cluster --tasks <task-arn>

# Check vLLM logs
aws logs tail /ecs/ftm-vllm --since 10m
```

### Map phase fails

```bash
# Check Batch job logs
aws logs tail /aws/batch/ftm --since 30m --filter-pattern "ERROR"

# Check failures manifest
aws s3 ls s3://ubc-torrin/progress/2023Q1/
aws s3 cp s3://ubc-torrin/progress/2023Q1/<batch_id>_failures.json -
```

### Reduce phase fails

```bash
# Check reduce job logs
aws logs tail /aws/batch/ftm --since 30m --filter-pattern "theme-aggregator"
```

---

## Configuration Reference

| Variable              | Default         | Description                         |
| --------------------- | --------------- | ----------------------------------- |
| `LLM_MODEL_NAME`      | `Qwen/Qwen3-8B` | Model served by vLLM                |
| `LLM_MAX_CONCURRENT`  | `10`            | Max concurrent LLM requests per job |
| `CHECKPOINT_INTERVAL` | `50`            | Firms per checkpoint in map phase   |

---

---

## Operational Notes

### Scale-to-Zero Behavior

`on_demand_base_capacity=1` only applies when ASG desired capacity > 0. When scaled to zero between runs, no instances run (as expected for cost savings).

**Runbook** - before each pipeline run:
```bash
# Scale UP before run (triggers on-demand base)
aws autoscaling set-desired-capacity --auto-scaling-group-name ftm-vllm-asg --desired-capacity 1
aws ecs update-service --cluster ftm-vllm-cluster --service ftm-vllm-service --desired-count 1

# Wait for vLLM healthy (~5 min for model load)
aws ecs wait services-stable --cluster ftm-vllm-cluster --services ftm-vllm-service

# Run pipeline...
aws stepfunctions start-execution --state-machine-arn $STATE_MACHINE_ARN --input '{"quarter":"2023Q1"}'

# Scale DOWN after run (cost savings)
aws ecs update-service --cluster ftm-vllm-cluster --service ftm-vllm-service --desired-count 0
aws autoscaling set-desired-capacity --auto-scaling-group-name ftm-vllm-asg --desired-capacity 0
```

### Shared Recovery Wait Behavior

When vLLM goes down mid-map (e.g., 50 concurrent LLM calls in flight):
- All 50 calls share the SAME `XAIClient` instance (created once per job)
- First coroutine to detect failure acquires the shared lock, starts polling `/health`
- Other 49 coroutines block on the shared event (no wasted polling)
- When vLLM recovers, all coroutines resume together
- Total wait time = ~3-5 min (recovery), not 50 × 10 min

### Health-Aware Retry Verification

To verify health-aware retry is working during a spot interruption test:
```bash
# Watch for recovery messages in Batch logs
aws logs tail /aws/batch/ftm --follow --filter-pattern "vLLM"

# Expected patterns:
# - "vLLM unhealthy, waiting... (30s / 600s)"
# - "vLLM recovered after X.Xs"
```

---

_Last updated: Sprint 8 - Research Account Migration (2026-02-09)_
