# Smoke Test Procedure

Sprint 7 smoke test for validating LLM integration (map phase) and reduce phase (cross-firm themes).

## Prerequisites

- AWS credentials configured
- Terraform infrastructure deployed (batch, stepfunctions, ecs modules)
- Docker installed locally
- vLLM ECS service exists (can be scaled to 0)

---

## Phase 1: Local Validation

```bash
# 1. Run unit tests
pytest tests/integration/test_batch_integration.py -v -k "not integration"

# 2. Validate Terraform
cd cloud/terraform/batch && terraform fmt && terraform validate
cd ../stepfunctions && terraform fmt && terraform validate
cd ../../..
```

---

## Phase 2: Infrastructure Updates

```bash
# 3. Get AWS account ID and region
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export AWS_REGION=us-east-1

# 4. Apply Terraform (batch module first - creates reduce ECR + job def)
cd cloud/terraform/batch
terraform apply -var="enable_vllm=true"
cd ../../..

# 5. Build and push MAP container (has LLM integration changes)
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

docker build -t ftm-map -f cloud/containers/map/Dockerfile .
docker tag ftm-map:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/ftm-map:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/ftm-map:latest

# 6. Build and push REDUCE container (new)
docker build -t ftm-reduce -f cloud/containers/reduce/Dockerfile .
docker tag ftm-reduce:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/ftm-reduce:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/ftm-reduce:latest

# 7. Apply Step Functions changes (wires reduce phase)
cd cloud/terraform/stepfunctions
terraform apply
cd ../../..
```

---

## Phase 3: Run Smoke Test

```bash
# 8. Scale up vLLM (wait ~3-5 min for model load)
aws ecs update-service --cluster ftm-vllm-cluster --service ftm-vllm-service --desired-count 1

# 9. Wait for vLLM to be healthy
aws ecs wait services-stable --cluster ftm-vllm-cluster --services ftm-vllm-service

# 10. Get state machine ARN
export STATE_MACHINE_ARN=$(cd cloud/terraform/stepfunctions && terraform output -raw state_machine_arn)

# 11. Start pipeline execution
aws stepfunctions start-execution \
  --state-machine-arn $STATE_MACHINE_ARN \
  --input '{"quarter": "2023Q1", "bucket": "ftm-pipeline-78ea68c8"}' \
  --name "smoke-test-$(date +%Y%m%d-%H%M%S)"
```

---

## Phase 4: Monitor Execution

```bash
# 12. Get execution ARN from output above, then monitor
export EXECUTION_ARN="<paste-execution-arn-from-step-11>"

# Watch execution status (poll every 30s)
watch -n 30 "aws stepfunctions describe-execution --execution-arn $EXECUTION_ARN --query '{status:status,started:startDate}'"

# Or check Step Functions console:
# https://console.aws.amazon.com/states/home?region=us-east-1

# 13. Monitor vLLM logs for LLM calls
aws logs tail /ecs/ftm-vllm --follow --filter-pattern "POST /v1/chat/completions"
```

---

## Phase 5: Evaluate Outputs

```bash
# 14. After execution completes, check map phase output (topic summaries)
aws s3 cp s3://ftm-pipeline-78ea68c8/intermediate/firm-topics/quarter=2023Q1/ /tmp/map-output/ --recursive
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
aws s3 ls s3://ftm-pipeline-78ea68c8/processed/themes/quarter=2023Q1/

aws s3 cp s3://ftm-pipeline-78ea68c8/processed/themes/quarter=2023Q1/themes.parquet /tmp/
aws s3 cp s3://ftm-pipeline-78ea68c8/processed/themes/quarter=2023Q1/theme_contributions.parquet /tmp/

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

## Phase 6: Cleanup

```bash
# 16. Scale down vLLM
aws ecs update-service --cluster ftm-vllm-cluster --service ftm-vllm-service --desired-count 0
```

---

## Success Criteria Checklist

- [ ] Map output has `summary` field with natural language (not keywords)
- [ ] vLLM logs show `/v1/chat/completions` requests
- [ ] `themes.parquet` exists with `theme_id` format `theme_2023Q1_XXX`
- [ ] `theme_contributions.parquet` has columns: `permno`, `gvkey`, `earnings_call_date`
- [ ] Step Functions execution completes with status `SUCCEEDED`

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
aws s3 ls s3://ftm-pipeline-78ea68c8/progress/2023Q1/
aws s3 cp s3://ftm-pipeline-78ea68c8/progress/2023Q1/<batch_id>_failures.json -
```

### Reduce phase fails
```bash
# Check reduce job logs
aws logs tail /aws/batch/ftm --since 30m --filter-pattern "theme-aggregator"
```

---

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL_NAME` | `Qwen/Qwen3-8B` | Model served by vLLM |
| `LLM_MAX_CONCURRENT` | `10` | Max concurrent LLM requests per job |
| `CHECKPOINT_INTERVAL` | `50` | Firms per checkpoint in map phase |

---

*Last updated: Sprint 7 (2026-01-30)*
