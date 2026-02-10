# Sprint 5 Deployment Guide: vLLM + Step Functions

This guide documents the deployment of Sprint 5 infrastructure for the Financial Topic Modeling project.

## Overview

Sprint 5 adds:
- **vLLM on ECS**: Self-hosted LLM (Qwen3-8B) for topic naming, replacing Grok API
- **Step Functions**: Visual workflow orchestration for multi-quarter batch processing
- **Lambda helpers**: Prefetch check, batch manifest creation, result summarization, notifications

## Prerequisites

Before deploying Sprint 5, ensure:
1. AWS CLI configured with appropriate credentials
2. Terraform >= 1.0 installed
3. Main Terraform deployed (S3 bucket exists)
4. Batch module deployed (job queue and definition exist)

```bash
# Verify AWS access
aws sts get-caller-identity

# Get your S3 bucket name (from main terraform)
cd cloud/terraform
terraform output s3_bucket_name
# Example output: ubc-torrin

# Verify Batch infrastructure
aws batch describe-job-queues --job-queues ftm-queue-main
aws batch describe-job-definitions --job-definition-name ftm-firm-processor --status ACTIVE
```

## Deployment Order

**IMPORTANT**: Deploy in this order due to cross-module dependencies.

```
1. ECS Module (vLLM) → Creates SSM parameter
2. Batch Module Update → Reads SSM parameter, injects LLM_BASE_URL
3. Step Functions Module → References Batch job queue/definition
```

---

## Phase 1: Deploy ECS/vLLM Module

### 1.1 Initialize and Plan

```bash
cd cloud/terraform/ecs

terraform init
terraform plan -var="aws_region=us-west-2"
```

### 1.2 Review Plan

Expect ~22 resources:
- ECS cluster, service, task definition
- ALB (internal), target group, listener
- ASG with GPU launch template (g5.xlarge)
- Security groups (ALB, instance, task)
- IAM roles (execution, task, instance)
- SSM parameters (vLLM base URL)
- CloudWatch log group

### 1.3 Apply

```bash
terraform apply -var="aws_region=us-west-2"
```

### 1.4 Verify Deployment

```bash
# Check SSM parameter created
aws ssm get-parameter --name /ftm/vllm/base_url --query 'Parameter.Value' --output text

# Check ASG instance launched
aws autoscaling describe-auto-scaling-groups --auto-scaling-group-names ftm-vllm-asg \
  --query 'AutoScalingGroups[0].Instances'

# Check ECS service status
aws ecs describe-services --cluster ftm-vllm-cluster --services ftm-vllm-service \
  --query 'services[0].{desired:desiredCount,running:runningCount,pending:pendingCount}'

# Watch logs for model loading (takes 5-10 minutes)
aws logs tail /ecs/ftm-vllm --since 10m --follow
```

### 1.5 Verify vLLM Health

```bash
# Check target health
TG_ARN=$(aws elbv2 describe-target-groups --names ftm-vllm-tg2 --query 'TargetGroups[0].TargetGroupArn' --output text)
aws elbv2 describe-target-health --target-group-arn $TG_ARN
```

**Expected log output when ready:**
```
vLLM 0.14.1 | Qwen/Qwen3-8B | max_model_len: 4096
...
Server: Running on http://0.0.0.0:8000
Health: 200 OK (heartbeat active)
```

---

## Phase 2: Update Batch Module for vLLM

### 2.1 Plan and Apply

```bash
cd cloud/terraform/batch

# Replace with your actual bucket name
terraform plan -var="s3_bucket_name=ubc-torrin" -var="enable_vllm=true"
terraform apply -var="s3_bucket_name=ubc-torrin" -var="enable_vllm=true"
```

### 2.2 Verify Job Definition Updated

```bash
aws batch describe-job-definitions --job-definition-name ftm-firm-processor --status ACTIVE \
  --query 'jobDefinitions[0].containerProperties.environment[?name==`LLM_BASE_URL`]'
```

Should show the vLLM ALB URL.

---

## Phase 3: Deploy Step Functions Module

### 3.1 Initialize and Plan

```bash
cd cloud/terraform/stepfunctions

terraform init
terraform plan \
  -var="s3_bucket_name=ubc-torrin" \
  -var="job_queue_name=ftm-queue-main" \
  -var="job_definition_name=ftm-firm-processor"
```

### 3.2 Review Plan

Expect ~18 resources:
- Step Functions state machine
- Lambda functions (4): prefetch_check, create_batch_manifest, summarize_results, notify_completion
- IAM roles (Lambda execution, Step Functions execution)
- CloudWatch log groups

### 3.3 Apply

```bash
terraform apply \
  -var="s3_bucket_name=ubc-torrin" \
  -var="job_queue_name=ftm-queue-main" \
  -var="job_definition_name=ftm-firm-processor"
```

### 3.4 Capture Outputs

```bash
terraform output
```

Save the `state_machine_arn` and `console_url` for later use.

---

## Phase 4: Integration Testing

### 4.1 Test Lambda Functions Individually

```bash
# Test prefetch check (should return exists: true/false)
aws lambda invoke --function-name ftm-prefetch-check \
  --payload '{"quarter": "2023Q1", "bucket": "ubc-torrin"}' \
  --cli-binary-format raw-in-base64-out \
  /dev/stdout

# Test batch manifest creation (only if prefetch exists)
aws lambda invoke --function-name ftm-create-batch-manifest \
  --payload '{"quarter": "2023Q1", "bucket": "ubc-torrin", "batch_size": 100}' \
  --cli-binary-format raw-in-base64-out \
  /dev/stdout
```

### 4.2 Test Step Functions Execution

```bash
# Start execution with a quarter that has prefetch data
STATE_MACHINE_ARN="arn:aws:states:us-west-2:015705018204:stateMachine:ftm-quarter-processor"

aws stepfunctions start-execution \
  --state-machine-arn $STATE_MACHINE_ARN \
  --name "test-$(date +%Y%m%d-%H%M%S)" \
  --input '{"quarter": "2023Q1", "bucket": "ubc-torrin", "batch_size": 100}'
```

### 4.3 Monitor Execution

```bash
# List recent executions
aws stepfunctions list-executions \
  --state-machine-arn $STATE_MACHINE_ARN \
  --max-results 5

# Or use the console URL from terraform output
```

---

## Operational Commands

### Scale vLLM Up (Before Processing)

```bash
aws ecs update-service --cluster ftm-vllm-cluster --service ftm-vllm-service --desired-count 1
aws autoscaling set-desired-capacity --auto-scaling-group-name ftm-vllm-asg --desired-capacity 1
```

### Scale vLLM Down (After Processing)

```bash
aws ecs update-service --cluster ftm-vllm-cluster --service ftm-vllm-service --desired-count 0
aws autoscaling set-desired-capacity --auto-scaling-group-name ftm-vllm-asg --desired-capacity 0
```

### Check vLLM Status

```bash
# Service status
aws ecs describe-services --cluster ftm-vllm-cluster --services ftm-vllm-service \
  --query 'services[0].{desired:desiredCount,running:runningCount}'

# Instance status
aws autoscaling describe-auto-scaling-groups --auto-scaling-group-names ftm-vllm-asg \
  --query 'AutoScalingGroups[0].{desired:DesiredCapacity,instances:Instances[*].InstanceId}'

# Recent logs
aws logs tail /ecs/ftm-vllm --since 30m
```

---

## Cost Management

| Resource | Hourly (Spot) | Daily | Monthly |
|----------|---------------|-------|---------|
| g5.xlarge (vLLM) | ~$0.35 | ~$8.40 | ~$252 |
| ALB | ~$0.02 | ~$0.50 | ~$15 |
| **Scaled Up** | ~$0.37 | ~$8.90 | ~$267 |
| **Scaled Down** | ~$0.02 | ~$0.50 | ~$15 |

**IMPORTANT**: Always scale down vLLM when not in use!

---

## Troubleshooting

### vLLM Can't Download Model

**Symptom**: Logs show `Connection to huggingface.co timed out`

**Cause**: Task doesn't have internet access

**Fix**: Ensure task definition uses `host` network mode (not `awsvpc`)

### Target Group Unhealthy

**Symptom**: `aws elbv2 describe-target-health` shows `unhealthy`

**Possible causes**:
1. Model still loading (wait 5-10 minutes)
2. Container crashed (check CloudWatch logs)
3. Security group blocking traffic

### Step Functions Fails at PrefetchRequired

**Symptom**: Execution fails with `PrefetchRequiredError`

**Cause**: No prefetch data exists for the quarter

**Fix**: Run prefetch first:
```bash
python -c "
from cloud.src.orchestrate.quarter_orchestrator import QuarterOrchestrator
o = QuarterOrchestrator('ubc-torrin', 'ftm-firm-processor', 'ftm-queue-main')
o.run_prefetch('2023Q1')
"
```

### Batch Job Fails

**Symptom**: Step Functions shows job failed

**Debug**:
```bash
# Check Batch job logs
aws logs tail /aws/batch/ftm --since 1h

# Check job status
aws batch describe-jobs --jobs <job-id>
```

---

## Cleanup / Teardown

To completely remove Sprint 5 infrastructure:

```bash
# 1. Scale down vLLM first
aws ecs update-service --cluster ftm-vllm-cluster --service ftm-vllm-service --desired-count 0
aws autoscaling set-desired-capacity --auto-scaling-group-name ftm-vllm-asg --desired-capacity 0

# Wait for instances to terminate
sleep 60

# 2. Destroy Step Functions
cd cloud/terraform/stepfunctions
terraform destroy \
  -var="s3_bucket_name=ubc-torrin" \
  -var="job_queue_name=ftm-queue-main" \
  -var="job_definition_name=ftm-firm-processor"

# 3. Revert Batch module (disable vLLM)
cd cloud/terraform/batch
terraform apply -var="s3_bucket_name=ubc-torrin" -var="enable_vllm=false"

# 4. Destroy ECS
cd cloud/terraform/ecs
terraform destroy -var="aws_region=us-west-2"
```

---

## Configuration Reference

### ECS Module Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `aws_region` | us-west-2 | AWS region |
| `vllm_model` | Qwen/Qwen3-8B | HuggingFace model ID |
| `instance_type` | g5.xlarge | GPU instance type |
| `min_capacity` | 1 | Minimum ASG capacity |
| `max_capacity` | 4 | Maximum ASG capacity |
| `container_memory` | 14000 | Container memory (MB) |

### Step Functions Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `s3_bucket_name` | (required) | S3 bucket for manifests |
| `job_queue_name` | ftm-queue-main | Batch job queue |
| `job_definition_name` | ftm-firm-processor | Batch job definition |
| `batch_size` | 1000 | Firms per batch job |
| `max_concurrency` | 5 | Parallel batch jobs |

---

## Architecture Summary

```
Step Functions (ftm-quarter-processor)
    │
    ├── Lambda: prefetch_check
    │       └── S3: prefetch/transcripts/quarter={Q}/manifest.json
    │
    ├── Lambda: create_batch_manifest
    │       └── S3: manifests/{Q}/manifest_*.jsonl
    │
    ├── Map State (5 concurrent)
    │       └── Batch: ftm-firm-processor
    │               ├── S3: Read prefetch data
    │               ├── vLLM: Topic naming (via ALB)
    │               └── S3: Write intermediate/firm-topics/
    │
    ├── Lambda: summarize_results
    │
    └── Lambda: notify_completion
            └── SNS: (optional)
```
