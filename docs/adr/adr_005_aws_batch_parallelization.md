# ADR-005: AWS Batch Parallelization

## Status
Accepted

## Date
2026-01-26

## Context

The current Financial Topic Modeling pipeline processes firms sequentially on a single `g4dn.2xlarge` GPU instance. This approach:
- Takes ~48 hours per quarter for ~5000 firms
- Requires manual babysitting during Spot instance interruptions
- Cannot meet the deadline of processing 8 quarters by end of week

### Processing Time Breakdown (Sequential)
| Phase | Time per Firm | Bottleneck |
|-------|---------------|------------|
| Transcript preprocessing (SpaCy) | ~2s | CPU |
| Embedding generation | ~3s | GPU |
| BERTopic clustering | ~5s | CPU/GPU |
| LLM topic naming (Grok API) | ~5s | API rate limit |
| **Total** | **~15s** | Mixed |

At 15 seconds per firm × 5000 firms = 75,000 seconds = **20.8 hours** pure processing. With overhead, checkpointing, and Spot interruptions, this becomes ~48 hours.

### Why Sequential Was Correct for MVP
The prior architecture decision (see `docs/ai-log/claude-2026-batch-convo.md`) validated that sequential processing was appropriate because:
1. Grok API rate limit of 500 req/min was nearly saturated at ~300-400 req/min sequential
2. Even 2x parallelism would have exceeded API limits
3. Single GPU instance kept embedding model in memory (load once, use many times)

### What Changed
1. We now have self-hosted vLLM (no API rate limits)
2. We need 3-5x speedup to meet timeline
3. AWS Batch provides managed Spot instance handling

## Decision

**Implement distributed firm processing using AWS Batch with Spot instances, targeting 3-5x parallelism per quarter.**

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     AWS Batch Architecture                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐                                           │
│  │   Job Submitter  │  (Python script or Lambda)                │
│  │   - Partitions   │                                           │
│  │     5000 firms   │                                           │
│  │     into batches │                                           │
│  └────────┬─────────┘                                           │
│           │                                                      │
│           ▼                                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │         AWS Batch Job Queue (Spot Priority)               │   │
│  │                                                           │   │
│  │   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐        │   │
│  │   │ Job 0   │ │ Job 1   │ │ Job 2   │ │ Job 3   │ ...    │   │
│  │   │ 1000    │ │ 1000    │ │ 1000    │ │ 1000    │        │   │
│  │   │ firms   │ │ firms   │ │ firms   │ │ firms   │        │   │
│  │   └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘        │   │
│  └────────│──────────│──────────│──────────│─────────────────┘   │
│           │          │          │          │                     │
│           ▼          ▼          ▼          ▼                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │   Compute Environment (Spot g4dn.xlarge / g5.xlarge)     │   │
│  │                                                           │   │
│  │   Each job:                                               │   │
│  │   1. Loads embedding model (once)                         │   │
│  │   2. Fetches transcripts from WRDS                        │   │
│  │   3. Processes 1000 firms sequentially (reusing GPU)     │   │
│  │   4. Calls shared vLLM service for topic naming          │   │
│  │   5. Writes Parquet to S3 every ~50 firms                │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

#### 1. Batch Size: ~1000 firms per job
- **Why not 1 firm per job?** Container cold start (~60s) dominates actual work (~15s). Also, embedding model reload per job is wasteful.
- **Why not 5000 firms per job?** No parallelism benefit; still sequential.
- **1000 firms = ~4 hours per job** - reasonable for Spot interruption recovery

#### 2. Compute Environment: Spot with Fallback
```hcl
compute_resources {
  type                = "SPOT"
  allocation_strategy = "SPOT_PRICE_CAPACITY_OPTIMIZED"
  instance_type       = ["g4dn.xlarge", "g5.xlarge", "g4dn.2xlarge"]
  max_vcpus           = 64
  min_vcpus           = 0
}
```
- Multiple instance types for better Spot availability
- `SPOT_PRICE_CAPACITY_OPTIMIZED` balances cost and availability
- Falls back to larger instances if small ones unavailable

#### 3. Retry Strategy: Handle Spot Interruptions
```hcl
retry_strategy {
  attempts = 3
  evaluate_on_exit {
    action           = "RETRY"
    on_status_reason = "Host EC2*"  # Spot termination
  }
  evaluate_on_exit {
    action    = "EXIT"
    on_exit_code = "1"  # Application error - don't retry
  }
}
```

#### 4. Checkpoint Strategy: Per-Firm Progress
Each job writes Parquet chunks every ~50 firms. If interrupted:
1. Job restarts from beginning (AWS Batch retry)
2. Reads S3 to find last completed chunk
3. Resumes from next firm

This is simpler than database-based checkpointing and works with S3's eventual consistency.

## Consequences

### Positive
- **3-5x speedup**: 5 parallel jobs = ~10 hours instead of ~48 hours
- **Managed Spot handling**: AWS Batch handles interruptions automatically
- **Cost efficiency**: Spot instances are 60-70% cheaper than On-Demand
- **Scalability**: Can increase parallelism if needed

### Negative
- **Increased complexity**: More moving parts than single instance
- **Debugging harder**: Logs spread across multiple jobs
- **Cold start overhead**: Each job pays ~60s container startup
- **Coordination needed**: Jobs must write to distinct S3 paths

### Cost Analysis (5000 firms, 5 parallel jobs)
| Component | Sequential | Batch (5x) |
|-----------|------------|------------|
| GPU hours | 20 hrs × $0.50 = $10 | 4 hrs × 5 × $0.16 = $3.20 |
| Total time | ~48 hours | ~10 hours |
| **Savings** | - | **68% cost, 80% time** |

## Alternatives Considered

### 1. AWS Step Functions Map State (Distributed)
- **Pro**: Better for truly parallel embarrassingly-parallel workloads
- **Con**: 10,000 concurrent execution limit; better for smaller jobs
- **Decision**: Use Step Functions for orchestration, Batch for compute

### 2. AWS Lambda (Containerized)
- **Pro**: True pay-per-use, no cold start management
- **Con**: 15-minute timeout, 10GB memory limit, no GPU support
- **Decision**: Not suitable for GPU workloads

### 3. EKS with Karpenter
- **Pro**: More control, potentially better GPU scheduling
- **Con**: Significant operational complexity, overkill for batch processing
- **Decision**: AWS Batch is purpose-built for this use case

### 4. SageMaker Processing Jobs
- **Pro**: Managed, integrates with ML workflows
- **Con**: Less control over instance types, pricing less transparent
- **Decision**: Batch is simpler for this specific use case

## Implementation Notes

### Implementation Updates (2026-01-28)

The implementation landed in `cloud/terraform/batch/` and `cloud/containers/map/entrypoint.py`. The following updates reflect the current, validated behavior:

- **GPU AMI pinned**: Batch compute environment explicitly uses `ECS_AL2_NVIDIA` to ensure NVIDIA drivers are present.
- **Secrets Manager access**: Both execution **and job roles** have `secretsmanager:GetSecretValue` (plus `kms:Decrypt`) to support WRDS credential fallback.
- **Failure tolerance + circuit breaker**:
  - `ALLOW_FAILURES=true` by default (production tolerance for rare per‑firm failures).
  - Circuit breaker trips on critical errors, excessive consecutive failures, or high failure rate (env‑configurable).
- **Job definition env vars**: Job‑specific vars are passed at submission (manifest key, batch ID, quarter). Static vars include `S3_BUCKET` and `CHECKPOINT_INTERVAL`.

These changes were validated end‑to‑end with real WRDS data and Batch integration tests.

### Job Definition
```hcl
resource "aws_batch_job_definition" "firm_processor" {
  name = "ftm-firm-processor"
  type = "container"

  container_properties = jsonencode({
    image = "${aws_ecr_repository.map.repository_url}:latest"

    resourceRequirements = [
      { type = "VCPU", value = "4" },
      { type = "MEMORY", value = "16384" },
      { type = "GPU", value = "1" }
    ]

    environment = [
      { name = "S3_BUCKET", value = var.s3_bucket },
      { name = "CHECKPOINT_INTERVAL", value = "50" }
    ]

    secrets = [
      {
        name      = "WRDS_USERNAME",
        valueFrom = data.aws_secretsmanager_secret.wrds.arn
      },
      {
        name      = "WRDS_PASSWORD",
        valueFrom = data.aws_secretsmanager_secret.wrds.arn
      }
    ]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = "/aws/batch/ftm"
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "firm-processor"
      }
    }
  })

  retry_strategy {
    attempts = 3
  }

  timeout {
    attempt_duration_seconds = 18000  # 5 hours
  }
}
```

### Job Submission Pattern
```python
def submit_quarter_jobs(quarter: str, firm_ids: List[str], batch_size: int = 1000):
    """Submit Batch jobs for a quarter."""
    batches = [firm_ids[i:i+batch_size] for i in range(0, len(firm_ids), batch_size)]

    job_ids = []
    for idx, batch in enumerate(batches):
        response = batch_client.submit_job(
            jobName=f"ftm-{quarter}-batch-{idx:03d}",
            jobQueue="ftm-queue-main",
            jobDefinition="ftm-firm-processor",
            containerOverrides={
                "environment": [
                    {"name": "QUARTER", "value": quarter},
                    {"name": "FIRM_IDS", "value": ",".join(batch)},
                    {"name": "BATCH_ID", "value": str(idx)},
                ]
            }
        )
        job_ids.append(response["jobId"])

    return job_ids
```

## References

- [AWS Batch User Guide](https://docs.aws.amazon.com/batch/latest/userguide/)
- [AWS Batch Spot Best Practices](https://docs.aws.amazon.com/batch/latest/userguide/bestpractice6.html)
- [Batch Allocation Strategies](https://docs.aws.amazon.com/batch/latest/userguide/allocation-strategies.html)
- Prior architecture discussion: `docs/ai-log/claude-2026-batch-convo.md`
