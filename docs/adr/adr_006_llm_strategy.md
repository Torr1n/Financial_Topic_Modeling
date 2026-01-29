# ADR-006: Self-Hosted LLM Strategy

## Status
Accepted

## Date
2026-01-26

## Context

The Financial Topic Modeling pipeline uses LLM calls for two purposes:
1. **Topic naming**: Generate human-readable summaries for firm-level topics (~20-50 topics per firm)
2. **Theme descriptions**: Generate descriptions for cross-firm themes (~50-200 themes per quarter)

### Current Implementation
- Uses xAI's Grok API via OpenAI-compatible client (`cloud/src/llm/xai_client.py`)
- Model: `grok-4-1-fast-reasoning`
- Rate limit: ~500 requests/minute (observed ~300-400 req/min at capacity)
- Cost: ~$80 per quarter (3.85M tokens/hour under load)

### Problem with API for Parallel Processing
With 5 parallel Batch jobs (per ADR-005), LLM request rate would be:
- Sequential: ~300 req/min (within API limits)
- 5x parallel: ~1500 req/min (3x over API limit)

**This would cause widespread 429 rate limit errors and job failures.**

### Options Evaluated in Prior Discussion (`docs/ai-log/claude-2026-batch-convo.md`)

| Option | Throughput | Cost/Quarter | Complexity |
|--------|------------|--------------|------------|
| Grok API (current) | 500 req/min | ~$80 | Low |
| Request rate increase | 2000 req/min | ~$80 | Low (if approved) |
| AWS Bedrock (Llama) | Varies | ~$100+ | Medium |
| Self-hosted vLLM (8B) | 2000+ req/min | ~$25-50 | High |
| Self-hosted vLLM (70B) | 500 req/min | ~$100+ | High |

## Decision

**Implement self-hosted LLM inference using vLLM on Amazon ECS with Qwen3-8B model.**

### Embedding Model Hosting (Decision Boundary)

**Decision**: Keep the embedding model **local to each Batch job container** for now. Do **not** host Qwen3-Embedding as a shared ECS service in the initial production pivot.

**Rationale**:
- **Throughput volume**: Embeddings are per-sentence (very high volume) vs. LLM naming per-topic (low volume). A shared endpoint risks becoming the primary bottleneck.
- **Network overhead**: Shipping large text payloads to a remote embedding service adds latency and bandwidth cost.
- **Amortized load**: Map jobs run for hours; loading the embedding model once per job is acceptable.
- **Operational simplicity**: Avoids a second GPU service (autoscaling, batching, retries, payload limits, observability) during the deadline window.

**Revisit Trigger**: If GPU costs dominate after the first multi-quarter run, consider an ECS embedding service to enable CPU-only map jobs. This would require a dedicated throughput benchmark and a new ADR.

### Why vLLM
1. **OpenAI-compatible API**: Minimal client code changes (just change `base_url`)
2. **Continuous batching**: 2-3x throughput vs naive serving
3. **PagedAttention**: Efficient GPU memory management
4. **Proven at scale**: Used by major inference providers

### Why Qwen3-8B
1. **Size vs quality tradeoff**: 8B parameters fits on g5.xlarge (24GB A10G)
2. **Instruction following**: Good at structured output generation
3. **Cost effective**: Single GPU instance handles 2000+ req/min
4. **Open source**: No API keys or licensing concerns

### Quality Acceptance Criteria

**Decision**: Qwen3-8B quality tradeoff is acceptable.

**Validation approach**: Re-run the prior quarter (already processed with Grok) using Qwen3-8B and compare topic naming quality. This provides a direct baseline comparison.

**Rationale**: Scalability and cost reduction are higher priorities than marginal quality differences. The task complexity (financial topic naming) is expected to produce comparable results across capable models.

### Throughput Requirements

| Metric | Target | Range |
|--------|--------|-------|
| Requests per minute | 3,000 | 2,000 - 5,000 |
| Concurrent batch jobs | 5 | 3 - 10 |
| Topics per firm | ~20-50 | varies |

**Sizing calculation**:
- 5 parallel batch jobs × ~300 req/min per job = 1,500 req/min baseline
- Theme descriptions add ~50-200 req/quarter (negligible)
- Buffer for retries and bursts: 2x = 3,000 req/min target

### Why ECS over SageMaker
| Factor | ECS + vLLM | SageMaker |
|--------|------------|-----------|
| Spot support | Native | Async only |
| Cold start | ~2-3 min | ~5-10 min |
| Throughput | vLLM optimized | Standard |
| Control | Full | Limited |
| Complexity | Medium | Low |

For a batch workload needing maximum throughput per dollar, ECS wins.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   ECS vLLM Inference Service                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │               Application Load Balancer                     │ │
│  │                    Port 8000                                │ │
│  │               http://vllm-alb.internal:8000                 │ │
│  └────────────────────────┬───────────────────────────────────┘ │
│                           │                                      │
│                           ▼                                      │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                   ECS Cluster                               │ │
│  │                                                             │ │
│  │   ┌─────────────────────────────────────────────────────┐  │ │
│  │   │              vLLM Service                           │  │ │
│  │   │                                                     │  │ │
│  │   │   Container: vllm/vllm-openai:latest                │  │ │
│  │   │   Model: Qwen/Qwen3-8B                              │  │ │
│  │   │   GPU: 1x A10G (g5.xlarge)                          │  │ │
│  │   │   Endpoints:                                        │  │ │
│  │   │     POST /v1/chat/completions                       │  │ │
│  │   │     POST /v1/completions                            │  │ │
│  │   │     GET /health                                     │  │ │
│  │   │                                                     │  │ │
│  │   │   Capacity Provider:                                │  │ │
│  │   │     80% Spot (g5.xlarge)                            │  │ │
│  │   │     20% On-Demand (baseline)                        │  │ │
│  │   │                                                     │  │ │
│  │   └─────────────────────────────────────────────────────┘  │ │
│  │                                                             │ │
│  │   Auto Scaling:                                             │ │
│  │     Target: CPU 70% (keep warm during batch)               │ │
│  │     Min: 0 (scale to zero when idle)                       │ │
│  │     Max: 4 (handle peak parallel load)                     │ │
│  │                                                             │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Client Migration (Minimal Change)

**Current code** (`cloud/src/llm/xai_client.py` line 105-109):
```python
self._client = AsyncOpenAI(
    api_key=api_key,
    base_url=DEFAULT_BASE_URL,  # "https://api.x.ai/v1"
    timeout=self._timeout,
)
```

**Updated code**:
```python
self._client = AsyncOpenAI(
    api_key=os.environ.get("LLM_API_KEY", api_key),  # vLLM doesn't require key
    base_url=os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL),
    timeout=self._timeout,
)
```

That's it. The rest of the client code (prompts, retry logic, batching) remains unchanged.

## Consequences

### Positive
- **No rate limits**: Self-hosted means unlimited requests
- **Cost reduction**: ~$25-50/quarter vs ~$80 (40% savings)
- **Predictable latency**: No shared API variability
- **Privacy**: Transcript data stays within AWS

### Negative
- **Operational burden**: Must manage ECS cluster, GPU instances
- **Cold start**: 2-3 minutes to load model on new instance
- **Quality tradeoff**: 8B model may be slightly worse than Grok
- **GPU dependency**: Must have g5/g4dn Spot availability

### Mitigations
- **Cold start**: Keep service warm during batch runs (don't scale to zero mid-job)
- **Quality**: Run blind evaluation on sample before production
- **Availability**: Use multiple instance types and On-Demand baseline

## Alternatives Considered

### 1. Request Grok API Rate Limit Increase
- **Pro**: Simplest solution, proven quality
- **Con**: No guarantee of approval; still have API dependency

### 2. AWS Bedrock with Provisioned Throughput
- **Pro**: Managed, no GPU ops
- **Con**: $1,500/month minimum; overkill for batch workload

### 3. Larger Model (Qwen3-70B)
- **Pro**: Better quality outputs
- **Con**: Requires multiple A100s; 4x cost for 2x quality improvement

### 4. Queue-Based Batching with API
- **Pro**: Stay within rate limits
- **Con**: Adds 10-20 minutes latency; complexity

## Implementation Notes

### Terraform (ECS + vLLM)
```hcl
resource "aws_ecs_task_definition" "vllm" {
  family                   = "ftm-vllm"
  requires_compatibilities = ["EC2"]
  network_mode             = "awsvpc"

  container_definitions = jsonencode([{
    name  = "vllm"
    image = "vllm/vllm-openai:latest"

    command = [
      "--model", "Qwen/Qwen3-8B",
      "--host", "0.0.0.0",
      "--port", "8000",
      "--tensor-parallel-size", "1"
    ]

    resourceRequirements = [{
      type  = "GPU"
      value = "1"
    }]

    portMappings = [{
      containerPort = 8000
      hostPort      = 8000
      protocol      = "tcp"
    }]

    environment = [
      { name = "HF_TOKEN", value = var.huggingface_token }
    ]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = "/aws/ecs/ftm-vllm"
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "vllm"
      }
    }
  }])

  cpu    = 4096
  memory = 30720  # 30GB for 8B model
}
```

### Health Check Pattern
```python
async def check_vllm_health(endpoint: str) -> bool:
    """Check if vLLM service is ready."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{endpoint}/health", timeout=10)
            return response.status_code == 200
    except Exception:
        return False
```

### Keep-Warm Strategy: Quarter Overlap

**Key insight**: When one quarter's themes are being aggregated (reduce phase), the next quarter's batch jobs (map phase) can begin. This keeps the vLLM continuously utilized.

```
Quarter 1: [=== MAP ===][= REDUCE =]
Quarter 2:              [=== MAP ===][= REDUCE =]
Quarter 3:                           [=== MAP ===][= REDUCE =]
                                                              ^
                                                         Only idle here
                                                         (final quarter)
```

**Orchestration pattern**:
1. Start Quarter N map phase
2. When Quarter N map completes, start Quarter N reduce AND Quarter N+1 map
3. Repeat until final quarter
4. Final quarter: keep vLLM warm during reduce (only idle period)

**Implementation**: Step Functions parallel states or separate state machine invocations.

### Operational Note: Manual Scaling During Runs (2026-01-29)

For early production runs, keep vLLM **manually scaled** rather than relying on autoscaling. The pipeline is scheduled and batch‑oriented, so predictability matters more than elasticity. Manual scaling keeps vLLM warm throughout a run, avoids cold‑start stalls, and reduces operational complexity.

**Recommended approach:**
- Set **min_capacity = 1** during runs (vLLM stays warm)
- Scale down to 0 **after** runs to save costs (manual or a simple post‑run step)

Autoscaling can be reintroduced later if/when we need dynamic concurrency management.

### Warm-Up Lambda
Before starting the first batch job, ensure vLLM is warm:
```python
# Lambda: check_vllm_health
def handler(event, context):
    ecs = boto3.client("ecs")

    # Ensure at least 1 task running
    ecs.update_service(
        cluster="ftm-cluster",
        service="ftm-vllm",
        desiredCount=1
    )

    # Wait for healthy
    for _ in range(30):  # 5 minutes max
        if check_endpoint_health():
            return {"status": "READY"}
        time.sleep(10)

    raise Exception("vLLM failed to become healthy")
```

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [Qwen3 Model Card](https://huggingface.co/Qwen/Qwen3-8B)
- [ECS GPU Task Definitions](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ecs-gpu.html)
- Prior discussion: `docs/ai-log/claude-2026-batch-convo.md`
