# ADR-001: Single GPU Instance vs AWS Batch Distributed Processing

**Status**: Accepted
**Decision Makers**: Project Lead + AI Architecture Consultation (Gemini)

---

## Context

The Financial Topic Modeling pipeline processes earnings call transcripts from 3,000-5,000 firms quarterly. The original architecture design followed a distributed computing pattern:

**Original Design (AWS Batch):**

- 3,000+ containers (one per firm)
- Each container: loads embedding model, processes ~500 sentences, writes to S3
- Separate "Reduce" container aggregates results from S3
- Final output to DynamoDB

This design was influenced by the intuition that "more parallelism = faster execution" and that distributed systems are appropriate for "big data" workloads.

---

## Decision

**We pivoted to a single GPU instance (`g4dn.2xlarge`) running all processing sequentially.**

- One EC2 Spot instance processes all 3,000-5,000 firms in a loop
- Embedding model loaded ONCE, reused for all firms
- No intermediate S3 storage; direct write to PostgreSQL
- Theme aggregation runs on same instance immediately after firm processing

---

## Rationale

### 1. Amdahl's Law and the "Overhead Beast"

The critical realization came from analyzing where time was actually spent:

| Step                 | CPU Container | GPU Container |
| -------------------- | ------------- | ------------- |
| Image Pull           | ~5 sec        | ~45 sec       |
| Python Import        | ~3 sec        | ~5 sec        |
| Model Load           | ~4 sec        | ~6 sec        |
| **Actual Inference** | ~10 sec       | **~0.5 sec**  |
| UMAP/HDBSCAN         | ~2 sec        | ~2 sec        |
| **TOTAL**            | ~25 sec       | ~59 sec       |

**Key Insight from Gemini Consultation:**

> "You are currently invoking the 'Overhead Beast' (Container provisioning + Python startup + Model loading) for every tiny slice of 'Actual Work' (Inference)."
>
> "The Inefficiency Ratio: Current Setup: 15 seconds of overhead for 10 seconds of work."

### 2. The 60-Second Billing Minimum

AWS bills EC2 in 1-second increments with a **minimum of 60 seconds**. Both CPU and GPU containers finish a single firm in under 60 seconds, meaning:

- CPU Spot: ~$0.0008 per firm
- GPU Spot: ~$0.0030 per firm
- **GPU is 3.7x more expensive per container** due to idle billing

### 3. Model Loading is the Bottleneck

With 3,000 firms × container cold starts:

- Model loaded 3,000+ times
- Total model load overhead: 3,000 × 5 seconds = **4+ hours** (just loading!)

With a single instance:

- Model loaded **1 time** = 5 seconds
- Model stays in VRAM for all 3,000 firms

### 4. The Scale Reality Check

**From Gemini:**

> "This scale is actually quite small in the world of NLP. You are currently treating this as a 'Big Data' problem requiring orchestration (containers, queues, batch jobs), but it fits comfortably on a single machine."

The math:

- Total sentences: ~2,000,000
- GPU throughput: ~1,200 sentences/sec
- Inference time: ~28 minutes
- Total runtime with overhead: ~2-4 hours
- Total cost: **<$1.00**

---

## Consequences

### Positive

1. **90% Cost Reduction**: From estimated $5-10 (distributed) to <$1.00 (single instance)
2. **Dramatic Simplicity**: No AWS Batch, Step Functions, ECR, or container orchestration
3. **Faster Debugging**: One script, one log file, one machine to SSH into
4. **Better Resource Utilization**: GPU stays 100% utilized; no idle billing
5. **Native Global Context**: All topic summaries already in memory for theme aggregation
6. **Easier Model Upgrades**: Change one line of code vs. rebuilding Docker images

### Negative

1. **Sequential Processing**: Slightly longer wall-clock time than theoretical distributed max
2. **Spot Instance Risk**: Single point of failure (mitigated by checkpoint/resume)
3. **Memory Constraints**: Limited to what fits in 32GB RAM (sufficient for this workload)

### Mitigations

- **Checkpoint/Resume**: Write to PostgreSQL after each firm; re-run resumes from last checkpoint
- **Spot Instance Tolerance**: Batch jobs are interruptible; ~75% cost savings worth the risk

---

## Alternatives Considered

### Alternative 1: AWS Batch with Larger Batches (50 firms/container)

- Reduces cold starts to ~60 containers
- Still requires S3 intermediate storage
- More complex code to handle batch splitting
- **Rejected**: Added complexity for marginal benefit

### Alternative 2: ECS Fargate with Shared Model Cache

- Sidesteps model loading with shared EFS volume
- Higher operational complexity
- Still requires container orchestration
- **Rejected**: Complexity without significant benefit

### Alternative 3: SageMaker Processing Jobs

- Managed infrastructure
- Higher cost ($0.50+/hour minimum)
- Less control over instance selection
- **Rejected**: Cost prohibitive for quarterly batch job

---

## Evidence

### Source Documents

1. **Gemini Architecture Consultation**: `docs/ai-log/gemini-conversation.md`
2. **Original AWS Batch Plan**: `docs/packages/cloud_migration_package/plan.md`
3. **Pivot Mission Briefing**: `docs/packages/architecture_pivot_package/Mission_Briefing_architecture_pivot.md`

### Key Quote

> "For a quarterly job of this size, distributed computing (containers/AWS Batch) introduces more problems than it solves. Best Approach: Use a single `g4dn.xlarge` Spot instance. Run the whole job sequentially in one go. It will cost you about 30 cents and take less than 2 hours. Complexity is minimized, and performance is maximized."
>
> — Gemini Architecture Consultation

### Validated Results

| Metric              | Target   | Actual                 |
| ------------------- | -------- | ---------------------- |
| Firms Processed     | 3,000+   | 11 (MAG7 validation)   |
| Processing Time     | <4 hours | ~15 minutes (11 firms) |
| Infrastructure Cost | <$5      | ~$1.30                 |

---

## Learning Outcome

This ADR documents the most significant learning from this project: **applying parallel computing principles (Amdahl's Law, Gustafson's Law) to cloud architecture decisions.**

The intuition that "distributed = better" for batch processing was wrong for this workload because:

1. The **sequential portion** (model loading, container startup) dominated the **parallelizable portion** (inference)
2. The **data scale** (2M sentences) was "small data" that fits easily on a single machine
3. The **orchestration overhead** (AWS Batch, S3, Step Functions) added complexity without performance benefit

This realization fundamentally changed the project architecture and demonstrated that **simplicity often beats sophistication** in cloud system design.
