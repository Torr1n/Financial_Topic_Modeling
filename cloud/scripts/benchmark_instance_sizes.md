# Instance Sizing Benchmark Guide

## Overview

Before locking instance sizes for AWS Batch job definitions, run this benchmarking
process to validate memory requirements for BERTopic + sentence-transformers.

## Recommended Starting Points

Per the approved plan:
- **Map Phase:** m5.xlarge (16GB RAM, 4 vCPU)
- **Reduce Phase:** m5.2xlarge (32GB RAM, 8 vCPU)

These are conservative estimates. Actual requirements depend on:
- Number of sentences per firm
- Embedding model size (all-mpnet-base-v2 ~420MB)
- BERTopic internal data structures

## Local Benchmarking

### 1. Install memory profiling tools

```bash
pip install memory_profiler psutil
```

### 2. Run local memory benchmark

```python
# benchmark_memory.py
import os
import psutil
from memory_profiler import profile

# Set environment
os.environ["LOCAL_MODE"] = "true"
os.environ["FIRM_ID"] = "1001"  # Or your test firm
os.environ["LOCAL_INPUT"] = "transcripts_2023-01-01_to_2023-03-31_enriched.csv"
os.environ["LOCAL_OUTPUT"] = "output/benchmark"

@profile
def run_map_phase():
    from cloud.containers.map.entrypoint import main
    main()

if __name__ == "__main__":
    process = psutil.Process()
    print(f"Starting memory: {process.memory_info().rss / 1024**3:.2f} GB")
    run_map_phase()
    print(f"Peak memory: {process.memory_info().rss / 1024**3:.2f} GB")
```

### 3. Measure with different firm sizes

Test with firms of varying sentence counts:
- Small firm: ~100 sentences
- Medium firm: ~500 sentences
- Large firm: ~2000+ sentences

## AWS Benchmarking

After local testing, validate on AWS with actual instance types:

```bash
# Test with different instance types
for INSTANCE in t3.large m5.large m5.xlarge; do
  echo "Testing $INSTANCE..."

  aws batch submit-job \
    --job-name "benchmark-map-${INSTANCE}" \
    --job-definition map-benchmark-${INSTANCE} \
    --job-queue ftm-benchmark-queue \
    --container-overrides '{
      "environment": [
        {"name": "FIRM_ID", "value": "YOUR_TEST_FIRM_ID"}
      ]
    }'
done
```

## Metrics to Capture

From CloudWatch Container Insights:

1. **Memory Utilization (%)** - Peak during processing
2. **Memory Reservation (bytes)** - Total allocated
3. **CPU Utilization (%)** - Processing efficiency

From job logs:

4. **Processing Time** - Total duration
5. **OOM Errors** - Any out-of-memory failures

## Decision Criteria

Lock instance sizes when:

| Instance | Memory OK | No OOM | Time <5min |
|----------|-----------|--------|------------|
| t3.large (8GB) | ? | ? | ? |
| m5.large (8GB) | ? | ? | ? |
| m5.xlarge (16GB) | ? | ? | ? |

Choose the smallest instance that passes all criteria.

## Cost Comparison (us-east-1, Spot pricing)

| Instance | Memory | vCPU | Spot $/hr | 100 firms |
|----------|--------|------|-----------|-----------|
| t3.large | 8GB | 2 | ~$0.02 | ~$0.10 |
| m5.large | 8GB | 2 | ~$0.03 | ~$0.15 |
| m5.xlarge | 16GB | 4 | ~$0.06 | ~$0.30 |
| m5.2xlarge | 32GB | 8 | ~$0.12 | ~$0.60 |

*Assumes 3 minutes per firm processing time*

## Recommendation

Start with m5.xlarge for map phase to ensure stability, then right-size down
based on actual memory usage if cost optimization is needed.
