# Sprint 5 Bootstrap: vLLM + Step Functions Orchestration

## Context: You Are Continuing a Research Project

This is a Financial Topic Modeling research project that identifies cross-firm investment themes from earnings call transcripts. You are picking up from Sprint 4, which solved the WRDS MFA blocker via a "prefetch to S3" pattern.

**Your role:** Implement Sprint 5 - deploy vLLM for topic naming and implement Step Functions orchestration. Production validation moves to Sprint 6.

---

## Guiding Principles (Non-Negotiable)

These principles come from the project's founding document and must guide every decision:

> "The best engineers write code my mom could read."

1. **Simplicity over complexity** - Boring technology, obvious patterns, no clever tricks
2. **Document the "why"** - Code comments explain reasoning, not mechanics
3. **Under-engineer the "how"** - Minimum viable complexity for current requirements
4. **Test-driven validation** - Validate as you go, don't accumulate technical debt
5. **Modular design** - Components should be swappable without rewiring

**Anti-patterns to avoid:**

- Over-abstraction for hypothetical futures
- Feature flags or backwards-compatibility shims
- Adding features beyond what's requested
- Clever code that requires mental compilation

---

## What Sprint 4 Delivered (Your Foundation)

### WRDS Prefetch Infrastructure

The core problem was WRDS IP-based MFA (Duo Push) - each AWS Batch container gets a different IP, triggering new MFA challenges. Sprint 4 solved this with "prefetch to S3":

| Component | File | Purpose |
|-----------|------|---------|
| WRDSPrefetcher | `cloud/src/prefetch/wrds_prefetcher.py` | Fetch all transcripts for a quarter to S3 |
| S3TranscriptConnector | `cloud/src/connectors/s3_connector.py` | Manifest-based selective loading |
| DATA_SOURCE switch | `cloud/containers/map/entrypoint.py` | Route to S3 or WRDS connector |
| QuarterOrchestrator | `cloud/src/orchestrate/quarter_orchestrator.py` | Coordinate prefetch → batch → monitor |

### Data Flow

```
[Once per quarter, fixed-IP machine]
WRDSPrefetcher → WRDS (MFA once) → s3://bucket/prefetch/transcripts/quarter=.../

[Parallel Batch jobs]
Batch Job → S3TranscriptConnector → manifest → selective chunks → FirmProcessor → intermediate/
```

### Key Design Decisions Made in Sprint 4

1. **Preprocessed storage:** Prefetch stores `cleaned_text` (already lemmatized, stopwords removed) - no re-preprocessing in batch jobs
2. **Manifest-based loading:** `manifest.json` maps firm_id → chunk file for O(1) lookup
3. **Explicit DATA_SOURCE:** Default `wrds` for local dev, `s3` for Terraform (production)
4. **Prefetch required:** QuarterOrchestrator fails explicitly if manifest missing (no silent WRDS fallback)

### Verified Working

- All 47 unit tests passing
- Integration tests for prefetch pipeline
- Terraform updated with DATA_SOURCE and prefetch S3 permissions
- Documentation updated (ADR-007, spec docs)

---

## Sprint 5 Scope

### Phase 1: vLLM Deployment (P1)

**Objective:** Replace Grok API with self-hosted vLLM for topic naming (removes rate limits).

1. **ECS Task Definition for vLLM**
   - Qwen3-8B model (fits on g5.xlarge)
   - OpenAI-compatible API endpoint
   - Terraform in `cloud/terraform/ecs/`

2. **xai_client.py Base URL Switch**
   - Add `LLM_BASE_URL` environment variable
   - Default to Grok API for backward compatibility
   - Production: point to ECS vLLM endpoint

3. **Integration Testing**
   - Verify topic naming with self-hosted vLLM
   - Compare quality to Grok API output

### Phase 2: Step Functions (P2)

**Objective:** Visual orchestration for multi-quarter processing.

1. **State Machine Definition**
   - Prefetch check → Batch submission → Monitoring → Completion

2. **Error Handling**
   - Automatic retries with backoff
   - Alerting on failures

**Note:** Step Functions are part of Sprint 5 per the updated plan.

---

## Key Files to Understand

Before writing code, read these files to understand existing patterns:

```
# Sprint 4 deliverables (your foundation)
cloud/src/prefetch/wrds_prefetcher.py     # WRDSPrefetcher class
cloud/src/connectors/s3_connector.py      # S3TranscriptConnector
cloud/src/orchestrate/quarter_orchestrator.py  # QuarterOrchestrator
cloud/containers/map/entrypoint.py        # DATA_SOURCE switch

# Core pipeline (unchanged from Sprint 3)
cloud/src/firm_processor.py               # FirmProcessor.process()
cloud/src/topic_models/bertopic_model.py  # BERTopicModel (GPU)
cloud/src/connectors/wrds_connector.py    # WRDSConnector + get_firm_ids_in_range()

# LLM client (to be modified for vLLM)
cloud/src/llm/xai_client.py               # Current Grok API client

# Terraform
cloud/terraform/batch/*.tf                # Batch infrastructure

# Documentation
docs/packages/production_pivot/sprint4_instance_summary.md  # Detailed Sprint 4 learnings
docs/adr/adr_006_llm_strategy.md          # vLLM design decisions
docs/adr/adr_007_storage_strategy.md      # S3 layout including prefetch

# Tests
tests/unit/test_wrds_prefetcher.py        # Prefetcher tests
tests/unit/test_s3_connector.py           # S3 connector tests
tests/unit/test_quarter_orchestrator.py   # Orchestrator tests
tests/integration/test_prefetch_integration.py  # Integration tests
```

---

## Architecture Constraints

### S3 Bucket Layout (Post-Sprint 4)

```
s3://ftm-pipeline-78ea68c8/
├── prefetch/                              # WRDS prefetch (Sprint 4)
│   └── transcripts/
│       └── quarter=2023Q1/
│           ├── chunk_0000.parquet         # ~200 firms each
│           ├── chunk_0001.parquet
│           ├── ...
│           ├── manifest.json              # firm_to_chunk mapping (gzip)
│           └── _checkpoint.json           # Resume state (deleted on completion)
├── manifests/                             # Batch job manifests
│   └── quarter=2023Q1/
│       └── manifest_YYYYMMDD_HHMMSS.jsonl
├── progress/                              # Checkpoints and failures
│   └── quarter=2023Q1/
│       ├── 2023Q1_batch_0000_checkpoint.json
│       └── 2023Q1_batch_0000_failures.json
└── intermediate/                          # Firm-level topic output
    └── firm-topics/
        └── quarter=2023Q1/
            └── 2023Q1_batch_0000_part_0000.parquet
```

### Prefetch Parquet Schema

```
firm_id (STRING)           # Capital IQ companyid
firm_name (STRING)         # Company name
permno (INT64)             # CRSP PERMNO (required)
gvkey (STRING)             # Compustat GVKEY
transcript_id (STRING)     # Transcript identifier
earnings_call_date (DATE)  # Earnings call date
sentence_id (STRING)       # Format: {firm_id}_{transcript_id}_{position:04d}
raw_text (STRING)          # Original sentence
cleaned_text (STRING)      # Preprocessed for embeddings
speaker_type (STRING)      # CEO, CFO, Analyst, etc.
position (INT32)           # Order in transcript
quarter (STRING)           # Partition key
```

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `DATA_SOURCE` | `wrds` (entrypoint) / `s3` (Terraform) | Route to connector |
| `MANIFEST_S3_KEY` | Required | Batch manifest location |
| `BATCH_ID` | Required | This batch's identifier |
| `QUARTER` | Required | Quarter being processed |
| `S3_BUCKET` | Required | Output bucket |
| `CHECKPOINT_INTERVAL` | `50` | Firms per checkpoint |
| `ALLOW_FAILURES` | `true` | Tolerate per-firm errors |

---

## Implementation Guidance

### For Production Validation (Sprint 6)

1. **Prefetch Machine Setup**
   - EC2 instance with Elastic IP (fixed IP for MFA), OR
   - Local development machine with stable IP
   - Ensure WRDS credentials available (env vars or .pgpass)

2. **Monitoring Prefetch Progress**
   - Check `_checkpoint.json` for progress
   - Watch CloudWatch logs if running from EC2
   - Prefetch is resumable - safe to interrupt and restart

3. **Validating Output**
   ```python
   import boto3
   import gzip
   import json

   s3 = boto3.client("s3")

   # Check manifest
   response = s3.get_object(
       Bucket="ftm-pipeline-78ea68c8",
       Key="prefetch/transcripts/quarter=2023Q1/manifest.json"
   )
   manifest = json.loads(gzip.decompress(response["Body"].read()))
   print(f"Firms: {manifest['n_firms']}, Chunks: {manifest['n_chunks']}")
   ```

### For vLLM Deployment

1. **ECS Setup (Terraform)**
   ```hcl
   resource "aws_ecs_task_definition" "vllm" {
     family = "ftm-vllm"
     requires_compatibilities = ["EC2"]

     container_definitions = jsonencode([{
       name  = "vllm"
       image = "vllm/vllm-openai:latest"
       command = ["--model", "Qwen/Qwen3-8B", "--max-model-len", "4096"]

       resourceRequirements = [
         { type = "GPU", value = "1" }
       ]

       portMappings = [{
         containerPort = 8000
         hostPort      = 8000
       }]
     }])
   }
   ```

2. **xai_client.py Modification**
   ```python
   LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.x.ai/v1")

   # In XAIClient.__init__:
   self.client = OpenAI(
       api_key=self.api_key,
       base_url=LLM_BASE_URL,  # Now configurable
   )
   ```

3. **Integration Test**
   ```python
   # Point to local vLLM for testing
   os.environ["LLM_BASE_URL"] = "http://localhost:8000/v1"
   os.environ["XAI_API_KEY"] = "dummy"  # vLLM doesn't require auth

   client = XAIClient()
   result = client.generate_topic_name(keywords, sentences)
   ```

---

## Testing Strategy

### Phase 1: vLLM

- Unit tests: Mock OpenAI client, verify base_url routing
- Integration tests: Local vLLM container, verify topic naming
- Comparison test: Same inputs to Grok API and vLLM, compare quality

### Phase 2: Step Functions

- Unit tests: Mock AWS Step Functions client
- Integration tests: Deploy to AWS, trigger execution
- Only if Python orchestrator proves insufficient

---

## Known Issues & Workarounds

### Test Patch Paths

When mocking connectors, patch at the **source module**:

```python
# CORRECT:
with patch("cloud.src.connectors.s3_connector.S3TranscriptConnector") as mock:

# WRONG:
with patch("cloud.containers.map.entrypoint.S3TranscriptConnector") as mock:
```

### describe_jobs AWS Limit

AWS Batch `describe_jobs` only accepts 100 job IDs. QuarterOrchestrator already handles this with chunking.

### WRDS MFA Timing

WRDS MFA session expires. If prefetch runs for many hours, MFA may need re-approval. The checkpoint system handles this - just restart prefetch.

---

## Definition of Done (Sprint 5)

### Phase 1: vLLM
- [ ] ECS task definition for vLLM deployed
- [ ] xai_client.py supports LLM_BASE_URL
- [ ] Topic naming works with self-hosted vLLM
- [ ] Unit and integration tests

### Phase 2: Step Functions
- [ ] State machine definition
- [ ] Error handling and retries
- [ ] Visual monitoring via AWS console

---

## Questions to Resolve Early

1. **Prefetch machine:** EC2 with EIP or local development machine?
2. **vLLM model:** Qwen3-8B confirmed? Any alternatives?
3. **Step Functions:** Confirm state machine scope for multi-quarter orchestration.
4. **Multi-quarter:** Process 2023Q1-Q4 serially or in parallel?

---

## Reference: Sprint 4 Component Summary

| Component | Status | Notes |
|-----------|--------|-------|
| WRDSPrefetcher | Complete | Chunks, manifest, checkpoint |
| S3TranscriptConnector | Complete | Selective loading, no re-preprocessing |
| DATA_SOURCE switch | Complete | Factory pattern in entrypoint |
| QuarterOrchestrator | Complete | Prefetch check, job submission, monitoring |
| Terraform updates | Complete | DATA_SOURCE env, prefetch S3 permissions |
| Unit tests | Complete | 47 tests passing |
| Integration tests | Complete | Prefetch pipeline tests |
| Documentation | Complete | ADR-007, spec docs updated |

---

## Final Notes

- Read `docs/packages/production_pivot/sprint4_instance_summary.md` for detailed learnings
- Production validation is the first priority - it validates all Sprint 4 work
- vLLM deployment follows the pattern in ADR-006
- Keep solutions simple - only add Step Functions if truly needed
- Test incrementally - validate each phase before moving to the next
