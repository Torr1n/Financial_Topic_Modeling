# Sprint 4 Bootstrap: Orchestration & WRDS Prefetch

## Context: You Are Continuing a Research Project

This is a Financial Topic Modeling research project that identifies cross-firm investment themes from earnings call transcripts. You are picking up from Sprint 3, which successfully delivered AWS Batch parallelization for firm-level processing.

**Your role:** Implement Sprint 4 - orchestration layer (Step Functions or equivalent) and critically, solve the WRDS MFA blocker that prevents production runs.

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

## What Sprint 3 Delivered (Your Foundation)

### AWS Batch Infrastructure (`cloud/terraform/batch/`)

- GPU Spot compute environment with `ECS_AL2_NVIDIA` AMI
- Job definition with secrets injection, retry strategy, and circuit breaker
- ECR repository `ftm-map` with working container image
- All IAM roles configured (batch_service, execution, job, instance, spot_fleet)

### Batch Container (`cloud/containers/map/`)

- CUDA 11.8 + cuML 24.04 + PyTorch 2.2.2 (hard-won compatibility)
- `entrypoint.py` with checkpointing, failure tracking, `ALLOW_FAILURES`, and circuit breaker
- Processes firms via WRDSConnector, outputs Parquet to S3

### Job Submitter (`cloud/src/batch/job_submitter.py`)

- `BatchJobSubmitter` class for manifest creation and job submission
- `submit_quarter(quarter, firm_ids, batch_size)` orchestrates full quarter

### Verified Working

- GPU acceleration (cuML UMAP/HDBSCAN)
- Checkpoint/resume across Spot interruptions
- End-to-end pipeline with real WRDS data (Apple, Lamb Weston tested)
- Integration tests with real Capital IQ company IDs

---

## The Critical Blocker: WRDS MFA

**Problem:** WRDS uses IP-based session retention for MFA (Duo Push). Each AWS Batch container gets a different IP, triggering a new MFA challenge. This makes production runs impossible without manual approval per container.

**Recommended Solution (Prefetch to S3):**

```
1. Run WRDS fetcher ONCE per quarter from fixed-IP machine (EC2 or local)
2. Approve MFA once, dump raw transcripts to S3
3. Batch jobs read from S3, never touch WRDS directly
```

**Why this approach:**

- Minimal infrastructure change (no NAT gateway needed)
- Aligns with ADR-007's `raw/` prefix pattern
- Decouples data ingestion from processing
- One MFA approval per quarter instead of per container

**Alternative (more complex):**

- NAT Gateway with Elastic IP for fixed egress
- Requires moving Batch to private subnets
- Adds ~$32/month cost

---

## Sprint 4 Scope

### Must-Have (P0)

1. **WRDS Prefetch Mechanism**
   - Script or lightweight job that fetches all transcripts for a quarter
   - Writes to `s3://bucket/raw/transcripts/quarter=YYYYQN/*.parquet`
   - Designed to run from fixed-IP machine with MFA approval
   - Includes progress tracking for resumability

2. **DATA_SOURCE Environment Variable**
   - Add `DATA_SOURCE=wrds|s3` switch to `entrypoint.py`
   - `wrds`: Current behavior (direct WRDS connection)
   - `s3`: Read pre-fetched transcripts from S3
   - S3 mode skips WRDSConnector entirely

3. **Orchestration Layer**
   - Coordinate: prefetch → manifest creation → batch submission → monitoring
   - Options: Step Functions, simple Python script, or Airflow DAG
   - Recommendation: Start simple (Python script), add Step Functions if needed

### Should-Have (P1)

4. **Production Quarter Test**
   - Run full 2023Q1 (~5000 firms) end-to-end
   - Validate batch parallelization (3-5 jobs)
   - Measure actual runtime and costs

5. **Pin Remaining Package Versions**
   - bertopic, hdbscan, umap-learn currently floating
   - Pin for reproducible builds

### Nice-to-Have (P2)

6. **Monitoring Dashboard**
   - CloudWatch metrics for job progress
   - Cost tracking per quarter

---

## Key Files to Understand

Before writing code, read these files to understand existing patterns:

```
# Core pipeline
cloud/src/firm_processor.py          # FirmProcessor.process() interface
cloud/src/connectors/wrds_connector.py  # WRDSConnector.fetch_transcripts()
cloud/src/topic_models/bertopic_model.py  # BERTopicModel (GPU acceleration)

# Batch infrastructure
cloud/containers/map/entrypoint.py   # Current batch job logic
cloud/src/batch/job_submitter.py     # BatchJobSubmitter class

# Configuration
cloud/terraform/batch/*.tf           # Deployed infrastructure

# Tests
tests/integration/test_batch_integration.py  # Integration test patterns

# Documentation
docs/packages/production_pivot/sprint3_instance_summary.md     # Detailed Sprint 3 learnings
docs/adr/                            # ADRs and design decisions
docs/specs/                          # Interface specs
docs/diagrams/                       # PlantUML architecture diagrams
```

---

## Architecture Constraints

### S3 Bucket Layout (ADR-007)

```
s3://ftm-pipeline-78ea68c8/
├── raw/                    # NEW: Pre-fetched WRDS data
│   └── transcripts/
│       └── quarter=2023Q1/
│           └── *.parquet
├── manifests/              # Batch job manifests
│   └── quarter=2023Q1/
│       └── batches.jsonl
├── progress/               # Checkpoints and failures
│   └── quarter=2023Q1/
│       ├── batch_000_checkpoint.json
│       └── batch_000_failures.json
└── intermediate/           # Firm-level topic output
    └── firm-topics/
        └── quarter=2023Q1/
            └── batch_000_part_0000.parquet
```

### Data Flow (Current → Target)

**Current (Sprint 3):**

```
Batch Job → WRDSConnector → WRDS Database (MFA!) → Process → S3
```

**Target (Sprint 4):**

```
Prefetch Script → WRDS Database (MFA once) → S3 raw/
Batch Job → S3 raw/ → Process → S3 intermediate/
```

### Existing Data Models

**TranscriptData** (from WRDSConnector):

```python
@dataclass
class TranscriptData:
    firms: Dict[str, FirmTranscriptData]  # firm_id → data
    metadata: Dict[str, Any]

@dataclass
class FirmTranscriptData:
    firm_id: str
    firm_name: str
    sentences: List[TranscriptSentence]
    metadata: Dict[str, Any]  # permno, gvkey, earnings_call_date, etc.
```

**Parquet Output Schema** (from entrypoint.py):

```
firm_id, firm_name, quarter, permno, gvkey, earnings_call_date,
topic_id, representation, keywords, n_sentences, sentence_ids, processing_timestamp
```

---

## Implementation Guidance

### For WRDS Prefetch

The prefetcher should:

1. Accept quarter as input (e.g., "2023Q1")
2. Query WRDS for all firms with transcripts in date range
3. Fetch transcript components, run preprocessing (sentence splitting, cleaning)
4. Write FirmTranscriptData-compatible Parquet files to S3
5. Track progress for resumability (in case of interruption)

**Key question:** Should prefetch store raw components or preprocessed sentences?

- **Recommendation:** Store preprocessed (post-WRDSConnector) to avoid duplicating NLP work
- This means prefetch output is directly usable by FirmProcessor

### For DATA_SOURCE Switch

Minimal change to `entrypoint.py`:

```python
data_source = get_env_optional("DATA_SOURCE", "wrds")

if data_source == "wrds":
    with WRDSConnector() as connector:
        transcript_data = connector.fetch_transcripts(...)
elif data_source == "s3":
    transcript_data = load_transcripts_from_s3(bucket, quarter, firm_id)
```

The `load_transcripts_from_s3` function should return the same `TranscriptData` structure that `WRDSConnector.fetch_transcripts()` returns.

### For Orchestration

Start simple:

```python
# orchestrate_quarter.py
def run_quarter(quarter: str):
    # 1. Check if prefetch exists
    if not prefetch_exists(quarter):
        raise RuntimeError(f"Run prefetch for {quarter} first")

    # 2. Get firm IDs from prefetched data
    firm_ids = list_prefetched_firms(quarter)

    # 3. Submit batch jobs
    submitter = BatchJobSubmitter(...)
    results = submitter.submit_quarter(quarter, firm_ids, batch_size=1000)

    # 4. Wait for completion
    final_status = submitter.wait_for_completion([r.job_id for r in results])

    # 5. Report results
    return final_status
```

Add Step Functions later if you need:

- Automatic retries with backoff
- Visual workflow monitoring
- Integration with other AWS services

---

## Testing Strategy

### Unit Tests (Local, No AWS)

- Prefetch logic (Parquet schema, progress tracking)
- DATA_SOURCE switch routing
- Orchestration logic (mock submitter)

### Integration Tests (Requires AWS)

- Prefetch small firm set → verify S3 output
- Batch job with DATA_SOURCE=s3 → verify processing
- Full orchestration flow

### Production Validation

- Run 2023Q1 (~5000 firms) end-to-end
- Target: 3-5 parallel jobs completing in ~10 hours
- Verify output Parquet files are complete

---

## Known Issues & Workarounds

### WRDS Port

WRDS uses port **9737**, not 5432. The `.pgpass` file should be:

```
wrds-pgdata.wharton.upenn.edu:9737:wrds:username:password
```

(WRDSConnector already handles this correctly)

### Package Compatibility

The current Docker image has a stable stack. Don't change versions unless necessary:

```
cuml-cu11==24.04.*
torch==2.2.2
pandas>=2.0,<2.2.2
scikit-learn==1.2
transformers>=4.34.0,<4.43.0
```

### Failure Handling Defaults

```
ALLOW_FAILURES=true  (default; tolerates per-firm errors, writes failures manifest)
ALLOW_FAILURES=false (strict mode; any per-firm error fails the job)
MAX_CONSECUTIVE_FAILURES=5
MAX_FAILURE_RATE=0.05 (5% after MIN_PROCESSED_FOR_RATE=100 firms)
```

---

## Definition of Done (Sprint 4)

- [ ] WRDS prefetch script works from local/EC2 with single MFA approval
- [ ] Prefetched data lands in `s3://bucket/raw/transcripts/quarter=.../`
- [ ] `entrypoint.py` supports `DATA_SOURCE=s3` mode
- [ ] Orchestration script coordinates prefetch → batch → monitoring
- [ ] Full 2023Q1 quarter processed successfully
- [ ] Documentation updated with prefetch workflow
- [ ] Unit and integration tests for new code

---

## Questions to Resolve Early

1. **Prefetch granularity:** One Parquet file per firm, or batch files with multiple firms?
2. **Prefetch machine:** EC2 instance with EIP, or run from local development machine?
3. **Orchestration complexity:** Simple Python script sufficient, or Step Functions needed?
4. **Monitoring:** CloudWatch sufficient, or need custom dashboard?

---

## Reference: Sprint 3 Integration Test Output

This is what success looks like:

```
cuML available - GPU acceleration enabled for UMAP/HDBSCAN
Batch ID: batch_000
Quarter: 2023Q1
Processing 10 pending firms (0 already complete)
Processing firm 374372246 (1/10)
WRDS query returned 40 rows
Built TranscriptData for 1 firms
Processing firm 374372246 (Lamb Weston Holdings, Inc.)
Running topic model on 259 sentences
Using cuML GPU-accelerated UMAP and HDBSCAN
Discovered 20 topics
Wrote 47 rows to s3://ftm-pipeline-78ea68c8/intermediate/firm-topics/quarter=2023Q1/batch_000_part_0000.parquet
Map phase complete: 10 firms processed, 8 skipped (no data)
```

---

## Final Notes

- Read `docs/packages/production_pivot/sprint3_instance_summary.md` for detailed learnings from the previous session
- The WRDS MFA issue is the highest priority - without solving it, production runs are blocked
- Keep solutions simple - a Python script that works is better than a complex Step Functions workflow that doesn't
- Test incrementally - verify prefetch works before building orchestration on top
