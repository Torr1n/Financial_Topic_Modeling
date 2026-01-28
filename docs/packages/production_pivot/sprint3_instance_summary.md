# Sprint 3 Instance Summary: AWS Batch Parallelization

**Session Date:** 2026-01-28
**Objective:** Parallelize firm-level processing via AWS Batch (5,000 firms/quarter → 3-5 parallel jobs → ~10 hours vs 48 hours sequential)

---

## Executive Summary

Sprint 3 successfully delivered a production-ready AWS Batch infrastructure for parallelizing firm-level topic modeling. The system processes earnings call transcripts through a GPU-accelerated pipeline (cuML + BERTopic) with robust checkpointing, failure tracking, and a circuit breaker to halt systemic failures. Integration tests passed with real WRDS data (Lamb Weston: 20 topics, Apple: 27 topics).

**Critical Blocker Identified:** WRDS MFA requirements may trigger Duo push for each batch container due to IP-based session retention. Recommended solution (WRDS prefetch to S3) deferred to Sprint 4.

---

## What Was Delivered

### Terraform Infrastructure (`cloud/terraform/batch/`)

| File | Purpose |
|------|---------|
| `main.tf` | Provider config, data sources (default VPC, existing S3 bucket, WRDS secret) |
| `compute.tf` | GPU Spot compute environment with `ECS_AL2_NVIDIA` AMI, job queue |
| `job_definition.tf` | Job definition with GPU, secrets injection, 3-attempt retry, 5-hour timeout |
| `iam.tf` | 5 IAM roles with S3 and Secrets Manager permissions |
| `ecr.tf` | ECR repository `ftm-map` with lifecycle policy (keep last 5 images) |
| `variables.tf` | Input variables (aws_region, s3_bucket_name) |
| `outputs.tf` | Outputs for deployment scripts |

**Key Terraform Decisions:**
- **Separate root directory** (`cloud/terraform/batch/`) isolates from existing live state to prevent drift
- **Default VPC with public subnets** avoids NAT gateway cost (~$32/month) per ADR-005
- **WRDS secret created out-of-band** keeps credentials out of terraform state
- **GPU AMI explicitly pinned** (`ECS_AL2_NVIDIA`) prevents "nvidia-smi not found" failures

### Docker Container (`cloud/containers/map/`)

| File | Purpose |
|------|---------|
| `Dockerfile` | CUDA 11.8 base, strict 5-step install order, build-time verification |
| `requirements.txt` | Pinned versions for cuML 24.04 compatibility |
| `entrypoint.py` | Batch orchestration with checkpointing, failure tracking, `ALLOW_FAILURES`, and circuit breaker |

**Critical Package Compatibility (Hard-Won Knowledge):**

```
# INSTALL ORDER MATTERS - DO NOT REORDER
Step 1: Base numerical (numpy>=1.23,<2.0, pandas>=2.0,<2.2.2, scikit-learn==1.2)
Step 2: RAPIDS cuML 24.04 (NOT 24.02 - has pandas 2.x compatibility)
Step 3: PyTorch 2.2.2 (sentence-transformers 3.x requires >=2.2)
Step 4: Transformers ecosystem (transformers>=4.34.0,<4.43.0)
Step 5: Application dependencies
```

**Why These Specific Versions:**
- `cuML 24.02` → `24.04`: Fixed `pandas.api.types.is_extension_type` deprecation error
- `PyTorch 2.0.1` → `2.2.2`: sentence-transformers 3.x requires PyTorch ≥2.2
- `pandas <2.2.2`: Upper bound for cuML 24.04 compatibility
- `scikit-learn==1.2`: Required by cuML, no NumPy 2.0 support

### Python Module (`cloud/src/batch/`)

| File | Purpose |
|------|---------|
| `__init__.py` | Package exports |
| `job_submitter.py` | `BatchJobSubmitter` class for manifest creation, job submission, monitoring |

**BatchJobSubmitter Capabilities:**
- `create_manifest(quarter, firm_ids, batch_size)` → JSONL manifest in S3
- `submit_job(quarter, batch_id, manifest_key)` → Single job submission
- `submit_quarter(quarter, firm_ids, batch_size)` → Full quarter orchestration
- `get_job_status(job_ids)` → Status polling (handles >100 jobs via pagination)
- `wait_for_completion(job_ids, poll_interval, timeout)` → Blocking wait

### Entrypoint Design (`entrypoint.py`)

**Core Flow:**
1. Parse environment variables (MANIFEST_S3_KEY, BATCH_ID, QUARTER, S3_BUCKET)
2. Load manifest from S3, find assigned firm IDs
3. Load checkpoint if exists (Spot interruption resume)
4. Initialize models ONCE (SentenceTransformer + BERTopicModel on GPU)
5. Process firms with WRDSConnector, checkpoint every 50 firms
6. Write Parquet chunks to S3, track failures
7. Exit 0 (success) or 1 (circuit breaker tripped or `ALLOW_FAILURES=false`)

**Circuit Breaker Pattern:**
```python
# Fail job if:
# 1. Critical error (WRDS connection/auth, CUDA OOM, AWS access) → immediate fail
# 2. Consecutive failures >= MAX_CONSECUTIVE_FAILURES (default: 5)
# 3. Failure rate >= MAX_FAILURE_RATE after MIN_PROCESSED_FOR_RATE (default: 5% after 100 firms)
```

**Failure Handling:**
- `ALLOW_FAILURES=true` (default): Tolerates per-firm errors, writes failures manifest, exits 0
- `ALLOW_FAILURES=false`: Any per-firm error fails the job
- Circuit breaker always fails regardless of `ALLOW_FAILURES`
- Failures manifest is always written when errors occur

**Parquet Output Schema:**
```
firm_id, firm_name, quarter, permno, gvkey, earnings_call_date,
topic_id, representation, keywords, n_sentences, sentence_ids, processing_timestamp
```
- Path: `s3://bucket/intermediate/firm-topics/quarter=2023Q1/batch_000_part_0000.parquet`
- `quarter` column included for downstream tooling (even though it's a partition key)

### Tests (`tests/integration/test_batch_integration.py`)

**Unit Tests (local, no AWS required):**
- Manifest creation and JSONL format
- Checkpoint save/load round-trip
- ClientError handling for missing checkpoints
- Circuit breaker config defaults + env overrides
- Quarter to date range conversion
- Parquet schema validation
- Job submitter API calls

**Integration Tests (2 tests, requires AWS + deployed infrastructure):**
- `test_submit_job_to_batch` - Single firm smoke test
- `test_process_10_firms_via_batch` - End-to-end with 10 Mag 7 companies

**Real Capital IQ Company IDs (Verified in WRDS):**
```python
"18749"     # Amazon.com, Inc.
"19691"     # Cisco Systems, Inc.
"21835"     # Microsoft Corporation
"24937"     # Apple Inc.
"29096"     # Alphabet Inc.
"32307"     # NVIDIA Corporation
"20765463"  # Meta Platforms, Inc.
"25016048"  # Broadcom Inc.
"27444752"  # Tesla, Inc.
"33348547"  # Arista Networks Inc
```

---

## The Reasoning (Why We Made These Choices)

### 1. Separate Terraform Root
**Problem:** Existing `cloud/terraform/` has live state with RDS, S3, SGs.
**Risk:** Mixing Batch resources could cause drift or accidental deletion.
**Decision:** New root at `cloud/terraform/batch/` with clean isolation.
**Validation:** Terraform apply created 20 resources without touching existing infra.

### 2. Default VPC (No Bespoke VPC)
**Problem:** NAT gateway costs ~$32/month and adds complexity.
**Decision:** Per ADR-005, use default VPC with public subnets.
**Trade-off:** Batch instances have public IPs but only egress (no ingress rules).
**Future Note:** If WRDS MFA becomes blocking, may need NAT gateway for fixed egress IP.

### 3. cuML 24.04 Over 24.02
**Problem:** cuML 24.02 calls `pandas.api.types.is_extension_type()` which was removed in pandas 2.1.
**Root Cause:** RAPIDS 24.02 was in pandas 1.x → 2.x transition period.
**Fix:** cuML 24.04 has full pandas 2.x support.
**Lesson:** Always check RAPIDS release notes for pandas compatibility.

### 4. Strict Dockerfile Install Order
**Problem:** pip dependency resolution can install incompatible versions if order is wrong.
**Critical Insight:** cuML must install BEFORE PyTorch to avoid CUDA package conflicts.
**Solution:** 5-step install order with explicit version pins at each step.
**Validation:** Build-time import verification catches errors before push.

### 5. Failure Tolerance + Circuit Breaker
**Problem:** Sequential single-quarter run had 3-7 firm failures out of ~7000 (rare edge cases).
**Decision:** Allow scattered failures in production while halting systemic issues quickly.
**Implementation:**
  - `ALLOW_FAILURES=true` (default): job succeeds but writes a failures manifest
  - Critical errors → immediate fail
  - Consecutive failures / failure rate thresholds → circuit breaker
**Rationale:** Avoids silent data loss while tolerating rare per-firm issues.

### 6. Single Model Instantiation
**Problem:** Loading SentenceTransformer + BERTopicModel per firm would be extremely slow.
**Decision:** Load once at job start, reuse for all firms in batch.
**Implementation:** `embedding_model` passed to `BERTopicModel(config, embedding_model=embedding_model)`
**Benefit:** ~3-4 seconds per firm instead of ~30+ seconds.

### 7. Local Temp File + boto3 Upload
**Problem:** s3fs and pyarrow.fs have compatibility issues with some environments.
**Decision:** Write Parquet to `/tmp`, upload with boto3, delete local file.
**Trade-off:** Slightly more I/O but much simpler and more reliable.

### 8. ALLOW_FAILURES Optional (Default = true)
**Problem:** Strict failure mode would fail jobs for rare edge cases.
**Decision:** Default to tolerant mode for production, allow strict mode via `ALLOW_FAILURES=false`.
**Safety Net:** Failures manifest is always written; circuit breaker halts systemic issues.

---

## What Went Well

1. **Terraform Deployment** - 20 resources created cleanly, no conflicts with existing infra
2. **GPU Acceleration** - cuML detected and used for UMAP/HDBSCAN (`cuML available - GPU acceleration enabled`)
3. **Package Compatibility** - After debugging, stable stack with cuML 24.04 + PyTorch 2.2.2
4. **Checkpoint/Resume** - Second job correctly detected "10 already complete" and skipped
5. **End-to-End Pipeline** - Real WRDS data processed (Lamb Weston: 259 sentences → 20 topics)
6. **Codex Reviews** - Multiple rounds caught issues (ClientError handling, Secrets Manager on job role, GPU AMI)

---

## Challenges & Debugging Journey

### Challenge 1: PyTorch Version Mismatch
**Symptom:** `Disabling PyTorch because PyTorch >= 2.2 is required but found 2.0.1+cu118`
**Root Cause:** sentence-transformers 3.x dropped support for older PyTorch
**Fix:** Upgraded `torch==2.0.1` → `torch==2.2.2` in Dockerfile
**Time to Diagnose:** ~10 minutes (clear error message)

### Challenge 2: cuML/pandas Incompatibility
**Symptom:** `AttributeError: module 'pandas.api.types' has no attribute 'is_extension_type'`
**Root Cause:** cuML 24.02 built against pandas 1.x API
**Debugging Path:**
  1. First attempted `pandas<2.1.0` pin (didn't work - cudf pulls its own pandas)
  2. Dispatched api-docs-synthesizer for deep research
  3. Discovered cuML 24.04 has pandas 2.x support
**Fix:** `cuml-cu11==24.02.*` → `cuml-cu11==24.04.*` + strict install order
**Time to Diagnose:** ~1 hour (required package ecosystem research)

### Challenge 3: Test Firm IDs
**Symptom:** 8 of 10 firms skipped with "No transcript data"
**Root Cause:** Placeholder IDs (1045810, 1418135, etc.) weren't real Capital IQ company IDs
**Fix:** User provided verified Mag 7 + other company IDs
**Lesson:** Always use real identifiers for integration tests

---

## Confidence Levels

### High Confidence ✓
- Terraform infrastructure is correctly configured
- Docker container builds and runs with GPU acceleration
- Package versions are compatible (cuML 24.04 + PyTorch 2.2.2 + pandas 2.x)
- Checkpoint/resume mechanism works correctly
- Circuit breaker + failure handling are working
- IAM permissions are sufficient (tested end-to-end)

### Medium Confidence ~
- Large-scale behavior (only tested 10 firms, not 1000+ per batch)
- Memory allocation (14GB should be fine but not stress-tested with large firms)
- Spot interruption recovery (checkpoint logic is correct but not tested with real interruption)

### Low Confidence / Known Risks ⚠
- **WRDS MFA** - Each batch container may trigger Duo push (IP-based session retention)
- **Floating package versions** - bertopic, hdbscan, umap-learn not pinned
- **Production quarter run** - Not yet attempted with full ~5000 firms

---

## What Wasn't Completed (Deferred to Sprint 4)

### 1. WRDS Prefetch to S3
**Problem:** WRDS MFA triggers for each batch container (different IPs)
**Proposed Solution:**
  - Run WRDS fetcher once per quarter from fixed-IP machine
  - Write raw transcripts to `s3://bucket/raw/transcripts/quarter=.../`
  - Batch jobs read from S3, never touch WRDS
**Rationale for Deferral:** Sprint 4 already introduces orchestration (Step Functions), natural place to add prefetch step

### 2. DATA_SOURCE=wrds|s3 Switch
**Purpose:** Allow entrypoint to read from prefetched S3 instead of live WRDS
**Implementation:** Environment variable switch in entrypoint.py
**Dependency:** Requires prefetch mechanism above

### 3. NAT Gateway for Fixed Egress IP
**Alternative to prefetch:** Route all Batch traffic through NAT with Elastic IP
**Cost:** ~$0.045/hr + data transfer
**Trade-off:** More infrastructure complexity vs simpler prefetch approach

### 4. Pin Remaining Package Versions
**Currently floating:** bertopic, hdbscan, umap-learn
**Risk:** Future pip installs may get incompatible versions
**Mitigation:** Low risk short-term, should pin for reproducible builds

---

## Residual Risks for Sprint 4

| Risk | Severity | Mitigation |
|------|----------|------------|
| WRDS MFA per container | HIGH | Implement prefetch-to-S3 workflow |
| Floating package versions | LOW | Pin bertopic, hdbscan, umap-learn |
| Large firm memory usage | LOW | Monitor first production runs, increase if needed |
| Spot interruption untested | LOW | Checkpoint logic is correct, will validate in production |

---

## Deployment Artifacts

### Terraform State
- Location: `cloud/terraform/batch/terraform.tfstate` (local, not committed)
- Resources: 20 created
- Key outputs: `ecr_repository_url`, `job_definition_name`, `job_queue_name`, `s3_bucket_name`

### ECR Image
- Repository: `666938415731.dkr.ecr.us-east-1.amazonaws.com/ftm-map`
- Tag: `latest`
- Status: Pushed and tested

### S3 Outputs (from integration test)
- Manifest: `s3://ftm-pipeline-78ea68c8/manifests/quarter=2023Q1/batches.jsonl`
- Output: `s3://ftm-pipeline-78ea68c8/intermediate/firm-topics/quarter=2023Q1/batch_000_part_0000.parquet`
- Checkpoint: `s3://ftm-pipeline-78ea68c8/progress/2023Q1/batch_000_checkpoint.json`

---

## Files Modified/Created This Session

```
cloud/terraform/batch/
├── main.tf              # NEW
├── compute.tf           # NEW
├── job_definition.tf    # NEW
├── iam.tf               # NEW
├── ecr.tf               # NEW
├── variables.tf         # NEW
└── outputs.tf           # NEW

cloud/containers/map/
├── Dockerfile           # NEW
├── requirements.txt     # NEW
└── entrypoint.py        # NEW

cloud/src/batch/
├── __init__.py          # NEW
└── job_submitter.py     # NEW

tests/integration/
└── test_batch_integration.py  # NEW

pytest.ini               # NEW
```

---

## Commands Reference

```bash
# Deploy infrastructure
cd cloud/terraform/batch
terraform init
terraform plan -var="s3_bucket_name=ftm-pipeline-78ea68c8"
terraform apply -var="s3_bucket_name=ftm-pipeline-78ea68c8"

# Build and push container
ECR_URL=$(terraform output -raw ecr_repository_url)
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ${ECR_URL%/*}
docker build --no-cache -t ftm-map:latest -f cloud/containers/map/Dockerfile .
docker tag ftm-map:latest $ECR_URL:latest
docker push $ECR_URL:latest

# Run tests
pytest tests/integration/test_batch_integration.py -v -m "not integration"  # Unit tests
pytest tests/integration/test_batch_integration.py -v -m integration         # Integration tests

# Submit production quarter
python -c "
from cloud.src.batch import BatchJobSubmitter
submitter = BatchJobSubmitter(
    job_definition='ftm-firm-processor',
    job_queue='ftm-queue-main',
    s3_bucket='ftm-pipeline-78ea68c8'
)
results = submitter.submit_quarter('2023Q1', firm_ids=real_firm_ids, batch_size=1000)
"
```

---

## Key Learnings for Future Instances

1. **RAPIDS/cuML version compatibility is fragile** - Always check release notes for pandas support
2. **Dockerfile install order matters for CUDA packages** - cuML before PyTorch
3. **Build-time verification catches issues early** - Add import checks in Dockerfile
4. **Test with real identifiers** - Placeholder IDs cause confusing "no data" results
5. **Circuit breaker prevents systemic failure loops** - Essential for production robustness
6. **Codex reviews catch edge cases** - Multiple rounds improved error handling significantly
7. **WRDS MFA is a production blocker** - Must address before full-scale runs
