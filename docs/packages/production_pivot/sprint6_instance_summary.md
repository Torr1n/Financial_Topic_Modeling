# Sprint 6 Instance Summary

**Session Date**: 2026-01-29
**Agent**: Claude Opus 4.5
**Session Purpose**: Integration testing of vLLM + Step Functions pipeline

---

## Executive Summary

Sprint 6 was intended as an integration testing phase to validate the vLLM + Step Functions infrastructure deployed in Sprint 5. The smoke test (10 firms, 1 batch) **succeeded mechanically** but revealed two fundamental gaps: **LLM topic naming is not wired into the map phase**, and **the reduce phase (cross-firm theme aggregation) does not exist in the workflow**.

These gaps mean the pipeline produces firm-level topics with keyword representations only, without LLM-generated summaries, and does not aggregate them into cross-firm investment themes—the core research objective.

---

## What Was Accomplished

### Infrastructure Validation (Successful)

| Component | Status | Evidence |
|-----------|--------|----------|
| Prefetch → S3 | ✅ Working | 10 firms, 3,996 sentences prefetched |
| S3TranscriptConnector | ✅ Working | Map phase reads from S3, not WRDS |
| Step Functions orchestration | ✅ Working | CheckPrefetch → CreateBatchManifest → Map → SummarizeResults → NotifyCompletion → CheckForFailures → Success/Fail |
| Batch job execution | ✅ Working | 10/10 firms processed, 0 failures |
| vLLM deployment | ✅ Working | Scales up, serves health checks, scales down |
| Parquet output schema | ✅ Correct | firm_id, permno, gvkey, topic_id, representation, keywords, etc. |

### Issues Diagnosed and Fixed

**Issue: Batch jobs using WRDS connector despite `DATA_SOURCE=s3`**

- **Symptom**: Map phase failed with `WRDSConnectionError` (interactive prompt for credentials)
- **Root Cause**: ECR container image was stale—old code didn't respect `DATA_SOURCE` environment variable
- **Fix Applied**: User rebuilt and pushed container image with updated `entrypoint.py`
- **Verification**: Re-run succeeded with logs showing `Using S3TranscriptConnector`

**Issue: vLLM Spot interruption during test**

- **Symptom**: vLLM task shut down mid-test
- **Root Cause**: Expected Spot instance behavior (100% Spot configuration)
- **Resolution**: ASG auto-reprovisions; manual scale-up restored service

---

## What Was NOT Accomplished (Critical Gaps)

### Gap 1: LLM Topic Naming Not Integrated in Map Phase

**The Problem**:
The `representation` field in output contains BERTopic's default c-TF-IDF keyword concatenations:
```
"physical store, grocery, store think, people, store, physical"
```

Instead of LLM-generated summaries:
```
"Discussion of Amazon's physical retail and grocery store expansion strategy"
```

**Why This Matters**:
- ADR-006 explicitly states vLLM is for "topic naming" and "theme descriptions"
- The entire vLLM infrastructure (~$0.37/hr) provides no value if not called
- Keyword representations are not human-readable or research-quality

**What We Know**:
- `LLM_BASE_URL` environment variable IS correctly set in Batch job
- `XAIClient` exists at `cloud/src/llm/xai_client.py`
- The call to generate topic summaries is either missing or not wired in `FirmProcessor`/`BERTopicModel`

**What We Don't Know**:
- Exactly where in the code the LLM naming call should be inserted
- Whether there's existing integration code that's disabled or incomplete

### Gap 2: Reduce Phase Does Not Exist

**The Problem**:
The Step Functions workflow ends after the Map phase:
```
CheckPrefetch → CreateBatchManifest → ProcessBatches (Map) → CheckForFailures → Success
```

There is no:
- Theme aggregation step (clustering firm-level topics into cross-firm themes)
- Theme naming step (LLM-generated descriptions for themes)
- Output to `intermediate/themes/quarter=...`

**Why This Matters**:
- The core research objective is "cross-firm investment theme identification"
- Without the reduce phase, the pipeline produces disconnected firm-level topics
- ADR-005 describes a "Map-Reduce" architecture that's only half-implemented

**What Exists**:
- `cloud/terraform/stepfunctions/state_machine.json` - Map only
- `cloud/src/theme_identification/` - May contain reduce logic (not investigated)

**What Doesn't Exist** (or isn't wired):
- Reduce phase in Step Functions state machine
- Batch job or Lambda for theme aggregation
- LLM integration for theme naming

---

## Reasoning Behind Decisions Made

### Why We Used Known Firm IDs Instead of WRDS Query

The prefetch phase requires WRDS access with MFA from a fixed-IP machine. Rather than running a WRDS query to discover firm IDs, we used 10 known Capital IQ company IDs from `test_batch_integration.py`:
- Amazon (18749), Cisco (19691), Microsoft (21835), Apple (24937), Alphabet (29096)
- NVIDIA (32307), Meta (20765463), Broadcom (25016048), Tesla (27444752), Arista (33348547)

These are large-cap tech companies with guaranteed transcript coverage in 2023Q1, ensuring the smoke test wouldn't fail due to missing data.

### Why We Rebuilt the Container Instead of Debugging Further

When the Batch job used WRDS despite `DATA_SOURCE=s3`:
1. The Terraform had `DATA_SOURCE=s3` correctly configured
2. The deployed job definition (revision 2) had the correct environment variable
3. The running job received the correct environment variable

The only remaining explanation was stale container code. Rebuilding was faster and more reliable than attempting to debug the discrepancy further.

### Why We Didn't Proceed to Cross-Quarter Test

After discovering the two gaps, proceeding to cross-quarter testing would validate the same incomplete pipeline at larger scale—wasteful of compute and time. The correct decision was to pause and address the fundamental gaps.

---

## Confidence Assessment

### High Confidence

- **Infrastructure works**: Step Functions, Batch, vLLM, S3 connectors all function correctly
- **Data flow is correct**: Prefetch → S3 → Map → Parquet output path is validated
- **Schema is correct**: Output Parquet has all required fields (firm_id, permno, gvkey, topic_id, etc.)
- **Environment configuration works**: `DATA_SOURCE`, `LLM_BASE_URL`, `QUARTER`, etc. all propagate correctly

### Medium Confidence

- **LLM client works**: `XAIClient` exists and is designed for OpenAI-compatible endpoints, but we didn't verify it actually calls vLLM successfully
- **Theme identification code exists**: `cloud/src/theme_identification/` directory exists but wasn't investigated

### Low Confidence

- **Where LLM naming should be called**: Need to trace through `FirmProcessor` → `BERTopicModel` to find integration point
- **Reduce phase architecture**: Need to understand if code exists and just needs Step Functions wiring, or if it needs to be built

---

## Files Changed/Created This Session

| File | Action | Purpose |
|------|--------|---------|
| S3: `prefetch/transcripts/quarter=2023Q1/` | Created | Prefetch data for 10 firms |
| S3: `intermediate/firm-topics/quarter=2023Q1/` | Created | Smoke test output |
| ECR: `ftm-map:latest` | Rebuilt | Fixed stale container with S3 connector support |

## Files NOT Changed (Planned but Deferred)

| File | Planned Change | Status |
|------|----------------|--------|
| `docs/adr/adr_006_llm_strategy.md` | Update Spot baseline from "80/20" to "100% Spot" | Deferred |
| vLLM service | Scale down to 0 | **ACTION REQUIRED** |

---

## Cleanup Required

**IMPORTANT**: vLLM may still be scaled up and costing ~$0.37/hr.

```bash
# Scale down vLLM immediately
aws ecs update-service --cluster ftm-vllm-cluster --service ftm-vllm-service --desired-count 0
aws autoscaling set-desired-capacity --auto-scaling-group-name ftm-vllm-asg --desired-capacity 0
```

---

## Key Learnings for Future Sprints

1. **Validate end-to-end before infrastructure**: The vLLM infrastructure was deployed without verifying the code actually calls it. Always trace the full code path before deploying supporting infrastructure.

2. **Container images can silently become stale**: Even with correct Terraform and environment variables, old container images run old code. Always verify image push timestamps match recent code changes.

3. **"Pipeline succeeds" ≠ "Pipeline produces expected output"**: The smoke test returned success, but output inspection revealed the core functionality (LLM naming) wasn't being exercised.

4. **Keyword representations are a red flag**: If you see c-TF-IDF keyword concatenations instead of natural language summaries, the LLM integration isn't working.

---

## Alignment with Original Plan

| Original Plan Item | Status | Notes |
|--------------------|--------|-------|
| Phase 0: Scoped Prefetch | ✅ Complete | 10 firms for smoke test |
| Phase 1: Pre-Flight Verification | ✅ Complete | All checks passed |
| Phase 2: vLLM Service Validation | ✅ Complete | Healthy after scale-up |
| Phase 3A: Smoke Test | ⚠️ Partial | Mechanically succeeded, but revealed gaps |
| Phase 3B: Cross-Quarter Test | ⏸️ Deferred | Blocked on gap resolution |
| Phase 4: Quality Validation | ⏸️ Deferred | Output quality is poor (keywords only) |
| Phase 5: ADR-006 Verification | ⏸️ Deferred | Minor update still needed |
| Phase 6: Cleanup | ⏸️ Pending | vLLM scale-down required |

---

## Handover to Sprint 7

Sprint 7 must address:

1. **Wire LLM topic naming into map phase** (`FirmProcessor`/`BERTopicModel`)
2. **Implement reduce phase** (cross-firm theme aggregation with LLM naming)
3. **Re-run smoke test** to validate LLM integration
4. **Run cross-quarter test** (100 firms × 2 quarters)
5. **Complete ADR-006 update** (Spot baseline)
6. **Gate to full quarter run** (only after above pass)

The infrastructure is ready. The gap is in the application code integration.
