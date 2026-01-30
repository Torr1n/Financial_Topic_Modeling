# Sprint 7: LLM Integration & Reduce Phase Implementation

## Why This Sprint Exists

Sprint 6 validated that the vLLM + Step Functions infrastructure **works mechanically**—but revealed that it **produces no value**. The smoke test succeeded, yet:

1. **Topic naming uses keywords, not LLM summaries**: The `representation` field contains BERTopic's c-TF-IDF output like `"physical store, grocery, store think, people, store, physical"` instead of human-readable summaries like `"Discussion of Amazon's physical retail and grocery store expansion strategy"`.

2. **The reduce phase doesn't exist**: The Step Functions workflow ends after the map phase. There is no cross-firm theme aggregation—the core research objective.

**Bottom line**: We deployed a $0.37/hr GPU inference service that isn't being called. The pipeline produces disconnected firm-level keyword lists, not cross-firm investment themes with LLM-generated descriptions.

Sprint 7 closes these gaps.

---

## Guiding Principles (Non-Negotiable)

### Simplicity Over Cleverness
> "The best engineers write code my mom could read."

- If a solution requires explanation, simplify it
- Prefer boring, proven patterns over elegant abstractions
- One clear code path beats three configurable options

### Trace Before You Build
Before writing any code, trace the full execution path:
- Where does data enter?
- What transforms it?
- Where does it exit?
- Where should the LLM call be inserted?

### Validate Incrementally
Don't implement both gaps at once. Wire map-phase LLM naming first, verify it works, then implement the reduce phase.

### Output Inspection Over Status Codes
A pipeline that returns "SUCCESS" but produces keyword concatenations is **broken**. Always inspect actual output, not just exit codes.

---

## Objective

Complete LLM integration in both pipeline phases:

1. **Map Phase**: Wire `XAIClient` into `FirmProcessor`/`BERTopicModel` to generate topic summaries
2. **Reduce Phase**: Implement cross-firm theme aggregation with LLM-generated theme descriptions
3. **Validation**: Re-run smoke test and cross-quarter test to verify end-to-end LLM usage

---

## Context: What Sprint 6 Established

### Infrastructure (Working)
- vLLM on ECS: Qwen3-8B, scales up/down, serves `/v1/chat/completions`
- Step Functions: Orchestrates prefetch check → batch manifest → map phase
- Batch jobs: GPU-enabled, `DATA_SOURCE=s3`, `LLM_BASE_URL` correctly set
- S3 connectors: Prefetch data flows correctly through map phase
- Output schema: Parquet with firm_id, permno, gvkey, topic_id, representation, keywords

### Application Code (Gap)
- `XAIClient` exists at `cloud/src/llm/xai_client.py` but isn't called during map phase
- `FirmProcessor` at `cloud/src/firm_processor.py` processes firms but uses keyword representations
- `BERTopicModel` at `cloud/src/topic_models/bertopic_model.py` generates topics without LLM naming
- `cloud/src/theme_identification/` exists but isn't wired into Step Functions

### Prefetch Data (Available)
- 10 firms prefetched for 2023Q1 (Amazon, Apple, Microsoft, NVIDIA, Tesla, etc.)
- Manifest: `s3://ftm-pipeline-78ea68c8/prefetch/transcripts/quarter=2023Q1/manifest.json`

---

## Phase 1: Wire LLM Topic Naming into Map Phase

### 1.1 Investigate Current Code Path

**Goal**: Understand exactly where topic representations are generated and where LLM naming should be inserted.

**Files to examine**:
- `cloud/src/firm_processor.py` - Entry point for firm processing
- `cloud/src/topic_models/bertopic_model.py` - BERTopic wrapper
- `cloud/src/llm/xai_client.py` - LLM client (should already support vLLM via `LLM_BASE_URL`)
- `cloud/containers/map/entrypoint.py` - Batch job entrypoint (already sets up LLM_BASE_URL)

**Questions to answer**:
1. Where does the `representation` field get populated?
2. Is there existing code that calls `XAIClient` for topic naming that's disabled/incomplete?
3. What prompt template should be used for topic naming?

### 1.2 Implement LLM Topic Naming

**Expected change**: After BERTopic generates topics with keyword representations, call `XAIClient` to generate a human-readable summary for each topic.

**Design considerations**:
- Batch LLM calls if possible (multiple topics per request)
- **Bound LLM concurrency** to avoid swamping vLLM during batch processing
- Handle LLM failures gracefully (fall back to keyword representation)
- Log LLM usage for cost tracking

**Output change**: The `representation` field should contain LLM-generated summaries:
```
Before: "physical store, grocery, store think, people, store, physical"
After:  "Discussion of Amazon's physical retail and grocery store expansion strategy"
```

### 1.3 Validate Map Phase LLM Integration

**Test procedure**:
1. Ensure vLLM is scaled up and healthy
2. Run smoke test: `aws stepfunctions start-execution --input '{"quarter": "2023Q1", ...}'`
3. Watch vLLM logs for `POST /v1/chat/completions HTTP/1.1" 200`
4. Inspect output Parquet—representations should be natural language summaries
5. Scale down vLLM when done

---

## Phase 2: Implement Reduce Phase (Cross-Firm Theme Aggregation)

### 2.1 Understand Reduce Phase Architecture

**Goal**: Aggregate firm-level topics into cross-firm themes.

**Input**: `s3://ftm-pipeline-78ea68c8/intermediate/firm-topics/quarter=2023Q1/*.parquet`
**Output**: `s3://ftm-pipeline-78ea68c8/intermediate/themes/quarter=2023Q1/themes.parquet`

**Process**:
1. Load all firm-level topics for the quarter
2. Cluster topics across firms (by embedding similarity or keyword overlap)
3. Generate theme descriptions via LLM
4. Write theme output with firm-topic mappings

### 2.2 Investigate Existing Code

**Files to examine**:
- `cloud/src/theme_identification/` - May contain existing reduce logic
- `Local_BERTopic_MVP/src/theme_identification/` - Reference implementation

**Questions to answer**:
1. Does theme aggregation code exist?
2. What clustering method is used (re-embedding, keyword matching, etc.)?
3. How should themes be stored (schema)?

### 2.3 Wire Reduce Phase into Step Functions

**Current workflow**:
```
CheckPrefetch → CreateBatchManifest → ProcessBatches (Map) → CheckForFailures → Success
```

**Target workflow**:
```
CheckPrefetch → CreateBatchManifest → ProcessBatches (Map) → CheckForFailures →
AggregateThemes (Reduce) → CheckThemeResults → Success
```

**Implementation options**:
- **Lambda**: If reduce phase is fast (<15 min, <10GB memory)
- **Batch job**: If reduce phase needs GPU or longer runtime

### 2.4 Validate Reduce Phase

**Test procedure**:
1. Run smoke test with full pipeline (map + reduce)
2. Verify theme output exists in S3
3. Inspect theme descriptions—should be LLM-generated
4. Verify theme-to-firm-topic mappings are correct

---

## Phase 3: Integration Testing (Deferred from Sprint 6)

### 3.1 Smoke Test (10 firms, 1 batch)

**Prerequisites**:
- Map phase LLM naming working
- Reduce phase implemented

**Success criteria**:
- [ ] Step Functions completes successfully
- [ ] Batch logs show `LLM_BASE_URL=http://internal-ftm-vllm-alb-...`
- [ ] vLLM logs show `POST /v1/chat/completions HTTP/1.1" 200` requests
- [ ] Firm topic `representation` fields contain LLM summaries (not keywords)
- [ ] Theme output exists with LLM-generated descriptions

### 3.2 Cross-Quarter Test (100 firms × 2 quarters)

**Prerequisites**:
- Smoke test passed
- Prefetch data created for 100 firms in 2023Q1 and 2023Q2

**Success criteria**:
- [ ] Both quarters complete successfully
- [ ] LLM naming works at scale (no rate limit issues)
- [ ] Themes are consistent across quarters

---

## Phase 4: Cleanup and Documentation

### 4.1 ADR-006 Verification

Verify ADR-006 matches deployed configuration; update only the Spot baseline delta if still present:
- From: `"80% Spot (g5.xlarge) / 20% On-Demand (baseline)"`
- To: `"100% Spot (g5.xlarge) - cost optimization for batch workloads"`

### 4.2 Scale Down vLLM

**IMPORTANT**: vLLM may still be running from Sprint 6. Scale down immediately if not testing:

```bash
aws ecs update-service --cluster ftm-vllm-cluster --service ftm-vllm-service --desired-count 0
aws autoscaling set-desired-capacity --auto-scaling-group-name ftm-vllm-asg --desired-capacity 0
```

---

## Critical Files Reference

| Purpose | Path |
|---------|------|
| Firm processor | `cloud/src/firm_processor.py` |
| BERTopic model | `cloud/src/topic_models/bertopic_model.py` |
| LLM client | `cloud/src/llm/xai_client.py` |
| Theme identification | `cloud/src/theme_identification/` |
| Batch entrypoint | `cloud/containers/map/entrypoint.py` |
| Step Functions state machine | `cloud/terraform/stepfunctions/state_machine.json` |
| ADR-006 (LLM strategy) | `docs/adr/adr_006_llm_strategy.md` |
| Sprint 6 summary | `docs/packages/production_pivot/sprint6_instance_summary.md` |

---

## Definition of Done

Sprint 7 is complete when:

1. **Map phase produces LLM summaries**: Topic `representation` fields contain natural language descriptions, not keyword concatenations
2. **Reduce phase exists and runs**: Step Functions includes theme aggregation step
3. **Themes have LLM descriptions**: Cross-firm themes have human-readable descriptions
4. **Smoke test passes with LLM verification**: Output inspection confirms LLM usage
5. **Cross-quarter test passes**: 100 firms × 2 quarters completes successfully
6. **ADR-006 updated**: Spot baseline corrected
7. **vLLM scaled down**: Cost control verified

---

## Estimated Effort

| Phase | Scope | Estimate |
|-------|-------|----------|
| Phase 1: Map LLM naming | Code investigation + integration | Medium |
| Phase 2: Reduce phase | May be significant if code doesn't exist | Medium-High |
| Phase 3: Testing | Smoke + cross-quarter | Low (execution time) |
| Phase 4: Cleanup | ADR update, scale down | Low |

**Key risk**: If reduce phase code doesn't exist, this sprint scope expands significantly.

---

## Success Metrics

| Metric | Sprint 6 | Sprint 7 Target |
|--------|----------|-----------------|
| Topic representations | Keywords | LLM summaries |
| vLLM requests during smoke test | 0 | >0 (100+ expected) |
| Cross-firm themes generated | 0 | >0 |
| Theme descriptions | N/A | LLM-generated |
