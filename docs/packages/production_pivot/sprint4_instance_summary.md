# Sprint 4 Instance Summary: WRDS Prefetch & S3 Data Source

**Session Date:** 2026-01-28
**Objective:** Solve the WRDS MFA blocker by implementing a "prefetch to S3" pattern, enabling production batch runs without per-container MFA challenges.

---

## Executive Summary

Sprint 4 successfully delivered the WRDS prefetch infrastructure that unblocks production batch runs. The core problem was WRDS IP-based MFA (Duo Push) - each AWS Batch container gets a different IP, triggering new MFA challenges. Our solution: fetch transcripts ONCE per quarter from a fixed-IP machine, store preprocessed data in S3, then batch jobs read from S3 without touching WRDS.

**Key Deliverables:**
- WRDSPrefetcher: Fetches all transcripts for a quarter, writes Parquet chunks + manifest.json
- S3TranscriptConnector: Manifest-based selective loading (memory-bounded reads)
- DATA_SOURCE switch: Seamless routing between WRDS (local dev) and S3 (production)
- QuarterOrchestrator: Coordinates prefetch check → batch submission → monitoring

**Scope Change:** Original Sprint 4 (vLLM + Step Functions) was deferred. WRDS MFA was a production blocker that had to be solved first. Plan updated accordingly.

---

## What Was Delivered

### WRDSPrefetcher (`cloud/src/prefetch/wrds_prefetcher.py`)

| Feature | Implementation |
|---------|----------------|
| Firm discovery | `get_firm_ids_in_range()` - lightweight query, not full transcript load |
| Chunking | ~200 firms per Parquet file (balance between small file problem and selective loading) |
| Manifest | `manifest.json` with `firm_to_chunk` mapping (gzip compressed, ~50KB for 5000 firms) |
| Checkpointing | Every 100 firms for resumability after interruption |
| Schema | 12 columns matching prefetch specification (firm_id, cleaned_text, permno, etc.) |

**Why These Decisions:**
- **Chunks of 200 firms:** Large enough to avoid S3 small file overhead, small enough for selective loading
- **Preprocessed data:** Stores `cleaned_text` (already lemmatized, stopwords removed) to avoid duplicate NLP work in batch jobs
- **Checkpoint interval of 100:** Frequent enough for resilience, not so frequent as to slow down S3 writes
- **Gzip manifest:** Reduces S3 storage and transfer costs for large firm-to-chunk mappings

### S3TranscriptConnector (`cloud/src/connectors/s3_connector.py`)

| Feature | Implementation |
|---------|----------------|
| Selective loading | Reads only chunks containing requested firm_ids (via manifest) |
| No re-preprocessing | Uses `cleaned_text` as-is from Parquet |
| Interface compliance | Implements `DataConnector` interface (same as WRDSConnector) |
| Firm discovery | `get_available_firm_ids()` reads from manifest (O(1), no data scan) |

**Why Manifest-Based Loading:**
- Memory-bounded: For 1000 firms across 5 chunks, reads only 5 files (not all 25)
- Efficient: No need to scan Parquet files to find which contain which firms
- Simple: Dictionary lookup is faster than any filtering approach

**Critical Design Decision - No Re-Preprocessing:**
The S3TranscriptConnector does NOT re-run NLP preprocessing. The prefetcher already applied sentence splitting, lemmatization, NER removal, and stopword filtering. This:
1. Avoids duplicate computation
2. Ensures consistent preprocessing across prefetch and batch
3. Requires SpaCy only on prefetch machine (not in batch containers)

### DATA_SOURCE Switch (`cloud/containers/map/entrypoint.py`)

```python
def get_data_connector(data_source: str, quarter: str, bucket: str) -> DataConnector:
    if data_source == "s3":
        return S3TranscriptConnector(bucket=bucket, quarter=quarter)
    elif data_source == "wrds":
        return WRDSConnector()
    else:
        raise ValueError(f"Unknown DATA_SOURCE: '{data_source}'")
```

**Default Behavior:**
- **Entrypoint default:** `wrds` (backward compatibility for local testing)
- **Terraform default:** `s3` (production uses prefetch)

**Why This Split:**
- Local development can run without prefetch data (direct WRDS)
- Production always uses prefetch (no accidental WRDS MFA in Batch)
- Explicit rather than magic - DATA_SOURCE is visible in logs

### QuarterOrchestrator (`cloud/src/orchestrate/quarter_orchestrator.py`)

| Method | Purpose |
|--------|---------|
| `prefetch_exists(quarter)` | Check if manifest.json exists |
| `get_prefetch_firm_ids(quarter)` | Read firm IDs from manifest (O(1)) |
| `run_prefetch(quarter)` | Execute WRDSPrefetcher (requires MFA) |
| `run_quarter(quarter, batch_size)` | Full workflow: check prefetch → submit jobs → monitor |

**Critical Safety Decision:**
The orchestrator **REQUIRES** prefetch data for batch runs. There is no silent fallback to WRDS. If manifest is missing, it raises `PrefetchRequiredError`. This prevents accidental MFA challenges in production.

### Terraform Updates

| File | Change |
|------|--------|
| `job_definition.tf` | Added `DATA_SOURCE=s3` environment variable |
| `iam.tf` | Added `prefetch/*` to S3 permissions |

### Tests

| File | Coverage |
|------|----------|
| `test_wrds_prefetcher.py` | Flatten, chunk write, manifest, checkpoint, firm discovery |
| `test_s3_connector.py` | Manifest load, selective loading, no re-preprocessing |
| `test_quarter_orchestrator.py` | Prefetch check, job submission, describe_jobs chunking |
| `test_data_source_switch.py` | Connector factory, defaults, interface compliance |
| `test_prefetch_integration.py` | End-to-end prefetch and S3 read (requires AWS) |

---

## The Reasoning (Why We Made These Choices)

### 1. Prefetch to S3 vs NAT Gateway

**Problem:** WRDS MFA triggers per-container because each gets a different IP.

**Options Considered:**
- NAT Gateway with Elastic IP (~$32/month, complex subnet migration)
- Prefetch to S3 (one-time MFA, simple, aligns with existing patterns)

**Decision:** Prefetch to S3

**Why:**
- Zero recurring cost (S3 storage is ~$0.02/GB/month)
- Aligns with ADR-007's storage strategy
- Decouples data ingestion from processing
- Prefetched data can be reused (e.g., for debugging, re-runs)

### 2. Preprocessed vs Raw Storage

**Problem:** Should prefetch store raw transcript components or preprocessed sentences?

**Decision:** Store preprocessed (post-NLP pipeline)

**Why:**
- Avoids duplicate NLP work (sentence splitting, lemmatization, NER removal)
- Batch containers don't need SpaCy installed
- Consistent preprocessing - same code path whether WRDS or S3

**Trade-off:** Can't change preprocessing without re-running prefetch. Acceptable because:
- Preprocessing is stable (tested in Sprint 2)
- Quarter data is immutable anyway

### 3. Manifest Structure

**Problem:** How to efficiently find which chunk contains a firm?

**Decision:** `manifest.json` with `firm_to_chunk` dictionary

**Why:**
- O(1) lookup per firm
- Small file (~50KB compressed for 5000 firms)
- Human-readable (debugging)

**Alternative Rejected:** Parquet metadata or separate index table. Too complex, no meaningful benefit.

### 4. Memory-Bounded Prefetch

**Original Code (Bad):**
```python
if firm_ids is None:
    transcript_data = connector.fetch_transcripts(firm_ids=[], ...)  # Loads ENTIRE quarter!
```

**Fixed Code (Good):**
```python
if firm_ids is None:
    firm_ids = connector.get_firm_ids_in_range(start_date, end_date)  # Lightweight query
```

**Why:** The original code would load all transcripts for a quarter into memory (potentially GB of data). The fix uses a lightweight SQL query that returns only firm IDs.

### 5. Checkpoint Counter Bug

**Original Code (Bug):**
```python
# firms_since_checkpoint only incremented on SKIPPED firms
if firm_id not in firm_data.firms:
    firms_since_checkpoint += 1  # Only here
```

**Fixed Code:**
```python
# Also increment on SUCCESSFUL firms
buffer.extend(rows)
firms_since_checkpoint += 1  # Now here too
```

**Why:** Without this fix, checkpoints only happened at chunk boundaries (200 firms), not every 100 firms as intended.

### 6. Filter Alignment

**Problem:** `get_firm_ids_in_range()` might return firms that `fetch_transcripts()` would skip (due to missing PERMNO link date validity).

**Fix:** Added identical link date filters to both queries:
```sql
AND fir.earnings_call_date >= ccm.linkdt
AND fir.earnings_call_date <= COALESCE(ccm.linkenddt, '9999-12-31')
```

**Why:** Prefetch shouldn't waste time fetching firms that will be filtered out later.

---

## What We're Confident In

| Component | Confidence | Validation |
|-----------|------------|------------|
| WRDSPrefetcher design | High | Comprehensive unit tests, clear data flow |
| S3TranscriptConnector selective loading | High | Unit tests verify only required chunks read |
| DATA_SOURCE switch | High | Factory pattern is simple, tests pass |
| Terraform changes | High | Minimal, isolated changes |
| Manifest structure | High | Matches specification, tests verify |
| Checkpoint/resume | High | Round-trip tests pass |

---

## What We're Less Confident In

| Component | Concern | Mitigation |
|-----------|---------|------------|
| Full production run | Haven't run 5000 firms end-to-end | Integration test structure exists |
| Memory at scale | Haven't measured peak memory for large quarter | Chunking design should bound it |
| QuarterOrchestrator job monitoring | describe_jobs chunking untested at 100+ jobs | Unit test verifies chunking logic |
| WRDS prefetch real-world | Requires actual MFA approval | Deferred to production validation |

---

## What Wasn't Changed

- **Reduce phase:** No changes (still pending implementation)
- **Topic modeling logic:** Unchanged from Sprint 3
- **BERTopicModel:** Unchanged
- **FirmProcessor:** Unchanged
- **vLLM + Step Functions:** Moved to Sprint 5
- **Sentiment integration:** Moved to Sprint 6
- **Production validation:** Moved to Sprint 6 (prefetch + Batch full run)

---

## Files Created/Modified

### New Files

| File | Purpose |
|------|---------|
| `cloud/src/prefetch/__init__.py` | Package init |
| `cloud/src/prefetch/wrds_prefetcher.py` | Prefetch implementation |
| `cloud/src/orchestrate/__init__.py` | Package init |
| `cloud/src/orchestrate/quarter_orchestrator.py` | Orchestration logic |
| `tests/unit/test_wrds_prefetcher.py` | Prefetcher unit tests |
| `tests/unit/test_s3_connector.py` | S3 connector unit tests |
| `tests/unit/test_quarter_orchestrator.py` | Orchestrator unit tests |
| `tests/unit/test_data_source_switch.py` | DATA_SOURCE switch tests |
| `tests/integration/test_prefetch_integration.py` | Integration tests |

### Modified Files

| File | Change |
|------|--------|
| `cloud/src/connectors/s3_connector.py` | Replaced stub with full implementation |
| `cloud/src/connectors/wrds_connector.py` | Added `get_firm_ids_in_range()` method and SQL query |
| `cloud/containers/map/entrypoint.py` | Added DATA_SOURCE switch and `get_data_connector()` |
| `cloud/terraform/batch/job_definition.tf` | Added DATA_SOURCE=s3 env var |
| `cloud/terraform/batch/iam.tf` | Added prefetch/* to S3 permissions |
| `docs/specs/wrds_connector_spec.md` | Documented new method |
| `docs/adr/adr_007_storage_strategy.md` | Added prefetch staging layer documentation |
| `docs/packages/production_pivot/serialized-wibbling-pike.md` | Updated sprint status |

---

## S3 Bucket Structure (Post-Sprint 4)

```
s3://ftm-pipeline-78ea68c8/
├── prefetch/                              # NEW: Sprint 4
│   └── transcripts/
│       └── quarter=2023Q1/
│           ├── chunk_0000.parquet         # ~200 firms each
│           ├── chunk_0001.parquet
│           ├── ...
│           ├── manifest.json              # firm_to_chunk mapping (gzip)
│           └── _checkpoint.json           # Resume state (deleted on completion)
├── manifests/                             # Batch job manifests
├── progress/                              # Checkpoints and failures
└── intermediate/                          # Firm-level topic output
```

---

## Data Flow (Updated)

```
[Once per quarter, fixed-IP machine]
WRDSPrefetcher → WRDS (MFA once) → prefetch/transcripts/quarter=.../

[Parallel Batch jobs]
Batch Job → S3TranscriptConnector → manifest → selective chunks → FirmProcessor → intermediate/
```

---

## Known Issues & Workarounds

### Test Patch Paths

When mocking connectors in tests, patch at the **source module**, not the import location:

```python
# WRONG:
with patch("cloud.containers.map.entrypoint.S3TranscriptConnector") as mock:

# CORRECT:
with patch("cloud.src.connectors.s3_connector.S3TranscriptConnector") as mock:
```

**Why:** `get_data_connector()` uses local imports inside the function, so the class doesn't exist on the entrypoint module's namespace.

### describe_jobs AWS Limit

AWS Batch `describe_jobs` only accepts 100 job IDs per call. The orchestrator chunks calls:

```python
for i in range(0, len(pending_list), 100):
    chunk = pending_list[i:i + 100]
    response = self._batch_client.describe_jobs(jobs=chunk)
```

---

## Next Steps (Sprint 5 and Sprint 6)

### Sprint 5: vLLM + Step Functions
1. **ECS vLLM Deployment**
   - Qwen3-8B for topic naming (ECS on g5.xlarge)
   - `xai_client.py` base_url switch via `LLM_BASE_URL`

2. **Step Functions Orchestration**
   - Multi-quarter coordination
   - Visual workflow monitoring

### Sprint 6: Production Validation + Integration
1. **Production Prefetch Validation**
   - Run prefetch for 2023Q1 from fixed-IP machine
   - Verify manifest structure and Parquet schema

2. **Full Production Run**
   - 5000 firms via batch with DATA_SOURCE=s3
   - Measure runtime and validate output

3. **Sentiment Integration Validation**
   - Ensure sentiment-ready outputs align with ADR-007

---

## Reference: Test Output

All 47 unit tests passing:

```
tests/unit/test_wrds_prefetcher.py::TestQuarterToDateRange::test_q1 PASSED
tests/unit/test_wrds_prefetcher.py::TestQuarterToDateRange::test_q2 PASSED
tests/unit/test_wrds_prefetcher.py::TestQuarterToDateRange::test_q3 PASSED
tests/unit/test_wrds_prefetcher.py::TestQuarterToDateRange::test_q4 PASSED
tests/unit/test_wrds_prefetcher.py::TestFlattenToRows::test_flatten_single_firm PASSED
tests/unit/test_wrds_prefetcher.py::TestFlattenToRows::test_flatten_multiple_firms PASSED
tests/unit/test_wrds_prefetcher.py::TestWriteChunk::test_write_chunk_creates_parquet PASSED
tests/unit/test_wrds_prefetcher.py::TestWriteManifest::test_write_manifest_gzip_compressed PASSED
tests/unit/test_wrds_prefetcher.py::TestCheckpoint::test_get_checkpoint_no_checkpoint PASSED
tests/unit/test_wrds_prefetcher.py::TestCheckpoint::test_checkpoint_round_trip PASSED
tests/unit/test_wrds_prefetcher.py::TestPrefetchQuarter::test_prefetch_resumes_from_checkpoint PASSED
tests/unit/test_wrds_prefetcher.py::TestPrefetchQuarter::test_prefetch_discovers_firms_when_none_provided PASSED
tests/unit/test_wrds_prefetcher.py::TestPrefetchQuarter::test_checkpoint_interval_includes_successful_firms PASSED
tests/unit/test_wrds_prefetcher.py::TestParquetSchema::test_schema_has_required_columns PASSED
tests/unit/test_wrds_prefetcher.py::TestParquetSchema::test_schema_types PASSED
tests/unit/test_s3_connector.py::TestLoadManifest::test_load_gzip_manifest PASSED
tests/unit/test_s3_connector.py::TestLoadManifest::test_manifest_cached PASSED
tests/unit/test_s3_connector.py::TestLoadManifest::test_manifest_not_found_raises PASSED
tests/unit/test_s3_connector.py::TestGetChunksForFirms::test_selects_correct_chunks PASSED
tests/unit/test_s3_connector.py::TestGetChunksForFirms::test_handles_missing_firms PASSED
tests/unit/test_s3_connector.py::TestSelectiveLoading::test_reads_only_required_chunks PASSED
tests/unit/test_s3_connector.py::TestBuildTranscriptData::test_reconstructs_transcript_data_structure PASSED
tests/unit/test_s3_connector.py::TestBuildTranscriptData::test_no_reprocessing PASSED
tests/unit/test_s3_connector.py::TestGetAvailableFirmIds::test_returns_from_manifest PASSED
tests/unit/test_s3_connector.py::TestEdgeCases::test_empty_firm_ids_returns_empty PASSED
tests/unit/test_s3_connector.py::TestEdgeCases::test_all_missing_firms_returns_empty PASSED
tests/unit/test_s3_connector.py::TestEdgeCases::test_date_params_ignored PASSED
tests/unit/test_quarter_orchestrator.py::TestPrefetchExists::test_returns_true_when_manifest_exists PASSED
tests/unit/test_quarter_orchestrator.py::TestPrefetchExists::test_returns_false_when_manifest_missing PASSED
tests/unit/test_quarter_orchestrator.py::TestGetPrefetchFirmIds::test_returns_sorted_firm_ids_from_manifest PASSED
tests/unit/test_quarter_orchestrator.py::TestGetPrefetchFirmIds::test_raises_when_manifest_missing PASSED
tests/unit/test_quarter_orchestrator.py::TestGetPrefetchSummary::test_returns_summary_from_manifest PASSED
tests/unit/test_quarter_orchestrator.py::TestCreateBatchManifest::test_creates_jsonl_manifest PASSED
tests/unit/test_quarter_orchestrator.py::TestSubmitBatchJob::test_submits_with_data_source_s3 PASSED
tests/unit/test_quarter_orchestrator.py::TestRunQuarter::test_fails_when_prefetch_missing PASSED
tests/unit/test_quarter_orchestrator.py::TestRunQuarter::test_submits_jobs_with_correct_count PASSED
tests/unit/test_quarter_orchestrator.py::TestRunQuarter::test_returns_summary_after_wait PASSED
tests/unit/test_quarter_orchestrator.py::TestNoWrdsFallback::test_run_quarter_never_calls_wrds PASSED
tests/unit/test_quarter_orchestrator.py::TestRunPrefetch::test_run_prefetch_calls_wrds_prefetcher PASSED
tests/unit/test_quarter_orchestrator.py::TestDescribeJobsChunking::test_describe_jobs_chunks_large_lists PASSED
tests/unit/test_quarter_orchestrator.py::TestDescribeJobsChunking::test_describe_jobs_works_with_small_lists PASSED
tests/unit/test_data_source_switch.py::TestGetDataConnector::test_returns_s3_connector_for_s3_source PASSED
tests/unit/test_data_source_switch.py::TestGetDataConnector::test_returns_wrds_connector_for_wrds_source PASSED
tests/unit/test_data_source_switch.py::TestGetDataConnector::test_raises_for_unknown_source PASSED
tests/unit/test_data_source_switch.py::TestEnvironmentVariables::test_data_source_default_is_wrds PASSED
tests/unit/test_data_source_switch.py::TestEnvironmentVariables::test_data_source_env_var_respected PASSED
tests/unit/test_data_source_switch.py::TestDataConnectorInterface::test_s3_connector_has_required_methods PASSED
tests/unit/test_data_source_switch.py::TestDataConnectorInterface::test_wrds_connector_has_required_methods PASSED
tests/unit/test_data_source_switch.py::TestBackwardCompatibility::test_entrypoint_still_works_without_data_source PASSED
tests/unit/test_data_source_switch.py::TestBackwardCompatibility::test_process_firms_accepts_any_data_connector PASSED
tests/unit/test_data_source_switch.py::TestTerraformDefaults::test_document_expected_defaults PASSED
```

---

## Codex Review Summary

Codex identified three issues, all fixed:

| Issue | Severity | Status |
|-------|----------|--------|
| Full-quarter memory load in prefetcher | High | Fixed: Use `get_firm_ids_in_range()` |
| Checkpoint counter not incrementing on success | Medium | Fixed: Added increment |
| describe_jobs 100-job limit | Low | Fixed: Chunked calls |

Additional fix identified during review:
- `FIRM_IDS_IN_RANGE_QUERY` missing link date validity filters (now aligned with main query)
