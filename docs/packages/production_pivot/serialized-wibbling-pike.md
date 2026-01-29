# Production Pivot Plan: Financial Topic Modeling

## Executive Summary

Transform the Financial Topic Modeling pipeline from a single-GPU sequential processor (~48 hours/quarter) to a distributed AWS Batch architecture capable of processing 8 quarters in parallel, achieving the target deadline of end-of-week delivery.

**Key Transitions:**
1. CSV → WRDS/Capital IQ (with PERMNO/GVKEY linking)
2. Single EC2 → AWS Batch distributed processing
3. Grok API → Self-hosted vLLM on ECS (rate limit removal)
4. PostgreSQL → S3/Parquet + Athena (cost optimization)
5. Manual → Step Functions orchestration
6. Separate sentiment → Integrated pipeline

---

## Context Summary

### Current State (Validated MVP)
- Single `g4dn.2xlarge` GPU instance processing firms sequentially
- 48+ hours per quarter, requiring manual babysitting during Spot interruptions
- Grok API for LLM naming at ~300-500 requests/minute (near rate limit)
- PostgreSQL + pgvector for storage
- CSV-based transcript ingestion
- Sentiment analysis as separate post-processing (handoff to collaborator)

### Target State
- AWS Batch with 3-5x parallelism per quarter
- Self-hosted vLLM on ECS (Qwen3-8B) for LLM naming
- WRDS/Capital IQ direct ingestion with PERMNO/GVKEY linking
- S3/Parquet storage with Athena queries
- Step Functions for multi-quarter orchestration
- Integrated sentiment analysis consuming themes directly from S3

### Prior Architectural Decisions (from `docs/ai-log/claude-2026-batch-convo.md`)
- **ECS + vLLM** over SageMaker (better throughput with vLLM's batching)
- **S3 + Parquet + Athena** over Postgres (pay-per-query, cold storage)
- **Partition by quarter only** (avoid small file problem - NOT by firm)
- **Chunk writes every ~50 firms** (memory safety + atomic progress)
- **Budget target**: ~$100/quarter
- **Keep ECS warm** during batch run (don't over-engineer autoscaling)

---

## Sprint Breakdown

### Status Update (2026-01-28)
- **Sprint 1**: Complete (ADRs/specs/diagrams approved)
- **Sprint 2**: Complete (WRDSConnector + tests + real WRDS integration)
- **Sprint 3**: Complete (AWS Batch infra + container + end-to-end Batch test)
- **Sprint 4 (re-scoped)**: Complete (WRDS prefetch + S3 connector + DATA_SOURCE switch + orchestrator)
- **Sprint 5**: Next (ECS vLLM + Step Functions)
- **Sprint 6**: Final (Integration + sentiment validation)

### Sprint 1: Specification & Design (Current Sprint)
**Objective:** Complete architectural specifications before writing any code.

**Deliverables:**
| Document | Location | Status |
|----------|----------|--------|
| Production Pivot Plan | This document | Complete |
| ADR-004: WRDS Data Source | `docs/adr/adr_004_wrds_data_source.md` | Complete |
| ADR-005: AWS Batch Parallelization | `docs/adr/adr_005_aws_batch_parallelization.md` | Complete |
| ADR-006: Self-Hosted LLM Strategy | `docs/adr/adr_006_llm_strategy.md` | Complete |
| ADR-007: S3/Parquet Storage | `docs/adr/adr_007_storage_strategy.md` | Complete |
| WRDSConnector Interface Spec | `docs/specs/wrds_connector_spec.md` | Complete |
| Sentiment-Ready Schema Spec | `docs/specs/sentiment_ready_schema_spec.md` | Complete |
| PlantUML Architecture Diagrams | `docs/diagrams/` | Complete |
| Codex Reviewer Init Prompt | `docs/packages/production_pivot/codex_reviewer_init.md` | Complete |

**Halting Point:** All ADRs and specs reviewed and approved by user + Codex auditor.

---

### Sprint 2: WRDS Data Connector
**Objective:** Replace CSV ingestion with WRDS/Capital IQ, including PERMNO/GVKEY linking for sentiment analysis.

**Deliverables:**
| File | Description |
|------|-------------|
| `cloud/src/connectors/wrds_connector.py` | New WRDSConnector implementing DataConnector interface |
| `cloud/src/connectors/permno_linker.py` | GVKEY → PERMNO linking utility |
| `cloud/src/models.py` | Add permno, gvkey, link_date to FirmTranscriptData.metadata |
| `tests/unit/test_wrds_connector.py` | Unit tests with mocked WRDS responses |
| `tests/integration/test_wrds_integration.py` | Integration tests against real WRDS (10 firms) |

**Key WRDS Query (from API research):**
```sql
WITH transcript_data AS (
    SELECT t.companyid, t.transcriptid, tc.componenttext, tc.componentorder,
           t.mostimportantdateutc::date AS earnings_call_date, tc.speakertypename
    FROM ciq.ciqtranscript t
    JOIN ciq.ciqcompany c ON t.companyid = c.companyid
    JOIN ciq.ciqtranscriptcomponent tc ON t.transcriptid = tc.transcriptid
    WHERE t.mostimportantdateutc BETWEEN %(start_date)s AND %(end_date)s
),
gvkey_link AS (
    SELECT td.*, wg.gvkey FROM transcript_data td
    LEFT JOIN ciq.wrds_gvkey wg ON td.companyid = wg.companyid
),
permno_link AS (
    SELECT g.*, ccm.lpermno AS permno, ccm.linkdt AS link_date
    FROM gvkey_link g
    LEFT JOIN crsp.ccmxpf_linktable ccm ON g.gvkey = ccm.gvkey
        AND ccm.linktype IN ('LU', 'LC') AND ccm.linkprim IN ('P', 'C')
        AND g.earnings_call_date >= ccm.linkdt
        AND g.earnings_call_date <= COALESCE(ccm.linkenddt, '9999-12-31')
)
SELECT * FROM permno_link ORDER BY companyid, transcriptid, componentorder;
```

**Validation:**
- [ ] Unit tests pass with mocked responses
- [ ] Integration test with 10 real firms succeeds
- [ ] PERMNO appears in output for all linked firms
- [ ] Data structure matches existing `TranscriptData` format

**Halting Point:** WRDS connector validated locally before proceeding to AWS Batch.

---

### Sprint 3: AWS Batch Parallelization
**Objective:** Parallelize firm-level processing using AWS Batch array jobs with Spot instances.

**Deliverables:**
| File | Description |
|------|-------------|
| `cloud/terraform/batch.tf` | Batch compute environment, job queue, job definitions |
| `cloud/terraform/ecr.tf` | ECR repository for container images |
| `cloud/containers/map/Dockerfile` | Updated map container for Batch |
| `cloud/containers/map/entrypoint.py` | Batch-compatible entrypoint |
| `cloud/src/batch/job_submitter.py` | Python module for Batch job submission |
| `tests/integration/test_batch_integration.py` | Integration test (10 firms via Batch) |

**Terraform Resources:**
```hcl
# Compute Environment (Spot)
resource "aws_batch_compute_environment" "firm_processing" {
  type = "MANAGED"
  compute_resources {
    type                = "SPOT"
    allocation_strategy = "SPOT_PRICE_CAPACITY_OPTIMIZED"
    instance_type       = ["g4dn.xlarge", "g5.xlarge"]
    max_vcpus           = 64
    min_vcpus           = 0
  }
}

# Job Definition
resource "aws_batch_job_definition" "firm_processor" {
  container_properties = {
    image = "${aws_ecr_repository.map.repository_url}:latest"
    resourceRequirements = [
      { type = "VCPU", value = "4" },
      { type = "MEMORY", value = "16384" },
      { type = "GPU", value = "1" }
    ]
  }
  retry_strategy { attempts = 3 }
}
```

**Key Design Decisions:**
- Each Batch job processes a firm subset (chunked, not individual firms)
- Jobs write Parquet to S3 (not PostgreSQL) in chunks of ~50 firms
- XAI client remains as-is; vLLM integration deferred to Sprint 5

**Validation:**
- [ ] Batch compute environment deployed
- [ ] 10 firms processed successfully via Batch
- [ ] Results match single-instance baseline
- [ ] Processing time demonstrates parallelism benefit

**Halting Point:** Batch jobs validated before adding vLLM layer.

---

### Sprint 4: WRDS Prefetch + S3 Data Source (MFA Mitigation)
**Objective:** Eliminate per-container WRDS MFA by prefetching transcripts once per quarter from a fixed-IP machine and reading from S3 in Batch.

**Deliverables:**
| File | Description |
|------|-------------|
| `cloud/src/prefetch/wrds_prefetcher.py` | Prefetch WRDS transcripts to S3 (preprocessed) |
| `cloud/src/connectors/s3_connector.py` | Read prefetched Parquet via manifest |
| `cloud/src/orchestrate/quarter_orchestrator.py` | Prefetch + Batch orchestration |
| `cloud/containers/map/entrypoint.py` | DATA_SOURCE switch (`wrds` vs `s3`) |
| `cloud/terraform/batch/job_definition.tf` | Default DATA_SOURCE=s3 |
| `tests/unit/test_wrds_prefetcher.py` | Prefetch unit tests |
| `tests/unit/test_s3_connector.py` | S3 connector unit tests |
| `tests/unit/test_quarter_orchestrator.py` | Orchestrator unit tests |
| `tests/integration/test_prefetch_integration.py` | Prefetch + S3 read integration test |

**Validation:**
- [ ] Prefetch writes manifest + chunks to S3 (single MFA approval)
- [ ] S3TranscriptConnector reads prefetched data without re-preprocessing
- [ ] Batch jobs run with DATA_SOURCE=s3 (no WRDS in Batch)

**Halting Point:** Prefetch + S3-backed Batch validated before vLLM/Step Functions.

---

### Sprint 5: ECS vLLM + Step Functions Orchestration
**Objective:** Add self-hosted LLM inference and multi-quarter orchestration.

**Deliverables:**
| File | Description |
|------|-------------|
| `cloud/terraform/ecs_vllm.tf` | ECS cluster, service, task definition for vLLM |
| `cloud/terraform/step_functions.tf` | Step Functions state machine |
| `cloud/state_machines/multi_quarter.asl.json` | ASL definition |
| `cloud/src/lambda/generate_quarters.py` | Lambda: date range → quarter list |
| `cloud/src/lambda/partition_firms.py` | Lambda: query firms, partition into batches |
| `cloud/src/llm/xai_client.py` | Update to support configurable base_url |

**XAI Client Migration (minimal change):**
```python
# Line 105-109 in xai_client.py
# Change from:
base_url=DEFAULT_BASE_URL,  # "https://api.x.ai/v1"

# To:
base_url=os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL),
```

**ECS vLLM Configuration:**
- Image: `vllm/vllm-openai:latest` with Qwen3-8B model
- Instance: g5.xlarge (Spot, 80%) + g5.xlarge (On-Demand, 20% baseline)
- Auto-scaling: Target tracking on CPU (keep warm during batch run)
- OpenAI-compatible endpoint at `http://vllm-alb.internal:8000/v1`

**Validation:**
- [ ] vLLM endpoint responds to `/v1/chat/completions`
- [ ] XAI client works with vLLM base_url
- [ ] Step Functions state machine executes for 1 quarter
- [ ] 2 quarters process end-to-end without manual intervention

**Halting Point:** Orchestration validated before integration testing.

---

### Sprint 6: Integration & Validation
**Objective:** End-to-end validation with 100 firms × 2 quarters, plus sentiment analysis integration.

**Deliverables:**
| File | Description |
|------|-------------|
| `cloud/terraform/storage.tf` | S3 bucket structure, Glue catalog, Athena workgroup |
| `scripts/run_sentiment_postprocess.py` | Sentiment post-processor reading from S3 |
| `docs/validation_report.md` | Results from 100×2 validation |
| `docs/OPERATIONS.md` | Runbook for pipeline operation |
| Updated `README.md` | Production deployment instructions |

**S3 Bucket Structure:**
```
s3://financial-topic-modeling-prod/
├── intermediate/firm-topics/quarter=2023Q1/batch_{id}_part_{n}.parquet
├── processed/
│   ├── firms/quarter=2023Q1/firms.parquet
│   ├── sentences/quarter=2023Q1/sentences.parquet
│   ├── topics/quarter=2023Q1/topics.parquet
│   └── themes/quarter=2023Q1/themes.parquet
└── sentiment-ready/quarter=2023Q1/themes_for_sentiment.parquet
```

**Validation Test Plan:**
| Test | Firms | Quarters | Target Time | Success Criteria |
|------|-------|----------|-------------|------------------|
| Smoke | 10 | 1 | < 15 min | Completes without error |
| Medium | 50 | 1 | < 1 hour | All firms processed |
| **Final** | **100** | **2** | **< 4 hours** | **All firms processed, themes + sentiment generated** |

**Halting Point:** 100×2 validation passes before 8-quarter scale run.

---

## Critical Files

### Must Modify
| File | Sprint | Change |
|------|--------|--------|
| `cloud/src/interfaces.py` | 2 | Add WRDSConnector stub to docstring |
| `cloud/src/models.py` | 2 | Add permno/gvkey/link_date to metadata |
| `cloud/src/llm/xai_client.py` | 5 | Configurable base_url for vLLM |
| `cloud/containers/map/entrypoint.py` | 4 | DATA_SOURCE switch (wrds vs s3) |
| `cloud/terraform/batch/job_definition.tf` | 4 | Default DATA_SOURCE=s3 |
| `cloud/terraform/main.tf` | 5 | Extend with ECS/Step Functions modules |

### Must Create
| File | Sprint | Purpose |
|------|--------|---------|
| `cloud/src/connectors/wrds_connector.py` | 2 | WRDS data ingestion with PERMNO linking |
| `cloud/terraform/batch/` | 3 | AWS Batch infrastructure (separate root) |
| `cloud/src/prefetch/wrds_prefetcher.py` | 4 | WRDS prefetch to S3 (MFA mitigation) |
| `cloud/src/connectors/s3_connector.py` | 4 | S3TranscriptConnector (prefetch reader) |
| `cloud/src/orchestrate/quarter_orchestrator.py` | 4 | Prefetch + Batch orchestration |
| `cloud/terraform/ecs_vllm.tf` | 5 | ECS vLLM inference layer |
| `cloud/terraform/step_functions.tf` | 5 | Step Functions orchestration |
| `cloud/containers/map/Dockerfile` | 3 | Batch-compatible container |

### Reference Only
| File | Purpose |
|------|---------|
| `docs/ai-log/claude-2026-batch-convo.md` | Prior architecture decisions |
| `sentiment_analysis/handoff_package/` | Understand expected input schema |
| `legacy/containers/map/` | Legacy container patterns (deprecated) |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| WRDS schema differs from expectations | Medium | High | Validate against WRDS docs early in Sprint 2 |
| Spot instance capacity issues | Medium | Medium | Use SPOT_PRICE_CAPACITY_OPTIMIZED, multiple instance types |
| vLLM cold start delays | Medium | Medium | Keep ECS warm during batch run, don't scale to zero |
| Rate limiting from parallel LLM calls | Low | High | vLLM removes API limits; if issues, use queue-based batching |
| PERMNO linkage gaps | Medium | Low | Skip unlinked firms entirely (per Codex clarification) |
| Timeline pressure (8 quarters by end of week) | High | High | Focus on 100×2 validation first, scale after |

---

## Cost Projection

### Per-Quarter Processing (Target: ~$100)
| Component | Estimate |
|-----------|----------|
| AWS Batch (g4dn Spot, ~4 hrs) | $25-40 |
| ECS vLLM (g5.xlarge, ~4 hrs) | $20-30 |
| S3 Storage | $5 |
| Step Functions | $1 |
| Lambda/Athena | $1 |
| **Total** | **~$50-80** |

### Infrastructure (Monthly when active)
| Component | Estimate |
|-----------|----------|
| NAT Gateway | $45 (avoid by using public subnets) |
| RDS (if kept) | $95 (can stop when not in use) |
| S3 (100GB) | $2.30 |

---

## Verification Plan

### Sprint 2 Verification
```bash
# Run unit tests
pytest tests/unit/test_wrds_connector.py -v

# Run integration test (requires WRDS credentials)
WRDS_USERNAME=xxx WRDS_PASSWORD=xxx pytest tests/integration/test_wrds_integration.py -v
```

### Sprint 3 Verification
```bash
# Deploy Batch infrastructure
cd cloud/terraform/batch && terraform apply -var="s3_bucket_name=YOUR_BUCKET"

# Submit test job
python -c "from cloud.src.batch.job_submitter import BatchJobSubmitter; ..."
```

### Sprint 4 Verification
```bash
# Prefetch integration test (requires AWS creds)
pytest tests/integration/test_prefetch_integration.py -v -m integration
```

### Sprint 6 Verification (Final E2E)
```bash
# Start Step Functions execution
aws stepfunctions start-execution \
  --state-machine-arn arn:aws:states:us-east-1:XXX:stateMachine:ftm-pipeline \
  --input '{"start_date": "2023-01-01", "end_date": "2023-06-30"}'

# Monitor progress
aws stepfunctions describe-execution --execution-arn arn:aws:states:us-east-1:XXX:execution:...

# Validate results in S3
aws s3 ls s3://financial-topic-modeling-prod/processed/themes/quarter=2023Q1/
```

---

## Confirmed Decisions

1. **WRDS Access**: Confirmed - user has `ciq_transcripts` library access
2. **Sentiment Scope**: Integration only - modify to read from S3/Parquet (minimal changes)
3. **vLLM Model**: Qwen3-8B as planned in prior architecture discussion
4. **Existing Postgres Data**: Keep separate (not migrated) - MVP data remains for reference

---

## Codex Review Clarifications (2026-01-26)

The following decisions were confirmed during Codex reviewer onboarding:

| Decision | Detail |
|----------|--------|
| **Quality tolerance** | Qwen3-8B vs Grok acceptable; sample-based comparison by re-running prior quarter |
| **Unlinked firms** | **SKIP entirely** - non-US firms without PERMNO cannot be used in event studies |
| **vLLM throughput** | Target 3,000 req/min (range: 2,000-5,000) |
| **Keep-warm strategy** | Overlap quarters: start next quarter's map while current quarter's reduce runs |
| **Sentiment contract** | Sentiment module refactored to conform to Parquet schema (not vice versa) |
| **Reduce scope** | Theme aggregation reads only `topics.parquet`, not sentences |

These clarifications have been incorporated into:
- `docs/adr/adr_004_wrds_data_source.md` - Skip unlinked firms decision
- `docs/adr/adr_006_llm_strategy.md` - Throughput target and keep-warm strategy
- `docs/adr/adr_007_storage_strategy.md` - Sentiment contract and reduce scope
- `docs/specs/sentiment_ready_schema_spec.md` - New sentiment-ready Parquet schema

---

## Next Steps (Current)

1. **Sprint 5 Kickoff**: ECS vLLM + Step Functions orchestration
2. **Sprint 6 Prep**: Define 100×2 E2E validation plan + sentiment integration checkpoints
3. **Operational Readiness**: Confirm prefetch + S3-backed Batch runbooks and cleanup procedures
