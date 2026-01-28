# Sprint 3 Bootstrap: AWS Batch Parallelization

## Initialization Prompt for Next Instance

You are continuing work on the Financial Topic Modeling production pivot. Sprint 2 (WRDS Data Connector) is complete. Your task is to implement Sprint 3: AWS Batch Parallelization.

---

## Ground Truth Documents (READ THESE FIRST)

Before writing any code, read and internalize these documents in order:

1. **Master Plan**: `docs/packages/production_pivot/serialized-wibbling-pike.md`
   - Contains full sprint breakdown, architectural decisions, budget constraints
   - Sprint 3 section (lines ~113-180) has specific deliverables

2. **Sprint 2 Summary**: `docs/packages/production_pivot/instance_summary_sprint2.md`
   - Documents what was built, why, and key learnings
   - Critical: Contains WRDS credential management details for AWS

3. **ADR-005**: `docs/adr/adr_005_aws_batch_parallelization.md`
   - Batch compute environment specs, job definitions, Spot strategy

4. **Existing Cloud Code**: `cloud/src/` directory
   - `interfaces.py` - DataConnector, TopicModel interfaces
   - `connectors/wrds_connector.py` - WRDS connector (your Batch jobs will use this)
   - `topic_models/bertopic_model.py` - BERTopic implementation
   - `pipeline.py` - Unified pipeline orchestration

5. **Prior Batch Conversations**: `docs/ai-log/claude-2026-batch-convo.md`
   - Contains validated architectural decisions (ECS+vLLM, S3+Parquet, etc.)

---

## Sprint 3 Objective

Parallelize firm-level processing using AWS Batch array jobs with Spot instances.

**Why this matters**: The MVP processes firms sequentially on a single GPU, taking 48+ hours per quarter. With ~5,000 firms per quarter and 8 quarters to process, sequential processing is untenable. AWS Batch enables:

- 3-5x parallelism per quarter
- Spot instance cost savings (~70% vs on-demand)
- Automatic retry on Spot interruptions
- S3-based coordination (no shared state)

---

## Sprint 3 Deliverables

| File                                          | Description                                           |
| --------------------------------------------- | ----------------------------------------------------- |
| `cloud/terraform/batch.tf`                    | Batch compute environment, job queue, job definitions |
| `cloud/terraform/ecr.tf`                      | ECR repository for container images                   |
| `cloud/containers/map/Dockerfile`             | Map container for Batch (GPU-enabled)                 |
| `cloud/containers/map/entrypoint.py`          | Batch-compatible entrypoint                           |
| `cloud/src/batch/job_submitter.py`            | Python module for Batch job submission                |
| `tests/integration/test_batch_integration.py` | Integration test (10 firms via Batch)                 |

---

## Critical Integration Points

### 1. WRDS Connector in Batch

The `WRDSConnector` from Sprint 2 must work in Batch containers. Key requirements:

```python
# In entrypoint.py - credentials come from Secrets Manager
from cloud.src.connectors import WRDSConnector

# _setup_wrds_auth() automatically checks AWS Secrets Manager
# for secret named "wrds-credentials" with JSON {"username":"xxx","password":"xxx"}
with WRDSConnector() as conn:
    data = conn.fetch_transcripts(firm_ids=[firm_id], start_date=..., end_date=...)
```

**Action needed**: Create the `wrds-credentials` secret in Secrets Manager (or document in Terraform).

### 2. Firm Batch Assignment

Batch jobs should process **batches of firms** (not single firms) to amortize model load and reduce cold-start overhead.
If using array jobs, `AWS_BATCH_JOB_ARRAY_INDEX` should map to a **pre-partitioned batch** of firm IDs:

1. Partition firm IDs into batches (e.g., ~1000 firms per job)
2. Pass the batch mapping to jobs (e.g., via S3 manifest or job parameters)
3. Each job processes its assigned firm batch sequentially
4. Write chunked Parquet outputs to S3

### 3. SpaCy Model in Container

The container needs `en_core_web_sm`:

```dockerfile
RUN python -m spacy download en_core_web_sm
```

### 4. Output Format

Per ADR-007, output goes to S3 as Parquet **partitioned by quarter only**, with chunked writes to avoid memory spikes (e.g., every ~50 firms per file):

```
s3://bucket/intermediate/firm-topics/quarter=2023Q1/batch_000_part_00.parquet
```

---

## Guiding Principles (IMPERATIVE)

### Code Quality Standards

1. **Simplicity over cleverness** - Write code that's easy to understand at a glance. If a solution requires extensive comments to explain, it's too complex.

2. **Boring technology** - Use well-established patterns. AWS Batch, Terraform, Docker are boring and reliable.

3. **Test-driven development** - Write tests first. Define the contract, then implement.

4. **Spec-driven implementation** - The ADRs and specs are the source of truth. If you need to deviate, update the docs first.

5. **Minimal viable changes** - Don't refactor existing code unless necessary. Don't add features beyond the sprint scope.

### What NOT To Do

- Do NOT add fancy abstractions "for future flexibility"
- Do NOT over-engineer error handling for scenarios that can't happen
- Do NOT create new files unless explicitly required
- Do NOT modify existing working code without clear necessity
- Do NOT skip writing tests to "save time"

### Documentation

- Update ADRs if decisions change
- Keep the instance summary pattern - document the "why" not just the "what"
- If you discover something unexpected (like the WRDS schema issue in Sprint 2), document it

---

## Validation Criteria

Before marking Sprint 3 complete:

- [ ] Terraform plans successfully (`terraform plan` shows expected resources)
- [ ] Container builds and pushes to ECR
- [ ] Batch job submission works programmatically
- [ ] Integration test processes 10 firms via Batch array job
- [ ] Output appears in S3 with correct partitioning
- [ ] Spot interruption retry is configured
- [ ] Codex review passes

---

## Commands to Get Started

```bash
# Activate virtual environment
source venv/bin/activate

# Read the ground truth documents
cat docs/packages/production_pivot/serialized-wibbling-pike.md
cat docs/packages/production_pivot/instance_summary_sprint2.md
cat docs/adr/adr_005_aws_batch_parallelization.md

# Review existing cloud code
ls -la cloud/src/
cat cloud/src/connectors/wrds_connector.py

# Run existing tests to verify environment
pytest tests/unit/ -v --ignore=tests/integration/
```

---

## Budget Constraint

Target: ~$100/quarter for processing

This constrains:

- Spot instances only (no on-demand)
- g4dn.xlarge preferred over larger instances
- Minimize data transfer costs (process in same region as WRDS)

---

## Questions to Clarify Before Starting

If any of these are unclear, investigate or ask before implementing:

1. Does the user have AWS credentials configured locally?
2. Is there an existing Terraform state file, or starting fresh?
3. What AWS region should be used?
4. Should the ECR repository be public or private?
5. What's the S3 bucket name for outputs?

---

## Halting Point

Sprint 3 is complete when:

1. All deliverables exist and pass tests
2. Integration test with 10 real firms succeeds via Batch
3. Codex review approves
4. Instance summary document created for Sprint 4 handoff

<Note from User>
It is mandatory that during the planning and bootstrap stage you do your due dillegence both individually but also by leveraging context-synthesizer, api-docs-synthesizer and plan subagents to methodologically manage your context by offloading exploration-heavy tasks as you see fit. We are building production software, give this your all and leave nothing on the table. Proceed as such.
</Note from User>
