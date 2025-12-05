# SubAgent Strategy: Architecture Pivot

**Mission:** architecture-pivot-gpu-postgres
**Strategy Version:** 1.0

---

## Strategic Overview

This document provides tactical guidance for executing the architecture pivot. Each phase maps to specific tool invocations and validation checkpoints.

---

<Objective id="database-layer">
**Phase 1: Database Layer Design**

Design and implement PostgreSQL schema with pgvector support, establishing the foundation for all subsequent work.

<Phase id="1.1" name="Schema Research">
**Goal:** Understand pgvector and SQLAlchemy patterns before implementation.

<Invocation tool="WebSearch">
Research current pgvector SQLAlchemy integration patterns (2024-2025)
- Query: "pgvector sqlalchemy vector column type 2024"
- Query: "postgresql hnsw index bulk insert performance"
</Invocation>

<Invocation tool="Read">
Review existing data models to understand what we're replacing:
- `cloud/src/models.py` - Current data structures
- `cloud/src/dynamodb_utils.py` - Current storage patterns
</Invocation>

**Validation:** Document pgvector column syntax, index creation timing, and bulk insert patterns.
</Phase>

<Phase id="1.2" name="Schema Implementation">
**Goal:** Implement SQLAlchemy models with proper relationships.

<Invocation tool="Write">
Create `cloud/src/database/models.py`:
- Firm, Sentence, Topic, Theme models
- Foreign key relationships
- Vector(768) columns for embeddings
</Invocation>

<Invocation tool="Write">
Create `cloud/src/database/repository.py`:
- Session management
- CRUD operations per entity
- Bulk insert methods
- Hierarchical query methods
</Invocation>

<Invocation tool="Bash">
Create docker-compose.yml for local Postgres with pgvector:
```bash
# Test locally before cloud deployment
docker-compose up -d postgres
```
</Invocation>

**Validation:** Schema creates successfully, relationships enforce hierarchy.
</Phase>

<Phase id="1.3" name="Database Testing">
**Goal:** Comprehensive tests for the data layer.

<Invocation tool="Write">
Create `tests/unit/test_database.py`:
- Model creation tests
- Relationship constraint tests
- Bulk insert performance tests
- Hierarchical query tests
</Invocation>

<Invocation tool="Bash">
Run database tests:
```bash
pytest tests/unit/test_database.py -v
```
</Invocation>

**Validation:** All tests pass, bulk insert handles 10k+ sentences efficiently.
</Phase>

**HALT: Present Phase 1 deliverables for approval.**
</Objective>

---

<Objective id="pipeline-unification">
**Phase 2: Pipeline Unification**

Merge separate map/reduce containers into a single unified pipeline.

<Phase id="2.1" name="Code Analysis">
**Goal:** Understand current firm_processor and theme_aggregator interfaces.

<Invocation tool="Read">
Study existing implementations:
- `cloud/src/firm_processor.py` - Map phase logic
- `cloud/src/theme_aggregator.py` - Reduce phase logic
- `cloud/containers/map/entrypoint.py` - Current orchestration
- `cloud/containers/reduce/entrypoint.py` - Current orchestration
</Invocation>

**Validation:** Document which methods to preserve vs adapt.
</Phase>

<Phase id="2.2" name="FirmProcessor Adaptation">
**Goal:** Modify FirmProcessor to work with unified pipeline.

Key changes:
1. Accept pre-loaded embedding model (don't create new)
2. Return embeddings along with topics (for storage)
3. Remove S3/DynamoDB writes (caller handles storage)

<Invocation tool="Edit">
Modify `cloud/src/firm_processor.py`:
- Add `embedding_model` parameter to constructor
- Return sentence embeddings from `process()`
- Remove infrastructure dependencies
</Invocation>

**Validation:** FirmProcessor can be instantiated with external model.
</Phase>

<Phase id="2.3" name="ThemeAggregator Adaptation">
**Goal:** Modify ThemeAggregator for direct integration.

Key changes:
1. Accept pre-loaded embedding model
2. Accept topic data directly (not from S3)
3. Return theme embeddings (for storage)

<Invocation tool="Edit">
Modify `cloud/src/theme_aggregator.py`:
- Add `embedding_model` parameter
- Modify `aggregate()` to accept in-memory topic list
- Return embeddings with themes
</Invocation>

**Validation:** ThemeAggregator works with in-memory data.
</Phase>

<Phase id="2.4" name="Unified Pipeline">
**Goal:** Create the main orchestration script.

<Invocation tool="Write">
Create `cloud/src/pipeline/unified_pipeline.py`:
```python
# Structure:
# 1. Setup (load model once)
# 2. Firm loop (process, summarize, store)
# 3. Theme aggregation (cluster, describe, store)
# 4. Finalize (build indexes)
```
</Invocation>

<Invocation tool="Write">
Create `cloud/src/pipeline/checkpoint.py`:
- Query for processed firms
- Mark firm as processed after commit
- Resume logic for spot interruption
</Invocation>

**Validation:** Pipeline processes multiple firms, can be interrupted/resumed.
</Phase>

<Phase id="2.5" name="Integration Testing">
**Goal:** Verify unified pipeline with real data.

<Invocation tool="Write">
Create `tests/integration/test_unified_pipeline.py`:
- End-to-end test with subset of firms
- Interrupt/resume test
- Data integrity verification
</Invocation>

<Invocation tool="Bash">
Run integration tests:
```bash
pytest tests/integration/test_unified_pipeline.py -v
```
</Invocation>

**Validation:** Pipeline completes, data correct, resume works.
</Phase>

**HALT: Present Phase 2 deliverables for approval.**
</Objective>

---

<Objective id="llm-integration">
**Phase 3: LLM Integration**

Add async xAI API calls for topic/theme summarization.

<Phase id="3.1" name="xAI Client">
**Goal:** Implement async API client with rate limiting.

<Invocation tool="WebSearch">
Research xAI API patterns:
- Query: "xAI grok API async python 2024"
- Query: "openai compatible API asyncio semaphore rate limit"
</Invocation>

<Invocation tool="Write">
Create `cloud/src/llm/xai_client.py`:
- AsyncOpenAI client (xAI-compatible)
- Semaphore for rate limiting (50 concurrent)
- Retry with exponential backoff
- Graceful timeout handling
</Invocation>

**Validation:** Client handles rate limits, retries failures.
</Phase>

<Phase id="3.2" name="Summarization Integration">
**Goal:** Integrate LLM calls into pipeline.

<Invocation tool="Edit">
Update firm processing to include summarization:
- After BERTopic clustering, generate summaries
- Embed summaries using same model
- Store both summary text and embedding
</Invocation>

<Invocation tool="Edit">
Update theme aggregation:
- Generate theme descriptions after clustering
- Store descriptions with embeddings
</Invocation>

**Validation:** Summaries generated, stored with embeddings.
</Phase>

<Phase id="3.3" name="LLM Testing">
**Goal:** Test LLM integration with mocks.

<Invocation tool="Write">
Create `tests/unit/test_xai_client.py`:
- Mock API responses
- Rate limiting behavior
- Retry logic
- Timeout handling
</Invocation>

**Validation:** All LLM tests pass with mocked API.
</Phase>

**HALT: Present Phase 3 deliverables for approval.**
</Objective>

---

<Objective id="infrastructure">
**Phase 4: Infrastructure & Deployment**

Deploy simplified infrastructure and validate end-to-end.

<Phase id="4.1" name="Terraform">
**Goal:** Create simplified Terraform for RDS.

<Invocation tool="WebSearch">
Research current Terraform patterns:
- Query: "terraform aws rds postgresql pgvector 2024"
- Query: "terraform rds security group vpc"
</Invocation>

<Invocation tool="Write">
Create `cloud/terraform/` structure:
- `main.tf` - Provider and module composition
- `variables.tf` - Input variables
- `modules/database/main.tf` - RDS PostgreSQL
- `environments/dev.tfvars` - Dev configuration
</Invocation>

<Invocation tool="Bash">
Validate Terraform:
```bash
cd cloud/terraform && terraform init && terraform plan
```
</Invocation>

**Validation:** Terraform plan shows correct resources.
</Phase>

<Phase id="4.2" name="Launch Scripts">
**Goal:** Create scripts for EC2 and RDS management.

<Invocation tool="Write">
Create `cloud/scripts/launch_pipeline.sh`:
- Request spot instance
- Configure environment
- Run pipeline
</Invocation>

<Invocation tool="Write">
Create `cloud/scripts/stop_rds.sh`:
- Stop RDS instance (cost savings)
- Verify stopped state
</Invocation>

**Validation:** Scripts execute without errors.
</Phase>

<Phase id="4.3" name="End-to-End Validation">
**Goal:** Full pipeline run on cloud infrastructure.

<Invocation tool="Bash">
Deploy and run:
```bash
terraform apply
./launch_pipeline.sh
```
</Invocation>

Verify:
- All firms processed
- Data in Postgres
- Vector indexes built
- Hierarchical queries work

**Validation:** Complete run <4 hours, cost <$5.
</Phase>

**FINAL REVIEW: Present complete implementation.**
</Objective>

---

## Tool Usage Patterns

### For Research
- `WebSearch` for documentation and patterns
- `Read` for existing codebase understanding
- `Task` with `Explore` agent for codebase navigation

### For Implementation
- `Write` for new files
- `Edit` for modifications to existing files
- `Bash` for testing and validation

### For Validation
- `Bash` for running tests
- `Read` for verifying changes
- `Task` with `Plan` agent for complex decisions

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| pgvector compatibility | Test locally with Docker first |
| Spot interruption | Per-firm checkpoint, resume logic |
| API rate limits | Semaphore, exponential backoff |
| Memory pressure | Process firms sequentially, commit often |

---

## Quality Gates

Before each HALT:
1. All new tests pass
2. Existing tests still pass
3. Code coverage maintained >80%
4. Documentation updated
5. Changes committed with clear message
