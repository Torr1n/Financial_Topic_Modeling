# Mission Briefing: Architecture Pivot Implementation

**Agent Persona:** Cloud Infrastructure Engineer + ML Pipeline Specialist
**Mission ID:** architecture-pivot-gpu-postgres
**Status:** Ready for Execution

---

## Mission Overview

You are implementing an architectural pivot for the Financial Topic Modeling pipeline. The existing AWS Batch map-reduce design has been identified as over-engineered for our scale. Your mission is to migrate to a simpler, more cost-effective architecture: a single GPU instance with PostgreSQL + pgvector storage.

**Critical Context:** Phases 1-3 are COMPLETE. The core ML logic (BERTopicModel, FirmProcessor, ThemeAggregator) works. This is a **reorganization**, not a rewrite. Preserve working code; adapt the orchestration layer.

---

## Phase 1: Database Layer Design

**Objective:** Design and implement the PostgreSQL schema with pgvector support.

### Tasks

1. **Schema Design**
   - Create SQLAlchemy models for: `Firm`, `Sentence`, `Topic`, `Theme`
   - Include `vector(768)` columns for embeddings
   - Define foreign key relationships encoding the hierarchy

2. **Repository Layer**
   - Implement CRUD operations for all entities
   - Bulk insert methods for performance
   - Query methods for hierarchical traversal

3. **Local Testing**
   - Docker Compose for local Postgres with pgvector
   - Unit tests for all repository methods

### Deliverables
- [ ] `cloud/src/database/models.py` - SQLAlchemy models
- [ ] `cloud/src/database/repository.py` - Data access layer
- [ ] `docker-compose.yml` - Local Postgres for testing
- [ ] `tests/unit/test_database.py` - Repository tests

### Acceptance Criteria
- All models define correct relationships
- Bulk insert handles 10,000+ sentences efficiently
- Foreign key constraints enforce hierarchy
- Vector columns accept 768-dimensional embeddings

---

**== HALT: Await approval before proceeding to Phase 2 ==**

---

## Phase 2: Pipeline Unification

**Objective:** Merge map and reduce containers into a single unified pipeline script.

### Tasks

1. **Unified Pipeline Script**
   - Create `cloud/src/pipeline/unified_pipeline.py`
   - Load embedding model ONCE at startup
   - Sequential firm processing loop
   - Direct handoff to theme aggregation (no S3)

2. **Adapt Existing Code**
   - Modify `FirmProcessor` to accept pre-loaded model
   - Modify `FirmProcessor` to write to Postgres (not S3/DynamoDB)
   - Modify `ThemeAggregator` to read from Postgres (not S3)

3. **Checkpoint/Resume Logic**
   - Query Postgres for already-processed firms
   - Skip processed firms on resume
   - Commit after each firm (spot instance safety)

### Deliverables
- [ ] `cloud/src/pipeline/unified_pipeline.py` - Main orchestration
- [ ] `cloud/src/pipeline/checkpoint.py` - Resume logic
- [ ] Modified `firm_processor.py` - Postgres integration
- [ ] Modified `theme_aggregator.py` - Direct integration
- [ ] `tests/integration/test_unified_pipeline.py`

### Acceptance Criteria
- Embedding model loaded exactly once
- Pipeline can be interrupted and resumed
- No S3 intermediate storage required
- All existing ML logic preserved

---

**== HALT: Await approval before proceeding to Phase 3 ==**

---

## Phase 3: LLM Integration

**Objective:** Add async xAI API calls for topic/theme summarization.

### Tasks

1. **xAI Client**
   - Implement async client using OpenAI-compatible API
   - Rate limiting with asyncio.Semaphore
   - Retry logic with exponential backoff

2. **Topic Summarization**
   - Generate summaries for firm topics after clustering
   - Embed summaries using same model (GPU)
   - Store summaries and embeddings in Postgres

3. **Theme Description**
   - Generate descriptions for cross-firm themes
   - Store in themes table

### Deliverables
- [ ] `cloud/src/llm/xai_client.py` - Async API client
- [ ] Updated `firm_processor.py` - Topic summarization
- [ ] Updated `theme_aggregator.py` - Theme description
- [ ] `tests/unit/test_xai_client.py` - Mock API tests

### Acceptance Criteria
- Async calls don't block GPU processing
- Rate limiting prevents API throttling
- Graceful degradation if API unavailable
- Summaries stored with embeddings

---

**== HALT: Await approval before proceeding to Phase 4 ==**

---

## Phase 4: Infrastructure & Deployment

**Objective:** Deploy the simplified infrastructure and run end-to-end.

### Tasks

1. **Terraform for RDS**
   - PostgreSQL instance (db.t4g.large)
   - Security groups for access
   - pgvector extension enabled

2. **EC2 Launch Script**
   - g4dn.2xlarge spot instance
   - Deep Learning AMI
   - User data for setup

3. **End-to-End Validation**
   - Run pipeline on full CSV dataset
   - Verify all data in Postgres
   - Build vector indexes
   - Test hierarchical queries

### Deliverables
- [ ] `cloud/terraform/` - Simplified Terraform
- [ ] `cloud/scripts/launch_pipeline.sh` - EC2 launch
- [ ] `cloud/scripts/stop_rds.sh` - Cost management
- [ ] Documentation of full run

### Acceptance Criteria
- Pipeline completes in <4 hours
- Total compute cost <$5
- All data queryable in Postgres
- Vector search functional

---

**== FINAL REVIEW: Present complete implementation ==**

---

## Critical Constraints

### Preserve Existing Logic
The following must NOT be rewritten (only adapted):
- `BERTopicModel.fit_transform()` interface
- `FirmProcessor.process()` core logic
- `ThemeAggregator.aggregate()` core logic
- Data model structures

### Code Quality
- TDD: Tests before implementation
- 80%+ coverage maintained
- Documentation for all new modules
- Type hints throughout

### Cost Consciousness
- RDS must be stoppable (no Aurora Serverless)
- Spot instance for compute
- No unnecessary AWS services

---

## Reference Documents

| Document | Purpose |
|----------|---------|
| `Senior_Engineer_Plan_architecture_pivot.md` | Detailed technical specification |
| `raw_transcript_architecture_pivot.md` | Original vision and rationale |
| `docs/ai-log/gemini-conversation.md` | Architecture analysis |
| `cloud/src/firm_processor.py` | Existing map logic |
| `cloud/src/theme_aggregator.py` | Existing reduce logic |

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Processing Time | <4 hours for 3,000 firms |
| Compute Cost | <$5 per quarterly run |
| Code Coverage | >80% |
| Resume Capability | Full (from any firm) |
| Query Latency | <100ms for hierarchical queries |

---

**Design Philosophy Reminder:**

> "The best engineers write code my mom could read. They choose boring technology, over-document the 'why', and under-engineer the 'how'. Complexity is not a flex; it becomes a liability."

This pivot is a victory for simplicity. Execute it with the same rigor.
