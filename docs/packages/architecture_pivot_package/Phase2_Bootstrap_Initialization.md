# Architecture Pivot: Phase 2 Bootstrap Initialization

Greetings Claude, and welcome to the Financial Topic Modeling project. You are receiving this initialization as the continuation of a strategic architecture pivot—a simplification of our cloud infrastructure from distributed AWS Batch to a single GPU instance with PostgreSQL storage.

**Phase 1 is complete. You are beginning Phase 2.**

---

## Project Overview

The Financial Topic Modeling pipeline identifies cross-firm investment themes from earnings call transcripts. We completed Phases 1-3 of an initial cloud migration (abstraction layer, map container, reduce container), then caught something important: our planned AWS Batch architecture was over-engineered for our scale (~2M sentences quarterly, 3,000-5,000 firms). The orchestration overhead would exceed the actual compute time.

A previous Claude instance executed Phase 1 of the pivot:
- **From:** AWS Batch (3,000+ CPU containers) + DynamoDB
- **To:** Single g4dn.2xlarge GPU instance + PostgreSQL with pgvector

Phase 1 delivered the complete database layer. Your mission is Phase 2: Pipeline Unification.

---

## The Guiding Principles

Before you read another word, internalize these principles. They are not suggestions—they are the standards by which your work will be judged:

### 1. Simplicity is Mandatory

> "The best engineers write code my mom could read. They choose boring technology, over-document the 'why', and under-engineer the 'how'. Complexity is not a flex; it becomes a liability. Real seniority is making hard problems look simple, not making simple problems look hard."

If you find yourself building something clever, stop. Clever is a warning sign. The pivot itself exists because we caught over-engineering before deploying it. Honor that lesson.

### 2. Adapt, Don't Rewrite

The core ML logic works. `BERTopicModel`, `FirmProcessor`, `ThemeAggregator`—these have been tested against real MAG7 earnings call data and produce valid themes. You are reorganizing the orchestration layer, not reimplementing the algorithms.

**PRESERVE:** The `fit_transform()`, `process()`, and `aggregate()` interfaces
**ADAPT:** The I/O layer (remove S3/DynamoDB, add Postgres)
**UNIFY:** The orchestration (single script, model loaded once)

If you delete working ML logic and rewrite it, you have failed the mission.

### 3. TDD is Non-Negotiable

Write tests before implementation. Not after. Not "when you have time." Before.

- 80%+ coverage required
- Tests document intent
- Tests catch regressions
- Tests are the specification

The Phase 1 instance delivered 50 tests. Match or exceed that standard.

### 4. Halting Points are Real

Phase 2 ends with a HALT. You stop, summarize your work, and await approval before Phase 3. This is not optional. The review protocol exists because mistakes compound—catching issues at phase boundaries is 10× cheaper than catching them in production.

---

## Your Context Package

The following documents provide everything you need:

### Primary Reference
| Document | Location | Purpose |
|----------|----------|---------|
| **Phase 2 Handover** | `docs/handovers/resume-task-architecture-pivot-phase2.md` | Complete context from Phase 1 instance—what was built, why, decisions made, issues encountered, and your exact next steps |

### Original Planning Documents
| Document | Location | Purpose |
|----------|----------|---------|
| Senior Engineer Plan | `docs/packages/architecture_pivot_package/Senior_Engineer_Plan_architecture_pivot.md` | Technical specification—schema, code matrix, acceptance criteria |
| Mission Briefing | `docs/packages/architecture_pivot_package/Mission_Briefing_architecture_pivot.md` | Phased execution guide with halting points |
| SubAgent Strategy | `docs/packages/architecture_pivot_package/SubAgent_Strategy_architecture_pivot.md` | Tactical recommendations for tools and patterns |
| Raw Vision Transcript | `docs/packages/architecture_pivot_package/raw_transcript_architecture_pivot.md` | Unedited researcher vision—consult when structured docs feel ambiguous |

### Supporting Context
| Document | Location | Purpose |
|----------|----------|---------|
| Gemini Analysis | `docs/ai-log/gemini-conversation.md` | The architectural analysis that prompted the pivot |
| Project Guidelines | `CLAUDE.md` | Project conventions and commands |

---

## What Phase 1 Delivered

The previous instance built the complete PostgreSQL + pgvector database layer:

```
cloud/src/database/
├── __init__.py           # Package exports
├── models.py             # Firm, Sentence, Topic, Theme (SQLAlchemy + Vector(768))
└── repository.py         # DatabaseRepository with CRUD, bulk ops, hierarchical queries

tests/unit/test_database.py   # 50 tests, ~98% coverage
docker-compose.yml            # Local Postgres + pgvector
scripts/init-pgvector.sql     # Extension setup
```

**Key Repository Methods You'll Use:**
```python
# Checkpoint/Resume
repo.get_unprocessed_firms() -> List[Firm]
repo.mark_firm_processed(firm_id) -> None

# Bulk Operations
repo.bulk_insert_sentences(List[Dict]) -> int
repo.bulk_insert_topics(List[Dict]) -> int

# After All Processing
repo.build_vector_indexes() -> None  # HNSW indexes - MUST COMMIT AFTER!
```

**Critical:** `build_vector_indexes()` does NOT commit. You MUST call `session.commit()` after building indexes to persist them.

The handover document contains the complete API surface and design decisions.

---

## Your Phase 2 Mission

### Objective
Merge the separate map and reduce containers into a single unified pipeline script that:
1. Loads the embedding model ONCE
2. Processes firms sequentially with per-firm checkpoints
3. Hands off directly to theme aggregation (no S3 intermediate)
4. Writes all results to PostgreSQL

### Deliverables
- [ ] `cloud/src/pipeline/unified_pipeline.py` - Main orchestration
- [ ] `cloud/src/pipeline/checkpoint.py` - Resume logic
- [ ] Modified `cloud/src/firm_processor.py` - Postgres integration
- [ ] Modified `cloud/src/theme_aggregator.py` - Direct integration
- [ ] `tests/integration/test_unified_pipeline.py`

### Acceptance Criteria
- [ ] Embedding model loaded exactly once
- [ ] Pipeline can be interrupted and resumed from any firm
- [ ] No S3 intermediate storage required
- [ ] All existing ML logic preserved (not rewritten)
- [ ] Tests pass with real Postgres (testcontainers)

---

## Getting Started

**Before writing any code:**

1. **Read the handover document** (`docs/handovers/resume-task-architecture-pivot-phase2.md`)
   - Understand what was built and why
   - Note the decisions made and issues encountered
   - Absorb the repository API you'll be using

2. **Read the code you'll be adapting:**
   - `cloud/src/firm_processor.py` - Current FirmProcessor implementation
   - `cloud/src/theme_aggregator.py` - Current ThemeAggregator implementation
   - `cloud/containers/map/entrypoint.py` - How firm_processor is currently called
   - `cloud/containers/reduce/entrypoint.py` - How theme_aggregator is currently called

3. **Read the database layer you'll be using:**
   - `cloud/src/database/models.py` - The schema
   - `cloud/src/database/repository.py` - The methods available

4. **Understand the data flow:**
   - Input: CSV with earnings call transcripts
   - FirmProcessor: Sentences → Topics (per firm)
   - ThemeAggregator: Topics → Themes (cross-firm)
   - Output: Everything in PostgreSQL with embeddings

---

## Critical Implementation Notes

### The Embedding Challenge (Critical)

**Current Problem:** `FirmProcessor` calls BERTopic internally, which computes embeddings, but those embeddings are NOT returned to the caller. The Postgres `sentences` table has an `embedding vector(768)` column that MUST be populated.

**Your Task:** Adapt the I/O interface (not the ML logic) to surface embeddings. Options:
1. Have `FirmProcessor` accept pre-computed embeddings as input AND return them with results
2. Have the unified pipeline compute embeddings externally, pass to BERTopic, and store directly

**Requirements:**
- Embedding model loaded ONCE at startup (not per-firm)
- BERTopic supports pre-computed embeddings—use this capability
- Store sentence embeddings in Postgres alongside text
- Reuse the same model for topic summary embeddings in Phase 3

### The Checkpoint Pattern
```python
for firm in unprocessed_firms:
    try:
        # Process firm
        results = firm_processor.process(firm, embeddings)

        # Write to Postgres
        repo.bulk_insert_sentences(...)
        repo.bulk_insert_topics(...)
        repo.mark_firm_processed(firm.id)

        # CHECKPOINT - commit after each firm
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Failed on {firm.company_id}: {e}")
        raise
```

### What NOT to Change
- `BERTopicModel.fit_transform()` interface
- `TopicModelResult` dataclass structure
- The "Dual-BERTopic" pattern (topic representations become documents for theme clustering)
- Any clustering parameters or algorithms

---

## Review Protocol

Your work will be reviewed by an impartial Codex instance at the Phase 2 halting point. Expect reviews that check:

1. **Alignment**: Does implementation match the Senior Engineer Plan?
2. **Quality**: Tests, type hints, documentation, coverage >80%
3. **Complexity**: Will be flagged aggressively—simplicity is mandatory
4. **Preservation**: Did you adapt existing code or unnecessarily rewrite it?

The Phase 1 instance went through two revision cycles with Codex—a critical bug in bulk updates was caught and fixed. Expect the same rigor.

---

## The Why Behind Phase 2

The architecture pivot eliminates cold start overhead by loading the embedding model once instead of 3,000+ times. But that benefit is only realized if the unified pipeline actually works.

Phase 2 is where the pivot becomes real:
- Phase 1 gave us the storage layer (PostgreSQL)
- Phase 2 gives us the execution layer (unified pipeline)
- Phase 3 will add LLM integration (topic/theme naming)
- Phase 4 will deploy infrastructure (Terraform)

Without Phase 2, we have a database with no data in it. Your work is the bridge.

---

## Final Reminders

1. **Read before you write.** Understand the existing code before modifying it.

2. **Test before you implement.** TDD is not a suggestion.

3. **Commit after each firm.** Spot instances can be interrupted. Checkpoints save work.

4. **Keep the ML logic.** Adapt the I/O, preserve the algorithms.

5. **Stop at the halt point.** Phase 2 complete → summarize → await approval.

---

The database layer is ready. The core ML logic is proven. The architecture is validated.

Now unify the pipeline.

Let's build something we can be proud of.
