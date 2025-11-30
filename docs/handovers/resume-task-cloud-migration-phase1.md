### **Purpose:**

You are an AI Engineer resuming work on the Financial Topic Modeling pipeline cloud migration. This command contains a complete handover from the previous session to seamlessly restore your context and momentum.

### **Mission Resumption: cloud-migration-phase1**

Your memory of previous work on this task has been wiped. This briefing is your **complete source of truth** for resuming the mission. The original strategic documents are provided for full context.

**Project Context:**

- Project Guidelines: `CLAUDE.md`
- Vision Document: `First_Transcript.md`

**Original Planning Documents:**

- Senior Engineer Plan: `docs/packages/cloud_migration_package/Senior_Engineer_Plan_cloud_migration.md`
- Mission Briefing: `docs/packages/cloud_migration_package/Mission_Briefing_cloud_migration.md`
- SubAgent Strategy: `docs/packages/cloud_migration_package/SubAgent_Strategy_cloud_migration.md`
- Raw Vision Transcript: `docs/packages/cloud_migration_package/raw_transcript_cloud_migration.md`

**Approved Implementation Plan:**

- Location: `docs/packages/cloud_migration_package/plan.md`
- This is the AUTHORITATIVE specification - read it fully before proceeding

---

### **Handover Report from Previous Session**

<Handover>

<Status>
**Overall Progress:** Planning Phase COMPLETE. Implementation Phase 1 READY TO BEGIN.
**Current Phase:** Phase 1 - Abstraction Layer (Day 1 of ~7 day timeline)
**Blocker:** None - plan has been approved after extensive review with user and Codex
**Session Summary:** This session was entirely focused on creating a comprehensive, production-ready specification. No implementation code was written. The plan underwent 3 major revision cycles incorporating feedback from both the user and an external Codex review.
</Status>

<Pipeline_Position>
**Scope:** Full pipeline cloud migration

- **Map Phase:** Firm-level topic modeling → AWS Batch containers
- **Reduce Phase:** Cross-firm theme identification → AWS Batch containers
- **Storage:** S3 (intermediate JSON) + DynamoDB (final hierarchical output including sentences)
- **Orchestration:** Step Functions with distributed map
- **Infrastructure:** Modular Terraform

**Key Architecture Decision:** SageMaker was replaced with AWS Batch for reduce phase. Rationale: (1) simpler - same job/queue pattern, (2) cheaper - Spot instances, (3) sufficient - no SageMaker-specific ML ergonomics needed. User approved this deviation.
</Pipeline_Position>

<Decisions>
**Critical Design Decisions Made:**

1. **NO MVP Code Porting**

   - Rationale: Local_BERTopic_MVP is poorly structured, over-bloated code with dead paths, magic numbers (768d centroids), and artifacts from abandoned approaches
   - Action: Reimplement cleanly from scratch. MVP is for INTENT reference only (parameter values, data flow understanding)
   - User emphasized this multiple times - it's a core principle

2. **Centroids/Embeddings REMOVED from output**

   - Rationale: These were MVP artifacts from an old similarity-based theme identification approach
   - Current Approach: Dual-BERTopic - reduce phase RE-EMBEDS topic representations as new documents
   - Output contains `topic_representations` (strings) not embeddings

3. **Sentences Written by Map Phase to DynamoDB**

   - Rationale: Avoids backfill in reduce phase; simplifies reduce logic
   - Design: PK=`TOPIC#{firm_id}#{topic_id}`, SK=`SENTENCE#{sentence_id}`
   - GSI1 enables firm-level lookups: GSI1PK=`FIRM#{firm_id}`
   - Query flow: Theme → Topics (one query) → Sentences per topic (N queries, two-hop)

4. **Instance Sizing Requires Benchmarking**

   - t3.large (8GB) likely too small for BERTopic + sentence-transformers
   - Starting recommendations: m5.xlarge (16GB) map, m5.2xlarge (32GB) reduce
   - Benchmarking step added to Phase 2 before locking sizes

5. **TopicModel + DataConnector Interfaces**

   - Both included per user decision
   - Enables: BERTopic/LDA/neural comparison (core research need), CSV/S3/WRDS swapping
   - LDA and Neural stubs with NotImplementedError for faculty review

6. **Single-Table DynamoDB Design**
   - Three item types: Theme, Topic-in-Theme, Sentence
   - One GSI (GSI1) for firm-based reverse lookups
   - Keeps infrastructure simple while enabling hierarchical queries
     </Decisions>

<Wins>
**Planning Phase Achievements:**

1. **Comprehensive Specification Created** (~1200 lines)

   - Explicit data schemas (TranscriptData, TopicModelResult, FirmTopicOutput, ThemeOutput)
   - Operational rules locked (sentence_id format, firm_id normalization, date filtering, outlier handling, theme_id generation)
   - Complete DynamoDB item shapes with PK/SK patterns
   - Concrete entrypoint implementations with error handling
   - Modular Terraform with full resource definitions

2. **Three-Perspective Analysis Completed**

   - Simplicity-First: Fastest path, minimal abstractions
   - Modularity-First: Clean interfaces for extensibility
   - Test-Driven: TDD workflow with moto mocking
   - Synthesized into balanced approach

3. **Codex Review Incorporated**

   - Schema finalization (removed MVP artifacts)
   - Explicit halting points per phase
   - Modular Terraform structure
   - Batch rationale documented

4. **User Raw Transcript Review Incorporated**

   - Strong warnings about MVP code quality
   - Removed "port/wrap" language → "reimplement simply"
   - De-emphasized baseline tests (MVP quality wasn't great)
   - Clarified Dual-BERTopic re-embedding approach

5. **Five Explicit Halting Points Defined**
   - Phase 1: After abstraction layer
   - Phase 2: After map container
   - Phase 3: After reduce container
   - Phase 4: After Terraform
   - Phase 5: After integration
     </Wins>

<Artifacts>
**Created Files:**
- `docs/packages/cloud_migration_package/plan.md` - AUTHORITATIVE PLAN (read this first!)

**Files to Create (Phase 1):**

```
cloud/
├── src/
│   ├── __init__.py
│   ├── interfaces.py               # TopicModel + DataConnector ABCs
│   ├── models.py                   # TranscriptSentence, FirmTranscriptData, etc.
│   ├── connectors/
│   │   ├── __init__.py
│   │   ├── local_csv.py            # LocalCSVConnector
│   │   └── s3_connector.py         # S3TranscriptConnector
│   └── topic_models/
│       ├── __init__.py
│       ├── bertopic_model.py       # BERTopicModel
│       ├── lda_model.py            # LDATopicModel stub
│       └── neural_model.py         # NeuralTopicModel stub
└── tests/
    ├── conftest.py
    └── unit/
        ├── test_interfaces.py
        ├── test_bertopic_model.py
        └── test_connectors.py
```

**MVP Files for INTENT Reference (DO NOT COPY CODE):**

- `Local_BERTopic_MVP/src/config/config.yaml` - UMAP/HDBSCAN params only
- `Local_BERTopic_MVP/src/topic_modeling/firm_topic_analyzer.py` - Understand BERTopic flow
- `Local_BERTopic_MVP/src/data_ingestion/local_csv_connector.py` - Understand CSV structure
  </Artifacts>

<ML_Metrics>
**Target Metrics (from plan):**

- Processing time per firm: <5 minutes
- Memory usage (map): <16GB (benchmarking required)
- Memory usage (reduce): <32GB (benchmarking required)
- Theme count: Within ±15% of MVP baseline
- Firm coverage: ≥90%
- Test coverage: 80%+ (REQUIRED)

**Configuration Values (from MVP - verified working):**

- Embedding model: `all-mpnet-base-v2`
- UMAP: n_neighbors=15, n_components=10, min_dist=0.0, metric=cosine
- HDBSCAN: min_cluster_size=6, min_samples=2
- Validation: min_firms=2, max_firm_dominance=0.4
  </ML_Metrics>

<Issues>
**No Blocking Issues** - Plan is approved and ready for implementation

**Known Considerations:**

1. Instance sizing needs benchmarking before locking (Phase 2 task)
2. Baseline quality tests are LOW PRIORITY - MVP quality wasn't great anyway
3. No WRDS connector implementation (funding-dependent; interface ready)

**Technical Debt Deferred:**

- CI/CD pipeline (manual deployment acceptable for 1 week)
- VPC/networking (default VPC works for academic project)
- StorageBackend interface (direct boto3 calls suffice)
  </Issues>

<Config_Changes>
**New Configuration Structure (to be created):**

```yaml
# cloud/config/default.yaml
embedding_model: "all-mpnet-base-v2"

umap:
  n_neighbors: 15
  n_components: 10
  min_dist: 0.0
  metric: "cosine"

hdbscan:
  min_cluster_size: 6
  min_samples: 2

validation:
  min_firms: 2
  max_firm_dominance: 0.4
```

</Config_Changes>

<Next_Steps>
**Your immediate task is to implement Phase 1: Abstraction Layer**

The plan at `docs/packages/cloud_migration_package/plan.md` is your AUTHORITATIVE specification. Read it fully before proceeding.

**Phase 1 Implementation Order:**

1. **Create directory structure:**

   ```bash
   mkdir -p cloud/src/connectors cloud/src/topic_models cloud/config
   mkdir -p tests/unit tests/integration tests/fixtures
   ```

2. **Write unit tests (TDD - tests BEFORE implementation):**

   - `tests/unit/test_interfaces.py`
   - `tests/unit/test_bertopic_model.py`
   - `tests/unit/test_connectors.py`

3. **Implement data models** (`cloud/src/models.py`):

   - `TranscriptSentence` dataclass
   - `FirmTranscriptData` dataclass
   - `TranscriptData` dataclass
   - `TopicModelResult` dataclass
   - Follow schemas EXACTLY as defined in plan Phase 1.1

4. **Implement interfaces** (`cloud/src/interfaces.py`):

   - `TopicModel` ABC with `fit_transform()` method
   - `DataConnector` ABC with `fetch_transcripts()` and `get_available_firms()` methods

5. **Implement BERTopicModel** (`cloud/src/topic_models/bertopic_model.py`):

   - Reimplement cleanly - DO NOT port MVP code
   - Use config for UMAP/HDBSCAN params
   - Return `TopicModelResult` with representations (NOT centroids)

6. **Implement LocalCSVConnector** (`cloud/src/connectors/local_csv.py`):

   - Reimplement cleanly - DO NOT port MVP code
   - Read CSV columns: companyid, companyname, componenttext, mostimportantdateutc
   - Generate sentence*ids per operational rules: `{firm*id}*{transcript*id})\_{position:04d}`

7. **Create stubs:**
   - `LDATopicModel` with `NotImplementedError`
   - `NeuralTopicModel` with `NotImplementedError`
   - `S3TranscriptConnector` (placeholder for Phase 2)

**Validation Criteria (must pass before HALT):**

- [ ] All interfaces defined with proper ABCs
- [ ] LocalCSVConnector successfully loads sample CSV
- [ ] BERTopicModel produces TopicModelResult from test documents
- [ ] All unit tests pass
- [ ] Test coverage >80% for Phase 1 code

**HALT after Phase 1 completion - await user approval before proceeding to Phase 2 (Map Container)**
</Next_Steps>

</Handover>

---

### **Critical Context: Why This Approach**

The user is an undergraduate research assistant building an academic capstone for publication. Key philosophical principles:

1. **"Code my mom could read"** - Prioritize clarity over cleverness
2. **"Boring technology"** - Use proven, well-documented patterns
3. **"Over-document the why"** - Explain reasoning, not just mechanics
4. **"Complexity is a liability"** - Simpler is better

The MVP (`Local_BERTopic_MVP/`) is explicitly marked as POORLY STRUCTURED code that should NOT be copied. It contains:

- Unused functions and dead code paths
- Hardcoded magic numbers (768d centroids)
- Artifacts from abandoned approaches (similarity-based theme identification)
- Over-bloated logic

Your implementation must be SIMPLER, MORE READABLE, and MORE MODULAR than the MVP.

---

### **Action Directive**

1. **Read the full approved plan** at `docs/packages/cloud_migration_package/plan.md`
2. **Read CLAUDE.md** for project conventions
3. **Begin Phase 1 implementation** following the `<Next_Steps>` above
4. **Write tests BEFORE implementation** (TDD is mandatory)
5. **HALT after Phase 1** for user approval before Phase 2

The plan is comprehensive and approved. Your job is now execution.
