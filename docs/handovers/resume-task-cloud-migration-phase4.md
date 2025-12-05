### **Purpose:**

You are an AI Engineer resuming work on the Financial Topic Modeling cloud migration. This command contains a complete handover from the previous session to seamlessly restore your context and momentum.

### **Mission Resumption: cloud-migration-phase4**

Your memory of previous work on this task has been wiped. This briefing is your **complete source of truth** for resuming the mission. The original strategic documents are provided for full context.

**Project Context:**
- Project Guidelines: `CLAUDE.md`
- Vision Document: `First_Transcript.md`

**Original Planning Documents:**
- Approved Plan (AUTHORITATIVE): `docs/packages/cloud_migration_package/plan.md`
- Mission Briefing: `docs/packages/cloud_migration_package/Mission_Briefing_cloud_migration.md`
- Strategy: `docs/packages/cloud_migration_package/SubAgent_Strategy_cloud_migration.md`
- Phase 3 Handover: `docs/handovers/resume-task-cloud-migration-phase3.md`
- Phase 3 Review: `docs/handovers/phase3-reduce-container-review.md`

---

### **Handover Report from Previous Session**

<Handover>

<Status>
**Overall Progress:** Phases 1, 2, and 3 COMPLETE and APPROVED. Phase 4 (Terraform Infrastructure) is NEXT.
**Current Phase:** About to begin Phase 4 implementation
**Blocker:** None - ready to proceed

**Critical Context:** This is a CLOUD MIGRATION project. The MVP code at `Local_BERTopic_MVP/` is EXPLICITLY marked as poorly structured and should NOT be ported. We build CLEAN, SIMPLE code following the approved plan schemas. All containers (map + reduce) are tested and working locally.
</Status>

<Pipeline_Position>
**Completed Phases:**
- Phase 1: Abstraction Layer (models, interfaces, BERTopicModel, LocalCSVConnector)
- Phase 2: Map Container (FirmProcessor, s3_utils, dynamodb_utils, map entrypoint)
- Phase 3: Reduce Container (ThemeAggregator, reduce entrypoint, validation filters)

**Next Phase:** Phase 4 - Terraform Infrastructure
- `cloud/terraform/` - Modular Terraform structure
- Storage module: S3 bucket, DynamoDB table with GSI
- Compute module: AWS Batch environments, job definitions
- Orchestration module: Step Functions (if needed)

**Downstream:** Phase 5 is Integration Testing (after Phase 4 approval)
</Pipeline_Position>

<Decisions>
### Architectural Decisions (Locked - Phases 1-3)

1. **Dual-BERTopic** - Topic representations (strings) are RE-EMBEDDED as new documents in reduce phase. No centroids, no 768-dim embeddings carried forward. Same TopicModel interface for both phases.

2. **Single-Table DynamoDB Design** - One table with composite keys (PK/SK):
   - Theme metadata: `PK=THEME#theme_001, SK=METADATA`
   - Topic links: `PK=THEME#theme_001, SK=TOPIC#firm_id#topic_id`
   - Sentences: `PK=TOPIC#firm_id#topic_id, SK=SENTENCE#sentence_id`
   - GSI1: `GSI1PK=FIRM#{firm_id}` for firm-based lookups

3. **Validation Filters** - Themes require:
   - `min_firms=2`: At least 2 distinct firms
   - `max_firm_dominance=0.4`: No firm can have >40% of topics

4. **firm_id from companyid** - CSV `companyid` column is the canonical firm identifier

5. **SpaCy Sentence Splitting** - Transcript components split into sentences before topic modeling

6. **TDD Mandatory** - Tests written before implementation, target 80%+ coverage

### Phase 3 Decisions Made

7. **Module-Scoped Test Fixture** - Integration tests use a cached BERTopic run to reduce runtime from 15+ minutes to ~2-3 minutes.

8. **Real Data for Integration Tests** - Uses actual MAG7 firm outputs from `output/map_test/` instead of synthetic data to avoid CountVectorizer `min_df` errors.

9. **Theme ID Format** - `theme_YYYYMMDD_NNN` sorted by n_topics descending

10. **Graceful Error Handling** - Reduce phase skips corrupt/invalid JSON files and continues processing valid ones.
</Decisions>

<Wins>
### Phase 3 Completed (This Session)

**Core Implementation:**
- `cloud/src/theme_aggregator.py` (207 lines) - ThemeAggregator with Dual-BERTopic approach
- `cloud/containers/reduce/Dockerfile` - Container definition
- `cloud/containers/reduce/requirements.txt` - Same deps as map
- `cloud/containers/reduce/entrypoint.py` (255 lines) - Entry point with local/cloud modes
- `cloud/scripts/local_test_reduce.sh` - Local testing script

**Test Coverage:**
- `tests/unit/test_theme_aggregator.py` - 23 unit tests (all passing)
- `tests/integration/test_reduce_entrypoint.py` - 15 integration tests (all passing)
- Added fixtures to `tests/conftest.py`: `sample_firm_topic_outputs`, `validate_theme_output`

**Key Fixes During Session:**
1. Fixed unit tests that failed due to max_dominance filter being too strict for 2-firm scenarios (needed 3+ firms per theme)
2. Fixed integration tests that used synthetic data causing `min_df` errors (switched to real MAG7 data)
3. Optimized integration test runtime with module-scoped fixture (15+ min → ~2-3 min)
4. Fixed `datetime.utcnow()` deprecation warnings

### All Prior Phases Complete
- Phase 1: Abstraction layer with swappable TopicModel implementations
- Phase 2: Map container with sentence splitting and stopword removal
- Post-Phase 2: BERTopic enhancements (representations, probabilities, ordering)

### Test Results Summary
- Unit tests: 23 ThemeAggregator + prior tests = ~50+ total
- Integration tests: 15 reduce + prior map tests
- Coverage: Target 80%+ maintained
</Wins>

<Artifacts>
**Phase 3 Files Created:**
```
cloud/src/theme_aggregator.py
cloud/containers/reduce/Dockerfile
cloud/containers/reduce/requirements.txt
cloud/containers/reduce/entrypoint.py
cloud/scripts/local_test_reduce.sh
tests/unit/test_theme_aggregator.py
tests/integration/test_reduce_entrypoint.py
docs/handovers/phase3-reduce-container-review.md
```

**Files Modified:**
```
tests/conftest.py (added fixtures)
```

**Full Project Structure:**
```
cloud/
├── src/
│   ├── models.py                    # Data models
│   ├── interfaces.py                # TopicModel, DataConnector ABCs
│   ├── firm_processor.py            # Map phase logic
│   ├── theme_aggregator.py          # Reduce phase logic (NEW)
│   ├── s3_utils.py                  # S3 helpers
│   ├── dynamodb_utils.py            # DynamoDB writers (Map + Reduce)
│   ├── connectors/
│   │   ├── local_csv.py             # Local CSV connector
│   │   └── s3_connector.py          # S3 connector (stub)
│   └── topic_models/
│       ├── bertopic_model.py        # BERTopic implementation
│       ├── lda_model.py             # LDA stub
│       └── neural_model.py          # Neural stub
├── containers/
│   ├── map/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── entrypoint.py
│   └── reduce/                      # NEW
│       ├── Dockerfile
│       ├── requirements.txt
│       └── entrypoint.py
├── config/
│   └── default.yaml
├── scripts/
│   ├── local_test_map.sh
│   └── local_test_reduce.sh         # NEW
└── terraform/                       # TO BE CREATED IN PHASE 4
```

**Sample Output Available:**
- `output/map_test/*.json` - MAG7+ firm topic outputs (9 firms)
- `output/reduce_test/themes.json` - Cross-firm themes (after local test)
</Artifacts>

<ML_Metrics>
### Map Phase (Per Firm)
- Microsoft (21835): 447 sentences → 29 topics, 104 outliers
- Processing time: ~2-3 minutes per firm (with embedding)

### Reduce Phase (Cross-Firm)
- Input: 9 firms, ~250+ topics total
- Output: Themes filtered by min_firms=2, max_dominance=0.4
- Processing time: ~2-3 minutes for aggregation

### Validation Filters Applied
- `min_firms=2`: Ensures cross-firm themes only
- `max_dominance=0.4`: Prevents single-firm-dominated themes
</ML_Metrics>

<Issues>
### Known Limitations (Not Blockers)

1. **S3TranscriptConnector** - Still a stub. Needed for full cloud deployment but not blocking Phase 4 Terraform work.

2. **Integration Test Runtime** - ~2-3 minutes with cached fixture. Acceptable but noted.

3. **Container Images Not Published** - Dockerfiles exist but images not pushed to ECR. Will happen in Phase 4/5.

4. **Step Functions Not Implemented** - Orchestration is Phase 5 scope, not blocking.

### Technical Debt
- `map/entrypoint.py` at 73% coverage (guard branches untested)
- Some tests use `@pytest.mark.slow` but mark not registered (cosmetic warning)
</Issues>

<Config_Changes>
**cloud/config/default.yaml** (unchanged from Phase 2):
```yaml
embedding_model: "all-mpnet-base-v2"
device: "cpu"
umap:
  n_neighbors: 15
  n_components: 10
  min_dist: 0.0
  metric: "cosine"
  random_state: 42
hdbscan:
  min_cluster_size: 6
  min_samples: 2
  metric: "euclidean"
  cluster_selection_method: "leaf"
vectorizer:
  ngram_range: [1, 2]
  min_df: 2
representation:
  mmr_diversity: 0.3
  pos_model: "en_core_web_sm"
validation:
  min_firms: 2
  max_firm_dominance: 0.4
```
</Config_Changes>

<Understanding>
### WHY This Architecture Works

**Dual-BERTopic Elegance:**
The key insight is that topic representations (strings like "ai investment strategy") can be treated as documents themselves. In the reduce phase, we:
1. Collect all topic representations from all firms
2. Embed them using the SAME embedding model
3. Cluster them using the SAME BERTopic pipeline
4. Get "themes" which are clusters of similar topics

This means ONE interface (`TopicModel`) and ONE implementation (`BERTopicModel`) serve BOTH phases. No special theme-clustering code needed.

**Validation Filters Purpose:**
- `min_firms=2`: A "cross-firm theme" with only one firm is meaningless
- `max_dominance=0.4`: If 80% of topics come from one firm, it's not a cross-firm pattern

**Module-Scoped Test Fixture:**
BERTopic is slow (~2 min per run). Running it per-test took 15+ minutes. By caching the result at module scope, we run it once and validate many aspects of the same output.

### HOW Components Connect

```
LocalCSVConnector
       ↓
TranscriptData (firms dict)
       ↓
FirmProcessor.process(FirmTranscriptData)
       ↓
BERTopicModel.fit_transform(sentences)
       ↓
FirmTopicOutput JSON → S3
       ↓
ThemeAggregator.aggregate(List[FirmTopicOutput])
       ↓
BERTopicModel.fit_transform(topic representations)
       ↓
ThemeOutput JSON → DynamoDB
```

### WHAT Phase 4 Needs to Build

From the approved plan (lines 1025-1200):
1. **Storage Module**: S3 bucket with lifecycle, DynamoDB with GSI
2. **Compute Module**: AWS Batch compute environments, job definitions for map + reduce containers
3. **Orchestration Module**: Step Functions for workflow (may be Phase 5)
4. **IAM**: Roles for Batch, S3, DynamoDB access
5. **Variables/Outputs**: Proper Terraform structure

The containers are ready. Phase 4 is purely infrastructure.
</Understanding>

<Next_Steps>
### Your Immediate Task: Phase 4 - Terraform Infrastructure

**Read the approved plan first:** `docs/packages/cloud_migration_package/plan.md` lines 1025-1200

**CRITICAL FIRST STEP: Research Official Terraform Documentation**

Before writing ANY Terraform code, you MUST use `WebSearch` and `WebFetch` to research official HashiCorp/AWS provider documentation (registry.terraform.io) for each resource type:

| Resource | Documentation to Research |
|----------|---------------------------|
| `aws_s3_bucket` | Current bucket creation syntax |
| `aws_s3_bucket_lifecycle_configuration` | Lifecycle rules (separate resource) |
| `aws_dynamodb_table` | Table with GSI configuration |
| `aws_batch_compute_environment` | Managed SPOT environment setup |
| `aws_batch_job_queue` | Job queue configuration |
| `aws_batch_job_definition` | Container job definitions |
| `aws_ecr_repository` | Container registry |
| `aws_iam_role` / `aws_iam_policy` | IAM resources and policies |

**Why:** The plan's Terraform snippets are *guidance* that may be outdated. Terraform AWS provider updates frequently with deprecations and new required arguments. Verify syntax against current official docs.

**Implementation Order:**

1. **Research Terraform documentation** (MANDATORY - see above)

2. **Create Terraform directory structure:**
   ```
   cloud/terraform/
   ├── main.tf
   ├── variables.tf
   ├── outputs.tf
   ├── modules/
   │   ├── storage/
   │   ├── compute/
   │   ├── iam/
   │   └── orchestration/
   └── environments/
       └── dev.tfvars
   ```

3. **Storage Module (First):**
   - S3 bucket for intermediate results (firm topic JSONs)
   - DynamoDB table with PK/SK and GSI1 (per schema in plan)
   - Lifecycle rules for S3 cleanup

4. **Compute Module (Second):**
   - ECR repositories for map and reduce containers
   - Batch compute environment (SPOT instances - required)
   - Batch job queues
   - Batch job definitions referencing containers

5. **IAM Module (Third):**
   - Batch service role
   - Batch execution role
   - Job role with S3/DynamoDB permissions

6. **Verification:**
   - Run `terraform init`
   - Run `terraform plan` (should succeed without errors)
   - Do NOT run `terraform apply` without user approval

**Halting Point:** After `terraform plan` succeeds, HALT for approval before Phase 5.

**Key Resources:**
- DynamoDB schema: plan.md lines 727-831
- Terraform modules: plan.md lines 1025-1200
- Instance sizing: m5.xlarge (map), m5.2xlarge (reduce)
- **Official docs**: registry.terraform.io/providers/hashicorp/aws/latest/docs
</Next_Steps>

</Handover>

---

### **Action Directive**

1. Review the Handover Report above to fully absorb the context.
2. Read Phase 4 section of the approved plan: `docs/packages/cloud_migration_package/plan.md` (lines 1025-1200)
3. Verify existing code works: `python -m pytest tests/unit/ tests/integration/ -v`
4. **MANDATORY: Use WebSearch and WebFetch to research official Terraform AWS provider documentation** for each resource type before writing code
5. Create Terraform directory structure and implement modules
6. Run `terraform plan` to validate before requesting approval
7. HALT after Phase 4 - do not proceed to Phase 5 without explicit approval

**The containers are ready and tested. Phase 4 is purely infrastructure-as-code. Do NOT copy plan snippets verbatim - verify against official Terraform docs.**
