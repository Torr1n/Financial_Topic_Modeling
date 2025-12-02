### **Purpose:**

You are an AI Engineer resuming work on the Financial Topic Modeling cloud migration. This command contains a complete handover from the previous session to seamlessly restore your context and momentum.

### **Mission Resumption: cloud-migration-phase3**

Your memory of previous work on this task has been wiped. This briefing is your **complete source of truth** for resuming the mission. The original strategic documents are provided for full context.

**Project Context:**
- Project Guidelines: `CLAUDE.md`
- Vision Document: `First_Transcript.md`

**Original Planning Documents:**
- Approved Plan (AUTHORITATIVE): `docs/packages/cloud_migration_package/plan.md`
- Mission Briefing: `docs/packages/cloud_migration_package/Mission_Briefing_cloud_migration.md`
- Strategy: `docs/packages/cloud_migration_package/SubAgent_Strategy_cloud_migration.md`
- Previous Handover: `docs/handovers/resume-task-cloud-migration-phase1.md`

---

### **Handover Report from Previous Session**

<Handover>

<Status>
**Overall Progress:** Phases 1 and 2 COMPLETE and APPROVED. BERTopic improvements COMPLETE. Phase 3 (Reduce Container) is NEXT.
**Current Phase:** About to begin Phase 3 implementation
**Blocker:** None - ready to proceed

**Critical Context:** This is a CLOUD MIGRATION project. The MVP code at `Local_BERTopic_MVP/` is EXPLICITLY marked as poorly structured and should NOT be ported. We build CLEAN, SIMPLE code following the approved plan schemas.

**Recent Work (Post-Phase 2):** BERTopic marginal improvements were implemented before Phase 3:
- TopicModelResult.probabilities is now REQUIRED (not optional)
- BERTopicModel uses enhanced representations (KeyBERT, MMR, POS)
- FirmProcessor orders sentence_ids by probability (highest first)
- CountVectorizer added with ngram_range=(1,2), min_df=2
</Status>

<Pipeline_Position>
**Completed Phases:**
- Phase 1: Abstraction Layer (models, interfaces, BERTopicModel, LocalCSVConnector)
- Phase 2: Map Container (FirmProcessor, s3_utils, dynamodb_utils, entrypoint, sentence splitting fix)
- Post-Phase 2: BERTopic improvements (representations, probabilities, sentence ordering)

**Next Phase:** Phase 3 - Reduce Container
- `cloud/src/theme_aggregator.py` - Dual-BERTopic theme aggregation
- `cloud/containers/reduce/` - Dockerfile, requirements.txt, entrypoint.py
- Integration with existing `ReducePhaseDynamoDBWriter` (already implemented and tested)

**Downstream:** Phase 4 is Terraform/AWS infrastructure (after Phase 3 approval)
</Pipeline_Position>

<Decisions>
### Architectural Decisions (Locked)

1. **NO CENTROIDS** - The old MVP used 768-dim centroid embeddings for cross-firm comparison. The approved plan uses Dual-BERTopic instead: topic representations (strings) are RE-EMBEDDED as new documents in the reduce phase. This is cleaner and model-agnostic.

2. **Single-Table DynamoDB Design** - One table with composite keys (PK/SK) enables hierarchical queries:
   - Theme metadata: `PK=THEME#theme_001, SK=METADATA`
   - Topic links: `PK=THEME#theme_001, SK=TOPIC#firm_id#topic_id`
   - Sentences: `PK=TOPIC#firm_id#topic_id, SK=SENTENCE#sentence_id` (written by MAP phase)

3. **firm_id from companyid** - We use the CSV `companyid` column as `firm_id`, NOT company names. This enables reliable lookups.

4. **SpaCy Sentence Splitting** - Each CSV row is a "transcript component" containing multiple sentences. We use SpaCy's English sentencizer to split components into individual sentences before topic modeling.

5. **Stopword Removal** - Topic modeling receives stopword-cleaned text for better topic quality. SpaCy's built-in stopword list is used.

6. **TDD Mandatory** - Tests are written BEFORE implementation. Target 80%+ coverage.

### Phase 2 Corrections Made

During Phase 2, we discovered and fixed two critical issues:
1. **Transcript components ≠ sentences**: Original code treated each CSV row as a sentence. Fixed by adding SpaCy sentence splitting.
2. **Stopword noise**: Topics were dominated by stopwords. Fixed by adding stopword removal preprocessing.

These fixes improved Microsoft's sentence count from 41 → 447 and topic quality significantly improved.

### Post-Phase 2: BERTopic Improvements

Before Phase 3, marginal BERTopic improvements were implemented:

7. **Probabilities Required** - TopicModelResult.probabilities is now a REQUIRED field (n_docs x n_topics matrix). This enables probability-based sentence ordering and downstream processing.

8. **Enhanced Representations** - BERTopicModel uses multiple representation models:
   - KeyBERTInspired: Embedding-based keyword extraction
   - MaximalMarginalRelevance (diversity=0.3): Diverse keyword selection
   - PartOfSpeech: Noun/noun-phrase filtering
   - Custom labels concatenate top keywords from all three

9. **Sentence Ordering** - FirmProcessor now orders sentence_ids by probability (highest first) within each topic. The first sentence_id is the most representative for that topic.

10. **CountVectorizer** - Explicit vectorizer with ngram_range=(1,2), min_df=2 for richer topic representations.

11. **Pre-computed Embeddings** - BERTopicModel pre-computes embeddings before fit_transform for efficiency.
</Decisions>

<Wins>
### Phase 1 Completed
- `cloud/src/models.py` - TranscriptSentence, FirmTranscriptData, TranscriptData, TopicModelResult
- `cloud/src/interfaces.py` - TopicModel ABC, DataConnector ABC
- `cloud/src/topic_models/bertopic_model.py` - BERTopicModel implementation
- `cloud/src/connectors/local_csv.py` - LocalCSVConnector with SpaCy sentence splitting
- Stubs: LDATopicModel, NeuralTopicModel, S3TranscriptConnector
- `cloud/config/default.yaml` - UMAP/HDBSCAN parameters from MVP

### Phase 2 Completed
- `cloud/src/firm_processor.py` - FirmProcessor producing FirmTopicOutput schema
- `cloud/src/s3_utils.py` - upload_json, download_json, list_json_files
- `cloud/src/dynamodb_utils.py` - MapPhaseDynamoDBWriter AND ReducePhaseDynamoDBWriter (both tested)
- `cloud/containers/map/` - Dockerfile, requirements.txt, entrypoint.py
- `cloud/scripts/local_test_map.sh` - Local testing script
- `cloud/scripts/benchmark_instance_sizes.md` - Benchmarking guide

### Post-Phase 2: BERTopic Improvements
- Updated `cloud/src/models.py` - TopicModelResult.probabilities now REQUIRED
- Updated `cloud/src/topic_models/bertopic_model.py` - Representations, CountVectorizer, pre-computed embeddings
- Updated `cloud/src/firm_processor.py` - Probability-based sentence ordering
- Updated `cloud/config/default.yaml` - Added vectorizer and representation config sections
- Updated tests in conftest.py, test_models.py, test_firm_processor.py, test_bertopic_model.py

### Test Coverage: 92%
- 27+ tests passing (unit tests for models, firm_processor updated for new behavior)
- All unit tests for models, interfaces, connectors, firm_processor, s3_utils, dynamodb_utils
- Integration tests for map entrypoint in local mode
- Note: map/entrypoint.py at 73% (guard branches untested - not a blocker)
</Wins>

<Artifacts>
**Phase 1 Files:**
```
cloud/src/models.py
cloud/src/interfaces.py
cloud/src/topic_models/bertopic_model.py
cloud/src/topic_models/lda_model.py (stub)
cloud/src/topic_models/neural_model.py (stub)
cloud/src/connectors/local_csv.py
cloud/src/connectors/s3_connector.py (stub)
cloud/config/default.yaml
cloud/requirements.txt
```

**Phase 2 Files:**
```
cloud/src/firm_processor.py
cloud/src/s3_utils.py
cloud/src/dynamodb_utils.py
cloud/containers/map/Dockerfile
cloud/containers/map/requirements.txt
cloud/containers/map/entrypoint.py
cloud/scripts/local_test_map.sh
cloud/scripts/benchmark_instance_sizes.md
```

**Test Files:**
```
tests/conftest.py
tests/unit/test_models.py
tests/unit/test_interfaces.py
tests/unit/test_bertopic_model.py
tests/unit/test_connectors.py
tests/unit/test_firm_processor.py
tests/unit/test_s3_utils.py
tests/unit/test_dynamodb_utils.py
tests/unit/test_stubs.py
tests/integration/test_map_entrypoint.py
```

**Sample Output Generated:**
- `output/map_test/21835_topics.json` - Microsoft with 29 topics from 447 sentences
</Artifacts>

<ML_Metrics>
**Map Phase Validated:**
- Microsoft (firm 21835): 36 components → 447 sentences → 29 topics, 104 outliers
- Topics are cleaner without stopword noise
- BERTopic parameters from MVP config work well: UMAP(n_neighbors=15, n_components=10), HDBSCAN(min_cluster_size=6, min_samples=2)
- **NEW:** Topic representations now use combined KeyBERT+MMR+POS keywords
- **NEW:** sentence_ids ordered by probability (first = most representative)
- **NEW:** Full (n_docs x n_topics) probability matrix available in TopicModelResult

**Reduce Phase Metrics (To Be Validated):**
- Theme validation filters: min_firms=2, max_firm_dominance=0.4
- These will filter out single-firm themes and themes dominated by one firm
- Note: ThemeAggregator will receive TopicModelResult with required probabilities
</ML_Metrics>

<Issues>
### Known Limitations (Not Blockers)

1. **SpaCy Double Parse** - The connector parses text twice (once for sentence splitting, once for stopword removal). Acceptable for now but could be optimized.

2. **Coverage Gaps** - Overall 92%. map/entrypoint.py at 73% (guard branches untested). Not blocking.

3. **S3TranscriptConnector** - Still a stub. Will need implementation for full cloud deployment but not needed for Phase 3.

4. **Map Runtime** - ~2.5 minutes for 447 docs due to pre-computed embeddings + approximate_distribution. Acceptable but noted.

### Not Started
- Phase 3: Reduce Container (ThemeAggregator, reduce entrypoint)
- Phase 4: Terraform infrastructure
- Phase 5: Orchestration
</Issues>

<Config_Changes>
**cloud/config/default.yaml** contains locked parameters:
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
# NEW: CountVectorizer settings (Post-Phase 2)
vectorizer:
  ngram_range: [1, 2]
  min_df: 2
# NEW: Representation model settings (Post-Phase 2)
representation:
  mmr_diversity: 0.3
  pos_model: "en_core_web_sm"
validation:
  min_firms: 2
  max_firm_dominance: 0.4
```

**cloud/requirements.txt** and **cloud/containers/map/requirements.txt** include:
- bertopic, sentence-transformers, umap-learn, hdbscan, scikit-learn, numpy
- spacy (added for sentence splitting)
- pandas, pyyaml, boto3, tqdm
- pytest, pytest-cov, moto (testing)
</Config_Changes>

<Next_Steps>
**Your immediate task is to implement Phase 3: Reduce Container.**

### Step-by-Step Instructions:

1. **Re-read Phase 3 of the approved plan** (lines 645-1024 of `docs/packages/cloud_migration_package/plan.md`)

2. **Write tests FIRST (TDD):**
   - `tests/unit/test_theme_aggregator.py` - ThemeAggregator unit tests
   - `tests/integration/test_reduce_entrypoint.py` - Reduce entrypoint integration tests
   - Use moto for S3/DynamoDB mocks

3. **Implement ThemeAggregator** (`cloud/src/theme_aggregator.py`):
   - Dual-BERTopic approach: topic representations → re-embed → cluster into themes
   - Apply validation filters: min_firms, max_firm_dominance
   - Generate ThemeOutput dicts with theme_id, name, keywords, topics list

4. **Create reduce container:**
   - `cloud/containers/reduce/Dockerfile`
   - `cloud/containers/reduce/requirements.txt`
   - `cloud/containers/reduce/entrypoint.py`
   - Supports LOCAL_MODE for testing, cloud mode for S3/DynamoDB

5. **Test locally:**
   - Use output from map phase (`output/map_test/*.json`) as input
   - Verify themes are generated and match schema
   - Run with mocked DynamoDB to verify writes

6. **HALT after Phase 3** for approval before Terraform (Phase 4)

### Key Schema Reference (from plan):

**ThemeOutput:**
```python
{
    "theme_id": "theme_20241130_001",
    "name": "AI Investment Strategy",
    "keywords": ["ai", "machine", "learning", ...],
    "n_firms": 5,
    "n_topics": 12,
    "topics": [
        {"firm_id": "1001", "topic_id": 0, "representation": "...", "size": 25},
        ...
    ],
    "metadata": {...}
}
```

### Important Reminders:
- ReducePhaseDynamoDBWriter is ALREADY IMPLEMENTED and TESTED (96% coverage)
- DO NOT port MVP code - write clean, simple implementations
- Use dependency injection for TopicModel (same pattern as FirmProcessor)
- Handle missing/corrupt firm files gracefully (skip + log)
</Next_Steps>

</Handover>

---

### **Action Directive**

1. Review the Handover Report above to fully absorb the context.
2. Read Phase 3 of the approved plan: `docs/packages/cloud_migration_package/plan.md` (lines 645-1024)
3. Begin with TDD: write tests for ThemeAggregator and reduce entrypoint
4. Implement the reduce container following the plan specification
5. Test locally with map phase output files
6. HALT for approval before proceeding to Phase 4 (Terraform)

**The reduce phase reuses the same TopicModel interface and BERTopicModel implementation from Phase 1. The key insight is that topic representations become the "documents" for theme-level clustering.**
