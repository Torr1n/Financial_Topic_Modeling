### **Purpose:**

You are an AI Engineer resuming work on the Financial Topic Modeling pipeline. This command contains a complete handover from the previous session to seamlessly restore your context and momentum.

### **Mission Resumption: architecture-pivot-phase3**

Your memory of previous work on this task has been wiped. This briefing is your **complete source of truth** for resuming the mission. The original strategic documents are provided for full context.

**Project Context:**
- Project Guidelines: `CLAUDE.md`

**Original Planning Documents:**
- Senior Engineer Plan: `docs/packages/architecture_pivot_package/Senior_Engineer_Plan_architecture_pivot.md`
- Mission Briefing: `docs/packages/architecture_pivot_package/Mission_Briefing_architecture_pivot.md`
- SubAgent Strategy: `docs/packages/architecture_pivot_package/SubAgent_Strategy_architecture_pivot.md`
- Raw Vision Transcript: `docs/packages/architecture_pivot_package/raw_transcript_architecture_pivot.md`
- Gemini Analysis: `docs/ai-log/gemini-conversation.md`

**Previous Handovers:**
- Phase 1 → Phase 2: `docs/handovers/resume-task-architecture-pivot-phase2.md`

---

### **Handover Report from Previous Session (Phase 2)**

<Handover>

<Status>
**Overall Progress:** Phase 2 (Pipeline Unification) is 100% COMPLETE. All tests passing, validated with real data.
**Current Phase:** Ready for Phase 3 (LLM Integration)
**Blocker:** None - Phase 2 complete, user ran full pipeline successfully with 11 tech firms
**Validation:** User verified results in DBeaver - 19 themes, 350 topics, 5,014 sentences with embeddings
</Status>

<Pipeline_Position>
**Completed Phases:**
- Phase 1 - Database Layer Design ✅
- Phase 2 - Pipeline Unification ✅

**Next Phase:** Phase 3 - LLM Integration

**Phase 2 Modules Implemented:**
- `cloud/src/pipeline/unified_pipeline.py` - Main orchestration (320 lines)
- `cloud/src/pipeline/checkpoint.py` - Resume logic
- `cloud/src/pipeline/__init__.py` - Package exports
- Modified `cloud/src/firm_processor.py` - Now accepts/returns embeddings
- Modified `cloud/src/topic_models/bertopic_model.py` - Accepts pre-computed embeddings
- Modified `cloud/src/interfaces.py` - Updated interface for embeddings
- `tests/integration/test_unified_pipeline.py` - Integration tests
- `tests/unit/test_checkpoint_manager.py` - Unit tests
</Pipeline_Position>

<Decisions>
**Critical Phase 2 Decisions Made:**

1. **Embedding Model Singleton Pattern:**
   - `SentenceTransformer` loaded ONCE in `UnifiedPipeline.__init__()`
   - Passed to `FirmProcessor` via pre-computed embeddings
   - Eliminates cold start overhead from 3,000× model loads

2. **Pre-computed Embeddings Flow:**
   - Pipeline computes embeddings externally before calling `fit_transform()`
   - BERTopicModel accepts optional `embeddings` param, skips internal encoding when provided
   - Same embeddings stored to Postgres `sentences.embedding` column

3. **Topic Embeddings NOW (Phase 2):**
   - Embed topic representations (keywords) immediately in Phase 2
   - Store in `topics.embedding` column
   - **DOCUMENTED AS TEMPORARY** - Will be replaced with richer LLM summary embeddings in Phase 3

4. **Sentence→Topic Mapping (Codex Review Pattern):**
   - Insert topics FIRST, flush to get IDs
   - Insert sentences WITH `topic_id` already set
   - No insert-then-update pattern (per Codex review)

5. **Session Management:**
   - Context manager per firm for clean checkpoints
   - `mark_firm_processed()` + `commit()` after each firm
   - Enables spot instance resume

6. **Separate Topic Model Instances:**
   - `firm_topic_model` and `theme_topic_model` are separate instances
   - Avoids internal state carryover between firm and theme processing

7. **FirmProcessor Return Type Change:**
   - Changed from `Dict` to `Tuple[Dict, np.ndarray]`
   - Returns `(output_dict, topic_assignments)` for sentence→topic mapping
   - All tests and callers updated to handle tuple

8. **NLP Preprocessing Enhancement (Session End):**
   - User identified "garbage" themes (thank you, question, operator, etc.)
   - Enhanced `LocalCSVConnector` with full SpaCy pipeline:
     - Load `en_core_web_sm` for NER + lemmatization
     - Custom stopwords: "yes, thank, thanks, question, questions, afternoon, operator, welcome"
     - Filter PERSON, DATE, TIME, CARDINAL, ORDINAL, MONEY, PERCENT entities
     - Filter Operator speaker_type rows
     - Filter single-word sentences after preprocessing

9. **Dual-Text Model (Session End):**
   - `TranscriptSentence` now has `raw_text` and `cleaned_text`
   - `raw_text`: Original unprocessed sentence (for observability in DBeaver)
   - `cleaned_text`: Preprocessed text used for embeddings/topic modeling
   - Database `Sentence` model updated with both columns
</Decisions>

<Wins>
**Phase 2 Deliverables (All Complete):**
- ✅ `cloud/src/pipeline/__init__.py` - Package exports
- ✅ `cloud/src/pipeline/unified_pipeline.py` - Main orchestration (320 lines)
- ✅ `cloud/src/pipeline/checkpoint.py` - Resume logic wrapper
- ✅ `cloud/src/topic_models/bertopic_model.py` - Pre-computed embeddings support
- ✅ `cloud/src/interfaces.py` - Updated TopicModel interface
- ✅ `cloud/src/firm_processor.py` - Embeddings handling, tuple return
- ✅ `cloud/src/connectors/local_csv.py` - Enhanced NLP preprocessing
- ✅ `cloud/src/models.py` - `raw_text` + `cleaned_text` fields
- ✅ `cloud/src/database/models.py` - Database schema with both text columns
- ✅ `tests/integration/test_unified_pipeline.py` - Integration tests
- ✅ `tests/unit/test_checkpoint_manager.py` - 10 unit tests
- ✅ `scripts/run_unified_pipeline_mag7.py` - Local test runner

**Real Data Validation:**
User ran full pipeline with 11 tech firms (MAG7 + Broadcom, Arista, Cisco, Oracle):
- 11 firms processed in ~3 minutes
- 350 topics discovered
- 19 validated themes (after min_firms=2, max_dominance=0.4 filters)
- 5,014 sentences with embeddings
- All results verified in DBeaver

**Test Results:**
- All existing tests pass after TranscriptSentence signature update
- 42 tests pass in test_firm_processor.py
- Integration tests pass with testcontainers
</Wins>

<Artifacts>
**Files Created in Phase 2:**
```
cloud/src/pipeline/__init__.py           # Package exports
cloud/src/pipeline/unified_pipeline.py   # Main orchestration (320 lines)
cloud/src/pipeline/checkpoint.py         # Resume logic wrapper
tests/integration/test_unified_pipeline.py
tests/unit/test_checkpoint_manager.py
scripts/run_unified_pipeline_mag7.py     # Local test runner
```

**Files Modified in Phase 2:**
```
cloud/src/topic_models/bertopic_model.py  # Added embeddings param to fit_transform()
cloud/src/interfaces.py                   # Updated TopicModel interface
cloud/src/firm_processor.py               # Accept embeddings, return tuple
cloud/src/connectors/local_csv.py         # Enhanced NLP preprocessing
cloud/src/models.py                       # raw_text + cleaned_text
cloud/src/database/models.py              # Schema with both text columns
cloud/src/database/repository.py          # Updated for raw_text/cleaned_text in hierarchy queries
cloud/containers/map/entrypoint.py        # Updated for tuple return
tests/unit/test_bertopic_model.py         # 8 new tests for embeddings
tests/unit/test_firm_processor.py         # Updated all tests + 7 new
tests/unit/test_models.py                 # Updated for new signature
tests/unit/test_dynamodb_utils.py         # Updated for new signature
tests/unit/test_interfaces.py             # Updated for new signature
tests/conftest.py                         # Added make_sentence() helper
```

**Files Preserved (NOT Modified):**
```
cloud/src/theme_aggregator.py             # Core reduce logic unchanged
cloud/src/bertopic_model.py               # Core ML unchanged (only I/O adapted)
```
</Artifacts>

<ML_Metrics>
**Real Pipeline Run (11 firms):**
| Metric | Value |
|--------|-------|
| Total sentences | 5,014 |
| Total topics | 350 (avg 32/firm) |
| Validated themes | 19 |
| Outlier sentences | ~20% per firm |
| Processing time | ~3 minutes (CPU) |

**Theme Quality Observations:**
User identified 4 "garbage" themes before NLP preprocessing enhancement:
- Theme 4: "afternoon, day, excited, today, turn, thank, good"
- Theme 7: "question comes, questions, line, guess, second"
- Theme 15: "yes, better, course, know, right yes, high"
- Theme 19: "operator, question comes, line question"

~50/350 topics contained garbage words. NLP preprocessing enhancement addresses this by:
- Filtering Operator speaker_type
- Custom stopwords
- NER removal (names, dates)
- Single-word sentence filtering
</ML_Metrics>

<Issues>
**Resolved Issues:**
1. ✅ Embedding gap - FirmProcessor now accepts/returns embeddings
2. ✅ Sentence→topic mapping - Insert topics first, then sentences with topic_id set
3. ✅ TranscriptSentence signature - All tests updated to new 5-arg signature
4. ✅ Garbage themes - NLP preprocessing enhanced with custom stopwords, NER, filtering

**Known Technical Debt (Minor):**
1. Topic embeddings are keyword-based (Phase 2) - **Phase 3 will replace with LLM summaries**
2. Theme names are from BERTopic keywords - **Phase 3 will generate LLM descriptions**

**Not Yet Implemented (Phase 3 Scope):**
1. Async xAI client for LLM calls
2. Topic summary generation
3. Theme description generation
4. LLM-based embeddings for topics/themes
</Issues>

<Config_Changes>
**Custom Stopwords Added (`local_csv.py`):**
```python
CUSTOM_STOPWORDS = {"yes", "thank", "thanks", "question", "questions", "afternoon", "operator", "welcome"}
```

**NER Entity Types Filtered (`local_csv.py`):**
```python
_FILTERED_ENTITY_TYPES = {"PERSON", "DATE", "TIME", "CARDINAL", "ORDINAL", "MONEY", "PERCENT"}
```

**SpaCy Model Required:**
```bash
python -m spacy download en_core_web_sm
```

**Local Test Runner (`scripts/run_unified_pipeline_mag7.py`):**
```python
# 11 tech firms for testing
TEST_FIRM_IDS = [
    "21835",      # Microsoft
    "29096",      # Alphabet (Google)
    "27444752",   # Tesla, Inc.
    "18749",      # Amazon.com, Inc.
    "20765463",   # Meta Platforms, Inc.
    "32307",      # NVIDIA Corporation
    "24937",      # Apple Inc
    "25016048",   # Broadcom Inc.
    "33348547",   # Arista Networks Inc
    "19691",      # Cisco Systems, Inc.
    "22247",      # Oracle Corporation
]
```
</Config_Changes>

<Key_Code_Patterns>
**1. Pre-computed Embeddings Pattern:**
```python
# UnifiedPipeline._process_single_firm()
texts = [s.cleaned_text for s in firm_data.sentences]
sentence_embeddings = self.embedding_model.encode(texts)  # Computed ONCE
output, topic_assignments = self.firm_processor.process(firm_data, embeddings=sentence_embeddings)
```

**2. Topic-First Insert Pattern:**
```python
# UnifiedPipeline._write_firm_results()
# 1. Insert topics FIRST
for topic in output["topics"]:
    db_topic = Topic(firm_id=firm.id, ...)
    session.add(db_topic)
    session.flush()  # Get ID
    topic_id_map[topic["topic_id"]] = db_topic.id

# 2. Insert sentences WITH topic_id already set
for i, sentence in enumerate(firm_data.sentences):
    sentence_records.append({
        "topic_id": topic_id_map.get(int(topic_assignments[i])),  # Already set
        ...
    })
repo.bulk_insert_sentences(sentence_records)
```

**3. Dual-Text Model Pattern:**
```python
# TranscriptSentence
@dataclass
class TranscriptSentence:
    sentence_id: str
    raw_text: str        # Original for observability
    cleaned_text: str    # Preprocessed for ML
    speaker_type: Optional[str]
    position: int
```

**4. NLP Preprocessing Pattern:**
```python
# LocalCSVConnector._preprocess_text()
doc = self._nlp(text)
for ent in doc.ents:
    if ent.label_ in self._FILTERED_ENTITY_TYPES:
        entity_token_indices.add(...)  # Mark for removal
for token in doc:
    lemma = token.lemma_.lower()
    if lemma not in self._stopwords and not token.like_num:
        filtered_tokens.append(lemma)
```
</Key_Code_Patterns>

<Next_Steps>
**Your immediate mission is to execute Phase 3: LLM Integration.**

Per the Senior Engineer Plan, Phase 3 involves:

### Phase 3 Deliverables:
- [ ] `cloud/src/llm/xai_client.py` - Async xAI API client with rate limiting
- [ ] `cloud/src/llm/__init__.py` - Package exports
- [ ] Topic summary generation in firm processing
- [ ] Theme description generation in aggregation
- [ ] Replace keyword-based topic embeddings with LLM summary embeddings
- [ ] Tests for LLM integration

### Phase 3 Implementation:

1. **Create LLM Client Package:**
   ```
   cloud/src/llm/
   ├── __init__.py
   └── xai_client.py    # Async client with semaphore rate limiting
   ```

2. **Implement Async xAI Client:**
   ```python
   # Per Senior Engineer Plan
   import asyncio
   from openai import AsyncOpenAI  # xAI uses OpenAI-compatible API

   sem = asyncio.Semaphore(50)  # Rate limit protection

   async def generate_summary(client, topic_info: dict) -> str:
       async with sem:
           response = await client.chat.completions.create(
               model="grok-beta",
               messages=[{
                   "role": "user",
                   "content": f"Summarize this topic in 1-2 sentences: {topic_info}"
               }]
           )
           return response.choices[0].message.content
   ```

3. **Integrate into UnifiedPipeline:**
   - After firm processing: Generate topic summaries via LLM
   - Update topic embeddings from keyword-based to summary-based
   - After theme aggregation: Generate theme descriptions via LLM

4. **Update Topic/Theme Models:**
   - `Topic.summary` field already exists (nullable)
   - `Theme.description` field already exists (nullable)
   - Replace keyword embeddings with summary embeddings

5. **Environment Variables Needed:**
   ```
   XAI_API_KEY=your_key_here
   XAI_BASE_URL=https://api.x.ai/v1  # or appropriate endpoint
   ```

### Phase 3 Acceptance Criteria:
- [ ] Topic summaries generated via LLM (1-2 sentences each)
- [ ] Theme descriptions generated via LLM
- [ ] Topic embeddings based on LLM summaries (not keywords)
- [ ] Rate limiting prevents API throttling
- [ ] Async processing for performance
- [ ] Tests mock LLM calls appropriately

### HALT Point:
After completing Phase 3, STOP and await approval before proceeding to Phase 4 (Infrastructure/Terraform).
</Next_Steps>

</Handover>

---

### **Critical Context: The Architecture Pivot Philosophy**

The entire pivot exists because complexity was caught before deployment. From the original analysis:

> "For our scale (~2M sentences quarterly, 3,000-5,000 firms), the orchestration overhead of AWS Batch exceeds the compute time. A single GPU processes all firms in ~2-4 hours for ~$1.00."

**Key Insight:** Model loading once vs 3,000× eliminates cold start tax. This principle extends to LLM integration:
- Batch topic summaries efficiently
- Use async with semaphore for rate limiting
- Minimize API round-trips where possible

---

### **Why the Dual-Text Model Exists**

During Phase 2 testing, user discovered that preprocessed text (lowercase, lemmatized, stopwords removed) was being stored in Postgres. This made it hard to understand topics when viewing in DBeaver.

**Solution:** Store BOTH:
- `raw_text`: Original sentence for human readability
- `cleaned_text`: Preprocessed text for ML pipeline

The ML pipeline uses `cleaned_text` for embeddings and topic modeling. Humans view `raw_text` for understanding. This is a clean separation of concerns.

---

### **Why NLP Preprocessing Was Enhanced**

Initial run revealed "garbage" themes dominated by:
- Operator introductions ("Thank you, next question from...")
- Courtesies ("Thank you", "Good afternoon")
- Question logistics ("Our next question comes from...")

**Root Cause:** These high-frequency, low-signal phrases were forming their own clusters.

**Solution:** Enhanced preprocessing in `LocalCSVConnector`:
1. Filter Operator speaker_type entirely
2. Custom stopwords for earnings call patterns
3. NER removal for names/dates (filter "Thank you Sarah")
4. Single-word sentence filtering

This is preprocessing, not ML changes. The topic modeling algorithm remains unchanged.

---

### **Action Directive**

1. Review this Handover Report to fully absorb the context
2. Read the original Senior Engineer Plan for Phase 3 specifications
3. Read `cloud/src/pipeline/unified_pipeline.py` to understand where LLM calls integrate
4. Create the LLM client package with async xAI integration
5. Integrate topic summarization into firm processing
6. Integrate theme descriptions into aggregation
7. Update embeddings to use LLM summaries
8. Write tests with mocked LLM calls
9. HALT at the end of Phase 3 and await approval before Phase 4
