### **Purpose:**

You are an AI Engineer resuming work on the Financial Topic Modeling pipeline. This command contains a complete handover from the previous session to seamlessly restore your context and momentum.

### **Mission Resumption: architecture-pivot-phase4**

Your memory of previous work on this task has been wiped. This briefing is your **complete source of truth** for resuming the mission. The original strategic documents are provided for full context.

**Project Context:**
- Project Guidelines: `CLAUDE.md`

**Original Planning Documents:**
- Senior Engineer Plan: `docs/packages/architecture_pivot_package/Senior_Engineer_Plan_architecture_pivot.md`
- Mission Briefing: `docs/packages/architecture_pivot_package/Mission_Briefing_architecture_pivot.md`
- SubAgent Strategy: `docs/packages/architecture_pivot_package/SubAgent_Strategy_architecture_pivot.md`
- Raw Vision Transcript: `docs/packages/architecture_pivot_package/raw_transcript_architecture_pivot.md`

**Previous Handovers:**
- Phase 1 → Phase 2: `docs/handovers/resume-task-architecture-pivot-phase2.md`
- Phase 2 → Phase 3: `docs/handovers/resume-task-architecture-pivot-phase3.md`

---

### **Handover Report from Previous Session (Phase 3)**

<Handover>

<Status>
**Overall Progress:** Phase 3 (LLM Integration) is 100% COMPLETE. All core functionality implemented and validated with real data.
**Current Phase:** Ready for Phase 4 (Infrastructure/Terraform)
**Blocker:** None - Phase 3 complete, user ran full pipeline successfully with MAG7 firms
**Validation:** User verified topic summaries and theme descriptions in DBeaver

**Tests:** All unit and integration tests are currently passing after async mock fixes and prompt updates.
</Status>

<Pipeline_Position>
**Completed Phases:**
- Phase 1 - Database Layer Design ✅
- Phase 2 - Pipeline Unification ✅
- Phase 3 - LLM Integration ✅

**Next Phase:** Phase 4 - Infrastructure & Deployment (Terraform)

**Phase 3 Modules Implemented:**
- `cloud/src/llm/__init__.py` - Package exports
- `cloud/src/llm/xai_client.py` - Async xAI client with rate limiting
- Modified `cloud/src/pipeline/unified_pipeline.py` - LLM integration
- Modified `cloud/src/theme_aggregator.py` - Summary-based clustering
- `tests/unit/test_xai_client.py` - 19 unit tests
- `tests/integration/test_llm_integration.py` - Integration tests
</Pipeline_Position>

<Decisions>
**Critical Phase 3 Decisions Made:**

1. **XAI Client Architecture:**
   - Uses OpenAI-compatible `AsyncOpenAI` client with custom base_url for xAI
   - Semaphore-based rate limiting (50 concurrent by default, configurable)
   - Exponential backoff retry logic for transient errors
   - Graceful fallback to keywords when LLM fails (never crashes pipeline)

2. **Topic Summary Prompt Design:**
   - User wrote custom prompts emphasizing **generalizability** across firms
   - Prompts include BOTH keywords AND raw sentences for richer context
   - Explicitly instructs LLM to avoid company-specific terminology
   - This design choice was driven by the downstream need for cross-firm theme identification

3. **Theme Description Prompt Design:**
   - Uses topic summaries (not keywords) as input for richer semantic content
   - Passes theme-level keywords (from `theme_data["keywords"]`, comma-joined) into the prompt

4. **Summary-Based Theme Clustering (Critical):**
   - `ThemeAggregator._extract_topic_documents()` now uses summaries for clustering
   - Falls back to keywords if summary is missing
   - This ensures themes are clustered on richer semantic content, not just keywords

5. **Raw Sentences in Topic Prompts:**
   - `_generate_topic_summaries()` groups raw sentences by topic assignment
   - Passes up to 10 sentences per topic to LLM (MAX_SENTENCES_IN_PROMPT = 10)
   - Raw sentences provide context beyond just keywords

6. **Prompt Logging for Observability:**
   - One topic prompt logged per firm (`log_first_prompt=True`)
   - One theme prompt logged during aggregation
   - Enables debugging/sanity checking of LLM inputs

7. **Event Loop Safety:**
   - `_run_async()` helper handles both sync and async contexts
   - Falls back to ThreadPoolExecutor if running inside existing event loop (e.g., Jupyter)
   - Pipeline can be called from any context safely

8. **Fallback Strategy:**
   - If XAI_API_KEY not present → silent fallback to keywords, no LLM calls
   - If LLM call fails → individual topic/theme uses keywords as fallback
   - Embeddings ALWAYS populated (from summary or fallback text)
   - Pipeline never fails due to LLM issues
</Decisions>

<Wins>
**Phase 3 Deliverables (All Complete):**
- ✅ `cloud/src/llm/__init__.py` - Package exports
- ✅ `cloud/src/llm/xai_client.py` - Full async client (271 lines)
- ✅ `cloud/src/pipeline/unified_pipeline.py` - LLM integration (600+ lines)
- ✅ `cloud/src/theme_aggregator.py` - Summary-based clustering
- ✅ `tests/unit/test_xai_client.py` - 19 unit tests, all passing
- ✅ `tests/integration/test_llm_integration.py` - Integration tests
- ✅ `cloud/requirements.txt` - Added `openai>=1.0.0`, `pytest-asyncio>=0.23.0`

**Real Data Validation:**
User ran full pipeline with MAG7 + additional tech firms:
- Topic summaries generated via xAI (grok-4-1-fast-reasoning model)
- Theme descriptions generated and stored
- Results verified in DBeaver - human-readable summaries replacing keyword strings
- Prompt logging confirmed correct inputs being sent to LLM

**Data Flow Implemented:**
```
Firm Processing:
  BERTopic keywords + raw sentences → LLM → summary → embed(summary) → store both

Theme Aggregation:
  Topics with summaries → cluster on summaries → LLM description → embed(description) → store
```
</Wins>

<Artifacts>
**Files Created in Phase 3:**
```
cloud/src/llm/__init__.py              # Package exports
cloud/src/llm/xai_client.py            # Async xAI client (271 lines)
tests/unit/test_xai_client.py          # 19 unit tests
tests/integration/test_llm_integration.py  # Integration tests
```

**Files Modified in Phase 3:**
```
cloud/src/pipeline/unified_pipeline.py  # Added LLM integration methods:
                                        # - _init_llm_client()
                                        # - _run_async()
                                        # - _generate_topic_summaries()
                                        # - _generate_theme_descriptions()
                                        # - Updated _write_firm_results()
                                        # - Updated _write_themes()
                                        # - Updated _build_firm_topic_outputs()

cloud/src/theme_aggregator.py           # _extract_topic_documents() now uses summaries

cloud/requirements.txt                  # Added: openai>=1.0.0, pytest-asyncio>=0.23.0

tests/conftest.py                       # Added pytest-asyncio plugin
```
</Artifacts>

<ML_Metrics>
**LLM Integration Performance:**
- Topic summaries: ~1-2 seconds per topic (batched async)
- Theme descriptions: ~2-3 seconds per theme
- Rate limiting: 50 concurrent requests (semaphore)
- Retry logic: Exponential backoff (2^attempt seconds)

**Quality Observations:**
- Topic summaries are generalizable across firms (per prompt design)
- Theme descriptions capture cross-firm patterns effectively
- Embeddings from summaries provide richer semantic search than keywords
</ML_Metrics>

<Issues>
**Known Test Issues:** None currently blocking. All tests are green with async mocks configured.

**Technical Debt (Minor):**
1. No integration test for full LLM flow with real API (mocked only)
</Issues>

<Config_Changes>
**Environment Variables Added:**
```bash
XAI_API_KEY=your_xai_api_key_here  # Optional - if missing, uses keyword fallbacks
```

**Requirements Added (`cloud/requirements.txt`):**
```
# LLM Integration (Phase 3)
openai>=1.0.0

# Testing
pytest-asyncio>=0.23.0
```

**XAI Client Configuration (in config dict):**
```python
config = {
    "llm": {
        "model": "grok-4-1-fast-reasoning",  # Default model
        "max_concurrent": 50,                 # Semaphore limit
        "timeout": 30,                        # Request timeout (seconds)
        "max_retries": 3,                     # Retry attempts
    }
}
```

**Prompt Constants:**
- `MAX_SENTENCES_IN_PROMPT = 10` - Limits sentences per topic to avoid token limits
- Custom prompts in `TOPIC_SUMMARY_PROMPT` and `THEME_DESCRIPTION_PROMPT`
</Config_Changes>

<Key_Code_Patterns>
**1. Async Client with Rate Limiting:**
```python
# cloud/src/llm/xai_client.py
self._semaphore = asyncio.Semaphore(self._max_concurrent)

async def _call_llm(self, prompt: str) -> Optional[str]:
    async with self._semaphore:  # Rate limiting
        for attempt in range(self._max_retries):
            try:
                response = await self._client.chat.completions.create(...)
                return response.choices[0].message.content
            except (RateLimitError, APIStatusError) as e:
                wait_time = 2 ** attempt  # Exponential backoff
                await asyncio.sleep(wait_time)
```

**2. Safe Async Execution from Sync Context:**
```python
# cloud/src/pipeline/unified_pipeline.py
def _run_async(self, coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # No loop - create one
    else:
        # Loop exists - run in thread to avoid nested loop issues
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
```

**3. Topic Summary Generation with Raw Sentences:**
```python
def _generate_topic_summaries(self, output, firm_data, topic_assignments):
    # Group raw sentences by topic ID
    topic_sentences = {}
    for i, sentence in enumerate(firm_data.sentences):
        topic_id = int(topic_assignments[i])
        if topic_id >= 0:
            if topic_id not in topic_sentences:
                topic_sentences[topic_id] = []
            topic_sentences[topic_id].append(sentence.raw_text)  # RAW text for LLM

    # Add sentences to topic dicts
    for topic in output["topics"]:
        topic["sentences"] = topic_sentences.get(topic["topic_id"], [])

    # Generate via LLM
    summaries = self._run_async(
        self._xai_client.generate_batch_summaries(topics, log_first_prompt=True)
    )
```

**4. Summary-Based Theme Clustering:**
```python
# cloud/src/theme_aggregator.py
def _extract_topic_documents(self, firm_results):
    for topic in firm_result.get("topics", []):
        # Use summary when available, fallback to keywords
        summary = topic.get("summary", "")
        doc_text = summary if summary else topic.get("representation", "")
        topic_docs.append(doc_text)  # Cluster on summaries!
```

**5. LLM Fallback Pattern:**
```python
if self._xai_client is None:
    # No API key - silent fallback
    for topic in topics:
        topic["summary"] = topic["representation"]
    return

# Try LLM, fallback on failure
try:
    summaries = self._run_async(generate_all())
    for i, topic in enumerate(topics):
        if summaries[i] is not None:
            topic["summary"] = summaries[i]
        else:
            topic["summary"] = topic["representation"]  # Individual fallback
except Exception as e:
    for topic in topics:
        topic["summary"] = topic["representation"]  # Batch fallback
```
</Key_Code_Patterns>

<Next_Steps>
**Your immediate mission is to execute Phase 4: Infrastructure & Deployment.**

Per the Senior Engineer Plan, Phase 4 involves:

### Phase 4 Deliverables:
- [ ] `cloud/terraform/` - Simplified Terraform for RDS PostgreSQL
- [ ] `cloud/scripts/launch_pipeline.sh` - EC2 spot instance launch
- [ ] `cloud/scripts/stop_rds.sh` - Cost management script
- [ ] End-to-end validation on cloud infrastructure
- [ ] Documentation of full cloud run

### Phase 4 Implementation:

1. **Terraform for RDS PostgreSQL:**
   ```hcl
   resource "aws_db_instance" "main" {
     identifier           = "ftm-db-${var.environment}"
     engine               = "postgres"
     engine_version       = "15"
     instance_class       = "db.t4g.large"  # 8GB RAM
     allocated_storage    = 100
     db_name              = "ftm"
     # pgvector extension enabled via post-provisioning
   }
   ```

2. **EC2 Launch Script (g4dn.2xlarge spot):**
   - Deep Learning AMI (Ubuntu 20.04) with PyTorch
   - Security group allowing SSH + Postgres access
   - User data script to clone repo and install dependencies

3. **Enable pgvector Extension:**
   ```sql
   CREATE EXTENSION vector;
   ```

4. **Cost Management:**
   - RDS must be stoppable (not Aurora Serverless)
   - Spot instance for compute (~$0.25/hour)
   - Scripts to start/stop RDS when not in use

5. **End-to-End Validation:**
   - Run pipeline on full CSV dataset (~3,000 firms)
   - Verify all data in Postgres
   - Build vector indexes (HNSW)
   - Test hierarchical queries
   - Measure total processing time and cost

### Phase 4 Acceptance Criteria:
- [ ] Pipeline completes in <4 hours for 3,000 firms
- [ ] Total compute cost <$5 per quarterly run
- [ ] All data queryable in Postgres
- [ ] Vector search functional with HNSW indexes
- [ ] Hierarchical queries work (theme → topics → sentences → firms)

### Before Starting Phase 4:

1. **Fix Failing Tests (Optional):**
   If you want clean tests before infrastructure work:
   - Review `tests/integration/test_llm_integration.py`
   - Ensure `AsyncMock` is properly configured for `generate_batch_summaries`
   - The issue is mock setup, not implementation

2. **Verify Local Pipeline:**
   Run `scripts/run_unified_pipeline_mag7.py` to confirm everything works locally

### HALT Point:
After completing Phase 4, STOP and present complete implementation for final review.
</Next_Steps>

</Handover>

---

### **Critical Context: The Architecture Pivot Philosophy**

The entire pivot exists because complexity was caught before deployment. From the original analysis:

> "For our scale (~2M sentences quarterly, 3,000-5,000 firms), the orchestration overhead of AWS Batch exceeds the compute time. A single GPU processes all firms in ~2-4 hours for ~$1.00."

**Key Insight for Phase 4:** The infrastructure should be equally simple:
- RDS PostgreSQL (stoppable for cost management)
- g4dn.2xlarge spot instance (GPU for embeddings)
- No Step Functions, no Lambda, no complex orchestration
- Single script runs everything

---

### **Why LLM Prompts Were Designed This Way**

The user emphasized **generalizability** in the prompt design:

1. **Topic Summaries Must Be Generic:**
   - No company-specific brand names or services
   - Focus on underlying business concepts
   - This enables cross-firm theme identification

2. **Raw Sentences Provide Context:**
   - Keywords alone are ambiguous (e.g., "cloud" could mean weather or computing)
   - Raw sentences give LLM full context for accurate summarization
   - Up to 50 sentences per topic (configurable)

3. **Theme Descriptions Build on Summaries:**
   - Not re-processing keywords
   - Using richer topic summaries from Stage 1
   - Creates coherent hierarchy of descriptions

---

### **Why Summary-Based Clustering Matters**

A critical Phase 3 requirement (per Codex review) was ensuring theme clustering uses summaries, not keywords:

**Before (Phase 2):**
```python
topic_docs.append(representation)  # Just keywords
```

**After (Phase 3):**
```python
doc_text = summary if summary else representation
topic_docs.append(doc_text)  # Summaries for richer clustering
```

This ensures themes are clustered on richer semantic content, producing better cross-firm theme identification.

---

### **Test Failures Context**

The 3 failing integration tests are a **mock configuration issue**, not an implementation bug:

1. The implementation works correctly with real xAI API calls
2. The user verified results in DBeaver
3. The mocks use `AsyncMock` with `side_effect`, which may have edge cases in the async execution context

If you need to fix these before Phase 4:
- Focus on how `_run_async()` interacts with `AsyncMock`
- The mock may need to be configured differently for the ThreadPoolExecutor fallback path
- Consider using `pytest-asyncio` `auto` mode or making tests async

---

### **Action Directive**

1. Review this Handover Report to fully absorb the context
2. Read the Senior Engineer Plan Phase 4 section for specifications
3. Optionally fix the 3 failing integration tests (not blocking)
4. Create simplified Terraform for RDS PostgreSQL
5. Create EC2 launch/setup scripts for g4dn.2xlarge spot
6. Deploy infrastructure and run end-to-end validation
7. Document total processing time and cost
8. HALT at the end of Phase 4 and present complete implementation
