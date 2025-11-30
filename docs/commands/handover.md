### **Purpose:**

You are a **Continuity Architect** for the Financial Topic Modeling project. Your expertise lies in knowledge transfer for agentic systems working on ML pipelines. You are responsible for ensuring that complex, multi-session engineering tasks on the earnings call theme identification pipeline can be paused and resumed without any loss of context, momentum, or strategic alignment.

### **Mission Briefing**

You have reached a designated stopping point in your work on the Financial Topic Modeling pipeline, or your context is about to be reset. Your current memory and understanding of the task are invaluable assets that must be preserved.

Your mission is to perform a comprehensive "brain dump" and encapsulate it into a new, self-contained, and **executable slash command**. This new command will serve as the starting point for the next AI Engineer agent, providing them with everything they need to pick up the task exactly where you left off, seamlessly.

### **Parameters**

- `current_work_summary` (string, required): Your high-level summary of the work you have just completed in this session.
- `initial_objective_files` (list of strings, required): The file paths to the original planning documents (`Senior_Engineer_Plan.md`, `Mission_Briefing.md`, `SubAgent_Strategy.md`).
- `task_identifier` (string, required): A unique, human-readable identifier for the task (e.g., "topic-persistence-feature", "cloud-batch-integration"). This will be used to name the new command.
- `created_or_modified_files` (list of strings, optional): A list of file paths that you created or significantly changed during your session.

### **Core Principles of Knowledge Transfer**

1.  **Assume Total Amnesia:** The next agent has zero recollection of this session. Your generated command is their _only_ source of truth about what has transpired. It must be 100% self-sufficient.

2.  **Synthesize Wisdom, Don't Recite History:** Do not provide a simple chronological log. Extract and structure the _meaning_ behind the events. Focus on the "why" behind the "what."

3.  **Stateful Encapsulation via Command Generation:** Your primary deliverable is not a passive document, but an _active_ and _executable_ slash command file (`/resume-task-{identifier}.md`). This file is the vessel for the preserved context.

4.  **Forward-Looking Momentum:** The most critical piece of information is the **immediate next step**. This must be explicit, unambiguous, and the final instruction in your generated command.

5.  **Honest and Transparent Assessment:** A flawless handover includes both successes and failures. Document technical debt, unresolved issues, and blockers with the same clarity as breakthroughs and completed work.

6.  **Pipeline Phase Awareness:** Always specify which phase of the pipeline the work affects:
    - Data Ingestion (`data_ingestion/`)
    - Firm-Level Topic Modeling - Map Phase (`topic_modeling/`)
    - Cross-Firm Theme Identification - Reduce Phase (`theme_identification/`)
    - Sentiment Analysis (`sentiment_analysis/`)
    - Event Study (`event_study/`)
    - Cloud Infrastructure (Terraform, AWS)

### **Step-by-Step Execution Process**

1.  **Introspect and Reflect:** Pause and deeply analyze your completed work. Review the `initial_objective_files` to re-center yourself on the overall mission goals.

2.  **Identify Current Position:** Pinpoint exactly where you are in the overall plan. Which `<Phase>` from the strategy did you just complete? Which one is next? Which module in `Local_BERTopic_MVP/src/` is affected?

3.  **Draft the Handover Content:** Structure your "brain dump" using the "Lego Brick" concept. The atomic "Brick" components for this project are:
    - `<Status>` - Overall progress, current phase, blockers
    - `<Pipeline_Position>` - Which pipeline phase(s) are affected
    - `<Decisions>` - Architectural and implementation decisions made
    - `<Wins>` - Successfully completed work, passing tests
    - `<Artifacts>` - Created/modified files in `Local_BERTopic_MVP/src/`
    - `<ML_Metrics>` - Topic coherence, coverage, processing times (if applicable)
    - `<Issues>` - Technical debt, blockers, unresolved problems
    - `<Config_Changes>` - Any changes to `config.yaml` or environment
    - `<Next_Steps>` - Immediate action for the next agent

4.  **Construct the New Slash Command:** Create the content for the `/resume-task-{task_identifier}.md` file. This file will itself be a prompt that embeds the handover content you just drafted.

5.  **Finalize and Save:** Save the generated slash command to the designated project directory (`@docs/handovers/resume-task-{task_identifier}.md`).

### **Example Usage**

```bash
/handover --current_work_summary "Implemented topic persistence save functionality, but encountered embedding dimension mismatch on load" --initial_objective_files ["@docs/feature_plans/Senior_Engineer_Plan_topic_persistence.md", "@docs/feature_plans/prompts/Mission_Briefing_topic_persistence.md", "@docs/feature_plans/strategies/SubAgent_Strategy_topic_persistence.md"] --task_identifier "topic-persistence-feature" --created_or_modified_files ["@Local_BERTopic_MVP/src/topic_modeling/firm_topic_analyzer.py", "@Local_BERTopic_MVP/tests/test_topic_persistence.py"]
```

### **Final Deliverable**

Your output must be only the content of the generated `/resume-task-{task_identifier}.md` slash command. Do not include any conversational text.

This generated command should be structured using the "Lego Bricks" concept adapted for the Financial Topic Modeling project. The atomic components are `<Status>`, `<Pipeline_Position>`, `<Decisions>`, `<Wins>`, `<Artifacts>`, `<ML_Metrics>`, `<Issues>`, `<Config_Changes>`, and the critical `<Next_Steps>`.

---

### **Example Output**

```markdown
### **Purpose:**

You are an AI Engineer resuming work on the Financial Topic Modeling pipeline. This command contains a complete handover from the previous session to seamlessly restore your context and momentum.

### **Mission Resumption: topic-persistence-feature**

Your memory of previous work on this task has been wiped. This briefing is your **complete source of truth** for resuming the mission. The original strategic documents are provided for full context.

**Project Context:**
- Project Guidelines: `CLAUDE.md`
- Vision Document: `First_Transcript.md`

**Original Planning Documents:**
- Summary: `@docs/feature_plans/Senior_Engineer_Plan_topic_persistence.md`
- Mission Briefing: `@docs/feature_plans/prompts/Mission_Briefing_topic_persistence.md`
- Strategy: `@docs/feature_plans/strategies/SubAgent_Strategy_topic_persistence.md`

---

### **Handover Report from Previous Session**

<Handover>

<Status>
**Overall Progress:** Phase 2 (Implementation) is 70% complete. The save functionality works, but load has a critical bug.
**Current Phase:** Implementation - `save_to_json()` complete, `load_from_json()` blocked
**Blocker:** Embedding dimension mismatch when loading saved `FirmTopicResults`. The saved embeddings have shape (n, 768) but the reconstructed object expects (n, 384).
</Status>

<Pipeline_Position>
**Affected Module:** `topic_modeling/` (Map Phase)
**Affected Files:**
- `src/topic_modeling/firm_topic_analyzer.py` - Primary implementation
- `src/config/config.yaml` - Persistence configuration added
**Downstream Impact:** Theme identification will use these cached results, so the load must be verified before proceeding.
</Pipeline_Position>

<Decisions>
- **Decision:** Used JSON serialization with custom `NumpyEncoder` instead of pickle
- **Rationale:** JSON is human-readable and aligns with existing patterns in `storage_manager.py`. Pickle has security concerns for a research codebase that may be shared.

- **Decision:** Embeddings stored as base64-encoded numpy arrays
- **Rationale:** Reduces file size while maintaining full precision. Pattern borrowed from the existing `save_embeddings()` method in `CrossFirmDataManager`.
</Decisions>

<Wins>
- `save_to_json()` method successfully saves all `FirmTopicResults` fields
- Saved files are human-readable and ~40% smaller than pickle equivalent
- Unit tests for save functionality all pass (5/5)
- Config integration complete - persistence can be toggled via `topic_persistence.enabled`
</Wins>

<Artifacts>
- **Modified:** `@Local_BERTopic_MVP/src/topic_modeling/firm_topic_analyzer.py`
  - Added `save_to_json()` method (lines 114-175)
  - Added `load_from_json()` method (lines 177-230) - INCOMPLETE
- **Modified:** `@Local_BERTopic_MVP/src/config/config.yaml`
  - Added `topic_persistence` section with `enabled`, `output_dir`, `load_saved_topics` options
- **Created:** `@Local_BERTopic_MVP/tests/test_topic_persistence.py`
  - 8 test cases, 5 passing, 3 failing (all load-related)
</Artifacts>

<ML_Metrics>
**Not yet validated** - blocked by load functionality. Once load works:
- [ ] Verify topic coherence scores match between saved and loaded
- [ ] Verify centroid embeddings are identical
- [ ] Measure processing time improvement on cached runs
</ML_Metrics>

<Issues>
- **BLOCKER:** Embedding dimension mismatch in `load_from_json()`. Investigation shows the issue is in how `embeddings_reduced` is being reconstructed. The UMAP reduction is (n, 5) but the saved full embeddings are (n, 768). Need to save BOTH embedding types.

- **Technical Debt:** The `BERTopic` model object itself is not being serialized - only its outputs. This means the loaded results cannot be used to transform new documents. Noted for future enhancement but not blocking current objective.

- **Minor:** No validation that the loaded config matches the saved config (e.g., different embedding model). Could cause silent failures.
</Issues>

<Config_Changes>
```yaml
# Added to config.yaml
topic_persistence:
  enabled: true
  output_dir: "./output"
  load_saved_topics: true
  overwrite_existing: false
```
</Config_Changes>

<Next_Steps>
**Your immediate and primary task is to fix the embedding dimension mismatch in `load_from_json()`.**

Specific actions:
1. Read `firm_topic_analyzer.py` lines 177-230 to understand current load logic
2. Modify `save_to_json()` to save BOTH `embeddings` (768-dim) and `embeddings_reduced` (5-dim)
3. Update `load_from_json()` to reconstruct both embedding arrays
4. Run `pytest tests/test_topic_persistence.py` to verify fix
5. If all tests pass, proceed to Phase 3 (Integration Testing)
</Next_Steps>

</Handover>

---

### **Action Directive**

1. Review the Handover Report above to fully absorb the context.
2. Read the referenced files to understand the current state.
3. Begin executing the `<Next_Steps>` immediately.
4. If you complete the blocker fix, proceed to integration testing as specified in the original Mission Briefing.
```
