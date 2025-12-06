# Idempotency Plan

## Overview

The pipeline is designed to be idempotent—running it multiple times produces the same result without duplicating data or causing errors.

---

## Idempotency Guarantees

| Operation          | Idempotent? | Mechanism                      |
| ------------------ | ----------- | ------------------------------ |
| Firm processing    | ✅ Yes      | Skip if `processed_at` is set  |
| Sentence insertion | ✅ Yes      | Foreign key to firm + topic    |
| Topic creation     | ✅ Yes      | Deleted and recreated per firm |
| Theme aggregation  | ✅ Yes      | Full recompute from topics     |
| Index building     | ✅ Yes      | DROP IF EXISTS + CREATE        |

---

## Firm-Level Idempotency

### Check Before Processing

```python
# From unified_pipeline.py
def _get_unprocessed_firms(self, all_firms: List[str]) -> List[str]:
    """Return firms that haven't been processed yet."""
    processed = (
        session.query(Firm.company_id)
        .filter(Firm.processed_at.isnot(None))
        .all()
    )
    processed_ids = {f.company_id for f in processed}

    return [f for f in all_firms if f not in processed_ids]
```

### Behavior on Re-run

```
First Run:
  Processing firm 1/11: AAPL
  Processing firm 2/11: MSFT
  ... (all firms processed)

Second Run (same data):
  Skipping AAPL (already processed)
  Skipping MSFT (already processed)
  ... (all firms skipped)
  No new firms to process. Pipeline complete.
```

---

## Database Idempotency

### Sentences

Sentences are tied to firms and topics. If a firm is reprocessed:

```python
# Delete existing data for this firm
session.query(Sentence).filter(Sentence.firm_id == firm_id).delete()
session.query(Topic).filter(Topic.firm_id == firm_id).delete()

# Reinsert fresh data
session.add_all(new_sentences)
session.add_all(new_topics)
session.commit()
```

**Note**: In normal operation, firms are skipped, not reprocessed. Reprocessing is only triggered by explicit `--force` flag (not implemented) or database reset.

### Topics

Topics are recreated during theme aggregation:

```python
# Theme aggregation clears and rebuilds theme assignments
session.query(Topic).update({Topic.theme_id: None})  # Clear old themes
# ... run theme clustering ...
# Assign new theme_ids
```

### Themes

Themes are fully recomputed each run:

```python
# Clear all existing themes
session.query(Theme).delete()

# Create new themes from topic clustering
for theme_data in aggregation_results:
    theme = Theme(
        name=theme_data.name,
        description=theme_data.description,
        ...
    )
    session.add(theme)
```

---

## Index Idempotency

### Vector Indexes

```sql
-- Safe to run multiple times
DROP INDEX IF EXISTS idx_sentences_embedding;
DROP INDEX IF EXISTS idx_topics_embedding;
DROP INDEX IF EXISTS idx_themes_embedding;

-- Rebuild
CREATE INDEX idx_sentences_embedding ON sentences
USING hnsw (embedding vector_cosine_ops);
-- ...
```

### Rationale

HNSW indexes are expensive to build incrementally. Full rebuild after bulk insert is faster and ensures optimal structure.

---

## LLM Call Idempotency

### Problem

LLM calls are inherently non-deterministic. Same prompt may produce different outputs.

### Solution

LLM summaries are tied to topics. If topic exists with summary, don't re-call LLM:

```python
# Check for existing summary
if topic.summary is not None:
    continue  # Skip LLM call

# Generate new summary
summary = await llm_client.generate_summary(topic)
topic.summary = summary
```

### Trade-off

We accept that:

- Re-running after theme deletion will produce different summaries
- This is acceptable for research use case
- Consistent results would require caching (complexity not justified)

---

## Edge Cases

### Partial Firm Processing

**Scenario**: Instance terminates mid-firm (e.g., during LLM calls)

**Handling**:

- Transaction not committed = firm not marked as processed
- Re-run will process the firm again from scratch
- No partial data in database

```python
def _process_single_firm(self, firm_id: str):
    try:
        # All processing in one transaction
        with session.begin():
            # ... embed, cluster, summarize ...
            firm.processed_at = datetime.utcnow()
            session.add_all(sentences)
            session.add_all(topics)
        # Commit only on success
    except Exception:
        # Rollback implicit, firm not marked processed
        raise
```

### Duplicate Firm IDs

**Scenario**: Same firm appears twice in input data

**Handling**:

- `company_id` has UNIQUE constraint
- Second occurrence skipped as "already processed"

```sql
CREATE TABLE firms (
    company_id VARCHAR(255) UNIQUE NOT NULL,
    ...
);
```

### Theme Count Changes

**Scenario**: Running with different hyperparameters produces different theme count

**Handling**:

- Themes fully replaced each run
- Old theme IDs not reused
- Downstream references (if any) would need update

---

## Testing Idempotency

### Test 1: Double Run

```bash
# First run
TEST_MODE=mag7 ./launch_pipeline.sh
# Verify: 11 firms, ~350 topics, ~19 themes

# Second run (same data)
TEST_MODE=mag7 ./launch_pipeline.sh
# Verify: Still 11 firms, ~350 topics, ~19 themes (not doubled)
```

### Test 2: Partial Failure Recovery

```bash
# Run with limit
MAX_FIRMS=5 ./launch_pipeline.sh
# Verify: 5 firms processed

# Complete run
./launch_pipeline.sh
# Verify: 6 new firms processed (total 11)
# Verify: First 5 firms not reprocessed
```

### Test 3: Index Rebuild

```sql
-- Run multiple times
SELECT recreate_indexes();  -- Hypothetical function

-- Verify no duplicates
SELECT COUNT(*) FROM pg_indexes WHERE indexname LIKE 'idx_%_embedding';
-- Should return 3 (sentences, topics, themes)
```

---

## Recommendations

1. **Never manually insert data** - Use pipeline for all data population
2. **Use `processed_at` as source of truth** - Don't modify manually
3. **Reset by truncating tables** - Not by deleting individual records
4. **Test idempotency after changes** - Run pipeline twice, verify no duplicates
