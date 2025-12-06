# Clause → Control → Test Evidence

## Overview

This document demonstrates the implementation of ethical guardrails using the **Clause → Control → Test** framework. Each guardrail starts as a promise (clause), is enforced through a technical control, and validated by a failing test that becomes passing after implementation.

---

## Guardrail 1: No PII in LLM Prompts

### Clause

**Promise**: "The pipeline will not send personally identifiable information (PII) or company-specific identifiers to external LLM APIs."

**Stakeholder**: Data subjects whose information appears in earnings calls (executives, employees mentioned by name).

**Risk**: Leaking sensitive information to third-party AI providers.

### Control

**Enforcement Point**: LLM prompt construction in `xai_client.py`

**Implementation**:

```python
# From cloud/src/llm/xai_client.py

def _build_topic_prompt(self, topic_info: dict) -> str:
    """
    Build prompt using ONLY:
    - Topic keywords (derived, not raw)
    - Sentence count (metadata)

    NOT included:
    - Raw transcript text
    - Company name
    - Person names
    - Specific dates
    """
    keywords = topic_info.get("representation", [])[:10]
    sentence_count = topic_info.get("sentence_count", 0)

    return f"""Summarize this earnings call topic in 1-2 sentences.
Focus on the business concept, not specific companies.

Keywords: {', '.join(keywords)}
Number of sentences: {sentence_count}

Provide a general business summary."""
```

**Verification**: Manual audit of prompt templates confirms:

- No `company_name` variable in prompts
- No `raw_text` from sentences in prompts
- Keywords are derived statistical representations

### Test

**Red-Bar Test** (Failing before control):

```python
# tests/test_llm_privacy.py

def test_no_company_names_in_prompts():
    """Verify LLM prompts don't contain company identifiers."""
    client = XAIClient(config)

    # Simulate topic from "Apple Inc."
    topic_info = {
        "firm_name": "Apple Inc.",
        "representation": ["iPhone", "services", "growth"],
        "sentence_count": 25
    }

    prompt = client._build_topic_prompt(topic_info)

    # This should FAIL if firm_name leaks into prompt
    assert "Apple" not in prompt
    assert "Inc." not in prompt
```

**Status**: 🟢 **PASSING**

The prompt construction intentionally excludes `firm_name` from the template.

---

## Guardrail 2: Transparent AI Attribution

### Clause

**Promise**: "All AI-generated content will be clearly marked as such in the database, distinguishing human-derived data from machine-generated summaries."

**Stakeholder**: Downstream analysts who consume the data and need to understand its provenance.

**Risk**: Mistaking LLM-generated text for human-written analysis.

### Control

**Enforcement Point**: Database schema and topic/theme insertion

**Implementation**:

```python
# From cloud/src/database/models.py

class Topic(Base):
    __tablename__ = "topics"

    # ... other fields ...
    summary = Column(Text)  # LLM-generated
    summary_source = Column(String(50), default="llm")  # Attribution

class Theme(Base):
    __tablename__ = "themes"

    # ... other fields ...
    description = Column(Text)  # LLM-generated
    description_source = Column(String(50), default="llm")  # Attribution
```

**Database Evidence**:

```sql
-- Query to verify attribution
SELECT
    id,
    summary,
    summary_source
FROM topics
WHERE summary IS NOT NULL
LIMIT 5;

-- Expected output:
-- id | summary                              | summary_source
-- ---+--------------------------------------+----------------
-- 1  | Companies are investing in AI...    | llm
-- 2  | Supply chain challenges discussed...| llm
```

### Test

**Red-Bar Test** (Failing before control):

```python
# tests/test_ai_attribution.py

def test_llm_summaries_are_attributed():
    """Verify all LLM content has source attribution."""
    session = get_test_session()

    # Insert topic with LLM summary
    topic = Topic(
        firm_id=1,
        representation={"keywords": ["AI", "GPU"]},
        summary="AI infrastructure investment discussion"
    )
    session.add(topic)
    session.commit()

    # Retrieve and verify attribution
    saved_topic = session.query(Topic).first()

    # This should FAIL if summary_source column doesn't exist or isn't populated
    assert saved_topic.summary_source == "llm"
```

**Status**: 🟢 **PASSING**

Default value ensures all LLM content is attributed.

---

## Guardrail 3: Equal Treatment Across Firm Size

### Clause

**Promise**: "The pipeline will process all firms equally, regardless of market capitalization. Small-cap firms will receive the same analytical treatment as large-cap firms."

**Stakeholder**: Small-cap firms and their investors (the "empty chair") who are often overlooked in traditional analysis.

**Risk**: Algorithms inadvertently favoring large firms with more data.

### Control

**Enforcement Point**: Theme validation filters in `theme_aggregator.py`

**Implementation**:

```python
# From cloud/src/pipeline/theme_aggregator.py

class ThemeAggregator:
    def __init__(self, config):
        self.min_firms = config.get("min_firms", 2)
        self.max_firm_dominance = config.get("max_firm_dominance", 0.4)

    def _validate_theme(self, theme_topics: List[Topic]) -> bool:
        """
        Validate theme doesn't over-represent any single firm.

        A theme where 1 firm contributes >40% of topics is rejected.
        This prevents large firms from dominating theme identification.
        """
        firm_counts = Counter(t.firm_id for t in theme_topics)
        total_topics = len(theme_topics)

        for firm_id, count in firm_counts.items():
            dominance = count / total_topics
            if dominance > self.max_firm_dominance:
                logger.warning(
                    f"Theme rejected: Firm {firm_id} has {dominance:.1%} dominance"
                )
                return False

        return True
```

**Configuration** (`production.yaml`):

```yaml
theme_validation:
  min_firms: 2 # Theme must span at least 2 firms
  max_firm_dominance: 0.4 # No firm > 40% of theme topics
```

### Test

**Red-Bar Test** (Failing before control):

```python
# tests/test_theme_validation.py

def test_no_single_firm_dominance():
    """Verify themes don't over-represent large firms."""
    aggregator = ThemeAggregator(config)

    # Create topics where Apple dominates (60% of topics)
    topics = [
        Topic(firm_id=1, summary="Apple topic 1"),
        Topic(firm_id=1, summary="Apple topic 2"),
        Topic(firm_id=1, summary="Apple topic 3"),
        Topic(firm_id=2, summary="Microsoft topic"),
        Topic(firm_id=3, summary="Google topic"),
    ]

    # This should FAIL if validation doesn't catch dominance
    is_valid = aggregator._validate_theme(topics)
    assert is_valid == False  # Apple has 60% > 40% threshold
```

**Status**: 🟢 **PASSING**

Validation filter rejects themes dominated by single firms.

---

## Summary

| Guardrail       | Clause                              | Control                              | Test Status |
| --------------- | ----------------------------------- | ------------------------------------ | ----------- |
| No PII in LLM   | Don't send identifying info to APIs | Prompt template excludes names/text  | 🟢 Passing  |
| AI Attribution  | Mark all AI content as such         | `summary_source` column with default | 🟢 Passing  |
| Equal Treatment | Same processing for all firms       | Max dominance filter (40%)           | 🟢 Passing  |

---

## Evidence Files

| Guardrail       | Implementation File                      | Test File                        |
| --------------- | ---------------------------------------- | -------------------------------- |
| No PII          | `cloud/src/llm/xai_client.py`            | `tests/test_llm_privacy.py`      |
| AI Attribution  | `cloud/src/database/models.py`           | `tests/test_ai_attribution.py`   |
| Equal Treatment | `cloud/src/pipeline/theme_aggregator.py` | `tests/test_theme_validation.py` |

---

## Red-Bar → Green-Bar Process

The development followed test-driven development (TDD):

1. **Write failing test** that proves guardrail is missing
2. **Run test** → RED (fails as expected)
3. **Implement control** in production code
4. **Run test** → GREEN (passes)
5. **Document** in ethics ledger

This ensures guardrails are not just documented but actively enforced through automated testing.
