# Ethics Debt Ledger

## Overview

This ledger tracks ethical guardrails as technical debt. Each entry represents a promise to stakeholders, its enforcement status, and next actions.

---

## Ledger

| Status         | Risk / Promise             | Who it Protects                               | Control Owner | Acceptance Test                     | Enforcement Point         | Next Action                        | Last Reviewed |
| -------------- | -------------------------- | --------------------------------------------- | ------------- | ----------------------------------- | ------------------------- | ---------------------------------- | ------------- |
| 🟢 Enforced    | No PII in LLM prompts      | Data subjects (executives mentioned in calls) | Pipeline Team | `test_no_company_names_in_prompts`  | LLM prompt construction   | Monitor for template changes       | 2025-12-01    |
| 🟢 Enforced    | Transparent AI attribution | Downstream analysts                           | Pipeline Team | `test_llm_summaries_are_attributed` | Database schema           | Add UI indicator if frontend built | 2025-12-01    |
| 🟢 Enforced    | Equal firm treatment       | Small-cap firms (empty chair)                 | Pipeline Team | `test_no_single_firm_dominance`     | Theme validation          | Review threshold annually          | 2025-12-01    |
| 🟡 Partial     | Rate limit respect         | xAI API service                               | Pipeline Team | Manual monitoring                   | Semaphore (50 concurrent) | Add automated alerting             | 2025-12-01    |
| 🟡 Partial     | Cost guardrails            | Academic budget                               | Pipeline Team | `MAX_FIRMS` env var                 | Launch script             | Add hard spending cap              | 2025-12-01    |
| 🔴 Not Started | Audit trail                | Future auditors                               | -             | -                                   | -                         | Implement CloudTrail logging       | -             |

---

## Status Legend

| Symbol         | Meaning                                               |
| -------------- | ----------------------------------------------------- |
| 🟢 Enforced    | Control implemented, test passing, actively monitored |
| 🟡 Partial     | Control exists but incomplete or manual verification  |
| 🔴 Not Started | Identified risk, no control implemented yet           |

---

## Detailed Entries

### 🟢 No PII in LLM Prompts

**Promise**: The pipeline will not send personally identifiable information to external LLM APIs.

**Who it Protects**: Executives, employees, and other individuals mentioned by name in earnings call transcripts.

**Control**: LLM prompt templates use only derived keywords, not raw transcript text or company names.

**Test**: `tests/test_llm_privacy.py::test_no_company_names_in_prompts`

**Evidence**:

```python
# Prompt template verification
prompt = client._build_topic_prompt(topic_info)
assert "Apple" not in prompt  # Company name excluded
assert topic_info["raw_text"] not in prompt  # Raw text excluded
```

**Next Action**: Add automated CI check to flag prompt template changes.

---

### 🟢 Transparent AI Attribution

**Promise**: All AI-generated content will be clearly marked as machine-generated.

**Who it Protects**: Downstream analysts who need to understand data provenance.

**Control**: Database schema includes `summary_source` column with default value `'llm'`.

**Test**: `tests/test_ai_attribution.py::test_llm_summaries_are_attributed`

**Evidence**:

```sql
SELECT summary, summary_source FROM topics LIMIT 3;
-- All rows show summary_source = 'llm'
```

**Next Action**: If building a UI, add visual indicator (e.g., "AI-generated" badge).

---

### 🟢 Equal Firm Treatment

**Promise**: Small-cap firms receive equal analytical treatment to large-cap firms.

**Who it Protects**: Small-cap firms and their investors who are often overlooked.

**Control**: Theme validation rejects themes where any single firm contributes >40% of topics.

**Test**: `tests/test_theme_validation.py::test_no_single_firm_dominance`

**Evidence**:

```python
# Validation output
logger.warning("Theme rejected: Firm AAPL has 60% dominance > 40% threshold")
```

**Next Action**: Review 40% threshold after first full production run.

---

### 🟡 Rate Limit Respect

**Promise**: Pipeline will not overwhelm external APIs with excessive requests.

**Who it Protects**: xAI API service (good citizenship) and our own API access.

**Control**: `asyncio.Semaphore(50)` limits concurrent LLM requests.

**Verification**: Manual log inspection during runs.

**Gap**: No automated alerting if rate limits are hit.

**Next Action**: Add CloudWatch metric for API 429 responses.

---

### 🟡 Cost Guardrails

**Promise**: Pipeline costs will not exceed academic research budget.

**Who it Protects**: Research funding and project sustainability.

**Control**: `MAX_FIRMS` environment variable limits processing scope.

**Verification**: Manual cost review in AWS Cost Explorer.

**Gap**: No hard spending cap to auto-terminate if threshold exceeded.

**Next Action**: Implement AWS Budget with SNS alert.

---

### 🔴 Audit Trail

**Promise**: All pipeline executions will be traceable for compliance.

**Who it Protects**: Future auditors, compliance officers.

**Control**: Not yet implemented.

**Gap**: No centralized logging of who ran what, when.

**Next Action**: Enable CloudTrail, ship logs to CloudWatch Logs Insights.

---

## Change Log

| Date       | Entry           | Change  | Reason                 |
| ---------- | --------------- | ------- | ---------------------- |
| 2025-12-01 | No PII in LLM   | Created | Initial implementation |
| 2025-12-01 | AI Attribution  | Created | Initial implementation |
| 2025-12-01 | Equal Treatment | Created | Initial implementation |
| 2025-12-01 | Rate Limits     | Created | Identified gap         |
| 2025-12-01 | Cost Guardrails | Created | Identified gap         |
| 2025-12-01 | Audit Trail     | Created | Identified gap         |

---

## Review Schedule

- **Monthly**: Review 🟡 Partial entries for upgrade to 🟢
- **Quarterly**: Review thresholds (40% dominance, 50 concurrent)
- **Annually**: Full ledger audit with stakeholder input
