# Telemetry Matrix

## Overview

This document defines what data the pipeline collects, why, and what it explicitly does NOT collect. The goal is to balance operational observability with privacy minimization.

---

## What We Log

| Data Category          | Specific Data                                    | Purpose                   | Retention | Location |
| ---------------------- | ------------------------------------------------ | ------------------------- | --------- | -------- |
| **Processing Metrics** | Firm count, sentence count, topic count          | Performance monitoring    | Session   | EC2 logs |
| **Timing**             | Stage durations (embed, cluster, LLM, aggregate) | Bottleneck identification | Session   | EC2 logs |
| **Errors**             | Exception messages, stack traces                 | Debugging                 | Session   | EC2 logs |
| **Model Info**         | Embedding model name, dimension                  | Reproducibility           | Permanent | Database |
| **API Usage**          | LLM request count, token estimates               | Cost tracking             | Session   | EC2 logs |
| **Progress**           | Current firm being processed                     | Resume capability         | Session   | EC2 logs |

### Sample Log Output

```
2024-12-01 14:32:15 INFO  Starting pipeline with 11 firms
2024-12-01 14:32:16 INFO  Loading embedding model: all-mpnet-base-v2 (768 dim)
2024-12-01 14:32:45 INFO  Model loaded in 30.2s
2024-12-01 14:33:00 INFO  Processing firm 1/11: AAPL
2024-12-01 14:33:15 INFO  Embedded 450 sentences in 15.3s
2024-12-01 14:33:20 INFO  Generated 23 topics
2024-12-01 14:33:25 INFO  LLM summaries: 23 requests, ~2300 tokens
2024-12-01 14:33:30 INFO  Firm AAPL complete, 23 topics saved
...
2024-12-01 14:45:00 INFO  Theme aggregation: 250 topics → 19 themes
2024-12-01 14:45:30 INFO  Pipeline complete. Total time: 13m 15s
```

---

## What We Do NOT Log

| Data Category             | Reason for Exclusion        | Alternative                 |
| ------------------------- | --------------------------- | --------------------------- |
| **Raw transcript text**   | Privacy - may contain names | Store in DB only, not logs  |
| **User queries**          | No user-facing interface    | N/A                         |
| **Individual embeddings** | High volume, not actionable | Aggregate statistics only   |
| **LLM prompt content**    | May contain derived PII     | Log request count only      |
| **LLM response content**  | Stored in DB, not logs      | Database is source of truth |
| **Database credentials**  | Security                    | Masked in all outputs       |
| **API keys**              | Security                    | Never logged                |
| **IP addresses**          | Privacy                     | Not collected               |

### Log Sanitization

```python
# From unified_pipeline.py
logger.info(f"Processing firm {i}/{total}: {firm.company_id}")
# Logs: "Processing firm 5/11: MSFT"

# NOT logged:
# - Full company name ("Microsoft Corporation")
# - CEO name mentioned in transcript
# - Raw sentence text
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA COLLECTION                              │
│                                                                      │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐          │
│   │ Processing  │     │  Errors &   │     │    API      │          │
│   │  Metrics    │     │  Warnings   │     │   Usage     │          │
│   └──────┬──────┘     └──────┬──────┘     └──────┬──────┘          │
│          │                   │                   │                  │
│          └───────────────────┼───────────────────┘                  │
│                              │                                       │
│                              ▼                                       │
│                    ┌─────────────────┐                              │
│                    │   Log Formatter │                              │
│                    │  (Sanitization) │                              │
│                    └────────┬────────┘                              │
│                             │                                        │
└─────────────────────────────┼────────────────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │   /var/log/ftm-pipeline.log   │
              │   (Session retention only)    │
              └───────────────────────────────┘
```

---

## Retention Policy

| Data Type        | Retention Period               | Deletion Method            |
| ---------------- | ------------------------------ | -------------------------- |
| EC2 Logs         | Session (instance termination) | Auto-deleted with instance |
| Database Records | Indefinite                     | Manual deletion if needed  |
| Terraform State  | Project lifetime               | Manual deletion            |
| S3 Code/Data     | Project lifetime               | Manual deletion            |

### Rationale

- **Session-only logs**: Spot instances terminate automatically; logs are ephemeral
- **Permanent DB records**: Research data needs long-term availability
- **No log aggregation**: CloudWatch Logs not configured (cost savings)

---

## Privacy Controls

### Minimization

```python
# Example: Log summary, not content
def _log_topic_creation(self, topic):
    # DO:
    logger.info(f"Created topic {topic.id} with {topic.sentence_count} sentences")

    # DON'T:
    # logger.info(f"Topic keywords: {topic.representation}")  # May reveal firm
    # logger.info(f"Sample sentence: {topic.sentences[0].text}")  # PII risk
```

### Anonymization

```python
# Firm identifiers use stock tickers, not full names
logger.info(f"Processing firm: {firm.company_id}")  # "AAPL"
# Not: logger.info(f"Processing firm: {firm.name}")  # "Apple Inc."
```

### Access Control

| Log Location | Access                                |
| ------------ | ------------------------------------- |
| EC2 instance | SSH (developer only)                  |
| Database     | Developer credentials                 |
| S3 bucket    | IAM role (EC2), developer credentials |

---

## Observability Gaps

| Gap                  | Impact                   | Mitigation                      | Priority |
| -------------------- | ------------------------ | ------------------------------- | -------- |
| No CloudWatch Logs   | Logs lost on termination | SSH and save before termination | Low      |
| No APM (X-Ray)       | No distributed tracing   | Not needed (single instance)    | N/A      |
| No metrics dashboard | Manual log inspection    | Add Grafana if scaling          | Low      |
| No alerting          | Silent failures          | Add SNS for critical errors     | Medium   |

---

## Telemetry Ethics Checklist

| Question                           | Answer                         |
| ---------------------------------- | ------------------------------ |
| Do we collect only necessary data? | Yes - operational metrics only |
| Is raw content logged?             | No - summaries and counts only |
| Are credentials ever logged?       | No - explicitly excluded       |
| Is retention appropriate?          | Yes - session-only for logs    |
| Can users opt out?                 | N/A - no user-facing interface |
| Is collection documented?          | Yes - this document            |
