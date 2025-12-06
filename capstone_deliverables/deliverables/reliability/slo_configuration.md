# SLO Configuration

## Overview

Service Level Objectives (SLOs) define measurable targets for pipeline performance, cost, and reliability. These are appropriate for a batch processing workload.

---

## SLO Definitions

### SLO-1: Processing Throughput

**Target**: Process 3,000 firms in under 4 hours

| Metric         | Target   | Measurement                       |
| -------------- | -------- | --------------------------------- |
| Firms per hour | >750     | `firms_processed / elapsed_hours` |
| Total runtime  | <4 hours | End-to-end wall clock             |

**Rationale**: Based on extrapolation from MAG7 run (11 firms in 15 min = 44 firms/hour). Full run should complete in single work session.

**Measurement**:

```python
# From pipeline logs
start_time = datetime.now()
# ... processing ...
elapsed = datetime.now() - start_time
throughput = firms_processed / (elapsed.total_seconds() / 3600)
logger.info(f"Throughput: {throughput:.1f} firms/hour")
```

**Alert Threshold**: If throughput drops below 500 firms/hour, investigate bottleneck.

---

### SLO-2: Cost Efficiency

**Target**: Infrastructure cost under $5 per quarterly run (excluding LLM)

| Component     | Budget     | Measurement       |
| ------------- | ---------- | ----------------- |
| EC2 (spot)    | <$2.00     | AWS Cost Explorer |
| RDS (running) | <$1.00     | AWS Cost Explorer |
| S3            | <$0.10     | AWS Cost Explorer |
| **Total**     | **<$5.00** |                   |

**Rationale**: Academic research budget requires cost consciousness. LLM costs tracked separately (variable based on usage).

**Measurement**:

```bash
# Post-run cost check
aws ce get-cost-and-usage \
  --time-period Start=2025-12-01,End=2025-12-02 \
  --granularity DAILY \
  --metrics "BlendedCost" \
  --filter '{"Dimensions":{"Key":"SERVICE","Values":["EC2","RDS","S3"]}}'
```

**Alert Threshold**: If run exceeds $10, investigate resource sizing.

---

### SLO-3: Data Completeness

**Target**: 100% of input firms produce output topics

| Metric                | Target | Measurement                           |
| --------------------- | ------ | ------------------------------------- |
| Firm completion rate  | 100%   | `processed_firms / input_firms`       |
| Topic generation rate | >90%   | `firms_with_topics / processed_firms` |

**Rationale**: Pipeline should not silently skip firms.

**Measurement**:

```sql
-- Completion rate
SELECT
    (SELECT COUNT(*) FROM firms WHERE processed_at IS NOT NULL) as processed,
    (SELECT COUNT(DISTINCT firm_id) FROM topics) as with_topics;
```

**Alert Threshold**: Any firm without topics requires investigation.

---

### SLO-4: Theme Quality

**Target**: Themes span multiple firms (no single-firm "themes")

| Metric              | Target | Measurement                      |
| ------------------- | ------ | -------------------------------- |
| Min firms per theme | >=2    | `min(n_firms)` from themes table |
| Max firm dominance  | <=40%  | Validation filter output         |

**Rationale**: Single-firm themes are not "cross-firm themes" by definition.

**Measurement**:

```sql
SELECT
    MIN(n_firms) as min_firms,
    MAX(n_firms) as max_firms,
    AVG(n_firms) as avg_firms
FROM themes;
```

**Alert Threshold**: Any theme with `n_firms < 2` indicates validation failure.

---

## SLO Dashboard (Manual)

Since this is a batch job without continuous monitoring, SLOs are verified post-run:

### Post-Run Checklist

```markdown
## SLO Verification - Run Date: \***\*\_\_\_\_\*\***

### SLO-1: Throughput

- [ ] Total runtime: \_\_\_ hours (target: <4)
- [ ] Throughput: \_\_\_ firms/hour (target: >750)

### SLO-2: Cost

- [ ] EC2 cost: $**\_** (target: <$2.00)
- [ ] RDS cost: $**\_** (target: <$1.00)
- [ ] Total infra: $**\_** (target: <$5.00)
- [ ] LLM cost: $**\_** (tracked separately)

### SLO-3: Completeness

- [ ] Firms processed: **_/_** (target: 100%)
- [ ] Firms with topics: **_/_** (target: >90%)

### SLO-4: Quality

- [ ] Min firms per theme: \_\_\_ (target: >=2)
- [ ] Themes count: \_\_\_ (expected: 10-100)
```

---

## Alerting

### Current State

No automated alerting configured. Pipeline logs errors to stdout/file.

### Recommended Implementation

```python
# Alert on SLO breach
def check_slos_and_alert():
    violations = []

    # SLO-1: Throughput
    if throughput < 500:
        violations.append(f"Throughput {throughput} < 500 firms/hour")

    # SLO-3: Completeness
    if completion_rate < 1.0:
        violations.append(f"Completion rate {completion_rate:.1%} < 100%")

    # SLO-4: Quality
    min_firms = session.query(func.min(Theme.n_firms)).scalar()
    if min_firms < 2:
        violations.append(f"Theme with only {min_firms} firms")

    if violations:
        # Send SNS notification (future implementation)
        logger.error(f"SLO VIOLATIONS: {violations}")
```

---

## Error Budget

### Definition

Error budget = tolerated deviation from SLO targets.

| SLO          | Target      | Error Budget                             |
| ------------ | ----------- | ---------------------------------------- |
| Throughput   | <4 hours    | 30 min buffer (4.5 hours acceptable)     |
| Cost         | <$5         | $2 buffer ($7 acceptable for single run) |
| Completeness | 100%        | 0% (all firms must process)              |
| Quality      | min 2 firms | 0% (validation enforced)                 |

### Budget Consumption

Track across runs:

| Run Date   | Throughput | Cost  | Completeness | Quality | Notes     |
| ---------- | ---------- | ----- | ------------ | ------- | --------- |
| 2025-12-01 | 15 min     | $1.30 | 100%         | Pass    | MAG7 test |
| TBD        | -          | -     | -            | -       | Full run  |

---

## Continuous Improvement

### Review Cadence

- **After each run**: Verify SLOs against checklist
- **Quarterly**: Review thresholds based on accumulated data
- **On failure**: Root cause analysis and threshold adjustment

### SLO Evolution

| Version | Date       | Change            | Reason                 |
| ------- | ---------- | ----------------- | ---------------------- |
| 1.0     | 2025-12-01 | Initial SLOs      | Baseline from MAG7 run |
| -       | TBD        | Adjust throughput | After full run data    |
