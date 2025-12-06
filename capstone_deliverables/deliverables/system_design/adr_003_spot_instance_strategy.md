# ADR-003: Spot Instance Strategy with Checkpoint/Resume

**Status**: Accepted
**Decision Makers**: Project Lead

---

## Context

The pipeline runs on a GPU instance (`g4dn.2xlarge`) for 2-4 hours per quarterly execution. AWS offers two pricing models:

| Pricing   | Hourly Cost | Interruption Risk |
| --------- | ----------- | ----------------- |
| On-Demand | ~$1.01/hr   | None              |
| Spot      | ~$0.25/hr   | 2-minute warning  |

For a 4-hour job:

- On-Demand: ~$4.00
- Spot: ~$1.00 (75% savings)

---

## Decision

**We use Spot instances with a checkpoint/resume pattern for resilience.**

- EC2 Spot with `one-time` request (terminates when job completes)
- Per-firm checkpoint writes to PostgreSQL
- Re-running the pipeline resumes from last completed firm

---

## Rationale

### 1. Cost Savings

Spot instances provide 60-75% cost reduction:

```
On-Demand: $1.01/hr × 4 hours = $4.04
Spot:      $0.25/hr × 4 hours = $1.00
Savings:   75%
```

For a quarterly batch job that isn't time-critical, this is acceptable risk.

### 2. Batch Job Tolerance

The pipeline processes firms independently. If interrupted:

- No data corruption (PostgreSQL transactions)
- No lost work (checkpoint after each firm)
- Easy recovery (re-run same command)

### 3. Checkpoint Pattern

```python
# Pseudocode for checkpoint/resume
processed_firms = db.query(Firm.id).filter(Firm.processed_at != None)

for firm in all_firms:
    if firm.id in processed_firms:
        continue  # Skip already-processed firms

    # Process firm
    topics = process_firm(firm)

    # Checkpoint: Write immediately, committed transaction
    db.add_all(topics)
    db.commit()  # <-- Checkpoint complete
```

If interrupted between firms:

- All completed firms are in database
- Resume picks up from next unprocessed firm

### 4. Two-Minute Warning

AWS provides a 2-minute termination notice via instance metadata. While we don't implement active handling (unnecessary for this workload), the current firm being processed will be lost at worst—a ~30 second penalty.

---

## Consequences

### Positive

1. **75% Cost Reduction**: $1.00 vs $4.00 per run
2. **Automatic Cleanup**: Spot instances terminate; no orphaned resources
3. **No Lock-In**: Can switch to on-demand with `USE_SPOT=false` flag
4. **Resilient Design**: Checkpoint pattern useful beyond spot instances

### Negative

1. **Interruption Risk**: ~5-10% chance of interruption during 4-hour job
2. **Resume Overhead**: If interrupted, must re-launch instance (~3 min)
3. **Partial Work Loss**: Current firm processing lost on interrupt (~30 sec)

### Mitigations

- **Checkpoint Granularity**: Per-firm (not per-batch) ensures minimal loss
- **Easy Re-run**: Same command resumes automatically
- **Optional Override**: `USE_SPOT=false` for guaranteed completion

---

## Alternatives Considered

### Alternative 1: On-Demand Instances

- Zero interruption risk
- 4x higher cost
- **Rejected**: Cost prohibitive for budget-conscious academic project

### Alternative 2: Spot Fleet with Diversification

- Multiple instance types for availability
- Complex configuration
- **Rejected**: Over-engineering for single quarterly job

### Alternative 3: Reserved Instances

- Discounted rates for commitment
- Requires 1-year or 3-year commitment
- **Rejected**: Usage too infrequent to justify

### Alternative 4: Lambda with Step Functions

- No infrastructure to manage
- 15-minute timeout per Lambda
- **Rejected**: Would require complex orchestration; still expensive

---

## Implementation Details

### Launch Script (`launch_pipeline.sh`)

```bash
# Spot instance configuration
aws ec2 run-instances \
  --instance-type g4dn.2xlarge \
  --instance-market-options '{
    "MarketType": "spot",
    "SpotOptions": {
      "SpotInstanceType": "one-time"
    }
  }' \
  ...
```

### Override for On-Demand

```bash
# Use on-demand if spot unavailable or guaranteed completion needed
USE_SPOT=false ./launch_pipeline.sh
```

### Database Checkpoint

From `unified_pipeline.py`:

```python
def _process_single_firm(self, firm_id: str, ...):
    # ... processing logic ...

    # Atomic checkpoint
    with self.session.begin():
        self.session.add(firm)
        self.session.add_all(sentences)
        self.session.add_all(topics)
    # Firm is now checkpointed; safe to interrupt
```

---

## Evidence

### Cost Validation

From validated MAG7 run (11 firms, ~15 minutes):

| Component       | Cost       |
| --------------- | ---------- |
| EC2 Spot        | ~$0.06     |
| RDS             | ~$0.02     |
| S3              | <$0.01     |
| **Total Infra** | **~$0.09** |
| xAI API         | ~$1.20     |
| **Total**       | **~$1.30** |

### Spot Price History

`g4dn.2xlarge` in us-east-1 (30-day range):

- Min: $0.22/hr
- Max: $0.31/hr
- Average: $0.25/hr
- On-Demand: $1.01/hr

### Interruption Rate

Historical data for `g4dn.2xlarge`:

- Interruption frequency: ~5-10% over 4 hours
- Acceptable for batch workloads with checkpoint/resume

---

## Learning Outcome

Spot instances are ideal for:

- Batch jobs with checkpointing
- Fault-tolerant workloads
- Cost-sensitive projects
- Non-time-critical processing

The key insight: **design for failure from the start**. The checkpoint/resume pattern not only enables spot instances but also provides resilience against any failure mode (power outage, OOM, network issues).
