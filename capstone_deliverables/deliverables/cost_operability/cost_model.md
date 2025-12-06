# Cost Model

## Overview

This document provides a detailed breakdown of pipeline costs, comparing the original distributed design with the final single-instance architecture.

---

## Cost Summary

### Per Quarterly Run (Validated)

| Component                | Unit Cost    | Usage        | Total       |
| ------------------------ | ------------ | ------------ | ----------- |
| EC2 g4dn.2xlarge (spot)  | $0.25/hr     | 3 hours      | $0.75       |
| RDS db.t4g.large         | $0.08/hr     | 3 hours      | $0.24       |
| RDS Storage (100GB)      | $0.10/GB/mo  | Prorated     | $0.08       |
| S3 Storage               | $0.023/GB/mo | <1GB         | $0.02       |
| S3 Requests              | $0.005/1K    | <100         | <$0.01      |
| **Infrastructure Total** |              |              | **~$1.10**  |
| xAI API (LLM)            | Variable     | ~3,500 calls | ~$10-50     |
| **Grand Total**          |              |              | **~$12-56** |

### MAG7 Validation Run (11 firms)

| Component    | Cost       |
| ------------ | ---------- |
| EC2 (15 min) | $0.06      |
| RDS (15 min) | $0.02      |
| S3           | <$0.01     |
| xAI API      | ~$0.28     |
| **Total**    | **~$0.37** |

---

## Cost Comparison: Original vs Final Architecture

### Original Design (AWS Batch)

```
3,000 containers × $0.003/container = $9.00 (compute)
+ S3 intermediate storage = $0.50
+ DynamoDB writes = $1.25
+ Step Functions = $0.50
─────────────────────────────────
Estimated Total: $11.25+
```

### Final Design (Single GPU)

```
1 instance × 4 hours × $0.25/hr = $1.00 (compute)
+ RDS running = $0.32
+ S3 code = $0.02
─────────────────────────────────
Total: $1.34
```

### Savings

| Metric    | Original   | Final     | Savings |
| --------- | ---------- | --------- | ------- |
| Compute   | $9.00      | $1.00     | 89%     |
| Storage   | $1.75      | $0.34     | 81%     |
| **Total** | **$11.25** | **$1.34** | **88%** |

---

## Detailed Cost Breakdown

### EC2 Compute

| Instance Type | On-Demand | Spot     | Savings |
| ------------- | --------- | -------- | ------- |
| g4dn.xlarge   | $0.526/hr | $0.18/hr | 66%     |
| g4dn.2xlarge  | $1.01/hr  | $0.25/hr | 75%     |
| g4dn.4xlarge  | $1.94/hr  | $0.45/hr | 77%     |

**Selected**: g4dn.2xlarge (spot) - best balance of performance and cost.

### RDS Database

| State   | Cost                               |
| ------- | ---------------------------------- |
| Running | $0.08/hr                           |
| Stopped | $0/hr (compute) + $10/mo (storage) |

**Strategy**: Stop RDS between runs. Start 5 min before pipeline.

### S3 Storage

| Item           | Size   | Cost          |
| -------------- | ------ | ------------- |
| Code package   | ~50MB  | <$0.01/mo     |
| Transcript CSV | ~400MB | ~$0.01/mo     |
| **Total**      | ~450MB | **~$0.02/mo** |

### LLM API (xAI)

| Operation          | Requests | Tokens (est.) | Cost       |
| ------------------ | -------- | ------------- | ---------- |
| Topic summaries    | ~350     | ~35,000       | ~$0.50     |
| Theme descriptions | ~20      | ~2,000        | ~$0.05     |
| Theme naming       | ~20      | ~1,000        | ~$0.03     |
| **Total**          | ~390     | ~38,000       | **~$0.60** |

_Note: Costs vary based on xAI pricing tier and actual token usage._

---

## Monthly Cost Projections

### Active Development Month

| Activity    | Runs | Cost     |
| ----------- | ---- | -------- |
| MAG7 tests  | 10   | $13      |
| Full runs   | 2    | $12      |
| RDS storage | -    | $10      |
| **Total**   |      | **~$35** |

### Idle Month (No Runs)

| Component   | Cost     |
| ----------- | -------- |
| RDS storage | $10      |
| S3 storage  | $0.02    |
| **Total**   | **~$10** |

### Production Quarter

| Month             | Activity | Cost     |
| ----------------- | -------- | -------- |
| Month 1           | Full run | $6       |
| Month 2           | Idle     | $10      |
| Month 3           | Idle     | $10      |
| **Quarter Total** |          | **~$26** |

---

## Cost Optimization Strategies

### 1. RDS Start/Stop

```bash
# After pipeline completes
./stop_rds.sh
# Saves: $0.08/hr × 720 hrs/mo = $57.60/mo

# Before next run
./start_rds.sh
# Wait 5 minutes
```

### 2. Spot Instances

```bash
# Default (spot)
./launch_pipeline.sh

# Override for guaranteed completion
USE_SPOT=false ./launch_pipeline.sh
# Extra cost: ~$3 per run
```

### 3. Instance Right-Sizing

| Workload           | Recommended Instance | Cost     |
| ------------------ | -------------------- | -------- |
| MAG7 (11 firms)    | g4dn.xlarge          | $0.18/hr |
| Full (3,000 firms) | g4dn.2xlarge         | $0.25/hr |
| Large batch        | g4dn.4xlarge         | $0.45/hr |

### 4. LLM Cost Control

```bash
# Skip LLM summaries for testing
XAI_API_KEY="" ./launch_pipeline.sh
# Saves: ~$1-5 per run (topics have keywords only)
```

---

## Cost Alerts

### AWS Budget Configuration

```bash
aws budgets create-budget \
  --account-id $ACCOUNT_ID \
  --budget '{
    "BudgetName": "ftm-monthly",
    "BudgetLimit": {"Amount": "50", "Unit": "USD"},
    "TimeUnit": "MONTHLY",
    "BudgetType": "COST"
  }' \
  --notifications-with-subscribers '[{
    "Notification": {
      "NotificationType": "ACTUAL",
      "ComparisonOperator": "GREATER_THAN",
      "Threshold": 80
    },
    "Subscribers": [{
      "SubscriptionType": "EMAIL",
      "Address": "your@email.com"
    }]
  }]'
```

### Thresholds

| Alert    | Threshold | Action                      |
| -------- | --------- | --------------------------- |
| Warning  | $40/mo    | Review usage                |
| Critical | $75/mo    | Investigate immediately     |
| Hard Cap | $100/mo   | Consider stopping resources |

---

## ROI Analysis

### Cost to Build

| Phase          | Hours  | Notes                 |
| -------------- | ------ | --------------------- |
| Design         | 20     | Architecture planning |
| Implementation | 40     | Code development      |
| Testing        | 15     | Validation runs       |
| Documentation  | 10     | This capstone         |
| **Total**      | **85** |                       |

### Value Delivered

| Capability                   | Manual Effort | Automated |
| ---------------------------- | ------------- | --------- |
| Process 3,000 earnings calls | ~500 hours    | 4 hours   |
| Identify cross-firm themes   | Impossible    | Automatic |
| Semantic search              | Not feasible  | Native    |

### Payback

If time valued at $50/hour:

- Manual: 500 hrs × $50 = $25,000
- Automated: 85 hrs dev + $6/quarter = $4,274/year
- **Savings**: ~$20,000/year (for quarterly analysis)

---

## Cost Governance

### Budget Owner

Project lead is responsible for:

- Monitoring AWS Cost Explorer weekly
- Stopping unused resources
- Approving runs exceeding $10

### Approval Thresholds

| Run Type               | Est. Cost | Approval         |
| ---------------------- | --------- | ---------------- |
| MAG7 test              | <$2       | Self-approved    |
| Full run               | <$10      | Self-approved    |
| Multiple full runs     | >$20      | Advisor approval |
| Infrastructure changes | Variable  | Review required  |
