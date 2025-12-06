# Capstone Deliverables: Financial Topic Modeling

**Cloud Computing Capstone Project**

This directory contains all deliverables for the capstone project submission, demonstrating the design, implementation, and deployment of a cloud-native Financial Topic Modeling pipeline.

---

## Project Overview

**Research Question**: What universal investment themes emerge across thousands of quarterly earnings calls?

**Solution**: A hierarchical NLP pipeline that:

1. Ingests earnings call transcripts from 3,000+ firms
2. Clusters sentences into firm-level topics using BERTopic
3. Generates human-readable summaries via LLM (xAI/Grok)
4. Aggregates topics into cross-firm themes
5. Stores results in PostgreSQL with pgvector for semantic search

**Key Learning**: Applying parallel computing principles (Amdahl's Law) to realize that a single GPU instance outperforms a distributed AWS Batch architecture for this workload, achieving 90% cost reduction with equivalent performance.

---

## Validated Results

| Metric              | Value            |
| ------------------- | ---------------- |
| Firms Processed     | 11 (MAG7 + tech) |
| Sentences Analyzed  | ~5,000           |
| Firm-Level Topics   | ~350             |
| Cross-Firm Themes   | ~19              |
| Processing Time     | ~15 minutes      |
| Infrastructure Cost | ~$1.30           |

---

## Deliverables Structure

```
capstone_deliverables/
├── README.md                           # This file
├── course_documents_and_resources/     # Assignment guidelines and templates
└── deliverables/
    ├── system_design/                  # Architecture and ADRs
    │   ├── architecture_diagram.md
    │   ├── adr_001_single_gpu_vs_batch.md
    │   ├── adr_002_postgres_vs_dynamodb.md
    │   ├── adr_003_spot_instance_strategy.md
    │   ├── trust_model.md
    │   └── api_schema.md
    ├── ethics/                         # Clause → Control → Test
    │   ├── clause_control_test.md
    │   ├── ethics_ledger.md
    │   └── telemetry_matrix.md
    ├── reliability/                    # Testing and operations
    │   ├── test_notes.md
    │   ├── idempotency_plan.md
    │   ├── slo_configuration.md
    │   └── dr_runbook.md
    ├── cost_operability/               # Cost model and incident response
    │   ├── cost_model.md
    │   ├── backpressure_killswitches.md
    │   └── incident_response.md
    └── ai_collaboration/               # AI assistance log
        └── ai_log.md
```

---

## Quick Setup

### Prerequisites

- AWS CLI configured with valid credentials
- Terraform 1.0+
- EC2 key pair created in us-east-1
- Python 3.10+ (for local testing)

### Sample Data

**Download transcript data**: [Google Drive Link - [Zipped CSV](https://drive.google.com/file/d/1pndWY7ApEEXusoody2hWeCz25yCsFMMo/view?usp=sharing)]

Place the CSV file in the project root:

```
Financial_Topic_Modeling/
├── transcripts_2023-01-01_to_2023-03-31_enriched.csv
```

### Environment Setup

```bash
# Clone repository
git clone https://github.com/Torr1n/Financial_Topic_Modeling.git
cd Financial_Topic_Modeling

# Configure Terraform
cd cloud/terraform
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars`:

```hcl
aws_region    = "us-east-1"
db_password   = "your-secure-password"  # Must be 8+ chars
my_ip         = "YOUR.IP.ADDRESS/32"    # For SSH access
key_pair_name = "your-key-pair"         # Existing EC2 key pair
```

### Deploy Infrastructure

```bash
# Initialize and apply Terraform
terraform init
terraform plan
terraform apply

# Upload code to S3
cd ../scripts
./upload_to_s3.sh
```

### Smoke Test (MAG7 Validation)

```bash
# Run pipeline with 11 test firms
TEST_MODE=mag7 ./launch_pipeline.sh
```

Expected output:

```
Launching EC2 spot instance...
Instance ID: i-0abc123def456
Public IP: 54.xxx.xxx.xxx

SSH: ssh -i ~/.ssh/your-key.pem ubuntu@54.xxx.xxx.xxx
Logs: sudo tail -f /var/log/ftm-pipeline.log
```

### Verify Results

Connect to RDS via any PostgreSQL client (e.g., DBeaver):

```
Host: ftm-db.xxxxx.us-east-1.rds.amazonaws.com
Port: 5432
Database: ftm
User: ftm
Password: (from terraform.tfvars)
```

Validation queries:

```sql
-- Check firm count
SELECT COUNT(*) FROM firms;  -- Expected: 11

-- Check theme count
SELECT COUNT(*) FROM themes;  -- Expected: ~19

-- Sample theme with topics
SELECT
  th.name AS theme,
  th.n_firms,
  COUNT(t.id) AS topic_count
FROM themes th
JOIN topics t ON t.theme_id = th.id
GROUP BY th.id
ORDER BY th.n_firms DESC
LIMIT 5;
```

### Cost Management

```bash
# Stop RDS when not in use (saves ~$0.08/hr)
./stop_rds.sh

# Start RDS before next run
./start_rds.sh
```

---

## Key Evidence

### Architecture Decision Records

| ADR                                                                 | Decision                | Rationale                                   |
| ------------------------------------------------------------------- | ----------------------- | ------------------------------------------- |
| [001](deliverables/system_design/adr_001_single_gpu_vs_batch.md)    | Single GPU vs AWS Batch | Amdahl's Law: sequential overhead dominates |
| [002](deliverables/system_design/adr_002_postgres_vs_dynamodb.md)   | PostgreSQL vs DynamoDB  | Hierarchical queries, vector search         |
| [003](deliverables/system_design/adr_003_spot_instance_strategy.md) | Spot instances          | 75% cost savings with checkpoint/resume     |

### Ethics & Guardrails

| Clause                | Control                       | Status   |
| --------------------- | ----------------------------- | -------- |
| No PII in LLM prompts | Keywords only, no raw text    | Enforced |
| AI attribution        | `source='llm'` in database    | Enforced |
| Equal firm treatment  | Same processing for all firms | Enforced |

### Reliability

| SLO             | Target                   | Actual                                |
| --------------- | ------------------------ | ------------------------------------- |
| Processing time | <4 hours for 3,000 firms | ~15 min for 11 firms (linear scaling) |
| Cost per run    | <$5 (excluding LLM)      | ~$1.30                                |
| Uptime          | N/A (batch job)          | N/A                                   |

---

## Repository Links

| Resource         | Path                                                 |
| ---------------- | ---------------------------------------------------- |
| Main README      | [`/cloud/README.md`](../README.md)                   |
| Architecture     | [`/docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md)   |
| Deployment Guide | [`/docs/DEPLOYMENT.md`](../docs/DEPLOYMENT.md)       |
| Configuration    | [`/docs/CONFIGURATION.md`](../docs/CONFIGURATION.md) |
| Terraform        | [`/cloud/terraform/`](../cloud/terraform/)           |
| Pipeline Code    | [`/cloud/src/pipeline/`](../cloud/src/pipeline/)     |

---

## Submission Checklist

- [ ] All deliverable files created
- [ ] Smoke test passes (MAG7)
- [ ] Results visible in PostgreSQL
- [ ] RDS stopped (cost savings)
- [ ] Repository URL + commit hash submitted to Canvas
