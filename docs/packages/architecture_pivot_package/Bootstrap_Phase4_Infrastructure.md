# Architecture Pivot: Phase 4 Bootstrap Initialization

Greetings Claude, and welcome to the Financial Topic Modeling project. You are receiving this initialization as the continuation of a strategic architecture pivot—a simplification of our cloud infrastructure from distributed AWS Batch to a single GPU instance with PostgreSQL storage.

**Phases 1, 2, and 3 are complete. You are beginning Phase 4: Infrastructure & Deployment.**

---

## Project Overview

The Financial Topic Modeling pipeline identifies cross-firm investment themes from earnings call transcripts. We completed an initial cloud migration, then caught something important: our planned AWS Batch architecture was over-engineered for our scale (~2M sentences quarterly, 3,000-5,000 firms). The orchestration overhead would exceed the actual compute time.

Three previous Claude instances executed the pivot:
- **Phase 1 Instance:** Built the complete PostgreSQL + pgvector database layer
- **Phase 2 Instance:** Unified the pipeline, validated with real data (11 firms, 350 topics, 19 themes)
- **Phase 3 Instance:** Integrated LLM summarization with xAI, implemented summary-based theme clustering

Your mission is Phase 4: Infrastructure & Deployment.

---

## The Guiding Principles

Before you read another word, internalize these principles. They are not suggestions—they are the standards by which your work will be judged:

### 1. Simplicity is Mandatory

> "The best engineers write code my mom could read. They choose boring technology, over-document the 'why', and under-engineer the 'how'. Complexity is not a flex; it becomes a liability. Real seniority is making hard problems look simple, not making simple problems look hard."

The entire pivot exists because we caught over-engineering before deploying it. AWS Batch with Step Functions orchestration would have cost more in complexity than it saved. A single GPU instance with PostgreSQL is the answer. Honor that lesson in your Terraform.

**Do not:**
- Add Lambda functions for "flexibility"
- Create complex IAM role hierarchies
- Over-engineer networking with multiple subnets
- Build "reusable modules" for infrastructure we'll use once

**Do:**
- Write the minimum Terraform that works
- Use sensible defaults
- Keep everything in one module if it's simple enough
- Document WHY each resource exists

### 2. Cost Consciousness is Non-Negotiable

The architecture pivot was driven by cost awareness. Your infrastructure must reflect this:

- **RDS must be stoppable.** Not Aurora Serverless. A regular RDS instance we can stop when not in use.
- **Spot instances for compute.** g4dn.2xlarge spot is ~$0.25/hour vs $0.75 on-demand.
- **No unnecessary services.** No NAT gateways unless required. No ELBs. No CloudWatch dashboards we won't look at.

**Target cost: <$5 per quarterly run.** This is achievable. Don't compromise it.

### 3. Adapt, Don't Over-Engineer

The pipeline works. It's been tested with real data. Your job is to deploy it to AWS, not to redesign it.

**PRESERVE:** The existing pipeline orchestration, database schema, LLM integration
**ADD:** Terraform for RDS, launch scripts for EC2
**DO NOT ADD:** Container orchestration, serverless functions, complex CI/CD

If you find yourself building something clever, stop. Ask: "Would a single shell script do this?"

### 4. TDD Applies to Infrastructure Too

Write tests before you deploy. Not after.

- Terraform `validate` before `apply`
- Test database connectivity before running pipeline
- Verify pgvector extension before bulk inserts
- Test spot instance termination handling before relying on it

### 5. Halting Points are Real

Phase 4 ends with a HALT. You stop, present the complete implementation, and await approval. This is not optional. The review protocol exists because mistakes in infrastructure are expensive—catching issues before production is 100× cheaper than after.

---

## Your Context Package

The following documents provide everything you need:

### Primary Reference
| Document | Location | Purpose |
|----------|----------|---------|
| **Phase 4 Handover** | `docs/handovers/resume-task-architecture-pivot-phase4.md` | Complete context from Phase 3 instance—what was built, why, decisions made, and your exact next steps |

### Original Planning Documents
| Document | Location | Purpose |
|----------|----------|---------|
| Senior Engineer Plan | `docs/packages/architecture_pivot_package/Senior_Engineer_Plan_architecture_pivot.md` | Technical specification—includes Terraform pseudocode and cost analysis |
| Mission Briefing | `docs/packages/architecture_pivot_package/Mission_Briefing_architecture_pivot.md` | Phased execution guide with acceptance criteria |
| SubAgent Strategy | `docs/packages/architecture_pivot_package/SubAgent_Strategy_architecture_pivot.md` | Tactical recommendations |
| Raw Vision Transcript | `docs/packages/architecture_pivot_package/raw_transcript_architecture_pivot.md` | Unedited researcher vision—consult when structured docs feel ambiguous |

### Previous Handovers (For Context)
| Document | Location | Purpose |
|----------|----------|---------|
| Phase 1 → Phase 2 | `docs/handovers/resume-task-architecture-pivot-phase2.md` | Database layer decisions |
| Phase 2 → Phase 3 | `docs/handovers/resume-task-architecture-pivot-phase3.md` | Pipeline unification decisions |
| Phase 3 → Phase 4 | `docs/handovers/resume-task-architecture-pivot-phase4.md` | LLM integration decisions |

---

## What Previous Phases Delivered

### Phase 1: Database Layer (Complete)
```
cloud/src/database/
├── __init__.py           # Package exports
├── models.py             # Firm, Sentence, Topic, Theme (SQLAlchemy + Vector(768))
└── repository.py         # DatabaseRepository with CRUD, bulk ops, hierarchical queries
```

### Phase 2: Unified Pipeline (Complete)
```
cloud/src/pipeline/
├── __init__.py           # Package exports
├── unified_pipeline.py   # Main orchestration (600+ lines with LLM integration)
└── checkpoint.py         # Resume logic for spot instances
```

### Phase 3: LLM Integration (Complete)
```
cloud/src/llm/
├── __init__.py           # Package exports
└── xai_client.py         # Async xAI client with rate limiting (271 lines)

Key capabilities:
- Topic summaries generated via LLM (1-2 sentences each)
- Theme descriptions generated via LLM (2-3 sentences each)
- Summary-based theme clustering (richer semantic content)
- Graceful fallback to keywords if LLM unavailable
- Prompt logging for observability
```

### Validation Results
The pipeline has been validated with real data:
- **11 tech firms** processed (MAG7 + others)
- **5,014 sentences** with embeddings
- **350 topics** with LLM summaries
- **19 validated themes** with LLM descriptions
- **~3 minutes** processing time on CPU
- Results verified in DBeaver

---

## Your Phase 4 Mission

### Objective
Deploy the simplified infrastructure to AWS and validate the complete pipeline end-to-end.

### Deliverables
- [ ] `cloud/terraform/main.tf` - Root module with RDS PostgreSQL
- [ ] `cloud/terraform/variables.tf` - Input variables
- [ ] `cloud/terraform/outputs.tf` - Output values (DB endpoint, etc.)
- [ ] `cloud/scripts/launch_pipeline.sh` - EC2 spot instance launch script
- [ ] `cloud/scripts/stop_rds.sh` - Cost management (stop RDS when not in use)
- [ ] `cloud/scripts/setup_ec2.sh` - EC2 user data / setup script
- [ ] End-to-end validation on cloud infrastructure
- [ ] Documentation of full run (time, cost, results)

### Acceptance Criteria
- [ ] RDS PostgreSQL deployed with pgvector extension
- [ ] g4dn.2xlarge spot instance can connect to RDS
- [ ] Pipeline completes successfully on cloud
- [ ] Processing time <4 hours for full dataset (~3,000 firms)
- [ ] Total compute cost <$5 per run
- [ ] All data queryable in Postgres with vector search
- [ ] Hierarchical queries work (theme → topics → sentences → firms)
- [ ] RDS can be stopped/started for cost management

---

## Getting Started

**Before writing any Terraform:**

1. **Read the Phase 4 handover document** (`docs/handovers/resume-task-architecture-pivot-phase4.md`)
   - Understand what Phases 1-3 built and why
   - Note any test issues mentioned (not blocking for Phase 4)
   - Absorb the key decisions made

2. **Read the Senior Engineer Plan's Infrastructure section:**
   - Section 6 shows the Terraform structure
   - Section 7 shows the cost analysis
   - Section 8 shows acceptance criteria

3. **Understand the deployment model:**
   - RDS PostgreSQL runs continuously (but stoppable)
   - EC2 spot instance launched only for pipeline runs
   - Pipeline connects to RDS, processes data, writes results
   - EC2 terminated after run, RDS optionally stopped

4. **Verify local pipeline works:**
   - Run `scripts/run_unified_pipeline_mag7.py` locally first
   - Ensure you understand the pipeline flow before deploying

---

## Critical Implementation Notes

### The Infrastructure Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         VPC (Default or Simple)                      │
│                                                                      │
│   ┌─────────────────┐              ┌──────────────────────────────┐ │
│   │   RDS Postgres  │◄─────────────│   g4dn.2xlarge (Spot)        │ │
│   │   + pgvector    │   port 5432  │   - Deep Learning AMI        │ │
│   │                 │              │   - PyTorch + CUDA           │ │
│   │   db.t4g.large  │              │   - Pipeline code            │ │
│   └─────────────────┘              └──────────────────────────────┘ │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### RDS Configuration

```hcl
resource "aws_db_instance" "main" {
  identifier           = "ftm-db"
  engine               = "postgres"
  engine_version       = "15"
  instance_class       = "db.t4g.large"  # 8GB RAM, ARM-based (cheaper)
  allocated_storage    = 100              # GB, enough for embeddings

  db_name              = "ftm"
  username             = var.db_username
  password             = var.db_password

  # CRITICAL: Must be stoppable for cost management
  # Aurora Serverless cannot be stopped - do NOT use it

  skip_final_snapshot  = true  # Dev environment
  publicly_accessible  = true  # For EC2 access (or use VPC peering)

  # pgvector installed via: CREATE EXTENSION vector;
}
```

### EC2 Spot Instance Launch

```bash
#!/bin/bash
# launch_pipeline.sh

aws ec2 run-instances \
  --image-id ami-0xxxxxxxxxx  # Deep Learning AMI (Ubuntu 20.04)
  --instance-type g4dn.2xlarge \
  --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time"}}' \
  --key-name your-key \
  --security-group-ids sg-xxxxxxxx \
  --user-data file://setup_ec2.sh \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=ftm-pipeline}]'
```

### EC2 Setup Script

```bash
#!/bin/bash
# setup_ec2.sh (user data)

# Clone repo
git clone https://github.com/your-repo/Financial_Topic_Modeling.git
cd Financial_Topic_Modeling

# Install dependencies
pip install -r cloud/requirements.txt
python -m spacy download en_core_web_sm

# Set environment variables
export DATABASE_URL="postgresql://user:pass@rds-endpoint:5432/ftm"
export XAI_API_KEY="your-key"

# Run pipeline
python scripts/run_unified_pipeline.py
```

### pgvector Extension

After RDS is created, connect and enable pgvector:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

This must be done before the pipeline runs. Can be automated in setup script.

### Security Group Rules

```hcl
# Allow PostgreSQL from EC2
ingress {
  from_port   = 5432
  to_port     = 5432
  protocol    = "tcp"
  cidr_blocks = ["0.0.0.0/0"]  # Or restrict to EC2 security group
}

# Allow SSH for debugging
ingress {
  from_port   = 22
  to_port     = 22
  protocol    = "tcp"
  cidr_blocks = ["your-ip/32"]
}
```

---

## Cost Management

### Stop RDS When Not In Use

```bash
#!/bin/bash
# stop_rds.sh

aws rds stop-db-instance --db-instance-identifier ftm-db

echo "RDS stopped. Will auto-start after 7 days if not manually started."
echo "Storage cost while stopped: ~$0.10/GB/month = ~$10/month for 100GB"
```

### Start RDS Before Pipeline Run

```bash
#!/bin/bash
# start_rds.sh

aws rds start-db-instance --db-instance-identifier ftm-db

echo "RDS starting. Wait ~5 minutes for availability."
aws rds wait db-instance-available --db-instance-identifier ftm-db
echo "RDS available."
```

### Cost Breakdown

| Component | Cost |
|-----------|------|
| g4dn.2xlarge spot (4 hours) | ~$1.00 |
| RDS db.t4g.large (4 hours running) | ~$0.30 |
| RDS storage (100GB, always) | ~$10/month |
| xAI API calls | ~$20-100 (varies) |
| **Total per quarterly run** | **~$25-100** (dominated by LLM) |

The compute infrastructure cost is negligible. LLM costs dominate.

---

## What NOT to Build

The following are explicitly out of scope. Do not add them:

1. **CI/CD Pipeline** - Manual deployment is fine for quarterly runs
2. **Auto Scaling** - Single instance is sufficient
3. **Load Balancer** - No web traffic
4. **CloudWatch Dashboards** - Pipeline logs to stdout
5. **Lambda Functions** - No event-driven processing needed
6. **Step Functions** - The whole point of the pivot was to eliminate this
7. **ECS/EKS** - No containers needed
8. **NAT Gateway** - Use public subnets or VPC endpoints if needed
9. **Multiple Environments** - Dev only for now

---

## Validation Checklist

Before declaring Phase 4 complete, verify:

### Infrastructure
- [ ] RDS PostgreSQL accessible from EC2
- [ ] pgvector extension enabled
- [ ] Security groups allow required traffic
- [ ] Spot instance launches successfully
- [ ] RDS can be stopped and started

### Pipeline
- [ ] Pipeline connects to cloud RDS
- [ ] All firms process without errors
- [ ] Topics have LLM summaries
- [ ] Themes have LLM descriptions
- [ ] Embeddings stored correctly

### Queries
- [ ] Hierarchical query works: `SELECT * FROM themes JOIN topics ON ... JOIN sentences ON ...`
- [ ] Vector search works: `SELECT * FROM topics ORDER BY embedding <-> query_embedding LIMIT 10`

### Cost
- [ ] Total run cost documented
- [ ] RDS stop script works
- [ ] Spot instance terminates after pipeline

---

## Review Protocol

Your work will be reviewed at the Phase 4 completion point. Expect reviews that check:

1. **Simplicity**: Is this the minimum infrastructure needed? Could anything be removed?
2. **Cost**: Can RDS be stopped? Are we using spot instances? No unnecessary services?
3. **Security**: Are credentials handled properly? No secrets in code?
4. **Documentation**: Can someone else deploy this? Are the scripts documented?
5. **Validation**: Did the pipeline actually run end-to-end? What were the results?

---

## The Why Behind Phase 4

The pipeline produces valuable research output: cross-firm investment themes identified from earnings call transcripts. But it's useless if it can't run at scale.

Phase 4 is where the pipeline becomes production-ready:
1. **Scalability**: Process 3,000+ firms, not just 11
2. **Reliability**: Spot instance interruption handling via checkpoints
3. **Cost-effectiveness**: <$5 compute per quarterly run
4. **Reproducibility**: Anyone can deploy and run with the scripts

This is not about building impressive infrastructure. It's about deploying the minimum infrastructure that lets the research happen.

---

## Final Reminders

1. **Read before you write.** Understand the existing pipeline before deploying it.

2. **Keep it simple.** If you're writing more than 200 lines of Terraform, something is wrong.

3. **Test locally first.** Run the pipeline locally before cloud deployment.

4. **Document everything.** The next person (or your future self) needs to understand this.

5. **Stop at the halt point.** Phase 4 complete → present implementation → await approval.

6. **Cost consciousness.** Every AWS resource costs money. Justify each one.

---

The database layer is ready. The unified pipeline is proven. The LLM integration works.

Now deploy it to the cloud and validate at scale.

Let's build something we can be proud of.
