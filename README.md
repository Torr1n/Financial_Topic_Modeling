# Financial Topic Modeling Pipeline

**Identifying Cross-Firm Investment Themes from Earnings Call Transcripts**

An NLP research pipeline that discovers universal investment themes by analyzing thousands of earnings call transcripts using hierarchical topic modeling with BERTopic, GPU acceleration, and LLM-powered summarization.

---

## Research Motivation

Earnings calls contain rich, unstructured information about corporate strategy, market conditions, and industry trends. While individual transcripts are valuable, the real insight comes from identifying **cross-firm themes**—topics that emerge across multiple companies, revealing broader market narratives.

This pipeline answers: *"What are the universal themes being discussed across all firms in a given quarter?"*

---

## Key Results

**Validated Cloud Run (MAG7 + Tech Firms):**

| Metric | Value |
|--------|-------|
| Firms Processed | 11 |
| Sentences Analyzed | ~5,000 |
| Firm-Level Topics | ~350 |
| Cross-Firm Themes | ~19 |
| Processing Time | ~15 minutes |
| Infrastructure Cost | ~$1.30 |

**Example Themes Discovered:**
- AI infrastructure investment and GPU demand
- Cloud migration and digital transformation
- Supply chain optimization
- Regulatory compliance and data privacy
- Customer acquisition and retention strategies

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FINANCIAL TOPIC MODELING PIPELINE                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌─────────────┐    ┌──────────────────┐    ┌──────────────────────────────┐│
│  │  Earnings   │───▶│  Sentence        │───▶│  Firm-Level Topic Modeling   ││
│  │  Transcripts│    │  Embeddings      │    │  (BERTopic per firm)         ││
│  │  (CSV/S3)   │    │  (all-mpnet-v2)  │    │  ~25 topics per firm         ││
│  └─────────────┘    └──────────────────┘    └──────────────────────────────┘│
│                                                        │                      │
│                                                        ▼                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                     LLM Topic Summarization (xAI/Grok)                   ││
│  │     Keywords + Sentences → "AI infrastructure investment discussion"     ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                        │                      │
│                                                        ▼                      │
│  ┌──────────────────────────────────────────────────────────────────────────┐│
│  │                    Cross-Firm Theme Aggregation                          ││
│  │     Re-cluster ~350 topics → ~19 universal themes                        ││
│  │     Validation: min_firms=2, max_dominance=0.4                           ││
│  └──────────────────────────────────────────────────────────────────────────┘│
│                                                        │                      │
│                                                        ▼                      │
│  ┌──────────────────────────────────────────────────────────────────────────┐│
│  │                    PostgreSQL + pgvector Storage                          ││
│  │     Hierarchy: Theme → Topics → Sentences → Firms                        ││
│  │     Vector search enabled for semantic queries                           ││
│  └──────────────────────────────────────────────────────────────────────────┘│
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Infrastructure (Simplified Cloud Architecture):**

```
┌─────────────────────────────────────────────────────────────────┐
│                      AWS (us-east-1)                             │
│                                                                  │
│   ┌─────────────────┐         ┌────────────────────────────┐   │
│   │  RDS PostgreSQL │◄────────│  EC2 g4dn.2xlarge (Spot)   │   │
│   │  + pgvector     │         │  - Deep Learning AMI       │   │
│   │  db.t4g.large   │         │  - GPU: NVIDIA T4          │   │
│   │  (stoppable)    │         │  - cuML acceleration       │   │
│   └─────────────────┘         └────────────────────────────┘   │
│                                         │                       │
│   ┌─────────────────┐                   │                       │
│   │  S3 Bucket      │───────────────────┘                       │
│   │  (code + data)  │                                           │
│   └─────────────────┘                                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Docker (for local PostgreSQL)
- AWS CLI (for cloud deployment)
- Terraform 1.0+ (for cloud deployment)

### Local Development

```bash
# Clone repository
git clone <repository-url>
cd Financial_Topic_Modeling

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r cloud/requirements.txt
python -m spacy download en_core_web_sm

# Start local PostgreSQL
docker-compose up -d

# Run pipeline with test firms
DATABASE_URL="postgresql://ftm:ftm_password@localhost:5432/ftm" \
python scripts/run_unified_pipeline_mag7.py
```

### Cloud Deployment

```bash
# 1. Configure Terraform
cd cloud/terraform
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your AWS settings

# 2. Deploy infrastructure
terraform init && terraform apply

# 3. Upload code to S3
cd ../scripts && ./upload_to_s3.sh

# 4. Run pipeline (MAG7 validation)
TEST_MODE=mag7 ./launch_pipeline.sh

# 5. Stop RDS when done (save costs)
./stop_rds.sh
```

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for detailed instructions.

---

## Project Structure

```
Financial_Topic_Modeling/
├── cloud/                      # Cloud-native implementation
│   ├── config/                 # Pipeline configuration
│   │   └── production.yaml     # Main config file
│   ├── src/                    # Source code
│   │   ├── pipeline/           # Unified pipeline orchestration
│   │   ├── topic_models/       # BERTopic implementation
│   │   ├── database/           # PostgreSQL + pgvector models
│   │   ├── llm/                # xAI/Grok integration
│   │   └── connectors/         # Data source connectors
│   ├── terraform/              # Infrastructure as code
│   └── scripts/                # Deployment scripts
├── scripts/                    # Pipeline entry points
│   ├── run_unified_pipeline.py # Cloud/local runner
│   └── run_unified_pipeline_mag7.py  # Test runner
├── docs/                       # Documentation
│   ├── ARCHITECTURE.md         # Technical architecture
│   ├── DEPLOYMENT.md           # Cloud deployment guide
│   ├── CONFIGURATION.md        # Config reference
│   └── DEVELOPMENT.md          # Contributor guide
├── tests/                      # Test suite
└── legacy/                     # Legacy MVP code (reference only)
```

---

## Configuration

All pipeline settings are in `cloud/config/production.yaml`:

```yaml
# Embedding model
embedding:
  model: "all-mpnet-base-v2"  # or "Qwen/Qwen3-Embedding-8B" for SOTA
  dimension: 768
  device: "cuda"

# Firm-level topic modeling (per-firm)
firm_topic_model:
  umap: { n_neighbors: 15, n_components: 10 }
  hdbscan: { min_cluster_size: 6 }

# Theme-level topic modeling (cross-firm)
theme_topic_model:
  umap: { n_neighbors: 30, n_components: 15 }
  hdbscan: { min_cluster_size: 20 }

# LLM for summarization
llm:
  model: "grok-4-1-fast-reasoning"
  max_concurrent: 50
```

See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for full reference.

---

## Data Schema

**Hierarchical Output Structure:**

```
themes (cross-firm)
  └── topics (firm-level)
        └── sentences (transcript)
              └── firms (companies)
```

**PostgreSQL Tables:**

| Table | Description | Key Fields |
|-------|-------------|------------|
| `firms` | Companies processed | company_id, name, processed_at |
| `sentences` | Transcript sentences | raw_text, cleaned_text, embedding |
| `topics` | Firm-level topics | representation, summary, embedding |
| `themes` | Cross-firm themes | name, description, embedding, n_firms |

**Vector Search Enabled:**
```sql
-- Find similar topics by semantic meaning
SELECT * FROM topics
ORDER BY embedding <-> '[query_embedding]'
LIMIT 10;
```

---

## Key Features

- **Hierarchical Topic Modeling**: Two-stage BERTopic (firm → theme)
- **GPU Acceleration**: cuML for 10-100x faster UMAP/HDBSCAN
- **LLM Summarization**: Human-readable topic/theme descriptions via xAI
- **Checkpoint/Resume**: Spot instance resilience with per-firm checkpoints
- **Vector Search**: pgvector enables semantic similarity queries
- **Configurable**: Separate hyperparameters for firm vs theme clustering
- **Cost-Optimized**: Spot instances + stoppable RDS = ~$1/run

---

## Future Work

- [ ] Sentiment analysis integration (FinBERT)
- [ ] Event study framework for theme-based trading signals
- [ ] Time-series theme evolution tracking
- [ ] Interactive dashboard for theme exploration
- [ ] WRDS direct integration for live transcript data
- [ ] Multi-quarter longitudinal analysis

---

## Research Team

Built as part of financial NLP research. For questions or collaboration, contact the repository maintainers.

---

## License

[Add appropriate license]

---

## Acknowledgments

- BERTopic by Maarten Grootendorst
- Sentence Transformers by UKPLab
- RAPIDS/cuML by NVIDIA
- xAI for Grok API access
