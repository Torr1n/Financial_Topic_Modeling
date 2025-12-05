# Senior Engineer Plan: Architecture Pivot

**Feature:** Cloud Architecture Simplification - AWS Batch → Single GPU Instance
**Priority:** Critical (Blocks Phase 4 Infrastructure)
**Estimated Scope:** Medium (Refactor, not rewrite)

---

## 1. Executive Summary

### The Pivot
Migrate from a distributed AWS Batch map-reduce architecture to a single GPU instance (g4dn.2xlarge) with PostgreSQL + pgvector storage, replacing DynamoDB.

### Why Now
- Caught over-engineering before deploying infrastructure
- Phases 1-3 complete with working containers (firm_processor, theme_aggregator)
- ~2 hours from deploying infrastructure we would have regretted

### Core Insight
For our scale (~2M sentences quarterly, 3,000-5,000 firms), the orchestration overhead of AWS Batch exceeds the compute time. A single GPU processes all firms in ~2-4 hours for ~$1.00.

---

## 2. Architectural Changes

### Before (Planned)
```
┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ S3 (Input)  │────▶│   AWS Batch     │────▶│   AWS Batch     │────▶ DynamoDB
│             │     │   MAP (N pods)  │     │   REDUCE (1)    │
└─────────────┘     └─────────────────┘     └─────────────────┘
                          │                        │
                    Model loads N×            Model loads 1×
                    Cold starts N×            S3 round-trip
```

### After (Pivot)
```
┌─────────────┐     ┌─────────────────────────────────────────┐
│ S3/Local    │────▶│         g4dn.2xlarge (Spot)            │────▶ PostgreSQL
│ (Input)     │     │  ┌─────────────┐  ┌─────────────────┐  │     + pgvector
└─────────────┘     │  │ Stage 1:    │  │ Stage 2:        │  │
                    │  │ Firm Topics │─▶│ Cross-Firm      │  │
                    │  │ (Loop)      │  │ Themes          │  │
                    │  └─────────────┘  └─────────────────┘  │
                    │       Model loads ONCE                  │
                    └─────────────────────────────────────────┘
```

---

## 3. What Changes

### 3.1 Infrastructure (Replace)

| Component | Old | New |
|-----------|-----|-----|
| Compute | AWS Batch (CPU spot) | EC2 g4dn.2xlarge (GPU spot) |
| Storage | DynamoDB (single-table) | RDS PostgreSQL + pgvector |
| Orchestration | Step Functions | Single Python script |
| Intermediate | S3 JSON files | In-memory + Postgres checkpoint |

### 3.2 Code (Adapt vs Replace)

| Module | Action | Rationale |
|--------|--------|-----------|
| `models.py` | **Keep** | Data models unchanged |
| `interfaces.py` | **Keep** | TopicModel abstraction still valid |
| `bertopic_model.py` | **Keep** | Core ML logic unchanged |
| `firm_processor.py` | **Adapt** | Remove S3 I/O, add Postgres writes |
| `theme_aggregator.py` | **Adapt** | Runs in same process, direct handoff |
| `dynamodb_utils.py` | **Replace** | New PostgreSQL layer |
| `s3_utils.py` | **Simplify** | Optional checkpoint only |
| Container Dockerfiles | **Eliminate** | Single script, no containers |
| Terraform (Phase 4) | **Rewrite** | RDS + EC2 instead of Batch |

### 3.3 New Components (Add)

| Component | Purpose |
|-----------|---------|
| `database/models.py` | SQLAlchemy + pgvector schema |
| `database/repository.py` | CRUD operations for all entities |
| `pipeline/unified_pipeline.py` | Single orchestration script |
| `pipeline/checkpoint.py` | Spot instance resume logic |
| `llm/xai_client.py` | Async xAI API for topic/theme naming |

---

## 4. Database Schema Design

### 4.1 PostgreSQL Tables (Relational)

```sql
-- Enable pgvector
CREATE EXTENSION vector;

-- Firms table
CREATE TABLE firms (
    id SERIAL PRIMARY KEY,
    company_id VARCHAR(50) UNIQUE NOT NULL,  -- From CSV companyid
    ticker VARCHAR(20),
    name VARCHAR(255),
    quarter VARCHAR(10),  -- e.g., "2023Q1"
    processed_at TIMESTAMP
);

-- Sentences table (bulk of data)
CREATE TABLE sentences (
    id BIGSERIAL PRIMARY KEY,
    firm_id INTEGER REFERENCES firms(id),
    text TEXT NOT NULL,
    position INTEGER,
    speaker_type VARCHAR(50),
    topic_id INTEGER,  -- Nullable until assigned
    embedding vector(768)  -- pgvector column
);

-- Topics table (firm-level)
CREATE TABLE topics (
    id SERIAL PRIMARY KEY,
    firm_id INTEGER REFERENCES firms(id),
    local_topic_id INTEGER,  -- BERTopic's topic number
    representation TEXT,  -- Keywords
    summary TEXT,  -- LLM-generated (nullable)
    n_sentences INTEGER,
    theme_id INTEGER,  -- Nullable until reduce phase
    embedding vector(768)
);

-- Themes table (cross-firm)
CREATE TABLE themes (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    description TEXT,
    n_topics INTEGER,
    n_firms INTEGER,
    embedding vector(768),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes (built AFTER bulk insert)
-- CREATE INDEX ON sentences USING hnsw (embedding vector_cosine_ops);
-- CREATE INDEX ON topics USING hnsw (embedding vector_cosine_ops);
-- CREATE INDEX ON themes USING hnsw (embedding vector_cosine_ops);
```

### 4.2 Why This Schema

1. **Natural hierarchy**: Foreign keys encode Theme → Topics → Sentences → Firms
2. **Traceability queries**: Simple JOINs replace N+1 DynamoDB lookups
3. **Vector search**: pgvector enables semantic RAG without additional infrastructure
4. **Deferred indexing**: Bulk insert to heap, build HNSW index once at end

---

## 5. Processing Architecture

### 5.1 Two-Stage Monolith

```python
# Pseudo-code for unified_pipeline.py

async def main():
    # === SETUP (Once) ===
    db = init_database()
    embedding_model = load_embedding_model("cuda")  # Loaded ONCE
    xai_client = init_xai_client()

    # === STAGE 1: Firm Processing ===
    firms = get_unprocessed_firms(db)  # Resume support

    for firm in firms:
        # 1. Load & embed sentences (GPU)
        sentences = load_firm_sentences(firm)
        embeddings = embedding_model.encode(sentences, device="cuda")

        # 2. Cluster into topics (BERTopic)
        topics, assignments = bertopic.fit_transform(sentences, embeddings)

        # 3. Generate topic summaries (async LLM)
        summaries = await generate_summaries(xai_client, topics)

        # 4. Embed summaries (GPU - reuse model)
        topic_embeddings = embedding_model.encode(summaries, device="cuda")

        # 5. Write to Postgres (checkpoint)
        write_firm_results(db, firm, sentences, embeddings, topics, topic_embeddings)

    # === STAGE 2: Theme Aggregation ===
    all_topics = db.query(Topic).all()
    topic_texts = [t.summary for t in all_topics]

    # Re-embed all topic summaries
    global_embeddings = embedding_model.encode(topic_texts, device="cuda")

    # Cluster into themes
    themes, assignments = bertopic.fit_transform(topic_texts, global_embeddings)

    # Generate theme descriptions (async LLM)
    theme_descriptions = await generate_theme_descriptions(xai_client, themes)

    # Write themes, update topic.theme_id
    write_themes(db, themes, theme_descriptions, assignments)

    # === FINALIZE ===
    build_vector_indexes(db)  # HNSW indexes built at end

if __name__ == "__main__":
    asyncio.run(main())
```

### 5.2 Spot Instance Resume Logic

```python
def get_unprocessed_firms(db) -> List[Firm]:
    """Resume from where we left off if spot instance was interrupted."""
    processed_ids = db.query(Firm.company_id).filter(Firm.processed_at.isnot(None)).all()
    all_firm_ids = get_all_firm_ids_from_source()
    return [f for f in all_firm_ids if f not in processed_ids]
```

### 5.3 Async LLM Integration

```python
import asyncio
from openai import AsyncOpenAI  # xAI uses OpenAI-compatible API

sem = asyncio.Semaphore(50)  # Rate limit protection

async def generate_summary(client, topic_info: dict) -> str:
    async with sem:
        response = await client.chat.completions.create(
            model="grok-beta",
            messages=[{
                "role": "user",
                "content": f"Summarize this topic in 1-2 sentences: {topic_info}"
            }]
        )
        return response.choices[0].message.content
```

---

## 6. Infrastructure (Simplified Terraform)

### 6.1 What We Need

```
cloud/terraform/
├── main.tf           # Root module
├── variables.tf      # Input variables
├── outputs.tf        # Output values
├── modules/
│   ├── database/     # RDS PostgreSQL
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── outputs.tf
│   └── compute/      # EC2 spot instance (for reference)
│       └── ...
└── environments/
    └── dev.tfvars
```

### 6.2 RDS PostgreSQL Module

```hcl
resource "aws_db_instance" "main" {
  identifier           = "${var.project_name}-db-${var.environment}"
  engine               = "postgres"
  engine_version       = "15"
  instance_class       = "db.t4g.large"  # 8GB RAM
  allocated_storage    = 100

  db_name              = "ftm"
  username             = var.db_username
  password             = var.db_password

  vpc_security_group_ids = [aws_security_group.db.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name

  skip_final_snapshot  = true  # Dev environment

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}
```

### 6.3 Compute (Manual for Now)

The g4dn.2xlarge spot instance will be launched manually or via a simple script:
- Deep Learning AMI (Ubuntu 20.04) with PyTorch
- Security group allowing SSH + Postgres access
- User data script to clone repo and install dependencies

---

## 7. Cost Analysis

### Old Architecture (AWS Batch)
- 3,000 container cold starts × 60s minimum billing = massive waste
- DynamoDB write capacity for 2M+ items
- S3 I/O between map and reduce phases
- Step Functions execution costs

### New Architecture (Single GPU)
- g4dn.2xlarge spot: ~$0.25/hour × 4 hours = **~$1.00**
- RDS db.t4g.large: $0 when stopped, ~$2/month storage
- xAI API: $20-100 (dominates regardless of architecture)

**Total compute cost reduction: 90%+**

---

## 8. Acceptance Criteria

### Functional Requirements
- [ ] Process all firms from CSV dataset
- [ ] Generate firm-level topics with LLM summaries
- [ ] Aggregate topics into cross-firm themes
- [ ] Store all results in PostgreSQL with embeddings
- [ ] Support resume from spot instance interruption
- [ ] Enable hierarchical queries (theme → topics → sentences → firms)

### Non-Functional Requirements
- [ ] Complete processing in <4 hours for 3,000 firms
- [ ] Compute cost <$5 per quarterly run
- [ ] Code coverage >80%
- [ ] All existing unit tests pass (adapted)

### Technical Requirements
- [ ] Embedding model loaded once, reused for all operations
- [ ] Async LLM calls with rate limiting (semaphore)
- [ ] Per-firm checkpoint to Postgres (resume support)
- [ ] Deferred vector index build (performance)

---

## 9. Migration Strategy

### Phase 1: Database Layer
1. Design and implement SQLAlchemy models
2. Write repository layer with CRUD operations
3. Test with local Postgres (Docker)

### Phase 2: Pipeline Unification
1. Create unified_pipeline.py combining firm_processor + theme_aggregator
2. Adapt existing logic (remove S3 I/O, add Postgres writes)
3. Add checkpoint/resume logic

### Phase 3: LLM Integration
1. Implement async xAI client with rate limiting
2. Add topic summarization to firm processing
3. Add theme description to aggregation

### Phase 4: Infrastructure
1. Simplified Terraform for RDS only
2. Launch script for g4dn.2xlarge spot
3. End-to-end testing on cloud

---

## 10. What We Preserve

The core ML logic remains unchanged:
- `BERTopicModel.fit_transform()` - Same interface
- `FirmProcessor.process()` - Adapted, not rewritten
- `ThemeAggregator.aggregate()` - Adapted, not rewritten
- Data models (`FirmTopicOutput`, `ThemeOutput`) - Extended with embeddings
- Test fixtures and validation logic - Adapted for new storage

**This is a reorganization, not a rewrite.**
