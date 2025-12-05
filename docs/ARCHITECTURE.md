# Architecture Documentation

This document provides a technical deep-dive into the Financial Topic Modeling pipeline architecture, design decisions, and implementation details.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Pipeline Stages](#pipeline-stages)
3. [Data Flow](#data-flow)
4. [Key Components](#key-components)
5. [Database Schema](#database-schema)
6. [GPU Acceleration](#gpu-acceleration)
7. [LLM Integration](#llm-integration)
8. [Infrastructure](#infrastructure)
9. [Design Decisions](#design-decisions)

---

## System Overview

The Financial Topic Modeling pipeline is a hierarchical NLP system that:

1. **Ingests** earnings call transcripts (CSV or S3)
2. **Embeds** sentences using sentence-transformers
3. **Clusters** sentences into firm-level topics (BERTopic)
4. **Summarizes** topics using LLM (xAI/Grok)
5. **Aggregates** topics into cross-firm themes
6. **Stores** results in PostgreSQL with vector search capability

### Architecture Principles

| Principle | Implementation |
|-----------|----------------|
| **Simplicity** | Single GPU instance vs distributed batch |
| **Cost-efficiency** | Spot instances + stoppable RDS |
| **Resumability** | Per-firm checkpoints for fault tolerance |
| **Configurability** | YAML-driven hyperparameters |
| **Extensibility** | Abstract interfaces for topic models |

---

## Pipeline Stages

### Stage 1: Data Ingestion

```
CSV/S3 → LocalCSVConnector → FirmTranscriptData
```

**Input Format:**
```
companyid, companyname, transcriptid, componenttext, componentorder, speakertypename
```

**Processing:**
1. Load raw transcript components
2. Split into sentences (SpaCy)
3. Filter operator/boilerplate text
4. NLP preprocessing (lemmatization, NER filtering)
5. Generate `TranscriptSentence` objects

**Output:** `FirmTranscriptData` with `raw_text` and `cleaned_text` per sentence

### Stage 2: Sentence Embedding

```
FirmTranscriptData → SentenceTransformer → embeddings[N, 768]
```

**Model:** `all-mpnet-base-v2` (768 dimensions)
- Loaded ONCE at pipeline initialization
- Reused for all firms (no redundant loading)
- GPU-accelerated encoding

**Critical Design Decision:** Embeddings are computed from `cleaned_text` (preprocessed) but `raw_text` is preserved for LLM context.

### Stage 3: Firm-Level Topic Modeling

```
embeddings → UMAP → HDBSCAN → BERTopic → FirmTopicOutput
```

**BERTopic Configuration (firm-level):**
```yaml
umap:
  n_neighbors: 15      # Local structure
  n_components: 10     # Reduced dimensions
  min_dist: 0.01
  metric: "cosine"

hdbscan:
  min_cluster_size: 6  # Small clusters OK
  min_samples: 2
  cluster_selection_method: "leaf"  # Prefer smaller clusters
```

**Output per firm:**
- ~25 topics (varies by firm size)
- Topic keywords/representation
- Sentence-to-topic assignments
- Topic probability distributions

### Stage 4: LLM Topic Summarization

```
topics + sentences → xAI/Grok → topic.summary
```

**Prompt Design:**
```
Keywords: {keywords}
Sentences: {sample_sentences}

Generate a 1-2 sentence summary that:
- Captures the underlying theme
- Avoids company-specific terms
- Is generalizable across firms
```

**Why this matters:** Raw keywords like "cloud, azure, infrastructure" are ambiguous. LLM summaries like "Enterprise cloud migration strategy and infrastructure modernization" enable better theme clustering.

### Stage 5: Cross-Firm Theme Aggregation

```
all_topics → embed(summaries) → UMAP → HDBSCAN → themes
```

**BERTopic Configuration (theme-level):**
```yaml
umap:
  n_neighbors: 30      # Global structure
  n_components: 15     # More dimensions for complexity
  min_dist: 0.025
  metric: "cosine"

hdbscan:
  min_cluster_size: 20  # Themes must be substantial
  min_samples: 6
  cluster_selection_method: "eom"  # Larger clusters
```

**Validation Rules:**
- `min_firms: 2` — Themes must span multiple companies
- `max_firm_dominance: 0.4` — No single firm > 40% of theme

### Stage 6: Storage & Indexing

```
themes, topics, sentences → PostgreSQL + pgvector
```

**Deferred Indexing:** HNSW vector indexes built AFTER bulk insert (100x faster than incremental).

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            DATA FLOW                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  CSV File                                                                │
│      │                                                                   │
│      ▼                                                                   │
│  LocalCSVConnector.fetch_transcripts()                                   │
│      │                                                                   │
│      ▼                                                                   │
│  TranscriptData { firm_id → FirmTranscriptData }                        │
│      │                                                                   │
│      ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  FOR EACH FIRM:                                                  │    │
│  │                                                                  │    │
│  │  sentences.cleaned_text                                          │    │
│  │      │                                                           │    │
│  │      ▼                                                           │    │
│  │  SentenceTransformer.encode() → embeddings[N, 768]               │    │
│  │      │                                                           │    │
│  │      ▼                                                           │    │
│  │  FirmProcessor.process(firm_data, embeddings)                    │    │
│  │      │                                                           │    │
│  │      ├── BERTopicModel.fit_transform()                          │    │
│  │      │       │                                                   │    │
│  │      │       ├── UMAP (dim reduction)                           │    │
│  │      │       ├── HDBSCAN (clustering)                           │    │
│  │      │       └── Representations (KeyBERT, MMR, POS)            │    │
│  │      │                                                           │    │
│  │      └── FirmTopicOutput { topics, assignments }                │    │
│  │              │                                                   │    │
│  │              ▼                                                   │    │
│  │  XAIClient.generate_batch_summaries(topics)                     │    │
│  │      │                                                           │    │
│  │      ▼                                                           │    │
│  │  topics[].summary = "LLM generated summary"                     │    │
│  │      │                                                           │    │
│  │      ▼                                                           │    │
│  │  DatabaseRepository.write(firm, topics, sentences)              │    │
│  │      │                                                           │    │
│  │      ▼                                                           │    │
│  │  CHECKPOINT: firm.processed_at = now()                          │    │
│  │                                                                  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ALL FIRMS COMPLETE                                                      │
│      │                                                                   │
│      ▼                                                                   │
│  ThemeAggregator.aggregate(all_firm_topics)                             │
│      │                                                                   │
│      ├── Extract topic summaries as documents                           │
│      ├── Embed summaries                                                │
│      ├── BERTopicModel.fit_transform() [theme-level config]            │
│      ├── Validate themes (min_firms, max_dominance)                    │
│      └── XAIClient.generate_theme_descriptions()                       │
│              │                                                          │
│              ▼                                                          │
│  DatabaseRepository.write_themes()                                      │
│      │                                                                   │
│      ▼                                                                   │
│  DatabaseRepository.build_vector_indexes() [HNSW]                       │
│      │                                                                   │
│      ▼                                                                   │
│  PIPELINE COMPLETE                                                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Key Components

### UnifiedPipeline (`cloud/src/pipeline/unified_pipeline.py`)

The main orchestrator that:
- Loads embedding model ONCE
- Creates separate BERTopicModel instances for firm/theme
- Manages database sessions
- Coordinates LLM calls
- Handles checkpointing

```python
class UnifiedPipeline:
    def __init__(self, database_url, config, device):
        self.embedding_model = SentenceTransformer(...)  # Loaded ONCE
        self.firm_topic_model = BERTopicModel(firm_config, embedding_model)
        self.theme_topic_model = BERTopicModel(theme_config, embedding_model)
        self._xai_client = XAIClient(...)

    def run(self, data_source):
        self._process_firms(data_source)   # Stage 1-4
        self._aggregate_themes()            # Stage 5-6
```

### BERTopicModel (`cloud/src/topic_models/bertopic_model.py`)

BERTopic wrapper implementing the `TopicModel` interface:
- Accepts pre-computed embeddings (efficiency)
- Configurable UMAP/HDBSCAN parameters
- Multiple representation models (KeyBERT, MMR, POS)
- GPU acceleration via cuML (when available)

```python
class BERTopicModel(TopicModel):
    def fit_transform(self, documents, embeddings=None) -> TopicModelResult:
        # Uses injected embeddings OR computes internally
        # Returns: topic_assignments, representations, probabilities
```

### XAIClient (`cloud/src/llm/xai_client.py`)

Async LLM client for topic/theme summarization:
- OpenAI-compatible API (xAI uses same protocol)
- Semaphore-based rate limiting (50 concurrent)
- Exponential backoff retry
- Graceful fallback to keywords on failure

```python
class XAIClient:
    async def generate_topic_summary(self, keywords, sentences) -> str
    async def generate_theme_description(self, keywords, topic_summaries) -> str
```

### DatabaseRepository (`cloud/src/database/repository.py`)

Data access layer with:
- Bulk insert operations (optimized for performance)
- Hierarchical queries (theme → topics → sentences)
- Vector index management
- Checkpoint tracking

---

## Database Schema

```sql
-- Firms: Companies in the dataset
CREATE TABLE firms (
    id SERIAL PRIMARY KEY,
    company_id VARCHAR(50) UNIQUE NOT NULL,
    ticker VARCHAR(20),
    name VARCHAR(255),
    quarter VARCHAR(10),
    processed_at TIMESTAMP  -- NULL = not yet processed
);

-- Sentences: Individual transcript sentences
CREATE TABLE sentences (
    id BIGSERIAL PRIMARY KEY,
    firm_id INTEGER REFERENCES firms(id),
    raw_text TEXT NOT NULL,      -- Original for LLM context
    cleaned_text TEXT NOT NULL,  -- Preprocessed for embedding
    position INTEGER NOT NULL,
    speaker_type VARCHAR(50),
    topic_id INTEGER REFERENCES topics(id),
    embedding VECTOR(768)        -- Configurable dimension
);

-- Topics: Firm-level topic clusters
CREATE TABLE topics (
    id SERIAL PRIMARY KEY,
    firm_id INTEGER REFERENCES firms(id),
    local_topic_id INTEGER NOT NULL,
    representation TEXT NOT NULL,  -- Keywords
    summary TEXT,                  -- LLM-generated
    n_sentences INTEGER DEFAULT 0,
    theme_id INTEGER REFERENCES themes(id),
    embedding VECTOR(768)
);

-- Themes: Cross-firm theme clusters
CREATE TABLE themes (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,              -- LLM-generated
    n_topics INTEGER DEFAULT 0,
    n_firms INTEGER DEFAULT 0,
    embedding VECTOR(768),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Vector indexes (built after bulk insert)
CREATE INDEX ix_sentences_embedding ON sentences USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ix_topics_embedding ON topics USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ix_themes_embedding ON themes USING hnsw (embedding vector_cosine_ops);
```

**Why pgvector?**
- Native PostgreSQL integration
- HNSW indexes for fast approximate nearest neighbor
- Enables semantic search: "Find topics similar to X"

---

## GPU Acceleration

### Embedding (SentenceTransformer)

```python
model = SentenceTransformer("all-mpnet-base-v2", device="cuda")
embeddings = model.encode(texts)  # 10x faster than CPU
```

### Dimensionality Reduction (cuML UMAP)

```python
from cuml.manifold import UMAP as cuUMAP

umap_model = cuUMAP(
    n_neighbors=15,
    n_components=10,
    metric="cosine"
)
```

### Clustering (cuML HDBSCAN)

```python
from cuml.cluster import HDBSCAN as cuHDBSCAN

hdbscan_model = cuHDBSCAN(
    min_cluster_size=6,
    min_samples=2,
    gen_min_span_tree=True,
    prediction_data=True
)
```

**Fallback:** If cuML unavailable, uses CPU implementations (umap-learn, hdbscan).

---

## LLM Integration

### Topic Summary Prompt

```
## ROLE
Financial analyst specializing in earnings call themes.

## CONTEXT
Topic keywords: {keywords}
Sample sentences:
- {sentence_1}
- {sentence_2}
...

## OBJECTIVE
Generate 1-2 sentence summary capturing the underlying theme.
AVOID company-specific terms (must be generalizable).

## OUTPUT
Raw text only.
```

### Theme Description Prompt

```
## ROLE
Cross-firm theme analyst.

## CONTEXT
Theme keywords: {keywords}
Contributing topic summaries:
- {topic_summary_1}
- {topic_summary_2}
...

## OBJECTIVE
Generate 2-3 sentence description of the cross-firm theme.

## OUTPUT
Raw text only.
```

---

## Infrastructure

### AWS Resources

| Resource | Type | Purpose |
|----------|------|---------|
| RDS | db.t4g.large | PostgreSQL + pgvector |
| EC2 | g4dn.2xlarge (spot) | GPU compute |
| S3 | Bucket | Code + data delivery |
| IAM | Role + Instance Profile | EC2 → S3 access |

### Cost Optimization

| Strategy | Savings |
|----------|---------|
| Spot instances | 60-70% vs on-demand |
| Stoppable RDS | $0/hr when stopped |
| Single instance | No orchestration overhead |
| Deferred indexing | 100x faster bulk insert |

---

## Design Decisions

### Why Single GPU vs Distributed?

**Problem:** Original design used AWS Batch for parallel firm processing.

**Insight:** For ~3,000 firms with ~300 sentences each:
- Container cold starts: 60s × 3,000 = 50 hours of wasted compute
- Model loading: 3,000 redundant loads
- Orchestration: Step Functions + S3 intermediate storage

**Solution:** Single g4dn.2xlarge processes all firms sequentially in ~4 hours for ~$1.

### Why Separate Firm/Theme Configs?

**Problem:** Same hyperparameters for both stages.

**Insight:**
- Firm-level: ~300 sentences → ~25 topics (local structure important)
- Theme-level: ~750+ topics → ~100 themes (global structure important)

**Solution:** Separate `firm_topic_model` and `theme_topic_model` configs.

### Why LLM Summaries?

**Problem:** Topic keywords are ambiguous ("cloud, infrastructure, scaling").

**Insight:** Same keywords could mean:
- Cloud computing infrastructure
- Weather patterns
- Conceptual cloud/abstract thinking

**Solution:** LLM reads keywords + actual sentences → unambiguous summary.

### Why Summary-Based Theme Clustering?

**Problem:** Clustering on keywords produces poor themes.

**Insight:** LLM summaries contain richer semantic content than keyword lists.

**Solution:** Embed summaries (not keywords) for theme clustering.

---

## Performance Characteristics

| Stage | Time (11 firms) | Time (3,000 firms est.) |
|-------|-----------------|-------------------------|
| Setup & Dependencies | 3 min | 3 min |
| Embedding Load | 1 min | 1 min |
| Firm Processing | 8 min | ~3 hours |
| Theme Aggregation | 2 min | ~30 min |
| Index Building | 1 min | ~10 min |
| **Total** | **~15 min** | **~4 hours** |

---

## Extension Points

1. **New Topic Models**: Implement `TopicModel` interface
2. **New Data Sources**: Implement `DataConnector` interface
3. **New LLM Providers**: Modify `XAIClient` (OpenAI-compatible)
4. **New Embedding Models**: Change `embedding.model` in config
5. **Downstream Analysis**: Query themes table, join with market data
