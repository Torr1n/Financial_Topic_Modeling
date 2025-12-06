# API & Database Schema

## Overview

This document defines the database schema, vector configurations, and key API contracts for the Financial Topic Modeling pipeline.

---

## Database Schema

### Entity Relationship Diagram

```
┌─────────────────┐
│     themes      │
├─────────────────┤
│ id (PK)         │
│ name            │
│ description     │
│ embedding       │──── VECTOR(768)
│ n_topics        │
│ n_firms         │
│ created_at      │
└────────┬────────┘
         │ 1:N
         ▼
┌─────────────────┐       ┌─────────────────┐
│     topics      │       │      firms      │
├─────────────────┤       ├─────────────────┤
│ id (PK)         │       │ id (PK)         │
│ theme_id (FK)   │───────│ company_id (UK) │
│ firm_id (FK)    │──────▶│ name            │
│ representation  │       │ processed_at    │
│ summary         │       │ created_at      │
│ embedding       │       └─────────────────┘
│ sentence_count  │
│ created_at      │
└────────┬────────┘
         │ 1:N
         ▼
┌─────────────────┐
│   sentences     │
├─────────────────┤
│ id (PK)         │
│ topic_id (FK)   │
│ firm_id (FK)    │
│ raw_text        │
│ cleaned_text    │
│ embedding       │──── VECTOR(768)
│ probability     │
│ created_at      │
└─────────────────┘
```

---

## Table Definitions

### `firms`

Stores company metadata.

```sql
CREATE TABLE firms (
    id SERIAL PRIMARY KEY,
    company_id VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(500) NOT NULL,
    processed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_firms_company_id ON firms(company_id);
CREATE INDEX idx_firms_processed_at ON firms(processed_at);
```

| Column       | Type         | Description                               |
| ------------ | ------------ | ----------------------------------------- |
| id           | SERIAL       | Auto-increment primary key                |
| company_id   | VARCHAR(255) | Unique identifier (e.g., "AAPL")          |
| name         | VARCHAR(500) | Full company name                         |
| processed_at | TIMESTAMP    | When firm was processed (NULL if pending) |
| created_at   | TIMESTAMP    | Record creation time                      |

### `sentences`

Stores transcript sentences with embeddings.

```sql
CREATE TABLE sentences (
    id SERIAL PRIMARY KEY,
    firm_id INTEGER REFERENCES firms(id),
    topic_id INTEGER REFERENCES topics(id),
    raw_text TEXT NOT NULL,
    cleaned_text TEXT NOT NULL,
    embedding VECTOR(768),
    probability FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_sentences_firm_id ON sentences(firm_id);
CREATE INDEX idx_sentences_topic_id ON sentences(topic_id);
```

| Column       | Type        | Description                                  |
| ------------ | ----------- | -------------------------------------------- |
| id           | SERIAL      | Auto-increment primary key                   |
| firm_id      | INTEGER     | Foreign key to firms                         |
| topic_id     | INTEGER     | Foreign key to topics (assigned by BERTopic) |
| raw_text     | TEXT        | Original sentence                            |
| cleaned_text | TEXT        | Preprocessed sentence                        |
| embedding    | VECTOR(768) | Sentence embedding vector                    |
| probability  | FLOAT       | Topic assignment probability                 |

### `topics`

Stores firm-level topics with LLM summaries.

```sql
CREATE TABLE topics (
    id SERIAL PRIMARY KEY,
    firm_id INTEGER REFERENCES firms(id),
    theme_id INTEGER REFERENCES themes(id),
    representation JSONB NOT NULL,
    summary TEXT,
    embedding VECTOR(768),
    sentence_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_topics_firm_id ON topics(firm_id);
CREATE INDEX idx_topics_theme_id ON topics(theme_id);
```

| Column         | Type        | Description                                                 |
| -------------- | ----------- | ----------------------------------------------------------- |
| id             | SERIAL      | Auto-increment primary key                                  |
| firm_id        | INTEGER     | Foreign key to firms                                        |
| theme_id       | INTEGER     | Foreign key to themes (NULL until aggregation)              |
| representation | JSONB       | BERTopic keywords (e.g., `["AI", "GPU", "infrastructure"]`) |
| summary        | TEXT        | LLM-generated topic summary                                 |
| embedding      | VECTOR(768) | Summary embedding vector                                    |
| sentence_count | INTEGER     | Number of sentences in topic                                |

### `themes`

Stores cross-firm themes.

```sql
CREATE TABLE themes (
    id SERIAL PRIMARY KEY,
    name VARCHAR(500) NOT NULL,
    description TEXT,
    embedding VECTOR(768),
    n_topics INTEGER,
    n_firms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

| Column      | Type         | Description                      |
| ----------- | ------------ | -------------------------------- |
| id          | SERIAL       | Auto-increment primary key       |
| name        | VARCHAR(500) | Short theme name (LLM-generated) |
| description | TEXT         | Extended theme description       |
| embedding   | VECTOR(768)  | Theme embedding vector           |
| n_topics    | INTEGER      | Number of topics in theme        |
| n_firms     | INTEGER      | Number of unique firms in theme  |

---

## Vector Configuration

### pgvector Extension

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### HNSW Indexes

Built after bulk insert for performance:

```sql
-- Sentence embeddings (largest table)
CREATE INDEX idx_sentences_embedding ON sentences
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 128);

-- Topic embeddings
CREATE INDEX idx_topics_embedding ON topics
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Theme embeddings
CREATE INDEX idx_themes_embedding ON themes
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

### Index Parameters

| Parameter           | Value  | Description                                                |
| ------------------- | ------ | ---------------------------------------------------------- |
| `m`                 | 16     | Max connections per node                                   |
| `ef_construction`   | 64-128 | Build-time accuracy (higher = slower build, better recall) |
| `vector_cosine_ops` | -      | Cosine distance operator                                   |

---

## Key Queries

### Hierarchical Traversal

**Theme → Topics → Sentences**

```sql
-- Get all sentences for a theme
SELECT
    th.name AS theme_name,
    t.summary AS topic_summary,
    f.name AS firm_name,
    s.cleaned_text AS sentence
FROM themes th
JOIN topics t ON t.theme_id = th.id
JOIN sentences s ON s.topic_id = t.id
JOIN firms f ON s.firm_id = f.id
WHERE th.id = :theme_id
ORDER BY f.name, t.id;
```

**Firm → Topics → Themes**

```sql
-- Get all themes a firm contributes to
SELECT DISTINCT
    f.name AS firm_name,
    th.name AS theme_name,
    th.n_firms AS firms_in_theme
FROM firms f
JOIN topics t ON t.firm_id = f.id
JOIN themes th ON t.theme_id = th.id
WHERE f.company_id = :company_id;
```

### Semantic Search

**Find similar topics**

```sql
-- Input: query_embedding (768-dim vector)
SELECT
    t.summary,
    f.name AS firm,
    1 - (t.embedding <=> :query_embedding) AS similarity
FROM topics t
JOIN firms f ON t.firm_id = f.id
ORDER BY t.embedding <=> :query_embedding
LIMIT 10;
```

**Find similar themes**

```sql
SELECT
    name,
    description,
    n_firms,
    1 - (embedding <=> :query_embedding) AS similarity
FROM themes
ORDER BY embedding <=> :query_embedding
LIMIT 5;
```

### Analytics

**Theme distribution**

```sql
SELECT
    th.name,
    th.n_firms,
    th.n_topics,
    ROUND(th.n_firms::numeric / (SELECT COUNT(*) FROM firms) * 100, 2) AS firm_coverage_pct
FROM themes th
ORDER BY th.n_firms DESC;
```

**Topic quality (sentence coverage)**

```sql
SELECT
    t.id,
    t.summary,
    t.sentence_count,
    COUNT(s.id) AS actual_sentences,
    ROUND(AVG(s.probability), 3) AS avg_probability
FROM topics t
LEFT JOIN sentences s ON s.topic_id = t.id
GROUP BY t.id
ORDER BY t.sentence_count DESC
LIMIT 20;
```

---

## API Contracts

### Pipeline Entry Point

**Script**: `scripts/run_unified_pipeline.py`

**Environment Variables**:

| Variable              | Required | Description                     |
| --------------------- | -------- | ------------------------------- |
| `DATABASE_URL`        | Yes      | PostgreSQL connection string    |
| `XAI_API_KEY`         | No       | xAI API key for summaries       |
| `CONFIG_PATH`         | No       | Path to YAML config             |
| `TEST_MODE`           | No       | Set to `mag7` for test firms    |
| `MAX_FIRMS`           | No       | Limit number of firms           |
| `EMBEDDING_DIMENSION` | No       | Vector dimension (default: 768) |

### LLM API (xAI)

**Endpoint**: `https://api.x.ai/v1/chat/completions`

**Request (Topic Summary)**:

```json
{
  "model": "grok-4-1-fast-reasoning",
  "messages": [
    {
      "role": "system",
      "content": "You are a financial analyst. Summarize the following topic in 1-2 sentences. Focus on the business concept, not specific companies."
    },
    {
      "role": "user",
      "content": "Keywords: AI, GPU, infrastructure, investment, capacity\nSample sentences: [...]"
    }
  ],
  "max_tokens": 150
}
```

**Response**:

```json
{
  "choices": [
    {
      "message": {
        "content": "Companies are investing heavily in AI infrastructure, particularly GPU capacity, to support growing computational demands."
      }
    }
  ]
}
```

---

## Data Validation Rules

### Theme Validation

```python
# From theme_aggregator.py
MIN_FIRMS = 2           # Theme must span at least 2 firms
MAX_FIRM_DOMINANCE = 0.4  # No single firm > 40% of theme topics
```

### Topic Quality

- Minimum 3 sentences per topic
- Probability threshold: 0.1 (outliers below this assigned to -1)

### Embedding Consistency

- All embeddings must be 768-dimensional (or configured dimension)
- Null embeddings indicate processing failure

---

## Migration Notes

### Changing Embedding Dimension

If upgrading to a larger embedding model (e.g., Qwen3 with 4096 dim):

1. Update `cloud/config/production.yaml`:

   ```yaml
   embedding:
     dimension: 4096
   ```

2. Set environment variable:

   ```bash
   export EMBEDDING_DIMENSION=4096
   ```

3. Recreate tables (dimension is fixed at creation):
   ```sql
   DROP TABLE sentences, topics, themes CASCADE;
   -- Re-run pipeline to recreate
   ```

### Bulk Insert Performance

For large loads, indexes are built after insertion:

```python
# From repository.py
def rebuild_vector_indexes(self):
    # Drop existing indexes
    session.execute("DROP INDEX IF EXISTS idx_sentences_embedding")
    # ... bulk insert ...
    # Rebuild with HNSW
    session.execute("CREATE INDEX idx_sentences_embedding ...")
```
