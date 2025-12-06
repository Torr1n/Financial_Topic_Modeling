# ADR-002: PostgreSQL + pgvector vs DynamoDB

**Status**: Accepted
**Decision Makers**: Project Lead + AI Architecture Consultation (Gemini)

---

## Context

The pipeline produces hierarchical data with complex traceability requirements:

```
Theme → (N) Topics → (N) Firms → (N) Sentences
```

Users need to:

1. **Traverse up**: "Which theme does this sentence belong to?"
2. **Traverse down**: "Show me all sentences in Theme X"
3. **Semantic search**: "Find topics similar to 'AI infrastructure'"

The original design used **DynamoDB** following AWS best practices for serverless architectures. However, this decision needed re-evaluation given the specific data model.

---

## Decision

**We chose PostgreSQL with pgvector extension over DynamoDB.**

- RDS PostgreSQL 15 (`db.t4g.large`)
- pgvector extension for vector similarity search
- Relational schema with foreign keys
- HNSW indexes for vector queries

---

## Rationale

### 1. The "Traceability Problem"

From Gemini consultation:

> "Your requirements explicitly describe a highly relational, hierarchical dataset with a need for deep traversal (joins). Using DynamoDB here would force you into 'Single Table Design' gymnastics, making your application code brittle and your queries expensive."

**In PostgreSQL:**

```sql
-- "Show me all sentences in Theme X"
SELECT s.text, f.name
FROM sentences s
JOIN topics t ON s.topic_id = t.id
JOIN themes th ON t.theme_id = th.id
JOIN firms f ON s.firm_id = f.id
WHERE th.id = 'theme_123';
```

**In DynamoDB (N+1 Query Hell):**

```python
# Step 1: Get theme
theme = table.get_item(PK='THEME#123')
# Step 2: Query all topics in theme
topics = table.query(PK='THEME#123', SK__begins_with='TOPIC#')
# Step 3: For EACH topic, query sentences
for topic in topics:
    sentences = table.query(PK=f'TOPIC#{topic.id}', SK__begins_with='SENT#')
    # ... this explodes into hundreds of queries
```

### 2. Vector Search Requirements

Future work requires semantic search across themes, topics, and sentences. pgvector provides native support:

```sql
-- Find topics similar to a query embedding
SELECT * FROM topics
ORDER BY embedding <=> '[0.1, 0.2, ...]'::vector
LIMIT 10;
```

**DynamoDB alternative**: Would require a separate vector database (Pinecone, Weaviate) adding:

- Additional service cost
- Data synchronization complexity
- Multiple points of failure

### 3. Cost Model

| Aspect       | DynamoDB       | PostgreSQL (RDS)  |
| ------------ | -------------- | ----------------- |
| Write cost   | $1.25/M writes | Included          |
| Read cost    | $0.25/M reads  | Included          |
| Storage      | $0.25/GB/month | ~$0.10/GB/month   |
| Idle cost    | $0 (on-demand) | ~$0.08/hr running |
| Stopped cost | N/A            | $0/hr + storage   |

**For this workload (quarterly batch)**:

- DynamoDB: Pay per request, no "off" switch
- PostgreSQL: Stop instance between runs, pay only storage (~$10/month)

### 4. Query Flexibility

DynamoDB requires knowing access patterns upfront. PostgreSQL allows ad-hoc queries:

```sql
-- "Which firms contribute most to AI-related themes?"
SELECT f.name, COUNT(DISTINCT th.id) as theme_count
FROM firms f
JOIN topics t ON t.firm_id = f.id
JOIN themes th ON t.theme_id = th.id
WHERE th.description ILIKE '%AI%'
GROUP BY f.id
ORDER BY theme_count DESC;
```

This query would require a complex GSI design in DynamoDB or be impossible without data duplication.

---

## Consequences

### Positive

1. **Natural Hierarchy**: Foreign keys enforce data integrity
2. **Flexible Queries**: Any SQL query without upfront GSI planning
3. **Vector Search**: Native pgvector support for embeddings
4. **Cost Control**: Stop RDS when not in use
5. **Familiar Tooling**: Standard SQL clients (DBeaver, psql)
6. **ACID Transactions**: Consistent writes during pipeline execution

### Negative

1. **Infrastructure Management**: Must manage RDS instance (vs. serverless DynamoDB)
2. **Start/Stop Latency**: RDS takes ~5 minutes to start
3. **Scaling Limits**: Single instance (sufficient for this workload, but no auto-scaling)
4. **Connection Management**: Requires connection pooling for high concurrency

### Mitigations

- **Start/Stop Scripts**: `start_rds.sh` and `stop_rds.sh` for cost management
- **Terraform**: Infrastructure as code ensures reproducibility
- **Connection Limits**: Pipeline uses single connection; not an issue

---

## Alternatives Considered

### Alternative 1: DynamoDB with Single Table Design

- Complex partition/sort key overloading
- GSIs for each access pattern
- Data denormalization required
- **Rejected**: Complexity doesn't match workload

### Alternative 2: DynamoDB + Pinecone (Vector DB)

- DynamoDB for relational data
- Pinecone for vector search
- **Rejected**: Two services to manage, sync complexity

### Alternative 3: Aurora Serverless v2

- Serverless PostgreSQL
- Auto-scaling
- **Rejected**: Minimum ~$45/month; doesn't scale to zero

### Alternative 4: SQLite (Local)

- Zero infrastructure
- Single file database
- **Rejected**: No remote access for analytics; no vector support

---

## Evidence

### Source Documents

1. **Gemini Consultation**: `docs/ai-log/gemini-conversation.md` (lines 585-667)
2. **Database Models**: `cloud/src/database/models.py`
3. **Repository Layer**: `cloud/src/database/repository.py`

### Key Quotes

> "DynamoDB is excellent for high-throughput, simple key-value lookups, your requirements explicitly describe a highly relational, hierarchical dataset with a need for deep traversal (joins)."

> "In a Relational Database (SQL), this is a simple JOIN. In DynamoDB, to answer 'Show me all sentences in Theme X,' you have two bad options."

### Schema Design

```sql
-- Hierarchical schema with vector columns
CREATE TABLE themes (
    id SERIAL PRIMARY KEY,
    name TEXT,
    description TEXT,
    embedding VECTOR(768),
    n_topics INT,
    n_firms INT
);

CREATE TABLE topics (
    id SERIAL PRIMARY KEY,
    theme_id INT REFERENCES themes(id),
    firm_id INT REFERENCES firms(id),
    representation JSONB,
    summary TEXT,
    embedding VECTOR(768)
);

CREATE TABLE sentences (
    id SERIAL PRIMARY KEY,
    topic_id INT REFERENCES topics(id),
    firm_id INT REFERENCES firms(id),
    raw_text TEXT,
    cleaned_text TEXT,
    embedding VECTOR(768)
);
```

---

## Learning Outcome

This decision reinforced that **technology choices should match data models**, not follow industry trends:

- DynamoDB is excellent for: High-scale key-value, event streams, session stores
- PostgreSQL is excellent for: Hierarchical data, complex queries, vector search

The "serverless" appeal of DynamoDB was outweighed by the complexity cost of forcing a relational data model into a key-value store.
