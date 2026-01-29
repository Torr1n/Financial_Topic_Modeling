# ADR-007: S3/Parquet Storage Strategy

## Status
Accepted

## Date
2026-01-26

## Context

The Financial Topic Modeling pipeline currently stores results in PostgreSQL with pgvector:
- Sentences, topics, themes, and firms as relational tables
- Embeddings stored as vector columns
- RDS `db.t4g.large` instance (~$95/month when running)

### Current Pain Points
1. **Cost**: RDS costs ~$95/month even when stopped 7 days max
2. **Scaling**: Multi-quarter data will grow PostgreSQL storage significantly
3. **Cold data**: After processing, data is queried infrequently (sentiment analysis only)
4. **Query patterns**: Primarily analytical (aggregate by theme, firm, quarter)

### Requirements
1. Store 8+ quarters of earnings call data
2. Enable efficient queries for sentiment analysis
3. Minimize storage costs for "cold" historical data
4. Support Athena-style SQL queries when needed
5. Allow parallel writes from Batch jobs without coordination

## Decision

**Migrate from PostgreSQL to S3/Parquet with Athena for queries, partitioned by quarter only.**

### Why Parquet
- **Columnar format**: Excellent compression (~10x vs JSON), efficient column scans
- **Schema-on-read**: Athena handles schema without explicit catalog
- **Ecosystem support**: PyArrow, Pandas, Spark, Athena, Glue all native

### Why S3
- **Cost**: $0.023/GB/month (vs $0.115/GB for RDS gp3)
- **Durability**: 99.999999999% (11 nines)
- **Scalability**: Unlimited storage, no provisioning
- **Parallel writes**: Multiple Batch jobs can write concurrently

### Why Athena
- **Pay-per-query**: $5/TB scanned (Parquet + partitions = minimal scan)
- **Standard SQL**: Familiar syntax for sentiment analysis joins
- **Serverless**: No infrastructure to manage
- **Integration**: Works directly with S3 Parquet files

### Sentiment Integration Contract

**Decision**: Sentiment analysis module will be **refactored to conform to the Parquet schema**, not the other way around.

**Rationale**: The existing sentiment analysis code has an unnecessarily complex input contract (JSON with specific structure). Since we're designing the data lake from scratch, the Parquet schema becomes the source of truth. Minimal refactoring of the sentiment module to read Parquet is simpler than contorting the data lake to match legacy JSON expectations.

**Contract**: The `sentiment-ready/` prefix contains denormalized Parquet files that directly provide all fields needed for:
1. Theme-level sentiment aggregation
2. Firm-within-theme sentiment scoring
3. Event study date alignment (via `earnings_call_date`)
4. CRSP joins (via `permno`)

**Build timing**: `sentiment-ready/` is produced by a **post-reduce packaging step** that joins `processed/theme_contributions.parquet` with `processed/sentences.parquet`. This keeps the reduce phase lightweight.

### Partitioning Strategy

**Partition by quarter only. NOT by firm.**

The prior architecture discussion (`docs/ai-log/claude-2026-batch-convo.md`) explicitly warned against firm-level partitioning:

> "The 'Small File' Problem: Athena suffers massive performance penalties when reading thousands of tiny files. 5,000 firms × 8 quarters = 40,000 files."

```
# CORRECT - Partition by quarter only
s3://bucket/sentences/quarter=2023Q1/batch_001.parquet
s3://bucket/sentences/quarter=2023Q1/batch_002.parquet

# WRONG - Do not partition by firm
s3://bucket/sentences/quarter=2023Q1/firm=AAPL/data.parquet  # ❌
```

### S3 Bucket Structure

```
s3://financial-topic-modeling-prod/
│
├── prefetch/
│   └── transcripts/                         # WRDS prefetch (preprocessed, MFA mitigation)
│       └── quarter=2023Q1/
│           ├── chunk_0000.parquet           # ~100-200 firms per chunk
│           ├── chunk_0001.parquet
│           ├── ...
│           ├── manifest.json                # firm_id -> chunk mapping (gzip)
│           └── _checkpoint.json             # resumable prefetch state
│
├── intermediate/
│   └── firm-topics/                         # Map phase output
│       └── quarter=2023Q1/
│           ├── batch_000_part_00.parquet    # Firms 0-49
│           ├── batch_000_part_01.parquet    # Firms 50-99
│           ├── batch_001_part_00.parquet    # Firms 100-149
│           └── ...
│
├── processed/
│   ├── firms/                               # Firm metadata
│   │   └── quarter=2023Q1/
│   │       └── firms.parquet
│   │
│   ├── sentences/                           # All sentences with assignments
│   │   └── quarter=2023Q1/
│   │       └── sentences.parquet
│   │
│   ├── topics/                              # Firm-level topics
│   │   └── quarter=2023Q1/
│   │       └── topics.parquet
│   │
│   └── themes/                              # Cross-firm themes
│       └── quarter=2023Q1/
│           ├── themes.parquet               # Theme metadata
│           └── theme_contributions.parquet  # Theme → firm mappings
│
└── sentiment-ready/                         # Denormalized for sentiment
    └── quarter=2023Q1/
        └── themes_for_sentiment.parquet
```

### Prefetch Staging Layer (MFA Mitigation)

**Decision**: Use a prefetch staging layer to avoid WRDS MFA in Batch jobs.

**Details**:
- Prefetch runs once per quarter from a fixed-IP machine.
- Data is **preprocessed** (cleaned_text, sentence_id, etc.) and written to `prefetch/transcripts/`.
- Batch jobs read via `S3TranscriptConnector` using `manifest.json` (firm_id → chunk file mapping).
- Prefetch data is **not** used by reduce directly; it only feeds the map phase.

### Parquet Schemas

#### firms.parquet
| Column | Type | Description |
|--------|------|-------------|
| firm_id | STRING | Capital IQ companyid |
| firm_name | STRING | Company name |
| ticker | STRING | Stock ticker |
| gvkey | STRING | Compustat identifier |
| permno | INT64 | **CRSP PERMNO** |
| link_date | DATE | GVKEY-PERMNO link date |
| earnings_call_date | DATE | Transcript date |
| quarter | STRING | Partition key |
| n_sentences | INT32 | Sentence count |
| n_topics | INT32 | Topic count |

#### sentences.parquet
| Column | Type | Description |
|--------|------|-------------|
| sentence_id | STRING | `{firm_id}_{transcript}_{position}` |
| firm_id | STRING | FK to firms |
| permno | INT64 | Denormalized for joins |
| transcript_id | STRING | Source transcript |
| earnings_call_date | DATE | For event study |
| raw_text | STRING | Original text |
| cleaned_text | STRING | Preprocessed |
| speaker_type | STRING | CEO, CFO, Analyst, etc. |
| position | INT32 | Order in transcript |
| local_topic_id | INT32 | Topic ID within firm |
| quarter | STRING | Partition key |

#### topics.parquet
| Column | Type | Description |
|--------|------|-------------|
| topic_id | STRING | `{firm_id}_{local_topic_id}` |
| firm_id | STRING | FK to firms |
| local_topic_id | INT32 | BERTopic topic number |
| permno | INT64 | Denormalized |
| gvkey | STRING | Denormalized |
| earnings_call_date | DATE | For event study |
| representation | STRING | BERTopic keywords |
| summary | STRING | LLM-generated |
| keywords | LIST<STRING> | Top keywords |
| n_sentences | INT32 | Sentence count |
| theme_id | STRING | Assigned after reduce |
| quarter | STRING | Partition key |

#### themes.parquet
| Column | Type | Description |
|--------|------|-------------|
| theme_id | STRING | `theme_{quarter}_{seq}` |
| name | STRING | BERTopic representation |
| description | STRING | LLM-generated |
| keywords | LIST<STRING> | Theme keywords |
| n_topics | INT32 | Contributing topics |
| n_firms | INT32 | Distinct firms |
| quarter | STRING | Partition key |

#### theme_contributions.parquet
| Column | Type | Description |
|--------|------|-------------|
| theme_id | STRING | FK to themes |
| firm_id | STRING | FK to firms |
| firm_name | STRING | Denormalized |
| permno | INT64 | **For event study** |
| gvkey | STRING | For Compustat joins |
| earnings_call_date | DATE | **Event date** |
| topic_id | STRING | Which topic contributed |
| n_sentences | INT32 | Sentences in contribution |
| quarter | STRING | Partition key |

### Reduce Phase Input Scope

**Decision**: Theme aggregation (reduce phase) reads **only `topics.parquet`**, not sentences.

**Rationale**:
- Theme clustering uses topic summaries and keywords, not raw sentences
- Sentences are only needed for sentiment analysis (downstream)
- Reading only topics.parquet minimizes S3 read costs and memory usage
- Topics table is ~100x smaller than sentences table

### Sentiment-Ready Build Step

**Decision**: Build `sentiment-ready/` **after** reduce completes.

**Rationale**:
- Reduce phase should remain small and deterministic (topics only)
- Sentiment packaging requires sentence text and should not expand reduce scope

**Implementation Note**:
Run a lightweight packaging job that:
1. Reads `processed/theme_contributions.parquet` for theme/topic mappings
2. Reads `processed/sentences.parquet` for sentence text
3. Joins on `(firm_id, local_topic_id)` (topic_id = `{firm_id}_{local_topic_id}`)
4. Writes `sentiment-ready/quarter={Q}/themes_for_sentiment.parquet`

**Data flow**:
```
Prefetch (fixed IP) → prefetch/transcripts/ (preprocessed)
           ↓
Map Phase → intermediate/firm-topics/ (contains topics + sentences)
           ↓
Reduce Phase → reads topics only → writes themes.parquet
           ↓
Post-process → writes sentiment-ready/ (joins themes + sentences)
```

### Chunk Writing Pattern

Per prior discussion, Batch jobs should write in chunks to avoid memory issues:

```python
buffer = []
chunk_counter = 0

for firm in firms_to_process:
    firm_data = process_firm(firm)
    buffer.append(firm_data)

    if len(buffer) >= 50:  # Every 50 firms
        write_parquet_chunk(buffer, chunk_counter)
        buffer = []
        chunk_counter += 1

# Write remaining
if buffer:
    write_parquet_chunk(buffer, chunk_counter)
```

## Consequences

### Positive
- **90% cost reduction**: ~$2/month storage vs ~$95/month RDS
- **Parallel writes**: Batch jobs write independently
- **Pay-per-query**: Only pay when analyzing data
- **Compression**: Parquet compresses 10x vs JSON
- **Scalability**: No storage provisioning limits

### Negative
- **No ACID transactions**: S3 is eventually consistent
- **Query latency**: Athena has 1-5 second startup overhead
- **No indexes**: Full partition scans (mitigated by column pruning)
- **Schema evolution**: Requires careful Parquet schema versioning

### Mitigations
- **Consistency**: Use unique filenames per Batch job; no overwrites
- **Latency**: Acceptable for batch analytics (not real-time)
- **Indexes**: Partition pruning + Parquet predicate pushdown sufficient
- **Schema**: Use backward-compatible additions only

## Athena Query Examples

### Theme-Level Sentiment (for sentiment analysis)
```sql
SELECT
    theme_id,
    theme_name,
    COUNT(*) as total_sentences,
    SUM(CASE WHEN sentiment = 'positive' THEN 1 ELSE 0 END) as positive,
    (positive - negative) * 1.0 / COUNT(*) as sentiment_score
FROM processed.themes t
JOIN processed.sentences s ON t.theme_id = s.theme_id
WHERE t.quarter = '2023Q1'
GROUP BY theme_id, theme_name;
```

### Firm-Within-Theme Sentiment
```sql
SELECT
    tc.theme_id,
    tc.firm_id,
    tc.firm_name,
    tc.permno,
    tc.earnings_call_date,
    COUNT(*) as n_sentences
FROM processed.theme_contributions tc
JOIN processed.sentences s
    ON tc.firm_id = s.firm_id AND tc.topic_id = s.local_topic_id
WHERE tc.quarter = '2023Q1'
    AND tc.permno IS NOT NULL
GROUP BY tc.theme_id, tc.firm_id, tc.firm_name, tc.permno, tc.earnings_call_date;
```

## Alternatives Considered

### 1. Keep PostgreSQL + pgvector
- **Pro**: Familiar, relational, vector search
- **Con**: $95/month, doesn't scale well for cold data

### 2. Aurora Serverless v2
- **Pro**: Scales to near-zero, relational
- **Con**: Still ~$43/month minimum, cold start latency

### 3. DynamoDB On-Demand
- **Pro**: True pay-per-use, scales to zero
- **Con**: Not relational, requires denormalization, expensive for scans

### 4. S3 + JSON
- **Pro**: Simple, flexible schema
- **Con**: 10x larger files, slower queries, no column pruning

## Implementation Notes

### PyArrow Write Pattern
```python
import pyarrow as pa
import pyarrow.parquet as pq

def write_firms_parquet(firms: List[dict], s3_path: str):
    schema = pa.schema([
        ("firm_id", pa.string()),
        ("firm_name", pa.string()),
        ("permno", pa.int64()),
        ("gvkey", pa.string()),
        ("earnings_call_date", pa.date32()),
        ("quarter", pa.string()),
        # ... other columns
    ])

    table = pa.Table.from_pylist(firms, schema=schema)

    pq.write_table(
        table,
        s3_path,
        compression="snappy",
        use_deprecated_int96_timestamps=False,
    )
```

### Glue Catalog Setup
```hcl
resource "aws_glue_catalog_database" "ftm" {
  name = "financial_topic_modeling"
}

resource "aws_glue_catalog_table" "sentences" {
  name          = "sentences"
  database_name = aws_glue_catalog_database.ftm.name

  table_type = "EXTERNAL_TABLE"

  parameters = {
    "classification" = "parquet"
  }

  storage_descriptor {
    location      = "s3://${aws_s3_bucket.data.id}/processed/sentences/"
    input_format  = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat"
    output_format = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat"

    ser_de_info {
      serialization_library = "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"
    }

    columns {
      name = "sentence_id"
      type = "string"
    }
    # ... other columns
  }

  partition_keys {
    name = "quarter"
    type = "string"
  }
}
```

## References

- [Apache Parquet Format](https://parquet.apache.org/)
- [Amazon Athena Best Practices](https://docs.aws.amazon.com/athena/latest/ug/performance-tuning.html)
- [PyArrow Parquet Documentation](https://arrow.apache.org/docs/python/parquet.html)
- Prior discussion: `docs/ai-log/claude-2026-batch-convo.md` (small file problem)
