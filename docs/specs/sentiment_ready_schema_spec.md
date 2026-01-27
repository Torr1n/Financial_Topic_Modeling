# Sentiment-Ready Parquet Schema Specification

## Overview

The `sentiment-ready/` S3 prefix contains denormalized Parquet files designed for direct consumption by the sentiment analysis module. This schema is the **source of truth** - the sentiment module will be refactored to conform to it.

**Build timing**: The sentiment-ready dataset is produced by a **post-reduce packaging step** that joins `processed/theme_contributions.parquet` with `processed/sentences.parquet`. The reduce phase itself still reads **only** `topics.parquet` (per ADR-007).

## Location

```
s3://financial-topic-modeling-prod/sentiment-ready/quarter={quarter}/themes_for_sentiment.parquet
```

## Schema Definition

### PyArrow Schema

```python
import pyarrow as pa

# Sentence struct (nested within theme contributions)
sentence_struct = pa.struct([
    ("sentence_id", pa.string()),      # "{firm_id}_{transcript_id}_{position:04d}"
    ("text", pa.string()),             # Raw text for FinBERT classification
    ("speaker_type", pa.string()),     # CEO, CFO, Analyst, Operator, Unknown
    ("position", pa.int32()),          # Order in transcript
])

# Main schema
sentiment_ready_schema = pa.schema([
    # Theme identification
    ("theme_id", pa.string()),         # "theme_{quarter}_{seq:03d}"
    ("theme_name", pa.string()),       # BERTopic representation
    ("theme_description", pa.string()),# LLM-generated description
    ("theme_keywords", pa.list_(pa.string())),  # Top keywords

    # Firm identification (REQUIRED for event study)
    ("firm_id", pa.string()),          # Capital IQ companyid
    ("firm_name", pa.string()),        # Company name
    ("permno", pa.int64()),            # CRSP PERMNO - REQUIRED, NOT NULL
    ("gvkey", pa.string()),            # Compustat GVKEY

    # Event study timing
    ("earnings_call_date", pa.date32()),  # Event date for CAR calculation
    ("link_date", pa.date32()),        # GVKEY-PERMNO link validity date

    # Topic linkage
    ("topic_id", pa.string()),         # Firm-level topic that contributed
    ("topic_summary", pa.string()),    # LLM-generated topic summary

    # Sentences for sentiment classification
    ("sentences", pa.list_(sentence_struct)),  # Nested sentence data
    ("n_sentences", pa.int32()),       # Count for validation

    # Partition key
    ("quarter", pa.string()),          # "2023Q1" - also S3 partition
])
```

### Column Descriptions

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `theme_id` | string | No | Unique theme identifier |
| `theme_name` | string | No | BERTopic keyword representation |
| `theme_description` | string | Yes | LLM description (may fail) |
| `theme_keywords` | list<string> | No | Top theme keywords |
| `firm_id` | string | No | Capital IQ company ID |
| `firm_name` | string | No | Company name |
| `permno` | int64 | **No** | CRSP identifier (firms without PERMNO are excluded) |
| `gvkey` | string | No | Compustat identifier |
| `earnings_call_date` | date | No | Transcript date = event date |
| `link_date` | date | No | GVKEY-PERMNO link validity |
| `topic_id` | string | No | Contributing firm-level topic (row grain key) |
| `topic_summary` | string | Yes | LLM topic summary |
| `sentences` | list<struct> | No | Nested sentences for FinBERT |
| `n_sentences` | int32 | No | Sentence count |
| `quarter` | string | No | Partition key |

### Important Constraints

1. **PERMNO is NOT NULL**: Firms without PERMNO linkage are excluded during WRDS ingestion (per ADR-004)
2. **Row grain**: One row per **(theme_id, firm_id, topic_id)** contribution
3. **Sentences are nested**: Each row contains sentences from that firm-topic only

## Example Data

```python
{
    "theme_id": "theme_2023Q1_007",
    "theme_name": "supply chain, logistics, inventory",
    "theme_description": "Discussion of supply chain challenges...",
    "theme_keywords": ["supply", "chain", "logistics", "inventory", "shipping"],

    "firm_id": "374372246",
    "firm_name": "Lamb Weston Holdings, Inc.",
    "permno": 16431,
    "gvkey": "123456",

    "earnings_call_date": date(2023, 1, 5),
    "link_date": date(2022, 1, 1),

    "topic_id": "374372246_3",
    "topic_summary": "Supply chain improvements and inventory management...",

    "sentences": [
        {
            "sentence_id": "374372246_T001_0045",
            "text": "We've seen significant improvements in our supply chain.",
            "speaker_type": "CEO",
            "position": 45
        },
        {
            "sentence_id": "374372246_T001_0046",
            "text": "Inventory levels are now normalized.",
            "speaker_type": "CEO",
            "position": 46
        }
    ],
    "n_sentences": 2,

    "quarter": "2023Q1"
}
```

## Athena Query Examples

### Theme-Level Sentiment (after FinBERT classification)

```sql
SELECT
    theme_id,
    theme_name,
    COUNT(*) as total_sentences,
    SUM(CASE WHEN sentiment = 'positive' THEN 1 ELSE 0 END) as positive,
    SUM(CASE WHEN sentiment = 'negative' THEN 1 ELSE 0 END) as negative,
    (positive - negative) * 1.0 / COUNT(*) as sentiment_score
FROM sentiment_ready
CROSS JOIN UNNEST(sentences) AS t(sentence)
WHERE quarter = '2023Q1'
GROUP BY theme_id, theme_name;
```

### Firm-Within-Theme Sentiment

```sql
SELECT
    theme_id,
    theme_name,
    firm_id,
    firm_name,
    permno,
    earnings_call_date,
    COUNT(*) as total_sentences,
    SUM(CASE WHEN sentiment = 'positive' THEN 1 ELSE 0 END) as positive,
    SUM(CASE WHEN sentiment = 'negative' THEN 1 ELSE 0 END) as negative,
    (positive - negative) * 1.0 / COUNT(*) as sentiment_score
FROM sentiment_ready
CROSS JOIN UNNEST(sentences) AS t(sentence)
WHERE quarter = '2023Q1'
GROUP BY theme_id, theme_name, firm_id, firm_name, permno, earnings_call_date;
```

### Event Study Input

```sql
SELECT DISTINCT
    permno,
    earnings_call_date as event_date,
    theme_id,
    theme_name
FROM sentiment_ready
WHERE quarter = '2023Q1'
ORDER BY permno, event_date;
```

## Comparison with Legacy JSON Format

### Legacy Format (from handoff_package)

```json
{
  "themes": [{
    "theme_id": "theme_001",
    "theme_name": "...",
    "firm_contributions": [{
      "firm_id": "374372246",
      "firm_name": "Apple Inc.",
      "permno": 14593,
      "earnings_call_date": "2023-01-28",
      "sentences": [{"text": "..."}]
    }]
  }]
}
```

### New Parquet Format

- **Flattened**: One row per **(theme, firm, topic)** contribution
- **Columnar**: Efficient for analytical queries
- **Partitioned**: By quarter for fast scans
- **Additional fields**: `gvkey`, `link_date`, `topic_id`, `topic_summary`

### Migration Path

The sentiment analysis module needs these changes:
1. Replace JSON loading with Parquet/Athena reading
2. Remove the nested `firm_contributions` iteration
3. Group by `theme_id` when needed (already flat)
4. Use `earnings_call_date` (date type) instead of string parsing

## Write Pattern (Post-Reduce Sentiment Builder)

```python
def write_sentiment_ready(
    theme_contributions: List[dict],
    sentences_by_topic: Dict[Tuple[str, int], List[dict]],
    s3_bucket: str,
    quarter: str,
):
    """Write denormalized sentiment-ready Parquet (post-reduce)."""
    rows = []

    for contrib in theme_contributions:
        firm_id = contrib["firm_id"]
        # topic_id format: "{firm_id}_{local_topic_id}"
        local_topic_id = int(contrib["topic_id"].split("_")[-1])
        topic_key = (firm_id, local_topic_id)
        topic_sentences = sentences_by_topic.get(topic_key, [])

        rows.append({
            "theme_id": contrib["theme_id"],
            "theme_name": contrib["theme_name"],
            "theme_description": contrib.get("theme_description"),
            "theme_keywords": contrib["theme_keywords"],

            "firm_id": firm_id,
            "firm_name": contrib["firm_name"],
            "permno": contrib["permno"],
            "gvkey": contrib["gvkey"],

            "earnings_call_date": contrib["earnings_call_date"],
            "link_date": contrib["link_date"],

            "topic_id": contrib["topic_id"],
            "topic_summary": contrib.get("topic_summary"),

            "sentences": topic_sentences,
            "n_sentences": len(topic_sentences),

            "quarter": quarter,
        })

    table = pa.Table.from_pylist(rows, schema=sentiment_ready_schema)

    pq.write_table(
        table,
        f"s3://{s3_bucket}/sentiment-ready/quarter={quarter}/themes_for_sentiment.parquet",
        compression="snappy",
    )
```

## Validation Criteria

- [ ] All rows have non-null `permno`
- [ ] `earnings_call_date` is within the quarter range
- [ ] `sentences` list is non-empty for all rows
- [ ] `n_sentences` matches `len(sentences)`
- [ ] No duplicate `(theme_id, firm_id, topic_id)` pairs within a quarter
- [ ] All `theme_id` values exist in `processed/themes/` table

## References

- `docs/adr/adr_007_storage_strategy.md` - Storage design rationale
- `docs/adr/adr_004_wrds_data_source.md` - PERMNO requirement and skip-unlinked decision
- `sentiment_analysis/handoff_package/` - Legacy format reference
