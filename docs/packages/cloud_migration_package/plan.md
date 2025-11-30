# Cloud Migration Plan: Financial Topic Modeling Pipeline

## Executive Summary

Build a production-ready AWS cloud architecture for financial topic modeling using a Map-Reduce pattern. This is a **clean reimplementation** informed by MVP learnings, NOT a port of MVP code.

**Timeline:** ~6-7 days intensive development
**Philosophy:** Code my mom could read, boring technology, over-document the "why"

**Critical Principle:** The Local_BERTopic_MVP is for **intent and data flow reference ONLY**. It is poorly structured, over-bloated code written under time pressure. We will reimplement simply and cleanly, not wrap or port MVP code. Our implementation must be simpler, more readable, and more modular than the MVP.

---

## Architecture Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Data Source    │     │   MAP PHASE     │     │  REDUCE PHASE   │
│  (CSV/S3)       │────▶│  AWS Batch      │────▶│  AWS Batch      │────▶ DynamoDB
│                 │     │  Per-Firm Topics│     │  Cross-Firm     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                              │                        │
                              ▼                        ▼
                         S3 (JSON)              Final Output
```

**AWS Services:**

- **Map Phase:** AWS Batch with spot instances (m5.xlarge recommended, 16GB - see sizing note)
- **Reduce Phase:** AWS Batch (m5.2xlarge recommended, 32GB - see sizing note)
- **Orchestration:** Step Functions with distributed map
- **Storage:** S3 (intermediate JSON), DynamoDB (final hierarchical output including sentences)
- **Infrastructure:** Terraform (modular for auditability)

**Instance Sizing Note:** t3.large (8GB) may be tight for BERTopic + sentence-transformers. A benchmarking step is required in Phase 2 to validate memory requirements before locking instance types. Start with m5.xlarge (16GB) for map and m5.2xlarge (32GB) for reduce, then right-size based on actual usage.

**Intentional Deviation from Original Spec:** The raw transcript mentioned SageMaker for reduce phase. After analysis, AWS Batch is approved because: (1) simpler - same job/queue pattern as map, fewer IAM roles; (2) cheaper - Batch on Spot for infrequent reduce runs; (3) sufficient - reduce is Python aggregation with BERTopic, no SageMaker-specific ML ergonomics needed.

---

## Explicit Halting Points (Per Mission Briefing)

| Phase | Gate                         | Criteria                                                                  |
| ----- | ---------------------------- | ------------------------------------------------------------------------- |
| 1     | HALT after abstraction layer | Interfaces defined, local CSV connector works, tests pass                 |
| 2     | HALT after map container     | Container builds, processes single firm locally, output schema validated  |
| 3     | HALT after reduce container  | Container builds, aggregates test data locally, DynamoDB schema validated |
| 4     | HALT after Terraform         | `terraform plan` succeeds, resources correctly defined                    |
| 5     | HALT after integration       | Single-firm cloud test passes, full pipeline completes                    |

**Do not proceed past a halting point without explicit user approval.**

---

## Directory Structure

```
Financial_Topic_Modeling/
├── cloud/                              # New cloud migration code
│   ├── README.md                       # Quick start guide
│   │
│   ├── src/                            # Core pipeline code
│   │   ├── __init__.py
│   │   ├── interfaces.py               # TopicModel + DataConnector ABCs
│   │   │
│   │   ├── connectors/                 # DataConnector implementations
│   │   │   ├── __init__.py
│   │   │   ├── local_csv.py            # Local CSV for testing
│   │   │   └── s3_connector.py         # S3 for cloud deployment
│   │   │
│   │   ├── topic_models/               # TopicModel implementations
│   │   │   ├── __init__.py
│   │   │   ├── bertopic_model.py       # BERTopic (primary)
│   │   │   ├── lda_model.py            # LDA stub (NotImplemented)
│   │   │   └── neural_model.py         # Neural stub (NotImplemented)
│   │   │
│   │   ├── firm_processor.py           # Map phase logic
│   │   ├── theme_aggregator.py         # Reduce phase logic
│   │   ├── s3_utils.py                 # S3 read/write helpers
│   │   └── dynamodb_utils.py           # DynamoDB write helpers
│   │
│   ├── containers/
│   │   ├── map/
│   │   │   ├── Dockerfile
│   │   │   ├── requirements.txt
│   │   │   └── entrypoint.py           # CLI for firm processing
│   │   └── reduce/
│   │       ├── Dockerfile
│   │       ├── requirements.txt
│   │       └── entrypoint.py           # CLI for theme aggregation
│   │
│   ├── terraform/
│   │   ├── main.tf                     # All resources (single file)
│   │   ├── variables.tf
│   │   └── outputs.tf
│   │
│   └── scripts/
│       ├── local_test_map.sh
│       ├── local_test_reduce.sh
│       └── deploy.sh
│
├── tests/                              # Test suite
│   ├── conftest.py                     # Pytest fixtures, moto setup
│   ├── unit/
│   │   ├── test_bertopic_model.py
│   │   ├── test_firm_processor.py
│   │   └── test_theme_aggregator.py
│   ├── integration/
│   │   ├── test_map_pipeline.py
│   │   └── test_reduce_pipeline.py
│   ├── baseline/
│   │   ├── test_topic_quality.py       # Compare to MVP
│   │   └── test_theme_quality.py
│   └── fixtures/
│       ├── sample_transcripts.csv
│       └── baseline/                   # MVP outputs for comparison
│
└── Local_BERTopic_MVP/                 # Existing (reference only)
```

---

## Phase 1: Abstraction Layer (Day 1)

### 1.1 Explicit Data Schemas (CRITICAL - Define Before Coding)

**TranscriptData** - Output of DataConnector:

```python
@dataclass
class TranscriptSentence:
    """Single sentence from an earnings call."""
    sentence_id: str              # Unique identifier
    text: str                     # The actual sentence
    speaker_type: Optional[str]   # CEO, CFO, Analyst, etc.
    position: int                 # Order in transcript

@dataclass
class FirmTranscriptData:
    """All transcript sentences for a single firm."""
    firm_id: str
    firm_name: str
    sentences: List[TranscriptSentence]
    metadata: Dict[str, Any]      # date_range, transcript_count, etc.

@dataclass
class TranscriptData:
    """Complete dataset from DataConnector."""
    firms: Dict[str, FirmTranscriptData]  # firm_id -> FirmTranscriptData

    def get_firm_sentences(self, firm_id: str) -> List[str]:
        """Get sentence texts for a firm."""
        return [s.text for s in self.firms[firm_id].sentences]

    def get_all_firm_ids(self) -> List[str]:
        return list(self.firms.keys())
```

**TopicModelResult** - Output of TopicModel (model-agnostic, NO hardcoded dimensions):

```python
@dataclass
class TopicModelResult:
    """
    Standardized output for ALL topic model implementations.

    Design: Documents in -> Topic assignments + representations out.
    NO centroids or hardcoded embedding dimensions (those were MVP artifacts).
    """
    # Required: Every implementation must provide these
    topic_assignments: np.ndarray          # (n_docs,) - topic ID per document (-1 = outlier)
    n_topics: int                          # Number of topics discovered (excluding outliers)
    topic_representations: Dict[int, str]  # topic_id -> human-readable representation
    topic_keywords: Dict[int, List[str]]   # topic_id -> top keywords

    # Optional: Model-specific, may not be available for all implementations
    probabilities: Optional[np.ndarray] = None     # (n_docs, n_topics) - LDA/neural provide this
    topic_sizes: Optional[Dict[int, int]] = None   # topic_id -> document count

    # Metadata for debugging/audit
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**FirmTopicOutput** - Map phase output (JSON to S3):

```json
{
  "firm_id": "AAPL",
  "firm_name": "Apple Inc.",
  "n_topics": 8,
  "topics": [
    {
      "topic_id": 0,
      "representation": "AI and Machine Learning Investment",
      "keywords": ["ai", "machine learning", "neural", "model", "training"],
      "size": 45,
      "sentence_ids": ["s001", "s015", "s023", ...]
    },
    ...
  ],
  "outlier_sentence_ids": ["s003", "s007", ...],
  "metadata": {
    "processing_timestamp": "2024-11-30T10:00:00Z",
    "model_config": {"umap_n_neighbors": 15, "hdbscan_min_cluster_size": 6},
    "n_sentences_processed": 500
  }
}
```

**ThemeOutput** - Reduce phase output (to DynamoDB):

```json
{
  "theme_id": "theme_001",
  "name": "Supply Chain Resilience",
  "keywords": ["supply", "chain", "logistics", "disruption"],
  "n_firms": 5,
  "n_topics": 7,
  "topics": [
    {"firm_id": "AAPL", "topic_id": 2, "representation": "Supply chain diversification"},
    {"firm_id": "MSFT", "topic_id": 5, "representation": "Logistics optimization"},
    ...
  ],
  "metadata": {"avg_similarity": 0.82, "discovery_method": "dual_bertopic"}
}
```

### 1.2 Operational Rules (LOCK BEFORE CODING)

**sentence_id Generation:**

- Format: `{firm_id}_{transcript_id}_{position:04d}` (e.g., `AAPL_T12345_0023`)
- Properties: unique, stable (same input → same ID), order-preserving within transcript
- Position is 0-indexed, zero-padded to 4 digits

**firm_id Normalization:**

- Use company ID from source data (e.g., CSV `companyid` column), NOT company name
- Store `firm_name` separately for display purposes
- Case-sensitive, no whitespace trimming on IDs

**Date Filtering Semantics:**

- Both `start_date` and `end_date` are INCLUSIVE
- Dates are in UTC, format: `YYYY-MM-DD`
- Filter on transcript date (`mostimportantdateutc` column in CSV)

**Outlier Handling (-1 topic):**

- Map phase: Outliers tracked in `outlier_sentence_ids` array
- Reduce phase: Outliers SKIPPED - only process topics with topic_id >= 0
- Reason: Outliers are sentences that didn't cluster; no representation to re-embed

**theme_id Generation:**

- Format: `theme_{run_id}_{index:03d}` (e.g., `theme_20241130_001`)
- `run_id`: YYYYMMDD timestamp of reduce phase execution
- `index`: 0-indexed, zero-padded to 3 digits, sorted by theme size (largest first)

**topic_id Within Firm:**

- 0-indexed integers assigned by BERTopic
- -1 reserved for outliers
- Not globally unique - must be combined with firm_id

### 1.3 TopicModel Interface

```python
# cloud/src/interfaces.py
from abc import ABC, abstractmethod

class TopicModel(ABC):
    """
    Abstract interface for topic models.

    Contract: Takes documents, returns topic assignments and representations.
    The reduce phase will RE-EMBED topic representations - we do NOT carry
    embeddings forward (that was an MVP artifact from the old similarity-based approach).
    """

    @abstractmethod
    def fit_transform(self, documents: List[str]) -> TopicModelResult:
        """
        Fit the topic model and transform documents to topics.

        Args:
            documents: List of document texts

        Returns:
            TopicModelResult with assignments, representations, and keywords
        """
        pass
```

### 1.3 DataConnector Interface

```python
class DataConnector(ABC):
    """
    Abstract interface for transcript data sources.
    Enables swapping: CSV (testing) -> S3 (cloud) -> WRDS (future).
    """

    @abstractmethod
    def fetch_transcripts(
        self,
        firms: List[str],
        start_date: str,
        end_date: str
    ) -> TranscriptData:
        """
        Fetch transcript sentences for specified firms and date range.

        Args:
            firms: List of company names (case-insensitive matching)
            start_date: YYYY-MM-DD format
            end_date: YYYY-MM-DD format (inclusive)

        Returns:
            TranscriptData with firms mapped to their sentences
        """
        pass

    @abstractmethod
    def get_available_firms(self) -> List[str]:
        """List all firms available in the data source."""
        pass

    def close(self) -> None:
        """Clean up resources. Default: no-op."""
        pass
```

### 1.4 Implementations (Reimplement Simply, Do NOT Port MVP Code)

**BERTopicModel** - Primary implementation:

```python
# cloud/src/topic_models/bertopic_model.py
class BERTopicModel(TopicModel):
    """
    BERTopic implementation.

    Reimplemented cleanly - MVP code is for intent reference only.
    """

    def __init__(self, config: dict):
        self.config = config
        # Initialize embedding model, UMAP, HDBSCAN from config
        # Config provides: embedding_model_name, umap_*, hdbscan_*

    def fit_transform(self, documents: List[str]) -> TopicModelResult:
        # 1. Embed documents (embedding model from config)
        # 2. Reduce dimensions (UMAP)
        # 3. Cluster (HDBSCAN)
        # 4. Extract representations and keywords
        # 5. Return TopicModelResult (NO centroids, NO hardcoded dims)
```

**LocalCSVConnector** - For local testing:

```python
# cloud/src/connectors/local_csv.py
class LocalCSVConnector(DataConnector):
    """
    Local CSV connector for testing and development.

    Reimplemented cleanly - MVP local_csv_connector.py is for reference only.
    """
```

**S3TranscriptConnector** - For cloud deployment:

```python
# cloud/src/connectors/s3_connector.py
class S3TranscriptConnector(DataConnector):
    """S3 connector for cloud deployment."""
```

### 1.5 Topic Model Stubs (Minimal Placeholders)

```python
# cloud/src/topic_models/lda_model.py
class LDATopicModel(TopicModel):
    """LDA placeholder - demonstrates interface extensibility."""
    def fit_transform(self, documents: List[str]) -> TopicModelResult:
        raise NotImplementedError("LDA not yet implemented")

# cloud/src/topic_models/neural_model.py
class NeuralTopicModel(TopicModel):
    """Neural topic model placeholder - demonstrates interface extensibility."""
    def fit_transform(self, documents: List[str]) -> TopicModelResult:
        raise NotImplementedError("Neural topic model not yet implemented")
```

**MVP Files for Intent Reference (NOT code to copy):**

- `Local_BERTopic_MVP/src/config/config.yaml` - UMAP/HDBSCAN parameter values
- `Local_BERTopic_MVP/src/data_ingestion/local_csv_connector.py` - Understand CSV structure
- `Local_BERTopic_MVP/src/topic_modeling/firm_topic_analyzer.py` - Understand BERTopic flow

**== HALT: Await approval before proceeding to Phase 2 ==**

---

## Phase 2: Map Phase Container (Days 2-3)

### 2.1 Firm Processor

```python
# cloud/src/firm_processor.py
class FirmProcessor:
    """Process single firm's transcripts into topics."""

    def __init__(self, topic_model: TopicModel, config: dict):
        self.model = topic_model  # Dependency injection
        self.config = config

    def process(self, firm_data: FirmTranscriptData) -> dict:
        """
        Run topic modeling and return FirmTopicOutput (see schema in Phase 1).

        Args:
            firm_data: FirmTranscriptData from DataConnector

        Returns:
            Dict matching FirmTopicOutput schema (JSON-serializable)
        """
        sentences = [s.text for s in firm_data.sentences]
        sentence_ids = [s.sentence_id for s in firm_data.sentences]

        result = self.model.fit_transform(sentences)

        return self._to_output(firm_data, result, sentence_ids)

    def _to_output(self, firm_data, result: TopicModelResult, sentence_ids: List[str]) -> dict:
        """Convert to FirmTopicOutput schema defined in Phase 1."""
        topics = []
        for topic_id in range(result.n_topics):
            mask = result.topic_assignments == topic_id
            topics.append({
                "topic_id": topic_id,
                "representation": result.topic_representations[topic_id],
                "keywords": result.topic_keywords[topic_id],
                "size": int(mask.sum()),
                "sentence_ids": [sid for sid, m in zip(sentence_ids, mask) if m]
            })

        outlier_mask = result.topic_assignments == -1
        return {
            "firm_id": firm_data.firm_id,
            "firm_name": firm_data.firm_name,
            "n_topics": result.n_topics,
            "topics": topics,
            "outlier_sentence_ids": [sid for sid, m in zip(sentence_ids, outlier_mask) if m],
            "metadata": {
                "processing_timestamp": datetime.utcnow().isoformat() + "Z",
                "model_config": self.config,
                "n_sentences_processed": len(sentences)
            }
        }
```

### 2.2 Map Container Entrypoint (Concrete Implementation)

```python
# cloud/containers/map/entrypoint.py
"""
Map phase: Process single firm into topics and write sentences to DynamoDB.

Environment Variables:
  Required:
    FIRM_ID           - Firm ID to process (from CSV companyid column)

  Cloud Mode (default):
    S3_INPUT_BUCKET   - Bucket containing transcript CSV
    S3_INPUT_KEY      - Key to transcript CSV file
    S3_OUTPUT_BUCKET  - Bucket for FirmTopicOutput JSON
    S3_OUTPUT_PREFIX  - Key prefix for output (default: "firm-topics/")
    DYNAMODB_TABLE    - Table name for sentence writes

  Local Mode (LOCAL_MODE=true):
    LOCAL_INPUT       - Path to local CSV file
    LOCAL_OUTPUT      - Directory for output JSON
    (DynamoDB writes skipped in local mode)

  Optional:
    CONFIG_PATH       - Path to config YAML (default: /app/config/default.yaml)
    LOG_LEVEL         - Logging level (default: INFO)
"""
import os
import sys
import json
import logging
import yaml
from datetime import datetime

# Setup logging early
logging.basicConfig(
    level=os.environ.get('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('map-entrypoint')

def load_config() -> dict:
    """Load configuration from YAML file."""
    config_path = os.environ.get('CONFIG_PATH', '/app/config/default.yaml')
    if os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f)
    # Fallback to hardcoded defaults
    return {
        'embedding_model': 'all-mpnet-base-v2',
        'umap': {'n_neighbors': 15, 'n_components': 10, 'min_dist': 0.0, 'metric': 'cosine'},
        'hdbscan': {'min_cluster_size': 6, 'min_samples': 2},
        'validation': {'min_firms': 2, 'max_firm_dominance': 0.4}
    }

def main():
    try:
        # 1. Parse environment
        firm_id = os.environ['FIRM_ID']
        local_mode = os.environ.get('LOCAL_MODE', 'false').lower() == 'true'
        logger.info(f"Processing firm: {firm_id}, local_mode={local_mode}")

        # 2. Load config
        config = load_config()

        # 3. Initialize topic model
        from cloud.src.topic_models.bertopic_model import BERTopicModel
        model = BERTopicModel(config)

        # 4. Initialize processor
        from cloud.src.firm_processor import FirmProcessor
        processor = FirmProcessor(model, config)

        # 5. Load data
        if local_mode:
            from cloud.src.connectors.local_csv import LocalCSVConnector
            connector = LocalCSVConnector(os.environ['LOCAL_INPUT'])
        else:
            from cloud.src.connectors.s3_connector import S3TranscriptConnector
            connector = S3TranscriptConnector(
                bucket=os.environ['S3_INPUT_BUCKET'],
                key=os.environ['S3_INPUT_KEY']
            )

        transcript_data = connector.fetch_transcripts(
            firms=[firm_id],
            start_date='1900-01-01',  # No date filter for single firm processing
            end_date='2100-01-01'
        )

        if firm_id not in transcript_data.firms:
            logger.error(f"Firm {firm_id} not found in data source")
            sys.exit(1)

        firm_data = transcript_data.firms[firm_id]
        logger.info(f"Loaded {len(firm_data.sentences)} sentences for {firm_id}")

        # 6. Process
        result = processor.process(firm_data)
        logger.info(f"Discovered {result['n_topics']} topics")

        # 7. Save output
        if local_mode:
            output_dir = os.environ['LOCAL_OUTPUT']
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{firm_id}_topics.json")
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            logger.info(f"Saved to {output_path}")
        else:
            # Write JSON to S3
            from cloud.src.s3_utils import upload_json
            s3_key = f"{os.environ.get('S3_OUTPUT_PREFIX', 'firm-topics/')}{firm_id}_topics.json"
            upload_json(os.environ['S3_OUTPUT_BUCKET'], s3_key, result)
            logger.info(f"Uploaded to s3://{os.environ['S3_OUTPUT_BUCKET']}/{s3_key}")

            # Write sentences to DynamoDB
            from cloud.src.dynamodb_utils import MapPhaseDynamoDBWriter
            dynamo_writer = MapPhaseDynamoDBWriter(os.environ['DYNAMODB_TABLE'])
            dynamo_writer.write_firm_sentences(result, firm_data.sentences)
            logger.info(f"Wrote sentences to DynamoDB")

        logger.info("Map phase completed successfully")

    except KeyError as e:
        logger.error(f"Missing required environment variable: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Map phase failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
```

### 2.3 Benchmarking Step (Required Before Locking Instance Sizes)

After local testing, run a single-firm benchmark on AWS:

```bash
# Test with different instance types
for INSTANCE in t3.large m5.large m5.xlarge; do
  aws batch submit-job \
    --job-name benchmark-${INSTANCE} \
    --job-definition map-benchmark \
    --container-overrides "resourceRequirements=[{type=VCPU,value=2},{type=MEMORY,value=8192}]" \
    --parameters firm_id=AAPL
done
```

Measure:

- Peak memory usage (CloudWatch Metrics)
- Processing time
- Any OOM errors

Lock instance sizes after benchmarking validates memory requirements.

### 2.3 Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY cloud/src/ ./cloud/src/
COPY cloud/containers/map/entrypoint.py .

CMD ["python", "entrypoint.py"]
```

**== HALT: Await approval before proceeding to Phase 3 ==**

---

## Phase 3: Reduce Phase Container (Days 3-4)

### 3.1 Theme Aggregator (Dual-BERTopic Approach)

**Key Insight:** The reduce phase treats topic representations as NEW DOCUMENTS and re-embeds them. We do NOT carry embeddings forward from the map phase - that was an MVP artifact from an old similarity-based approach.

```python
# cloud/src/theme_aggregator.py
class ThemeAggregator:
    """
    Aggregate firm topics into cross-firm themes using Dual-BERTopic.

    Process:
    1. Collect topic representations (strings) from all firms
    2. Embed these representations as new documents
    3. Run BERTopic again to cluster topics into themes
    4. Validate themes (diversity, dominance filters)
    5. Return ThemeOutput dicts (see schema in Phase 1)
    """

    def __init__(self, topic_model: TopicModel, config: dict):
        self.model = topic_model  # Same interface, reused for theme clustering
        self.config = config
        self.min_firms = config.get('min_firms', 2)
        self.max_dominance = config.get('max_firm_dominance', 0.4)

    def aggregate(self, firm_results: List[dict]) -> List[dict]:
        """
        Args:
            firm_results: List of FirmTopicOutput dicts from S3

        Returns:
            List of ThemeOutput dicts matching schema in Phase 1
        """
        # 1. Extract topic representations as "documents" for theme modeling
        topic_docs = []  # List of representation strings
        topic_metadata = []  # Track which firm/topic each doc came from
        for firm_result in firm_results:
            for topic in firm_result['topics']:
                topic_docs.append(topic['representation'])
                topic_metadata.append({
                    'firm_id': firm_result['firm_id'],
                    'topic_id': topic['topic_id'],
                    'representation': topic['representation'],
                    'size': topic['size']
                })

        # 2. Run BERTopic on topic representations (RE-EMBEDS them)
        theme_result = self.model.fit_transform(topic_docs)

        # 3. Group topics by theme assignment
        raw_themes = self._group_into_themes(theme_result, topic_metadata)

        # 4. Apply validation filters
        validated_themes = self._validate_themes(raw_themes)

        return validated_themes

    def _validate_themes(self, themes: List[dict]) -> List[dict]:
        """Apply two-tier validation: min_firms and max_dominance."""
        validated = []
        for theme in themes:
            firms = set(t['firm_id'] for t in theme['topics'])
            # Tier 1: Minimum firms filter
            if len(firms) < self.min_firms:
                continue
            # Tier 2: Maximum dominance filter
            firm_counts = Counter(t['firm_id'] for t in theme['topics'])
            max_share = max(firm_counts.values()) / len(theme['topics'])
            if max_share > self.max_dominance:
                continue
            validated.append(theme)
        return validated
```

### 3.2 DynamoDB Schema (Single-Table Design with Sentences)

**Design Decision:** Single table with composite keys enables hierarchical queries (theme → topics → sentences) while keeping infrastructure simple. Sentences are written by the MAP phase (no backfill needed in reduce).

**Table Schema:**

```
Table: financial-topics-{env}
- PK (Partition Key): String
- SK (Sort Key): String
- GSI1PK, GSI1SK: For reverse lookups (firm → themes, firm → sentences)
```

**Item Types:**

| Item Type      | PK                | SK                        | Attributes                                      | Written By |
| -------------- | ----------------- | ------------------------- | ----------------------------------------------- | ---------- |
| Theme          | `THEME#theme_001` | `METADATA`                | name, keywords[], n_firms, n_topics, metadata{} | Reduce     |
| Topic in Theme | `THEME#theme_001` | `TOPIC#AAPL#2`            | firm_id, topic_id, representation, size         | Reduce     |
| Sentence       | `TOPIC#AAPL#2`    | `SENTENCE#AAPL_T123_0001` | text, position, speaker_type, firm_id, topic_id | **Map**    |

**GSI1 (Firm Index):**
| Item Type | GSI1PK | GSI1SK |
|-----------|--------|--------|
| Topic in Theme | `FIRM#AAPL` | `THEME#theme_001` |
| Sentence | `FIRM#AAPL` | `SENTENCE#AAPL_T123_0001` |

**Query Patterns:**

- Get theme + all topics: `PK = "THEME#theme_001"` → returns metadata + all topics
- Get all sentences for a topic: `PK = "TOPIC#AAPL#2"` → returns all sentences
- Get all themes for a firm: `GSI1PK = "FIRM#AAPL", SK begins_with "THEME#"` → returns theme associations
- Get all sentences for a firm: `GSI1PK = "FIRM#AAPL", SK begins_with "SENTENCE#"` → returns all sentences

**Query Flow (Theme → Sentences):**

1. Query `PK = "THEME#theme_001"` → get theme metadata + list of topics
2. For each topic, query `PK = "TOPIC#{firm_id}#{topic_id}"` → get sentences
3. (Two-hop query, but keeps reduce phase simple)

### 3.3 Map Phase DynamoDB Writer (Sentences)

```python
# cloud/src/dynamodb_utils.py
class MapPhaseDynamoDBWriter:
    """Write sentences to DynamoDB during map phase."""

    def __init__(self, table_name: str):
        self.dynamodb = boto3.resource('dynamodb')
        self.table = self.dynamodb.Table(table_name)

    def write_firm_sentences(self, firm_result: dict, sentences: List[TranscriptSentence]):
        """Write sentence records for all topics in a firm."""
        with self.table.batch_writer() as batch:
            for topic in firm_result['topics']:
                firm_id = firm_result['firm_id']
                topic_id = topic['topic_id']

                for sentence_id in topic['sentence_ids']:
                    # Find the sentence by ID
                    sentence = next(s for s in sentences if s.sentence_id == sentence_id)
                    batch.put_item(Item={
                        'PK': f"TOPIC#{firm_id}#{topic_id}",
                        'SK': f"SENTENCE#{sentence_id}",
                        'text': sentence.text,
                        'position': sentence.position,
                        'speaker_type': sentence.speaker_type or 'UNKNOWN',
                        'firm_id': firm_id,
                        'topic_id': topic_id,
                        'GSI1PK': f"FIRM#{firm_id}",
                        'GSI1SK': f"SENTENCE#{sentence_id}"
                    })

class ReducePhaseDynamoDBWriter:
    """Write themes to DynamoDB during reduce phase."""

    def __init__(self, table_name: str):
        self.dynamodb = boto3.resource('dynamodb')
        self.table = self.dynamodb.Table(table_name)

    def write_themes(self, themes: List[dict]):
        """Write theme and topic records (sentences already written by map)."""
        with self.table.batch_writer() as batch:
            for theme in themes:
                theme_id = theme['theme_id']

                # 1. Theme metadata record
                batch.put_item(Item={
                    'PK': f"THEME#{theme_id}",
                    'SK': 'METADATA',
                    'name': theme['name'],
                    'keywords': theme['keywords'][:20],
                    'n_firms': theme['n_firms'],
                    'n_topics': theme['n_topics'],
                    'metadata': theme.get('metadata', {})
                })

                # 2. Topic records (link theme to topics)
                for topic in theme['topics']:
                    batch.put_item(Item={
                        'PK': f"THEME#{theme_id}",
                        'SK': f"TOPIC#{topic['firm_id']}#{topic['topic_id']}",
                        'firm_id': topic['firm_id'],
                        'topic_id': topic['topic_id'],
                        'representation': topic['representation'],
                        'GSI1PK': f"FIRM#{topic['firm_id']}",
                        'GSI1SK': f"THEME#{theme_id}"
                    })
```

### 3.4 Reduce Container Entrypoint (With Error Handling)

```python
# cloud/containers/reduce/entrypoint.py
"""
Reduce phase: Aggregate firm topics into cross-firm themes.

Environment Variables:
  Cloud Mode (default):
    S3_INPUT_BUCKET   - Bucket containing firm topic JSONs
    S3_INPUT_PREFIX   - Key prefix for firm results (default: "firm-topics/")
    DYNAMODB_TABLE    - Table name for theme writes

  Local Mode (LOCAL_MODE=true):
    LOCAL_INPUT       - Directory containing firm topic JSONs
    LOCAL_OUTPUT      - Path for output themes JSON

  Optional:
    CONFIG_PATH       - Path to config YAML
    LOG_LEVEL         - Logging level (default: INFO)
    MIN_FIRMS         - Minimum firms required to proceed (default: 1)
"""
import os
import sys
import json
import logging
from datetime import datetime
from typing import List, Dict, Tuple

logging.basicConfig(
    level=os.environ.get('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('reduce-entrypoint')

def load_firm_results_from_s3(bucket: str, prefix: str) -> Tuple[List[dict], List[str]]:
    """
    Load all firm topic results from S3.

    Returns:
        (loaded_results, skipped_files) - Successfully loaded results and list of skipped file keys
    """
    import boto3
    s3 = boto3.client('s3')

    loaded = []
    skipped = []

    # List all JSON files
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if not key.endswith('_topics.json'):
                continue

            try:
                response = s3.get_object(Bucket=bucket, Key=key)
                data = json.loads(response['Body'].read().decode('utf-8'))

                # Validate required fields
                if 'firm_id' not in data or 'topics' not in data:
                    logger.warning(f"Skipping {key}: missing required fields")
                    skipped.append(key)
                    continue

                loaded.append(data)
                logger.debug(f"Loaded {key}: {len(data['topics'])} topics")

            except json.JSONDecodeError as e:
                logger.warning(f"Skipping {key}: invalid JSON - {e}")
                skipped.append(key)
            except Exception as e:
                logger.warning(f"Skipping {key}: {e}")
                skipped.append(key)

    return loaded, skipped

def load_firm_results_local(input_dir: str) -> Tuple[List[dict], List[str]]:
    """Load all firm topic results from local directory."""
    from pathlib import Path

    loaded = []
    skipped = []

    for json_file in Path(input_dir).glob('*_topics.json'):
        try:
            with open(json_file) as f:
                data = json.load(f)

            if 'firm_id' not in data or 'topics' not in data:
                logger.warning(f"Skipping {json_file}: missing required fields")
                skipped.append(str(json_file))
                continue

            loaded.append(data)
        except Exception as e:
            logger.warning(f"Skipping {json_file}: {e}")
            skipped.append(str(json_file))

    return loaded, skipped

def main():
    try:
        local_mode = os.environ.get('LOCAL_MODE', 'false').lower() == 'true'
        min_firms = int(os.environ.get('MIN_FIRMS', '1'))

        # 1. Load firm results
        logger.info("Loading firm topic results...")
        if local_mode:
            firm_results, skipped = load_firm_results_local(os.environ['LOCAL_INPUT'])
        else:
            firm_results, skipped = load_firm_results_from_s3(
                os.environ['S3_INPUT_BUCKET'],
                os.environ.get('S3_INPUT_PREFIX', 'firm-topics/')
            )

        logger.info(f"Loaded {len(firm_results)} firms, skipped {len(skipped)} files")
        if skipped:
            logger.warning(f"Skipped files: {skipped[:10]}{'...' if len(skipped) > 10 else ''}")

        # 2. Check minimum firms requirement
        if len(firm_results) < min_firms:
            logger.error(f"Insufficient firms: {len(firm_results)} < {min_firms}")
            sys.exit(1)

        # 3. Filter out firms with only outliers (no valid topics)
        valid_results = [r for r in firm_results if r['n_topics'] > 0]
        logger.info(f"{len(valid_results)} firms have valid topics (n_topics > 0)")

        if len(valid_results) < min_firms:
            logger.error(f"Insufficient valid firms after filtering: {len(valid_results)}")
            sys.exit(1)

        # 4. Initialize topic model and aggregator
        from cloud.src.topic_models.bertopic_model import BERTopicModel
        from cloud.src.theme_aggregator import ThemeAggregator

        config = load_config()
        model = BERTopicModel(config)
        aggregator = ThemeAggregator(model, config)

        # 5. Aggregate themes
        logger.info("Aggregating topics into themes...")
        themes = aggregator.aggregate(valid_results)
        logger.info(f"Discovered {len(themes)} themes")

        # 6. Generate theme IDs
        run_id = datetime.utcnow().strftime('%Y%m%d')
        for i, theme in enumerate(sorted(themes, key=lambda t: -t['n_topics'])):
            theme['theme_id'] = f"theme_{run_id}_{i:03d}"

        # 7. Save output
        if local_mode:
            output_path = os.environ['LOCAL_OUTPUT']
            with open(output_path, 'w') as f:
                json.dump(themes, f, indent=2, default=str)
            logger.info(f"Saved {len(themes)} themes to {output_path}")
        else:
            from cloud.src.dynamodb_utils import ReducePhaseDynamoDBWriter
            writer = ReducePhaseDynamoDBWriter(os.environ['DYNAMODB_TABLE'])
            writer.write_themes(themes)
            logger.info(f"Wrote {len(themes)} themes to DynamoDB")

        # 8. Log summary
        logger.info("=== Reduce Phase Summary ===")
        logger.info(f"  Firms loaded: {len(firm_results)}")
        logger.info(f"  Firms with topics: {len(valid_results)}")
        logger.info(f"  Files skipped: {len(skipped)}")
        logger.info(f"  Themes discovered: {len(themes)}")
        logger.info("Reduce phase completed successfully")

    except KeyError as e:
        logger.error(f"Missing required environment variable: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Reduce phase failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
```

**MVP Files for Intent Reference (NOT code to copy):**

- `Local_BERTopic_MVP/src/theme_identification/cross_firm_analyzer.py` - Understand Dual-BERTopic flow
- `Local_BERTopic_MVP/src/theme_identification/theme_processor.py` - Understand validation logic

**== HALT: Await approval before proceeding to Phase 4 ==**

---

## Phase 4: Terraform Infrastructure (Day 4-5)

### 4.1 Modular Structure (Per Codex Recommendation)

Modest modularity for auditability - each module is self-contained and documented:

```
cloud/terraform/
├── main.tf                    # Root module, composes all
├── variables.tf               # Input variables
├── outputs.tf                 # Output values
├── modules/
│   ├── storage/               # S3 + DynamoDB
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── outputs.tf
│   ├── compute/               # Batch environment + job definitions
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── outputs.tf
│   └── orchestration/         # Step Functions
│       ├── main.tf
│       ├── variables.tf
│       └── outputs.tf
└── environments/
    └── dev.tfvars
```

### 4.2 Storage Module (Complete DynamoDB Schema)

```hcl
# cloud/terraform/modules/storage/main.tf

# S3 Bucket - Intermediate Results
resource "aws_s3_bucket" "intermediate" {
  bucket = "${var.project_name}-intermediate-${var.environment}"

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "intermediate" {
  bucket = aws_s3_bucket.intermediate.id

  rule {
    id     = "expire-old-results"
    status = "Enabled"
    expiration {
      days = 30  # Auto-cleanup after 30 days
    }
  }
}

# DynamoDB - Single Table Design with GSI
resource "aws_dynamodb_table" "topics" {
  name         = "${var.project_name}-topics-${var.environment}"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "PK"
  range_key    = "SK"

  # Primary key attributes
  attribute {
    name = "PK"
    type = "S"
  }

  attribute {
    name = "SK"
    type = "S"
  }

  # GSI1 attributes (for firm lookups)
  attribute {
    name = "GSI1PK"
    type = "S"
  }

  attribute {
    name = "GSI1SK"
    type = "S"
  }

  # Global Secondary Index for firm-based queries
  global_secondary_index {
    name            = "GSI1"
    hash_key        = "GSI1PK"
    range_key       = "GSI1SK"
    projection_type = "ALL"
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

# ECR Repositories
resource "aws_ecr_repository" "map" {
  name = "${var.project_name}-map"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = false
  }
}

resource "aws_ecr_repository" "reduce" {
  name = "${var.project_name}-reduce"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = false
  }
}
```

### 4.3 Compute Module

```hcl
# cloud/terraform/modules/compute/main.tf

# AWS Batch - Compute Environment (Spot instances)
resource "aws_batch_compute_environment" "main" {
  compute_environment_name = "${var.project_name}-compute-${var.environment}"
  type                     = "MANAGED"
  state                    = "ENABLED"
  service_role             = var.batch_service_role_arn

  compute_resources {
    type                = "SPOT"
    allocation_strategy = "SPOT_CAPACITY_OPTIMIZED"
    bid_percentage      = 100

    min_vcpus     = 0
    max_vcpus     = 64
    desired_vcpus = 0

    # Start with larger instances; right-size after benchmarking
    instance_type = ["m5.xlarge", "m5.2xlarge", "m5.4xlarge"]

    subnets            = var.subnet_ids
    security_group_ids = [var.security_group_id]
    instance_role      = var.batch_instance_role_arn
  }
}

resource "aws_batch_job_queue" "main" {
  name                 = "${var.project_name}-queue-${var.environment}"
  state                = "ENABLED"
  priority             = 1
  compute_environments = [aws_batch_compute_environment.main.arn]
}

# Map Job Definition
resource "aws_batch_job_definition" "map" {
  name = "${var.project_name}-map-${var.environment}"
  type = "container"

  container_properties = jsonencode({
    image      = "${var.map_ecr_url}:latest"
    vcpus      = 4
    memory     = 15360  # 15GB (leave headroom on m5.xlarge)
    jobRoleArn = var.job_role_arn

    environment = [
      { name = "S3_INPUT_BUCKET", value = var.intermediate_bucket },
      { name = "S3_OUTPUT_BUCKET", value = var.intermediate_bucket },
      { name = "S3_OUTPUT_PREFIX", value = "firm-topics/" },
      { name = "DYNAMODB_TABLE", value = var.dynamodb_table_name },
      { name = "LOG_LEVEL", value = "INFO" }
    ]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = "/aws/batch/${var.project_name}-map"
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "map"
      }
    }
  })

  retry_strategy {
    attempts = 2
  }

  timeout {
    attempt_duration_seconds = 1800  # 30 minutes max per firm
  }
}

# Reduce Job Definition
resource "aws_batch_job_definition" "reduce" {
  name = "${var.project_name}-reduce-${var.environment}"
  type = "container"

  container_properties = jsonencode({
    image      = "${var.reduce_ecr_url}:latest"
    vcpus      = 8
    memory     = 30720  # 30GB (leave headroom on m5.2xlarge)
    jobRoleArn = var.job_role_arn

    environment = [
      { name = "S3_INPUT_BUCKET", value = var.intermediate_bucket },
      { name = "S3_INPUT_PREFIX", value = "firm-topics/" },
      { name = "DYNAMODB_TABLE", value = var.dynamodb_table_name },
      { name = "LOG_LEVEL", value = "INFO" }
    ]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = "/aws/batch/${var.project_name}-reduce"
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "reduce"
      }
    }
  })

  retry_strategy {
    attempts = 1  # Reduce is expensive; fail fast
  }

  timeout {
    attempt_duration_seconds = 7200  # 2 hours max
  }
}
```

### 4.4 Orchestration Module

```hcl
# cloud/terraform/modules/orchestration/main.tf

resource "aws_sfn_state_machine" "pipeline" {
  name     = "${var.project_name}-pipeline-${var.environment}"
  role_arn = var.step_functions_role_arn

  definition = jsonencode({
    Comment = "Financial Topic Modeling Pipeline"
    StartAt = "MapPhase"
    States = {
      MapPhase = {
        Type           = "Map"
        ItemsPath      = "$.firms"
        MaxConcurrency = 10
        ItemSelector = {
          "firm_id.$" = "$$.Map.Item.Value"
        }
        Iterator = {
          StartAt = "ProcessFirm"
          States = {
            ProcessFirm = {
              Type     = "Task"
              Resource = "arn:aws:states:::batch:submitJob.sync"
              Parameters = {
                JobName       = "map-firm"
                JobDefinition = var.map_job_definition_arn
                JobQueue      = var.job_queue_arn
                ContainerOverrides = {
                  Environment = [
                    { Name = "FIRM_ID", "Value.$" = "$.firm_id" }
                  ]
                }
              }
              End = true
              Catch = [{
                ErrorEquals = ["States.ALL"]
                Next        = "MapFailed"
              }]
            }
            MapFailed = {
              Type  = "Pass"
              End   = true
              Result = { "status" = "failed" }
            }
          }
        }
        ResultPath = "$.mapResults"
        Next       = "ReducePhase"
      }
      ReducePhase = {
        Type     = "Task"
        Resource = "arn:aws:states:::batch:submitJob.sync"
        Parameters = {
          JobName       = "reduce-themes"
          JobDefinition = var.reduce_job_definition_arn
          JobQueue      = var.job_queue_arn
        }
        End = true
      }
    }
  })
}
```

---

**== HALT: Await approval before proceeding to Phase 5 ==**

---

## Phase 5: Testing Strategy (Throughout)

**Coverage Target: 80%+ (REQUIRED)**

TDD for core logic. Integration tests for containers. Baseline validation is LOW PRIORITY (MVP quality wasn't great anyway).

### 5.1 Test Priority

| Priority | Test Type           | Focus                                        |
| -------- | ------------------- | -------------------------------------------- |
| P0       | Unit tests          | Interfaces, processors, aggregator, utils    |
| P0       | Integration tests   | Container entrypoints, S3/DynamoDB with moto |
| P1       | Schema validation   | Output matches defined schemas               |
| P2       | Baseline comparison | Optional - empirical review is sufficient    |

### 5.2 Test-Driven Development

Write tests BEFORE implementation for all core logic:

```python
# tests/unit/test_interfaces.py - Test abstract contracts
# tests/unit/test_bertopic_model.py - Test TopicModel implementation
# tests/unit/test_connectors.py - Test DataConnector implementations
# tests/unit/test_firm_processor.py - Test map phase logic
# tests/unit/test_theme_aggregator.py - Test reduce phase logic

class TestFirmProcessor:
    def test_process_returns_valid_schema(self, mock_topic_model, sample_firm_data):
        processor = FirmProcessor(mock_topic_model, config)
        result = processor.process(sample_firm_data)

        # Validate against FirmTopicOutput schema from Phase 1
        assert 'firm_id' in result
        assert 'topics' in result
        assert all('representation' in t for t in result['topics'])
        assert result['n_topics'] > 0

    def test_handles_empty_input(self, mock_topic_model):
        # Edge case: firm with no sentences

    def test_uses_injected_model(self, mock_topic_model, sample_firm_data):
        # Verify dependency injection works
        processor = FirmProcessor(mock_topic_model, config)
        processor.process(sample_firm_data)
        assert mock_topic_model.fit_transform.called
```

### 5.2 AWS Mocking with Moto

```python
# tests/conftest.py
import pytest
from moto import mock_aws

@pytest.fixture
def mock_s3():
    with mock_aws():
        s3 = boto3.client('s3', region_name='us-east-1')
        s3.create_bucket(Bucket='test-bucket')
        yield s3

@pytest.fixture
def mock_dynamodb():
    with mock_aws():
        dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
        table = dynamodb.create_table(
            TableName='themes',
            KeySchema=[
                {'AttributeName': 'PK', 'KeyType': 'HASH'},
                {'AttributeName': 'SK', 'KeyType': 'RANGE'}
            ],
            # ...
        )
        yield table
```

### 5.3 Baseline Validation

Compare cloud results to MVP:

```python
# tests/baseline/test_theme_quality.py
def test_theme_count_matches_baseline(cloud_themes, baseline_themes):
    """Theme count should be within 15% of MVP."""
    ratio = len(cloud_themes) / len(baseline_themes)
    assert 0.85 <= ratio <= 1.15

def test_firm_coverage_matches(cloud_themes, baseline_themes):
    """Same firms should appear in themes."""
    cloud_firms = set(f for t in cloud_themes for f in t['firms'])
    baseline_firms = set(f for t in baseline_themes for f in t['firms'])
    overlap = len(cloud_firms & baseline_firms) / len(baseline_firms)
    assert overlap >= 0.90
```

---

## Phase 6: Integration & Validation (Days 5-7)

### 6.1 Local Testing Sequence

```bash
# 1. Run unit tests
pytest tests/unit/ -v

# 2. Build containers
docker build -t map:local ./cloud/containers/map/
docker build -t reduce:local ./cloud/containers/reduce/

# 3. Test map container locally
docker run -e LOCAL_MODE=true -e FIRM_ID="Tesla, Inc." \
    -v $(pwd)/data:/data map:local

# 4. Test reduce container locally
docker run -e LOCAL_MODE=true -v $(pwd)/output:/data reduce:local

# 5. Run baseline validation
pytest tests/baseline/ -v
```

### 6.2 Cloud Deployment

```bash
# 1. Deploy infrastructure
cd cloud/terraform
terraform init && terraform plan
terraform apply

# 2. Push containers to ECR
aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_URL
docker push $ECR_URL/map:latest
docker push $ECR_URL/reduce:latest

# 3. Single firm test
aws batch submit-job --job-name test-single --job-definition map ...

# 4. Full pipeline via Step Functions
aws stepfunctions start-execution --state-machine-arn $SM_ARN --input '{"firms": [...]}'
```

### 6.3 Validation Criteria

| Metric                  | Target    | Tolerance |
| ----------------------- | --------- | --------- |
| Theme count             | Match MVP | ±15%      |
| Firm coverage           | 100%      | ≥90%      |
| Processing time/firm    | <5 min    | -         |
| Memory usage            | <8GB      | -         |
| AWS cost (full quarter) | <$50      | -         |

---

## Implementation Timeline

| Day | Tasks                                                                   | Deliverables                 |
| --- | ----------------------------------------------------------------------- | ---------------------------- |
| 1   | Create directory structure, TopicModel interface, BERTopicModel wrapper | Abstraction layer complete   |
| 2   | FirmProcessor, S3 utils, map entrypoint                                 | Map container builds locally |
| 3   | Tests for map phase, Docker testing                                     | Map container validated      |
| 4   | ThemeAggregator, DynamoDB utils, reduce entrypoint                      | Reduce container builds      |
| 5   | Terraform infrastructure, deploy to AWS                                 | Infrastructure live          |
| 6   | Single-firm cloud test, Step Functions                                  | End-to-end works             |
| 7   | Full pipeline, baseline validation, documentation                       | Mission complete             |

---

## What We Explicitly Skip (Simplicity)

| Feature                        | Rationale                                             | Add Later If Needed               |
| ------------------------------ | ----------------------------------------------------- | --------------------------------- |
| StorageBackend interface       | Direct boto3 calls suffice for S3/DynamoDB            | If storage becomes complex        |
| VPC/networking                 | Default VPC works for academic project                | For production security           |
| CI/CD pipeline                 | Manual deployment fine for 1 week                     | After first success               |
| WRDS connector                 | Funding-dependent; interface ready                    | When WRDS access granted          |
| Baseline quality tests         | MVP quality wasn't great; empirical review sufficient | If quantitative comparison needed |
| Centroids/embeddings in output | MVP artifact from old similarity approach             | Never - use representations       |

## What We Include (User Decisions)

| Feature                   | Rationale                                                         |
| ------------------------- | ----------------------------------------------------------------- |
| DataConnector interface   | Enables CSV/S3 today, WRDS tomorrow without refactors             |
| TopicModel interface      | Enables BERTopic/LDA/neural comparison (core research need)       |
| LDA/neural stubs          | Minimal NotImplemented placeholders for faculty review            |
| 80%+ test coverage        | TDD for core logic, integration for containers                    |
| AWS Batch for both phases | Simpler than mixing Batch + SageMaker                             |
| Modular Terraform         | Audit-friendly structure per Codex recommendation                 |
| Explicit schemas          | FirmTopicOutput, ThemeOutput, DynamoDB items locked before coding |

---

## MVP Reference (INTENT ONLY - Do NOT Copy Code)

**WARNING:** The Local_BERTopic_MVP is poorly structured, over-bloated code written under time pressure with an inferior model. It contains:

- Functions that aren't used in the actual data flow
- Bloated logic with dead code paths
- Hard-coded magic numbers (e.g., 768d centroids)
- Artifacts from abandoned approaches (similarity-based theme identification)

**Use MVP ONLY to understand:**

1. What the pipeline should DO (not how to code it)
2. Parameter values that worked (UMAP/HDBSCAN config)
3. CSV column structure for data ingestion

| MVP File                                      | What to Extract                                                          | What to IGNORE                             |
| --------------------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------ |
| `config/config.yaml`                          | UMAP/HDBSCAN params: n_neighbors=15, n_components=10, min_cluster_size=6 | Everything else                            |
| `topic_modeling/firm_topic_analyzer.py`       | Understand BERTopic flow                                                 | Centroid calculation, serialization format |
| `theme_identification/cross_firm_analyzer.py` | Understand Dual-BERTopic concept                                         | Complex similarity-based fallback code     |
| `theme_identification/theme_processor.py`     | Validation thresholds: min_firms=2, max_dominance=0.4                    | Over-complicated filtering logic           |
| `data_ingestion/local_csv_connector.py`       | CSV column names                                                         | Sentence tokenization approach             |

**Our implementation MUST be simpler, more readable, and more modular than the MVP.**

---

## User Decisions (Resolved)

- **Abstractions:** Include both TopicModel + DataConnector interfaces
- **Reduce Phase:** AWS Batch (simpler than SageMaker, same pattern as map)
- **Test Coverage:** 80%+ required - TDD for core, integration for containers
- **Model Stubs:** Include minimal LDA/neural stubs with NotImplementedError
- **Schemas:** Locked before coding (see Phase 1.1)
- **Centroids:** Do NOT use - MVP artifact from old approach
- **Baseline Tests:** Low priority - empirical review sufficient
