# Development Guide

Guide for contributing to the Financial Topic Modeling pipeline.

---

## Table of Contents

1. [Development Setup](#development-setup)
2. [Project Structure](#project-structure)
3. [Running Tests](#running-tests)
4. [Code Style](#code-style)
5. [Adding New Features](#adding-new-features)
6. [Database Migrations](#database-migrations)
7. [Debugging](#debugging)
8. [Common Tasks](#common-tasks)

---

## Development Setup

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- Git

### Local Environment

```bash
# 1. Clone repository
git clone <repository-url>
cd Financial_Topic_Modeling

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r cloud/requirements.txt
pip install -e .  # If setup.py exists

# 4. Install SpaCy model
python -m spacy download en_core_web_sm

# 5. Start local PostgreSQL
docker-compose up -d

# 6. Verify setup
python -c "from cloud.src.pipeline import UnifiedPipeline; print('OK')"
```

### IDE Setup (VS Code)

Recommended extensions:

- Python (Microsoft)
- Pylance
- Black Formatter
- GitLens

`.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true
}
```

---

## Project Structure

```
cloud/
├── config/                 # Configuration files
│   └── production.yaml     # Main pipeline config
├── src/
│   ├── pipeline/           # Pipeline orchestration
│   │   ├── unified_pipeline.py   # Main pipeline class
│   │   └── checkpoint.py         # Checkpoint/resume logic
│   ├── topic_models/       # Topic model implementations
│   │   ├── bertopic_model.py     # Primary implementation
│   │   ├── lda_model.py          # LDA stub (not implemented)
│   │   └── neural_model.py       # Neural stub (not implemented)
│   ├── database/           # Database layer
│   │   ├── models.py             # SQLAlchemy ORM models
│   │   └── repository.py         # Data access layer
│   ├── llm/                # LLM integration
│   │   └── xai_client.py         # xAI/Grok client
│   ├── connectors/         # Data source connectors
│   │   ├── local_csv.py          # CSV file connector
│   │   └── s3_connector.py       # S3 connector (stub)
│   ├── interfaces.py       # Abstract base classes
│   ├── models.py           # Data transfer objects
│   ├── firm_processor.py   # Per-firm processing logic
│   └── theme_aggregator.py # Cross-firm aggregation
├── terraform/              # AWS infrastructure
├── scripts/                # Deployment scripts
└── tests/                  # Test suite
```

### Key Abstractions

**TopicModel Interface** (`interfaces.py`):

```python
class TopicModel(ABC):
    @abstractmethod
    def fit_transform(self, documents, embeddings=None) -> TopicModelResult:
        """Fit model and return topic assignments."""
        pass
```

**DataConnector Interface** (`interfaces.py`):

```python
class DataConnector(ABC):
    @abstractmethod
    def fetch_transcripts(self, firm_ids, start_date, end_date) -> TranscriptData:
        pass

    @abstractmethod
    def get_available_firm_ids(self) -> List[str]:
        pass
```

---

## Running Tests

### Unit Tests

```bash
# Run all tests
pytest cloud/tests/ -v

# Run specific test file
pytest cloud/tests/test_bertopic_model.py -v

# Run with coverage
pytest cloud/tests/ --cov=cloud/src --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Integration Tests

```bash
# Requires running PostgreSQL
docker-compose up -d

# Run integration tests
pytest cloud/tests/integration/ -v
```

### Test Configuration

Tests use `testcontainers` for isolated PostgreSQL instances:

```python
from testcontainers.postgres import PostgresContainer

def test_with_database():
    with PostgresContainer("postgres:16") as postgres:
        engine = create_engine(postgres.get_connection_url())
        # Test code here
```

---

## Code Style

### Formatting

We use Black for code formatting:

```bash
# Format all files
black cloud/src/ scripts/

# Check without modifying
black --check cloud/src/
```

### Linting

```bash
# Run pylint
pylint cloud/src/

# Run flake8
flake8 cloud/src/
```

### Type Hints

Use type hints for all public functions:

```python
def process_firm(
    firm_id: str,
    data: FirmTranscriptData,
    embeddings: Optional[np.ndarray] = None,
) -> Tuple[FirmTopicOutput, np.ndarray]:
    """Process a single firm's transcripts."""
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def fit_transform(
    self,
    documents: List[str],
    embeddings: Optional[np.ndarray] = None,
) -> TopicModelResult:
    """
    Fit the topic model and transform documents.

    Args:
        documents: List of document texts
        embeddings: Pre-computed embeddings (optional)

    Returns:
        TopicModelResult with assignments and metadata

    Raises:
        ValueError: If documents list is empty
    """
```

---

## Adding New Features

### Adding a New Topic Model

1. **Create implementation** in `cloud/src/topic_models/`:

```python
# cloud/src/topic_models/my_model.py
from cloud.src.interfaces import TopicModel
from cloud.src.models import TopicModelResult

class MyTopicModel(TopicModel):
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def fit_transform(
        self,
        documents: List[str],
        embeddings: Optional[np.ndarray] = None,
    ) -> TopicModelResult:
        # Implementation here
        return TopicModelResult(
            topic_assignments=assignments,
            n_topics=n_topics,
            topic_representations=representations,
            topic_keywords=keywords,
            probabilities=probs,
        )
```

2. **Register in `__init__.py`**:

```python
# cloud/src/topic_models/__init__.py
from .my_model import MyTopicModel
```

3. **Add tests**:

```python
# cloud/tests/test_my_model.py
def test_my_model_fit_transform():
    model = MyTopicModel(config={})
    result = model.fit_transform(documents)
    assert result.n_topics > 0
```

### Adding a New Data Connector

1. **Implement interface** in `cloud/src/connectors/`:

```python
# cloud/src/connectors/my_connector.py
from cloud.src.interfaces import DataConnector

class MyConnector(DataConnector):
    def fetch_transcripts(self, firm_ids, start_date, end_date):
        # Fetch from your data source
        return TranscriptData(firms=firms_dict)

    def get_available_firm_ids(self) -> List[str]:
        # Return all available firm IDs
        return firm_ids
```

2. **Update pipeline runner** to support new connector.

### Adding Configuration Options

1. **Add to `production.yaml`**:

```yaml
my_feature:
  enabled: true
  parameter: value
```

2. **Read in code**:

```python
config = self.config.get("my_feature", {})
if config.get("enabled", False):
    # Feature logic
```

3. **Document in `CONFIGURATION.md`**.

---

## Database Migrations

### Schema Changes

The pipeline uses SQLAlchemy ORM. Schema changes require:

1. **Update models** in `cloud/src/database/models.py`

2. **Recreate tables** (development):

```python
Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)
```

3. **For production**, use Alembic migrations (not currently set up).

### Adding a New Table

```python
# cloud/src/database/models.py
class NewEntity(Base):
    __tablename__ = "new_entities"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    # ... other columns
```

### Changing Embedding Dimension

If using a different embedding model with different dimensions:

```bash
# Set environment variable BEFORE importing models
export EMBEDDING_DIMENSION=4096

# Drop and recreate tables
python -c "
from sqlalchemy import create_engine
from cloud.src.database.models import Base
engine = create_engine('postgresql://...')
Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)
"
```

---

## Debugging

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or set environment variable:

```bash
export LOG_LEVEL=DEBUG
```

### Database Queries

Enable SQLAlchemy query logging:

```python
import logging
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
```

### LLM Prompts

The XAIClient logs prompts when `log_prompt=True`:

```python
summary = await client.generate_topic_summary(
    keywords=keywords,
    sentences=sentences,
    log_prompt=True,  # Logs full prompt
)
```

### GPU Memory Issues

Monitor GPU memory:

```python
import torch
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

Clear cache if needed:

```python
torch.cuda.empty_cache()
```

---

## Common Tasks

### Run Pipeline with Subset of Firms

```bash
# First 10 firms
MAX_FIRMS=10 python scripts/run_unified_pipeline.py

# Specific test firms (MAG7)
TEST_MODE=mag7 python scripts/run_unified_pipeline.py
```

### Reset Database

```bash
# Drop all tables and recreate
python -c "
from sqlalchemy import create_engine
from cloud.src.database.models import Base
import os

engine = create_engine(os.environ['DATABASE_URL'])
Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)
print('Database reset complete')
"
```

### Export Results to CSV

```python
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine(DATABASE_URL)

# Export themes
themes_df = pd.read_sql("SELECT * FROM themes", engine)
themes_df.to_csv("themes_export.csv", index=False)

# Export with hierarchy
query = """
SELECT t.name as theme, top.representation as topic,
       top.summary, f.name as firm
FROM themes t
JOIN topics top ON top.theme_id = t.id
JOIN firms f ON top.firm_id = f.id
"""
full_df = pd.read_sql(query, engine)
full_df.to_csv("full_hierarchy.csv", index=False)
```

### Query Similar Topics

```python
from sqlalchemy import text

# Find topics similar to a given topic
query = text("""
SELECT id, representation, summary,
       1 - (embedding <=> (SELECT embedding FROM topics WHERE id = :topic_id)) as similarity
FROM topics
WHERE id != :topic_id
ORDER BY embedding <=> (SELECT embedding FROM topics WHERE id = :topic_id)
LIMIT 10
""")

with engine.connect() as conn:
    results = conn.execute(query, {"topic_id": 123})
    for row in results:
        print(f"{row.similarity:.3f}: {row.representation[:50]}")
```

### Profile Performance

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Run pipeline
pipeline.run(connector)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

---

## Pull Request Checklist

Before submitting a PR:

- [ ] Tests pass: `pytest cloud/tests/ -v`
- [ ] Code formatted: `black --check cloud/src/`
- [ ] No lint errors: `flake8 cloud/src/`
- [ ] Documentation updated if needed
- [ ] Commit messages are descriptive
- [ ] No sensitive data (API keys, passwords) committed
