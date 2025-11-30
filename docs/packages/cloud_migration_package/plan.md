Cloud Migration Plan: Financial Topic Modeling Pipeline

Executive Summary

Migrate the Local_BERTopic_MVP to a production-ready AWS cloud architecture using a Map-Reduce pattern. This plan balances simplicity with necessary modularity for future topic model comparison.

Timeline: ~6-7 days intensive development
Philosophy: Code my mom could read, boring technology, over-document the "why"

---

Architecture Overview

┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ Data Source │ │ MAP PHASE │ │ REDUCE PHASE │
│ (CSV/S3) │────▶│ AWS Batch │────▶│ AWS Batch │────▶ DynamoDB
│ │ │ Per-Firm Topics│ │ Cross-Firm │
└─────────────────┘ └─────────────────┘ └─────────────────┘
│ │
▼ ▼
S3 (JSON) Final Output

AWS Services:

- Map Phase: AWS Batch with spot instances (t3.large, 8GB)
- Reduce Phase: AWS Batch (c5.xlarge, 16GB) - simpler than SageMaker
- Orchestration: Step Functions with distributed map
- Storage: S3 (intermediate JSON), DynamoDB (final hierarchical output)
- Infrastructure: Terraform (single file for academic auditability)

---

Directory Structure

Financial_Topic_Modeling/
├── cloud/ # New cloud migration code
│ ├── README.md # Quick start guide
│ │
│ ├── src/ # Core pipeline code
│ │ ├── **init**.py
│ │ ├── interfaces.py # TopicModel + DataConnector ABCs
│ │ │
│ │ ├── connectors/ # DataConnector implementations
│ │ │ ├── **init**.py
│ │ │ ├── local_csv.py # Local CSV for testing
│ │ │ └── s3_connector.py # S3 for cloud deployment
│ │ │
│ │ ├── topic_models/ # TopicModel implementations
│ │ │ ├── **init**.py
│ │ │ ├── bertopic_model.py # BERTopic (primary)
│ │ │ ├── lda_model.py # LDA stub (NotImplemented)
│ │ │ └── neural_model.py # Neural stub (NotImplemented)
│ │ │
│ │ ├── firm_processor.py # Map phase logic
│ │ ├── theme_aggregator.py # Reduce phase logic
│ │ ├── s3_utils.py # S3 read/write helpers
│ │ └── dynamodb_utils.py # DynamoDB write helpers
│ │
│ ├── containers/
│ │ ├── map/
│ │ │ ├── Dockerfile
│ │ │ ├── requirements.txt
│ │ │ └── entrypoint.py # CLI for firm processing
│ │ └── reduce/
│ │ ├── Dockerfile
│ │ ├── requirements.txt
│ │ └── entrypoint.py # CLI for theme aggregation
│ │
│ ├── terraform/
│ │ ├── main.tf # All resources (single file)
│ │ ├── variables.tf
│ │ └── outputs.tf
│ │
│ └── scripts/
│ ├── local_test_map.sh
│ ├── local_test_reduce.sh
│ └── deploy.sh
│
├── tests/ # Test suite
│ ├── conftest.py # Pytest fixtures, moto setup
│ ├── unit/
│ │ ├── test_bertopic_model.py
│ │ ├── test_firm_processor.py
│ │ └── test_theme_aggregator.py
│ ├── integration/
│ │ ├── test_map_pipeline.py
│ │ └── test_reduce_pipeline.py
│ ├── baseline/
│ │ ├── test_topic_quality.py # Compare to MVP
│ │ └── test_theme_quality.py
│ └── fixtures/
│ ├── sample_transcripts.csv
│ └── baseline/ # MVP outputs for comparison
│
└── Local_BERTopic_MVP/ # Existing (reference only)

---

Phase 1: Abstraction Layer (Day 1)

1.1 TopicModel Interface

Create abstract interface for topic model swapping (BERTopic, LDA, neural):

# cloud/src/interfaces.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

@dataclass
class TopicModelResult:
"""Standardized output for all topic model implementations."""
topics: np.ndarray # (n_docs,) topic assignments
topic_embeddings: Dict[int, np.ndarray] # topic_id -> centroid (768d)
n_topics: int
topic_keywords: Dict[int, List[str]] # topic_id -> keywords
topic_names: Dict[int, str] # topic_id -> name
probabilities: Optional[np.ndarray] = None
metadata: Dict = None

class TopicModel(ABC):
"""Abstract interface for topic models (BERTopic, LDA, neural)."""

     @abstractmethod
     def fit_transform(
         self,
         documents: List[str],
         embeddings: Optional[np.ndarray] = None
     ) -> TopicModelResult:
         """Fit model and return standardized results."""
         pass

class DataConnector(ABC):
"""
Abstract interface for transcript data sources.

     Enables swapping between CSV (local testing), S3 (cloud),
     and WRDS (future, pending funding) without pipeline changes.
     """

     @abstractmethod
     def fetch_transcripts(
         self,
         firms: List[str],
         start_date: str,
         end_date: str
     ) -> 'TranscriptDataset':
         """Fetch transcript sentences for specified firms and date range."""
         pass

     @abstractmethod
     def get_available_firms(self) -> List[str]:
         """List all available firms in the data source."""
         pass

     def close(self) -> None:
         """Clean up resources. Default: no-op."""
         pass

1.2 DataConnector Implementations

# cloud/src/connectors/local_csv.py

class LocalCSVConnector(DataConnector):
"""Local CSV connector for testing and development.""" # Port from Local_BERTopic_MVP/src/data_ingestion/local_csv_connector.py

# cloud/src/connectors/s3_connector.py

class S3TranscriptConnector(DataConnector):
"""S3 connector for cloud deployment.""" # Read from S3 bucket with firm-partitioned data

1.3 BERTopic Implementation

Wrap existing MVP logic:

# cloud/src/bertopic_model.py

class BERTopicModel(TopicModel):
"""BERTopic implementation wrapping MVP logic."""

     def __init__(self, config: dict, sentence_model: SentenceTransformer):
         self.config = config
         self.sentence_model = sentence_model
         # Configure UMAP, HDBSCAN from config

     def fit_transform(self, documents, embeddings=None) -> TopicModelResult:
         # Port logic from FirmTopicAnalyzer.analyze_firm()
         # Return standardized TopicModelResult

1.4 Topic Model Stubs (Extensibility)

Minimal stubs to demonstrate interface compliance for faculty review:

# cloud/src/topic_models/lda_model.py

class LDATopicModel(TopicModel):
"""LDA implementation placeholder for future model comparison."""

     def fit_transform(self, documents, embeddings=None) -> TopicModelResult:
         raise NotImplementedError(
             "LDA model not yet implemented. "
             "See BERTopicModel for reference implementation."
         )

# cloud/src/topic_models/neural_model.py

class NeuralTopicModel(TopicModel):
"""Neural topic model placeholder for future research."""

     def fit_transform(self, documents, embeddings=None) -> TopicModelResult:
         raise NotImplementedError(
             "Neural topic model not yet implemented. "
             "See BERTopicModel for reference implementation."
         )

Critical MVP Files to Reference:

- Local_BERTopic_MVP/src/topic_modeling/firm_topic_analyzer.py (lines 114-251 for serialization)
- Local_BERTopic_MVP/src/data_ingestion/local_csv_connector.py (DataConnector pattern)
- Local_BERTopic_MVP/src/config/config.yaml (UMAP/HDBSCAN params)

---

Phase 2: Map Phase Container (Days 2-3)

2.1 Firm Processor

# cloud/src/firm_processor.py

class FirmProcessor:
"""Process single firm's transcripts into topics."""

     def __init__(self, topic_model: TopicModel, config: dict):
         self.model = topic_model  # Dependency injection
         self.config = config

     def process(self, firm_id: str, sentences: List[str]) -> dict:
         """Run topic modeling and return serializable results."""
         result = self.model.fit_transform(sentences)
         return self._to_dict(firm_id, result)

     def _to_dict(self, firm_id: str, result: TopicModelResult) -> dict:
         """Convert to JSON-serializable dict matching MVP format."""
         # Match FirmTopicResults.save_to_json() format

2.2 Map Container Entrypoint

# cloud/containers/map/entrypoint.py

"""
Map phase: Process single firm into topics.

Usage:
LOCAL_MODE=true FIRM_ID="Tesla, Inc." LOCAL_INPUT=./data.csv LOCAL_OUTPUT=./out/ python entrypoint.py
FIRM_ID="Tesla, Inc." S3_INPUT=s3://bucket/data.csv S3_OUTPUT_BUCKET=bucket python entrypoint.py
"""
import os
import argparse
from cloud.src.firm_processor import FirmProcessor
from cloud.src.bertopic_model import BERTopicModel
from cloud.src.s3_utils import download_csv, upload_json

def main():
parser = argparse.ArgumentParser()
parser.add_argument('--firm-id', required=True)
parser.add_argument('--local-mode', action='store_true') # ... more args

     # Initialize model (shared sentence transformer)
     model = BERTopicModel(config, sentence_model)
     processor = FirmProcessor(model, config)

     # Load data (local or S3)
     sentences = load_firm_sentences(args)

     # Process
     result = processor.process(args.firm_id, sentences)

     # Save (local or S3)
     save_result(result, args)

if **name** == '**main**':
main()

2.3 Dockerfile

FROM python:3.10-slim

WORKDIR /app

# System dependencies

RUN apt-get update && apt-get install -y --no-install-recommends \
 build-essential && rm -rf /var/lib/apt/lists/\*

# Python dependencies

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code

COPY cloud/src/ ./cloud/src/
COPY cloud/containers/map/entrypoint.py .

CMD ["python", "entrypoint.py"]

---

Phase 3: Reduce Phase Container (Days 3-4)

3.1 Theme Aggregator

# cloud/src/theme_aggregator.py

class ThemeAggregator:
"""Aggregate firm topics into cross-firm themes."""

     def __init__(self, topic_model: TopicModel, config: dict):
         self.model = topic_model  # For re-clustering
         self.config = config
         self.min_firms = config.get('min_firms', 2)
         self.max_dominance = config.get('max_firm_dominance', 0.4)

     def aggregate(self, firm_results: Dict[str, dict]) -> List[dict]:
         """
         1. Extract topic centroids from all firms
         2. Re-cluster using Dual-BERTopic
         3. Validate themes (diversity, dominance filters)
         4. Return theme dicts ready for DynamoDB
         """
         # Port logic from CrossFirmThemeAnalyzer.identify_themes()

3.2 DynamoDB Writer

# cloud/src/dynamodb_utils.py

class DynamoDBWriter:
"""Write hierarchical themes to DynamoDB."""

     def __init__(self, table_name: str):
         self.dynamodb = boto3.resource('dynamodb')
         self.table = self.dynamodb.Table(table_name)

     def write_themes(self, themes: List[dict]):
         """Write theme records with hierarchical structure."""
         with self.table.batch_writer() as batch:
             for theme in themes:
                 # Theme metadata record
                 batch.put_item(Item={
                     'PK': f"THEME#{theme['theme_id']}",
                     'SK': 'METADATA',
                     'name': theme['name'],
                     'n_firms': theme['n_firms'],
                     'keywords': theme['keywords'][:20],
                     # ...
                 })
                 # Topic records for this theme
                 for topic in theme['topics']:
                     batch.put_item(Item={
                         'PK': f"THEME#{theme['theme_id']}",
                         'SK': f"TOPIC#{topic['firm_id']}#{topic['topic_id']}",
                         # ...
                     })

Critical MVP Files to Reference:

- Local_BERTopic_MVP/src/theme_identification/cross_firm_analyzer.py (Dual-BERTopic logic)
- Local_BERTopic_MVP/src/theme_identification/theme_processor.py (validation filters)

---

Phase 4: Terraform Infrastructure (Day 4-5)

4.1 Single main.tf (~250 lines)

All resources in one auditable file:

# cloud/terraform/main.tf

# S3 Bucket - Intermediate Results

resource "aws_s3_bucket" "intermediate" {
bucket = "${var.project_name}-intermediate-${var.environment}"
}

# DynamoDB - Final Themes

resource "aws_dynamodb_table" "themes" {
name = "${var.project_name}-themes"
billing_mode = "PAY_PER_REQUEST"
hash_key = "PK"
range_key = "SK"

# ...

}

# ECR Repositories

resource "aws_ecr_repository" "map" { ... }
resource "aws_ecr_repository" "reduce" { ... }

# AWS Batch - Compute Environment (Spot instances)

resource "aws_batch_compute_environment" "main" {
compute_resources {
type = "SPOT"
bid_percentage = 100
max_vcpus = 32
instance_type = ["t3.large", "m5.large"]
}
}

# AWS Batch - Job Definitions

resource "aws_batch_job_definition" "map" { ... }
resource "aws_batch_job_definition" "reduce" { ... }

# Step Functions - Orchestration

resource "aws_sfn_state_machine" "pipeline" {
definition = jsonencode({
StartAt = "MapPhase"
States = {
MapPhase = {
Type = "Map"
MaxConcurrency = 10
Iterator = { ... }
Next = "ReducePhase"
}
ReducePhase = { ... }
}
})
}

---

Phase 5: Testing Strategy (Throughout)

Coverage Target: 80%+ (REQUIRED)

TDD for core logic and abstractions. Integration tests for containers. Baseline validation for quality.

5.1 Test-Driven Development

Write tests BEFORE implementation for all core logic:

# tests/unit/test_interfaces.py - Test abstract contracts

# tests/unit/test_bertopic_model.py - Test TopicModel implementation

# tests/unit/test_connectors.py - Test DataConnector implementations

# tests/unit/test_firm_processor.py - Test map phase logic

# tests/unit/test_theme_aggregator.py - Test reduce phase logic

class TestFirmProcessor:
def test_process_returns_valid_schema(self, mock_topic_model):
processor = FirmProcessor(mock_topic_model, config)
result = processor.process("AAPL", sample_sentences)

         assert 'firm_id' in result
         assert 'centroids' in result
         assert result['n_topics'] > 0

     def test_handles_empty_input(self, mock_topic_model):
         # Edge case testing

     def test_uses_injected_connector(self, mock_connector, mock_topic_model):
         # Verify dependency injection works

5.2 AWS Mocking with Moto

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
], # ...
)
yield table

5.3 Baseline Validation

Compare cloud results to MVP:

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

---

Phase 6: Integration & Validation (Days 5-7)

6.1 Local Testing Sequence

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

6.2 Cloud Deployment

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

6.3 Validation Criteria

| Metric                  | Target    | Tolerance |
| ----------------------- | --------- | --------- |
| Theme count             | Match MVP | ±15%      |
| Firm coverage           | 100%      | ≥90%      |
| Processing time/firm    | <5 min    | -         |
| Memory usage            | <8GB      | -         |
| AWS cost (full quarter) | <$50      | -         |

---

Implementation Timeline

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

What We Explicitly Skip (Simplicity)

| Feature                  | Rationale                                  | Add Later If Needed        |
| ------------------------ | ------------------------------------------ | -------------------------- |
| StorageBackend interface | Direct boto3 calls suffice for S3/DynamoDB | If storage becomes complex |
| Terraform modules        | ~250 lines doesn't need modularity         | For multi-environment      |
| VPC/networking           | Default VPC works for academic project     | For production security    |
| CI/CD pipeline           | Manual deployment fine for 1 week          | After first success        |
| WRDS connector           | Funding-dependent; interface ready         | When WRDS access granted   |

What We Include (User Decisions)

| Feature                   | Rationale                                                   |
| ------------------------- | ----------------------------------------------------------- |
| DataConnector interface   | Enables CSV/S3 today, WRDS tomorrow without refactors       |
| TopicModel interface      | Enables BERTopic/LDA/neural comparison (core research need) |
| LDA/neural stubs          | Minimal NotImplemented placeholders for faculty review      |
| 80%+ test coverage        | TDD for core logic, integration for containers              |
| AWS Batch for both phases | Simpler than mixing Batch + SageMaker                       |

---

Critical Files from MVP

1.  Local_BERTopic_MVP/src/topic_modeling/firm_topic_analyzer.py

- FirmTopicAnalyzer.analyze_firm() - core map logic
- FirmTopicResults.save_to_json() - serialization format (lines 114-182)

2.  Local_BERTopic_MVP/src/theme_identification/cross_firm_analyzer.py

- CrossFirmThemeAnalyzer.identify_themes() - core reduce logic
- Dual-BERTopic approach (lines 235-262)

3.  Local_BERTopic_MVP/src/theme_identification/theme_processor.py

- Two-tier frequency validation (min_firms, max_dominance)

4.  Local_BERTopic_MVP/src/data_ingestion/data_structures.py

- TranscriptSentence, Theme, ThemeCluster schemas

5.  Local_BERTopic_MVP/src/config/config.yaml

- UMAP: n_neighbors=15, n_components=10
- HDBSCAN: min_cluster_size=6
- Validation: min_firms=2, max_firm_dominance=0.4

---

User Decisions (Resolved)

- Abstractions: Include both TopicModel + DataConnector interfaces
- Reduce Phase: AWS Batch (simpler than SageMaker, same pattern as map)
- Test Coverage: 80%+ required - TDD for core, integration for containers
- Model Stubs: Include minimal LDA/neural stubs with NotImplementedError
