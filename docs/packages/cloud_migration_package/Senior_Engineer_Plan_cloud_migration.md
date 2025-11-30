# Senior Engineer Plan: Cloud Migration for Financial Topic Modeling Pipeline

## 1. Executive Summary & Project Context

This plan outlines the migration of the Financial Topic Modeling pipeline from a local MVP implementation to a production-ready, cloud-native architecture on AWS. The project transforms an earnings call theme identification system that currently requires manual distribution across multiple team members' computers into a scalable, automated batch processing pipeline capable of handling 10,000+ firms across multiple quarters.

**Project Type:** Academic capstone research with publication intent
**Timeline Constraint:** Approximately one week of intensive development
**Team Context:** Undergraduate research assistant with finance, CS, and statistics background; faculty collaboration in finance and NLP

**Related Context:**
- Project Vision: `First_Transcript.md`
- Project Guidelines: `CLAUDE.md`
- Legacy Reference: `Local_BERTopic_MVP/` (use for ideas and data flow only, not code patterns)

## 2. Core Problem & Architectural Vision

### 2.1 The Problem

The existing `Local_BERTopic_MVP` successfully demonstrates the earnings call theme identification pipeline but cannot scale:
- Processing a single quarter (~1,000 firms) required distributing work across multiple team members' computers
- No automated orchestration; manual file management for intermediate results
- Cannot support multi-quarter longitudinal analysis (target: 10,000+ firms)
- Not suitable for academic publication or downstream use by others

### 2.2 Architectural Vision

Implement a **Map-Reduce pattern** using AWS cloud services:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Data Ingestion │     │   MAP PHASE     │     │  REDUCE PHASE   │
│  (CSV/WRDS/S3)  │────▶│  Firm-Level     │────▶│  Cross-Firm     │────▶ DynamoDB
│                 │     │  Topic Modeling │     │  Theme ID       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                              │                        │
                              ▼                        ▼
                         S3 (intermediate)      Final Output
```

**Key Design Decisions:**
- **Map Phase:** AWS Batch with spot instances (cost-optimized, parallelizable)
- **Reduce Phase:** SageMaker instance (larger compute for aggregation)
- **Orchestration:** AWS Step Functions with distributed map
- **Storage:** S3 for intermediates, DynamoDB for final hierarchical output
- **Infrastructure:** Terraform for reproducibility and auditability

## 3. Pipeline Architecture

### 3.1 Data Ingestion Layer

**Current State:** Local CSV file (`transcripts_2023-01-01_to_2023-03-31_enriched.csv`)
**Future State:** WRDS connector for direct database queries

**Modularity Requirement:** Design data ingestion as a pluggable interface to support:
- Local CSV files (current)
- WRDS database connector (future, pending funding)
- S3/Athena cloud sources

**Affected Module:** `Local_BERTopic_MVP/src/data_ingestion/`

### 3.2 Map Phase: Firm-Level Topic Modeling

**Purpose:** Process each firm's earnings call transcripts independently to identify firm-level topics.

**Processing Flow per Firm:**
1. Load transcript data for single firm
2. Preprocess sentences (filtering, cleaning)
3. Generate sentence embeddings (sentence-transformers)
4. Run BERTopic clustering
5. Calculate cluster centroids in original embedding space
6. Write results to S3 intermediate storage

**Container Specification:**
- Dependencies: BERTopic, sentence-transformers, PyTorch (minimal)
- Input: Firm identifier + data source reference
- Output: JSON with topics, embeddings, centroids, metadata

**Modularity Requirement:** Topic modeling should be abstracted to support swappable implementations:
- BERTopic (current)
- LDA (future comparison)
- Neural topic models (future comparison)

**Affected Module:** `Local_BERTopic_MVP/src/topic_modeling/`

### 3.3 Reduce Phase: Cross-Firm Theme Identification

**Purpose:** Aggregate firm-level topics to discover universal cross-firm themes.

**Processing Flow:**
1. Load all firm topic results from S3
2. Re-embed topic representations
3. Run secondary BERTopic clustering on topic centroids
4. Validate themes (diversity, concentration filters)
5. Write hierarchical results to DynamoDB

**Output Schema (DynamoDB):**
```
Themes → Topics → Sentences → Firms
```

**Container Specification:**
- Larger instance than map phase (more memory for aggregation)
- Dependencies: Same as map phase + boto3 for DynamoDB
- Input: S3 path to intermediate results
- Output: DynamoDB records

**Affected Module:** `Local_BERTopic_MVP/src/theme_identification/`

### 3.4 Downstream Analysis (Out of Scope)

The following are explicitly **out of scope** for this migration but should not be broken:
- Sentiment analysis (`sentiment_analysis/`) - FinBERT integration
- Event study (`event_study/`) - Statistical analysis framework

## 4. Cloud Infrastructure Design

### 4.1 AWS Services Selection

| Component | Service | Rationale |
|-----------|---------|-----------|
| Orchestration | Step Functions | Distributed map support, visual workflow |
| Map Phase | AWS Batch (Spot) | Cost-optimized batch processing |
| Reduce Phase | SageMaker | ML-optimized instance, managed environment |
| Intermediate Storage | S3 | Standard, cost-effective object storage |
| Final Storage | DynamoDB | Hierarchical queries, serverless scaling |
| Infrastructure | Terraform | Reproducible, auditable, team-friendly |

### 4.2 Container Architecture

**Map Container (`firm-topic-processor`):**
```dockerfile
# Minimal dependencies for firm-level processing
- Python 3.10+
- sentence-transformers
- BERTopic
- PyTorch (CPU or GPU based on instance)
- boto3 (S3 writes)
```

**Reduce Container (`theme-aggregator`):**
```dockerfile
# Dependencies for cross-firm aggregation
- Same as map container
- Additional memory allocation
- DynamoDB client
```

### 4.3 Cost Optimization Strategy

- Use Spot instances for map phase (interruptible, 60-90% savings)
- Right-size instances based on actual memory/compute needs
- S3 lifecycle policies for intermediate file cleanup
- DynamoDB on-demand pricing for variable workloads

## 5. Epic Breakdown

### Part 5.1: Local Development Environment & Abstractions

**Objective:** Create the modular abstractions that will be used in both local testing and cloud deployment.

**Deliverables:**
- Abstract `TopicModelInterface` class for swappable topic models
- Abstract `DataConnector` interface for pluggable data sources
- Local execution wrapper for testing without cloud

**Acceptance Criteria:**
- [ ] Can run full pipeline locally on sample CSV
- [ ] Topic model is injectable/swappable
- [ ] Data source is injectable/swappable
- [ ] Unit test coverage > 80%

### Part 5.2: Map Phase Container Implementation

**Objective:** Build and test the firm-level topic modeling container.

**Deliverables:**
- Dockerfile for `firm-topic-processor`
- Entry point script with CLI arguments
- S3 output writer
- Local testing harness

**Acceptance Criteria:**
- [ ] Container builds successfully
- [ ] Processes single firm and writes to S3 (or local mock)
- [ ] Output format matches expected schema
- [ ] Handles edge cases (empty transcripts, missing data)
- [ ] Processing time benchmarked per firm

### Part 5.3: Reduce Phase Container Implementation

**Objective:** Build and test the cross-firm theme aggregation container.

**Deliverables:**
- Dockerfile for `theme-aggregator`
- S3 reader for intermediate results
- DynamoDB writer with hierarchical schema
- Local testing harness

**Acceptance Criteria:**
- [ ] Container builds successfully
- [ ] Reads intermediate results from S3 (or local mock)
- [ ] Produces valid cross-firm themes
- [ ] Writes hierarchical data to DynamoDB (or local mock)
- [ ] Theme quality metrics match MVP baseline

### Part 5.4: Terraform Infrastructure

**Objective:** Define all AWS infrastructure as code.

**Deliverables:**
- Terraform modules for: VPC, S3, DynamoDB, ECR, Batch, SageMaker, Step Functions
- Variable files for different environments (dev, prod)
- Documentation of all resources

**Acceptance Criteria:**
- [ ] `terraform plan` succeeds without errors
- [ ] `terraform apply` creates all required resources
- [ ] Resources are tagged appropriately for cost tracking
- [ ] Team members can audit and understand the infrastructure

### Part 5.5: End-to-End Integration & Validation

**Objective:** Run complete pipeline on baseline CSV and validate results.

**Deliverables:**
- Step Functions workflow definition
- Integration test suite
- Performance and cost benchmarks
- Documentation of results

**Acceptance Criteria:**
- [ ] Full pipeline completes on Q1 2023 CSV
- [ ] Processing time documented
- [ ] AWS cost documented
- [ ] Theme quality matches or exceeds MVP baseline
- [ ] All tests pass

## 6. Testing & Validation Mandate

### 6.1 Test-Driven Development Requirements

- **Unit Tests:** Required for all new functions and classes
- **Integration Tests:** Required for container entry points
- **Local Mocks:** S3 and DynamoDB must be mockable for local testing
- **Coverage Target:** >80% for all new code

### 6.2 ML Validation Requirements

- Topic coherence scores must match MVP baseline
- Firm coverage (% of firms with meaningful topics) must be maintained
- Cross-firm theme diversity must meet existing thresholds

### 6.3 Validation Sequence

1. Local unit tests pass
2. Local integration tests with mocks pass
3. Single-firm cloud test succeeds
4. Full pipeline cloud test succeeds
5. Results validation against MVP baseline

## 7. Technical Debt & Future Considerations

### 7.1 Deferred for Future Work

- **LLM Integration:** Sentence filtering, stop-word generation, topic/theme naming
- **Alternative Topic Models:** LDA, neural topic models for comparison
- **WRDS Integration:** Direct database connector (pending funding)
- **Multi-quarter Analysis:** Temporal trend analysis across quarters

### 7.2 Known Limitations

- Embedding model loaded per-container (consider shared endpoint for optimization)
- Spot instance interruption handling not fully robust
- DynamoDB schema optimized for reads, not complex queries

## 8. Design Philosophy Principles

Per the project vision, all implementation must adhere to:

1. **Simplicity over Complexity:** "The best engineers write code my mom could read"
2. **Boring Technology:** Use proven, well-documented AWS services
3. **Over-document the Why:** Every architectural decision must be justified
4. **Under-engineer the How:** Minimal code, maximum clarity
5. **Validation-as-you-go:** TDD, continuous testing, no surprises at integration
6. **Auditability:** Faculty and team members must be able to review and understand
