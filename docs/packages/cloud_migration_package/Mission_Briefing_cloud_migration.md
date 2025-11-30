---
**To:** Claude, Cloud Infrastructure Engineer
**From:** Senior Quantitative Researcher
**Subject:** Mission Briefing - Cloud Migration for Financial Topic Modeling Pipeline
**Date:** November 2024
**Priority:** Critical
**Timeline:** ~1 week intensive development
---

## Mission Overview

You are a **Cloud Infrastructure Engineer** working on the Financial Topic Modeling pipeline. Your mission is to migrate the existing local MVP (`Local_BERTopic_MVP`) to a production-ready, cloud-native AWS architecture as specified in `Senior_Engineer_Plan_cloud_migration.md`.

**Primary Context Documents:**
- Project Guidelines: `CLAUDE.md`
- Feature Specification: `docs/packages/cloud_migration_package/Senior_Engineer_Plan_cloud_migration.md`
- Legacy Reference: `Local_BERTopic_MVP/` (ideas and data flow only)
- Vision Document: `First_Transcript.md`

**Design Philosophy Reminder:**
- Prioritize simplicity over complexity
- Use boring technology
- Over-document the "why", under-engineer the "how"
- Code my mom could read
- Complexity is a liability, not a flex

---

## Code Quality Standards

1. All new code must have corresponding unit tests (>80% coverage)
2. Follow existing patterns in `Local_BERTopic_MVP/src/` where they make sense
3. Document the "why" in comments, not the "what"
4. No over-engineering - implement only what is specified
5. Local testing must pass before any cloud deployment
6. All infrastructure must be defined in Terraform (auditable by faculty)
7. Containers must be testable locally with mocks

---

## Phase 1: Codebase Analysis & Architecture Design

**Objective:** Deeply understand the existing MVP and validate the proposed cloud architecture before writing any code.

**Tasks:**
1. Analyze `Local_BERTopic_MVP/src/` to understand the data flow:
   - `data_ingestion/` - How transcripts are loaded and processed
   - `topic_modeling/` - How BERTopic is configured and run per-firm
   - `theme_identification/` - How cross-firm aggregation works
2. Identify the exact inputs and outputs at each pipeline stage
3. Map existing code to the proposed Map-Reduce architecture
4. Document any gaps or inconsistencies between MVP and proposed design
5. Validate AWS service selections against actual requirements:
   - Memory requirements for topic modeling
   - Estimated processing time per firm
   - S3 intermediate file size estimates

**Deliverables:**
- [ ] Architecture validation report (confirm or revise proposed design)
- [ ] Data flow diagram with exact schemas at each stage
- [ ] List of MVP code to reuse vs. rewrite
- [ ] Revised cost/time estimates if needed

== END OF PHASE 1 ==
**STOP and await my review and approval before proceeding to abstraction layer implementation.**

---

## Phase 2: Abstraction Layer & Local Testing Framework

**Objective:** Create the modular abstractions that enable both local testing and cloud deployment.

**Tasks:**
1. Design and implement `TopicModelInterface` abstract class:
   - `fit(documents: List[str]) -> TopicModelResult`
   - `get_topics() -> List[Topic]`
   - `get_embeddings() -> np.ndarray`
   - Support for BERTopic (current) and future models (LDA, neural)

2. Design and implement `DataConnector` abstract class:
   - `get_firms(start_date, end_date) -> List[str]`
   - `get_transcripts(firm_id) -> TranscriptData`
   - Implementations: `LocalCSVConnector`, `S3Connector`, (future: `WRDSConnector`)

3. Create local execution wrapper:
   - Run full pipeline without cloud dependencies
   - Mock S3 and DynamoDB for local testing
   - CLI interface matching cloud container entry points

4. Write comprehensive unit tests for all abstractions

**Validation Criteria:**
- [ ] Full pipeline runs locally on sample CSV
- [ ] Topic model is injectable (can swap BERTopic implementation)
- [ ] Data source is injectable (can swap CSV for S3)
- [ ] All unit tests pass
- [ ] Test coverage >80%

== END OF PHASE 2 ==
**STOP and await my review and approval before proceeding to container implementation.**

---

## Phase 3: Map Phase Container Implementation

**Objective:** Build, test, and validate the firm-level topic modeling container.

**Tasks:**
1. Create `containers/firm-topic-processor/` directory structure:
   ```
   containers/firm-topic-processor/
   ├── Dockerfile
   ├── requirements.txt
   ├── entrypoint.py
   ├── src/
   │   ├── processor.py
   │   └── s3_writer.py
   └── tests/
       └── test_processor.py
   ```

2. Implement `entrypoint.py` with CLI arguments:
   - `--firm-id`: Firm to process
   - `--data-source`: S3 path or local path
   - `--output-bucket`: S3 bucket for results
   - `--output-prefix`: S3 key prefix

3. Implement S3 output writer with schema:
   ```json
   {
     "firm_id": "...",
     "firm_name": "...",
     "n_topics": 10,
     "topics": [...],
     "centroids": [...],
     "metadata": {...}
   }
   ```

4. Build Docker image and test locally:
   - Process single firm from sample CSV
   - Write to local directory (mock S3)
   - Validate output schema

5. Test with actual S3 (manual, single firm)

**Validation Criteria:**
- [ ] Docker build succeeds
- [ ] Container processes single firm correctly
- [ ] Output matches expected JSON schema
- [ ] Handles edge cases (empty transcript, missing firm)
- [ ] Processing time: < 5 minutes per firm (benchmark)
- [ ] Memory usage: < 8GB (benchmark)

== END OF PHASE 3 ==
**STOP and await my review and approval before proceeding to reduce phase.**

---

## Phase 4: Reduce Phase Container Implementation

**Objective:** Build, test, and validate the cross-firm theme aggregation container.

**Tasks:**
1. Create `containers/theme-aggregator/` directory structure:
   ```
   containers/theme-aggregator/
   ├── Dockerfile
   ├── requirements.txt
   ├── entrypoint.py
   ├── src/
   │   ├── aggregator.py
   │   ├── s3_reader.py
   │   └── dynamodb_writer.py
   └── tests/
       └── test_aggregator.py
   ```

2. Implement `entrypoint.py` with CLI arguments:
   - `--input-bucket`: S3 bucket with firm results
   - `--input-prefix`: S3 key prefix to read
   - `--dynamodb-table`: Output table name

3. Implement S3 reader:
   - List all firm result files
   - Load and parse JSON results
   - Handle missing/corrupt files gracefully

4. Implement DynamoDB writer with hierarchical schema:
   - Theme records (PK: theme_id)
   - Topic records (PK: theme_id, SK: topic_id)
   - Firm-topic mappings

5. Build Docker image and test locally:
   - Read from local directory (mock S3)
   - Write to local DynamoDB (localstack or mock)
   - Validate output schema

6. Test with actual AWS services (manual, small dataset)

**Validation Criteria:**
- [ ] Docker build succeeds
- [ ] Container aggregates firm results correctly
- [ ] Cross-firm themes match MVP quality
- [ ] DynamoDB records follow hierarchical schema
- [ ] Handles edge cases (missing firms, no themes found)
- [ ] Processing time: < 30 minutes for full quarter (benchmark)

== END OF PHASE 4 ==
**STOP and await my review and approval before proceeding to Terraform infrastructure.**

---

## Phase 5: Terraform Infrastructure

**Objective:** Define all AWS infrastructure as auditable, reproducible code.

**Tasks:**
1. Create `terraform/` directory structure:
   ```
   terraform/
   ├── main.tf
   ├── variables.tf
   ├── outputs.tf
   ├── modules/
   │   ├── networking/
   │   ├── storage/
   │   ├── compute/
   │   └── orchestration/
   └── environments/
       ├── dev.tfvars
       └── prod.tfvars
   ```

2. Implement networking module:
   - VPC with public/private subnets
   - Security groups for containers

3. Implement storage module:
   - S3 bucket for intermediate results (with lifecycle policy)
   - DynamoDB table with appropriate indexes
   - ECR repositories for containers

4. Implement compute module:
   - AWS Batch compute environment (spot instances)
   - AWS Batch job definitions for map container
   - SageMaker processing job configuration for reduce

5. Implement orchestration module:
   - Step Functions state machine
   - Distributed map for parallel firm processing
   - Error handling and retry logic

6. Document all resources with comments explaining "why"

**Validation Criteria:**
- [ ] `terraform init` succeeds
- [ ] `terraform plan` shows expected resources
- [ ] `terraform apply` creates all resources (dev environment)
- [ ] Resources are tagged for cost tracking
- [ ] Documentation is clear for faculty review

== END OF PHASE 5 ==
**STOP and await my review and approval before proceeding to integration testing.**

---

## Phase 6: End-to-End Integration & Validation

**Objective:** Run complete pipeline on baseline CSV and validate against MVP results.

**Tasks:**
1. Push container images to ECR

2. Deploy infrastructure to dev environment

3. Run single-firm test:
   - Trigger map container for one firm
   - Verify S3 output
   - Document any issues

4. Run full pipeline test:
   - Trigger Step Functions workflow
   - Process all firms from Q1 2023 CSV
   - Aggregate results in reduce phase
   - Verify DynamoDB output

5. Validate results against MVP baseline:
   - Compare number of themes identified
   - Compare topic coherence scores
   - Compare firm coverage

6. Document performance and cost:
   - Total processing time
   - AWS cost breakdown
   - Comparison to local MVP runtime

7. Create runbook for pipeline execution

**Final Validation:**
- [ ] Full pipeline completes without errors
- [ ] Processing time < 2 hours for full quarter
- [ ] AWS cost < $50 for full quarter (estimate)
- [ ] Theme quality matches MVP baseline
- [ ] All integration tests pass
- [ ] Runbook is complete and tested

== FINAL HALTING POINT ==

**Required Deliverables for Mission Completion:**

1. **Code Deliverables:**
   - [ ] Abstraction layer (`src/abstractions/`)
   - [ ] Map container (`containers/firm-topic-processor/`)
   - [ ] Reduce container (`containers/theme-aggregator/`)
   - [ ] Terraform infrastructure (`terraform/`)

2. **Documentation Deliverables:**
   - [ ] Architecture validation report
   - [ ] Data flow diagram with schemas
   - [ ] Terraform resource documentation
   - [ ] Pipeline runbook
   - [ ] Performance/cost benchmark report

3. **Test Deliverables:**
   - [ ] Unit test suite (>80% coverage)
   - [ ] Integration test suite
   - [ ] Validation report comparing to MVP

4. **Evidence:**
   - [ ] Screenshot/logs of successful pipeline run
   - [ ] DynamoDB query results showing hierarchical data
   - [ ] AWS Cost Explorer showing actual costs

**Await final approval before marking mission complete.**
