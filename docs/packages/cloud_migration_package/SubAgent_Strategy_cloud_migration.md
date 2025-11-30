<Objective title="Cloud Migration for Financial Topic Modeling Pipeline" summary="Migrate the Local_BERTopic_MVP to a production-ready AWS cloud architecture using a Map-Reduce pattern with AWS Batch, SageMaker, S3, DynamoDB, and Terraform infrastructure-as-code">

This document provides a recommended sequence of agent invocations and tool usage to complement the main Mission Briefing. It is designed to guide the AI Engineer through implementing the Cloud Migration for the Financial Topic Modeling pipeline.

**Project Context:**
- Vision Document: `First_Transcript.md`
- Project Guidelines: `CLAUDE.md`
- Feature Specification: `docs/packages/cloud_migration_package/Senior_Engineer_Plan_cloud_migration.md`
- Legacy Reference: `Local_BERTopic_MVP/src/`

If any sub-agents are invoked, ensure they return focused reports summarizing their findings for the main agent stream.

**Critical Reminder:** The `Local_BERTopic_MVP` code is poorly structured legacy code. Extract only the ideas, data flow, and specifications. Do not copy code patterns directly.

---

<Phase 1 title="Deep Research & Codebase Analysis" purpose="Understanding the existing MVP architecture and validating the proposed cloud design before any implementation" relevant_modules="data_ingestion/, topic_modeling/, theme_identification/, config/">

<Invocation 1 title="Comprehensive MVP Analysis">

- **Agent:** `Explore` (via Task tool)
- **Approach:** `"very thorough"` - complete analysis of all pipeline modules
- **Focus Areas:**
  - `Local_BERTopic_MVP/src/main.py` - Pipeline orchestration flow
  - `Local_BERTopic_MVP/src/data_ingestion/` - Data loading and processing
  - `Local_BERTopic_MVP/src/topic_modeling/firm_topic_analyzer.py` - Per-firm topic modeling
  - `Local_BERTopic_MVP/src/theme_identification/cross_firm_analyzer.py` - Cross-firm aggregation
  - `Local_BERTopic_MVP/src/config/config.yaml` - Configuration structure
- **Expected Output:** Report detailing:
  - Exact data flow from ingestion to final themes
  - Input/output schemas at each stage
  - BERTopic configuration and parameters used
  - Memory and compute requirements observed
- **Reasoning:** Before implementing cloud architecture, we must deeply understand what the MVP actually does. This analysis will validate our Map-Reduce design and identify any gaps. The "very thorough" approach ensures we don't miss edge cases or implicit dependencies.

</Invocation 1>

<Invocation 2 title="AWS Architecture Research">

- **Agent:** `api-docs-synthesizer` (via Task tool)
- **Focus Areas:**
  - AWS Batch job array patterns and spot instance configuration
  - Step Functions distributed map for parallel processing
  - SageMaker Processing Jobs for ML workloads
  - DynamoDB data modeling for hierarchical data
  - Terraform AWS provider best practices
- **Expected Output:** Synthesized documentation on:
  - Recommended instance types for BERTopic workloads
  - Step Functions distributed map limits and patterns
  - DynamoDB schema design for theme→topic→firm hierarchy
- **Reasoning:** The researcher is using this as an educational exercise. We need to ensure our AWS service selections are optimal and well-documented. This research validates or refines the initial architecture proposal.

</Invocation 2>

<Invocation 3 title="Create Architecture Validation Report">

- **Tools:** `Read` MVP files, `Write` validation report
- **Actions:**
  - Synthesize findings from Invocations 1 and 2
  - Create data flow diagram (ASCII or Mermaid)
  - Document exact schemas at each pipeline stage
  - List MVP code to reuse (minimal) vs. rewrite (most)
  - Confirm or revise AWS service selections
- **Output Location:** `docs/architecture_validation_report.md`
- **Reasoning:** This report is the foundation for all subsequent work. It ensures alignment between MVP understanding and cloud design before any implementation begins. The researcher explicitly requested thorough planning before coding.

</Invocation 3>

</Phase 1>

---

<Phase 2 title="Abstraction Layer & Local Testing Framework" purpose="Building the modular foundations that enable both local testing and cloud deployment" relevant_modules="new: src/abstractions/, src/local_runner/">

<Invocation 1 title="Design Abstract Interfaces">

- **Tools:** `Write` for interface files
- **Files to Create:**
  - `src/abstractions/__init__.py`
  - `src/abstractions/topic_model_interface.py` - Abstract base class
  - `src/abstractions/data_connector_interface.py` - Abstract base class
  - `src/abstractions/storage_interface.py` - S3/local abstraction
- **Design Pattern:** Python ABC (Abstract Base Class) with type hints
- **Reasoning:** These abstractions are the key to modularity. They enable swapping BERTopic for LDA, swapping CSV for WRDS, and testing locally without AWS. TDD starts here - interfaces first, then tests, then implementation.

</Invocation 1>

<Invocation 2 title="Implement BERTopic Adapter">

- **Approach:** Test-Driven Development
- **Steps:**
  1. Write tests first: `tests/test_bertopic_adapter.py`
  2. Implement: `src/adapters/bertopic_adapter.py`
  3. Validate against MVP output format
- **Tools:** `Write` for files, `Bash` for running pytest
- **Reasoning:** The BERTopic adapter is the first concrete implementation of our interface. Writing tests first ensures we understand the expected behavior before implementation.

</Invocation 2>

<Invocation 3 title="Implement Data Connectors">

- **Approach:** Test-Driven Development
- **Files to Create:**
  - `src/connectors/local_csv_connector.py` - For local testing
  - `src/connectors/s3_connector.py` - For cloud deployment
  - `tests/test_connectors.py`
- **Tools:** `Write` for files, `Bash` for pytest
- **Reasoning:** Data connectors must be testable locally (CSV) and work identically in the cloud (S3). This abstraction enables the same processing code to work in both environments.

</Invocation 3>

<Invocation 4 title="Create Local Execution Wrapper">

- **Files to Create:**
  - `src/local_runner/run_local.py` - CLI for local pipeline execution
  - `src/local_runner/mock_aws.py` - Mock S3 and DynamoDB
- **Tools:** `Write` for files, `Bash` for testing
- **Validation:**
  - Run full pipeline locally on sample CSV
  - Verify output matches MVP format
- **Reasoning:** The local runner is essential for rapid iteration. We must be able to test the full pipeline without deploying to AWS. This aligns with the "validation-as-you-go" principle.

</Invocation 4>

</Phase 2>

---

<Phase 3 title="Map Phase Container Implementation" purpose="Building and testing the firm-level topic modeling container for AWS Batch" relevant_modules="new: containers/firm-topic-processor/">

<Invocation 1 title="Create Container Structure">

- **Tools:** `Write` for all container files
- **Directory Structure:**
  ```
  containers/firm-topic-processor/
  ├── Dockerfile
  ├── requirements.txt
  ├── entrypoint.py
  ├── src/
  │   ├── __init__.py
  │   ├── processor.py
  │   └── s3_writer.py
  └── tests/
      ├── __init__.py
      └── test_processor.py
  ```
- **Reasoning:** Clean container structure with clear separation. Tests live with the container so they can be run during CI/CD.

</Invocation 1>

<Invocation 2 title="Implement Firm Processor with TDD">

- **Approach:** Test-Driven Development
- **Steps:**
  1. Write `tests/test_processor.py` with expected behavior
  2. Implement `src/processor.py` using abstractions from Phase 2
  3. Implement `src/s3_writer.py` with local fallback
  4. Implement `entrypoint.py` with CLI arguments
- **Tools:** `Write`, `Edit`, `Bash` for pytest
- **Reasoning:** TDD ensures each component works before integration. The processor uses our abstraction layer, so it should be straightforward if Phase 2 was done correctly.

</Invocation 2>

<Invocation 3 title="Build and Test Docker Container">

- **Tools:** `Bash` for Docker commands
- **Commands:**
  ```bash
  cd containers/firm-topic-processor
  docker build -t firm-topic-processor:local .
  docker run firm-topic-processor:local --help
  docker run -v $(pwd)/test_data:/data firm-topic-processor:local \
    --firm-id "Tesla, Inc." --data-source /data/sample.csv --output-dir /data/output
  ```
- **Validation:**
  - Container builds successfully
  - Help command shows expected arguments
  - Single firm processes correctly
  - Output JSON matches expected schema
- **Reasoning:** Local Docker testing catches issues before cloud deployment. This is much faster and cheaper than debugging in AWS Batch.

</Invocation 3>

</Phase 3>

---

<Phase 4 title="Reduce Phase Container Implementation" purpose="Building and testing the cross-firm theme aggregation container for SageMaker" relevant_modules="new: containers/theme-aggregator/">

<Invocation 1 title="Create Container Structure">

- **Tools:** `Write` for all container files
- **Directory Structure:**
  ```
  containers/theme-aggregator/
  ├── Dockerfile
  ├── requirements.txt
  ├── entrypoint.py
  ├── src/
  │   ├── __init__.py
  │   ├── aggregator.py
  │   ├── s3_reader.py
  │   └── dynamodb_writer.py
  └── tests/
      ├── __init__.py
      └── test_aggregator.py
  ```
- **Reasoning:** Same clean structure as map container. The aggregator is more complex due to cross-firm logic, so tests are even more important.

</Invocation 1>

<Invocation 2 title="Implement Theme Aggregator with TDD">

- **Approach:** Test-Driven Development
- **Steps:**
  1. Write `tests/test_aggregator.py` - especially cross-firm theme logic
  2. Implement `src/s3_reader.py` - read and parse firm results
  3. Implement `src/aggregator.py` - core theme identification
  4. Implement `src/dynamodb_writer.py` - hierarchical output
  5. Implement `entrypoint.py` with CLI arguments
- **Tools:** `Write`, `Edit`, `Bash` for pytest
- **Key Challenge:** DynamoDB schema must support hierarchical queries (theme→topics→firms)
- **Reasoning:** The aggregator is the most complex component. TDD and thorough testing here prevent issues during integration.

</Invocation 2>

<Invocation 3 title="Build and Test Docker Container">

- **Tools:** `Bash` for Docker commands
- **Commands:**
  ```bash
  cd containers/theme-aggregator
  docker build -t theme-aggregator:local .
  docker run -v $(pwd)/test_data:/data theme-aggregator:local \
    --input-dir /data/firm_results --output-dir /data/themes
  ```
- **Validation:**
  - Container builds successfully
  - Aggregates multiple firm results correctly
  - Theme quality matches MVP baseline
  - Output format ready for DynamoDB
- **Reasoning:** Local testing with mock data before deploying to SageMaker. This validates the core cross-firm logic independent of AWS.

</Invocation 3>

</Phase 4>

---

<Phase 5 title="Terraform Infrastructure" purpose="Defining all AWS resources as auditable, reproducible infrastructure-as-code" relevant_modules="new: terraform/">

<Invocation 1 title="Research Terraform Patterns">

- **Agent:** `api-docs-synthesizer` (via Task tool)
- **Focus:**
  - Terraform AWS provider best practices
  - AWS Batch with Terraform examples
  - Step Functions state machine definition in Terraform
  - DynamoDB table design with GSIs
- **Reasoning:** Infrastructure-as-code is new to this project. Research ensures we follow best practices and don't reinvent solutions that already exist.

</Invocation 1>

<Invocation 2 title="Implement Terraform Modules">

- **Tools:** `Write` for all Terraform files
- **Module Order (dependencies):**
  1. `modules/networking/` - VPC, subnets, security groups
  2. `modules/storage/` - S3, DynamoDB, ECR
  3. `modules/compute/` - Batch, SageMaker
  4. `modules/orchestration/` - Step Functions
  5. `main.tf` - Root module composing all
- **Reasoning:** Modular Terraform structure allows team members to understand and modify individual components. Each module is self-contained and documented.

</Invocation 2>

<Invocation 3 title="Validate Terraform">

- **Tools:** `Bash` for Terraform commands
- **Commands:**
  ```bash
  cd terraform
  terraform init
  terraform validate
  terraform plan -var-file=environments/dev.tfvars
  ```
- **Validation:**
  - No syntax errors
  - All resources defined correctly
  - Plan shows expected resource creation
- **Reasoning:** `terraform plan` is a dry run that shows exactly what will be created. This must be reviewed before any `apply`.

</Invocation 3>

</Phase 5>

---

<Phase 6 title="End-to-End Integration & Validation" purpose="Deploying to AWS, running the full pipeline, and validating against MVP baseline" relevant_modules="all">

<Invocation 1 title="Deploy Infrastructure">

- **Tools:** `Bash` for Terraform and AWS CLI
- **Steps:**
  1. `terraform apply -var-file=environments/dev.tfvars`
  2. Push container images to ECR
  3. Verify all resources created correctly
- **Reasoning:** This is the first real AWS deployment. Document everything that happens, especially any errors or unexpected behavior.

</Invocation 1>

<Invocation 2 title="Run Single-Firm Test">

- **Tools:** `Bash` for AWS CLI
- **Steps:**
  1. Trigger single Batch job for one firm
  2. Monitor CloudWatch logs
  3. Verify S3 output
  4. Compare to local test results
- **Reasoning:** Single-firm test validates the map phase in isolation before running at scale. Much cheaper and faster to debug.

</Invocation 2>

<Invocation 3 title="Run Full Pipeline">

- **Tools:** `Bash` for AWS CLI, Step Functions console
- **Steps:**
  1. Trigger Step Functions workflow
  2. Monitor distributed map progress
  3. Monitor reduce phase
  4. Verify DynamoDB output
  5. Query hierarchical data
- **Document:**
  - Total processing time
  - AWS cost (from Cost Explorer)
  - Any errors or retries
- **Reasoning:** Full pipeline test on Q1 2023 data is the ultimate validation. This is what we're delivering.

</Invocation 3>

<Invocation 4 title="Validate Against MVP Baseline">

- **Tools:** `Read` MVP results, `Bash` for queries
- **Comparisons:**
  - Number of themes identified
  - Theme keyword quality
  - Firm coverage percentage
  - Topic coherence scores (if available)
- **Output:** Validation report comparing cloud results to MVP
- **Reasoning:** The cloud pipeline must produce results at least as good as the MVP. Any degradation requires investigation.

</Invocation 4>

<Invocation 5 title="Final Documentation">

- **Tools:** `Write` for documentation files
- **Deliverables:**
  - `docs/pipeline_runbook.md` - How to run the pipeline
  - `docs/architecture_diagram.md` - Final architecture with actual resources
  - `docs/cost_analysis.md` - Actual vs estimated costs
  - Updated `CLAUDE.md` with new commands and structure
- **Reasoning:** Documentation is a required deliverable. The researcher explicitly wants faculty and team members to be able to understand and audit the work.

</Invocation 5>

</Phase 6>

</Objective>

---

**Important Notes for AI Engineer Execution:**

1. **Respect Halting Points:** Do not proceed past a phase boundary without explicit approval. The researcher wants to validate each phase.

2. **TDD is Mandatory:** Write tests before implementation. No exceptions.

3. **Document the "Why":** Every file should have comments explaining decisions, not just what the code does.

4. **Local First:** Always test locally before deploying to AWS. Use mocks for AWS services.

5. **Cost Awareness:** Monitor AWS costs during testing. Use spot instances and clean up resources.

6. **MVP is Reference Only:** Do not copy code patterns from `Local_BERTopic_MVP`. Extract ideas and data flow, then implement cleanly.

7. **Simplicity:** If a simpler solution works, use it. Complexity is a liability.
