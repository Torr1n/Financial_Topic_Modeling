# AI Collaboration Log

## Overview

This document records how AI assistants contributed to the Financial Topic Modeling project, following the Socratic Log format. It captures key prompts, inflection points, and decisions—not full transcripts.

---

## AI Tools Used

| Tool             | Role                    | Contribution                                |
| ---------------- | ----------------------- | ------------------------------------------- |
| **Gemini 3 Pro** | Architecture Consultant | Cloud architecture reasoning, cost analysis |
| **Codex**        | Code Reviewer           | Security review, best practices             |
| **Claude Code**  | Implementation          | Code writing, documentation                 |

---

## Session 1: Architecture Pivot Discovery

**AI**: Gemini 3 Pro
**Context**: Planning cloud migration, debating AWS Batch vs alternatives

### Prompt A: Design Alternatives

> "My main use case is large-scale topic modeling for firms earnings calls (using BERTopic), where I have a container defined to process 1 firm's earnings call. Each one of these earnings calls would be MAXIMUM 1000 sentences with a typical range of 250-500 sentences. Give me a comparison and tradeoffs analysis."

**Gemini Response** (summarized):

- Identified "Container Cold Start" problem
- Showed CPU wins at N=500 sentences due to 60-second billing minimum
- Introduced "Overhead Beast" concept

### Prompt B: Red-Team (Challenging My Assumption)

> "If I were to also intend to extend these firm topic-modelling steps to involve an LLM call per topic... would this change anything?"

**Gemini Response**:

- Async LLM calls enable "pipelining" on single instance
- GPU processes Firm B while CPU awaits LLM response for Firm A
- Single instance still superior

### Inflection Point

**What changed my thinking**: The realization that model loading time (~5 seconds × 3,000 containers = 4+ hours wasted) dwarfs the actual inference time. Amdahl's Law applies directly: the sequential portion (startup) dominates.

**Quote that crystallized the insight**:

> "For a quarterly job of this size, distributed computing (containers/AWS Batch) introduces more problems than it solves. Best Approach: Use a single g4dn.xlarge Spot instance."

### What I Tested

- Created cost comparison spreadsheet
- Calculated: 3,000 containers × $0.003 = $9 vs 1 instance × 4 hrs × $0.25 = $1
- Validated with MAG7 test run

### Outcome

**Kept**: Single GPU instance architecture
**Discarded**: AWS Batch distributed design
**Modified**: Added checkpoint/resume for spot instance resilience

---

## Session 2: Database Architecture

**AI**: Gemini 3
**Context**: Choosing between DynamoDB and PostgreSQL

### Prompt A: Design Alternatives

> "Is DynamoDB the correct choice for storing our results given the data type... we want to be able to trace from themes → topics → firms → sentences?"

**Gemini Response** (summarized):

- Identified "N+1 Query Hell" problem in DynamoDB
- Recommended PostgreSQL for hierarchical data
- Introduced pgvector for semantic search

### Red-Team Prompt

> "What about Aurora Serverless v2 for zero-cost when idle?"

**Gemini Response**:

- Aurora Serverless doesn't scale to zero
- Minimum ~$45/month even when idle
- Standard RDS with stop/start is cheaper for quarterly batch

### Inflection Point

**What changed my thinking**: DynamoDB's single-table design pattern, while elegant for high-scale key-value, creates query complexity that doesn't match our hierarchical data model.

**Key quote**:

> "Your requirements explicitly describe a highly relational, hierarchical dataset with a need for deep traversal (joins). Using DynamoDB here would force you into 'Single Table Design' gymnastics."

### What I Tested

- Wrote sample queries in both DynamoDB (GSI-based) and PostgreSQL (JOIN-based)
- PostgreSQL query: 5 lines, obvious. DynamoDB: 30+ lines, multiple queries
- pgvector enabled semantic search not possible in DynamoDB

### Outcome

**Kept**: PostgreSQL with pgvector
**Discarded**: DynamoDB single-table design
**Added**: HNSW indexes for vector search

---

## Session 3: Code Review

**AI**: Codex
**Context**: Terraform and deployment script review

### Prompt

> "Review our Terraform configuration and deployment scripts for security issues and best practices."

### Codex Findings

1. **Security Group**: RDS had `0.0.0.0/0` on port 5432 (blocker)
2. **IAM**: Missing instance profile for EC2
3. **Region**: Hardcoded us-east-1 in scripts
4. **AMI**: Hardcoded AMI ID (will become stale)
5. **Secrets**: DB password in multiple places

### What I Changed

- Scoped RDS security group to EC2 SG only
- Added `aws_iam_instance_profile` resource
- Parameterized region from Terraform outputs
- Added dynamic AMI lookup via `describe-images`
- Single source of truth for DB password (Terraform output)

### Outcome

All Codex blockers resolved. Security posture improved.

---

## Session 4: Implementation

**AI**: Claude Code
**Context**: Implementation of unified pipeline and documentation

### Contributions

| Component           | AI Contribution           | My Review                     |
| ------------------- | ------------------------- | ----------------------------- |
| Terraform main.tf   | Full implementation       | Reviewed security             |
| Deployment scripts  | Full implementation       | Tested on AWS                 |
| unified_pipeline.py | Architecture design       | Integrated with existing code |
| production.yaml     | Configuration structure   | Validated values from MVP     |
| Documentation       | All capstone deliverables | Reviewed for accuracy         |

### Prompt Pattern

Prompts followed a pattern of:

1. Context from previous sessions
2. Specific task description
3. Constraints (simplicity, cost, no over-engineering)

### Example Prompt

> "Create the ADR for the single GPU vs AWS Batch decision. Include key quotes from the Gemini consultation and the economic analysis."

### Quality Control

- All generated code was run and tested
- Documentation cross-referenced with actual implementation
- Numbers validated against real AWS bills

---

## Key Decisions Made with AI Assistance

| Decision                  | AI Tool | Keep/Modify/Discard    |
| ------------------------- | ------- | ---------------------- |
| Single GPU over AWS Batch | Gemini  | Keep                   |
| PostgreSQL over DynamoDB  | Gemini  | Keep                   |
| pgvector for embeddings   | Gemini  | Keep                   |
| Spot instances            | Gemini  | Keep                   |
| Checkpoint per firm       | Gemini  | Keep                   |
| Security group scoping    | Codex   | Modified (implemented) |
| Dynamic AMI lookup        | Codex   | Modified (implemented) |
| asyncio for LLM calls     | Gemini  | Keep                   |
| Theme validation filters  | None    | N/A (my decision)      |

---

## What AI Did NOT Do

- **Final architecture decision**: I evaluated AI recommendations and chose
- **Hyperparameter tuning**: Values from MVP experiments, not AI suggestions
- **Data analysis**: Theme interpretation requires domain expertise
- **Testing in production**: I ran and validated all AWS deployments
- **Cost approval**: I approved all spending decisions

---

## Lessons Learned

### On AI Collaboration

1. **Specify personas**: "Impartial cloud architect" got better analysis than generic prompts
2. **Challenge assumptions**: Red-team prompts revealed blind spots
3. **Ask for tradeoffs**: Not just "what should I do" but "what are the tradeoffs"
4. **Verify numbers**: AI cost estimates were directionally correct but needed validation

### On Code Generation

1. **Context matters**: Providing existing code structure improved generation quality
2. **Iterate**: First drafts needed refinement
3. **Test everything**: Generated code compiled but sometimes had logical issues
4. **Document why**: AI helped document "why" better than "what"

---

## Source Documents

Full AI conversations are available in:

| Document                             | Content                                      |
| ------------------------------------ | -------------------------------------------- |
| `docs/ai-log/gemini-conversation.md` | Full architecture consultation (~1000 lines) |
| `docs/ai-log/codex-consultation.md`  | Code review findings                         |
| `docs/packages/*/`                   | AI-generated planning documents              |
| `docs/handovers/`                    | AI-generated handover notes                  |

---

## Attribution

Per course requirements, all substantial AI contributions are documented. Code and documentation were generated with AI assistance and reviewed by the project lead.
