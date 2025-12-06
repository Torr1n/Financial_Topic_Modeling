# Architecture Diagram

## System Overview

The Financial Topic Modeling Pipeline is a hierarchical NLP system that identifies cross-firm investment themes from earnings call transcripts.

---

## Pipeline Architecture

```mermaid
flowchart TB
    subgraph Stage1["Stage 1: Data Ingestion"]
        CSV["Transcript CSV<br/>(S3 or Local)"]
        Parse["Parse & Clean<br/>Sentences"]
    end

    subgraph Stage2["Stage 2: Embedding"]
        Embed["Sentence Transformer<br/>(all-mpnet-base-v2)"]
        GPU1["GPU Accelerated<br/>768-dim vectors"]
    end

    subgraph Stage3["Stage 3: Firm-Level Clustering"]
        UMAP1["UMAP Reduction<br/>(cuML GPU)"]
        HDBSCAN1["HDBSCAN Clustering<br/>(cuML GPU)"]
        Topics["~25 Topics per Firm"]
    end

    subgraph Stage4["Stage 4: LLM Summarization"]
        LLM["xAI Grok API<br/>(async, 50 concurrent)"]
        Summary["Topic Summaries<br/>(human-readable)"]
    end

    subgraph Stage5["Stage 5: Theme Aggregation"]
        ReEmbed["Re-embed Summaries"]
        UMAP2["UMAP Reduction"]
        HDBSCAN2["HDBSCAN Clustering"]
        Validate["Validation Filters<br/>min_firms=2<br/>max_dominance=0.4"]
        Themes["~19 Cross-Firm Themes"]
    end

    subgraph Stage6["Stage 6: Storage"]
        PG["PostgreSQL<br/>+ pgvector"]
        Index["HNSW Vector Index"]
    end

    CSV --> Parse --> Embed --> GPU1
    GPU1 --> UMAP1 --> HDBSCAN1 --> Topics
    Topics --> LLM --> Summary
    Summary --> ReEmbed --> UMAP2 --> HDBSCAN2 --> Validate --> Themes
    Themes --> PG --> Index
```

---

## Data Flow (Per Firm)

```mermaid
sequenceDiagram
    participant S3 as S3 Bucket
    participant EC2 as EC2 Instance
    participant Model as BERTopic Model
    participant LLM as xAI API
    participant RDS as PostgreSQL

    S3->>EC2: Download transcripts
    EC2->>Model: Load embedding model (ONCE)

    loop For Each Firm
        EC2->>Model: Embed sentences (GPU)
        Model->>Model: UMAP + HDBSCAN clustering
        Model->>EC2: Return topics + keywords
        EC2->>LLM: Generate summary (async)
        LLM->>EC2: Topic summary
        EC2->>RDS: Write firm + sentences + topics
    end

    EC2->>RDS: Read all topics
    EC2->>Model: Re-embed summaries
    Model->>Model: Theme clustering
    EC2->>RDS: Write themes
    EC2->>RDS: Build vector indexes
```

---

## Infrastructure Architecture

```mermaid
flowchart TB
    subgraph AWS["AWS Cloud (us-east-1)"]
        subgraph VPC["Default VPC"]
            subgraph EC2["EC2 Compute"]
                Spot["g4dn.2xlarge<br/>Spot Instance"]
                GPU["NVIDIA T4 GPU"]
                DLAMI["Deep Learning AMI<br/>PyTorch 2.7"]
            end

            subgraph RDS["RDS Database"]
                PG["PostgreSQL 15<br/>db.t4g.large"]
                Vector["pgvector Extension"]
            end

            Spot --> |"Port 5432"| PG
        end

        subgraph S3["S3 Storage"]
            Code["ftm-pipeline.tar.gz<br/>(code)"]
            Data["transcripts.csv<br/>(data)"]
        end

        S3 --> |"Download on boot"| Spot
    end

    subgraph Local["Local Machine"]
        TF["Terraform<br/>(deploy)"]
        SSH["SSH<br/>(monitor)"]
        DBeaver["DBeaver<br/>(query)"]
    end

    TF --> |"terraform apply"| AWS
    SSH --> |"Port 22"| Spot
    DBeaver --> |"Port 5432<br/>(via tunnel)"| PG
```

---

## Security Architecture

```mermaid
flowchart LR
    subgraph Internet
        User["Developer"]
        XAI["xAI API"]
    end

    subgraph AWS
        subgraph SG_EC2["EC2 Security Group"]
            EC2["EC2 Instance"]
        end

        subgraph SG_RDS["RDS Security Group"]
            RDS["PostgreSQL"]
        end

        S3["S3 Bucket<br/>(Private)"]
    end

    User -->|"SSH:22<br/>Your IP only"| EC2
    EC2 -->|"HTTPS:443"| XAI
    EC2 -->|"PostgreSQL:5432"| RDS
    EC2 -->|"IAM Role"| S3

    style SG_RDS fill:#f9f,stroke:#333
    style SG_EC2 fill:#bbf,stroke:#333
```

**Security Controls:**

- EC2: SSH restricted to single IP (developer workstation)
- RDS: Accepts connections only from EC2 security group
- S3: Block all public access, IAM role-based access
- LLM: Outbound HTTPS only, no PII transmitted

---

## Database Schema (ERD)

```mermaid
erDiagram
    FIRMS ||--o{ SENTENCES : contains
    FIRMS ||--o{ TOPICS : generates
    TOPICS }o--|| THEMES : aggregates_to
    TOPICS ||--o{ SENTENCES : assigns

    FIRMS {
        int id PK
        string company_id UK
        string name
        timestamp processed_at
    }

    SENTENCES {
        int id PK
        int firm_id FK
        int topic_id FK
        text raw_text
        text cleaned_text
        vector embedding "768-dim"
        float probability
    }

    TOPICS {
        int id PK
        int firm_id FK
        int theme_id FK
        jsonb representation
        text summary
        vector embedding "768-dim"
        int sentence_count
    }

    THEMES {
        int id PK
        string name
        text description
        vector embedding "768-dim"
        int n_topics
        int n_firms
    }
```

---

## Processing Timeline (MAG7 Run)

```mermaid
gantt
    title Pipeline Execution Timeline (11 Firms)
    dateFormat mm:ss
    axisFormat %M:%S

    section Setup
    EC2 Boot + Download     :setup, 00:00, 3m
    Load Embedding Model    :load, after setup, 1m

    section Firm Processing
    Embed Sentences         :embed, after load, 4m
    Topic Clustering        :cluster, after embed, 2m
    LLM Summarization       :llm, after cluster, 2m

    section Aggregation
    Theme Clustering        :theme, after llm, 2m
    Build Vector Index      :index, after theme, 1m
```

**Total: ~15 minutes**

---

## Comparison: Original vs Final Architecture

### Original Design (AWS Batch)

```mermaid
flowchart LR
    S3_In["S3 Input"]
    Batch["AWS Batch<br/>(3,000 containers)"]
    S3_Mid["S3 Intermediate"]
    Reduce["Reduce Container"]
    DDB["DynamoDB"]

    S3_In --> Batch
    Batch --> S3_Mid
    S3_Mid --> Reduce
    Reduce --> DDB
```

**Problems:**

- 3,000 cold starts (60s each)
- Model loaded 3,000 times
- S3 round-trips for intermediate data
- DynamoDB N+1 query issues

### Final Design (Single GPU)

```mermaid
flowchart LR
    S3["S3 Input"]
    EC2["Single EC2<br/>(GPU)"]
    PG["PostgreSQL"]

    S3 --> EC2 --> PG
```

**Benefits:**

- 1 model load
- No intermediate storage
- Relational queries
- 90% cost reduction

---

## Key Metrics

| Component            | Specification                    |
| -------------------- | -------------------------------- |
| Embedding Model      | all-mpnet-base-v2 (768-dim)      |
| GPU                  | NVIDIA T4 (16GB VRAM)            |
| Database             | PostgreSQL 15 + pgvector         |
| Vector Index         | HNSW (ef_construction=128, m=16) |
| LLM                  | xAI grok-4-1-fast-reasoning      |
| Concurrent LLM Calls | 50 (semaphore-limited)           |

---

_Source: [docs/ARCHITECTURE.md](../../../docs/ARCHITECTURE.md)_
