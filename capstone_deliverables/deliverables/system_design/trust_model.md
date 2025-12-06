# Trust Model

## Overview

This document describes the trust boundaries, data flows, and security controls in the Financial Topic Modeling pipeline.

---

## Trust Boundary Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRUST BOUNDARY: AWS Account                        │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    TRUST BOUNDARY: VPC                                 │ │
│  │                                                                        │ │
│  │   ┌──────────────────────┐        ┌──────────────────────────────┐   │ │
│  │   │  EC2 Security Group  │        │   RDS Security Group         │   │ │
│  │   │  ┌────────────────┐  │        │   ┌────────────────────┐     │   │ │
│  │   │  │ g4dn.2xlarge   │──┼────────┼──▶│ PostgreSQL 15      │     │   │ │
│  │   │  │ (compute)      │  │  5432  │   │ (data storage)     │     │   │ │
│  │   │  └────────────────┘  │        │   └────────────────────┘     │   │ │
│  │   │         │            │        │                               │   │ │
│  │   │         │ IAM Role   │        │   Accepts ONLY from           │   │ │
│  │   │         ▼            │        │   EC2 Security Group          │   │ │
│  │   └──────────────────────┘        └──────────────────────────────┘   │ │
│  │              │                                                        │ │
│  │              │ S3 Read                                                │ │
│  │              ▼                                                        │ │
│  │   ┌──────────────────────┐                                           │ │
│  │   │  S3 Bucket           │                                           │ │
│  │   │  (code + data)       │                                           │ │
│  │   │  Block Public Access │                                           │ │
│  │   └──────────────────────┘                                           │ │
│  │                                                                        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
          │                                    │
          │ SSH (Port 22)                      │ HTTPS (Port 443)
          │ Developer IP Only                  │ Outbound Only
          ▼                                    ▼
┌──────────────────┐                 ┌──────────────────┐
│ Developer        │                 │ xAI API          │
│ Workstation      │                 │ (LLM Service)    │
│                  │                 │                  │
│ - Terraform      │                 │ - Topic Summary  │
│ - SSH Access     │                 │ - Theme Naming   │
│ - DBeaver        │                 │                  │
└──────────────────┘                 └──────────────────┘
```

---

## Data Classification

| Data Type            | Classification | Storage         | Sensitivity          |
| -------------------- | -------------- | --------------- | -------------------- |
| Earnings Transcripts | Public         | S3, RDS         | Low (already public) |
| Sentence Embeddings  | Derived        | RDS (pgvector)  | Low                  |
| Topic Keywords       | Derived        | RDS             | Low                  |
| LLM Summaries        | Derived        | RDS             | Low                  |
| Theme Descriptions   | Derived        | RDS             | Low                  |
| AWS Credentials      | Secret         | Environment     | High                 |
| Database Password    | Secret         | Terraform State | High                 |
| xAI API Key          | Secret         | Environment     | High                 |

---

## Trust Relationships

### 1. Developer → AWS Account

- **Authentication**: AWS IAM credentials
- **Authorization**: IAM policies (admin for Terraform)
- **Controls**: MFA recommended, credential rotation

### 2. EC2 → S3

- **Authentication**: IAM Instance Profile
- **Authorization**: `AmazonS3ReadOnlyAccess` policy
- **Controls**: No credentials stored on instance

### 3. EC2 → RDS

- **Authentication**: Username/password
- **Authorization**: Database user with full access
- **Controls**: Security group restricts to EC2 only

### 4. EC2 → xAI API

- **Authentication**: API key (bearer token)
- **Authorization**: API key permissions
- **Controls**: HTTPS only, key in environment variable

### 5. Developer → EC2

- **Authentication**: SSH key pair
- **Authorization**: Security group allows developer IP only
- **Controls**: No password authentication

### 6. Developer → RDS (Optional)

- **Authentication**: Database credentials
- **Authorization**: Via SSH tunnel through EC2
- **Controls**: No direct internet access to RDS

---

## Security Controls

### Network Security

| Control       | Implementation                     |
| ------------- | ---------------------------------- |
| VPC Isolation | Default VPC with security groups   |
| EC2 Ingress   | SSH (22) from developer IP only    |
| EC2 Egress    | All traffic (required for S3, xAI) |
| RDS Ingress   | PostgreSQL (5432) from EC2 SG only |
| RDS Egress    | None required                      |
| S3 Access     | Block all public access            |

### Identity & Access Management

| Principal         | Access Level | Scope            |
| ----------------- | ------------ | ---------------- |
| Developer         | Admin        | Full AWS account |
| EC2 Instance Role | Read-only    | S3 bucket        |
| RDS Master User   | Full         | ftm database     |

### Data Protection

| Layer                | Control                     |
| -------------------- | --------------------------- |
| Data at Rest (S3)    | SSE-S3 encryption (default) |
| Data at Rest (RDS)   | Encrypted storage           |
| Data in Transit      | HTTPS/TLS for all APIs      |
| Data in Transit (DB) | SSL connection (optional)   |

### Secrets Management

| Secret          | Storage                  | Access Pattern     |
| --------------- | ------------------------ | ------------------ |
| AWS Credentials | Local `.aws/credentials` | AWS CLI/SDK        |
| DB Password     | `TF_VAR_db_password` env | Terraform, scripts |
| xAI API Key     | `.env` file              | Pipeline script    |

**Note**: For production, migrate to AWS Secrets Manager.

---

## Threat Model

### Threat 1: Unauthorized Access to RDS

- **Mitigation**: Security group restricts to EC2 only
- **Residual Risk**: Compromised EC2 could access DB
- **Detection**: CloudTrail, RDS logs

### Threat 2: API Key Exposure

- **Mitigation**: Keys in environment, not code
- **Residual Risk**: Keys in EC2 user-data (encrypted in transit)
- **Detection**: xAI usage monitoring

### Threat 3: Spot Instance Termination

- **Mitigation**: Checkpoint/resume pattern
- **Residual Risk**: Minimal data loss (current firm)
- **Detection**: Not a security threat

### Threat 4: S3 Bucket Exposure

- **Mitigation**: Block all public access
- **Residual Risk**: IAM misconfiguration
- **Detection**: S3 access logs

### Threat 5: SSH Key Compromise

- **Mitigation**: IP restriction in security group
- **Residual Risk**: Developer IP spoofing (unlikely)
- **Detection**: CloudTrail, EC2 logs

---

## Data Flow Analysis

### Input Data (Transcripts)

```
Source: WRDS (academic) or Local CSV
Classification: Public (SEC filings)
Processing: Sentence splitting, embedding
No PII handling required
```

### LLM Data Flow

```
Sent to xAI:
- Topic keywords (derived from public data)
- Sentence counts (metadata)

NOT sent to xAI:
- Raw transcript text
- Company identifiers in prompts
- Any PII

Response from xAI:
- Topic summaries (1-2 sentences)
- Theme descriptions
```

### Output Data (Themes)

```
Storage: PostgreSQL (RDS)
Access: Developer only (SSH tunnel)
Retention: Indefinite (academic research)
Sharing: None (internal research)
```

---

## Compliance Considerations

### Academic Use

- Data from public SEC filings
- No GDPR/CCPA scope (no personal data)
- WRDS terms of service compliance

### AWS Shared Responsibility

- AWS: Physical security, hypervisor
- Customer: Network config, access control, data encryption

### API Terms

- xAI: Standard API terms
- No training on customer data (per xAI policy)

---

## Recommendations for Production

1. **Secrets Manager**: Migrate all secrets to AWS Secrets Manager
2. **VPC Endpoints**: Add S3 VPC endpoint to eliminate internet transit
3. **IAM Policies**: Scope down from `S3ReadOnlyAccess` to specific bucket
4. **SSL/TLS**: Enforce SSL connections to RDS
5. **Audit Logging**: Enable CloudTrail, RDS audit logs
6. **Encryption**: Enable customer-managed KMS keys
