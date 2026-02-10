# Financial Topic Modeling - Phase 4 Infrastructure
# Minimal setup: RDS PostgreSQL + S3 + Security Groups + IAM
# Total: ~130 lines (honoring simplicity constraint)

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Owner   = "Torrin"
      Project = "financial-topic-modeling"
    }
  }
}

# Random suffix for globally unique S3 bucket name
resource "random_id" "suffix" {
  byte_length = 4
}

# -----------------------------------------------------------------------------
# S3 BUCKET - Code and data delivery to EC2
# -----------------------------------------------------------------------------
resource "aws_s3_bucket" "pipeline" {
  bucket = "ftm-pipeline-${random_id.suffix.hex}"

  tags = {
    Name    = "ftm-pipeline"
    Project = "financial-topic-modeling"
  }
}

# Block public access (code bucket should be private)
resource "aws_s3_bucket_public_access_block" "pipeline" {
  bucket = aws_s3_bucket.pipeline.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# -----------------------------------------------------------------------------
# SECURITY GROUPS
# -----------------------------------------------------------------------------

# EC2 Security Group - SSH access from your IP only
resource "aws_security_group" "ec2" {
  name        = "ftm-ec2-sg"
  description = "Allow SSH from specified IP"

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.my_ip]
  }

  egress {
    description = "All outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "ftm-ec2-sg"
    Project = "financial-topic-modeling"
  }
}

# RDS Security Group - PostgreSQL access from EC2 only (NOT public)
resource "aws_security_group" "db" {
  name        = "ftm-db-sg"
  description = "Allow PostgreSQL from EC2 security group only"

  ingress {
    description     = "PostgreSQL from EC2"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.ec2.id]
  }

  tags = {
    Name    = "ftm-db-sg"
    Project = "financial-topic-modeling"
  }
}

# -----------------------------------------------------------------------------
# RDS POSTGRESQL - Main database with pgvector support
# -----------------------------------------------------------------------------
resource "aws_db_instance" "main" {
  identifier = "ftm-db"

  # Engine
  engine         = "postgres"
  engine_version = "15"

  # Instance size (stoppable - NOT Aurora Serverless)
  instance_class    = "db.t4g.large" # 8GB RAM, ARM-based (cost-effective)
  allocated_storage = 100            # GB, enough for embeddings

  # Database
  db_name  = "ftm"
  username = var.db_username
  password = var.db_password

  # Network - publicly accessible but restricted by security group
  publicly_accessible    = true
  vpc_security_group_ids = [aws_security_group.db.id]

  # Maintenance
  skip_final_snapshot = true # Dev environment
  apply_immediately   = true

  tags = {
    Name    = "ftm-db"
    Project = "financial-topic-modeling"
  }
}

# -----------------------------------------------------------------------------
# IAM ROLE & INSTANCE PROFILE - EC2 access to S3
# -----------------------------------------------------------------------------
resource "aws_iam_role" "ec2_pipeline" {
  name = "ftm-ec2-pipeline"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ec2.amazonaws.com"
      }
    }]
  })

  tags = {
    Project = "financial-topic-modeling"
  }
}

# Attach S3 read-only policy for code download
resource "aws_iam_role_policy_attachment" "s3_read" {
  role       = aws_iam_role.ec2_pipeline.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess"
}

# Instance profile (required for EC2 to use IAM role)
resource "aws_iam_instance_profile" "ec2_profile" {
  name = "ftm-ec2-profile"
  role = aws_iam_role.ec2_pipeline.name
}
