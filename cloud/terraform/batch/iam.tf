# Financial Topic Modeling - Batch IAM Roles
# Five roles: batch_service, batch_execution, batch_job, batch_instance, spot_fleet

# -----------------------------------------------------------------------------
# BATCH SERVICE ROLE - Used by AWS Batch to manage compute resources
# -----------------------------------------------------------------------------
resource "aws_iam_role" "batch_service" {
  name = "ftm-batch-service"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "batch.amazonaws.com"
      }
    }]
  })

  tags = {
    Name    = "ftm-batch-service"
    Project = "financial-topic-modeling"
  }
}

resource "aws_iam_role_policy_attachment" "batch_service" {
  role       = aws_iam_role.batch_service.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSBatchServiceRole"
}

# -----------------------------------------------------------------------------
# BATCH EXECUTION ROLE - ECS task execution (ECR pull, Secrets Manager, CloudWatch)
# -----------------------------------------------------------------------------
resource "aws_iam_role" "batch_execution" {
  name = "ftm-batch-execution"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ecs-tasks.amazonaws.com"
      }
    }]
  })

  tags = {
    Name    = "ftm-batch-execution"
    Project = "financial-topic-modeling"
  }
}

# Basic ECS execution role policy (ECR, CloudWatch)
resource "aws_iam_role_policy_attachment" "batch_execution_ecs" {
  role       = aws_iam_role.batch_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# Secrets Manager access for WRDS credentials
# Note: kms:Decrypt included for customer-managed KMS keys.
# If using AWS-managed key (default), kms:Decrypt is not required but harmless.
resource "aws_iam_role_policy" "batch_execution_secrets" {
  name = "ftm-batch-execution-secrets"
  role = aws_iam_role.batch_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = [data.aws_secretsmanager_secret.wrds.arn]
      },
      {
        Effect = "Allow"
        Action = [
          "kms:Decrypt"
        ]
        Resource = ["*"]
        Condition = {
          StringEquals = {
            "kms:ViaService" = "secretsmanager.${data.aws_region.current.name}.amazonaws.com"
          }
        }
      }
    ]
  })
}

# -----------------------------------------------------------------------------
# BATCH JOB ROLE - Used by running containers (S3 read/write)
# -----------------------------------------------------------------------------
resource "aws_iam_role" "batch_job" {
  name = "ftm-batch-job"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ecs-tasks.amazonaws.com"
      }
    }]
  })

  tags = {
    Name    = "ftm-batch-job"
    Project = "financial-topic-modeling"
  }
}

# S3 access for manifests, checkpoints, prefetch data, and output
resource "aws_iam_role_policy" "batch_job_s3" {
  name = "ftm-batch-job-s3"
  role = aws_iam_role.batch_job.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Resource = [
          "${data.aws_s3_bucket.pipeline.arn}/manifests/*",
          "${data.aws_s3_bucket.pipeline.arn}/progress/*",
          "${data.aws_s3_bucket.pipeline.arn}/intermediate/*",
          "${data.aws_s3_bucket.pipeline.arn}/prefetch/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket"
        ]
        Resource = [data.aws_s3_bucket.pipeline.arn]
        Condition = {
          StringLike = {
            "s3:prefix" = [
              "manifests/*",
              "progress/*",
              "intermediate/*",
              "prefetch/*"
            ]
          }
        }
      }
    ]
  })
}

# Secrets Manager access for WRDS credentials (fallback auth in WRDSConnector)
# This allows the job to fetch credentials if env var injection fails or for local testing.
resource "aws_iam_role_policy" "batch_job_secrets" {
  name = "ftm-batch-job-secrets"
  role = aws_iam_role.batch_job.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = [data.aws_secretsmanager_secret.wrds.arn]
      },
      {
        Effect = "Allow"
        Action = [
          "kms:Decrypt"
        ]
        Resource = ["*"]
        Condition = {
          StringEquals = {
            "kms:ViaService" = "secretsmanager.${data.aws_region.current.name}.amazonaws.com"
          }
        }
      }
    ]
  })
}

# -----------------------------------------------------------------------------
# BATCH INSTANCE ROLE - EC2 instances in compute environment
# -----------------------------------------------------------------------------
resource "aws_iam_role" "batch_instance" {
  name = "ftm-batch-instance"

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
    Name    = "ftm-batch-instance"
    Project = "financial-topic-modeling"
  }
}

# ECS container instance policy (required for Batch)
resource "aws_iam_role_policy_attachment" "batch_instance_ecs" {
  role       = aws_iam_role.batch_instance.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}

# Instance profile for EC2 instances
resource "aws_iam_instance_profile" "batch_instance" {
  name = "ftm-batch-instance"
  role = aws_iam_role.batch_instance.name
}

# -----------------------------------------------------------------------------
# SPOT FLEET ROLE - For Spot instance management
# -----------------------------------------------------------------------------
resource "aws_iam_role" "spot_fleet" {
  name = "ftm-spot-fleet"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "spotfleet.amazonaws.com"
      }
    }]
  })

  tags = {
    Name    = "ftm-spot-fleet"
    Project = "financial-topic-modeling"
  }
}

resource "aws_iam_role_policy_attachment" "spot_fleet" {
  role       = aws_iam_role.spot_fleet.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2SpotFleetTaggingRole"
}
