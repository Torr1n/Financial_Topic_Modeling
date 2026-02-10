# Financial Topic Modeling - Batch IAM Roles
# Five roles: batch_service, batch_execution, batch_job, batch_instance, spot_fleet
#
# When use_precreated_roles=true (default for research account):
#   - Skips role creation, uses precreated role ARNs via data sources
# When use_precreated_roles=false (original behavior):
#   - Creates all roles and policies as before

# -----------------------------------------------------------------------------
# LOCALS - Resolve to either precreated or self-managed role ARNs
# Note: We construct ARNs directly instead of using data sources because
# the research account doesn't have iam:GetRole permission.
# -----------------------------------------------------------------------------
locals {
  account_id = data.aws_caller_identity.current.account_id

  batch_execution_role_arn = var.use_precreated_roles ? "arn:aws:iam::${local.account_id}:role/${var.precreated_batch_execution_role_name}" : aws_iam_role.batch_execution[0].arn
  batch_job_role_arn       = var.use_precreated_roles ? "arn:aws:iam::${local.account_id}:role/${var.precreated_batch_job_role_name}" : aws_iam_role.batch_job[0].arn

  # service_role and spot_fleet_role: set to null when using precreated roles.
  # AWS Batch automatically uses its service-linked role (AWSServiceRoleForBatch)
  # when service_role is unset. This avoids needing iam:PassRole for these roles.
  batch_service_role_arn     = var.use_precreated_roles ? null : aws_iam_role.batch_service[0].arn
  spot_fleet_role_arn        = var.use_precreated_roles ? null : aws_iam_role.spot_fleet[0].arn

  # Instance profile: constructed from explicit variable (not inferred from role name)
  batch_instance_profile_arn = var.use_precreated_roles ? "arn:aws:iam::${local.account_id}:instance-profile/${var.precreated_batch_instance_profile_name}" : aws_iam_instance_profile.batch_instance[0].arn
}

# =============================================================================
# SELF-MANAGED ROLES (only created when use_precreated_roles=false)
# =============================================================================

# -----------------------------------------------------------------------------
# BATCH SERVICE ROLE - Used by AWS Batch to manage compute resources
# -----------------------------------------------------------------------------
resource "aws_iam_role" "batch_service" {
  count = var.use_precreated_roles ? 0 : 1
  name  = "ftm-batch-service"

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
    Name = "ftm-batch-service"
  }
}

resource "aws_iam_role_policy_attachment" "batch_service" {
  count      = var.use_precreated_roles ? 0 : 1
  role       = aws_iam_role.batch_service[0].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSBatchServiceRole"
}

# -----------------------------------------------------------------------------
# BATCH EXECUTION ROLE - ECS task execution (ECR pull, Secrets Manager, CloudWatch)
# -----------------------------------------------------------------------------
resource "aws_iam_role" "batch_execution" {
  count = var.use_precreated_roles ? 0 : 1
  name  = "ftm-batch-execution"

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
    Name = "ftm-batch-execution"
  }
}

# Basic ECS execution role policy (ECR, CloudWatch)
resource "aws_iam_role_policy_attachment" "batch_execution_ecs" {
  count      = var.use_precreated_roles ? 0 : 1
  role       = aws_iam_role.batch_execution[0].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# Secrets Manager access for WRDS credentials
resource "aws_iam_role_policy" "batch_execution_secrets" {
  count = var.use_precreated_roles ? 0 : (var.enable_wrds_secrets ? 1 : 0)
  name  = "ftm-batch-execution-secrets"
  role  = aws_iam_role.batch_execution[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = [data.aws_secretsmanager_secret.wrds[0].arn]
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
  count = var.use_precreated_roles ? 0 : 1
  name  = "ftm-batch-job"

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
    Name = "ftm-batch-job"
  }
}

# S3 access for manifests, checkpoints, prefetch data, intermediate, and processed output
resource "aws_iam_role_policy" "batch_job_s3" {
  count = var.use_precreated_roles ? 0 : 1
  name  = "ftm-batch-job-s3"
  role  = aws_iam_role.batch_job[0].id

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
          "${data.aws_s3_bucket.pipeline.arn}/prefetch/*",
          "${data.aws_s3_bucket.pipeline.arn}/processed/*"
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
              "prefetch/*",
              "processed/*"
            ]
          }
        }
      }
    ]
  })
}

# Secrets Manager access for WRDS credentials (fallback auth in WRDSConnector)
resource "aws_iam_role_policy" "batch_job_secrets" {
  count = var.use_precreated_roles ? 0 : (var.enable_wrds_secrets ? 1 : 0)
  name  = "ftm-batch-job-secrets"
  role  = aws_iam_role.batch_job[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = [data.aws_secretsmanager_secret.wrds[0].arn]
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

# SSM Parameter Store access for vLLM configuration
resource "aws_iam_role_policy" "batch_job_ssm" {
  count = var.use_precreated_roles ? 0 : (var.enable_vllm ? 1 : 0)
  name  = "ftm-batch-job-ssm"
  role  = aws_iam_role.batch_job[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ssm:GetParameter",
          "ssm:GetParameters"
        ]
        Resource = [
          "arn:aws:ssm:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:parameter/ftm/vllm/*"
        ]
      }
    ]
  })
}

# -----------------------------------------------------------------------------
# BATCH INSTANCE ROLE - EC2 instances in compute environment
# -----------------------------------------------------------------------------
resource "aws_iam_role" "batch_instance" {
  count = var.use_precreated_roles ? 0 : 1
  name  = "ftm-batch-instance"

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
    Name = "ftm-batch-instance"
  }
}

# ECS container instance policy (required for Batch)
resource "aws_iam_role_policy_attachment" "batch_instance_ecs" {
  count      = var.use_precreated_roles ? 0 : 1
  role       = aws_iam_role.batch_instance[0].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}

# Instance profile for EC2 instances
resource "aws_iam_instance_profile" "batch_instance" {
  count = var.use_precreated_roles ? 0 : 1
  name  = "ftm-batch-instance"
  role  = aws_iam_role.batch_instance[0].name
}

# -----------------------------------------------------------------------------
# SPOT FLEET ROLE - For Spot instance management
# -----------------------------------------------------------------------------
resource "aws_iam_role" "spot_fleet" {
  count = var.use_precreated_roles ? 0 : 1
  name  = "ftm-spot-fleet"

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
    Name = "ftm-spot-fleet"
  }
}

resource "aws_iam_role_policy_attachment" "spot_fleet" {
  count      = var.use_precreated_roles ? 0 : 1
  role       = aws_iam_role.spot_fleet[0].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2SpotFleetTaggingRole"
}
