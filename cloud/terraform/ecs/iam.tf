# Financial Topic Modeling - ECS IAM Roles for vLLM
# Two roles: execution role (ECR pull, logs) and task role (application permissions)
#
# When use_precreated_roles=true: uses precreated role ARNs
# When use_precreated_roles=false: creates roles (original behavior)

# -----------------------------------------------------------------------------
# LOCALS - Resolve to either precreated or self-managed role ARNs
# Note: We construct ARNs directly instead of using data sources because
# the research account doesn't have iam:GetRole permission.
# -----------------------------------------------------------------------------
locals {
  account_id                 = data.aws_caller_identity.current.account_id
  ecs_execution_role_arn     = var.use_precreated_roles ? "arn:aws:iam::${local.account_id}:role/${var.precreated_ecs_execution_role_name}" : aws_iam_role.ecs_execution[0].arn
  ecs_task_role_arn          = var.use_precreated_roles ? "arn:aws:iam::${local.account_id}:role/${var.precreated_ecs_task_role_name}" : aws_iam_role.ecs_task[0].arn
  vllm_instance_profile_arn  = var.use_precreated_roles ? "arn:aws:iam::${local.account_id}:instance-profile/${var.precreated_ecs_instance_profile_name}" : aws_iam_instance_profile.vllm_instance[0].arn
}

# =============================================================================
# SELF-MANAGED ROLES (only created when use_precreated_roles=false)
# =============================================================================

# -----------------------------------------------------------------------------
# ECS EXECUTION ROLE - Used by ECS to pull images and write logs
# -----------------------------------------------------------------------------
resource "aws_iam_role" "ecs_execution" {
  count = var.use_precreated_roles ? 0 : 1
  name  = "ftm-vllm-execution"

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
    Name = "ftm-vllm-execution"
  }
}

# Basic ECS execution role policy (ECR, CloudWatch)
resource "aws_iam_role_policy_attachment" "ecs_execution_policy" {
  count      = var.use_precreated_roles ? 0 : 1
  role       = aws_iam_role.ecs_execution[0].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# -----------------------------------------------------------------------------
# ECS TASK ROLE - Used by running containers (HuggingFace model downloads)
# -----------------------------------------------------------------------------
resource "aws_iam_role" "ecs_task" {
  count = var.use_precreated_roles ? 0 : 1
  name  = "ftm-vllm-task"

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
    Name = "ftm-vllm-task"
  }
}

# Note: vLLM downloads models from HuggingFace (public), no special AWS permissions needed
# If using private S3-hosted models in future, add S3 read permissions here
