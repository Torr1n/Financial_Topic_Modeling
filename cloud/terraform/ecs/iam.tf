# Financial Topic Modeling - ECS IAM Roles for vLLM
# Two roles: execution role (ECR pull, logs) and task role (application permissions)

# -----------------------------------------------------------------------------
# ECS EXECUTION ROLE - Used by ECS to pull images and write logs
# -----------------------------------------------------------------------------
resource "aws_iam_role" "ecs_execution" {
  name = "ftm-vllm-execution"

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
    Name    = "ftm-vllm-execution"
    Project = "financial-topic-modeling"
  }
}

# Basic ECS execution role policy (ECR, CloudWatch)
resource "aws_iam_role_policy_attachment" "ecs_execution_policy" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# -----------------------------------------------------------------------------
# ECS TASK ROLE - Used by running containers (HuggingFace model downloads)
# -----------------------------------------------------------------------------
resource "aws_iam_role" "ecs_task" {
  name = "ftm-vllm-task"

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
    Name    = "ftm-vllm-task"
    Project = "financial-topic-modeling"
  }
}

# Note: vLLM downloads models from HuggingFace (public), no special AWS permissions needed
# If using private S3-hosted models in future, add S3 read permissions here
