# Financial Topic Modeling - ECS Cluster, Task Definition, and Service for vLLM

# -----------------------------------------------------------------------------
# CLOUDWATCH LOG GROUP - For vLLM container logs
# -----------------------------------------------------------------------------
resource "aws_cloudwatch_log_group" "vllm" {
  name              = "/ecs/ftm-vllm"
  retention_in_days = 14

  tags = {
    Name    = "ftm-vllm-logs"
    Project = "financial-topic-modeling"
  }
}

# -----------------------------------------------------------------------------
# ECS CLUSTER - vLLM cluster with GPU capacity provider
# -----------------------------------------------------------------------------
resource "aws_ecs_cluster" "vllm" {
  name = "ftm-vllm-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = {
    Name    = "ftm-vllm-cluster"
    Project = "financial-topic-modeling"
  }
}

# -----------------------------------------------------------------------------
# TASK DEFINITION - vLLM container with GPU
# -----------------------------------------------------------------------------
resource "aws_ecs_task_definition" "vllm" {
  family                   = "ftm-vllm"
  requires_compatibilities = ["EC2"]
  network_mode             = "awsvpc"
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  # GPU requirement
  cpu    = var.container_cpu
  memory = var.container_memory

  container_definitions = jsonencode([
    {
      name      = "vllm"
      image     = "vllm/vllm-openai:latest"
      essential = true

      # vLLM command with model configuration
      command = [
        "--model", var.vllm_model,
        "--port", tostring(var.vllm_port),
        "--trust-remote-code",
        "--max-model-len", "4096"
      ]

      portMappings = [
        {
          containerPort = var.vllm_port
          hostPort      = var.vllm_port
          protocol      = "tcp"
        }
      ]

      # GPU resource requirement
      resourceRequirements = [
        {
          type  = "GPU"
          value = "1"
        }
      ]

      # Shared memory for PyTorch (required for large models)
      linuxParameters = {
        sharedMemorySize = var.shared_memory_size
      }

      # Environment variables
      environment = [
        {
          name  = "VLLM_WORKER_MULTIPROC_METHOD"
          value = "spawn"
        }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.vllm.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "vllm"
        }
      }

      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:${var.vllm_port}/health || exit 1"]
        interval    = 30
        timeout     = 10
        retries     = 3
        startPeriod = var.health_check_startup_period
      }
    }
  ])

  tags = {
    Name    = "ftm-vllm-task"
    Project = "financial-topic-modeling"
  }
}

# -----------------------------------------------------------------------------
# ECS SERVICE - vLLM service with ALB integration
# -----------------------------------------------------------------------------
resource "aws_ecs_service" "vllm" {
  name            = "ftm-vllm-service"
  cluster         = aws_ecs_cluster.vllm.id
  task_definition = aws_ecs_task_definition.vllm.arn
  desired_count   = var.min_capacity

  # Use EC2 launch type for GPU support
  launch_type = "EC2"

  network_configuration {
    subnets         = data.aws_subnets.default.ids
    security_groups = [aws_security_group.vllm_task.id]
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.vllm.arn
    container_name   = "vllm"
    container_port   = var.vllm_port
  }

  # Health check grace period for model loading
  health_check_grace_period_seconds = var.health_check_startup_period

  # Allow task to be replaced when task definition changes
  force_new_deployment = true

  deployment_configuration {
    maximum_percent         = 200
    minimum_healthy_percent = 50
  }

  # Wait for ALB target group before starting service
  depends_on = [
    aws_lb_listener.vllm
  ]

  tags = {
    Name    = "ftm-vllm-service"
    Project = "financial-topic-modeling"
  }
}
