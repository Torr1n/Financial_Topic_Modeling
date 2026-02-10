# Financial Topic Modeling - Batch Job Definition

# -----------------------------------------------------------------------------
# CLOUDWATCH LOG GROUP - For job logs
# -----------------------------------------------------------------------------
resource "aws_cloudwatch_log_group" "batch" {
  name              = "/aws/batch/ftm"
  retention_in_days = 14

  tags = {
    Name    = "ftm-batch-logs"
    Project = "financial-topic-modeling"
  }
}

# -----------------------------------------------------------------------------
# JOB DEFINITION - Firm processor container
# -----------------------------------------------------------------------------
resource "aws_batch_job_definition" "firm_processor" {
  name = "ftm-firm-processor"
  type = "container"

  platform_capabilities = ["EC2"]

  # Fail early if external images are enabled but URI is empty
  lifecycle {
    precondition {
      condition     = !var.use_external_images || length(var.map_image_uri) > 0
      error_message = "map_image_uri must be set when use_external_images=true. Provide the full ECR image URI (e.g., 015705018204.dkr.ecr.us-west-2.amazonaws.com/ftm-map:latest)."
    }
  }

  container_properties = jsonencode({
    image = local.map_image

    resourceRequirements = [
      { type = "VCPU", value = "4" },
      { type = "MEMORY", value = "14000" },
      { type = "GPU", value = "1" }
    ]

    # Static environment variables (job-specific vars passed at submission)
    # DATA_SOURCE=s3 for production - uses prefetch data, no WRDS/MFA needed
    # Override with DATA_SOURCE=wrds for local development if needed
    # LLM_BASE_URL is injected when vLLM is enabled (from SSM parameter)
    environment = concat(
      [
        { name = "S3_BUCKET", value = var.s3_bucket_name },
        { name = "CHECKPOINT_INTERVAL", value = tostring(var.checkpoint_interval) },
        { name = "DATA_SOURCE", value = "s3" }
      ],
      var.enable_vllm ? [
        { name = "LLM_BASE_URL", value = data.aws_ssm_parameter.vllm_base_url[0].value },
        { name = "LLM_MODEL_NAME", value = var.vllm_model },
        { name = "LLM_MAX_CONCURRENT", value = tostring(var.llm_max_concurrent) }
      ] : []
    )

    # WRDS credentials injected from Secrets Manager (gated by enable_wrds_secrets)
    secrets = var.enable_wrds_secrets ? [
      {
        name      = "WRDS_USERNAME"
        valueFrom = "${data.aws_secretsmanager_secret.wrds[0].arn}:username::"
      },
      {
        name      = "WRDS_PASSWORD"
        valueFrom = "${data.aws_secretsmanager_secret.wrds[0].arn}:password::"
      }
    ] : []

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.batch.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "firm-processor"
      }
    }

    executionRoleArn = local.batch_execution_role_arn
    jobRoleArn       = local.batch_job_role_arn
  })

  retry_strategy {
    attempts = 3

    evaluate_on_exit {
      action           = "RETRY"
      on_status_reason = "Host EC2*" # Spot interruption
    }
    evaluate_on_exit {
      action    = "RETRY"
      on_reason = "CannotInspectContainerError*" # Docker issues
    }
    evaluate_on_exit {
      action       = "EXIT"
      on_exit_code = "1" # Application error - don't retry
    }
  }

  timeout {
    attempt_duration_seconds = var.job_timeout_seconds
  }

  tags = {
    Name    = "ftm-firm-processor"
    Project = "financial-topic-modeling"
  }
}

# -----------------------------------------------------------------------------
# JOB DEFINITION - Theme aggregator (reduce phase)
# -----------------------------------------------------------------------------
resource "aws_batch_job_definition" "theme_aggregator" {
  name = "ftm-theme-aggregator"
  type = "container"

  platform_capabilities = ["EC2"]

  # Fail early if external images are enabled but URI is empty
  lifecycle {
    precondition {
      condition     = !var.use_external_images || length(var.reduce_image_uri) > 0
      error_message = "reduce_image_uri must be set when use_external_images=true. Provide the full ECR image URI (e.g., 015705018204.dkr.ecr.us-west-2.amazonaws.com/ftm-reduce:latest)."
    }
  }

  container_properties = jsonencode({
    image = local.reduce_image

    resourceRequirements = [
      { type = "VCPU", value = "4" },
      { type = "MEMORY", value = "14000" },
      { type = "GPU", value = "1" }
    ]

    # Environment variables for reduce phase
    # QUARTER is passed at job submission via ContainerOverrides
    environment = concat(
      [
        { name = "S3_BUCKET", value = var.s3_bucket_name }
      ],
      var.enable_vllm ? [
        { name = "LLM_BASE_URL", value = data.aws_ssm_parameter.vllm_base_url[0].value },
        { name = "LLM_MODEL_NAME", value = var.vllm_model },
        { name = "LLM_MAX_CONCURRENT", value = tostring(var.llm_max_concurrent) }
      ] : []
    )

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.batch.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "theme-aggregator"
      }
    }

    executionRoleArn = local.batch_execution_role_arn
    jobRoleArn       = local.batch_job_role_arn
  })

  retry_strategy {
    attempts = 2

    evaluate_on_exit {
      action           = "RETRY"
      on_status_reason = "Host EC2*" # Spot interruption
    }
    evaluate_on_exit {
      action       = "EXIT"
      on_exit_code = "1" # Application error - don't retry
    }
  }

  timeout {
    attempt_duration_seconds = 7200  # 2 hours for reduce phase (includes health-aware retry)
  }

  tags = {
    Name    = "ftm-theme-aggregator"
    Project = "financial-topic-modeling"
  }
}
