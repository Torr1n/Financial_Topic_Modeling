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

  container_properties = jsonencode({
    image = "${aws_ecr_repository.map.repository_url}:latest"

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
        { name = "LLM_BASE_URL", value = data.aws_ssm_parameter.vllm_base_url[0].value }
      ] : []
    )

    # WRDS credentials injected from Secrets Manager
    secrets = [
      {
        name      = "WRDS_USERNAME"
        valueFrom = "${data.aws_secretsmanager_secret.wrds.arn}:username::"
      },
      {
        name      = "WRDS_PASSWORD"
        valueFrom = "${data.aws_secretsmanager_secret.wrds.arn}:password::"
      }
    ]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.batch.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "firm-processor"
      }
    }

    executionRoleArn = aws_iam_role.batch_execution.arn
    jobRoleArn       = aws_iam_role.batch_job.arn
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
