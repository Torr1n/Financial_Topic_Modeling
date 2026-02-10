# Financial Topic Modeling - Lambda Functions for Step Functions

# -----------------------------------------------------------------------------
# LAMBDA LAYER - Shared dependencies (boto3, etc.)
# Using runtime boto3 (built-in) - no custom layer needed
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# LAMBDA: Prefetch Check
# -----------------------------------------------------------------------------
data "archive_file" "prefetch_check" {
  type        = "zip"
  source_file = "${path.module}/../../src/lambdas/prefetch_check.py"
  output_path = "${path.module}/builds/prefetch_check.zip"
}

resource "aws_lambda_function" "prefetch_check" {
  function_name = "ftm-prefetch-check"
  description   = "Check if prefetch manifest exists for a quarter"

  filename         = data.archive_file.prefetch_check.output_path
  source_code_hash = data.archive_file.prefetch_check.output_base64sha256

  handler = "prefetch_check.handler"
  runtime = "python3.11"
  timeout = var.lambda_timeout
  memory_size = var.lambda_memory_size

  role = local.lambda_execution_role_arn

  environment {
    variables = {
      S3_BUCKET = var.s3_bucket_name
    }
  }

  tags = {
    Name    = "ftm-prefetch-check"
    Project = "financial-topic-modeling"
  }
}

# -----------------------------------------------------------------------------
# LAMBDA: Create Batch Manifest
# -----------------------------------------------------------------------------
data "archive_file" "create_batch_manifest" {
  type        = "zip"
  source_file = "${path.module}/../../src/lambdas/create_batch_manifest.py"
  output_path = "${path.module}/builds/create_batch_manifest.zip"
}

resource "aws_lambda_function" "create_batch_manifest" {
  function_name = "ftm-create-batch-manifest"
  description   = "Create batch manifest JSONL and return batch_ids"

  filename         = data.archive_file.create_batch_manifest.output_path
  source_code_hash = data.archive_file.create_batch_manifest.output_base64sha256

  handler = "create_batch_manifest.handler"
  runtime = "python3.11"
  timeout = var.lambda_timeout
  memory_size = var.lambda_memory_size

  role = local.lambda_execution_role_arn

  environment {
    variables = {
      S3_BUCKET  = var.s3_bucket_name
      BATCH_SIZE = tostring(var.batch_size)
    }
  }

  tags = {
    Name    = "ftm-create-batch-manifest"
    Project = "financial-topic-modeling"
  }
}

# -----------------------------------------------------------------------------
# LAMBDA: Summarize Results
# -----------------------------------------------------------------------------
data "archive_file" "summarize_results" {
  type        = "zip"
  source_file = "${path.module}/../../src/lambdas/summarize_results.py"
  output_path = "${path.module}/builds/summarize_results.zip"
}

resource "aws_lambda_function" "summarize_results" {
  function_name = "ftm-summarize-results"
  description   = "Count succeeded/failed batches from Map state output"

  filename         = data.archive_file.summarize_results.output_path
  source_code_hash = data.archive_file.summarize_results.output_base64sha256

  handler     = "summarize_results.handler"
  runtime     = "python3.11"
  timeout     = var.lambda_timeout
  memory_size = var.lambda_memory_size

  role = local.lambda_execution_role_arn

  tags = {
    Name    = "ftm-summarize-results"
    Project = "financial-topic-modeling"
  }
}

# -----------------------------------------------------------------------------
# LAMBDA: Notify Completion
# -----------------------------------------------------------------------------
data "archive_file" "notify_completion" {
  type        = "zip"
  source_file = "${path.module}/../../src/lambdas/notify_completion.py"
  output_path = "${path.module}/builds/notify_completion.zip"
}

resource "aws_lambda_function" "notify_completion" {
  function_name = "ftm-notify-completion"
  description   = "Send SNS notification on quarter completion"

  filename         = data.archive_file.notify_completion.output_path
  source_code_hash = data.archive_file.notify_completion.output_base64sha256

  handler = "notify_completion.handler"
  runtime = "python3.11"
  timeout = var.lambda_timeout
  memory_size = var.lambda_memory_size

  role = local.lambda_execution_role_arn

  environment {
    variables = {
      SNS_TOPIC_ARN = var.sns_topic_arn
    }
  }

  tags = {
    Name    = "ftm-notify-completion"
    Project = "financial-topic-modeling"
  }
}

# -----------------------------------------------------------------------------
# CLOUDWATCH LOG GROUPS - For Lambda functions
# -----------------------------------------------------------------------------
resource "aws_cloudwatch_log_group" "prefetch_check" {
  name              = "/aws/lambda/ftm-prefetch-check"
  retention_in_days = 14

  tags = {
    Name    = "ftm-prefetch-check-logs"
    Project = "financial-topic-modeling"
  }
}

resource "aws_cloudwatch_log_group" "create_batch_manifest" {
  name              = "/aws/lambda/ftm-create-batch-manifest"
  retention_in_days = 14

  tags = {
    Name    = "ftm-create-batch-manifest-logs"
    Project = "financial-topic-modeling"
  }
}

resource "aws_cloudwatch_log_group" "summarize_results" {
  name              = "/aws/lambda/ftm-summarize-results"
  retention_in_days = 14

  tags = {
    Name    = "ftm-summarize-results-logs"
    Project = "financial-topic-modeling"
  }
}

resource "aws_cloudwatch_log_group" "notify_completion" {
  name              = "/aws/lambda/ftm-notify-completion"
  retention_in_days = 14

  tags = {
    Name    = "ftm-notify-completion-logs"
    Project = "financial-topic-modeling"
  }
}
