# Financial Topic Modeling - Step Functions State Machine

# -----------------------------------------------------------------------------
# CLOUDWATCH LOG GROUP - For state machine execution logs
# -----------------------------------------------------------------------------
resource "aws_cloudwatch_log_group" "sfn" {
  name              = "/aws/states/ftm-quarter-processor"
  retention_in_days = 14

  tags = {
    Name    = "ftm-sfn-logs"
    Project = "financial-topic-modeling"
  }
}

# -----------------------------------------------------------------------------
# STATE MACHINE - Quarter processing pipeline
# -----------------------------------------------------------------------------
resource "aws_sfn_state_machine" "quarter_processor" {
  name     = "ftm-quarter-processor"
  role_arn = local.sfn_execution_role_arn

  definition = templatefile("${path.module}/state_machine.json", {
    prefetch_check_arn           = aws_lambda_function.prefetch_check.arn
    create_batch_manifest_arn    = aws_lambda_function.create_batch_manifest.arn
    summarize_results_arn        = aws_lambda_function.summarize_results.arn
    notify_completion_arn        = aws_lambda_function.notify_completion.arn
    job_queue_arn                = data.aws_batch_job_queue.main.arn
    job_definition_arn           = data.aws_batch_job_definition.firm_processor.arn
    reduce_job_definition_arn    = data.aws_batch_job_definition.theme_aggregator.arn
    max_concurrency              = var.max_concurrency
  })

  logging_configuration {
    log_destination        = "${aws_cloudwatch_log_group.sfn.arn}:*"
    include_execution_data = true
    level                  = "ALL"
  }

  tracing_configuration {
    enabled = true
  }

  tags = {
    Name    = "ftm-quarter-processor"
    Project = "financial-topic-modeling"
  }
}
