# Financial Topic Modeling - Step Functions Outputs

output "state_machine_arn" {
  description = "Step Functions state machine ARN"
  value       = aws_sfn_state_machine.quarter_processor.arn
}

output "state_machine_name" {
  description = "Step Functions state machine name"
  value       = aws_sfn_state_machine.quarter_processor.name
}

output "prefetch_check_lambda_arn" {
  description = "Prefetch check Lambda function ARN"
  value       = aws_lambda_function.prefetch_check.arn
}

output "create_batch_manifest_lambda_arn" {
  description = "Create batch manifest Lambda function ARN"
  value       = aws_lambda_function.create_batch_manifest.arn
}

output "summarize_results_lambda_arn" {
  description = "Summarize results Lambda function ARN"
  value       = aws_lambda_function.summarize_results.arn
}

output "notify_completion_lambda_arn" {
  description = "Notify completion Lambda function ARN"
  value       = aws_lambda_function.notify_completion.arn
}

# Helper commands for execution
output "start_execution_command" {
  description = "AWS CLI command to start a quarter processing execution"
  value       = <<-EOT
    aws stepfunctions start-execution \
      --state-machine-arn ${aws_sfn_state_machine.quarter_processor.arn} \
      --name "2023Q1-$(date +%Y%m%d-%H%M%S)" \
      --input '{"quarter": "2023Q1", "bucket": "${var.s3_bucket_name}", "batch_size": ${var.batch_size}}'
  EOT
}

output "list_executions_command" {
  description = "AWS CLI command to list recent executions"
  value       = "aws stepfunctions list-executions --state-machine-arn ${aws_sfn_state_machine.quarter_processor.arn} --max-results 10"
}

output "console_url" {
  description = "AWS Console URL for the state machine"
  value       = "https://${data.aws_region.current.name}.console.aws.amazon.com/states/home?region=${data.aws_region.current.name}#/statemachines/view/${aws_sfn_state_machine.quarter_processor.arn}"
}
