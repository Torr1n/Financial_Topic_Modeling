# Financial Topic Modeling - Batch Infrastructure Outputs
# Used by deployment scripts and job submitter

output "ecr_repository_url" {
  description = "ECR repository URL for pushing container images"
  value       = aws_ecr_repository.map.repository_url
}

output "ecr_repository_name" {
  description = "ECR repository name"
  value       = aws_ecr_repository.map.name
}

output "job_definition_name" {
  description = "Batch job definition name"
  value       = aws_batch_job_definition.firm_processor.name
}

output "job_definition_arn" {
  description = "Batch job definition ARN"
  value       = aws_batch_job_definition.firm_processor.arn
}

output "job_queue_name" {
  description = "Batch job queue name"
  value       = aws_batch_job_queue.main.name
}

output "job_queue_arn" {
  description = "Batch job queue ARN"
  value       = aws_batch_job_queue.main.arn
}

output "compute_environment_name" {
  description = "Batch compute environment name"
  value       = aws_batch_compute_environment.gpu_spot.compute_environment_name
}

output "log_group_name" {
  description = "CloudWatch log group for batch jobs"
  value       = aws_cloudwatch_log_group.batch.name
}

output "s3_bucket_name" {
  description = "S3 bucket for manifests and output (passed through from input)"
  value       = var.s3_bucket_name
}
