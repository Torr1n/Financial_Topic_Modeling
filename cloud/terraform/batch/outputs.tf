# Financial Topic Modeling - Batch Infrastructure Outputs
# Used by deployment scripts and job submitter

output "map_image_uri" {
  description = "Container image URI for map phase"
  value       = local.map_image
}

output "reduce_image_uri" {
  description = "Container image URI for reduce phase"
  value       = local.reduce_image
}

output "ecr_repository_url" {
  description = "ECR repository URL for pushing map container images (empty if using external images)"
  value       = var.use_external_images ? "" : aws_ecr_repository.map[0].repository_url
}

output "ecr_repository_name" {
  description = "ECR repository name (empty if using external images)"
  value       = var.use_external_images ? "" : aws_ecr_repository.map[0].name
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

# Reduce phase outputs
output "reduce_ecr_repository_url" {
  description = "ECR repository URL for reduce phase (empty if using external images)"
  value       = var.use_external_images ? "" : aws_ecr_repository.reduce[0].repository_url
}

output "reduce_ecr_repository_name" {
  description = "ECR repository name for reduce phase (empty if using external images)"
  value       = var.use_external_images ? "" : aws_ecr_repository.reduce[0].name
}

output "reduce_job_definition_name" {
  description = "Batch job definition name for reduce phase"
  value       = aws_batch_job_definition.theme_aggregator.name
}

output "reduce_job_definition_arn" {
  description = "Batch job definition ARN for reduce phase"
  value       = aws_batch_job_definition.theme_aggregator.arn
}
