# Financial Topic Modeling - Step Functions Variables

variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-east-1"
}

variable "s3_bucket_name" {
  description = "S3 bucket name for manifests and output"
  type        = string
}

variable "job_queue_name" {
  description = "AWS Batch job queue name"
  type        = string
  default     = "ftm-queue-main"
}

variable "job_definition_name" {
  description = "AWS Batch job definition name for map phase"
  type        = string
  default     = "ftm-firm-processor"
}

variable "reduce_job_definition_name" {
  description = "AWS Batch job definition name for reduce phase"
  type        = string
  default     = "ftm-theme-aggregator"
}

variable "sns_topic_arn" {
  description = "SNS topic ARN for completion notifications (optional)"
  type        = string
  default     = ""
}

variable "batch_size" {
  description = "Number of firms per batch job"
  type        = number
  default     = 1000
}

variable "max_concurrency" {
  description = "Maximum concurrent batch jobs in Map state"
  type        = number
  default     = 5
}

variable "lambda_memory_size" {
  description = "Memory size for Lambda functions in MB"
  type        = number
  default     = 256
}

variable "lambda_timeout" {
  description = "Timeout for Lambda functions in seconds"
  type        = number
  default     = 60
}
