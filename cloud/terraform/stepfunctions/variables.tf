# Financial Topic Modeling - Step Functions Variables

variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-west-2"
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

# -----------------------------------------------------------------------------
# RESEARCH ACCOUNT COMPATIBILITY
# -----------------------------------------------------------------------------

variable "use_precreated_roles" {
  description = "Use precreated IAM roles instead of creating new ones (for restricted accounts)"
  type        = bool
  default     = true
}

variable "precreated_lambda_execution_role_name" {
  description = "Name of precreated Lambda execution role"
  type        = string
  default     = "role-torrin-lambda-exec"
}

variable "precreated_sfn_execution_role_name" {
  description = "Name of precreated Step Functions execution role"
  type        = string
  default     = "role-torrin-sfn-exec"
}
