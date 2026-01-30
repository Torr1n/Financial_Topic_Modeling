# Financial Topic Modeling - Batch Infrastructure Variables

variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-east-1"
}

variable "s3_bucket_name" {
  description = "Existing S3 bucket name from main terraform (get via: cd ../; terraform output -raw s3_bucket_name)"
  type        = string
}

variable "max_vcpus" {
  description = "Maximum vCPUs for the compute environment"
  type        = number
  default     = 64
}

variable "job_timeout_seconds" {
  description = "Maximum job duration in seconds (default: 5 hours)"
  type        = number
  default     = 18000
}

variable "checkpoint_interval" {
  description = "Number of firms to process before checkpointing"
  type        = number
  default     = 50
}

variable "enable_vllm" {
  description = "Enable vLLM integration (reads base URL from SSM, requires ECS module deployed first)"
  type        = bool
  default     = false
}

variable "vllm_model" {
  description = "Model name served by vLLM (must match ECS vllm_model)"
  type        = string
  default     = "Qwen/Qwen3-8B"
}
