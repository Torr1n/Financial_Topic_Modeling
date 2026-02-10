# Financial Topic Modeling - Batch Infrastructure Variables

variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-west-2"
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

variable "llm_max_concurrent" {
  description = "Maximum concurrent LLM requests per batch job (conservative for single vLLM instance)"
  type        = number
  default     = 10
}

# -----------------------------------------------------------------------------
# RESEARCH ACCOUNT COMPATIBILITY - Use precreated roles instead of creating new ones
# -----------------------------------------------------------------------------

variable "use_precreated_roles" {
  description = "Use precreated IAM roles instead of creating new ones (for restricted accounts)"
  type        = bool
  default     = true
}

variable "precreated_batch_execution_role_name" {
  description = "Name of precreated ECS task execution role (only used when use_precreated_roles=true)"
  type        = string
  default     = "role-torrin-ecs-task-exec"
}

variable "precreated_batch_job_role_name" {
  description = "Name of precreated Batch job role (only used when use_precreated_roles=true)"
  type        = string
  default     = "role-torrin-batch-job"
}

variable "precreated_batch_instance_profile_name" {
  description = "Name of precreated Batch EC2 instance profile (only used when use_precreated_roles=true). Must have AmazonEC2ContainerServiceforEC2Role attached. NOTE: Instance profiles and roles are separate AWS resources — confirm the exact profile name with your admin."
  type        = string
  default     = "role-torrin-batch-job"  # TODO: confirm with David
}

# -----------------------------------------------------------------------------
# EXTERNAL IMAGES - Use pre-built container images instead of ECR repos
# -----------------------------------------------------------------------------

variable "use_external_images" {
  description = "Use external image URIs instead of creating ECR repos (for restricted accounts)"
  type        = bool
  default     = true
}

variable "map_image_uri" {
  description = "Full image URI for map container (required when use_external_images=true). Example: 015705018204.dkr.ecr.us-west-2.amazonaws.com/ftm-map:latest"
  type        = string
  default     = ""
}

variable "reduce_image_uri" {
  description = "Full image URI for reduce container (required when use_external_images=true). Example: 015705018204.dkr.ecr.us-west-2.amazonaws.com/ftm-reduce:latest"
  type        = string
  default     = ""
}

# -----------------------------------------------------------------------------
# WRDS SECRETS - Gate behind feature flag
# -----------------------------------------------------------------------------

variable "enable_wrds_secrets" {
  description = "Enable WRDS Secrets Manager injection in job definitions (disable for S3-only mode)"
  type        = bool
  default     = false
}
