# Financial Topic Modeling - ECS vLLM Variables

variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-west-2"
}

variable "vllm_model" {
  description = "HuggingFace model ID for vLLM to serve"
  type        = string
  default     = "Qwen/Qwen3-8B"
}

variable "instance_type" {
  description = "EC2 instance type for vLLM (must have GPU)"
  type        = string
  default     = "g5.xlarge"
}

variable "spot_percentage" {
  description = "Percentage of Spot instances in capacity provider (0-100)"
  type        = number
  default     = 100
}

variable "on_demand_base_capacity" {
  description = "Number of on-demand instances to maintain (guaranteed availability)"
  type        = number
  default     = 1
}

variable "min_capacity" {
  description = "Minimum number of vLLM tasks (set to 1 during runs, 0 when idle)"
  type        = number
  default     = 1
}

variable "max_capacity" {
  description = "Maximum number of vLLM tasks for auto-scaling"
  type        = number
  default     = 4
}

variable "container_memory" {
  description = "Memory for vLLM container in MB"
  type        = number
  default     = 14000
}

variable "container_cpu" {
  description = "CPU units for vLLM container (1024 = 1 vCPU)"
  type        = number
  default     = 4096
}

variable "vllm_port" {
  description = "Port for vLLM OpenAI-compatible API"
  type        = number
  default     = 8000
}

variable "health_check_startup_period" {
  description = "Health check grace period in seconds (model loading time)"
  type        = number
  default     = 300
}

variable "shared_memory_size" {
  description = "Shared memory size in MB (PyTorch requirement)"
  type        = number
  default     = 2048
}

# -----------------------------------------------------------------------------
# RESEARCH ACCOUNT COMPATIBILITY
# -----------------------------------------------------------------------------

variable "use_precreated_roles" {
  description = "Use precreated IAM roles instead of creating new ones (for restricted accounts)"
  type        = bool
  default     = true
}

variable "precreated_ecs_execution_role_name" {
  description = "Name of precreated ECS task execution role"
  type        = string
  default     = "role-torrin-ecs-task-exec"
}

variable "precreated_ecs_task_role_name" {
  description = "Name of precreated ECS task role"
  type        = string
  default     = "role-torrin-ecs-task"
}

variable "precreated_ecs_instance_profile_name" {
  description = "Name of precreated ECS instance profile for vLLM GPU instances. NOTE: Instance profiles and roles are separate AWS resources — confirm the exact profile name with your admin."
  type        = string
  default     = "role-torrin-ecs-task"  # TODO: confirm with David
}
