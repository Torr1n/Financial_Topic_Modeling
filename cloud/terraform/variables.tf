# Financial Topic Modeling - Terraform Variables

variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-west-2"
}

variable "db_username" {
  description = "RDS master username"
  type        = string
  default     = "ftm"
}

variable "db_password" {
  description = "RDS master password (use terraform.tfvars or env var TF_VAR_db_password)"
  type        = string
  sensitive   = true
}

variable "my_ip" {
  description = "Your IP address in CIDR notation for SSH access (e.g., 1.2.3.4/32)"
  type        = string
}

variable "key_pair_name" {
  description = "Name of existing EC2 key pair for SSH access"
  type        = string
}
