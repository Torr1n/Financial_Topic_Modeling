# Financial Topic Modeling - Terraform Outputs
# Used by deployment scripts to configure EC2 and pipeline
# Scripts read these values to avoid hardcoding

output "db_endpoint" {
  description = "RDS endpoint (host:port)"
  value       = aws_db_instance.main.endpoint
}

output "db_host" {
  description = "RDS hostname only (without port)"
  value       = aws_db_instance.main.address
}

output "db_name" {
  description = "Database name"
  value       = aws_db_instance.main.db_name
}

output "s3_bucket_name" {
  description = "S3 bucket for pipeline code and data"
  value       = aws_s3_bucket.pipeline.id
}

output "ec2_security_group_id" {
  description = "Security group ID for EC2 instances"
  value       = aws_security_group.ec2.id
}

output "db_security_group_id" {
  description = "Security group ID for RDS"
  value       = aws_security_group.db.id
}

output "instance_profile_name" {
  description = "IAM instance profile for EC2"
  value       = aws_iam_instance_profile.ec2_profile.name
}

output "db_password" {
  description = "RDS password (sensitive - for script use only)"
  value       = var.db_password
  sensitive   = true
}

output "key_pair_name" {
  description = "EC2 key pair name"
  value       = var.key_pair_name
}

output "aws_region" {
  description = "AWS region for deployment"
  value       = var.aws_region
}
