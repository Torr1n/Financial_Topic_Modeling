# Financial Topic Modeling - ECS vLLM Outputs

output "cluster_name" {
  description = "ECS cluster name"
  value       = aws_ecs_cluster.vllm.name
}

output "cluster_arn" {
  description = "ECS cluster ARN"
  value       = aws_ecs_cluster.vllm.arn
}

output "service_name" {
  description = "ECS service name"
  value       = aws_ecs_service.vllm.name
}

output "alb_dns_name" {
  description = "ALB DNS name for vLLM API"
  value       = aws_lb.vllm.dns_name
}

output "alb_arn" {
  description = "ALB ARN"
  value       = aws_lb.vllm.arn
}

output "vllm_base_url" {
  description = "vLLM OpenAI-compatible API base URL"
  value       = "http://${aws_lb.vllm.dns_name}/v1"
}

output "ssm_base_url_parameter" {
  description = "SSM parameter name for vLLM base URL"
  value       = aws_ssm_parameter.vllm_base_url.name
}

output "task_security_group_id" {
  description = "Security group ID for vLLM tasks"
  value       = aws_security_group.vllm_task.id
}

output "alb_security_group_id" {
  description = "Security group ID for vLLM ALB"
  value       = aws_security_group.vllm_alb.id
}

# Helper commands for managing vLLM service
output "scale_down_command" {
  description = "Command to scale down vLLM service to save costs"
  value       = "aws ecs update-service --cluster ${aws_ecs_cluster.vllm.name} --service ${aws_ecs_service.vllm.name} --desired-count 0"
}

output "scale_up_command" {
  description = "Command to scale up vLLM service for processing"
  value       = "aws ecs update-service --cluster ${aws_ecs_cluster.vllm.name} --service ${aws_ecs_service.vllm.name} --desired-count 1"
}
