# Financial Topic Modeling - SSM Parameter Store for Cross-Module Communication
# Writes ALB DNS to SSM so Batch module can read it (decouples Terraform roots)

# -----------------------------------------------------------------------------
# SSM PARAMETERS - vLLM endpoint for Batch jobs
# -----------------------------------------------------------------------------
resource "aws_ssm_parameter" "vllm_base_url" {
  name  = "/ftm/vllm/base_url"
  type  = "String"
  value = "http://${aws_lb.vllm.dns_name}/v1"

  description = "vLLM OpenAI-compatible API base URL (used by Batch jobs)"

  tags = {
    Name    = "ftm-vllm-base-url"
    Project = "financial-topic-modeling"
  }
}

resource "aws_ssm_parameter" "vllm_alb_dns" {
  name  = "/ftm/vllm/alb_dns"
  type  = "String"
  value = aws_lb.vllm.dns_name

  description = "vLLM ALB DNS name (for debugging and health checks)"

  tags = {
    Name    = "ftm-vllm-alb-dns"
    Project = "financial-topic-modeling"
  }
}
