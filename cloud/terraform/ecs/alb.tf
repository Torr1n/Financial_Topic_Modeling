# Financial Topic Modeling - ALB for vLLM (Internal Only)
# Accessed by Batch jobs, not exposed to internet

# -----------------------------------------------------------------------------
# DATA SOURCE - Reference Batch security group (created by batch module)
# -----------------------------------------------------------------------------
data "aws_security_group" "batch" {
  filter {
    name   = "group-name"
    values = ["ftm-batch-sg"]
  }
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# -----------------------------------------------------------------------------
# SECURITY GROUPS
# -----------------------------------------------------------------------------

# ALB Security Group - Allows inbound from Batch SG only (least privilege)
resource "aws_security_group" "vllm_alb" {
  name        = "ftm-vllm-alb-sg"
  description = "Security group for vLLM ALB"
  vpc_id      = data.aws_vpc.default.id

  # Inbound: Allow HTTP from Batch SG only (not entire VPC)
  ingress {
    description     = "HTTP from Batch instances"
    from_port       = 80
    to_port         = 80
    protocol        = "tcp"
    security_groups = [data.aws_security_group.batch.id]
  }

  # Outbound: Allow all (needed to reach vLLM tasks)
  egress {
    description = "All outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "ftm-vllm-alb-sg"
    Project = "financial-topic-modeling"
  }
}

# vLLM Task Security Group - Allows inbound from ALB
resource "aws_security_group" "vllm_task" {
  name        = "ftm-vllm-task-sg"
  description = "Security group for vLLM ECS tasks"
  vpc_id      = data.aws_vpc.default.id

  # Inbound: Allow vLLM port from ALB
  ingress {
    description     = "vLLM API from ALB"
    from_port       = var.vllm_port
    to_port         = var.vllm_port
    protocol        = "tcp"
    security_groups = [aws_security_group.vllm_alb.id]
  }

  # Outbound: Allow all (needed for HuggingFace model downloads)
  egress {
    description = "All outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "ftm-vllm-task-sg"
    Project = "financial-topic-modeling"
  }
}

# -----------------------------------------------------------------------------
# APPLICATION LOAD BALANCER - Internal only
# -----------------------------------------------------------------------------
resource "aws_lb" "vllm" {
  name               = "ftm-vllm-alb"
  internal           = true  # INTERNAL ONLY - not exposed to internet
  load_balancer_type = "application"
  security_groups    = [aws_security_group.vllm_alb.id]
  subnets            = data.aws_subnets.default.ids

  enable_deletion_protection = false

  tags = {
    Name    = "ftm-vllm-alb"
    Project = "financial-topic-modeling"
  }
}

# -----------------------------------------------------------------------------
# TARGET GROUP - vLLM tasks
# -----------------------------------------------------------------------------
resource "aws_lb_target_group" "vllm" {
  name        = "ftm-vllm-tg"
  port        = var.vllm_port
  protocol    = "HTTP"
  vpc_id      = data.aws_vpc.default.id
  target_type = "instance"  # Required for host network mode

  health_check {
    enabled             = true
    healthy_threshold   = 2
    unhealthy_threshold = 3
    timeout             = 10
    interval            = 30
    path                = "/health"
    protocol            = "HTTP"
    matcher             = "200"
  }

  # Long deregistration delay for graceful shutdown
  deregistration_delay = 60

  tags = {
    Name    = "ftm-vllm-tg"
    Project = "financial-topic-modeling"
  }
}

# -----------------------------------------------------------------------------
# LISTENER - Forward HTTP to vLLM
# -----------------------------------------------------------------------------
resource "aws_lb_listener" "vllm" {
  load_balancer_arn = aws_lb.vllm.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.vllm.arn
  }

  tags = {
    Name    = "ftm-vllm-listener"
    Project = "financial-topic-modeling"
  }
}
