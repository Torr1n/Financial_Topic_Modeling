# Financial Topic Modeling - ECS Capacity Provider for vLLM
# GPU instances with Spot pricing
# NOTE: Scaling is MANUAL - no auto-scaling configured
# Use scale_up_command/scale_down_command from outputs to manage capacity

# -----------------------------------------------------------------------------
# LAUNCH TEMPLATE - GPU instances for vLLM
# -----------------------------------------------------------------------------
data "aws_ami" "ecs_gpu" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["amzn2-ami-ecs-gpu-hvm-*-x86_64-ebs"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

resource "aws_launch_template" "vllm" {
  name_prefix   = "ftm-vllm-"
  image_id      = data.aws_ami.ecs_gpu.id
  instance_type = var.instance_type

  # ECS cluster registration
  user_data = base64encode(<<-EOF
    #!/bin/bash
    echo "ECS_CLUSTER=${aws_ecs_cluster.vllm.name}" >> /etc/ecs/ecs.config
    echo "ECS_ENABLE_GPU_SUPPORT=true" >> /etc/ecs/ecs.config
  EOF
  )

  iam_instance_profile {
    arn = aws_iam_instance_profile.vllm_instance.arn
  }

  network_interfaces {
    associate_public_ip_address = true  # Required for HuggingFace downloads
    security_groups             = [aws_security_group.vllm_instance.id]
  }

  # EBS optimized for model loading
  ebs_optimized = true

  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      volume_size           = 100  # Space for model cache
      volume_type           = "gp3"
      delete_on_termination = true
    }
  }

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name    = "ftm-vllm-instance"
      Project = "financial-topic-modeling"
    }
  }

  tags = {
    Name    = "ftm-vllm-launch-template"
    Project = "financial-topic-modeling"
  }
}

# -----------------------------------------------------------------------------
# INSTANCE SECURITY GROUP - For EC2 instances hosting ECS tasks
# -----------------------------------------------------------------------------
resource "aws_security_group" "vllm_instance" {
  name        = "ftm-vllm-instance-sg"
  description = "Security group for vLLM ECS instances"
  vpc_id      = data.aws_vpc.default.id

  # Inbound: Allow vLLM port from ALB (for health checks)
  ingress {
    description     = "vLLM API from ALB"
    from_port       = var.vllm_port
    to_port         = var.vllm_port
    protocol        = "tcp"
    security_groups = [aws_security_group.vllm_alb.id]
  }

  # Outbound: Allow all (ECR, HuggingFace, CloudWatch)
  egress {
    description = "All outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "ftm-vllm-instance-sg"
    Project = "financial-topic-modeling"
  }
}

# -----------------------------------------------------------------------------
# IAM INSTANCE PROFILE - For EC2 instances in Auto Scaling Group
# -----------------------------------------------------------------------------
resource "aws_iam_role" "vllm_instance" {
  name = "ftm-vllm-instance"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ec2.amazonaws.com"
      }
    }]
  })

  tags = {
    Name    = "ftm-vllm-instance"
    Project = "financial-topic-modeling"
  }
}

resource "aws_iam_role_policy_attachment" "vllm_instance_ecs" {
  role       = aws_iam_role.vllm_instance.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}

resource "aws_iam_instance_profile" "vllm_instance" {
  name = "ftm-vllm-instance"
  role = aws_iam_role.vllm_instance.name
}

# -----------------------------------------------------------------------------
# AUTO SCALING GROUP - GPU instances with Spot pricing
# -----------------------------------------------------------------------------
resource "aws_autoscaling_group" "vllm" {
  name                = "ftm-vllm-asg"
  vpc_zone_identifier = data.aws_subnets.default.ids
  min_size            = 0
  max_size            = var.max_capacity
  desired_capacity    = var.min_capacity

  # Use mixed instances policy for Spot
  mixed_instances_policy {
    instances_distribution {
      on_demand_base_capacity                  = var.on_demand_base_capacity
      on_demand_percentage_above_base_capacity = 100 - var.spot_percentage
      spot_allocation_strategy                 = "capacity-optimized"
    }

    launch_template {
      launch_template_specification {
        launch_template_id = aws_launch_template.vllm.id
        version            = "$Latest"
      }

      # g5.xlarge only for now - Qwen3-8B verified to fit
      # Add g4dn.xlarge later after confirming model fits in 16GB T4 memory
      override {
        instance_type = "g5.xlarge"
      }
    }
  }

  # Protect from scale-in during active processing
  protect_from_scale_in = false

  tag {
    key                 = "Name"
    value               = "ftm-vllm-instance"
    propagate_at_launch = true
  }

  tag {
    key                 = "Project"
    value               = "financial-topic-modeling"
    propagate_at_launch = true
  }

  tag {
    key                 = "AmazonECSManaged"
    value               = "true"
    propagate_at_launch = true
  }

  lifecycle {
    create_before_destroy = true
  }
}

# -----------------------------------------------------------------------------
# CAPACITY PROVIDER - Link ASG to ECS cluster
# -----------------------------------------------------------------------------
resource "aws_ecs_capacity_provider" "vllm" {
  name = "ftm-vllm-capacity-provider"

  auto_scaling_group_provider {
    auto_scaling_group_arn         = aws_autoscaling_group.vllm.arn
    managed_termination_protection = "DISABLED"

    managed_scaling {
      maximum_scaling_step_size = 2
      minimum_scaling_step_size = 1
      status                    = "ENABLED"
      target_capacity           = 100
    }
  }

  tags = {
    Name    = "ftm-vllm-capacity-provider"
    Project = "financial-topic-modeling"
  }
}

resource "aws_ecs_cluster_capacity_providers" "vllm" {
  cluster_name = aws_ecs_cluster.vllm.name

  capacity_providers = [aws_ecs_capacity_provider.vllm.name]

  default_capacity_provider_strategy {
    base              = 1
    weight            = 100
    capacity_provider = aws_ecs_capacity_provider.vllm.name
  }
}
