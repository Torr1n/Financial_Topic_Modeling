# Financial Topic Modeling - Batch Compute Environment and Job Queue

# -----------------------------------------------------------------------------
# SECURITY GROUP - Batch compute instances
# -----------------------------------------------------------------------------
resource "aws_security_group" "batch" {
  name        = "ftm-batch-sg"
  description = "Security group for Batch compute instances"
  vpc_id      = data.aws_vpc.default.id

  # Outbound: Allow all (needed for ECR, S3, WRDS, Secrets Manager)
  egress {
    description = "All outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # No inbound rules needed - instances only make outbound connections

  tags = {
    Name    = "ftm-batch-sg"
    Project = "financial-topic-modeling"
  }
}

# -----------------------------------------------------------------------------
# COMPUTE ENVIRONMENT - GPU Spot instances
# -----------------------------------------------------------------------------
resource "aws_batch_compute_environment" "gpu_spot" {
  compute_environment_name = "ftm-gpu-spot"
  type                     = "MANAGED"
  state                    = "ENABLED"
  service_role             = aws_iam_role.batch_service.arn

  compute_resources {
    type                = "SPOT"
    allocation_strategy = "SPOT_CAPACITY_OPTIMIZED"
    bid_percentage      = 100 # Pay up to on-demand price

    # GPU instance types (T4 and A10G)
    instance_type = ["g4dn.xlarge", "g5.xlarge", "g4dn.2xlarge"]

    max_vcpus = var.max_vcpus
    min_vcpus = 0 # Scale to zero when idle

    subnets            = data.aws_subnets.default.ids
    security_group_ids = [aws_security_group.batch.id]

    instance_role       = aws_iam_instance_profile.batch_instance.arn
    spot_iam_fleet_role = aws_iam_role.spot_fleet.arn

    # Explicitly use GPU-optimized AMI with NVIDIA drivers
    ec2_configuration {
      image_type = "ECS_AL2_NVIDIA"
    }

    tags = {
      Name    = "ftm-batch-instance"
      Project = "financial-topic-modeling"
    }
  }

  tags = {
    Name    = "ftm-gpu-spot"
    Project = "financial-topic-modeling"
  }

  # Prevent replacement when compute_resources changes
  lifecycle {
    create_before_destroy = true
  }
}

# -----------------------------------------------------------------------------
# JOB QUEUE - Main queue for firm processing jobs
# -----------------------------------------------------------------------------
resource "aws_batch_job_queue" "main" {
  name     = "ftm-queue-main"
  state    = "ENABLED"
  priority = 10

  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.gpu_spot.arn
  }

  tags = {
    Name    = "ftm-queue-main"
    Project = "financial-topic-modeling"
  }
}
