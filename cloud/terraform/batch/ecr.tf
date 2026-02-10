# Financial Topic Modeling - ECR Repository
#
# When use_external_images=true: skips ECR creation, uses image URIs from variables
# When use_external_images=false: creates ECR repos (original behavior)

# -----------------------------------------------------------------------------
# LOCALS - Resolve to either external image URIs or self-managed ECR repos
# -----------------------------------------------------------------------------
locals {
  map_image    = var.use_external_images ? var.map_image_uri : "${aws_ecr_repository.map[0].repository_url}:latest"
  reduce_image = var.use_external_images ? var.reduce_image_uri : "${aws_ecr_repository.reduce[0].repository_url}:latest"
}

# =============================================================================
# SELF-MANAGED ECR (only created when use_external_images=false)
# =============================================================================

# -----------------------------------------------------------------------------
# ECR REPOSITORY - Container images for map phase
# -----------------------------------------------------------------------------
resource "aws_ecr_repository" "map" {
  count                = var.use_external_images ? 0 : 1
  name                 = "ftm-map"
  image_tag_mutability = "MUTABLE" # Allow :latest tag updates

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Name = "ftm-map"
  }
}

# Lifecycle policy - keep only last 5 images to control costs
resource "aws_ecr_lifecycle_policy" "map" {
  count      = var.use_external_images ? 0 : 1
  repository = aws_ecr_repository.map[0].name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep only last 5 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 5
      }
      action = {
        type = "expire"
      }
    }]
  })
}

# -----------------------------------------------------------------------------
# ECR REPOSITORY - Container images for reduce phase (theme aggregation)
# -----------------------------------------------------------------------------
resource "aws_ecr_repository" "reduce" {
  count                = var.use_external_images ? 0 : 1
  name                 = "ftm-reduce"
  image_tag_mutability = "MUTABLE" # Allow :latest tag updates

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Name = "ftm-reduce"
  }
}

# Lifecycle policy - keep only last 5 images to control costs
resource "aws_ecr_lifecycle_policy" "reduce" {
  count      = var.use_external_images ? 0 : 1
  repository = aws_ecr_repository.reduce[0].name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep only last 5 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 5
      }
      action = {
        type = "expire"
      }
    }]
  })
}
