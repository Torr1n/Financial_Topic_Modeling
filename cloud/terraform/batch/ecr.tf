# Financial Topic Modeling - ECR Repository

# -----------------------------------------------------------------------------
# ECR REPOSITORY - Container images for map phase
# -----------------------------------------------------------------------------
resource "aws_ecr_repository" "map" {
  name                 = "ftm-map"
  image_tag_mutability = "MUTABLE" # Allow :latest tag updates

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Name    = "ftm-map"
    Project = "financial-topic-modeling"
  }
}

# Lifecycle policy - keep only last 5 images to control costs
resource "aws_ecr_lifecycle_policy" "map" {
  repository = aws_ecr_repository.map.name

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
  name                 = "ftm-reduce"
  image_tag_mutability = "MUTABLE" # Allow :latest tag updates

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Name    = "ftm-reduce"
    Project = "financial-topic-modeling"
  }
}

# Lifecycle policy - keep only last 5 images to control costs
resource "aws_ecr_lifecycle_policy" "reduce" {
  repository = aws_ecr_repository.reduce.name

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
