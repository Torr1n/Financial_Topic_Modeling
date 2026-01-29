# Financial Topic Modeling - Step Functions IAM Roles

# -----------------------------------------------------------------------------
# LAMBDA EXECUTION ROLE - For Lambda functions
# -----------------------------------------------------------------------------
resource "aws_iam_role" "lambda_execution" {
  name = "ftm-sfn-lambda-execution"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })

  tags = {
    Name    = "ftm-sfn-lambda-execution"
    Project = "financial-topic-modeling"
  }
}

# Basic Lambda execution (CloudWatch Logs)
resource "aws_iam_role_policy_attachment" "lambda_basic_execution" {
  role       = aws_iam_role.lambda_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# S3 access for Lambda functions (read prefetch manifest, write batch manifest)
resource "aws_iam_role_policy" "lambda_s3" {
  name = "ftm-sfn-lambda-s3"
  role = aws_iam_role.lambda_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:HeadObject",
          "s3:PutObject"
        ]
        Resource = [
          "${data.aws_s3_bucket.pipeline.arn}/prefetch/*",
          "${data.aws_s3_bucket.pipeline.arn}/manifests/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket"
        ]
        Resource = [data.aws_s3_bucket.pipeline.arn]
        Condition = {
          StringLike = {
            "s3:prefix" = [
              "prefetch/*",
              "manifests/*"
            ]
          }
        }
      }
    ]
  })
}

# SNS publish for notifications (optional)
resource "aws_iam_role_policy" "lambda_sns" {
  count = var.sns_topic_arn != "" ? 1 : 0
  name  = "ftm-sfn-lambda-sns"
  role  = aws_iam_role.lambda_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = "sns:Publish"
        Resource = var.sns_topic_arn
      }
    ]
  })
}

# -----------------------------------------------------------------------------
# STEP FUNCTIONS EXECUTION ROLE
# -----------------------------------------------------------------------------
resource "aws_iam_role" "sfn_execution" {
  name = "ftm-sfn-execution"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "states.amazonaws.com"
      }
    }]
  })

  tags = {
    Name    = "ftm-sfn-execution"
    Project = "financial-topic-modeling"
  }
}

# Lambda invoke permissions
resource "aws_iam_role_policy" "sfn_lambda" {
  name = "ftm-sfn-lambda-invoke"
  role = aws_iam_role.sfn_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "lambda:InvokeFunction"
        ]
        Resource = [
          aws_lambda_function.prefetch_check.arn,
          aws_lambda_function.create_batch_manifest.arn,
          aws_lambda_function.summarize_results.arn,
          aws_lambda_function.notify_completion.arn
        ]
      }
    ]
  })
}

# Batch submit and describe job permissions
resource "aws_iam_role_policy" "sfn_batch" {
  name = "ftm-sfn-batch"
  role = aws_iam_role.sfn_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "batch:SubmitJob",
          "batch:DescribeJobs",
          "batch:TerminateJob"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "events:PutTargets",
          "events:PutRule",
          "events:DescribeRule"
        ]
        Resource = [
          "arn:aws:events:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:rule/StepFunctionsGetEventsForBatchJobsRule"
        ]
      }
    ]
  })
}

# CloudWatch Logs for Step Functions execution
resource "aws_iam_role_policy" "sfn_logs" {
  name = "ftm-sfn-logs"
  role = aws_iam_role.sfn_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogDelivery",
          "logs:GetLogDelivery",
          "logs:UpdateLogDelivery",
          "logs:DeleteLogDelivery",
          "logs:ListLogDeliveries",
          "logs:PutLogEvents",
          "logs:PutResourcePolicy",
          "logs:DescribeResourcePolicies",
          "logs:DescribeLogGroups"
        ]
        Resource = "*"
      }
    ]
  })
}

# X-Ray tracing (optional)
resource "aws_iam_role_policy" "sfn_xray" {
  name = "ftm-sfn-xray"
  role = aws_iam_role.sfn_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "xray:PutTraceSegments",
          "xray:PutTelemetryRecords",
          "xray:GetSamplingRules",
          "xray:GetSamplingTargets"
        ]
        Resource = "*"
      }
    ]
  })
}
