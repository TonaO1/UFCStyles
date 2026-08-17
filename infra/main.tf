terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ============================================================================
# VARIABLES
# ============================================================================

variable "aws_region" {
  description = "AWS region"
  default     = "us-east-1"
}

variable "suffix" {
  description = "Suffix for resource names (e.g., your name or username)"
  default     = "dev"
}

variable "budget_limit_usd" {
  description = "Monthly budget limit in USD"
  default     = 10
}

# ============================================================================
# S3: Data Storage
# ============================================================================

resource "aws_s3_bucket" "data" {
  bucket = "ufc-style-${var.suffix}-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

output "s3_bucket" {
  value = aws_s3_bucket.data.id
}

# ============================================================================
# DYNAMODB: Fighter Embeddings
# ============================================================================

resource "aws_dynamodb_table" "fighter_embeddings" {
  name           = "ufc-fighter-embeddings-${var.suffix}"
  billing_mode   = "PAY_PER_REQUEST"  # On-demand
  hash_key       = "fighter_id"
  
  attribute {
    name = "fighter_id"
    type = "S"
  }
  
  ttl {
    attribute_name = "expires_at"
    enabled        = false
  }
  
  tags = {
    Name = "Fighter Embeddings"
  }
}

output "dynamodb_table" {
  value = aws_dynamodb_table.fighter_embeddings.name
}

# ============================================================================
# BUDGET ALERTS
# ============================================================================

resource "aws_budgets_budget" "monthly_limit" {
  name              = "ufc-style-monthly-${var.suffix}"
  budget_type       = "MONTHLY"
  limit_unit        = "USD"
  limit_amount      = var.budget_limit_usd
  time_period_start = "2024-01-01_00:00"
  time_period_end   = "2087-12-31_23:59"
  
  notification {
    comparison_operator        = "GREATER_THAN"
    notification_type          = "FORECASTED"
    threshold                  = 50
    threshold_type             = "PERCENTAGE"
    notification_channels      = ["arn:aws:sns:${var.aws_region}:${data.aws_caller_identity.current.account_id}:Default_CloudWatch_Alarms_Topic"]
  }
  
  notification {
    comparison_operator        = "GREATER_THAN"
    notification_type          = "FORECASTED"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_channels      = ["arn:aws:sns:${var.aws_region}:${data.aws_caller_identity.current.account_id}:Default_CloudWatch_Alarms_Topic"]
  }
}

# ============================================================================
# CLOUDWATCH: Billing Alarm
# ============================================================================

resource "aws_cloudwatch_metric_alarm" "billing" {
  alarm_name          = "ufc-style-billing-${var.suffix}"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = 1
  metric_name         = "EstimatedCharges"
  namespace           = "AWS/Billing"
  period              = 86400
  statistic           = "Maximum"
  threshold           = 5.0  # $5 hard limit
  
  dimensions = {
    Currency = "USD"
  }
  
  alarm_actions = []  # Add SNS topic if desired
}

# ============================================================================
# IAM: Least-Privilege Roles (for Lambda)
# ============================================================================

resource "aws_iam_role" "lambda_role" {
  name = "ufc-style-lambda-${var.suffix}"
  
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
}

resource "aws_iam_role_policy" "lambda_policy" {
  name = "ufc-style-lambda-policy-${var.suffix}"
  role = aws_iam_role.lambda_role.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket",
        ]
        Resource = [
          aws_s3_bucket.data.arn,
          "${aws_s3_bucket.data.arn}/*",
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "dynamodb:GetItem",
          "dynamodb:Query",
          "dynamodb:Scan",
        ]
        Resource = aws_dynamodb_table.fighter_embeddings.arn
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
        ]
        Resource = "arn:aws:logs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:log-group:/aws/lambda/*"
      },
    ]
  })
}

# ============================================================================
# LAMBDA (Placeholder)
# ============================================================================

# Note: Lambda function created via ECR image or ZIP file
# This is just the role and placeholder for the function definition

output "lambda_role_arn" {
  value = aws_iam_role.lambda_role.arn
}

# ============================================================================
# API GATEWAY (Placeholder)
# ============================================================================

# API Gateway setup would go here (routes to /similar and /matchup)

# ============================================================================
# DATA SOURCE: Current AWS Account
# ============================================================================

data "aws_caller_identity" "current" {}

output "account_id" {
  value = data.aws_caller_identity.current.account_id
}
