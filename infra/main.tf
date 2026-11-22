# Terraform settings and providers
terraform {
  required_version = ">= 1.5" # Terraform must be version 1.5 or greater

  required_providers {
    aws = {                     # There is a = here because aws is a chosen name for a provider not a fixed keyword
      source  = "hashicorp/aws" # download from the registry publisher
      version = "~> 6.0"        # Allows 6. anything but not anything else like 7.0
    }
  }
}
# Important lesson, the order at which variables are created does not matter because Terraform is not running code, it's building a config for infrastructure

provider "aws" { # Settings for "aws" plugin that is our required provider
  region = var.aws_region
}

variable "aws_region" { # reference it by using var.aws_region
  description = "AWS region"
  default     = "us-east-2" # used when nobody overrides it, overrides possible through TF_VARs or command line args
  type        = string
}

variable "suffix" { # used to prevent resource name conflicts!
  description = "var used to prevent resource name conflicts!"
  default     = "dev"
  type        = string
}


resource "aws_ecr_repository" "serving" {
  name                 = "ufc-styles-serving-${var.suffix}"
  image_tag_mutability = "MUTABLE" # I will rebuild the AWS ECR image frequently so I need mutability to avoid having latter pushes rejected
  force_delete         = true      # This is a personal project of little signifcance so no need for precautions
}

output "ecr_repository_url" {
  value = aws_ecr_repository.serving.repository_url # CMD Terraform output returns value which is our "serving" URL
}

resource "aws_dynamodb_table" "fighters" {
  name         = "ufc-fighter-embeddings-${var.suffix}" # Matching convention used in handler.py
  billing_mode = "PAY_PER_REQUEST"                      # Default is Provisioned which makes you pay even when service is idle
  hash_key     = "fighter_id"                           # Primary key to differentiate different embeddings

  attribute { # Attribute for the hashkey
    name = "fighter_id"
    type = "S"

  }
}

output "dynamodb_table_name" {
  value = aws_dynamodb_table.fighters.name
}

# Configurations for IAM role used to make AWS Lambda function without storing credentials in image
resource "aws_iam_role" "lambda" {
  name = "ufc-styles-lambda-${var.suffix}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17" # The version of IAM language being used by AWS currently
    Statement = [{
      Effect    = "Allow"
      Action    = "sts:AssumeRole"                     # STS is AWS Secure Token Service used to grant temporary credentials and AssumeRole tells AWS Lambda it can borrow this role
      Principal = { Service = "lambda.amazonaws.com" } # AWS Lambda posses this IAM role (Principal means who is granted authority)

    }]
  })
}
# Configurations for Authorization on DynamoDB resource
resource "aws_iam_role_policy" "lambda_dynamodb" { # Permissions for a specific iam role (the one we configured)
  name = "read-fighters-table"
  role = aws_iam_role.lambda.name # Use IAM role we configured to access dynamodb and then use this policy to authorize calls to dynamodb
  policy = jsonencode({
    Version = "2012-10-17" # The version of IAM language being used by AWS currently
    Statement = [{
      Effect   = "Allow"
      Action   = "dynamodb:Scan" # We onyl need to read the entire table into memory cache once so this is least permission needed
      Resource = aws_dynamodb_table.fighters.arn
      # This resource has no principal value in the statement because it is a policy for someone already authorized
      # This means we do not need to specify who this permision is being authorized to (we already check using role = aws iam role)
    }]
  })
}
# Permissions to write logs
resource "aws_iam_role_policy_attachment" "lambda_logs" {
  # aws iam role policy attachment is a type of readymade policy AWS maintains
  role       = aws_iam_role.lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
  # Ready-made policy for allowing Lambda to send logs to AWS CloudWatch
  # Note: We don't need to provision a CloudWatch resource because it is on by default for every account and region

}

locals {
  function_name = "ufc-styles-api-${var.suffix}"
}
# Adds AWS Cloudwatch group to Terraform so we can set retention data (default forever)
# Without defining it, Terraform cannot tear down the log group or affect it anyway
resource "aws_cloudwatch_log_group" "lambda" {
  name = "/aws/lambda/${local.function_name}"
  retention_in_days = 7
}

