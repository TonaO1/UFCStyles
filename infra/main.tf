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
  name              = "/aws/lambda/${local.function_name}"
  retention_in_days = 7
}

resource "aws_lambda_function" "api" {
  function_name = local.function_name
  role          = aws_iam_role.lambda.arn
  # Role needs ARN because AWS Lambda can only look up a role by ARN unlike the policies which can look each other up by name

  package_type  = "Image"                                               # Specifies that this lambda function is in a docker image
  image_uri     = "${aws_ecr_repository.serving.repository_url}:latest" # ECR REPO URL where Lambda function runs
  architectures = ["arm64"]                                             # Specifies the CPU architecture for the mac

  memory_size = 512 # Amount of memory allocated to Lambda function (determines speed of lambda on cold starts)
  timeout     = 10  # Default is 3 seconds for timeout on a lambda function request, 10 seconds gives more leeway

  environment {
    variables = {
      TABLE_NAME = aws_dynamodb_table.fighters.name # ENV VAR that handler.py uses
    }
  }
  logging_config { # ensures log group is the resource we designed and that log format is not JSON default
    log_format = "Text"
    log_group  = aws_cloudwatch_log_group.lambda.name
  }
}
# Note: Output is used to give out about resource information used by CI scripts for monitoring or other modules/scripts for logic

output "lambda_function_name" {
  value = aws_lambda_function.api.function_name
}

# Defines the API gateway
# We are using v2 because our handler.py calls for rawPath and requestContext.http.method which only exist on v2's payload
resource "aws_apigatewayv2_api" "http" {
  name          = "ufc-styles-http-${var.suffix}"
  protocol_type = "HTTP" # Must explicitly define HTTP protocol type because default is Websocket protocol
}
# Defines how the API talks to Lambda (HTTP Method,etc)
resource "aws_apigatewayv2_integration" "lambda" {
  api_id                 = aws_apigatewayv2_api.http.id       # Identifier for the API gateway resource we provisioned
  integration_type       = "AWS_PROXY"                        # API Gateway sends the request as is; handler.py reads the HTTP request and returns the HTTP response itself
  integration_uri        = aws_lambda_function.api.invoke_arn # The special api gateway resource location for the Lambda function
  integration_method     = "POST"                             # Any call to Lambda Function is a post regardless of what request user sends to API Gateway
  payload_format_version = "2.0"
}

# Defines the request API routes available at API gateway
resource "aws_apigatewayv2_route" "similar" {
  api_id    = aws_apigatewayv2_api.http.id                             # Identifier for the API gateway the route belongs to
  route_key = "GET /similar"                                           # API endpoint URL path as expected by handler.py
  target    = "integrations/${aws_apigatewayv2_integration.lambda.id}" # Delineates which API integration handles the endpoint
}

resource "aws_apigatewayv2_route" "matchup" {
  api_id    = aws_apigatewayv2_api.http.id
  route_key = "GET /matchup"
  target    = "integrations/${aws_apigatewayv2_integration.lambda.id}"
}

# The "Stage" where the API URL is live
resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.http.id
  name        = "$default"
  auto_deploy = true # Makes sure changes to routes go live without manual overhead

  # Rate-limiting since this is a publicly accesible URL
  default_route_settings {
    throttling_burst_limit = 10
    throttling_rate_limit  = 5
  }

}

# Gives API gateway permission to invoke lambda function
resource "aws_lambda_permission" "apigw" {
  statement_id  = "AllowAPIGatewayInvoke"               # Permission name for permission granted to api gateway
  action        = "lambda:InvokeFunction"               # Action that api gateway can invoke on function
  function_name = aws_lambda_function.api.function_name # name of lambda function being invoked
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.http.execution_arn}/*/*" # Limits invoke permission to our api gateway only
}

output "api_url" {
  value = aws_apigatewayv2_stage.default.invoke_url
}
