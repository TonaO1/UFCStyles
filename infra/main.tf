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
  force_delete         = true # This is a personal project of little signifcance so no need for precautions
}

output "ecr_repository_url" {
  value = aws_ecr_repository.serving.repository_url # CMD Terraform output returns value which is our "serving" URL
}

resource "aws_dynamodb_table" "fighters" {
  name = "ufc_fighter_embeddings-${var.suffix}" # Matching convention used in handler.py
  billing_mode = "PAY_PER_REQUEST" # Default is Provisioned which makes you pay even when service is idle
  hash_key = "fighter_id" # Primary key to differentiate different embeddings

  attribute { # Attribute for the hashkey
    name = "fighter_id"
    type = "S"

  }
}

output "dynamodb_table_name" {
  value = aws_dynamodb_table.fighters.name
}
