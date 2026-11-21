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
  default = "us-east-1" # used when nobody overrides it, overrides possible through TF_VARs or command line args
  type    = string
}

variable "suffix" { # used to prevent resource name conflicts!
  default = "dev"
  type    = string
}