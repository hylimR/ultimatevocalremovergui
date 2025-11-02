terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "VocalRemoverWeb"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# Data sources
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# S3 Buckets
resource "aws_s3_bucket" "upload_bucket" {
  bucket = "${var.project_name}-upload-${var.environment}-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket" "output_bucket" {
  bucket = "${var.project_name}-output-${var.environment}-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket" "frontend_bucket" {
  bucket = "${var.project_name}-frontend-${var.environment}-${data.aws_caller_identity.current.account_id}"
}

# S3 Bucket Lifecycle Policies
resource "aws_s3_bucket_lifecycle_configuration" "upload_bucket_lifecycle" {
  bucket = aws_s3_bucket.upload_bucket.id

  rule {
    id     = "delete-after-1-day"
    status = "Enabled"

    expiration {
      days = 1
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "output_bucket_lifecycle" {
  bucket = aws_s3_bucket.output_bucket.id

  rule {
    id     = "delete-after-7-days"
    status = "Enabled"

    expiration {
      days = 7
    }
  }
}

# S3 Bucket CORS
resource "aws_s3_bucket_cors_configuration" "upload_bucket_cors" {
  bucket = aws_s3_bucket.upload_bucket.id

  cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["PUT", "POST"]
    allowed_origins = ["*"]
    expose_headers  = ["ETag"]
    max_age_seconds = 3000
  }
}

resource "aws_s3_bucket_cors_configuration" "output_bucket_cors" {
  bucket = aws_s3_bucket.output_bucket.id

  cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["GET"]
    allowed_origins = ["*"]
    max_age_seconds = 3000
  }
}

# S3 Bucket Public Access Block
resource "aws_s3_bucket_public_access_block" "upload_bucket_pab" {
  bucket = aws_s3_bucket.upload_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_public_access_block" "output_bucket_pab" {
  bucket = aws_s3_bucket.output_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# S3 Bucket Notification for triggering workflow
resource "aws_s3_bucket_notification" "upload_notification" {
  bucket = aws_s3_bucket.upload_bucket.id

  lambda_function {
    lambda_function_arn = aws_lambda_function.start_workflow.arn
    events              = ["s3:ObjectCreated:*"]
    filter_prefix       = "uploads/"
  }

  depends_on = [aws_lambda_permission.allow_s3_invoke_start_workflow]
}

# DynamoDB Tables
resource "aws_dynamodb_table" "rate_limit_table" {
  name         = "${var.project_name}-rate-limit-${var.environment}"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "client_ip"
  range_key    = "date"

  attribute {
    name = "client_ip"
    type = "S"
  }

  attribute {
    name = "date"
    type = "S"
  }

  ttl {
    attribute_name = "ttl"
    enabled        = true
  }
}

resource "aws_dynamodb_table" "jobs_table" {
  name         = "${var.project_name}-jobs-${var.environment}"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "job_id"

  attribute {
    name = "job_id"
    type = "S"
  }

  ttl {
    attribute_name = "ttl"
    enabled        = true
  }
}

# SQS Queue (for buffering)
resource "aws_sqs_queue" "processing_queue" {
  name                       = "${var.project_name}-processing-${var.environment}.fifo"
  fifo_queue                 = true
  content_based_deduplication = true
  visibility_timeout_seconds = 900 # 15 minutes
  message_retention_seconds  = 86400 # 1 day
}

# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "presigned_url_logs" {
  name              = "/aws/lambda/${var.project_name}-presigned-url-${var.environment}"
  retention_in_days = 7
}

resource "aws_cloudwatch_log_group" "start_workflow_logs" {
  name              = "/aws/lambda/${var.project_name}-start-workflow-${var.environment}"
  retention_in_days = 7
}

resource "aws_cloudwatch_log_group" "status_logs" {
  name              = "/aws/lambda/${var.project_name}-status-${var.environment}"
  retention_in_days = 7
}

resource "aws_cloudwatch_log_group" "video_converter_logs" {
  name              = "/aws/lambda/${var.project_name}-video-converter-${var.environment}"
  retention_in_days = 7
}

resource "aws_cloudwatch_log_group" "vocal_separator_logs" {
  name              = "/aws/lambda/${var.project_name}-vocal-separator-${var.environment}"
  retention_in_days = 7
}

resource "aws_cloudwatch_log_group" "voice_converter_logs" {
  name              = "/aws/lambda/${var.project_name}-voice-converter-${var.environment}"
  retention_in_days = 7
}

resource "aws_cloudwatch_log_group" "cleanup_logs" {
  name              = "/aws/lambda/${var.project_name}-cleanup-${var.environment}"
  retention_in_days = 7
}

resource "aws_cloudwatch_log_group" "state_machine_logs" {
  name              = "/aws/states/${var.project_name}-processing-${var.environment}"
  retention_in_days = 7
}
