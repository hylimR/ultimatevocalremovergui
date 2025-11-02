variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "prod"
}

variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "vocal-remover"
}

variable "max_uploads_per_day" {
  description = "Maximum number of uploads per IP per day"
  type        = number
  default     = 5
}

variable "max_file_size" {
  description = "Maximum file size in bytes (default 500MB)"
  type        = number
  default     = 524288000
}

variable "enable_waf" {
  description = "Enable AWS WAF for additional rate limiting"
  type        = bool
  default     = false
}
