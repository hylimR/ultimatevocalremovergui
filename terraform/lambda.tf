# ECR Repositories for Lambda Container Images
resource "aws_ecr_repository" "video_converter" {
  name                 = "${var.project_name}-video-converter"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "vocal_separator" {
  name                 = "${var.project_name}-vocal-separator"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "voice_converter" {
  name                 = "${var.project_name}-voice-converter"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

# Lambda Functions - API
resource "aws_lambda_function" "presigned_url" {
  function_name = "${var.project_name}-presigned-url-${var.environment}"
  role          = aws_iam_role.presigned_url_lambda_role.arn
  timeout       = 30
  memory_size   = 256

  filename         = data.archive_file.presigned_url_lambda.output_path
  source_code_hash = data.archive_file.presigned_url_lambda.output_base64sha256

  runtime = "python3.11"
  handler = "handler.lambda_handler"

  environment {
    variables = {
      UPLOAD_BUCKET       = aws_s3_bucket.upload_bucket.id
      RATE_LIMIT_TABLE    = aws_dynamodb_table.rate_limit_table.name
      JOBS_TABLE          = aws_dynamodb_table.jobs_table.name
      STATE_MACHINE_ARN   = aws_sfn_state_machine.processing_workflow.arn
      MAX_UPLOADS_PER_DAY = var.max_uploads_per_day
      MAX_FILE_SIZE       = var.max_file_size
    }
  }

  depends_on = [aws_cloudwatch_log_group.presigned_url_logs]
}

resource "aws_lambda_function" "status_handler" {
  function_name = "${var.project_name}-status-${var.environment}"
  role          = aws_iam_role.status_lambda_role.arn
  timeout       = 15
  memory_size   = 256

  filename         = data.archive_file.status_lambda.output_path
  source_code_hash = data.archive_file.status_lambda.output_base64sha256

  runtime = "python3.11"
  handler = "status_handler.lambda_handler"

  environment {
    variables = {
      JOBS_TABLE = aws_dynamodb_table.jobs_table.name
    }
  }

  depends_on = [aws_cloudwatch_log_group.status_logs]
}

resource "aws_lambda_function" "start_workflow" {
  function_name = "${var.project_name}-start-workflow-${var.environment}"
  role          = aws_iam_role.start_workflow_lambda_role.arn
  timeout       = 30
  memory_size   = 256

  filename         = data.archive_file.start_workflow_lambda.output_path
  source_code_hash = data.archive_file.start_workflow_lambda.output_base64sha256

  runtime = "python3.11"
  handler = "start_workflow.lambda_handler"

  environment {
    variables = {
      STATE_MACHINE_ARN = aws_sfn_state_machine.processing_workflow.arn
      JOBS_TABLE        = aws_dynamodb_table.jobs_table.name
    }
  }

  depends_on = [aws_cloudwatch_log_group.start_workflow_logs]
}

# Lambda Functions - Processing (Container Images)
resource "aws_lambda_function" "video_converter" {
  function_name = "${var.project_name}-video-converter-${var.environment}"
  role          = aws_iam_role.video_converter_lambda_role.arn
  timeout       = 300 # 5 minutes
  memory_size   = 3008
  package_type  = "Image"

  image_uri = "${aws_ecr_repository.video_converter.repository_url}:latest"

  ephemeral_storage {
    size = 5120 # 5 GB
  }

  depends_on = [aws_cloudwatch_log_group.video_converter_logs]
}

resource "aws_lambda_function" "vocal_separator" {
  function_name = "${var.project_name}-vocal-separator-${var.environment}"
  role          = aws_iam_role.vocal_separator_lambda_role.arn
  timeout       = 900 # 15 minutes
  memory_size   = 10240
  package_type  = "Image"

  image_uri = "${aws_ecr_repository.vocal_separator.repository_url}:latest"

  ephemeral_storage {
    size = 10240 # 10 GB
  }

  depends_on = [aws_cloudwatch_log_group.vocal_separator_logs]
}

resource "aws_lambda_function" "voice_converter" {
  function_name = "${var.project_name}-voice-converter-${var.environment}"
  role          = aws_iam_role.voice_converter_lambda_role.arn
  timeout       = 600 # 10 minutes
  memory_size   = 10240
  package_type  = "Image"

  image_uri = "${aws_ecr_repository.voice_converter.repository_url}:latest"

  ephemeral_storage {
    size = 10240 # 10 GB
  }

  depends_on = [aws_cloudwatch_log_group.voice_converter_logs]
}

resource "aws_lambda_function" "cleanup" {
  function_name = "${var.project_name}-cleanup-${var.environment}"
  role          = aws_iam_role.cleanup_lambda_role.arn
  timeout       = 60
  memory_size   = 512

  filename         = data.archive_file.cleanup_lambda.output_path
  source_code_hash = data.archive_file.cleanup_lambda.output_base64sha256

  runtime = "python3.11"
  handler = "handler.lambda_handler"

  environment {
    variables = {
      OUTPUT_BUCKET      = aws_s3_bucket.output_bucket.id
      JOBS_TABLE         = aws_dynamodb_table.jobs_table.name
      PROCESSING_BUCKET  = aws_s3_bucket.upload_bucket.id
    }
  }

  depends_on = [aws_cloudwatch_log_group.cleanup_logs]
}

# Lambda Permission for S3 to invoke start_workflow
resource "aws_lambda_permission" "allow_s3_invoke_start_workflow" {
  statement_id  = "AllowS3Invoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.start_workflow.function_name
  principal     = "s3.amazonaws.com"
  source_arn    = aws_s3_bucket.upload_bucket.arn
}

# Lambda Permissions for API Gateway
resource "aws_lambda_permission" "allow_api_gateway_presigned_url" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.presigned_url.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.vocal_remover_api.execution_arn}/*/*"
}

resource "aws_lambda_permission" "allow_api_gateway_status" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.status_handler.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.vocal_remover_api.execution_arn}/*/*"
}

# Archive Lambda source code
data "archive_file" "presigned_url_lambda" {
  type        = "zip"
  source_dir  = "${path.module}/../lambda/presigned-url"
  output_path = "${path.module}/.builds/presigned-url.zip"
  excludes    = ["start_workflow.py", "status_handler.py"]
}

data "archive_file" "status_lambda" {
  type        = "zip"
  source_file = "${path.module}/../lambda/presigned-url/status_handler.py"
  output_path = "${path.module}/.builds/status.zip"
}

data "archive_file" "start_workflow_lambda" {
  type        = "zip"
  source_file = "${path.module}/../lambda/presigned-url/start_workflow.py"
  output_path = "${path.module}/.builds/start-workflow.zip"
}

data "archive_file" "cleanup_lambda" {
  type        = "zip"
  source_dir  = "${path.module}/../lambda/cleanup"
  output_path = "${path.module}/.builds/cleanup.zip"
}
