# IAM Role for Presigned URL Lambda
resource "aws_iam_role" "presigned_url_lambda_role" {
  name = "${var.project_name}-presigned-url-lambda-${var.environment}"

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

resource "aws_iam_role_policy_attachment" "presigned_url_lambda_basic" {
  role       = aws_iam_role.presigned_url_lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "presigned_url_lambda_policy" {
  name = "${var.project_name}-presigned-url-policy-${var.environment}"
  role = aws_iam_role.presigned_url_lambda_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:GetObject"
        ]
        Resource = "${aws_s3_bucket.upload_bucket.arn}/*"
      },
      {
        Effect = "Allow"
        Action = [
          "dynamodb:GetItem",
          "dynamodb:PutItem",
          "dynamodb:UpdateItem"
        ]
        Resource = [
          aws_dynamodb_table.rate_limit_table.arn,
          aws_dynamodb_table.jobs_table.arn
        ]
      }
    ]
  })
}

# IAM Role for Status Lambda
resource "aws_iam_role" "status_lambda_role" {
  name = "${var.project_name}-status-lambda-${var.environment}"

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

resource "aws_iam_role_policy_attachment" "status_lambda_basic" {
  role       = aws_iam_role.status_lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "status_lambda_policy" {
  name = "${var.project_name}-status-policy-${var.environment}"
  role = aws_iam_role.status_lambda_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "dynamodb:GetItem"
        ]
        Resource = aws_dynamodb_table.jobs_table.arn
      }
    ]
  })
}

# IAM Role for Start Workflow Lambda
resource "aws_iam_role" "start_workflow_lambda_role" {
  name = "${var.project_name}-start-workflow-lambda-${var.environment}"

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

resource "aws_iam_role_policy_attachment" "start_workflow_lambda_basic" {
  role       = aws_iam_role.start_workflow_lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "start_workflow_lambda_policy" {
  name = "${var.project_name}-start-workflow-policy-${var.environment}"
  role = aws_iam_role.start_workflow_lambda_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "states:StartExecution"
        ]
        Resource = aws_sfn_state_machine.processing_workflow.arn
      },
      {
        Effect = "Allow"
        Action = [
          "dynamodb:GetItem",
          "dynamodb:UpdateItem"
        ]
        Resource = aws_dynamodb_table.jobs_table.arn
      }
    ]
  })
}

# IAM Role for Video Converter Lambda
resource "aws_iam_role" "video_converter_lambda_role" {
  name = "${var.project_name}-video-converter-lambda-${var.environment}"

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

resource "aws_iam_role_policy_attachment" "video_converter_lambda_basic" {
  role       = aws_iam_role.video_converter_lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "video_converter_lambda_policy" {
  name = "${var.project_name}-video-converter-policy-${var.environment}"
  role = aws_iam_role.video_converter_lambda_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject"
        ]
        Resource = "${aws_s3_bucket.upload_bucket.arn}/*"
      }
    ]
  })
}

# IAM Role for Vocal Separator Lambda
resource "aws_iam_role" "vocal_separator_lambda_role" {
  name = "${var.project_name}-vocal-separator-lambda-${var.environment}"

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

resource "aws_iam_role_policy_attachment" "vocal_separator_lambda_basic" {
  role       = aws_iam_role.vocal_separator_lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "vocal_separator_lambda_policy" {
  name = "${var.project_name}-vocal-separator-policy-${var.environment}"
  role = aws_iam_role.vocal_separator_lambda_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject"
        ]
        Resource = "${aws_s3_bucket.upload_bucket.arn}/*"
      }
    ]
  })
}

# IAM Role for Voice Converter Lambda
resource "aws_iam_role" "voice_converter_lambda_role" {
  name = "${var.project_name}-voice-converter-lambda-${var.environment}"

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

resource "aws_iam_role_policy_attachment" "voice_converter_lambda_basic" {
  role       = aws_iam_role.voice_converter_lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "voice_converter_lambda_policy" {
  name = "${var.project_name}-voice-converter-policy-${var.environment}"
  role = aws_iam_role.voice_converter_lambda_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject"
        ]
        Resource = "${aws_s3_bucket.upload_bucket.arn}/*"
      }
    ]
  })
}

# IAM Role for Cleanup Lambda
resource "aws_iam_role" "cleanup_lambda_role" {
  name = "${var.project_name}-cleanup-lambda-${var.environment}"

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

resource "aws_iam_role_policy_attachment" "cleanup_lambda_basic" {
  role       = aws_iam_role.cleanup_lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "cleanup_lambda_policy" {
  name = "${var.project_name}-cleanup-policy-${var.environment}"
  role = aws_iam_role.cleanup_lambda_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:CopyObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          "${aws_s3_bucket.upload_bucket.arn}/*",
          "${aws_s3_bucket.output_bucket.arn}/*",
          aws_s3_bucket.upload_bucket.arn,
          aws_s3_bucket.output_bucket.arn
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "dynamodb:GetItem",
          "dynamodb:UpdateItem"
        ]
        Resource = aws_dynamodb_table.jobs_table.arn
      }
    ]
  })
}

# IAM Role for Step Functions
resource "aws_iam_role" "step_functions_role" {
  name = "${var.project_name}-step-functions-${var.environment}"

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
}

resource "aws_iam_role_policy" "step_functions_policy" {
  name = "${var.project_name}-step-functions-policy-${var.environment}"
  role = aws_iam_role.step_functions_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "lambda:InvokeFunction"
        ]
        Resource = [
          aws_lambda_function.video_converter.arn,
          aws_lambda_function.vocal_separator.arn,
          aws_lambda_function.voice_converter.arn,
          aws_lambda_function.cleanup.arn
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "dynamodb:GetItem",
          "dynamodb:PutItem",
          "dynamodb:UpdateItem"
        ]
        Resource = aws_dynamodb_table.jobs_table.arn
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogDelivery",
          "logs:GetLogDelivery",
          "logs:UpdateLogDelivery",
          "logs:DeleteLogDelivery",
          "logs:ListLogDeliveries",
          "logs:PutResourcePolicy",
          "logs:DescribeResourcePolicies",
          "logs:DescribeLogGroups"
        ]
        Resource = "*"
      }
    ]
  })
}
