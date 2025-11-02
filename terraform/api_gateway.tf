# API Gateway HTTP API
resource "aws_apigatewayv2_api" "vocal_remover_api" {
  name          = "${var.project_name}-api-${var.environment}"
  protocol_type = "HTTP"

  cors_configuration {
    allow_origins = ["*"]
    allow_methods = ["GET", "POST", "OPTIONS"]
    allow_headers = ["*"]
    max_age       = 300
  }
}

# API Gateway Stage
resource "aws_apigatewayv2_stage" "prod" {
  api_id      = aws_apigatewayv2_api.vocal_remover_api.id
  name        = "prod"
  auto_deploy = true

  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.api_gateway_logs.arn
    format = jsonencode({
      requestId      = "$context.requestId"
      ip             = "$context.identity.sourceIp"
      requestTime    = "$context.requestTime"
      httpMethod     = "$context.httpMethod"
      routeKey       = "$context.routeKey"
      status         = "$context.status"
      protocol       = "$context.protocol"
      responseLength = "$context.responseLength"
    })
  }

  default_route_settings {
    throttling_burst_limit = 100
    throttling_rate_limit  = 50
  }
}

# CloudWatch Log Group for API Gateway
resource "aws_cloudwatch_log_group" "api_gateway_logs" {
  name              = "/aws/apigateway/${var.project_name}-${var.environment}"
  retention_in_days = 7
}

# API Gateway Integration for Presigned URL
resource "aws_apigatewayv2_integration" "presigned_url_integration" {
  api_id           = aws_apigatewayv2_api.vocal_remover_api.id
  integration_type = "AWS_PROXY"

  integration_uri    = aws_lambda_function.presigned_url.invoke_arn
  integration_method = "POST"
}

# API Gateway Route for Upload
resource "aws_apigatewayv2_route" "upload_route" {
  api_id    = aws_apigatewayv2_api.vocal_remover_api.id
  route_key = "POST /upload"
  target    = "integrations/${aws_apigatewayv2_integration.presigned_url_integration.id}"
}

# API Gateway Integration for Status
resource "aws_apigatewayv2_integration" "status_integration" {
  api_id           = aws_apigatewayv2_api.vocal_remover_api.id
  integration_type = "AWS_PROXY"

  integration_uri    = aws_lambda_function.status_handler.invoke_arn
  integration_method = "POST"
}

# API Gateway Route for Status
resource "aws_apigatewayv2_route" "status_route" {
  api_id    = aws_apigatewayv2_api.vocal_remover_api.id
  route_key = "GET /status/{jobId}"
  target    = "integrations/${aws_apigatewayv2_integration.status_integration.id}"
}

# WAF Web ACL for Rate Limiting (Optional but recommended)
resource "aws_wafv2_web_acl" "api_rate_limit" {
  count = var.enable_waf ? 1 : 0

  name  = "${var.project_name}-rate-limit-${var.environment}"
  scope = "REGIONAL"

  default_action {
    allow {}
  }

  rule {
    name     = "RateLimitRule"
    priority = 1

    action {
      block {}
    }

    statement {
      rate_based_statement {
        limit              = 100
        aggregate_key_type = "IP"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "${var.project_name}-rate-limit"
      sampled_requests_enabled   = true
    }
  }

  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name                = "${var.project_name}-waf"
    sampled_requests_enabled   = true
  }
}

# Associate WAF with API Gateway
resource "aws_wafv2_web_acl_association" "api_waf_association" {
  count = var.enable_waf ? 1 : 0

  resource_arn = aws_apigatewayv2_stage.prod.arn
  web_acl_arn  = aws_wafv2_web_acl.api_rate_limit[0].arn
}
