output "api_endpoint" {
  description = "API Gateway endpoint URL"
  value       = aws_apigatewayv2_api.vocal_remover_api.api_endpoint
}

output "api_invoke_url" {
  description = "API Gateway invoke URL"
  value       = "${aws_apigatewayv2_api.vocal_remover_api.api_endpoint}/prod"
}

output "cloudfront_domain" {
  description = "CloudFront distribution domain name"
  value       = aws_cloudfront_distribution.frontend.domain_name
}

output "cloudfront_url" {
  description = "CloudFront distribution URL"
  value       = "https://${aws_cloudfront_distribution.frontend.domain_name}"
}

output "frontend_bucket" {
  description = "Frontend S3 bucket name"
  value       = aws_s3_bucket.frontend_bucket.id
}

output "upload_bucket" {
  description = "Upload S3 bucket name"
  value       = aws_s3_bucket.upload_bucket.id
}

output "output_bucket" {
  description = "Output S3 bucket name"
  value       = aws_s3_bucket.output_bucket.id
}

output "ecr_video_converter_url" {
  description = "ECR repository URL for video converter"
  value       = aws_ecr_repository.video_converter.repository_url
}

output "ecr_vocal_separator_url" {
  description = "ECR repository URL for vocal separator"
  value       = aws_ecr_repository.vocal_separator.repository_url
}

output "ecr_voice_converter_url" {
  description = "ECR repository URL for voice converter"
  value       = aws_ecr_repository.voice_converter.repository_url
}

output "state_machine_arn" {
  description = "Step Functions state machine ARN"
  value       = aws_sfn_state_machine.processing_workflow.arn
}

output "jobs_table_name" {
  description = "DynamoDB jobs table name"
  value       = aws_dynamodb_table.jobs_table.name
}

output "rate_limit_table_name" {
  description = "DynamoDB rate limit table name"
  value       = aws_dynamodb_table.rate_limit_table.name
}
