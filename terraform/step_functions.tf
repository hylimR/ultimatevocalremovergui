# Step Functions State Machine
resource "aws_sfn_state_machine" "processing_workflow" {
  name     = "${var.project_name}-processing-${var.environment}"
  role_arn = aws_iam_role.step_functions_role.arn

  definition = templatefile("${path.module}/../lambda/step-functions-definition.json", {
    JobsTable               = aws_dynamodb_table.jobs_table.name
    VideoConverterFunction  = aws_lambda_function.video_converter.arn
    VocalSeparatorFunction  = aws_lambda_function.vocal_separator.arn
    VoiceConverterFunction  = aws_lambda_function.voice_converter.arn
    CleanupFunction         = aws_lambda_function.cleanup.arn
  })

  logging_configuration {
    log_destination        = "${aws_cloudwatch_log_group.state_machine_logs.arn}:*"
    include_execution_data = true
    level                  = "ALL"
  }

  tracing_configuration {
    enabled = true
  }
}
