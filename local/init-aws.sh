#!/bin/bash
# LocalStack initialization script
# This runs when LocalStack is ready

set -e

echo "🔧 Initializing LocalStack resources..."

# Set AWS config for LocalStack
export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export AWS_DEFAULT_REGION=us-east-1

# Helper function to run AWS CLI against LocalStack
aws_local() {
    aws --endpoint-url=http://localhost:4566 "$@"
}

# Create S3 buckets
echo "📦 Creating S3 buckets..."
aws_local s3 mb s3://vocal-remover-upload-local || true
aws_local s3 mb s3://vocal-remover-output-local || true
aws_local s3 mb s3://vocal-remover-frontend-local || true

# Create DynamoDB tables
echo "💾 Creating DynamoDB tables..."

# Jobs table
aws_local dynamodb create-table \
    --table-name vocal-remover-jobs-local \
    --attribute-definitions \
        AttributeName=job_id,AttributeType=S \
    --key-schema \
        AttributeName=job_id,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST \
    --region us-east-1 || true

# Rate limit table
aws_local dynamodb create-table \
    --table-name vocal-remover-rate-limit-local \
    --attribute-definitions \
        AttributeName=client_ip,AttributeType=S \
        AttributeName=date,AttributeType=S \
    --key-schema \
        AttributeName=client_ip,KeyType=HASH \
        AttributeName=date,KeyType=RANGE \
    --billing-mode PAY_PER_REQUEST \
    --region us-east-1 || true

echo "✅ LocalStack initialization complete!"
echo ""
echo "📍 LocalStack services available at: http://localhost:4566"
echo "   S3 buckets:"
echo "     - vocal-remover-upload-local"
echo "     - vocal-remover-output-local"
echo "     - vocal-remover-frontend-local"
echo ""
echo "   DynamoDB tables:"
echo "     - vocal-remover-jobs-local"
echo "     - vocal-remover-rate-limit-local"
echo ""
