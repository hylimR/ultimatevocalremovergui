#!/bin/bash
set -e

# Deployment script for Vocal Remover Web Application
# This script deploys the entire serverless infrastructure to AWS

echo "🚀 Starting deployment..."

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
AWS_REGION=${AWS_REGION:-us-east-1}
ENVIRONMENT=${ENVIRONMENT:-prod}
PROJECT_NAME="vocal-remover"

echo -e "${YELLOW}Configuration:${NC}"
echo "  AWS Region: $AWS_REGION"
echo "  Environment: $ENVIRONMENT"
echo "  Project: $PROJECT_NAME"
echo ""

# Step 1: Initialize Terraform
echo -e "${GREEN}Step 1: Initializing Terraform...${NC}"
cd terraform
terraform init

# Step 2: Plan Terraform deployment
echo -e "${GREEN}Step 2: Planning Terraform deployment...${NC}"
terraform plan \
  -var="aws_region=$AWS_REGION" \
  -var="environment=$ENVIRONMENT" \
  -var="project_name=$PROJECT_NAME" \
  -out=tfplan

# Step 3: Apply Terraform (create infrastructure)
echo -e "${GREEN}Step 3: Applying Terraform (creating infrastructure)...${NC}"
terraform apply tfplan

# Get outputs
API_ENDPOINT=$(terraform output -raw api_invoke_url)
FRONTEND_BUCKET=$(terraform output -raw frontend_bucket)
ECR_VIDEO_CONVERTER=$(terraform output -raw ecr_video_converter_url)
ECR_VOCAL_SEPARATOR=$(terraform output -raw ecr_vocal_separator_url)
ECR_VOICE_CONVERTER=$(terraform output -raw ecr_voice_converter_url)
CLOUDFRONT_URL=$(terraform output -raw cloudfront_url)

cd ..

echo -e "${GREEN}Step 4: Building and pushing Docker images...${NC}"

# Login to ECR
echo "Logging in to ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_VIDEO_CONVERTER

# Build and push Video Converter
echo -e "${YELLOW}Building Video Converter Lambda...${NC}"
docker build -f docker/Dockerfile.video-converter -t $ECR_VIDEO_CONVERTER:latest .
docker push $ECR_VIDEO_CONVERTER:latest

# Build and push Vocal Separator
echo -e "${YELLOW}Building Vocal Separator Lambda...${NC}"
docker build -f docker/Dockerfile.vocal-separator -t $ECR_VOCAL_SEPARATOR:latest .
docker push $ECR_VOCAL_SEPARATOR:latest

# Build and push Voice Converter
echo -e "${YELLOW}Building Voice Converter Lambda...${NC}"
docker build -f docker/Dockerfile.voice-converter -t $ECR_VOICE_CONVERTER:latest .
docker push $ECR_VOICE_CONVERTER:latest

# Step 5: Update Lambda functions to use new images
echo -e "${GREEN}Step 5: Updating Lambda functions...${NC}"
aws lambda update-function-code \
  --function-name $PROJECT_NAME-video-converter-$ENVIRONMENT \
  --image-uri $ECR_VIDEO_CONVERTER:latest \
  --region $AWS_REGION

aws lambda update-function-code \
  --function-name $PROJECT_NAME-vocal-separator-$ENVIRONMENT \
  --image-uri $ECR_VOCAL_SEPARATOR:latest \
  --region $AWS_REGION

aws lambda update-function-code \
  --function-name $PROJECT_NAME-voice-converter-$ENVIRONMENT \
  --image-uri $ECR_VOICE_CONVERTER:latest \
  --region $AWS_REGION

# Wait for functions to update
echo "Waiting for Lambda functions to update..."
sleep 10

# Step 6: Update frontend with API endpoint
echo -e "${GREEN}Step 6: Updating frontend configuration...${NC}"
sed -i.bak "s|API_ENDPOINT: '.*'|API_ENDPOINT: '$API_ENDPOINT'|" web-ui/app.js

# Step 7: Deploy frontend to S3
echo -e "${GREEN}Step 7: Deploying frontend to S3...${NC}"
aws s3 sync web-ui/ s3://$FRONTEND_BUCKET/ --delete

# Step 8: Invalidate CloudFront cache
echo -e "${GREEN}Step 8: Invalidating CloudFront cache...${NC}"
DISTRIBUTION_ID=$(aws cloudfront list-distributions --query "DistributionList.Items[?Origins.Items[?DomainName=='$FRONTEND_BUCKET.s3.amazonaws.com']].Id" --output text)
if [ ! -z "$DISTRIBUTION_ID" ]; then
  aws cloudfront create-invalidation --distribution-id $DISTRIBUTION_ID --paths "/*"
fi

echo ""
echo -e "${GREEN}✅ Deployment complete!${NC}"
echo ""
echo "📱 Your application is available at:"
echo -e "   ${GREEN}$CLOUDFRONT_URL${NC}"
echo ""
echo "🔗 API Endpoint:"
echo -e "   ${GREEN}$API_ENDPOINT${NC}"
echo ""
echo -e "${YELLOW}⚠️  Important Next Steps:${NC}"
echo "1. Add ONNX vocal separation models to the vocal-separator Lambda"
echo "2. Add RVC voice models to the voice-converter Lambda (if using voice conversion)"
echo "3. Test the application with a sample audio/video file"
echo "4. Configure custom domain (optional)"
echo "5. Enable WAF for additional security (set enable_waf=true in variables.tf)"
echo ""
