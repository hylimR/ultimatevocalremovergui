# 🎵 Vocal Remover Web Application

A serverless web application for AI-powered vocal removal and voice conversion, built on AWS with a cost-effective architecture.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Cost Estimation](#cost-estimation)
- [Prerequisites](#prerequisites)
- [Deployment](#deployment)
- [Configuration](#configuration)
- [Usage](#usage)
- [Development](#development)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

This application provides a simple web interface for:
- **Vocal Separation**: Extract vocals and instrumentals from audio/video files
- **Voice Conversion**: Convert vocals to AI-cloned voices using RVC models
- **Video Support**: Automatically converts video files to audio
- **Rate Limiting**: Built-in protection against abuse
- **Cost-Effective**: Serverless architecture that scales to zero

## 🏗️ Architecture

```
┌─────────────────┐
│   CloudFront    │  Frontend (Static Website)
│    + S3         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  API Gateway    │  REST API
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Lambda         │  Presigned URL + Rate Limiting
│  (Upload)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  S3             │  File Upload
│  (Upload)       │
└────────┬────────┘
         │
         ▼ (S3 Event)
┌─────────────────┐
│  Lambda         │  Start Workflow
│  (Trigger)      │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│         Step Functions State Machine        │
│  ┌──────────────────────────────────────┐  │
│  │  1. Video → Audio (FFmpeg)           │  │
│  │  2. Vocal Separation (ONNX)          │  │
│  │  3. Voice Conversion (RVC)           │  │
│  │  4. Cleanup & Generate URLs          │  │
│  └──────────────────────────────────────┘  │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
         ┌─────────────────┐
         │  S3 (Output)    │  Download Links
         └─────────────────┘
```

### Components

- **Frontend**: React/Vanilla JS SPA hosted on S3 + CloudFront
- **API Gateway**: HTTP API for upload and status endpoints
- **Lambda Functions**:
  - Presigned URL generator (with rate limiting)
  - S3 event trigger → Start workflow
  - Video converter (FFmpeg in container)
  - Vocal separator (ONNX Runtime)
  - Voice converter (RVC)
  - Cleanup & finalize
- **Step Functions**: Orchestrates processing workflow
- **DynamoDB**: Rate limiting and job tracking
- **S3 Buckets**: Upload, processing, and output storage

## ✨ Features

### Frontend
- ✅ Drag-and-drop file upload
- ✅ Client-side validation (size, duration, format)
- ✅ Real-time progress tracking
- ✅ Model selection (MDX-Net, VR, Demucs)
- ✅ Voice conversion model picker
- ✅ Audio preview and download
- ✅ Responsive design

### Backend
- ✅ Video to audio conversion (FFmpeg)
- ✅ Vocal separation (MDX-Net, VR, Demucs)
- ✅ Voice conversion (RVC models)
- ✅ Rate limiting (5 uploads/day per IP)
- ✅ File validation (max 500MB, 10 minutes)
- ✅ Automatic cleanup (7-day retention)
- ✅ Error handling and retry logic

## 💰 Cost Estimation

### Monthly Costs (based on usage)

| Usage Level | Requests/Month | Estimated Cost |
|-------------|----------------|----------------|
| **Low** | 100 | $2-5 |
| **Medium** | 1,000 | $6-15 |
| **High** | 10,000 | $40-80 |
| **Very High** | 100,000 | $400-800 |

### Cost Breakdown (1,000 requests/month)

| Service | Usage | Cost |
|---------|-------|------|
| S3 Storage | 100 GB | $2.30 |
| S3 Requests | 10K PUT, 20K GET | $0.15 |
| CloudFront | 10 GB transfer | $0.85 |
| API Gateway | 1K requests | $0.004 |
| Step Functions | 1K executions | $1.00 |
| Lambda | ~120K GB-seconds | $2.00 |
| DynamoDB | On-demand | $0.25 |
| **Total** | | **~$6.50/month** |

### Cost Optimization Tips

1. **Use Express Workflows**: 96% cheaper than Standard ($1 vs $25 per million)
2. **Enable S3 Lifecycle Policies**: Auto-delete old files
3. **Use ONNX Runtime**: 50% smaller than PyTorch, 3x faster
4. **Implement Chunking**: Process long audio in parallel
5. **Cache Models**: Use EFS or container layers
6. **Set Reserved Concurrency**: Prevent runaway costs

## 📦 Prerequisites

### Required
- **AWS Account** with appropriate permissions
- **AWS CLI** configured with credentials
- **Terraform** >= 1.0
- **Docker** installed and running
- **Bash** shell (Linux/macOS) or Git Bash (Windows)

### Optional
- **Custom Domain** (for CloudFront)
- **SSL Certificate** (for custom domain via ACM)

## 🚀 Deployment

### Quick Start

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd ultimatevocalremovergui

# 2. Make deploy script executable
chmod +x deploy.sh

# 3. Set AWS region (optional, defaults to us-east-1)
export AWS_REGION=us-east-1

# 4. Run deployment
./deploy.sh
```

### Manual Deployment Steps

If you prefer manual deployment:

#### 1. Deploy Infrastructure

```bash
cd terraform
terraform init
terraform plan -var="aws_region=us-east-1"
terraform apply
```

#### 2. Build and Push Docker Images

```bash
# Get ECR URLs from Terraform output
ECR_VIDEO=$(terraform output -raw ecr_video_converter_url)
ECR_VOCAL=$(terraform output -raw ecr_vocal_separator_url)
ECR_VOICE=$(terraform output -raw ecr_voice_converter_url)

# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin $ECR_VIDEO

# Build and push
cd ..
docker build -f docker/Dockerfile.video-converter -t $ECR_VIDEO:latest .
docker push $ECR_VIDEO:latest

docker build -f docker/Dockerfile.vocal-separator -t $ECR_VOCAL:latest .
docker push $ECR_VOCAL:latest

docker build -f docker/Dockerfile.voice-converter -t $ECR_VOICE:latest .
docker push $ECR_VOICE:latest
```

#### 3. Update Lambda Functions

```bash
aws lambda update-function-code \
  --function-name vocal-remover-video-converter-prod \
  --image-uri $ECR_VIDEO:latest

aws lambda update-function-code \
  --function-name vocal-remover-vocal-separator-prod \
  --image-uri $ECR_VOCAL:latest

aws lambda update-function-code \
  --function-name vocal-remover-voice-converter-prod \
  --image-uri $ECR_VOICE:latest
```

#### 4. Deploy Frontend

```bash
# Get API endpoint and bucket name
API_ENDPOINT=$(cd terraform && terraform output -raw api_invoke_url)
FRONTEND_BUCKET=$(cd terraform && terraform output -raw frontend_bucket)

# Update frontend config
sed -i "s|API_ENDPOINT: '.*'|API_ENDPOINT: '$API_ENDPOINT'|" web-ui/app.js

# Upload to S3
aws s3 sync web-ui/ s3://$FRONTEND_BUCKET/ --delete
```

#### 5. Get CloudFront URL

```bash
cd terraform
terraform output cloudfront_url
```

## ⚙️ Configuration

### Environment Variables

Edit `terraform/variables.tf`:

```hcl
variable "max_uploads_per_day" {
  default = 5  # Change to your desired limit
}

variable "max_file_size" {
  default = 524288000  # 500MB in bytes
}

variable "enable_waf" {
  default = false  # Set to true for WAF protection
}
```

### Add AI Models

#### Vocal Separation Models

1. Download ONNX models (e.g., from UVR model repository)
2. Add to Dockerfile:

```dockerfile
# In docker/Dockerfile.vocal-separator
COPY models/onnx/MDX23C-8KFFT-InstVoc_HQ.onnx /opt/models/
COPY models/onnx/Kim_Vocal_2.onnx /opt/models/
```

3. Rebuild and redeploy

#### Voice Conversion Models (RVC)

1. Train or download RVC models
2. Export to ONNX format
3. Add to Dockerfile:

```dockerfile
# In docker/Dockerfile.voice-converter
COPY models/rvc/default_voice.onnx /opt/models/rvc/
COPY models/rvc/deep_male.onnx /opt/models/rvc/
```

4. Update `lambda/voice-converter/handler.py` with model names

## 🎮 Usage

### Upload a File

1. Open the CloudFront URL in your browser
2. Drag-and-drop or click to select an audio/video file
3. Choose separation model (default: MDX-Net Karaoke)
4. (Optional) Select voice conversion model
5. Click "Process Audio"
6. Wait for processing (1-3 minutes for 3-minute song)
7. Download vocals and instrumental

### API Usage

#### Get Upload URL

```bash
curl -X POST https://your-api-endpoint/prod/upload \
  -H "Content-Type: application/json" \
  -d '{
    "fileName": "song.mp3",
    "fileSize": 5000000,
    "contentType": "audio/mpeg",
    "model": "mdx_karaoke",
    "voiceModel": "none"
  }'
```

Response:
```json
{
  "jobId": "uuid-here",
  "uploadUrl": "https://s3-presigned-url...",
  "s3Key": "uploads/..."
}
```

#### Check Status

```bash
curl https://your-api-endpoint/prod/status/{jobId}
```

Response:
```json
{
  "job_id": "uuid",
  "status": "COMPLETED",
  "progress": 100,
  "results": {
    "vocalsUrl": "https://...",
    "instrumentalUrl": "https://..."
  }
}
```

## 🛠️ Development

### Local Testing

#### Test Lambda Locally

```bash
# Install SAM CLI
pip install aws-sam-cli

# Test presigned URL function
sam local invoke PresignedUrlFunction \
  --event test-events/upload.json
```

#### Test Frontend Locally

```bash
cd web-ui
python -m http.server 8000

# Open http://localhost:8000
```

### Update Lambda Code

```bash
# After modifying Lambda code
cd terraform
terraform apply

# For container images, rebuild and push
docker build -f docker/Dockerfile.vocal-separator -t $ECR_VOCAL:latest .
docker push $ECR_VOCAL:latest

aws lambda update-function-code \
  --function-name vocal-remover-vocal-separator-prod \
  --image-uri $ECR_VOCAL:latest
```

## 🐛 Troubleshooting

### Issue: Lambda timeout

**Solution**: Increase timeout in `terraform/lambda.tf`

```hcl
resource "aws_lambda_function" "vocal_separator" {
  timeout = 900  # Increase to 15 minutes
}
```

### Issue: Out of memory

**Solution**: Increase memory in `terraform/lambda.tf`

```hcl
resource "aws_lambda_function" "vocal_separator" {
  memory_size = 10240  # Use max 10GB
}
```

### Issue: Rate limit exceeded

**Solution**: Adjust in `terraform/variables.tf`

```hcl
variable "max_uploads_per_day" {
  default = 10  # Increase limit
}
```

### Issue: Frontend can't connect to API

**Solution**: Check CORS configuration and API endpoint

```bash
# Verify API endpoint in web-ui/app.js
grep API_ENDPOINT web-ui/app.js

# Check API Gateway CORS settings
cd terraform
terraform show | grep cors_configuration
```

### View Logs

```bash
# Lambda logs
aws logs tail /aws/lambda/vocal-remover-vocal-separator-prod --follow

# Step Functions execution
aws stepfunctions describe-execution \
  --execution-arn <execution-arn>

# API Gateway logs
aws logs tail /aws/apigateway/vocal-remover-prod --follow
```

## 📊 Monitoring

### CloudWatch Dashboards

Create custom dashboard:

```bash
aws cloudwatch put-dashboard \
  --dashboard-name vocal-remover \
  --dashboard-body file://monitoring/dashboard.json
```

### Metrics to Monitor

- **Lambda Invocations**: Number of processing jobs
- **Lambda Duration**: Processing time per job
- **Lambda Errors**: Failed jobs
- **API Gateway 4xx/5xx**: Client/server errors
- **Step Functions Failed Executions**: Workflow failures
- **DynamoDB Consumed Capacity**: Database usage

## 🔒 Security

### Best Practices

1. **Enable WAF**: Set `enable_waf = true`
2. **Use VPC**: For sensitive workloads
3. **Encrypt S3**: Enable default encryption
4. **Rotate Credentials**: Use IAM roles, not keys
5. **Enable CloudTrail**: Audit all API calls
6. **Set Bucket Policies**: Restrict public access
7. **Use HTTPS Only**: Enforce SSL/TLS

### Enable S3 Encryption

```hcl
# In terraform/main.tf
resource "aws_s3_bucket_server_side_encryption_configuration" "upload_bucket_encryption" {
  bucket = aws_s3_bucket.upload_bucket.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}
```

## 📝 License

See the main README.md for license information.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📧 Support

For issues and questions:
- Create an issue on GitHub
- Check existing documentation
- Review AWS CloudWatch logs

---

**Built with ❤️ using AWS Serverless**
