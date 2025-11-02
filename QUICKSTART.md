# 🚀 Quick Start Guide

Get the Vocal Remover Web App running in 15 minutes!

## Prerequisites Checklist

- [ ] AWS Account with admin access
- [ ] AWS CLI installed and configured (`aws configure`)
- [ ] Terraform >= 1.0 installed
- [ ] Docker installed and running
- [ ] Bash shell (Linux/macOS/WSL)

## Step 1: Verify Prerequisites

```bash
# Check AWS CLI
aws --version
aws sts get-caller-identity

# Check Terraform
terraform --version

# Check Docker
docker --version
docker ps
```

## Step 2: Clone and Setup

```bash
# Clone repository
git clone <your-repo-url>
cd ultimatevocalremovergui

# Make deploy script executable
chmod +x deploy.sh
```

## Step 3: Deploy Infrastructure

```bash
# Set AWS region (optional, defaults to us-east-1)
export AWS_REGION=us-east-1

# Run deployment (takes ~10-15 minutes)
./deploy.sh
```

The script will:
1. ✅ Initialize Terraform
2. ✅ Create AWS infrastructure (S3, Lambda, API Gateway, etc.)
3. ✅ Build Docker images for Lambda functions
4. ✅ Push images to ECR
5. ✅ Deploy frontend to S3
6. ✅ Invalidate CloudFront cache

## Step 4: Get Your App URL

At the end of deployment, you'll see:

```
✅ Deployment complete!

📱 Your application is available at:
   https://d1234567890.cloudfront.net

🔗 API Endpoint:
   https://abcdefghij.execute-api.us-east-1.amazonaws.com/prod
```

## Step 5: Test the Application

1. Open the CloudFront URL in your browser
2. Upload a sample audio file (< 500MB, < 10 min)
3. Select model: "MDX-Net Karaoke"
4. Click "Process Audio"
5. Wait 1-3 minutes
6. Download vocals and instrumental!

## Common Issues

### Issue: "Terraform not found"
```bash
# Install Terraform
wget https://releases.hashicorp.com/terraform/1.6.0/terraform_1.6.0_linux_amd64.zip
unzip terraform_1.6.0_linux_amd64.zip
sudo mv terraform /usr/local/bin/
```

### Issue: "AWS credentials not configured"
```bash
aws configure
# Enter your AWS Access Key ID, Secret Key, and region
```

### Issue: "Docker not running"
```bash
# Start Docker
sudo systemctl start docker  # Linux
# or
open -a Docker  # macOS
```

### Issue: "Lambda returns error"
Check the placeholder implementations in:
- `lambda/vocal-separator/handler.py` (needs actual ONNX model integration)
- `lambda/voice-converter/handler.py` (needs RVC model integration)

These are placeholder implementations. For production:
1. Add ONNX models to Docker images
2. Integrate actual inference code from `lib_v5/`
3. Rebuild and redeploy

## What's Next?

### Add AI Models

The current deployment uses placeholder implementations. To add real vocal separation:

1. **Download MDX-Net ONNX models**:
   ```bash
   # Example: Download from UVR model repository
   mkdir -p models/onnx
   # Add your .onnx files here
   ```

2. **Update Dockerfile**:
   ```dockerfile
   # In docker/Dockerfile.vocal-separator
   COPY models/onnx/MDX23C-8KFFT-InstVoc_HQ.onnx /opt/models/
   ```

3. **Update handler**:
   ```python
   # In lambda/vocal-separator/handler.py
   # Replace simple_separation() with actual ONNX inference
   # Use code from lib_v5/mdxnet.py
   ```

4. **Rebuild and deploy**:
   ```bash
   ./deploy.sh
   ```

### Configure Custom Domain

1. Request SSL certificate in ACM (us-east-1)
2. Update CloudFront distribution with custom domain
3. Create Route53 record pointing to CloudFront

### Increase Rate Limits

```bash
cd terraform
# Edit variables.tf
# Change max_uploads_per_day from 5 to 10 (or any number)
terraform apply
```

### Enable WAF

```bash
cd terraform
# Edit terraform.tfvars
echo 'enable_waf = true' >> terraform.tfvars
terraform apply
```

## Cleanup

To remove all resources:

```bash
cd terraform
terraform destroy
```

⚠️ Warning: This will delete all data, including uploaded files and processing results.

## Support

- 📖 Full documentation: See `README_WEB.md`
- 🏗️ Architecture details: See `ARCHITECTURE.md`
- 🐛 Issues: Create GitHub issue
- 💬 Questions: Check CloudWatch logs

## Cost Estimate

With the free tier and low usage:
- **First year**: ~$0-5/month (mostly free tier)
- **After free tier**: ~$6-15/month for 1000 requests
- **High usage**: ~$40-80/month for 10,000 requests

Track costs: https://console.aws.amazon.com/billing/

---

**Ready to deploy?** Run `./deploy.sh` and you're live in 15 minutes! 🚀
