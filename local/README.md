# Local Development Files

This directory contains all the files needed to run the Vocal Remover application locally using LocalStack and Docker Compose.

## Quick Start

From the project root:

```bash
# Start local environment
./local-dev.sh start

# Open application
open http://localhost:8080

# View logs
./local-dev.sh logs

# Stop environment
./local-dev.sh stop
```

## Directory Structure

```
local/
├── api-server/
│   └── server.py           # Flask API server (replaces API Gateway)
├── worker/
│   └── worker.py           # Background worker (replaces Lambda/Step Functions)
├── Dockerfile.api-server   # API server Docker image
├── Dockerfile.worker       # Worker Docker image
├── init-aws.sh            # LocalStack initialization script
├── nginx.conf             # Nginx config for frontend
├── localstack-data/       # Persistent data (auto-created, gitignored)
└── README.md              # This file
```

## Components

### API Server (`api-server/server.py`)
- Flask application running on port 3000
- Mimics AWS API Gateway endpoints
- Handles upload URL generation and status checks
- Connects to LocalStack for S3 and DynamoDB

### Worker (`worker/worker.py`)
- Background process that watches S3 for uploads
- Automatically processes new files
- Simulates Lambda + Step Functions workflow
- Supports two modes:
  - `mock` - Fast, creates fake outputs
  - `real` - Actual processing (requires models)

### LocalStack Init (`init-aws.sh`)
- Runs when LocalStack starts
- Creates S3 buckets
- Creates DynamoDB tables
- Sets up local AWS environment

## Environment Variables

### API Server
- `AWS_ENDPOINT` - LocalStack endpoint (default: http://localstack:4566)
- `UPLOAD_BUCKET` - S3 bucket name (default: vocal-remover-upload-local)
- `JOBS_TABLE` - DynamoDB jobs table (default: vocal-remover-jobs-local)
- `RATE_LIMIT_TABLE` - DynamoDB rate limit table

### Worker
- `AWS_ENDPOINT` - LocalStack endpoint
- `PROCESSING_MODE` - `mock` or `real` (default: mock)
- `UPLOAD_BUCKET` - S3 bucket name

## Development

### Modifying the API Server

1. Edit `local/api-server/server.py`
2. Save (Flask auto-reloads in debug mode)
3. Test at http://localhost:3000

### Modifying the Worker

1. Edit `local/worker/worker.py`
2. Rebuild: `docker-compose build worker`
3. Restart: `docker-compose restart worker`
4. View logs: `docker-compose logs -f worker`

### Testing Processing Logic

To test with actual vocal separation:

1. Change `PROCESSING_MODE=real` in `docker-compose.yml`
2. Add ONNX models to worker Docker image
3. Integrate `lib_v5` code in worker

## Debugging

### View LocalStack Resources

```bash
# Set AWS CLI to use LocalStack
export AWS_ENDPOINT=http://localhost:4566
export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test

# List S3 buckets
aws --endpoint-url=$AWS_ENDPOINT s3 ls

# List files in upload bucket
aws --endpoint-url=$AWS_ENDPOINT s3 ls s3://vocal-remover-upload-local/

# Scan jobs table
aws --endpoint-url=$AWS_ENDPOINT dynamodb scan \
  --table-name vocal-remover-jobs-local
```

### Common Issues

**Port conflicts:**
```bash
# Check what's using port 3000
lsof -i :3000

# Change ports in docker-compose.yml if needed
```

**Services not starting:**
```bash
# View all service logs
docker-compose logs

# Restart everything
docker-compose down && docker-compose up -d
```

**Worker not processing:**
```bash
# Check worker logs
docker-compose logs -f worker

# Verify file was uploaded to S3
aws --endpoint-url=http://localhost:4566 s3 ls \
  s3://vocal-remover-upload-local/uploads/ --recursive
```

## Data Persistence

LocalStack data is stored in `localstack-data/` and persists between restarts.

To start fresh:
```bash
# Clean all data
./local-dev.sh clean

# Or manually
rm -rf local/localstack-data
```

## See Also

- **[LOCAL_DEVELOPMENT.md](../LOCAL_DEVELOPMENT.md)** - Complete development guide
- **[README_WEB.md](../README_WEB.md)** - Production deployment guide
- **[QUICKSTART.md](../QUICKSTART.md)** - Quick production deployment

---

**Happy local development! 🎉**
