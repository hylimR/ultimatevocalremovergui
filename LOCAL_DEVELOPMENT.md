# 🏠 Local Development Guide

Run the entire Vocal Remover application locally without deploying to AWS!

## 🎯 Overview

This local development environment allows you to:
- ✅ Test the complete application flow locally
- ✅ Develop and debug without AWS costs
- ✅ Fast iteration without deployment delays
- ✅ Mock AWS services using LocalStack
- ✅ Run everything in Docker containers

## 🛠️ Prerequisites

### Required
- **Docker** (>= 20.x) - [Install Docker](https://docs.docker.com/get-docker/)
- **Docker Compose** (>= 2.x) - Usually included with Docker Desktop
- **Bash** shell (Linux/macOS/WSL)

### Optional
- **FFmpeg** - For test file creation
- **curl** - For API testing
- **Python 3** - For JSON formatting

## ⚡ Quick Start

### 1. Start Local Environment

```bash
# Start all services
./local-dev.sh start
```

This will start:
- **LocalStack** (AWS services emulator)
- **API Server** (Flask server replacing API Gateway)
- **Worker** (Processes uploads automatically)
- **Frontend** (Nginx serving the web UI)

### 2. Open the Application

```
🌐 Frontend:   http://localhost:8080
🔌 API:        http://localhost:3000
☁️  LocalStack: http://localhost:4566
```

### 3. Upload a File

1. Open http://localhost:8080 in your browser
2. Upload an audio or video file
3. Select a model
4. Click "Process Audio"
5. Watch it process locally!

## 📋 Commands

### Service Management

```bash
# Start services
./local-dev.sh start

# Stop services
./local-dev.sh stop

# Restart services
./local-dev.sh restart

# Check service status
./local-dev.sh status

# Check health
./local-dev.sh health
```

### Logs and Debugging

```bash
# Show all logs
./local-dev.sh logs

# Show specific service logs
./local-dev.sh logs api-server
./local-dev.sh logs worker
./local-dev.sh logs localstack
./local-dev.sh logs frontend

# Follow logs in real-time (Ctrl+C to exit)
docker-compose logs -f api-server
```

### Testing

```bash
# Run automated upload test
./local-dev.sh test

# Clean local data (S3, DynamoDB)
./local-dev.sh clean
```

## 🏗️ Architecture

### Local vs Production

| Component | Production | Local Equivalent |
|-----------|-----------|------------------|
| API Gateway | AWS API Gateway | Flask server (:3000) |
| S3 | AWS S3 | LocalStack S3 (:4566) |
| DynamoDB | AWS DynamoDB | LocalStack DynamoDB (:4566) |
| Lambda | AWS Lambda | Python worker container |
| Step Functions | AWS Step Functions | Simulated in worker |
| CloudFront | AWS CloudFront | Nginx (:8080) |

### Services

#### 1. **LocalStack** (Port 4566)
Emulates AWS services:
- S3 buckets
- DynamoDB tables
- Step Functions (optional)
- CloudWatch Logs (optional)

#### 2. **API Server** (Port 3000)
Flask server that mimics API Gateway:
- `POST /upload` - Generate presigned S3 URL
- `GET /status/{jobId}` - Get job status
- `POST /trigger/{jobId}` - Manually trigger processing

#### 3. **Worker**
Background service that:
- Watches S3 for new uploads
- Processes files automatically
- Updates job status in DynamoDB
- Runs in `mock` mode by default (fast, no actual processing)

#### 4. **Frontend** (Port 8080)
Nginx serving the web UI:
- Automatically detects local environment
- Uses `http://localhost:3000` as API endpoint
- Serves static files from `web-ui/`

## 🔧 Configuration

### Environment Detection

The frontend automatically detects if it's running locally:

```javascript
// web-ui/config.js
const ENV = {
    isLocal: window.location.hostname === 'localhost',

    local: {
        apiEndpoint: 'http://localhost:3000',
        pollInterval: 2000
    },

    production: {
        apiEndpoint: 'https://your-api-url...',
        pollInterval: 3000
    }
};
```

### Processing Modes

The worker supports two modes:

#### Mock Mode (Default) - Fast Testing
```bash
# In docker-compose.yml
environment:
  - PROCESSING_MODE=mock
```

- Creates fake output files instantly
- Good for testing UI/API flow
- No actual vocal separation

#### Real Mode - Actual Processing
```bash
# In docker-compose.yml
environment:
  - PROCESSING_MODE=real
```

- Runs actual vocal separation (requires models)
- Uses FFmpeg, ONNX, etc.
- Slower but tests full pipeline

### Rate Limiting

Local environment has relaxed rate limits:

```python
# local/api-server/server.py
MAX_UPLOADS_PER_DAY = 100  # vs 5 in production
```

## 🧪 Testing

### Manual Testing

1. **Upload via UI**
   ```
   1. Open http://localhost:8080
   2. Upload a file
   3. Monitor progress
   4. Download results
   ```

2. **Upload via API**
   ```bash
   # 1. Get upload URL
   curl -X POST http://localhost:3000/upload \
     -H "Content-Type: application/json" \
     -d '{
       "fileName": "test.mp3",
       "fileSize": 1000000,
       "contentType": "audio/mpeg",
       "model": "mdx_karaoke",
       "voiceModel": "none"
     }'

   # Response: {"jobId": "...", "uploadUrl": "...", ...}

   # 2. Upload file
   curl -X PUT "<uploadUrl>" \
     -H "Content-Type: audio/mpeg" \
     --data-binary @test.mp3

   # 3. Check status
   curl http://localhost:3000/status/<jobId>
   ```

3. **Automated Test**
   ```bash
   ./local-dev.sh test
   ```

### LocalStack CLI

Access LocalStack services:

```bash
# Set endpoint
export AWS_ENDPOINT=http://localhost:4566
export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export AWS_DEFAULT_REGION=us-east-1

# List S3 buckets
aws --endpoint-url=$AWS_ENDPOINT s3 ls

# List DynamoDB tables
aws --endpoint-url=$AWS_ENDPOINT dynamodb list-tables

# Scan jobs table
aws --endpoint-url=$AWS_ENDPOINT dynamodb scan \
  --table-name vocal-remover-jobs-local

# Download file from S3
aws --endpoint-url=$AWS_ENDPOINT s3 cp \
  s3://vocal-remover-upload-local/uploads/... \
  output.wav
```

## 🐛 Debugging

### Common Issues

#### 1. Services won't start

```bash
# Check Docker is running
docker ps

# Check ports are free
lsof -i :3000  # API Server
lsof -i :4566  # LocalStack
lsof -i :8080  # Frontend

# View startup logs
docker-compose logs
```

#### 2. LocalStack not initializing

```bash
# Check initialization logs
docker-compose logs localstack

# Manually run init script
docker-compose exec localstack bash /etc/localstack/init/ready.d/init-aws.sh
```

#### 3. Worker not processing files

```bash
# Check worker logs
docker-compose logs worker

# Verify S3 upload
aws --endpoint-url=http://localhost:4566 s3 ls \
  s3://vocal-remover-upload-local/uploads/ --recursive

# Check job status in DynamoDB
aws --endpoint-url=http://localhost:4566 dynamodb scan \
  --table-name vocal-remover-jobs-local
```

#### 4. Frontend can't connect to API

```bash
# Check API health
curl http://localhost:3000/health

# Check browser console for CORS errors
# Open browser DevTools -> Console

# Verify config.js is loaded
# Open http://localhost:8080 -> check Network tab
```

### Debug Modes

Enable verbose logging:

```bash
# In docker-compose.yml, set DEBUG=1 for LocalStack
environment:
  - DEBUG=1

# Restart to apply
./local-dev.sh restart
```

## 📊 Monitoring

### View Logs in Real-Time

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api-server
docker-compose logs -f worker

# With timestamps
docker-compose logs -f --timestamps

# Last N lines
docker-compose logs --tail=100 worker
```

### Check Service Health

```bash
# Health check command
./local-dev.sh health

# Output:
# ✓ LocalStack is healthy
# ✓ API Server is healthy
# ✓ Frontend is serving
```

### Inspect Containers

```bash
# Enter API server container
docker-compose exec api-server bash

# Enter worker container
docker-compose exec worker bash

# Enter LocalStack container
docker-compose exec localstack bash

# View container stats
docker stats
```

## 🗂️ File Structure

```
local/
├── api-server/
│   └── server.py           # Flask API server
├── worker/
│   └── worker.py           # Background processor
├── Dockerfile.api-server   # API server image
├── Dockerfile.worker       # Worker image
├── init-aws.sh            # LocalStack initialization
├── nginx.conf             # Frontend nginx config
└── localstack-data/       # Persistent LocalStack data

docker-compose.yml         # Service orchestration
local-dev.sh              # Management script
```

## 💾 Data Persistence

LocalStack data is persisted in `local/localstack-data/`:

```bash
# List S3 files
ls -la local/localstack-data/s3/

# Clean all data
./local-dev.sh clean
```

## 🚀 Development Workflow

### Typical Development Cycle

1. **Start environment**
   ```bash
   ./local-dev.sh start
   ```

2. **Make code changes**
   - Edit Lambda functions in `lambda/`
   - Edit frontend in `web-ui/`
   - Edit worker logic in `local/worker/worker.py`

3. **Rebuild changed services**
   ```bash
   # Rebuild and restart worker
   docker-compose build worker
   docker-compose up -d worker

   # Frontend auto-reloads (no rebuild needed)

   # API server auto-reloads (Flask debug mode)
   ```

4. **Test changes**
   ```bash
   ./local-dev.sh test
   # Or test via browser at http://localhost:8080
   ```

5. **View logs**
   ```bash
   ./local-dev.sh logs worker
   ```

6. **Iterate!**

### Hot Reload

Services with hot reload:
- ✅ **Frontend** - Nginx serves files directly, changes visible immediately
- ✅ **API Server** - Flask debug mode auto-reloads
- ❌ **Worker** - Requires rebuild (`docker-compose build worker && docker-compose restart worker`)

## 🔄 Switching Between Local and Production

The frontend automatically switches based on hostname:

```javascript
// Automatic detection
if (window.location.hostname === 'localhost') {
    // Use http://localhost:3000
} else {
    // Use production API Gateway URL
}
```

No code changes needed when deploying!

## 📈 Performance

### Local Environment Performance

| Operation | Mock Mode | Real Mode |
|-----------|-----------|-----------|
| **Video Conversion** | Instant | 3-8 sec |
| **Vocal Separation** | Instant | 30-60 sec |
| **Voice Conversion** | Instant | 20-40 sec |
| **Total Time** | ~1 sec | 1-3 min |

### Resource Usage

```
LocalStack:   ~500 MB RAM
API Server:   ~100 MB RAM
Worker:       ~200 MB RAM (mock) / ~2 GB (real)
Frontend:     ~50 MB RAM
───────────────────────────────────
Total:        ~1 GB RAM (mock) / ~3 GB (real)
```

## 🎓 Tips & Tricks

### 1. Fast Testing

Use mock mode for UI/API testing:
```yaml
# docker-compose.yml
worker:
  environment:
    - PROCESSING_MODE=mock
```

### 2. Persist Data Between Restarts

Data persists automatically in `local/localstack-data/`. To start fresh:
```bash
./local-dev.sh clean
```

### 3. Debug Worker Processing

```bash
# Watch worker process files in real-time
docker-compose logs -f worker
```

### 4. Test Error Handling

Manually fail a job:
```bash
# 1. Upload a file
# 2. Get job ID
# 3. Manually set to FAILED
aws --endpoint-url=http://localhost:4566 dynamodb update-item \
  --table-name vocal-remover-jobs-local \
  --key '{"job_id": {"S": "YOUR-JOB-ID"}}' \
  --update-expression "SET #status = :failed" \
  --expression-attribute-names '{"#status": "status"}' \
  --expression-attribute-values '{":failed": {"S": "FAILED"}}'
```

### 5. Custom Test Files

Place test files in `local/`:
```bash
local/
├── test-audio.mp3
├── test-video.mp4
└── test-long.wav
```

Then upload via UI or API.

## 🆘 Troubleshooting

### Problem: Port already in use

```bash
# Find what's using the port
lsof -i :3000  # or :4566, :8080

# Kill the process
kill -9 <PID>

# Or change ports in docker-compose.yml
```

### Problem: Docker out of space

```bash
# Clean up Docker
docker system prune -a

# Remove old containers
docker-compose down --rmi all
```

### Problem: LocalStack not responding

```bash
# Restart LocalStack
docker-compose restart localstack

# Or full restart
./local-dev.sh restart
```

## 📚 Additional Resources

- **LocalStack Docs**: https://docs.localstack.cloud/
- **Docker Compose Docs**: https://docs.docker.com/compose/
- **Flask Docs**: https://flask.palletsprojects.com/

## 🎉 You're All Set!

```bash
# Start developing
./local-dev.sh start

# Open in browser
open http://localhost:8080

# Happy coding! 🚀
```

---

**Questions?** Check the logs: `./local-dev.sh logs`
**Issues?** See troubleshooting section above
