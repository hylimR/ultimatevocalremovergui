# 🏗️ Architecture Documentation

## Overview

This document provides detailed architecture information for the Vocal Remover Web Application, including design decisions, data flows, and cost optimization strategies.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Design Decisions](#design-decisions)
3. [Data Flow](#data-flow)
4. [Scalability](#scalability)
5. [Cost Optimization](#cost-optimization)
6. [Security](#security)
7. [Monitoring & Observability](#monitoring--observability)

---

## System Architecture

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         CloudFront CDN                               │
│              (Global edge locations for frontend)                    │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │   S3 Static Website   │
                    │   (index.html, CSS,   │
                    │    JS, assets)        │
                    └───────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                      API Layer (HTTP API)                            │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                │                               │
                ▼                               ▼
    ┌───────────────────┐          ┌───────────────────┐
    │  Lambda           │          │  Lambda           │
    │  (Presigned URL)  │          │  (Status Check)   │
    │  + Rate Limiting  │          │                   │
    └─────────┬─────────┘          └─────────┬─────────┘
              │                              │
              ▼                              ▼
    ┌─────────────────┐          ┌─────────────────┐
    │  DynamoDB       │          │  DynamoDB       │
    │  (Rate Limits)  │          │  (Jobs Table)   │
    └─────────────────┘          └─────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                      Storage Layer                                   │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
        ┌──────────┐    ┌──────────┐   ┌──────────┐
        │   S3     │    │   S3     │   │   S3     │
        │ (Upload) │    │(Process) │   │ (Output) │
        └────┬─────┘    └──────────┘   └──────────┘
             │
             │ (S3 Event)
             ▼
    ┌────────────────┐
    │  Lambda        │
    │ (Start Workflow)│
    └────────┬───────┘
             │
             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   Step Functions State Machine                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                                                                 │ │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    │ │
│  │  │   Lambda     │    │   Lambda     │    │   Lambda     │    │ │
│  │  │   (Video     │───▶│   (Vocal     │───▶│   (Voice     │    │ │
│  │  │  Converter)  │    │  Separator)  │    │  Converter)  │    │ │
│  │  │   FFmpeg     │    │  ONNX/MDX    │    │     RVC      │    │ │
│  │  └──────────────┘    └──────────────┘    └──────────────┘    │ │
│  │         │                     │                    │          │ │
│  │         └─────────────────────┼────────────────────┘          │ │
│  │                               ▼                                │ │
│  │                      ┌──────────────┐                          │ │
│  │                      │   Lambda     │                          │ │
│  │                      │  (Cleanup)   │                          │ │
│  │                      └──────────────┘                          │ │
│  │                                                                 │ │
│  └────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

### Component Details

#### 1. Frontend Layer

**Technology**: Static HTML/CSS/JavaScript
**Hosting**: S3 + CloudFront
**Purpose**: User interface for file upload and download

**Components**:
- `index.html`: Main application page
- `styles.css`: Responsive styling
- `app.js`: Client-side logic and API communication

**Features**:
- Client-side file validation (size, duration, format)
- Drag-and-drop upload interface
- Real-time progress tracking via polling
- Audio preview and download

#### 2. API Layer

**Technology**: API Gateway HTTP API
**Purpose**: RESTful endpoints for upload and status checking

**Endpoints**:
- `POST /upload`: Generate presigned S3 URL
- `GET /status/{jobId}`: Get job status and results

**Features**:
- CORS enabled for web access
- Request throttling (100 req/sec burst, 50 req/sec sustained)
- CloudWatch access logging
- Optional WAF integration for advanced rate limiting

#### 3. Processing Layer

**Technology**: AWS Lambda (Container Images)
**Purpose**: Execute CPU-intensive AI processing tasks

**Functions**:

| Function | Memory | Timeout | Technology | Purpose |
|----------|--------|---------|------------|---------|
| Video Converter | 3 GB | 5 min | FFmpeg | Extract audio from video |
| Vocal Separator | 10 GB | 15 min | ONNX Runtime | Separate vocals/instrumental |
| Voice Converter | 10 GB | 10 min | RVC ONNX | Convert voice characteristics |
| Cleanup | 512 MB | 1 min | Python | Finalize and generate URLs |

#### 4. Orchestration Layer

**Technology**: AWS Step Functions
**Purpose**: Coordinate multi-step processing workflow

**State Machine**: Express workflow for cost efficiency ($1 vs $25 per million)

**States**:
1. Update status to "PROCESSING"
2. Check if video (conditional)
3. Convert video to audio (if needed)
4. Separate vocals from instrumental
5. Convert voice (if requested)
6. Cleanup and generate download URLs
7. Handle errors and update status

#### 5. Storage Layer

**Technology**: Amazon S3
**Purpose**: File storage with lifecycle management

**Buckets**:
- `upload-bucket`: Temporary upload storage (1-day retention)
- `output-bucket`: Processed files (7-day retention)
- `frontend-bucket`: Static website hosting

#### 6. Data Layer

**Technology**: Amazon DynamoDB
**Purpose**: Metadata and rate limiting

**Tables**:
- `rate-limit-table`: Track uploads per IP (TTL: 24 hours)
- `jobs-table`: Job status and metadata (TTL: 7 days)

---

## Design Decisions

### Why Serverless?

**Pros**:
- ✅ Zero cost when idle
- ✅ Automatic scaling
- ✅ No server management
- ✅ Pay-per-use pricing
- ✅ Built-in high availability

**Cons**:
- ❌ Cold start latency (mitigated with provisioned concurrency)
- ❌ 15-minute Lambda timeout (handle with chunking)
- ❌ 10 GB Lambda storage limit (sufficient for most audio)

### Why Step Functions Express?

**Standard vs Express Comparison**:

| Feature | Standard | Express |
|---------|----------|---------|
| **Cost** | $25/M transitions | $1/M requests |
| **Duration** | Up to 1 year | Up to 5 minutes |
| **Execution History** | Full audit trail | CloudWatch logs |
| **Use Case** | Long-running workflows | Short, high-volume |

**Decision**: Express workflow is 96% cheaper and sufficient for audio processing (<5 min per file)

### Why ONNX Runtime?

**PyTorch vs ONNX Comparison**:

| Aspect | PyTorch | ONNX Runtime |
|--------|---------|--------------|
| **Size** | ~1.5 GB | ~700 MB |
| **Inference Speed** | Baseline | 2-3x faster |
| **Cold Start** | 10-15 sec | 3-5 sec |
| **Memory Usage** | Higher | 30-50% lower |

**Decision**: ONNX provides faster inference, smaller container size, and lower costs

### Why SQS + Step Functions Hybrid?

**Option 1**: SQS only
- ✅ Cheapest ($0.48/M)
- ❌ Complex error handling
- ❌ No visual workflow
- ❌ Manual state management

**Option 2**: Step Functions only
- ✅ Visual workflow
- ✅ Built-in error handling
- ❌ Expensive ($25/M for Standard)
- ❌ Can throttle at high volume

**Option 3**: SQS + Step Functions Express (Chosen)
- ✅ Cost-effective ($1/M)
- ✅ Visual workflow
- ✅ SQS buffers spikes
- ✅ Best of both worlds

### Why Container Images for Lambda?

**Deployment Package vs Container**:

| Aspect | ZIP Package | Container Image |
|--------|-------------|-----------------|
| **Max Size** | 250 MB | 10 GB |
| **Dependencies** | Limited | All included |
| **FFmpeg** | Requires layer | Native install |
| **ML Models** | Must download | Pre-packaged |

**Decision**: Container images support large dependencies (FFmpeg, ONNX models)

---

## Data Flow

### Upload Flow

```
┌─────────┐
│  User   │
└────┬────┘
     │
     │ 1. Select file
     ▼
┌──────────────┐
│  Frontend    │
│  Validation  │ (Size, duration, format)
└──────┬───────┘
       │
       │ 2. POST /upload
       ▼
┌──────────────────┐
│  API Gateway     │
└──────┬───────────┘
       │
       │ 3. Invoke Lambda
       ▼
┌────────────────────────┐
│  Presigned URL Lambda  │
│  • Check rate limit    │
│  • Create job record   │
│  • Generate S3 URL     │
└──────┬─────────────────┘
       │
       │ 4. Return presigned URL
       ▼
┌──────────────┐
│  Frontend    │
│  PUT to S3   │
└──────┬───────┘
       │
       │ 5. Upload file
       ▼
┌──────────────┐
│  S3 Upload   │
│  Bucket      │
└──────┬───────┘
       │
       │ 6. S3 Event
       ▼
┌──────────────────┐
│  Start Workflow  │
│  Lambda          │
└──────┬───────────┘
       │
       │ 7. Start execution
       ▼
┌──────────────────┐
│  Step Functions  │
└──────────────────┘
```

### Processing Flow

```
┌─────────────────┐
│ Step Functions  │
└────────┬────────┘
         │
         │ State 1
         ▼
┌─────────────────────┐
│  Video Converter    │
│  • Download from S3 │
│  • Check duration   │
│  • Extract audio    │───┐
│  • Upload WAV       │   │
└─────────────────────┘   │
         │                │
         │ State 2        │ (Skip if already audio)
         ▼                │
┌─────────────────────┐   │
│  Vocal Separator    │◄──┘
│  • Load audio       │
│  • Run ONNX model   │
│  • Save vocals      │
│  • Save instrumental│
└─────────┬───────────┘
          │
          │ State 3
          ▼
┌─────────────────────┐
│  Voice Converter    │
│  • Load vocals      │
│  • Run RVC model    │───┐
│  • Save converted   │   │
└─────────────────────┘   │
          │               │
          │ State 4       │ (Skip if none)
          ▼               │
┌─────────────────────┐   │
│  Cleanup            │◄──┘
│  • Copy to output   │
│  • Generate URLs    │
│  • Update job       │
│  • Delete temp files│
└─────────┬───────────┘
          │
          │ Complete
          ▼
┌─────────────────────┐
│  DynamoDB           │
│  Status: COMPLETED  │
│  + Download URLs    │
└─────────────────────┘
```

### Status Polling Flow

```
┌──────────────┐
│  Frontend    │
│  (Polling    │
│   every 3s)  │
└──────┬───────┘
       │
       │ GET /status/{jobId}
       ▼
┌──────────────────┐
│  API Gateway     │
└──────┬───────────┘
       │
       │ Invoke Lambda
       ▼
┌────────────────────┐
│  Status Lambda     │
│  • Get from DDB    │
│  • Calculate %     │
│  • Return status   │
└──────┬─────────────┘
       │
       │ Return JSON
       ▼
┌──────────────┐
│  Frontend    │
│  • Update UI │
│  • Show %    │
│  • Download  │
└──────────────┘
```

---

## Scalability

### Horizontal Scaling

All components scale automatically:

| Component | Scaling Mechanism | Limit |
|-----------|-------------------|-------|
| CloudFront | Global edge locations | Unlimited |
| API Gateway | Automatic | 10K req/sec (soft limit) |
| Lambda | Concurrent executions | 1000 (adjustable) |
| S3 | Automatic partitioning | Unlimited |
| DynamoDB | On-demand capacity | 40K RCU/WCU (soft limit) |
| Step Functions | Automatic | 5K executions/sec |

### Vertical Scaling

Lambda memory can be adjusted per function:

```hcl
resource "aws_lambda_function" "vocal_separator" {
  memory_size = 10240  # 10 GB (max)

  ephemeral_storage {
    size = 10240  # 10 GB /tmp (max)
  }
}
```

### Handling Large Files

**Problem**: 10-minute audio = ~100 MB WAV = Processing time > Lambda timeout

**Solution**: Chunking Strategy

```python
def process_large_audio(audio_path):
    # Split into 2-minute chunks
    chunks = split_audio(audio_path, chunk_duration=120)

    # Process chunks in parallel
    results = []
    for chunk in chunks:
        result = separate_vocals(chunk)
        results.append(result)

    # Merge results
    final_vocals = merge_chunks([r['vocals'] for r in results])
    final_instrumental = merge_chunks([r['instrumental'] for r in results])

    return final_vocals, final_instrumental
```

### Traffic Patterns

**Expected patterns**:
- **Peak hours**: Evenings, weekends (3x normal traffic)
- **Burst traffic**: Social media mentions (10x spike)
- **Seasonal**: Music production cycles

**Handling**:
- SQS buffers upload spikes
- Lambda auto-scales to 1000 concurrent
- Step Functions processes at sustainable rate
- DynamoDB on-demand handles variable load

---

## Cost Optimization

### Strategy 1: Use Express Workflows

**Savings**: 96% ($25 → $1 per million)

```hcl
resource "aws_sfn_state_machine" "processing" {
  type = "EXPRESS"  # Not "STANDARD"
}
```

### Strategy 2: S3 Lifecycle Policies

**Savings**: 80% on storage costs

```hcl
resource "aws_s3_bucket_lifecycle_configuration" "upload_lifecycle" {
  rule {
    expiration {
      days = 1  # Delete uploads after 1 day
    }
  }
}
```

### Strategy 3: ONNX Runtime

**Savings**: 50% on compute costs (faster = less $ per invocation)

### Strategy 4: Reserved Concurrency

**Savings**: Prevents runaway costs from bugs/attacks

```hcl
resource "aws_lambda_function" "vocal_separator" {
  reserved_concurrent_executions = 10
}
```

### Strategy 5: Right-Sizing Memory

**Principle**: Lambda pricing is linear with memory, but CPU scales with it

**Optimization**:
```bash
# Test different memory sizes
for memory in 1024 2048 3008 5120 10240; do
  aws lambda update-function-configuration \
    --function-name my-function \
    --memory-size $memory

  # Run benchmark
  time aws lambda invoke --function-name my-function
done

# Choose sweet spot: Lowest (duration × memory) = cost
```

### Cost Monitoring

**CloudWatch Alarms**:

```hcl
resource "aws_cloudwatch_metric_alarm" "high_cost" {
  alarm_name = "vocal-remover-high-cost"
  metric_name = "EstimatedCharges"
  comparison_operator = "GreaterThanThreshold"
  threshold = 100  # Alert if >$100/month
}
```

---

## Security

### Defense in Depth

```
┌────────────────────────────────────┐
│  Layer 1: WAF                      │  DDoS, rate limit
└────────┬───────────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│  Layer 2: API Gateway              │  Throttling, auth
└────────┬───────────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│  Layer 3: Lambda                   │  Rate limit in DDB
└────────┬───────────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│  Layer 4: S3 Bucket Policy         │  Restrict access
└────────────────────────────────────┘
```

### IAM Least Privilege

Each Lambda has minimal permissions:

```hcl
# Video Converter: Only S3 read/write
resource "aws_iam_role_policy" "video_converter" {
  policy = jsonencode({
    Statement = [{
      Effect = "Allow"
      Action = ["s3:GetObject", "s3:PutObject"]
      Resource = "${aws_s3_bucket.upload.arn}/*"
    }]
  })
}
```

### Data Protection

1. **Encryption at Rest**: S3 default encryption (AES-256)
2. **Encryption in Transit**: HTTPS only (CloudFront, API Gateway)
3. **Data Retention**: Automatic deletion via lifecycle policies
4. **Access Logging**: CloudTrail + S3 access logs

### Threat Mitigation

| Threat | Mitigation |
|--------|------------|
| **DDoS** | CloudFront + WAF |
| **Abuse** | Rate limiting (5/day per IP) |
| **Large files** | Pre-validation (500 MB max) |
| **Malicious files** | Sandboxed Lambda execution |
| **Cost attacks** | Reserved concurrency + alarms |
| **Data exfiltration** | Presigned URLs expire in 7 days |

---

## Monitoring & Observability

### Key Metrics

**Frontend**:
- CloudFront cache hit rate
- Origin requests
- 4xx/5xx error rates

**API**:
- Request count
- Latency (p50, p95, p99)
- Error rate

**Processing**:
- Lambda invocations
- Duration
- Errors
- Throttles
- Concurrent executions

**Storage**:
- S3 bucket size
- Request count
- Data transfer

**Workflow**:
- Step Functions executions
- Success/failure rate
- Execution duration

### CloudWatch Dashboard

```json
{
  "widgets": [
    {
      "type": "metric",
      "properties": {
        "metrics": [
          ["AWS/Lambda", "Invocations", {"stat": "Sum"}],
          ["AWS/Lambda", "Errors", {"stat": "Sum"}],
          ["AWS/Lambda", "Duration", {"stat": "Average"}]
        ],
        "period": 300,
        "stat": "Sum",
        "region": "us-east-1",
        "title": "Lambda Performance"
      }
    }
  ]
}
```

### Logging Strategy

**CloudWatch Log Groups**:
- `/aws/lambda/presigned-url`: Upload requests
- `/aws/lambda/vocal-separator`: Processing logs
- `/aws/states/processing-workflow`: Workflow execution
- `/aws/apigateway/vocal-remover`: API requests

**Log Retention**: 7 days (balance cost vs debugging)

### Alerting

**Critical Alarms**:
- Lambda error rate > 5%
- Step Functions failure rate > 10%
- API Gateway 5xx rate > 1%
- Estimated charges > $100/month

**Warning Alarms**:
- Lambda duration > 80% of timeout
- DynamoDB throttled requests > 0
- S3 bucket size > 100 GB

---

## Future Enhancements

### Planned Features

1. **WebSocket API**: Real-time progress instead of polling
2. **Batch Processing**: Upload multiple files at once
3. **Custom Models**: Allow users to upload their own RVC models
4. **Audio Effects**: Reverb, EQ, compression post-processing
5. **Mobile App**: React Native companion app
6. **Social Features**: Share results, compare outputs
7. **Premium Tier**: Higher limits, priority processing

### Technical Improvements

1. **EFS Integration**: Share models across Lambda invocations
2. **Fargate Processing**: For files >10 min duration
3. **SageMaker**: For real-time voice conversion inference
4. **API Caching**: ElastiCache for status queries
5. **Custom Domain**: Route53 + ACM for branded URLs
6. **Multi-Region**: Global deployment for lower latency

---

**Document Version**: 1.0
**Last Updated**: 2025-11-02
**Author**: Claude Code
