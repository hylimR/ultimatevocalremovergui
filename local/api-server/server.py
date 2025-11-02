#!/usr/bin/env python3
"""
Local API Server - Simulates API Gateway for local development
"""
import os
import json
import uuid
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
import boto3
from botocore.config import Config

app = Flask(__name__)
CORS(app)

# Configure boto3 for LocalStack
endpoint_url = os.environ.get('AWS_ENDPOINT', 'http://localstack:4566')
config = Config(
    region_name=os.environ.get('AWS_REGION', 'us-east-1'),
    signature_version='s3v4',
    retries={'max_attempts': 0}
)

s3_client = boto3.client('s3', endpoint_url=endpoint_url, config=config)
dynamodb = boto3.resource('dynamodb', endpoint_url=endpoint_url)
stepfunctions = boto3.client('stepfunctions', endpoint_url=endpoint_url)

# Environment variables
UPLOAD_BUCKET = os.environ.get('UPLOAD_BUCKET', 'vocal-remover-upload-local')
JOBS_TABLE = os.environ.get('JOBS_TABLE', 'vocal-remover-jobs-local')
RATE_LIMIT_TABLE = os.environ.get('RATE_LIMIT_TABLE', 'vocal-remover-rate-limit-local')
STATE_MACHINE_ARN = os.environ.get('STATE_MACHINE_ARN')
MAX_UPLOADS_PER_DAY = int(os.environ.get('MAX_UPLOADS_PER_DAY', '100'))  # Relaxed for local
MAX_FILE_SIZE = int(os.environ.get('MAX_FILE_SIZE', '524288000'))  # 500MB


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'vocal-remover-api-local'})


@app.route('/upload', methods=['POST', 'OPTIONS'])
def upload():
    """Generate presigned URL for upload"""
    if request.method == 'OPTIONS':
        return '', 204

    try:
        data = request.get_json()
        file_name = data.get('fileName')
        file_size = data.get('fileSize')
        content_type = data.get('contentType')
        model = data.get('model', 'mdx_karaoke')
        voice_model = data.get('voiceModel', 'none')

        # Get client IP
        client_ip = request.remote_addr

        print(f"Upload request from {client_ip}: {file_name} ({file_size} bytes)")

        # Validate
        if not file_name or not file_size:
            return jsonify({'error': 'fileName and fileSize required'}), 400

        if file_size > MAX_FILE_SIZE:
            return jsonify({'error': f'File size exceeds {MAX_FILE_SIZE / 1024 / 1024}MB limit'}), 400

        # Check rate limit (relaxed for local)
        if not check_rate_limit(client_ip):
            return jsonify({'error': f'Rate limit exceeded ({MAX_UPLOADS_PER_DAY}/day)'}), 429

        # Generate job ID
        job_id = str(uuid.uuid4())

        # Create S3 key
        extension = file_name.split('.')[-1] if '.' in file_name else 'bin'
        s3_key = f'uploads/{client_ip}/{job_id}/original.{extension}'

        # Generate presigned URL (LocalStack format)
        presigned_url = s3_client.generate_presigned_url(
            'put_object',
            Params={
                'Bucket': UPLOAD_BUCKET,
                'Key': s3_key,
                'ContentType': content_type or 'application/octet-stream'
            },
            ExpiresIn=3600
        )

        # Create job record
        create_job_record(job_id, client_ip, file_name, file_size, s3_key, model, voice_model)

        # Increment rate limit
        increment_upload_count(client_ip)

        print(f"Generated presigned URL for job {job_id}")

        return jsonify({
            'jobId': job_id,
            'uploadUrl': presigned_url,
            's3Key': s3_key,
            'message': 'Upload URL generated successfully'
        })

    except Exception as e:
        print(f"Error in upload: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/status/<job_id>', methods=['GET', 'OPTIONS'])
def status(job_id):
    """Get job status"""
    if request.method == 'OPTIONS':
        return '', 204

    try:
        # Get job from DynamoDB
        table = dynamodb.Table(JOBS_TABLE)
        response = table.get_item(Key={'job_id': job_id})

        if 'Item' not in response:
            return jsonify({'error': 'Job not found'}), 404

        job_data = response['Item']

        # Calculate progress
        status_val = job_data.get('status', 'UNKNOWN')
        progress = calculate_progress(status_val, job_data)
        current_step = get_current_step(status_val)

        result = {
            'job_id': job_id,
            'status': status_val,
            'progress': progress,
            'currentStep': current_step,
            'created_at': job_data.get('created_at'),
            'updated_at': job_data.get('updated_at')
        }

        # Add results if completed
        if status_val == 'COMPLETED' and 'results' in job_data:
            result['results'] = job_data['results']

        # Add error if failed
        if status_val == 'FAILED' and 'results' in job_data:
            result['error'] = job_data['results'].get('error', 'Unknown error')

        return jsonify(result)

    except Exception as e:
        print(f"Error in status: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/trigger/<job_id>', methods=['POST'])
def trigger_processing(job_id):
    """Manually trigger processing (for local testing)"""
    try:
        # Get job from DynamoDB
        table = dynamodb.Table(JOBS_TABLE)
        response = table.get_item(Key={'job_id': job_id})

        if 'Item' not in response:
            return jsonify({'error': 'Job not found'}), 404

        job_data = response['Item']

        # Check if file was uploaded
        s3_key = job_data.get('s3_key')

        # Update status to PROCESSING
        table.update_item(
            Key={'job_id': job_id},
            UpdateExpression='SET #status = :status, updated_at = :updated_at',
            ExpressionAttributeNames={'#status': 'status'},
            ExpressionAttributeValues={
                ':status': 'PROCESSING',
                ':updated_at': datetime.utcnow().isoformat()
            }
        )

        print(f"Triggered processing for job {job_id}")

        return jsonify({
            'message': 'Processing triggered',
            'job_id': job_id,
            'status': 'PROCESSING'
        })

    except Exception as e:
        print(f"Error triggering processing: {e}")
        return jsonify({'error': str(e)}), 500


def check_rate_limit(client_ip):
    """Check rate limit (relaxed for local dev)"""
    try:
        table = dynamodb.Table(RATE_LIMIT_TABLE)
        today = datetime.utcnow().strftime('%Y-%m-%d')

        response = table.get_item(Key={'client_ip': client_ip, 'date': today})

        if 'Item' in response:
            upload_count = int(response['Item'].get('upload_count', 0))
            if upload_count >= MAX_UPLOADS_PER_DAY:
                return False

        return True
    except Exception as e:
        print(f"Rate limit check error: {e}")
        return True  # Allow on error


def increment_upload_count(client_ip):
    """Increment upload count"""
    try:
        table = dynamodb.Table(RATE_LIMIT_TABLE)
        today = datetime.utcnow().strftime('%Y-%m-%d')
        ttl = int((datetime.utcnow() + timedelta(days=1)).timestamp())

        table.update_item(
            Key={'client_ip': client_ip, 'date': today},
            UpdateExpression='ADD upload_count :inc SET #ttl = :ttl',
            ExpressionAttributeNames={'#ttl': 'ttl'},
            ExpressionAttributeValues={':inc': 1, ':ttl': ttl}
        )
    except Exception as e:
        print(f"Increment count error: {e}")


def create_job_record(job_id, client_ip, file_name, file_size, s3_key, model, voice_model):
    """Create job record in DynamoDB"""
    try:
        table = dynamodb.Table(JOBS_TABLE)
        ttl = int((datetime.utcnow() + timedelta(days=7)).timestamp())

        table.put_item(
            Item={
                'job_id': job_id,
                'client_ip': client_ip,
                'file_name': file_name,
                'file_size': file_size,
                's3_key': s3_key,
                'model': model,
                'voice_model': voice_model,
                'status': 'UPLOADING',
                'progress': 0,
                'created_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat(),
                'ttl': ttl
            }
        )
        print(f"Created job record: {job_id}")
    except Exception as e:
        print(f"Create job record error: {e}")
        raise


def calculate_progress(status, job_data):
    """Calculate progress percentage"""
    progress_map = {
        'UPLOADING': 10,
        'PROCESSING': job_data.get('progress', 40),
        'CONVERTING': 30,
        'SEPARATING': 50,
        'VOICE_CONVERSION': 70,
        'FINALIZING': 90,
        'COMPLETED': 100,
        'FAILED': job_data.get('progress', 0)
    }
    return progress_map.get(status, 0)


def get_current_step(status):
    """Get current step number"""
    step_map = {
        'UPLOADING': 1,
        'CONVERTING': 2,
        'SEPARATING': 3,
        'VOICE_CONVERSION': 4,
        'FINALIZING': 5,
        'PROCESSING': 2,
        'COMPLETED': 5,
        'FAILED': 0
    }
    return step_map.get(status, 0)


if __name__ == '__main__':
    print("🚀 Starting Local API Server on http://localhost:3000")
    print(f"   LocalStack endpoint: {endpoint_url}")
    print(f"   Upload bucket: {UPLOAD_BUCKET}")
    print(f"   Jobs table: {JOBS_TABLE}")
    app.run(host='0.0.0.0', port=3000, debug=True)
