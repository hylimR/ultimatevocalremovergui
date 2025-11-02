#!/usr/bin/env python3
"""
Local Worker - Simulates Lambda processing for local development
Watches S3 for uploads and processes them
"""
import os
import sys
import time
import json
import boto3
from datetime import datetime
from botocore.config import Config

# Configure boto3 for LocalStack
endpoint_url = os.environ.get('AWS_ENDPOINT', 'http://localstack:4566')
config = Config(region_name='us-east-1', signature_version='s3v4')

s3_client = boto3.client('s3', endpoint_url=endpoint_url, config=config)
dynamodb = boto3.resource('dynamodb', endpoint_url=endpoint_url)

UPLOAD_BUCKET = os.environ.get('UPLOAD_BUCKET', 'vocal-remover-upload-local')
JOBS_TABLE = os.environ.get('JOBS_TABLE', 'vocal-remover-jobs-local')
PROCESSING_MODE = os.environ.get('PROCESSING_MODE', 'mock')  # 'mock' or 'real'

# Track processed files
processed_files = set()


def watch_for_uploads():
    """Poll S3 for new uploads"""
    print(f"👀 Watching bucket '{UPLOAD_BUCKET}' for uploads...")
    print(f"   Processing mode: {PROCESSING_MODE}")
    print(f"   LocalStack: {endpoint_url}")

    while True:
        try:
            # List objects in upload bucket
            response = s3_client.list_objects_v2(
                Bucket=UPLOAD_BUCKET,
                Prefix='uploads/'
            )

            if 'Contents' in response:
                for obj in response['Contents']:
                    s3_key = obj['Key']

                    # Skip if already processed
                    if s3_key in processed_files:
                        continue

                    # Extract job_id from key
                    parts = s3_key.split('/')
                    if len(parts) >= 3 and parts[0] == 'uploads':
                        job_id = parts[2]

                        # Check if job exists and is in UPLOADING status
                        table = dynamodb.Table(JOBS_TABLE)
                        job_response = table.get_item(Key={'job_id': job_id})

                        if 'Item' in job_response:
                            job = job_response['Item']
                            status = job.get('status', '')

                            if status == 'UPLOADING':
                                print(f"\n📥 New upload detected: {s3_key}")
                                print(f"   Job ID: {job_id}")
                                print(f"   File: {job.get('file_name', 'unknown')}")

                                # Mark as processed
                                processed_files.add(s3_key)

                                # Start processing
                                process_job(job_id, s3_key, job)

            # Wait before next poll
            time.sleep(2)

        except Exception as e:
            print(f"❌ Error watching uploads: {e}")
            time.sleep(5)


def process_job(job_id, s3_key, job_data):
    """Process a job"""
    try:
        table = dynamodb.Table(JOBS_TABLE)

        print(f"🔄 Processing job {job_id}...")

        # Update status to PROCESSING
        update_status(job_id, 'PROCESSING', 20)
        time.sleep(1)

        # Step 1: Video conversion (if needed)
        is_video = is_video_file(job_data.get('file_name', ''))
        if is_video:
            print(f"   📹 Converting video to audio...")
            update_status(job_id, 'CONVERTING', 30)
            time.sleep(2)  # Simulate processing

        # Step 2: Vocal separation
        print(f"   🎤 Separating vocals from instrumental...")
        update_status(job_id, 'SEPARATING', 50)

        if PROCESSING_MODE == 'real':
            # TODO: Actual vocal separation
            # vocals, instrumental = separate_vocals(audio_file)
            pass

        time.sleep(3)  # Simulate processing

        # Step 3: Voice conversion (if requested)
        voice_model = job_data.get('voice_model', 'none')
        if voice_model != 'none':
            print(f"   🎵 Converting voice using {voice_model}...")
            update_status(job_id, 'VOICE_CONVERSION', 70)
            time.sleep(2)  # Simulate processing

        # Step 4: Finalize
        print(f"   ✨ Finalizing...")
        update_status(job_id, 'FINALIZING', 90)
        time.sleep(1)

        # Create mock output files
        vocals_url = create_mock_output(job_id, 'vocals')
        instrumental_url = create_mock_output(job_id, 'instrumental')

        # Mark as completed
        table.update_item(
            Key={'job_id': job_id},
            UpdateExpression='SET #status = :status, progress = :progress, results = :results, updated_at = :updated_at',
            ExpressionAttributeNames={'#status': 'status'},
            ExpressionAttributeValues={
                ':status': 'COMPLETED',
                ':progress': 100,
                ':results': {
                    'vocalsUrl': vocals_url,
                    'instrumentalUrl': instrumental_url,
                    'voice_converted': voice_model != 'none'
                },
                ':updated_at': datetime.utcnow().isoformat()
            }
        )

        print(f"✅ Job {job_id} completed successfully!")
        print(f"   Vocals: {vocals_url}")
        print(f"   Instrumental: {instrumental_url}")

    except Exception as e:
        print(f"❌ Error processing job {job_id}: {e}")

        # Mark as failed
        try:
            table = dynamodb.Table(JOBS_TABLE)
            table.update_item(
                Key={'job_id': job_id},
                UpdateExpression='SET #status = :status, results = :results, updated_at = :updated_at',
                ExpressionAttributeNames={'#status': 'status'},
                ExpressionAttributeValues={
                    ':status': 'FAILED',
                    ':results': {'error': str(e)},
                    ':updated_at': datetime.utcnow().isoformat()
                }
            )
        except Exception as update_error:
            print(f"❌ Failed to update job status: {update_error}")


def update_status(job_id, status, progress):
    """Update job status"""
    try:
        table = dynamodb.Table(JOBS_TABLE)
        table.update_item(
            Key={'job_id': job_id},
            UpdateExpression='SET #status = :status, progress = :progress, updated_at = :updated_at',
            ExpressionAttributeNames={'#status': 'status'},
            ExpressionAttributeValues={
                ':status': status,
                ':progress': progress,
                ':updated_at': datetime.utcnow().isoformat()
            }
        )
    except Exception as e:
        print(f"Error updating status: {e}")


def create_mock_output(job_id, output_type):
    """Create mock output file (for demo)"""
    try:
        # Create a small text file as placeholder
        content = f"Mock {output_type} for job {job_id}\nGenerated at {datetime.utcnow().isoformat()}"

        # Upload to S3
        output_key = f'outputs/{job_id}/{output_type}.wav'
        s3_client.put_object(
            Bucket=UPLOAD_BUCKET,
            Key=output_key,
            Body=content.encode('utf-8'),
            ContentType='audio/wav'
        )

        # Generate presigned URL
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': UPLOAD_BUCKET, 'Key': output_key},
            ExpiresIn=604800  # 7 days
        )

        return url

    except Exception as e:
        print(f"Error creating mock output: {e}")
        return None


def is_video_file(filename):
    """Check if file is video"""
    video_extensions = ['mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 'm4v']
    extension = filename.split('.')[-1].lower() if '.' in filename else ''
    return extension in video_extensions


if __name__ == '__main__':
    print("🚀 Starting Local Worker")
    print(f"   Endpoint: {endpoint_url}")
    print(f"   Bucket: {UPLOAD_BUCKET}")
    print(f"   Jobs Table: {JOBS_TABLE}")
    print(f"   Mode: {PROCESSING_MODE}")
    print()

    # Wait for LocalStack to be ready
    print("⏳ Waiting for LocalStack...")
    time.sleep(5)

    # Start watching
    watch_for_uploads()
