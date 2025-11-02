import json
import uuid
import os
import boto3
from datetime import datetime, timedelta
from decimal import Decimal

s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
stepfunctions = boto3.client('stepfunctions')

# Environment variables
UPLOAD_BUCKET = os.environ['UPLOAD_BUCKET']
RATE_LIMIT_TABLE = os.environ['RATE_LIMIT_TABLE']
JOBS_TABLE = os.environ['JOBS_TABLE']
STATE_MACHINE_ARN = os.environ['STATE_MACHINE_ARN']
MAX_UPLOADS_PER_DAY = int(os.environ.get('MAX_UPLOADS_PER_DAY', '5'))
MAX_FILE_SIZE = int(os.environ.get('MAX_FILE_SIZE', '524288000'))  # 500MB

def lambda_handler(event, context):
    """
    Generate presigned URL for file upload with rate limiting
    """
    try:
        # Parse request body
        body = json.loads(event.get('body', '{}'))
        file_name = body.get('fileName')
        file_size = body.get('fileSize')
        content_type = body.get('contentType')
        model = body.get('model', 'mdx_karaoke')
        voice_model = body.get('voiceModel', 'none')

        # Get client IP for rate limiting
        client_ip = event['requestContext']['identity']['sourceIp']

        # Validate input
        if not file_name or not file_size:
            return error_response(400, 'fileName and fileSize are required')

        # Validate file size
        if file_size > MAX_FILE_SIZE:
            return error_response(400, f'File size exceeds {MAX_FILE_SIZE / 1024 / 1024}MB limit')

        # Check rate limit
        if not check_rate_limit(client_ip):
            return error_response(429, f'Rate limit exceeded. Maximum {MAX_UPLOADS_PER_DAY} uploads per day.')

        # Generate job ID
        job_id = str(uuid.uuid4())

        # Create S3 key
        extension = file_name.split('.')[-1]
        s3_key = f'uploads/{client_ip}/{job_id}/original.{extension}'

        # Generate presigned URL for upload
        presigned_url = s3_client.generate_presigned_url(
            'put_object',
            Params={
                'Bucket': UPLOAD_BUCKET,
                'Key': s3_key,
                'ContentType': content_type
            },
            ExpiresIn=3600  # 1 hour
        )

        # Store job metadata
        create_job_record(job_id, client_ip, file_name, file_size, s3_key, model, voice_model)

        # Increment rate limit counter
        increment_upload_count(client_ip)

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'jobId': job_id,
                'uploadUrl': presigned_url,
                's3Key': s3_key,
                'message': 'Upload URL generated successfully'
            })
        }

    except Exception as e:
        print(f'Error: {str(e)}')
        return error_response(500, 'Internal server error')


def check_rate_limit(client_ip):
    """
    Check if client has exceeded rate limit
    """
    try:
        table = dynamodb.Table(RATE_LIMIT_TABLE)
        today = datetime.utcnow().strftime('%Y-%m-%d')

        response = table.get_item(
            Key={'client_ip': client_ip, 'date': today}
        )

        if 'Item' in response:
            upload_count = int(response['Item'].get('upload_count', 0))
            if upload_count >= MAX_UPLOADS_PER_DAY:
                return False

        return True

    except Exception as e:
        print(f'Rate limit check error: {str(e)}')
        # Allow on error to prevent blocking legitimate users
        return True


def increment_upload_count(client_ip):
    """
    Increment upload count for client
    """
    try:
        table = dynamodb.Table(RATE_LIMIT_TABLE)
        today = datetime.utcnow().strftime('%Y-%m-%d')

        # TTL: expire after 24 hours
        ttl = int((datetime.utcnow() + timedelta(days=1)).timestamp())

        table.update_item(
            Key={'client_ip': client_ip, 'date': today},
            UpdateExpression='ADD upload_count :inc SET #ttl = :ttl',
            ExpressionAttributeNames={'#ttl': 'ttl'},
            ExpressionAttributeValues={
                ':inc': 1,
                ':ttl': ttl
            }
        )

    except Exception as e:
        print(f'Increment upload count error: {str(e)}')


def create_job_record(job_id, client_ip, file_name, file_size, s3_key, model, voice_model):
    """
    Create job record in DynamoDB
    """
    try:
        table = dynamodb.Table(JOBS_TABLE)

        # TTL: expire after 7 days
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
                'created_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat(),
                'ttl': ttl
            }
        )

    except Exception as e:
        print(f'Create job record error: {str(e)}')
        raise


def error_response(status_code, message):
    """
    Return error response
    """
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps({'error': message})
    }
