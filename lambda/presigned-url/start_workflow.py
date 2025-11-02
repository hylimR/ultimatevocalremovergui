"""
S3 Event Trigger Lambda - Starts Step Functions workflow when file is uploaded
"""
import json
import os
import boto3
from urllib.parse import unquote_plus

stepfunctions = boto3.client('stepfunctions')
dynamodb = boto3.resource('dynamodb')

STATE_MACHINE_ARN = os.environ['STATE_MACHINE_ARN']
JOBS_TABLE = os.environ['JOBS_TABLE']

def lambda_handler(event, context):
    """
    Triggered by S3 upload event, starts Step Functions workflow
    """
    try:
        # Parse S3 event
        for record in event['Records']:
            bucket = record['s3']['bucket']['name']
            key = unquote_plus(record['s3']['object']['key'])

            print(f'File uploaded: s3://{bucket}/{key}')

            # Extract job_id from S3 key (format: uploads/{client_ip}/{job_id}/original.{ext})
            parts = key.split('/')
            if len(parts) >= 3:
                job_id = parts[2]

                # Get job metadata from DynamoDB
                job_data = get_job_metadata(job_id)

                if job_data:
                    # Start Step Functions workflow
                    execution_arn = start_step_function(job_id, bucket, key, job_data)

                    # Update job status
                    update_job_status(job_id, 'PROCESSING', execution_arn)

                    print(f'Started workflow for job {job_id}: {execution_arn}')

        return {'statusCode': 200, 'body': 'Workflow started'}

    except Exception as e:
        print(f'Error: {str(e)}')
        return {'statusCode': 500, 'body': str(e)}


def get_job_metadata(job_id):
    """
    Get job metadata from DynamoDB
    """
    try:
        table = dynamodb.Table(JOBS_TABLE)
        response = table.get_item(Key={'job_id': job_id})

        if 'Item' in response:
            return response['Item']

        return None

    except Exception as e:
        print(f'Error getting job metadata: {str(e)}')
        return None


def start_step_function(job_id, bucket, key, job_data):
    """
    Start Step Functions execution
    """
    input_data = {
        'job_id': job_id,
        'bucket': bucket,
        'input_key': key,
        'file_name': job_data.get('file_name', ''),
        'model': job_data.get('model', 'mdx_karaoke'),
        'voice_model': job_data.get('voice_model', 'none'),
        'is_video': is_video_file(key)
    }

    response = stepfunctions.start_execution(
        stateMachineArn=STATE_MACHINE_ARN,
        name=f'job-{job_id}',
        input=json.dumps(input_data)
    )

    return response['executionArn']


def update_job_status(job_id, status, execution_arn=None):
    """
    Update job status in DynamoDB
    """
    try:
        table = dynamodb.Table(JOBS_TABLE)

        update_expr = 'SET #status = :status, updated_at = :updated_at'
        expr_values = {
            ':status': status,
            ':updated_at': boto3.dynamodb.types.DYNAMODB_CONTEXT.create_datetime().isoformat()
        }

        if execution_arn:
            update_expr += ', execution_arn = :execution_arn'
            expr_values[':execution_arn'] = execution_arn

        table.update_item(
            Key={'job_id': job_id},
            UpdateExpression=update_expr,
            ExpressionAttributeNames={'#status': 'status'},
            ExpressionAttributeValues=expr_values
        )

    except Exception as e:
        print(f'Error updating job status: {str(e)}')


def is_video_file(key):
    """
    Check if file is video based on extension
    """
    video_extensions = ['mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 'm4v', 'mpeg', 'mpg']
    extension = key.split('.')[-1].lower()
    return extension in video_extensions
