"""
Status Check Lambda
Returns job status and progress for frontend polling
"""
import json
import os
import boto3

dynamodb = boto3.resource('dynamodb')

JOBS_TABLE = os.environ['JOBS_TABLE']


def lambda_handler(event, context):
    """
    Get job status
    """
    try:
        # Get job_id from path parameters
        job_id = event['pathParameters']['jobId']

        # Get job from DynamoDB
        job_data = get_job(job_id)

        if not job_data:
            return {
                'statusCode': 404,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({'error': 'Job not found'})
            }

        # Calculate progress based on status
        status = job_data.get('status', 'UNKNOWN')
        progress = calculate_progress(status, job_data)

        response_body = {
            'job_id': job_id,
            'status': status,
            'progress': progress,
            'currentStep': get_current_step(status),
            'created_at': job_data.get('created_at'),
            'updated_at': job_data.get('updated_at')
        }

        # Add results if completed
        if status == 'COMPLETED' and 'results' in job_data:
            response_body['results'] = job_data['results']

        # Add error if failed
        if status == 'FAILED' and 'results' in job_data:
            response_body['error'] = job_data['results'].get('error', 'Unknown error')

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(response_body)
        }

    except Exception as e:
        print(f'Error: {str(e)}')
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({'error': 'Internal server error'})
        }


def get_job(job_id):
    """
    Get job data from DynamoDB
    """
    try:
        table = dynamodb.Table(JOBS_TABLE)
        response = table.get_item(Key={'job_id': job_id})

        if 'Item' in response:
            return response['Item']

        return None

    except Exception as e:
        print(f'Error getting job: {str(e)}')
        return None


def calculate_progress(status, job_data):
    """
    Calculate progress percentage based on status
    """
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
    """
    Get current step number for frontend display
    """
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
