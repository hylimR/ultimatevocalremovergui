"""
Cleanup Lambda
Finalizes processing, generates presigned URLs for download, and updates job status
"""
import json
import os
import boto3
from datetime import datetime, timedelta

s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

OUTPUT_BUCKET = os.environ['OUTPUT_BUCKET']
JOBS_TABLE = os.environ['JOBS_TABLE']
PROCESSING_BUCKET = os.environ.get('PROCESSING_BUCKET', OUTPUT_BUCKET)


def lambda_handler(event, context):
    """
    Finalize processing and generate download URLs
    """
    try:
        job_id = event['job_id']
        bucket = event['bucket']
        converted_vocals_key = event.get('converted_vocals_key')
        vocals_key = event.get('vocals_key')
        instrumental_key = event['instrumental_key']
        voice_converted = event.get('voice_converted', False)

        print(f'Finalizing job {job_id}')

        # Determine final vocals key
        final_vocals_key = converted_vocals_key if voice_converted else vocals_key

        # Copy files to output bucket
        output_vocals_key = f'outputs/{job_id}/vocals.wav'
        output_instrumental_key = f'outputs/{job_id}/instrumental.wav'

        print(f'Copying files to output bucket')

        s3_client.copy_object(
            CopySource={'Bucket': bucket, 'Key': final_vocals_key},
            Bucket=OUTPUT_BUCKET,
            Key=output_vocals_key
        )

        s3_client.copy_object(
            CopySource={'Bucket': bucket, 'Key': instrumental_key},
            Bucket=OUTPUT_BUCKET,
            Key=output_instrumental_key
        )

        # Generate presigned URLs (valid for 7 days)
        vocals_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': OUTPUT_BUCKET, 'Key': output_vocals_key},
            ExpiresIn=604800  # 7 days
        )

        instrumental_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': OUTPUT_BUCKET, 'Key': output_instrumental_key},
            ExpiresIn=604800  # 7 days
        )

        print('Generated download URLs')

        # Update job status in DynamoDB
        update_job_status(job_id, 'COMPLETED', {
            'vocals_url': vocals_url,
            'instrumental_url': instrumental_url,
            'vocals_key': output_vocals_key,
            'instrumental_key': output_instrumental_key,
            'voice_converted': voice_converted
        })

        # Cleanup processing files
        cleanup_processing_files(bucket, job_id)

        print(f'Job {job_id} completed successfully')

        return {
            'job_id': job_id,
            'status': 'COMPLETED',
            'vocals_url': vocals_url,
            'instrumental_url': instrumental_url,
            'voice_converted': voice_converted
        }

    except Exception as e:
        print(f'Error: {str(e)}')

        # Update job status to failed
        try:
            update_job_status(event['job_id'], 'FAILED', {'error': str(e)})
        except Exception as db_error:
            print(f'Failed to update job status: {str(db_error)}')

        raise


def update_job_status(job_id, status, results=None):
    """
    Update job status in DynamoDB
    """
    try:
        table = dynamodb.Table(JOBS_TABLE)

        update_expr = 'SET #status = :status, updated_at = :updated_at, progress = :progress'
        expr_names = {'#status': 'status'}
        expr_values = {
            ':status': status,
            ':updated_at': datetime.utcnow().isoformat(),
            ':progress': 100 if status == 'COMPLETED' else 0
        }

        if results:
            update_expr += ', results = :results'
            expr_values[':results'] = results

        table.update_item(
            Key={'job_id': job_id},
            UpdateExpression=update_expr,
            ExpressionAttributeNames=expr_names,
            ExpressionAttributeValues=expr_values
        )

        print(f'Updated job {job_id} status to {status}')

    except Exception as e:
        print(f'Error updating job status: {str(e)}')
        raise


def cleanup_processing_files(bucket, job_id):
    """
    Delete processing files from S3
    """
    try:
        # List all processing files for this job
        prefix = f'processing/{job_id}/'
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)

        if 'Contents' in response:
            for obj in response['Contents']:
                s3_client.delete_object(Bucket=bucket, Key=obj['Key'])
                print(f'Deleted {obj["Key"]}')

    except Exception as e:
        print(f'Error cleaning up processing files: {str(e)}')
        # Don't raise - cleanup is not critical
