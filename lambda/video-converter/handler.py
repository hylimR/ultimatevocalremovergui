"""
Video to Audio Converter Lambda
Converts video files to audio using FFmpeg
"""
import json
import os
import subprocess
import boto3
from pathlib import Path

s3_client = boto3.client('s3')

# Constants
TMP_DIR = '/tmp'
MAX_DURATION = 600  # 10 minutes

def lambda_handler(event, context):
    """
    Convert video to audio using FFmpeg
    """
    try:
        job_id = event['job_id']
        bucket = event['bucket']
        input_key = event['input_key']
        is_video = event.get('is_video', False)

        print(f'Processing job {job_id}, is_video: {is_video}')

        # If not video, skip conversion
        if not is_video:
            print('File is audio, skipping conversion')
            return {
                **event,
                'audio_key': input_key,
                'skipped_conversion': True
            }

        # Download video from S3
        input_path = f'{TMP_DIR}/{job_id}_input{Path(input_key).suffix}'
        output_path = f'{TMP_DIR}/{job_id}_audio.wav'

        print(f'Downloading from s3://{bucket}/{input_key}')
        s3_client.download_file(bucket, input_key, input_path)

        # Check video duration
        duration = get_video_duration(input_path)
        if duration > MAX_DURATION:
            raise Exception(f'Video duration ({duration}s) exceeds maximum ({MAX_DURATION}s)')

        print(f'Video duration: {duration}s')

        # Convert video to audio
        convert_to_audio(input_path, output_path, MAX_DURATION)

        # Upload audio to S3
        audio_key = f'processing/{job_id}/audio.wav'
        print(f'Uploading to s3://{bucket}/{audio_key}')
        s3_client.upload_file(output_path, bucket, audio_key)

        # Cleanup temp files
        cleanup_files([input_path, output_path])

        return {
            **event,
            'audio_key': audio_key,
            'duration': duration,
            'converted': True
        }

    except Exception as e:
        print(f'Error: {str(e)}')
        raise


def get_video_duration(input_path):
    """
    Get video duration using ffprobe
    """
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            input_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())
        return duration

    except Exception as e:
        print(f'Error getting duration: {str(e)}')
        raise


def convert_to_audio(input_path, output_path, max_duration):
    """
    Convert video to audio using FFmpeg
    Extract audio channel, convert to WAV, 44100 Hz, stereo, limit to max_duration
    """
    try:
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM 16-bit
            '-ar', '44100',  # Sample rate 44.1 kHz
            '-ac', '2',  # Stereo
            '-t', str(max_duration),  # Max duration
            '-y',  # Overwrite output
            output_path
        ]

        print(f'Running FFmpeg: {" ".join(cmd)}')

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=300  # 5 minute timeout
        )

        print('FFmpeg conversion successful')

        # Verify output file exists
        if not os.path.exists(output_path):
            raise Exception('Output file was not created')

        file_size = os.path.getsize(output_path)
        print(f'Output file size: {file_size / 1024 / 1024:.2f} MB')

    except subprocess.TimeoutExpired:
        raise Exception('FFmpeg conversion timed out')
    except subprocess.CalledProcessError as e:
        print(f'FFmpeg error: {e.stderr}')
        raise Exception(f'FFmpeg failed: {e.stderr}')
    except Exception as e:
        print(f'Conversion error: {str(e)}')
        raise


def cleanup_files(file_paths):
    """
    Remove temporary files
    """
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f'Removed {file_path}')
        except Exception as e:
            print(f'Error removing {file_path}: {str(e)}')
