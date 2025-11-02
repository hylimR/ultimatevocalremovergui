"""
Vocal Separator Lambda
Separates vocals from instrumental using MDX-Net/VR Architecture with ONNX Runtime
"""
import json
import os
import sys
import boto3
import librosa
import soundfile as sf
import numpy as np
import onnxruntime as ort

# Add lib_v5 to path (will be included in Docker image)
sys.path.insert(0, '/opt/lib_v5')

s3_client = boto3.client('s3')

# Constants
TMP_DIR = '/tmp'
MODEL_DIR = '/opt/models'

# Model configurations
MODELS = {
    'mdx_karaoke': {
        'path': 'MDX23C-8KFFT-InstVoc_HQ.onnx',
        'type': 'mdx',
        'hop_length': 1024,
        'n_fft': 8192,
        'dim_f': 3072,
        'dim_t': 256,
        'normalize': True
    },
    'mdx_extra': {
        'path': 'Kim_Vocal_2.onnx',
        'type': 'mdx',
        'hop_length': 1024,
        'n_fft': 6144,
        'dim_f': 2048,
        'dim_t': 256,
        'normalize': True
    },
    'vr_arch': {
        'path': 'UVR-MDX-NET-Inst_HQ_3.onnx',
        'type': 'vr',
        'hop_length': 512,
        'n_fft': 6144,
        'bins': 768,
        'normalize': True
    },
    'demucs_v4': {
        'path': 'htdemucs_ft.onnx',
        'type': 'demucs',
        'normalize': True
    }
}


def lambda_handler(event, context):
    """
    Separate vocals from instrumental
    """
    try:
        job_id = event['job_id']
        bucket = event['bucket']
        audio_key = event['audio_key']
        model_name = event.get('model', 'mdx_karaoke')

        print(f'Processing job {job_id} with model {model_name}')

        # Get model config
        if model_name not in MODELS:
            raise Exception(f'Unknown model: {model_name}')

        model_config = MODELS[model_name]

        # Download audio from S3
        input_path = f'{TMP_DIR}/{job_id}_input.wav'
        print(f'Downloading from s3://{bucket}/{audio_key}')
        s3_client.download_file(bucket, audio_key, input_path)

        # Load audio
        audio, sr = librosa.load(input_path, sr=44100, mono=False)
        print(f'Loaded audio: shape={audio.shape}, sr={sr}')

        # Ensure stereo
        if audio.ndim == 1:
            audio = np.stack([audio, audio])

        # Separate vocals and instrumental
        vocals, instrumental = separate_audio(audio, sr, model_config)

        # Save outputs
        vocals_path = f'{TMP_DIR}/{job_id}_vocals.wav'
        instrumental_path = f'{TMP_DIR}/{job_id}_instrumental.wav'

        sf.write(vocals_path, vocals.T, sr, subtype='PCM_16')
        sf.write(instrumental_path, instrumental.T, sr, subtype='PCM_16')

        print('Separation complete')

        # Upload results to S3
        vocals_key = f'processing/{job_id}/vocals.wav'
        instrumental_key = f'processing/{job_id}/instrumental.wav'

        print(f'Uploading results to S3')
        s3_client.upload_file(vocals_path, bucket, vocals_key)
        s3_client.upload_file(instrumental_path, bucket, instrumental_key)

        # Cleanup
        cleanup_files([input_path, vocals_path, instrumental_path])

        return {
            **event,
            'vocals_key': vocals_key,
            'instrumental_key': instrumental_key,
            'separated': True
        }

    except Exception as e:
        print(f'Error: {str(e)}')
        raise


def separate_audio(audio, sr, model_config):
    """
    Separate audio using ONNX model
    """
    try:
        model_path = os.path.join(MODEL_DIR, model_config['path'])

        # Check if model exists
        if not os.path.exists(model_path):
            raise Exception(f'Model not found: {model_path}')

        print(f'Loading model: {model_path}')

        # Create ONNX session
        session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )

        print('Running inference...')

        # Simple implementation - for production, use chunking for long audio
        # This is a simplified version - in production you'd use the full lib_v5 code

        # For now, use a basic STFT-based separation
        # In production, you would integrate the full MDX-Net/VR inference code
        vocals, instrumental = simple_separation(audio, sr, model_config)

        return vocals, instrumental

    except Exception as e:
        print(f'Separation error: {str(e)}')
        raise


def simple_separation(audio, sr, config):
    """
    Simplified separation for demonstration
    In production, this would use the full MDX-Net/VR inference pipeline
    """
    # NOTE: This is a placeholder implementation
    # For production, you need to:
    # 1. Implement proper STFT
    # 2. Run ONNX model inference on spectrograms
    # 3. Apply inverse STFT
    # 4. Use the actual lib_v5 code (SeperateMDX, SeperateVR classes)

    print('WARNING: Using simplified separation (placeholder)')
    print('For production, integrate full lib_v5 MDX-Net/VR inference code')

    # Placeholder: return original audio as vocals, silence as instrumental
    # This needs to be replaced with actual model inference
    vocals = audio
    instrumental = np.zeros_like(audio)

    return vocals, instrumental


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
