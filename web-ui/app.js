// Configuration - Update these with your deployed endpoints
const CONFIG = {
    API_ENDPOINT: 'https://your-api-gateway-url.execute-api.us-east-1.amazonaws.com/prod',
    MAX_FILE_SIZE: 500 * 1024 * 1024, // 500MB
    MAX_DURATION: 600, // 10 minutes in seconds
    POLL_INTERVAL: 3000, // 3 seconds
    ALLOWED_AUDIO_FORMATS: ['mp3', 'wav', 'flac', 'm4a', 'aac', 'ogg', 'opus', 'webm'],
    ALLOWED_VIDEO_FORMATS: ['mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 'm4v']
};

// State
let currentFile = null;
let currentJobId = null;
let pollInterval = null;

// DOM Elements
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const fileInfo = document.getElementById('fileInfo');
const fileName = document.getElementById('fileName');
const fileMeta = document.getElementById('fileMeta');
const removeFile = document.getElementById('removeFile');
const modelSelect = document.getElementById('modelSelect');
const voiceSelect = document.getElementById('voiceSelect');
const processBtn = document.getElementById('processBtn');
const uploadSection = document.getElementById('uploadSection');
const progressSection = document.getElementById('progressSection');
const resultsSection = document.getElementById('resultsSection');
const errorSection = document.getElementById('errorSection');
const progressStatus = document.getElementById('progressStatus');
const progressBar = document.getElementById('progressBar');
const progressPercent = document.getElementById('progressPercent');
const errorMessage = document.getElementById('errorMessage');
const retryBtn = document.getElementById('retryBtn');
const newProcessing = document.getElementById('newProcessing');
const vocalsPreview = document.getElementById('vocalsPreview');
const instrumentalPreview = document.getElementById('instrumentalPreview');
const downloadVocals = document.getElementById('downloadVocals');
const downloadInstrumental = document.getElementById('downloadInstrumental');

// Initialize
init();

function init() {
    // Drag and drop
    uploadZone.addEventListener('click', () => fileInput.click());
    uploadZone.addEventListener('dragover', handleDragOver);
    uploadZone.addEventListener('dragleave', handleDragLeave);
    uploadZone.addEventListener('drop', handleDrop);

    // File input
    fileInput.addEventListener('change', handleFileSelect);

    // Remove file
    removeFile.addEventListener('click', clearFile);

    // Process button
    processBtn.addEventListener('click', processFile);

    // Retry button
    retryBtn.addEventListener('click', reset);

    // New processing button
    newProcessing.addEventListener('click', reset);
}

function handleDragOver(e) {
    e.preventDefault();
    uploadZone.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadZone.classList.remove('dragover');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

async function handleFile(file) {
    // Validate file size
    if (file.size > CONFIG.MAX_FILE_SIZE) {
        showError(`File size exceeds 500MB limit. Your file is ${formatFileSize(file.size)}`);
        return;
    }

    // Validate file type
    const extension = file.name.split('.').pop().toLowerCase();
    const isAudio = CONFIG.ALLOWED_AUDIO_FORMATS.includes(extension);
    const isVideo = CONFIG.ALLOWED_VIDEO_FORMATS.includes(extension);

    if (!isAudio && !isVideo) {
        showError(`Unsupported file format: .${extension}`);
        return;
    }

    // Validate duration for audio/video
    try {
        const duration = await getMediaDuration(file);
        if (duration > CONFIG.MAX_DURATION) {
            showError(`Duration exceeds 10 minutes limit. Your file is ${formatDuration(duration)}`);
            return;
        }

        // Success - show file info
        currentFile = file;
        displayFileInfo(file, duration, isVideo);
        processBtn.disabled = false;
    } catch (error) {
        console.error('Error reading file:', error);
        showError('Unable to read file metadata. Please ensure the file is valid.');
    }
}

function getMediaDuration(file) {
    return new Promise((resolve, reject) => {
        const url = URL.createObjectURL(file);
        const extension = file.name.split('.').pop().toLowerCase();
        const isVideo = CONFIG.ALLOWED_VIDEO_FORMATS.includes(extension);

        const element = isVideo ? document.createElement('video') : document.createElement('audio');

        element.addEventListener('loadedmetadata', () => {
            URL.revokeObjectURL(url);
            resolve(element.duration);
        });

        element.addEventListener('error', () => {
            URL.revokeObjectURL(url);
            reject(new Error('Failed to load media'));
        });

        element.src = url;
    });
}

function displayFileInfo(file, duration, isVideo) {
    uploadZone.style.display = 'none';
    fileInfo.style.display = 'flex';

    fileName.textContent = file.name;
    fileMeta.textContent = `${formatFileSize(file.size)} • ${formatDuration(duration)} • ${isVideo ? 'Video' : 'Audio'}`;
}

function clearFile() {
    currentFile = null;
    uploadZone.style.display = 'block';
    fileInfo.style.display = 'none';
    fileInput.value = '';
    processBtn.disabled = true;
}

async function processFile() {
    if (!currentFile) return;

    try {
        processBtn.disabled = true;
        processBtn.innerHTML = '<span>Processing...</span>';

        // Step 1: Get presigned URL
        updateProgress(5, 'Requesting upload URL...', 1);
        const uploadData = await getPresignedUrl(currentFile);

        // Step 2: Upload to S3
        updateProgress(10, 'Uploading file...', 1);
        await uploadToS3(uploadData.uploadUrl, currentFile);

        // Step 3: Start processing
        updateProgress(30, 'Starting processing...', 2);
        currentJobId = uploadData.jobId;

        // Step 4: Poll for status
        startPolling();

    } catch (error) {
        console.error('Processing error:', error);
        showError(error.message || 'Failed to process file. Please try again.');
        processBtn.disabled = false;
        processBtn.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Process Audio
        `;
    }
}

async function getPresignedUrl(file) {
    const response = await fetch(`${CONFIG.API_ENDPOINT}/upload`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            fileName: file.name,
            fileSize: file.size,
            contentType: file.type,
            model: modelSelect.value,
            voiceModel: voiceSelect.value
        })
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Failed to get upload URL');
    }

    return response.json();
}

async function uploadToS3(presignedUrl, file) {
    const response = await fetch(presignedUrl, {
        method: 'PUT',
        body: file,
        headers: {
            'Content-Type': file.type
        }
    });

    if (!response.ok) {
        throw new Error('Failed to upload file to S3');
    }
}

function startPolling() {
    uploadSection.style.display = 'none';
    progressSection.style.display = 'block';

    pollInterval = setInterval(checkStatus, CONFIG.POLL_INTERVAL);
    checkStatus(); // Initial check
}

async function checkStatus() {
    try {
        const response = await fetch(`${CONFIG.API_ENDPOINT}/status/${currentJobId}`);

        if (!response.ok) {
            throw new Error('Failed to get job status');
        }

        const data = await response.json();
        handleStatusUpdate(data);

    } catch (error) {
        console.error('Status check error:', error);
        // Continue polling unless critical error
    }
}

function handleStatusUpdate(data) {
    const { status, progress, currentStep, error, results } = data;

    // Update progress
    if (progress !== undefined) {
        updateProgress(progress, getStepMessage(currentStep), currentStep);
    }

    // Handle completion
    if (status === 'COMPLETED') {
        clearInterval(pollInterval);
        showResults(results);
    }

    // Handle failure
    if (status === 'FAILED') {
        clearInterval(pollInterval);
        showError(error || 'Processing failed. Please try again.');
    }
}

function updateProgress(percent, message, step) {
    progressBar.style.width = `${percent}%`;
    progressPercent.textContent = `${Math.round(percent)}%`;
    progressStatus.textContent = message;

    // Update step indicators
    for (let i = 1; i <= 5; i++) {
        const stepEl = document.getElementById(`step${i}`);
        if (i < step) {
            stepEl.classList.add('completed');
            stepEl.classList.remove('active');
        } else if (i === step) {
            stepEl.classList.add('active');
            stepEl.classList.remove('completed');
        } else {
            stepEl.classList.remove('active', 'completed');
        }
    }
}

function getStepMessage(step) {
    const messages = {
        1: 'Uploading file...',
        2: 'Converting video to audio...',
        3: 'Separating vocals from instrumental...',
        4: 'Converting voice with AI...',
        5: 'Finalizing and preparing downloads...'
    };
    return messages[step] || 'Processing...';
}

function showResults(results) {
    progressSection.style.display = 'none';
    resultsSection.style.display = 'block';

    // Set audio previews
    if (results.vocalsUrl) {
        vocalsPreview.src = results.vocalsUrl;
        downloadVocals.onclick = () => downloadFile(results.vocalsUrl, 'vocals.wav');
    }

    if (results.instrumentalUrl) {
        instrumentalPreview.src = results.instrumentalUrl;
        downloadInstrumental.onclick = () => downloadFile(results.instrumentalUrl, 'instrumental.wav');
    }
}

function showError(message) {
    uploadSection.style.display = 'none';
    progressSection.style.display = 'none';
    resultsSection.style.display = 'none';
    errorSection.style.display = 'block';

    errorMessage.textContent = message;
}

function reset() {
    clearInterval(pollInterval);
    currentFile = null;
    currentJobId = null;

    uploadSection.style.display = 'block';
    progressSection.style.display = 'none';
    resultsSection.style.display = 'none';
    errorSection.style.display = 'none';

    clearFile();

    processBtn.disabled = true;
    processBtn.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        Process Audio
    `;
}

function downloadFile(url, filename) {
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

function formatDuration(seconds) {
    const minutes = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
}
