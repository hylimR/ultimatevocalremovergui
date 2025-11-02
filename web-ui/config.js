// Configuration for local vs production
const ENV = {
    // Detect if running locally
    isLocal: window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1',

    // API endpoints
    local: {
        apiEndpoint: 'http://localhost:3000',
        pollInterval: 2000  // Poll every 2 seconds locally
    },

    production: {
        apiEndpoint: 'https://your-api-gateway-url.execute-api.us-east-1.amazonaws.com/prod',
        pollInterval: 3000  // Poll every 3 seconds in production
    }
};

// Export current config
const CONFIG = ENV.isLocal ? ENV.local : ENV.production;

// Add common config
CONFIG.MAX_FILE_SIZE = 500 * 1024 * 1024; // 500MB
CONFIG.MAX_DURATION = 600; // 10 minutes
CONFIG.ALLOWED_AUDIO_FORMATS = ['mp3', 'wav', 'flac', 'm4a', 'aac', 'ogg', 'opus', 'webm'];
CONFIG.ALLOWED_VIDEO_FORMATS = ['mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 'm4v'];

console.log(`🌍 Environment: ${ENV.isLocal ? 'LOCAL' : 'PRODUCTION'}`);
console.log(`🔗 API Endpoint: ${CONFIG.apiEndpoint}`);
