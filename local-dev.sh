#!/bin/bash
# Local Development Environment Manager
# Easily start/stop/manage local development environment

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "${BLUE}"
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║     Vocal Remover - Local Development Environment         ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

check_dependencies() {
    print_info "Checking dependencies..."

    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi

    print_success "All dependencies are installed"
}

start_services() {
    print_header
    check_dependencies

    echo ""
    print_info "Starting local development environment..."
    echo ""

    # Create local directories if they don't exist
    mkdir -p local/localstack-data

    # Start Docker Compose
    docker-compose up -d

    echo ""
    print_success "Services started!"
    echo ""

    # Wait for services to be ready
    print_info "Waiting for services to be ready..."
    sleep 10

    # Check health
    check_health

    echo ""
    print_success "Local development environment is ready!"
    echo ""

    print_usage
}

stop_services() {
    print_header
    print_info "Stopping local development environment..."

    docker-compose down

    echo ""
    print_success "Services stopped"
}

restart_services() {
    print_header
    print_info "Restarting local development environment..."

    docker-compose restart

    echo ""
    print_success "Services restarted"

    sleep 5
    check_health
}

check_health() {
    print_info "Checking service health..."

    # Check LocalStack
    if curl -s http://localhost:4566/_localstack/health > /dev/null 2>&1; then
        print_success "LocalStack is healthy"
    else
        print_warning "LocalStack might not be ready yet"
    fi

    # Check API Server
    if curl -s http://localhost:3000/health > /dev/null 2>&1; then
        print_success "API Server is healthy"
    else
        print_warning "API Server might not be ready yet"
    fi

    # Check Frontend
    if curl -s http://localhost:8080 > /dev/null 2>&1; then
        print_success "Frontend is serving"
    else
        print_warning "Frontend might not be ready yet"
    fi
}

show_logs() {
    service=$1

    if [ -z "$service" ]; then
        docker-compose logs -f
    else
        docker-compose logs -f "$service"
    fi
}

show_status() {
    print_header
    echo ""
    docker-compose ps
    echo ""
}

clean_data() {
    print_header
    print_warning "This will delete all local data (S3 files, DynamoDB tables, etc.)"
    read -p "Are you sure? (y/N) " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Cleaning local data..."
        rm -rf local/localstack-data
        print_success "Local data cleaned"
    else
        print_info "Cancelled"
    fi
}

test_upload() {
    print_header
    print_info "Testing file upload..."

    # Create a test audio file (1 second of silence)
    print_info "Creating test audio file..."
    ffmpeg -f lavfi -i anullsrc=r=44100:cl=stereo -t 1 -q:a 9 -acodec libmp3lame local/test-audio.mp3 -y 2>/dev/null

    if [ ! -f local/test-audio.mp3 ]; then
        print_error "Failed to create test audio file (ffmpeg required)"
        exit 1
    fi

    print_success "Test file created: local/test-audio.mp3"

    # Get file info
    file_size=$(stat -f%z local/test-audio.mp3 2>/dev/null || stat -c%s local/test-audio.mp3)

    print_info "Requesting upload URL..."

    # Request presigned URL
    response=$(curl -s -X POST http://localhost:3000/upload \
        -H "Content-Type: application/json" \
        -d "{
            \"fileName\": \"test-audio.mp3\",
            \"fileSize\": $file_size,
            \"contentType\": \"audio/mpeg\",
            \"model\": \"mdx_karaoke\",
            \"voiceModel\": \"none\"
        }")

    job_id=$(echo "$response" | grep -o '"jobId":"[^"]*' | cut -d'"' -f4)
    upload_url=$(echo "$response" | grep -o '"uploadUrl":"[^"]*' | cut -d'"' -f4)

    if [ -z "$job_id" ]; then
        print_error "Failed to get upload URL"
        echo "$response"
        exit 1
    fi

    print_success "Got upload URL for job: $job_id"

    # Upload file
    print_info "Uploading file..."
    curl -s -X PUT "$upload_url" \
        -H "Content-Type: audio/mpeg" \
        --data-binary @local/test-audio.mp3 > /dev/null

    print_success "File uploaded!"

    # Check status
    print_info "Checking job status..."
    sleep 2

    status_response=$(curl -s http://localhost:3000/status/$job_id)
    echo "$status_response" | python3 -m json.tool 2>/dev/null || echo "$status_response"

    echo ""
    print_success "Test upload completed! Job ID: $job_id"
    print_info "Monitor progress at: http://localhost:3000/status/$job_id"
}

print_usage() {
    echo -e "${BLUE}📍 Services are running at:${NC}"
    echo ""
    echo "   🌐 Frontend:     http://localhost:8080"
    echo "   🔌 API Server:   http://localhost:3000"
    echo "   ☁️  LocalStack:   http://localhost:4566"
    echo ""
    echo -e "${BLUE}📚 Quick Commands:${NC}"
    echo ""
    echo "   ./local-dev.sh status    - Show service status"
    echo "   ./local-dev.sh logs      - Show all logs"
    echo "   ./local-dev.sh logs api  - Show API server logs"
    echo "   ./local-dev.sh test      - Test file upload"
    echo "   ./local-dev.sh stop      - Stop all services"
    echo ""
}

# Main command dispatcher
case "$1" in
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs "$2"
        ;;
    health)
        print_header
        check_health
        ;;
    clean)
        clean_data
        ;;
    test)
        test_upload
        ;;
    *)
        print_header
        echo "Usage: $0 {start|stop|restart|status|logs|health|clean|test}"
        echo ""
        echo "Commands:"
        echo "  start      - Start local development environment"
        echo "  stop       - Stop all services"
        echo "  restart    - Restart all services"
        echo "  status     - Show service status"
        echo "  logs       - Show logs (optionally specify service: api, worker, frontend)"
        echo "  health     - Check service health"
        echo "  clean      - Clean local data (S3, DynamoDB)"
        echo "  test       - Run upload test"
        echo ""
        exit 1
        ;;
esac
