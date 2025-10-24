#!/bin/bash

# Solar Rooftop Analysis - Deployment Script
# Supports development, staging, and production environments

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="development"
BUILD_CACHE=""
CLEAN_BUILD=false
SKIP_TESTS=false
MONITORING=false

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -e, --environment ENV    Set environment (development|staging|production) [default: development]"
    echo "  -c, --clean              Clean build (remove cache and rebuild)"
    echo "  -m, --monitoring         Enable monitoring stack (Prometheus + Grafana)"
    echo "  -s, --skip-tests         Skip running tests"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --environment production --clean"
    echo "  $0 --environment development --monitoring"
    echo "  $0 --environment staging --skip-tests"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -c|--clean)
            CLEAN_BUILD=true
            shift
            ;;
        -m|--monitoring)
            MONITORING=true
            shift
            ;;
        -s|--skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(development|staging|production)$ ]]; then
    print_error "Invalid environment: $ENVIRONMENT"
    print_error "Must be one of: development, staging, production"
    exit 1
fi

print_status "Starting deployment for environment: $ENVIRONMENT"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose > /dev/null 2>&1; then
    print_error "Docker Compose is not installed. Please install Docker Compose and try again."
    exit 1
fi

# Set build cache option
if [ "$CLEAN_BUILD" = true ]; then
    BUILD_CACHE="--no-cache"
    print_status "Clean build enabled - removing cache"
fi

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p outputs uploads logs ssl monitoring

# Set permissions
chmod 755 outputs uploads logs

# Load environment variables
if [ -f ".env.$ENVIRONMENT" ]; then
    print_status "Loading environment variables from .env.$ENVIRONMENT"
    export $(cat .env.$ENVIRONMENT | grep -v '^#' | xargs)
elif [ -f ".env" ]; then
    print_status "Loading environment variables from .env"
    export $(cat .env | grep -v '^#' | xargs)
else
    print_warning "No environment file found. Using default values."
fi

# Stop existing containers
print_status "Stopping existing containers..."
docker-compose -f docker-compose.yml down 2>/dev/null || true
if [ "$ENVIRONMENT" = "development" ]; then
    docker-compose -f docker-compose.dev.yml down 2>/dev/null || true
fi

# Clean up if requested
if [ "$CLEAN_BUILD" = true ]; then
    print_status "Cleaning up Docker resources..."
    docker system prune -f
    docker volume prune -f
fi

# Run tests if not skipped
if [ "$SKIP_TESTS" = false ]; then
    print_status "Running tests..."
    
    # Backend tests
    if [ -f "backend/test_main.py" ]; then
        print_status "Running backend tests..."
        cd backend
        python -m pytest test_main.py -v || print_warning "Backend tests failed"
        cd ..
    fi
    
    # Frontend tests
    if [ -f "frontend/package.json" ]; then
        print_status "Running frontend tests..."
        cd frontend
        npm test -- --passWithNoTests || print_warning "Frontend tests failed"
        cd ..
    fi
else
    print_warning "Skipping tests"
fi

# Build and start services
print_status "Building and starting services..."

if [ "$ENVIRONMENT" = "development" ]; then
    print_status "Starting development environment..."
    docker-compose -f docker-compose.dev.yml up --build $BUILD_CACHE -d
elif [ "$ENVIRONMENT" = "staging" ]; then
    print_status "Starting staging environment..."
    docker-compose -f docker-compose.yml up --build $BUILD_CACHE -d
elif [ "$ENVIRONMENT" = "production" ]; then
    print_status "Starting production environment..."
    docker-compose -f docker-compose.yml up --build $BUILD_CACHE -d
fi

# Start monitoring if requested
if [ "$MONITORING" = true ]; then
    print_status "Starting monitoring stack..."
    docker-compose -f docker-compose.yml --profile monitoring up -d
fi

# Wait for services to be healthy
print_status "Waiting for services to be healthy..."
sleep 30

# Check service health
print_status "Checking service health..."

# Check backend
if curl -f http://localhost:8000/api/health > /dev/null 2>&1; then
    print_success "Backend is healthy"
else
    print_error "Backend health check failed"
    docker-compose logs backend
    exit 1
fi

# Check frontend
if curl -f http://localhost:3000 > /dev/null 2>&1; then
    print_success "Frontend is healthy"
else
    print_error "Frontend health check failed"
    docker-compose logs frontend
    exit 1
fi

# Check Redis
if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
    print_success "Redis is healthy"
else
    print_error "Redis health check failed"
    docker-compose logs redis
    exit 1
fi

# Display service URLs
print_success "Deployment completed successfully!"
echo ""
echo "Service URLs:"
echo "  Frontend: http://localhost:3000"
echo "  Backend API: http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
echo "  Nginx: http://localhost"

if [ "$MONITORING" = true ]; then
    echo "  Prometheus: http://localhost:9090"
    echo "  Grafana: http://localhost:3001"
fi

echo ""
echo "Useful commands:"
echo "  View logs: docker-compose logs -f [service]"
echo "  Stop services: docker-compose down"
echo "  Restart service: docker-compose restart [service]"
echo "  Scale service: docker-compose up --scale [service]=N"
