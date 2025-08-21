#!/bin/bash

# Fire Prediction System - Docker Management Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
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

# Help function
show_help() {
    echo "Fire Prediction System - Docker Management"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build       Build the Docker image"
    echo "  start       Start the services"
    echo "  stop        Stop the services"
    echo "  restart     Restart the services"
    echo "  logs        Show service logs"
    echo "  status      Show service status"
    echo "  clean       Clean up containers and images"
    echo "  test        Test the services"
    echo "  shell       Access the container shell"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build && $0 start"
    echo "  $0 logs"
    echo "  $0 test"
}

# Build function
build_image() {
    print_status "Building Docker image..."
    docker-compose build --no-cache
    print_success "Docker image built successfully!"
}

# Start function
start_services() {
    print_status "Starting Fire Prediction services..."
    
    # Check if .env file exists, if not create from example
    if [ ! -f .env ]; then
        print_warning ".env file not found, creating from .env.example"
        cp .env.example .env
    fi
    
    docker-compose up -d
    print_success "Services started successfully!"
    
    print_status "Waiting for services to be ready..."
    sleep 10
    
    # Check if services are healthy
    if curl -f http://localhost:5001/ > /dev/null 2>&1; then
        print_success "Main API is responding on http://localhost:5001"
    else
        print_warning "Main API might not be ready yet"
    fi
    
    if curl -f http://localhost:5002/status > /dev/null 2>&1; then
        print_success "Telegram API is responding on http://localhost:5002"
    else
        print_warning "Telegram API might not be ready yet"
    fi
}

# Stop function
stop_services() {
    print_status "Stopping Fire Prediction services..."
    docker-compose down
    print_success "Services stopped successfully!"
}

# Restart function
restart_services() {
    print_status "Restarting Fire Prediction services..."
    docker-compose restart
    print_success "Services restarted successfully!"
}

# Logs function
show_logs() {
    print_status "Showing service logs (Ctrl+C to exit)..."
    docker-compose logs -f
}

# Status function
show_status() {
    print_status "Service Status:"
    docker-compose ps
    
    echo ""
    print_status "API Health Checks:"
    
    # Check main API
    if curl -f http://localhost:5001/ > /dev/null 2>&1; then
        print_success "✅ Main API (http://localhost:5001) - Healthy"
    else
        print_error "❌ Main API (http://localhost:5001) - Not responding"
    fi
    
    # Check Telegram API
    if curl -f http://localhost:5002/status > /dev/null 2>&1; then
        print_success "✅ Telegram API (http://localhost:5002) - Healthy"
    else
        print_error "❌ Telegram API (http://localhost:5002) - Not responding"
    fi
}

# Clean function
clean_system() {
    print_status "Cleaning up Docker system..."
    
    # Stop and remove containers
    docker-compose down --remove-orphans
    
    # Remove images
    docker-compose down --rmi all --volumes
    
    # Clean up unused Docker resources
    docker system prune -f
    
    print_success "Docker system cleaned successfully!"
}

# Test function
test_services() {
    print_status "Testing Fire Prediction services..."
    
    # Test main API
    print_status "Testing Main API..."
    response=$(curl -s http://localhost:5001/ || echo "FAILED")
    if [[ $response == *"Welcome to the Satellite Image Analysis API"* ]]; then
        print_success "✅ Main API test passed"
    else
        print_error "❌ Main API test failed"
        return 1
    fi
    
    # Test Telegram API
    print_status "Testing Telegram API..."
    response=$(curl -s http://localhost:5002/status || echo "FAILED")
    if [[ $response == *"status"* ]]; then
        print_success "✅ Telegram API test passed"
    else
        print_error "❌ Telegram API test failed"
        return 1
    fi
    
    # Test Telegram integration
    print_status "Testing Telegram integration..."
    response=$(curl -s -X POST "http://localhost:5001/test-telegram-alert?message=Docker%20Test" || echo "FAILED")
    if [[ $response == *"test_results"* ]]; then
        print_success "✅ Telegram integration test passed"
    else
        print_warning "⚠️  Telegram integration test partially failed (might be normal if no subscribers)"
    fi
    
    print_success "All tests completed!"
}

# Shell function
access_shell() {
    print_status "Accessing container shell..."
    docker-compose exec backend /bin/bash
}

# Main script logic
case "${1:-help}" in
    build)
        build_image
        ;;
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    logs)
        show_logs
        ;;
    status)
        show_status
        ;;
    clean)
        clean_system
        ;;
    test)
        test_services
        ;;
    shell)
        access_shell
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
