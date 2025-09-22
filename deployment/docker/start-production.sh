#!/bin/bash

# VecLite Production Deployment Startup Script
# Automates the deployment process with proper checks and configurations

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}" >&2
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi

    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running. Please start Docker."
        exit 1
    fi

    success "Prerequisites check passed"
}

# Create necessary directories
create_directories() {
    log "Creating necessary directories..."

    mkdir -p "$SCRIPT_DIR/data"
    mkdir -p "$SCRIPT_DIR/logs"
    mkdir -p "$SCRIPT_DIR/nginx/ssl"
    mkdir -p "$SCRIPT_DIR/grafana/dashboards"
    mkdir -p "$SCRIPT_DIR/grafana/datasources"

    success "Directories created"
}

# Generate SSL certificates if they don't exist
setup_ssl() {
    if [ ! -f "$SCRIPT_DIR/nginx/ssl/cert.pem" ] || [ ! -f "$SCRIPT_DIR/nginx/ssl/key.pem" ]; then
        log "Generating SSL certificates..."
        chmod +x "$SCRIPT_DIR/generate-ssl.sh"
        "$SCRIPT_DIR/generate-ssl.sh"
    else
        log "SSL certificates already exist, skipping generation"
    fi
}

# Setup Grafana configuration
setup_grafana() {
    log "Setting up Grafana configuration..."

    # Create datasource configuration
    cat > "$SCRIPT_DIR/grafana/datasources/prometheus.yml" << EOF
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF

    # Create dashboard configuration
    cat > "$SCRIPT_DIR/grafana/dashboards/dashboard.yml" << EOF
apiVersion: 1
providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    options:
      path: /etc/grafana/provisioning/dashboards
EOF

    success "Grafana configuration created"
}

# Build VecLite image
build_image() {
    log "Building VecLite Docker image..."

    cd "$PROJECT_DIR"
    if docker build -f Dockerfile -t veclite:latest .; then
        success "VecLite image built successfully"
    else
        error "Failed to build VecLite image"
        exit 1
    fi
}

# Deploy with Docker Compose
deploy() {
    log "Deploying VecLite stack..."

    cd "$SCRIPT_DIR"

    # Pull latest images for other services
    docker-compose pull prometheus grafana nginx

    # Start the stack
    if docker-compose up -d; then
        success "VecLite stack deployed successfully"
    else
        error "Failed to deploy VecLite stack"
        exit 1
    fi
}

# Wait for services to be healthy
wait_for_services() {
    log "Waiting for services to become healthy..."

    local timeout=120
    local elapsed=0
    local interval=5

    while [ $elapsed -lt $timeout ]; do
        if docker-compose ps | grep -q "healthy\|Up"; then
            log "Checking VecLite health..."
            if curl -f http://localhost:8080/health &> /dev/null; then
                success "VecLite is healthy and ready"
                return 0
            fi
        fi

        log "Waiting for services... ($elapsed/$timeout seconds)"
        sleep $interval
        elapsed=$((elapsed + interval))
    done

    error "Services failed to become healthy within $timeout seconds"
    docker-compose logs veclite
    exit 1
}

# Show deployment information
show_info() {
    echo ""
    echo "=========================================="
    echo "  VecLite Production Deployment Complete"
    echo "=========================================="
    echo ""
    echo "Services:"
    echo "  VecLite API:     http://localhost:8080"
    echo "  VecLite HTTPS:   https://veclite.local (with /etc/hosts entry)"
    echo "  Prometheus:      http://localhost:9090"
    echo "  Grafana:         http://localhost:3000 (admin/admin)"
    echo ""
    echo "Useful Commands:"
    echo "  View logs:       docker-compose logs -f veclite"
    echo "  Check status:    docker-compose ps"
    echo "  Stop services:   docker-compose down"
    echo "  Restart:         docker-compose restart veclite"
    echo ""
    echo "Data Locations:"
    echo "  Database:        $SCRIPT_DIR/data/"
    echo "  Logs:            $SCRIPT_DIR/logs/"
    echo "  Configuration:   $SCRIPT_DIR/veclite.toml"
    echo ""
    echo "For production use, remember to:"
    echo "  - Replace self-signed certificates with trusted ones"
    echo "  - Configure proper authentication"
    echo "  - Set up log rotation"
    echo "  - Configure backups"
    echo "  - Review and adjust resource limits"
    echo ""
}

# Cleanup function for errors
cleanup() {
    if [ $? -ne 0 ]; then
        error "Deployment failed. Cleaning up..."
        docker-compose down 2>/dev/null || true
    fi
}

# Main deployment process
main() {
    trap cleanup EXIT

    echo "Starting VecLite production deployment..."
    echo ""

    check_prerequisites
    create_directories
    setup_ssl
    setup_grafana
    build_image
    deploy
    wait_for_services
    show_info

    trap - EXIT
    success "Deployment completed successfully!"
}

# Parse command line arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "stop")
        log "Stopping VecLite services..."
        cd "$SCRIPT_DIR"
        docker-compose down
        success "Services stopped"
        ;;
    "restart")
        log "Restarting VecLite services..."
        cd "$SCRIPT_DIR"
        docker-compose restart
        success "Services restarted"
        ;;
    "logs")
        cd "$SCRIPT_DIR"
        docker-compose logs -f veclite
        ;;
    "status")
        cd "$SCRIPT_DIR"
        docker-compose ps
        echo ""
        echo "Health status:"
        curl -s http://localhost:8080/health && echo "VecLite: Healthy" || echo "VecLite: Unhealthy"
        ;;
    "clean")
        log "Cleaning up VecLite deployment..."
        cd "$SCRIPT_DIR"
        docker-compose down -v
        docker image rm veclite:latest 2>/dev/null || true
        success "Cleanup completed"
        ;;
    "help")
        echo "VecLite Production Deployment Script"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  deploy     Deploy VecLite stack (default)"
        echo "  stop       Stop all services"
        echo "  restart    Restart all services"
        echo "  logs       View VecLite logs"
        echo "  status     Show service status"
        echo "  clean      Remove all containers and data"
        echo "  help       Show this help message"
        ;;
    *)
        error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac