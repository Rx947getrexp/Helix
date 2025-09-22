#!/bin/bash
set -e

# VecLite Docker Entrypoint Script
# Provides flexible container startup options

# Print banner
echo "==========================================="
echo "         VecLite Container Starting        "
echo "==========================================="

# Show environment info
echo "VecLite Version: $(veclite --version)"
echo "Data Directory: ${VECLITE_DATA_DIR:-/data}"
echo "Config Path: ${VECLITE_CONFIG_PATH:-/config/veclite.toml}"
echo "Log Level: ${VECLITE_LOG_LEVEL:-info}"

# Create data directory if it doesn't exist
if [ ! -d "${VECLITE_DATA_DIR:-/data}" ]; then
    echo "Creating data directory: ${VECLITE_DATA_DIR:-/data}"
    mkdir -p "${VECLITE_DATA_DIR:-/data}"
fi

# Create logs directory if it doesn't exist
if [ ! -d "/logs" ]; then
    echo "Creating logs directory: /logs"
    mkdir -p "/logs"
fi

# Function to run VecLite server
run_server() {
    echo "Starting VecLite server..."

    # Check if database exists, create if not
    if [ ! -f "${VECLITE_DATA_DIR:-/data}/database.vlt" ]; then
        echo "No existing database found, will create new one on first use"
    else
        echo "Using existing database: ${VECLITE_DATA_DIR:-/data}/database.vlt"

        # Validate database integrity
        if ! veclite database validate "${VECLITE_DATA_DIR:-/data}/database.vlt"; then
            echo "WARNING: Database validation failed!"
            echo "Database may be corrupted. Please check manually."
        else
            echo "Database validation passed"
        fi
    fi

    # Start the server
    exec veclite server --config "${VECLITE_CONFIG_PATH:-/config/veclite.toml}"
}

# Function to run database operations
run_database() {
    echo "Running database operation: $*"
    exec veclite database "$@"
}

# Function to run vector operations
run_vector() {
    echo "Running vector operation: $*"
    exec veclite vector "$@"
}

# Function to run benchmark
run_benchmark() {
    echo "Running benchmark: $*"
    exec veclite benchmark "$@"
}

# Function to show help
show_help() {
    echo "VecLite Docker Container Usage:"
    echo ""
    echo "Commands:"
    echo "  server                    Start VecLite server (default)"
    echo "  database <subcommand>     Run database operations"
    echo "  vector <subcommand>       Run vector operations"
    echo "  benchmark <options>       Run performance benchmark"
    echo "  help                      Show this help message"
    echo ""
    echo "Examples:"
    echo "  docker run veclite/veclite:latest"
    echo "  docker run veclite/veclite:latest database info /data/database.vlt"
    echo "  docker run veclite/veclite:latest benchmark --vectors 1000"
    echo ""
    echo "Environment Variables:"
    echo "  VECLITE_CONFIG_PATH       Path to configuration file"
    echo "  VECLITE_DATA_DIR          Data directory path"
    echo "  VECLITE_LOG_LEVEL         Log level (debug, info, warn, error)"
    echo ""
}

# Main command handler
case "$1" in
    "server")
        run_server
        ;;
    "database")
        shift
        run_database "$@"
        ;;
    "vector")
        shift
        run_vector "$@"
        ;;
    "benchmark")
        shift
        run_benchmark "$@"
        ;;
    "help"|"--help"|"-h")
        show_help
        ;;
    "")
        # Default to server if no command specified
        run_server
        ;;
    *)
        echo "Unknown command: $1"
        echo "Use 'help' for usage information"
        exit 1
        ;;
esac