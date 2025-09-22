# Multi-stage Docker build for VecLite
# Optimized for production deployment with minimal image size and security

# Build stage
FROM rust:1.75-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy source code
COPY . .

# Build with optimizations and all features
RUN cargo build --release --features performance,cli,compression

# Create a minimal runtime image
FROM ubuntu:22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    libssl3 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create veclite user for security
RUN useradd -r -s /bin/false -d /data veclite

# Copy binary from builder stage
COPY --from=builder /app/target/release/veclite /usr/local/bin/veclite
RUN chmod +x /usr/local/bin/veclite

# Create necessary directories
RUN mkdir -p /data /config /logs && \
    chown -R veclite:veclite /data /config /logs

# Copy default configuration
COPY docker/veclite.toml /config/veclite.toml
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set up environment
ENV VECLITE_CONFIG_PATH=/config/veclite.toml
ENV VECLITE_DATA_DIR=/data
ENV VECLITE_LOG_LEVEL=info

# Switch to non-root user
USER veclite
WORKDIR /data

# Expose default port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD veclite database info /data/database.vlt || exit 1

# Use entrypoint script for flexibility
ENTRYPOINT ["/entrypoint.sh"]
CMD ["server"]