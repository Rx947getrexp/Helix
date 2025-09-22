# Multi-stage Dockerfile for Helix vector search library
# Security-hardened container with non-root user

# Build stage
FROM rust:1.75-slim-bookworm as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy dependency files
COPY Cargo.toml Cargo.lock ./

# Copy source code
COPY src ./src
COPY tests ./tests
COPY benches ./benches
COPY examples ./examples

# Build the release binary
RUN cargo build --release --bin helix

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r helix && useradd -r -g helix -s /bin/false helix

# Create directories with proper permissions
RUN mkdir -p /app/data /app/logs \
    && chown -R helix:helix /app

# Copy binary from builder stage
COPY --from=builder --chown=helix:helix /app/target/release/helix /usr/local/bin/helix

# Set working directory
WORKDIR /app

# Switch to non-root user
USER helix

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD helix --help || exit 1

# Default command
CMD ["helix", "--help"]

# Expose default port (if CLI tool has server mode)
EXPOSE 8080

# Labels for metadata
LABEL maintainer="Helix Team"
LABEL description="Helix vector search library - lightweight, embeddable vector database"
LABEL version="0.1.0"
LABEL security.non-root="true"