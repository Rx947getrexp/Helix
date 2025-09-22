# VecLite Production Deployment Guide

This guide provides comprehensive instructions for deploying VecLite in production environments, including Docker containerization, performance tuning, and operational best practices.

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation Methods](#installation-methods)
- [Docker Deployment](#docker-deployment)
- [Configuration](#configuration)
- [Performance Tuning](#performance-tuning)
- [Monitoring and Observability](#monitoring-and-observability)
- [Security Considerations](#security-considerations)
- [Backup and Recovery](#backup-and-recovery)
- [Scaling Strategies](#scaling-strategies)
- [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements
- **CPU**: 2 cores, x86_64 architecture
- **Memory**: 4GB RAM
- **Storage**: 10GB available disk space
- **OS**: Linux (Ubuntu 20.04+, CentOS 8+, RHEL 8+), macOS 11+, Windows 10+

### Recommended Production Requirements
- **CPU**: 8+ cores with AVX2 support for SIMD optimizations
- **Memory**: 16GB+ RAM for optimal performance
- **Storage**: SSD with 100GB+ available space
- **Network**: 1Gbps+ network interface
- **OS**: Linux (Ubuntu 22.04 LTS recommended)

### Hardware Optimization
- **SIMD Support**: Intel CPUs with AVX2 or ARM with NEON for optimal distance calculations
- **Memory**: ECC RAM recommended for data integrity
- **Storage**: NVMe SSD for fast I/O operations
- **CPU Cache**: Larger L3 cache improves vector search performance

## Installation Methods

### Method 1: Binary Release (Recommended)

```bash
# Download the latest release
wget https://github.com/user/veclite/releases/latest/download/veclite-linux-x86_64.tar.gz

# Extract and install
tar -xzf veclite-linux-x86_64.tar.gz
sudo mv veclite /usr/local/bin/
sudo chmod +x /usr/local/bin/veclite

# Verify installation
veclite --version
```

### Method 2: Build from Source

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Clone and build
git clone https://github.com/user/veclite.git
cd veclite
cargo build --release --features performance

# Install binary
sudo cp target/release/veclite /usr/local/bin/
```

### Method 3: Package Managers

```bash
# Ubuntu/Debian (when available)
sudo apt update
sudo apt install veclite

# CentOS/RHEL (when available)
sudo yum install veclite

# macOS with Homebrew (when available)
brew install veclite
```

## Docker Deployment

### Quick Start with Docker

```bash
# Pull the official image
docker pull veclite/veclite:latest

# Run with default configuration
docker run -d \
  --name veclite \
  -p 8080:8080 \
  -v veclite_data:/data \
  veclite/veclite:latest
```

### Production Docker Deployment

Create a `docker-compose.yml` file:

```yaml
version: '3.8'

services:
  veclite:
    image: veclite/veclite:latest
    container_name: veclite-prod
    restart: unless-stopped
    ports:
      - "8080:8080"
    volumes:
      - ./data:/data
      - ./config:/config
      - ./logs:/logs
    environment:
      - VECLITE_CONFIG_PATH=/config/veclite.toml
      - VECLITE_LOG_LEVEL=info
      - VECLITE_DATA_DIR=/data
    healthcheck:
      test: ["CMD", "veclite", "database", "info", "/data/database.vlt"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    resources:
      limits:
        memory: 8G
        cpus: '4'
      reservations:
        memory: 2G
        cpus: '1'

  # Optional: Monitoring with Prometheus
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

volumes:
  prometheus_data:
```

### Multi-Stage Dockerfile

```dockerfile
# Build stage
FROM rust:1.75 as builder

WORKDIR /app
COPY . .
RUN cargo build --release --features performance

# Runtime stage
FROM ubuntu:22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create veclite user
RUN useradd -r -s /bin/false veclite

# Copy binary
COPY --from=builder /app/target/release/veclite /usr/local/bin/veclite
RUN chmod +x /usr/local/bin/veclite

# Create directories
RUN mkdir -p /data /config /logs && \
    chown -R veclite:veclite /data /config /logs

# Copy default configuration
COPY docker/veclite.toml /config/veclite.toml

USER veclite
WORKDIR /data

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD veclite database info /data/database.vlt || exit 1

CMD ["veclite", "server", "--config", "/config/veclite.toml"]
```

## Configuration

### Production Configuration File

Create `/etc/veclite/veclite.toml`:

```toml
[server]
host = "0.0.0.0"
port = 8080
workers = 8
max_connections = 1000

[storage]
data_dir = "/var/lib/veclite"
max_vectors = 10_000_000
max_dimensions = 4096
memory_limit_bytes = 8_589_934_592  # 8GB
enable_checksums = true

[memory]
max_memory_bytes = 8_589_934_592    # 8GB
enable_pooling = true
pool_max_vectors = 1000
pool_max_pools = 10000
enable_monitoring = true
warning_threshold_percent = 80
cleanup_threshold_percent = 90
enable_auto_cleanup = true
cleanup_interval_seconds = 300      # 5 minutes

[index]
index_type = "HNSW"

[index.hnsw]
m = 16
max_m = 32
max_m_l = 16
ef_construction = 400
ef_search = 100
ml = 1.44269504088896

[query]
default_k = 10
max_k = 1000
ef_search = 100
enable_metadata_filter = true

[persistence]
compression_enabled = true
compression_level = 6
checksum_enabled = true
backup_count = 5

[logging]
level = "info"
file = "/var/log/veclite/veclite.log"
max_size = "100MB"
max_files = 10

[metrics]
enabled = true
endpoint = "/metrics"
interval_seconds = 60
```

### Environment Variables

```bash
# Core settings
export VECLITE_CONFIG_PATH="/etc/veclite/veclite.toml"
export VECLITE_DATA_DIR="/var/lib/veclite"
export VECLITE_LOG_LEVEL="info"

# Performance settings
export VECLITE_MEMORY_LIMIT="8G"
export VECLITE_WORKER_THREADS="8"
export VECLITE_ENABLE_SIMD="true"

# Security settings
export VECLITE_TLS_CERT_PATH="/etc/ssl/veclite/cert.pem"
export VECLITE_TLS_KEY_PATH="/etc/ssl/veclite/key.pem"
export VECLITE_API_KEY_FILE="/etc/veclite/api_keys"
```

## Performance Tuning

### Memory Configuration

```toml
[memory]
# Set to 70-80% of available system RAM
max_memory_bytes = 13_421_772_800   # 12.5GB for 16GB system

# Optimize pool settings for workload
pool_max_vectors = 2000             # Higher for read-heavy workloads
pool_max_pools = 20000              # More pools for diverse dimensions

# Aggressive cleanup for memory-constrained environments
warning_threshold_percent = 70
cleanup_threshold_percent = 80
cleanup_interval_seconds = 120
```

### HNSW Index Tuning

```toml
[index.hnsw]
# Balanced settings for most workloads
m = 16                  # Good balance of speed and memory
ef_construction = 400   # Higher for better accuracy
ef_search = 100         # Adjust based on speed vs accuracy needs

# High-performance settings (more memory, better accuracy)
# m = 32
# ef_construction = 800
# ef_search = 200

# Memory-constrained settings (less memory, faster but less accurate)
# m = 8
# ef_construction = 200
# ef_search = 50
```

### Operating System Tuning

```bash
# Increase file descriptor limits
echo "veclite soft nofile 65536" >> /etc/security/limits.conf
echo "veclite hard nofile 65536" >> /etc/security/limits.conf

# Optimize virtual memory
echo "vm.swappiness = 10" >> /etc/sysctl.conf
echo "vm.dirty_ratio = 15" >> /etc/sysctl.conf
echo "vm.dirty_background_ratio = 5" >> /etc/sysctl.conf

# Network optimizations
echo "net.core.rmem_max = 134217728" >> /etc/sysctl.conf
echo "net.core.wmem_max = 134217728" >> /etc/sysctl.conf
echo "net.core.netdev_max_backlog = 5000" >> /etc/sysctl.conf

# Apply changes
sysctl -p
```

## Monitoring and Observability

### Metrics Collection

VecLite exposes Prometheus-compatible metrics:

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'veclite'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
    scrape_interval: 10s
```

### Key Metrics to Monitor

- **veclite_memory_usage_bytes**: Current memory usage
- **veclite_vector_count**: Total vectors in database
- **veclite_search_duration_seconds**: Search operation latency
- **veclite_insert_duration_seconds**: Insert operation latency
- **veclite_pool_efficiency_ratio**: Memory pool efficiency
- **veclite_index_build_time_seconds**: Index construction time

### Logging Configuration

```toml
[logging]
level = "info"
format = "json"
file = "/var/log/veclite/veclite.log"
max_size = "100MB"
max_files = 10
rotate_daily = true

# Log shipping to centralized logging
[logging.shipping]
enabled = true
endpoint = "https://logs.example.com/api/v1/logs"
buffer_size = 1000
flush_interval = "30s"
```

### Health Checks

```bash
#!/bin/bash
# healthcheck.sh

# Check if VecLite is responding
if ! curl -f http://localhost:8080/health >/dev/null 2>&1; then
    echo "VecLite health check failed"
    exit 1
fi

# Check memory usage
MEMORY_USAGE=$(curl -s http://localhost:8080/metrics | grep veclite_memory_usage_bytes | cut -d' ' -f2)
MEMORY_LIMIT=8589934592  # 8GB

if [ "$MEMORY_USAGE" -gt "$MEMORY_LIMIT" ]; then
    echo "Memory usage exceeded limit: $MEMORY_USAGE > $MEMORY_LIMIT"
    exit 1
fi

echo "Health check passed"
exit 0
```

## Security Considerations

### Network Security

```bash
# Firewall configuration
sudo ufw allow 8080/tcp  # VecLite API
sudo ufw allow 9090/tcp  # Prometheus (if used)
sudo ufw enable

# Use reverse proxy for TLS termination
# nginx configuration example
server {
    listen 443 ssl http2;
    server_name veclite.example.com;

    ssl_certificate /etc/ssl/certs/veclite.crt;
    ssl_certificate_key /etc/ssl/private/veclite.key;

    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Access Control

```toml
[security]
enable_auth = true
api_key_file = "/etc/veclite/api_keys"
rate_limit_per_minute = 1000
enable_cors = false
allowed_origins = ["https://app.example.com"]

[security.tls]
enabled = true
cert_file = "/etc/ssl/veclite/cert.pem"
key_file = "/etc/ssl/veclite/key.pem"
min_version = "1.2"
```

### Data Encryption

```bash
# Encrypt data at rest using LUKS
sudo cryptsetup luksFormat /dev/sdb1
sudo cryptsetup luksOpen /dev/sdb1 veclite_data
sudo mkfs.ext4 /dev/mapper/veclite_data
sudo mount /dev/mapper/veclite_data /var/lib/veclite

# Add to /etc/fstab for persistence
echo "/dev/mapper/veclite_data /var/lib/veclite ext4 defaults 0 2" >> /etc/fstab
```

## Backup and Recovery

### Automated Backup Script

```bash
#!/bin/bash
# backup_veclite.sh

BACKUP_DIR="/backup/veclite"
DATA_DIR="/var/lib/veclite"
RETENTION_DAYS=30

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Create timestamped backup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/veclite_backup_$TIMESTAMP.tar.gz"

# Stop VecLite gracefully
systemctl stop veclite

# Create compressed backup
tar -czf "$BACKUP_FILE" -C "$DATA_DIR" .

# Start VecLite
systemctl start veclite

# Clean old backups
find "$BACKUP_DIR" -name "veclite_backup_*.tar.gz" -mtime +$RETENTION_DAYS -delete

# Verify backup
if [ -f "$BACKUP_FILE" ]; then
    echo "Backup created successfully: $BACKUP_FILE"

    # Upload to cloud storage (optional)
    # aws s3 cp "$BACKUP_FILE" s3://veclite-backups/
else
    echo "Backup failed!"
    exit 1
fi
```

### Recovery Procedure

```bash
#!/bin/bash
# restore_veclite.sh

BACKUP_FILE="$1"
DATA_DIR="/var/lib/veclite"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file.tar.gz>"
    exit 1
fi

# Stop VecLite
systemctl stop veclite

# Backup current data (just in case)
mv "$DATA_DIR" "${DATA_DIR}.bak.$(date +%s)"

# Create new data directory
mkdir -p "$DATA_DIR"

# Restore from backup
tar -xzf "$BACKUP_FILE" -C "$DATA_DIR"

# Set permissions
chown -R veclite:veclite "$DATA_DIR"

# Start VecLite
systemctl start veclite

echo "Restore completed. Verify database integrity."
```

## Scaling Strategies

### Vertical Scaling

1. **Memory Scaling**: Add more RAM and increase `max_memory_bytes`
2. **CPU Scaling**: Add more cores and increase worker threads
3. **Storage Scaling**: Use faster SSDs and increase capacity

### Horizontal Scaling (Future Enhancement)

```yaml
# Multi-instance deployment with load balancing
version: '3.8'

services:
  veclite-1:
    image: veclite/veclite:latest
    volumes:
      - ./data1:/data
    environment:
      - VECLITE_INSTANCE_ID=1

  veclite-2:
    image: veclite/veclite:latest
    volumes:
      - ./data2:/data
    environment:
      - VECLITE_INSTANCE_ID=2

  load-balancer:
    image: nginx:alpine
    ports:
      - "8080:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - veclite-1
      - veclite-2
```

### Read Replicas (Future Enhancement)

```bash
# Script for creating read replicas
#!/bin/bash
create_read_replica() {
    MASTER_DATA="/var/lib/veclite/master"
    REPLICA_DATA="/var/lib/veclite/replica$1"

    # Create replica data directory
    mkdir -p "$REPLICA_DATA"

    # Copy data from master
    rsync -av "$MASTER_DATA/" "$REPLICA_DATA/"

    # Start replica in read-only mode
    docker run -d \
        --name "veclite-replica-$1" \
        -v "$REPLICA_DATA:/data" \
        -e VECLITE_READ_ONLY=true \
        veclite/veclite:latest
}
```

## Troubleshooting

### Common Issues and Solutions

#### High Memory Usage

```bash
# Check memory statistics
curl -s http://localhost:8080/metrics | grep memory

# Reduce memory limits
veclite config set memory.max_memory_bytes 4294967296  # 4GB

# Trigger manual cleanup
curl -X POST http://localhost:8080/admin/cleanup
```

#### Slow Search Performance

```bash
# Check index statistics
veclite database info /data/database.vlt

# Rebuild index with better parameters
veclite index rebuild /data/database.vlt \
    --ef-construction 800 \
    --max-m 32

# Check for memory pressure
curl -s http://localhost:8080/metrics | grep veclite_pool_efficiency
```

#### Database Corruption

```bash
# Validate database integrity
veclite database validate /data/database.vlt

# Repair database (if possible)
veclite database repair /data/database.vlt \
    --backup /backup/database_backup.vlt

# Restore from backup if repair fails
./restore_veclite.sh /backup/veclite_backup_20240101_120000.tar.gz
```

### Diagnostic Commands

```bash
# System information
veclite system info

# Performance benchmark
veclite benchmark \
    --vectors 10000 \
    --dimensions 768 \
    --k 10 \
    --threads 8

# Memory analysis
veclite memory analyze /data/database.vlt

# Export logs for analysis
journalctl -u veclite --since "1 hour ago" > /tmp/veclite.log
```

### Monitoring Alerts

```yaml
# Prometheus alerting rules
groups:
  - name: veclite
    rules:
      - alert: VecLiteHighMemoryUsage
        expr: veclite_memory_usage_bytes / veclite_memory_limit_bytes > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "VecLite memory usage is high"

      - alert: VecLiteSlowQueries
        expr: histogram_quantile(0.95, veclite_search_duration_seconds) > 1.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "VecLite queries are slow"
```

## Support and Maintenance

### Regular Maintenance Tasks

1. **Weekly**: Check logs for errors and warnings
2. **Monthly**: Review performance metrics and tune configuration
3. **Quarterly**: Update VecLite to latest version
4. **Annually**: Full backup verification and disaster recovery testing

### Getting Help

- **Documentation**: https://docs.veclite.io
- **GitHub Issues**: https://github.com/user/veclite/issues
- **Community**: https://discord.gg/veclite
- **Enterprise Support**: support@veclite.io

---

This deployment guide provides a solid foundation for running VecLite in production. Adjust configurations based on your specific workload requirements and infrastructure constraints.