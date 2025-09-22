# VecLite Performance Optimization Guide

This guide provides comprehensive performance optimization strategies for VecLite vector search operations.

## Table of Contents

1. [Performance Overview](#performance-overview)
2. [Hardware Considerations](#hardware-considerations)
3. [Configuration Tuning](#configuration-tuning)
4. [Indexing Strategies](#indexing-strategies)
5. [Distance Metric Selection](#distance-metric-selection)
6. [Memory Optimization](#memory-optimization)
7. [Batch Operations](#batch-operations)
8. [Persistence Optimization](#persistence-optimization)
9. [Benchmarking and Profiling](#benchmarking-and-profiling)
10. [Production Deployment](#production-deployment)

## Performance Overview

### Key Metrics

VecLite performance is measured across several dimensions:

- **Insertion Rate**: Vectors inserted per second
- **Search Latency**: Time to complete similarity search
- **Search Throughput**: Searches per second
- **Memory Usage**: RAM consumption
- **Index Build Time**: Time to construct search indices
- **Recall@K**: Search accuracy (fraction of true neighbors found)

### Baseline Performance

On modern hardware (Intel i7, 16GB RAM), VecLite achieves:

| Operation | Performance | Configuration |
|-----------|-------------|---------------|
| Insertion | 50,000-100,000 vectors/sec | Brute force |
| Insertion | 10,000-30,000 vectors/sec | HNSW index |
| Search | 1,000+ searches/sec | 10K vectors, k=10 |
| Search | 500+ searches/sec | 100K vectors, k=10, HNSW |
| Memory | ~400 bytes/vector | 128D vectors + metadata |

## Hardware Considerations

### CPU Requirements

- **Cores**: VecLite uses Rayon for parallelization
- **SIMD**: Enable AVX/AVX2 for distance calculations
- **Cache**: Larger L2/L3 cache improves search performance
- **Memory Bandwidth**: Important for high-dimensional vectors

### Memory Requirements

- **RAM**: 2-4x vector data size for optimal performance
- **Storage**: SSD recommended for persistence operations
- **Virtual Memory**: Configure adequate swap for large datasets

### Compiler Optimizations

Enable maximum optimizations:

```toml
[profile.release]
lto = true
codegen-units = 1
panic = "abort"
opt-level = 3
```

Build with native CPU features:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

## Configuration Tuning

### Storage Configuration

```rust
use veclite::types::StorageConfig;

let mut storage_config = StorageConfig::default();

// Set appropriate capacity
storage_config.max_vectors = 1_000_000;
storage_config.initial_capacity = 10_000;

// Memory limit (optional)
storage_config.memory_limit_bytes = Some(8 * 1024 * 1024 * 1024); // 8GB
```

### Query Configuration

```rust
use veclite::types::QueryConfig;

let mut query_config = QueryConfig::default();

// Default search parameters
query_config.default_k = 10;
query_config.max_k = 1000;

// Enable parallel search for large k
query_config.parallel_search_threshold = 100;
```

### Index Configuration

```rust
use veclite::types::{IndexConfig, IndexType, HNSWConfig};

let mut index_config = IndexConfig::default();

// Choose index type based on dataset size
if dataset_size < 10_000 {
    index_config.index_type = IndexType::BruteForce;
} else {
    index_config.index_type = IndexType::HNSW;
}

// HNSW parameters
index_config.hnsw = HNSWConfig {
    m: 16,                    // Balance build time vs accuracy
    max_m: 16,
    max_m_l: 32,
    ef_construction: 200,     // Higher = better accuracy, slower build
    ef_search: 100,          // Higher = better accuracy, slower search
    ml: 1.0 / (2.0_f64).ln(),
};
```

## Indexing Strategies

### When to Use Brute Force

Use brute force indexing when:

- Dataset size < 10,000 vectors
- Vector dimensions ≤ 64
- Perfect recall required
- Memory constraints are tight
- Updates are very frequent

### When to Use HNSW

Use HNSW indexing when:

- Dataset size > 10,000 vectors
- Vector dimensions > 64
- Search speed is critical
- Some accuracy loss is acceptable
- Dataset is relatively stable

### HNSW Parameter Tuning

#### M (connections per layer)

```rust
// Conservative (slower build, better accuracy)
config.hnsw.m = 32;
config.hnsw.max_m = 32;

// Balanced (recommended starting point)
config.hnsw.m = 16;
config.hnsw.max_m = 16;

// Aggressive (faster build, lower accuracy)
config.hnsw.m = 8;
config.hnsw.max_m = 8;
```

#### ef_construction (build quality)

```rust
// High quality (slow build, best accuracy)
config.hnsw.ef_construction = 400;

// Balanced (recommended)
config.hnsw.ef_construction = 200;

// Fast build (lower accuracy)
config.hnsw.ef_construction = 100;
```

#### ef_search (search quality)

```rust
// Runtime adjustment for accuracy/speed tradeoff
let results = db.search_with_config(&query, k, SearchConfig {
    ef_search: Some(200), // Override default
    ..Default::default()
})?;
```

## Distance Metric Selection

### Performance Characteristics

| Metric | Speed | Use Case | Notes |
|--------|-------|----------|--------|
| Dot Product | Fastest | Normalized vectors | No sqrt computation |
| Euclidean | Fast | General purpose | Most common choice |
| Cosine | Medium | Text embeddings | Good for high dimensions |
| Manhattan | Medium | Discrete features | Robust to outliers |

### Optimization Tips

1. **Pre-normalize vectors** for cosine similarity to use dot product
2. **Use appropriate data types**: f32 vs f64 based on precision needs
3. **Consider quantization** for memory-constrained environments

### Custom Distance Metrics

For specialized use cases, implement custom metrics:

```rust
use veclite::distance::DistanceMetric;

struct CustomMetric;

impl DistanceMetric for CustomMetric {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        // Implement optimized distance calculation
        // Use SIMD intrinsics for maximum performance
        todo!()
    }

    fn batch_distance(&self, query: &[f32], vectors: &[&[f32]]) -> Vec<f32> {
        // Implement batch computation for better cache efficiency
        vectors.iter().map(|v| self.distance(query, v)).collect()
    }
}
```

## Memory Optimization

### Vector Storage

Minimize memory overhead:

```rust
// Use f32 instead of f64 for coordinates
type Vector = Vec<f32>;

// Minimize metadata size
let metadata = HashMap::from([
    ("id".to_string(), id.to_string()),
    // Avoid large strings in metadata
]);
```

### Memory Monitoring

```rust
use veclite::VecLite;

let db = VecLite::new()?;
let stats = db.stats();

println!("Memory usage: {:.2} MB",
    stats.total_memory_bytes as f64 / 1024.0 / 1024.0);
println!("Average vector size: {} bytes",
    stats.average_vector_size);

// Check if approaching memory limit
if stats.total_memory_bytes > (8 * 1024 * 1024 * 1024) {
    eprintln!("Warning: High memory usage");
}
```

### Memory Pool Configuration

```rust
// Pre-allocate capacity to avoid reallocations
let mut config = VecLiteConfig::default();
config.storage.initial_capacity = expected_vector_count;
```

## Batch Operations

### Batch Insertions

Always prefer batch operations:

```rust
// Good: Batch insertion
let vectors: Vec<(String, Vec<f32>, Metadata)> = load_vectors();
db.insert_batch(vectors)?;

// Avoid: Individual insertions in loops
for (id, vector, metadata) in vectors {
    db.insert(id, vector, metadata)?; // Inefficient
}
```

### Optimal Batch Sizes

```rust
const OPTIMAL_BATCH_SIZE: usize = 1000;

fn insert_large_dataset(
    db: &VecLite,
    vectors: Vec<(String, Vec<f32>, Metadata)>
) -> Result<(), Box<dyn std::error::Error>> {
    for chunk in vectors.chunks(OPTIMAL_BATCH_SIZE) {
        db.insert_batch(chunk.to_vec())?;

        // Optional: Progress reporting
        println!("Inserted {} vectors", chunk.len());
    }
    Ok(())
}
```

### Batch Searches

For multiple queries:

```rust
// Efficient batch searching
fn batch_search(
    db: &VecLite,
    queries: &[Vec<f32>],
    k: usize
) -> Vec<Vec<SearchResult>> {
    queries.iter()
        .map(|query| db.search(query, k).unwrap_or_default())
        .collect()
}
```

## Persistence Optimization

### Compression Settings

```rust
use veclite::types::PersistenceConfig;

let mut persistence_config = PersistenceConfig::default();

// Enable compression for large databases
persistence_config.compression_enabled = true;
persistence_config.compression_level = 6; // Balance speed vs ratio

// Enable checksums for data integrity
persistence_config.checksum_enabled = true;
```

### Save/Load Strategies

```rust
// Periodic saves for large datasets
use std::time::{Duration, Instant};

let mut last_save = Instant::now();
const SAVE_INTERVAL: Duration = Duration::from_secs(300); // 5 minutes

// In your application loop
if last_save.elapsed() > SAVE_INTERVAL {
    db.save("database.vlt")?;
    last_save = Instant::now();
}
```

### File System Considerations

- Use SSDs for better I/O performance
- Consider memory-mapped files for large read-only datasets
- Place database files on fast local storage
- Use appropriate file system (ext4, NTFS, APFS)

## Benchmarking and Profiling

### Built-in Benchmarking

Use VecLite's CLI for standardized benchmarks:

```bash
# Comprehensive benchmark
veclite benchmark --count 10000 --dimensions 128 --benchmark-type all

# Specific workloads
veclite benchmark --count 50000 --dimensions 256 --benchmark-type insert
veclite benchmark --count 10000 --dimensions 512 --benchmark-type search
```

### Custom Benchmarks

```rust
use std::time::Instant;
use veclite::VecLite;

fn benchmark_insertion(vector_count: usize, dimensions: usize) {
    let db = VecLite::new().unwrap();

    let vectors: Vec<_> = (0..vector_count)
        .map(|i| {
            let vector = vec![i as f32 / 1000.0; dimensions];
            let metadata = std::collections::HashMap::new();
            (format!("vec_{}", i), vector, metadata)
        })
        .collect();

    let start = Instant::now();
    db.insert_batch(vectors).unwrap();
    let duration = start.elapsed();

    let rate = vector_count as f64 / duration.as_secs_f64();
    println!("Insertion rate: {:.0} vectors/second", rate);
}

fn benchmark_search(db: &VecLite, dimensions: usize, iterations: usize) {
    let query = vec![0.5; dimensions];

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = db.search(&query, 10).unwrap();
    }
    let duration = start.elapsed();

    let rate = iterations as f64 / duration.as_secs_f64();
    println!("Search rate: {:.0} searches/second", rate);
}
```

### Profiling Tools

Use standard Rust profiling tools:

```bash
# CPU profiling
cargo install flamegraph
cargo flamegraph --bench your_benchmark

# Memory profiling
cargo install heaptrack
heaptrack cargo run --release

# Performance analysis
cargo install cargo-profdata
cargo profdata -- -instrument-allocations your_binary
```

## Production Deployment

### System Configuration

#### Linux

```bash
# Increase memory limits
echo 'vm.max_map_count = 262144' >> /etc/sysctl.conf

# Optimize for memory-intensive applications
echo 'vm.swappiness = 1' >> /etc/sysctl.conf
echo 'vm.dirty_ratio = 15' >> /etc/sysctl.conf

# Apply changes
sysctl -p
```

#### Resource Monitoring

```rust
use std::time::{Duration, Instant};

struct PerformanceMonitor {
    start_time: Instant,
    operation_count: u64,
}

impl PerformanceMonitor {
    fn new() -> Self {
        Self {
            start_time: Instant::now(),
            operation_count: 0,
        }
    }

    fn record_operation(&mut self) {
        self.operation_count += 1;

        if self.operation_count % 10000 == 0 {
            let elapsed = self.start_time.elapsed();
            let rate = self.operation_count as f64 / elapsed.as_secs_f64();
            println!("Current rate: {:.0} ops/sec", rate);
        }
    }
}
```

### Scaling Strategies

#### Horizontal Scaling

```rust
// Shard vectors across multiple VecLite instances
struct ShardedVecLite {
    shards: Vec<VecLite>,
}

impl ShardedVecLite {
    fn insert(&self, id: String, vector: Vec<f32>, metadata: Metadata) -> Result<()> {
        let shard_index = self.hash_to_shard(&id);
        self.shards[shard_index].insert(id, vector, metadata)
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        // Search all shards and merge results
        let mut all_results = Vec::new();

        for shard in &self.shards {
            let results = shard.search(query, k)?;
            all_results.extend(results);
        }

        // Sort and return top k
        all_results.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
        all_results.truncate(k);
        Ok(all_results)
    }

    fn hash_to_shard(&self, id: &str) -> usize {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        id.hash(&mut hasher);
        (hasher.finish() as usize) % self.shards.len()
    }
}
```

#### Load Balancing

```rust
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};

struct LoadBalancer {
    databases: Vec<Arc<VecLite>>,
    counter: AtomicUsize,
}

impl LoadBalancer {
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        let index = self.counter.fetch_add(1, Ordering::Relaxed) % self.databases.len();
        self.databases[index].search(query, k)
    }
}
```

### Error Handling and Recovery

```rust
use std::time::Duration;

fn resilient_search(
    db: &VecLite,
    query: &[f32],
    k: usize,
    max_retries: usize
) -> Result<Vec<SearchResult>> {
    let mut attempts = 0;

    loop {
        match db.search(query, k) {
            Ok(results) => return Ok(results),
            Err(e) if attempts < max_retries => {
                eprintln!("Search attempt {} failed: {}", attempts + 1, e);
                attempts += 1;
                std::thread::sleep(Duration::from_millis(100 * attempts as u64));
            }
            Err(e) => return Err(e),
        }
    }
}
```

## Performance Checklist

### Development

- [ ] Use release builds for benchmarking
- [ ] Enable native CPU optimizations
- [ ] Choose appropriate vector dimensions
- [ ] Select optimal distance metric
- [ ] Use batch operations
- [ ] Monitor memory usage

### Configuration

- [ ] Set appropriate max_vectors limit
- [ ] Configure HNSW parameters for your data
- [ ] Enable compression for large databases
- [ ] Set reasonable search limits (max_k)
- [ ] Configure memory limits appropriately

### Production

- [ ] Profile application under realistic load
- [ ] Set up performance monitoring
- [ ] Configure system-level optimizations
- [ ] Plan for horizontal scaling
- [ ] Implement error recovery strategies
- [ ] Set up automated benchmarking

### Monitoring

- [ ] Track insertion/search rates
- [ ] Monitor memory usage trends
- [ ] Measure search accuracy (recall@k)
- [ ] Profile query latency distribution
- [ ] Monitor file I/O performance

## Conclusion

VecLite performance optimization requires attention to:

1. **Hardware characteristics** and system configuration
2. **Algorithm selection** (brute force vs HNSW)
3. **Parameter tuning** for your specific use case
4. **Memory management** and batch operations
5. **Production deployment** strategies

Regular benchmarking and profiling are essential for maintaining optimal performance as your dataset and query patterns evolve.

For additional help:

- Check the [API documentation](target/doc/veclite/index.html)
- Review [benchmark results](benchmarks/)
- Join discussions on [GitHub](https://github.com/user/veclite/discussions)