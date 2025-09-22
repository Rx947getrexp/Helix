# VecLite Examples

This directory contains practical examples demonstrating how to use VecLite for various vector search scenarios.

## Available Examples

### 📚 Basic Usage Examples
- **[basic_usage.rs](basic_usage.rs)** - Getting started with VecLite
  - Creating a database
  - Inserting vectors
  - Basic search operations
  - Working with metadata

### 🔍 Advanced Search Examples
- **[advanced_search.rs](advanced_search.rs)** - Advanced search techniques
  - Multiple distance metrics
  - Metadata filtering
  - Batch operations
  - Result customization

### ⚡ Performance Examples
- **[performance_benchmark.rs](performance_benchmark.rs)** - Performance optimization
  - Large-scale vector insertion
  - Search performance measurement
  - Memory usage optimization
  - Index configuration tuning

### 🔧 Integration Examples
- **[integration_example.rs](integration_example.rs)** - Real-world integration
  - Document similarity search
  - Recommendation system
  - Data persistence
  - Configuration management

## Running Examples

### Prerequisites
Make sure you have Rust installed and VecLite dependencies available:

```bash
# Clone the repository
git clone https://github.com/your-username/veclite.git
cd veclite

# Build the project
cargo build --release
```

### Running Individual Examples

```bash
# Basic usage example
cargo run --example basic_usage

# Advanced search example
cargo run --example advanced_search

# Performance benchmark
cargo run --example performance_benchmark

# Integration example
cargo run --example integration_example
```

### Running All Examples

```bash
# Run all examples in sequence
cargo run --example basic_usage
cargo run --example advanced_search
cargo run --example performance_benchmark
cargo run --example integration_example
```

## Example Data

Some examples use sample datasets included in the `examples/data/` directory:
- `sample_vectors.json` - Small dataset for testing
- `documents.json` - Text documents with embeddings
- `large_dataset.json` - Large dataset for performance testing

## Common Patterns

### 1. Creating a VecLite Instance
```rust
use veclite::VecLite;

// Default configuration
let db = VecLite::new()?;

// Custom configuration
let mut config = VecLiteConfig::default();
config.storage.max_vectors = 1_000_000;
let db = VecLite::with_config(config)?;
```

### 2. Working with Vectors
```rust
use std::collections::HashMap;

// Insert single vector
let vector = vec![1.0, 2.0, 3.0];
let metadata = HashMap::from([
    ("title".to_string(), "Example Document".to_string()),
    ("category".to_string(), "text".to_string()),
]);
db.insert("doc1".to_string(), vector, metadata)?;

// Search for similar vectors
let query = vec![1.1, 2.1, 3.1];
let results = db.search(&query, 5)?;
```

### 3. Persistence
```rust
// Save database
db.save("my_vectors.vlt")?;

// Load database
let db = VecLite::open("my_vectors.vlt")?;
```

## Best Practices

1. **Vector Normalization**: Normalize vectors for cosine similarity
2. **Batch Operations**: Use batch insertion for better performance
3. **Memory Management**: Monitor memory usage with large datasets
4. **Index Configuration**: Tune HNSW parameters for your use case
5. **Error Handling**: Always handle VecLiteResult properly

## Performance Tips

- Use SIMD-optimized distance metrics when available
- Configure memory limits based on your system
- Choose appropriate index type for your data size
- Use metadata filtering efficiently
- Consider vector quantization for large datasets

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `max_vectors` or enable memory pooling
2. **Slow Search**: Tune HNSW `ef_search` parameter
3. **High Memory Usage**: Enable compression or use smaller batches
4. **Dimension Mismatch**: Ensure all vectors have the same dimensions

### Getting Help

- Check the [Performance Guide](../docs/guides/PERFORMANCE_GUIDE.md)
- Read the [Deployment Guide](../docs/guides/DEPLOYMENT_GUIDE.md)
- Visit the [GitHub Issues](https://github.com/your-username/veclite/issues)
- Join our [Discussions](https://github.com/your-username/veclite/discussions)

## Contributing Examples

We welcome contributions of new examples! Please:

1. Follow the existing code style
2. Include comprehensive comments
3. Add appropriate error handling
4. Test your example thoroughly
5. Update this README if needed

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed guidelines.