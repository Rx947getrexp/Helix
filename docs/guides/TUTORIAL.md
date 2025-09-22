# VecLite Tutorial

A comprehensive guide to using VecLite, the lightweight embeddable vector search library.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Operations](#basic-operations)
3. [Advanced Features](#advanced-features)
4. [Command Line Interface](#command-line-interface)
5. [Performance Optimization](#performance-optimization)
6. [Go Bindings](#go-bindings)
7. [Examples](#examples)

## Getting Started

### Installation

Add VecLite to your `Cargo.toml`:

```toml
[dependencies]
veclite = "0.1"

# Optional features
[features]
default = ["compression"]
compression = ["zstd"]
cli = ["clap", "serde_json"]
ffi = []
```

### Quick Start

```rust
use veclite::{VecLite, VecLiteConfig};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a new VecLite database
    let db = VecLite::new()?;

    // Insert vectors with metadata
    let vector = vec![1.0, 2.0, 3.0];
    let metadata = HashMap::from([
        ("type".to_string(), "document".to_string()),
        ("title".to_string(), "My First Vector".to_string()),
    ]);

    db.insert("doc1".to_string(), vector, metadata)?;

    // Search for similar vectors
    let query = vec![1.1, 2.1, 3.1];
    let results = db.search(&query, 5)?;

    for result in results {
        println!("Found: {} (score: {:.4})", result.id, result.score);
    }

    Ok(())
}
```

## Basic Operations

### Creating a Database

VecLite can be created with default or custom configuration:

```rust
use veclite::{VecLite, VecLiteConfig};

// Default configuration
let db = VecLite::new()?;

// Custom configuration
let mut config = VecLiteConfig::default();
config.storage.max_vectors = 100_000;
config.query.default_k = 20;

let db = VecLite::with_config(config)?;
```

### Inserting Vectors

Single vector insertion:

```rust
use std::collections::HashMap;

let vector = vec![1.0, 2.0, 3.0, 4.0];
let metadata = HashMap::from([
    ("category".to_string(), "example".to_string()),
    ("timestamp".to_string(), "2024-01-01".to_string()),
]);

db.insert("example_1".to_string(), vector, metadata)?;
```

Batch insertion for better performance:

```rust
let vectors = vec![
    ("vec1".to_string(), vec![1.0, 0.0, 0.0], HashMap::new()),
    ("vec2".to_string(), vec![0.0, 1.0, 0.0], HashMap::new()),
    ("vec3".to_string(), vec![0.0, 0.0, 1.0], HashMap::new()),
];

db.insert_batch(vectors)?;
```

### Retrieving Vectors

Get a specific vector by ID:

```rust
if let Some(item) = db.get("vec1")? {
    println!("Vector: {:?}", item.vector);
    println!("Metadata: {:?}", item.metadata);
    println!("Timestamp: {}", item.timestamp);
}
```

Check if a vector exists:

```rust
if db.exists("vec1")? {
    println!("Vector exists!");
}
```

### Searching Vectors

Basic similarity search:

```rust
let query = vec![0.9, 0.1, 0.1];
let results = db.search(&query, 10)?; // Find top 10 similar vectors

for result in results {
    println!("{}: {:.4}", result.id, result.score);
}
```

Search with specific distance metric:

```rust
let results = db.search_with_metric(&query, 10, "cosine")?;
```

Search with metadata filtering:

```rust
let results = db.search_with_filter(&query, 10, |metadata| {
    metadata.get("category") == Some(&"document".to_string())
})?;
```

### Deleting Vectors

```rust
if db.delete("vec1")? {
    println!("Vector deleted successfully");
} else {
    println!("Vector not found");
}
```

### Database Statistics

```rust
let stats = db.stats();
println!("Vector count: {}", stats.vector_count);
println!("Memory usage: {} bytes", stats.total_memory_bytes);
println!("Dimensions: {:?}", stats.dimensions);
```

## Advanced Features

### Distance Metrics

VecLite supports multiple distance metrics:

```rust
// Available metrics
let metrics = VecLite::available_metrics();
println!("Available: {:?}", metrics);

// Use different metrics
let euclidean_results = db.search_with_metric(&query, 5, "euclidean")?;
let cosine_results = db.search_with_metric(&query, 5, "cosine")?;
let manhattan_results = db.search_with_metric(&query, 5, "manhattan")?;
let dot_product_results = db.search_with_metric(&query, 5, "dot_product")?;
```

### HNSW Index for Large Datasets

For better performance with large datasets, enable HNSW indexing:

```rust
use veclite::types::{IndexType, HNSWConfig};

let mut config = VecLiteConfig::default();
config.index.index_type = IndexType::HNSW;
config.index.hnsw = HNSWConfig {
    m: 16,                    // Number of connections
    max_m: 16,               // Max connections at layer 0
    max_m_l: 32,             // Max connections at higher layers
    ef_construction: 200,     // Size of dynamic candidate list
    ef_search: 100,          // Size of dynamic candidate list for search
    ml: 1.0 / (2.0_f64).ln(), // Level generation factor
};

let db = VecLite::with_config(config)?;
```

### Persistence

Save and load databases:

```rust
// Save to file
db.save("my_database.vlt")?;

// Load from file
let loaded_db = VecLite::load("my_database.vlt")?;

// Verify loaded data
assert_eq!(db.len(), loaded_db.len());
```

### Configuration Options

```rust
use veclite::types::*;

let mut config = VecLiteConfig::default();

// Storage configuration
config.storage.max_vectors = 1_000_000;
config.storage.initial_capacity = 10_000;

// Query configuration
config.query.default_k = 20;
config.query.max_k = 1000;

// Index configuration
config.index.index_type = IndexType::HNSW;
config.index.hnsw.ef_construction = 400;

// Persistence configuration
config.persistence.compression_enabled = true;
config.persistence.compression_level = 6;
config.persistence.checksum_enabled = true;

let db = VecLite::with_config(config)?;
```

## Command Line Interface

VecLite includes a comprehensive CLI tool:

### Database Management

```bash
# Create a new database
veclite database create --max-vectors 100000 --default-k 10

# Show database information
veclite database info

# Validate database integrity
veclite database validate

# Compact database
veclite database compact
```

### Vector Operations

```bash
# Insert a vector
veclite vector insert my_vec --vector "1.0,2.0,3.0" --metadata '{"type":"test"}'

# Get a vector
veclite vector get my_vec --format json

# Search for similar vectors
veclite vector search "1.1,2.1,3.1" --k 5 --metric euclidean

# Delete a vector
veclite vector delete my_vec

# Import vectors from JSON lines file
veclite vector import vectors.jsonl --batch-size 1000

# Export vectors
veclite vector export output.jsonl --format json-lines
```

### Performance Benchmarking

```bash
# Run insertion benchmark
veclite benchmark --count 10000 --dimensions 128 --benchmark-type insert

# Run search benchmark
veclite benchmark --count 10000 --dimensions 128 --benchmark-type search

# Run mixed workload
veclite benchmark --count 5000 --dimensions 128 --benchmark-type mixed

# Run all benchmarks
veclite benchmark --count 10000 --dimensions 128 --benchmark-type all
```

## Performance Optimization

### Vector Dimensions

- **Small dimensions (≤64)**: Use brute force search for best accuracy
- **Medium dimensions (64-512)**: HNSW provides good speed/accuracy tradeoff
- **Large dimensions (≥512)**: HNSW is essential for reasonable performance

### Batch Operations

Always prefer batch operations when possible:

```rust
// Good: Batch insertion
let vectors = vec![/* ... */];
db.insert_batch(vectors)?;

// Less efficient: Individual insertions
for (id, vector, metadata) in vectors {
    db.insert(id, vector, metadata)?;
}
```

### Memory Management

Monitor memory usage:

```rust
let stats = db.stats();
println!("Memory usage: {:.2} MB", stats.total_memory_bytes as f64 / 1024.0 / 1024.0);

// Clear database when needed
db.clear()?;
```

### Distance Metric Selection

Choose the right metric for your use case:

- **Euclidean**: General-purpose, good for normalized vectors
- **Cosine**: Best for text embeddings and high-dimensional sparse vectors
- **Manhattan**: Robust to outliers, good for discrete features
- **Dot Product**: Fast, good when vectors have meaningful magnitudes

### HNSW Parameters

Tune HNSW parameters based on your needs:

```rust
// High accuracy, slower build
config.index.hnsw.ef_construction = 400;
config.index.hnsw.m = 32;

// Fast build, lower accuracy
config.index.hnsw.ef_construction = 100;
config.index.hnsw.m = 8;

// Balanced (recommended starting point)
config.index.hnsw.ef_construction = 200;
config.index.hnsw.m = 16;
```

## Go Bindings

VecLite provides complete Go language bindings:

### Installation

```go
import "veclite"
```

### Basic Usage

```go
package main

import (
    "fmt"
    "log"
    "veclite"
)

func main() {
    // Create database
    db, err := veclite.New()
    if err != nil {
        log.Fatal(err)
    }
    defer db.Close()

    // Insert vector
    vector := veclite.Vector{1.0, 2.0, 3.0}
    metadata := veclite.Metadata{
        "type":  "document",
        "title": "Example",
    }

    err = db.Insert("doc1", vector, metadata)
    if err != nil {
        log.Fatal(err)
    }

    // Search
    query := veclite.Vector{1.1, 2.1, 3.1}
    results, err := db.Search(query, 5)
    if err != nil {
        log.Fatal(err)
    }

    for _, result := range results {
        fmt.Printf("Found: %s (score: %.4f)\n", result.ID, result.Score)
    }
}
```

## Examples

### Example 1: Document Search

```rust
use veclite::VecLite;
use std::collections::HashMap;

fn document_search_example() -> Result<(), Box<dyn std::error::Error>> {
    let db = VecLite::new()?;

    // Insert document vectors (simulated embeddings)
    let documents = vec![
        ("doc1", vec![0.1, 0.2, 0.3], "Introduction to Rust"),
        ("doc2", vec![0.2, 0.1, 0.4], "Advanced Rust Programming"),
        ("doc3", vec![0.8, 0.1, 0.1], "Python Machine Learning"),
        ("doc4", vec![0.1, 0.9, 0.0], "JavaScript Web Development"),
    ];

    for (id, vector, title) in documents {
        let metadata = HashMap::from([
            ("title".to_string(), title.to_string()),
            ("type".to_string(), "document".to_string()),
        ]);
        db.insert(id.to_string(), vector, metadata)?;
    }

    // Search for Rust-related documents
    let rust_query = vec![0.15, 0.15, 0.35]; // Similar to Rust docs
    let results = db.search(&rust_query, 2)?;

    println!("Rust-related documents:");
    for result in results {
        if let Some(title) = result.metadata.get("title") {
            println!("  {}: {} (score: {:.3})", result.id, title, result.score);
        }
    }

    Ok(())
}
```

### Example 2: Image Feature Matching

```rust
use veclite::{VecLite, VecLiteConfig};
use veclite::types::IndexType;

fn image_matching_example() -> Result<(), Box<dyn std::error::Error>> {
    // Configure for large-scale image search
    let mut config = VecLiteConfig::default();
    config.index.index_type = IndexType::HNSW;
    config.storage.max_vectors = 1_000_000;

    let db = VecLite::with_config(config)?;

    // Simulate image feature vectors (e.g., from CNN)
    let images = generate_image_features(1000, 2048)?;

    // Batch insert for performance
    db.insert_batch(images)?;

    // Find similar images
    let query_image = vec![0.1; 2048]; // Query image features
    let similar_images = db.search_with_metric(&query_image, 10, "cosine")?;

    println!("Similar images:");
    for result in similar_images {
        println!("  Image {}: similarity {:.3}", result.id, 1.0 - result.score);
    }

    Ok(())
}

fn generate_image_features(count: usize, dimensions: usize)
    -> Result<Vec<(String, Vec<f32>, std::collections::HashMap<String, String>)>, Box<dyn std::error::Error>> {
    use rand::Rng;

    let mut rng = rand::thread_rng();
    let mut images = Vec::new();

    for i in 0..count {
        let features: Vec<f32> = (0..dimensions)
            .map(|_| rng.gen::<f32>())
            .collect();

        let metadata = std::collections::HashMap::from([
            ("type".to_string(), "image".to_string()),
            ("category".to_string(), format!("cat_{}", i % 10)),
        ]);

        images.push((format!("img_{}", i), features, metadata));
    }

    Ok(images)
}
```

### Example 3: Real-time Recommendation System

```rust
use veclite::{VecLite, VecLiteConfig};
use std::collections::HashMap;
use std::time::Instant;

fn recommendation_system_example() -> Result<(), Box<dyn std::error::Error>> {
    let mut config = VecLiteConfig::default();
    config.query.default_k = 50;

    let db = VecLite::with_config(config)?;

    // Simulate user profiles and item features
    populate_recommendation_data(&db)?;

    // Real-time recommendation
    let user_profile = vec![0.8, 0.2, 0.6, 0.1, 0.9]; // User preferences

    let start = Instant::now();
    let recommendations = db.search_with_filter(&user_profile, 10, |metadata| {
        // Filter for available items only
        metadata.get("status") == Some(&"available".to_string())
    })?;
    let duration = start.elapsed();

    println!("Recommendations (found in {:.2}ms):", duration.as_millis());
    for (i, result) in recommendations.iter().enumerate() {
        if let Some(title) = result.metadata.get("title") {
            println!("  {}. {}: {:.3} match", i + 1, title, 1.0 - result.score);
        }
    }

    Ok(())
}

fn populate_recommendation_data(db: &VecLite) -> Result<(), Box<dyn std::error::Error>> {
    let items = vec![
        ("item1", vec![0.9, 0.1, 0.7, 0.0, 0.8], "Rust Programming Book", "available"),
        ("item2", vec![0.2, 0.8, 0.1, 0.9, 0.3], "Cooking Recipes", "available"),
        ("item3", vec![0.7, 0.3, 0.8, 0.2, 0.9], "Advanced Algorithms", "out_of_stock"),
        ("item4", vec![0.8, 0.2, 0.5, 0.1, 0.7], "System Design Guide", "available"),
    ];

    for (id, features, title, status) in items {
        let metadata = HashMap::from([
            ("title".to_string(), title.to_string()),
            ("status".to_string(), status.to_string()),
            ("type".to_string(), "product".to_string()),
        ]);
        db.insert(id.to_string(), features, metadata)?;
    }

    Ok(())
}
```

## Best Practices

1. **Choose appropriate vector dimensions**: Start with smaller dimensions and increase as needed
2. **Use batch operations**: Always prefer batch insertions for better performance
3. **Monitor memory usage**: Check statistics regularly in production
4. **Select the right distance metric**: Test different metrics with your data
5. **Tune HNSW parameters**: Balance build time vs search accuracy
6. **Enable compression**: Use compression for persistent storage
7. **Regular validation**: Validate database integrity after major operations
8. **Error handling**: Always handle errors gracefully in production code

## Troubleshooting

### Common Issues

1. **Dimension mismatch**: Ensure all vectors have the same dimensions
2. **Memory issues**: Monitor memory usage and adjust `max_vectors` accordingly
3. **Poor search accuracy**: Try different distance metrics or HNSW parameters
4. **Slow performance**: Use HNSW for large datasets, batch operations
5. **File corruption**: Enable checksums for persistent storage

### Performance Tips

1. Use Release builds: `cargo build --release`
2. Enable appropriate CPU features: `RUSTFLAGS="-C target-cpu=native"`
3. Batch operations when possible
4. Choose optimal HNSW parameters for your use case
5. Consider memory-mapping for very large databases

## Next Steps

- Explore the [API documentation](target/doc/veclite/index.html)
- Check out the [Go bindings](go/README.md)
- Read the [performance benchmarks](benchmarks/README.md)
- Contribute to the project on [GitHub](https://github.com/user/veclite)