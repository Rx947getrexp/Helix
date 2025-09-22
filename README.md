# Helix - Lightweight Embeddable Vector Search Library

[![Crates.io](https://img.shields.io/crates/v/helix.svg)](https://crates.io/crates/helix)
[![Documentation](https://docs.rs/helix/badge.svg)](https://docs.rs/helix)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Build Status](https://github.com/Rx947getrexp/Helix/workflows/CI/badge.svg)](https://github.com/Rx947getrexp/Helix/actions)

**SQLite for Vector Search** - A lightweight, embeddable vector search library inspired by SQLite's philosophy of being embedded, zero-configuration, and dependency-free.

## 🚀 Quick Start

```rust
use helix::Helix;
use std::collections::HashMap;

// Open or create a database (like SQLite)
let db = Helix::open("my_vectors.hlx")?;

// Insert vectors with metadata
let vector = vec![1.0, 2.0, 3.0];
let metadata = HashMap::from([
    ("title".to_string(), "My Document".to_string()),
    ("category".to_string(), "text".to_string()),
]);

db.insert("doc1".to_string(), vector, metadata)?;

// Search for similar vectors
let query = vec![1.1, 2.1, 3.1];
let results = db.search(&query, 5)?; // Find top 5 similar vectors

for result in results {
    println!("ID: {}, Score: {:.4}", result.id, result.score);
}
```

## ✨ Features

- **🔧 Zero Configuration**: Works out of the box with sensible defaults
- **📦 Single File Database**: All data stored in portable `.hlx` files
- **⚡ High Performance**: HNSW indexing with <100ms query latency
- **🔄 Dynamic Updates**: Real-time insertions and deletions
- **📏 Multiple Distance Metrics**: Euclidean, Cosine, Dot Product, Manhattan
- **🔒 Thread Safe**: Concurrent read/write operations
- **💾 Memory Efficient**: Configurable limits and memory pooling
- **🌐 Cross-Language**: Rust native with Go bindings
- **🛡️ Production Ready**: Comprehensive testing and validation

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [Tutorial](docs/guides/TUTORIAL.md) | Step-by-step introduction |
| [Performance Guide](docs/guides/PERFORMANCE_GUIDE.md) | Optimization strategies |
| [Deployment Guide](docs/guides/DEPLOYMENT_GUIDE.md) | Production deployment |
| [API Documentation](https://docs.rs/helix) | Complete API reference |

## 🎯 Use Cases

- **Desktop RAG Applications**: Local vector search without external services
- **Code Analysis Tools**: Privacy-sensitive source code search
- **Edge Devices**: Lightweight vector search with minimal footprint
- **Rapid Prototyping**: Quick vector search integration
- **Embedded Systems**: Single-file deployment with your application

## 🏗️ Architecture

Helix follows a modular architecture:

```
┌─────────────────────────────────────────────────────┐
│                Application Layer                    │
├─────────────────────────────────────────────────────┤
│     Rust API          │        Go Bindings         │
├─────────────────────────────────────────────────────┤
│  Query Engine  │  HNSW Index  │  Storage Manager   │
├─────────────────────────────────────────────────────┤
│              VLT File Format (.hlx)                 │
└─────────────────────────────────────────────────────┘
```

## 📦 Installation

### Rust

Add to your `Cargo.toml`:

```toml
[dependencies]
helix = "0.1.0"
```

### Go

```bash
go get github.com/Rx947getrexp/Helix/go
```

### CLI Tool

```bash
cargo install helix --features cli
```

## 🔧 Configuration

Helix works with zero configuration, but can be customized:

```rust
use helix::{Helix, VecLiteConfig, IndexType};

let mut config = VecLiteConfig::default();
config.storage.max_vectors = 1_000_000;
config.index.index_type = IndexType::HNSW;
config.index.hnsw.ef_construction = 200;

let db = Helix::with_config(config)?;
```

## 🎮 CLI Usage

```bash
# Create a new database
helix database create my_vectors.hlx --dimensions 768

# Insert vectors from a file
helix vectors import my_vectors.hlx vectors.json

# Search for similar vectors
helix vectors search my_vectors.hlx --query-file query.json --k 10

# Run performance benchmarks
helix benchmark my_vectors.hlx --insert-count 10000
```

## 📊 Performance

Helix delivers excellent performance for embedded use cases:

| Operation | Performance | Scale |
|-----------|-------------|-------|
| Vector Insert | <1ms single, <0.1ms batch | O(log n) |
| KNN Search | <100ms | 100k vectors, k=10 |
| Index Build | <30s | 100k vectors |
| Memory Usage | <2GB | 1M vectors (768-dim) |

## 🛡️ Security

Helix prioritizes security and privacy:

- **Local-only processing**: No network communication required
- **Memory safety**: Built with Rust's memory safety guarantees
- **Input validation**: Comprehensive validation at all boundaries
- **Secure file format**: Checksums and integrity validation

See [SECURITY.md](security/SECURITY.md) for details.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

### Development Setup

```bash
git clone https://github.com/Rx947getrexp/Helix.git
cd helix
cargo build
cargo test
```

### Project Structure

```
helix/
├── src/                 # Core Rust implementation
├── go/                  # Go language bindings
├── examples/            # Usage examples
├── docs/                # Documentation
├── tests/               # Integration tests
└── deployment/          # Deployment configurations
```

## 📄 License

This project is licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## 🙏 Acknowledgments

- Inspired by [SQLite](https://sqlite.org/)'s embedded database philosophy
- HNSW algorithm based on ["Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"](https://arxiv.org/abs/1603.09320)
- Built with the amazing Rust ecosystem

## 📞 Support

- 📖 [Documentation](https://docs.rs/helix)
- 🐛 [Issue Tracker](https://github.com/Rx947getrexp/Helix/issues)
- 💬 [Discussions](https://github.com/Rx947getrexp/Helix/discussions)

---

**Helix** - Making vector search as simple as SQLite 🚀