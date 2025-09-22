//! Performance Benchmark Example
//!
//! This example demonstrates Helix performance characteristics:
//! - Large-scale vector insertion performance
//! - Search performance across different dataset sizes
//! - Memory usage optimization
//! - Index configuration tuning
//! - Concurrency and throughput testing
//!
//! Run with: cargo run --example performance_benchmark --release

use helix::{Helix, IndexType, VecLiteConfig};
use std::collections::HashMap;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

const DIMENSIONS: usize = 128;
const SMALL_DATASET: usize = 1_000;
const MEDIUM_DATASET: usize = 10_000;
const _LARGE_DATASET: usize = 100_000;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ Helix Performance Benchmark");
    println!("================================\n");

    // Warmup
    println!("🔥 Performing warmup...");
    warmup_benchmark()?;
    println!("✅ Warmup completed\n");

    // Benchmark 1: Insertion Performance
    println!("📥 Insertion Performance Benchmarks");
    println!("-----------------------------------");
    benchmark_insertion_performance()?;

    // Benchmark 2: Search Performance
    println!("\n🔍 Search Performance Benchmarks");
    println!("--------------------------------");
    benchmark_search_performance()?;

    // Benchmark 3: Index Configuration Impact
    println!("\n⚙️  Index Configuration Benchmarks");
    println!("----------------------------------");
    benchmark_index_configurations()?;

    // Benchmark 4: Memory Usage Analysis
    println!("\n💾 Memory Usage Analysis");
    println!("-----------------------");
    benchmark_memory_usage()?;

    // Benchmark 5: Concurrent Access Performance
    println!("\n🔄 Concurrent Access Benchmarks");
    println!("-------------------------------");
    benchmark_concurrent_access()?;

    // Benchmark 6: Batch vs Individual Operations
    println!("\n📦 Batch vs Individual Operations");
    println!("---------------------------------");
    benchmark_batch_operations()?;

    // Benchmark 7: Distance Metric Performance
    println!("\n📐 Distance Metric Performance");
    println!("------------------------------");
    benchmark_distance_metrics()?;

    println!("\n🎉 Performance benchmarking completed!");
    print_performance_recommendations();

    Ok(())
}

fn warmup_benchmark() -> Result<(), Box<dyn std::error::Error>> {
    let db = Helix::new()?;
    let vectors = generate_test_vectors(100, DIMENSIONS);

    // Insert some vectors
    db.insert_batch(vectors)?;

    // Perform some searches
    for _ in 0..10 {
        let query = generate_random_vector(DIMENSIONS);
        let _ = db.search(&query, 5)?;
    }

    Ok(())
}

fn benchmark_insertion_performance() -> Result<(), Box<dyn std::error::Error>> {
    let dataset_sizes = vec![SMALL_DATASET, MEDIUM_DATASET];

    for &size in &dataset_sizes {
        println!(
            "📊 Testing insertion of {} vectors ({} dimensions)",
            size, DIMENSIONS
        );

        // Test individual insertions
        let db = Helix::new()?;
        let vectors = generate_test_vectors(size, DIMENSIONS);

        let start_time = Instant::now();
        for (id, vector, metadata) in &vectors {
            db.insert(id.clone(), vector.clone(), metadata.clone())?;
        }
        let individual_duration = start_time.elapsed();

        let individual_rate = size as f64 / individual_duration.as_secs_f64();

        // Test batch insertions
        let db_batch = Helix::new()?;
        let start_time = Instant::now();
        db_batch.insert_batch(vectors)?;
        let batch_duration = start_time.elapsed();

        let batch_rate = size as f64 / batch_duration.as_secs_f64();

        println!(
            "   Individual: {:>8?} ({:>8.0} vectors/sec)",
            individual_duration, individual_rate
        );
        println!(
            "   Batch:      {:>8?} ({:>8.0} vectors/sec)",
            batch_duration, batch_rate
        );
        println!("   Speedup:    {:.2}x\n", batch_rate / individual_rate);
    }

    Ok(())
}

fn benchmark_search_performance() -> Result<(), Box<dyn std::error::Error>> {
    let dataset_sizes = vec![SMALL_DATASET, MEDIUM_DATASET];
    let k_values = vec![1, 5, 10, 50];

    for &size in &dataset_sizes {
        println!("🔍 Search performance on {} vectors:", size);

        // Create and populate database
        let db = Helix::new()?;
        let vectors = generate_test_vectors(size, DIMENSIONS);
        db.insert_batch(vectors)?;

        // Generate query vectors
        let queries: Vec<Vec<f32>> = (0..100)
            .map(|_| generate_random_vector(DIMENSIONS))
            .collect();

        for &k in &k_values {
            let start_time = Instant::now();

            for query in &queries {
                let _ = db.search(query, k)?;
            }

            let duration = start_time.elapsed();
            let avg_latency = duration / queries.len() as u32;
            let qps = queries.len() as f64 / duration.as_secs_f64();

            println!(
                "   K={:2} | Avg latency: {:>6?} | QPS: {:>8.0}",
                k, avg_latency, qps
            );
        }

        println!();
    }

    Ok(())
}

fn benchmark_index_configurations() -> Result<(), Box<dyn std::error::Error>> {
    let vectors = generate_test_vectors(MEDIUM_DATASET, DIMENSIONS);
    let query = generate_random_vector(DIMENSIONS);

    let configurations = vec![
        ("BruteForce", create_brute_force_config()),
        ("HNSW-Small", create_hnsw_config(8, 100, 50)),
        ("HNSW-Medium", create_hnsw_config(16, 200, 100)),
        ("HNSW-Large", create_hnsw_config(32, 400, 200)),
    ];

    println!(
        "⚙️  Comparing index configurations on {} vectors:",
        MEDIUM_DATASET
    );
    println!();

    for (name, config) in configurations {
        println!("📋 Testing {} configuration:", name);

        // Build time
        let start_time = Instant::now();
        let db = Helix::with_config(config)?;
        db.insert_batch(vectors.clone())?;
        let build_duration = start_time.elapsed();

        // Search time
        let start_time = Instant::now();
        for _ in 0..100 {
            let _ = db.search(&query, 10)?;
        }
        let search_duration = start_time.elapsed();
        let avg_search_time = search_duration / 100;

        // Memory usage
        let stats = db.stats();

        println!("   🏗️  Build time: {:>8?}", build_duration);
        println!("   🔍 Avg search: {:>8?}", avg_search_time);
        println!(
            "   💾 Memory:     {:>8} bytes",
            stats.memory.current_allocated
        );
        println!();
    }

    Ok(())
}

fn benchmark_memory_usage() -> Result<(), Box<dyn std::error::Error>> {
    let dataset_sizes = vec![1_000, 5_000, 10_000, 25_000];

    println!("💾 Memory usage scaling:");
    println!("   Vectors  |   Memory   | Per Vector");
    println!("   ---------|------------|----------");

    for &size in &dataset_sizes {
        let db = Helix::new()?;
        let vectors = generate_test_vectors(size, DIMENSIONS);

        db.insert_batch(vectors)?;

        let stats = db.stats();
        let memory_per_vector = stats.memory.current_allocated / size;

        println!(
            "   {:>8} | {:>9} B | {:>8} B",
            size, stats.memory.current_allocated, memory_per_vector
        );
    }

    println!();

    // Memory efficiency with different vector dimensions
    println!("💾 Memory usage by dimension:");
    let dimensions = vec![32, 64, 128, 256, 512];
    let vector_count = 1000;

    println!("   Dims |   Memory   | Per Vector | Per Dim");
    println!("   -----|------------|------------|--------");

    for &dim in &dimensions {
        let db = Helix::new()?;
        let vectors = generate_test_vectors(vector_count, dim);

        db.insert_batch(vectors)?;

        let stats = db.stats();
        let memory_per_vector = stats.memory.current_allocated / vector_count;
        let memory_per_dim = memory_per_vector / dim;

        println!(
            "   {:>4} | {:>9} B | {:>8} B | {:>6} B",
            dim, stats.memory.current_allocated, memory_per_vector, memory_per_dim
        );
    }

    Ok(())
}

fn benchmark_concurrent_access() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 Testing concurrent read performance...");

    // Setup database
    let db = Arc::new(Helix::new()?);
    let vectors = generate_test_vectors(MEDIUM_DATASET, DIMENSIONS);
    db.insert_batch(vectors)?;

    let thread_counts = vec![1, 2, 4, 8];

    for &thread_count in &thread_counts {
        let queries_per_thread = 50;
        let total_queries = thread_count * queries_per_thread;

        let start_time = Instant::now();

        let handles: Vec<_> = (0..thread_count)
            .map(|_| {
                let db_clone = Arc::clone(&db);
                thread::spawn(move || {
                    for _ in 0..queries_per_thread {
                        let query = generate_random_vector(DIMENSIONS);
                        let _ = db_clone.search(&query, 10);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        let duration = start_time.elapsed();
        let qps = total_queries as f64 / duration.as_secs_f64();

        println!(
            "   {} threads: {:>6} QPS ({:>6?} total)",
            thread_count, qps as u64, duration
        );
    }

    println!();

    Ok(())
}

fn benchmark_batch_operations() -> Result<(), Box<dyn std::error::Error>> {
    let batch_sizes = vec![1, 10, 100, 1000];
    let total_vectors = 5000;

    println!("📦 Batch operation performance:");

    for &batch_size in &batch_sizes {
        let db = Helix::new()?;
        let batches = total_vectors / batch_size;

        let start_time = Instant::now();

        for batch_idx in 0..batches {
            let batch_vectors =
                generate_test_vectors_with_offset(batch_size, DIMENSIONS, batch_idx * batch_size);
            db.insert_batch(batch_vectors)?;
        }

        let duration = start_time.elapsed();
        let rate = total_vectors as f64 / duration.as_secs_f64();

        println!(
            "   Batch size {:>4}: {:>8?} ({:>8.0} vectors/sec)",
            batch_size, duration, rate
        );
    }

    // Batch search performance
    println!("\n📦 Batch search performance:");
    let db = Helix::new()?;
    let vectors = generate_test_vectors(MEDIUM_DATASET, DIMENSIONS);
    db.insert_batch(vectors)?;

    let query_batch_sizes = vec![1, 5, 10, 25, 50];

    for &query_count in &query_batch_sizes {
        let queries: Vec<Vec<f32>> = (0..query_count)
            .map(|_| generate_random_vector(DIMENSIONS))
            .collect();

        // Individual searches
        let start_time = Instant::now();
        for query in &queries {
            let _ = db.search(query, 10)?;
        }
        let individual_duration = start_time.elapsed();

        // Batch search
        let start_time = Instant::now();
        let _ = db.batch_search(&queries, 10)?;
        let batch_duration = start_time.elapsed();

        let speedup = individual_duration.as_secs_f64() / batch_duration.as_secs_f64();

        println!(
            "   {} queries: Individual {:>6?} | Batch {:>6?} | Speedup {:.2}x",
            query_count, individual_duration, batch_duration, speedup
        );
    }

    Ok(())
}

fn benchmark_distance_metrics() -> Result<(), Box<dyn std::error::Error>> {
    let db = Helix::new()?;
    let vectors = generate_test_vectors(MEDIUM_DATASET, DIMENSIONS);
    db.insert_batch(vectors)?;

    let query = generate_random_vector(DIMENSIONS);
    let metrics = vec!["euclidean", "cosine", "dot_product", "manhattan"];
    let iterations = 100;

    println!(
        "📐 Distance metric performance ({} searches each):",
        iterations
    );

    for metric in metrics {
        let start_time = Instant::now();

        for _ in 0..iterations {
            match db.search_with_metric(&query, 10, metric) {
                Ok(_) => {}
                Err(_) => continue, // Skip unsupported metrics
            }
        }

        let duration = start_time.elapsed();
        let avg_time = duration / iterations;

        println!(
            "   {:>12}: {:>6?} avg ({:>6?} total)",
            metric, avg_time, duration
        );
    }

    Ok(())
}

// Helper functions

fn generate_test_vectors(
    count: usize,
    dimensions: usize,
) -> Vec<(String, Vec<f32>, HashMap<String, String>)> {
    generate_test_vectors_with_offset(count, dimensions, 0)
}

fn generate_test_vectors_with_offset(
    count: usize,
    dimensions: usize,
    offset: usize,
) -> Vec<(String, Vec<f32>, HashMap<String, String>)> {
    (0..count)
        .map(|i| {
            let id = format!("vec_{:06}", offset + i);
            let vector = generate_random_vector(dimensions);
            let metadata = HashMap::from([
                ("batch".to_string(), (i / 1000).to_string()),
                ("index".to_string(), (offset + i).to_string()),
            ]);
            (id, vector, metadata)
        })
        .collect()
}

fn generate_random_vector(dimensions: usize) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    std::thread::current().id().hash(&mut hasher);
    let seed = hasher.finish();

    // Simple deterministic random number generator
    let mut rng_state = seed;
    (0..dimensions)
        .map(|i| {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let normalized = (rng_state ^ (i as u64)) as f32 / u64::MAX as f32;
            normalized * 2.0 - 1.0 // Range [-1, 1]
        })
        .collect()
}

fn create_brute_force_config() -> VecLiteConfig {
    let mut config = VecLiteConfig::default();
    config.index.index_type = IndexType::BruteForce;
    config
}

fn create_hnsw_config(max_m: usize, ef_construction: usize, ef_search: usize) -> VecLiteConfig {
    let mut config = VecLiteConfig::default();
    config.index.index_type = IndexType::HNSW;
    config.index.hnsw.max_m = max_m;
    config.index.hnsw.ef_construction = ef_construction;
    config.index.hnsw.ef_search = ef_search;
    config
}

fn print_performance_recommendations() {
    println!("\n💡 Performance Recommendations");
    println!("==============================");

    println!("\n🚀 For High Throughput:");
    println!("   • Use batch operations for insertions");
    println!("   • Configure HNSW with higher ef_construction for better quality");
    println!("   • Use appropriate vector dimensions (128-512 often optimal)");
    println!("   • Enable SIMD features when available");

    println!("\n🎯 For Low Latency:");
    println!("   • Use smaller ef_search values");
    println!("   • Consider brute force for small datasets (<1000 vectors)");
    println!("   • Pre-warm the database with a few queries");
    println!("   • Use concurrent searches for parallel workloads");

    println!("\n💾 For Memory Efficiency:");
    println!("   • Use vector quantization for large datasets");
    println!("   • Enable compression for persistence");
    println!("   • Monitor memory usage with built-in statistics");
    println!("   • Consider smaller vector dimensions if accuracy permits");

    println!("\n⚙️  Configuration Guidelines:");
    println!("   • Small datasets (<10K): BruteForce or HNSW with M=8");
    println!("   • Medium datasets (10K-100K): HNSW with M=16, ef_construction=200");
    println!("   • Large datasets (>100K): HNSW with M=32, ef_construction=400");
    println!("   • Real-time applications: Lower ef_search (50-100)");
    println!("   • Batch processing: Higher ef_search (200-400)");
}
