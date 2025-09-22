// Stress tests for VecLite performance validation
// Tests memory monitoring, batch operations, and production readiness

use helix::{Helix, VecLiteConfig};
use std::collections::HashMap;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

// Test configuration constants
const STRESS_TEST_VECTORS: usize = 50_000;
const STRESS_TEST_DIMENSIONS: usize = 384;
const STRESS_TEST_THREADS: usize = 8;
const MEMORY_LIMIT_MB: usize = 512; // 512MB limit for stress tests

#[test]
fn test_high_volume_insertions() {
    let mut config = VecLiteConfig::default();
    config.memory.max_memory_bytes = MEMORY_LIMIT_MB * 1024 * 1024;
    config.memory.enable_monitoring = true;
    config.memory.warning_threshold_percent = 80;

    let db = Helix::with_config(config).unwrap();

    println!(
        "Starting high-volume insertion test with {} vectors",
        STRESS_TEST_VECTORS
    );
    let start = Instant::now();

    // Insert vectors in batches
    let batch_size = 1000;
    let mut total_inserted = 0;

    for batch_idx in 0..(STRESS_TEST_VECTORS / batch_size) {
        let mut batch = Vec::new();

        for i in 0..batch_size {
            let vector_id = format!("stress_vector_{}_{}", batch_idx, i);
            let vector = (0..STRESS_TEST_DIMENSIONS)
                .map(|j| (batch_idx * batch_size + i + j) as f32 * 0.01)
                .collect::<Vec<f32>>();
            let metadata = HashMap::from([
                ("batch".to_string(), batch_idx.to_string()),
                ("index".to_string(), i.to_string()),
                ("test".to_string(), "stress".to_string()),
            ]);

            batch.push((vector_id, vector, metadata));
        }

        db.insert_batch(batch).unwrap();
        total_inserted += batch_size;

        // Check memory usage periodically
        if batch_idx % 10 == 0 {
            let stats = db.stats();
            let memory_usage_mb = stats.memory.current_allocated / (1024 * 1024);
            println!(
                "Batch {}: {} vectors inserted, {} MB memory used",
                batch_idx, total_inserted, memory_usage_mb
            );

            // Ensure memory is within limits
            assert!(
                db.is_memory_within_limits(),
                "Memory usage exceeded limits: {} MB",
                memory_usage_mb
            );
        }
    }

    let duration = start.elapsed();
    let final_stats = db.stats();

    println!("High-volume insertion completed:");
    println!("  Vectors inserted: {}", total_inserted);
    println!("  Time taken: {:?}", duration);
    println!(
        "  Insertion rate: {:.0} vectors/sec",
        total_inserted as f64 / duration.as_secs_f64()
    );
    println!(
        "  Memory usage: {} MB",
        final_stats.memory.current_allocated / (1024 * 1024)
    );
    println!(
        "  Peak memory: {} MB",
        final_stats.memory.peak_allocated / (1024 * 1024)
    );

    // Validate final state
    assert_eq!(db.len(), total_inserted);
    assert!(
        duration.as_secs() < 60,
        "Insertion took too long: {:?}",
        duration
    );
    assert!(db.is_memory_within_limits());
}

#[test]
fn test_concurrent_operations() {
    let config = VecLiteConfig::default();
    let db = Arc::new(Helix::with_config(config).unwrap());

    println!(
        "Starting concurrent operations test with {} threads",
        STRESS_TEST_THREADS
    );

    // Pre-populate with some data
    let mut initial_batch = Vec::new();
    for i in 0..1000 {
        let vector_id = format!("initial_{}", i);
        let vector = (0..STRESS_TEST_DIMENSIONS)
            .map(|j| (i + j) as f32 * 0.01)
            .collect();
        let metadata = HashMap::from([("type".to_string(), "initial".to_string())]);
        initial_batch.push((vector_id, vector, metadata));
    }
    db.insert_batch(initial_batch).unwrap();

    let barrier = Arc::new(Barrier::new(STRESS_TEST_THREADS));
    let mut handles = Vec::new();

    let start = Instant::now();

    // Spawn concurrent threads
    for thread_id in 0..STRESS_TEST_THREADS {
        let db_clone = Arc::clone(&db);
        let barrier_clone = Arc::clone(&barrier);

        let handle = thread::spawn(move || {
            barrier_clone.wait(); // Synchronize start

            let operations_per_thread = 1000;
            let mut insert_count = 0;
            let mut search_count = 0;
            let mut get_count = 0;

            for i in 0..operations_per_thread {
                match i % 4 {
                    0 | 1 => {
                        // Insert operation (50% of operations)
                        let vector_id = format!("thread_{}_{}", thread_id, i);
                        let vector = (0..STRESS_TEST_DIMENSIONS)
                            .map(|j| (thread_id * 1000 + i + j) as f32 * 0.01)
                            .collect();
                        let metadata = HashMap::from([
                            ("thread".to_string(), thread_id.to_string()),
                            ("operation".to_string(), i.to_string()),
                        ]);

                        if db_clone.insert(vector_id, vector, metadata).is_ok() {
                            insert_count += 1;
                        }
                    }
                    2 => {
                        // Search operation (25% of operations)
                        let query = (0..STRESS_TEST_DIMENSIONS)
                            .map(|j| ((thread_id + i + j) as f32 * 0.01) % 1.0)
                            .collect();

                        if db_clone.search(&query, 10).is_ok() {
                            search_count += 1;
                        }
                    }
                    3 => {
                        // Get operation (25% of operations)
                        let vector_id = format!("initial_{}", i % 1000);
                        if db_clone.get(&vector_id).is_ok() {
                            get_count += 1;
                        }
                    }
                    _ => unreachable!(),
                }
            }

            (insert_count, search_count, get_count)
        });

        handles.push(handle);
    }

    // Wait for all threads to complete
    let mut total_inserts = 0;
    let mut total_searches = 0;
    let mut total_gets = 0;

    for handle in handles {
        let (inserts, searches, gets) = handle.join().unwrap();
        total_inserts += inserts;
        total_searches += searches;
        total_gets += gets;
    }

    let duration = start.elapsed();
    let stats = db.stats();

    println!("Concurrent operations completed:");
    println!("  Total inserts: {}", total_inserts);
    println!("  Total searches: {}", total_searches);
    println!("  Total gets: {}", total_gets);
    println!("  Time taken: {:?}", duration);
    println!(
        "  Operations/sec: {:.0}",
        (total_inserts + total_searches + total_gets) as f64 / duration.as_secs_f64()
    );
    println!(
        "  Memory usage: {} MB",
        stats.memory.current_allocated / (1024 * 1024)
    );

    // Validate database consistency
    assert!(db.len() >= 1000); // At least initial data
    assert!(
        duration.as_secs() < 30,
        "Concurrent operations took too long"
    );
    assert!(db.is_memory_within_limits());
}

#[test]
fn test_memory_pressure_and_cleanup() {
    let mut config = VecLiteConfig::default();
    config.memory.max_memory_bytes = 64 * 1024 * 1024; // 64MB limit
    config.memory.enable_monitoring = true;
    config.memory.enable_auto_cleanup = true;
    config.memory.warning_threshold_percent = 70;
    config.memory.cleanup_threshold_percent = 85;
    config.memory.cleanup_interval_seconds = 1;

    let max_memory_limit = config.memory.max_memory_bytes;
    let db = Helix::with_config(config).unwrap();

    println!("Starting memory pressure test with 64MB limit");

    let mut batch_count = 0;
    let batch_size = 500;
    let vector_size = 1024; // Larger vectors to consume more memory

    loop {
        let mut batch = Vec::new();

        for i in 0..batch_size {
            let vector_id = format!("pressure_{}_{}", batch_count, i);
            let vector = (0..vector_size).map(|j| (i + j) as f32 * 0.001).collect();
            let metadata = HashMap::from([
                ("batch".to_string(), batch_count.to_string()),
                ("pressure_test".to_string(), "true".to_string()),
            ]);

            batch.push((vector_id, vector, metadata));
        }

        if db.insert_batch(batch).is_err() {
            println!(
                "Insert failed at batch {} due to memory pressure",
                batch_count
            );
            break;
        }

        batch_count += 1;

        let stats = db.stats();
        let memory_mb = stats.memory.current_allocated / (1024 * 1024);

        if batch_count % 5 == 0 {
            println!("Batch {}: {} MB memory used", batch_count, memory_mb);
        }

        // Check if we're approaching limits
        if memory_mb > 50 {
            // Getting close to 64MB limit
            println!(
                "Approaching memory limit at batch {}: {} MB",
                batch_count, memory_mb
            );

            // Verify memory monitoring is working
            if !stats.memory.warnings.is_empty() {
                println!("Memory warnings detected: {:?}", stats.memory.warnings);
            }

            // Give some time for automatic cleanup to trigger
            thread::sleep(Duration::from_millis(100));
        }

        if batch_count > 100 {
            // Safety limit
            break;
        }
    }

    let final_stats = db.stats();
    println!("Memory pressure test completed:");
    println!("  Batches processed: {}", batch_count);
    println!("  Final vectors: {}", db.len());
    println!(
        "  Final memory: {} MB",
        final_stats.memory.current_allocated / (1024 * 1024)
    );
    println!(
        "  Peak memory: {} MB",
        final_stats.memory.peak_allocated / (1024 * 1024)
    );
    println!(
        "  Total allocations: {}",
        final_stats.memory.total_allocations
    );
    println!(
        "  Total deallocations: {}",
        final_stats.memory.total_deallocations
    );

    // Validate memory management worked
    assert!(batch_count > 10, "Should have processed multiple batches");
    assert!(
        final_stats.memory.peak_allocated <= max_memory_limit,
        "Peak memory exceeded configured limit"
    );
}

#[test]
fn test_large_batch_operations() {
    let config = VecLiteConfig::default();
    let db = Helix::with_config(config).unwrap();

    println!("Starting large batch operations test");

    // Test very large batch insert
    let large_batch_size = 10_000;
    let start = Instant::now();

    let mut large_batch = Vec::new();
    for i in 0..large_batch_size {
        let vector_id = format!("large_batch_{}", i);
        let vector = (0..STRESS_TEST_DIMENSIONS)
            .map(|j| (i + j) as f32 * 0.001)
            .collect();
        let metadata = HashMap::from([
            ("batch_type".to_string(), "large".to_string()),
            ("index".to_string(), i.to_string()),
        ]);

        large_batch.push((vector_id, vector, metadata));
    }

    db.insert_batch(large_batch).unwrap();
    let insert_duration = start.elapsed();

    println!("Large batch insert completed in {:?}", insert_duration);
    assert_eq!(db.len(), large_batch_size);

    // Test batch search operations
    let search_start = Instant::now();
    let mut total_results = 0;

    for i in 0..100 {
        let query = (0..STRESS_TEST_DIMENSIONS)
            .map(|j| (i + j) as f32 * 0.001)
            .collect();

        let results = db.search(&query, 50).unwrap();
        total_results += results.len();
    }

    let search_duration = search_start.elapsed();

    println!("Batch search completed:");
    println!("  100 searches performed");
    println!("  Total results: {}", total_results);
    println!("  Search duration: {:?}", search_duration);
    println!("  Average search time: {:?}", search_duration / 100);

    let stats = db.stats();
    println!("Final statistics:");
    println!(
        "  Memory usage: {} MB",
        stats.memory.current_allocated / (1024 * 1024)
    );
    println!(
        "  Average vector size: {:.2} bytes",
        stats.average_vector_size
    );

    // Performance assertions
    assert!(
        insert_duration.as_secs() < 10,
        "Large batch insert too slow"
    );
    assert!(
        search_duration.as_millis() < 5000,
        "Batch searches too slow"
    );
    assert!(total_results > 0, "No search results found");
}

#[test]
fn test_memory_configuration_updates() {
    let mut config = VecLiteConfig::default();
    config.memory.max_memory_bytes = 128 * 1024 * 1024; // 128MB
    config.memory.warning_threshold_percent = 75;

    let db = Helix::with_config(config.clone()).unwrap();

    // Verify initial configuration
    let initial_config = db.memory_monitor().get_config();
    assert_eq!(initial_config.max_memory_bytes, 128 * 1024 * 1024);
    assert_eq!(initial_config.warning_threshold_percent, 75);

    // Insert some data
    let mut batch = Vec::new();
    for i in 0..1000 {
        let vector_id = format!("config_test_{}", i);
        let vector = (0..256).map(|j| (i + j) as f32 * 0.01).collect();
        let metadata = HashMap::from([("test".to_string(), "config".to_string())]);
        batch.push((vector_id, vector, metadata));
    }
    db.insert_batch(batch).unwrap();

    // Update memory configuration
    let mut new_memory_config = config.memory.clone();
    new_memory_config.max_memory_bytes = 256 * 1024 * 1024; // 256MB
    new_memory_config.warning_threshold_percent = 80;
    new_memory_config.cleanup_threshold_percent = 95;

    db.update_memory_config(new_memory_config.clone());

    // Verify configuration update
    let updated_config = db.memory_monitor().get_config();
    assert_eq!(updated_config.max_memory_bytes, 256 * 1024 * 1024);
    assert_eq!(updated_config.warning_threshold_percent, 80);
    assert_eq!(updated_config.cleanup_threshold_percent, 95);

    // Verify database still functions correctly
    let query = (0..256).map(|i| i as f32 * 0.01).collect();
    let results = db.search(&query, 10).unwrap();
    assert!(!results.is_empty());

    println!("Memory configuration update test completed successfully");
}

#[test]
fn test_performance_regression() {
    let config = VecLiteConfig::default();
    let db = Helix::with_config(config).unwrap();

    // Baseline performance test
    let test_vectors = 5_000;
    let dimensions = 384;

    // Insert performance test
    let insert_start = Instant::now();
    let mut batch = Vec::new();

    for i in 0..test_vectors {
        let vector_id = format!("perf_test_{}", i);
        let vector = (0..dimensions).map(|j| (i + j) as f32 * 0.001).collect();
        let metadata = HashMap::from([("test".to_string(), "performance".to_string())]);
        batch.push((vector_id, vector, metadata));
    }

    db.insert_batch(batch).unwrap();
    let insert_duration = insert_start.elapsed();
    let insert_rate = test_vectors as f64 / insert_duration.as_secs_f64();

    // Search performance test
    let search_start = Instant::now();
    let search_iterations = 100;

    for i in 0..search_iterations {
        let query = (0..dimensions).map(|j| (i + j) as f32 * 0.001).collect();
        db.search(&query, 10).unwrap();
    }

    let search_duration = search_start.elapsed();
    let search_rate = search_iterations as f64 / search_duration.as_secs_f64();

    println!("Performance regression test results:");
    println!("  Insert rate: {:.0} vectors/sec", insert_rate);
    println!("  Search rate: {:.0} searches/sec", search_rate);
    println!(
        "  Average search time: {:.2}ms",
        search_duration.as_millis() as f64 / search_iterations as f64
    );

    // Performance regression assertions (these are baseline minimums)
    assert!(
        insert_rate > 1000.0,
        "Insert performance regression: {} < 1000 vec/sec",
        insert_rate
    );
    assert!(
        search_rate > 100.0,
        "Search performance regression: {} < 100 searches/sec",
        search_rate
    );
    assert!(
        search_duration.as_millis() / search_iterations < 50,
        "Individual search too slow: > 50ms average"
    );

    let final_stats = db.stats();
    println!(
        "  Memory efficiency: {:.2} bytes/vector",
        final_stats.memory.current_allocated as f64 / test_vectors as f64
    );
}
