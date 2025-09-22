// Cross-platform compatibility tests for VecLite
// Ensures consistent behavior across different operating systems and architectures

use std::collections::HashMap;
use tempfile::tempdir;
use veclite::{VecLite, VecLiteConfig};

// Test configuration constants
const TEST_VECTORS: usize = 1000;
const TEST_DIMENSIONS: usize = 128;

fn create_test_data() -> Vec<(String, Vec<f32>, HashMap<String, String>)> {
    (0..TEST_VECTORS)
        .map(|i| {
            let vector = (0..TEST_DIMENSIONS)
                .map(|j| (i * TEST_DIMENSIONS + j) as f32 * 0.001)
                .collect();
            let metadata = HashMap::from([
                ("id".to_string(), i.to_string()),
                ("platform".to_string(), std::env::consts::OS.to_string()),
                ("arch".to_string(), std::env::consts::ARCH.to_string()),
            ]);
            (format!("vector_{}", i), vector, metadata)
        })
        .collect()
}

#[test]
fn test_basic_operations_cross_platform() {
    let db = VecLite::new().unwrap();
    let test_data = create_test_data();

    // Test batch insertion
    db.insert_batch(test_data.clone()).unwrap();
    assert_eq!(db.len(), TEST_VECTORS);

    // Test search functionality
    let query = vec![0.5; TEST_DIMENSIONS];
    let results = db.search(&query, 10).unwrap();
    assert_eq!(results.len(), 10);

    // Test individual operations
    let (id, vector, metadata) = &test_data[0];
    let retrieved = db.get(id).unwrap().unwrap();
    assert_eq!(retrieved.vector, *vector);
    assert_eq!(retrieved.metadata, *metadata);

    // Test deletion
    assert!(db.delete(id).unwrap());
    assert_eq!(db.len(), TEST_VECTORS - 1);
}

#[test]
fn test_file_persistence_cross_platform() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_db.vlt");

    // Create and populate database
    {
        let db = VecLite::new().unwrap();
        let test_data = create_test_data();
        db.insert_batch(test_data).unwrap();

        // Save to file
        db.save(&db_path).unwrap();
    }

    // Load from file
    let loaded_db = VecLite::load(&db_path).unwrap();
    assert_eq!(loaded_db.len(), TEST_VECTORS);

    // Verify data integrity
    let query = vec![0.5; TEST_DIMENSIONS];
    let results = loaded_db.search(&query, 5).unwrap();
    assert_eq!(results.len(), 5);

    // Check that the file exists and has reasonable size
    assert!(db_path.exists());
    let file_size = std::fs::metadata(&db_path).unwrap().len();
    assert!(file_size > 1000); // Should be at least 1KB for 1000 vectors
}

#[test]
fn test_path_handling_cross_platform() {
    let temp_dir = tempdir().unwrap();

    // Test various path formats
    let test_paths = vec![
        temp_dir.path().join("simple.vlt"),
        temp_dir.path().join("with spaces.vlt"),
        temp_dir.path().join("with-dashes.vlt"),
        temp_dir.path().join("with_underscores.vlt"),
        temp_dir.path().join("UPPERCASE.vlt"),
        temp_dir.path().join("subdirectory").join("nested.vlt"),
    ];

    for path in test_paths {
        // Create directory if needed
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }

        let db = VecLite::new().unwrap();
        let test_vector = vec![1.0, 2.0, 3.0];
        let metadata = HashMap::from([("test".to_string(), "path".to_string())]);

        db.insert(
            "test_vector".to_string(),
            test_vector.clone(),
            metadata.clone(),
        )
        .unwrap();

        // Save and load
        db.save(&path).unwrap();
        let loaded_db = VecLite::load(&path).unwrap();

        // Verify
        let retrieved = loaded_db.get(&"test_vector".to_string()).unwrap().unwrap();
        assert_eq!(retrieved.vector, test_vector);
        assert_eq!(retrieved.metadata, metadata);
    }
}

#[test]
fn test_memory_behavior_cross_platform() {
    let mut config = VecLiteConfig::default();

    // Set platform-appropriate memory limits
    let memory_limit = if cfg!(target_pointer_width = "32") {
        512 * 1024 * 1024 // 512MB for 32-bit systems
    } else {
        2 * 1024 * 1024 * 1024 // 2GB for 64-bit systems
    };

    config.memory.max_memory_bytes = memory_limit;
    config.memory.enable_monitoring = true;

    let db = VecLite::with_config(config).unwrap();

    // Insert data gradually and monitor memory
    let batch_size = 100;
    let batches = TEST_VECTORS / batch_size;

    for batch_idx in 0..batches {
        let mut batch = Vec::new();
        for i in 0..batch_size {
            let vector_id = format!("mem_test_{}_{}", batch_idx, i);
            let vector = vec![batch_idx as f32; TEST_DIMENSIONS];
            let metadata = HashMap::from([("batch".to_string(), batch_idx.to_string())]);
            batch.push((vector_id, vector, metadata));
        }

        db.insert_batch(batch).unwrap();

        // Check memory usage
        let stats = db.stats();
        assert!(stats.memory.current_allocated <= memory_limit);
        assert!(db.is_memory_within_limits());
    }
}

#[test]
fn test_floating_point_precision_cross_platform() {
    let db = VecLite::new().unwrap();

    // Test various floating point edge cases
    let test_cases = vec![
        ("zero", vec![0.0; 10]),
        ("positive", vec![1.0, 2.0, 3.0, 4.0, 5.0]),
        ("negative", vec![-1.0, -2.0, -3.0, -4.0, -5.0]),
        ("mixed", vec![1.0, -1.0, 2.0, -2.0, 0.0]),
        ("small", vec![1e-6, 1e-7, 1e-8, 1e-9, 1e-10]),
        ("large", vec![1e6, 1e7, 1e8, 1e9, 1e10]),
        ("fractional", vec![0.1, 0.2, 0.3, 0.4, 0.5]),
        ("pi_e", vec![std::f32::consts::PI, std::f32::consts::E]),
    ];

    for (name, vector) in test_cases {
        let metadata = HashMap::from([("type".to_string(), name.to_string())]);
        db.insert(name.to_string(), vector.clone(), metadata.clone())
            .unwrap();

        // Retrieve and verify exact equality
        let retrieved = db.get(&name.to_string()).unwrap().unwrap();
        assert_eq!(retrieved.vector, vector, "Vector mismatch for {}", name);
        assert_eq!(retrieved.metadata, metadata);
    }

    // Test search with precise floating point queries
    for (name, vector) in [
        ("zero", vec![0.0; 10]),
        ("positive", vec![1.0, 2.0, 3.0, 4.0, 5.0]),
    ] {
        let results = db.search(&vector, 1).unwrap();
        assert_eq!(results[0].id, name);
        // Score should be exactly 0.0 for identical vectors in Euclidean distance
        if name == "zero" {
            assert_eq!(results[0].score, 0.0);
        }
    }
}

#[test]
fn test_unicode_support_cross_platform() {
    let db = VecLite::new().unwrap();

    // Test various Unicode strings in metadata and IDs
    let unicode_tests = vec![
        ("ascii", "Hello World"),
        ("chinese", "你好世界"),
        ("japanese", "こんにちは世界"),
        ("emoji", "🚀🌍🔬💡"),
        ("mixed", "Hello 世界 🌍"),
        ("accents", "café résumé naïve"),
        ("symbols", "α β γ δ ε ∞ ∑ ∫"),
        ("right_to_left", "مرحبا بالعالم"),
    ];

    for (test_name, unicode_text) in unicode_tests {
        let vector_id = format!("unicode_{}", test_name);
        let vector = vec![1.0, 2.0, 3.0];
        let metadata = HashMap::from([
            ("text".to_string(), unicode_text.to_string()),
            ("type".to_string(), test_name.to_string()),
        ]);

        // Insert with Unicode metadata
        db.insert(vector_id.clone(), vector.clone(), metadata.clone())
            .unwrap();

        // Retrieve and verify Unicode preservation
        let retrieved = db.get(&vector_id).unwrap().unwrap();
        assert_eq!(retrieved.metadata.get("text").unwrap(), unicode_text);
        assert_eq!(retrieved.vector, vector);
    }

    // Test search with Unicode filters
    let results = db
        .search_with_filter(&vec![1.0, 2.0, 3.0], 10, |metadata| {
            metadata
                .get("text")
                .map_or(false, |text| text.contains("世界"))
        })
        .unwrap();

    // Should find Chinese and mixed entries
    assert!(results.len() >= 2);
}

#[test]
fn test_concurrent_access_cross_platform() {
    use std::sync::Arc;
    use std::thread;

    let db = Arc::new(VecLite::new().unwrap());
    let thread_count = if cfg!(target_os = "windows") { 4 } else { 8 };

    // Pre-populate with some data
    let initial_data = (0..100)
        .map(|i| {
            let vector = vec![i as f32; 10];
            let metadata = HashMap::from([("initial".to_string(), i.to_string())]);
            (format!("initial_{}", i), vector, metadata)
        })
        .collect();
    db.insert_batch(initial_data).unwrap();

    let mut handles = Vec::new();

    // Spawn concurrent threads
    for thread_id in 0..thread_count {
        let db_clone = Arc::clone(&db);

        let handle = thread::spawn(move || {
            let operations_per_thread = 50;

            for i in 0..operations_per_thread {
                match i % 3 {
                    0 => {
                        // Insert operation
                        let vector_id = format!("thread_{}_{}", thread_id, i);
                        let vector = vec![thread_id as f32; 10];
                        let metadata = HashMap::from([
                            ("thread".to_string(), thread_id.to_string()),
                            ("op".to_string(), i.to_string()),
                        ]);

                        if let Err(e) = db_clone.insert(vector_id, vector, metadata) {
                            eprintln!("Insert failed in thread {}: {}", thread_id, e);
                        }
                    }
                    1 => {
                        // Search operation
                        let query = vec![thread_id as f32; 10];
                        if let Err(e) = db_clone.search(&query, 5) {
                            eprintln!("Search failed in thread {}: {}", thread_id, e);
                        }
                    }
                    2 => {
                        // Get operation
                        let vector_id = format!("initial_{}", i % 100);
                        if let Err(e) = db_clone.get(&vector_id) {
                            eprintln!("Get failed in thread {}: {}", thread_id, e);
                        }
                    }
                    _ => unreachable!(),
                }
            }
        });

        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }

    // Verify database consistency
    let final_count = db.len();
    assert!(final_count >= 100); // At least initial data should remain
    println!("Final vector count: {}", final_count);

    // Verify search still works
    let results = db.search(&vec![0.5; 10], 10).unwrap();
    assert!(!results.is_empty());
}

#[test]
fn test_large_file_handling_cross_platform() {
    let temp_dir = tempdir().unwrap();
    let large_db_path = temp_dir.path().join("large_test.vlt");

    // Create a larger dataset
    let large_test_size = 5000;
    let large_dimensions = 256;

    let db = VecLite::new().unwrap();

    // Insert data in chunks to avoid memory issues
    let chunk_size = 500;
    for chunk_start in (0..large_test_size).step_by(chunk_size) {
        let chunk_end = std::cmp::min(chunk_start + chunk_size, large_test_size);
        let chunk_data: Vec<_> = (chunk_start..chunk_end)
            .map(|i| {
                let vector = (0..large_dimensions)
                    .map(|j| (i * large_dimensions + j) as f32 * 0.0001)
                    .collect();
                let metadata = HashMap::from([
                    ("chunk".to_string(), (i / chunk_size).to_string()),
                    ("index".to_string(), i.to_string()),
                ]);
                (format!("large_vector_{}", i), vector, metadata)
            })
            .collect();

        db.insert_batch(chunk_data).unwrap();
    }

    assert_eq!(db.len(), large_test_size);

    // Save large database
    db.save(&large_db_path).unwrap();

    // Verify file size is reasonable
    let file_size = std::fs::metadata(&large_db_path).unwrap().len();
    let expected_min_size = (large_test_size * large_dimensions * 4) as u64; // 4 bytes per f32
    assert!(file_size >= expected_min_size / 2); // Allow for compression

    // Load and verify
    let loaded_db = VecLite::load(&large_db_path).unwrap();
    assert_eq!(loaded_db.len(), large_test_size);

    // Test search on large database
    let query = vec![0.1; large_dimensions];
    let results = loaded_db.search(&query, 20).unwrap();
    assert_eq!(results.len(), 20);
}

#[test]
fn test_error_handling_cross_platform() {
    // Test various error conditions across platforms

    // Test invalid file paths
    let invalid_paths = if cfg!(windows) {
        vec![
            "C:\\invalid\\path\\test.vlt",
            "\\\\invalid\\unc\\path.vlt",
            "CON.vlt", // Reserved name on Windows
        ]
    } else {
        vec![
            "/root/invalid/path/test.vlt", // Likely no permission
            "/dev/null/test.vlt",          // Invalid parent
        ]
    };

    for invalid_path in invalid_paths {
        let db = VecLite::new().unwrap();
        let result = db.save(invalid_path);
        assert!(
            result.is_err(),
            "Should fail to save to invalid path: {}",
            invalid_path
        );
    }

    // Test loading non-existent files
    let result = VecLite::load("non_existent_file.vlt");
    assert!(result.is_err());

    // Test invalid vector dimensions
    let db = VecLite::new().unwrap();
    db.insert("test1".to_string(), vec![1.0, 2.0], HashMap::new())
        .unwrap();

    let result = db.insert("test2".to_string(), vec![1.0, 2.0, 3.0], HashMap::new());
    assert!(result.is_err()); // Should fail due to dimension mismatch
}

#[test]
fn test_performance_consistency_cross_platform() {
    use std::time::Instant;

    let db = VecLite::new().unwrap();
    let test_data = create_test_data();

    // Measure insertion performance
    let insert_start = Instant::now();
    db.insert_batch(test_data).unwrap();
    let insert_duration = insert_start.elapsed();

    // Measure search performance
    let query = vec![0.5; TEST_DIMENSIONS];
    let search_start = Instant::now();

    let search_iterations = 100;
    for _ in 0..search_iterations {
        let _results = db.search(&query, 10).unwrap();
    }
    let search_duration = search_start.elapsed();

    // Performance should be reasonable on all platforms
    let insert_rate = TEST_VECTORS as f64 / insert_duration.as_secs_f64();
    let search_rate = search_iterations as f64 / search_duration.as_secs_f64();

    println!(
        "Platform: {} ({})",
        std::env::consts::OS,
        std::env::consts::ARCH
    );
    println!("Insert rate: {:.0} vectors/sec", insert_rate);
    println!("Search rate: {:.0} searches/sec", search_rate);

    // Basic performance thresholds (adjust based on platform capabilities)
    let min_insert_rate = if cfg!(target_arch = "aarch64") {
        500.0
    } else {
        1000.0
    };
    let min_search_rate = if cfg!(target_arch = "aarch64") {
        50.0
    } else {
        100.0
    };

    assert!(
        insert_rate > min_insert_rate,
        "Insert performance too slow: {} < {}",
        insert_rate,
        min_insert_rate
    );
    assert!(
        search_rate > min_search_rate,
        "Search performance too slow: {} < {}",
        search_rate,
        min_search_rate
    );
}

#[test]
fn test_configuration_platform_specific() {
    let mut config = VecLiteConfig::default();

    // Adjust configuration based on platform capabilities
    if cfg!(target_pointer_width = "32") {
        config.memory.max_memory_bytes = 512 * 1024 * 1024; // 512MB for 32-bit
        config.storage.max_vectors = 100_000;
    } else {
        config.memory.max_memory_bytes = 2 * 1024 * 1024 * 1024; // 2GB for 64-bit
        config.storage.max_vectors = 1_000_000;
    }

    // Platform-specific thread counts
    if cfg!(target_os = "windows") {
        // Windows might have different threading characteristics
        config.memory.pool_max_pools = 1000;
    } else {
        config.memory.pool_max_pools = 2000;
    }

    let db = VecLite::with_config(config).unwrap();

    // Test that the configuration is properly applied
    let db_config = db.config();
    if cfg!(target_pointer_width = "32") {
        assert_eq!(db_config.memory.max_memory_bytes, 512 * 1024 * 1024);
    } else {
        assert_eq!(db_config.memory.max_memory_bytes, 2 * 1024 * 1024 * 1024);
    }

    // Test basic functionality with platform-specific config
    let test_vector = vec![1.0; 50];
    let metadata = HashMap::from([("platform".to_string(), std::env::consts::OS.to_string())]);

    db.insert(
        "platform_test".to_string(),
        test_vector.clone(),
        metadata.clone(),
    )
    .unwrap();

    let retrieved = db.get(&"platform_test".to_string()).unwrap().unwrap();
    assert_eq!(retrieved.vector, test_vector);
    assert_eq!(retrieved.metadata, metadata);
}
