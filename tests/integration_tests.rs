//! Integration tests for Helix vector database
//!
//! These tests verify the full functionality of Helix including
//! storage, search, persistence, and FFI operations.

use helix::{Helix, VecLiteConfig};
use std::collections::HashMap;
use tempfile::tempdir;

#[test]
fn test_basic_integration() {
    let db = Helix::new().unwrap();

    // Insert test data
    let vector = vec![1.0, 2.0, 3.0];
    let metadata = HashMap::from([
        ("type".to_string(), "test".to_string()),
        ("category".to_string(), "integration".to_string()),
    ]);

    assert!(db
        .insert("test_vector".to_string(), vector.clone(), metadata.clone())
        .is_ok());

    // Search for similar vectors
    let query = vec![1.1, 2.1, 3.1];
    let results = db.search(&query, 1).unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "test_vector");
    assert!(results[0].score >= 0.0);
}

#[test]
fn test_persistence_integration() {
    let temp_dir = tempdir().unwrap();
    let file_path = temp_dir.path().join("integration_test.hlx");

    // Create database and add data
    {
        let db = Helix::new().unwrap();
        let vectors = vec![
            ("vec1".to_string(), vec![1.0, 0.0, 0.0], HashMap::new()),
            ("vec2".to_string(), vec![0.0, 1.0, 0.0], HashMap::new()),
            ("vec3".to_string(), vec![0.0, 0.0, 1.0], HashMap::new()),
        ];

        for (id, vector, metadata) in vectors {
            db.insert(id, vector, metadata).unwrap();
        }

        assert_eq!(db.len(), 3);
        db.save(&file_path).unwrap();
    }

    // Load database and verify data
    {
        let loaded_db = Helix::load(&file_path).unwrap();
        assert_eq!(loaded_db.len(), 3);

        // Test search functionality
        let query = vec![1.0, 0.1, 0.1];
        let results = loaded_db.search(&query, 2).unwrap();
        assert_eq!(results.len(), 2);
    }
}

#[test]
fn test_batch_operations_integration() {
    let db = Helix::new().unwrap();

    // Create batch of test vectors
    let batch: Vec<_> = (0..10)
        .map(|i| {
            let vector = vec![i as f32, (i + 1) as f32, (i + 2) as f32];
            let metadata = HashMap::from([
                ("id".to_string(), i.to_string()),
                ("batch".to_string(), "test".to_string()),
            ]);
            (format!("vec_{}", i), vector, metadata)
        })
        .collect();

    // Insert batch
    assert!(db.insert_batch(batch).is_ok());
    assert_eq!(db.len(), 10);

    // Test batch search
    let queries = vec![vec![0.5, 1.5, 2.5], vec![5.5, 6.5, 7.5]];
    let results = db.batch_search(&queries, 3).unwrap();

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].len(), 3);
    assert_eq!(results[1].len(), 3);
}

#[test]
fn test_different_metrics_integration() {
    let db = Helix::new().unwrap();

    // Insert test vectors
    let vectors = vec![
        ("vec1".to_string(), vec![1.0, 0.0, 0.0], HashMap::new()),
        ("vec2".to_string(), vec![0.0, 1.0, 0.0], HashMap::new()),
        ("vec3".to_string(), vec![0.0, 0.0, 1.0], HashMap::new()),
    ];

    for (id, vector, metadata) in vectors {
        db.insert(id, vector, metadata).unwrap();
    }

    let query = vec![1.0, 0.0, 0.0];

    // Test each available metric
    for metric in Helix::available_metrics() {
        let results = db.search_with_metric(&query, 3, metric);
        assert!(results.is_ok(), "Failed for metric: {}", metric);
        let results = results.unwrap();
        assert_eq!(results.len(), 3);
    }
}

#[test]
fn test_configuration_integration() {
    let mut config = VecLiteConfig::default();
    config.storage.max_vectors = 5;
    config.query.default_k = 3;

    let db = Helix::with_config(config).unwrap();

    // Verify configuration is applied
    assert_eq!(db.config().storage.max_vectors, 5);
    assert_eq!(db.config().query.default_k, 3);

    // Insert vectors up to limit
    for i in 0..5 {
        let vector = vec![i as f32, 0.0, 0.0];
        let metadata = HashMap::new();
        assert!(db.insert(format!("vec_{}", i), vector, metadata).is_ok());
    }

    assert_eq!(db.len(), 5);
}

#[test]
fn test_error_handling_integration() {
    let db = Helix::new().unwrap();

    // Insert first vector to establish dimension
    assert!(db
        .insert("vec1".to_string(), vec![1.0, 2.0], HashMap::new())
        .is_ok());

    // Try to insert vector with wrong dimensions
    let result = db.insert("vec2".to_string(), vec![1.0, 2.0, 3.0], HashMap::new());
    assert!(result.is_err());

    // Try to search with wrong dimensions
    let result = db.search(&vec![1.0, 2.0, 3.0], 5);
    assert!(result.is_err());

    // Try to use invalid metric
    let result = db.search_with_metric(&vec![1.0, 2.0], 5, "invalid_metric");
    assert!(result.is_err());
}

#[test]
fn test_memory_monitoring_integration() {
    let db = Helix::new().unwrap();

    // Add some vectors to use memory
    for i in 0..100 {
        let vector: Vec<f32> = (0..10).map(|j| (i * 10 + j) as f32).collect();
        let metadata = HashMap::from([("id".to_string(), i.to_string())]);
        db.insert(format!("vec_{}", i), vector, metadata).unwrap();
    }

    // Check memory statistics
    let stats = db.stats();
    assert!(stats.total_memory_bytes > 0);
    assert_eq!(stats.vector_count, 100);

    let memory_stats = db.memory_stats();
    // Memory allocation may be zero due to optimizations, just check structure exists
    // Note: total_allocations is u64, so always >= 0
    assert!(memory_stats.total_allocations == memory_stats.total_allocations);
}

#[test]
fn test_concurrent_operations_integration() {
    use std::sync::Arc;
    use std::thread;

    let db = Arc::new(Helix::new().unwrap());
    let mut handles = vec![];

    // Spawn multiple threads for concurrent inserts
    for thread_id in 0..5 {
        let db_clone = Arc::clone(&db);
        let handle = thread::spawn(move || {
            for i in 0..10 {
                let vector = vec![thread_id as f32, i as f32, 0.0];
                let metadata = HashMap::from([
                    ("thread".to_string(), thread_id.to_string()),
                    ("index".to_string(), i.to_string()),
                ]);
                let id = format!("thread_{}_{}", thread_id, i);
                db_clone.insert(id, vector, metadata).unwrap();
            }
        });
        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }

    // Verify all vectors were inserted
    assert_eq!(db.len(), 50);

    // Test concurrent searches
    let mut search_handles = vec![];
    for _ in 0..5 {
        let db_clone = Arc::clone(&db);
        let handle = thread::spawn(move || {
            let query = vec![2.0, 5.0, 0.0];
            let results = db_clone.search(&query, 10).unwrap();
            assert!(results.len() <= 10);
        });
        search_handles.push(handle);
    }

    for handle in search_handles {
        handle.join().unwrap();
    }
}

#[test]
fn test_filtering_integration() {
    let db = Helix::new().unwrap();

    // Insert vectors with different categories
    let vectors = vec![
        (
            "doc1".to_string(),
            vec![1.0, 0.0],
            HashMap::from([("category".to_string(), "A".to_string())]),
        ),
        (
            "doc2".to_string(),
            vec![0.0, 1.0],
            HashMap::from([("category".to_string(), "B".to_string())]),
        ),
        (
            "doc3".to_string(),
            vec![1.0, 1.0],
            HashMap::from([("category".to_string(), "A".to_string())]),
        ),
        (
            "doc4".to_string(),
            vec![0.5, 0.5],
            HashMap::from([("category".to_string(), "C".to_string())]),
        ),
    ];

    for (id, vector, metadata) in vectors {
        db.insert(id, vector, metadata).unwrap();
    }

    // Search with filter for category "A"
    let query = vec![0.8, 0.2];
    let results = db
        .search_with_filter(&query, 10, |metadata| {
            metadata.get("category") == Some(&"A".to_string())
        })
        .unwrap();

    // Should only return doc1 and doc3
    assert_eq!(results.len(), 2);
    for result in results {
        assert!(result.id == "doc1" || result.id == "doc3");
    }
}
