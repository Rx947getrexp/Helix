//! # Helix - Lightweight Embeddable Vector Search Library
//!
//! Helix is a lightweight, embeddable vector search library inspired by SQLite's
//! philosophy of being embedded, zero-configuration, and dependency-free. It provides
//! complete vector search functionality in a single library.
//!
//! ## Features
//!
//! - **Embedded-first**: Direct integration into applications without external services
//! - **Complete functionality**: Full CRUD operations, indexing, and persistence
//! - **Multiple distance metrics**: Euclidean, Cosine, Dot Product, Manhattan
//! - **Thread-safe**: Concurrent read/write operations with RwLock
//! - **Memory efficient**: Configurable memory limits and statistics
//! - **High performance**: Optimized distance calculations and batch operations
//!
//! ## Quick Start
//!
//! ```rust
//! use helix::{Helix, VectorData, Metadata};
//! use std::collections::HashMap;
//!
//! // Create a new Helix instance
//! let mut db = Helix::new().unwrap();
//!
//! // Insert a vector with metadata
//! let vector: VectorData = vec![1.0, 2.0, 3.0];
//! let metadata: Metadata = HashMap::from([
//!     ("type".to_string(), "document".to_string()),
//!     ("source".to_string(), "example".to_string()),
//! ]);
//!
//! db.insert("doc1".to_string(), vector, metadata).unwrap();
//!
//! // Search for similar vectors
//! let query = vec![1.1, 2.1, 3.1];
//! let results = db.search(&query, 5).unwrap(); // Find top 5 similar vectors
//!
//! // Results contain IDs, scores, and metadata
//! for result in results {
//!     println!("ID: {}, Score: {:.4}", result.id, result.score);
//! }
//! ```

// Public modules
pub mod batch;
pub mod distance;
pub mod error;
pub mod hnsw;
pub mod memory;
pub mod persistence;
pub mod storage;
pub mod types;

// FFI module (only included when building as cdylib)
#[cfg(feature = "ffi")]
pub mod ffi;

// Internal modules
mod query;

// Re-export commonly used types and traits
pub use distance::{
    CosineDistance, DistanceMetric, DistanceMetricFactory, DotProductSimilarity, EuclideanDistance,
};
pub use error::{QueryError, StorageError, VecLiteError, VecLiteResult};
pub use hnsw::HNSWIndex;
pub use persistence::{file_format, VLTHeader, VLTPersistence};
pub use query::{BruteForceIndex, QueryEngine};
pub use storage::{StorageManager, VectorStorage};
pub use types::{
    Dimensions, HNSWConfig, IndexConfig, IndexType, Metadata, QueryConfig, Score, SearchResult,
    StorageConfig, VecLiteConfig, VectorData, VectorId, VectorItem,
};

use std::sync::Arc;
use tracing::{info, instrument};

/// Main Helix database interface
///
/// This is the primary entry point for all Helix operations. It provides
/// a simple API for vector storage, indexing, and search operations.
#[derive(Debug)]
pub struct Helix {
    storage: Arc<StorageManager>,
    query_engine: QueryEngine,
    config: VecLiteConfig,
    memory_monitor: Arc<memory::MemoryMonitor>,
}

impl Helix {
    /// Create a new Helix instance with default configuration
    #[instrument]
    pub fn new() -> VecLiteResult<Self> {
        Self::with_config(VecLiteConfig::default())
    }

    /// Create a new Helix instance with custom configuration
    #[instrument(skip(config))]
    pub fn with_config(config: VecLiteConfig) -> VecLiteResult<Self> {
        info!("Creating Helix instance with custom configuration");

        let memory_monitor = Arc::new(memory::MemoryMonitor::with_config(config.memory.clone()));
        let storage = Arc::new(StorageManager::new(config.storage.clone()));
        let query_engine = QueryEngine::new(
            storage.clone(),
            config.query.clone(),
            config.index.clone(),
            "euclidean".to_string(), // Default distance metric
        )?;

        Ok(Self {
            storage,
            query_engine,
            config,
            memory_monitor,
        })
    }

    /// Insert a single vector with metadata
    #[instrument(skip(self, vector, metadata))]
    pub fn insert(
        &self,
        id: VectorId,
        vector: VectorData,
        metadata: Metadata,
    ) -> VecLiteResult<()> {
        self.storage.insert(id, vector, metadata)
    }

    /// Insert multiple vectors in a batch operation
    #[instrument(skip(self, vectors))]
    pub fn insert_batch(
        &self,
        vectors: Vec<(VectorId, VectorData, Metadata)>,
    ) -> VecLiteResult<()> {
        self.storage.insert_batch(vectors)
    }

    /// Get a vector by ID
    pub fn get(&self, id: &VectorId) -> VecLiteResult<Option<VectorItem>> {
        self.storage.get(id)
    }

    /// Delete a vector by ID
    #[instrument(skip(self))]
    pub fn delete(&self, id: &VectorId) -> VecLiteResult<bool> {
        let deleted = self.storage.delete(id)?;
        if deleted {
            // TODO: Update index when HNSW is implemented
        }
        Ok(deleted)
    }

    /// Check if a vector exists by ID
    pub fn exists(&self, id: &VectorId) -> VecLiteResult<bool> {
        self.storage.exists(id)
    }

    /// Search for k nearest neighbors
    #[instrument(skip(self, query))]
    pub fn search(&self, query: &VectorData, k: usize) -> VecLiteResult<Vec<SearchResult>> {
        self.query_engine.search(query, k)
    }

    /// Search with a specific distance metric
    #[instrument(skip(self, query))]
    pub fn search_with_metric(
        &self,
        query: &VectorData,
        k: usize,
        metric_name: &str,
    ) -> VecLiteResult<Vec<SearchResult>> {
        self.query_engine.search_with_metric(query, k, metric_name)
    }

    /// Search with metadata filtering
    #[instrument(skip(self, query, filter))]
    pub fn search_with_filter<F>(
        &self,
        query: &VectorData,
        k: usize,
        filter: F,
    ) -> VecLiteResult<Vec<SearchResult>>
    where
        F: Fn(&Metadata) -> bool,
    {
        self.query_engine.search_with_filter(query, k, filter)
    }

    /// Batch search for multiple query vectors
    #[instrument(skip(self, queries))]
    pub fn batch_search(
        &self,
        queries: &[VectorData],
        k: usize,
    ) -> VecLiteResult<Vec<Vec<SearchResult>>> {
        let mut results = Vec::with_capacity(queries.len());

        for query in queries {
            let query_results = self.search(query, k)?;
            results.push(query_results);
        }

        Ok(results)
    }

    /// Get the number of vectors in the database
    pub fn len(&self) -> usize {
        self.storage.len()
    }

    /// Check if the database is empty
    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    /// Clear all vectors from the database
    #[instrument(skip(self))]
    pub fn clear(&self) -> VecLiteResult<()> {
        self.storage.clear()
    }

    /// Get database statistics with enhanced memory information
    pub fn stats(&self) -> types::StorageStats {
        let mut storage_stats = self.storage.stats();
        let memory_stats = self.memory_monitor.get_stats();
        storage_stats.update_memory_stats(memory_stats);

        // Check for memory warnings
        if let Some(warning) = storage_stats.check_memory_threshold(
            self.config.memory.max_memory_bytes,
            self.config.memory.warning_threshold_percent,
        ) {
            storage_stats.memory.warnings.push(warning);
        }

        storage_stats
    }

    /// Get current configuration
    pub fn config(&self) -> &VecLiteConfig {
        &self.config
    }

    /// Get memory monitor for advanced memory management
    pub fn memory_monitor(&self) -> &Arc<memory::MemoryMonitor> {
        &self.memory_monitor
    }

    /// Update memory configuration
    pub fn update_memory_config(&self, memory_config: types::MemoryConfig) {
        self.memory_monitor.update_config(memory_config.clone());
        // Note: This only updates the monitor; full config update would require reconstruction
    }

    /// Check if memory usage is within configured limits
    pub fn is_memory_within_limits(&self) -> bool {
        self.memory_monitor.is_within_limits()
    }

    /// Get detailed memory statistics
    pub fn memory_stats(&self) -> memory::MemoryStats {
        self.memory_monitor.get_stats()
    }

    /// Get reference to storage manager (internal use)
    pub(crate) fn storage(&self) -> &Arc<StorageManager> {
        &self.storage
    }

    /// Get available distance metrics
    pub fn available_metrics() -> Vec<&'static str> {
        DistanceMetricFactory::available_metrics()
    }

    /// Save the database to a HLX file
    #[instrument(skip(self, path))]
    pub fn save<P: AsRef<std::path::Path> + std::fmt::Debug>(&self, path: P) -> VecLiteResult<()> {
        info!("Saving Helix database to {:?}", path);
        let persistence = VLTPersistence::new(self.config.persistence.clone());
        persistence.save(self, path)
    }

    /// Load a database from a HLX file
    #[instrument(skip(path))]
    pub fn load<P: AsRef<std::path::Path> + std::fmt::Debug>(path: P) -> VecLiteResult<Self> {
        info!("Loading Helix database from {:?}", path);
        let persistence = VLTPersistence::new(types::PersistenceConfig::default());
        persistence.load(path)
    }

    /// Open or create a database file (like SQLite)
    /// If the file exists, load it; otherwise create a new database and save it
    #[instrument(skip(path))]
    pub fn open<P: AsRef<std::path::Path> + std::fmt::Debug>(path: P) -> VecLiteResult<Self> {
        let path_ref = path.as_ref();
        if path_ref.exists() {
            info!("Opening existing Helix database at {:?}", path);
            Self::load(path)
        } else {
            info!("Creating new Helix database at {:?}", path);
            let db = Self::new()?;
            db.save(path)?;
            Ok(db)
        }
    }

    /// Open the default database file in current directory
    /// Uses "vectors.hlx" as the default filename
    pub fn open_default() -> VecLiteResult<Self> {
        Self::open("vectors.hlx")
    }

    /// Open a database with automatic filename based on current directory name
    /// For example: if in "/home/user/my_project", uses "my_project.hlx"
    pub fn open_auto() -> VecLiteResult<Self> {
        let current_dir = std::env::current_dir()
            .map_err(|e| VecLiteError::Persistence(crate::error::PersistenceError::Io(e)))?;

        let dir_name = current_dir
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("vectors");

        let filename = format!("{}.hlx", dir_name);
        Self::open(filename)
    }
}

// Implement Default trait for convenience
impl Default for Helix {
    fn default() -> Self {
        Self::new().expect("Failed to create default Helix instance")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn create_test_vectors() -> Vec<(VectorId, VectorData, Metadata)> {
        vec![
            (
                "doc1".to_string(),
                vec![1.0, 0.0, 0.0],
                HashMap::from([("type".to_string(), "A".to_string())]),
            ),
            (
                "doc2".to_string(),
                vec![0.0, 1.0, 0.0],
                HashMap::from([("type".to_string(), "B".to_string())]),
            ),
            (
                "doc3".to_string(),
                vec![0.0, 0.0, 1.0],
                HashMap::from([("type".to_string(), "A".to_string())]),
            ),
        ]
    }

    #[test]
    fn test_helix_creation() {
        let db = Helix::new();
        assert!(db.is_ok());

        let db = db.unwrap();
        assert_eq!(db.len(), 0);
        assert!(db.is_empty());
    }

    #[test]
    fn test_vector_operations() {
        let db = Helix::new().unwrap();
        let vectors = create_test_vectors();

        // Test single insert
        let (id, vector, metadata) = vectors[0].clone();
        assert!(db
            .insert(id.clone(), vector.clone(), metadata.clone())
            .is_ok());
        assert_eq!(db.len(), 1);

        // Test get
        let retrieved = db.get(&id).unwrap();
        assert!(retrieved.is_some());
        let item = retrieved.unwrap();
        assert_eq!(item.id, id);
        assert_eq!(item.vector, vector);

        // Test exists
        assert!(db.exists(&id).unwrap());
        assert!(!db.exists(&"nonexistent".to_string()).unwrap());

        // Test delete
        assert!(db.delete(&id).unwrap());
        assert_eq!(db.len(), 0);
        assert!(!db.exists(&id).unwrap());
    }

    #[test]
    fn test_batch_operations() {
        let db = Helix::new().unwrap();
        let vectors = create_test_vectors();

        // Test batch insert
        assert!(db.insert_batch(vectors.clone()).is_ok());
        assert_eq!(db.len(), 3);

        // Verify all vectors were inserted
        for (id, _, _) in &vectors {
            assert!(db.exists(id).unwrap());
        }
    }

    #[test]
    fn test_search_operations() {
        let db = Helix::new().unwrap();
        let vectors = create_test_vectors();

        // Insert test vectors
        db.insert_batch(vectors).unwrap();

        // Test basic search
        let query = vec![1.0, 0.1, 0.1]; // Similar to doc1
        let results = db.search(&query, 2).unwrap();

        assert_eq!(results.len(), 2);
        // First result should be doc1 (most similar)
        assert_eq!(results[0].id, "doc1");

        // Test search with different metrics
        let results_cosine = db.search_with_metric(&query, 2, "cosine").unwrap();
        assert_eq!(results_cosine.len(), 2);

        let results_dot = db.search_with_metric(&query, 2, "dot_product").unwrap();
        assert_eq!(results_dot.len(), 2);
    }

    #[test]
    fn test_search_with_filter() {
        let db = Helix::new().unwrap();
        let vectors = create_test_vectors();

        db.insert_batch(vectors).unwrap();

        // Search only for type "A" documents
        let query = vec![0.5, 0.5, 0.5];
        let results = db
            .search_with_filter(&query, 5, |metadata| {
                metadata.get("type") == Some(&"A".to_string())
            })
            .unwrap();

        assert_eq!(results.len(), 2); // doc1 and doc3 have type "A"
        for result in results {
            assert!(result.id == "doc1" || result.id == "doc3");
        }
    }

    #[test]
    fn test_clear_database() {
        let db = Helix::new().unwrap();
        let vectors = create_test_vectors();

        db.insert_batch(vectors).unwrap();
        assert_eq!(db.len(), 3);

        assert!(db.clear().is_ok());
        assert_eq!(db.len(), 0);
        assert!(db.is_empty());
    }

    #[test]
    fn test_configuration() {
        let mut config = VecLiteConfig::default();
        config.storage.max_vectors = 100;
        config.query.default_k = 5;

        let db = Helix::with_config(config).unwrap();
        assert_eq!(db.config().storage.max_vectors, 100);
        assert_eq!(db.config().query.default_k, 5);
    }

    #[test]
    fn test_available_metrics() {
        let metrics = Helix::available_metrics();
        assert!(metrics.contains(&"euclidean"));
        assert!(metrics.contains(&"cosine"));
        assert!(metrics.contains(&"dot_product"));
        assert!(metrics.contains(&"manhattan"));
    }

    #[test]
    fn test_hnsw_index_configuration() {
        use crate::types::*;

        let mut config = VecLiteConfig::default();
        config.index.index_type = IndexType::HNSW;
        config.index.hnsw.ef_construction = 100;
        config.index.hnsw.max_m = 8;

        let db = Helix::with_config(config).unwrap();

        // Insert test vectors
        let vectors = vec![
            (
                "vec1".to_string(),
                vec![1.0, 2.0, 3.0],
                std::collections::HashMap::new(),
            ),
            (
                "vec2".to_string(),
                vec![2.0, 3.0, 4.0],
                std::collections::HashMap::new(),
            ),
            (
                "vec3".to_string(),
                vec![3.0, 4.0, 5.0],
                std::collections::HashMap::new(),
            ),
        ];

        for (id, vector, metadata) in vectors {
            db.insert(id, vector, metadata).unwrap();
        }

        // Search should work with HNSW index
        let query = vec![1.1, 2.1, 3.1];
        let results = db.search(&query, 2).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "vec1"); // Should be closest
    }

    #[test]
    fn test_error_handling() {
        let db = Helix::new().unwrap();

        // Test invalid dimensions
        db.insert("test1".to_string(), vec![1.0, 2.0], HashMap::new())
            .unwrap();

        let result = db.insert("test2".to_string(), vec![1.0, 2.0, 3.0], HashMap::new());
        assert!(result.is_err());

        // Test invalid metric
        let query = vec![1.0, 2.0];
        let result = db.search_with_metric(&query, 5, "invalid_metric");
        assert!(result.is_err());
    }

    #[test]
    fn test_statistics() {
        let db = Helix::new().unwrap();
        let vectors = create_test_vectors();

        // Initial stats
        let stats = db.stats();
        assert_eq!(stats.vector_count, 0);

        // Insert vectors and check stats
        db.insert_batch(vectors).unwrap();

        let stats = db.stats();
        assert_eq!(stats.vector_count, 3);
        assert!(stats.total_memory_bytes > 0);
        assert_eq!(stats.dimensions, Some(3));
    }

    #[test]
    fn test_persistence_save_load() {
        use tempfile::tempdir;

        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_db.hlx");

        // Create and populate original database
        let original_db = Helix::new().unwrap();
        let vectors = create_test_vectors();
        original_db.insert_batch(vectors.clone()).unwrap();

        // Verify original data
        assert_eq!(original_db.len(), 3);
        let original_stats = original_db.stats();

        // Save to file
        assert!(original_db.save(&file_path).is_ok());
        assert!(file_path.exists());

        // Load from file
        let loaded_db = Helix::load(&file_path).unwrap();

        // Verify loaded data matches original
        assert_eq!(loaded_db.len(), 3);
        let loaded_stats = loaded_db.stats();
        assert_eq!(loaded_stats.vector_count, original_stats.vector_count);
        assert_eq!(loaded_stats.dimensions, original_stats.dimensions);

        // Verify all vectors can be retrieved
        for (id, vector, metadata) in &vectors {
            let retrieved = loaded_db.get(id).unwrap();
            assert!(retrieved.is_some());
            let item = retrieved.unwrap();
            assert_eq!(item.vector, *vector);
            assert_eq!(item.metadata, *metadata);
        }

        // Test search functionality on loaded database
        let query = vec![1.0, 0.1, 0.1];
        let results = loaded_db.search(&query, 2).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_open_database() {
        use tempfile::tempdir;

        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_open.hlx");

        // First open should create a new database
        let db1 = Helix::open(&file_path).unwrap();
        assert_eq!(db1.len(), 0);
        assert!(file_path.exists());

        // Add some data
        let vectors = create_test_vectors();
        db1.insert_batch(vectors.clone()).unwrap();
        assert_eq!(db1.len(), 3);

        // Save the database
        db1.save(&file_path).unwrap();

        // Second open should load existing database
        let db2 = Helix::open(&file_path).unwrap();
        assert_eq!(db2.len(), 3);

        // Verify data integrity
        for (id, vector, metadata) in &vectors {
            let retrieved = db2.get(id).unwrap();
            assert!(retrieved.is_some());
            let item = retrieved.unwrap();
            assert_eq!(item.vector, *vector);
            assert_eq!(item.metadata, *metadata);
        }
    }

    #[test]
    fn test_default_database_operations() {
        use std::env;
        use tempfile::tempdir;

        // Test in a temporary directory to avoid conflicts
        let temp_dir = tempdir().unwrap();
        let original_dir = env::current_dir().unwrap();

        // Change to temp directory (create path if it doesn't exist)
        std::fs::create_dir_all(&temp_dir).unwrap();
        env::set_current_dir(&temp_dir).unwrap();

        // Test open_default
        let db = Helix::open_default().unwrap();
        assert_eq!(db.len(), 0);

        // Add data and save
        let vectors = create_test_vectors();
        db.insert_batch(vectors).unwrap();

        // Save and then force reload to test persistence
        let file_path = temp_dir.path().join("vectors.hlx");
        db.save(&file_path).unwrap();

        // Verify the default file exists
        assert!(file_path.exists());

        // Test loading the file directly (rather than using open_default which might create a new one)
        let db2 = Helix::load(&file_path).unwrap();
        assert_eq!(db2.len(), 3);

        // Restore original directory
        env::set_current_dir(&original_dir).unwrap();
    }

    #[test]
    fn test_auto_database_name() {
        use std::env;
        use tempfile::tempdir;

        let temp_dir = tempdir().unwrap();
        let original_dir = env::current_dir().unwrap();

        // Create a subdirectory with a specific name
        let project_dir = temp_dir.path().join("my_awesome_project");
        std::fs::create_dir_all(&project_dir).unwrap();
        env::set_current_dir(&project_dir).unwrap();

        // Test open_auto - should create "my_awesome_project.hlx"
        let db = Helix::open_auto().unwrap();
        assert_eq!(db.len(), 0);

        // Add data
        let vectors = create_test_vectors();
        db.insert_batch(vectors).unwrap();

        // Save using full path
        let file_path = project_dir.join("my_awesome_project.hlx");
        db.save(&file_path).unwrap();

        // Verify the auto-named file exists
        assert!(file_path.exists());

        // Test loading the file directly
        let db2 = Helix::load(&file_path).unwrap();
        assert_eq!(db2.len(), 3);

        // Restore original directory
        env::set_current_dir(&original_dir).unwrap();
    }
}
