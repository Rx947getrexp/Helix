use crate::error::{StorageError, VecLiteError, VecLiteResult};
use crate::types::{
    Dimensions, Metadata, StorageConfig, StorageStats, VectorData, VectorId, VectorItem,
};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tracing::{debug, info, instrument, trace, warn};

/// Vector storage trait - unified interface for storage implementations
/// This trait MUST be implemented by all storage backends
/// DO NOT create alternative storage interfaces
pub trait VectorStorage: Send + Sync {
    fn insert(&self, id: VectorId, vector: VectorData, metadata: Metadata) -> VecLiteResult<()>;
    fn get(&self, id: &VectorId) -> VecLiteResult<Option<VectorItem>>;
    fn delete(&self, id: &VectorId) -> VecLiteResult<bool>;
    fn exists(&self, id: &VectorId) -> VecLiteResult<bool>;
    fn iter(&self) -> Box<dyn Iterator<Item = VectorItem> + '_>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn clear(&self) -> VecLiteResult<()>;
    fn stats(&self) -> StorageStats;
    fn validate_vector(&self, vector: &VectorData) -> VecLiteResult<()>;
}

/// Thread-safe in-memory storage manager
/// Primary storage implementation using HashMap with RwLock for concurrency
#[derive(Debug)]
pub struct StorageManager {
    /// Thread-safe vector storage
    vectors: Arc<RwLock<HashMap<VectorId, VectorItem>>>,
    /// Storage configuration
    config: StorageConfig,
    /// Runtime statistics
    stats: Arc<RwLock<StorageStats>>,
    /// Expected vector dimensions (set on first insert)
    expected_dimensions: Arc<RwLock<Option<Dimensions>>>,
}

impl StorageManager {
    /// Create new storage manager with configuration
    pub fn new(config: StorageConfig) -> Self {
        info!(
            max_vectors = config.max_vectors,
            max_dimensions = config.max_dimensions,
            memory_limit_mb = config.memory_limit_bytes / 1024 / 1024,
            "Creating new StorageManager"
        );

        Self {
            vectors: Arc::new(RwLock::new(HashMap::with_capacity(1024))),
            config,
            stats: Arc::new(RwLock::new(StorageStats::default())),
            expected_dimensions: Arc::new(RwLock::new(None)),
        }
    }

    /// Create storage manager with default configuration
    pub fn with_defaults() -> Self {
        Self::new(StorageConfig::default())
    }

    /// Get current memory usage estimation in bytes
    fn estimate_memory_usage(&self) -> VecLiteResult<usize> {
        let vectors = self.vectors.read().map_err(|_| {
            VecLiteError::Storage(StorageError::InvalidValue {
                index: 0,
                value: 0.0,
            })
        })?; // Lock error handling

        let total_size = vectors
            .values()
            .map(|item| item.memory_size())
            .sum::<usize>();

        Ok(total_size)
    }

    /// Update internal statistics
    #[instrument(skip(self))]
    fn update_stats(&self) -> VecLiteResult<()> {
        let vectors = self.vectors.read().map_err(|_| {
            VecLiteError::Storage(StorageError::InvalidValue {
                index: 0,
                value: 0.0,
            })
        })?;

        let vector_count = vectors.len();
        let memory_bytes = self.estimate_memory_usage()?;

        // Get dimensions from first vector if available
        let dimensions = vectors.values().next().map(|v| v.dimensions());

        drop(vectors); // Release read lock

        let mut stats = self.stats.write().map_err(|_| {
            VecLiteError::Storage(StorageError::InvalidValue {
                index: 0,
                value: 0.0,
            })
        })?;

        if let Some(dims) = dimensions {
            stats.update(vector_count, memory_bytes, dims);
        } else {
            stats.vector_count = vector_count;
            stats.total_memory_bytes = memory_bytes;
        }

        trace!(
            vector_count = vector_count,
            memory_mb = memory_bytes / 1024 / 1024,
            dimensions = ?dimensions,
            "Updated storage statistics"
        );

        Ok(())
    }

    /// Validate vector against storage constraints
    fn validate_vector_internal(&self, vector: &VectorData) -> VecLiteResult<()> {
        // Check for empty vector
        if vector.is_empty() {
            return Err(VecLiteError::Storage(StorageError::EmptyVector));
        }

        // Check dimensions limit
        if vector.len() > self.config.max_dimensions {
            return Err(VecLiteError::Storage(StorageError::InvalidDimensions {
                expected: self.config.max_dimensions,
                actual: vector.len(),
            }));
        }

        // Check for invalid values (NaN, infinite)
        for (i, &value) in vector.iter().enumerate() {
            if !value.is_finite() {
                return Err(VecLiteError::Storage(StorageError::InvalidValue {
                    index: i,
                    value,
                }));
            }
        }

        // Check dimensions consistency
        let expected_dims = self.expected_dimensions.read().map_err(|_| {
            VecLiteError::Storage(StorageError::InvalidValue {
                index: 0,
                value: 0.0,
            })
        })?;

        if let Some(expected) = *expected_dims {
            if vector.len() != expected {
                return Err(VecLiteError::Storage(StorageError::InvalidDimensions {
                    expected,
                    actual: vector.len(),
                }));
            }
        }

        Ok(())
    }

    /// Set expected dimensions for all vectors
    fn set_expected_dimensions(&self, dimensions: Dimensions) -> VecLiteResult<()> {
        let mut expected_dims = self.expected_dimensions.write().map_err(|_| {
            VecLiteError::Storage(StorageError::InvalidValue {
                index: 0,
                value: 0.0,
            })
        })?;

        if expected_dims.is_none() {
            *expected_dims = Some(dimensions);
            debug!(dimensions = dimensions, "Set expected vector dimensions");
        }

        Ok(())
    }

    /// Batch insert vectors for improved performance
    #[instrument(skip(self, vectors))]
    pub fn insert_batch(
        &self,
        vectors: Vec<(VectorId, VectorData, Metadata)>,
    ) -> VecLiteResult<()> {
        if vectors.is_empty() {
            return Ok(());
        }

        // Validate all vectors first
        for (_id, vector, metadata) in &vectors {
            self.validate_vector_internal(vector)?;

            // Validate metadata size
            let metadata_size = metadata
                .iter()
                .map(|(k, v)| k.len() + v.len())
                .sum::<usize>();

            const MAX_METADATA_SIZE: usize = 1024;
            if metadata_size > MAX_METADATA_SIZE {
                return Err(VecLiteError::Storage(StorageError::MetadataTooLarge {
                    size: metadata_size,
                    max_size: MAX_METADATA_SIZE,
                }));
            }
        }

        // Check memory limit before batch insert
        let current_memory = self.estimate_memory_usage()?;
        let batch_memory_estimate: usize = vectors
            .iter()
            .map(|(id, vector, metadata)| {
                id.len() + vector.len() * 4 + // 4 bytes per f32
                metadata.iter().map(|(k, v)| k.len() + v.len()).sum::<usize>() + 64
            })
            .sum();

        if current_memory + batch_memory_estimate > self.config.memory_limit_bytes {
            return Err(VecLiteError::Storage(StorageError::MemoryLimitExceeded {
                current: current_memory + batch_memory_estimate,
            }));
        }

        // Check vector count limit
        let current_count = self.len();
        if current_count + vectors.len() > self.config.max_vectors {
            return Err(VecLiteError::Storage(StorageError::MemoryLimitExceeded {
                current: current_count + vectors.len(),
            }));
        }

        // Set dimensions if this is first batch
        if let Some((_, first_vector, _)) = vectors.first() {
            self.set_expected_dimensions(first_vector.len())?;
        }

        // Perform batch insert
        let mut storage = self.vectors.write().map_err(|_| {
            VecLiteError::Storage(StorageError::InvalidValue {
                index: 0,
                value: 0.0,
            })
        })?;

        let batch_size = vectors.len();
        for (id, vector, metadata) in vectors {
            // Check for duplicate ID
            if storage.contains_key(&id) {
                return Err(VecLiteError::Storage(StorageError::VectorAlreadyExists {
                    id: id.clone(),
                }));
            }

            let item = VectorItem::new(id.clone(), vector, metadata);
            storage.insert(id, item);
        }

        drop(storage); // Release write lock

        // Update statistics
        self.update_stats()?;

        info!(batch_size = batch_size, "Batch insert completed");
        Ok(())
    }

    /// Get multiple vectors by IDs
    pub fn get_batch(&self, ids: &[VectorId]) -> VecLiteResult<Vec<Option<VectorItem>>> {
        let vectors = self.vectors.read().map_err(|_| {
            VecLiteError::Storage(StorageError::InvalidValue {
                index: 0,
                value: 0.0,
            })
        })?;

        let results = ids.iter().map(|id| vectors.get(id).cloned()).collect();

        Ok(results)
    }

    /// Delete multiple vectors by IDs
    #[instrument(skip(self, ids))]
    pub fn delete_batch(&self, ids: &[VectorId]) -> VecLiteResult<usize> {
        if ids.is_empty() {
            return Ok(0);
        }

        let mut vectors = self.vectors.write().map_err(|_| {
            VecLiteError::Storage(StorageError::InvalidValue {
                index: 0,
                value: 0.0,
            })
        })?;

        let mut deleted_count = 0;
        for id in ids {
            if vectors.remove(id).is_some() {
                deleted_count += 1;
            }
        }

        drop(vectors); // Release write lock

        if deleted_count > 0 {
            self.update_stats()?;
            info!(deleted_count = deleted_count, "Batch delete completed");
        }

        Ok(deleted_count)
    }

    /// Get all vector IDs
    pub fn get_all_ids(&self) -> VecLiteResult<Vec<VectorId>> {
        let vectors = self.vectors.read().map_err(|_| {
            VecLiteError::Storage(StorageError::InvalidValue {
                index: 0,
                value: 0.0,
            })
        })?;

        Ok(vectors.keys().cloned().collect())
    }

    /// Filter vectors by metadata predicate
    pub fn filter_by_metadata<F>(&self, predicate: F) -> VecLiteResult<Vec<VectorItem>>
    where
        F: Fn(&Metadata) -> bool,
    {
        let vectors = self.vectors.read().map_err(|_| {
            VecLiteError::Storage(StorageError::InvalidValue {
                index: 0,
                value: 0.0,
            })
        })?;

        let filtered: Vec<VectorItem> = vectors
            .values()
            .filter(|item| predicate(&item.metadata))
            .cloned()
            .collect();

        Ok(filtered)
    }

    /// Get storage configuration
    pub fn config(&self) -> &StorageConfig {
        &self.config
    }

    /// Get expected vector dimensions
    pub fn expected_dimensions(&self) -> VecLiteResult<Option<Dimensions>> {
        let dims = self.expected_dimensions.read().map_err(|_| {
            VecLiteError::Storage(StorageError::InvalidValue {
                index: 0,
                value: 0.0,
            })
        })?;
        Ok(*dims)
    }
}

impl VectorStorage for StorageManager {
    #[instrument(skip(self, vector, metadata))]
    fn insert(&self, id: VectorId, vector: VectorData, metadata: Metadata) -> VecLiteResult<()> {
        // Validate vector
        self.validate_vector_internal(&vector)?;

        // Validate metadata size
        let metadata_size = metadata
            .iter()
            .map(|(k, v)| k.len() + v.len())
            .sum::<usize>();

        const MAX_METADATA_SIZE: usize = 1024;
        if metadata_size > MAX_METADATA_SIZE {
            return Err(VecLiteError::Storage(StorageError::MetadataTooLarge {
                size: metadata_size,
                max_size: MAX_METADATA_SIZE,
            }));
        }

        // Check memory limit
        let current_memory = self.estimate_memory_usage()?;
        let item_memory = id.len() + vector.len() * 4 + metadata_size + 64; // Estimate

        if current_memory + item_memory > self.config.memory_limit_bytes {
            return Err(VecLiteError::Storage(StorageError::MemoryLimitExceeded {
                current: current_memory + item_memory,
            }));
        }

        // Check vector count limit
        if self.len() >= self.config.max_vectors {
            return Err(VecLiteError::Storage(StorageError::MemoryLimitExceeded {
                current: self.len() + 1,
            }));
        }

        // Set expected dimensions if this is the first vector
        self.set_expected_dimensions(vector.len())?;

        // Insert vector
        let mut vectors = self.vectors.write().map_err(|_| {
            VecLiteError::Storage(StorageError::InvalidValue {
                index: 0,
                value: 0.0,
            })
        })?;

        // Check for duplicate ID
        if vectors.contains_key(&id) {
            return Err(VecLiteError::Storage(StorageError::VectorAlreadyExists {
                id: id.clone(),
            }));
        }

        let item = VectorItem::new(id.clone(), vector, metadata);
        vectors.insert(id.clone(), item);

        drop(vectors); // Release write lock

        // Update statistics
        self.update_stats()?;

        debug!(id = %id, "Vector inserted successfully");
        Ok(())
    }

    fn get(&self, id: &VectorId) -> VecLiteResult<Option<VectorItem>> {
        let vectors = self.vectors.read().map_err(|_| {
            VecLiteError::Storage(StorageError::InvalidValue {
                index: 0,
                value: 0.0,
            })
        })?;

        Ok(vectors.get(id).cloned())
    }

    #[instrument(skip(self))]
    fn delete(&self, id: &VectorId) -> VecLiteResult<bool> {
        let mut vectors = self.vectors.write().map_err(|_| {
            VecLiteError::Storage(StorageError::InvalidValue {
                index: 0,
                value: 0.0,
            })
        })?;

        let deleted = vectors.remove(id).is_some();

        drop(vectors); // Release write lock

        if deleted {
            self.update_stats()?;
            debug!(id = %id, "Vector deleted successfully");
        }

        Ok(deleted)
    }

    fn exists(&self, id: &VectorId) -> VecLiteResult<bool> {
        let vectors = self.vectors.read().map_err(|_| {
            VecLiteError::Storage(StorageError::InvalidValue {
                index: 0,
                value: 0.0,
            })
        })?;

        Ok(vectors.contains_key(id))
    }

    fn iter(&self) -> Box<dyn Iterator<Item = VectorItem> + '_> {
        // Note: This is a simplified implementation for testing
        // In production, we'd want a more sophisticated iterator that doesn't hold locks
        let vectors = self.vectors.read().unwrap();
        let items: Vec<VectorItem> = vectors.values().cloned().collect();
        Box::new(items.into_iter())
    }

    fn len(&self) -> usize {
        self.vectors.read().map(|v| v.len()).unwrap_or(0)
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[instrument(skip(self))]
    fn clear(&self) -> VecLiteResult<()> {
        let mut vectors = self.vectors.write().map_err(|_| {
            VecLiteError::Storage(StorageError::InvalidValue {
                index: 0,
                value: 0.0,
            })
        })?;

        let count = vectors.len();
        vectors.clear();

        drop(vectors); // Release write lock

        // Reset expected dimensions
        let mut expected_dims = self.expected_dimensions.write().map_err(|_| {
            VecLiteError::Storage(StorageError::InvalidValue {
                index: 0,
                value: 0.0,
            })
        })?;
        *expected_dims = None;

        drop(expected_dims);

        // Update statistics
        self.update_stats()?;

        info!(cleared_count = count, "Storage cleared");
        Ok(())
    }

    fn stats(&self) -> StorageStats {
        self.stats.read().unwrap().clone()
    }

    fn validate_vector(&self, vector: &VectorData) -> VecLiteResult<()> {
        self.validate_vector_internal(vector)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn create_test_config() -> StorageConfig {
        StorageConfig {
            max_vectors: 1000,
            max_dimensions: 128,
            memory_limit_bytes: 10 * 1024 * 1024, // 10MB for testing
            enable_checksums: true,
        }
    }

    fn create_test_vector(id: &str, dims: usize) -> (VectorId, VectorData, Metadata) {
        let vector: VectorData = (0..dims).map(|i| i as f32).collect();
        let metadata = HashMap::from([
            ("type".to_string(), "test".to_string()),
            ("id".to_string(), id.to_string()),
        ]);
        (id.to_string(), vector, metadata)
    }

    #[test]
    fn test_storage_manager_creation() {
        let config = create_test_config();
        let storage = StorageManager::new(config.clone());

        assert_eq!(storage.len(), 0);
        assert!(storage.is_empty());
        assert_eq!(storage.config().max_vectors, 1000);
    }

    #[test]
    fn test_vector_insert_and_get() {
        let storage = StorageManager::new(create_test_config());
        let (id, vector, metadata) = create_test_vector("test_1", 64);

        // Insert vector
        let result = storage.insert(id.clone(), vector.clone(), metadata.clone());
        assert!(result.is_ok());

        // Check storage state
        assert_eq!(storage.len(), 1);
        assert!(!storage.is_empty());

        // Get vector
        let retrieved = storage.get(&id).unwrap();
        assert!(retrieved.is_some());

        let item = retrieved.unwrap();
        assert_eq!(item.id, id);
        assert_eq!(item.vector, vector);
        assert_eq!(item.metadata, metadata);
        assert_eq!(item.dimensions(), 64);
    }

    #[test]
    fn test_vector_delete() {
        let storage = StorageManager::new(create_test_config());
        let (id, vector, metadata) = create_test_vector("test_delete", 32);

        // Insert then delete
        storage.insert(id.clone(), vector, metadata).unwrap();
        assert_eq!(storage.len(), 1);

        let deleted = storage.delete(&id).unwrap();
        assert!(deleted);
        assert_eq!(storage.len(), 0);

        // Delete non-existent vector
        let not_deleted = storage.delete(&id).unwrap();
        assert!(!not_deleted);
    }

    #[test]
    fn test_vector_exists() {
        let storage = StorageManager::new(create_test_config());
        let (id, vector, metadata) = create_test_vector("exists_test", 16);

        assert!(!storage.exists(&id).unwrap());

        storage.insert(id.clone(), vector, metadata).unwrap();
        assert!(storage.exists(&id).unwrap());

        storage.delete(&id).unwrap();
        assert!(!storage.exists(&id).unwrap());
    }

    #[test]
    fn test_batch_operations() {
        let storage = StorageManager::new(create_test_config());

        // Prepare batch data
        let batch_data = (0..10)
            .map(|i| create_test_vector(&format!("batch_{}", i), 32))
            .collect::<Vec<_>>();

        // Batch insert
        let result = storage.insert_batch(batch_data.clone());
        assert!(result.is_ok());
        assert_eq!(storage.len(), 10);

        // Batch get
        let ids: Vec<VectorId> = batch_data.iter().map(|(id, _, _)| id.clone()).collect();
        let retrieved = storage.get_batch(&ids).unwrap();
        assert_eq!(retrieved.len(), 10);
        assert!(retrieved.iter().all(|item| item.is_some()));

        // Batch delete
        let deleted_count = storage.delete_batch(&ids[0..5]).unwrap();
        assert_eq!(deleted_count, 5);
        assert_eq!(storage.len(), 5);
    }

    #[test]
    fn test_dimension_validation() {
        let storage = StorageManager::new(create_test_config());

        // Insert first vector with 64 dimensions
        let (id1, vector1, metadata1) = create_test_vector("dim_test_1", 64);
        storage.insert(id1, vector1, metadata1).unwrap();

        // Try to insert vector with different dimensions - should fail
        let (id2, vector2, metadata2) = create_test_vector("dim_test_2", 32);
        let result = storage.insert(id2, vector2, metadata2);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            VecLiteError::Storage(StorageError::InvalidDimensions { .. })
        ));
    }

    #[test]
    fn test_invalid_vector_values() {
        let storage = StorageManager::new(create_test_config());

        // Test NaN values
        let nan_vector = vec![1.0, f32::NAN, 3.0];
        let result = storage.insert("nan_test".to_string(), nan_vector, HashMap::new());
        assert!(result.is_err());

        // Test infinite values
        let inf_vector = vec![1.0, f32::INFINITY, 3.0];
        let result = storage.insert("inf_test".to_string(), inf_vector, HashMap::new());
        assert!(result.is_err());

        // Test empty vector
        let empty_vector = vec![];
        let result = storage.insert("empty_test".to_string(), empty_vector, HashMap::new());
        assert!(result.is_err());
    }

    #[test]
    fn test_duplicate_id_handling() {
        let storage = StorageManager::new(create_test_config());
        let (id, vector, metadata) = create_test_vector("duplicate_test", 16);

        // First insert should succeed
        let result1 = storage.insert(id.clone(), vector.clone(), metadata.clone());
        assert!(result1.is_ok());

        // Second insert with same ID should fail
        let result2 = storage.insert(id, vector, metadata);
        assert!(result2.is_err());
        assert!(matches!(
            result2.unwrap_err(),
            VecLiteError::Storage(StorageError::VectorAlreadyExists { .. })
        ));
    }

    #[test]
    fn test_metadata_size_limit() {
        let storage = StorageManager::new(create_test_config());

        // Create metadata that's too large (>1KB) - 1020 + 9 = 1029 > 1024
        let large_metadata =
            HashMap::from([("large_key".to_string(), "x".repeat(1020).to_string())]);

        let result = storage.insert(
            "large_metadata_test".to_string(),
            vec![1.0, 2.0, 3.0],
            large_metadata,
        );

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            VecLiteError::Storage(StorageError::MetadataTooLarge { .. })
        ));
    }

    #[test]
    fn test_storage_limits() {
        let mut limited_config = create_test_config();
        limited_config.max_vectors = 2; // Very small limit for testing

        let storage = StorageManager::new(limited_config);

        // Insert up to limit
        let (id1, vector1, metadata1) = create_test_vector("limit_1", 16);
        let (id2, vector2, metadata2) = create_test_vector("limit_2", 16);
        let (id3, vector3, metadata3) = create_test_vector("limit_3", 16);

        assert!(storage.insert(id1, vector1, metadata1).is_ok());
        assert!(storage.insert(id2, vector2, metadata2).is_ok());

        // Third insert should fail due to limit
        let result = storage.insert(id3, vector3, metadata3);
        assert!(result.is_err());
    }

    #[test]
    fn test_clear_storage() {
        let storage = StorageManager::new(create_test_config());

        // Insert some vectors
        for i in 0..5 {
            let (id, vector, metadata) = create_test_vector(&format!("clear_test_{}", i), 32);
            storage.insert(id, vector, metadata).unwrap();
        }

        assert_eq!(storage.len(), 5);

        // Clear storage
        let result = storage.clear();
        assert!(result.is_ok());
        assert_eq!(storage.len(), 0);
        assert!(storage.is_empty());

        // Should be able to insert new vectors with different dimensions after clear
        let (id, vector, metadata) = create_test_vector("after_clear", 64);
        let result = storage.insert(id, vector, metadata);
        assert!(result.is_ok());
    }

    #[test]
    fn test_statistics() {
        let storage = StorageManager::new(create_test_config());

        // Check initial stats
        let initial_stats = storage.stats();
        assert_eq!(initial_stats.vector_count, 0);
        assert_eq!(initial_stats.total_memory_bytes, 0);

        // Insert vectors and check updated stats
        for i in 0..10 {
            let (id, vector, metadata) = create_test_vector(&format!("stats_test_{}", i), 128);
            storage.insert(id, vector, metadata).unwrap();
        }

        let stats = storage.stats();
        assert_eq!(stats.vector_count, 10);
        assert!(stats.total_memory_bytes > 0);
        assert_eq!(stats.dimensions, Some(128));
        assert!(stats.average_vector_size > 0.0);
    }

    #[test]
    fn test_metadata_filtering() {
        let storage = StorageManager::new(create_test_config());

        // Insert vectors with different metadata
        for i in 0..5 {
            let vector: VectorData = vec![i as f32; 32];
            let metadata = HashMap::from([
                (
                    "type".to_string(),
                    if i % 2 == 0 {
                        "even".to_string()
                    } else {
                        "odd".to_string()
                    },
                ),
                ("value".to_string(), i.to_string()),
            ]);
            storage
                .insert(format!("filter_test_{}", i), vector, metadata)
                .unwrap();
        }

        // Filter by type
        let even_vectors = storage
            .filter_by_metadata(|meta| meta.get("type") == Some(&"even".to_string()))
            .unwrap();

        assert_eq!(even_vectors.len(), 3); // 0, 2, 4

        let odd_vectors = storage
            .filter_by_metadata(|meta| meta.get("type") == Some(&"odd".to_string()))
            .unwrap();

        assert_eq!(odd_vectors.len(), 2); // 1, 3
    }
}
