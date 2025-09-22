use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Unique identifier for vectors
/// Must be used throughout the codebase - DO NOT create alternatives
pub type VectorId = String;

/// Vector dimensions type
/// Must be used for all dimension-related operations
pub type Dimensions = usize;

/// Vector data representation
/// Must be used for all vector data - DO NOT create alternatives
pub type VectorData = Vec<f32>;

/// Distance/similarity score
/// Used for all distance and similarity calculations
pub type Score = f32;

/// Metadata key-value pairs
/// Must be used for all vector metadata - DO NOT create alternatives
pub type Metadata = HashMap<String, String>;

/// Core vector item structure
/// Represents a single vector with its associated data
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VectorItem {
    /// Unique identifier for this vector
    pub id: VectorId,
    /// The vector data (high-dimensional embedding)
    pub vector: VectorData,
    /// Associated metadata as key-value pairs
    pub metadata: Metadata,
    /// Timestamp when vector was created/updated
    pub timestamp: u64,
}

impl VectorItem {
    /// Create a new vector item with current timestamp
    pub fn new(id: VectorId, vector: VectorData, metadata: Metadata) -> Self {
        Self {
            id,
            vector,
            metadata,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }

    /// Create a new vector item with specified timestamp
    pub fn with_timestamp(
        id: VectorId,
        vector: VectorData,
        metadata: Metadata,
        timestamp: u64,
    ) -> Self {
        Self {
            id,
            vector,
            metadata,
            timestamp,
        }
    }

    /// Get vector dimensions
    pub fn dimensions(&self) -> Dimensions {
        self.vector.len()
    }

    /// Estimate memory usage in bytes
    pub fn memory_size(&self) -> usize {
        let id_size = self.id.len();
        let vector_size = self.vector.len() * std::mem::size_of::<f32>();
        let metadata_size = self
            .metadata
            .iter()
            .map(|(k, v)| k.len() + v.len())
            .sum::<usize>();
        let timestamp_size = std::mem::size_of::<u64>();

        id_size + vector_size + metadata_size + timestamp_size + 64 // struct overhead
    }

    /// Validate vector data
    pub fn validate(&self) -> crate::error::VecLiteResult<()> {
        use crate::error::{StorageError, VecLiteError};

        // Check for empty vector
        if self.vector.is_empty() {
            return Err(VecLiteError::Storage(StorageError::EmptyVector));
        }

        // Check for invalid values (NaN, infinite)
        for (i, &value) in self.vector.iter().enumerate() {
            if !value.is_finite() {
                return Err(VecLiteError::Storage(StorageError::InvalidValue {
                    index: i,
                    value,
                }));
            }
        }

        // Check metadata size
        let metadata_size = self
            .metadata
            .iter()
            .map(|(k, v)| k.len() + v.len())
            .sum::<usize>();

        const MAX_METADATA_SIZE: usize = 1024; // 1KB as per requirements
        if metadata_size > MAX_METADATA_SIZE {
            return Err(VecLiteError::Storage(StorageError::MetadataTooLarge {
                size: metadata_size,
                max_size: MAX_METADATA_SIZE,
            }));
        }

        Ok(())
    }
}

/// Search result with score and metadata
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    /// Vector ID that matched the query
    pub id: VectorId,
    /// Similarity/distance score (meaning depends on metric)
    pub score: Score,
    /// Associated metadata
    pub metadata: Metadata,
}

impl SearchResult {
    pub fn new(id: VectorId, score: Score, metadata: Metadata) -> Self {
        Self {
            id,
            score,
            metadata,
        }
    }
}

/// Configuration for VecLite
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VecLiteConfig {
    pub storage: StorageConfig,
    pub index: IndexConfig,
    pub query: QueryConfig,
    pub persistence: PersistenceConfig,
    pub memory: MemoryConfig,
}

/// Memory management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Maximum total memory usage in bytes
    pub max_memory_bytes: usize,
    /// Enable memory pooling for allocations
    pub enable_pooling: bool,
    /// Maximum vectors per memory pool
    pub pool_max_vectors: usize,
    /// Maximum pools to maintain
    pub pool_max_pools: usize,
    /// Enable memory monitoring and statistics
    pub enable_monitoring: bool,
    /// Memory warning threshold (percentage of max_memory_bytes)
    pub warning_threshold_percent: u8,
    /// Memory cleanup threshold (percentage of max_memory_bytes)
    pub cleanup_threshold_percent: u8,
    /// Enable automatic garbage collection
    pub enable_auto_cleanup: bool,
    /// Cleanup interval in seconds
    pub cleanup_interval_seconds: u64,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_memory_bytes: 2_147_483_648, // 2GB default
            enable_pooling: true,
            pool_max_vectors: 100,
            pool_max_pools: 1000,
            enable_monitoring: true,
            warning_threshold_percent: 80,
            cleanup_threshold_percent: 90,
            enable_auto_cleanup: true,
            cleanup_interval_seconds: 60,
        }
    }
}

/// Storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub max_vectors: usize,
    pub max_dimensions: Dimensions,
    pub memory_limit_bytes: usize,
    pub enable_checksums: bool,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            max_vectors: 1_000_000,            // 1M vectors max
            max_dimensions: 4096,              // Up to 4096 dimensions
            memory_limit_bytes: 2_147_483_648, // 2GB default limit
            enable_checksums: true,
        }
    }
}

/// Index configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    pub index_type: IndexType,
    pub hnsw: HNSWConfig,
    pub vp_tree: VPTreeConfig,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            index_type: IndexType::BruteForce, // Start with brute force in Stage 1
            hnsw: HNSWConfig::default(),
            vp_tree: VPTreeConfig::default(),
        }
    }
}

/// Index algorithm type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum IndexType {
    BruteForce,
    HNSW,
    VPTree,
}

/// HNSW-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HNSWConfig {
    pub m: usize,               // Maximum connections per layer 0 node
    pub max_m: usize,           // Maximum connections per layer 0 node
    pub max_m_l: usize,         // Maximum connections per upper layer node
    pub ef_construction: usize, // Size of dynamic candidate list during construction
    pub ef_search: usize,       // Size of dynamic candidate list during search
    pub ml: f64,                // Level generation factor
}

impl Default for HNSWConfig {
    fn default() -> Self {
        Self {
            m: 16,
            max_m: 16,
            max_m_l: 16,
            ef_construction: 200,
            ef_search: 50,
            ml: 1.0 / (2.0_f64).ln(),
        }
    }
}

/// VP-Tree specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VPTreeConfig {
    pub leaf_size: usize,   // Maximum items in leaf nodes
    pub sample_size: usize, // Number of samples for vantage point selection
}

impl Default for VPTreeConfig {
    fn default() -> Self {
        Self {
            leaf_size: 32,
            sample_size: 10,
        }
    }
}

/// Query engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryConfig {
    pub default_k: usize,
    pub max_k: usize,
    pub ef_search: usize,
    pub enable_metadata_filter: bool,
}

impl Default for QueryConfig {
    fn default() -> Self {
        Self {
            default_k: 10,
            max_k: 10000,
            ef_search: 50,
            enable_metadata_filter: true,
        }
    }
}

/// Persistence configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceConfig {
    pub compression_enabled: bool,
    pub compression_level: i32,
    pub checksum_enabled: bool,
    pub backup_count: usize,
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        Self {
            compression_enabled: true,
            compression_level: 3,
            checksum_enabled: true,
            backup_count: 3,
        }
    }
}

/// Statistics about storage usage
#[derive(Debug, Clone, Default)]
pub struct StorageStats {
    pub vector_count: usize,
    pub total_memory_bytes: usize,
    pub average_vector_size: f64,
    pub dimensions: Option<Dimensions>,
    pub last_updated: u64,
    pub memory: MemoryStats,
}

/// Detailed memory usage statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Current allocated memory in bytes
    pub current_allocated: usize,
    /// Peak memory usage in bytes
    pub peak_allocated: usize,
    /// Total number of allocations performed
    pub total_allocations: u64,
    /// Total number of deallocations performed
    pub total_deallocations: u64,
    /// Memory pool statistics
    pub pool_stats: PoolStats,
    /// Memory usage by component
    pub component_usage: ComponentMemoryUsage,
    /// Memory warnings and alerts
    pub warnings: Vec<MemoryWarning>,
}

/// Memory pool usage statistics
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Total vectors currently pooled
    pub total_pooled_vectors: usize,
    /// Number of active pools
    pub active_pools: usize,
    /// Pool efficiency (reuse rate)
    pub pool_efficiency_percent: f32,
    /// Memory saved by pooling in bytes
    pub memory_saved_bytes: usize,
}

/// Memory usage breakdown by component
#[derive(Debug, Clone, Default)]
pub struct ComponentMemoryUsage {
    /// Memory used by vector data
    pub vectors_bytes: usize,
    /// Memory used by metadata
    pub metadata_bytes: usize,
    /// Memory used by HNSW index
    pub index_bytes: usize,
    /// Memory used by internal structures
    pub overhead_bytes: usize,
}

/// Memory warning types
#[derive(Debug, Clone)]
pub enum MemoryWarning {
    HighUsage {
        current_bytes: usize,
        threshold_bytes: usize,
        percent_used: u8,
    },
    PoolInefficiency {
        pool_id: usize,
        efficiency_percent: f32,
    },
    FrequentAllocations {
        allocations_per_second: f32,
    },
}

impl StorageStats {
    pub fn update(&mut self, vector_count: usize, memory_bytes: usize, dimensions: Dimensions) {
        self.vector_count = vector_count;
        self.total_memory_bytes = memory_bytes;
        self.average_vector_size = if vector_count > 0 {
            memory_bytes as f64 / vector_count as f64
        } else {
            0.0
        };
        self.dimensions = Some(dimensions);
        self.last_updated = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
    }

    /// Update memory statistics from memory monitor
    pub fn update_memory_stats(&mut self, memory_stats: crate::memory::MemoryStats) {
        self.memory.current_allocated = memory_stats.current_allocated;
        self.memory.peak_allocated = memory_stats.peak_allocated;
        self.memory.total_allocations = memory_stats.total_allocations;
        self.memory.total_deallocations = memory_stats.total_deallocations;
    }

    /// Check if memory usage exceeds threshold
    pub fn check_memory_threshold(
        &self,
        max_memory: usize,
        threshold_percent: u8,
    ) -> Option<MemoryWarning> {
        let threshold_bytes = (max_memory as f64 * threshold_percent as f64 / 100.0) as usize;
        if self.memory.current_allocated > threshold_bytes {
            let percent_used =
                (self.memory.current_allocated as f64 / max_memory as f64 * 100.0) as u8;
            Some(MemoryWarning::HighUsage {
                current_bytes: self.memory.current_allocated,
                threshold_bytes,
                percent_used,
            })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_item_creation() {
        let vector = vec![1.0, 2.0, 3.0];
        let metadata = HashMap::from([("key".to_string(), "value".to_string())]);
        let item = VectorItem::new("test_id".to_string(), vector.clone(), metadata.clone());

        assert_eq!(item.id, "test_id");
        assert_eq!(item.vector, vector);
        assert_eq!(item.metadata, metadata);
        assert_eq!(item.dimensions(), 3);
        assert!(item.timestamp > 0);
    }

    #[test]
    fn test_vector_item_validation() {
        // Valid vector
        let valid_item = VectorItem::new("valid".to_string(), vec![1.0, 2.0, 3.0], HashMap::new());
        assert!(valid_item.validate().is_ok());

        // Empty vector
        let empty_item = VectorItem::new("empty".to_string(), vec![], HashMap::new());
        assert!(empty_item.validate().is_err());

        // Invalid values (NaN)
        let nan_item = VectorItem::new("nan".to_string(), vec![1.0, f32::NAN, 3.0], HashMap::new());
        assert!(nan_item.validate().is_err());

        // Invalid values (infinite)
        let inf_item = VectorItem::new(
            "inf".to_string(),
            vec![1.0, f32::INFINITY, 3.0],
            HashMap::new(),
        );
        assert!(inf_item.validate().is_err());
    }

    #[test]
    fn test_vector_item_memory_size() {
        let vector = vec![1.0; 768]; // 768-dimensional vector
        let metadata = HashMap::from([
            ("type".to_string(), "document".to_string()),
            ("source".to_string(), "test".to_string()),
        ]);
        let item = VectorItem::new("test_vector_123".to_string(), vector, metadata);

        let size = item.memory_size();

        // Should include ID + vector + metadata + timestamp + overhead
        let expected_min = "test_vector_123".len() + 768 * 4 + "typedocumentsourcetest".len() + 8;
        assert!(size >= expected_min);
    }

    #[test]
    fn test_metadata_size_limit() {
        let vector = vec![1.0, 2.0, 3.0];
        let large_metadata = HashMap::from([
            ("large_key".to_string(), "x".repeat(1020).to_string()), // >1KB metadata (1020 + 9 = 1029 > 1024)
        ]);
        let item = VectorItem::new("test".to_string(), vector, large_metadata);

        assert!(item.validate().is_err());
    }

    #[test]
    fn test_search_result() {
        let metadata = HashMap::from([("type".to_string(), "test".to_string())]);
        let result = SearchResult::new("result_1".to_string(), 0.85, metadata.clone());

        assert_eq!(result.id, "result_1");
        assert_eq!(result.score, 0.85);
        assert_eq!(result.metadata, metadata);
    }

    #[test]
    fn test_config_defaults() {
        let config = VecLiteConfig::default();

        assert_eq!(config.storage.max_vectors, 1_000_000);
        assert_eq!(config.storage.max_dimensions, 4096);
        assert_eq!(config.index.index_type, IndexType::BruteForce);
        assert_eq!(config.query.default_k, 10);
        assert_eq!(config.query.max_k, 10000);
        assert!(config.persistence.compression_enabled);
        assert_eq!(config.memory.max_memory_bytes, 2_147_483_648);
        assert!(config.memory.enable_pooling);
        assert!(config.memory.enable_monitoring);
    }

    #[test]
    fn test_storage_stats() {
        let mut stats = StorageStats::default();
        stats.update(1000, 4_000_000, 768);

        assert_eq!(stats.vector_count, 1000);
        assert_eq!(stats.total_memory_bytes, 4_000_000);
        assert_eq!(stats.average_vector_size, 4000.0);
        assert_eq!(stats.dimensions, Some(768));
        assert!(stats.last_updated > 0);
    }

    #[test]
    fn test_memory_threshold_checking() {
        let mut stats = StorageStats::default();
        stats.memory.current_allocated = 800_000_000; // 800MB

        let max_memory = 1_000_000_000; // 1GB
        let warning = stats.check_memory_threshold(max_memory, 70); // 70% threshold

        assert!(warning.is_some());
        if let Some(MemoryWarning::HighUsage { percent_used, .. }) = warning {
            assert_eq!(percent_used, 80); // 800MB / 1GB = 80%
        } else {
            panic!("Expected HighUsage warning");
        }

        let no_warning = stats.check_memory_threshold(max_memory, 90); // 90% threshold
        assert!(no_warning.is_none());
    }

    #[test]
    fn test_memory_config_defaults() {
        let config = MemoryConfig::default();

        assert_eq!(config.max_memory_bytes, 2_147_483_648);
        assert!(config.enable_pooling);
        assert!(config.enable_monitoring);
        assert_eq!(config.warning_threshold_percent, 80);
        assert_eq!(config.cleanup_threshold_percent, 90);
        assert!(config.enable_auto_cleanup);
        assert_eq!(config.cleanup_interval_seconds, 60);
    }

    #[test]
    fn test_type_aliases_consistency() {
        // Ensure type aliases work as expected
        let _id: VectorId = "test".to_string();
        let _dims: Dimensions = 768;
        let _vector: VectorData = vec![1.0, 2.0, 3.0];
        let _score: Score = 0.95;
        let _meta: Metadata = HashMap::new();

        // Compilation success means type aliases are working correctly
    }
}
