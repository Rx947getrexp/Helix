use crate::distance::{DistanceMetric, DistanceMetricFactory};
use crate::error::{QueryError, VecLiteError, VecLiteResult};
use crate::hnsw::{HNSWConfig, HNSWIndex};
use crate::storage::{StorageManager, VectorStorage};
use crate::types::{
    IndexConfig, IndexType, Metadata, QueryConfig, Score, SearchResult, VectorData, VectorId,
};
use std::sync::Arc;
use tracing::{debug, instrument, trace};

/// Search index trait - unified interface for all indexing algorithms
/// This trait MUST be implemented by all index types
/// DO NOT create alternative index interfaces
pub trait SearchIndex: Send + Sync + std::fmt::Debug {
    fn build(&mut self, vectors: &[(VectorId, VectorData)]) -> VecLiteResult<()>;
    fn insert(&mut self, id: VectorId, vector: VectorData) -> VecLiteResult<()>;
    fn delete(&mut self, id: &VectorId) -> VecLiteResult<bool>;
    fn search(&self, query: &VectorData, k: usize) -> VecLiteResult<Vec<SearchResult>>;
    fn stats(&self) -> IndexStats;
    fn clear(&mut self) -> VecLiteResult<()>;
    fn name(&self) -> &'static str;
}

/// Index statistics
#[derive(Debug, Clone, Default)]
pub struct IndexStats {
    pub vector_count: usize,
    pub index_size_bytes: usize,
    pub build_time_ms: u64,
    pub last_build_time: u64,
    pub search_count: u64,
    pub average_search_time_ms: f64,
    /// HNSW-specific: average node degree across all layers
    pub average_degree: f64,
    /// HNSW-specific: maximum layer in the index
    pub max_level: usize,
}

/// Brute force index implementation
/// Linear search through all vectors - O(n) complexity but guaranteed exact results
#[derive(Debug)]
pub struct BruteForceIndex {
    vectors: Vec<(VectorId, VectorData)>,
    distance_metric: Box<dyn DistanceMetric>,
    stats: IndexStats,
}

impl BruteForceIndex {
    /// Create new brute force index with specified distance metric
    pub fn new(metric_name: &str) -> VecLiteResult<Self> {
        let distance_metric = DistanceMetricFactory::create(metric_name)?;

        Ok(Self {
            vectors: Vec::new(),
            distance_metric,
            stats: IndexStats::default(),
        })
    }

    /// Create with default Euclidean distance
    pub fn with_euclidean() -> VecLiteResult<Self> {
        Self::new("euclidean")
    }

    /// Update search statistics
    #[allow(dead_code)]
    fn update_search_stats(&mut self, search_time_ms: u64) {
        self.stats.search_count += 1;
        let total_time = self.stats.average_search_time_ms * (self.stats.search_count - 1) as f64
            + search_time_ms as f64;
        self.stats.average_search_time_ms = total_time / self.stats.search_count as f64;
    }
}

impl SearchIndex for BruteForceIndex {
    #[instrument(skip(self, vectors))]
    fn build(&mut self, vectors: &[(VectorId, VectorData)]) -> VecLiteResult<()> {
        let start_time = std::time::Instant::now();

        self.vectors.clear();
        self.vectors.extend_from_slice(vectors);

        let build_time = start_time.elapsed();
        self.stats.vector_count = self.vectors.len();
        self.stats.build_time_ms = build_time.as_millis() as u64;
        self.stats.last_build_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Estimate index size (vector data + IDs)
        self.stats.index_size_bytes = vectors
            .iter()
            .map(|(id, vec)| id.len() + vec.len() * std::mem::size_of::<f32>())
            .sum();

        debug!(
            vector_count = vectors.len(),
            build_time_ms = build_time.as_millis(),
            "Built brute force index"
        );

        Ok(())
    }

    #[instrument(skip(self, vector))]
    fn insert(&mut self, id: VectorId, vector: VectorData) -> VecLiteResult<()> {
        // Check for duplicate ID
        if self
            .vectors
            .iter()
            .any(|(existing_id, _)| existing_id == &id)
        {
            return Err(VecLiteError::Index(
                crate::error::IndexError::InsertFailed {
                    reason: format!("Vector with ID '{}' already exists in index", id),
                },
            ));
        }

        // Validate dimensions against existing vectors
        if let Some((_, first_vector)) = self.vectors.first() {
            if vector.len() != first_vector.len() {
                return Err(VecLiteError::Query(QueryError::DimensionMismatch {
                    expected: first_vector.len(),
                    actual: vector.len(),
                }));
            }
        }

        self.vectors.push((id.clone(), vector));
        self.stats.vector_count = self.vectors.len();

        trace!(id = %id, vector_count = self.vectors.len(), "Inserted vector into brute force index");
        Ok(())
    }

    #[instrument(skip(self))]
    fn delete(&mut self, id: &VectorId) -> VecLiteResult<bool> {
        let initial_len = self.vectors.len();
        self.vectors.retain(|(existing_id, _)| existing_id != id);

        let deleted = self.vectors.len() < initial_len;
        if deleted {
            self.stats.vector_count = self.vectors.len();
            trace!(id = %id, vector_count = self.vectors.len(), "Deleted vector from brute force index");
        }

        Ok(deleted)
    }

    #[instrument(skip(self, query))]
    fn search(&self, query: &VectorData, k: usize) -> VecLiteResult<Vec<SearchResult>> {
        let start_time = std::time::Instant::now();

        if self.vectors.is_empty() {
            return Err(VecLiteError::Query(QueryError::EmptyDataset));
        }

        // Validate k parameter
        let effective_k = k.min(self.vectors.len());
        if k == 0 {
            return Err(VecLiteError::Query(QueryError::InvalidK {
                k,
                max: self.vectors.len(),
            }));
        }

        // Validate query dimensions
        if let Some((_, first_vector)) = self.vectors.first() {
            if query.len() != first_vector.len() {
                return Err(VecLiteError::Query(QueryError::DimensionMismatch {
                    expected: first_vector.len(),
                    actual: query.len(),
                }));
            }
        }

        // Calculate distances to all vectors
        let vector_refs: Vec<&VectorData> = self.vectors.iter().map(|(_, v)| v).collect();
        let distances = self.distance_metric.batch_distance(query, &vector_refs);

        // Create (distance, index) pairs and sort
        let mut scored_results: Vec<(Score, usize)> = distances
            .into_iter()
            .enumerate()
            .map(|(idx, dist)| (dist, idx))
            .collect();

        // Sort by score (ascending for distance metrics, descending for similarity metrics)
        if self.distance_metric.is_similarity() {
            scored_results
                .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            scored_results
                .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        }

        // Convert to SearchResult objects
        let results: Vec<SearchResult> = scored_results
            .into_iter()
            .take(effective_k)
            .map(|(score, idx)| {
                let (id, _) = &self.vectors[idx];
                SearchResult::new(id.clone(), score, std::collections::HashMap::new())
            })
            .collect();

        let search_time = start_time.elapsed().as_millis() as u64;
        // Note: We can't modify self in a &self method, so stats update would need to be handled differently
        // For now, we'll track this separately in the QueryEngine

        trace!(
            query_dims = query.len(),
            k = k,
            results_count = results.len(),
            search_time_ms = search_time,
            "Completed brute force search"
        );

        Ok(results)
    }

    fn stats(&self) -> IndexStats {
        self.stats.clone()
    }

    fn clear(&mut self) -> VecLiteResult<()> {
        self.vectors.clear();
        self.stats = IndexStats::default();
        debug!("Cleared brute force index");
        Ok(())
    }

    fn name(&self) -> &'static str {
        "brute_force"
    }
}

/// Query engine that coordinates search operations
/// Manages the index and provides high-level search interfaces
#[derive(Debug)]
pub struct QueryEngine {
    storage: Arc<StorageManager>,
    index: Box<dyn SearchIndex>,
    config: QueryConfig,
    default_metric: String,
}

impl QueryEngine {
    /// Create new query engine with storage reference and configuration
    pub fn new(
        storage: Arc<StorageManager>,
        query_config: QueryConfig,
        index_config: IndexConfig,
        default_metric: String,
    ) -> VecLiteResult<Self> {
        let index = Self::create_index(&index_config, &default_metric)?;

        Ok(Self {
            storage,
            index,
            config: query_config,
            default_metric,
        })
    }

    /// Create appropriate index based on configuration
    fn create_index(
        config: &IndexConfig,
        default_metric: &str,
    ) -> VecLiteResult<Box<dyn SearchIndex>> {
        match config.index_type {
            IndexType::BruteForce => Ok(Box::new(BruteForceIndex::new(default_metric)?)),
            IndexType::HNSW => {
                let distance_metric: Arc<dyn DistanceMetric> = match default_metric {
                    "euclidean" | "l2" => Arc::new(crate::distance::EuclideanDistance),
                    "cosine" => Arc::new(crate::distance::CosineDistance),
                    "dot_product" | "dot" => Arc::new(crate::distance::DotProductSimilarity),
                    "manhattan" | "l1" => Arc::new(crate::distance::ManhattanDistance),
                    _ => {
                        return Err(VecLiteError::Query(QueryError::UnsupportedMetric {
                            metric: default_metric.to_string(),
                        }))
                    }
                };
                let hnsw_config = HNSWConfig {
                    max_connections_0: config.hnsw.max_m_l,
                    max_connections: config.hnsw.max_m,
                    ef_construction: config.hnsw.ef_construction,
                    ef_search: config.hnsw.ef_search,
                    ml: config.hnsw.ml,
                    seed: 42, // Use configurable seed in future
                };
                Ok(Box::new(HNSWIndex::new(hnsw_config, distance_metric)))
            }
            IndexType::VPTree => {
                // TODO: Implement VP-Tree in Stage 2.5 or later
                Err(VecLiteError::Query(QueryError::QueryFailed {
                    reason: "VP-Tree index not yet implemented".to_string(),
                }))
            }
        }
    }

    /// Search for k nearest neighbors using default metric
    #[instrument(skip(self, query))]
    pub fn search(&self, query: &VectorData, k: usize) -> VecLiteResult<Vec<SearchResult>> {
        self.search_with_metric(query, k, &self.default_metric)
    }

    /// Search with specified distance metric
    #[instrument(skip(self, query))]
    pub fn search_with_metric(
        &self,
        query: &VectorData,
        k: usize,
        metric_name: &str,
    ) -> VecLiteResult<Vec<SearchResult>> {
        // Validate k parameter
        if k == 0 {
            return Err(VecLiteError::Query(QueryError::InvalidK {
                k,
                max: self.config.max_k,
            }));
        }

        if k > self.config.max_k {
            return Err(VecLiteError::Query(QueryError::InvalidK {
                k,
                max: self.config.max_k,
            }));
        }

        // For now, we'll use brute force search directly with storage
        // In future stages, this will use the appropriate index
        self.brute_force_search(query, k, metric_name)
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
        // Get filtered vectors from storage
        let filtered_items = self.storage.filter_by_metadata(filter)?;

        if filtered_items.is_empty() {
            return Ok(Vec::new());
        }

        // Create temporary dataset for search
        let _filtered_vectors: Vec<(VectorId, VectorData)> = filtered_items
            .iter()
            .map(|item| (item.id.clone(), item.vector.clone()))
            .collect();

        // Perform brute force search on filtered dataset
        let metric = DistanceMetricFactory::create(&self.default_metric)?;
        let mut scored_results: Vec<(Score, &crate::types::VectorItem)> = filtered_items
            .iter()
            .map(|item| {
                let distance = metric.distance(query, &item.vector);
                (distance, item)
            })
            .collect();

        // Sort results
        if metric.is_similarity() {
            scored_results
                .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            scored_results
                .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        }

        // Convert to SearchResult and take top k
        let results: Vec<SearchResult> = scored_results
            .into_iter()
            .take(k.min(filtered_items.len()))
            .map(|(score, item)| SearchResult::new(item.id.clone(), score, item.metadata.clone()))
            .collect();

        Ok(results)
    }

    /// Perform brute force search directly on storage
    #[instrument(skip(self, query))]
    fn brute_force_search(
        &self,
        query: &VectorData,
        k: usize,
        metric_name: &str,
    ) -> VecLiteResult<Vec<SearchResult>> {
        let metric = DistanceMetricFactory::create(metric_name)?;

        // Get all vectors from storage
        let all_items: Vec<_> = self.storage.iter().collect();

        if all_items.is_empty() {
            return Err(VecLiteError::Query(QueryError::EmptyDataset));
        }

        // Validate query dimensions
        if let Some(first_item) = all_items.first() {
            if query.len() != first_item.vector.len() {
                return Err(VecLiteError::Query(QueryError::DimensionMismatch {
                    expected: first_item.vector.len(),
                    actual: query.len(),
                }));
            }
        }

        // Calculate distances to all vectors
        let mut scored_results: Vec<(Score, &crate::types::VectorItem)> = all_items
            .iter()
            .map(|item| {
                let distance = metric.distance(query, &item.vector);
                (distance, item)
            })
            .collect();

        // Sort by score
        if metric.is_similarity() {
            scored_results
                .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            scored_results
                .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        }

        // Convert to SearchResult and take top k
        let effective_k = k.min(all_items.len());
        let results: Vec<SearchResult> = scored_results
            .into_iter()
            .take(effective_k)
            .map(|(score, item)| SearchResult::new(item.id.clone(), score, item.metadata.clone()))
            .collect();

        debug!(
            query_dims = query.len(),
            k = k,
            effective_k = effective_k,
            results_count = results.len(),
            metric = metric_name,
            "Completed brute force search"
        );

        Ok(results)
    }

    /// Batch search for multiple queries
    #[instrument(skip(self, queries))]
    pub fn batch_search(
        &self,
        queries: &[VectorData],
        k: usize,
    ) -> VecLiteResult<Vec<Vec<SearchResult>>> {
        queries.iter().map(|query| self.search(query, k)).collect()
    }

    /// Get query engine statistics
    pub fn stats(&self) -> IndexStats {
        self.index.stats()
    }

    /// Get current configuration
    pub fn config(&self) -> &QueryConfig {
        &self.config
    }

    /// Update the distance metric used for searches
    pub fn set_default_metric(&mut self, metric_name: String) -> VecLiteResult<()> {
        // Validate metric exists
        DistanceMetricFactory::create(&metric_name)?;
        self.default_metric = metric_name;
        Ok(())
    }

    /// Rebuild index from current storage contents
    /// This will be more important when we have sophisticated indices like HNSW
    #[instrument(skip(self))]
    pub fn rebuild_index(&mut self) -> VecLiteResult<()> {
        let all_items: Vec<_> = self.storage.iter().collect();
        let vectors: Vec<(VectorId, VectorData)> = all_items
            .iter()
            .map(|item| (item.id.clone(), item.vector.clone()))
            .collect();

        self.index.build(&vectors)?;
        debug!(vector_count = vectors.len(), "Rebuilt search index");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::StorageManager;
    use crate::types::{HNSWConfig, IndexConfig, IndexType, StorageConfig, VPTreeConfig};
    use std::collections::HashMap;

    fn create_test_storage() -> Arc<StorageManager> {
        Arc::new(StorageManager::new(StorageConfig::default()))
    }

    fn create_test_index_config() -> IndexConfig {
        IndexConfig {
            index_type: IndexType::BruteForce,
            hnsw: HNSWConfig::default(),
            vp_tree: VPTreeConfig::default(),
        }
    }

    fn create_test_vectors() -> Vec<(VectorId, VectorData, Metadata)> {
        vec![
            (
                "vec1".to_string(),
                vec![1.0, 0.0, 0.0],
                HashMap::from([("type".to_string(), "A".to_string())]),
            ),
            (
                "vec2".to_string(),
                vec![0.0, 1.0, 0.0],
                HashMap::from([("type".to_string(), "B".to_string())]),
            ),
            (
                "vec3".to_string(),
                vec![0.0, 0.0, 1.0],
                HashMap::from([("type".to_string(), "A".to_string())]),
            ),
            (
                "vec4".to_string(),
                vec![0.5, 0.5, 0.0],
                HashMap::from([("type".to_string(), "C".to_string())]),
            ),
        ]
    }

    #[test]
    fn test_brute_force_index_creation() {
        let index = BruteForceIndex::new("euclidean");
        assert!(index.is_ok());

        let index = index.unwrap();
        assert_eq!(index.name(), "brute_force");
        assert_eq!(index.stats().vector_count, 0);
    }

    #[test]
    fn test_brute_force_index_build() {
        let mut index = BruteForceIndex::new("euclidean").unwrap();

        let vectors = vec![
            ("vec1".to_string(), vec![1.0, 0.0]),
            ("vec2".to_string(), vec![0.0, 1.0]),
        ];

        let result = index.build(&vectors);
        assert!(result.is_ok());

        let stats = index.stats();
        assert_eq!(stats.vector_count, 2);
        assert!(stats.index_size_bytes > 0);
    }

    #[test]
    fn test_brute_force_index_operations() {
        let mut index = BruteForceIndex::new("euclidean").unwrap();

        // Test insert
        let result = index.insert("vec1".to_string(), vec![1.0, 0.0]);
        assert!(result.is_ok());
        assert_eq!(index.stats().vector_count, 1);

        // Test duplicate insert
        let result = index.insert("vec1".to_string(), vec![2.0, 0.0]);
        assert!(result.is_err());

        // Test search
        let results = index.search(&vec![1.1, 0.1], 1);
        assert!(results.is_ok());
        let results = results.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "vec1");

        // Test delete
        let deleted = index.delete(&"vec1".to_string());
        assert!(deleted.is_ok());
        assert!(deleted.unwrap());
        assert_eq!(index.stats().vector_count, 0);
    }

    #[test]
    fn test_brute_force_search_sorting() {
        let mut index = BruteForceIndex::new("euclidean").unwrap();

        // Insert vectors at different distances from origin
        index.insert("far".to_string(), vec![3.0, 4.0]).unwrap(); // Distance 5.0
        index.insert("near".to_string(), vec![1.0, 1.0]).unwrap(); // Distance ~1.41
        index.insert("medium".to_string(), vec![2.0, 2.0]).unwrap(); // Distance ~2.83

        // Search from origin - should return in order of increasing distance
        let results = index.search(&vec![0.0, 0.0], 3).unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].id, "near"); // Closest
        assert_eq!(results[1].id, "medium"); // Middle
        assert_eq!(results[2].id, "far"); // Farthest
    }

    #[test]
    fn test_query_engine_creation() {
        let storage = create_test_storage();
        let config = QueryConfig::default();

        let engine = QueryEngine::new(
            storage,
            config,
            create_test_index_config(),
            "euclidean".to_string(),
        );
        assert!(engine.is_ok());

        let engine = engine.unwrap();
        assert_eq!(engine.config().default_k, 10);
    }

    #[test]
    fn test_query_engine_search() {
        let storage = create_test_storage();
        let vectors = create_test_vectors();

        // Insert vectors into storage
        for (id, vector, metadata) in vectors {
            storage.insert(id, vector, metadata).unwrap();
        }

        let config = QueryConfig::default();
        let engine = QueryEngine::new(
            storage,
            config,
            create_test_index_config(),
            "euclidean".to_string(),
        )
        .unwrap();

        // Test search
        let query = vec![1.0, 0.1, 0.1]; // Close to vec1
        let results = engine.search(&query, 2).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "vec1"); // Should be closest
    }

    #[test]
    fn test_search_with_different_metrics() {
        let storage = create_test_storage();
        let vectors = create_test_vectors();

        for (id, vector, metadata) in vectors {
            storage.insert(id, vector, metadata).unwrap();
        }

        let config = QueryConfig::default();
        let engine = QueryEngine::new(
            storage,
            config,
            create_test_index_config(),
            "euclidean".to_string(),
        )
        .unwrap();

        let query = vec![1.0, 0.5, 0.0];

        // Test different metrics
        let euclidean_results = engine.search_with_metric(&query, 2, "euclidean").unwrap();
        let cosine_results = engine.search_with_metric(&query, 2, "cosine").unwrap();
        let dot_results = engine.search_with_metric(&query, 2, "dot_product").unwrap();

        assert_eq!(euclidean_results.len(), 2);
        assert_eq!(cosine_results.len(), 2);
        assert_eq!(dot_results.len(), 2);

        // Results might be in different orders due to different metrics
        // but should contain valid vector IDs
        for results in [&euclidean_results, &cosine_results, &dot_results] {
            for result in results {
                assert!(["vec1", "vec2", "vec3", "vec4"].contains(&result.id.as_str()));
            }
        }
    }

    #[test]
    fn test_search_with_metadata_filter() {
        let storage = create_test_storage();
        let vectors = create_test_vectors();

        for (id, vector, metadata) in vectors {
            storage.insert(id, vector, metadata).unwrap();
        }

        let config = QueryConfig::default();
        let engine = QueryEngine::new(
            storage,
            config,
            create_test_index_config(),
            "euclidean".to_string(),
        )
        .unwrap();

        let query = vec![0.5, 0.5, 0.5];

        // Filter for type "A" only
        let results = engine
            .search_with_filter(&query, 5, |metadata| {
                metadata.get("type") == Some(&"A".to_string())
            })
            .unwrap();

        assert_eq!(results.len(), 2); // vec1 and vec3 have type "A"
        for result in results {
            assert!(result.id == "vec1" || result.id == "vec3");
            assert_eq!(result.metadata.get("type"), Some(&"A".to_string()));
        }
    }

    #[test]
    fn test_batch_search() {
        let storage = create_test_storage();
        let vectors = create_test_vectors();

        for (id, vector, metadata) in vectors {
            storage.insert(id, vector, metadata).unwrap();
        }

        let config = QueryConfig::default();
        let engine = QueryEngine::new(
            storage,
            config,
            create_test_index_config(),
            "euclidean".to_string(),
        )
        .unwrap();

        let queries = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];

        let batch_results = engine.batch_search(&queries, 2).unwrap();

        assert_eq!(batch_results.len(), 3);
        for results in batch_results {
            assert_eq!(results.len(), 2);
        }
    }

    #[test]
    fn test_query_validation() {
        let storage = create_test_storage();
        storage
            .insert("vec1".to_string(), vec![1.0, 2.0], HashMap::new())
            .unwrap();

        let config = QueryConfig::default();
        let engine = QueryEngine::new(
            storage,
            config,
            create_test_index_config(),
            "euclidean".to_string(),
        )
        .unwrap();

        // Test invalid k values
        let query = vec![1.0, 2.0];
        assert!(engine.search(&query, 0).is_err()); // k = 0
        assert!(engine.search(&query, 99999).is_err()); // k > max_k

        // Test dimension mismatch
        let wrong_dim_query = vec![1.0, 2.0, 3.0];
        assert!(engine.search(&wrong_dim_query, 1).is_err());
    }

    #[test]
    fn test_empty_dataset_search() {
        let storage = create_test_storage();
        let config = QueryConfig::default();
        let engine = QueryEngine::new(
            storage,
            config,
            create_test_index_config(),
            "euclidean".to_string(),
        )
        .unwrap();

        let query = vec![1.0, 2.0];
        let result = engine.search(&query, 1);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            VecLiteError::Query(QueryError::EmptyDataset)
        ));
    }

    #[test]
    fn test_invalid_metric() {
        let storage = create_test_storage();
        let config = QueryConfig::default();
        let engine = QueryEngine::new(
            storage,
            config,
            create_test_index_config(),
            "euclidean".to_string(),
        )
        .unwrap();

        let query = vec![1.0, 2.0];
        let result = engine.search_with_metric(&query, 1, "invalid_metric");
        assert!(result.is_err());
    }
}
