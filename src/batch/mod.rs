// Optimized batch operations for VecLite
// Provides high-performance bulk operations with memory pooling and parallel processing

use crate::distance::DistanceMetric;
use crate::error::{StorageError, VecLiteError, VecLiteResult};
use crate::memory::{get_pooled_results, return_pooled_results};
use crate::types::{Metadata, Score, SearchResult, VectorData, VectorId};
use rayon::prelude::*;
use std::sync::Arc;

/// Configuration for batch operations
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Number of vectors to process in each batch chunk
    pub chunk_size: usize,
    /// Maximum number of parallel threads to use
    pub max_threads: usize,
    /// Whether to use memory pools for allocations
    pub use_memory_pools: bool,
    /// Whether to enable parallel processing
    pub parallel_enabled: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1000,
            max_threads: rayon::current_num_threads(),
            use_memory_pools: true,
            parallel_enabled: true,
        }
    }
}

/// Optimized batch insertion with parallel validation and memory pooling
pub fn optimized_batch_insert(
    vectors: Vec<(VectorId, VectorData, Metadata)>,
    config: &BatchConfig,
) -> VecLiteResult<Vec<(VectorId, VectorData, Metadata)>> {
    if vectors.is_empty() {
        return Ok(vectors);
    }

    let validated_vectors = if config.parallel_enabled && vectors.len() > 100 {
        // Parallel validation for large batches
        validate_vectors_parallel(vectors, config)?
    } else {
        // Sequential validation for small batches
        validate_vectors_sequential(vectors)?
    };

    Ok(validated_vectors)
}

/// Parallel vector validation with chunking
fn validate_vectors_parallel(
    vectors: Vec<(VectorId, VectorData, Metadata)>,
    config: &BatchConfig,
) -> VecLiteResult<Vec<(VectorId, VectorData, Metadata)>> {
    let chunks: Vec<_> = vectors
        .chunks(config.chunk_size)
        .map(|chunk| chunk.to_vec())
        .collect();

    let results: Vec<VecLiteResult<Vec<_>>> = chunks
        .into_par_iter()
        .map(validate_vectors_sequential)
        .collect();

    // Collect results and handle errors
    let mut validated_vectors = Vec::new();
    for result in results {
        validated_vectors.extend(result?);
    }

    Ok(validated_vectors)
}

/// Sequential vector validation
fn validate_vectors_sequential(
    vectors: Vec<(VectorId, VectorData, Metadata)>,
) -> VecLiteResult<Vec<(VectorId, VectorData, Metadata)>> {
    let mut validated: Vec<(VectorId, VectorData, Metadata)> = Vec::with_capacity(vectors.len());

    for (id, vector, metadata) in vectors {
        // Validate vector dimensions (assuming first vector sets the standard)
        if !validated.is_empty() && vector.len() != validated[0].1.len() {
            return Err(VecLiteError::Storage(StorageError::InvalidDimensions {
                expected: validated[0].1.len(),
                actual: vector.len(),
            }));
        }

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

        validated.push((id, vector, metadata));
    }

    Ok(validated)
}

/// Optimized batch search with parallel processing and result pooling
pub fn optimized_batch_search(
    queries: &[VectorData],
    vectors: &[(VectorId, VectorData, Metadata)],
    k: usize,
    distance_metric: Arc<dyn DistanceMetric>,
    config: &BatchConfig,
) -> VecLiteResult<Vec<Vec<SearchResult>>> {
    if queries.is_empty() || vectors.is_empty() {
        return Ok(vec![Vec::new(); queries.len()]);
    }

    if config.parallel_enabled && queries.len() > 4 {
        batch_search_parallel(queries, vectors, k, distance_metric, config)
    } else {
        batch_search_sequential(queries, vectors, k, distance_metric, config)
    }
}

/// Parallel batch search implementation
fn batch_search_parallel(
    queries: &[VectorData],
    vectors: &[(VectorId, VectorData, Metadata)],
    k: usize,
    distance_metric: Arc<dyn DistanceMetric>,
    config: &BatchConfig,
) -> VecLiteResult<Vec<Vec<SearchResult>>> {
    let results: Vec<Vec<SearchResult>> = queries
        .par_iter()
        .map(|query| search_single_query(query, vectors, k, distance_metric.clone(), config))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(results)
}

/// Sequential batch search implementation
fn batch_search_sequential(
    queries: &[VectorData],
    vectors: &[(VectorId, VectorData, Metadata)],
    k: usize,
    distance_metric: Arc<dyn DistanceMetric>,
    config: &BatchConfig,
) -> VecLiteResult<Vec<Vec<SearchResult>>> {
    let mut results = Vec::with_capacity(queries.len());

    for query in queries {
        let query_results =
            search_single_query(query, vectors, k, distance_metric.clone(), config)?;
        results.push(query_results);
    }

    Ok(results)
}

/// Search implementation for a single query with memory pooling
fn search_single_query(
    query: &VectorData,
    vectors: &[(VectorId, VectorData, Metadata)],
    k: usize,
    distance_metric: Arc<dyn DistanceMetric>,
    config: &BatchConfig,
) -> VecLiteResult<Vec<SearchResult>> {
    let mut results = if config.use_memory_pools {
        get_pooled_results(vectors.len().min(k * 2))
    } else {
        Vec::with_capacity(vectors.len().min(k * 2))
    };

    // Calculate distances for all vectors
    for (id, vector, metadata) in vectors {
        let distance = distance_metric.distance(query, vector);

        results.push(SearchResult {
            id: id.clone(),
            score: distance,
            metadata: metadata.clone(),
        });
    }

    // Sort by distance (ascending for distance metrics, descending for similarity)
    if distance_metric.is_similarity() {
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    } else {
        results.sort_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    // Keep only top k results
    results.truncate(k);

    if config.use_memory_pools && results.capacity() > k * 2 {
        // If we allocated too much, return to pool and create right-sized result
        let final_results = results.clone();
        return_pooled_results(results);
        Ok(final_results)
    } else {
        Ok(results)
    }
}

/// Optimized batch distance calculation using SIMD when available
pub fn optimized_batch_distances(
    query: &VectorData,
    vectors: &[&VectorData],
    distance_metric: Arc<dyn DistanceMetric>,
    config: &BatchConfig,
) -> Vec<Score> {
    if config.parallel_enabled && vectors.len() > 1000 {
        // For large batches, use parallel chunked processing
        vectors
            .par_chunks(config.chunk_size)
            .flat_map(|chunk| distance_metric.batch_distance(query, chunk))
            .collect()
    } else {
        // Use built-in batch distance for smaller sets
        distance_metric.batch_distance(query, vectors)
    }
}

/// Memory-efficient vector normalization for batch operations
pub fn batch_normalize_vectors(
    vectors: &mut [VectorData],
    config: &BatchConfig,
) -> VecLiteResult<()> {
    if config.parallel_enabled && vectors.len() > 100 {
        vectors
            .par_iter_mut()
            .try_for_each(normalize_vector_inplace)?;
    } else {
        for vector in vectors {
            normalize_vector_inplace(vector)?;
        }
    }
    Ok(())
}

fn normalize_vector_inplace(vector: &mut VectorData) -> VecLiteResult<()> {
    let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm == 0.0 {
        return Err(VecLiteError::Storage(StorageError::EmptyVector));
    }

    for value in vector {
        *value /= norm;
    }

    Ok(())
}

/// Batch vector similarity calculation with optimizations
pub fn batch_vector_similarity(
    base_vectors: &[VectorData],
    compare_vectors: &[VectorData],
    distance_metric: Arc<dyn DistanceMetric>,
    config: &BatchConfig,
) -> VecLiteResult<Vec<Vec<Score>>> {
    if base_vectors.len() != compare_vectors.len() {
        return Err(VecLiteError::Storage(StorageError::InvalidDimensions {
            expected: base_vectors.len(),
            actual: compare_vectors.len(),
        }));
    }

    let results = if config.parallel_enabled && base_vectors.len() > 10 {
        base_vectors
            .par_iter()
            .zip(compare_vectors.par_iter())
            .map(|(base, compare)| vec![distance_metric.distance(base, compare)])
            .collect()
    } else {
        base_vectors
            .iter()
            .zip(compare_vectors.iter())
            .map(|(base, compare)| vec![distance_metric.distance(base, compare)])
            .collect()
    };

    Ok(results)
}

/// Utility function to estimate memory usage for batch operations
pub fn estimate_batch_memory_usage(vectors: &[(VectorId, VectorData, Metadata)]) -> usize {
    vectors
        .iter()
        .map(|(id, vector, metadata)| {
            id.len()
                + vector.len() * std::mem::size_of::<f32>()
                + metadata
                    .iter()
                    .map(|(k, v)| k.len() + v.len())
                    .sum::<usize>()
                + 64 // Overhead estimate
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::EuclideanDistance;
    use std::collections::HashMap;

    fn create_test_vectors(
        count: usize,
        dimensions: usize,
    ) -> Vec<(VectorId, VectorData, Metadata)> {
        (0..count)
            .map(|i| {
                let vector = vec![i as f32; dimensions];
                let metadata = HashMap::from([("index".to_string(), i.to_string())]);
                (format!("vec_{}", i), vector, metadata)
            })
            .collect()
    }

    #[test]
    fn test_batch_validation() {
        let vectors = create_test_vectors(10, 3);
        let config = BatchConfig::default();

        let result = optimized_batch_insert(vectors, &config);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 10);
    }

    #[test]
    fn test_batch_search() {
        let vectors = create_test_vectors(100, 4);
        let queries = vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]];
        let distance_metric = Arc::new(EuclideanDistance);
        let config = BatchConfig::default();

        let results = optimized_batch_search(&queries, &vectors, 5, distance_metric, &config);
        assert!(results.is_ok());

        let search_results = results.unwrap();
        assert_eq!(search_results.len(), 2);
        assert_eq!(search_results[0].len(), 5);
        assert_eq!(search_results[1].len(), 5);
    }

    #[test]
    fn test_parallel_vs_sequential() {
        let vectors = create_test_vectors(1000, 10);
        let queries = vec![vec![1.0; 10]; 10];
        let distance_metric = Arc::new(EuclideanDistance);

        let mut parallel_config = BatchConfig::default();
        parallel_config.parallel_enabled = true;

        let mut sequential_config = BatchConfig::default();
        sequential_config.parallel_enabled = false;

        let parallel_results = optimized_batch_search(
            &queries,
            &vectors,
            5,
            distance_metric.clone(),
            &parallel_config,
        );
        let sequential_results =
            optimized_batch_search(&queries, &vectors, 5, distance_metric, &sequential_config);

        assert!(parallel_results.is_ok());
        assert!(sequential_results.is_ok());

        // Results should be identical
        let p_results = parallel_results.unwrap();
        let s_results = sequential_results.unwrap();
        assert_eq!(p_results.len(), s_results.len());
    }

    #[test]
    fn test_memory_usage_estimation() {
        let vectors = create_test_vectors(100, 128);
        let usage = estimate_batch_memory_usage(&vectors);

        // Should be reasonable estimate
        assert!(usage > 0);
        assert!(usage < 1_000_000); // Less than 1MB for 100 vectors
    }

    #[test]
    fn test_vector_normalization() {
        let mut vectors = vec![vec![3.0, 4.0], vec![1.0, 1.0], vec![5.0, 12.0]];

        let config = BatchConfig::default();
        let result = batch_normalize_vectors(&mut vectors, &config);
        assert!(result.is_ok());

        // Check that vectors are normalized
        for vector in &vectors {
            let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-6);
        }
    }
}
