use crate::types::{Score, VectorData};
use std::fmt::Debug;

/// Distance metric trait - unified interface for all distance calculations
/// This trait MUST be implemented by all distance metrics
/// DO NOT create alternative distance calculation interfaces
pub trait DistanceMetric: Send + Sync + Debug {
    /// Calculate distance between two vectors
    /// Lower values indicate more similar vectors for distance metrics
    /// Higher values indicate more similar vectors for similarity metrics
    fn distance(&self, a: &VectorData, b: &VectorData) -> Score;

    /// Calculate distances from query to multiple vectors (batch optimization)
    fn batch_distance(&self, query: &VectorData, vectors: &[&VectorData]) -> Vec<Score>;

    /// Get the name of this distance metric
    fn name(&self) -> &'static str;

    /// Returns true for similarity metrics (higher = more similar)
    /// Returns false for distance metrics (lower = more similar)
    fn is_similarity(&self) -> bool;

    /// Validate that vectors have compatible dimensions
    fn validate_dimensions(
        &self,
        a: &VectorData,
        b: &VectorData,
    ) -> crate::error::VecLiteResult<()> {
        use crate::error::{QueryError, VecLiteError};

        if a.len() != b.len() {
            return Err(VecLiteError::Query(QueryError::DimensionMismatch {
                expected: a.len(),
                actual: b.len(),
            }));
        }
        Ok(())
    }
}

/// Euclidean distance implementation
/// L2 norm: sqrt(sum((a[i] - b[i])^2))
#[derive(Debug, Default, Clone)]
pub struct EuclideanDistance;

impl DistanceMetric for EuclideanDistance {
    fn distance(&self, a: &VectorData, b: &VectorData) -> Score {
        debug_assert_eq!(a.len(), b.len());

        let mut sum = 0.0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            let diff = x - y;
            sum += diff * diff;
        }
        sum.sqrt()
    }

    fn batch_distance(&self, query: &VectorData, vectors: &[&VectorData]) -> Vec<Score> {
        vectors
            .iter()
            .map(|vector| self.distance(query, vector))
            .collect()
    }

    fn name(&self) -> &'static str {
        "euclidean"
    }

    fn is_similarity(&self) -> bool {
        false // Distance metric: lower = more similar
    }
}

/// Cosine distance implementation
/// Cosine distance = 1 - cosine_similarity
/// Cosine similarity = dot(a, b) / (||a|| * ||b||)
#[derive(Debug, Default, Clone)]
pub struct CosineDistance;

impl DistanceMetric for CosineDistance {
    fn distance(&self, a: &VectorData, b: &VectorData) -> Score {
        debug_assert_eq!(a.len(), b.len());

        let (mut dot, mut norm_a, mut norm_b) = (0.0f32, 0.0f32, 0.0f32);

        for (x, y) in a.iter().zip(b.iter()) {
            dot += x * y;
            norm_a += x * x;
            norm_b += y * y;
        }

        // Handle zero vectors
        if norm_a == 0.0 || norm_b == 0.0 {
            return 1.0; // Maximum distance for zero vectors
        }

        let cosine_similarity = dot / (norm_a.sqrt() * norm_b.sqrt());
        1.0 - cosine_similarity
    }

    fn batch_distance(&self, query: &VectorData, vectors: &[&VectorData]) -> Vec<Score> {
        // Pre-compute query norm for efficiency
        let query_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();

        if query_norm == 0.0 {
            return vec![1.0; vectors.len()]; // Max distance for zero query
        }

        vectors
            .iter()
            .map(|vector| {
                let (mut dot, mut vector_norm) = (0.0f32, 0.0f32);

                for (q, v) in query.iter().zip(vector.iter()) {
                    dot += q * v;
                    vector_norm += v * v;
                }

                if vector_norm == 0.0 {
                    1.0 // Max distance for zero vector
                } else {
                    let cosine_similarity = dot / (query_norm * vector_norm.sqrt());
                    1.0 - cosine_similarity
                }
            })
            .collect()
    }

    fn name(&self) -> &'static str {
        "cosine"
    }

    fn is_similarity(&self) -> bool {
        false // Distance metric: lower = more similar
    }
}

/// Dot product similarity implementation
/// Higher dot product = more similar vectors
#[derive(Debug, Default, Clone)]
pub struct DotProductSimilarity;

impl DistanceMetric for DotProductSimilarity {
    fn distance(&self, a: &VectorData, b: &VectorData) -> Score {
        debug_assert_eq!(a.len(), b.len());

        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    fn batch_distance(&self, query: &VectorData, vectors: &[&VectorData]) -> Vec<Score> {
        vectors
            .iter()
            .map(|vector| query.iter().zip(vector.iter()).map(|(q, v)| q * v).sum())
            .collect()
    }

    fn name(&self) -> &'static str {
        "dot_product"
    }

    fn is_similarity(&self) -> bool {
        true // Similarity metric: higher = more similar
    }
}

/// Manhattan distance implementation (L1 norm)
/// Sum of absolute differences: sum(|a[i] - b[i]|)
#[derive(Debug, Default, Clone)]
pub struct ManhattanDistance;

impl DistanceMetric for ManhattanDistance {
    fn distance(&self, a: &VectorData, b: &VectorData) -> Score {
        debug_assert_eq!(a.len(), b.len());

        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
    }

    fn batch_distance(&self, query: &VectorData, vectors: &[&VectorData]) -> Vec<Score> {
        vectors
            .iter()
            .map(|vector| self.distance(query, vector))
            .collect()
    }

    fn name(&self) -> &'static str {
        "manhattan"
    }

    fn is_similarity(&self) -> bool {
        false // Distance metric: lower = more similar
    }
}

/// Factory for creating distance metrics
pub struct DistanceMetricFactory;

impl DistanceMetricFactory {
    /// Create distance metric by name
    pub fn create(name: &str) -> crate::error::VecLiteResult<Box<dyn DistanceMetric>> {
        use crate::error::{QueryError, VecLiteError};

        match name.to_lowercase().as_str() {
            "euclidean" | "l2" => Ok(Box::new(EuclideanDistance)),
            "cosine" => Ok(Box::new(CosineDistance)),
            "dot_product" | "dot" => Ok(Box::new(DotProductSimilarity)),
            "manhattan" | "l1" => Ok(Box::new(ManhattanDistance)),
            _ => Err(VecLiteError::Query(QueryError::UnsupportedMetric {
                metric: name.to_string(),
            })),
        }
    }

    /// List all available distance metrics
    pub fn available_metrics() -> Vec<&'static str> {
        vec!["euclidean", "cosine", "dot_product", "manhattan"]
    }
}

// SIMD-optimized implementations for supported platforms
// Only compiled when simd feature is enabled
#[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
mod simd_optimizations {
    use super::*;

    impl EuclideanDistance {
        /// SIMD-optimized Euclidean distance for AVX2
        pub fn distance_simd(&self, a: &VectorData, b: &VectorData) -> Score {
            debug_assert_eq!(a.len(), b.len());

            unsafe {
                use std::arch::x86_64::*;

                let mut sum = _mm256_setzero_ps();
                let chunks = a.len() / 8;

                // Process 8 floats at a time
                for i in 0..chunks {
                    let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
                    let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
                    let diff = _mm256_sub_ps(va, vb);
                    sum = _mm256_fmadd_ps(diff, diff, sum);
                }

                // Reduce and handle remainder
                let mut result = [0.0f32; 8];
                _mm256_storeu_ps(result.as_mut_ptr(), sum);
                let mut final_sum = result.iter().sum::<f32>();

                // Handle remaining elements
                for i in (chunks * 8)..a.len() {
                    let diff = a[i] - b[i];
                    final_sum += diff * diff;
                }

                final_sum.sqrt()
            }
        }
    }
}

/// Utility functions for vector operations
pub mod vector_ops {
    use super::*;

    /// Normalize vector to unit length
    pub fn normalize(vector: &mut VectorData) {
        let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in vector.iter_mut() {
                *x /= norm;
            }
        }
    }

    /// Calculate vector magnitude
    pub fn magnitude(vector: &VectorData) -> f32 {
        vector.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Check if vector has valid dimensions and values
    pub fn validate_vector(
        vector: &VectorData,
        expected_dims: usize,
    ) -> crate::error::VecLiteResult<()> {
        use crate::error::{StorageError, VecLiteError};

        // Check dimensions
        if vector.len() != expected_dims {
            return Err(VecLiteError::Storage(StorageError::InvalidDimensions {
                expected: expected_dims,
                actual: vector.len(),
            }));
        }

        // Check for invalid values
        for (i, &value) in vector.iter().enumerate() {
            if !value.is_finite() {
                return Err(VecLiteError::Storage(StorageError::InvalidValue {
                    index: i,
                    value,
                }));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_vectors() -> (VectorData, VectorData, VectorData) {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let zero = vec![0.0, 0.0, 0.0];
        (a, b, zero)
    }

    #[test]
    fn test_euclidean_distance() {
        let (a, b, _) = create_test_vectors();
        let metric = EuclideanDistance;

        let distance = metric.distance(&a, &b);
        let expected =
            ((4.0f32 - 1.0).powi(2) + (5.0f32 - 2.0).powi(2) + (6.0f32 - 3.0).powi(2)).sqrt();

        assert!((distance - expected).abs() < 1e-6);
        assert!(!metric.is_similarity());
        assert_eq!(metric.name(), "euclidean");
    }

    #[test]
    fn test_cosine_distance() {
        let (a, b, zero) = create_test_vectors();
        let metric = CosineDistance;

        // Normal vectors
        let distance = metric.distance(&a, &b);
        assert!(distance >= 0.0 && distance <= 2.0);

        // Zero vector handling
        let distance_zero = metric.distance(&a, &zero);
        assert_eq!(distance_zero, 1.0); // Maximum distance

        assert!(!metric.is_similarity());
        assert_eq!(metric.name(), "cosine");
    }

    #[test]
    fn test_dot_product_similarity() {
        let (a, b, _) = create_test_vectors();
        let metric = DotProductSimilarity;

        let similarity = metric.distance(&a, &b);
        let expected = 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0; // 32.0

        assert_eq!(similarity, expected);
        assert!(metric.is_similarity());
        assert_eq!(metric.name(), "dot_product");
    }

    #[test]
    fn test_manhattan_distance() {
        let (a, b, _) = create_test_vectors();
        let metric = ManhattanDistance;

        let distance = metric.distance(&a, &b);
        let expected = (4.0f32 - 1.0).abs() + (5.0f32 - 2.0).abs() + (6.0f32 - 3.0).abs(); // 9.0

        assert_eq!(distance, expected);
        assert!(!metric.is_similarity());
        assert_eq!(metric.name(), "manhattan");
    }

    #[test]
    fn test_batch_distance_optimization() {
        let query = vec![1.0, 2.0, 3.0];
        let vectors = vec![
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
            vec![0.0, 1.0, 2.0],
        ];
        let vector_refs: Vec<&VectorData> = vectors.iter().collect();

        let metric = EuclideanDistance;
        let batch_distances = metric.batch_distance(&query, &vector_refs);

        assert_eq!(batch_distances.len(), 3);

        // Verify batch results match individual calculations
        for (i, vector) in vectors.iter().enumerate() {
            let individual_distance = metric.distance(&query, vector);
            assert!((batch_distances[i] - individual_distance).abs() < 1e-6);
        }
    }

    #[test]
    fn test_distance_metric_factory() {
        // Valid metrics
        let euclidean = DistanceMetricFactory::create("euclidean").unwrap();
        assert_eq!(euclidean.name(), "euclidean");

        let cosine = DistanceMetricFactory::create("COSINE").unwrap(); // Case insensitive
        assert_eq!(cosine.name(), "cosine");

        // Invalid metric
        let invalid = DistanceMetricFactory::create("invalid_metric");
        assert!(invalid.is_err());

        // Available metrics
        let available = DistanceMetricFactory::available_metrics();
        assert!(available.contains(&"euclidean"));
        assert!(available.contains(&"cosine"));
        assert!(available.contains(&"dot_product"));
        assert!(available.contains(&"manhattan"));
    }

    #[test]
    fn test_dimension_validation() {
        let metric = EuclideanDistance;

        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0]; // Different dimensions

        let result = metric.validate_dimensions(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_operations() {
        use vector_ops::*;

        // Normalization
        let mut vector = vec![3.0, 4.0]; // Magnitude = 5.0
        normalize(&mut vector);
        let norm = magnitude(&vector);
        assert!((norm - 1.0).abs() < 1e-6);

        // Magnitude calculation
        let vector = vec![3.0, 4.0];
        let mag = magnitude(&vector);
        assert_eq!(mag, 5.0);

        // Vector validation
        let valid_vector = vec![1.0, 2.0, 3.0];
        assert!(validate_vector(&valid_vector, 3).is_ok());

        let invalid_dims = vec![1.0, 2.0];
        assert!(validate_vector(&invalid_dims, 3).is_err());

        let invalid_values = vec![1.0, f32::NAN, 3.0];
        assert!(validate_vector(&invalid_values, 3).is_err());
    }

    #[test]
    fn test_cosine_distance_edge_cases() {
        let metric = CosineDistance;

        // Identical vectors should have 0 distance
        let a = vec![1.0, 2.0, 3.0];
        let distance = metric.distance(&a, &a);
        assert!(distance.abs() < 1e-6);

        // Opposite vectors should have maximum distance (2.0)
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let distance = metric.distance(&a, &b);
        assert!((distance - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_all_metrics_consistency() {
        let query = vec![1.0, 2.0, 3.0];
        let vectors = vec![vec![1.0, 2.0, 3.0]]; // Identical to query
        let vector_refs: Vec<&VectorData> = vectors.iter().collect();

        // Test all metrics with identical vectors
        let euclidean = EuclideanDistance;
        let distance = euclidean.distance(&query, &vectors[0]);
        assert!(distance.abs() < 1e-6); // Should be ~0 for identical vectors

        let cosine = CosineDistance;
        let distance = cosine.distance(&query, &vectors[0]);
        assert!(distance.abs() < 1e-6); // Should be ~0 for identical vectors

        // Batch operations should match individual calculations
        let batch_results = euclidean.batch_distance(&query, &vector_refs);
        assert_eq!(batch_results.len(), 1);
        assert!(batch_results[0].abs() < 1e-6);
    }
}
