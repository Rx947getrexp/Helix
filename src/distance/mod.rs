// Distance calculation module for VecLite
// This module provides both standard and SIMD-optimized distance calculations

pub mod legacy;
pub mod simd;

// Re-export legacy implementations for compatibility
pub use legacy::*;

use crate::types::{Score, VectorData};

/// SIMD-optimized versions of distance metrics
/// These provide significant performance improvements on supported hardware
#[derive(Debug, Default, Clone)]
pub struct SIMDEuclideanDistance;

#[derive(Debug, Default, Clone)]
pub struct SIMDCosineDistance;

#[derive(Debug, Default, Clone)]
pub struct SIMDDotProductSimilarity;

#[derive(Debug, Default, Clone)]
pub struct SIMDManhattanDistance;

impl DistanceMetric for SIMDEuclideanDistance {
    fn distance(&self, a: &VectorData, b: &VectorData) -> Score {
        simd::euclidean_distance_simd(a, b)
    }

    fn batch_distance(&self, query: &VectorData, vectors: &[&VectorData]) -> Vec<Score> {
        simd::euclidean_distance_batch_simd(query, vectors)
    }

    fn name(&self) -> &'static str {
        "euclidean_simd"
    }

    fn is_similarity(&self) -> bool {
        false
    }
}

impl DistanceMetric for SIMDCosineDistance {
    fn distance(&self, a: &VectorData, b: &VectorData) -> Score {
        simd::cosine_distance_simd(a, b)
    }

    fn batch_distance(&self, query: &VectorData, vectors: &[&VectorData]) -> Vec<Score> {
        simd::cosine_distance_batch_simd(query, vectors)
    }

    fn name(&self) -> &'static str {
        "cosine_simd"
    }

    fn is_similarity(&self) -> bool {
        false
    }
}

impl DistanceMetric for SIMDDotProductSimilarity {
    fn distance(&self, a: &VectorData, b: &VectorData) -> Score {
        simd::dot_product_simd(a, b)
    }

    fn batch_distance(&self, query: &VectorData, vectors: &[&VectorData]) -> Vec<Score> {
        simd::dot_product_batch_simd(query, vectors)
    }

    fn name(&self) -> &'static str {
        "dot_product_simd"
    }

    fn is_similarity(&self) -> bool {
        true
    }
}

impl DistanceMetric for SIMDManhattanDistance {
    fn distance(&self, a: &VectorData, b: &VectorData) -> Score {
        // Manhattan distance calculation with SIMD (when available)
        manhattan_distance_simd(a, b)
    }

    fn batch_distance(&self, query: &VectorData, vectors: &[&VectorData]) -> Vec<Score> {
        vectors
            .iter()
            .map(|vector| self.distance(query, vector))
            .collect()
    }

    fn name(&self) -> &'static str {
        "manhattan_simd"
    }

    fn is_similarity(&self) -> bool {
        false
    }
}

/// SIMD-optimized Manhattan distance
fn manhattan_distance_simd(a: &VectorData, b: &VectorData) -> Score {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { manhattan_distance_avx2(a, b) }
        } else {
            manhattan_distance_scalar(a, b)
        }
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        manhattan_distance_scalar(a, b)
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn manhattan_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    debug_assert_eq!(a.len(), b.len());

    let len = a.len();
    let chunks = len / 8;
    let _remainder = len % 8;

    let mut sum_vec = _mm256_setzero_ps();
    let sign_mask = _mm256_set1_ps(-0.0); // 0x80000000 pattern

    // Process 8 elements at a time
    for i in 0..chunks {
        let offset = i * 8;

        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));

        // Calculate difference
        let diff = _mm256_sub_ps(va, vb);

        // Absolute value using bit manipulation
        let abs_diff = _mm256_andnot_ps(sign_mask, diff);

        // Accumulate
        sum_vec = _mm256_add_ps(sum_vec, abs_diff);
    }

    // Horizontal sum
    let sum_array: [f32; 8] = std::mem::transmute(sum_vec);
    let mut sum = sum_array.iter().sum::<f32>();

    // Handle remainder
    let remainder_start = chunks * 8;
    for i in remainder_start..len {
        sum += (a[i] - b[i]).abs();
    }

    sum
}

fn manhattan_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

/// Factory function to create SIMD-optimized distance metrics
pub fn create_simd_distance_metric(name: &str) -> Option<Box<dyn DistanceMetric>> {
    match name {
        "euclidean_simd" => Some(Box::new(SIMDEuclideanDistance)),
        "cosine_simd" => Some(Box::new(SIMDCosineDistance)),
        "dot_product_simd" => Some(Box::new(SIMDDotProductSimilarity)),
        "manhattan_simd" => Some(Box::new(SIMDManhattanDistance)),
        _ => None,
    }
}

/// Check if SIMD instructions are available on current hardware
pub fn simd_features_available() -> Vec<String> {
    let mut features = Vec::new();

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("sse2") {
            features.push("sse2".to_string());
        }
        if is_x86_feature_detected!("avx") {
            features.push("avx".to_string());
        }
        if is_x86_feature_detected!("avx2") {
            features.push("avx2".to_string());
        }
        if is_x86_feature_detected!("fma") {
            features.push("fma".to_string());
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            features.push("neon".to_string());
        }
    }

    features
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_distance_metrics() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];

        // Test SIMD Euclidean
        let simd_euclidean = SIMDEuclideanDistance;
        let euclidean_result = simd_euclidean.distance(&a, &b);
        assert!(euclidean_result > 0.0);

        // Test SIMD Cosine
        let simd_cosine = SIMDCosineDistance;
        let cosine_result = simd_cosine.distance(&a, &b);
        assert!(cosine_result >= 0.0 && cosine_result <= 2.0);

        // Test SIMD Dot Product
        let simd_dot = SIMDDotProductSimilarity;
        let dot_result = simd_dot.distance(&a, &b);
        assert!(dot_result > 0.0);

        // Test SIMD Manhattan
        let simd_manhattan = SIMDManhattanDistance;
        let manhattan_result = simd_manhattan.distance(&a, &b);
        assert!(manhattan_result > 0.0);
    }

    #[test]
    fn test_simd_factory() {
        let euclidean = create_simd_distance_metric("euclidean_simd");
        assert!(euclidean.is_some());

        let invalid = create_simd_distance_metric("invalid");
        assert!(invalid.is_none());
    }

    #[test]
    fn test_simd_features() {
        let features = simd_features_available();
        // Just verify it doesn't crash
        println!("Available SIMD features: {:?}", features);
    }
}
