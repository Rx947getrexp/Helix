// SIMD-optimized distance calculations for VecLite
// This module provides vectorized implementations of distance metrics
// using platform-specific SIMD instructions for maximum performance

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86")]
use std::arch::x86::*;

use crate::types::{Score, VectorData};

/// SIMD-optimized Euclidean distance calculation
/// Uses AVX2 instructions when available, falling back to SSE2 or scalar
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn euclidean_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let len = a.len();
    let chunks = len / 8; // AVX2 processes 8 f32 at a time
    let _remainder = len % 8;

    let mut sum_vec = _mm256_setzero_ps();

    // Process 8 elements at a time with AVX2
    for i in 0..chunks {
        let offset = i * 8;

        // Load vectors
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));

        // Calculate difference
        let diff = _mm256_sub_ps(va, vb);

        // Square and accumulate
        let squared = _mm256_mul_ps(diff, diff);
        sum_vec = _mm256_add_ps(sum_vec, squared);
    }

    // Horizontal sum of the vector
    let sum_array: [f32; 8] = std::mem::transmute(sum_vec);
    let mut sum = sum_array.iter().sum::<f32>();

    // Handle remaining elements
    let remainder_start = chunks * 8;
    for i in remainder_start..len {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }

    sum.sqrt()
}

/// SIMD-optimized Euclidean distance with SSE2 fallback
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn euclidean_distance_sse2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let len = a.len();
    let chunks = len / 4; // SSE2 processes 4 f32 at a time
    let _remainder = len % 4;

    let mut sum_vec = _mm_setzero_ps();

    // Process 4 elements at a time with SSE2
    for i in 0..chunks {
        let offset = i * 4;

        // Load vectors
        let va = _mm_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm_loadu_ps(b.as_ptr().add(offset));

        // Calculate difference
        let diff = _mm_sub_ps(va, vb);

        // Square and accumulate
        let squared = _mm_mul_ps(diff, diff);
        sum_vec = _mm_add_ps(sum_vec, squared);
    }

    // Horizontal sum of the vector
    let sum_array: [f32; 4] = std::mem::transmute(sum_vec);
    let mut sum = sum_array.iter().sum::<f32>();

    // Handle remaining elements
    let remainder_start = chunks * 4;
    for i in remainder_start..len {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }

    sum.sqrt()
}

/// SIMD-optimized dot product calculation
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let len = a.len();
    let chunks = len / 8;
    let _remainder = len % 8;

    let mut sum_vec = _mm256_setzero_ps();

    // Process 8 elements at a time
    for i in 0..chunks {
        let offset = i * 8;

        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));

        let product = _mm256_mul_ps(va, vb);
        sum_vec = _mm256_add_ps(sum_vec, product);
    }

    // Horizontal sum
    let sum_array: [f32; 8] = std::mem::transmute(sum_vec);
    let mut sum = sum_array.iter().sum::<f32>();

    // Handle remainder
    let remainder_start = chunks * 8;
    for i in remainder_start..len {
        sum += a[i] * b[i];
    }

    sum
}

/// SIMD-optimized cosine distance calculation
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn cosine_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let len = a.len();
    let chunks = len / 8;
    let _remainder = len % 8;

    let mut dot_vec = _mm256_setzero_ps();
    let mut norm_a_vec = _mm256_setzero_ps();
    let mut norm_b_vec = _mm256_setzero_ps();

    // Process 8 elements at a time
    for i in 0..chunks {
        let offset = i * 8;

        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));

        // Dot product
        let product = _mm256_mul_ps(va, vb);
        dot_vec = _mm256_add_ps(dot_vec, product);

        // Norms
        let a_squared = _mm256_mul_ps(va, va);
        let b_squared = _mm256_mul_ps(vb, vb);
        norm_a_vec = _mm256_add_ps(norm_a_vec, a_squared);
        norm_b_vec = _mm256_add_ps(norm_b_vec, b_squared);
    }

    // Horizontal sums
    let dot_array: [f32; 8] = std::mem::transmute(dot_vec);
    let norm_a_array: [f32; 8] = std::mem::transmute(norm_a_vec);
    let norm_b_array: [f32; 8] = std::mem::transmute(norm_b_vec);

    let mut dot = dot_array.iter().sum::<f32>();
    let mut norm_a = norm_a_array.iter().sum::<f32>();
    let mut norm_b = norm_b_array.iter().sum::<f32>();

    // Handle remainder
    let remainder_start = chunks * 8;
    for i in remainder_start..len {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    // Handle zero vectors
    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0;
    }

    let cosine_similarity = dot / (norm_a.sqrt() * norm_b.sqrt());
    1.0 - cosine_similarity
}

/// Public interface for SIMD-optimized Euclidean distance
pub fn euclidean_distance_simd(a: &VectorData, b: &VectorData) -> Score {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 function is safe when feature is detected and vectors are validated
            unsafe { euclidean_distance_avx2(a, b) }
        } else if is_x86_feature_detected!("sse2") {
            // SAFETY: SSE2 function is safe when feature is detected and vectors are validated
            unsafe { euclidean_distance_sse2(a, b) }
        } else {
            euclidean_distance_scalar(a, b)
        }
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        euclidean_distance_scalar(a, b)
    }
}

/// Public interface for SIMD-optimized dot product
pub fn dot_product_simd(a: &VectorData, b: &VectorData) -> Score {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 function is safe when feature is detected and vectors are validated
            unsafe { dot_product_avx2(a, b) }
        } else {
            dot_product_scalar(a, b)
        }
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        dot_product_scalar(a, b)
    }
}

/// Public interface for SIMD-optimized cosine distance
pub fn cosine_distance_simd(a: &VectorData, b: &VectorData) -> Score {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 function is safe when feature is detected and vectors are validated
            unsafe { cosine_distance_avx2(a, b) }
        } else {
            cosine_distance_scalar(a, b)
        }
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        cosine_distance_scalar(a, b)
    }
}

/// Batch SIMD operations for multiple vectors
pub fn euclidean_distance_batch_simd(query: &VectorData, vectors: &[&VectorData]) -> Vec<Score> {
    vectors
        .iter()
        .map(|vector| euclidean_distance_simd(query, vector))
        .collect()
}

pub fn dot_product_batch_simd(query: &VectorData, vectors: &[&VectorData]) -> Vec<Score> {
    vectors
        .iter()
        .map(|vector| dot_product_simd(query, vector))
        .collect()
}

pub fn cosine_distance_batch_simd(query: &VectorData, vectors: &[&VectorData]) -> Vec<Score> {
    // Pre-compute query norm for efficiency
    let query_norm_squared = query.iter().map(|x| x * x).sum::<f32>();

    if query_norm_squared == 0.0 {
        return vec![1.0; vectors.len()];
    }

    let query_norm = query_norm_squared.sqrt();

    vectors
        .iter()
        .map(|vector| {
            let dot = dot_product_simd(query, vector);
            let vector_norm_squared = vector.iter().map(|x| x * x).sum::<f32>();

            if vector_norm_squared == 0.0 {
                1.0
            } else {
                let vector_norm = vector_norm_squared.sqrt();
                let cosine_similarity = dot / (query_norm * vector_norm);
                1.0 - cosine_similarity
            }
        })
        .collect()
}

// Fallback scalar implementations
fn euclidean_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let diff = x - y;
        sum += diff * diff;
    }
    sum.sqrt()
}

fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn cosine_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut norm_a, mut norm_b) = (0.0f32, 0.0f32, 0.0f32);

    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0;
    }

    let cosine_similarity = dot / (norm_a.sqrt() * norm_b.sqrt());
    1.0 - cosine_similarity
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_euclidean_accuracy() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let simd_result = euclidean_distance_simd(&a, &b);
        let scalar_result = euclidean_distance_scalar(&a, &b);

        assert!((simd_result - scalar_result).abs() < 1e-6);
    }

    #[test]
    fn test_simd_dot_product_accuracy() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];

        let simd_result = dot_product_simd(&a, &b);
        let scalar_result = dot_product_scalar(&a, &b);

        assert!((simd_result - scalar_result).abs() < 1e-6);
    }

    #[test]
    fn test_simd_cosine_accuracy() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];

        let simd_result = cosine_distance_simd(&a, &b);
        let scalar_result = cosine_distance_scalar(&a, &b);

        assert!((simd_result - scalar_result).abs() < 1e-6);
    }

    #[test]
    fn test_batch_operations() {
        let query = vec![1.0, 2.0, 3.0, 4.0];
        let vectors = vec![
            vec![2.0, 3.0, 4.0, 5.0],
            vec![3.0, 4.0, 5.0, 6.0],
            vec![1.0, 1.0, 1.0, 1.0],
        ];
        let vector_refs: Vec<&Vec<f32>> = vectors.iter().collect();

        let batch_results = euclidean_distance_batch_simd(&query, &vector_refs);

        assert_eq!(batch_results.len(), 3);
        for result in batch_results {
            assert!(result > 0.0);
        }
    }
}
