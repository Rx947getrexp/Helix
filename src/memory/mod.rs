// Memory management module for VecLite
// Provides memory pools and allocation optimization for frequent operations

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// Memory pool for vector allocations
/// Reduces allocation overhead by reusing vector memory
#[derive(Debug)]
pub struct VectorPool {
    pools: Vec<Arc<Mutex<VecDeque<Vec<f32>>>>>,
    max_size_per_pool: usize,
    #[allow(dead_code)]
    max_pools: usize,
}

impl VectorPool {
    /// Create a new vector pool with specified configuration
    pub fn new(max_size_per_pool: usize, max_pools: usize) -> Self {
        let pools = (0..32) // Support up to 32 different dimension sizes
            .map(|_| Arc::new(Mutex::new(VecDeque::new())))
            .collect();

        Self {
            pools,
            max_size_per_pool,
            max_pools,
        }
    }

    /// Get a vector with the specified capacity, reusing from pool if available
    pub fn get_vector(&self, dimensions: usize) -> Vec<f32> {
        let pool_index = self.dimension_to_pool_index(dimensions);

        if let Some(pool) = self.pools.get(pool_index) {
            if let Ok(mut pool_guard) = pool.lock() {
                if let Some(mut vector) = pool_guard.pop_front() {
                    vector.clear();
                    vector.reserve(dimensions);
                    return vector;
                }
            }
        }

        // Allocate new vector if pool is empty or unavailable
        Vec::with_capacity(dimensions)
    }

    /// Return a vector to the pool for reuse
    pub fn return_vector(&self, mut vector: Vec<f32>) {
        let dimensions = vector.capacity();
        let pool_index = self.dimension_to_pool_index(dimensions);

        if let Some(pool) = self.pools.get(pool_index) {
            if let Ok(mut pool_guard) = pool.lock() {
                if pool_guard.len() < self.max_size_per_pool {
                    vector.clear();
                    pool_guard.push_back(vector);
                }
                // If pool is full, let vector drop naturally
            }
        }
    }

    /// Map dimensions to pool index for similar-sized vectors
    fn dimension_to_pool_index(&self, dimensions: usize) -> usize {
        match dimensions {
            0..=16 => 0,
            17..=32 => 1,
            33..=64 => 2,
            65..=128 => 3,
            129..=256 => 4,
            257..=512 => 5,
            513..=1024 => 6,
            1025..=2048 => 7,
            _ => 8, // Large vectors
        }
    }

    /// Get pool statistics for monitoring
    pub fn statistics(&self) -> PoolStatistics {
        let mut stats = PoolStatistics::default();

        for (i, pool) in self.pools.iter().enumerate() {
            if let Ok(pool_guard) = pool.lock() {
                stats.pool_sizes.push((i, pool_guard.len()));
                stats.total_pooled_vectors += pool_guard.len();
            }
        }

        stats
    }

    /// Clear all pools and release memory
    pub fn clear(&self) {
        for pool in &self.pools {
            if let Ok(mut pool_guard) = pool.lock() {
                pool_guard.clear();
            }
        }
    }
}

/// Statistics about memory pool usage
#[derive(Debug, Default)]
pub struct PoolStatistics {
    pub total_pooled_vectors: usize,
    pub pool_sizes: Vec<(usize, usize)>, // (pool_index, size)
}

lazy_static::lazy_static! {
    pub static ref GLOBAL_VECTOR_POOL: VectorPool = VectorPool::new(100, 1000);
}

/// Convenient functions for using the global pool
pub fn get_pooled_vector(dimensions: usize) -> Vec<f32> {
    GLOBAL_VECTOR_POOL.get_vector(dimensions)
}

pub fn return_pooled_vector(vector: Vec<f32>) {
    GLOBAL_VECTOR_POOL.return_vector(vector);
}

/// Memory pool for search result allocations
#[derive(Debug)]
pub struct SearchResultPool {
    pool: Arc<Mutex<VecDeque<Vec<crate::types::SearchResult>>>>,
    max_size: usize,
}

impl SearchResultPool {
    pub fn new(max_size: usize) -> Self {
        Self {
            pool: Arc::new(Mutex::new(VecDeque::new())),
            max_size,
        }
    }

    pub fn get_result_vector(&self, capacity: usize) -> Vec<crate::types::SearchResult> {
        if let Ok(mut pool_guard) = self.pool.lock() {
            if let Some(mut vector) = pool_guard.pop_front() {
                vector.clear();
                vector.reserve(capacity);
                return vector;
            }
        }

        Vec::with_capacity(capacity)
    }

    pub fn return_result_vector(&self, mut vector: Vec<crate::types::SearchResult>) {
        if let Ok(mut pool_guard) = self.pool.lock() {
            if pool_guard.len() < self.max_size {
                vector.clear();
                pool_guard.push_back(vector);
            }
        }
    }
}

lazy_static::lazy_static! {
    pub static ref GLOBAL_RESULT_POOL: SearchResultPool = SearchResultPool::new(50);
}

pub fn get_pooled_results(capacity: usize) -> Vec<crate::types::SearchResult> {
    GLOBAL_RESULT_POOL.get_result_vector(capacity)
}

pub fn return_pooled_results(vector: Vec<crate::types::SearchResult>) {
    GLOBAL_RESULT_POOL.return_result_vector(vector);
}

/// Memory-efficient batch operation utilities
pub struct BatchAllocator {
    vector_pool: VectorPool,
    #[allow(dead_code)]
    batch_size: usize,
}

impl BatchAllocator {
    pub fn new(batch_size: usize) -> Self {
        Self {
            vector_pool: VectorPool::new(50, 200),
            batch_size,
        }
    }

    /// Pre-allocate vectors for batch operations
    pub fn allocate_batch(&self, dimensions: usize, count: usize) -> Vec<Vec<f32>> {
        let mut batch = Vec::with_capacity(count);
        for _ in 0..count {
            batch.push(self.vector_pool.get_vector(dimensions));
        }
        batch
    }

    /// Return batch of vectors to pool
    pub fn return_batch(&self, batch: Vec<Vec<f32>>) {
        for vector in batch {
            self.vector_pool.return_vector(vector);
        }
    }
}

/// Memory monitoring and statistics with configuration support
#[derive(Debug)]
pub struct MemoryMonitor {
    pub allocated_vectors: std::sync::atomic::AtomicUsize,
    pub peak_memory_usage: std::sync::atomic::AtomicUsize,
    pub total_allocations: std::sync::atomic::AtomicU64,
    pub total_deallocations: std::sync::atomic::AtomicU64,
    config: std::sync::Arc<std::sync::RwLock<crate::types::MemoryConfig>>,
    last_cleanup: std::sync::atomic::AtomicU64,
}

impl MemoryMonitor {
    pub fn new() -> Self {
        Self {
            allocated_vectors: std::sync::atomic::AtomicUsize::new(0),
            peak_memory_usage: std::sync::atomic::AtomicUsize::new(0),
            total_allocations: std::sync::atomic::AtomicU64::new(0),
            total_deallocations: std::sync::atomic::AtomicU64::new(0),
            config: std::sync::Arc::new(std::sync::RwLock::new(
                crate::types::MemoryConfig::default(),
            )),
            last_cleanup: std::sync::atomic::AtomicU64::new(0),
        }
    }

    pub fn with_config(config: crate::types::MemoryConfig) -> Self {
        Self {
            allocated_vectors: std::sync::atomic::AtomicUsize::new(0),
            peak_memory_usage: std::sync::atomic::AtomicUsize::new(0),
            total_allocations: std::sync::atomic::AtomicU64::new(0),
            total_deallocations: std::sync::atomic::AtomicU64::new(0),
            config: std::sync::Arc::new(std::sync::RwLock::new(config)),
            last_cleanup: std::sync::atomic::AtomicU64::new(0),
        }
    }

    pub fn record_allocation(&self, size: usize) {
        self.allocated_vectors
            .fetch_add(size, std::sync::atomic::Ordering::Relaxed);
        self.total_allocations
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Update peak if necessary
        let current = self
            .allocated_vectors
            .load(std::sync::atomic::Ordering::Relaxed);
        let mut peak = self
            .peak_memory_usage
            .load(std::sync::atomic::Ordering::Relaxed);

        while current > peak {
            match self.peak_memory_usage.compare_exchange_weak(
                peak,
                current,
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => peak = x,
            }
        }

        // Check memory limits and trigger cleanup if needed
        self.check_memory_limits(current);
    }

    pub fn record_deallocation(&self, size: usize) {
        self.allocated_vectors
            .fetch_sub(size, std::sync::atomic::Ordering::Relaxed);
        self.total_deallocations
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn get_stats(&self) -> MemoryStats {
        MemoryStats {
            current_allocated: self
                .allocated_vectors
                .load(std::sync::atomic::Ordering::Relaxed),
            peak_allocated: self
                .peak_memory_usage
                .load(std::sync::atomic::Ordering::Relaxed),
            total_allocations: self
                .total_allocations
                .load(std::sync::atomic::Ordering::Relaxed),
            total_deallocations: self
                .total_deallocations
                .load(std::sync::atomic::Ordering::Relaxed),
        }
    }

    /// Check memory limits and trigger cleanup if necessary
    fn check_memory_limits(&self, current_memory: usize) {
        if let Ok(config) = self.config.read() {
            if !config.enable_monitoring {
                return;
            }

            let max_memory = config.max_memory_bytes;
            let warning_threshold =
                (max_memory as f64 * config.warning_threshold_percent as f64 / 100.0) as usize;
            let cleanup_threshold =
                (max_memory as f64 * config.cleanup_threshold_percent as f64 / 100.0) as usize;

            if current_memory > cleanup_threshold && config.enable_auto_cleanup {
                self.trigger_cleanup();
            } else if current_memory > warning_threshold {
                tracing::warn!(
                    "Memory usage warning: {} bytes ({:.1}% of limit)",
                    current_memory,
                    current_memory as f64 / max_memory as f64 * 100.0
                );
            }
        }
    }

    /// Trigger memory cleanup operations
    fn trigger_cleanup(&self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let last_cleanup = self.last_cleanup.load(std::sync::atomic::Ordering::Relaxed);

        if let Ok(config) = self.config.read() {
            if now - last_cleanup < config.cleanup_interval_seconds {
                return; // Too soon since last cleanup
            }
        }

        if self
            .last_cleanup
            .compare_exchange(
                last_cleanup,
                now,
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
            )
            .is_ok()
        {
            // Clear memory pools to free up memory
            GLOBAL_VECTOR_POOL.clear();
            GLOBAL_RESULT_POOL.pool.lock().unwrap().clear();

            tracing::info!("Memory cleanup completed");
        }
    }

    /// Update memory configuration
    pub fn update_config(&self, new_config: crate::types::MemoryConfig) {
        if let Ok(mut config) = self.config.write() {
            *config = new_config;
        }
    }

    /// Get current memory configuration
    pub fn get_config(&self) -> crate::types::MemoryConfig {
        self.config.read().unwrap().clone()
    }

    /// Check if memory usage is within limits
    pub fn is_within_limits(&self) -> bool {
        let current = self
            .allocated_vectors
            .load(std::sync::atomic::Ordering::Relaxed);
        if let Ok(config) = self.config.read() {
            current <= config.max_memory_bytes
        } else {
            true
        }
    }
}

#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub current_allocated: usize,
    pub peak_allocated: usize,
    pub total_allocations: u64,
    pub total_deallocations: u64,
}

impl Default for MemoryMonitor {
    fn default() -> Self {
        Self::new()
    }
}

lazy_static::lazy_static! {
    pub static ref GLOBAL_MEMORY_MONITOR: MemoryMonitor = MemoryMonitor::new();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_pool_basic_operations() {
        let pool = VectorPool::new(10, 100);

        // Get and return vectors
        let vec1 = pool.get_vector(128);
        assert_eq!(vec1.capacity(), 128);

        let vec2 = pool.get_vector(256);
        assert_eq!(vec2.capacity(), 256);

        pool.return_vector(vec1);
        pool.return_vector(vec2);

        // Check statistics
        let stats = pool.statistics();
        assert!(stats.total_pooled_vectors > 0);
    }

    #[test]
    fn test_pool_dimension_mapping() {
        let pool = VectorPool::new(10, 100);

        assert_eq!(pool.dimension_to_pool_index(16), 0);
        assert_eq!(pool.dimension_to_pool_index(32), 1);
        assert_eq!(pool.dimension_to_pool_index(64), 2);
        assert_eq!(pool.dimension_to_pool_index(128), 3);
        assert_eq!(pool.dimension_to_pool_index(256), 4);
    }

    #[test]
    fn test_global_pools() {
        let vec1 = get_pooled_vector(128);
        assert_eq!(vec1.capacity(), 128);

        return_pooled_vector(vec1);

        // Test result pool - capacity should be at least what was requested
        // (may be larger if vector was reused from pool)
        let results = get_pooled_results(10);
        assert!(results.capacity() >= 10);

        return_pooled_results(results);
    }

    #[test]
    fn test_memory_monitor() {
        let monitor = MemoryMonitor::new();

        monitor.record_allocation(1000);
        monitor.record_allocation(500);

        let stats = monitor.get_stats();
        assert_eq!(stats.current_allocated, 1500);
        assert_eq!(stats.peak_allocated, 1500);
        assert_eq!(stats.total_allocations, 2);

        monitor.record_deallocation(500);
        let stats = monitor.get_stats();
        assert_eq!(stats.current_allocated, 1000);
        assert_eq!(stats.peak_allocated, 1500); // Peak remains
    }

    #[test]
    fn test_batch_allocator() {
        let allocator = BatchAllocator::new(10);

        let batch = allocator.allocate_batch(128, 5);
        assert_eq!(batch.len(), 5);

        for vector in &batch {
            assert_eq!(vector.capacity(), 128);
        }

        allocator.return_batch(batch);
    }
}
