# VecLite - Technical Design Document (TDD)

## 1. Technical Overview

VecLite is implemented as a high-performance vector search library using Rust for the core engine with Go FFI bindings for cross-language support. The architecture emphasizes memory efficiency, algorithmic performance, and maintainability while providing a simple embedded database experience similar to SQLite.

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            Client Applications                          │
├─────────────────────────────┬───────────────────────────────────────────┤
│         Rust Native API     │              Go FFI API                  │
├─────────────────────────────┴───────────────────────────────────────────┤
│                           VecLite Core Engine                          │
├─────────────────────────────────────────────────────────────────────────┤
│  Query Engine  │  Index Manager  │  Storage Manager  │  Persistence    │
├────────────────┼─────────────────┼──────────────────┼─────────────────┤
│  • Distance    │  • HNSW Index   │  • Vector Store  │  • VLT Format   │
│    Metrics     │  • VP-Tree      │  • Metadata      │  • Serialization│
│  • KNN Search  │  • Index Stats  │  • Memory Mgmt   │  • Compression  │
│  • Batch Ops   │  • Dynamic Upd  │  • Thread Safety │  • Versioning   │
└────────────────┴─────────────────┴──────────────────┴─────────────────┘
```

## 2. Core Component Design

### 2.1 Storage Manager

#### Data Structures

```rust
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

#[derive(Debug, Clone)]
pub struct VectorItem {
    pub id: String,
    pub vector: Vec<f32>,
    pub metadata: HashMap<String, String>,
    pub timestamp: u64,
}

#[derive(Debug)]
pub struct StorageManager {
    vectors: Arc<RwLock<HashMap<String, VectorItem>>>,
    config: StorageConfig,
    stats: StorageStats,
}

#[derive(Debug, Clone)]
pub struct StorageConfig {
    pub max_vectors: usize,
    pub max_vector_dim: usize,
    pub memory_limit_bytes: usize,
    pub enable_checksums: bool,
}
```

#### Implementation Details

**Vector Storage Strategy**:
- Primary storage in `HashMap<String, VectorItem>` for O(1) access by ID
- Memory allocation using arena-based approach for cache locality
- Copy-on-write semantics for thread safety
- Configurable memory limits with overflow handling

**Thread Safety**:
- Read-write locks (`RwLock`) for concurrent access
- Lock-free reads for query-heavy workloads
- Write batching to minimize lock contention
- Dead-lock prevention through lock ordering

**Memory Management**:
```rust
impl StorageManager {
    pub fn insert(&self, item: VectorItem) -> Result<(), VecLiteError> {
        // Check memory limits
        if self.estimate_memory_usage() + item.memory_size() > self.config.memory_limit_bytes {
            return Err(VecLiteError::MemoryLimitExceeded);
        }

        // Validate vector dimensions
        if item.vector.len() > self.config.max_vector_dim {
            return Err(VecLiteError::InvalidDimensions);
        }

        // Insert with write lock
        let mut vectors = self.vectors.write().unwrap();
        vectors.insert(item.id.clone(), item);

        self.update_stats();
        Ok(())
    }

    pub fn get(&self, id: &str) -> Option<VectorItem> {
        let vectors = self.vectors.read().unwrap();
        vectors.get(id).cloned()
    }

    pub fn delete(&self, id: &str) -> Result<bool, VecLiteError> {
        let mut vectors = self.vectors.write().unwrap();
        Ok(vectors.remove(id).is_some())
    }
}
```

### 2.2 Index Manager

#### HNSW Implementation

```rust
#[derive(Debug)]
pub struct HNSWIndex {
    layers: Vec<Layer>,
    entry_point: Option<NodeId>,
    m: usize,              // Maximum connections per node
    max_m: usize,          // Maximum connections for layer 0
    max_m_l: usize,        // Maximum connections for upper layers
    ml: f64,               // Level generation factor
    ef_construction: usize, // Size of dynamic candidate list
}

#[derive(Debug)]
struct Layer {
    graph: HashMap<NodeId, Node>,
}

#[derive(Debug)]
struct Node {
    id: String,
    vector: Vec<f32>,
    connections: Vec<NodeId>,
    level: usize,
}

type NodeId = u64;
```

**HNSW Algorithm Implementation**:

```rust
impl HNSWIndex {
    pub fn insert(&mut self, id: String, vector: Vec<f32>) -> Result<(), VecLiteError> {
        let level = self.generate_level();
        let node_id = self.next_node_id();

        // Create node
        let node = Node {
            id: id.clone(),
            vector: vector.clone(),
            connections: Vec::new(),
            level,
        };

        // Insert into appropriate layers
        for lev in 0..=level {
            self.ensure_layer(lev);

            if lev == 0 {
                // Layer 0: more connections for recall
                self.insert_at_layer(&node, lev, self.max_m)?;
            } else {
                // Upper layers: fewer connections for speed
                self.insert_at_layer(&node, lev, self.max_m_l)?;
            }
        }

        // Update entry point if necessary
        if self.entry_point.is_none() || level > self.get_entry_level() {
            self.entry_point = Some(node_id);
        }

        Ok(())
    }

    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<SearchResult>, VecLiteError> {
        if self.entry_point.is_none() {
            return Ok(Vec::new());
        }

        let entry_point = self.entry_point.unwrap();
        let mut current_closest = entry_point;

        // Search from top layer down to layer 1
        for layer in (1..self.layers.len()).rev() {
            current_closest = self.search_layer(query, 1, layer, current_closest)?[0];
        }

        // Search layer 0 with ef
        let candidates = self.search_layer(query, ef, 0, current_closest)?;

        // Convert to search results and take top k
        let mut results = Vec::new();
        for (distance, node_id) in candidates.into_iter().take(k) {
            if let Some(node) = self.get_node(node_id, 0) {
                results.push(SearchResult {
                    id: node.id.clone(),
                    score: distance,
                    metadata: HashMap::new(), // Filled by storage manager
                });
            }
        }

        Ok(results)
    }

    fn search_layer(&self, query: &[f32], ef: usize, layer: usize, entry_point: NodeId)
                   -> Result<Vec<(f32, NodeId)>, VecLiteError> {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new(); // Max heap for candidates
        let mut w = BinaryHeap::new();          // Min heap for results

        let entry_distance = self.distance(query, &self.get_node(entry_point, layer)?.vector);
        candidates.push(Reverse((OrderedFloat(entry_distance), entry_point)));
        w.push((OrderedFloat(entry_distance), entry_point));
        visited.insert(entry_point);

        while let Some(Reverse((current_dist, current))) = candidates.pop() {
            if current_dist.0 > w.peek().unwrap().0 .0 {
                break;
            }

            if let Some(node) = self.get_node(current, layer) {
                for &neighbor in &node.connections {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);

                        let distance = self.distance(query, &self.get_node(neighbor, layer)?.vector);

                        if distance < w.peek().unwrap().0 .0 || w.len() < ef {
                            candidates.push(Reverse((OrderedFloat(distance), neighbor)));
                            w.push((OrderedFloat(distance), neighbor));

                            if w.len() > ef {
                                w.pop();
                            }
                        }
                    }
                }
            }
        }

        Ok(w.into_sorted_vec().into_iter().map(|(d, n)| (d.0, n)).collect())
    }
}
```

#### Distance Metrics Implementation

```rust
pub trait DistanceMetric: Send + Sync {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32;
    fn name(&self) -> &'static str;
}

pub struct EuclideanDistance;
impl DistanceMetric for EuclideanDistance {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());

        let mut sum = 0.0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            let diff = x - y;
            sum += diff * diff;
        }
        sum.sqrt()
    }

    fn name(&self) -> &'static str { "euclidean" }
}

pub struct CosineDistance;
impl DistanceMetric for CosineDistance {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());

        let (mut dot, mut norm_a, mut norm_b) = (0.0f32, 0.0f32, 0.0f32);

        for (x, y) in a.iter().zip(b.iter()) {
            dot += x * y;
            norm_a += x * x;
            norm_b += y * y;
        }

        if norm_a == 0.0 || norm_b == 0.0 {
            return f32::INFINITY;
        }

        1.0 - (dot / (norm_a.sqrt() * norm_b.sqrt()))
    }

    fn name(&self) -> &'static str { "cosine" }
}

// SIMD-optimized versions for supported platforms
#[cfg(target_feature = "avx2")]
impl EuclideanDistance {
    fn distance_simd(&self, a: &[f32], b: &[f32]) -> f32 {
        // AVX2 implementation for 8 floats at a time
        unsafe {
            use std::arch::x86_64::*;

            let mut sum = _mm256_setzero_ps();
            let chunks = a.len() / 8;

            for i in 0..chunks {
                let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
                let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
                let diff = _mm256_sub_ps(va, vb);
                sum = _mm256_fmadd_ps(diff, diff, sum);
            }

            // Handle remainder and reduce
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
```

### 2.3 Query Engine

```rust
pub struct QueryEngine {
    storage: Arc<StorageManager>,
    index: Arc<RwLock<Box<dyn SearchIndex>>>,
    distance_metric: Box<dyn DistanceMetric>,
    config: QueryConfig,
}

#[derive(Debug, Clone)]
pub struct QueryConfig {
    pub default_k: usize,
    pub max_k: usize,
    pub ef_search: usize,
    pub enable_metadata_filter: bool,
}

impl QueryEngine {
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>, VecLiteError> {
        // Validate inputs
        if k > self.config.max_k {
            return Err(VecLiteError::InvalidK(k));
        }

        // Search using index
        let index = self.index.read().unwrap();
        let mut results = index.search(query, k, self.config.ef_search)?;

        // Enrich with metadata from storage
        for result in &mut results {
            if let Some(item) = self.storage.get(&result.id) {
                result.metadata = item.metadata;
            }
        }

        Ok(results)
    }

    pub fn search_with_filter<F>(&self, query: &[f32], k: usize, filter: F)
                                -> Result<Vec<SearchResult>, VecLiteError>
    where F: Fn(&HashMap<String, String>) -> bool {
        // Get more candidates to account for filtering
        let expanded_k = k * 3; // Heuristic expansion
        let raw_results = self.search(query, expanded_k)?;

        // Apply filter and take top k
        let filtered_results: Vec<_> = raw_results
            .into_iter()
            .filter(|result| filter(&result.metadata))
            .take(k)
            .collect();

        Ok(filtered_results)
    }

    pub fn batch_search(&self, queries: &[Vec<f32>], k: usize) -> Result<Vec<Vec<SearchResult>>, VecLiteError> {
        // Parallel processing for batch queries
        use rayon::prelude::*;

        queries.par_iter()
               .map(|query| self.search(query, k))
               .collect()
    }
}
```

### 2.4 Persistence Layer

#### VLT File Format Implementation

```rust
#[repr(C)]
#[derive(Debug, Clone)]
pub struct VLTHeader {
    pub magic: [u8; 4],        // "VLT1"
    pub version: u32,           // Format version
    pub flags: u32,             // Feature flags
    pub checksum: u64,          // Header checksum
    pub metadata_offset: u64,   // Offset to metadata section
    pub metadata_size: u64,     // Size of metadata section
    pub vector_offset: u64,     // Offset to vector section
    pub vector_size: u64,       // Size of vector section
    pub index_offset: u64,      // Offset to index section
    pub index_size: u64,        // Size of index section
    pub reserved: [u8; 32],     // Reserved for future use
}

pub struct VLTPersistence {
    compression: CompressionType,
    checksum_enabled: bool,
}

impl VLTPersistence {
    pub fn save(&self, db: &VecLite, path: &Path) -> Result<(), VecLiteError> {
        let mut file = BufWriter::new(File::create(path)?);

        // Write header (placeholder, will update after writing sections)
        let mut header = VLTHeader::default();
        header.magic = *b"VLT1";
        header.version = VLT_VERSION;
        file.write_all(unsafe {
            std::slice::from_raw_parts(
                &header as *const _ as *const u8,
                std::mem::size_of::<VLTHeader>()
            )
        })?;

        // Write metadata section
        header.metadata_offset = file.stream_position()?;
        let metadata = self.serialize_metadata(db)?;
        let compressed_metadata = self.compress(&metadata)?;
        file.write_all(&compressed_metadata)?;
        header.metadata_size = compressed_metadata.len() as u64;

        // Write vector section
        header.vector_offset = file.stream_position()?;
        let vectors = self.serialize_vectors(db)?;
        let compressed_vectors = self.compress(&vectors)?;
        file.write_all(&compressed_vectors)?;
        header.vector_size = compressed_vectors.len() as u64;

        // Write index section
        header.index_offset = file.stream_position()?;
        let index_data = self.serialize_index(db)?;
        let compressed_index = self.compress(&index_data)?;
        file.write_all(&compressed_index)?;
        header.index_size = compressed_index.len() as u64;

        // Update header with correct offsets
        if self.checksum_enabled {
            header.checksum = self.calculate_checksum(&header)?;
        }

        file.seek(SeekFrom::Start(0))?;
        file.write_all(unsafe {
            std::slice::from_raw_parts(
                &header as *const _ as *const u8,
                std::mem::size_of::<VLTHeader>()
            )
        })?;

        file.flush()?;
        Ok(())
    }

    pub fn load(&self, path: &Path) -> Result<VecLite, VecLiteError> {
        let mut file = BufReader::new(File::open(path)?);

        // Read and validate header
        let mut header_bytes = vec![0u8; std::mem::size_of::<VLTHeader>()];
        file.read_exact(&mut header_bytes)?;

        let header: VLTHeader = unsafe {
            std::ptr::read(header_bytes.as_ptr() as *const VLTHeader)
        };

        if header.magic != *b"VLT1" {
            return Err(VecLiteError::InvalidFileFormat);
        }

        if self.checksum_enabled && !self.verify_checksum(&header)? {
            return Err(VecLiteError::ChecksumMismatch);
        }

        // Load sections
        let metadata = self.load_section(&mut file, header.metadata_offset, header.metadata_size)?;
        let vectors = self.load_section(&mut file, header.vector_offset, header.vector_size)?;
        let index_data = self.load_section(&mut file, header.index_offset, header.index_size)?;

        // Reconstruct database
        let config = self.deserialize_metadata(&metadata)?;
        let mut db = VecLite::with_config(config)?;

        self.deserialize_vectors(&vectors, &mut db)?;
        self.deserialize_index(&index_data, &mut db)?;

        Ok(db)
    }

    fn compress(&self, data: &[u8]) -> Result<Vec<u8>, VecLiteError> {
        match self.compression {
            CompressionType::None => Ok(data.to_vec()),
            CompressionType::Zstd => {
                let mut encoder = zstd::Encoder::new(Vec::new(), 3)?;
                encoder.write_all(data)?;
                Ok(encoder.finish()?)
            }
        }
    }
}

#[derive(Debug, Clone)]
enum CompressionType {
    None,
    Zstd,
}
```

### 2.5 Go FFI Interface

#### C-Compatible Interface

```rust
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float, c_int, c_void};

#[repr(C)]
pub struct CVecLite {
    inner: Box<VecLite>,
}

#[repr(C)]
pub struct CSearchResult {
    id: *mut c_char,
    score: c_float,
    metadata_json: *mut c_char,
}

#[repr(C)]
pub struct CSearchResults {
    results: *mut CSearchResult,
    count: c_int,
}

#[no_mangle]
pub extern "C" fn veclite_new() -> *mut CVecLite {
    match VecLite::new() {
        Ok(db) => Box::into_raw(Box::new(CVecLite {
            inner: Box::new(db),
        })),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn veclite_insert(
    db: *mut CVecLite,
    id: *const c_char,
    vector: *const c_float,
    vector_len: c_int,
    metadata_json: *const c_char,
) -> c_int {
    if db.is_null() || id.is_null() || vector.is_null() {
        return -1;
    }

    let db = unsafe { &mut *db };
    let id = unsafe { CStr::from_ptr(id) }.to_string_lossy().into_owned();
    let vector = unsafe { std::slice::from_raw_parts(vector, vector_len as usize) }.to_vec();

    let metadata = if metadata_json.is_null() {
        HashMap::new()
    } else {
        let json_str = unsafe { CStr::from_ptr(metadata_json) }.to_string_lossy();
        serde_json::from_str(&json_str).unwrap_or_default()
    };

    match db.inner.insert(id, vector, metadata) {
        Ok(_) => 0,
        Err(_) => -1,
    }
}

#[no_mangle]
pub extern "C" fn veclite_search(
    db: *mut CVecLite,
    query: *const c_float,
    query_len: c_int,
    k: c_int,
) -> *mut CSearchResults {
    if db.is_null() || query.is_null() {
        return std::ptr::null_mut();
    }

    let db = unsafe { &*db };
    let query = unsafe { std::slice::from_raw_parts(query, query_len as usize) };

    match db.inner.search(query, k as usize) {
        Ok(results) => {
            let c_results: Vec<CSearchResult> = results.into_iter().map(|r| {
                CSearchResult {
                    id: CString::new(r.id).unwrap().into_raw(),
                    score: r.score,
                    metadata_json: CString::new(serde_json::to_string(&r.metadata).unwrap_or_default())
                        .unwrap().into_raw(),
                }
            }).collect();

            let count = c_results.len() as c_int;
            let results_ptr = c_results.into_boxed_slice();
            Box::into_raw(Box::new(CSearchResults {
                results: Box::into_raw(results_ptr) as *mut CSearchResult,
                count,
            }))
        },
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn veclite_free(db: *mut CVecLite) {
    if !db.is_null() {
        unsafe { Box::from_raw(db) };
    }
}

#[no_mangle]
pub extern "C" fn veclite_free_search_results(results: *mut CSearchResults) {
    if !results.is_null() {
        let results = unsafe { Box::from_raw(results) };

        // Free individual result strings
        let results_slice = unsafe {
            std::slice::from_raw_parts_mut(results.results, results.count as usize)
        };

        for result in results_slice {
            if !result.id.is_null() {
                unsafe { CString::from_raw(result.id) };
            }
            if !result.metadata_json.is_null() {
                unsafe { CString::from_raw(result.metadata_json) };
            }
        }

        unsafe { Box::from_raw(results.results) };
    }
}
```

#### Go Wrapper Implementation

```go
package veclite

/*
#cgo LDFLAGS: -L. -lveclite
#include "veclite.h"
#include <stdlib.h>
*/
import "C"
import (
    "encoding/json"
    "runtime"
    "unsafe"
)

type VecLite struct {
    handle *C.CVecLite
}

type SearchResult struct {
    ID       string            `json:"id"`
    Score    float32           `json:"score"`
    Metadata map[string]string `json:"metadata"`
}

func New() (*VecLite, error) {
    handle := C.veclite_new()
    if handle == nil {
        return nil, fmt.Errorf("failed to create VecLite instance")
    }

    vl := &VecLite{handle: handle}
    runtime.SetFinalizer(vl, (*VecLite).Close)
    return vl, nil
}

func (vl *VecLite) Insert(id string, vector []float32, metadata map[string]string) error {
    if vl.handle == nil {
        return fmt.Errorf("VecLite instance is closed")
    }

    cID := C.CString(id)
    defer C.free(unsafe.Pointer(cID))

    var cVector *C.c_float
    if len(vector) > 0 {
        cVector = (*C.c_float)(unsafe.Pointer(&vector[0]))
    }

    var cMetadata *C.char
    if metadata != nil {
        metadataJSON, _ := json.Marshal(metadata)
        cMetadata = C.CString(string(metadataJSON))
        defer C.free(unsafe.Pointer(cMetadata))
    }

    result := C.veclite_insert(vl.handle, cID, cVector, C.c_int(len(vector)), cMetadata)
    if result != 0 {
        return fmt.Errorf("failed to insert vector")
    }

    return nil
}

func (vl *VecLite) Search(query []float32, k int) ([]SearchResult, error) {
    if vl.handle == nil {
        return nil, fmt.Errorf("VecLite instance is closed")
    }

    if len(query) == 0 {
        return nil, fmt.Errorf("query vector is empty")
    }

    cQuery := (*C.c_float)(unsafe.Pointer(&query[0]))
    cResults := C.veclite_search(vl.handle, cQuery, C.c_int(len(query)), C.c_int(k))

    if cResults == nil {
        return nil, fmt.Errorf("search failed")
    }
    defer C.veclite_free_search_results(cResults)

    count := int(cResults.count)
    if count == 0 {
        return []SearchResult{}, nil
    }

    cResultsSlice := (*[1 << 28]C.CSearchResult)(unsafe.Pointer(cResults.results))[:count:count]
    results := make([]SearchResult, count)

    for i, cResult := range cResultsSlice {
        id := C.GoString(cResult.id)
        score := float32(cResult.score)

        var metadata map[string]string
        if cResult.metadata_json != nil {
            metadataJSON := C.GoString(cResult.metadata_json)
            json.Unmarshal([]byte(metadataJSON), &metadata)
        }

        results[i] = SearchResult{
            ID:       id,
            Score:    score,
            Metadata: metadata,
        }
    }

    return results, nil
}

func (vl *VecLite) Close() {
    if vl.handle != nil {
        C.veclite_free(vl.handle)
        vl.handle = nil
        runtime.SetFinalizer(vl, nil)
    }
}
```

## 3. Performance Optimization

### 3.1 Memory Layout Optimization

```rust
// Align structures for SIMD operations
#[repr(align(32))]
pub struct AlignedVector {
    data: Vec<f32>,
}

// Use memory pools for frequent allocations
pub struct VectorPool {
    pools: HashMap<usize, Vec<Vec<f32>>>,
    max_pool_size: usize,
}

impl VectorPool {
    pub fn get_vector(&mut self, size: usize) -> Vec<f32> {
        if let Some(pool) = self.pools.get_mut(&size) {
            if let Some(vec) = pool.pop() {
                return vec;
            }
        }

        let mut vec = Vec::with_capacity(size);
        vec.resize(size, 0.0);
        vec
    }

    pub fn return_vector(&mut self, mut vec: Vec<f32>) {
        let size = vec.capacity();
        vec.clear();

        let pool = self.pools.entry(size).or_insert_with(Vec::new);
        if pool.len() < self.max_pool_size {
            pool.push(vec);
        }
    }
}
```

### 3.2 Algorithmic Optimizations

```rust
// Batch distance computations for better cache utilization
impl DistanceMetric for EuclideanDistance {
    fn batch_distance(&self, query: &[f32], vectors: &[&[f32]]) -> Vec<f32> {
        let mut results = Vec::with_capacity(vectors.len());

        // Process in chunks for better cache usage
        const CHUNK_SIZE: usize = 64;
        for chunk in vectors.chunks(CHUNK_SIZE) {
            for vector in chunk {
                results.push(self.distance(query, vector));
            }
        }

        results
    }
}

// Early termination for top-k search
impl HNSWIndex {
    fn search_with_early_termination(&self, query: &[f32], k: usize, target_recall: f32)
                                   -> Result<Vec<SearchResult>, VecLiteError> {
        let mut ef = self.config.ef_search;
        let max_ef = ef * 4;

        while ef <= max_ef {
            let results = self.search(query, k, ef)?;

            // Estimate recall based on score distribution
            if self.estimate_recall(&results) >= target_recall {
                return Ok(results);
            }

            ef = (ef as f32 * 1.5) as usize;
        }

        // Fallback to maximum ef
        self.search(query, k, max_ef)
    }
}
```

## 4. Error Handling Strategy

### 4.1 Error Types

```rust
#[derive(Debug, thiserror::Error)]
pub enum VecLiteError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] bincode::Error),

    #[error("Invalid vector dimensions: expected {expected}, got {actual}")]
    InvalidDimensions { expected: usize, actual: usize },

    #[error("Vector not found: {id}")]
    VectorNotFound { id: String },

    #[error("Memory limit exceeded: {current_usage} bytes")]
    MemoryLimitExceeded { current_usage: usize },

    #[error("Invalid k value: {0} (must be > 0 and <= {max})", max = "max_k")]
    InvalidK(usize),

    #[error("Database is corrupted: {reason}")]
    CorruptedDatabase { reason: String },

    #[error("File format error: {0}")]
    FileFormat(String),

    #[error("Checksum mismatch")]
    ChecksumMismatch,

    #[error("Index build failed: {0}")]
    IndexBuildFailed(String),

    #[error("Configuration error: {0}")]
    Configuration(String),
}

pub type Result<T> = std::result::Result<T, VecLiteError>;
```

### 4.2 Error Recovery

```rust
impl VecLite {
    pub fn recover_from_corruption(&mut self, backup_path: Option<&Path>) -> Result<()> {
        eprintln!("Attempting to recover from database corruption...");

        if let Some(backup) = backup_path {
            eprintln!("Restoring from backup: {}", backup.display());
            *self = Self::load(backup)?;
            return Ok(());
        }

        // Attempt partial recovery
        eprintln!("Attempting partial recovery...");

        // Reset index and rebuild
        self.index_manager.clear();
        let vectors: Vec<_> = self.storage_manager.iter().collect();

        for (id, vector) in vectors {
            if let Err(e) = self.index_manager.insert(id.clone(), vector.vector.clone()) {
                eprintln!("Failed to reindex vector {}: {}", id, e);
                // Continue with other vectors
            }
        }

        eprintln!("Partial recovery completed");
        Ok(())
    }
}
```

## 5. Testing Strategy

### 5.1 Unit Test Framework

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_vector_insertion_and_retrieval() {
        let mut db = VecLite::new().unwrap();
        let vector = vec![1.0, 2.0, 3.0];
        let metadata = HashMap::from([("key".to_string(), "value".to_string())]);

        db.insert("test_id".to_string(), vector.clone(), metadata.clone()).unwrap();

        let results = db.search(&vector, 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "test_id");
        assert_eq!(results[0].metadata, metadata);
    }

    proptest! {
        #[test]
        fn test_distance_metric_properties(
            a in prop::collection::vec(-1000.0f32..1000.0f32, 10..1000),
            b in prop::collection::vec(-1000.0f32..1000.0f32, 10..1000)
        ) {
            prop_assume!(a.len() == b.len());

            let metric = EuclideanDistance;
            let d_ab = metric.distance(&a, &b);
            let d_ba = metric.distance(&b, &a);
            let d_aa = metric.distance(&a, &a);

            // Symmetry
            assert!((d_ab - d_ba).abs() < f32::EPSILON);

            // Identity
            assert!(d_aa < f32::EPSILON);

            // Non-negativity
            assert!(d_ab >= 0.0);
        }
    }
}
```

### 5.2 Integration Tests

```rust
#[cfg(test)]
mod integration_tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_persistence_roundtrip() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.vlt");

        // Create and populate database
        let mut db1 = VecLite::new().unwrap();
        for i in 0..1000 {
            let vector = vec![i as f32; 768];
            let metadata = HashMap::from([("index".to_string(), i.to_string())]);
            db1.insert(format!("doc_{}", i), vector, metadata).unwrap();
        }

        // Save to disk
        db1.save(&db_path).unwrap();

        // Load from disk
        let db2 = VecLite::load(&db_path).unwrap();

        // Verify data integrity
        let query = vec![500.0; 768];
        let results1 = db1.search(&query, 10).unwrap();
        let results2 = db2.search(&query, 10).unwrap();

        assert_eq!(results1.len(), results2.len());
        for (r1, r2) in results1.iter().zip(results2.iter()) {
            assert_eq!(r1.id, r2.id);
            assert!((r1.score - r2.score).abs() < 1e-6);
        }
    }

    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let db = Arc::new(VecLite::new().unwrap());
        let mut handles = vec![];

        // Concurrent insertions
        for thread_id in 0..10 {
            let db_clone = db.clone();
            let handle = thread::spawn(move || {
                for i in 0..100 {
                    let vector = vec![(thread_id * 100 + i) as f32; 768];
                    let id = format!("thread_{}_{}", thread_id, i);
                    db_clone.insert(id, vector, HashMap::new()).unwrap();
                }
            });
            handles.push(handle);
        }

        // Wait for all insertions
        for handle in handles {
            handle.join().unwrap();
        }

        // Verify all vectors were inserted
        let query = vec![500.0; 768];
        let results = db.search(&query, 1000).unwrap();
        assert_eq!(results.len(), 1000);
    }
}
```

### 5.3 Benchmark Framework

```rust
#[cfg(test)]
mod benchmarks {
    use super::*;
    use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

    fn benchmark_insertion(c: &mut Criterion) {
        let mut group = c.benchmark_group("insertion");

        for vector_count in [1000, 10000, 100000].iter() {
            group.bench_with_input(
                BenchmarkId::new("sequential", vector_count),
                vector_count,
                |b, &vector_count| {
                    b.iter_with_setup(
                        || VecLite::new().unwrap(),
                        |mut db| {
                            for i in 0..vector_count {
                                let vector = vec![i as f32; 768];
                                db.insert(format!("doc_{}", i), vector, HashMap::new()).unwrap();
                            }
                        }
                    );
                }
            );
        }

        group.finish();
    }

    fn benchmark_search(c: &mut Criterion) {
        let mut group = c.benchmark_group("search");

        // Setup database with different sizes
        let sizes = [1000, 10000, 100000];
        let databases: Vec<_> = sizes.iter().map(|&size| {
            let mut db = VecLite::new().unwrap();
            for i in 0..size {
                let vector = vec![i as f32; 768];
                db.insert(format!("doc_{}", i), vector, HashMap::new()).unwrap();
            }
            (size, db)
        }).collect();

        for (size, db) in databases.iter() {
            group.bench_with_input(
                BenchmarkId::new("knn_search", size),
                size,
                |b, _| {
                    let query = vec![5000.0; 768];
                    b.iter(|| {
                        db.search(&query, 10).unwrap()
                    });
                }
            );
        }

        group.finish();
    }

    criterion_group!(benches, benchmark_insertion, benchmark_search);
    criterion_main!(benches);
}
```

## 6. Build and Deployment

### 6.1 Build Configuration

```toml
[package]
name = "veclite"
version = "0.1.0"
edition = "2021"
license = "MIT OR Apache-2.0"
description = "Lightweight embeddable vector search library"
homepage = "https://github.com/user/veclite"
repository = "https://github.com/user/veclite"
keywords = ["vector", "search", "embedding", "similarity", "database"]
categories = ["database", "algorithms", "science"]

[lib]
name = "veclite"
crate-type = ["cdylib", "rlib"]

[dependencies]
serde = { version = "1.0", features = ["derive"] }
bincode = "1.3"
thiserror = "1.0"
rayon = "1.7"
zstd = { version = "0.12", optional = true }
clap = { version = "4.0", features = ["derive"], optional = true }

[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
proptest = "1.0"
tempfile = "3.0"

[features]
default = ["compression"]
compression = ["zstd"]
cli = ["clap"]
simd = []

[profile.release]
lto = true
codegen-units = 1
panic = "abort"

[profile.bench]
debug = true

[[bench]]
name = "benchmarks"
harness = false
```

### 6.2 Cross-Language Build

```makefile
.PHONY: all clean test bench rust go

all: rust go

rust:
	cargo build --release

go: rust
	cbindgen --config cbindgen.toml --crate veclite --output veclite.h
	cd go && go build

test:
	cargo test
	cd go && go test -v

bench:
	cargo bench
	cd go && go test -bench=.

clean:
	cargo clean
	cd go && go clean
	rm -f veclite.h

install:
	cargo install --path .

docker:
	docker build -t veclite .

cross-compile:
	cross build --target x86_64-pc-windows-gnu --release
	cross build --target x86_64-apple-darwin --release
	cross build --target aarch64-apple-darwin --release
	cross build --target x86_64-unknown-linux-gnu --release
	cross build --target aarch64-unknown-linux-gnu --release
```

---

*This TDD provides the complete technical specification for implementing VecLite according to Google Level 5 standards, with comprehensive architecture, algorithms, error handling, testing, and deployment strategies.*