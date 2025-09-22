/// Hierarchical Navigable Small World (HNSW) index implementation
/// Based on "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"
/// by Yu. A. Malkov and D. A. Yashunin
use crate::{
    distance::DistanceMetric,
    error::{IndexError, VecLiteError, VecLiteResult},
    query::{IndexStats, SearchIndex},
    types::{Score, SearchResult, VectorData, VectorId},
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap, HashSet},
    f64::consts::LN_2,
    sync::Arc,
};

/// HNSW algorithm parameters
#[derive(Debug, Clone)]
pub struct HNSWConfig {
    /// Maximum connections per node in layer 0
    pub max_connections_0: usize,
    /// Maximum connections per node in layers 1+
    pub max_connections: usize,
    /// Size of dynamic candidate list during construction
    pub ef_construction: usize,
    /// Size of dynamic candidate list during search (can be overridden per query)
    pub ef_search: usize,
    /// Level generation factor (1/ln(2))
    pub ml: f64,
    /// Random seed for reproducible builds
    pub seed: u64,
}

impl Default for HNSWConfig {
    fn default() -> Self {
        Self {
            max_connections_0: 32, // M_L from paper
            max_connections: 16,   // M from paper
            ef_construction: 200,  // efConstruction from paper
            ef_search: 50,         // ef from paper (runtime parameter)
            ml: 1.0 / LN_2,        // mL from paper
            seed: 42,
        }
    }
}

/// Node identifier within HNSW graph
pub type NodeId = u32;

/// Connection with distance information
#[derive(Debug, Clone, Copy)]
pub struct Connection {
    pub node_id: NodeId,
    pub distance: Score,
}

impl PartialEq for Connection {
    fn eq(&self, other: &Self) -> bool {
        self.node_id == other.node_id
    }
}

impl Eq for Connection {}

impl PartialOrd for Connection {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Connection {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // For min-heap behavior in BinaryHeap (closest connections first)
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| self.node_id.cmp(&other.node_id))
    }
}

/// HNSW node containing vector data and connections at each layer
#[derive(Debug, Clone)]
pub struct HNSWNode {
    pub id: NodeId,
    pub vector_id: VectorId,
    pub vector: VectorData,
    pub level: usize,
    /// Connections at each layer (layer -> set of connections)
    pub connections: Vec<HashSet<NodeId>>,
}

impl HNSWNode {
    pub fn new(id: NodeId, vector_id: VectorId, vector: VectorData, level: usize) -> Self {
        let mut connections = Vec::with_capacity(level + 1);
        for _ in 0..=level {
            connections.push(HashSet::new());
        }

        Self {
            id,
            vector_id,
            vector,
            level,
            connections,
        }
    }

    /// Get connections at a specific layer
    pub fn get_connections(&self, layer: usize) -> Option<&HashSet<NodeId>> {
        self.connections.get(layer)
    }

    /// Add bidirectional connection at a specific layer
    pub fn add_connection(&mut self, layer: usize, node_id: NodeId) {
        if let Some(connections) = self.connections.get_mut(layer) {
            connections.insert(node_id);
        }
    }

    /// Remove connection at a specific layer
    pub fn remove_connection(&mut self, layer: usize, node_id: NodeId) {
        if let Some(connections) = self.connections.get_mut(layer) {
            connections.remove(&node_id);
        }
    }
}

/// Priority queue elements for search algorithms
#[derive(Debug, Clone)]
pub struct Candidate {
    pub node_id: NodeId,
    pub distance: Score,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.node_id == other.node_id
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // For max-heap behavior in BinaryHeap (farthest candidates first for working queue)
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| self.node_id.cmp(&other.node_id))
    }
}

/// HNSW Index implementation
#[derive(Debug)]
pub struct HNSWIndex {
    /// HNSW configuration parameters
    config: HNSWConfig,
    /// All nodes in the graph, indexed by NodeId
    nodes: HashMap<NodeId, HNSWNode>,
    /// Mapping from VectorId to NodeId for lookups
    vector_to_node: HashMap<VectorId, NodeId>,
    /// Entry point node (highest level node)
    entry_point: Option<NodeId>,
    /// Next available node ID
    next_node_id: NodeId,
    /// Distance metric for vector comparisons
    distance_metric: Arc<dyn DistanceMetric>,
    /// Random number generator for level assignment
    rng: ChaCha8Rng,
    /// Index statistics
    stats: IndexStats,
}

impl HNSWIndex {
    /// Create a new HNSW index with specified configuration and distance metric
    pub fn new(config: HNSWConfig, distance_metric: Arc<dyn DistanceMetric>) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(config.seed),
            config,
            nodes: HashMap::new(),
            vector_to_node: HashMap::new(),
            entry_point: None,
            next_node_id: 0,
            distance_metric,
            stats: IndexStats::default(),
        }
    }

    /// Generate random level for new node based on ml parameter
    fn generate_level(&mut self) -> usize {
        let mut level = 0;
        while self.rng.gen::<f64>() < 1.0 / self.config.ml && level < 16 {
            level += 1;
        }
        level
    }

    /// Get distance between two nodes
    fn node_distance(&self, node_a: &HNSWNode, node_b: &HNSWNode) -> Score {
        self.distance_metric
            .distance(&node_a.vector, &node_b.vector)
    }

    /// Get distance between query vector and node
    fn query_distance(&self, query: &VectorData, node: &HNSWNode) -> Score {
        self.distance_metric.distance(query, &node.vector)
    }

    /// Search for closest nodes starting from entry points at a given layer
    fn search_layer(
        &self,
        query: &VectorData,
        entry_points: Vec<NodeId>,
        num_closest: usize,
        layer: usize,
    ) -> VecLiteResult<Vec<Candidate>> {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new(); // Max-heap for working queue
        let mut dynamic_candidates = BinaryHeap::new(); // Min-heap for result candidates

        // Initialize with entry points
        for entry_id in entry_points {
            if let Some(entry_node) = self.nodes.get(&entry_id) {
                let distance = self.query_distance(query, entry_node);
                let candidate = Candidate {
                    node_id: entry_id,
                    distance,
                };

                visited.insert(entry_id);
                candidates.push(candidate.clone());
                dynamic_candidates.push(Reverse(candidate));
            }
        }

        while let Some(current) = candidates.pop() {
            // Stop if current candidate is farther than the farthest in dynamic list
            if dynamic_candidates.len() >= num_closest {
                if let Some(Reverse(farthest)) = dynamic_candidates.peek() {
                    if current.distance > farthest.distance {
                        break;
                    }
                }
            }

            // Explore connections of current node
            if let Some(current_node) = self.nodes.get(&current.node_id) {
                if let Some(connections) = current_node.get_connections(layer) {
                    for &neighbor_id in connections {
                        if !visited.contains(&neighbor_id) {
                            visited.insert(neighbor_id);

                            if let Some(neighbor_node) = self.nodes.get(&neighbor_id) {
                                let distance = self.query_distance(query, neighbor_node);
                                let neighbor_candidate = Candidate {
                                    node_id: neighbor_id,
                                    distance,
                                };

                                // Add to working queue
                                candidates.push(neighbor_candidate.clone());

                                // Maintain dynamic candidate list
                                if dynamic_candidates.len() < num_closest {
                                    dynamic_candidates.push(Reverse(neighbor_candidate));
                                } else if let Some(Reverse(farthest)) = dynamic_candidates.peek() {
                                    if distance < farthest.distance {
                                        dynamic_candidates.pop();
                                        dynamic_candidates.push(Reverse(neighbor_candidate));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Convert min-heap results to sorted vec (closest first)
        let mut results: Vec<_> = dynamic_candidates.into_iter().map(|Reverse(c)| c).collect();
        results.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(results)
    }

    /// Select M connections from candidates using simple heuristic
    fn select_connections_simple(&self, candidates: Vec<Candidate>, m: usize) -> Vec<NodeId> {
        candidates.into_iter().take(m).map(|c| c.node_id).collect()
    }

    /// Prune connections of a node to maintain degree constraints
    fn prune_connections(&mut self, node_id: NodeId, layer: usize) {
        let max_connections = if layer == 0 {
            self.config.max_connections_0
        } else {
            self.config.max_connections
        };

        if let Some(node) = self.nodes.get(&node_id) {
            if let Some(connections) = node.get_connections(layer) {
                if connections.len() > max_connections {
                    // Collect all connections with distances
                    let mut conn_distances: Vec<_> = connections
                        .iter()
                        .filter_map(|&conn_id| {
                            self.nodes.get(&conn_id).map(|conn_node| {
                                let distance = self.node_distance(node, conn_node);
                                Candidate {
                                    node_id: conn_id,
                                    distance,
                                }
                            })
                        })
                        .collect();

                    // Sort by distance and keep only the closest ones
                    conn_distances.sort_by(|a, b| {
                        a.distance
                            .partial_cmp(&b.distance)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });

                    let new_connections: HashSet<_> = conn_distances
                        .into_iter()
                        .take(max_connections)
                        .map(|c| c.node_id)
                        .collect();

                    // Update connections
                    if let Some(node) = self.nodes.get_mut(&node_id) {
                        if let Some(connections) = node.connections.get_mut(layer) {
                            *connections = new_connections;
                        }
                    }
                }
            }
        }
    }

    /// Insert a new node into the HNSW graph
    fn insert_node(&mut self, vector_id: VectorId, vector: VectorData) -> VecLiteResult<()> {
        let node_id = self.next_node_id;
        self.next_node_id += 1;

        let level = self.generate_level();
        let mut new_node = HNSWNode::new(node_id, vector_id.clone(), vector, level);

        // If this is the first node, make it the entry point
        if self.entry_point.is_none() {
            self.entry_point = Some(node_id);
            self.nodes.insert(node_id, new_node);
            self.vector_to_node.insert(vector_id, node_id);
            self.stats.vector_count += 1;
            return Ok(());
        }

        let entry_point = self.entry_point.unwrap();
        let entry_level = self
            .nodes
            .get(&entry_point)
            .ok_or_else(|| {
                VecLiteError::Index(IndexError::Corrupted {
                    details: "Entry point not found".to_string(),
                })
            })?
            .level;

        let mut current_closest = vec![entry_point];

        // Search from top layer down to layer level+1
        for layer in (level + 1..=entry_level).rev() {
            let candidates = self.search_layer(&new_node.vector, current_closest, 1, layer)?;
            current_closest = vec![candidates[0].node_id];
        }

        // Search and connect from level down to 0
        for layer in (0..=level).rev() {
            let candidates = self.search_layer(
                &new_node.vector,
                current_closest.clone(),
                self.config.ef_construction,
                layer,
            )?;

            let max_connections = if layer == 0 {
                self.config.max_connections_0
            } else {
                self.config.max_connections
            };

            let selected = self.select_connections_simple(candidates.clone(), max_connections);

            // Add connections to new node
            for &conn_id in &selected {
                new_node.add_connection(layer, conn_id);

                // Add reciprocal connection
                if let Some(conn_node) = self.nodes.get_mut(&conn_id) {
                    conn_node.add_connection(layer, node_id);

                    // Prune connections if necessary
                    let conn_count = conn_node
                        .get_connections(layer)
                        .map(|c| c.len())
                        .unwrap_or(0);

                    if conn_count > max_connections {
                        // Release mutable borrow before calling prune_connections
                        self.prune_connections(conn_id, layer);
                    }
                }
            }

            current_closest = candidates.into_iter().map(|c| c.node_id).collect();
        }

        // Update entry point if new node has higher level
        if level > entry_level {
            self.entry_point = Some(node_id);
        }

        self.nodes.insert(node_id, new_node);
        self.vector_to_node.insert(vector_id, node_id);
        self.stats.vector_count += 1;

        Ok(())
    }
}

impl SearchIndex for HNSWIndex {
    fn build(&mut self, vectors: &[(VectorId, VectorData)]) -> VecLiteResult<()> {
        let start_time = std::time::Instant::now();

        self.nodes.clear();
        self.vector_to_node.clear();
        self.entry_point = None;
        self.next_node_id = 0;
        self.stats = IndexStats::default();

        for (vector_id, vector_data) in vectors {
            self.insert_node(vector_id.clone(), vector_data.clone())?;
        }

        self.stats.build_time_ms = start_time.elapsed().as_millis() as u64;
        Ok(())
    }

    fn insert(&mut self, id: VectorId, vector: VectorData) -> VecLiteResult<()> {
        // Check if vector already exists
        if self.vector_to_node.contains_key(&id) {
            return Err(VecLiteError::Index(IndexError::InsertFailed {
                reason: format!("Vector with id '{}' already exists", id),
            }));
        }

        self.insert_node(id, vector)
    }

    fn delete(&mut self, id: &VectorId) -> VecLiteResult<bool> {
        if let Some(&node_id) = self.vector_to_node.get(id) {
            // Remove node from all layers
            if let Some(node) = self.nodes.remove(&node_id) {
                // Remove connections from other nodes
                for layer in 0..=node.level {
                    if let Some(connections) = node.get_connections(layer) {
                        for &conn_id in connections {
                            if let Some(conn_node) = self.nodes.get_mut(&conn_id) {
                                conn_node.remove_connection(layer, node_id);
                            }
                        }
                    }
                }

                // Update entry point if necessary
                if Some(node_id) == self.entry_point {
                    // Find new entry point (node with highest level)
                    self.entry_point = self.nodes.values().max_by_key(|n| n.level).map(|n| n.id);
                }

                self.vector_to_node.remove(id);
                self.stats.vector_count -= 1;
                Ok(true)
            } else {
                Ok(false)
            }
        } else {
            Ok(false)
        }
    }

    fn search(&self, query: &VectorData, k: usize) -> VecLiteResult<Vec<SearchResult>> {
        if self.entry_point.is_none() || self.nodes.is_empty() {
            return Ok(Vec::new());
        }

        let entry_point = self.entry_point.unwrap();
        let entry_level = self
            .nodes
            .get(&entry_point)
            .ok_or_else(|| {
                VecLiteError::Index(IndexError::Corrupted {
                    details: "Entry point not found".to_string(),
                })
            })?
            .level;

        let mut current_closest = vec![entry_point];

        // Search from top layer down to layer 1
        for layer in (1..=entry_level).rev() {
            let candidates = self.search_layer(query, current_closest, 1, layer)?;
            current_closest = vec![candidates[0].node_id];
        }

        // Search at layer 0 with ef_search candidates
        let candidates =
            self.search_layer(query, current_closest, self.config.ef_search.max(k), 0)?;

        // Convert to SearchResults
        let results = candidates
            .into_iter()
            .take(k)
            .filter_map(|candidate| {
                self.nodes.get(&candidate.node_id).map(|node| {
                    SearchResult::new(
                        node.vector_id.clone(),
                        candidate.distance,
                        std::collections::HashMap::new(), // TODO: Add metadata support
                    )
                })
            })
            .collect();

        Ok(results)
    }

    fn stats(&self) -> IndexStats {
        let mut stats = self.stats.clone();

        // Calculate average degree and layer distribution
        if !self.nodes.is_empty() {
            let total_connections: usize = self
                .nodes
                .values()
                .flat_map(|node| &node.connections)
                .map(|connections| connections.len())
                .sum();
            stats.average_degree = total_connections as f64 / self.nodes.len() as f64;

            let max_level = self
                .nodes
                .values()
                .map(|node| node.level)
                .max()
                .unwrap_or(0);
            stats.max_level = max_level;
        }

        stats
    }

    fn clear(&mut self) -> VecLiteResult<()> {
        self.nodes.clear();
        self.vector_to_node.clear();
        self.entry_point = None;
        self.next_node_id = 0;
        self.stats = IndexStats::default();
        Ok(())
    }

    fn name(&self) -> &'static str {
        "hnsw"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::EuclideanDistance;

    fn create_test_vectors(count: usize, dimensions: usize) -> Vec<(VectorId, VectorData)> {
        (0..count)
            .map(|i| {
                let id = format!("vec_{}", i);
                let vector: VectorData =
                    (0..dimensions).map(|j| ((i + j) as f32) / 100.0).collect();
                (id, vector)
            })
            .collect()
    }

    #[test]
    fn test_hnsw_config_default() {
        let config = HNSWConfig::default();
        assert_eq!(config.max_connections_0, 32);
        assert_eq!(config.max_connections, 16);
        assert_eq!(config.ef_construction, 200);
        assert_eq!(config.ef_search, 50);
    }

    #[test]
    fn test_hnsw_node_creation() {
        let vector = vec![1.0, 2.0, 3.0];
        let node = HNSWNode::new(0, "test".to_string(), vector.clone(), 2);

        assert_eq!(node.id, 0);
        assert_eq!(node.vector_id, "test");
        assert_eq!(node.vector, vector);
        assert_eq!(node.level, 2);
        assert_eq!(node.connections.len(), 3); // levels 0, 1, 2
    }

    #[test]
    fn test_hnsw_index_creation() {
        let config = HNSWConfig::default();
        let distance_metric = Arc::new(EuclideanDistance);
        let index = HNSWIndex::new(config, distance_metric);

        assert_eq!(index.nodes.len(), 0);
        assert!(index.entry_point.is_none());
        assert_eq!(index.next_node_id, 0);
    }

    #[test]
    fn test_hnsw_build_small_dataset() {
        let config = HNSWConfig::default();
        let distance_metric = Arc::new(EuclideanDistance);
        let mut index = HNSWIndex::new(config, distance_metric);

        let vectors = create_test_vectors(10, 3);
        let result = index.build(&vectors);

        assert!(result.is_ok());
        assert_eq!(index.stats.vector_count, 10);
        assert!(index.entry_point.is_some());
        // Build time may be 0 for small datasets due to timing precision
    }

    #[test]
    fn test_hnsw_insert_and_search() {
        let config = HNSWConfig::default();
        let distance_metric = Arc::new(EuclideanDistance);
        let mut index = HNSWIndex::new(config, distance_metric);

        // Build initial index
        let vectors = create_test_vectors(20, 4);
        index.build(&vectors).unwrap();

        // Test search
        let query = vec![0.5, 1.5, 2.5, 3.5];
        let results = index.search(&query, 5).unwrap();

        assert!(!results.is_empty());
        assert!(results.len() <= 5);

        // Results should be sorted by distance
        for window in results.windows(2) {
            assert!(window[0].score <= window[1].score);
        }
    }

    #[test]
    fn test_hnsw_dynamic_insertion() {
        let config = HNSWConfig::default();
        let distance_metric = Arc::new(EuclideanDistance);
        let mut index = HNSWIndex::new(config, distance_metric);

        // Build initial index
        let vectors = create_test_vectors(10, 3);
        index.build(&vectors).unwrap();

        let initial_count = index.stats.vector_count;

        // Insert new vector
        let new_vector = vec![10.0, 20.0, 30.0];
        let result = index.insert("new_vec".to_string(), new_vector);

        assert!(result.is_ok());
        assert_eq!(index.stats.vector_count, initial_count + 1);

        // Should be able to find the new vector
        let query = vec![10.0, 20.0, 30.0];
        let results = index.search(&query, 1).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "new_vec");
        assert!(results[0].score < 0.01); // Should be very close
    }

    #[test]
    fn test_hnsw_deletion() {
        let config = HNSWConfig::default();
        let distance_metric = Arc::new(EuclideanDistance);
        let mut index = HNSWIndex::new(config, distance_metric);

        // Build initial index
        let vectors = create_test_vectors(10, 3);
        index.build(&vectors).unwrap();

        let initial_count = index.stats.vector_count;

        // Delete a vector
        let deleted = index.delete(&"vec_5".to_string()).unwrap();

        assert!(deleted);
        assert_eq!(index.stats.vector_count, initial_count - 1);

        // Should not find deleted vector in results
        let query = create_test_vectors(1, 3)[0].1.clone();
        let results = index.search(&query, 10).unwrap();

        assert!(!results.iter().any(|r| r.id == "vec_5"));
    }

    #[test]
    fn test_hnsw_clear() {
        let config = HNSWConfig::default();
        let distance_metric = Arc::new(EuclideanDistance);
        let mut index = HNSWIndex::new(config, distance_metric);

        // Build initial index
        let vectors = create_test_vectors(10, 3);
        index.build(&vectors).unwrap();

        assert_eq!(index.stats.vector_count, 10);

        // Clear index
        let result = index.clear();

        assert!(result.is_ok());
        assert_eq!(index.stats.vector_count, 0);
        assert!(index.entry_point.is_none());
        assert_eq!(index.nodes.len(), 0);

        // Search should return empty results
        let query = vec![1.0, 2.0, 3.0];
        let results = index.search(&query, 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_connection_ordering() {
        let conn1 = Connection {
            node_id: 1,
            distance: 0.5,
        };
        let conn2 = Connection {
            node_id: 2,
            distance: 0.3,
        };
        let conn3 = Connection {
            node_id: 3,
            distance: 0.7,
        };

        let mut heap = BinaryHeap::new();
        heap.push(conn1);
        heap.push(conn2);
        heap.push(conn3);

        // Should pop in order of increasing distance (min-heap behavior)
        let popped = heap.pop().unwrap();
        assert_eq!(popped.distance, 0.3);
    }

    #[test]
    fn test_candidate_ordering() {
        let cand1 = Candidate {
            node_id: 1,
            distance: 0.5,
        };
        let cand2 = Candidate {
            node_id: 2,
            distance: 0.3,
        };
        let cand3 = Candidate {
            node_id: 3,
            distance: 0.7,
        };

        let mut heap = BinaryHeap::new();
        heap.push(cand1);
        heap.push(cand2);
        heap.push(cand3);

        // Should pop in order of decreasing distance (max-heap behavior)
        let popped = heap.pop().unwrap();
        assert_eq!(popped.distance, 0.7);
    }
}
