/// VecLite Persistence Layer (.vlt file format)
///
/// The .vlt format is designed for efficient storage and retrieval of vector databases
/// with support for compression, versioning, and integrity validation.
use crate::{
    error::{PersistenceError, VecLiteError, VecLiteResult},
    query::SearchIndex,
    storage::{StorageManager, VectorStorage},
    types::{Dimensions, Metadata, PersistenceConfig, VectorData, VectorId, VectorItem},
};
use bincode;
use serde::{Deserialize, Serialize};
use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::Path,
};

/// VLT file format constants
pub mod file_format {
    /// VLT file magic bytes "VLT1"
    pub const VLT_MAGIC: &[u8; 4] = b"VLT1";

    /// Current file format version
    pub const VLT_VERSION: u32 = 1;

    /// Minimum supported version for backward compatibility
    pub const MIN_SUPPORTED_VERSION: u32 = 1;

    /// Default compression level for zstd
    pub const DEFAULT_COMPRESSION_LEVEL: i32 = 3;

    /// File alignment for better I/O performance (4KB)
    pub const FILE_ALIGNMENT: usize = 4096;

    /// Maximum metadata size (1KB)
    pub const MAX_METADATA_SIZE: usize = 1024;

    /// CRC32 checksum size
    pub const CHECKSUM_SIZE: usize = 4;
}

/// VLT file header structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VLTHeader {
    /// Magic bytes (should be "VLT1")
    pub magic: [u8; 4],
    /// File format version
    pub version: u32,
    /// Configuration used when creating this file
    pub config: PersistenceConfig,
    /// Vector dimensions (all vectors must have same dimensions)
    pub dimensions: Dimensions,
    /// Total number of vectors in the database
    pub vector_count: usize,
    /// Index type used (for reconstruction)
    pub index_type: crate::types::IndexType,
    /// Distance metric used
    pub distance_metric: String,
    /// Whether data is compressed
    pub compressed: bool,
    /// Offset to vector data section
    pub vectors_offset: u64,
    /// Size of vector data section (compressed or uncompressed)
    pub vectors_size: u64,
    /// Offset to index data section (if any)
    pub index_offset: u64,
    /// Size of index data section
    pub index_size: u64,
    /// CRC32 checksum of the entire file (excluding header)
    pub checksum: u32,
    /// Timestamp when file was created (Unix timestamp)
    pub created_at: u64,
    /// Timestamp when file was last modified
    pub modified_at: u64,
}

impl Default for VLTHeader {
    fn default() -> Self {
        Self {
            magic: *file_format::VLT_MAGIC,
            version: file_format::VLT_VERSION,
            config: PersistenceConfig::default(),
            dimensions: 0,
            vector_count: 0,
            index_type: crate::types::IndexType::BruteForce,
            distance_metric: "euclidean".to_string(),
            compressed: false,
            vectors_offset: 0,
            vectors_size: 0,
            index_offset: 0,
            index_size: 0,
            checksum: 0,
            created_at: 0,
            modified_at: 0,
        }
    }
}

impl VLTHeader {
    /// Validate header magic bytes and version
    pub fn validate(&self) -> VecLiteResult<()> {
        // Check magic bytes
        if &self.magic != file_format::VLT_MAGIC {
            return Err(VecLiteError::Persistence(PersistenceError::InvalidFormat {
                reason: format!(
                    "Invalid magic bytes: expected {:?}, got {:?}",
                    file_format::VLT_MAGIC,
                    self.magic
                ),
            }));
        }

        // Check version compatibility
        if self.version < file_format::MIN_SUPPORTED_VERSION {
            return Err(VecLiteError::Persistence(
                PersistenceError::VersionMismatch {
                    expected: file_format::VLT_VERSION,
                    actual: self.version,
                },
            ));
        }

        // Basic sanity checks
        if self.dimensions == 0 && self.vector_count > 0 {
            return Err(VecLiteError::Persistence(PersistenceError::InvalidFormat {
                reason: "Invalid dimensions: cannot be 0 with vectors present".to_string(),
            }));
        }

        if self.vectors_offset == 0 && self.vector_count > 0 {
            return Err(VecLiteError::Persistence(PersistenceError::InvalidFormat {
                reason: "Invalid vectors_offset: cannot be 0 with vectors present".to_string(),
            }));
        }

        Ok(())
    }

    /// Calculate expected file size based on header information
    pub fn calculate_expected_size(&self) -> u64 {
        let header_size = bincode::serialized_size(self).unwrap_or(256) as u64;
        header_size + self.vectors_size + self.index_size + file_format::CHECKSUM_SIZE as u64
    }
}

/// Serializable representation of vector data for persistence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableVectorItem {
    pub id: VectorId,
    pub vector: VectorData,
    pub metadata: Metadata,
    pub timestamp: u64,
}

impl From<&VectorItem> for SerializableVectorItem {
    fn from(item: &VectorItem) -> Self {
        Self {
            id: item.id.clone(),
            vector: item.vector.clone(),
            metadata: item.metadata.clone(),
            timestamp: item.timestamp,
        }
    }
}

impl From<SerializableVectorItem> for VectorItem {
    fn from(item: SerializableVectorItem) -> Self {
        Self {
            id: item.id,
            vector: item.vector,
            metadata: item.metadata,
            timestamp: item.timestamp,
        }
    }
}

/// Database snapshot for serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseSnapshot {
    /// All vectors in the database
    pub vectors: Vec<SerializableVectorItem>,
    /// Index-specific serialized data (if any)
    pub index_data: Option<Vec<u8>>,
}

/// VLT file format reader/writer
pub struct VLTPersistence {
    config: PersistenceConfig,
}

impl VLTPersistence {
    /// Create new VLT persistence handler
    pub fn new(config: PersistenceConfig) -> Self {
        Self { config }
    }

    /// Save database to VLT file
    pub fn save_database<P: AsRef<Path>>(
        &self,
        path: P,
        storage: &StorageManager,
        _index: Option<&dyn SearchIndex>,
        dimensions: Dimensions,
        index_type: crate::types::IndexType,
        distance_metric: &str,
    ) -> VecLiteResult<()> {
        let path = path.as_ref();
        let file =
            File::create(path).map_err(|e| VecLiteError::Persistence(PersistenceError::Io(e)))?;
        let mut writer = BufWriter::new(file);

        // Collect all vectors from storage
        let mut vectors = Vec::new();
        for item in storage.iter() {
            vectors.push(SerializableVectorItem::from(&item));
        }

        // Create database snapshot
        let snapshot = DatabaseSnapshot {
            vectors,
            index_data: None, // TODO: Implement index serialization
        };

        // Serialize vector data
        let serialized_data = bincode::serialize(&snapshot)
            .map_err(|e| VecLiteError::Persistence(PersistenceError::Serialization(e)))?;

        // Compress data if enabled
        let (final_data, compressed) = if self.config.compression_enabled {
            #[cfg(feature = "compression")]
            {
                let compressed_data =
                    zstd::bulk::compress(&serialized_data, self.config.compression_level).map_err(
                        |e| {
                            VecLiteError::Persistence(PersistenceError::Compression(format!(
                                "Compression failed: {}",
                                e
                            )))
                        },
                    )?;
                (compressed_data, true)
            }
            #[cfg(not(feature = "compression"))]
            {
                (serialized_data, false)
            }
        } else {
            (serialized_data, false)
        };

        // Calculate checksum
        let checksum = if self.config.checksum_enabled {
            crc32fast::hash(&final_data)
        } else {
            0
        };

        // Create header
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let header_serialized = bincode::serialize(&VLTHeader {
            magic: *file_format::VLT_MAGIC,
            version: file_format::VLT_VERSION,
            config: self.config.clone(),
            dimensions,
            vector_count: snapshot.vectors.len(),
            index_type,
            distance_metric: distance_metric.to_string(),
            compressed,
            vectors_offset: 0, // Will be updated after header is written
            vectors_size: final_data.len() as u64,
            index_offset: 0,
            index_size: 0,
            checksum,
            created_at: now,
            modified_at: now,
        })
        .map_err(|e| VecLiteError::Persistence(PersistenceError::Serialization(e)))?;

        // Calculate actual offsets
        let vectors_offset = header_serialized.len() as u64;

        // Update header with correct offsets
        let mut final_header = bincode::deserialize::<VLTHeader>(&header_serialized)
            .map_err(|e| VecLiteError::Persistence(PersistenceError::Serialization(e)))?;
        final_header.vectors_offset = vectors_offset;

        let final_header_data = bincode::serialize(&final_header)
            .map_err(|e| VecLiteError::Persistence(PersistenceError::Serialization(e)))?;

        // Write header
        writer
            .write_all(&final_header_data)
            .map_err(|e| VecLiteError::Persistence(PersistenceError::Io(e)))?;

        // Write vector data
        writer
            .write_all(&final_data)
            .map_err(|e| VecLiteError::Persistence(PersistenceError::Io(e)))?;

        // Flush to ensure data is written
        writer
            .flush()
            .map_err(|e| VecLiteError::Persistence(PersistenceError::Io(e)))?;

        Ok(())
    }

    /// Load database from VLT file
    pub fn load_database<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> VecLiteResult<(DatabaseSnapshot, VLTHeader)> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(VecLiteError::Persistence(PersistenceError::FileNotFound {
                path: path.to_string_lossy().to_string(),
            }));
        }

        let file =
            File::open(path).map_err(|e| VecLiteError::Persistence(PersistenceError::Io(e)))?;
        let mut reader = BufReader::new(file);

        // Read and parse header
        let header = self.read_header(&mut reader)?;
        header.validate()?;

        // Seek to vector data
        reader
            .seek(SeekFrom::Start(header.vectors_offset))
            .map_err(|e| VecLiteError::Persistence(PersistenceError::Io(e)))?;

        // Read vector data
        let mut data_buffer = vec![0u8; header.vectors_size as usize];
        reader
            .read_exact(&mut data_buffer)
            .map_err(|e| VecLiteError::Persistence(PersistenceError::Io(e)))?;

        // Verify checksum if enabled
        if self.config.checksum_enabled {
            let calculated_checksum = crc32fast::hash(&data_buffer);
            if calculated_checksum != header.checksum {
                return Err(VecLiteError::Persistence(
                    PersistenceError::ChecksumMismatch,
                ));
            }
        }

        // Decompress data if necessary
        let final_data = if header.compressed {
            #[cfg(feature = "compression")]
            {
                zstd::bulk::decompress(&data_buffer, header.vectors_size as usize * 4) // Estimate decompressed size
                    .map_err(|e| {
                        VecLiteError::Persistence(PersistenceError::Compression(format!(
                            "Decompression failed: {}",
                            e
                        )))
                    })?
            }
            #[cfg(not(feature = "compression"))]
            {
                return Err(VecLiteError::Persistence(PersistenceError::InvalidFormat {
                    reason: "File is compressed but compression feature is not enabled".to_string(),
                }));
            }
        } else {
            data_buffer
        };

        // Deserialize database snapshot
        let snapshot = bincode::deserialize::<DatabaseSnapshot>(&final_data)
            .map_err(|e| VecLiteError::Persistence(PersistenceError::Serialization(e)))?;

        // Validate vector count matches header
        if snapshot.vectors.len() != header.vector_count {
            return Err(VecLiteError::Persistence(PersistenceError::InvalidFormat {
                reason: format!(
                    "Vector count mismatch: header says {}, data has {}",
                    header.vector_count,
                    snapshot.vectors.len()
                ),
            }));
        }

        Ok((snapshot, header))
    }

    /// Read header from file
    fn read_header<R: Read>(&self, reader: &mut R) -> VecLiteResult<VLTHeader> {
        // Read magic bytes first
        let mut magic = [0u8; 4];
        reader
            .read_exact(&mut magic)
            .map_err(|e| VecLiteError::Persistence(PersistenceError::Io(e)))?;

        if &magic != file_format::VLT_MAGIC {
            return Err(VecLiteError::Persistence(PersistenceError::InvalidFormat {
                reason: format!(
                    "Invalid magic bytes: expected {:?}, got {:?}",
                    file_format::VLT_MAGIC,
                    magic
                ),
            }));
        }

        // Read version
        let mut version_bytes = [0u8; 4];
        reader
            .read_exact(&mut version_bytes)
            .map_err(|e| VecLiteError::Persistence(PersistenceError::Io(e)))?;
        let version = u32::from_le_bytes(version_bytes);

        // Check version compatibility before proceeding
        if version > file_format::VLT_VERSION {
            return Err(VecLiteError::Persistence(
                PersistenceError::VersionMismatch {
                    expected: file_format::VLT_VERSION,
                    actual: version,
                },
            ));
        }

        // Create a temporary buffer to read the rest of the header
        // We'll use a reasonable size estimate for the header
        let mut header_buffer = vec![0u8; 1024]; // Should be enough for header
        let bytes_read = reader
            .read(&mut header_buffer)
            .map_err(|e| VecLiteError::Persistence(PersistenceError::Io(e)))?;
        header_buffer.truncate(bytes_read);

        // Reconstruct the full header data with magic and version
        let mut full_header_data = Vec::new();
        full_header_data.extend_from_slice(&magic);
        full_header_data.extend_from_slice(&version_bytes);
        full_header_data.extend_from_slice(&header_buffer);

        // Try to deserialize the header
        bincode::deserialize::<VLTHeader>(&full_header_data)
            .map_err(|e| VecLiteError::Persistence(PersistenceError::Serialization(e)))
    }

    /// Create backup of existing file
    pub fn create_backup<P: AsRef<Path>>(&self, path: P) -> VecLiteResult<()> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(()); // Nothing to backup
        }

        let mut backup_path = path.to_path_buf();
        backup_path.set_extension(format!(
            "{}.backup",
            path.extension().and_then(|s| s.to_str()).unwrap_or("vlt")
        ));

        std::fs::copy(path, backup_path)
            .map_err(|e| VecLiteError::Persistence(PersistenceError::Io(e)))?;

        Ok(())
    }

    /// Verify file integrity
    pub fn verify_file<P: AsRef<Path>>(&self, path: P) -> VecLiteResult<VLTHeader> {
        let (_snapshot, header) = self.load_database(path)?;
        Ok(header)
    }

    /// Convenience method to save a VecLite instance
    pub fn save<P: AsRef<Path>>(&self, helix: &crate::Helix, path: P) -> VecLiteResult<()> {
        // Extract required information from Helix instance
        let config = helix.config();
        let stats = helix.stats();
        let dimensions = stats.dimensions.unwrap_or(0);

        // Get the distance metric name (default to euclidean if not available)
        let distance_metric = "euclidean"; // TODO: Extract actual metric from config

        // Access storage through the internal getter method
        self.save_database(
            path,
            helix.storage().as_ref(),
            None, // No index serialization yet
            dimensions,
            config.index.index_type.clone(),
            distance_metric,
        )
    }

    /// Convenience method to load a VecLite instance
    pub fn load<P: AsRef<Path>>(&self, path: P) -> VecLiteResult<crate::Helix> {
        let (snapshot, _header) = self.load_database(path)?;

        // Create a new Helix instance with appropriate configuration
        let mut config = crate::types::VecLiteConfig::default();
        config.persistence = self.config.clone();

        let helix = crate::Helix::with_config(config)?;

        // Insert all vectors from the snapshot
        let vectors: Vec<(
            crate::types::VectorId,
            crate::types::VectorData,
            crate::types::Metadata,
        )> = snapshot
            .vectors
            .into_iter()
            .map(|item| (item.id, item.vector, item.metadata))
            .collect();

        helix.insert_batch(vectors)?;

        Ok(helix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::StorageConfig;
    use std::collections::HashMap;
    use tempfile::tempdir;

    fn create_test_storage() -> StorageManager {
        StorageManager::new(StorageConfig::default())
    }

    fn create_test_vectors() -> Vec<(VectorId, VectorData, Metadata)> {
        vec![
            (
                "vec1".to_string(),
                vec![1.0, 2.0, 3.0],
                HashMap::from([("type".to_string(), "test".to_string())]),
            ),
            (
                "vec2".to_string(),
                vec![4.0, 5.0, 6.0],
                HashMap::from([("category".to_string(), "example".to_string())]),
            ),
            ("vec3".to_string(), vec![7.0, 8.0, 9.0], HashMap::new()),
        ]
    }

    #[test]
    fn test_vlt_header_validation() {
        let header = VLTHeader::default();
        assert!(header.validate().is_ok());

        let mut invalid_magic = header.clone();
        invalid_magic.magic = [0, 0, 0, 0];
        assert!(invalid_magic.validate().is_err());

        let mut invalid_version = header.clone();
        invalid_version.version = 0;
        assert!(invalid_version.validate().is_err());
    }

    #[test]
    fn test_serializable_vector_conversion() {
        let metadata = HashMap::from([("key".to_string(), "value".to_string())]);
        let vector_item =
            VectorItem::new("test_id".to_string(), vec![1.0, 2.0, 3.0], metadata.clone());

        let serializable = SerializableVectorItem::from(&vector_item);
        assert_eq!(serializable.id, "test_id");
        assert_eq!(serializable.vector, vec![1.0, 2.0, 3.0]);
        assert_eq!(serializable.metadata, metadata);

        let converted_back = VectorItem::from(serializable);
        assert_eq!(converted_back.id, vector_item.id);
        assert_eq!(converted_back.vector, vector_item.vector);
        assert_eq!(converted_back.metadata, vector_item.metadata);
    }

    #[test]
    fn test_database_snapshot_serialization() {
        let vectors = create_test_vectors();
        let serializable_vectors: Vec<_> = vectors
            .into_iter()
            .map(|(id, vector, metadata)| SerializableVectorItem {
                id,
                vector,
                metadata,
                timestamp: 123456789,
            })
            .collect();

        let snapshot = DatabaseSnapshot {
            vectors: serializable_vectors,
            index_data: None,
        };

        let serialized = bincode::serialize(&snapshot).unwrap();
        let deserialized: DatabaseSnapshot = bincode::deserialize(&serialized).unwrap();

        assert_eq!(deserialized.vectors.len(), 3);
        assert_eq!(deserialized.vectors[0].id, "vec1");
        assert_eq!(deserialized.vectors[1].vector, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_save_and_load_database() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_database.vlt");

        let storage = create_test_storage();
        let vectors = create_test_vectors();

        // Insert test vectors
        for (id, vector, metadata) in vectors.clone() {
            storage.insert(id, vector, metadata).unwrap();
        }

        let persistence = VLTPersistence::new(PersistenceConfig::default());

        // Save database
        let result = persistence.save_database(
            &file_path,
            &storage,
            None, // No index for this test
            3,    // 3 dimensions
            crate::types::IndexType::BruteForce,
            "euclidean",
        );
        assert!(result.is_ok());
        assert!(file_path.exists());

        // Load database
        let (snapshot, header) = persistence.load_database(&file_path).unwrap();

        // Verify header
        assert_eq!(header.vector_count, 3);
        assert_eq!(header.dimensions, 3);
        assert_eq!(header.distance_metric, "euclidean");
        assert_eq!(header.index_type, crate::types::IndexType::BruteForce);

        // Verify vectors
        assert_eq!(snapshot.vectors.len(), 3);

        let loaded_vec1 = snapshot.vectors.iter().find(|v| v.id == "vec1").unwrap();
        assert_eq!(loaded_vec1.vector, vec![1.0, 2.0, 3.0]);
        assert_eq!(loaded_vec1.metadata.get("type"), Some(&"test".to_string()));
    }

    #[test]
    fn test_file_not_found() {
        let persistence = VLTPersistence::new(PersistenceConfig::default());
        let result = persistence.load_database("nonexistent_file.vlt");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            VecLiteError::Persistence(PersistenceError::FileNotFound { .. })
        ));
    }

    #[test]
    fn test_backup_creation() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test.vlt");
        let backup_path = temp_dir.path().join("test.vlt.backup");

        // Create a test file
        std::fs::write(&file_path, "test data").unwrap();

        let persistence = VLTPersistence::new(PersistenceConfig::default());
        let result = persistence.create_backup(&file_path);

        assert!(result.is_ok());
        assert!(backup_path.exists());

        let backup_content = std::fs::read_to_string(backup_path).unwrap();
        assert_eq!(backup_content, "test data");
    }

    #[test]
    fn test_compression_flag() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("compressed_test.vlt");

        let storage = create_test_storage();
        let vectors = create_test_vectors();

        for (id, vector, metadata) in vectors {
            storage.insert(id, vector, metadata).unwrap();
        }

        let config = PersistenceConfig {
            compression_enabled: true,
            ..Default::default()
        };

        let persistence = VLTPersistence::new(config);

        // Save with compression enabled
        let result = persistence.save_database(
            &file_path,
            &storage,
            None,
            3,
            crate::types::IndexType::BruteForce,
            "euclidean",
        );

        #[cfg(feature = "compression")]
        {
            assert!(result.is_ok());

            // Verify compression flag in header
            let (_, header) = persistence.load_database(&file_path).unwrap();
            assert!(header.compressed);
        }

        #[cfg(not(feature = "compression"))]
        {
            // Should work but data won't actually be compressed
            assert!(result.is_ok());
        }
    }
}
