use thiserror::Error;

/// Primary result type for all VecLite operations
/// NEVER create new Result types - always use this unified type
pub type VecLiteResult<T> = Result<T, VecLiteError>;

/// Specialized result type for FFI operations
/// Used only for cross-language boundaries
pub type FFIResult<T> = Result<T, FFIError>;

/// Main error type for VecLite operations
/// Follows hierarchy from COMMON_CODE_GUIDE.md
#[derive(Debug, Error)]
pub enum VecLiteError {
    #[error("Storage error: {0}")]
    Storage(#[from] StorageError),

    #[error("Index error: {0}")]
    Index(#[from] IndexError),

    #[error("Query error: {0}")]
    Query(#[from] QueryError),

    #[error("Persistence error: {0}")]
    Persistence(#[from] PersistenceError),

    #[error("Configuration error: {0}")]
    Config(#[from] ConfigError),
}

/// Storage-related errors
#[derive(Debug, Error)]
pub enum StorageError {
    #[error("Vector not found: {id}")]
    VectorNotFound { id: String },

    #[error("Invalid dimensions: expected {expected}, got {actual}")]
    InvalidDimensions { expected: usize, actual: usize },

    #[error("Memory limit exceeded: {current} bytes")]
    MemoryLimitExceeded { current: usize },

    #[error("Invalid vector value at index {index}: {value}")]
    InvalidValue { index: usize, value: f32 },

    #[error("Vector ID already exists: {id}")]
    VectorAlreadyExists { id: String },

    #[error("Empty vector data")]
    EmptyVector,

    #[error("Metadata too large: {size} bytes (max: {max_size})")]
    MetadataTooLarge { size: usize, max_size: usize },
}

/// Index-related errors
#[derive(Debug, Error)]
pub enum IndexError {
    #[error("Index build failed: {reason}")]
    BuildFailed { reason: String },

    #[error("Index not initialized")]
    NotInitialized,

    #[error("Index corruption detected: {details}")]
    Corrupted { details: String },

    #[error("Insert failed: {reason}")]
    InsertFailed { reason: String },

    #[error("Delete failed: vector {id} not in index")]
    DeleteFailed { id: String },
}

/// Query-related errors
#[derive(Debug, Error)]
pub enum QueryError {
    #[error("Invalid k value: {k} (must be > 0 and <= {max})")]
    InvalidK { k: usize, max: usize },

    #[error("Query vector dimensions mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Empty dataset: cannot perform search")]
    EmptyDataset,

    #[error("Query failed: {reason}")]
    QueryFailed { reason: String },

    #[error("Unsupported distance metric: {metric}")]
    UnsupportedMetric { metric: String },
}

/// Persistence-related errors
#[derive(Debug, Error)]
pub enum PersistenceError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] bincode::Error),

    #[error("Invalid file format: {reason}")]
    InvalidFormat { reason: String },

    #[error("File format version mismatch: expected {expected}, got {actual}")]
    VersionMismatch { expected: u32, actual: u32 },

    #[error("Checksum mismatch: file may be corrupted")]
    ChecksumMismatch,

    #[error("Compression error: {0}")]
    Compression(String),

    #[error("File not found: {path}")]
    FileNotFound { path: String },
}

/// Configuration-related errors
#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("Invalid configuration: {field} = {value} (reason: {reason})")]
    InvalidValue {
        field: String,
        value: String,
        reason: String,
    },

    #[error("Missing required configuration: {field}")]
    MissingField { field: String },

    #[error("Configuration validation failed: {errors:?}")]
    ValidationFailed { errors: Vec<String> },
}

/// FFI-specific errors for cross-language boundaries
#[derive(Debug, Error)]
pub enum FFIError {
    #[error("VecLite error: {0}")]
    VecLite(#[from] VecLiteError),

    #[error("Null pointer error: {context}")]
    NullPointer { context: String },

    #[error("Memory allocation failed: {size} bytes")]
    MemoryAllocation { size: usize },

    #[error("String conversion error: {0}")]
    StringConversion(#[from] std::str::Utf8Error),

    #[error("Handle not found: {handle_id}")]
    HandleNotFound { handle_id: u64 },

    #[error("Thread safety violation: {context}")]
    ThreadSafety { context: String },
}

impl VecLiteError {
    /// Check if error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            VecLiteError::Storage(e) => match e {
                StorageError::VectorNotFound { .. } => true,
                StorageError::VectorAlreadyExists { .. } => true,
                StorageError::InvalidDimensions { .. } => false,
                StorageError::MemoryLimitExceeded { .. } => false,
                _ => false,
            },
            VecLiteError::Query(e) => match e {
                QueryError::InvalidK { .. } => true,
                QueryError::EmptyDataset => true,
                QueryError::DimensionMismatch { .. } => false,
                _ => false,
            },
            VecLiteError::Index(_) => false,
            VecLiteError::Persistence(e) => match e {
                PersistenceError::FileNotFound { .. } => true,
                PersistenceError::ChecksumMismatch => false,
                _ => false,
            },
            VecLiteError::Config(_) => false,
        }
    }

    /// Get error category for logging and metrics
    pub fn category(&self) -> &'static str {
        match self {
            VecLiteError::Storage(_) => "storage",
            VecLiteError::Index(_) => "index",
            VecLiteError::Query(_) => "query",
            VecLiteError::Persistence(_) => "persistence",
            VecLiteError::Config(_) => "configuration",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_error_creation() {
        let error = StorageError::VectorNotFound {
            id: "test_id".to_string(),
        };
        assert_eq!(error.to_string(), "Vector not found: test_id");
    }

    #[test]
    fn test_error_hierarchy() {
        let storage_error = StorageError::InvalidDimensions {
            expected: 768,
            actual: 512,
        };
        let vec_error: VecLiteError = storage_error.into();

        assert!(matches!(vec_error, VecLiteError::Storage(_)));
        assert!(!vec_error.is_recoverable());
        assert_eq!(vec_error.category(), "storage");
    }

    #[test]
    fn test_result_type_usage() {
        fn example_function() -> VecLiteResult<String> {
            Err(VecLiteError::Storage(StorageError::EmptyVector))
        }

        let result = example_function();
        assert!(result.is_err());
    }

    #[test]
    fn test_error_recoverability() {
        let recoverable = VecLiteError::Storage(StorageError::VectorNotFound {
            id: "missing".to_string(),
        });
        assert!(recoverable.is_recoverable());

        let non_recoverable = VecLiteError::Storage(StorageError::InvalidDimensions {
            expected: 768,
            actual: 512,
        });
        assert!(!non_recoverable.is_recoverable());
    }

    #[test]
    fn test_ffi_error_conversion() {
        let vec_error = VecLiteError::Storage(StorageError::EmptyVector);
        let ffi_error: FFIError = vec_error.into();

        assert!(matches!(ffi_error, FFIError::VecLite(_)));
    }
}
