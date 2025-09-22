// Package veclite provides Go bindings for the VecLite vector search library.
//
// VecLite is a lightweight, embeddable vector search library inspired by SQLite's
// philosophy of being embedded, zero-configuration, and dependency-free.
package veclite

/*
#cgo LDFLAGS: -L../../target/release -lveclite
#cgo CFLAGS: -I../../target/release

#include <stdlib.h>

typedef void* VecLiteHandle;

typedef enum {
    FFI_SUCCESS = 0,
    FFI_NULL_POINTER = 1,
    FFI_INVALID_UTF8 = 2,
    FFI_INVALID_DIMENSIONS = 3,
    FFI_VECTOR_NOT_FOUND = 4,
    FFI_INVALID_METRIC = 5,
    FFI_IO_ERROR = 6,
    FFI_SERIALIZATION_ERROR = 7,
    FFI_UNKNOWN_ERROR = 99,
} FFIErrorCode;

typedef struct {
    FFIErrorCode error_code;
    char* error_message;
} FFIResult;

typedef struct {
    const float* data;
    unsigned int len;
} FFIVector;

typedef struct {
    const char* key;
    const char* value;
} FFIMetadataEntry;

typedef struct {
    const FFIMetadataEntry* entries;
    unsigned int len;
} FFIMetadata;

typedef struct {
    char* id;
    float score;
    FFIMetadata metadata;
} FFISearchResult;

typedef struct {
    FFISearchResult* results;
    unsigned int len;
} FFISearchResults;

// FFI function declarations
VecLiteHandle veclite_new();
VecLiteHandle veclite_new_with_config(unsigned int max_vectors, unsigned int default_k);
void veclite_free(VecLiteHandle handle);

FFIResult veclite_insert(VecLiteHandle handle, const char* id, FFIVector vector, FFIMetadata metadata);
FFIResult veclite_get(VecLiteHandle handle, const char* id, FFIVector* vector, FFIMetadata* metadata);
FFIResult veclite_delete(VecLiteHandle handle, const char* id);
FFIResult veclite_search(VecLiteHandle handle, FFIVector query, unsigned int k, FFISearchResults* results);

unsigned int veclite_len(VecLiteHandle handle);
int veclite_is_empty(VecLiteHandle handle);

FFIResult veclite_save(VecLiteHandle handle, const char* path);
VecLiteHandle veclite_load(const char* path);

void veclite_free_string(char* s);
void veclite_free_search_results(FFISearchResults* results);

FFIResult veclite_get_available_metrics(char*** metrics, unsigned int* count);
*/
import "C"
import (
	"errors"
	"fmt"
	"runtime"
	"unsafe"
)

// Error definitions
var (
	ErrNullPointer      = errors.New("null pointer")
	ErrInvalidUTF8      = errors.New("invalid UTF-8 string")
	ErrInvalidDimension = errors.New("invalid dimensions")
	ErrVectorNotFound   = errors.New("vector not found")
	ErrInvalidMetric    = errors.New("invalid distance metric")
	ErrIO               = errors.New("I/O error")
	ErrSerialization    = errors.New("serialization error")
	ErrUnknown          = errors.New("unknown error")
)

// Vector represents a vector with its data
type Vector []float32

// Metadata represents key-value metadata associated with a vector
type Metadata map[string]string

// SearchResult represents a single search result
type SearchResult struct {
	ID       string
	Score    float32
	Metadata Metadata
}

// VecLite represents a VecLite database instance
type VecLite struct {
	handle C.VecLiteHandle
}

// Config represents configuration options for VecLite
type Config struct {
	MaxVectors uint
	DefaultK   uint
}

// DefaultConfig returns the default configuration
func DefaultConfig() *Config {
	return &Config{
		MaxVectors: 1000000,
		DefaultK:   10,
	}
}

// New creates a new VecLite instance with default configuration
func New() (*VecLite, error) {
	handle := C.veclite_new()
	if handle == nil {
		return nil, ErrUnknown
	}

	db := &VecLite{handle: handle}
	runtime.SetFinalizer(db, (*VecLite).finalize)
	return db, nil
}

// NewWithConfig creates a new VecLite instance with custom configuration
func NewWithConfig(config *Config) (*VecLite, error) {
	if config == nil {
		config = DefaultConfig()
	}

	handle := C.veclite_new_with_config(
		C.uint(config.MaxVectors),
		C.uint(config.DefaultK),
	)
	if handle == nil {
		return nil, ErrUnknown
	}

	db := &VecLite{handle: handle}
	runtime.SetFinalizer(db, (*VecLite).finalize)
	return db, nil
}

// Load loads a VecLite database from a file
func Load(path string) (*VecLite, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	handle := C.veclite_load(cPath)
	if handle == nil {
		return nil, ErrIO
	}

	db := &VecLite{handle: handle}
	runtime.SetFinalizer(db, (*VecLite).finalize)
	return db, nil
}

// Close explicitly frees the VecLite instance
func (v *VecLite) Close() error {
	if v.handle != nil {
		runtime.SetFinalizer(v, nil)
		C.veclite_free(v.handle)
		v.handle = nil
	}
	return nil
}

// finalize is called by the Go runtime finalizer
func (v *VecLite) finalize() {
	v.Close()
}

// Insert inserts a vector with its ID and metadata into the database
func (v *VecLite) Insert(id string, vector Vector, metadata Metadata) error {
	if v.handle == nil {
		return ErrNullPointer
	}

	cID := C.CString(id)
	defer C.free(unsafe.Pointer(cID))

	// Convert vector to C format
	cVector := C.FFIVector{
		data: (*C.float)(unsafe.Pointer(&vector[0])),
		len:  C.uint(len(vector)),
	}

	// Convert metadata to C format
	cMetadata := metadataToC(metadata)
	defer freeMetadata(cMetadata)

	result := C.veclite_insert(v.handle, cID, cVector, cMetadata)
	return resultToError(result)
}

// Get retrieves a vector and its metadata by ID
func (v *VecLite) Get(id string) (Vector, Metadata, error) {
	if v.handle == nil {
		return nil, nil, ErrNullPointer
	}

	cID := C.CString(id)
	defer C.free(unsafe.Pointer(cID))

	var cVector C.FFIVector
	var cMetadata C.FFIMetadata

	result := C.veclite_get(v.handle, cID, &cVector, &cMetadata)
	if err := resultToError(result); err != nil {
		return nil, nil, err
	}

	// Convert C vector to Go
	vector := cVectorToGo(cVector)
	metadata := cMetadataToGo(cMetadata)

	return vector, metadata, nil
}

// Delete deletes a vector by ID
func (v *VecLite) Delete(id string) error {
	if v.handle == nil {
		return ErrNullPointer
	}

	cID := C.CString(id)
	defer C.free(unsafe.Pointer(cID))

	result := C.veclite_delete(v.handle, cID)
	return resultToError(result)
}

// Search searches for k nearest neighbors to the query vector
func (v *VecLite) Search(query Vector, k uint) ([]SearchResult, error) {
	if v.handle == nil {
		return nil, ErrNullPointer
	}

	// Convert query vector to C format
	cQuery := C.FFIVector{
		data: (*C.float)(unsafe.Pointer(&query[0])),
		len:  C.uint(len(query)),
	}

	var cResults C.FFISearchResults
	result := C.veclite_search(v.handle, cQuery, C.uint(k), &cResults)
	if err := resultToError(result); err != nil {
		return nil, err
	}

	// Convert C results to Go
	results := cSearchResultsToGo(cResults)

	// Free C results
	C.veclite_free_search_results(&cResults)

	return results, nil
}

// Len returns the number of vectors in the database
func (v *VecLite) Len() uint {
	if v.handle == nil {
		return 0
	}
	return uint(C.veclite_len(v.handle))
}

// IsEmpty returns true if the database contains no vectors
func (v *VecLite) IsEmpty() bool {
	if v.handle == nil {
		return true
	}
	return C.veclite_is_empty(v.handle) != 0
}

// Save saves the database to a file
func (v *VecLite) Save(path string) error {
	if v.handle == nil {
		return ErrNullPointer
	}

	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	result := C.veclite_save(v.handle, cPath)
	return resultToError(result)
}

// GetAvailableMetrics returns the list of available distance metrics
func GetAvailableMetrics() ([]string, error) {
	var cMetrics **C.char
	var count C.uint

	result := C.veclite_get_available_metrics(&cMetrics, &count)
	if err := resultToError(result); err != nil {
		return nil, err
	}

	// Convert C string array to Go slice
	metrics := make([]string, int(count))
	cMetricSlice := (*[1000]*C.char)(unsafe.Pointer(cMetrics))[:count:count]

	for i, cMetric := range cMetricSlice {
		metrics[i] = C.GoString(cMetric)
		C.veclite_free_string(cMetric)
	}

	C.free(unsafe.Pointer(cMetrics))
	return metrics, nil
}

// Helper functions

// resultToError converts a C FFIResult to a Go error
func resultToError(result C.FFIResult) error {
	if result.error_code == C.FFI_SUCCESS {
		return nil
	}

	var err error
	switch result.error_code {
	case C.FFI_NULL_POINTER:
		err = ErrNullPointer
	case C.FFI_INVALID_UTF8:
		err = ErrInvalidUTF8
	case C.FFI_INVALID_DIMENSIONS:
		err = ErrInvalidDimension
	case C.FFI_VECTOR_NOT_FOUND:
		err = ErrVectorNotFound
	case C.FFI_INVALID_METRIC:
		err = ErrInvalidMetric
	case C.FFI_IO_ERROR:
		err = ErrIO
	case C.FFI_SERIALIZATION_ERROR:
		err = ErrSerialization
	default:
		err = ErrUnknown
	}

	if result.error_message != nil {
		message := C.GoString(result.error_message)
		C.veclite_free_string(result.error_message)
		err = fmt.Errorf("%w: %s", err, message)
	}

	return err
}

// metadataToC converts Go metadata to C format
func metadataToC(metadata Metadata) C.FFIMetadata {
	if len(metadata) == 0 {
		return C.FFIMetadata{
			entries: nil,
			len:     0,
		}
	}

	entries := make([]C.FFIMetadataEntry, 0, len(metadata))
	for key, value := range metadata {
		cKey := C.CString(key)
		cValue := C.CString(value)
		entries = append(entries, C.FFIMetadataEntry{
			key:   cKey,
			value: cValue,
		})
	}

	return C.FFIMetadata{
		entries: (*C.FFIMetadataEntry)(unsafe.Pointer(&entries[0])),
		len:     C.uint(len(entries)),
	}
}

// freeMetadata frees C metadata entries
func freeMetadata(cMetadata C.FFIMetadata) {
	if cMetadata.entries == nil || cMetadata.len == 0 {
		return
	}

	entries := (*[1000]C.FFIMetadataEntry)(unsafe.Pointer(cMetadata.entries))[:cMetadata.len:cMetadata.len]
	for _, entry := range entries {
		C.free(unsafe.Pointer(entry.key))
		C.free(unsafe.Pointer(entry.value))
	}
}

// cVectorToGo converts C vector to Go format
func cVectorToGo(cVector C.FFIVector) Vector {
	if cVector.data == nil || cVector.len == 0 {
		return nil
	}

	data := (*[1000000]C.float)(unsafe.Pointer(cVector.data))[:cVector.len:cVector.len]
	vector := make(Vector, len(data))
	for i, v := range data {
		vector[i] = float32(v)
	}
	return vector
}

// cMetadataToGo converts C metadata to Go format
func cMetadataToGo(cMetadata C.FFIMetadata) Metadata {
	if cMetadata.entries == nil || cMetadata.len == 0 {
		return make(Metadata)
	}

	entries := (*[1000]C.FFIMetadataEntry)(unsafe.Pointer(cMetadata.entries))[:cMetadata.len:cMetadata.len]
	metadata := make(Metadata, len(entries))
	for _, entry := range entries {
		key := C.GoString(entry.key)
		value := C.GoString(entry.value)
		metadata[key] = value
	}
	return metadata
}

// cSearchResultsToGo converts C search results to Go format
func cSearchResultsToGo(cResults C.FFISearchResults) []SearchResult {
	if cResults.results == nil || cResults.len == 0 {
		return nil
	}

	cResultsSlice := (*[1000]C.FFISearchResult)(unsafe.Pointer(cResults.results))[:cResults.len:cResults.len]
	results := make([]SearchResult, len(cResultsSlice))

	for i, cResult := range cResultsSlice {
		results[i] = SearchResult{
			ID:       C.GoString(cResult.id),
			Score:    float32(cResult.score),
			Metadata: cMetadataToGo(cResult.metadata),
		}
	}

	return results
}