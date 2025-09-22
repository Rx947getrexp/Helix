//! Foreign Function Interface (FFI) for VecLite
//!
//! This module provides C-compatible bindings for VecLite, enabling usage from
//! other languages like Go, Python, C++, etc.

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float, c_int, c_uint, c_void};
use std::ptr;
use std::slice;

use crate::types::{Metadata, SearchResult, VectorData};
use crate::{Helix, VecLiteConfig};

/// FFI Error codes
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FFIErrorCode {
    Success = 0,
    NullPointer = 1,
    InvalidUtf8 = 2,
    InvalidDimensions = 3,
    VectorNotFound = 4,
    InvalidMetric = 5,
    IoError = 6,
    SerializationError = 7,
    UnknownError = 99,
}

/// FFI result wrapper
#[repr(C)]
pub struct FFIResult {
    pub error_code: FFIErrorCode,
    pub error_message: *mut c_char,
}

/// FFI Vector structure
#[repr(C)]
pub struct FFIVector {
    pub data: *const c_float,
    pub len: c_uint,
}

/// FFI Metadata entry
#[repr(C)]
pub struct FFIMetadataEntry {
    pub key: *const c_char,
    pub value: *const c_char,
}

/// FFI Metadata structure
#[repr(C)]
pub struct FFIMetadata {
    pub entries: *const FFIMetadataEntry,
    pub len: c_uint,
}

/// FFI Search Result
#[repr(C)]
pub struct FFISearchResult {
    pub id: *mut c_char,
    pub score: c_float,
    pub metadata: FFIMetadata,
}

/// FFI Search Results Array
#[repr(C)]
pub struct FFISearchResults {
    pub results: *mut FFISearchResult,
    pub len: c_uint,
}

/// Opaque handle for Helix instance
pub type VecLiteHandle = *mut c_void;

impl FFIResult {
    fn success() -> Self {
        Self {
            error_code: FFIErrorCode::Success,
            error_message: ptr::null_mut(),
        }
    }

    fn error(code: FFIErrorCode, message: &str) -> Self {
        let c_message = CString::new(message).unwrap_or_default();
        Self {
            error_code: code,
            error_message: c_message.into_raw(),
        }
    }
}

/// Convert Rust string to C string
fn string_to_c_char(s: String) -> *mut c_char {
    CString::new(s).unwrap_or_default().into_raw()
}

/// Convert C string to Rust string
fn c_char_to_string(c_str: *const c_char) -> Result<String, FFIErrorCode> {
    if c_str.is_null() {
        return Err(FFIErrorCode::NullPointer);
    }

    // SAFETY: We've checked that c_str is not null above.
    // CStr::from_ptr is safe when the pointer is valid and null-terminated.
    unsafe {
        CStr::from_ptr(c_str)
            .to_str()
            .map(|s| s.to_string())
            .map_err(|_| FFIErrorCode::InvalidUtf8)
    }
}

/// Convert FFIVector to VectorData
fn ffi_vector_to_vector_data(ffi_vec: FFIVector) -> Result<VectorData, FFIErrorCode> {
    if ffi_vec.data.is_null() {
        return Err(FFIErrorCode::NullPointer);
    }

    // SAFETY: We've checked that ffi_vec.data is not null above.
    // slice::from_raw_parts is safe when the pointer is valid and the length is correct.
    unsafe {
        let slice = slice::from_raw_parts(ffi_vec.data, ffi_vec.len as usize);
        Ok(slice.to_vec())
    }
}

/// Convert FFIMetadata to Metadata
fn ffi_metadata_to_metadata(ffi_meta: FFIMetadata) -> Result<Metadata, FFIErrorCode> {
    if ffi_meta.entries.is_null() && ffi_meta.len > 0 {
        return Err(FFIErrorCode::NullPointer);
    }

    let mut metadata = HashMap::new();

    if ffi_meta.len > 0 {
        // SAFETY: We've validated that entries is not null when len > 0.
        // slice::from_raw_parts is safe when pointer is valid and length is correct.
        unsafe {
            let entries = slice::from_raw_parts(ffi_meta.entries, ffi_meta.len as usize);
            for entry in entries {
                let key = c_char_to_string(entry.key)?;
                let value = c_char_to_string(entry.value)?;
                metadata.insert(key, value);
            }
        }
    }

    Ok(metadata)
}

/// Convert SearchResult to FFISearchResult
fn search_result_to_ffi(result: SearchResult) -> FFISearchResult {
    let metadata_entries: Vec<FFIMetadataEntry> = result
        .metadata
        .iter()
        .map(|(k, v)| FFIMetadataEntry {
            key: string_to_c_char(k.clone()),
            value: string_to_c_char(v.clone()),
        })
        .collect();

    let metadata = FFIMetadata {
        entries: metadata_entries.as_ptr(),
        len: metadata_entries.len() as c_uint,
    };

    // Leak the metadata_entries to prevent deallocation
    std::mem::forget(metadata_entries);

    FFISearchResult {
        id: string_to_c_char(result.id),
        score: result.score,
        metadata,
    }
}

/// Create a new Helix instance
///
/// # Safety
/// This function is safe to call and doesn't dereference any pointers.
#[no_mangle]
pub unsafe extern "C" fn veclite_new() -> VecLiteHandle {
    match Helix::new() {
        Ok(db) => Box::into_raw(Box::new(db)) as VecLiteHandle,
        Err(_) => ptr::null_mut(),
    }
}

/// Create a new Helix instance with custom configuration
///
/// # Safety
/// This function is safe to call and doesn't dereference any pointers.
#[no_mangle]
pub unsafe extern "C" fn veclite_new_with_config(
    max_vectors: c_uint,
    default_k: c_uint,
) -> VecLiteHandle {
    let mut config = VecLiteConfig::default();
    config.storage.max_vectors = max_vectors as usize;
    config.query.default_k = default_k as usize;

    match Helix::with_config(config) {
        Ok(db) => Box::into_raw(Box::new(db)) as VecLiteHandle,
        Err(_) => ptr::null_mut(),
    }
}

/// Free a Helix instance
///
/// # Safety
/// Caller must ensure handle is a valid VecLiteHandle returned from veclite_new.
#[no_mangle]
pub unsafe extern "C" fn veclite_free(handle: VecLiteHandle) {
    if !handle.is_null() {
        // SAFETY: handle came from Box::into_raw in veclite_new, so it's valid to convert back
        let _db = unsafe { Box::from_raw(handle as *mut Helix) };
        // Automatic cleanup when Box is dropped
    }
}

/// Insert a vector into the database
///
/// # Safety
/// Caller must ensure all pointers are valid and handle is from veclite_new.
#[no_mangle]
pub unsafe extern "C" fn veclite_insert(
    handle: VecLiteHandle,
    id: *const c_char,
    vector: FFIVector,
    metadata: FFIMetadata,
) -> FFIResult {
    if handle.is_null() {
        return FFIResult::error(FFIErrorCode::NullPointer, "VecLite handle is null");
    }

    let db = &*(handle as *const Helix);

    let vector_id = match c_char_to_string(id) {
        Ok(s) => s,
        Err(code) => return FFIResult::error(code, "Invalid ID string"),
    };

    let vector_data = match ffi_vector_to_vector_data(vector) {
        Ok(v) => v,
        Err(code) => return FFIResult::error(code, "Invalid vector data"),
    };

    let metadata_map = match ffi_metadata_to_metadata(metadata) {
        Ok(m) => m,
        Err(code) => return FFIResult::error(code, "Invalid metadata"),
    };

    match db.insert(vector_id, vector_data, metadata_map) {
        Ok(_) => FFIResult::success(),
        Err(e) => FFIResult::error(FFIErrorCode::UnknownError, &format!("{}", e)),
    }
}

/// Get a vector from the database
///
/// # Safety
/// Caller must ensure all pointers are valid and handle is from veclite_new.
#[no_mangle]
pub unsafe extern "C" fn veclite_get(
    handle: VecLiteHandle,
    id: *const c_char,
    vector: *mut FFIVector,
    metadata: *mut FFIMetadata,
) -> FFIResult {
    if handle.is_null() {
        return FFIResult::error(FFIErrorCode::NullPointer, "VecLite handle is null");
    }

    let db = &*(handle as *const Helix);

    let vector_id = match c_char_to_string(id) {
        Ok(s) => s,
        Err(code) => return FFIResult::error(code, "Invalid ID string"),
    };

    match db.get(&vector_id) {
        Ok(Some(item)) => {
            if !vector.is_null() {
                let vec_data = item.vector.into_boxed_slice();
                (*vector).data = vec_data.as_ptr();
                (*vector).len = vec_data.len() as c_uint;
                std::mem::forget(vec_data); // Prevent deallocation
            }

            if !metadata.is_null() {
                let metadata_entries: Vec<FFIMetadataEntry> = item
                    .metadata
                    .iter()
                    .map(|(k, v)| FFIMetadataEntry {
                        key: string_to_c_char(k.clone()),
                        value: string_to_c_char(v.clone()),
                    })
                    .collect();

                (*metadata).entries = metadata_entries.as_ptr();
                (*metadata).len = metadata_entries.len() as c_uint;
                std::mem::forget(metadata_entries);
            }

            FFIResult::success()
        }
        Ok(None) => FFIResult::error(FFIErrorCode::VectorNotFound, "Vector not found"),
        Err(e) => FFIResult::error(FFIErrorCode::UnknownError, &format!("{}", e)),
    }
}

/// Delete a vector from the database
///
/// # Safety
/// Caller must ensure handle and id pointers are valid.
#[no_mangle]
pub unsafe extern "C" fn veclite_delete(handle: VecLiteHandle, id: *const c_char) -> FFIResult {
    if handle.is_null() {
        return FFIResult::error(FFIErrorCode::NullPointer, "VecLite handle is null");
    }

    let db = &*(handle as *const Helix);

    let vector_id = match c_char_to_string(id) {
        Ok(s) => s,
        Err(code) => return FFIResult::error(code, "Invalid ID string"),
    };

    match db.delete(&vector_id) {
        Ok(true) => FFIResult::success(),
        Ok(false) => FFIResult::error(FFIErrorCode::VectorNotFound, "Vector not found"),
        Err(e) => FFIResult::error(FFIErrorCode::UnknownError, &format!("{}", e)),
    }
}

/// Search for similar vectors
///
/// # Safety
/// Caller must ensure all pointers are valid and handle is from veclite_new.
#[no_mangle]
pub unsafe extern "C" fn veclite_search(
    handle: VecLiteHandle,
    query: FFIVector,
    k: c_uint,
    results: *mut FFISearchResults,
) -> FFIResult {
    if handle.is_null() {
        return FFIResult::error(FFIErrorCode::NullPointer, "VecLite handle is null");
    }

    if results.is_null() {
        return FFIResult::error(FFIErrorCode::NullPointer, "Results pointer is null");
    }

    let db = &*(handle as *const Helix);

    let query_vector = match ffi_vector_to_vector_data(query) {
        Ok(v) => v,
        Err(code) => return FFIResult::error(code, "Invalid query vector"),
    };

    match db.search(&query_vector, k as usize) {
        Ok(search_results) => {
            let ffi_results: Vec<FFISearchResult> = search_results
                .into_iter()
                .map(search_result_to_ffi)
                .collect();

            (*results).results = ffi_results.as_ptr() as *mut FFISearchResult;
            (*results).len = ffi_results.len() as c_uint;
            std::mem::forget(ffi_results);

            FFIResult::success()
        }
        Err(e) => FFIResult::error(FFIErrorCode::UnknownError, &format!("{}", e)),
    }
}

/// Get database statistics
///
/// # Safety
/// Caller must ensure handle is valid and from veclite_new.
#[no_mangle]
pub unsafe extern "C" fn veclite_len(handle: VecLiteHandle) -> c_uint {
    if handle.is_null() {
        return 0;
    }

    let db = &*(handle as *const Helix);
    db.len() as c_uint
}

/// Check if database is empty
///
/// # Safety
/// Caller must ensure handle is valid and from veclite_new.
#[no_mangle]
pub unsafe extern "C" fn veclite_is_empty(handle: VecLiteHandle) -> c_int {
    if handle.is_null() {
        return 1; // Consider null handle as "empty"
    }

    let db = &*(handle as *const Helix);
    if db.is_empty() {
        1
    } else {
        0
    }
}

/// Save database to file
///
/// # Safety
/// Caller must ensure handle and path pointers are valid.
#[no_mangle]
pub unsafe extern "C" fn veclite_save(handle: VecLiteHandle, path: *const c_char) -> FFIResult {
    if handle.is_null() {
        return FFIResult::error(FFIErrorCode::NullPointer, "VecLite handle is null");
    }

    let db = &*(handle as *const Helix);

    let file_path = match c_char_to_string(path) {
        Ok(s) => s,
        Err(code) => return FFIResult::error(code, "Invalid path string"),
    };

    match db.save(file_path) {
        Ok(_) => FFIResult::success(),
        Err(e) => FFIResult::error(FFIErrorCode::IoError, &format!("{}", e)),
    }
}

/// Load database from file
///
/// # Safety
/// Caller must ensure path pointer is valid.
#[no_mangle]
pub unsafe extern "C" fn veclite_load(path: *const c_char) -> VecLiteHandle {
    let file_path = match c_char_to_string(path) {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    match Helix::load(file_path) {
        Ok(db) => Box::into_raw(Box::new(db)) as VecLiteHandle,
        Err(_) => ptr::null_mut(),
    }
}

/// Free C string allocated by this library
///
/// # Safety
/// Caller must ensure s was allocated by this library or is null.
#[no_mangle]
pub unsafe extern "C" fn veclite_free_string(s: *mut c_char) {
    if !s.is_null() {
        // SAFETY: s was allocated by this library using CString::into_raw
        let _ = unsafe { CString::from_raw(s) };
    }
}

/// Free search results
///
/// # Safety
/// Caller must ensure results was allocated by this library or is null.
#[no_mangle]
pub unsafe extern "C" fn veclite_free_search_results(results: *mut FFISearchResults) {
    if results.is_null() {
        return;
    }

    let results_ref = unsafe { &mut *results };
    if !results_ref.results.is_null() {
        let results_slice =
            unsafe { slice::from_raw_parts_mut(results_ref.results, results_ref.len as usize) };

        for result in &mut *results_slice {
            veclite_free_string(result.id);

            // Free metadata entries
            if !result.metadata.entries.is_null() {
                let metadata_slice = unsafe {
                    slice::from_raw_parts_mut(
                        result.metadata.entries as *mut FFIMetadataEntry,
                        result.metadata.len as usize,
                    )
                };

                for entry in &mut *metadata_slice {
                    veclite_free_string(entry.key as *mut c_char);
                    veclite_free_string(entry.value as *mut c_char);
                }

                let _ = unsafe { Box::from_raw(metadata_slice.as_mut_ptr()) };
            }
        }

        let _ = unsafe { Box::from_raw(results_slice.as_mut_ptr()) };
    }

    results_ref.results = ptr::null_mut();
    results_ref.len = 0;
}

/// Get available distance metrics
///
/// # Safety
/// Caller must ensure metrics and count pointers are valid.
#[no_mangle]
pub unsafe extern "C" fn veclite_get_available_metrics(
    metrics: *mut *mut *mut c_char,
    count: *mut c_uint,
) -> FFIResult {
    if metrics.is_null() || count.is_null() {
        return FFIResult::error(FFIErrorCode::NullPointer, "Output pointers are null");
    }

    let available = Helix::available_metrics();
    let c_strings: Vec<*mut c_char> = available
        .into_iter()
        .map(|s| string_to_c_char(s.to_string()))
        .collect();

    unsafe {
        *count = c_strings.len() as c_uint;
        let boxed = c_strings.into_boxed_slice();
        *metrics = Box::into_raw(boxed) as *mut *mut c_char;
    }

    FFIResult::success()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffi_create_and_free() {
        unsafe {
            let handle = veclite_new();
            assert!(!handle.is_null());
            veclite_free(handle);
        }
    }

    #[test]
    fn test_ffi_insert_and_get() {
        unsafe {
            let handle = veclite_new();
            assert!(!handle.is_null());

            // Prepare test data
            let id_str = CString::new("test_vector").unwrap();
            let vector_data = [1.0, 2.0, 3.0];
            let ffi_vector = FFIVector {
                data: vector_data.as_ptr(),
                len: vector_data.len() as c_uint,
            };

            // Create empty metadata
            let ffi_metadata = FFIMetadata {
                entries: ptr::null(),
                len: 0,
            };

            // Insert vector
            let result = veclite_insert(handle, id_str.as_ptr(), ffi_vector, ffi_metadata);
            assert_eq!(result.error_code, FFIErrorCode::Success);

            // Check length
            assert_eq!(veclite_len(handle), 1);
            assert_eq!(veclite_is_empty(handle), 0);

            veclite_free(handle);
        }
    }

    #[test]
    fn test_ffi_error_codes() {
        unsafe {
            // Test null handle
            let result = veclite_insert(
                ptr::null_mut(),
                ptr::null(),
                FFIVector {
                    data: ptr::null(),
                    len: 0,
                },
                FFIMetadata {
                    entries: ptr::null(),
                    len: 0,
                },
            );
            assert_eq!(result.error_code, FFIErrorCode::NullPointer);

            // Test null ID
            let handle = veclite_new();
            let result = veclite_insert(
                handle,
                ptr::null(),
                FFIVector {
                    data: ptr::null(),
                    len: 0,
                },
                FFIMetadata {
                    entries: ptr::null(),
                    len: 0,
                },
            );
            assert_eq!(result.error_code, FFIErrorCode::NullPointer);

            veclite_free(handle);
        }
    }
}
