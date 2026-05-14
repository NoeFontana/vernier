//! Convert any Python-side manifest input to canonical JSON bytes
//! (ADR-0046).
//!
//! Accepts four shapes:
//!
//! - `dict` (canonical JSON-records shape) — re-serialized to JSON
//!   via the Python stdlib `json` module.
//! - `str` / `PathLike` (file path; `.json` only — CSV is rejected
//!   here pending a `vernier-core` CSV adapter, callers get a
//!   typed error directing them to the JSON form).
//! - Any object exposing the Arrow PyCapsule Interface
//!   (`__arrow_c_array__` or `__arrow_c_stream__`) — a polars /
//!   pandas / duckdb / pyarrow DataFrame passes straight in.
//!
//! The Arrow path needs a `key_kind` discriminator that isn't carried
//! in the table itself; callers pass an explicit `key_kind=` kwarg on
//! the wrapping pyfunction, defaulting to `"image_id"` for the common
//! image-keyed manifest case.

use std::fs;
use std::path::PathBuf;

use arrow_array::ffi::{from_ffi, FFI_ArrowArray, FFI_ArrowSchema};
use arrow_array::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use arrow_array::{
    array::{
        Int32Array, Int64Array, LargeStringArray, StringArray, StringViewArray, UInt32Array,
        UInt64Array,
    },
    Array, RecordBatch, StructArray,
};
use arrow_schema::DataType;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyDict, PyString};

/// Build canonical JSON-bytes for a manifest input.
///
/// `manifest` is any of: a Python `dict`, a file path (str /
/// PathLike), or an Arrow PyCapsule object. `key_kind` is consumed
/// only for the Arrow path (where the wire shape carries no
/// discriminator); the dict and file-path forms ignore it because the
/// canonical JSON already names a `key_kind`.
pub(crate) fn manifest_to_canonical_json(
    py: Python<'_>,
    manifest: &Bound<'_, PyAny>,
    key_kind: &str,
) -> PyResult<Vec<u8>> {
    // 1) `dict` — round-trip through `json.dumps` to get canonical
    //    bytes the core parser already understands.
    if let Ok(d) = manifest.cast::<PyDict>() {
        return dict_to_json_bytes(py, d);
    }

    // 2) File path. Accept both `str` and any `os.PathLike` via
    //    `os.fspath`. Only `.json` is wired today; `.csv` is rejected
    //    with a typed message pointing at the canonical JSON form
    //    (the core-side CSV adapter is a follow-up).
    if let Some(bytes) = try_load_path(py, manifest)? {
        return Ok(bytes);
    }

    // 3) Arrow PyCapsule Interface — stream form first, then array.
    if manifest.hasattr("__arrow_c_stream__")? {
        return arrow_stream_to_json_bytes(py, manifest, key_kind);
    }
    if manifest.hasattr("__arrow_c_array__")? {
        return arrow_array_to_json_bytes(py, manifest, key_kind);
    }

    Err(PyTypeError::new_err(
        "manifest= must be one of: a dict, a file path (str / PathLike), \
         or an object exposing the Arrow PyCapsule Interface \
         (__arrow_c_array__ / __arrow_c_stream__)",
    ))
}

/// Re-emit a Python dict as canonical UTF-8 JSON bytes.
fn dict_to_json_bytes(py: Python<'_>, d: &Bound<'_, PyDict>) -> PyResult<Vec<u8>> {
    let json_mod = py.import("json")?;
    let dumps = json_mod.getattr("dumps")?;
    let s = dumps.call1((d,))?;
    let s: &str = s.cast::<PyString>()?.to_str()?;
    Ok(s.as_bytes().to_vec())
}

/// If `manifest` looks like a path (str / PathLike), open it and
/// return its bytes; otherwise return `Ok(None)` so the caller can
/// try the next path.
fn try_load_path(py: Python<'_>, manifest: &Bound<'_, PyAny>) -> PyResult<Option<Vec<u8>>> {
    // str is a fast common path.
    let path_str: Option<String> = if let Ok(s) = manifest.cast::<PyString>() {
        Some(s.to_str()?.to_owned())
    } else {
        // PathLike: try `os.fspath`. Anything that fails here is
        // simply not a path — fall through to the Arrow path.
        let os = py.import("os")?;
        match os.getattr("fspath")?.call1((manifest,)) {
            Ok(p) => match p.cast::<PyString>() {
                Ok(s) => Some(s.to_str()?.to_owned()),
                Err(_) => None,
            },
            Err(_) => None,
        }
    };
    let Some(path) = path_str else {
        return Ok(None);
    };
    let path = PathBuf::from(path);
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase);
    match ext.as_deref() {
        Some("json") => fs::read(&path)
            .map(Some)
            .map_err(|e| PyValueError::new_err(format!("manifest file read failed: {e}"))),
        Some("csv") => Err(PyValueError::new_err(
            "manifest CSV ingest is a follow-up; convert to the canonical \
             JSON shape (`{{\"manifest_version\": \"1\", \"key_kind\": ..., \
             \"rows\": [...]}}`) for now",
        )),
        Some(other) => Err(PyValueError::new_err(format!(
            "unknown manifest file extension .{other}; expected .json"
        ))),
        None => Err(PyValueError::new_err(
            "manifest file path is missing an extension; expected .json",
        )),
    }
}

/// Consume an `__arrow_c_array__`-exposing object as one
/// `RecordBatch` and project to canonical manifest JSON bytes.
fn arrow_array_to_json_bytes(
    py: Python<'_>,
    manifest: &Bound<'_, PyAny>,
    key_kind: &str,
) -> PyResult<Vec<u8>> {
    let tuple = manifest.call_method0("__arrow_c_array__")?;
    let (schema_cap, array_cap): (Bound<'_, PyCapsule>, Bound<'_, PyCapsule>) = tuple.extract()?;

    // SAFETY: the producer hands us live FFI structs whose ownership
    // we are about to take by moving them out of the capsules. We
    // reset the capsules' destructors to no-op so they don't double-
    // free when Python drops them — the FFI struct's `Drop` (via the
    // C-Data-Interface `release` callback) is the canonical free.
    let array = unsafe { take_capsule::<FFI_ArrowArray>(&array_cap)? };
    let schema = unsafe { take_capsule::<FFI_ArrowSchema>(&schema_cap)? };

    let array_data = unsafe { from_ffi(array, &schema) }
        .map_err(|e| PyValueError::new_err(format!("arrow ffi import failed: {e}")))?;
    let struct_array = StructArray::from(array_data);
    let batch = RecordBatch::from(struct_array);
    let _ = py;
    record_batches_to_json_bytes(&[batch], key_kind)
}

/// Consume an `__arrow_c_stream__`-exposing object as a sequence of
/// `RecordBatch`es and project to canonical manifest JSON bytes.
fn arrow_stream_to_json_bytes(
    py: Python<'_>,
    manifest: &Bound<'_, PyAny>,
    key_kind: &str,
) -> PyResult<Vec<u8>> {
    let cap_any = manifest.call_method0("__arrow_c_stream__")?;
    let cap: Bound<'_, PyCapsule> = cap_any.extract()?;
    let stream = unsafe { take_capsule::<FFI_ArrowArrayStream>(&cap)? };
    let reader = ArrowArrayStreamReader::try_new(stream)
        .map_err(|e| PyValueError::new_err(format!("arrow stream import failed: {e}")))?;
    let mut batches: Vec<RecordBatch> = Vec::new();
    for batch in reader {
        let b =
            batch.map_err(|e| PyValueError::new_err(format!("arrow stream read failed: {e}")))?;
        batches.push(b);
    }
    let _ = py;
    record_batches_to_json_bytes(&batches, key_kind)
}

/// Materialize an Arrow FFI struct out of a PyCapsule, neutering the
/// capsule's destructor so the producer's `release` callback runs
/// exactly once (when our owned `T` drops).
///
/// # Safety
///
/// The capsule must contain a valid pointer to a `T` produced by an
/// Arrow C-Data-Interface producer, and the caller must take
/// ownership of the `T` exactly once. The capsule's destructor is
/// reset to `None` here so the same `T` isn't released twice — the
/// owned `T` is now responsible for invoking the C-Data-Interface
/// `release` callback on drop, and the capsule's own destructor must
/// not also invoke it.
unsafe fn take_capsule<T: Send + 'static>(cap: &Bound<'_, PyCapsule>) -> PyResult<T> {
    // Read the capsule's *own* name and pass it back to
    // `PyCapsule_GetPointer`; that function refuses to return a
    // pointer when its `name` argument doesn't match the capsule's
    // stored name. The Arrow C-Data-Interface producers tag their
    // capsules `"arrow_schema"` / `"arrow_array"` /
    // `"arrow_array_stream"` — passing `null` would return a null
    // pointer on every implementation that names its capsules.
    let name = unsafe { ffi::PyCapsule_GetName(cap.as_ptr()) };
    if name.is_null() {
        unsafe { ffi::PyErr_Clear() };
    }
    let ptr = unsafe { ffi::PyCapsule_GetPointer(cap.as_ptr(), name) } as *mut T;
    if ptr.is_null() {
        return Err(PyValueError::new_err(
            "arrow PyCapsule pointer is null; producer is malformed",
        ));
    }
    // Move the value out (the FFI struct's `Drop` will release).
    let value = unsafe { std::ptr::read(ptr) };
    // Disarm the capsule's destructor so the producer's `release`
    // callback isn't double-invoked.
    let rc = unsafe { ffi::PyCapsule_SetDestructor(cap.as_ptr(), None) };
    if rc != 0 {
        return Err(PyErr::fetch(cap.py()));
    }
    Ok(value)
}

/// Walk one or more record batches and emit canonical manifest JSON.
fn record_batches_to_json_bytes(batches: &[RecordBatch], key_kind: &str) -> PyResult<Vec<u8>> {
    if batches.is_empty() {
        // Empty manifest: still well-formed.
        let doc = serde_json::json!({
            "manifest_version": "1",
            "key_kind": key_kind,
            "rows": [],
        });
        return serde_json::to_vec(&doc)
            .map_err(|e| PyValueError::new_err(format!("manifest serialize failed: {e}")));
    }
    let schema = batches[0].schema();
    let column_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    // The first column doesn't have to be `key` literally; require
    // a column named `key` because the core parser does. Producers
    // (polars / pandas) preserve column names, so requiring it is
    // robust and produces a targeted error otherwise.
    if !column_names.contains(&"key") {
        return Err(PyValueError::new_err(
            "Arrow manifest must carry a column named `key`; \
             rename your image-id / label column to `key`",
        ));
    }
    let mut rows: Vec<serde_json::Value> = Vec::new();
    for batch in batches {
        if !batches_share_schema(batches[0].schema_ref(), batch.schema_ref()) {
            return Err(PyValueError::new_err(
                "Arrow manifest stream emitted batches with differing schemas; \
                 manifests must be uniformly columnar",
            ));
        }
        let n_rows = batch.num_rows();
        let n_cols = batch.num_columns();
        let columns: Vec<&dyn Array> = (0..n_cols).map(|i| batch.column(i).as_ref()).collect();
        for row in 0..n_rows {
            let mut obj = serde_json::Map::with_capacity(n_cols);
            for (col_idx, name) in column_names.iter().enumerate() {
                let value = column_value(columns[col_idx], row, name)?;
                obj.insert((*name).to_owned(), value);
            }
            rows.push(serde_json::Value::Object(obj));
        }
    }
    let doc = serde_json::json!({
        "manifest_version": "1",
        "key_kind": key_kind,
        "rows": rows,
    });
    serde_json::to_vec(&doc)
        .map_err(|e| PyValueError::new_err(format!("manifest serialize failed: {e}")))
}

fn batches_share_schema(a: &arrow_schema::Schema, b: &arrow_schema::Schema) -> bool {
    let af = a.fields();
    let bf = b.fields();
    if af.len() != bf.len() {
        return false;
    }
    af.iter()
        .zip(bf.iter())
        .all(|(x, y)| x.name() == y.name() && x.data_type() == y.data_type())
}

/// Extract one cell of an Arrow column into a serde value. The
/// canonical manifest parser accepts strings + integers for the
/// `key` column and strings for axis columns; we map any common
/// integer / boolean / string Arrow type accordingly. Anything else
/// is rejected with a typed error.
fn column_value(col: &dyn Array, row: usize, name: &str) -> PyResult<serde_json::Value> {
    if col.is_null(row) {
        return Ok(serde_json::Value::Null);
    }
    match col.data_type() {
        DataType::Utf8 => {
            let s = col.as_any().downcast_ref::<StringArray>().ok_or_else(|| {
                PyValueError::new_err(format!("column {name:?}: utf8 downcast failed"))
            })?;
            Ok(serde_json::Value::String(s.value(row).to_owned()))
        }
        DataType::LargeUtf8 => {
            let s = col
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .ok_or_else(|| {
                    PyValueError::new_err(format!("column {name:?}: large utf8 downcast failed"))
                })?;
            Ok(serde_json::Value::String(s.value(row).to_owned()))
        }
        DataType::Utf8View => {
            let s = col
                .as_any()
                .downcast_ref::<StringViewArray>()
                .ok_or_else(|| {
                    PyValueError::new_err(format!("column {name:?}: utf8view downcast failed"))
                })?;
            Ok(serde_json::Value::String(s.value(row).to_owned()))
        }
        DataType::Int32 => {
            let a = col.as_any().downcast_ref::<Int32Array>().ok_or_else(|| {
                PyValueError::new_err(format!("column {name:?}: int32 downcast failed"))
            })?;
            Ok(serde_json::Value::from(a.value(row) as i64))
        }
        DataType::Int64 => {
            let a = col.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                PyValueError::new_err(format!("column {name:?}: int64 downcast failed"))
            })?;
            Ok(serde_json::Value::from(a.value(row)))
        }
        DataType::UInt32 => {
            let a = col.as_any().downcast_ref::<UInt32Array>().ok_or_else(|| {
                PyValueError::new_err(format!("column {name:?}: uint32 downcast failed"))
            })?;
            Ok(serde_json::Value::from(a.value(row) as i64))
        }
        DataType::UInt64 => {
            let a = col.as_any().downcast_ref::<UInt64Array>().ok_or_else(|| {
                PyValueError::new_err(format!("column {name:?}: uint64 downcast failed"))
            })?;
            Ok(serde_json::Value::from(a.value(row)))
        }
        other => Err(PyValueError::new_err(format!(
            "Arrow manifest column {name:?} has unsupported type {other:?}; \
             use Utf8 for axis values and Int64/Utf8 for the `key` column"
        ))),
    }
}
