//! Arrow PyCapsule plumbing shared by every paradigm's tables FFI
//! module. Defines the [`ArrowRecordBatchPy`] pyclass that implements
//! the `__arrow_c_array__` dunder polars / pandas / duckdb / pyarrow
//! consume, plus the small helpers each table builder uses to wrap a
//! `RecordBatch` for export. No business logic, no per-paradigm types.

use std::ffi::CString;

use arrow_array::ffi::to_ffi;
use arrow_array::{Array, RecordBatch, StructArray};
use arrow_data::ArrayData;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyCapsule};

/// Arrow-PyCapsule producer for a single `RecordBatch`. Implements the
/// `__arrow_c_array__` dunder polars / pandas / duckdb / pyarrow
/// consume.
///
/// The buffers are `Arc`-backed inside `ArrayData`, so each call mints
/// fresh `FFI_ArrowArray` / `FFI_ArrowSchema` handles cheaply; when a
/// capsule drops, the FFI struct's `Drop` runs the C-Data-Interface
/// `release` callback.
#[pyclass(module = "vernier._core", name = "ArrowRecordBatch", frozen)]
pub(crate) struct ArrowRecordBatchPy {
    data: ArrayData,
}

#[pymethods]
impl ArrowRecordBatchPy {
    /// Arrow PyCapsule Interface entry. Returns `(schema_capsule,
    /// array_capsule)`. `requested_schema` is accepted for protocol
    /// compatibility but ignored — we always emit our native schema
    /// (the same relaxed-producer posture polars/duckdb/pyarrow take).
    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_array__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<(Bound<'py, PyCapsule>, Bound<'py, PyCapsule>)> {
        let _ = requested_schema;
        let (ffi_array, ffi_schema) = to_ffi(&self.data)
            .map_err(|e| PyValueError::new_err(format!("arrow ffi export failed: {e}")))?;
        let schema_capsule = make_capsule(py, ffi_schema, "arrow_schema")?;
        let array_capsule = make_capsule(py, ffi_array, "arrow_array")?;
        Ok((schema_capsule, array_capsule))
    }

    fn __repr__(&self) -> String {
        let n_rows = self.data.len();
        let n_cols = self.data.child_data().len();
        format!("ArrowRecordBatch(rows={n_rows}, columns={n_cols})")
    }
}

/// Wrap a (Drop-runs-release) Arrow FFI struct in a PyCapsule with the
/// canonical C-Data-Interface name. Generic over the FFI type so the
/// schema and array sites share one helper.
fn make_capsule<'py, T>(py: Python<'py>, value: T, name: &str) -> PyResult<Bound<'py, PyCapsule>>
where
    T: Send + 'static,
{
    let cname = CString::new(name)
        .map_err(|e| PyValueError::new_err(format!("invalid capsule name: {e}")))?;
    PyCapsule::new(py, value, Some(cname))
}

/// Wrap an Arrow `RecordBatch` for PyCapsule export. Arrow treats
/// record batches and struct arrays identically on the C-Data-Interface
/// side; we go through `StructArray` to get a single `ArrayData` payload.
pub(crate) fn wrap_batch(batch: RecordBatch) -> ArrowRecordBatchPy {
    ArrowRecordBatchPy {
        data: StructArray::from(batch).to_data(),
    }
}

/// Translate an Arrow build error into the `ValueError` shape every
/// paradigm's table FFI returns.
pub(crate) fn arrow_err(e: &arrow_schema::ArrowError) -> PyErr {
    PyValueError::new_err(format!("arrow record batch build failed: {e}"))
}
