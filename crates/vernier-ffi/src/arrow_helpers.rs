//! Arrow PyCapsule plumbing shared by every paradigm's tables FFI
//! module. Defines the [`ArrowRecordBatchPy`] pyclass that implements
//! the `__arrow_c_array__` dunder polars / pandas / duckdb / pyarrow
//! consume, plus the small helpers each table builder uses to wrap a
//! `RecordBatch` for export. No business logic, no per-paradigm types.

use std::sync::Arc;

use arrow_array::ffi::to_ffi;
use arrow_array::{Array, RecordBatch, StructArray};
use arrow_data::ArrayData;
use arrow_schema::ffi::FFI_ArrowSchema;
use arrow_schema::{Schema, SchemaRef};
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
/// `release` callback. The `schema` reference is retained because
/// `to_ffi(&ArrayData)` re-derives an `FFI_ArrowSchema` from the
/// `ArrayData`'s data_type alone — it does not see the parent
/// `RecordBatch`'s field-level or schema-level metadata. We rebuild
/// the FFI schema from the retained `Schema` so the ADR-0019 /
/// ADR-0046 metadata (`vernier.schema_version`, `vernier.table`, ...)
/// survives the round trip.
#[pyclass(module = "vernier._core", name = "ArrowRecordBatch", frozen)]
pub(crate) struct ArrowRecordBatchPy {
    data: ArrayData,
    schema: SchemaRef,
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
        // Build the array's FFI buffer pair, then discard the
        // metadata-less schema and replace it with one derived from
        // the retained `Schema` (which carries metadata).
        let (ffi_array, _ffi_schema_no_meta) = to_ffi(&self.data)
            .map_err(|e| PyValueError::new_err(format!("arrow ffi export failed: {e}")))?;
        let ffi_schema = schema_to_ffi(&self.schema)?;
        // Capsule names are the canonical C-Data-Interface strings; the
        // FFI structs' `Drop` runs the interface's release callback when
        // a capsule is collected.
        let schema_capsule = PyCapsule::new_with_value(py, ffi_schema, c"arrow_schema")?;
        let array_capsule = PyCapsule::new_with_value(py, ffi_array, c"arrow_array")?;
        Ok((schema_capsule, array_capsule))
    }

    fn __repr__(&self) -> String {
        let n_rows = self.data.len();
        let n_cols = self.data.child_data().len();
        format!("ArrowRecordBatch(rows={n_rows}, columns={n_cols})")
    }
}

/// Convert a `Schema` to an `FFI_ArrowSchema` carrying both the
/// schema-level metadata (e.g., `vernier.schema_version`) and the
/// per-field metadata. The structure must be a `Struct` (the
/// C-Data-Interface representation of a record batch); field children
/// are derived from `schema.fields()` in order.
fn schema_to_ffi(schema: &Schema) -> PyResult<FFI_ArrowSchema> {
    // Build child FFI schemas (one per field), each named like the
    // field, with its data_type and nullability flag.
    let mut children: Vec<FFI_ArrowSchema> = Vec::with_capacity(schema.fields().len());
    for field in schema.fields() {
        let child = FFI_ArrowSchema::try_from(field.data_type())
            .and_then(|c| c.with_name(field.name()))
            .map_err(|e| {
                PyValueError::new_err(format!("arrow field {:?} to FFI failed: {e}", field.name()))
            })?;
        let flags = if field.is_nullable() {
            arrow_schema::ffi::Flags::NULLABLE
        } else {
            arrow_schema::ffi::Flags::empty()
        };
        let child = child
            .with_flags(flags)
            .map_err(|e| PyValueError::new_err(format!("arrow field flags failed: {e}")))?;
        children.push(child);
    }
    let parent = FFI_ArrowSchema::try_new("+s", children, None)
        .map_err(|e| PyValueError::new_err(format!("arrow parent FFI build failed: {e}")))?;
    let parent = if !schema.metadata().is_empty() {
        parent
            .with_metadata(
                schema
                    .metadata()
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone())),
            )
            .map_err(|e| PyValueError::new_err(format!("arrow schema metadata failed: {e}")))?
    } else {
        parent
    };
    Ok(parent)
}

/// Wrap an Arrow `RecordBatch` for PyCapsule export. Arrow treats
/// record batches and struct arrays identically on the C-Data-Interface
/// side; we go through `StructArray` to get a single `ArrayData` payload,
/// and retain the original `Schema` so schema-level metadata
/// (ADR-0019, ADR-0046) is preserved on export.
pub(crate) fn wrap_batch(batch: RecordBatch) -> ArrowRecordBatchPy {
    let schema = Arc::clone(batch.schema_ref());
    ArrowRecordBatchPy {
        data: StructArray::from(batch).to_data(),
        schema,
    }
}

/// Translate an Arrow build error into the `ValueError` shape every
/// paradigm's table FFI returns.
pub(crate) fn arrow_err(e: &arrow_schema::ArrowError) -> PyErr {
    PyValueError::new_err(format!("arrow record batch build failed: {e}"))
}
