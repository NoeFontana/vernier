//! Arrow PyCapsule wrappers for [`vernier_semantic::tables`].

use std::sync::Arc;

use arrow_array::{ArrayRef, Float64Array, Int64Array, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use pyo3::prelude::*;

use vernier_semantic::tables::{build_per_class, PerClassTable};

use crate::arrow_helpers::{arrow_err, wrap_batch, ArrowRecordBatchPy};
use crate::semantic::PySemanticSummary;

/// Build a [`vernier_semantic::tables::PerClassTable`] from a semantic
/// summary and return it as an Arrow record batch wrapped for
/// PyCapsule export. Reads the confusion-matrix diagonal off the
/// summary in-place; no clone.
#[pyfunction]
#[pyo3(signature = (summary))]
pub(crate) fn semantic_per_class_to_arrow_pycapsule(
    py: Python<'_>,
    summary: &PySemanticSummary,
) -> PyResult<ArrowRecordBatchPy> {
    let inner = summary.summary_ref();
    let table = py.detach(|| build_per_class(inner, &inner.confusion_matrix));
    let batch = per_class_record_batch(&table).map_err(|e| arrow_err(&e))?;
    Ok(wrap_batch(batch))
}

fn per_class_record_batch(table: &PerClassTable) -> Result<RecordBatch, arrow_schema::ArrowError> {
    let schema = Arc::new(per_class_schema());
    let category_id: ArrayRef = Arc::new(Int64Array::from(table.category_id.clone()));
    let iou: ArrayRef = Arc::new(Float64Array::from(table.iou.clone()));
    let accuracy: ArrayRef = Arc::new(Float64Array::from(table.accuracy.clone()));
    let precision: ArrayRef = Arc::new(Float64Array::from(table.precision.clone()));
    let n_gt_pixels: ArrayRef = Arc::new(UInt64Array::from(table.n_gt_pixels.clone()));
    let n_dt_pixels: ArrayRef = Arc::new(UInt64Array::from(table.n_dt_pixels.clone()));
    let tp_pixels: ArrayRef = Arc::new(UInt64Array::from(table.tp_pixels.clone()));
    let fp_pixels: ArrayRef = Arc::new(UInt64Array::from(table.fp_pixels.clone()));
    let fn_pixels: ArrayRef = Arc::new(UInt64Array::from(table.fn_pixels.clone()));
    RecordBatch::try_new(
        schema,
        vec![
            category_id,
            iou,
            accuracy,
            precision,
            n_gt_pixels,
            n_dt_pixels,
            tp_pixels,
            fp_pixels,
            fn_pixels,
        ],
    )
}

fn per_class_schema() -> Schema {
    Schema::new(vec![
        Field::new("category_id", DataType::Int64, false),
        Field::new("iou", DataType::Float64, false),
        Field::new("accuracy", DataType::Float64, false),
        Field::new("precision", DataType::Float64, false),
        Field::new("n_gt_pixels", DataType::UInt64, false),
        Field::new("n_dt_pixels", DataType::UInt64, false),
        Field::new("tp_pixels", DataType::UInt64, false),
        Field::new("fp_pixels", DataType::UInt64, false),
        Field::new("fn_pixels", DataType::UInt64, false),
    ])
}
