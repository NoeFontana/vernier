//! Arrow PyCapsule wrappers for [`vernier_panoptic::tables`].

use std::sync::Arc;

use arrow_array::{ArrayRef, Float64Array, Int64Array, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use pyo3::prelude::*;

use vernier_panoptic::tables::{build_per_class, PerClassTable};

use crate::arrow_helpers::{arrow_err, wrap_batch, ArrowRecordBatchPy};
use crate::panoptic::PyPanopticSummary;

/// Build a [`vernier_panoptic::tables::PerClassTable`] from a panoptic
/// summary and return it as an Arrow record batch wrapped for
/// PyCapsule export.
#[pyfunction]
#[pyo3(signature = (summary))]
pub(crate) fn panoptic_per_class_to_arrow_pycapsule(
    py: Python<'_>,
    summary: &PyPanopticSummary,
) -> PyResult<ArrowRecordBatchPy> {
    let inner = summary.summary_ref();
    let table = py.detach(|| build_per_class(inner));
    let batch = per_class_record_batch(&table).map_err(|e| arrow_err(&e))?;
    Ok(wrap_batch(batch))
}

fn per_class_record_batch(table: &PerClassTable) -> Result<RecordBatch, arrow_schema::ArrowError> {
    let schema = Arc::new(per_class_schema());
    let category_id: ArrayRef = Arc::new(Int64Array::from(table.category_id.clone()));
    let pq: ArrayRef = Arc::new(Float64Array::from(table.pq.clone()));
    let sq: ArrayRef = Arc::new(Float64Array::from(table.sq.clone()));
    let rq: ArrayRef = Arc::new(Float64Array::from(table.rq.clone()));
    let n_tp: ArrayRef = Arc::new(UInt64Array::from(table.n_tp.clone()));
    let n_fp: ArrayRef = Arc::new(UInt64Array::from(table.n_fp.clone()));
    let n_fn: ArrayRef = Arc::new(UInt64Array::from(table.n_fn.clone()));
    let iou_sum: ArrayRef = Arc::new(Float64Array::from(table.iou_sum.clone()));
    RecordBatch::try_new(
        schema,
        vec![category_id, pq, sq, rq, n_tp, n_fp, n_fn, iou_sum],
    )
}

fn per_class_schema() -> Schema {
    Schema::new(vec![
        Field::new("category_id", DataType::Int64, false),
        Field::new("pq", DataType::Float64, false),
        Field::new("sq", DataType::Float64, false),
        Field::new("rq", DataType::Float64, false),
        Field::new("n_tp", DataType::UInt64, false),
        Field::new("n_fp", DataType::UInt64, false),
        Field::new("n_fn", DataType::UInt64, false),
        Field::new("iou_sum", DataType::Float64, false),
    ])
}
