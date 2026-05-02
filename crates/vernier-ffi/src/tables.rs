//! FFI surface for result tables: pack `vernier_core::tables` column
//! buffers into Arrow `RecordBatch`es and expose them through the Arrow
//! PyCapsule Interface. No business logic — column shapes, sentinels,
//! and overflow live in `vernier-core`. Construction runs under
//! `py.detach` (per ADR-0006).

use std::ffi::CString;
use std::sync::Arc;

use arrow_array::ffi::to_ffi;
use arrow_array::types::Int32Type;
use arrow_array::{
    Array, ArrayRef, DictionaryArray, FixedSizeListArray, Float64Array, Int32Array, Int64Array,
    RecordBatch, StringArray, StructArray, UInt32Array,
};
use arrow_data::ArrayData;
use arrow_schema::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyCapsule};

use vernier_core::{
    aggregate_per_class_support, build_per_class, build_per_detection, build_per_image,
    build_per_pair, iou_thresholds, BboxColumns, CocoDetections, EvalError, MatchStatus,
    PerClassTable, PerDetectionTable, PerImageTable, PerPairTable, TablesConfig,
};

use crate::dataset::PyDataset;
use crate::PyAccumulated;
use crate::PyEvalGrid;

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
fn wrap_batch(batch: RecordBatch) -> ArrowRecordBatchPy {
    ArrowRecordBatchPy {
        data: StructArray::from(batch).to_data(),
    }
}

fn arrow_err(e: &arrow_schema::ArrowError) -> PyErr {
    PyValueError::new_err(format!("arrow record batch build failed: {e}"))
}

/// Build a [`vernier_core::PerClassTable`] and return it as an
/// Arrow record batch wrapped for PyCapsule export.
#[pyfunction]
#[pyo3(signature = (grid, accum, dataset))]
pub(crate) fn per_class_to_arrow_pycapsule(
    py: Python<'_>,
    grid: &PyEvalGrid,
    accum: &PyAccumulated,
    dataset: &PyDataset,
) -> PyResult<ArrowRecordBatchPy> {
    let inner_grid = grid.eval_grid_ref();
    let inner_accum = accum.accumulated_ref();
    let max_dets = accum.max_dets_slice().to_vec();
    let dataset_arc = dataset.dataset_ref();
    let table = py
        .detach(move || -> Result<PerClassTable, EvalError> {
            let support = aggregate_per_class_support(inner_grid, 0);
            build_per_class(
                inner_accum,
                &dataset_arc,
                iou_thresholds(),
                &max_dets,
                &support,
            )
        })
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;
    let batch = per_class_record_batch(&table).map_err(|e| arrow_err(&e))?;
    Ok(wrap_batch(batch))
}

fn per_class_record_batch(table: &PerClassTable) -> Result<RecordBatch, arrow_schema::ArrowError> {
    let schema = Arc::new(per_class_schema());

    let category_id: ArrayRef = Arc::new(Int64Array::from(table.category_id.clone()));
    let category_name: ArrayRef = Arc::new(StringArray::from_iter_values(
        table.category_name.iter().map(String::as_str),
    ));
    let ap: ArrayRef = Arc::new(Float64Array::from_iter(table.ap.iter().copied()));
    let ap50: ArrayRef = Arc::new(Float64Array::from_iter(table.ap50.iter().copied()));
    let ap75: ArrayRef = Arc::new(Float64Array::from_iter(table.ap75.iter().copied()));
    let ap_s: ArrayRef = Arc::new(Float64Array::from_iter(table.ap_s.iter().copied()));
    let ap_m: ArrayRef = Arc::new(Float64Array::from_iter(table.ap_m.iter().copied()));
    let ap_l: ArrayRef = Arc::new(Float64Array::from_iter(table.ap_l.iter().copied()));
    let ar_max_1: ArrayRef = Arc::new(Float64Array::from_iter(table.ar_max_1.iter().copied()));
    let ar_max_10: ArrayRef = Arc::new(Float64Array::from_iter(table.ar_max_10.iter().copied()));
    let ar_max_100: ArrayRef = Arc::new(Float64Array::from_iter(table.ar_max_100.iter().copied()));
    let n_gt: ArrayRef = Arc::new(UInt32Array::from(table.n_gt.clone()));
    let n_dt: ArrayRef = Arc::new(UInt32Array::from(table.n_dt.clone()));

    RecordBatch::try_new(
        schema,
        vec![
            category_id,
            category_name,
            ap,
            ap50,
            ap75,
            ap_s,
            ap_m,
            ap_l,
            ar_max_1,
            ar_max_10,
            ar_max_100,
            n_gt,
            n_dt,
        ],
    )
}

/// Build a [`vernier_core::PerImageTable`] and return it as an
/// Arrow record batch wrapped for PyCapsule export.
#[pyfunction]
#[pyo3(signature = (grid, dataset))]
pub(crate) fn per_image_to_arrow_pycapsule(
    py: Python<'_>,
    grid: &PyEvalGrid,
    dataset: &PyDataset,
) -> PyResult<ArrowRecordBatchPy> {
    let inner_grid = grid.eval_grid_ref();
    let dataset_arc = dataset.dataset_ref();
    let table = py
        .detach(move || -> Result<PerImageTable, EvalError> {
            build_per_image(inner_grid, &dataset_arc, iou_thresholds())
        })
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;
    let batch = per_image_record_batch(&table).map_err(|e| arrow_err(&e))?;
    Ok(wrap_batch(batch))
}

/// Convert an already-built [`PerImageTable`] to an Arrow record batch
/// wrapped for PyCapsule export. Used by the streaming path which holds
/// the table value-by-value.
pub(crate) fn per_image_table_to_arrow(
    table: &PerImageTable,
) -> Result<ArrowRecordBatchPy, arrow_schema::ArrowError> {
    Ok(wrap_batch(per_image_record_batch(table)?))
}

/// Convert an already-built [`PerClassTable`] to an Arrow record batch
/// wrapped for PyCapsule export.
pub(crate) fn per_class_table_to_arrow(
    table: &PerClassTable,
) -> Result<ArrowRecordBatchPy, arrow_schema::ArrowError> {
    Ok(wrap_batch(per_class_record_batch(table)?))
}

/// Convert an already-built [`PerDetectionTable`] to an Arrow record
/// batch wrapped for PyCapsule export.
pub(crate) fn per_detection_table_to_arrow(
    table: &PerDetectionTable,
) -> Result<ArrowRecordBatchPy, arrow_schema::ArrowError> {
    Ok(wrap_batch(per_detection_record_batch(table)?))
}

/// Convert an already-built [`PerPairTable`] to an Arrow record batch
/// wrapped for PyCapsule export.
pub(crate) fn per_pair_table_to_arrow(
    table: &PerPairTable,
) -> Result<ArrowRecordBatchPy, arrow_schema::ArrowError> {
    Ok(wrap_batch(per_pair_record_batch(table)?))
}

fn per_image_record_batch(table: &PerImageTable) -> Result<RecordBatch, arrow_schema::ArrowError> {
    let schema = Arc::new(per_image_schema());
    let image_id: ArrayRef = Arc::new(Int64Array::from(table.image_id.clone()));
    let n_gt: ArrayRef = Arc::new(UInt32Array::from(table.n_gt.clone()));
    let n_dt: ArrayRef = Arc::new(UInt32Array::from(table.n_dt.clone()));
    let tp_at_50: ArrayRef = Arc::new(UInt32Array::from(table.tp_at_50.clone()));
    let fp_at_50: ArrayRef = Arc::new(UInt32Array::from(table.fp_at_50.clone()));
    let fn_at_50: ArrayRef = Arc::new(UInt32Array::from(table.fn_at_50.clone()));
    let tp_at_75: ArrayRef = Arc::new(UInt32Array::from(table.tp_at_75.clone()));
    let fp_at_75: ArrayRef = Arc::new(UInt32Array::from(table.fp_at_75.clone()));
    let fn_at_75: ArrayRef = Arc::new(UInt32Array::from(table.fn_at_75.clone()));
    let tp_mean_iou: ArrayRef = Arc::new(UInt32Array::from(table.tp_mean_iou.clone()));
    RecordBatch::try_new(
        schema,
        vec![
            image_id,
            n_gt,
            n_dt,
            tp_at_50,
            fp_at_50,
            fn_at_50,
            tp_at_75,
            fp_at_75,
            fn_at_75,
            tp_mean_iou,
        ],
    )
}

/// Build a [`PerDetectionTable`] and return it as an Arrow record batch
/// wrapped for PyCapsule export.
#[pyfunction]
#[pyo3(signature = (grid, dt_json, with_geometry=false))]
pub(crate) fn per_detection_to_arrow_pycapsule(
    py: Python<'_>,
    grid: &PyEvalGrid,
    dt_json: &Bound<'_, pyo3::types::PyBytes>,
    with_geometry: bool,
) -> PyResult<ArrowRecordBatchPy> {
    let inner_grid = grid.eval_grid_ref();
    let dt_bytes = dt_json.as_bytes().to_vec();
    let cfg = TablesConfig {
        per_detection_with_geometry: with_geometry,
        ..TablesConfig::default()
    };
    let table = py
        .detach(move || -> Result<PerDetectionTable, EvalError> {
            let dets = CocoDetections::from_json_bytes(&dt_bytes)?;
            build_per_detection(
                inner_grid,
                &dets,
                iou_thresholds(),
                inner_grid.retained_ious.as_ref(),
                &cfg,
            )
        })
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;
    let batch = per_detection_record_batch(&table).map_err(|e| arrow_err(&e))?;
    Ok(wrap_batch(batch))
}

fn per_detection_record_batch(
    table: &PerDetectionTable,
) -> Result<RecordBatch, arrow_schema::ArrowError> {
    let with_geometry = table.bbox.is_some();
    let schema = Arc::new(per_detection_schema(with_geometry));

    let detection_id: ArrayRef = Arc::new(Int64Array::from(table.detection_id.clone()));
    let image_id: ArrayRef = Arc::new(Int64Array::from(table.image_id.clone()));
    let category_id: ArrayRef = Arc::new(Int64Array::from(table.category_id.clone()));
    let score: ArrayRef = Arc::new(Float64Array::from(table.score.clone()));
    let area: ArrayRef = Arc::new(Float64Array::from(table.area.clone()));

    let keys: Vec<i32> = table
        .match_status_at_50
        .iter()
        .map(|s| s.dict_index() as i32)
        .collect();
    let keys_arr = Int32Array::from(keys);
    let values_arr: ArrayRef = Arc::new(StringArray::from(MatchStatus::DICT_VALUES.to_vec()));
    let match_status: ArrayRef = Arc::new(DictionaryArray::<Int32Type>::try_new(
        keys_arr, values_arr,
    )?);

    let matched_gt_id: ArrayRef = Arc::new(Int64Array::from_iter(
        table.matched_gt_id_at_50.iter().copied(),
    ));
    let best_iou: ArrayRef = Arc::new(Float64Array::from_iter(table.best_iou.iter().copied()));

    let mut columns: Vec<ArrayRef> = vec![
        detection_id,
        image_id,
        category_id,
        score,
        area,
        match_status,
        matched_gt_id,
        best_iou,
    ];

    if let Some(BboxColumns { xywh }) = &table.bbox {
        let flat: Vec<f64> = xywh.iter().flat_map(|b| b.iter().copied()).collect();
        let values_arr: ArrayRef = Arc::new(Float64Array::from(flat));
        let field = Arc::new(Field::new("item", DataType::Float64, false));
        let bbox_arr: ArrayRef = Arc::new(FixedSizeListArray::new(field, 4, values_arr, None));
        columns.push(bbox_arr);
    }

    RecordBatch::try_new(schema, columns)
}

fn per_detection_schema(with_geometry: bool) -> Schema {
    let mut fields = vec![
        Field::new("detection_id", DataType::Int64, false),
        Field::new("image_id", DataType::Int64, false),
        Field::new("category_id", DataType::Int64, false),
        Field::new("score", DataType::Float64, false),
        Field::new("area", DataType::Float64, false),
        Field::new(
            "match_status_at_50",
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            false,
        ),
        Field::new("matched_gt_id_at_50", DataType::Int64, true),
        Field::new("best_iou", DataType::Float64, true),
    ];
    if with_geometry {
        let item = Arc::new(Field::new("item", DataType::Float64, false));
        fields.push(Field::new("bbox_xywh", DataType::FixedSizeList(item, 4), false));
    }
    Schema::new(fields)
}

/// Build a [`PerPairTable`] and return it as an Arrow record batch
/// wrapped for PyCapsule export. Raises a `ValueError` (carrying the
/// typed `EvalError::PerPairOverflow` message) when the row count
/// exceeds `max_rows`.
#[pyfunction]
#[pyo3(signature = (grid, iou_floor=0.1, max_rows=10_000_000))]
pub(crate) fn per_pair_to_arrow_pycapsule(
    py: Python<'_>,
    grid: &PyEvalGrid,
    iou_floor: f64,
    max_rows: usize,
) -> PyResult<ArrowRecordBatchPy> {
    let inner_grid = grid.eval_grid_ref();
    let cfg = TablesConfig {
        per_pair_iou_floor: iou_floor,
        per_pair_max_rows: max_rows,
        ..TablesConfig::default()
    };
    let table = py
        .detach(move || -> Result<PerPairTable, EvalError> {
            let retained = inner_grid.retained_ious.as_ref().ok_or_else(|| {
                EvalError::InvalidConfig {
                    detail: "per_pair requires the upstream grid to have been built with \
                             retain_iou=True"
                        .into(),
                }
            })?;
            build_per_pair(inner_grid, retained, &cfg)
        })
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;
    let batch = per_pair_record_batch(&table).map_err(|e| arrow_err(&e))?;
    Ok(wrap_batch(batch))
}

fn per_pair_record_batch(table: &PerPairTable) -> Result<RecordBatch, arrow_schema::ArrowError> {
    let schema = Arc::new(per_pair_schema());
    let detection_id: ArrayRef = Arc::new(Int64Array::from(table.detection_id.clone()));
    let ground_truth_id: ArrayRef = Arc::new(Int64Array::from(table.ground_truth_id.clone()));
    let image_id: ArrayRef = Arc::new(Int64Array::from(table.image_id.clone()));
    let category_id: ArrayRef = Arc::new(Int64Array::from(table.category_id.clone()));
    let iou: ArrayRef = Arc::new(Float64Array::from(table.iou.clone()));
    RecordBatch::try_new(
        schema,
        vec![detection_id, ground_truth_id, image_id, category_id, iou],
    )
}

fn per_pair_schema() -> Schema {
    Schema::new(vec![
        Field::new("detection_id", DataType::Int64, false),
        Field::new("ground_truth_id", DataType::Int64, false),
        Field::new("image_id", DataType::Int64, false),
        Field::new("category_id", DataType::Int64, false),
        Field::new("iou", DataType::Float64, false),
    ])
}

fn per_image_schema() -> Schema {
    // No `ap` / `ap_50` columns: per-image AP is degenerate; see
    // `docs/explanation/why-no-per-image-ap.md`. Tests guard the absence.
    Schema::new(vec![
        Field::new("image_id", DataType::Int64, false),
        Field::new("n_gt", DataType::UInt32, false),
        Field::new("n_dt", DataType::UInt32, false),
        Field::new("tp_at_50", DataType::UInt32, false),
        Field::new("fp_at_50", DataType::UInt32, false),
        Field::new("fn_at_50", DataType::UInt32, false),
        Field::new("tp_at_75", DataType::UInt32, false),
        Field::new("fp_at_75", DataType::UInt32, false),
        Field::new("fn_at_75", DataType::UInt32, false),
        Field::new("tp_mean_iou", DataType::UInt32, false),
    ])
}

fn per_class_schema() -> Schema {
    Schema::new(vec![
        Field::new("category_id", DataType::Int64, false),
        Field::new("category_name", DataType::Utf8, false),
        Field::new("ap", DataType::Float64, true),
        Field::new("ap50", DataType::Float64, true),
        Field::new("ap75", DataType::Float64, true),
        Field::new("ap_s", DataType::Float64, true),
        Field::new("ap_m", DataType::Float64, true),
        Field::new("ap_l", DataType::Float64, true),
        Field::new("ar_max_1", DataType::Float64, true),
        Field::new("ar_max_10", DataType::Float64, true),
        Field::new("ar_max_100", DataType::Float64, true),
        Field::new("n_gt", DataType::UInt32, false),
        Field::new("n_dt", DataType::UInt32, false),
    ])
}
