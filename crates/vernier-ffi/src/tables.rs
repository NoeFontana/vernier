//! FFI surface for result tables: pack `vernier_core::tables` column
//! buffers into Arrow `RecordBatch`es and expose them through the Arrow
//! PyCapsule Interface. No business logic — column shapes, sentinels,
//! and overflow live in `vernier-core`. Construction runs under
//! `py.detach` (per ADR-0006).
//!
//! Schema metadata. Every schema produced here carries
//! `vernier.schema_version = "1"` and `vernier.table = "<name>"`
//! metadata so consumers can pin against a stable contract — the
//! Arrow analogue of the JSON `version` field (ADR-0019 / ADR-0046).
//! Slice schemas additionally carry `vernier.paradigm` and
//! `vernier.metric` so the four wide variants
//! (instance-AP / instance-LRP / panoptic / semantic) can be
//! disambiguated from a single column-name list.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::types::Int32Type;
use arrow_array::{
    ArrayRef, DictionaryArray, FixedSizeListArray, Float64Array, Int32Array, Int64Array,
    RecordBatch, StringArray, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use vernier_core::partition::PartitionedSummary;
use vernier_core::tables::{
    aggregate_per_class_support, build_per_class, build_per_detection, build_per_image,
    build_per_pair, BboxColumns, MatchStatus, PerClassTable, PerDetectionTable, PerImageTable,
    PerPairTable, TablesConfig,
};
use vernier_core::EvalError;

use crate::arrow_helpers::{arrow_err, wrap_batch, ArrowRecordBatchPy};
use crate::dataset::PyDataset;
use crate::PyAccumulated;
use crate::PyEvalGrid;

/// Schema metadata key for the wire-format version of a table.
pub(crate) const META_SCHEMA_VERSION: &str = "vernier.schema_version";
/// Schema metadata key naming the table (`per_class`, `slices`, ...).
pub(crate) const META_TABLE: &str = "vernier.table";
/// Schema metadata key naming the paradigm for slice tables
/// (`instance`, `panoptic`, `semantic`).
pub(crate) const META_PARADIGM: &str = "vernier.paradigm";
/// Schema metadata key naming the headline metric for slice tables
/// (`ap`, `lrp`, `pq`, `miou`).
pub(crate) const META_METRIC: &str = "vernier.metric";

/// Bytes-keyed metadata stamp for a regular result table.
pub(crate) fn table_metadata(name: &str) -> HashMap<String, String> {
    let mut md = HashMap::with_capacity(2);
    md.insert(META_SCHEMA_VERSION.to_owned(), "1".to_owned());
    md.insert(META_TABLE.to_owned(), name.to_owned());
    md
}

/// Bytes-keyed metadata stamp for a slice table.
fn slices_metadata(paradigm: &str, metric: &str) -> HashMap<String, String> {
    let mut md = HashMap::with_capacity(4);
    md.insert(META_SCHEMA_VERSION.to_owned(), "1".to_owned());
    md.insert(META_TABLE.to_owned(), "slices".to_owned());
    md.insert(META_PARADIGM.to_owned(), paradigm.to_owned());
    md.insert(META_METRIC.to_owned(), metric.to_owned());
    md
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
    // ADR-0040: reuse the grid's resolved IoU ladder so t50 / t75
    // lookups land on the same T-axis the matcher saw.
    let iou_thr = grid.iou_thresholds();
    let table = py
        .detach(move || -> Result<PerClassTable, EvalError> {
            let support = aggregate_per_class_support(inner_grid, 0);
            build_per_class(inner_accum, &dataset_arc, iou_thr, &max_dets, &support)
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
    let iou_thr = grid.iou_thresholds();
    let table = py
        .detach(move || -> Result<PerImageTable, EvalError> {
            build_per_image(inner_grid, &dataset_arc, iou_thr)
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
///
/// Reads detections retained on `grid` at construction time;
/// `evaluate_*_grid(retain_iou=true)` is required because the
/// per-detection table also needs the retained IoU matrices for
/// best-IoU lookup. Both come together — there is no path that could
/// supply one without the other, so retention serves both consumers
/// without re-parsing `dt`.
#[pyfunction]
#[pyo3(signature = (grid, with_geometry=false))]
pub(crate) fn per_detection_to_arrow_pycapsule(
    py: Python<'_>,
    grid: &PyEvalGrid,
    with_geometry: bool,
) -> PyResult<ArrowRecordBatchPy> {
    let inner_grid = grid.eval_grid_ref();
    let dets = grid.retained_dt().ok_or_else(|| {
        PyValueError::new_err(
            "per_detection table requires the grid to be built with \
             retain_iou=true; rebuild via `tables=('per_detection', ...)`",
        )
    })?;
    let cfg = TablesConfig {
        per_detection_with_geometry: with_geometry,
        ..TablesConfig::default()
    };
    let iou_thr = grid.iou_thresholds();
    let table = py
        .detach(move || -> Result<PerDetectionTable, EvalError> {
            build_per_detection(
                inner_grid,
                dets,
                iou_thr,
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
    let match_status: ArrayRef =
        Arc::new(DictionaryArray::<Int32Type>::try_new(keys_arr, values_arr)?);

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
        fields.push(Field::new(
            "bbox_xywh",
            DataType::FixedSizeList(item, 4),
            false,
        ));
    }
    Schema::new(fields).with_metadata(table_metadata("per_detection"))
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
            let retained =
                inner_grid
                    .retained_ious
                    .as_ref()
                    .ok_or_else(|| EvalError::InvalidConfig {
                        detail: "per_pair requires the upstream grid to have been built with \
                             retain_iou=True"
                            .into(),
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
    .with_metadata(table_metadata("per_pair"))
}

// ---------------------------------------------------------------------------
// Slices RecordBatches (ADR-0046)
// ---------------------------------------------------------------------------

/// Schema for the instance-AP slices table. Columns: the 12 detection
/// stats wide plus n_images / n_detections, one row per
/// `(axis, value)` cell.
fn slices_instance_ap_schema() -> Schema {
    Schema::new(vec![
        Field::new("axis", DataType::Utf8, false),
        Field::new("value", DataType::Utf8, false),
        Field::new("n_images", DataType::UInt64, false),
        Field::new("n_detections", DataType::UInt64, false),
        Field::new("ap", DataType::Float64, false),
        Field::new("ap50", DataType::Float64, false),
        Field::new("ap75", DataType::Float64, false),
        Field::new("ap_s", DataType::Float64, false),
        Field::new("ap_m", DataType::Float64, false),
        Field::new("ap_l", DataType::Float64, false),
        Field::new("ar_max_1", DataType::Float64, false),
        Field::new("ar_max_10", DataType::Float64, false),
        Field::new("ar_max_100", DataType::Float64, false),
        Field::new("ar_s", DataType::Float64, false),
        Field::new("ar_m", DataType::Float64, false),
        Field::new("ar_l", DataType::Float64, false),
    ])
    .with_metadata(slices_metadata("instance", "ap"))
}

/// Schema for the panoptic slices table.
fn slices_panoptic_schema() -> Schema {
    Schema::new(vec![
        Field::new("axis", DataType::Utf8, false),
        Field::new("value", DataType::Utf8, false),
        Field::new("n_images", DataType::UInt64, false),
        Field::new("n_detections", DataType::UInt64, false),
        Field::new("pq", DataType::Float64, false),
        Field::new("sq", DataType::Float64, false),
        Field::new("rq", DataType::Float64, false),
    ])
    .with_metadata(slices_metadata("panoptic", "pq"))
}

/// Schema for the semantic slices table.
fn slices_semantic_schema() -> Schema {
    Schema::new(vec![
        Field::new("axis", DataType::Utf8, false),
        Field::new("value", DataType::Utf8, false),
        Field::new("n_images", DataType::UInt64, false),
        Field::new("n_detections", DataType::UInt64, false),
        Field::new("miou", DataType::Float64, false),
        Field::new("fwiou", DataType::Float64, false),
        Field::new("pixel_accuracy", DataType::Float64, false),
        Field::new("mean_accuracy", DataType::Float64, false),
    ])
    .with_metadata(slices_metadata("semantic", "miou"))
}

/// One row per slice in `summary.slices`, with the 12 detection-AP
/// stats laid out wide.
pub(crate) fn slices_record_batch_instance_ap(
    summary: &PartitionedSummary,
) -> Result<RecordBatch, arrow_schema::ArrowError> {
    let n = summary.slices.len();
    let mut axis: Vec<String> = Vec::with_capacity(n);
    let mut value: Vec<String> = Vec::with_capacity(n);
    let mut n_images: Vec<u64> = Vec::with_capacity(n);
    let mut n_detections: Vec<u64> = Vec::with_capacity(n);
    // 12 detection stats — flat per-cell vectors, then assembled by column.
    let mut cols: [Vec<f64>; 12] = Default::default();
    for slice in &summary.slices {
        axis.push(slice.slice.axis.clone());
        value.push(slice.slice.value.clone());
        n_images.push(slice.n_images);
        n_detections.push(slice.n_detections);
        let stats = slice.summary.stats();
        for (i, col) in cols.iter_mut().enumerate() {
            col.push(stats.get(i).copied().unwrap_or(f64::NAN));
        }
    }
    let schema = Arc::new(slices_instance_ap_schema());
    let columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(axis)),
        Arc::new(StringArray::from(value)),
        Arc::new(UInt64Array::from(n_images)),
        Arc::new(UInt64Array::from(n_detections)),
        Arc::new(Float64Array::from(std::mem::take(&mut cols[0]))),
        Arc::new(Float64Array::from(std::mem::take(&mut cols[1]))),
        Arc::new(Float64Array::from(std::mem::take(&mut cols[2]))),
        Arc::new(Float64Array::from(std::mem::take(&mut cols[3]))),
        Arc::new(Float64Array::from(std::mem::take(&mut cols[4]))),
        Arc::new(Float64Array::from(std::mem::take(&mut cols[5]))),
        Arc::new(Float64Array::from(std::mem::take(&mut cols[6]))),
        Arc::new(Float64Array::from(std::mem::take(&mut cols[7]))),
        Arc::new(Float64Array::from(std::mem::take(&mut cols[8]))),
        Arc::new(Float64Array::from(std::mem::take(&mut cols[9]))),
        Arc::new(Float64Array::from(std::mem::take(&mut cols[10]))),
        Arc::new(Float64Array::from(std::mem::take(&mut cols[11]))),
    ];
    RecordBatch::try_new(schema, columns)
}

/// Convert an [`ArrowError`] result into a PyCapsule wrapper.
pub(crate) fn slices_instance_ap_to_arrow(
    summary: &PartitionedSummary,
) -> Result<ArrowRecordBatchPy, arrow_schema::ArrowError> {
    Ok(wrap_batch(slices_record_batch_instance_ap(summary)?))
}

/// One row per `(axis, value)` slice for the panoptic paradigm.
///
/// `rows` is a parallel vector of `(axis, value, n_images, n_detections,
/// pq, sq, rq)` tuples produced by the per-slice loop in the Python
/// wrapper — panoptic does not flow through [`PartitionedSummary`].
pub(crate) fn slices_record_batch_panoptic(
    rows: &[PanopticSliceRow],
) -> Result<RecordBatch, arrow_schema::ArrowError> {
    let n = rows.len();
    let mut axis: Vec<String> = Vec::with_capacity(n);
    let mut value: Vec<String> = Vec::with_capacity(n);
    let mut n_images: Vec<u64> = Vec::with_capacity(n);
    let mut n_detections: Vec<u64> = Vec::with_capacity(n);
    let mut pq: Vec<f64> = Vec::with_capacity(n);
    let mut sq: Vec<f64> = Vec::with_capacity(n);
    let mut rq: Vec<f64> = Vec::with_capacity(n);
    for r in rows {
        axis.push(r.axis.clone());
        value.push(r.value.clone());
        n_images.push(r.n_images);
        n_detections.push(r.n_detections);
        pq.push(r.pq);
        sq.push(r.sq);
        rq.push(r.rq);
    }
    let schema = Arc::new(slices_panoptic_schema());
    let columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(axis)),
        Arc::new(StringArray::from(value)),
        Arc::new(UInt64Array::from(n_images)),
        Arc::new(UInt64Array::from(n_detections)),
        Arc::new(Float64Array::from(pq)),
        Arc::new(Float64Array::from(sq)),
        Arc::new(Float64Array::from(rq)),
    ];
    RecordBatch::try_new(schema, columns)
}

/// One row per `(axis, value)` slice for the semantic paradigm.
pub(crate) fn slices_record_batch_semantic(
    rows: &[SemanticSliceRow],
) -> Result<RecordBatch, arrow_schema::ArrowError> {
    let n = rows.len();
    let mut axis: Vec<String> = Vec::with_capacity(n);
    let mut value: Vec<String> = Vec::with_capacity(n);
    let mut n_images: Vec<u64> = Vec::with_capacity(n);
    let mut n_detections: Vec<u64> = Vec::with_capacity(n);
    let mut miou: Vec<f64> = Vec::with_capacity(n);
    let mut fwiou: Vec<f64> = Vec::with_capacity(n);
    let mut pa: Vec<f64> = Vec::with_capacity(n);
    let mut ma: Vec<f64> = Vec::with_capacity(n);
    for r in rows {
        axis.push(r.axis.clone());
        value.push(r.value.clone());
        n_images.push(r.n_images);
        n_detections.push(r.n_detections);
        miou.push(r.miou);
        fwiou.push(r.fwiou);
        pa.push(r.pixel_accuracy);
        ma.push(r.mean_accuracy);
    }
    let schema = Arc::new(slices_semantic_schema());
    let columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(axis)),
        Arc::new(StringArray::from(value)),
        Arc::new(UInt64Array::from(n_images)),
        Arc::new(UInt64Array::from(n_detections)),
        Arc::new(Float64Array::from(miou)),
        Arc::new(Float64Array::from(fwiou)),
        Arc::new(Float64Array::from(pa)),
        Arc::new(Float64Array::from(ma)),
    ];
    RecordBatch::try_new(schema, columns)
}

/// Plain-data row payload for the panoptic slices table. Built by
/// the Python orchestrator that loops the unchanged single-eval over
/// per-slice image subsets (the panoptic substitute for
/// `evaluate_partitioned`).
#[derive(Debug, Clone)]
pub(crate) struct PanopticSliceRow {
    pub(crate) axis: String,
    pub(crate) value: String,
    pub(crate) n_images: u64,
    pub(crate) n_detections: u64,
    pub(crate) pq: f64,
    pub(crate) sq: f64,
    pub(crate) rq: f64,
}

/// Plain-data row payload for the semantic slices table. Same
/// orchestration pattern as [`PanopticSliceRow`].
#[derive(Debug, Clone)]
pub(crate) struct SemanticSliceRow {
    pub(crate) axis: String,
    pub(crate) value: String,
    pub(crate) n_images: u64,
    pub(crate) n_detections: u64,
    pub(crate) miou: f64,
    pub(crate) fwiou: f64,
    pub(crate) pixel_accuracy: f64,
    pub(crate) mean_accuracy: f64,
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
    .with_metadata(table_metadata("per_image"))
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
    .with_metadata(table_metadata("per_class"))
}
