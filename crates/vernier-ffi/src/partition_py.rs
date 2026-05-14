//! Partitioned-evaluation FFI (ADR-0046, instance-AP only).
//!
//! Adds four sibling pyfunctions next to the existing
//! `evaluate_*_grid` family — `evaluate_bbox_partitioned`,
//! `evaluate_segm_partitioned`, `evaluate_boundary_partitioned`,
//! `evaluate_keypoints_partitioned`. Each runs today's
//! `evaluate_*_grid` orchestration once (the locked spine; C3 axiom),
//! then resolves the user-supplied manifest into a
//! [`PartitionSpec`] and dispatches into
//! [`vernier_core::partition::evaluate_partitioned`]. The wrapping
//! [`PyPartitionedSummary`] surfaces the overall summary (bit-
//! identical to the un-partitioned path) plus an Arrow `slices`
//! capsule the Python lane reads as a `RecordBatch`.
//!
//! LRP / panoptic / semantic partitioning are NOT in this module —
//! see the report comment at the top of `lib.rs` for the rationale.
//! The LRP variant pyfunctions return a typed `PyRuntimeError`
//! pointing the caller at the deferred work; panoptic / semantic
//! partition at the Python level by looping the unchanged single-
//! eval N+1 times (a C1 fallback, acceptable for those paradigms'
//! smaller K + cheaper re-matching).

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict};

use vernier_core::manifest::partition_spec_from_manifest;
use vernier_core::partition::{
    evaluate_partitioned, image_id_to_idx, GridDims, PartitionedSummary, SummaryPlan,
};

use crate::arrow_helpers::{wrap_batch, ArrowRecordBatchPy};
use crate::breakdown;
use crate::manifest_py::manifest_to_canonical_json;
use crate::tables::{
    slices_instance_ap_to_arrow, slices_record_batch_panoptic, slices_record_batch_semantic,
    PanopticSliceRow, SemanticSliceRow,
};
use crate::{
    boundary_iou_type, evaluate_grid_impl, parse_parity_mode, parse_sigmas, EvalIouType, PySummary,
};

/// Result of an instance-AP partitioned evaluate. The `overall`
/// summary is bit-identical to the un-partitioned path; `slices` is
/// the Arrow `RecordBatch` carrying one row per `(axis, value)` cell.
#[pyclass(module = "vernier._core", name = "PartitionedSummary", frozen)]
pub(crate) struct PyPartitionedSummary {
    inner: PartitionedSummary,
}

#[pymethods]
impl PyPartitionedSummary {
    /// Bit-identical to the un-partitioned summary.
    #[getter]
    fn overall(&self) -> PySummary {
        PySummary::new(self.inner.overall.clone())
    }

    /// Dataset image count behind `overall`.
    #[getter]
    fn overall_n_images(&self) -> u64 {
        self.inner.overall_n_images
    }

    /// Detection count behind `overall`.
    #[getter]
    fn overall_n_detections(&self) -> u64 {
        self.inner.overall_n_detections
    }

    /// Number of `(axis, value)` cells in the partition (excluding
    /// `overall`).
    #[getter]
    fn n_slices(&self) -> usize {
        self.inner.slices.len()
    }

    /// Build the slices Arrow RecordBatch and return its
    /// PyCapsule-exporting wrapper. Materialised on demand so an
    /// `evaluate(...)` call that never reads `.slices` pays no Arrow
    /// build cost.
    fn slices_capsule(&self) -> PyResult<ArrowRecordBatchPy> {
        slices_instance_ap_to_arrow(&self.inner)
            .map_err(|e| PyValueError::new_err(format!("arrow build failed: {e}")))
    }

    fn __repr__(&self) -> String {
        format!(
            "PartitionedSummary(overall_n_images={}, overall_n_detections={}, n_slices={})",
            self.inner.overall_n_images,
            self.inner.overall_n_detections,
            self.inner.slices.len()
        )
    }
}

/// Cross product helper — convert `list[list[str]]` from Python
/// into the `Vec<Vec<String>>` shape `PartitionSpec::build` consumes.
fn parse_cross_axes(cross_axes: Option<Vec<Vec<String>>>) -> Vec<Vec<String>> {
    cross_axes.unwrap_or_default()
}

/// Surface a `Vec<ManifestWarning>` through Python's `warnings` module
/// so the caller observes them at `evaluate(manifest=...)` call time.
fn warn_about_manifest(
    py: Python<'_>,
    warnings: &[vernier_core::manifest::ManifestWarning],
) -> PyResult<()> {
    if warnings.is_empty() {
        return Ok(());
    }
    let warnings_mod = py.import("warnings")?;
    for w in warnings {
        let msg = match w {
            vernier_core::manifest::ManifestWarning::UnknownKey { key } => {
                format!("manifest key {key:?} is not present in the dataset; skipping")
            }
        };
        // UserWarning category, stacklevel=2 so the warning blames
        // the caller's evaluate(...) call, not our wrapper.
        warnings_mod.call_method1("warn", (msg,))?;
    }
    Ok(())
}

/// Shared per-paradigm orchestration: run the grid pass, resolve the
/// manifest, dispatch into `evaluate_partitioned`.
#[allow(clippy::too_many_arguments)]
fn evaluate_instance_partitioned_impl(
    py: Python<'_>,
    iou_type: EvalIouType,
    gt_json: &Bound<'_, PyBytes>,
    dt: &Bound<'_, PyAny>,
    parity_mode: &str,
    max_dets_per_image: usize,
    use_cats: bool,
    cast_inputs: bool,
    iou_thresholds: Option<Vec<f64>>,
    recall_thresholds: Option<Vec<f64>>,
    area_ranges: Option<&Bound<'_, breakdown::PyBreakdown>>,
    manifest: &Bound<'_, PyAny>,
    cross_axes: Option<Vec<Vec<String>>>,
    key_kind: &str,
    is_keypoints: bool,
) -> PyResult<PyPartitionedSummary> {
    // 1) Build the grid (the locked-spine matching pass).
    let grid = evaluate_grid_impl(
        py,
        iou_type,
        gt_json,
        dt,
        parity_mode,
        max_dets_per_image,
        use_cats,
        false, // retain_iou
        cast_inputs,
        iou_thresholds,
        recall_thresholds,
        area_ranges,
    )?;

    // 2) Build image_id -> sorted-index map from the grid's GT
    //    snapshot. Image ordering matches `evaluate_with` exactly
    //    (id-ascending sort).
    let snapshot = grid.dataset_snapshot();
    let image_id_to_idx = image_id_to_idx(&*snapshot.gt);

    // 3) Materialise canonical JSON for the manifest input.
    let manifest_bytes = manifest_to_canonical_json(py, manifest, key_kind)?;

    // 4) Resolve into a PartitionSpec.
    let cross = parse_cross_axes(cross_axes);
    let (spec, warnings) = partition_spec_from_manifest(&manifest_bytes, &image_id_to_idx, &cross)
        .map_err(|e| PyValueError::new_err(format!("manifest resolution failed: {e}")))?;
    warn_about_manifest(py, &warnings)?;

    // 5) Run the partitioned summarize pass. Capture eval_imgs by
    //    reference into the GIL-released closure — at LVIS scale the
    //    dense vec is ~760 MB of pointer slots plus ~hundreds of MB
    //    of deep-cloned cells; the closure only needs a borrow.
    let parity = parse_parity_mode(parity_mode)?;
    let iou_thr = grid.iou_thresholds_vec();
    let grid_inner = grid.eval_grid_ref();
    let dims = GridDims {
        n_categories: grid_inner.n_categories,
        n_area_ranges: grid_inner.n_area_ranges,
        n_images: grid_inner.n_images,
    };
    let plan = if is_keypoints {
        SummaryPlan::KeypointsDefault
    } else {
        SummaryPlan::DetectionDefault
    };
    let eval_imgs = grid_inner.eval_imgs.as_slice();
    let summary = py
        .detach(
            move || -> Result<PartitionedSummary, vernier_core::EvalError> {
                evaluate_partitioned(eval_imgs, dims, &spec, &iou_thr, parity, plan)
            },
        )
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;

    Ok(PyPartitionedSummary { inner: summary })
}

/// Bbox partitioned eval (ADR-0046).
///
/// Symmetric to [`crate::evaluate_bbox_grid`] but extended with a
/// `manifest=` parameter and an optional `cross_axes=` list.
/// Returns a [`PyPartitionedSummary`] whose `.overall` is bit-
/// identical to the un-partitioned summary.
#[pyfunction]
#[pyo3(signature = (
    gt_json,
    dt,
    parity_mode,
    max_dets_per_image,
    use_cats,
    manifest,
    cast_inputs = false,
    iou_thresholds = None,
    recall_thresholds = None,
    area_ranges = None,
    cross_axes = None,
    key_kind = "image_id",
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_bbox_partitioned<'py>(
    py: Python<'py>,
    gt_json: &Bound<'py, PyBytes>,
    dt: &Bound<'py, PyAny>,
    parity_mode: &str,
    max_dets_per_image: usize,
    use_cats: bool,
    manifest: &Bound<'py, PyAny>,
    cast_inputs: bool,
    iou_thresholds: Option<Vec<f64>>,
    recall_thresholds: Option<Vec<f64>>,
    area_ranges: Option<&Bound<'py, breakdown::PyBreakdown>>,
    cross_axes: Option<Vec<Vec<String>>>,
    key_kind: &str,
) -> PyResult<PyPartitionedSummary> {
    evaluate_instance_partitioned_impl(
        py,
        EvalIouType::Bbox,
        gt_json,
        dt,
        parity_mode,
        max_dets_per_image,
        use_cats,
        cast_inputs,
        iou_thresholds,
        recall_thresholds,
        area_ranges,
        manifest,
        cross_axes,
        key_kind,
        false,
    )
}

/// Segm partitioned eval. Mirrors [`evaluate_bbox_partitioned`] for
/// the segmentation kernel.
#[pyfunction]
#[pyo3(signature = (
    gt_json,
    dt,
    parity_mode,
    max_dets_per_image,
    use_cats,
    manifest,
    cast_inputs = false,
    iou_thresholds = None,
    recall_thresholds = None,
    area_ranges = None,
    cross_axes = None,
    key_kind = "image_id",
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_segm_partitioned<'py>(
    py: Python<'py>,
    gt_json: &Bound<'py, PyBytes>,
    dt: &Bound<'py, PyAny>,
    parity_mode: &str,
    max_dets_per_image: usize,
    use_cats: bool,
    manifest: &Bound<'py, PyAny>,
    cast_inputs: bool,
    iou_thresholds: Option<Vec<f64>>,
    recall_thresholds: Option<Vec<f64>>,
    area_ranges: Option<&Bound<'py, breakdown::PyBreakdown>>,
    cross_axes: Option<Vec<Vec<String>>>,
    key_kind: &str,
) -> PyResult<PyPartitionedSummary> {
    evaluate_instance_partitioned_impl(
        py,
        EvalIouType::Segm,
        gt_json,
        dt,
        parity_mode,
        max_dets_per_image,
        use_cats,
        cast_inputs,
        iou_thresholds,
        recall_thresholds,
        area_ranges,
        manifest,
        cross_axes,
        key_kind,
        false,
    )
}

/// Boundary partitioned eval. Mirrors [`evaluate_bbox_partitioned`]
/// for the boundary-IoU kernel (ADR-0010).
#[pyfunction]
#[pyo3(signature = (
    gt_json,
    dt,
    parity_mode,
    max_dets_per_image,
    use_cats,
    dilation_ratio,
    manifest,
    cast_inputs = false,
    iou_thresholds = None,
    recall_thresholds = None,
    area_ranges = None,
    cross_axes = None,
    key_kind = "image_id",
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_boundary_partitioned<'py>(
    py: Python<'py>,
    gt_json: &Bound<'py, PyBytes>,
    dt: &Bound<'py, PyAny>,
    parity_mode: &str,
    max_dets_per_image: usize,
    use_cats: bool,
    dilation_ratio: f64,
    manifest: &Bound<'py, PyAny>,
    cast_inputs: bool,
    iou_thresholds: Option<Vec<f64>>,
    recall_thresholds: Option<Vec<f64>>,
    area_ranges: Option<&Bound<'py, breakdown::PyBreakdown>>,
    cross_axes: Option<Vec<Vec<String>>>,
    key_kind: &str,
) -> PyResult<PyPartitionedSummary> {
    let iou_type = boundary_iou_type(dilation_ratio)?;
    evaluate_instance_partitioned_impl(
        py,
        iou_type,
        gt_json,
        dt,
        parity_mode,
        max_dets_per_image,
        use_cats,
        cast_inputs,
        iou_thresholds,
        recall_thresholds,
        area_ranges,
        manifest,
        cross_axes,
        key_kind,
        false,
    )
}

/// Keypoints partitioned eval. Mirrors [`evaluate_bbox_partitioned`]
/// for the OKS kernel (ADR-0012); summarized with the keypoints
/// 10-stat plan.
#[pyfunction]
#[pyo3(signature = (
    gt_json,
    dt,
    parity_mode,
    max_dets_per_image,
    use_cats,
    sigmas,
    manifest,
    cast_inputs = false,
    iou_thresholds = None,
    recall_thresholds = None,
    area_ranges = None,
    cross_axes = None,
    key_kind = "image_id",
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_keypoints_partitioned<'py>(
    py: Python<'py>,
    gt_json: &Bound<'py, PyBytes>,
    dt: &Bound<'py, PyAny>,
    parity_mode: &str,
    max_dets_per_image: usize,
    use_cats: bool,
    sigmas: &Bound<'py, PyDict>,
    manifest: &Bound<'py, PyAny>,
    cast_inputs: bool,
    iou_thresholds: Option<Vec<f64>>,
    recall_thresholds: Option<Vec<f64>>,
    area_ranges: Option<&Bound<'py, breakdown::PyBreakdown>>,
    cross_axes: Option<Vec<Vec<String>>>,
    key_kind: &str,
) -> PyResult<PyPartitionedSummary> {
    let iou_type = EvalIouType::Keypoints {
        sigmas: parse_sigmas(sigmas)?,
    };
    evaluate_instance_partitioned_impl(
        py,
        iou_type,
        gt_json,
        dt,
        parity_mode,
        max_dets_per_image,
        use_cats,
        cast_inputs,
        iou_thresholds,
        recall_thresholds,
        area_ranges,
        manifest,
        cross_axes,
        key_kind,
        true,
    )
}

// ---------------------------------------------------------------------------
// LRP partitioned (deferred — typed error pointing at the follow-up)
// ---------------------------------------------------------------------------

const LRP_DEFERRED_MSG: &str = "partitioned LRP is wired but its decomposition pipeline is \
                                pending; this is the ADR-0046 phase-1 follow-up. Run un-partitioned \
                                LRP today (`Evaluator.lrp(...)` without `manifest=`) or, for slice-\
                                level LRP, loop the un-partitioned call with a filtered DT subset.";

/// Bbox LRP partitioned eval — deferred. Raises [`PyRuntimeError`].
#[pyfunction]
#[pyo3(signature = (
    _gt_bytes,
    _dt_bytes,
    _parity_mode,
    _tp_threshold,
    _tau_grid,
    _max_dets_per_image,
    _use_cats,
    _manifest,
    _cross_axes = None,
    _key_kind = "image_id",
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_bbox_partitioned_lrp(
    _gt_bytes: &Bound<'_, PyBytes>,
    _dt_bytes: &Bound<'_, PyBytes>,
    _parity_mode: &str,
    _tp_threshold: f64,
    _tau_grid: Vec<f64>,
    _max_dets_per_image: usize,
    _use_cats: bool,
    _manifest: &Bound<'_, PyAny>,
    _cross_axes: Option<Vec<Vec<String>>>,
    _key_kind: &str,
) -> PyResult<()> {
    Err(PyRuntimeError::new_err(LRP_DEFERRED_MSG))
}

/// Segm LRP partitioned eval — deferred. Raises [`PyRuntimeError`].
#[pyfunction]
#[pyo3(signature = (
    _gt_bytes,
    _dt_bytes,
    _parity_mode,
    _tp_threshold,
    _tau_grid,
    _max_dets_per_image,
    _use_cats,
    _manifest,
    _cross_axes = None,
    _key_kind = "image_id",
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_segm_partitioned_lrp(
    _gt_bytes: &Bound<'_, PyBytes>,
    _dt_bytes: &Bound<'_, PyBytes>,
    _parity_mode: &str,
    _tp_threshold: f64,
    _tau_grid: Vec<f64>,
    _max_dets_per_image: usize,
    _use_cats: bool,
    _manifest: &Bound<'_, PyAny>,
    _cross_axes: Option<Vec<Vec<String>>>,
    _key_kind: &str,
) -> PyResult<()> {
    Err(PyRuntimeError::new_err(LRP_DEFERRED_MSG))
}

/// Boundary LRP partitioned eval — deferred. Raises [`PyRuntimeError`].
#[pyfunction]
#[pyo3(signature = (
    _gt_bytes,
    _dt_bytes,
    _parity_mode,
    _tp_threshold,
    _tau_grid,
    _max_dets_per_image,
    _use_cats,
    _dilation_ratio,
    _manifest,
    _cross_axes = None,
    _key_kind = "image_id",
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_boundary_partitioned_lrp(
    _gt_bytes: &Bound<'_, PyBytes>,
    _dt_bytes: &Bound<'_, PyBytes>,
    _parity_mode: &str,
    _tp_threshold: f64,
    _tau_grid: Vec<f64>,
    _max_dets_per_image: usize,
    _use_cats: bool,
    _dilation_ratio: f64,
    _manifest: &Bound<'_, PyAny>,
    _cross_axes: Option<Vec<Vec<String>>>,
    _key_kind: &str,
) -> PyResult<()> {
    Err(PyRuntimeError::new_err(LRP_DEFERRED_MSG))
}

/// Keypoints LRP partitioned eval — deferred. Raises [`PyRuntimeError`].
#[pyfunction]
#[pyo3(signature = (
    _gt_bytes,
    _dt_bytes,
    _parity_mode,
    _tp_threshold,
    _tau_grid,
    _max_dets_per_image,
    _use_cats,
    _sigmas,
    _manifest,
    _cross_axes = None,
    _key_kind = "image_id",
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_keypoints_partitioned_lrp(
    _gt_bytes: &Bound<'_, PyBytes>,
    _dt_bytes: &Bound<'_, PyBytes>,
    _parity_mode: &str,
    _tp_threshold: f64,
    _tau_grid: Vec<f64>,
    _max_dets_per_image: usize,
    _use_cats: bool,
    _sigmas: &Bound<'_, PyDict>,
    _manifest: &Bound<'_, PyAny>,
    _cross_axes: Option<Vec<Vec<String>>>,
    _key_kind: &str,
) -> PyResult<()> {
    Err(PyRuntimeError::new_err(LRP_DEFERRED_MSG))
}

// ---------------------------------------------------------------------------
// Panoptic / semantic slice tables (Python-orchestrated; the Rust
// side just packs the rows).
//
// The panoptic and semantic paradigms compute their summaries from
// per-image accumulations rather than from an AP-shaped accumulator
// tensor. Cleanly subsetting that pipeline would require refactoring
// both paradigms' summarize stage to take an image-id filter. As a
// pragmatic fallback (per ADR-0046 §"Performance"), the Python
// wrapper loops `Evaluator.evaluate(...)` once per slice and feeds
// the resulting metrics back through these two builders, getting a
// uniform Arrow-RecordBatch surface across all three paradigms. This
// is the C1 path; LVIS-scale panoptic / semantic users who want C3
// performance for partitioned eval should file an issue.
// ---------------------------------------------------------------------------

/// Panoptic per-slice row tuple shape as it crosses the FFI boundary
/// from the Python loop. `(axis, value, n_images, n_detections, pq,
/// sq, rq)`.
type PanopticRowTuple = (String, String, u64, u64, f64, f64, f64);

/// Semantic per-slice row tuple. `(axis, value, n_images,
/// n_detections, miou, fwiou, pixel_accuracy, mean_accuracy)`.
type SemanticRowTuple = (String, String, u64, u64, f64, f64, f64, f64);

/// Build the panoptic slices RecordBatch from a Python list of
/// per-slice tuples `(axis, value, n_images, n_detections, pq, sq,
/// rq)`. The Python wrapper produces the tuples by running the
/// unchanged `evaluate_panoptic` over each slice's image subset.
#[pyfunction]
pub(crate) fn slices_batch_panoptic(rows: Vec<PanopticRowTuple>) -> PyResult<ArrowRecordBatchPy> {
    let rows: Vec<PanopticSliceRow> = rows
        .into_iter()
        .map(
            |(axis, value, n_images, n_detections, pq, sq, rq)| PanopticSliceRow {
                axis,
                value,
                n_images,
                n_detections,
                pq,
                sq,
                rq,
            },
        )
        .collect();
    let batch = slices_record_batch_panoptic(&rows)
        .map_err(|e| PyValueError::new_err(format!("arrow build failed: {e}")))?;
    Ok(wrap_batch(batch))
}

/// Build the semantic slices RecordBatch from a Python list of
/// per-slice tuples `(axis, value, n_images, n_detections, miou,
/// fwiou, pixel_accuracy, mean_accuracy)`.
#[pyfunction]
pub(crate) fn slices_batch_semantic(rows: Vec<SemanticRowTuple>) -> PyResult<ArrowRecordBatchPy> {
    let rows: Vec<SemanticSliceRow> = rows
        .into_iter()
        .map(
            |(axis, value, n_images, n_detections, miou, fwiou, pa, ma)| SemanticSliceRow {
                axis,
                value,
                n_images,
                n_detections,
                miou,
                fwiou,
                pixel_accuracy: pa,
                mean_accuracy: ma,
            },
        )
        .collect();
    let batch = slices_record_batch_semantic(&rows)
        .map_err(|e| PyValueError::new_err(format!("arrow build failed: {e}")))?;
    Ok(wrap_batch(batch))
}

/// Build the canonical manifest JSON bytes from any of the inputs
/// `manifest_to_canonical_json` accepts. Exposed so the Python lane
/// can pre-resolve a manifest to bytes for the panoptic / semantic
/// per-slice loop (which needs the image-id assignments but does not
/// go through `evaluate_partitioned`).
#[pyfunction]
#[pyo3(signature = (manifest, key_kind = "image_id"))]
pub(crate) fn manifest_to_json_bytes<'py>(
    py: Python<'py>,
    manifest: &Bound<'py, PyAny>,
    key_kind: &str,
) -> PyResult<Bound<'py, PyBytes>> {
    let bytes = manifest_to_canonical_json(py, manifest, key_kind)?;
    Ok(PyBytes::new(py, &bytes))
}
