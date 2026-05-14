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

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict, PyList};

use vernier_core::evaluate::AreaRange;
use vernier_core::lrp::{LrpKernelMarker, LrpParams, LrpPerClass, LrpReport};
use vernier_core::manifest::partition_spec_from_manifest;
use vernier_core::partition::{
    evaluate_partitioned, evaluate_partitioned_lrp, image_id_to_idx, GridDims,
    PartitionedLrpReport, PartitionedSummary, SummaryPlan,
};
use vernier_core::similarity::{BboxIou, BoundaryIou, OksSimilarity, SegmIou};
use vernier_core::{CocoDataset, CocoDetections, EvalError};

use crate::arrow_helpers::{wrap_batch, ArrowRecordBatchPy};
use crate::breakdown;
use crate::manifest_py::manifest_to_canonical_json;
use crate::tables::{
    slices_instance_ap_to_arrow, slices_instance_lrp_to_arrow, slices_record_batch_panoptic,
    slices_record_batch_semantic, PanopticSliceRow, SemanticSliceRow,
};
use crate::{
    boundary_iou_type, evaluate_grid_impl, parse_dt, parse_gt, parse_parity_mode, parse_sigmas,
    validate_dilation_ratio, EvalIouType, PySummary,
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
// LRP partitioned (ADR-0046 phase-1 follow-up — real implementation)
// ---------------------------------------------------------------------------

/// Result of an instance-LRP partitioned evaluate. The `overall`
/// report is bit-identical to the un-partitioned LRP path; `slices`
/// is the Arrow `RecordBatch` carrying one row per `(axis, value)`
/// cell with the four headline LRP stats.
#[pyclass(module = "vernier._core", name = "PartitionedLrpReport", frozen)]
pub(crate) struct PyPartitionedLrpReport {
    inner: PartitionedLrpReport,
}

#[pymethods]
impl PyPartitionedLrpReport {
    /// Bit-identical to the un-partitioned LRP report. Same dict shape
    /// as the per-kernel `optimal_lrp_*` pyfunctions return.
    #[getter]
    fn overall<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        lrp_report_to_dict(py, &self.inner.overall)
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

    /// Build the slices Arrow RecordBatch and return its PyCapsule-
    /// exporting wrapper. Materialised on demand so a call that never
    /// reads `.slices` pays no Arrow build cost.
    fn slices_capsule(&self) -> PyResult<ArrowRecordBatchPy> {
        slices_instance_lrp_to_arrow(&self.inner)
            .map_err(|e| PyValueError::new_err(format!("arrow build failed: {e}")))
    }

    fn __repr__(&self) -> String {
        format!(
            "PartitionedLrpReport(overall_n_images={}, overall_n_detections={}, n_slices={})",
            self.inner.overall_n_images,
            self.inner.overall_n_detections,
            self.inner.slices.len()
        )
    }
}

/// Closure invoked inside `py.detach` to dispatch the kernel-specific
/// [`evaluate_partitioned_lrp`] call. The four kernel-specific
/// pyfunctions below close over a `LrpKernelDispatch` so the shared
/// boilerplate (parse, manifest resolve, GIL release) lives in one
/// place.
type LrpKernelDispatch = Box<
    dyn FnOnce(
            &CocoDataset,
            &CocoDetections,
            LrpParams<'_>,
            vernier_core::ParityMode,
            &vernier_core::partition::PartitionSpec,
        ) -> Result<PartitionedLrpReport, EvalError>
        + Send,
>;

/// Shared per-kernel orchestration for partitioned LRP.
///
/// 1. Parse `gt` and `dt` off the GIL via `py.detach`.
/// 2. Build `image_id_to_idx` from the parsed GT, resolve the
///    manifest into a partition spec, emit warnings.
/// 3. Run [`evaluate_partitioned_lrp`] (1× matching pass; N+1 cheap
///    decompose passes) off the GIL.
/// 4. Wrap into a [`PyPartitionedLrpReport`] for the user.
#[allow(clippy::too_many_arguments)]
fn evaluate_instance_partitioned_lrp_impl(
    py: Python<'_>,
    gt_bytes: &Bound<'_, PyBytes>,
    dt_bytes: &Bound<'_, PyBytes>,
    parity_mode: &str,
    tp_threshold: f64,
    tau_grid: Vec<f64>,
    max_dets_per_image: usize,
    use_cats: bool,
    manifest: &Bound<'_, PyAny>,
    cross_axes: Option<Vec<Vec<String>>>,
    key_kind: &str,
    dispatch: LrpKernelDispatch,
) -> PyResult<PyPartitionedLrpReport> {
    let parity = parse_parity_mode(parity_mode)?;
    let gt_vec = gt_bytes.as_bytes().to_vec();
    let dt_vec = dt_bytes.as_bytes().to_vec();
    let manifest_bytes = manifest_to_canonical_json(py, manifest, key_kind)?;
    let cross = cross_axes.unwrap_or_default();

    // 1) Parse GT + DT off the GIL.
    type ParseResult = (CocoDataset, CocoDetections);
    let (gt, dt) = py
        .detach(move || -> Result<ParseResult, EvalError> {
            let gt = parse_gt(&gt_vec).map_err(|e| EvalError::InvalidConfig {
                detail: format!("{e}"),
            })?;
            let dt = parse_dt(&dt_vec).map_err(|e| EvalError::InvalidConfig {
                detail: format!("{e}"),
            })?;
            Ok((gt, dt))
        })
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;

    // 2) Resolve manifest under the GIL so manifest warnings can
    //    surface to Python's warnings module before the heavy work.
    let id_map = image_id_to_idx(&gt);
    let (spec, warnings) = partition_spec_from_manifest(&manifest_bytes, &id_map, &cross)
        .map_err(|e| PyValueError::new_err(format!("manifest resolution failed: {e}")))?;
    warn_about_manifest(py, &warnings)?;

    // 3) Run partition LRP off the GIL.
    let report = py
        .detach(move || -> Result<PartitionedLrpReport, EvalError> {
            // LRP runs the matching pass internally; `[tp_threshold]`
            // is the minimal IoU ladder so `evaluate_with` does not
            // reject an empty list.
            let iou_thresholds = [tp_threshold];
            let area_ranges = AreaRange::coco_default();
            let params = LrpParams {
                tp_threshold,
                tau_grid: &tau_grid,
                max_dets_per_image,
                use_cats,
                iou_thresholds: &iou_thresholds,
                area_ranges: &area_ranges,
            };
            dispatch(&gt, &dt, params, parity, &spec)
        })
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;

    Ok(PyPartitionedLrpReport { inner: report })
}

/// Bbox LRP partitioned eval (ADR-0046 phase-1 follow-up).
///
/// Builds a [`PartitionedLrpReport`] over a `(gt, dt)` pair partitioned
/// by `manifest`. The matching pass runs **once** internally; the
/// per-class decompose pipeline runs once for the overall report and
/// once per slice with an I-axis filter (C3).
#[pyfunction]
#[pyo3(signature = (
    gt_bytes,
    dt_bytes,
    parity_mode,
    tp_threshold,
    tau_grid,
    max_dets_per_image,
    use_cats,
    manifest,
    cross_axes = None,
    key_kind = "image_id",
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_bbox_partitioned_lrp(
    py: Python<'_>,
    gt_bytes: &Bound<'_, PyBytes>,
    dt_bytes: &Bound<'_, PyBytes>,
    parity_mode: &str,
    tp_threshold: f64,
    tau_grid: Vec<f64>,
    max_dets_per_image: usize,
    use_cats: bool,
    manifest: &Bound<'_, PyAny>,
    cross_axes: Option<Vec<Vec<String>>>,
    key_kind: &str,
) -> PyResult<PyPartitionedLrpReport> {
    evaluate_instance_partitioned_lrp_impl(
        py,
        gt_bytes,
        dt_bytes,
        parity_mode,
        tp_threshold,
        tau_grid,
        max_dets_per_image,
        use_cats,
        manifest,
        cross_axes,
        key_kind,
        Box::new(|gt, dt, params, parity, spec| {
            evaluate_partitioned_lrp(
                gt,
                dt,
                &BboxIou,
                LrpKernelMarker::Bbox,
                params,
                parity,
                spec,
            )
        }),
    )
}

/// Segm LRP partitioned eval. Mirrors [`evaluate_bbox_partitioned_lrp`]
/// for the segmentation-mask IoU kernel.
#[pyfunction]
#[pyo3(signature = (
    gt_bytes,
    dt_bytes,
    parity_mode,
    tp_threshold,
    tau_grid,
    max_dets_per_image,
    use_cats,
    manifest,
    cross_axes = None,
    key_kind = "image_id",
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_segm_partitioned_lrp(
    py: Python<'_>,
    gt_bytes: &Bound<'_, PyBytes>,
    dt_bytes: &Bound<'_, PyBytes>,
    parity_mode: &str,
    tp_threshold: f64,
    tau_grid: Vec<f64>,
    max_dets_per_image: usize,
    use_cats: bool,
    manifest: &Bound<'_, PyAny>,
    cross_axes: Option<Vec<Vec<String>>>,
    key_kind: &str,
) -> PyResult<PyPartitionedLrpReport> {
    evaluate_instance_partitioned_lrp_impl(
        py,
        gt_bytes,
        dt_bytes,
        parity_mode,
        tp_threshold,
        tau_grid,
        max_dets_per_image,
        use_cats,
        manifest,
        cross_axes,
        key_kind,
        Box::new(|gt, dt, params, parity, spec| {
            evaluate_partitioned_lrp(
                gt,
                dt,
                &SegmIou,
                LrpKernelMarker::Segm,
                params,
                parity,
                spec,
            )
        }),
    )
}

/// Boundary LRP partitioned eval (ADR-0010 + ADR-0046).
#[pyfunction]
#[pyo3(signature = (
    gt_bytes,
    dt_bytes,
    parity_mode,
    tp_threshold,
    tau_grid,
    max_dets_per_image,
    use_cats,
    dilation_ratio,
    manifest,
    cross_axes = None,
    key_kind = "image_id",
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_boundary_partitioned_lrp(
    py: Python<'_>,
    gt_bytes: &Bound<'_, PyBytes>,
    dt_bytes: &Bound<'_, PyBytes>,
    parity_mode: &str,
    tp_threshold: f64,
    tau_grid: Vec<f64>,
    max_dets_per_image: usize,
    use_cats: bool,
    dilation_ratio: f64,
    manifest: &Bound<'_, PyAny>,
    cross_axes: Option<Vec<Vec<String>>>,
    key_kind: &str,
) -> PyResult<PyPartitionedLrpReport> {
    validate_dilation_ratio(dilation_ratio)?;
    evaluate_instance_partitioned_lrp_impl(
        py,
        gt_bytes,
        dt_bytes,
        parity_mode,
        tp_threshold,
        tau_grid,
        max_dets_per_image,
        use_cats,
        manifest,
        cross_axes,
        key_kind,
        Box::new(move |gt, dt, params, parity, spec| {
            let kernel = BoundaryIou { dilation_ratio };
            evaluate_partitioned_lrp(
                gt,
                dt,
                &kernel,
                LrpKernelMarker::Boundary,
                params,
                parity,
                spec,
            )
        }),
    )
}

/// Keypoints (OKS) LRP partitioned eval (ADR-0045 + ADR-0046).
#[pyfunction]
#[pyo3(signature = (
    gt_bytes,
    dt_bytes,
    parity_mode,
    tp_threshold,
    tau_grid,
    max_dets_per_image,
    use_cats,
    sigmas,
    manifest,
    cross_axes = None,
    key_kind = "image_id",
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_keypoints_partitioned_lrp(
    py: Python<'_>,
    gt_bytes: &Bound<'_, PyBytes>,
    dt_bytes: &Bound<'_, PyBytes>,
    parity_mode: &str,
    tp_threshold: f64,
    tau_grid: Vec<f64>,
    max_dets_per_image: usize,
    use_cats: bool,
    sigmas: &Bound<'_, PyDict>,
    manifest: &Bound<'_, PyAny>,
    cross_axes: Option<Vec<Vec<String>>>,
    key_kind: &str,
) -> PyResult<PyPartitionedLrpReport> {
    let sigmas_map = parse_sigmas(sigmas)?;
    evaluate_instance_partitioned_lrp_impl(
        py,
        gt_bytes,
        dt_bytes,
        parity_mode,
        tp_threshold,
        tau_grid,
        max_dets_per_image,
        use_cats,
        manifest,
        cross_axes,
        key_kind,
        Box::new(move |gt, dt, params, parity, spec| {
            let kernel = OksSimilarity::new(sigmas_map);
            evaluate_partitioned_lrp(
                gt,
                dt,
                &kernel,
                LrpKernelMarker::Keypoints,
                params,
                parity,
                spec,
            )
        }),
    )
}

/// Translate an [`LrpReport`] into the Python dict shape pinned in
/// `crates/vernier-ffi/src/lrp.rs`. Kept in this module rather than
/// reaching into the LRP module to avoid a `pub(crate)` widening of
/// the existing private helper.
fn lrp_report_to_dict<'py>(py: Python<'py>, report: &LrpReport) -> PyResult<Bound<'py, PyDict>> {
    let per_class = PyList::empty(py);
    for entry in &report.per_class {
        per_class.append(per_class_to_dict(py, entry)?)?;
    }
    let config = PyDict::new(py);
    config.set_item("tp_threshold", report.config.tp_threshold)?;
    config.set_item("tau_grid_len", report.config.tau_grid_len)?;
    config.set_item("kernel", report.config.kernel.as_str())?;
    let out = PyDict::new(py);
    out.set_item("olrp", report.olrp)?;
    out.set_item("loc", report.olrp_loc)?;
    out.set_item("fp", report.olrp_fp)?;
    out.set_item("fn", report.olrp_fn)?;
    out.set_item("per_class", per_class)?;
    out.set_item("n_empty_classes", report.n_empty_classes)?;
    out.set_item("config", config)?;
    Ok(out)
}

fn per_class_to_dict<'py>(py: Python<'py>, entry: &LrpPerClass) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("category_id", entry.category_id)?;
    set_optional_f64(&d, "olrp", entry.olrp)?;
    set_optional_f64(&d, "olrp_loc", entry.olrp_loc)?;
    set_optional_f64(&d, "olrp_fp", entry.olrp_fp)?;
    set_optional_f64(&d, "olrp_fn", entry.olrp_fn)?;
    set_optional_f64(&d, "tau", entry.tau)?;
    Ok(d)
}

fn set_optional_f64(dict: &Bound<'_, PyDict>, key: &str, value: Option<f64>) -> PyResult<()> {
    match value {
        Some(v) => dict.set_item(key, v),
        None => dict.set_item(key, dict.py().None()),
    }
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
