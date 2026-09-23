//! Wraps the kernel-specific entry points in [`vernier_core::tide`] and
//! exposes them to Python: `error_decomposition_bbox`,
//! `error_decomposition_segm`, `error_decomposition_boundary`. By
//! policy, this module contains only data conversion — the eight-pass
//! orchestration and the cell-rewrite layer live in
//! [`vernier_core::tide`].
//!
//! ## Output shape
//!
//! Both kernels return a `dict` with the exact shape the numpy oracle in
//! `tests/python/oracle/tide/oracle.py` produces, so the parity test can
//! compare bin-for-bin without any field renames:
//!
//! ```text
//! {
//!     "baseline_map": float,
//!     "delta": {
//!         "cls": float, "loc": float, "both": float,
//!         "dupe": float, "bkg": float, "missed": float,
//!     },
//!     "delta_all_fp_removed": float,
//!     "config": {"t_f": float, "t_b": float, "kernel": str},
//! }
//! ```
//!
//! The `config.kernel` string is `"bbox"` / `"segm"` / `"boundary"` for
//! the corresponding kernel-specific entry points.
//!
//! [`vernier_core::tide::report::TideReport`] stores `delta_per_bin` as a
//! sparse `HashMap` (per its docstring, bins not populated by the rewrite
//! layer — e.g. structurally-zero `Cls`/`Both` on a single-class workload —
//! are simply absent). The FFI fills the six known bin keys with `0.0` on
//! absence so the dict shape stays stable for downstream consumers.

use numpy::ndarray::Array1;
use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use vernier_core::parity::{iou_thresholds, recall_thresholds};
use vernier_core::tide::{
    self, FpIouHistogram, KernelMarker, TideErrorBin, TideParams, TideReport,
};
use vernier_core::{AreaRange, CocoDetections, EvalError, ParityMode};

use crate::array_ingest::ArrayIouType;
use crate::dataset::DatasetSnapshot;
use crate::{parse_parity_mode, validate_dilation_ratio, DiagnosticInputs};

/// Shared plumbing for the TIDE entry points: resolve `gt` / `dt`
/// (ADR-0064), run `kernel_call` off the GIL, build the report dict.
/// `kernel_call` picks the kernel; the mask kernels reuse the snapshot's
/// GT caches across TIDE's eight evaluation passes.
#[allow(clippy::too_many_arguments)]
fn run_tide_pass<'py, F>(
    py: Python<'py>,
    gt: &Bound<'py, PyAny>,
    dt: &Bound<'py, PyAny>,
    kernel: ArrayIouType,
    parity_mode: &str,
    t_f: f64,
    t_b: f64,
    max_dets_per_image: usize,
    use_cats: bool,
    cast_inputs: bool,
    kernel_call: F,
) -> PyResult<Bound<'py, PyDict>>
where
    F: FnOnce(
            &DatasetSnapshot,
            &CocoDetections,
            TideParams<'_>,
            ParityMode,
        ) -> Result<TideReport, EvalError>
        + Send,
{
    let parity = parse_parity_mode(parity_mode)?;
    let inputs = DiagnosticInputs::extract(py, gt, dt, kernel, cast_inputs, "error_decomposition")?;

    let report = py.detach(move || -> PyResult<TideReport> {
        let (gt, dt) = inputs.realize()?;
        let area_ranges = AreaRange::coco_default();
        let params = TideParams {
            t_f,
            t_b,
            max_dets_per_image,
            use_cats,
            iou_thresholds: iou_thresholds(),
            recall_thresholds: recall_thresholds(),
            area_ranges: &area_ranges,
        };
        kernel_call(&gt, &dt, params, parity).map_err(|e| PyValueError::new_err(format!("{e}")))
    })?;

    report_to_dict(py, &report)
}

/// TIDE error decomposition for the bbox kernel (ADR-0021).
///
/// `gt` is the COCO ground-truth JSON payload as `bytes` or a parsed
/// `CocoDataset` handle; `dt` is any of the detection forms the
/// evaluator accepts — results JSON `bytes`, columnar `Detections`,
/// result dicts, or an `(N, 7)` matrix (ADR-0030, ADR-0057, ADR-0064).
/// `cast_inputs` converts array dtypes rather than refusing them, as on
/// `Evaluator`. `parity_mode` is `"strict"` or `"corrected"`
/// per ADR-0002. `t_f` and `t_b` are the foreground / background
/// thresholds; ADR-0022 pins the bbox defaults at `0.5` / `0.1`.
/// `max_dets_per_image` matches the oracle's per-image cap (the oracle
/// uses `100` by default). `use_cats` mirrors pycocotools' `useCats`.
///
/// Returns the report dict described in the module docstring.
#[pyfunction]
#[pyo3(signature = (gt, dt, parity_mode, t_f, t_b, max_dets_per_image, use_cats, *, cast_inputs = false))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn error_decomposition_bbox<'py>(
    py: Python<'py>,
    gt: &Bound<'py, PyAny>,
    dt: &Bound<'py, PyAny>,
    parity_mode: &str,
    t_f: f64,
    t_b: f64,
    max_dets_per_image: usize,
    use_cats: bool,
    cast_inputs: bool,
) -> PyResult<Bound<'py, PyDict>> {
    run_tide_pass(
        py,
        gt,
        dt,
        ArrayIouType::Bbox,
        parity_mode,
        t_f,
        t_b,
        max_dets_per_image,
        use_cats,
        cast_inputs,
        |gt, dt, params, parity| tide::error_decomposition_bbox(&gt.gt, dt, params, parity),
    )
}

/// TIDE error decomposition for the segm kernel (ADR-0021, Week 3).
///
/// Same signature as [`error_decomposition_bbox`] above; the only
/// per-call difference is the kernel — `gt` / `dt` must
/// carry COCO `segmentation` fields under `iouType="segm"` semantics
/// (polygon or RLE; the J2-strict path synthesizes a rectangle polygon
/// from a DT bbox when the DT lacks a `segmentation` field, matching
/// pycocotools).
///
/// ADR-0022 currently defaults segm `t_b` to `0.1` (anchored on the bbox
/// row pending real-world measurement on COCO val2017 — see the ADR for
/// the open caveat). Callers who want a different value pass it
/// explicitly.
///
/// Returns the report dict described in the module docstring (with
/// `config.kernel = "segm"`).
#[pyfunction]
#[pyo3(signature = (gt, dt, parity_mode, t_f, t_b, max_dets_per_image, use_cats, *, cast_inputs = false))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn error_decomposition_segm<'py>(
    py: Python<'py>,
    gt: &Bound<'py, PyAny>,
    dt: &Bound<'py, PyAny>,
    parity_mode: &str,
    t_f: f64,
    t_b: f64,
    max_dets_per_image: usize,
    use_cats: bool,
    cast_inputs: bool,
) -> PyResult<Bound<'py, PyDict>> {
    run_tide_pass(
        py,
        gt,
        dt,
        ArrayIouType::Segm,
        parity_mode,
        t_f,
        t_b,
        max_dets_per_image,
        use_cats,
        cast_inputs,
        |gt, dt, params, parity| {
            let kernel = gt.segm_kernel();
            tide::error_decomposition_with(&gt.gt, dt, &kernel, KernelMarker::Segm, params, parity)
        },
    )
}

/// TIDE error decomposition for the boundary-segm kernel (ADR-0010 +
/// ADR-0021).
///
/// Same shape as [`error_decomposition_bbox`] except `dilation_ratio`
/// pins the boundary band thickness (ADR-0010 default `0.02` for COCO,
/// `0.008` for LVIS) and the report's `kernel` field reads
/// `"boundary"`. ADR-0022 pins the boundary defaults at `t_f = 0.5`,
/// `t_b = 0.05` (tentative; see the ADR's "Decision gate (boundary
/// default)" section for the empirical-anchoring follow-up plan).
///
/// Both `gt` and `dt` must carry `segmentation` fields —
/// the same constraint `evaluate_boundary_summary` enforces; polygon
/// and RLE shapes both work.
#[pyfunction]
#[pyo3(signature = (gt, dt, parity_mode, t_f, t_b, max_dets_per_image, use_cats, dilation_ratio, *, cast_inputs = false))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn error_decomposition_boundary<'py>(
    py: Python<'py>,
    gt: &Bound<'py, PyAny>,
    dt: &Bound<'py, PyAny>,
    parity_mode: &str,
    t_f: f64,
    t_b: f64,
    max_dets_per_image: usize,
    use_cats: bool,
    dilation_ratio: f64,
    cast_inputs: bool,
) -> PyResult<Bound<'py, PyDict>> {
    validate_dilation_ratio(dilation_ratio)?;
    run_tide_pass(
        py,
        gt,
        dt,
        ArrayIouType::Boundary,
        parity_mode,
        t_f,
        t_b,
        max_dets_per_image,
        use_cats,
        cast_inputs,
        move |gt, dt, params, parity| {
            let kernel = gt.boundary_kernel(dilation_ratio);
            tide::error_decomposition_with(
                &gt.gt,
                dt,
                &kernel,
                KernelMarker::Boundary,
                params,
                parity,
            )
        },
    )
}

/// Shared plumbing for the FP-IoU histogram entry points (ADR-0022);
/// same shape as [`run_tide_pass`].
#[allow(clippy::too_many_arguments)]
fn run_fp_histogram_pass<'py, F>(
    py: Python<'py>,
    gt: &Bound<'py, PyAny>,
    dt: &Bound<'py, PyAny>,
    kernel: ArrayIouType,
    parity_mode: &str,
    t_f: f64,
    max_dets_per_image: usize,
    use_cats: bool,
    cast_inputs: bool,
    kernel_call: F,
) -> PyResult<Bound<'py, PyDict>>
where
    F: FnOnce(
            &DatasetSnapshot,
            &CocoDetections,
            TideParams<'_>,
            ParityMode,
        ) -> Result<FpIouHistogram, EvalError>
        + Send,
{
    let parity = parse_parity_mode(parity_mode)?;
    let inputs = DiagnosticInputs::extract(py, gt, dt, kernel, cast_inputs, "fp_iou_histogram")?;

    let mut histogram = py.detach(move || -> PyResult<FpIouHistogram> {
        let (gt, dt) = inputs.realize()?;
        let area_ranges = AreaRange::coco_default();
        // `t_b` rides along on TideParams but the histogram extractor
        // ignores it (Bkg cutoff is decided Python-side from the
        // emitted IoUs); 0.0 keeps the field defined without
        // implying anything.
        let params = TideParams {
            t_f,
            t_b: 0.0,
            max_dets_per_image,
            use_cats,
            iou_thresholds: iou_thresholds(),
            recall_thresholds: recall_thresholds(),
            area_ranges: &area_ranges,
        };
        kernel_call(&gt, &dt, params, parity).map_err(|e| PyValueError::new_err(format!("{e}")))
    })?;

    histogram_to_dict(py, &mut histogram)
}

/// FP-IoU histogram for the bbox kernel (ADR-0022).
///
/// Returns a dict with parallel `iou_same` / `iou_cross` numpy arrays
/// (one entry per FP detection — Cls / Loc / Both / Dupe / Bkg, but
/// not TP and not Ignore), the kernel marker, the `t_f` used to
/// identify TP / Ignore, the total surviving DT count, and the FP
/// count. Caller computes the bin-as-Bkg fraction at candidate `t_b`
/// values from this output (the `t_b` parameter on `error_decomposition_*`
/// is not consumed here).
#[pyfunction]
#[pyo3(signature = (gt, dt, parity_mode, t_f, max_dets_per_image, use_cats, *, cast_inputs = false))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn fp_iou_histogram_bbox<'py>(
    py: Python<'py>,
    gt: &Bound<'py, PyAny>,
    dt: &Bound<'py, PyAny>,
    parity_mode: &str,
    t_f: f64,
    max_dets_per_image: usize,
    use_cats: bool,
    cast_inputs: bool,
) -> PyResult<Bound<'py, PyDict>> {
    run_fp_histogram_pass(
        py,
        gt,
        dt,
        ArrayIouType::Bbox,
        parity_mode,
        t_f,
        max_dets_per_image,
        use_cats,
        cast_inputs,
        |gt, dt, params, parity| tide::compute_fp_iou_histogram_bbox(&gt.gt, dt, params, parity),
    )
}

/// FP-IoU histogram for the segm kernel.
#[pyfunction]
#[pyo3(signature = (gt, dt, parity_mode, t_f, max_dets_per_image, use_cats, *, cast_inputs = false))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn fp_iou_histogram_segm<'py>(
    py: Python<'py>,
    gt: &Bound<'py, PyAny>,
    dt: &Bound<'py, PyAny>,
    parity_mode: &str,
    t_f: f64,
    max_dets_per_image: usize,
    use_cats: bool,
    cast_inputs: bool,
) -> PyResult<Bound<'py, PyDict>> {
    run_fp_histogram_pass(
        py,
        gt,
        dt,
        ArrayIouType::Segm,
        parity_mode,
        t_f,
        max_dets_per_image,
        use_cats,
        cast_inputs,
        |gt, dt, params, parity| {
            let kernel = gt.segm_kernel();
            tide::compute_fp_iou_histogram_with(
                &gt.gt,
                dt,
                &kernel,
                KernelMarker::Segm,
                params,
                parity,
            )
        },
    )
}

/// FP-IoU histogram for the boundary-segm kernel. `dilation_ratio`
/// configures the band thickness (ADR-0010 default `0.02` for COCO).
#[pyfunction]
#[pyo3(signature = (gt, dt, parity_mode, t_f, max_dets_per_image, use_cats, dilation_ratio, *, cast_inputs = false))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn fp_iou_histogram_boundary<'py>(
    py: Python<'py>,
    gt: &Bound<'py, PyAny>,
    dt: &Bound<'py, PyAny>,
    parity_mode: &str,
    t_f: f64,
    max_dets_per_image: usize,
    use_cats: bool,
    dilation_ratio: f64,
    cast_inputs: bool,
) -> PyResult<Bound<'py, PyDict>> {
    validate_dilation_ratio(dilation_ratio)?;
    run_fp_histogram_pass(
        py,
        gt,
        dt,
        ArrayIouType::Boundary,
        parity_mode,
        t_f,
        max_dets_per_image,
        use_cats,
        cast_inputs,
        move |gt, dt, params, parity| {
            let kernel = gt.boundary_kernel(dilation_ratio);
            tide::compute_fp_iou_histogram_with(
                &gt.gt,
                dt,
                &kernel,
                KernelMarker::Boundary,
                params,
                parity,
            )
        },
    )
}

fn histogram_to_dict<'py>(py: Python<'py>, h: &mut FpIouHistogram) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    // Transfer ownership of the Rust Vecs into the numpy buffers
    // (`into_pyarray` is zero-copy; `to_pyarray` would copy). On COCO
    // val with ~100K FPs this saves 1.6 MB × 2 of allocation per call.
    let iou_same = Array1::from(std::mem::take(&mut h.iou_same)).into_pyarray(py);
    let iou_cross = Array1::from(std::mem::take(&mut h.iou_cross)).into_pyarray(py);
    out.set_item("iou_same", iou_same)?;
    out.set_item("iou_cross", iou_cross)?;
    out.set_item("kernel", h.kernel.as_str())?;
    out.set_item("t_f", h.t_f)?;
    out.set_item("n_total_dts", h.n_total_dts)?;
    out.set_item("n_fps", h.n_fps)?;
    Ok(out)
}

/// Materialize a [`TideReport`] into the Python dict shape pinned in the
/// module docstring. Missing bins fall back to `0.0` so the dict shape is
/// stable regardless of the rewrite layer's sparse population pattern.
fn report_to_dict<'py>(py: Python<'py>, report: &TideReport) -> PyResult<Bound<'py, PyDict>> {
    let delta = PyDict::new(py);
    for (key, bin) in [
        ("cls", TideErrorBin::Cls),
        ("loc", TideErrorBin::Loc),
        ("both", TideErrorBin::Both),
        ("dupe", TideErrorBin::Dupe),
        ("bkg", TideErrorBin::Bkg),
        ("missed", TideErrorBin::Missed),
    ] {
        let value = report.delta_per_bin.get(&bin).copied().unwrap_or(0.0);
        delta.set_item(key, value)?;
    }

    let config = PyDict::new(py);
    config.set_item("t_f", report.config.t_f)?;
    config.set_item("t_b", report.config.t_b)?;
    config.set_item("kernel", report.config.kernel.as_str())?;

    let out = PyDict::new(py);
    out.set_item("baseline_map", report.baseline_map)?;
    out.set_item("delta", delta)?;
    out.set_item("delta_all_fp_removed", report.delta_all_fp)?;
    out.set_item("config", config)?;
    Ok(out)
}
