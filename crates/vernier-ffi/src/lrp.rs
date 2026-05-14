//! Wraps the kernel-specific entry points in [`vernier_core::lrp`] and
//! exposes them to Python: `optimal_lrp_bbox`, `optimal_lrp_segm`,
//! `optimal_lrp_boundary`, `optimal_lrp_keypoints`. By policy, this
//! module contains only data conversion — the tau sweep, per-class
//! decomposition, and matching layer live in
//! [`vernier_core::lrp`].
//!
//! Per ADR-0043, panoptic LRP is a separate path through panoptic
//! matching and does NOT route through this module; the panoptic
//! kernel does not carry per-segment scores so the tau sweep
//! requires a dedicated matching pass.
//!
//! ## Output shape
//!
//! Each kernel returns a `dict` shape designed to round-trip with the
//! numpy oracle at `tests/python/oracle/lrp/oracle.py` (the
//! correctness contract per ADR-0043):
//!
//! ```text
//! {
//!     "olrp": float,
//!     "loc": float,
//!     "fp": float,
//!     "fn": float,
//!     "per_class": [
//!         {"category_id": int, "olrp": float | None,
//!          "olrp_loc": float | None, "olrp_fp": float | None,
//!          "olrp_fn": float | None, "tau": float | None},
//!         ...
//!     ],
//!     "n_empty_classes": int,
//!     "config": {"tp_threshold": float, "tau_grid_len": int, "kernel": str},
//! }
//! ```
//!
//! Per-class `None` values map to Python `None`; the oracle uses
//! `NaN` so the wrapper layer (`python/vernier/_lrp.py`) translates
//! between the two representations.

use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

use vernier_core::evaluate::AreaRange;
use vernier_core::lrp::{self, LrpKernelMarker, LrpParams, LrpPerClass, LrpReport};
use vernier_core::{CocoDataset, CocoDetections, EvalError, ParityMode};

use crate::{parse_dt, parse_gt, parse_parity_mode, validate_dilation_ratio};

/// Common per-call plumbing for the four LRP kernel entry points:
/// parse parity mode, copy JSON bytes off the GIL, run the kernel-
/// specific orchestrator inside `py.detach`, and materialise the
/// report dict. `kernel_call` carries the kernel-specific dispatch
/// (and any extra knobs like `dilation_ratio` / `sigmas`) closed
/// over by the per-kernel wrappers below.
#[allow(clippy::too_many_arguments)]
fn run_lrp_pass<'py, F>(
    py: Python<'py>,
    gt_bytes: &Bound<'py, PyBytes>,
    dt_bytes: &Bound<'py, PyBytes>,
    parity_mode: &str,
    tp_threshold: f64,
    tau_grid: Vec<f64>,
    max_dets_per_image: usize,
    use_cats: bool,
    kernel_call: F,
) -> PyResult<Bound<'py, PyDict>>
where
    F: FnOnce(
            &CocoDataset,
            &CocoDetections,
            LrpParams<'_>,
            ParityMode,
        ) -> Result<LrpReport, EvalError>
        + Send,
{
    let parity = parse_parity_mode(parity_mode)?;
    // Copy the JSON bytes off the GIL-tied PyBytes borrow so the
    // parse and the LRP orchestration can run inside `py.detach`.
    let gt_bytes = gt_bytes.as_bytes().to_vec();
    let dt_bytes = dt_bytes.as_bytes().to_vec();

    let report = py.detach(move || -> PyResult<LrpReport> {
        let gt = parse_gt(&gt_bytes)?;
        let dt = parse_dt(&dt_bytes)?;
        let area_ranges = AreaRange::coco_default();
        // The LRP pass only consumes the retained IoU matrices; the
        // matching engine's IoU-threshold ladder is irrelevant. Use
        // `[tp_threshold]` as the minimal slice so `evaluate_with`
        // does not error on an empty thresholds list.
        let iou_thresholds = [tp_threshold];
        let params = LrpParams {
            tp_threshold,
            tau_grid: &tau_grid,
            max_dets_per_image,
            use_cats,
            iou_thresholds: &iou_thresholds,
            area_ranges: &area_ranges,
        };
        kernel_call(&gt, &dt, params, parity).map_err(|e| PyValueError::new_err(format!("{e}")))
    })?;

    report_to_dict(py, &report)
}

/// LRP / oLRP for the bbox kernel (ADR-0043 + ADR-0044).
///
/// `gt_bytes` / `dt_bytes` are the COCO ground-truth and detection
/// JSON payloads as bytes (the same shape pycocotools' `COCO(...)` /
/// `loadRes(...)` consume). `parity_mode` is `"strict"` or
/// `"corrected"` per ADR-0002. `tp_threshold` is the IoU floor above
/// which a matched pair is a TP (default `0.5` per ADR-0044).
/// `tau_grid` is the confidence-threshold grid; the canonical default
/// the Python wrapper picks is the 101-point grid `0.00, 0.01, ...,
/// 1.00`. `max_dets_per_image` matches the oracle's per-image cap;
/// `use_cats` mirrors pycocotools' `useCats`.
///
/// Returns the report dict described in the module docstring.
#[pyfunction]
#[pyo3(signature = (gt_bytes, dt_bytes, parity_mode, tp_threshold, tau_grid, max_dets_per_image, use_cats))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn optimal_lrp_bbox<'py>(
    py: Python<'py>,
    gt_bytes: &Bound<'py, PyBytes>,
    dt_bytes: &Bound<'py, PyBytes>,
    parity_mode: &str,
    tp_threshold: f64,
    tau_grid: Vec<f64>,
    max_dets_per_image: usize,
    use_cats: bool,
) -> PyResult<Bound<'py, PyDict>> {
    run_lrp_pass(
        py,
        gt_bytes,
        dt_bytes,
        parity_mode,
        tp_threshold,
        tau_grid,
        max_dets_per_image,
        use_cats,
        lrp::optimal_lrp_bbox,
    )
}

/// LRP / oLRP for the segm (mask) kernel.
#[pyfunction]
#[pyo3(signature = (gt_bytes, dt_bytes, parity_mode, tp_threshold, tau_grid, max_dets_per_image, use_cats))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn optimal_lrp_segm<'py>(
    py: Python<'py>,
    gt_bytes: &Bound<'py, PyBytes>,
    dt_bytes: &Bound<'py, PyBytes>,
    parity_mode: &str,
    tp_threshold: f64,
    tau_grid: Vec<f64>,
    max_dets_per_image: usize,
    use_cats: bool,
) -> PyResult<Bound<'py, PyDict>> {
    run_lrp_pass(
        py,
        gt_bytes,
        dt_bytes,
        parity_mode,
        tp_threshold,
        tau_grid,
        max_dets_per_image,
        use_cats,
        lrp::optimal_lrp_segm,
    )
}

/// LRP / oLRP for the boundary-segm kernel.
///
/// `dilation_ratio` configures the boundary band thickness (ADR-0010
/// default `0.02` for COCO, `0.008` for LVIS).
#[pyfunction]
#[pyo3(signature = (gt_bytes, dt_bytes, parity_mode, tp_threshold, tau_grid, max_dets_per_image, use_cats, dilation_ratio))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn optimal_lrp_boundary<'py>(
    py: Python<'py>,
    gt_bytes: &Bound<'py, PyBytes>,
    dt_bytes: &Bound<'py, PyBytes>,
    parity_mode: &str,
    tp_threshold: f64,
    tau_grid: Vec<f64>,
    max_dets_per_image: usize,
    use_cats: bool,
    dilation_ratio: f64,
) -> PyResult<Bound<'py, PyDict>> {
    validate_dilation_ratio(dilation_ratio)?;
    run_lrp_pass(
        py,
        gt_bytes,
        dt_bytes,
        parity_mode,
        tp_threshold,
        tau_grid,
        max_dets_per_image,
        use_cats,
        move |gt, dt, params, parity| {
            lrp::optimal_lrp_boundary(gt, dt, params, parity, dilation_ratio)
        },
    )
}

/// LRP / oLRP for the keypoints (OKS) kernel.
///
/// Per ADR-0045 LRP-on-OKS ships in 0.5.x. `sigmas` is the per-
/// category sigma override map (`{category_id: [sigma_0, sigma_1,
/// ...]}`); an empty mapping means "use the COCO-person 17-sigma
/// table for every category".
#[pyfunction]
#[pyo3(signature = (gt_bytes, dt_bytes, parity_mode, tp_threshold, tau_grid, max_dets_per_image, use_cats, sigmas))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn optimal_lrp_keypoints<'py>(
    py: Python<'py>,
    gt_bytes: &Bound<'py, PyBytes>,
    dt_bytes: &Bound<'py, PyBytes>,
    parity_mode: &str,
    tp_threshold: f64,
    tau_grid: Vec<f64>,
    max_dets_per_image: usize,
    use_cats: bool,
    sigmas: HashMap<i64, Vec<f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    run_lrp_pass(
        py,
        gt_bytes,
        dt_bytes,
        parity_mode,
        tp_threshold,
        tau_grid,
        max_dets_per_image,
        use_cats,
        move |gt, dt, params, parity| lrp::optimal_lrp_keypoints(gt, dt, params, parity, sigmas),
    )
}

/// Return the canonical 101-point tau grid as a Python list. Surfaced
/// so the Python wrapper can resolve `tau_grid=None` to the same
/// default the Rust side uses without duplicating the constant.
#[pyfunction]
pub(crate) fn lrp_default_tau_grid(py: Python<'_>) -> PyResult<Bound<'_, PyList>> {
    let slice = lrp::default_tau_grid();
    let list = PyList::empty(py);
    for &v in slice {
        list.append(v)?;
    }
    Ok(list)
}

/// Materialise an [`LrpReport`] into the Python dict shape pinned in
/// the module docstring. `None` values pass through as Python `None`;
/// the wrapper translates to `NaN` if the caller prefers that shape.
fn report_to_dict<'py>(py: Python<'py>, report: &LrpReport) -> PyResult<Bound<'py, PyDict>> {
    let per_class = PyList::empty(py);
    for entry in &report.per_class {
        per_class.append(per_class_to_dict(py, entry)?)?;
    }

    let config = PyDict::new(py);
    config.set_item("tp_threshold", report.config.tp_threshold)?;
    config.set_item("tau_grid_len", report.config.tau_grid_len)?;
    config.set_item("kernel", kernel_marker_to_str(report.config.kernel))?;

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

fn kernel_marker_to_str(m: LrpKernelMarker) -> &'static str {
    m.as_str()
}
