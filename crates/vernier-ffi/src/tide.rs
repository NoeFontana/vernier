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

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use vernier_core::evaluate::AreaRange;
use vernier_core::tide::{self, TideErrorBin, TideParams, TideReport};
use vernier_core::{iou_thresholds, recall_thresholds, CocoDataset, CocoDetections, EvalError, ParityMode};

use crate::{parse_dt, parse_gt, parse_parity_mode, validate_dilation_ratio};

/// Common per-call plumbing for the three TIDE kernel entry points:
/// parse parity mode, copy JSON bytes off the GIL, run the kernel-
/// specific orchestrator inside `py.detach`, and materialize the
/// report dict. `kernel_call` carries the kernel-specific dispatch
/// (and any extra knobs like `dilation_ratio`) closed over by the
/// per-kernel wrappers below.
#[allow(clippy::too_many_arguments)]
fn run_tide_pass<'py, F>(
    py: Python<'py>,
    gt_bytes: &Bound<'py, PyBytes>,
    dt_bytes: &Bound<'py, PyBytes>,
    parity_mode: &str,
    t_f: f64,
    t_b: f64,
    max_dets_per_image: usize,
    use_cats: bool,
    kernel_call: F,
) -> PyResult<Bound<'py, PyDict>>
where
    F: FnOnce(&CocoDataset, &CocoDetections, TideParams<'_>, ParityMode)
            -> Result<TideReport, EvalError>
        + Send,
{
    let parity = parse_parity_mode(parity_mode)?;
    // Copy the JSON bytes off the GIL-tied PyBytes borrow so the parse
    // and the eight-pass orchestration can run inside `py.detach`.
    let gt_bytes = gt_bytes.as_bytes().to_vec();
    let dt_bytes = dt_bytes.as_bytes().to_vec();

    let report = py.detach(move || -> PyResult<TideReport> {
        let gt = parse_gt(&gt_bytes)?;
        let dt = parse_dt(&dt_bytes)?;
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
/// `gt_bytes` and `dt_bytes` are the COCO ground-truth and detection JSON
/// payloads as bytes (the same shapes pycocotools' `COCO(...)` /
/// `loadRes(...)` consume). `parity_mode` is `"strict"` or `"corrected"`
/// per ADR-0002. `t_f` and `t_b` are the foreground / background
/// thresholds; ADR-0022 pins the bbox defaults at `0.5` / `0.1`.
/// `max_dets_per_image` matches the oracle's per-image cap (the oracle
/// uses `100` by default). `use_cats` mirrors pycocotools' `useCats`.
///
/// Returns the report dict described in the module docstring.
#[pyfunction]
#[pyo3(signature = (gt_bytes, dt_bytes, parity_mode, t_f, t_b, max_dets_per_image, use_cats))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn error_decomposition_bbox<'py>(
    py: Python<'py>,
    gt_bytes: &Bound<'py, PyBytes>,
    dt_bytes: &Bound<'py, PyBytes>,
    parity_mode: &str,
    t_f: f64,
    t_b: f64,
    max_dets_per_image: usize,
    use_cats: bool,
) -> PyResult<Bound<'py, PyDict>> {
    run_tide_pass(
        py,
        gt_bytes,
        dt_bytes,
        parity_mode,
        t_f,
        t_b,
        max_dets_per_image,
        use_cats,
        tide::error_decomposition_bbox,
    )
}

/// TIDE error decomposition for the segm kernel (ADR-0021, Week 3).
///
/// Same signature as [`error_decomposition_bbox`] above; the only
/// per-call difference is the kernel — `gt_bytes` / `dt_bytes` must
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
#[pyo3(signature = (gt_bytes, dt_bytes, parity_mode, t_f, t_b, max_dets_per_image, use_cats))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn error_decomposition_segm<'py>(
    py: Python<'py>,
    gt_bytes: &Bound<'py, PyBytes>,
    dt_bytes: &Bound<'py, PyBytes>,
    parity_mode: &str,
    t_f: f64,
    t_b: f64,
    max_dets_per_image: usize,
    use_cats: bool,
) -> PyResult<Bound<'py, PyDict>> {
    run_tide_pass(
        py,
        gt_bytes,
        dt_bytes,
        parity_mode,
        t_f,
        t_b,
        max_dets_per_image,
        use_cats,
        tide::error_decomposition_segm,
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
/// Both `gt_bytes` and `dt_bytes` must carry `segmentation` fields —
/// the same constraint `evaluate_boundary_summary` enforces; polygon
/// and RLE shapes both work.
#[pyfunction]
#[pyo3(signature = (gt_bytes, dt_bytes, parity_mode, t_f, t_b, max_dets_per_image, use_cats, dilation_ratio))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn error_decomposition_boundary<'py>(
    py: Python<'py>,
    gt_bytes: &Bound<'py, PyBytes>,
    dt_bytes: &Bound<'py, PyBytes>,
    parity_mode: &str,
    t_f: f64,
    t_b: f64,
    max_dets_per_image: usize,
    use_cats: bool,
    dilation_ratio: f64,
) -> PyResult<Bound<'py, PyDict>> {
    validate_dilation_ratio(dilation_ratio)?;
    run_tide_pass(
        py,
        gt_bytes,
        dt_bytes,
        parity_mode,
        t_f,
        t_b,
        max_dets_per_image,
        use_cats,
        move |gt, dt, params, parity| {
            tide::error_decomposition_boundary(gt, dt, params, parity, dilation_ratio)
        },
    )
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
