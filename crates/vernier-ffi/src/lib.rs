//! PyO3 bindings for [`vernier_core`].
//!
//! By policy, this crate contains **no business logic** — only data conversion
//! between Python and Rust. All evaluation algorithms live in
//! [`vernier_core`]. Reviewers: please push back on any PR that adds
//! computational logic here rather than there.
//!
//! ## Threading
//!
//! Per ADR-0006, every entry point that runs non-trivial Rust compute
//! drops the GIL via [`Python::detach`] (the PyO3 0.28+ name for the
//! historical `allow_threads`). The wrapped closure may not touch Python
//! objects: data conversion happens at the boundary, before [`Python::detach`];
//! results are converted back after the GIL is re-acquired.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use vernier_core::{
    accumulate, evaluate_bbox, iou_thresholds as iou_thresholds_default, recall_thresholds,
    summarize_detection, AccumulateParams, AreaRange, CocoDataset, CocoDetections, EvalError,
    EvaluateBboxParams, ParityMode, Summary,
};

/// Returns the underlying `vernier-core` version string. Useful as a smoke
/// test that the FFI bridge is wired up and the dynamic linker can find the
/// extension module.
#[pyfunction]
fn version() -> &'static str {
    vernier_core::VERSION
}

/// Pythonic view over a [`vernier_core::Summary`]. Frozen — the underlying
/// value is constructed once by [`evaluate_bbox_summary`] and never
/// mutated (per ADR-0006).
#[pyclass(module = "vernier._core", name = "Summary", frozen)]
struct PySummary {
    inner: Summary,
}

#[pymethods]
impl PySummary {
    /// 12 detection stats in canonical pycocotools order.
    #[getter]
    fn stats(&self) -> Vec<f64> {
        self.inner.stats()
    }

    /// One pretty-printed line per stat, matching the pycocotools
    /// `Average Precision (AP) @[ ... ] = 0.xxx` shape.
    fn pretty_lines(&self) -> Vec<String> {
        self.inner.pretty_lines()
    }

    fn __repr__(&self) -> String {
        let n = self.inner.lines.len();
        format!("Summary(lines={n})")
    }
}

/// Run the bbox evaluation pipeline end-to-end and return a [`PySummary`].
///
/// `gt_json` and `dt_json` are the COCO ground-truth and detection JSON
/// payloads as bytes (the same shapes pycocotools' `COCO(...)` /
/// `loadRes(...)` consume). `parity_mode` is `"strict"` or `"corrected"`
/// per ADR-0002. `max_dets` is the maxDets ladder fed to accumulate /
/// summarize (pycocotools default `[1, 10, 100]`). `use_cats` mirrors
/// pycocotools' `useCats` (quirk **L4**).
#[pyfunction]
#[pyo3(signature = (gt_json, dt_json, parity_mode, max_dets, use_cats))]
fn evaluate_bbox_summary(
    py: Python<'_>,
    gt_json: &Bound<'_, PyBytes>,
    dt_json: &Bound<'_, PyBytes>,
    parity_mode: &str,
    max_dets: Vec<usize>,
    use_cats: bool,
) -> PyResult<PySummary> {
    let parity = parse_parity_mode(parity_mode)?;
    // Copy the JSON bytes out of Python ownership so the Rust closure
    // can run with the GIL released.
    let gt_bytes: Vec<u8> = gt_json.as_bytes().to_vec();
    let dt_bytes: Vec<u8> = dt_json.as_bytes().to_vec();

    let summary = py
        .detach(move || run_bbox_pipeline(&gt_bytes, &dt_bytes, parity, &max_dets, use_cats))
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;

    Ok(PySummary { inner: summary })
}

fn run_bbox_pipeline(
    gt_bytes: &[u8],
    dt_bytes: &[u8],
    parity: ParityMode,
    max_dets: &[usize],
    use_cats: bool,
) -> Result<Summary, EvalError> {
    let gt = CocoDataset::from_json_bytes(gt_bytes)?;
    let dt = CocoDetections::from_json_bytes(dt_bytes)?;

    let iou_thr = iou_thresholds_default();
    let area = AreaRange::coco_default();
    let max_det_top = max_dets.iter().copied().max().unwrap_or(100);
    let eval_params = EvaluateBboxParams {
        iou_thresholds: iou_thr,
        area_ranges: &area,
        max_dets_per_image: max_det_top,
        use_cats,
    };
    let grid = evaluate_bbox(&gt, &dt, eval_params, parity)?;

    let acc_params = AccumulateParams {
        iou_thresholds: iou_thr,
        recall_thresholds: recall_thresholds(),
        max_dets,
        n_categories: grid.n_categories,
        n_area_ranges: grid.n_area_ranges,
        n_images: grid.n_images,
    };
    let acc = accumulate(&grid.eval_imgs, acc_params, parity)?;
    summarize_detection(&acc, iou_thr, max_dets)
}

fn parse_parity_mode(s: &str) -> PyResult<ParityMode> {
    match s {
        "strict" => Ok(ParityMode::Strict),
        "corrected" => Ok(ParityMode::Corrected),
        other => Err(PyValueError::new_err(format!(
            "invalid parity_mode {other:?}; expected 'strict' or 'corrected'"
        ))),
    }
}

/// The native module exposed to Python as `vernier._core`.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_bbox_summary, m)?)?;
    m.add_class::<PySummary>()?;
    m.add("__version__", vernier_core::VERSION)?;
    Ok(())
}
