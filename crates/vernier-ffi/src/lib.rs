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

use numpy::ndarray::Array1;
use numpy::ToPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

use vernier_core::{
    accumulate, evaluate_bbox, evaluate_segm, iou_thresholds, recall_thresholds, sort_max_dets,
    summarize_detection, AccumulateParams, Accumulated, AreaRange, CocoDataset, CocoDetections,
    EvalError, EvalGrid, EvalImageMeta, EvaluateParams, ParityMode, PerImageEval, Summary,
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

/// Frozen wrapper around [`vernier_core::EvalGrid`]. Produced by
/// [`evaluate_bbox_grid`]; consumed by [`PyEvalGrid::accumulate`]. The
/// `eval_imgs` accessor materializes the pycocotools-shaped per-cell
/// dicts on demand.
#[pyclass(module = "vernier._core", name = "EvalGrid", frozen)]
struct PyEvalGrid {
    inner: EvalGrid,
    parity: ParityMode,
}

#[pymethods]
impl PyEvalGrid {
    /// `K`: number of categories evaluated (`1` when `use_cats=False`).
    #[getter]
    fn n_categories(&self) -> usize {
        self.inner.n_categories
    }

    /// `A`: number of area ranges (`4` for the COCO default grid).
    #[getter]
    fn n_area_ranges(&self) -> usize {
        self.inner.n_area_ranges
    }

    /// `I`: number of images (every image in the GT dataset).
    #[getter]
    fn n_images(&self) -> usize {
        self.inner.n_images
    }

    /// Flat `[k][a][i]` list of pycocotools-shaped per-cell dicts. Each
    /// entry is either `None` (cell did not run) or a dict with keys
    /// matching `pycocotools.cocoeval.COCOeval._ImageEvaluationResult`:
    /// `image_id`, `category_id`, `aRng`, `maxDet`, `dtIds`, `gtIds`,
    /// `dtMatches`, `gtMatches`, `dtScores`, `gtIgnore`, `dtIgnore`.
    fn eval_imgs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for (cell, meta) in self
            .inner
            .eval_imgs
            .iter()
            .zip(self.inner.eval_imgs_meta.iter())
        {
            match (cell, meta) {
                (Some(cell), Some(meta)) => list.append(eval_img_dict(py, cell, meta)?)?,
                _ => list.append(py.None())?,
            }
        }
        Ok(list)
    }

    /// Accumulate this grid into precision / recall / scores tensors.
    /// `max_dets` is the full ladder fed to pycocotools' `accumulate`
    /// (default `[1, 10, 100]`); each entry must be `<=` the
    /// `max_dets_per_image` used to build the grid.
    fn accumulate(&self, py: Python<'_>, max_dets: Vec<usize>) -> PyResult<PyAccumulated> {
        require_nonempty_max_dets(&max_dets)?;
        // Quirk A2 (aligned): mirror pycocotools' `cocoeval.py:137`
        // `p.maxDets = sorted(p.maxDets)`. The accumulator's M-axis and
        // the summarizer's positional `AR_*` slots both depend on
        // ascending order; normalize at the FFI boundary.
        let mut max_dets = max_dets;
        sort_max_dets(&mut max_dets);
        let parity = self.parity;
        let n_categories = self.inner.n_categories;
        let n_area_ranges = self.inner.n_area_ranges;
        let n_images = self.inner.n_images;
        let eval_imgs = &self.inner.eval_imgs;
        let acc = py
            .detach(|| {
                accumulate(
                    eval_imgs,
                    AccumulateParams {
                        iou_thresholds: iou_thresholds(),
                        recall_thresholds: recall_thresholds(),
                        max_dets: &max_dets,
                        n_categories,
                        n_area_ranges,
                        n_images,
                    },
                    parity,
                )
            })
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;
        Ok(PyAccumulated {
            inner: acc,
            max_dets,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "EvalGrid(n_categories={}, n_area_ranges={}, n_images={})",
            self.inner.n_categories, self.inner.n_area_ranges, self.inner.n_images
        )
    }
}

/// Frozen wrapper around [`vernier_core::Accumulated`]. Carries the
/// `max_dets` ladder used by `accumulate`; `summarize` reuses it.
#[pyclass(module = "vernier._core", name = "Accumulated", frozen)]
struct PyAccumulated {
    inner: Accumulated,
    max_dets: Vec<usize>,
}

#[pymethods]
impl PyAccumulated {
    /// 5-D precision tensor `(T, R, K, A, M)`. Right-monotonic precision
    /// interpolated at every recall threshold. Cells with no in-bucket
    /// data carry `-1.0` (quirk **C5**).
    #[getter]
    fn precision<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray5<f64>> {
        self.inner.precision.to_pyarray(py)
    }

    /// 4-D recall tensor `(T, K, A, M)`. Cells with no data carry `-1.0`.
    #[getter]
    fn recall<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray4<f64>> {
        self.inner.recall.to_pyarray(py)
    }

    /// 5-D scores tensor `(T, R, K, A, M)`. The DT score at each
    /// (threshold, recall sample) point.
    #[getter]
    fn scores<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray5<f64>> {
        self.inner.scores.to_pyarray(py)
    }

    /// Pycocotools-shaped `eval["counts"]`: `[T, R, K, A, M]`.
    #[getter]
    fn counts(&self) -> Vec<usize> {
        self.inner.precision.shape().to_vec()
    }

    /// Summarize this accumulator into the canonical 12-stat detection
    /// vector. `max_dets` defaults to the ladder this accumulator was
    /// built with; pass an explicit value to override.
    #[pyo3(signature = (max_dets=None))]
    fn summarize(&self, py: Python<'_>, max_dets: Option<Vec<usize>>) -> PyResult<PySummary> {
        let mut dets = max_dets.unwrap_or_else(|| self.max_dets.clone());
        require_nonempty_max_dets(&dets)?;
        // Quirk A2 (aligned): the accumulator was built with a sorted
        // ladder; an explicit override here must follow the same
        // contract or the M-axis lookups in `summarize_detection`
        // would silently misalign. `self.max_dets` is already sorted
        // (set by `PyEvalGrid::accumulate`), so the unwrap_or branch is
        // a no-op.
        sort_max_dets(&mut dets);
        let acc = &self.inner;
        let summary = py
            .detach(|| summarize_detection(acc, iou_thresholds(), &dets))
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;
        Ok(PySummary { inner: summary })
    }

    fn __repr__(&self) -> String {
        let p = &self.inner.precision;
        let s = p.shape();
        format!(
            "Accumulated(precision={}x{}x{}x{}x{})",
            s[0], s[1], s[2], s[3], s[4]
        )
    }
}

/// Build a single pycocotools-shaped `evalImgs` dict from one cell.
fn eval_img_dict<'py>(
    py: Python<'py>,
    cell: &PerImageEval,
    meta: &EvalImageMeta,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("image_id", meta.image_id)?;
    dict.set_item("category_id", meta.category_id)?;
    dict.set_item("aRng", &meta.area_rng[..])?;
    dict.set_item("maxDet", meta.max_det)?;
    dict.set_item("dtIds", &meta.dt_ids[..])?;
    dict.set_item("gtIds", &meta.gt_ids[..])?;
    dict.set_item("dtMatches", meta.dt_matches.to_pyarray(py))?;
    dict.set_item("gtMatches", meta.gt_matches.to_pyarray(py))?;
    dict.set_item("dtScores", &cell.dt_scores[..])?;
    let gt_ignore = Array1::from_iter(cell.gt_ignore.iter().map(|&b| u8::from(b)));
    dict.set_item("gtIgnore", gt_ignore.to_pyarray(py))?;
    let dt_ignore = cell.dt_ignore.map(|&b| u8::from(b));
    dict.set_item("dtIgnore", dt_ignore.to_pyarray(py))?;
    Ok(dict)
}

/// Type-erased eval kernel selector for the FFI surface. Each variant
/// dispatches to the corresponding `evaluate_*` function in
/// `vernier-core`.
#[derive(Debug, Clone, Copy)]
enum EvalIouType {
    Bbox,
    Segm,
}

impl EvalIouType {
    fn run(
        self,
        gt: &CocoDataset,
        dt: &CocoDetections,
        params: EvaluateParams<'_>,
        parity: ParityMode,
    ) -> Result<EvalGrid, EvalError> {
        match self {
            Self::Bbox => evaluate_bbox(gt, dt, params, parity),
            Self::Segm => evaluate_segm(gt, dt, params, parity),
        }
    }
}

/// Run the per-image evaluation pass and return the pycocotools-shaped
/// grid.
///
/// `max_dets_per_image` is the single-int top-N cap applied per
/// `(image, category)` cell — pass the *largest* entry of the eventual
/// `accumulate()` `max_dets` ladder. Smaller ladder entries are sliced
/// downstream by `accumulate`.
fn evaluate_grid_impl(
    py: Python<'_>,
    iou_type: EvalIouType,
    gt_json: &Bound<'_, PyBytes>,
    dt_json: &Bound<'_, PyBytes>,
    parity_mode: &str,
    max_dets_per_image: usize,
    use_cats: bool,
) -> PyResult<PyEvalGrid> {
    let parity = parse_parity_mode(parity_mode)?;
    let gt = CocoDataset::from_json_bytes(gt_json.as_bytes())
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;
    let dt = CocoDetections::from_json_bytes(dt_json.as_bytes())
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;
    let area = AreaRange::coco_default();
    let grid = py
        .detach(move || {
            iou_type.run(
                &gt,
                &dt,
                EvaluateParams {
                    iou_thresholds: iou_thresholds(),
                    area_ranges: &area,
                    max_dets_per_image,
                    use_cats,
                },
                parity,
            )
        })
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;
    Ok(PyEvalGrid {
        inner: grid,
        parity,
    })
}

/// Bbox per-image evaluation pass — see [`evaluate_grid_impl`].
#[pyfunction]
#[pyo3(signature = (gt_json, dt_json, parity_mode, max_dets_per_image, use_cats))]
fn evaluate_bbox_grid(
    py: Python<'_>,
    gt_json: &Bound<'_, PyBytes>,
    dt_json: &Bound<'_, PyBytes>,
    parity_mode: &str,
    max_dets_per_image: usize,
    use_cats: bool,
) -> PyResult<PyEvalGrid> {
    evaluate_grid_impl(
        py,
        EvalIouType::Bbox,
        gt_json,
        dt_json,
        parity_mode,
        max_dets_per_image,
        use_cats,
    )
}

/// Segm per-image evaluation pass. Both GT and DT JSON must carry a
/// `segmentation` field on every entry; absent fields raise a typed
/// `ValueError` instead of being silently treated as empty.
#[pyfunction]
#[pyo3(signature = (gt_json, dt_json, parity_mode, max_dets_per_image, use_cats))]
fn evaluate_segm_grid(
    py: Python<'_>,
    gt_json: &Bound<'_, PyBytes>,
    dt_json: &Bound<'_, PyBytes>,
    parity_mode: &str,
    max_dets_per_image: usize,
    use_cats: bool,
) -> PyResult<PyEvalGrid> {
    evaluate_grid_impl(
        py,
        EvalIouType::Segm,
        gt_json,
        dt_json,
        parity_mode,
        max_dets_per_image,
        use_cats,
    )
}

/// Run an end-to-end evaluation pipeline (grid → accumulate → summarize)
/// and return a [`PySummary`].
///
/// `gt_json` and `dt_json` are the COCO ground-truth and detection JSON
/// payloads as bytes (the same shapes pycocotools' `COCO(...)` /
/// `loadRes(...)` consume). `parity_mode` is `"strict"` or `"corrected"`
/// per ADR-0002. `max_dets` is the maxDets ladder fed to accumulate /
/// summarize (pycocotools default `[1, 10, 100]`). `use_cats` mirrors
/// pycocotools' `useCats` (quirk **L4**).
fn evaluate_summary_impl(
    py: Python<'_>,
    iou_type: EvalIouType,
    gt_json: &Bound<'_, PyBytes>,
    dt_json: &Bound<'_, PyBytes>,
    parity_mode: &str,
    max_dets: Vec<usize>,
    use_cats: bool,
) -> PyResult<PySummary> {
    let parity = parse_parity_mode(parity_mode)?;
    require_nonempty_max_dets(&max_dets)?;
    // Quirk A2 (aligned): mirror pycocotools' `cocoeval.py:137`
    // `p.maxDets = sorted(p.maxDets)`. Sort once here so the eval
    // pipeline's `max_dets_per_image` cap (the largest entry) and the
    // summarizer's positional `AR_*` lookups both see the canonical
    // ascending ladder.
    let mut max_dets = max_dets;
    sort_max_dets(&mut max_dets);
    let gt = CocoDataset::from_json_bytes(gt_json.as_bytes())
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;
    let dt = CocoDetections::from_json_bytes(dt_json.as_bytes())
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;

    let summary = py
        .detach(move || run_pipeline(iou_type, &gt, &dt, parity, &max_dets, use_cats))
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;

    Ok(PySummary { inner: summary })
}

/// Bbox end-to-end pipeline — see [`evaluate_summary_impl`].
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
    evaluate_summary_impl(
        py,
        EvalIouType::Bbox,
        gt_json,
        dt_json,
        parity_mode,
        max_dets,
        use_cats,
    )
}

/// Segm end-to-end pipeline — see [`evaluate_summary_impl`]. Both GT
/// and DT must carry segmentation fields.
#[pyfunction]
#[pyo3(signature = (gt_json, dt_json, parity_mode, max_dets, use_cats))]
fn evaluate_segm_summary(
    py: Python<'_>,
    gt_json: &Bound<'_, PyBytes>,
    dt_json: &Bound<'_, PyBytes>,
    parity_mode: &str,
    max_dets: Vec<usize>,
    use_cats: bool,
) -> PyResult<PySummary> {
    evaluate_summary_impl(
        py,
        EvalIouType::Segm,
        gt_json,
        dt_json,
        parity_mode,
        max_dets,
        use_cats,
    )
}

fn run_pipeline(
    iou_type: EvalIouType,
    gt: &CocoDataset,
    dt: &CocoDetections,
    parity: ParityMode,
    max_dets: &[usize],
    use_cats: bool,
) -> Result<Summary, EvalError> {
    let iou_thr = iou_thresholds();
    let area = AreaRange::coco_default();
    let max_det_top = max_dets.iter().copied().max().unwrap_or(100);
    let eval_params = EvaluateParams {
        iou_thresholds: iou_thr,
        area_ranges: &area,
        max_dets_per_image: max_det_top,
        use_cats,
    };
    let grid = iou_type.run(gt, dt, eval_params, parity)?;

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

fn require_nonempty_max_dets(max_dets: &[usize]) -> PyResult<()> {
    if max_dets.is_empty() {
        Err(PyValueError::new_err(
            "max_dets must contain at least one entry",
        ))
    } else {
        Ok(())
    }
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
    m.add_function(wrap_pyfunction!(evaluate_bbox_grid, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_segm_summary, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_segm_grid, m)?)?;
    m.add_class::<PySummary>()?;
    m.add_class::<PyEvalGrid>()?;
    m.add_class::<PyAccumulated>()?;
    m.add("__version__", vernier_core::VERSION)?;
    Ok(())
}
