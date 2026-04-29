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
//!
//! Two pyclasses extend the threading model beyond the immutable batch
//! evaluators:
//!
//! - [`PyStreamingEvaluator`] (ADR-0013) is mutable but **single-writer**:
//!   the `ThreadId` of the first caller is stashed on first `update()`,
//!   and submissions from any other thread raise `RuntimeError`.
//! - [`PyBackgroundEvaluator`] (ADR-0014) wraps a streaming evaluator in
//!   a dedicated worker thread. The worker owns the inner evaluator,
//!   so the single-writer rule is satisfied by construction; callers
//!   may submit from any thread. Best-effort `nice` and core-affinity
//!   are applied to the worker via `thread-priority` and `core_affinity`;
//!   any scheduling syscall failure surfaces as a one-shot `UserWarning`
//!   on the constructing thread.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use numpy::ndarray::Array1;
use numpy::ToPyArray;
use pyo3::create_exception;
use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError, PyUserWarning, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict, PyList};

use vernier_core::{
    accumulate, evaluate_bbox, evaluate_boundary, evaluate_keypoints, evaluate_segm,
    iou_thresholds, recall_thresholds, sort_max_dets, summarize_detection, summarize_with,
    AccumulateParams, Accumulated, AreaRange, BboxIou, BoundaryIou, CocoDataset, CocoDetections,
    EvalError, EvalGrid, EvalImageMeta, EvaluateParams, MemoryBudget, OksSimilarity,
    OwnedEvaluateParams, ParityMode, ParsedDetections, PerImageEval, SegmIou, StatRequest,
    StreamingEvaluator, Summary, UpdateReport,
};

mod background;

create_exception!(
    vernier._core,
    OutOfBudgetError,
    pyo3::exceptions::PyRuntimeError,
    "Memory budget for the streaming evaluator was exceeded.\n\nAttributes: used_bytes, budget_bytes, breakdown."
);
create_exception!(
    vernier._core,
    QueueFullError,
    pyo3::exceptions::PyRuntimeError,
    "Background evaluator's submit queue was full.\n\nAttributes: queue_capacity, timeout."
);
create_exception!(
    vernier._core,
    MemoryBudgetWarning,
    pyo3::exceptions::PyUserWarning,
    "Streaming evaluator's memory usage crossed the soft-warn threshold."
);

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

    /// Summarize this accumulator. `plan` selects the stat plan:
    /// `"detection"` (default) yields the canonical 12-stat detection
    /// vector via [`vernier_core::summarize_detection`]; `"keypoints"`
    /// yields the 10-stat keypoints vector via
    /// [`vernier_core::StatRequest::coco_keypoints_default`] (ADR-0012).
    /// Pairing the kp plan with a detection-grid accumulator (4-bucket
    /// A-axis) reads the kp re-indexed buckets, and vice-versa indexes
    /// off the end — match the plan to the kernel that built the grid.
    /// `max_dets` defaults to the ladder this accumulator was built
    /// with; pass an explicit value to override.
    #[pyo3(signature = (max_dets=None, *, plan=None))]
    fn summarize(
        &self,
        py: Python<'_>,
        max_dets: Option<Vec<usize>>,
        plan: Option<&str>,
    ) -> PyResult<PySummary> {
        let plan = parse_summarize_plan(plan.unwrap_or("detection"))?;
        let mut dets = max_dets.unwrap_or_else(|| self.max_dets.clone());
        require_nonempty_max_dets(&dets)?;
        // Quirk A2 (aligned): the accumulator was built with a sorted
        // ladder; an explicit override here must follow the same
        // contract or the M-axis lookups in the summarizer would
        // silently misalign. `self.max_dets` is already sorted (set by
        // `PyEvalGrid::accumulate`), so the unwrap_or branch is a no-op.
        sort_max_dets(&mut dets);
        let acc = &self.inner;
        let iou_thr = iou_thresholds();
        let summary = py
            .detach(|| match plan {
                SummarizePlan::Detection => summarize_detection(acc, iou_thr, &dets),
                SummarizePlan::Keypoints => {
                    summarize_with(acc, &StatRequest::coco_keypoints_default(), iou_thr, &dets)
                }
            })
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
/// `vernier-core`. Boundary carries its `dilation_ratio` here and
/// Keypoints its sigmas map (ADR-0012) so the FFI signature for each
/// Python entry point stays a function of exactly that kernel's
/// parameters (per ADR-0011).
///
/// The Keypoints variant carries a [`HashMap`], which is not `Copy`;
/// the enum is therefore `Clone` only.
#[derive(Debug, Clone)]
enum EvalIouType {
    Bbox,
    Segm,
    Boundary { dilation_ratio: f64 },
    Keypoints { sigmas: HashMap<i64, Vec<f64>> },
}

impl EvalIouType {
    fn run(
        &self,
        gt: &CocoDataset,
        dt: &CocoDetections,
        params: EvaluateParams<'_>,
        parity: ParityMode,
    ) -> Result<EvalGrid, EvalError> {
        match self {
            Self::Bbox => evaluate_bbox(gt, dt, params, parity),
            Self::Segm => evaluate_segm(gt, dt, params, parity),
            Self::Boundary { dilation_ratio } => {
                evaluate_boundary(gt, dt, params, parity, *dilation_ratio)
            }
            Self::Keypoints { sigmas } => {
                evaluate_keypoints(gt, dt, params, parity, sigmas.clone())
            }
        }
    }

    /// True for the Keypoints kernel — drives the kp-vs-detection grid
    /// and summarizer-plan dispatch in [`evaluate_grid_impl`] and
    /// [`run_pipeline`].
    fn is_keypoints(&self) -> bool {
        matches!(self, Self::Keypoints { .. })
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
    // Kp grid drops the small bucket (D5); detection grid keeps all
    // four. `Vec<AreaRange>` lets either size flow through the same
    // `&[AreaRange]` slice the eval pass consumes.
    let area: Vec<AreaRange> = area_ranges_for(&iou_type);
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

/// Per-kernel area-range default. Keypoints uses the 3-bucket kp grid
/// (quirk **D5** strict, ADR-0012); every other kernel uses the
/// 4-bucket detection grid.
fn area_ranges_for(iou_type: &EvalIouType) -> Vec<AreaRange> {
    if iou_type.is_keypoints() {
        AreaRange::keypoints_default().to_vec()
    } else {
        AreaRange::coco_default().to_vec()
    }
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

/// Boundary-IoU per-image evaluation pass (ADR-0010). Same
/// segmentation-field requirements as [`evaluate_segm_grid`].
/// `dilation_ratio` is the boundary band width as a fraction of the
/// image diagonal (`0.02` COCO default; `0.008` LVIS variant).
#[pyfunction]
#[pyo3(signature = (gt_json, dt_json, parity_mode, max_dets_per_image, use_cats, dilation_ratio))]
fn evaluate_boundary_grid(
    py: Python<'_>,
    gt_json: &Bound<'_, PyBytes>,
    dt_json: &Bound<'_, PyBytes>,
    parity_mode: &str,
    max_dets_per_image: usize,
    use_cats: bool,
    dilation_ratio: f64,
) -> PyResult<PyEvalGrid> {
    let iou_type = boundary_iou_type(dilation_ratio)?;
    evaluate_grid_impl(
        py,
        iou_type,
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
        .detach(move || run_pipeline(&iou_type, &gt, &dt, parity, &max_dets, use_cats))
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

/// Boundary end-to-end pipeline (ADR-0010) — see
/// [`evaluate_summary_impl`]. Both GT and DT must carry segmentation
/// fields. `dilation_ratio` matches [`evaluate_boundary_grid`].
#[pyfunction]
#[pyo3(signature = (gt_json, dt_json, parity_mode, max_dets, use_cats, dilation_ratio))]
fn evaluate_boundary_summary(
    py: Python<'_>,
    gt_json: &Bound<'_, PyBytes>,
    dt_json: &Bound<'_, PyBytes>,
    parity_mode: &str,
    max_dets: Vec<usize>,
    use_cats: bool,
    dilation_ratio: f64,
) -> PyResult<PySummary> {
    let iou_type = boundary_iou_type(dilation_ratio)?;
    evaluate_summary_impl(
        py,
        iou_type,
        gt_json,
        dt_json,
        parity_mode,
        max_dets,
        use_cats,
    )
}

/// Keypoints (OKS) end-to-end pipeline (ADR-0012) — see
/// [`evaluate_summary_impl`]. Both GT and DT must carry `keypoints`
/// fields.
///
/// `sigmas` is a `dict[int, list[float] | tuple[float, ...]]` mapping
/// `category_id` → per-keypoint sigmas. An empty dict means "use the
/// COCO-person 17-sigma table for every category" (quirk **F1**
/// `corrected`). Sigmas must be supplied already scaled (post-divide-
/// by-10 per pycocotools' internal handling).
#[pyfunction]
#[pyo3(signature = (gt_json, dt_json, parity_mode, max_dets, use_cats, sigmas))]
fn evaluate_keypoints_summary(
    py: Python<'_>,
    gt_json: &Bound<'_, PyBytes>,
    dt_json: &Bound<'_, PyBytes>,
    parity_mode: &str,
    max_dets: Vec<usize>,
    use_cats: bool,
    sigmas: &Bound<'_, PyDict>,
) -> PyResult<PySummary> {
    let iou_type = EvalIouType::Keypoints {
        sigmas: parse_sigmas(sigmas)?,
    };
    evaluate_summary_impl(
        py,
        iou_type,
        gt_json,
        dt_json,
        parity_mode,
        max_dets,
        use_cats,
    )
}

/// Keypoints per-image evaluation pass (ADR-0012). Both GT and DT must
/// carry `keypoints` fields. `sigmas` matches
/// [`evaluate_keypoints_summary`].
#[pyfunction]
#[pyo3(signature = (gt_json, dt_json, parity_mode, max_dets_per_image, use_cats, sigmas))]
fn evaluate_keypoints_grid(
    py: Python<'_>,
    gt_json: &Bound<'_, PyBytes>,
    dt_json: &Bound<'_, PyBytes>,
    parity_mode: &str,
    max_dets_per_image: usize,
    use_cats: bool,
    sigmas: &Bound<'_, PyDict>,
) -> PyResult<PyEvalGrid> {
    let iou_type = EvalIouType::Keypoints {
        sigmas: parse_sigmas(sigmas)?,
    };
    evaluate_grid_impl(
        py,
        iou_type,
        gt_json,
        dt_json,
        parity_mode,
        max_dets_per_image,
        use_cats,
    )
}

/// Decode a Python `dict[int, Sequence[float]]` into the
/// `HashMap<i64, Vec<f64>>` shape `OksSimilarity` consumes. Empty dict
/// is valid — `OksSimilarity` falls back to COCO-person sigmas.
fn parse_sigmas(d: &Bound<'_, PyDict>) -> PyResult<HashMap<i64, Vec<f64>>> {
    let mut out: HashMap<i64, Vec<f64>> = HashMap::with_capacity(d.len());
    for (k, v) in d.iter() {
        let cat: i64 = k.extract()?;
        let sigmas: Vec<f64> = v.extract()?;
        out.insert(cat, sigmas);
    }
    Ok(out)
}

fn run_pipeline(
    iou_type: &EvalIouType,
    gt: &CocoDataset,
    dt: &CocoDetections,
    parity: ParityMode,
    max_dets: &[usize],
    use_cats: bool,
) -> Result<Summary, EvalError> {
    let iou_thr = iou_thresholds();
    let area = area_ranges_for(iou_type);
    let max_det_top = max_dets.iter().copied().max().unwrap_or(100);
    let eval_params = EvaluateParams {
        iou_thresholds: iou_thr,
        area_ranges: &area,
        max_dets_per_image: max_det_top,
        use_cats,
    };
    let is_kp = iou_type.is_keypoints();
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
    if is_kp {
        // ADR-0012 / D5: kp summary is the 10-stat plan over the
        // 3-bucket area grid. Detection's 12-stat plan would index
        // off the end of the kp accumulator's A-axis.
        summarize_with(
            &acc,
            &StatRequest::coco_keypoints_default(),
            iou_thr,
            max_dets,
        )
    } else {
        summarize_detection(&acc, iou_thr, max_dets)
    }
}

fn boundary_iou_type(dilation_ratio: f64) -> PyResult<EvalIouType> {
    if !dilation_ratio.is_finite() || dilation_ratio <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "dilation_ratio must be a positive finite float, got {dilation_ratio}"
        )));
    }
    Ok(EvalIouType::Boundary { dilation_ratio })
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

/// Selects the canonical pycocotools stat plan for
/// [`PyAccumulated::summarize`]. The detection plan resolves to
/// [`vernier_core::summarize_detection`] (12 stats); the keypoints
/// plan resolves to [`vernier_core::StatRequest::coco_keypoints_default`]
/// (10 stats, ADR-0012).
enum SummarizePlan {
    Detection,
    Keypoints,
}

fn parse_summarize_plan(s: &str) -> PyResult<SummarizePlan> {
    match s {
        "detection" => Ok(SummarizePlan::Detection),
        "keypoints" => Ok(SummarizePlan::Keypoints),
        other => Err(PyValueError::new_err(format!(
            "invalid plan {other:?}; expected 'detection' or 'keypoints'"
        ))),
    }
}

/// Map a [`vernier_core::EvalError`] into a `PyErr`. The [`OutOfBudget`]
/// variant is materialized into an [`OutOfBudgetError`] with the
/// `used_bytes`, `budget_bytes`, and `breakdown` attributes set on the
/// exception instance so callers can read them programmatically. The
/// [`NotImplemented`] variant maps to [`PyNotImplementedError`]. All
/// other variants fall back to a `PyValueError` formatted via the
/// underlying [`std::fmt::Display`] impl.
fn eval_error_to_pyerr(py: Python<'_>, e: EvalError) -> PyErr {
    match e {
        EvalError::OutOfBudget {
            used_bytes,
            budget_bytes,
            breakdown,
        } => {
            let exc = OutOfBudgetError::new_err(format!(
                "memory budget exceeded: used {used_bytes} / budget {budget_bytes} bytes"
            ));
            let value = exc.value(py);
            // Best-effort attribute decoration: if any setattr fails we
            // surface the underlying PyErr instead of the OutOfBudget so
            // the user notices the bridge break, but in practice the
            // type/string conversions here cannot fail.
            let breakdown_dict = PyDict::new(py);
            for (k, v) in breakdown.iter() {
                if let Err(err) = breakdown_dict.set_item(*k, *v) {
                    return err;
                }
            }
            if let Err(err) = value.setattr("used_bytes", used_bytes) {
                return err;
            }
            if let Err(err) = value.setattr("budget_bytes", budget_bytes) {
                return err;
            }
            if let Err(err) = value.setattr("breakdown", breakdown_dict) {
                return err;
            }
            exc
        }
        EvalError::NotImplemented { feature } => {
            PyNotImplementedError::new_err(feature.to_string())
        }
        other => PyValueError::new_err(format!("{other}")),
    }
}

/// Kernel-erased wrapper around a [`StreamingEvaluator<K>`].
///
/// One variant per supported kernel plus a [`Self::Finalized`] sentinel
/// the FFI swaps in when [`StreamingEvaluator::finalize`] consumes the
/// inner value. The [`Self::Finalized`] state rejects every operation
/// with [`EvalError::InvalidConfig`] — the error surfaces in Python as
/// a `ValueError` via [`eval_error_to_pyerr`].
enum StreamingState {
    Bbox(StreamingEvaluator<BboxIou>),
    Segm(StreamingEvaluator<SegmIou>),
    Boundary(StreamingEvaluator<BoundaryIou>),
    Keypoints(StreamingEvaluator<OksSimilarity>),
    Finalized,
}

/// Build the `EvalError` returned for any operation attempted after
/// `finalize()`. Sole consumer of the error message; centralized so the
/// string stays consistent across dispatch arms.
fn finalized_error() -> EvalError {
    EvalError::InvalidConfig {
        detail: "StreamingEvaluator has already been finalized".into(),
    }
}

impl StreamingState {
    fn update(&mut self, json_bytes: &[u8]) -> Result<UpdateReport, EvalError> {
        match self {
            Self::Bbox(ev) => ev.update(json_bytes),
            Self::Segm(ev) => ev.update(json_bytes),
            Self::Boundary(ev) => ev.update(json_bytes),
            Self::Keypoints(ev) => ev.update(json_bytes),
            Self::Finalized => Err(finalized_error()),
        }
    }

    fn snapshot(&self, running: bool) -> Result<Summary, EvalError> {
        match self {
            Self::Bbox(ev) => {
                if running {
                    ev.snapshot_running()
                } else {
                    ev.snapshot()
                }
            }
            Self::Segm(ev) => {
                if running {
                    ev.snapshot_running()
                } else {
                    ev.snapshot()
                }
            }
            Self::Boundary(ev) => {
                if running {
                    ev.snapshot_running()
                } else {
                    ev.snapshot()
                }
            }
            Self::Keypoints(ev) => {
                if running {
                    ev.snapshot_running()
                } else {
                    ev.snapshot()
                }
            }
            Self::Finalized => Err(finalized_error()),
        }
    }

    fn take_and_finalize(&mut self) -> Result<Summary, EvalError> {
        // Swap a [`Self::Finalized`] sentinel into place so we can move
        // out of the variant by value. If the inner finalize errors, the
        // evaluator is still considered consumed — re-running finalize
        // would not produce a useful summary either.
        let prev = std::mem::replace(self, Self::Finalized);
        match prev {
            Self::Bbox(ev) => ev.finalize(),
            Self::Segm(ev) => ev.finalize(),
            Self::Boundary(ev) => ev.finalize(),
            Self::Keypoints(ev) => ev.finalize(),
            Self::Finalized => Err(finalized_error()),
        }
    }

    fn images_seen(&self) -> usize {
        match self {
            Self::Bbox(ev) => ev.images_seen(),
            Self::Segm(ev) => ev.images_seen(),
            Self::Boundary(ev) => ev.images_seen(),
            Self::Keypoints(ev) => ev.images_seen(),
            Self::Finalized => 0,
        }
    }

    fn detections_seen(&self) -> usize {
        match self {
            Self::Bbox(ev) => ev.detections_seen(),
            Self::Segm(ev) => ev.detections_seen(),
            Self::Boundary(ev) => ev.detections_seen(),
            Self::Keypoints(ev) => ev.detections_seen(),
            Self::Finalized => 0,
        }
    }

    fn images_pending(&self) -> usize {
        match self {
            Self::Bbox(ev) => ev.images_pending(),
            Self::Segm(ev) => ev.images_pending(),
            Self::Boundary(ev) => ev.images_pending(),
            Self::Keypoints(ev) => ev.images_pending(),
            Self::Finalized => 0,
        }
    }

    fn memory_used_bytes(&self) -> usize {
        match self {
            Self::Bbox(ev) => ev.memory_used_bytes(),
            Self::Segm(ev) => ev.memory_used_bytes(),
            Self::Boundary(ev) => ev.memory_used_bytes(),
            Self::Keypoints(ev) => ev.memory_used_bytes(),
            Self::Finalized => 0,
        }
    }

    fn memory_budget_bytes(&self) -> usize {
        match self {
            Self::Bbox(ev) => ev.budget().bytes,
            Self::Segm(ev) => ev.budget().bytes,
            Self::Boundary(ev) => ev.budget().bytes,
            Self::Keypoints(ev) => ev.budget().bytes,
            Self::Finalized => 0,
        }
    }
}

/// Streaming evaluator surface (ADR-0013). Single-writer per the runtime
/// `owner_thread` check; mutable state guarded by an internal `Mutex` so
/// the pyclass can stay non-frozen and accept `&self` on its methods.
#[pyclass(module = "vernier._core", name = "StreamingEvaluator")]
struct PyStreamingEvaluator {
    state: Mutex<StreamingState>,
    owner_thread: Mutex<Option<std::thread::ThreadId>>,
    warned_soft_budget: AtomicBool,
}

impl PyStreamingEvaluator {
    fn lock_state(&self) -> PyResult<std::sync::MutexGuard<'_, StreamingState>> {
        self.state
            .lock()
            .map_err(|_| PyRuntimeError::new_err("StreamingEvaluator state mutex poisoned"))
    }

    /// Single-writer guard. The first `update()` call records the
    /// calling thread; later calls verify it. Mismatch raises a
    /// `RuntimeError` that names both threads.
    fn check_owner_thread(&self) -> PyResult<()> {
        let mut owner = self.owner_thread.lock().map_err(|_| {
            PyRuntimeError::new_err("StreamingEvaluator owner_thread mutex poisoned")
        })?;
        let current = std::thread::current().id();
        match *owner {
            None => {
                *owner = Some(current);
                Ok(())
            }
            Some(prior) if prior == current => Ok(()),
            Some(prior) => Err(PyRuntimeError::new_err(format!(
                "StreamingEvaluator is single-writer; submitted from {current:?}, owned by {prior:?}"
            ))),
        }
    }
}

#[pymethods]
impl PyStreamingEvaluator {
    #[new]
    #[pyo3(signature = (
        gt_json,
        *,
        iou_type = "bbox",
        parity_mode = "corrected",
        max_dets = vec![1, 10, 100],
        use_cats = true,
        memory_budget_bytes = None,
        dilation_ratio = 0.02,
        sigmas = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        gt_json: &Bound<'_, PyBytes>,
        iou_type: &str,
        parity_mode: &str,
        max_dets: Vec<usize>,
        use_cats: bool,
        memory_budget_bytes: Option<usize>,
        dilation_ratio: f64,
        sigmas: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let parity = parse_parity_mode(parity_mode)?;
        require_nonempty_max_dets(&max_dets)?;
        // Quirk A2 (aligned): same sort the batch path applies — the
        // largest entry caps `max_dets_per_image`; smaller entries are
        // sliced downstream by `accumulate`.
        let mut max_dets = max_dets;
        sort_max_dets(&mut max_dets);
        let max_dets_per_image = max_dets.iter().copied().max().unwrap_or(100);

        let budget = match memory_budget_bytes {
            Some(b) => MemoryBudget {
                bytes: b,
                soft_warn_fraction: 0.80,
            },
            None => MemoryBudget::auto_default(),
        };

        let dataset = CocoDataset::from_json_bytes(gt_json.as_bytes())
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;

        // Build the kernel-typed evaluator. Each arm constructs the
        // matching `EvalIouType` solely to reuse `area_ranges_for`'s
        // detection-vs-keypoints fork; this keeps the area-bucket
        // selection logic in one place.
        let state = match iou_type {
            "bbox" => {
                let area = area_ranges_for(&EvalIouType::Bbox);
                let params = OwnedEvaluateParams {
                    iou_thresholds: iou_thresholds().to_vec(),
                    area_ranges: area,
                    max_dets_per_image,
                    use_cats,
                };
                let ev = StreamingEvaluator::new(dataset, BboxIou, params, parity, budget)
                    .map_err(|e| PyValueError::new_err(format!("{e}")))?;
                StreamingState::Bbox(ev)
            }
            "segm" => {
                let area = area_ranges_for(&EvalIouType::Segm);
                let params = OwnedEvaluateParams {
                    iou_thresholds: iou_thresholds().to_vec(),
                    area_ranges: area,
                    max_dets_per_image,
                    use_cats,
                };
                let ev = StreamingEvaluator::new(dataset, SegmIou, params, parity, budget)
                    .map_err(|e| PyValueError::new_err(format!("{e}")))?;
                StreamingState::Segm(ev)
            }
            "boundary" => {
                let iou_kind = boundary_iou_type(dilation_ratio)?;
                let area = area_ranges_for(&iou_kind);
                let params = OwnedEvaluateParams {
                    iou_thresholds: iou_thresholds().to_vec(),
                    area_ranges: area,
                    max_dets_per_image,
                    use_cats,
                };
                let kernel = BoundaryIou { dilation_ratio };
                let ev = StreamingEvaluator::new(dataset, kernel, params, parity, budget)
                    .map_err(|e| PyValueError::new_err(format!("{e}")))?;
                StreamingState::Boundary(ev)
            }
            "keypoints" => {
                let parsed_sigmas = match sigmas {
                    Some(d) => parse_sigmas(d)?,
                    None => HashMap::new(),
                };
                let iou_kind = EvalIouType::Keypoints {
                    sigmas: parsed_sigmas.clone(),
                };
                let area = area_ranges_for(&iou_kind);
                let params = OwnedEvaluateParams {
                    iou_thresholds: iou_thresholds().to_vec(),
                    area_ranges: area,
                    max_dets_per_image,
                    use_cats,
                };
                let kernel = OksSimilarity::new(parsed_sigmas);
                let ev = StreamingEvaluator::new(dataset, kernel, params, parity, budget)
                    .map_err(|e| PyValueError::new_err(format!("{e}")))?;
                StreamingState::Keypoints(ev)
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "invalid iou_type {other:?}; expected 'bbox', 'segm', 'boundary', or 'keypoints'"
                )));
            }
        };

        Ok(Self {
            state: Mutex::new(state),
            owner_thread: Mutex::new(None),
            warned_soft_budget: AtomicBool::new(false),
        })
    }

    /// Submit a batch of detections (loadRes-shaped JSON bytes). Returns
    /// an `_UpdateReportDict` describing what was accepted plus the
    /// post-update memory total. Single-writer: only the first calling
    /// thread is permitted to call this method.
    fn update<'py>(
        &self,
        py: Python<'py>,
        detections: &Bound<'py, PyBytes>,
    ) -> PyResult<Bound<'py, PyDict>> {
        self.check_owner_thread()?;
        // Take an owned copy of the bytes so we can drop the GIL for the
        // (potentially expensive) update_parsed call below. Cloning the
        // wire payload once is cheap relative to the parse + match work.
        let bytes: Vec<u8> = detections.as_bytes().to_vec();
        // Lock inside `py.detach` so the `MutexGuard` (which is `!Send`)
        // never crosses the closure boundary on Send-checking. The
        // `Mutex` itself is `Send + Sync`, so a borrow of it is fine to
        // capture into the GIL-released body.
        let state_mutex = &self.state;
        let (report, memory_used_bytes) = py
            .detach(move || {
                let mut guard = state_mutex.lock().map_err(|_| EvalError::InvalidConfig {
                    detail: "StreamingEvaluator state mutex poisoned".into(),
                })?;
                let report = guard.update(&bytes)?;
                let memory = guard.memory_used_bytes();
                Ok::<(UpdateReport, usize), EvalError>((report, memory))
            })
            .map_err(|e| eval_error_to_pyerr(py, e))?;

        // One-shot soft-warn emission. The atomic guard ensures even
        // re-entrant code paths (e.g., a warning filter that re-calls
        // update) won't fire the warning twice.
        if report.soft_warn_triggered && !self.warned_soft_budget.swap(true, Ordering::AcqRel) {
            let warnings = py.import("warnings")?;
            let warn_class = py.get_type::<MemoryBudgetWarning>();
            let msg = format!(
                "StreamingEvaluator memory usage crossed the soft-warn threshold ({} bytes used)",
                memory_used_bytes
            );
            warnings.getattr("warn")?.call1((msg, warn_class))?;
        }

        let dict = PyDict::new(py);
        // `n_images_in_batch` is the closest analog to "new_images" —
        // the streaming evaluator rejects re-submissions so every image
        // in a batch is necessarily new.
        dict.set_item("new_images", report.n_images_in_batch)?;
        dict.set_item("new_detections", report.n_detections_accepted)?;
        dict.set_item("memory_used_bytes", memory_used_bytes)?;
        dict.set_item("soft_warn_triggered", report.soft_warn_triggered)?;
        Ok(dict)
    }

    /// Compute a `Summary` over the current store. `running=True` selects
    /// the "fast" snapshot path (currently identical to the regular
    /// snapshot — see `StreamingEvaluator::snapshot_running` rustdoc).
    #[pyo3(signature = (*, running = false))]
    fn snapshot(&self, py: Python<'_>, running: bool) -> PyResult<PySummary> {
        let state_mutex = &self.state;
        let summary = py
            .detach(move || {
                let guard = state_mutex.lock().map_err(|_| EvalError::InvalidConfig {
                    detail: "StreamingEvaluator state mutex poisoned".into(),
                })?;
                guard.snapshot(running)
            })
            .map_err(|e| eval_error_to_pyerr(py, e))?;
        Ok(PySummary { inner: summary })
    }

    /// Consume the evaluator and return the final summary. Subsequent
    /// calls on this object error out with the "already finalized"
    /// message.
    fn finalize(&self, py: Python<'_>) -> PyResult<PySummary> {
        let state_mutex = &self.state;
        let summary = py
            .detach(move || {
                let mut guard = state_mutex.lock().map_err(|_| EvalError::InvalidConfig {
                    detail: "StreamingEvaluator state mutex poisoned".into(),
                })?;
                guard.take_and_finalize()
            })
            .map_err(|e| eval_error_to_pyerr(py, e))?;
        Ok(PySummary { inner: summary })
    }

    /// Distinct images that have received at least one detection.
    #[getter]
    fn images_seen(&self) -> PyResult<usize> {
        Ok(self.lock_state()?.images_seen())
    }

    /// Cumulative number of detections accepted across all batches.
    #[getter]
    fn detections_seen(&self) -> PyResult<usize> {
        Ok(self.lock_state()?.detections_seen())
    }

    /// GT images that have not yet received any detection.
    #[getter]
    fn images_pending(&self) -> PyResult<usize> {
        Ok(self.lock_state()?.images_pending())
    }

    /// Bytes the evaluator currently holds across cells, scores, and
    /// match flags.
    #[getter]
    fn memory_used_bytes(&self) -> PyResult<usize> {
        Ok(self.lock_state()?.memory_used_bytes())
    }

    /// Configured hard cap; an `update()` whose insert would push past
    /// this number raises `OutOfBudgetError`.
    #[getter]
    fn memory_budget_bytes(&self) -> PyResult<usize> {
        Ok(self.lock_state()?.memory_budget_bytes())
    }
}

/// Kernel-erased wrapper around a [`background::BackgroundEvaluator<K>`].
///
/// Mirrors [`StreamingState`] for the background surface. Each variant
/// owns a worker thread that in turn owns a `StreamingEvaluator<K>`. The
/// [`Self::Finalized`] sentinel rejects every operation with
/// [`EvalError::InvalidConfig`] (mapped to a `ValueError` in Python).
enum BackgroundEvalState {
    Bbox(background::BackgroundEvaluator<BboxIou>),
    Segm(background::BackgroundEvaluator<SegmIou>),
    Boundary(background::BackgroundEvaluator<BoundaryIou>),
    Keypoints(background::BackgroundEvaluator<OksSimilarity>),
    Finalized,
}

/// Build the `EvalError` returned for any operation attempted after
/// `finalize()` on the background surface. Mirrors [`finalized_error`].
fn background_finalized_error() -> EvalError {
    EvalError::InvalidConfig {
        detail: "BackgroundEvaluator has already been finalized".into(),
    }
}

impl BackgroundEvalState {
    /// Parse the wire-format detection bytes for the active kernel and
    /// post the resulting batch to the worker. `timeout` mirrors the
    /// Python `timeout=` parameter on `submit()`:
    ///
    /// - `None` → block forever (`submit_blocking`)
    /// - `Some(Duration::ZERO)` → single non-blocking attempt
    /// - `Some(t > 0)` → bounded wait
    fn submit(
        &self,
        bytes: &[u8],
        timeout: Option<Duration>,
    ) -> Result<(), background::SubmitError> {
        match self {
            Self::Bbox(ev) => {
                let parsed = ParsedDetections::<BboxIou>::from_json_bytes(bytes)
                    .map_err(background::SubmitError::Eval)?;
                match timeout {
                    None => ev
                        .submit_blocking(parsed)
                        .map_err(background::SubmitError::Eval),
                    Some(t) => ev.submit_timeout(parsed, t),
                }
            }
            Self::Segm(ev) => {
                let parsed = ParsedDetections::<SegmIou>::from_json_bytes(bytes)
                    .map_err(background::SubmitError::Eval)?;
                match timeout {
                    None => ev
                        .submit_blocking(parsed)
                        .map_err(background::SubmitError::Eval),
                    Some(t) => ev.submit_timeout(parsed, t),
                }
            }
            Self::Boundary(ev) => {
                let parsed = ParsedDetections::<BoundaryIou>::from_json_bytes(bytes)
                    .map_err(background::SubmitError::Eval)?;
                match timeout {
                    None => ev
                        .submit_blocking(parsed)
                        .map_err(background::SubmitError::Eval),
                    Some(t) => ev.submit_timeout(parsed, t),
                }
            }
            Self::Keypoints(ev) => {
                let parsed = ParsedDetections::<OksSimilarity>::from_json_bytes(bytes)
                    .map_err(background::SubmitError::Eval)?;
                match timeout {
                    None => ev
                        .submit_blocking(parsed)
                        .map_err(background::SubmitError::Eval),
                    Some(t) => ev.submit_timeout(parsed, t),
                }
            }
            Self::Finalized => Err(background::SubmitError::Eval(background_finalized_error())),
        }
    }

    fn snapshot(&self, peek: bool) -> Result<Summary, EvalError> {
        match self {
            Self::Bbox(ev) => ev.snapshot(peek),
            Self::Segm(ev) => ev.snapshot(peek),
            Self::Boundary(ev) => ev.snapshot(peek),
            Self::Keypoints(ev) => ev.snapshot(peek),
            Self::Finalized => Err(background_finalized_error()),
        }
    }

    fn take_and_finalize(&mut self) -> Result<Summary, EvalError> {
        let prev = std::mem::replace(self, Self::Finalized);
        match prev {
            Self::Bbox(ev) => ev.finalize(),
            Self::Segm(ev) => ev.finalize(),
            Self::Boundary(ev) => ev.finalize(),
            Self::Keypoints(ev) => ev.finalize(),
            Self::Finalized => Err(background_finalized_error()),
        }
    }

    fn take_scheduling_outcome(&self) -> Option<Result<(), String>> {
        match self {
            Self::Bbox(ev) => ev.take_scheduling_outcome(),
            Self::Segm(ev) => ev.take_scheduling_outcome(),
            Self::Boundary(ev) => ev.take_scheduling_outcome(),
            Self::Keypoints(ev) => ev.take_scheduling_outcome(),
            Self::Finalized => None,
        }
    }

    fn images_seen(&self) -> usize {
        match self {
            Self::Bbox(ev) => ev.images_seen(),
            Self::Segm(ev) => ev.images_seen(),
            Self::Boundary(ev) => ev.images_seen(),
            Self::Keypoints(ev) => ev.images_seen(),
            Self::Finalized => 0,
        }
    }

    fn detections_seen(&self) -> usize {
        match self {
            Self::Bbox(ev) => ev.detections_seen(),
            Self::Segm(ev) => ev.detections_seen(),
            Self::Boundary(ev) => ev.detections_seen(),
            Self::Keypoints(ev) => ev.detections_seen(),
            Self::Finalized => 0,
        }
    }

    fn queue_depth(&self) -> usize {
        match self {
            Self::Bbox(ev) => ev.queue_depth(),
            Self::Segm(ev) => ev.queue_depth(),
            Self::Boundary(ev) => ev.queue_depth(),
            Self::Keypoints(ev) => ev.queue_depth(),
            Self::Finalized => 0,
        }
    }

    fn memory_used_bytes(&self) -> usize {
        match self {
            Self::Bbox(ev) => ev.memory_used_bytes(),
            Self::Segm(ev) => ev.memory_used_bytes(),
            Self::Boundary(ev) => ev.memory_used_bytes(),
            Self::Keypoints(ev) => ev.memory_used_bytes(),
            Self::Finalized => 0,
        }
    }

    /// Best-effort cooperative shutdown. Used by `__exit__` and `__del__`
    /// when the evaluator hasn't already been finalized.
    fn shutdown(&mut self) {
        let prev = std::mem::replace(self, Self::Finalized);
        match prev {
            Self::Bbox(ev) => ev.shutdown(),
            Self::Segm(ev) => ev.shutdown(),
            Self::Boundary(ev) => ev.shutdown(),
            Self::Keypoints(ev) => ev.shutdown(),
            Self::Finalized => {}
        }
    }

    /// Test-only: forward to the worker's `_inject_poison_for_tests`.
    /// Gated behind the `test-poison` Cargo feature; only the panic-recovery
    /// test in `tests/python/background/test_background_worker_panic.py`
    /// reaches this.
    #[cfg(feature = "test-poison")]
    fn inject_poison_for_tests(&self) -> Result<(), EvalError> {
        match self {
            Self::Bbox(ev) => ev._inject_poison_for_tests(),
            Self::Segm(ev) => ev._inject_poison_for_tests(),
            Self::Boundary(ev) => ev._inject_poison_for_tests(),
            Self::Keypoints(ev) => ev._inject_poison_for_tests(),
            Self::Finalized => Err(background_finalized_error()),
        }
    }
}

/// Map a [`background::SubmitError`] to a Python exception. `Eval` is
/// routed through [`eval_error_to_pyerr`]; `Full` is materialized into a
/// [`QueueFullError`] with the `queue_capacity` and `timeout` (in
/// fractional seconds) attached as instance attributes.
fn submit_error_to_pyerr(py: Python<'_>, e: background::SubmitError) -> PyErr {
    match e {
        background::SubmitError::Eval(inner) => eval_error_to_pyerr(py, inner),
        background::SubmitError::Full(full) => {
            let exc = QueueFullError::new_err(format!(
                "background submit queue full (capacity={}, timeout={:?})",
                full.queue_capacity, full.timeout
            ));
            let value = exc.value(py);
            if let Err(err) = value.setattr("queue_capacity", full.queue_capacity) {
                return err;
            }
            // `timeout` is always finite here — `submit_blocking` (the
            // `None` Python case) cannot return `Full` — so report it as
            // a float in seconds, matching the docstring on
            // `QueueFullError`.
            let timeout_secs = full.timeout.as_secs_f64();
            if let Err(err) = value.setattr("timeout", timeout_secs) {
                return err;
            }
            exc
        }
    }
}

/// Background-evaluator surface (ADR-0014). Wraps a worker thread that
/// owns the `StreamingEvaluator<K>`; every public method either sends on
/// the channel or reads atomic counters. Not frozen — `finalize()` and
/// `__exit__` need to mutate state.
#[pyclass(module = "vernier._core", name = "BackgroundEvaluator")]
struct PyBackgroundEvaluator {
    state: Mutex<BackgroundEvalState>,
}

impl PyBackgroundEvaluator {
    fn lock_state(&self) -> PyResult<std::sync::MutexGuard<'_, BackgroundEvalState>> {
        self.state
            .lock()
            .map_err(|_| PyRuntimeError::new_err("BackgroundEvaluator state mutex poisoned"))
    }
}

#[pymethods]
impl PyBackgroundEvaluator {
    #[new]
    #[pyo3(signature = (
        gt_json,
        *,
        iou_type = "bbox",
        parity_mode = "corrected",
        max_dets = vec![1, 10, 100],
        use_cats = true,
        memory_budget_bytes = None,
        dilation_ratio = 0.02,
        sigmas = None,
        queue_capacity = 8,
        worker_affinity = None,
        worker_nice = 5,
        shutdown_timeout_seconds = 5.0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        gt_json: &Bound<'_, PyBytes>,
        iou_type: &str,
        parity_mode: &str,
        max_dets: Vec<usize>,
        use_cats: bool,
        memory_budget_bytes: Option<usize>,
        dilation_ratio: f64,
        sigmas: Option<&Bound<'_, PyDict>>,
        queue_capacity: usize,
        worker_affinity: Option<usize>,
        worker_nice: i32,
        shutdown_timeout_seconds: f64,
    ) -> PyResult<Self> {
        let parity = parse_parity_mode(parity_mode)?;
        require_nonempty_max_dets(&max_dets)?;
        // Quirk A2 (aligned): same sort the streaming/batch paths apply.
        let mut max_dets = max_dets;
        sort_max_dets(&mut max_dets);
        let max_dets_per_image = max_dets.iter().copied().max().unwrap_or(100);

        let budget = match memory_budget_bytes {
            Some(b) => MemoryBudget {
                bytes: b,
                soft_warn_fraction: 0.80,
            },
            None => MemoryBudget::auto_default(),
        };

        let dataset = CocoDataset::from_json_bytes(gt_json.as_bytes())
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;

        if !shutdown_timeout_seconds.is_finite() || shutdown_timeout_seconds < 0.0 {
            return Err(PyValueError::new_err(format!(
                "shutdown_timeout_seconds must be a non-negative finite float, got {shutdown_timeout_seconds}"
            )));
        }
        let config = background::BackgroundConfig {
            queue_capacity,
            worker_affinity,
            worker_nice,
            shutdown_timeout: Duration::from_secs_f64(shutdown_timeout_seconds),
        };

        let state = match iou_type {
            "bbox" => {
                let area = area_ranges_for(&EvalIouType::Bbox);
                let params = OwnedEvaluateParams {
                    iou_thresholds: iou_thresholds().to_vec(),
                    area_ranges: area,
                    max_dets_per_image,
                    use_cats,
                };
                let ev = StreamingEvaluator::new(dataset, BboxIou, params, parity, budget)
                    .map_err(|e| PyValueError::new_err(format!("{e}")))?;
                let bg = background::BackgroundEvaluator::spawn(ev, config)
                    .map_err(|e| eval_error_to_pyerr(py, e))?;
                BackgroundEvalState::Bbox(bg)
            }
            "segm" => {
                let area = area_ranges_for(&EvalIouType::Segm);
                let params = OwnedEvaluateParams {
                    iou_thresholds: iou_thresholds().to_vec(),
                    area_ranges: area,
                    max_dets_per_image,
                    use_cats,
                };
                let ev = StreamingEvaluator::new(dataset, SegmIou, params, parity, budget)
                    .map_err(|e| PyValueError::new_err(format!("{e}")))?;
                let bg = background::BackgroundEvaluator::spawn(ev, config)
                    .map_err(|e| eval_error_to_pyerr(py, e))?;
                BackgroundEvalState::Segm(bg)
            }
            "boundary" => {
                let iou_kind = boundary_iou_type(dilation_ratio)?;
                let area = area_ranges_for(&iou_kind);
                let params = OwnedEvaluateParams {
                    iou_thresholds: iou_thresholds().to_vec(),
                    area_ranges: area,
                    max_dets_per_image,
                    use_cats,
                };
                let kernel = BoundaryIou { dilation_ratio };
                let ev = StreamingEvaluator::new(dataset, kernel, params, parity, budget)
                    .map_err(|e| PyValueError::new_err(format!("{e}")))?;
                let bg = background::BackgroundEvaluator::spawn(ev, config)
                    .map_err(|e| eval_error_to_pyerr(py, e))?;
                BackgroundEvalState::Boundary(bg)
            }
            "keypoints" => {
                let parsed_sigmas = match sigmas {
                    Some(d) => parse_sigmas(d)?,
                    None => HashMap::new(),
                };
                let iou_kind = EvalIouType::Keypoints {
                    sigmas: parsed_sigmas.clone(),
                };
                let area = area_ranges_for(&iou_kind);
                let params = OwnedEvaluateParams {
                    iou_thresholds: iou_thresholds().to_vec(),
                    area_ranges: area,
                    max_dets_per_image,
                    use_cats,
                };
                let kernel = OksSimilarity::new(parsed_sigmas);
                let ev = StreamingEvaluator::new(dataset, kernel, params, parity, budget)
                    .map_err(|e| PyValueError::new_err(format!("{e}")))?;
                let bg = background::BackgroundEvaluator::spawn(ev, config)
                    .map_err(|e| eval_error_to_pyerr(py, e))?;
                BackgroundEvalState::Keypoints(bg)
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "invalid iou_type {other:?}; expected 'bbox', 'segm', 'boundary', or 'keypoints'"
                )));
            }
        };

        let this = Self {
            state: Mutex::new(state),
        };

        // Briefly poll for the worker's startup scheduling result. The
        // worker stamps `state.scheduling_outcome` before pulling its
        // first message; with a 1ms cadence and 10 rounds we give it up
        // to ~10ms of slack. If still `None`, drop it on the floor —
        // it'll be re-checked next time something reads scheduling.
        let outcome: Option<Result<(), String>> = {
            let mut found: Option<Result<(), String>> = None;
            for _ in 0..10 {
                {
                    let guard = this.lock_state()?;
                    if let Some(o) = guard.take_scheduling_outcome() {
                        found = Some(o);
                        break;
                    }
                }
                py.detach(|| std::thread::sleep(Duration::from_millis(1)));
            }
            found
        };
        if let Some(Err(msg)) = outcome {
            // Use plain `UserWarning` here (per ADR-0014 §"Worker
            // scheduling"): MemoryBudgetWarning is for soft-budget
            // crossings, not scheduling. Emit-once is implicit — the
            // worker only stamps the outcome once.
            let warnings = py.import("warnings")?;
            let warn_class = py.get_type::<PyUserWarning>();
            let full_msg = format!("BackgroundEvaluator scheduling: {msg}");
            warnings.getattr("warn")?.call1((full_msg, warn_class))?;
        }

        Ok(this)
    }

    /// Submit a parsed-JSON detection batch to the worker. `timeout`
    /// controls backpressure:
    ///
    /// - `None` (default) → block until a slot is free
    /// - `0.0` → single non-blocking attempt; raise `QueueFullError` if
    ///   the queue is full
    /// - `t > 0.0` → wait up to `t` seconds; raise `QueueFullError` on
    ///   timeout
    #[pyo3(signature = (detections, *, timeout = None))]
    fn submit(
        &self,
        py: Python<'_>,
        detections: &Bound<'_, PyBytes>,
        timeout: Option<f64>,
    ) -> PyResult<()> {
        let timeout_dur = match timeout {
            None => None,
            Some(t) => {
                if !t.is_finite() || t < 0.0 {
                    return Err(PyValueError::new_err(format!(
                        "timeout must be a non-negative finite float or None, got {t}"
                    )));
                }
                Some(Duration::from_secs_f64(t))
            }
        };
        // Copy bytes out under the GIL so the parse + send can run with
        // the GIL released. The kernel-typed parse happens inside
        // `BackgroundEvalState::submit` to keep the dispatch single-site.
        let bytes: Vec<u8> = detections.as_bytes().to_vec();
        let state_mutex = &self.state;
        let result = py.detach(move || {
            let guard = state_mutex.lock().map_err(|_| {
                background::SubmitError::Eval(EvalError::InvalidConfig {
                    detail: "BackgroundEvaluator state mutex poisoned".into(),
                })
            })?;
            guard.submit(&bytes, timeout_dur)
        });
        result.map_err(|e| submit_error_to_pyerr(py, e))
    }

    /// Compute a `Summary` against the worker's current store. `peek=True`
    /// uses the cheaper snapshot path (currently identical to the regular
    /// snapshot).
    #[pyo3(signature = (*, peek = false))]
    fn snapshot(&self, py: Python<'_>, peek: bool) -> PyResult<PySummary> {
        let state_mutex = &self.state;
        let summary = py
            .detach(move || {
                let guard = state_mutex.lock().map_err(|_| EvalError::InvalidConfig {
                    detail: "BackgroundEvaluator state mutex poisoned".into(),
                })?;
                guard.snapshot(peek)
            })
            .map_err(|e| eval_error_to_pyerr(py, e))?;
        Ok(PySummary { inner: summary })
    }

    /// Drain the queue, finalize the evaluator, and join the worker.
    /// Subsequent calls error with the "already finalized" message.
    fn finalize(&self, py: Python<'_>) -> PyResult<PySummary> {
        let state_mutex = &self.state;
        let summary = py
            .detach(move || {
                let mut guard = state_mutex.lock().map_err(|_| EvalError::InvalidConfig {
                    detail: "BackgroundEvaluator state mutex poisoned".into(),
                })?;
                guard.take_and_finalize()
            })
            .map_err(|e| eval_error_to_pyerr(py, e))?;
        Ok(PySummary { inner: summary })
    }

    /// Context-manager entry. Returns `self` so `with ev as e:` binds
    /// the same instance the user constructed.
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Context-manager exit: best-effort shutdown. Errors raised inside
    /// the `with` block are propagated; errors from the shutdown itself
    /// are silenced (the original exception is more important).
    #[pyo3(signature = (_exc_type=None, _exc=None, _tb=None))]
    fn __exit__(
        &self,
        py: Python<'_>,
        _exc_type: Option<Py<PyAny>>,
        _exc: Option<Py<PyAny>>,
        _tb: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        let state_mutex = &self.state;
        py.detach(|| {
            if let Ok(mut guard) = state_mutex.lock() {
                guard.shutdown();
            }
        });
        Ok(())
    }

    /// Best-effort cleanup if the user lets the wrapper go out of scope
    /// without explicit `finalize()` / `__exit__`. Silences all errors —
    /// raising from `__del__` is invisible to the caller anyway.
    ///
    /// `shutdown()` polls the worker's `JoinHandle` for up to
    /// `shutdown_timeout` (5 s by default), so dropping the GIL via
    /// `py.detach` is mandatory: otherwise garbage collection from the
    /// main interpreter thread can freeze for seconds.
    fn __del__(&self, py: Python<'_>) {
        let state_mutex = &self.state;
        py.detach(|| {
            if let Ok(mut guard) = state_mutex.lock() {
                guard.shutdown();
            }
        });
    }

    /// Mirror of `StreamingEvaluator::images_seen()`. Advisory — updated
    /// by the worker after each successful submit.
    #[getter]
    fn images_seen(&self) -> PyResult<usize> {
        Ok(self.lock_state()?.images_seen())
    }

    /// Mirror of `StreamingEvaluator::detections_seen()`. Advisory.
    #[getter]
    fn detections_seen(&self) -> PyResult<usize> {
        Ok(self.lock_state()?.detections_seen())
    }

    /// Approximate count of `Update` messages waiting in the channel.
    #[getter]
    fn queue_depth(&self) -> PyResult<usize> {
        Ok(self.lock_state()?.queue_depth())
    }

    /// Mirror of `StreamingEvaluator::memory_used_bytes()`. Advisory.
    #[getter]
    fn memory_used_bytes(&self) -> PyResult<usize> {
        Ok(self.lock_state()?.memory_used_bytes())
    }

    /// Test-only: post a `Poison` message that panics the worker. Visible
    /// only when the FFI crate is compiled with `--features test-poison`.
    /// The `tests/python/background/test_background_worker_panic.py` test
    /// uses `hasattr(...)` to skip itself when the feature is absent.
    #[cfg(feature = "test-poison")]
    fn _inject_poison_for_tests(&self, py: Python<'_>) -> PyResult<()> {
        let guard = self.lock_state()?;
        guard
            .inject_poison_for_tests()
            .map_err(|e| eval_error_to_pyerr(py, e))
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
    m.add_function(wrap_pyfunction!(evaluate_boundary_summary, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_boundary_grid, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_keypoints_summary, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_keypoints_grid, m)?)?;
    m.add_class::<PySummary>()?;
    m.add_class::<PyEvalGrid>()?;
    m.add_class::<PyAccumulated>()?;
    m.add_class::<PyStreamingEvaluator>()?;
    m.add_class::<PyBackgroundEvaluator>()?;
    m.add("OutOfBudgetError", m.py().get_type::<OutOfBudgetError>())?;
    m.add("QueueFullError", m.py().get_type::<QueueFullError>())?;
    m.add(
        "MemoryBudgetWarning",
        m.py().get_type::<MemoryBudgetWarning>(),
    )?;
    m.add("__version__", vernier_core::VERSION)?;
    Ok(())
}
