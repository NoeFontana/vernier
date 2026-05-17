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
//! Two threaded surfaces extend the immutable batch evaluators:
//!
//! - `InstanceStreamOrchestrator` (ADR-0035) is an internal Rust
//!   struct — not a pyclass. The `evaluate_instance_to_partial` and
//!   `merge_instance_partials` pyfunctions own one synchronously and
//!   drive it through a single construct + update + finalize-to-partial
//!   cycle.
//! - `PyBackgroundEvaluator` (ADR-0014) wraps a streaming evaluator in
//!   a dedicated worker thread. The worker owns the inner evaluator,
//!   so the single-writer rule is satisfied by construction; callers
//!   may submit from any thread. Best-effort `nice` and core-affinity
//!   are applied to the worker via `thread-priority` and `core_affinity`;
//!   any scheduling syscall failure surfaces as a one-shot `UserWarning`
//!   on the constructing thread.

// Allocator A/B knob (`mimalloc-global` feature); see Cargo.toml.
#[cfg(feature = "mimalloc-global")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

use numpy::ndarray::Array1;
use numpy::ToPyArray;
use pyo3::create_exception;
use pyo3::exceptions::{
    PyNotImplementedError, PyRuntimeError, PyTypeError, PyUserWarning, PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict, PyList};

use vernier_core::accumulate::{accumulate, sort_max_dets, AccumulateParams, PerImageEval};
use vernier_core::dataset::{CategoryId, DetectionInput};
use vernier_core::evaluate::{
    evaluate_boundary_cached, evaluate_segm_cached, BoundaryIouCached, EvalImageMeta, EvalKernel,
    OwnedEvaluateParams, SegmIouCached,
};
use vernier_core::parity::{iou_thresholds, recall_thresholds};
use vernier_core::similarity::{BboxIou, BoundaryIou, OksSimilarity, SegmIou};
use vernier_core::stream::{MemoryBudget, ParsedDetections, StreamingEvaluator, UpdateReport};
use vernier_core::summarize::{
    summarize_detection, summarize_with, summarize_with_lvis, StatRequest,
};
use vernier_core::tables::{Tables, TablesConfig, TablesRequest};
use vernier_core::{
    evaluate_bbox, evaluate_boundary, evaluate_keypoints, evaluate_segm, Accumulated, AreaRange,
    CocoDataset, CocoDetections, EvalDataset, EvalError, EvalGrid, EvaluateParams, ParityMode,
    Summary,
};

mod array_ingest;
mod arrow_helpers;
mod background;
mod background_streaming;
mod breakdown;
mod calibration;
mod confusion;
mod dataset;
mod dlpack;
mod lrp;
mod manifest_py;
mod numpy_utils;
mod panoptic;
mod panoptic_tables;
mod partition_py;
mod semantic;
mod semantic_tables;
mod tables;
mod thread_sched;
mod threads;
mod tide;

use dataset::{DatasetCaches, PyDataset};

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

create_exception!(
    vernier._core,
    InvalidAnnotationError,
    pyo3::exceptions::PyValueError,
    "Annotation could not be parsed or references an unknown image_id / category_id."
);
create_exception!(
    vernier._core,
    NonFiniteError,
    pyo3::exceptions::PyValueError,
    "A NaN or infinity reached arithmetic that cannot tolerate it (e.g., a detection score)."
);
create_exception!(
    vernier._core,
    DimensionMismatchError,
    pyo3::exceptions::PyValueError,
    "Two annotations or two RLEs disagree on dimensions in a way that makes the operation undefined."
);
create_exception!(
    vernier._core,
    InvalidConfigError,
    pyo3::exceptions::PyValueError,
    "Caller-supplied evaluation parameters are inconsistent with the data they are being applied to."
);

// ADR-0031 distributed-eval merge errors. Each carries typed
// attributes so a CI gate can match on them directly without
// parsing a string message.
create_exception!(
    vernier._core,
    PartialFormatMismatch,
    pyo3::exceptions::PyRuntimeError,
    "Distributed-eval partial blob is structurally malformed (magic / version / CRC / kernel kind / parity / retain_iou / grid dimensions / rkyv archive).\n\nAttributes: kind (string discriminator)."
);
create_exception!(
    vernier._core,
    PartialDatasetMismatch,
    pyo3::exceptions::PyRuntimeError,
    "Distributed-eval partial was computed against a different dataset than the receiving evaluator.\n\nAttributes: expected (bytes), actual (bytes)."
);
create_exception!(
    vernier._core,
    PartialParamsMismatch,
    pyo3::exceptions::PyRuntimeError,
    "Distributed-eval partial was computed against different evaluation params than the receiving evaluator.\n\nAttributes: expected (bytes), actual (bytes)."
);
create_exception!(
    vernier._core,
    PartialPartitionOverlap,
    pyo3::exceptions::PyRuntimeError,
    "Two distributed-eval partials cover the same image_id (sampler bug).\n\nAttributes: rank_a, rank_b, image_id."
);
create_exception!(
    vernier._core,
    PartialRankCollision,
    pyo3::exceptions::PyRuntimeError,
    "Two strict-mode distributed-eval partials share a rank_id.\n\nAttributes: rank_id."
);

/// Returns the underlying `vernier-core` version string. Useful as a smoke
/// test that the FFI bridge is wired up and the dynamic linker can find the
/// extension module.
#[pyfunction]
fn version() -> &'static str {
    vernier_core::VERSION
}

/// Stage-0 instrumentation hook for the bbox-IoU optimization plan.
///
/// Writes the `(kind, g, d, wall_ns)` records accumulated across every
/// `BboxIou::compute` and `BboxIou::compute_overlap_mask` call to
/// `path` as CSV (header `kind,g,d,wall_ns`; `kind` is the variant
/// label `FullIou` or `OverlapMask`), then clears the in-process
/// buffer. Returns the number of records written.
///
/// Only present when the FFI crate is compiled with `--features
/// bench-histogram`. Bench harness builds set this; the shipped wheel
/// never does, so production runs carry zero recording overhead.
#[cfg(feature = "bench-histogram")]
#[pyfunction]
fn dump_bbox_iou_histogram(path: &str) -> PyResult<usize> {
    vernier_core::dump_bbox_iou_histogram_csv(std::path::Path::new(path))
        .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))
}

/// Bench-timings hook: `(par_iter_ns, serial_post_ns, n_calls)` since last read.
#[cfg(feature = "bench-timings")]
#[pyfunction]
fn read_and_reset_evaluate_parallel_timings() -> (u64, u64, u64) {
    vernier_core::read_and_reset_evaluate_parallel_timings()
}

/// Bench-timings hook: `build_*_anns` call count since last read.
#[cfg(feature = "bench-timings")]
#[pyfunction]
fn read_and_reset_build_anns_count() -> u64 {
    vernier_core::read_and_reset_build_anns_count()
}

/// Bench-timings hook: `(gt_parse_ns, gt_from_parts_ns, dt_parse_ns, dt_from_inputs_ns)` since last read.
#[cfg(feature = "bench-timings")]
#[pyfunction]
fn read_and_reset_dataset_timings() -> (u64, u64, u64, u64) {
    vernier_core::read_and_reset_dataset_timings()
}

/// Pythonic view over a [`vernier_core::Summary`]. Frozen — the underlying
/// value is constructed once by [`evaluate_bbox_summary`] and never
/// mutated (per ADR-0006).
#[pyclass(module = "vernier._core", name = "Summary", frozen)]
pub(crate) struct PySummary {
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

impl PySummary {
    /// Construct from a borrowed [`Summary`] for crate-internal use
    /// (the partition orchestrator wraps the `overall` summary on its
    /// outbound path).
    pub(crate) fn new(inner: Summary) -> Self {
        Self { inner }
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
    /// Detections that produced `inner.eval_imgs`, retained when the
    /// grid was built with `retain_iou=true` so the per_detection table
    /// builder can read them without re-parsing `dt`. `None` when
    /// retention is off (per_image / per_class only); reading on the
    /// wrong path raises a typed error from
    /// [`per_detection_to_arrow_pycapsule`].
    retained_dt: Option<CocoDetections>,
    /// Dataset snapshot the grid was built against. Three `Arc` clones
    /// of `(gt, boundary_cache, segm_cache)`; reused by [`Self::dataset`]
    /// so the result-tables Python wrapper doesn't re-parse GT JSON.
    retained_dataset: dataset::DatasetSnapshot,
    /// Resolved IoU ladder this grid was built with (ADR-0040). Reused
    /// by [`Self::accumulate`] and the per-axis tables so T-axis
    /// indexing stays aligned with what the matcher saw.
    iou_thresholds: Vec<f64>,
    /// Resolved recall ladder, threaded into
    /// [`AccumulateParams::recall_thresholds`] by [`Self::accumulate`].
    recall_thresholds: Vec<f64>,
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

    /// Dataset handle the grid was built against. Returns the snapshot
    /// captured at construction time — no re-parsing. For result-table
    /// builders (`per_image`, `per_class`) that previously had to call
    /// `Dataset.from_json(gt)` again on the tables= path.
    fn dataset(&self) -> dataset::PyDataset {
        dataset::PyDataset::from_snapshot(self.retained_dataset.clone())
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
        // Quirk A2 (strict): mirror pycocotools' `cocoeval.py:137`
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
        let iou_thr = self.iou_thresholds.clone();
        let recall_thr = self.recall_thresholds.clone();
        let acc = py
            .detach(|| {
                accumulate(
                    eval_imgs,
                    AccumulateParams {
                        iou_thresholds: &iou_thr,
                        recall_thresholds: &recall_thr,
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
            iou_thresholds: iou_thr,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "EvalGrid(n_categories={}, n_area_ranges={}, n_images={})",
            self.inner.n_categories, self.inner.n_area_ranges, self.inner.n_images
        )
    }
}

impl PyEvalGrid {
    /// Crate-internal borrow used by the result-tables FFI module.
    pub(crate) fn eval_grid_ref(&self) -> &EvalGrid {
        &self.inner
    }

    /// Detections retained at grid-construction time. `Some` when the
    /// grid was built with `retain_iou=true` (which the per_detection
    /// table requires); `None` otherwise.
    pub(crate) fn retained_dt(&self) -> Option<&CocoDetections> {
        self.retained_dt.as_ref()
    }

    /// Resolved IoU threshold ladder this grid was built with (ADR-0040).
    /// Used by the result-tables module so its T-axis indexing matches
    /// the matcher's; reads as the kernel-canonical ladder when no
    /// custom value was supplied at evaluate-time.
    pub(crate) fn iou_thresholds(&self) -> &[f64] {
        &self.iou_thresholds
    }

    /// Owned copy of the resolved IoU threshold ladder, for callers
    /// that need to move it into a `py.detach` closure (the partition
    /// orchestrator does).
    pub(crate) fn iou_thresholds_vec(&self) -> Vec<f64> {
        self.iou_thresholds.clone()
    }

    /// Three-`Arc` snapshot of the dataset the grid was built against.
    /// Cheap clone; consumed by the partition orchestrator to look up
    /// the image-id → flat-index map without re-parsing GT.
    pub(crate) fn dataset_snapshot(&self) -> dataset::DatasetSnapshot {
        self.retained_dataset.clone()
    }

    /// Parity-mode this grid was built under. Threaded by
    /// [`calibration::cells_from_grid`] into the [`calibration::EvalCells`]
    /// handle so the calibration summarizer receives the same flag the
    /// matcher saw (ADR-0018 Unit 2).
    pub(crate) fn parity_mode(&self) -> ParityMode {
        self.parity
    }
}

/// Frozen wrapper around [`vernier_core::Accumulated`]. Carries the
/// `max_dets` ladder used by `accumulate`; `summarize` reuses it.
#[pyclass(module = "vernier._core", name = "Accumulated", frozen)]
struct PyAccumulated {
    inner: Accumulated,
    max_dets: Vec<usize>,
    /// Resolved IoU threshold ladder propagated from the source
    /// [`PyEvalGrid`] (ADR-0040). [`Self::summarize`] indexes into the
    /// 5-D `precision` tensor along this ladder; pairing it with the
    /// canonical ladder when the grid was built on a custom one would
    /// silently mis-slice.
    iou_thresholds: Vec<f64>,
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
        // Quirk A2 (strict): the accumulator was built with a sorted
        // ladder; an explicit override here must follow the same
        // contract or the M-axis lookups in the summarizer would
        // silently misalign. `self.max_dets` is already sorted (set by
        // `PyEvalGrid::accumulate`), so the unwrap_or branch is a no-op.
        sort_max_dets(&mut dets);
        let acc = &self.inner;
        let iou_thr = self.iou_thresholds.clone();
        let summary = py
            .detach(|| match plan {
                SummarizePlan::Detection => summarize_detection(acc, &iou_thr, &dets),
                SummarizePlan::Keypoints => {
                    summarize_with(acc, &StatRequest::coco_keypoints_default(), &iou_thr, &dets)
                }
            })
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;
        Ok(PySummary { inner: summary })
    }

    /// Summarize this accumulator with the canonical LVIS 13-entry
    /// plan (ADR-0026 AF1, AF4). Routes the K-axis Frequency filter
    /// through the dataset's federated metadata
    /// (`category_frequency`); a non-federated `Dataset` yields `-1`
    /// on every `AP_r`/`AP_c`/`AP_f` entry (quirk **AB6**) and
    /// vernier-rate AP/AR on the others.
    ///
    /// `max_dets` defaults to the ladder this accumulator was built
    /// with; pass `[300]` (or pair with an `evaluate_*_grid` call
    /// using `max_dets_per_image=300`) for byte-identity with
    /// `LVISEval`.
    #[pyo3(signature = (gt, max_dets=None))]
    fn summarize_lvis(
        &self,
        py: Python<'_>,
        gt: &PyDataset,
        max_dets: Option<Vec<usize>>,
    ) -> PyResult<PySummary> {
        let mut dets = max_dets.unwrap_or_else(|| self.max_dets.clone());
        require_nonempty_max_dets(&dets)?;
        sort_max_dets(&mut dets);
        let acc = &self.inner;
        let iou_thr = self.iou_thresholds.clone();
        let dataset = gt.dataset_ref();
        let summary = py
            .detach(move || -> Result<vernier_core::Summary, EvalError> {
                let mut category_ids: Vec<CategoryId> =
                    dataset.categories().iter().map(|c| c.id).collect();
                category_ids.sort_unstable_by_key(|c| c.0);
                summarize_with_lvis(
                    acc,
                    &StatRequest::lvis_default(),
                    &iou_thr,
                    &dets,
                    &category_ids,
                    dataset.category_frequency(),
                )
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

impl PyAccumulated {
    /// Crate-internal borrow used by the result-tables FFI module.
    pub(crate) fn accumulated_ref(&self) -> &Accumulated {
        &self.inner
    }
    /// Crate-internal borrow of the M-axis ladder this accumulator was
    /// built with. Result tables that cite specific maxDets entries
    /// look these up.
    pub(crate) fn max_dets_slice(&self) -> &[usize] {
        &self.max_dets
    }
}

/// Parse a COCO GT payload, lifting `EvalError` into `PyValueError`.
/// One helper per FFI entry — keeps the error mapping uniform.
pub(crate) fn parse_gt(bytes: &[u8]) -> PyResult<CocoDataset> {
    CocoDataset::from_json_bytes(bytes).map_err(coco_load_error_to_pyerr)
}

/// Parse a COCO detections payload (sibling of [`parse_gt`]).
pub(crate) fn parse_dt(bytes: &[u8]) -> PyResult<CocoDetections> {
    CocoDetections::from_json_bytes(bytes).map_err(coco_load_error_to_pyerr)
}

/// GIL-free subset of [`eval_error_to_pyerr`] for the variants that can
/// fire during JSON loading. The Partial* / OutOfBudget arms need a
/// `Python<'_>` token to attach attributes and are unreachable here, so
/// they collapse into the generic `PyValueError` fallback.
fn coco_load_error_to_pyerr(e: EvalError) -> PyErr {
    match e {
        EvalError::InvalidAnnotation { .. } => InvalidAnnotationError::new_err(format!("{e}")),
        EvalError::NonFinite { .. } => NonFiniteError::new_err(format!("{e}")),
        EvalError::DimensionMismatch { .. } => DimensionMismatchError::new_err(format!("{e}")),
        EvalError::InvalidConfig { .. } => InvalidConfigError::new_err(format!("{e}")),
        other => PyValueError::new_err(format!("{other}")),
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
pub(crate) enum EvalIouType {
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

    /// `run` against the dataset's GT-side derivation caches
    /// (ADR-0020). Mirrors [`Self::run`] but routes the kernels that
    /// have a cache slot through their `*_cached` variants.
    fn run_cached(
        &self,
        gt: &CocoDataset,
        dt: &CocoDetections,
        params: EvaluateParams<'_>,
        parity: ParityMode,
        caches: DatasetCaches<'_>,
    ) -> Result<EvalGrid, EvalError> {
        match self {
            Self::Bbox => evaluate_bbox(gt, dt, params, parity),
            Self::Segm => evaluate_segm_cached(gt, dt, params, parity, caches.segm),
            Self::Boundary { dilation_ratio } => {
                evaluate_boundary_cached(gt, dt, params, parity, *dilation_ratio, caches.boundary)
            }
            Self::Keypoints { sigmas } => {
                evaluate_keypoints(gt, dt, params, parity, sigmas.clone())
            }
        }
    }

    /// Parallel sibling of [`Self::run`] (ADR-0047). Caller is
    /// responsible for installing a `rayon::ThreadPool` of the
    /// requested size via `pool.install(|| ...)` — this method uses
    /// the parallel evaluator entries which read the ambient pool.
    ///
    /// # Errors
    ///
    /// Propagates [`EvalError`] from the underlying kernel calls.
    fn run_parallel(
        &self,
        gt: &CocoDataset,
        dt: &CocoDetections,
        params: EvaluateParams<'_>,
        parity: ParityMode,
    ) -> Result<EvalGrid, EvalError> {
        use vernier_core::{
            evaluate_bbox_parallel, evaluate_boundary_parallel, evaluate_keypoints_parallel,
            evaluate_segm_parallel,
        };
        match self {
            Self::Bbox => evaluate_bbox_parallel(gt, dt, params, parity),
            Self::Segm => evaluate_segm_parallel(gt, dt, params, parity),
            Self::Boundary { dilation_ratio } => {
                evaluate_boundary_parallel(gt, dt, params, parity, *dilation_ratio)
            }
            Self::Keypoints { sigmas } => {
                evaluate_keypoints_parallel(gt, dt, params, parity, sigmas.clone())
            }
        }
    }

    /// Parallel sibling of [`Self::run_cached`]. Same pool ownership
    /// contract as [`Self::run_parallel`].
    ///
    /// # Errors
    ///
    /// Propagates [`EvalError`] from the underlying kernel calls.
    fn run_cached_parallel(
        &self,
        gt: &CocoDataset,
        dt: &CocoDetections,
        params: EvaluateParams<'_>,
        parity: ParityMode,
        caches: DatasetCaches<'_>,
    ) -> Result<EvalGrid, EvalError> {
        use vernier_core::{
            evaluate_bbox_parallel, evaluate_boundary_cached_parallel, evaluate_keypoints_parallel,
            evaluate_segm_cached_parallel,
        };
        match self {
            Self::Bbox => evaluate_bbox_parallel(gt, dt, params, parity),
            Self::Segm => evaluate_segm_cached_parallel(gt, dt, params, parity, caches.segm),
            Self::Boundary { dilation_ratio } => evaluate_boundary_cached_parallel(
                gt,
                dt,
                params,
                parity,
                *dilation_ratio,
                caches.boundary,
            ),
            Self::Keypoints { sigmas } => {
                evaluate_keypoints_parallel(gt, dt, params, parity, sigmas.clone())
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
///
/// `num_threads` (ADR-0047) routes between the sequential
/// [`EvalIouType::run`] path (`None` / `Some(1)` / unset env var) and
/// the parallel [`EvalIouType::run_parallel`] path. The parallel arm
/// builds a scoped per-call `rayon::ThreadPool` of exactly the
/// requested thread count.
#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_grid_impl(
    py: Python<'_>,
    iou_type: EvalIouType,
    gt_json: &Bound<'_, PyBytes>,
    dt: &Bound<'_, PyAny>,
    parity_mode: &str,
    max_dets_per_image: usize,
    use_cats: bool,
    retain_iou: bool,
    cast_inputs: bool,
    iou_thresholds_arg: Option<Vec<f64>>,
    recall_thresholds_arg: Option<Vec<f64>>,
    area_ranges_arg: Option<&Bound<'_, breakdown::PyBreakdown>>,
    num_threads: Option<usize>,
) -> PyResult<PyEvalGrid> {
    let parity = parse_parity_mode(parity_mode)?;
    let (iou_thr, recall_thr, area) = resolve_grid_axes(
        &iou_type,
        iou_thresholds_arg,
        recall_thresholds_arg,
        area_ranges_arg,
    )?;
    // Resolve threading policy under the GIL so the env-var / re-entry
    // `UserWarning` can fire to Python before we detach.
    let thread_policy = threads::resolve_threads(py, num_threads);
    // Zero-copy borrow over the GT bytes — `PyBackedBytes` keeps the
    // underlying `Py<PyBytes>` alive across `py.detach` while exposing
    // `&[u8]` via `Deref`. Saves a 20 MB `to_vec()` per call on val2017
    // and is `Send + Sync` so the buffer crosses the GIL release safely
    // (the underlying object is Python-immutable and refcount-pinned).
    let gt_bytes = pyo3::pybacked::PyBackedBytes::from(gt_json.clone());
    let dt_payload = prepare_dt_payload(py, dt, &iou_type, cast_inputs)?;
    type GridParts = (EvalGrid, Option<CocoDetections>, dataset::DatasetSnapshot);
    let iou_for_run = iou_thr.clone();
    let (grid, retained_dt, retained_dataset) = py.detach(move || -> PyResult<GridParts> {
        let gt = parse_gt(&gt_bytes)?;
        let dt = realize_dt(dt_payload)?;
        let params = EvaluateParams {
            iou_thresholds: &iou_for_run,
            area_ranges: &area,
            max_dets_per_image,
            use_cats,
            retain_iou,
        };
        let grid = run_grid_with_policy(&iou_type, &gt, &dt, params, parity, thread_policy)
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;
        Ok((
            grid,
            retain_iou.then_some(dt),
            dataset::DatasetSnapshot::from_parsed(gt),
        ))
    })?;
    Ok(PyEvalGrid {
        inner: grid,
        parity,
        retained_dt,
        retained_dataset,
        iou_thresholds: iou_thr,
        recall_thresholds: recall_thr,
    })
}

/// Dispatch the right per-paradigm runner based on
/// [`threads::ThreadPolicy`]: sequential calls today's
/// [`EvalIouType::run`] unchanged; parallel builds a scoped pool of
/// exactly the requested thread count and `install`s the parallel
/// runner inside it.
fn run_grid_with_policy(
    iou_type: &EvalIouType,
    gt: &CocoDataset,
    dt: &CocoDetections,
    params: EvaluateParams<'_>,
    parity: ParityMode,
    thread_policy: threads::ThreadPolicy,
) -> Result<EvalGrid, EvalError> {
    match thread_policy.thread_count() {
        None => iou_type.run(gt, dt, params, parity),
        Some(n) => {
            let pool = threads::build_scoped_pool(n)
                .map_err(|detail| EvalError::InvalidConfig { detail })?;
            pool.install(|| iou_type.run_parallel(gt, dt, params, parity))
        }
    }
}

/// Cached-dataset sibling of [`run_grid_with_policy`].
fn run_grid_cached_with_policy(
    iou_type: &EvalIouType,
    gt: &CocoDataset,
    dt: &CocoDetections,
    params: EvaluateParams<'_>,
    parity: ParityMode,
    caches: DatasetCaches<'_>,
    thread_policy: threads::ThreadPolicy,
) -> Result<EvalGrid, EvalError> {
    match thread_policy.thread_count() {
        None => iou_type.run_cached(gt, dt, params, parity, caches),
        Some(n) => {
            let pool = threads::build_scoped_pool(n)
                .map_err(|detail| EvalError::InvalidConfig { detail })?;
            pool.install(|| iou_type.run_cached_parallel(gt, dt, params, parity, caches))
        }
    }
}

/// Same as [`evaluate_grid_impl`] but accepts a parsed-once
/// [`PyDataset`] (ADR-0020) — required when the GT carries LVIS
/// federated metadata that the JSON-bytes path would discard
/// (ADR-0026, the orchestrator's `gt.is_federated()` gate fires
/// only on a dataset built via `Dataset.from_lvis_json`).
///
/// `num_threads` (ADR-0047) has identical semantics to the
/// JSON-bytes path; see [`evaluate_grid_impl`].
#[allow(clippy::too_many_arguments)]
fn evaluate_grid_with_dataset_impl(
    py: Python<'_>,
    iou_type: EvalIouType,
    gt: &PyDataset,
    dt: &Bound<'_, PyAny>,
    parity_mode: &str,
    max_dets_per_image: usize,
    use_cats: bool,
    retain_iou: bool,
    cast_inputs: bool,
    iou_thresholds_arg: Option<Vec<f64>>,
    recall_thresholds_arg: Option<Vec<f64>>,
    area_ranges_arg: Option<&Bound<'_, breakdown::PyBreakdown>>,
    num_threads: Option<usize>,
) -> PyResult<PyEvalGrid> {
    let parity = parse_parity_mode(parity_mode)?;
    let (iou_thr, recall_thr, area) = resolve_grid_axes(
        &iou_type,
        iou_thresholds_arg,
        recall_thresholds_arg,
        area_ranges_arg,
    )?;
    let thread_policy = threads::resolve_threads(py, num_threads);
    let snapshot = gt.snapshot();
    let retained_dataset = snapshot.clone();
    let dt_payload = prepare_dt_payload(py, dt, &iou_type, cast_inputs)?;
    let iou_for_run = iou_thr.clone();
    let (grid, retained_dt) =
        py.detach(move || -> PyResult<(EvalGrid, Option<CocoDetections>)> {
            let dt = realize_dt(dt_payload)?;
            // ADR-0026 AC2: federated datasets trim DTs at input time
            // (mirrors `LVISResults.limit_dets_per_image` at construction).
            // The trim is a no-op when fewer than `max_dets_per_image`
            // DTs land on any single image, and is disabled with a
            // negative cap (AC5).
            let dt = if snapshot.gt.is_federated() {
                #[allow(clippy::cast_possible_wrap)]
                let cap = max_dets_per_image as i64;
                dt.lvis_trim(cap)
            } else {
                dt
            };
            let caches = snapshot.caches();
            let params = EvaluateParams {
                iou_thresholds: &iou_for_run,
                area_ranges: &area,
                max_dets_per_image,
                use_cats,
                retain_iou,
            };
            let grid = run_grid_cached_with_policy(
                &iou_type,
                &snapshot.gt,
                &dt,
                params,
                parity,
                caches,
                thread_policy,
            )
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;
            Ok((grid, retain_iou.then_some(dt)))
        })?;
    Ok(PyEvalGrid {
        inner: grid,
        parity,
        retained_dt,
        retained_dataset,
        iou_thresholds: iou_thr,
        recall_thresholds: recall_thr,
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

/// Resolve the three optional ADR-0040 grid axes against the
/// kernel-canonical defaults. `None` falls back to the canonical
/// ladder / area grid; an explicit value is taken verbatim.
///
/// `area_ranges` is supplied as a [`PyBreakdown`] (a range Breakdown
/// validated by the Python wrapper), unpacked here via
/// [`vernier_core::breakdown::Breakdown::area_ranges`]. The class-groups
/// variant is rejected — instance area_ranges accepts only range
/// breakdowns per ADR-0040.
fn resolve_grid_axes(
    iou_type: &EvalIouType,
    iou_thresholds_arg: Option<Vec<f64>>,
    recall_thresholds_arg: Option<Vec<f64>>,
    area_ranges_arg: Option<&Bound<'_, breakdown::PyBreakdown>>,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<AreaRange>)> {
    let iou = iou_thresholds_arg.unwrap_or_else(|| iou_thresholds().to_vec());
    let recall = recall_thresholds_arg.unwrap_or_else(|| recall_thresholds().to_vec());
    let area = match area_ranges_arg {
        None => area_ranges_for(iou_type),
        Some(bd) => match &bd.borrow().inner {
            breakdown::BreakdownInner::Range(b) => b.area_ranges(),
            breakdown::BreakdownInner::ClassGroups(_) => {
                return Err(PyValueError::new_err(
                    "area_ranges must be a range Breakdown (Breakdown.from_ranges); \
                     class-groups breakdowns belong on semantic / panoptic class_grouping (ADR-0040)",
                ));
            }
        },
    };
    Ok((iou, recall, area))
}

/// Bbox per-image evaluation pass — see [`evaluate_grid_impl`].
/// `retain_iou` (per ADR-0019 Week 2.3) keeps the per-`(category,
/// image)` IoU matrix on the returned grid for later table
/// construction; defaults to `False` so existing callers pay no extra
/// allocation.
#[pyfunction]
#[pyo3(signature = (gt_json, dt, parity_mode, max_dets_per_image, use_cats, retain_iou=false, cast_inputs=false, iou_thresholds=None, recall_thresholds=None, area_ranges=None, num_threads=None))]
#[allow(clippy::too_many_arguments)]
fn evaluate_bbox_grid<'py>(
    py: Python<'py>,
    gt_json: &Bound<'py, PyBytes>,
    dt: &Bound<'py, PyAny>,
    parity_mode: &str,
    max_dets_per_image: usize,
    use_cats: bool,
    retain_iou: bool,
    cast_inputs: bool,
    iou_thresholds: Option<Vec<f64>>,
    recall_thresholds: Option<Vec<f64>>,
    area_ranges: Option<&Bound<'py, breakdown::PyBreakdown>>,
    num_threads: Option<usize>,
) -> PyResult<PyEvalGrid> {
    evaluate_grid_impl(
        py,
        EvalIouType::Bbox,
        gt_json,
        dt,
        parity_mode,
        max_dets_per_image,
        use_cats,
        retain_iou,
        cast_inputs,
        iou_thresholds,
        recall_thresholds,
        area_ranges,
        num_threads,
    )
}

/// Bbox per-image evaluation pass against a parsed-once
/// [`Dataset`]. The federated form is the entry point the LVIS
/// parity harness consumes: the JSON-bytes [`evaluate_bbox_grid`]
/// strips ADR-0026 federated metadata at GT load, so the
/// orchestrator's AA3/AA4 branches never fire on that path.
#[pyfunction]
#[pyo3(signature = (gt, dt, parity_mode, max_dets_per_image, use_cats, retain_iou=false, cast_inputs=false, iou_thresholds=None, recall_thresholds=None, area_ranges=None, num_threads=None))]
#[allow(clippy::too_many_arguments)]
fn evaluate_bbox_grid_with_dataset<'py>(
    py: Python<'py>,
    gt: &PyDataset,
    dt: &Bound<'py, PyAny>,
    parity_mode: &str,
    max_dets_per_image: usize,
    use_cats: bool,
    retain_iou: bool,
    cast_inputs: bool,
    iou_thresholds: Option<Vec<f64>>,
    recall_thresholds: Option<Vec<f64>>,
    area_ranges: Option<&Bound<'py, breakdown::PyBreakdown>>,
    num_threads: Option<usize>,
) -> PyResult<PyEvalGrid> {
    evaluate_grid_with_dataset_impl(
        py,
        EvalIouType::Bbox,
        gt,
        dt,
        parity_mode,
        max_dets_per_image,
        use_cats,
        retain_iou,
        cast_inputs,
        iou_thresholds,
        recall_thresholds,
        area_ranges,
        num_threads,
    )
}

/// Segm per-image evaluation pass. Both GT and DT JSON must carry a
/// `segmentation` field on every entry; absent fields raise a typed
/// `ValueError` instead of being silently treated as empty.
#[pyfunction]
#[pyo3(signature = (gt_json, dt, parity_mode, max_dets_per_image, use_cats, retain_iou=false, cast_inputs=false, iou_thresholds=None, recall_thresholds=None, area_ranges=None, num_threads=None))]
#[allow(clippy::too_many_arguments)]
fn evaluate_segm_grid<'py>(
    py: Python<'py>,
    gt_json: &Bound<'py, PyBytes>,
    dt: &Bound<'py, PyAny>,
    parity_mode: &str,
    max_dets_per_image: usize,
    use_cats: bool,
    retain_iou: bool,
    cast_inputs: bool,
    iou_thresholds: Option<Vec<f64>>,
    recall_thresholds: Option<Vec<f64>>,
    area_ranges: Option<&Bound<'py, breakdown::PyBreakdown>>,
    num_threads: Option<usize>,
) -> PyResult<PyEvalGrid> {
    evaluate_grid_impl(
        py,
        EvalIouType::Segm,
        gt_json,
        dt,
        parity_mode,
        max_dets_per_image,
        use_cats,
        retain_iou,
        cast_inputs,
        iou_thresholds,
        recall_thresholds,
        area_ranges,
        num_threads,
    )
}

/// Boundary-IoU per-image evaluation pass (ADR-0010). Same
/// segmentation-field requirements as [`evaluate_segm_grid`].
/// `dilation_ratio` is the boundary band width as a fraction of the
/// image diagonal (`0.02` COCO default; `0.008` LVIS variant).
#[pyfunction]
#[pyo3(signature = (gt_json, dt, parity_mode, max_dets_per_image, use_cats, dilation_ratio, retain_iou=false, cast_inputs=false, iou_thresholds=None, recall_thresholds=None, area_ranges=None, num_threads=None))]
#[allow(clippy::too_many_arguments)]
fn evaluate_boundary_grid<'py>(
    py: Python<'py>,
    gt_json: &Bound<'py, PyBytes>,
    dt: &Bound<'py, PyAny>,
    parity_mode: &str,
    max_dets_per_image: usize,
    use_cats: bool,
    dilation_ratio: f64,
    retain_iou: bool,
    cast_inputs: bool,
    iou_thresholds: Option<Vec<f64>>,
    recall_thresholds: Option<Vec<f64>>,
    area_ranges: Option<&Bound<'py, breakdown::PyBreakdown>>,
    num_threads: Option<usize>,
) -> PyResult<PyEvalGrid> {
    let iou_type = boundary_iou_type(dilation_ratio)?;
    evaluate_grid_impl(
        py,
        iou_type,
        gt_json,
        dt,
        parity_mode,
        max_dets_per_image,
        use_cats,
        retain_iou,
        cast_inputs,
        iou_thresholds,
        recall_thresholds,
        area_ranges,
        num_threads,
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
/// pycocotools' `useCats` (quirk **L4**). `num_threads` (ADR-0047)
/// routes between the sequential and parallel paths; see
/// [`evaluate_grid_impl`].
#[allow(clippy::too_many_arguments)]
fn evaluate_summary_impl(
    py: Python<'_>,
    iou_type: EvalIouType,
    gt_json: &Bound<'_, PyBytes>,
    dt: &Bound<'_, PyAny>,
    parity_mode: &str,
    max_dets: Vec<usize>,
    use_cats: bool,
    cast_inputs: bool,
    num_threads: Option<usize>,
) -> PyResult<PySummary> {
    let parity = parse_parity_mode(parity_mode)?;
    require_nonempty_max_dets(&max_dets)?;
    let thread_policy = threads::resolve_threads(py, num_threads);
    // Quirk A2 (strict): mirror pycocotools' `cocoeval.py:137`
    // `p.maxDets = sorted(p.maxDets)`. Sort once here so the eval
    // pipeline's `max_dets_per_image` cap (the largest entry) and the
    // summarizer's positional `AR_*` lookups both see the canonical
    // ascending ladder.
    let mut max_dets = max_dets;
    sort_max_dets(&mut max_dets);
    // The `PyBytes` borrow is GIL-tied; copy so the JSON parse can run
    // inside `py.detach`. One memcpy per call buys a wider GIL-drop
    // window for multi-threaded callers.
    let gt_bytes = gt_json.as_bytes().to_vec();
    let dt_payload = prepare_dt_payload(py, dt, &iou_type, cast_inputs)?;

    let summary = py.detach(move || -> PyResult<Summary> {
        let gt = parse_gt(&gt_bytes)?;
        let dt = realize_dt(dt_payload)?;
        run_pipeline(
            &iou_type,
            &gt,
            &dt,
            parity,
            &max_dets,
            use_cats,
            thread_policy,
        )
        .map_err(|e| PyValueError::new_err(format!("{e}")))
    })?;

    Ok(PySummary { inner: summary })
}

/// Bbox end-to-end pipeline — see [`evaluate_summary_impl`].
#[pyfunction]
#[pyo3(signature = (gt_json, dt, parity_mode, max_dets, use_cats, cast_inputs=false, num_threads=None))]
#[allow(clippy::too_many_arguments)]
fn evaluate_bbox_summary(
    py: Python<'_>,
    gt_json: &Bound<'_, PyBytes>,
    dt: &Bound<'_, PyAny>,
    parity_mode: &str,
    max_dets: Vec<usize>,
    use_cats: bool,
    cast_inputs: bool,
    num_threads: Option<usize>,
) -> PyResult<PySummary> {
    evaluate_summary_impl(
        py,
        EvalIouType::Bbox,
        gt_json,
        dt,
        parity_mode,
        max_dets,
        use_cats,
        cast_inputs,
        num_threads,
    )
}

/// Segm end-to-end pipeline — see [`evaluate_summary_impl`]. Both GT
/// and DT must carry segmentation fields.
#[pyfunction]
#[pyo3(signature = (gt_json, dt, parity_mode, max_dets, use_cats, cast_inputs=false, num_threads=None))]
#[allow(clippy::too_many_arguments)]
fn evaluate_segm_summary(
    py: Python<'_>,
    gt_json: &Bound<'_, PyBytes>,
    dt: &Bound<'_, PyAny>,
    parity_mode: &str,
    max_dets: Vec<usize>,
    use_cats: bool,
    cast_inputs: bool,
    num_threads: Option<usize>,
) -> PyResult<PySummary> {
    evaluate_summary_impl(
        py,
        EvalIouType::Segm,
        gt_json,
        dt,
        parity_mode,
        max_dets,
        use_cats,
        cast_inputs,
        num_threads,
    )
}

/// Boundary end-to-end pipeline (ADR-0010) — see
/// [`evaluate_summary_impl`]. Both GT and DT must carry segmentation
/// fields. `dilation_ratio` matches [`evaluate_boundary_grid`].
#[pyfunction]
#[pyo3(signature = (gt_json, dt, parity_mode, max_dets, use_cats, dilation_ratio, cast_inputs=false, num_threads=None))]
#[allow(clippy::too_many_arguments)]
fn evaluate_boundary_summary(
    py: Python<'_>,
    gt_json: &Bound<'_, PyBytes>,
    dt: &Bound<'_, PyAny>,
    parity_mode: &str,
    max_dets: Vec<usize>,
    use_cats: bool,
    dilation_ratio: f64,
    cast_inputs: bool,
    num_threads: Option<usize>,
) -> PyResult<PySummary> {
    let iou_type = boundary_iou_type(dilation_ratio)?;
    evaluate_summary_impl(
        py,
        iou_type,
        gt_json,
        dt,
        parity_mode,
        max_dets,
        use_cats,
        cast_inputs,
        num_threads,
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
#[pyo3(signature = (gt_json, dt, parity_mode, max_dets, use_cats, sigmas, cast_inputs=false, num_threads=None))]
#[allow(clippy::too_many_arguments)]
fn evaluate_keypoints_summary(
    py: Python<'_>,
    gt_json: &Bound<'_, PyBytes>,
    dt: &Bound<'_, PyAny>,
    parity_mode: &str,
    max_dets: Vec<usize>,
    use_cats: bool,
    sigmas: &Bound<'_, PyDict>,
    cast_inputs: bool,
    num_threads: Option<usize>,
) -> PyResult<PySummary> {
    let iou_type = EvalIouType::Keypoints {
        sigmas: parse_sigmas(sigmas)?,
    };
    evaluate_summary_impl(
        py,
        iou_type,
        gt_json,
        dt,
        parity_mode,
        max_dets,
        use_cats,
        cast_inputs,
        num_threads,
    )
}

/// Bbox end-to-end pipeline against a parsed-once [`PyDataset`]
/// (ADR-0020). Reuses the dataset's parsed GT; bbox has no GT-side
/// derivation cache today, so the only saving over
/// [`evaluate_bbox_summary`] is the GT JSON parse.
#[pyfunction]
#[pyo3(signature = (dataset, dt, parity_mode, max_dets, use_cats, cast_inputs=false, num_threads=None))]
#[allow(clippy::too_many_arguments)]
fn evaluate_bbox_summary_with_dataset(
    py: Python<'_>,
    dataset: &PyDataset,
    dt: &Bound<'_, PyAny>,
    parity_mode: &str,
    max_dets: Vec<usize>,
    use_cats: bool,
    cast_inputs: bool,
    num_threads: Option<usize>,
) -> PyResult<PySummary> {
    evaluate_summary_with_dataset_impl(
        py,
        EvalIouType::Bbox,
        dataset,
        dt,
        parity_mode,
        max_dets,
        use_cats,
        cast_inputs,
        num_threads,
    )
}

/// Segm end-to-end pipeline against a parsed-once [`PyDataset`]
/// (ADR-0020). Threads the dataset's [`SegmGtCache`] into the
/// kernel so cross-call GT bbox+area derivation is reused.
#[pyfunction]
#[pyo3(signature = (dataset, dt, parity_mode, max_dets, use_cats, cast_inputs=false, num_threads=None))]
#[allow(clippy::too_many_arguments)]
fn evaluate_segm_summary_with_dataset(
    py: Python<'_>,
    dataset: &PyDataset,
    dt: &Bound<'_, PyAny>,
    parity_mode: &str,
    max_dets: Vec<usize>,
    use_cats: bool,
    cast_inputs: bool,
    num_threads: Option<usize>,
) -> PyResult<PySummary> {
    evaluate_summary_with_dataset_impl(
        py,
        EvalIouType::Segm,
        dataset,
        dt,
        parity_mode,
        max_dets,
        use_cats,
        cast_inputs,
        num_threads,
    )
}

/// Boundary end-to-end pipeline against a parsed-once [`PyDataset`]
/// (ADR-0020). Threads the dataset's [`BoundaryGtCache`] into the
/// kernel so cross-call GT band derivation (the dominant boundary
/// cost) is reused. The cache is cleared if `dilation_ratio` differs
/// from the previous call's, per ADR-0010.
#[pyfunction]
#[pyo3(signature = (dataset, dt, parity_mode, max_dets, use_cats, dilation_ratio, cast_inputs=false, num_threads=None))]
#[allow(clippy::too_many_arguments)]
fn evaluate_boundary_summary_with_dataset(
    py: Python<'_>,
    dataset: &PyDataset,
    dt: &Bound<'_, PyAny>,
    parity_mode: &str,
    max_dets: Vec<usize>,
    use_cats: bool,
    dilation_ratio: f64,
    cast_inputs: bool,
    num_threads: Option<usize>,
) -> PyResult<PySummary> {
    let iou_type = boundary_iou_type(dilation_ratio)?;
    evaluate_summary_with_dataset_impl(
        py,
        iou_type,
        dataset,
        dt,
        parity_mode,
        max_dets,
        use_cats,
        cast_inputs,
        num_threads,
    )
}

/// Keypoints (OKS) end-to-end pipeline against a parsed-once
/// [`PyDataset`] (ADR-0020). No keypoints-side cache today, so the
/// saving over [`evaluate_keypoints_summary`] is the GT JSON parse.
#[pyfunction]
#[pyo3(signature = (dataset, dt, parity_mode, max_dets, use_cats, sigmas, cast_inputs=false, num_threads=None))]
#[allow(clippy::too_many_arguments)]
fn evaluate_keypoints_summary_with_dataset(
    py: Python<'_>,
    dataset: &PyDataset,
    dt: &Bound<'_, PyAny>,
    parity_mode: &str,
    max_dets: Vec<usize>,
    use_cats: bool,
    sigmas: &Bound<'_, PyDict>,
    cast_inputs: bool,
    num_threads: Option<usize>,
) -> PyResult<PySummary> {
    let iou_type = EvalIouType::Keypoints {
        sigmas: parse_sigmas(sigmas)?,
    };
    evaluate_summary_with_dataset_impl(
        py,
        iou_type,
        dataset,
        dt,
        parity_mode,
        max_dets,
        use_cats,
        cast_inputs,
        num_threads,
    )
}

/// Shared dispatch for the `evaluate_*_summary_with_dataset` family
/// (ADR-0020). Mirrors [`evaluate_summary_impl`] but skips GT parse
/// (the dataset already holds one) and threads the per-kernel cache
/// from `dataset` through [`run_pipeline_with_dataset`].
#[allow(clippy::too_many_arguments)]
fn evaluate_summary_with_dataset_impl(
    py: Python<'_>,
    iou_type: EvalIouType,
    dataset: &PyDataset,
    dt: &Bound<'_, PyAny>,
    parity_mode: &str,
    max_dets: Vec<usize>,
    use_cats: bool,
    cast_inputs: bool,
    num_threads: Option<usize>,
) -> PyResult<PySummary> {
    let parity = parse_parity_mode(parity_mode)?;
    require_nonempty_max_dets(&max_dets)?;
    let thread_policy = threads::resolve_threads(py, num_threads);
    let mut max_dets = max_dets;
    sort_max_dets(&mut max_dets);
    let dt_payload = prepare_dt_payload(py, dt, &iou_type, cast_inputs)?;
    let snapshot = dataset.snapshot();
    let summary = py.detach(move || -> PyResult<Summary> {
        let dt = realize_dt(dt_payload)?;
        run_pipeline_with_dataset(
            &iou_type,
            &snapshot.gt,
            snapshot.caches(),
            &dt,
            parity,
            &max_dets,
            use_cats,
            thread_policy,
        )
        .map_err(|e| PyValueError::new_err(format!("{e}")))
    })?;
    Ok(PySummary { inner: summary })
}

/// Keypoints per-image evaluation pass (ADR-0012). Both GT and DT must
/// carry `keypoints` fields. `sigmas` matches
/// [`evaluate_keypoints_summary`].
#[pyfunction]
#[pyo3(signature = (gt_json, dt, parity_mode, max_dets_per_image, use_cats, sigmas, cast_inputs=false, iou_thresholds=None, recall_thresholds=None, area_ranges=None, num_threads=None))]
#[allow(clippy::too_many_arguments)]
fn evaluate_keypoints_grid<'py>(
    py: Python<'py>,
    gt_json: &Bound<'py, PyBytes>,
    dt: &Bound<'py, PyAny>,
    parity_mode: &str,
    max_dets_per_image: usize,
    use_cats: bool,
    sigmas: &Bound<'py, PyDict>,
    cast_inputs: bool,
    iou_thresholds: Option<Vec<f64>>,
    recall_thresholds: Option<Vec<f64>>,
    area_ranges: Option<&Bound<'py, breakdown::PyBreakdown>>,
    num_threads: Option<usize>,
) -> PyResult<PyEvalGrid> {
    let iou_type = EvalIouType::Keypoints {
        sigmas: parse_sigmas(sigmas)?,
    };
    evaluate_grid_impl(
        py,
        iou_type,
        gt_json,
        dt,
        parity_mode,
        max_dets_per_image,
        use_cats,
        false,
        cast_inputs,
        iou_thresholds,
        recall_thresholds,
        area_ranges,
        num_threads,
    )
}

/// Decode a Python `dict[int, Sequence[float]]` into the
/// `HashMap<i64, Vec<f64>>` shape `OksSimilarity` consumes. Empty dict
/// is valid — `OksSimilarity` falls back to COCO-person sigmas.
pub(crate) fn parse_sigmas(d: &Bound<'_, PyDict>) -> PyResult<HashMap<i64, Vec<f64>>> {
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
    thread_policy: threads::ThreadPolicy,
) -> Result<Summary, EvalError> {
    let area = area_ranges_for(iou_type);
    let max_det_top = max_dets.iter().copied().max().unwrap_or(100);
    let eval_params = EvaluateParams {
        iou_thresholds: iou_thresholds(),
        area_ranges: &area,
        max_dets_per_image: max_det_top,
        use_cats,
        retain_iou: false,
    };
    let grid = run_grid_with_policy(iou_type, gt, dt, eval_params, parity, thread_policy)?;
    summarize_grid(&grid, iou_type.is_keypoints(), parity, max_dets)
}

/// End-to-end pipeline against a parsed-once dataset (ADR-0020).
/// Mirrors [`run_pipeline`] but routes through
/// [`EvalIouType::run_cached`] so kernels with a cache slot
/// (`evaluate_segm_cached`, `evaluate_boundary_cached`) reuse GT-side
/// derivations across calls.
#[allow(clippy::too_many_arguments)]
fn run_pipeline_with_dataset(
    iou_type: &EvalIouType,
    gt: &CocoDataset,
    caches: DatasetCaches<'_>,
    dt: &CocoDetections,
    parity: ParityMode,
    max_dets: &[usize],
    use_cats: bool,
    thread_policy: threads::ThreadPolicy,
) -> Result<Summary, EvalError> {
    let area = area_ranges_for(iou_type);
    let max_det_top = max_dets.iter().copied().max().unwrap_or(100);
    let eval_params = EvaluateParams {
        iou_thresholds: iou_thresholds(),
        area_ranges: &area,
        max_dets_per_image: max_det_top,
        use_cats,
        retain_iou: false,
    };
    let grid =
        run_grid_cached_with_policy(iou_type, gt, dt, eval_params, parity, caches, thread_policy)?;
    summarize_grid(&grid, iou_type.is_keypoints(), parity, max_dets)
}

/// Shared accumulate + summarize tail for both pipeline shapes.
fn summarize_grid(
    grid: &EvalGrid,
    is_keypoints: bool,
    parity: ParityMode,
    max_dets: &[usize],
) -> Result<Summary, EvalError> {
    let iou_thr = iou_thresholds();
    let acc_params = AccumulateParams {
        iou_thresholds: iou_thr,
        recall_thresholds: recall_thresholds(),
        max_dets,
        n_categories: grid.n_categories,
        n_area_ranges: grid.n_area_ranges,
        n_images: grid.n_images,
    };
    let acc = accumulate(&grid.eval_imgs, acc_params, parity)?;
    if is_keypoints {
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

/// Reject non-positive / non-finite `dilation_ratio` values at the FFI
/// boundary so the boundary kernel never sees a value its band-radius
/// math (`round(ratio * sqrt(h^2 + w^2))`) would silently degenerate
/// on. Used by every boundary entry point (`evaluate_boundary_*`,
/// `tide::error_decomposition_boundary`).
pub(crate) fn validate_dilation_ratio(dilation_ratio: f64) -> PyResult<()> {
    if !dilation_ratio.is_finite() || dilation_ratio <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "dilation_ratio must be a positive finite float, got {dilation_ratio}"
        )));
    }
    Ok(())
}

pub(crate) fn boundary_iou_type(dilation_ratio: f64) -> PyResult<EvalIouType> {
    validate_dilation_ratio(dilation_ratio)?;
    Ok(EvalIouType::Boundary { dilation_ratio })
}

/// Tag a freshly-built [`StreamingEvaluator`] with `rank_id` when the
/// caller provided one, leaving it untagged otherwise. ADR-0031: rank
/// identity is a construction-time property; vernier never mutates it
/// after the first `update`.
fn maybe_tag_rank<K: EvalKernel>(
    ev: StreamingEvaluator<K>,
    rank_id: Option<u32>,
) -> PyResult<StreamingEvaluator<K>> {
    match rank_id {
        Some(rid) => ev
            .with_rank(rid)
            .map_err(|e| PyValueError::new_err(format!("{e}"))),
        None => Ok(ev),
    }
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

pub(crate) fn parse_parity_mode(s: &str) -> PyResult<ParityMode> {
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
pub(crate) fn eval_error_to_pyerr(py: Python<'_>, e: EvalError) -> PyErr {
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
        EvalError::PartialFormatMismatch { ref kind } => {
            partial_format_to_pyerr(py, format!("{e}"), kind)
        }
        EvalError::PartialDatasetMismatch { expected, actual } => {
            partial_hash_mismatch_to_pyerr(py, format!("{e}"), &expected, &actual, true)
        }
        EvalError::PartialParamsMismatch { expected, actual } => {
            partial_hash_mismatch_to_pyerr(py, format!("{e}"), &expected, &actual, false)
        }
        EvalError::PartialPartitionOverlap {
            rank_a,
            rank_b,
            image_id,
        } => partial_partition_overlap_to_pyerr(py, format!("{e}"), rank_a, rank_b, image_id),
        EvalError::PartialRankCollision { rank_id } => {
            partial_rank_collision_to_pyerr(py, format!("{e}"), rank_id)
        }
        EvalError::InvalidAnnotation { .. } => InvalidAnnotationError::new_err(format!("{e}")),
        EvalError::NonFinite { .. } => NonFiniteError::new_err(format!("{e}")),
        EvalError::DimensionMismatch { .. } => DimensionMismatchError::new_err(format!("{e}")),
        EvalError::InvalidConfig { .. } => InvalidConfigError::new_err(format!("{e}")),
        other => PyValueError::new_err(format!("{other}")),
    }
}

/// Map a leaf [`vernier_partial::PartialError`] to its Python
/// exception class. Shared between [`eval_error_to_pyerr`] (instance)
/// and [`crate::semantic::semantic_error_to_pyerr`] (semantic) so a
/// caller can `try/except vernier.PartialFormatMismatch` regardless
/// of which paradigm produced the error.
pub(crate) fn partial_error_to_pyerr(py: Python<'_>, err: &vernier_partial::PartialError) -> PyErr {
    use vernier_partial::PartialError;
    let msg = format!("{err}");
    match err {
        PartialError::Format { kind } => partial_format_to_pyerr(py, msg, kind),
        PartialError::DatasetMismatch { expected, actual } => {
            partial_hash_mismatch_to_pyerr(py, msg, expected, actual, true)
        }
        PartialError::ParamsMismatch { expected, actual } => {
            partial_hash_mismatch_to_pyerr(py, msg, expected, actual, false)
        }
        PartialError::PartitionOverlap {
            rank_a,
            rank_b,
            image_id,
        } => partial_partition_overlap_to_pyerr(py, msg, *rank_a, *rank_b, *image_id),
        PartialError::RankCollision { rank_id } => {
            partial_rank_collision_to_pyerr(py, msg, *rank_id)
        }
    }
}

fn partial_format_to_pyerr(
    py: Python<'_>,
    msg: String,
    kind: &vernier_partial::PartialFormatErrorKind,
) -> PyErr {
    let exc = PartialFormatMismatch::new_err(msg);
    // `kind.tag()` is the canonical snake_case discriminant;
    // exhaustive over PartialFormatErrorKind so adding a variant
    // fails compilation rather than silently falling through.
    if let Err(err) = exc.value(py).setattr("kind", kind.tag()) {
        return err;
    }
    exc
}

fn partial_hash_mismatch_to_pyerr(
    py: Python<'_>,
    msg: String,
    expected: &[u8; 32],
    actual: &[u8; 32],
    is_dataset: bool,
) -> PyErr {
    let exc = if is_dataset {
        PartialDatasetMismatch::new_err(msg)
    } else {
        PartialParamsMismatch::new_err(msg)
    };
    let value = exc.value(py);
    if let Err(err) = value.setattr("expected", PyBytes::new(py, expected)) {
        return err;
    }
    if let Err(err) = value.setattr("actual", PyBytes::new(py, actual)) {
        return err;
    }
    exc
}

fn partial_partition_overlap_to_pyerr(
    py: Python<'_>,
    msg: String,
    rank_a: u32,
    rank_b: u32,
    image_id: i64,
) -> PyErr {
    let exc = PartialPartitionOverlap::new_err(msg);
    let value = exc.value(py);
    if let Err(err) = value.setattr("rank_a", rank_a) {
        return err;
    }
    if let Err(err) = value.setattr("rank_b", rank_b) {
        return err;
    }
    if let Err(err) = value.setattr("image_id", image_id) {
        return err;
    }
    exc
}

fn partial_rank_collision_to_pyerr(py: Python<'_>, msg: String, rank_id: u32) -> PyErr {
    let exc = PartialRankCollision::new_err(msg);
    if let Err(err) = exc.value(py).setattr("rank_id", rank_id) {
        return err;
    }
    exc
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

/// `warnings.warn(msg, W)` from Rust — used by the cast-promotion latch,
/// the soft-budget crossing, and the worker-scheduling path.
pub(crate) fn emit_warning<W: pyo3::type_object::PyTypeInfo>(
    py: Python<'_>,
    msg: &str,
) -> PyResult<()> {
    let warnings = py.import("warnings")?;
    let warn_class = py.get_type::<W>();
    warnings.getattr("warn")?.call1((msg, warn_class))?;
    Ok(())
}

impl StreamingState {
    /// Run an `update`. The JSON arm defers parsing to the kernel-typed
    /// `from_json_bytes`; the array arm calls `from_inputs` here so the
    /// HashMap build runs without the GIL (we're already inside
    /// `py.detach` on every call site).
    fn run_update(&mut self, payload: UpdatePayload) -> Result<UpdateReport, EvalError> {
        macro_rules! dispatch {
            ($ev:expr) => {
                match payload {
                    UpdatePayload::Bytes(b) => $ev.update(&b),
                    UpdatePayload::Inputs(inputs) => {
                        let detections = CocoDetections::from_inputs(inputs)?;
                        $ev.update_parsed(ParsedDetections::from_detections(detections))
                    }
                }
            };
        }
        match self {
            Self::Bbox(ev) => dispatch!(ev),
            Self::Segm(ev) => dispatch!(ev),
            Self::Boundary(ev) => dispatch!(ev),
            Self::Keypoints(ev) => dispatch!(ev),
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

    /// ADR-0031: serialize spine state to a partial blob and consume
    /// the evaluator (swap [`Self::Finalized`] in).
    fn take_and_finalize_to_partial(&mut self) -> Result<Vec<u8>, EvalError> {
        let prev = std::mem::replace(self, Self::Finalized);
        match prev {
            Self::Bbox(ev) => ev.finalize_to_partial(),
            Self::Segm(ev) => ev.finalize_to_partial(),
            Self::Boundary(ev) => ev.finalize_to_partial(),
            Self::Keypoints(ev) => ev.finalize_to_partial(),
            Self::Finalized => Err(finalized_error()),
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
}

/// Return tuple shape for `snapshot_with_tables` / `finalize_with_tables`:
/// `(Summary, per_image, per_class, per_detection, per_pair)` — each
/// table column `Some` only when its flag was set on the call.
type StreamingTablesResult = (
    PySummary,
    Option<arrow_helpers::ArrowRecordBatchPy>,
    Option<arrow_helpers::ArrowRecordBatchPy>,
    Option<arrow_helpers::ArrowRecordBatchPy>,
    Option<arrow_helpers::ArrowRecordBatchPy>,
);

fn streaming_tables_result(summary: Summary, tables: Tables) -> PyResult<StreamingTablesResult> {
    let per_image = tables
        .per_image
        .map(|t| tables::per_image_table_to_arrow(&t))
        .transpose()
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;
    let per_class = tables
        .per_class
        .map(|t| tables::per_class_table_to_arrow(&t))
        .transpose()
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;
    let per_detection = tables
        .per_detection
        .map(|t| tables::per_detection_table_to_arrow(&t))
        .transpose()
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;
    let per_pair = tables
        .per_pair
        .map(|t| tables::per_pair_table_to_arrow(&t))
        .transpose()
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;
    Ok((
        PySummary { inner: summary },
        per_image,
        per_class,
        per_detection,
        per_pair,
    ))
}

/// One kernel-input payload bound for the streaming evaluator. The JSON
/// path holds the JSON bytes; the array path holds a flat
/// `Vec<DetectionInput>`. **Neither** variant carries a constructed
/// [`CocoDetections`] — the `from_inputs` HashMap build is
/// intentionally deferred to the consumer's `py.detach` block so it
/// doesn't hold the GIL.
pub(crate) enum UpdatePayload {
    Bytes(pyo3::pybacked::PyBackedBytes),
    Inputs(Vec<DetectionInput>),
}

/// Classify and materialize the Python `detections=` argument into an
/// [`UpdatePayload`] ready to cross `py.detach`. Shared by the streaming
/// `update`, background `submit`, and foreground `evaluate_*_*` entry
/// points.
pub(crate) fn build_update_payload<'py>(
    py: Python<'py>,
    detections: &Bound<'py, PyAny>,
    iou_type: array_ingest::ArrayIouType,
    cast_state: &array_ingest::CastState,
) -> PyResult<UpdatePayload> {
    Ok(match array_ingest::DetectionsArg::extract(detections)? {
        array_ingest::DetectionsArg::Bytes(b) => UpdatePayload::Bytes(b),
        array_ingest::DetectionsArg::Dicts(dicts) => UpdatePayload::Inputs(
            array_ingest::dicts_to_inputs(py, &dicts, iou_type, cast_state)?,
        ),
    })
}

/// Realize a Send-safe [`UpdatePayload`] inside `py.detach` into the
/// [`CocoDetections`] the foreground pipeline takes by reference.
/// `Bytes` is parsed via `from_json_bytes`; `Inputs` runs `from_inputs`
/// — both happen without the GIL.
pub(crate) fn realize_dt(payload: UpdatePayload) -> PyResult<CocoDetections> {
    match payload {
        UpdatePayload::Bytes(b) => parse_dt(&b),
        UpdatePayload::Inputs(inputs) => CocoDetections::from_inputs(inputs)
            .map_err(|e| PyValueError::new_err(format!("detections array ingest: {e}"))),
    }
}

/// Lighter-weight discriminator over [`EvalIouType`] used by the
/// array-ingest validator to decide which fields are required.
impl From<&EvalIouType> for array_ingest::ArrayIouType {
    fn from(iou: &EvalIouType) -> Self {
        match iou {
            EvalIouType::Bbox => Self::Bbox,
            EvalIouType::Segm => Self::Segm,
            EvalIouType::Boundary { .. } => Self::Boundary,
            EvalIouType::Keypoints { .. } => Self::Keypoints,
        }
    }
}

/// Resolve the Python `dt=` argument for the foreground evaluators.
/// Bundles cast-state construction with the shared
/// [`build_update_payload`] dispatch so each `*_impl` doesn't repeat the
/// two-line preamble.
fn prepare_dt_payload<'py>(
    py: Python<'py>,
    dt: &Bound<'py, PyAny>,
    iou_type: &EvalIouType,
    cast_inputs: bool,
) -> PyResult<UpdatePayload> {
    let cast_state = array_ingest::new_cast_state(cast_inputs);
    build_update_payload(py, dt, iou_type.into(), &cast_state)
}

/// Internal Rust orchestrator for the per-rank distributed-eval flow
/// (ADR-0035). Not exposed to Python: the `evaluate_instance_to_partial`
/// and `merge_instance_partials` pyfunctions own one synchronously and
/// drive it through a single construct + update + finalize-to-partial
/// cycle.
struct InstanceStreamOrchestrator {
    state: StreamingState,
    /// Cached at construction so `update` doesn't have to re-walk the
    /// dispatch enum to learn which fields each `Detections` dict
    /// requires.
    array_iou_type: array_ingest::ArrayIouType,
    /// `Some(latch)` when `cast_inputs=True` — the latch fires the
    /// `UserWarning` at most once. `None` when the strict ADR-0004
    /// boundary is enforced.
    cast_state: array_ingest::CastState,
}

impl InstanceStreamOrchestrator {
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
        retain_iou: bool,
        cast_inputs: bool,
        rank_id: Option<u32>,
    ) -> PyResult<Self> {
        let parity = parse_parity_mode(parity_mode)?;
        require_nonempty_max_dets(&max_dets)?;
        // Quirk A2 (strict): same sort the batch path applies — the
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

        let dataset = parse_gt(gt_json.as_bytes())?;

        // Build the kernel-typed evaluator. Each arm constructs the
        // matching `EvalIouType` solely to reuse `area_ranges_for`'s
        // detection-vs-keypoints fork; this keeps the area-bucket
        // selection logic in one place.
        let (state, array_iou_type) = match iou_type {
            "bbox" => {
                let area = area_ranges_for(&EvalIouType::Bbox);
                let params = OwnedEvaluateParams {
                    iou_thresholds: iou_thresholds().to_vec(),
                    area_ranges: area,
                    max_dets_per_image,
                    use_cats,
                    retain_iou,
                };
                let ev = maybe_tag_rank(
                    StreamingEvaluator::new(dataset, BboxIou, params, parity, budget)
                        .map_err(|e| PyValueError::new_err(format!("{e}")))?,
                    rank_id,
                )?;
                (StreamingState::Bbox(ev), array_ingest::ArrayIouType::Bbox)
            }
            "segm" => {
                let area = area_ranges_for(&EvalIouType::Segm);
                let params = OwnedEvaluateParams {
                    iou_thresholds: iou_thresholds().to_vec(),
                    area_ranges: area,
                    max_dets_per_image,
                    use_cats,
                    retain_iou,
                };
                let ev = maybe_tag_rank(
                    StreamingEvaluator::new(dataset, SegmIou, params, parity, budget)
                        .map_err(|e| PyValueError::new_err(format!("{e}")))?,
                    rank_id,
                )?;
                (StreamingState::Segm(ev), array_ingest::ArrayIouType::Segm)
            }
            "boundary" => {
                let iou_kind = boundary_iou_type(dilation_ratio)?;
                let area = area_ranges_for(&iou_kind);
                let params = OwnedEvaluateParams {
                    iou_thresholds: iou_thresholds().to_vec(),
                    area_ranges: area,
                    max_dets_per_image,
                    use_cats,
                    retain_iou,
                };
                let kernel = BoundaryIou { dilation_ratio };
                let ev = maybe_tag_rank(
                    StreamingEvaluator::new(dataset, kernel, params, parity, budget)
                        .map_err(|e| PyValueError::new_err(format!("{e}")))?,
                    rank_id,
                )?;
                (
                    StreamingState::Boundary(ev),
                    array_ingest::ArrayIouType::Boundary,
                )
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
                    retain_iou,
                };
                let kernel = OksSimilarity::new(parsed_sigmas);
                let ev = maybe_tag_rank(
                    StreamingEvaluator::new(dataset, kernel, params, parity, budget)
                        .map_err(|e| PyValueError::new_err(format!("{e}")))?,
                    rank_id,
                )?;
                (
                    StreamingState::Keypoints(ev),
                    array_ingest::ArrayIouType::Keypoints,
                )
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "invalid iou_type {other:?}; expected 'bbox', 'segm', 'boundary', or 'keypoints'"
                )));
            }
        };

        Ok(Self {
            state,
            array_iou_type,
            cast_state: array_ingest::new_cast_state(cast_inputs),
        })
    }

    /// Submit a batch of detections. Accepts either loadRes-shaped JSON
    /// `bytes` (legacy) or an ADR-0030 `Detections` dict / sequence of
    /// `Detections` dicts (numpy/DLPack). Returns an `_UpdateReportDict`
    /// describing what was accepted plus the post-update memory total.
    fn update<'py>(
        &mut self,
        py: Python<'py>,
        detections: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        // Build the kernel-input payload before dropping the GIL: the
        // array path borrows DLPack views to materialize `DetectionInput`s,
        // which requires Python-side reads.
        let parsed_payload =
            build_update_payload(py, detections, self.array_iou_type, &self.cast_state)?;

        let state = &mut self.state;
        let (report, memory_used_bytes) = py
            .detach(move || {
                let report = state.run_update(parsed_payload)?;
                let memory = state.memory_used_bytes();
                Ok::<(UpdateReport, usize), EvalError>((report, memory))
            })
            .map_err(|e| eval_error_to_pyerr(py, e))?;

        // `report.soft_warn_triggered` is set by the core evaluator
        // exactly once per stream — no FFI-side latch needed.
        if report.soft_warn_triggered {
            emit_warning::<MemoryBudgetWarning>(
                py,
                &format!(
                    "StreamingEvaluator memory usage crossed the soft-warn threshold \
                     ({memory_used_bytes} bytes used)"
                ),
            )?;
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

    /// Consume the orchestrator and return the final summary.
    fn finalize(mut self, py: Python<'_>) -> PyResult<PySummary> {
        let summary = py
            .detach(move || self.state.take_and_finalize())
            .map_err(|e| eval_error_to_pyerr(py, e))?;
        Ok(PySummary { inner: summary })
    }

    /// ADR-0031: consume the orchestrator and serialize its final state
    /// as a partial byte blob.
    fn finalize_to_partial<'py>(mut self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let blob = py
            .detach(move || self.state.take_and_finalize_to_partial())
            .map_err(|e| eval_error_to_pyerr(py, e))?;
        Ok(PyBytes::new(py, &blob))
    }

    /// ADR-0031: reconstruct an evaluator from N partial blobs
    /// (typically one per rank in a multi-process eval).
    ///
    /// Validation is strict per the ADR's "Validation order": each
    /// partial must agree on the receiving rank's `iou_type`,
    /// `parity_mode`, `max_dets`, `use_cats`, `retain_iou`, dataset
    /// hash, and params hash. In strict mode every partial must
    /// declare a distinct `rank_id`. Image-id sets across partials
    /// must be disjoint.
    #[allow(clippy::too_many_arguments)]
    fn from_partials(
        py: Python<'_>,
        gt_json: &Bound<'_, PyBytes>,
        partials: &Bound<'_, pyo3::types::PyList>,
        iou_type: &str,
        parity_mode: &str,
        max_dets: Vec<usize>,
        use_cats: bool,
        memory_budget_bytes: Option<usize>,
        dilation_ratio: f64,
        sigmas: Option<&Bound<'_, PyDict>>,
        retain_iou: bool,
        cast_inputs: bool,
    ) -> PyResult<Self> {
        let parity = parse_parity_mode(parity_mode)?;
        require_nonempty_max_dets(&max_dets)?;
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
        let dataset = parse_gt(gt_json.as_bytes())?;
        // Copy partial bytes off the GIL — we'll release it for the
        // rkyv decode + cell-store fold. Each list entry must be
        // bytes; we reject non-bytes inputs at the boundary.
        let mut partial_buffers: Vec<Vec<u8>> = Vec::with_capacity(partials.len());
        for entry in partials.iter() {
            let b = entry.cast::<PyBytes>().map_err(|_| {
                PyValueError::new_err("partials must be a sequence of bytes objects")
            })?;
            partial_buffers.push(b.as_bytes().to_vec());
        }

        let parsed_sigmas = match (iou_type, sigmas) {
            ("keypoints", Some(d)) => parse_sigmas(d)?,
            _ => HashMap::new(),
        };
        // Validate dilation_ratio eagerly under the GIL — the boundary
        // arm uses it and the bbox/segm/keypoints arms ignore it, but
        // an out-of-range value still indicates a caller bug and the
        // typed PyValueError surfaces cleaner now than buried inside
        // py.detach.
        boundary_iou_type(dilation_ratio)?;

        let (state, array_iou_type) = py
            .detach(move || -> Result<(StreamingState, array_ingest::ArrayIouType), EvalError> {
                let partial_refs: Vec<&[u8]> =
                    partial_buffers.iter().map(Vec::as_slice).collect();
                match iou_type {
                    "bbox" => {
                        let area = area_ranges_for(&EvalIouType::Bbox);
                        let params = OwnedEvaluateParams {
                            iou_thresholds: iou_thresholds().to_vec(),
                            area_ranges: area,
                            max_dets_per_image,
                            use_cats,
                            retain_iou,
                        };
                        let ev = StreamingEvaluator::from_partials(
                            dataset, BboxIou, params, parity, budget, &partial_refs,
                        )?;
                        Ok((StreamingState::Bbox(ev), array_ingest::ArrayIouType::Bbox))
                    }
                    "segm" => {
                        let area = area_ranges_for(&EvalIouType::Segm);
                        let params = OwnedEvaluateParams {
                            iou_thresholds: iou_thresholds().to_vec(),
                            area_ranges: area,
                            max_dets_per_image,
                            use_cats,
                            retain_iou,
                        };
                        let ev = StreamingEvaluator::from_partials(
                            dataset, SegmIou, params, parity, budget, &partial_refs,
                        )?;
                        Ok((StreamingState::Segm(ev), array_ingest::ArrayIouType::Segm))
                    }
                    "boundary" => {
                        let iou_kind_dyn = EvalIouType::Boundary { dilation_ratio };
                        let area = area_ranges_for(&iou_kind_dyn);
                        let params = OwnedEvaluateParams {
                            iou_thresholds: iou_thresholds().to_vec(),
                            area_ranges: area,
                            max_dets_per_image,
                            use_cats,
                            retain_iou,
                        };
                        let kernel = BoundaryIou { dilation_ratio };
                        let ev = StreamingEvaluator::from_partials(
                            dataset, kernel, params, parity, budget, &partial_refs,
                        )?;
                        Ok((
                            StreamingState::Boundary(ev),
                            array_ingest::ArrayIouType::Boundary,
                        ))
                    }
                    "keypoints" => {
                        let iou_kind_dyn = EvalIouType::Keypoints {
                            sigmas: parsed_sigmas.clone(),
                        };
                        let area = area_ranges_for(&iou_kind_dyn);
                        let params = OwnedEvaluateParams {
                            iou_thresholds: iou_thresholds().to_vec(),
                            area_ranges: area,
                            max_dets_per_image,
                            use_cats,
                            retain_iou,
                        };
                        let kernel = OksSimilarity::new(parsed_sigmas);
                        let ev = StreamingEvaluator::from_partials(
                            dataset, kernel, params, parity, budget, &partial_refs,
                        )?;
                        Ok((
                            StreamingState::Keypoints(ev),
                            array_ingest::ArrayIouType::Keypoints,
                        ))
                    }
                    _ => Err(EvalError::InvalidConfig {
                        detail: format!(
                            "invalid iou_type {iou_type:?}; expected 'bbox', 'segm', 'boundary', or 'keypoints'"
                        ),
                    }),
                }
            })
            .map_err(|e| eval_error_to_pyerr(py, e))?;

        Ok(Self {
            state,
            array_iou_type,
            cast_state: array_ingest::new_cast_state(cast_inputs),
        })
    }
}

/// Spawn a [`background::BackgroundEvaluator`] for a typed kernel,
/// folding the three error-mapping shapes — `with_rank`,
/// `StreamingEvaluator::new`, and `spawn_with_options` — that every
/// branch of [`PyBackgroundEvaluator::new`] shares.
#[allow(clippy::too_many_arguments)]
fn spawn_background_evaluator<K: EvalKernel + Send + 'static>(
    py: Python<'_>,
    dataset: CocoDataset,
    kernel: K,
    params: OwnedEvaluateParams,
    parity: ParityMode,
    budget: MemoryBudget,
    config: background::BackgroundConfig,
    rank_id: Option<u32>,
    record_latency_samples: bool,
) -> PyResult<background::BackgroundEvaluator<K>> {
    let ev = maybe_tag_rank(
        StreamingEvaluator::new(dataset, kernel, params, parity, budget)
            .map_err(|e| PyValueError::new_err(format!("{e}")))?,
        rank_id,
    )?;
    background::BackgroundEvaluator::spawn_with_options(ev, config, record_latency_samples)
        .map_err(|e| eval_error_to_pyerr(py, e))
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
    /// Segm path constructed from a `CocoDataset` handle (ADR-0020): the
    /// kernel owns an `Arc<SegmGtCache>` so per-annotation GT-side
    /// derivations are reused across `submit()` rounds.
    SegmCached(background::BackgroundEvaluator<SegmIouCached<'static>>),
    Boundary(background::BackgroundEvaluator<BoundaryIou>),
    /// Boundary path constructed from a `CocoDataset` handle
    /// (ADR-0020): the kernel owns an `Arc<BoundaryGtCache>`, so the
    /// ~36k Chebyshev erodes that build the GT band are paid once per
    /// dataset rather than once per submitted batch.
    BoundaryCached(background::BackgroundEvaluator<BoundaryIouCached<'static>>),
    Keypoints(background::BackgroundEvaluator<OksSimilarity>),
    Finalized,
}

/// Post a kernel-typed [`ParsedDetections`] to the worker, using either
/// the blocking or bounded-wait sender depending on `timeout`. Lifted
/// out of `BackgroundEvalState::submit` so the JSON and array paths can
/// share the same backpressure logic.
fn send_parsed<K: EvalKernel + Send + 'static>(
    ev: &background::BackgroundEvaluator<K>,
    parsed: ParsedDetections<K>,
    timeout: Option<Duration>,
) -> Result<(), background::SubmitError> {
    match timeout {
        None => ev
            .submit_blocking(parsed)
            .map_err(background::SubmitError::Eval),
        Some(t) => ev.submit_timeout(parsed, t),
    }
}

/// Build the `EvalError` returned for any operation attempted after
/// `finalize()` on the background surface. Mirrors [`finalized_error`].
fn background_finalized_error() -> EvalError {
    EvalError::InvalidConfig {
        detail: "BackgroundEvaluator has already been finalized".into(),
    }
}

/// Borrowing dispatch: `match self` over every kernel variant, binding
/// the inner `BackgroundEvaluator<K>` to `$ev` for `$body`. Used by the
/// read-only accessors (`images_seen`, `queue_depth`, ...) where every
/// arm calls the same method on `$ev` and returns the same scalar type.
/// Adding a new kernel paradigm means adding one line *here*, not 11
/// hand-written arms across the impl block.
macro_rules! dispatch_state_ref {
    ($self:expr, $ev:ident => $body:expr, $finalized:expr) => {
        match $self {
            Self::Bbox($ev) => $body,
            Self::Segm($ev) => $body,
            Self::SegmCached($ev) => $body,
            Self::Boundary($ev) => $body,
            Self::BoundaryCached($ev) => $body,
            Self::Keypoints($ev) => $body,
            Self::Finalized => $finalized,
        }
    };
}

/// Consuming dispatch: replace `*self` with [`BackgroundEvalState::Finalized`]
/// and `match` the prior value, binding the inner
/// `BackgroundEvaluator<K>` to `$ev` for `$body`. Used by `take_and_*`
/// and `shutdown` where each arm needs to consume the inner evaluator.
macro_rules! dispatch_state_consuming {
    ($self:expr, $ev:ident => $body:expr, $finalized:expr) => {
        match std::mem::replace($self, Self::Finalized) {
            Self::Bbox($ev) => $body,
            Self::Segm($ev) => $body,
            Self::SegmCached($ev) => $body,
            Self::Boundary($ev) => $body,
            Self::BoundaryCached($ev) => $body,
            Self::Keypoints($ev) => $body,
            Self::Finalized => $finalized,
        }
    };
}

impl BackgroundEvalState {
    /// Post a detection batch to the worker. The JSON arm parses to
    /// [`ParsedDetections`] inside the kernel-typed branch; the array arm
    /// wraps a pre-parsed [`CocoDetections`] (ADR-0030). `timeout`
    /// mirrors the Python `timeout=` parameter on `submit()`:
    ///
    /// - `None` → block forever (`submit_blocking`)
    /// - `Some(Duration::ZERO)` → single non-blocking attempt
    /// - `Some(t > 0)` → bounded wait
    fn submit_payload(
        &self,
        payload: UpdatePayload,
        timeout: Option<Duration>,
    ) -> Result<(), background::SubmitError> {
        macro_rules! dispatch {
            ($ev:expr, $K:ty) => {{
                let parsed = match payload {
                    UpdatePayload::Bytes(b) => ParsedDetections::<$K>::from_json_bytes(&b)
                        .map_err(background::SubmitError::Eval)?,
                    UpdatePayload::Inputs(inputs) => {
                        let detections = CocoDetections::from_inputs(inputs)
                            .map_err(background::SubmitError::Eval)?;
                        ParsedDetections::<$K>::from_detections(detections)
                    }
                };
                send_parsed($ev, parsed, timeout)
            }};
        }
        match self {
            Self::Bbox(ev) => dispatch!(ev, BboxIou),
            Self::Segm(ev) => dispatch!(ev, SegmIou),
            Self::SegmCached(ev) => dispatch!(ev, SegmIouCached<'static>),
            Self::Boundary(ev) => dispatch!(ev, BoundaryIou),
            Self::BoundaryCached(ev) => dispatch!(ev, BoundaryIouCached<'static>),
            Self::Keypoints(ev) => dispatch!(ev, OksSimilarity),
            Self::Finalized => Err(background::SubmitError::Eval(background_finalized_error())),
        }
    }

    fn take_and_finalize(&mut self) -> Result<Summary, EvalError> {
        dispatch_state_consuming!(self, ev => ev.finalize(), Err(background_finalized_error()))
    }

    fn take_and_finalize_with_tables(
        &mut self,
        request: TablesRequest,
        config: TablesConfig,
    ) -> Result<(Summary, Tables), EvalError> {
        dispatch_state_consuming!(
            self,
            ev => ev.finalize_with_tables(request, config),
            Err(background_finalized_error())
        )
    }

    fn take_and_finalize_to_partial(&mut self) -> Result<Vec<u8>, EvalError> {
        dispatch_state_consuming!(
            self,
            ev => ev.finalize_to_partial(),
            Err(background_finalized_error())
        )
    }

    /// ADR-0018 Unit 6: consume the inner background evaluator and
    /// return both the canonical [`Summary`] and the per-image cell
    /// store needed by the calibration summarizer. The summary axis is
    /// bit-identical to [`Self::take_and_finalize`]; this variant only
    /// adds cell retention.
    fn take_and_finalize_with_cells(
        &mut self,
    ) -> Result<vernier_core::stream::SnapshotWithCells, EvalError> {
        dispatch_state_consuming!(
            self,
            ev => ev.finalize_with_cells(),
            Err(background_finalized_error())
        )
    }

    fn take_scheduling_outcome(&self) -> Option<Result<(), String>> {
        dispatch_state_ref!(self, ev => ev.take_scheduling_outcome(), None)
    }

    fn images_seen(&self) -> usize {
        dispatch_state_ref!(self, ev => ev.images_seen(), 0)
    }

    fn detections_seen(&self) -> usize {
        dispatch_state_ref!(self, ev => ev.detections_seen(), 0)
    }

    fn queue_depth(&self) -> usize {
        dispatch_state_ref!(self, ev => ev.queue_depth(), 0)
    }

    fn memory_used_bytes(&self) -> usize {
        dispatch_state_ref!(self, ev => ev.memory_used_bytes(), 0)
    }

    /// B5: drain the worker's accumulated submit-latency samples (in
    /// nanoseconds). Returns an empty `Vec` when the evaluator was
    /// constructed without `record_latency_samples=True` or after
    /// finalize.
    fn latency_samples_drain(&self) -> Vec<u64> {
        dispatch_state_ref!(self, ev => ev.latency_samples_drain(), Vec::new())
    }

    /// Best-effort cooperative shutdown. Used by `__exit__` and `__del__`
    /// when the evaluator hasn't already been finalized.
    fn shutdown(&mut self) {
        dispatch_state_consuming!(self, ev => ev.shutdown(), ())
    }

    /// Test-only: forward to the worker's `_inject_poison_for_tests`.
    /// Gated behind the `test-poison` Cargo feature; only the panic-recovery
    /// test in `tests/python/background/test_background_worker_panic.py`
    /// reaches this.
    #[cfg(feature = "test-poison")]
    fn inject_poison_for_tests(&self) -> Result<(), EvalError> {
        dispatch_state_ref!(
            self,
            ev => ev._inject_poison_for_tests(),
            Err(background_finalized_error())
        )
    }
}

/// Map a [`background::SubmitError`] to a Python exception. `Eval` is
/// routed through [`eval_error_to_pyerr`]; `Full` is routed through
/// [`queue_full_to_pyerr`] (shared with the semantic / panoptic
/// background paradigms — the `QueueFull` shape is paradigm-agnostic).
fn submit_error_to_pyerr(py: Python<'_>, e: background::SubmitError) -> PyErr {
    match e {
        background::SubmitError::Eval(inner) => eval_error_to_pyerr(py, inner),
        background::SubmitError::Full(full) => queue_full_to_pyerr(py, full),
    }
}

/// Validate a `shutdown_timeout_seconds` constructor arg
/// (non-negative finite float). Shared across the three background
/// pyclasses so the message is identical and the `f64::is_finite` /
/// negative checks don't drift.
pub(crate) fn validate_shutdown_timeout(seconds: f64) -> PyResult<Duration> {
    if !seconds.is_finite() || seconds < 0.0 {
        return Err(PyValueError::new_err(format!(
            "shutdown_timeout_seconds must be a non-negative finite float, got {seconds}"
        )));
    }
    Ok(Duration::from_secs_f64(seconds))
}

/// Briefly poll the worker's startup scheduling outcome and emit a
/// single `UserWarning` on `Err`. Shared across the three background
/// pyclasses.
///
/// `read_outcome` is a closure the caller supplies that locks the
/// caller's lifecycle/state mutex and forwards
/// `take_scheduling_outcome()` — paradigm-specific because the
/// pyclass holds its own mutex shape, but the polling cadence and
/// warning emission are uniform.
pub(crate) fn poll_scheduling_warning<F>(
    py: Python<'_>,
    evaluator_name: &str,
    mut read_outcome: F,
) -> PyResult<()>
where
    F: FnMut() -> PyResult<Option<Result<(), String>>>,
{
    const POLL_TRIES: usize = 10;
    const POLL_INTERVAL: Duration = Duration::from_millis(1);
    let mut found: Option<Result<(), String>> = None;
    for _ in 0..POLL_TRIES {
        if let Some(o) = read_outcome()? {
            found = Some(o);
            break;
        }
        py.detach(|| std::thread::sleep(POLL_INTERVAL));
    }
    if let Some(Err(msg)) = found {
        emit_warning::<PyUserWarning>(py, &format!("{evaluator_name} scheduling: {msg}"))?;
    }
    Ok(())
}

/// Materialize a [`background::QueueFull`] into a [`QueueFullError`]
/// with `queue_capacity` and `timeout` (in fractional seconds)
/// attached as instance attributes. Shared across paradigms; the
/// non-`QueueFull` arm of any paradigm's submit-error mapping calls
/// the paradigm-specific error mapper.
pub(crate) fn queue_full_to_pyerr(py: Python<'_>, full: background::QueueFull) -> PyErr {
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

// ---------------------------------------------------------------------------
// One-shot streaming partial functions (ADR-0035).
//
// Public entry points for the per-rank distributed-eval flow on the
// instance paradigm. Each pyfunction owns one
// `InstanceStreamOrchestrator` synchronously and drives it through a
// single construct + update + finalize-to-partial cycle.
// ---------------------------------------------------------------------------

/// Construct a streaming evaluator, submit one batch, and serialize a
/// partial blob (ADR-0031, ADR-0035).
#[pyfunction]
#[pyo3(signature = (
    gt_json,
    detections,
    iou_type,
    rank_id,
    *,
    parity_mode = "corrected",
    max_dets = vec![1, 10, 100],
    use_cats = true,
    memory_budget_bytes = None,
    dilation_ratio = 0.02,
    sigmas = None,
    retain_iou = false,
    cast_inputs = false,
))]
#[allow(clippy::too_many_arguments)]
fn evaluate_instance_to_partial<'py>(
    py: Python<'py>,
    gt_json: &Bound<'py, PyBytes>,
    detections: &Bound<'py, PyAny>,
    iou_type: &str,
    rank_id: u32,
    parity_mode: &str,
    max_dets: Vec<usize>,
    use_cats: bool,
    memory_budget_bytes: Option<usize>,
    dilation_ratio: f64,
    sigmas: Option<&Bound<'py, PyDict>>,
    retain_iou: bool,
    cast_inputs: bool,
) -> PyResult<Bound<'py, PyBytes>> {
    let mut ev = InstanceStreamOrchestrator::new(
        gt_json,
        iou_type,
        parity_mode,
        max_dets,
        use_cats,
        memory_budget_bytes,
        dilation_ratio,
        sigmas,
        retain_iou,
        cast_inputs,
        Some(rank_id),
    )?;
    let _report = ev.update(py, detections)?;
    ev.finalize_to_partial(py)
}

/// Merge per-rank partials into a final summary (ADR-0031, ADR-0035).
#[pyfunction]
#[pyo3(signature = (
    gt_json,
    partials,
    iou_type,
    *,
    parity_mode = "corrected",
    max_dets = vec![1, 10, 100],
    use_cats = true,
    memory_budget_bytes = None,
    dilation_ratio = 0.02,
    sigmas = None,
    retain_iou = false,
    cast_inputs = false,
))]
#[allow(clippy::too_many_arguments)]
fn merge_instance_partials<'py>(
    py: Python<'py>,
    gt_json: &Bound<'py, PyBytes>,
    partials: &Bound<'py, pyo3::types::PyList>,
    iou_type: &str,
    parity_mode: &str,
    max_dets: Vec<usize>,
    use_cats: bool,
    memory_budget_bytes: Option<usize>,
    dilation_ratio: f64,
    sigmas: Option<&Bound<'py, PyDict>>,
    retain_iou: bool,
    cast_inputs: bool,
) -> PyResult<PySummary> {
    let merged = InstanceStreamOrchestrator::from_partials(
        py,
        gt_json,
        partials,
        iou_type,
        parity_mode,
        max_dets,
        use_cats,
        memory_budget_bytes,
        dilation_ratio,
        sigmas,
        retain_iou,
        cast_inputs,
    )?;
    merged.finalize(py)
}

/// Background-evaluator surface (ADR-0014). Wraps a worker thread that
/// owns the `StreamingEvaluator<K>`; every public method either sends on
/// the channel or reads atomic counters. Not frozen — `finalize()` and
/// `__exit__` need to mutate state.
#[pyclass(module = "vernier._core", name = "BackgroundEvaluator")]
struct PyBackgroundEvaluator {
    state: Mutex<BackgroundEvalState>,
    /// Cached at construction; same role as on `InstanceStreamOrchestrator`.
    array_iou_type: array_ingest::ArrayIouType,
    /// `Some` when `cast_inputs=True`. See `InstanceStreamOrchestrator::cast_state`.
    cast_state: array_ingest::CastState,
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
        gt,
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
        retain_iou = false,
        cast_inputs = false,
        rank_id = None,
        record_latency_samples = false,
        num_threads = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        gt: &Bound<'_, PyAny>,
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
        retain_iou: bool,
        cast_inputs: bool,
        rank_id: Option<u32>,
        record_latency_samples: bool,
        num_threads: Option<usize>,
    ) -> PyResult<Self> {
        let parity = parse_parity_mode(parity_mode)?;
        require_nonempty_max_dets(&max_dets)?;
        // Quirk A2 (strict): same sort the streaming/batch paths apply.
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

        // ADR-0020: a `CocoDataset` handle carries `Arc`-shared
        // per-kernel caches that the worker thread reuses across
        // `submit()` rounds; raw bytes carry no caches and the kernel
        // dispatch below stays on the uncached variant.
        let (snapshot, has_dataset_handle) = if let Ok(py_dataset) = gt.cast::<dataset::PyDataset>()
        {
            (py_dataset.borrow().snapshot(), true)
        } else if let Ok(bytes) = gt.cast::<PyBytes>() {
            (
                dataset::DatasetSnapshot::from_parsed(parse_gt(bytes.as_bytes())?),
                false,
            )
        } else {
            return Err(PyTypeError::new_err(
                "BackgroundEvaluator(...) gt must be `bytes` or `vernier.instance.CocoDataset`",
            ));
        };
        let (dataset, boundary_cache_arc, segm_cache_arc) = snapshot.into_parts();

        if !shutdown_timeout_seconds.is_finite() || shutdown_timeout_seconds < 0.0 {
            return Err(PyValueError::new_err(format!(
                "shutdown_timeout_seconds must be a non-negative finite float, got {shutdown_timeout_seconds}"
            )));
        }

        // ADR-0047 Stage B: resolve `num_threads` under the GIL so the
        // re-entry / env-var `UserWarning` can fire to Python *before*
        // the worker thread spawns. The resolved `NonZeroUsize` is
        // owned by `BackgroundConfig` and consumed by `worker_loop`
        // when it builds the per-worker scoped pool.
        let thread_policy = threads::resolve_threads(py, num_threads);
        let config = background::BackgroundConfig {
            queue_capacity,
            worker_affinity,
            worker_nice,
            shutdown_timeout: Duration::from_secs_f64(shutdown_timeout_seconds),
            num_threads: thread_policy.thread_count(),
        };

        let make_params = |area: Vec<AreaRange>| OwnedEvaluateParams {
            iou_thresholds: iou_thresholds().to_vec(),
            area_ranges: area,
            max_dets_per_image,
            use_cats,
            retain_iou,
        };

        let (state, array_iou_type) = match iou_type {
            "bbox" => {
                let params = make_params(area_ranges_for(&EvalIouType::Bbox));
                let bg = spawn_background_evaluator(
                    py,
                    dataset,
                    BboxIou,
                    params,
                    parity,
                    budget,
                    config,
                    rank_id,
                    record_latency_samples,
                )?;
                (
                    BackgroundEvalState::Bbox(bg),
                    array_ingest::ArrayIouType::Bbox,
                )
            }
            "segm" => {
                let params = make_params(area_ranges_for(&EvalIouType::Segm));
                let state = if has_dataset_handle {
                    let bg = spawn_background_evaluator(
                        py,
                        dataset,
                        SegmIouCached::with_arc_cache(segm_cache_arc),
                        params,
                        parity,
                        budget,
                        config,
                        rank_id,
                        record_latency_samples,
                    )?;
                    BackgroundEvalState::SegmCached(bg)
                } else {
                    let bg = spawn_background_evaluator(
                        py,
                        dataset,
                        SegmIou,
                        params,
                        parity,
                        budget,
                        config,
                        rank_id,
                        record_latency_samples,
                    )?;
                    BackgroundEvalState::Segm(bg)
                };
                (state, array_ingest::ArrayIouType::Segm)
            }
            "boundary" => {
                let iou_kind = boundary_iou_type(dilation_ratio)?;
                let params = make_params(area_ranges_for(&iou_kind));
                let state = if has_dataset_handle {
                    let bg = spawn_background_evaluator(
                        py,
                        dataset,
                        BoundaryIouCached::with_arc_cache(dilation_ratio, boundary_cache_arc),
                        params,
                        parity,
                        budget,
                        config,
                        rank_id,
                        record_latency_samples,
                    )?;
                    BackgroundEvalState::BoundaryCached(bg)
                } else {
                    let bg = spawn_background_evaluator(
                        py,
                        dataset,
                        BoundaryIou { dilation_ratio },
                        params,
                        parity,
                        budget,
                        config,
                        rank_id,
                        record_latency_samples,
                    )?;
                    BackgroundEvalState::Boundary(bg)
                };
                (state, array_ingest::ArrayIouType::Boundary)
            }
            "keypoints" => {
                let parsed_sigmas = match sigmas {
                    Some(d) => parse_sigmas(d)?,
                    None => HashMap::new(),
                };
                let iou_kind = EvalIouType::Keypoints {
                    sigmas: parsed_sigmas.clone(),
                };
                let params = make_params(area_ranges_for(&iou_kind));
                let bg = spawn_background_evaluator(
                    py,
                    dataset,
                    OksSimilarity::new(parsed_sigmas),
                    params,
                    parity,
                    budget,
                    config,
                    rank_id,
                    record_latency_samples,
                )?;
                (
                    BackgroundEvalState::Keypoints(bg),
                    array_ingest::ArrayIouType::Keypoints,
                )
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "invalid iou_type {other:?}; expected 'bbox', 'segm', 'boundary', or 'keypoints'"
                )));
            }
        };

        let this = Self {
            state: Mutex::new(state),
            array_iou_type,
            cast_state: array_ingest::new_cast_state(cast_inputs),
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
            emit_warning::<PyUserWarning>(py, &format!("BackgroundEvaluator scheduling: {msg}"))?;
        }

        Ok(this)
    }

    /// Submit a detection batch to the worker. Accepts either loadRes-
    /// shaped JSON `bytes` (legacy) or an ADR-0030 `Detections` dict /
    /// sequence of `Detections` dicts (numpy/DLPack). `timeout` controls
    /// backpressure:
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
        detections: &Bound<'_, PyAny>,
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

        let payload = build_update_payload(py, detections, self.array_iou_type, &self.cast_state)?;

        let state_mutex = &self.state;
        let result = py.detach(move || {
            let guard = state_mutex.lock().map_err(|_| {
                background::SubmitError::Eval(EvalError::InvalidConfig {
                    detail: "BackgroundEvaluator state mutex poisoned".into(),
                })
            })?;
            guard.submit_payload(payload, timeout_dur)
        });
        result.map_err(|e| submit_error_to_pyerr(py, e))
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

    /// Tables-aware finalize. Drains the queue and consumes the worker.
    #[pyo3(signature = (
        per_image=false,
        per_class=false,
        per_detection=false,
        per_pair=false,
        per_pair_iou_floor=0.1,
        per_pair_max_rows=10_000_000,
        per_detection_with_geometry=false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn finalize_with_tables(
        &self,
        py: Python<'_>,
        per_image: bool,
        per_class: bool,
        per_detection: bool,
        per_pair: bool,
        per_pair_iou_floor: f64,
        per_pair_max_rows: usize,
        per_detection_with_geometry: bool,
    ) -> PyResult<StreamingTablesResult> {
        let request = TablesRequest {
            per_image,
            per_class,
            per_detection,
            per_pair,
        };
        let cfg = TablesConfig {
            per_pair_iou_floor,
            per_pair_max_rows,
            per_detection_with_geometry,
        };
        let state_mutex = &self.state;
        let (summary, tables) = py
            .detach(move || {
                let mut guard = state_mutex.lock().map_err(|_| EvalError::InvalidConfig {
                    detail: "BackgroundEvaluator state mutex poisoned".into(),
                })?;
                guard.take_and_finalize_with_tables(request, cfg)
            })
            .map_err(|e| eval_error_to_pyerr(py, e))?;
        streaming_tables_result(summary, tables)
    }

    /// ADR-0018 Unit 6: drain the queue, finalize the evaluator, and
    /// return both the canonical :class:`Summary` and the opaque
    /// :class:`EvalCells` handle the calibration summarizer consumes.
    ///
    /// The summary axis is bit-identical to :meth:`finalize`; this
    /// variant only adds cell retention. Subsequent calls raise
    /// "already finalized". The Python-side adapter that wraps the
    /// returned tuple into a :class:`StreamingSnapshot` lives in
    /// ``vernier.calibration``.
    fn finalize_with_cells(&self, py: Python<'_>) -> PyResult<(PySummary, calibration::EvalCells)> {
        let state_mutex = &self.state;
        let bundle = py
            .detach(move || {
                let mut guard = state_mutex.lock().map_err(|_| EvalError::InvalidConfig {
                    detail: "BackgroundEvaluator state mutex poisoned".into(),
                })?;
                guard.take_and_finalize_with_cells()
            })
            .map_err(|e| eval_error_to_pyerr(py, e))?;
        let summary = PySummary {
            inner: bundle.summary,
        };
        let cells = calibration::EvalCells::new(
            bundle.eval_imgs,
            bundle.n_categories,
            bundle.n_area_ranges,
            bundle.iou_thresholds,
            bundle.parity_mode,
        );
        Ok((summary, cells))
    }

    /// ADR-0031 / ADR-0035: drain the queue, serialize the worker's
    /// final state as a partial blob, and shut the worker down.
    /// Subsequent calls raise "already finalized".
    fn finalize_to_partial<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let state_mutex = &self.state;
        let blob = py
            .detach(move || {
                let mut guard = state_mutex.lock().map_err(|_| EvalError::InvalidConfig {
                    detail: "BackgroundEvaluator state mutex poisoned".into(),
                })?;
                guard.take_and_finalize_to_partial()
            })
            .map_err(|e| eval_error_to_pyerr(py, e))?;
        Ok(PyBytes::new(py, &blob))
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

    /// Drain the worker's accumulated submit-latency samples (B5).
    ///
    /// Each sample is the wall-time of one ``submit()`` call's
    /// channel-send leg, in nanoseconds. The buffer is reset to empty
    /// on each call so subsequent submits keep accumulating; returns
    /// an empty list when the evaluator was constructed without
    /// ``record_latency_samples=True`` (the default) or after
    /// ``finalize`` has consumed the worker.
    fn drain_latency_samples_ns(&self) -> PyResult<Vec<u64>> {
        Ok(self.lock_state()?.latency_samples_drain())
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
    #[cfg(feature = "bench-histogram")]
    m.add_function(wrap_pyfunction!(dump_bbox_iou_histogram, m)?)?;
    #[cfg(feature = "bench-timings")]
    m.add_function(wrap_pyfunction!(
        read_and_reset_evaluate_parallel_timings,
        m
    )?)?;
    #[cfg(feature = "bench-timings")]
    m.add_function(wrap_pyfunction!(read_and_reset_build_anns_count, m)?)?;
    #[cfg(feature = "bench-timings")]
    m.add_function(wrap_pyfunction!(read_and_reset_dataset_timings, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_bbox_summary, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_bbox_grid, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_bbox_grid_with_dataset, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_segm_summary, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_segm_grid, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_boundary_summary, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_boundary_grid, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_keypoints_summary, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_keypoints_grid, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_bbox_summary_with_dataset, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_segm_summary_with_dataset, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_boundary_summary_with_dataset, m)?)?;
    m.add_function(wrap_pyfunction!(
        evaluate_keypoints_summary_with_dataset,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(calibration::cells_from_grid, m)?)?;
    m.add_function(wrap_pyfunction!(tables::per_class_to_arrow_pycapsule, m)?)?;
    m.add_function(wrap_pyfunction!(tables::per_image_to_arrow_pycapsule, m)?)?;
    m.add_function(wrap_pyfunction!(
        tables::per_detection_to_arrow_pycapsule,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(tables::per_pair_to_arrow_pycapsule, m)?)?;
    m.add_function(wrap_pyfunction!(
        panoptic_tables::panoptic_per_class_to_arrow_pycapsule,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        semantic_tables::semantic_per_class_to_arrow_pycapsule,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(tide::error_decomposition_bbox, m)?)?;
    m.add_function(wrap_pyfunction!(tide::error_decomposition_segm, m)?)?;
    m.add_function(wrap_pyfunction!(tide::error_decomposition_boundary, m)?)?;
    m.add_function(wrap_pyfunction!(tide::fp_iou_histogram_bbox, m)?)?;
    m.add_function(wrap_pyfunction!(tide::fp_iou_histogram_segm, m)?)?;
    m.add_function(wrap_pyfunction!(tide::fp_iou_histogram_boundary, m)?)?;
    m.add_function(wrap_pyfunction!(lrp::optimal_lrp_bbox, m)?)?;
    m.add_function(wrap_pyfunction!(lrp::optimal_lrp_segm, m)?)?;
    m.add_function(wrap_pyfunction!(lrp::optimal_lrp_boundary, m)?)?;
    m.add_function(wrap_pyfunction!(lrp::optimal_lrp_keypoints, m)?)?;
    m.add_function(wrap_pyfunction!(lrp::lrp_default_tau_grid, m)?)?;
    m.add_function(wrap_pyfunction!(confusion::confusion_matrix_bbox, m)?)?;
    m.add_function(wrap_pyfunction!(confusion::confusion_matrix_segm, m)?)?;
    m.add_function(wrap_pyfunction!(confusion::confusion_matrix_boundary, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_instance_to_partial, m)?)?;
    m.add_function(wrap_pyfunction!(merge_instance_partials, m)?)?;
    // ADR-0046 partitioned-eval surface (instance only; LRP variants
    // raise a typed `RuntimeError`, panoptic/semantic partition at
    // the Python level).
    m.add_function(wrap_pyfunction!(
        partition_py::evaluate_bbox_partitioned,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        partition_py::evaluate_segm_partitioned,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        partition_py::evaluate_boundary_partitioned,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        partition_py::evaluate_keypoints_partitioned,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        partition_py::evaluate_bbox_partitioned_lrp,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        partition_py::evaluate_segm_partitioned_lrp,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        partition_py::evaluate_boundary_partitioned_lrp,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        partition_py::evaluate_keypoints_partitioned_lrp,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(partition_py::slices_batch_panoptic, m)?)?;
    m.add_function(wrap_pyfunction!(partition_py::slices_batch_semantic, m)?)?;
    m.add_function(wrap_pyfunction!(partition_py::manifest_to_json_bytes, m)?)?;
    m.add_class::<partition_py::PyPartitionedSummary>()?;
    m.add_class::<partition_py::PyPartitionedLrpReport>()?;
    m.add_class::<PySummary>()?;
    m.add_class::<PyEvalGrid>()?;
    m.add_class::<PyAccumulated>()?;
    m.add_class::<PyDataset>()?;
    m.add_class::<PyBackgroundEvaluator>()?;
    m.add_class::<arrow_helpers::ArrowRecordBatchPy>()?;
    m.add_class::<breakdown::PyBreakdown>()?;
    m.add_class::<calibration::EvalCells>()?;
    m.add("OutOfBudgetError", m.py().get_type::<OutOfBudgetError>())?;
    m.add("QueueFullError", m.py().get_type::<QueueFullError>())?;
    m.add(
        "MemoryBudgetWarning",
        m.py().get_type::<MemoryBudgetWarning>(),
    )?;
    m.add(
        "InvalidAnnotationError",
        m.py().get_type::<InvalidAnnotationError>(),
    )?;
    m.add("NonFiniteError", m.py().get_type::<NonFiniteError>())?;
    m.add(
        "DimensionMismatchError",
        m.py().get_type::<DimensionMismatchError>(),
    )?;
    m.add(
        "InvalidConfigError",
        m.py().get_type::<InvalidConfigError>(),
    )?;
    m.add(
        "PartialFormatMismatch",
        m.py().get_type::<PartialFormatMismatch>(),
    )?;
    m.add(
        "PartialDatasetMismatch",
        m.py().get_type::<PartialDatasetMismatch>(),
    )?;
    m.add(
        "PartialParamsMismatch",
        m.py().get_type::<PartialParamsMismatch>(),
    )?;
    m.add(
        "PartialPartitionOverlap",
        m.py().get_type::<PartialPartitionOverlap>(),
    )?;
    m.add(
        "PartialRankCollision",
        m.py().get_type::<PartialRankCollision>(),
    )?;
    m.add("__version__", vernier_core::VERSION)?;
    panoptic::register(m)?;
    semantic::register(m)?;
    Ok(())
}
