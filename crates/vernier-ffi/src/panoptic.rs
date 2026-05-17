//! PyO3 bindings for [`vernier_panoptic`] (ADR-0025).
//!
//! Surface:
//! - [`PyPanopticDataset`] / [`PyPanopticPredictions`] — frozen
//!   parsed-once handles (mirrors the [`crate::dataset::PyDataset`]
//!   pattern). `from_arrays` builds a handle from a Python dict of
//!   uint32 ndarrays + JSON-string segments_info. A `from_files`
//!   variant (Rust-side `png` decode) is a documented follow-up;
//!   users currently round-trip through Pillow + np.array.
//! - [`PyPanopticSummary`] / [`PyClassPanopticStats`] — read-only
//!   wrappers over the [`vernier_panoptic::PanopticSummary`] /
//!   [`vernier_panoptic::ClassPanopticStats`] result rows.
//! - [`evaluate_panoptic`] — free pyfunction that runs the full
//!   pipeline (kernel + attribute + summarize) under `py.detach`.
//!
//! `PyPanopticDataset::from_arrays` is the **first** uint32 zero-copy
//! ndarray reader in this codebase. The pattern is `PyReadonlyArray2<u32>`
//! per dict value, `.as_array()` to a `numpy::ndarray::ArrayView2<u32>`,
//! `.to_owned()` to materialize a `Vec<u32>` for the dataset (the
//! kernel walks the buffer linearly so the row-major flatten is a
//! single allocation, no copy from the user's array). When boundary-PQ
//! lands and the kernel needs to call `vernier-mask::erode_chebyshev_ball`
//! on the prediction masks, the materialization will move into
//! `from_arrays` so the boundary path can avoid re-allocating.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use numpy::PyReadonlyArray2;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyDictMethods, PyList};
use serde::Deserialize;
use vernier_partial::PartialError;

use crate::arrow_helpers::{wrap_batch, ArrowRecordBatchPy};
use crate::background::BackgroundConfig;
use crate::background_streaming::{
    BackgroundCapable, BackgroundCore, BackgroundLifecycle, SubmitError,
};
use crate::manifest_py::manifest_to_canonical_json;
use crate::numpy_utils::parse_uint32_label_maps;
use crate::tables::{slices_record_batch_panoptic, PanopticSliceRow};
use crate::threads;
use crate::{poll_scheduling_warning, queue_full_to_pyerr, validate_shutdown_timeout};
use std::collections::HashSet;
use vernier_core::manifest::partition_spec_from_manifest;
use vernier_panoptic::attribute::PqStat;
use vernier_panoptic::dataset::{CategoryId, CategoryMeta, ImageEntry, ImageId, SegmentInfo};
use vernier_panoptic::decode::decode_panoptic_png;
use vernier_panoptic::stream::StreamingPanopticEvaluator;
use vernier_panoptic::{
    evaluate_per_image, evaluate_per_image_parallel, evaluate_with_options,
    evaluate_with_options_parallel, fold_per_image, summarize_from_acc_with_options,
    BoundaryConfig, ClassPanopticStats, EvaluateOptions, GroupPanopticStats, PanopticDataset,
    PanopticError, PanopticPredictions, PanopticSummary, ParityMode, SummarizeOptions,
    BOUNDARY_PANOPTIC_DILATION_RATIO_DEFAULT,
};

// ---------------------------------------------------------------------------
// JSON shapes for segments_info + categories. Serde derives keep the
// FFI free of hand-rolled object decoding; the Python side passes
// these as bytes via `json.dumps(...).encode()`.
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct SegmentInfoJson {
    id: u32,
    category_id: i64,
    #[serde(default)]
    iscrowd: serde_json::Value,
    #[serde(default)]
    area: u64,
}

#[derive(Deserialize)]
struct CategoryMetaJson {
    id: i64,
    isthing: serde_json::Value,
}

/// Convert the `iscrowd` / `isthing` field to bool. panopticapi accepts
/// `int` or bool (quirk **S5** for `iscrowd`; the COCO panoptic JSON
/// likewise stores `isthing` as `int 0/1`). We normalize to bool on
/// the FFI boundary.
fn parse_bool_or_int(v: &serde_json::Value) -> bool {
    match v {
        serde_json::Value::Bool(b) => *b,
        serde_json::Value::Number(n) => n.as_i64().is_some_and(|x| x != 0),
        _ => false,
    }
}

fn parse_segments_map(bytes: &[u8]) -> Result<HashMap<ImageId, Vec<SegmentInfo>>, PanopticError> {
    let raw: HashMap<String, Vec<SegmentInfoJson>> = serde_json::from_slice(bytes)?;
    raw.into_iter()
        .map(|(k, v)| {
            let id: ImageId = k.parse().map_err(|_| PanopticError::InvalidInput {
                detail: format!("segments_info key {k:?} is not an integer image id"),
            })?;
            let segments = v
                .into_iter()
                .map(|s| SegmentInfo {
                    id: s.id,
                    category_id: s.category_id,
                    iscrowd: parse_bool_or_int(&s.iscrowd),
                    area: s.area,
                })
                .collect();
            Ok((id, segments))
        })
        .collect()
}

fn parse_categories(bytes: &[u8]) -> Result<HashMap<CategoryId, CategoryMeta>, PanopticError> {
    let raw: Vec<CategoryMetaJson> = serde_json::from_slice(bytes)?;
    Ok(raw
        .into_iter()
        .map(|c| {
            (
                c.id,
                CategoryMeta {
                    id: c.id,
                    isthing: parse_bool_or_int(&c.isthing),
                },
            )
        })
        .collect())
}

// ---------------------------------------------------------------------------
// pyclasses
// ---------------------------------------------------------------------------

/// Parsed-once panoptic ground-truth handle. Build via
/// [`PyPanopticDataset::from_arrays`]; pass to
/// [`evaluate_panoptic`].
#[pyclass(module = "vernier._core", name = "PanopticDataset", frozen)]
pub(crate) struct PyPanopticDataset {
    inner: Arc<PanopticDataset>,
}

#[pymethods]
impl PyPanopticDataset {
    /// Build a dataset from pre-decoded uint32 label maps. `label_maps`
    /// maps image id (int) to a 2-D `numpy.ndarray` of dtype `uint32`
    /// whose pixel values are panoptic segment ids
    /// (`id = R + 256*G + 256²*B`, post-rgb2id). `segments_info` and
    /// `categories` are JSON byte strings.
    #[staticmethod]
    fn from_arrays<'py>(
        py: Python<'py>,
        label_maps: &Bound<'py, PyDict>,
        segments_info: &[u8],
        categories: &[u8],
    ) -> PyResult<Self> {
        let mut maps = parse_uint32_label_maps(label_maps, "panoptic")?;
        let segs = parse_segments_map(segments_info).map_err(|e| panoptic_error_to_pyerr(py, e))?;
        let cats = parse_categories(categories).map_err(|e| panoptic_error_to_pyerr(py, e))?;

        let mut images: HashMap<ImageId, ImageEntry> = HashMap::with_capacity(maps.len());
        for (image_id, segments) in segs {
            let (h, w, label_map) = maps.remove(&image_id).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "panoptic gt: segments_info has image_id={image_id} but label_maps does not"
                ))
            })?;
            let entry = ImageEntry::from_components(image_id, h, w, label_map, segments, "gt")
                .map_err(|e| panoptic_error_to_pyerr(py, e))?;
            images.insert(image_id, entry);
        }

        Ok(Self {
            inner: Arc::new(PanopticDataset::from_components(images, cats)),
        })
    }

    /// Number of images in the dataset.
    #[getter]
    fn num_images(&self) -> usize {
        self.inner.images.len()
    }

    /// Number of categories.
    #[getter]
    fn num_categories(&self) -> usize {
        self.inner.categories.len()
    }

    /// Image ids present in this dataset. Order is unspecified — the
    /// caller must sort if a deterministic order is required.
    fn image_ids(&self) -> Vec<i64> {
        self.inner.images.keys().copied().collect()
    }

    /// Build a new `PanopticDataset` retaining only the entries whose
    /// image id is in `ids` (ADR-0046 §"Panoptic / semantic partitioning").
    ///
    /// Ids absent from this dataset are skipped silently — the
    /// Python-level partition orchestrator has already done the
    /// known-key intersection at manifest parse time. Categories are
    /// preserved verbatim (they are a dataset-level taxonomy, not
    /// per-image).
    fn subset_by_image_ids(&self, ids: Vec<i64>) -> Self {
        let keep: std::collections::HashSet<ImageId> = ids.into_iter().collect();
        let images: HashMap<ImageId, ImageEntry> = self
            .inner
            .images
            .iter()
            .filter(|(id, _)| keep.contains(id))
            .map(|(id, entry)| (*id, entry.clone()))
            .collect();
        Self {
            inner: Arc::new(PanopticDataset::from_components(
                images,
                self.inner.categories.clone(),
            )),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PanopticDataset(images={}, categories={})",
            self.inner.images.len(),
            self.inner.categories.len(),
        )
    }
}

/// Parsed-once panoptic prediction handle. Sibling shape to
/// [`PyPanopticDataset`]; predictions never carry a category taxonomy
/// (quirk **S9**).
#[pyclass(module = "vernier._core", name = "PanopticPredictions", frozen)]
pub(crate) struct PyPanopticPredictions {
    inner: Arc<PanopticPredictions>,
}

#[pymethods]
impl PyPanopticPredictions {
    /// Build predictions from pre-decoded uint32 label maps. Mirrors
    /// [`PyPanopticDataset::from_arrays`] but takes no `categories`.
    /// The DT-side validation runs the full S1/S11 PNG-vs-segments_info
    /// cross-check, recomputes per-segment areas from PNG marginals
    /// (quirk **S3**), and rejects duplicate ids
    /// ([`PanopticError::DuplicateSegmentId`]).
    #[staticmethod]
    fn from_arrays<'py>(
        py: Python<'py>,
        label_maps: &Bound<'py, PyDict>,
        segments_info: &[u8],
    ) -> PyResult<Self> {
        let mut maps = parse_uint32_label_maps(label_maps, "panoptic")?;
        let segs = parse_segments_map(segments_info).map_err(|e| panoptic_error_to_pyerr(py, e))?;

        let mut images: HashMap<ImageId, ImageEntry> = HashMap::with_capacity(maps.len());
        for (image_id, segments) in segs {
            let (h, w, label_map) = maps.remove(&image_id).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "panoptic dt: segments_info has image_id={image_id} but label_maps does not"
                ))
            })?;
            let mut entry = ImageEntry::from_components(image_id, h, w, label_map, segments, "dt")
                .map_err(|e| panoptic_error_to_pyerr(py, e))?;
            // S3: pred area is always overwritten from the PNG. The
            // FFI is the canonical place to do this — kernel and
            // summarize rely on the area field being correct.
            entry.recompute_areas_from_png();
            images.insert(image_id, entry);
        }

        Ok(Self {
            inner: Arc::new(PanopticPredictions::from_components(images)),
        })
    }

    /// Number of images for which we have predictions.
    #[getter]
    fn num_images(&self) -> usize {
        self.inner.images.len()
    }

    /// Total number of DT segments across all images. Used as the
    /// `n_detections` cell on the panoptic slices table (ADR-0046).
    #[getter]
    fn num_segments(&self) -> u64 {
        self.inner
            .images
            .values()
            .map(|e| e.segments.len() as u64)
            .sum()
    }

    /// Image ids present in this predictions handle. Order is
    /// unspecified — the caller must sort if a deterministic order is
    /// required.
    fn image_ids(&self) -> Vec<i64> {
        self.inner.images.keys().copied().collect()
    }

    /// Total number of DT segments across the images in `ids`. Lets
    /// the Python-side partition loop count `n_detections` per slice
    /// without rebuilding the per-slice [`PanopticPredictions`] just
    /// to read its [`Self::num_segments`].
    fn num_segments_for(&self, ids: Vec<i64>) -> u64 {
        let keep: std::collections::HashSet<ImageId> = ids.into_iter().collect();
        self.inner
            .images
            .iter()
            .filter(|(id, _)| keep.contains(id))
            .map(|(_, e)| e.segments.len() as u64)
            .sum()
    }

    /// Build a new `PanopticPredictions` retaining only the entries
    /// whose image id is in `ids` (ADR-0046 §"Panoptic / semantic
    /// partitioning"). Mirrors
    /// [`PyPanopticDataset::subset_by_image_ids`]; predictions do not
    /// carry a category taxonomy (quirk **S9**), so nothing else
    /// needs to be threaded through.
    fn subset_by_image_ids(&self, ids: Vec<i64>) -> Self {
        let keep: std::collections::HashSet<ImageId> = ids.into_iter().collect();
        let images: HashMap<ImageId, ImageEntry> = self
            .inner
            .images
            .iter()
            .filter(|(id, _)| keep.contains(id))
            .map(|(id, entry)| (*id, entry.clone()))
            .collect();
        Self {
            inner: Arc::new(PanopticPredictions::from_components(images)),
        }
    }

    fn __repr__(&self) -> String {
        format!("PanopticPredictions(images={})", self.inner.images.len())
    }
}

/// Per-class PQ row exposed to Python (W8 strict-superset shape).
#[pyclass(
    module = "vernier._core",
    name = "ClassPanopticStats",
    frozen,
    skip_from_py_object
)]
#[derive(Clone, Copy)]
pub(crate) struct PyClassPanopticStats {
    inner: ClassPanopticStats,
}

#[pymethods]
impl PyClassPanopticStats {
    #[getter]
    fn pq(&self) -> f64 {
        self.inner.pq
    }
    #[getter]
    fn sq(&self) -> f64 {
        self.inner.sq
    }
    #[getter]
    fn rq(&self) -> f64 {
        self.inner.rq
    }
    #[getter]
    fn n_tp(&self) -> u64 {
        self.inner.n_tp
    }
    #[getter]
    fn n_fp(&self) -> u64 {
        self.inner.n_fp
    }
    #[getter]
    fn n_fn(&self) -> u64 {
        self.inner.n_fn
    }
    #[getter]
    fn iou_sum(&self) -> f64 {
        self.inner.iou_sum
    }

    fn __repr__(&self) -> String {
        format!(
            "ClassPanopticStats(pq={:.4}, sq={:.4}, rq={:.4}, n_tp={}, n_fp={}, n_fn={}, iou_sum={:.4})",
            self.inner.pq,
            self.inner.sq,
            self.inner.rq,
            self.inner.n_tp,
            self.inner.n_fp,
            self.inner.n_fn,
            self.inner.iou_sum,
        )
    }
}

/// Per-group rollup row (ADR-0042).
///
/// Mirrors [`vernier_panoptic::GroupPanopticStats`] one-to-one. Built
/// only when `class_grouping=` is passed to `evaluate_panoptic`.
#[pyclass(
    module = "vernier._core",
    name = "GroupPanopticStats",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyGroupPanopticStats {
    inner: GroupPanopticStats,
}

#[pymethods]
impl PyGroupPanopticStats {
    #[getter]
    fn label(&self) -> &str {
        &self.inner.label
    }
    #[getter]
    fn member_category_ids(&self) -> Vec<i64> {
        self.inner.member_category_ids.clone()
    }
    #[getter]
    fn pq(&self) -> f64 {
        self.inner.pq
    }
    #[getter]
    fn sq(&self) -> f64 {
        self.inner.sq
    }
    #[getter]
    fn rq(&self) -> f64 {
        self.inner.rq
    }
    #[getter]
    fn n(&self) -> usize {
        self.inner.n
    }

    fn __repr__(&self) -> String {
        format!(
            "GroupPanopticStats(label={:?}, pq={:.4}, sq={:.4}, rq={:.4}, n={})",
            self.inner.label, self.inner.pq, self.inner.sq, self.inner.rq, self.inner.n,
        )
    }
}

/// Top-level panoptic evaluation result. Read via field accessors.
#[pyclass(module = "vernier._core", name = "PanopticSummary", frozen)]
pub(crate) struct PyPanopticSummary {
    inner: PanopticSummary,
}

#[pymethods]
impl PyPanopticSummary {
    #[getter]
    fn pq(&self) -> f64 {
        self.inner.pq
    }
    #[getter]
    fn sq(&self) -> f64 {
        self.inner.sq
    }
    #[getter]
    fn rq(&self) -> f64 {
        self.inner.rq
    }
    #[getter]
    fn pq_things(&self) -> Option<f64> {
        self.inner.pq_things
    }
    #[getter]
    fn sq_things(&self) -> Option<f64> {
        self.inner.sq_things
    }
    #[getter]
    fn rq_things(&self) -> Option<f64> {
        self.inner.rq_things
    }
    #[getter]
    fn pq_stuff(&self) -> Option<f64> {
        self.inner.pq_stuff
    }
    #[getter]
    fn sq_stuff(&self) -> Option<f64> {
        self.inner.sq_stuff
    }
    #[getter]
    fn rq_stuff(&self) -> Option<f64> {
        self.inner.rq_stuff
    }
    #[getter]
    fn n(&self) -> usize {
        self.inner.n
    }
    #[getter]
    fn n_things(&self) -> Option<usize> {
        self.inner.n_things
    }
    #[getter]
    fn n_stuff(&self) -> Option<usize> {
        self.inner.n_stuff
    }

    /// Per-class rows keyed by category id. Returns a Python `dict`
    /// (constructed fresh on each call from the underlying BTreeMap).
    fn per_class<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (cat, row) in &self.inner.per_class {
            dict.set_item(*cat, PyClassPanopticStats { inner: *row })?;
        }
        Ok(dict)
    }

    /// Per-group rollup keyed by group label (ADR-0042). Empty when
    /// the evaluator was run without `class_grouping`. Returns a fresh
    /// dict on each call.
    fn per_group<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (label, row) in &self.inner.per_group {
            dict.set_item(label, PyGroupPanopticStats { inner: row.clone() })?;
        }
        Ok(dict)
    }

    /// `dict[str, dict | None]` matching panopticapi's `pq_compute`
    /// return shape exactly when `strict=True` (the count fields are
    /// dropped from per_class to match W8). Useful for round-tripping
    /// with downstream tools that expect the upstream dict layout.
    #[pyo3(signature = (*, strict=false))]
    fn to_dict<'py>(&self, py: Python<'py>, strict: bool) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);

        let bucket = |pq: f64, sq: f64, rq: f64, n: usize| -> PyResult<Bound<'py, PyDict>> {
            let d = PyDict::new(py);
            d.set_item("pq", pq)?;
            d.set_item("sq", sq)?;
            d.set_item("rq", rq)?;
            d.set_item("n", n)?;
            Ok(d)
        };

        out.set_item(
            "All",
            bucket(self.inner.pq, self.inner.sq, self.inner.rq, self.inner.n)?,
        )?;
        match (
            self.inner.pq_things,
            self.inner.sq_things,
            self.inner.rq_things,
            self.inner.n_things,
        ) {
            (Some(p), Some(s), Some(r), Some(n)) => {
                out.set_item("Things", bucket(p, s, r, n)?)?;
            }
            _ => out.set_item("Things", py.None())?,
        }
        match (
            self.inner.pq_stuff,
            self.inner.sq_stuff,
            self.inner.rq_stuff,
            self.inner.n_stuff,
        ) {
            (Some(p), Some(s), Some(r), Some(n)) => {
                out.set_item("Stuff", bucket(p, s, r, n)?)?;
            }
            _ => out.set_item("Stuff", py.None())?,
        }

        let per_class = PyDict::new(py);
        for (cat, row) in &self.inner.per_class {
            let row_dict = PyDict::new(py);
            row_dict.set_item("pq", row.pq)?;
            row_dict.set_item("sq", row.sq)?;
            row_dict.set_item("rq", row.rq)?;
            if !strict {
                row_dict.set_item("n_tp", row.n_tp)?;
                row_dict.set_item("n_fp", row.n_fp)?;
                row_dict.set_item("n_fn", row.n_fn)?;
            }
            per_class.set_item(*cat, row_dict)?;
        }
        out.set_item("per_class", per_class)?;
        Ok(out)
    }

    fn __repr__(&self) -> String {
        format!(
            "PanopticSummary(pq={:.4}, sq={:.4}, rq={:.4}, n={})",
            self.inner.pq, self.inner.sq, self.inner.rq, self.inner.n
        )
    }
}

impl PyPanopticSummary {
    pub(crate) fn summary_ref(&self) -> &PanopticSummary {
        &self.inner
    }
}

// ---------------------------------------------------------------------------
// Free pyfunctions
// ---------------------------------------------------------------------------

/// Run the full panoptic evaluation (kernel + attribute + summarize)
/// against pre-built [`PyPanopticDataset`] / [`PyPanopticPredictions`].
/// Drops the GIL via `py.detach` for the duration of the kernel walk
/// (ADR-0006).
#[pyfunction]
#[pyo3(signature = (
    gt,
    dt,
    parity_mode,
    things_stuff_split=true,
    *,
    pq_iou_threshold=None,
    category_filter=None,
    class_grouping=None,
    stuff_thing_partition=None,
    boundary=false,
    dilation_ratio=BOUNDARY_PANOPTIC_DILATION_RATIO_DEFAULT,
    num_threads=None,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_panoptic(
    py: Python<'_>,
    gt: &PyPanopticDataset,
    dt: &PyPanopticPredictions,
    parity_mode: &str,
    things_stuff_split: bool,
    pq_iou_threshold: Option<f64>,
    category_filter: Option<Vec<u32>>,
    class_grouping: Option<Vec<(String, Vec<u32>)>>,
    stuff_thing_partition: Option<(Vec<u32>, Vec<u32>)>,
    boundary: bool,
    dilation_ratio: f64,
    num_threads: Option<usize>,
) -> PyResult<PyPanopticSummary> {
    let mode = parse_panoptic_parity_mode(parity_mode)?;
    let boundary_cfg = boundary_cfg_from_ffi(boundary, dilation_ratio, mode)?;
    // Resolve `num_threads` under the GIL so the env-var / re-entry
    // `UserWarning` can fire to Python before we detach.
    let thread_policy = threads::resolve_threads(py, num_threads);
    let gt_arc = Arc::clone(&gt.inner);
    let dt_arc = Arc::clone(&dt.inner);
    // Build the isthing-overridden categories map up front so the
    // closure captures an owned value with the right lifetime.
    // ADR-0042 §"stuff_thing_partition": user-supplied membership
    // wins over dataset-derived `isthing` flags.
    let categories_override = stuff_thing_partition.map(|(stuff, things)| {
        let mut overridden = gt_arc.categories.clone();
        for cat in stuff {
            overridden
                .entry(CategoryId::from(cat))
                .and_modify(|m| m.isthing = false);
        }
        for cat in things {
            overridden
                .entry(CategoryId::from(cat))
                .and_modify(|m| m.isthing = true);
        }
        overridden
    });
    let summary = py
        .detach(move || {
            let summarize_opts = SummarizeOptions {
                category_filter: category_filter.as_deref(),
                class_groups: class_grouping.as_deref(),
            };
            let opts = EvaluateOptions {
                pq_iou_threshold,
                categories_override: categories_override.as_ref(),
                summarize: summarize_opts,
                boundary: boundary_cfg,
            };
            run_panoptic_with_policy(
                &gt_arc,
                &dt_arc,
                mode,
                things_stuff_split,
                &opts,
                thread_policy,
            )
        })
        .map_err(|e| panoptic_error_to_pyerr(py, e))?;
    Ok(PyPanopticSummary { inner: summary })
}

/// Dispatch between sequential [`evaluate_with_options`] and the
/// parallel sibling [`evaluate_with_options_parallel`] (ADR-0047).
/// `Sequential` is the zero-overhead path — no rayon symbol entered.
/// `Pool(n)` builds a scoped per-call `rayon::ThreadPool` of exactly
/// `n` threads, `install`s the parallel evaluator inside it, and
/// drops the pool on return.
fn run_panoptic_with_policy(
    gt: &PanopticDataset,
    dt: &PanopticPredictions,
    mode: ParityMode,
    things_stuff_split: bool,
    opts: &EvaluateOptions<'_>,
    thread_policy: threads::ThreadPolicy,
) -> Result<PanopticSummary, PanopticError> {
    match thread_policy.thread_count() {
        None => evaluate_with_options(gt, dt, mode, things_stuff_split, opts),
        Some(n) => {
            let pool = threads::build_scoped_pool(n).map_err(|detail| {
                PanopticError::Partial(vernier_partial::PartialError::Format {
                    kind: vernier_partial::PartialFormatErrorKind::Internal { detail },
                })
            })?;
            pool.install(|| evaluate_with_options_parallel(gt, dt, mode, things_stuff_split, opts))
        }
    }
}

// ---------------------------------------------------------------------------
// Partitioned panoptic eval (ADR-0046 C3).
//
// Mirrors the instance-AP `evaluate_*_partitioned` shape: one matching
// pass, then N+1 cheap summarize passes (one for `overall` plus one
// per slice). The per-image accumulator deltas are retained once and
// then folded under different image-id filters at summarize time —
// the image-axis analogue of ADR-0026's K-axis subset-at-summarize.
// ---------------------------------------------------------------------------

/// Test-only call counter for [`evaluate_per_image`] invocations.
///
/// The Python perf test asserts the matching pass runs **exactly
/// once** regardless of slice count. Wrapping the call in a counter
/// here is the cheapest way to surface the invariant to Python —
/// timing-based assertions are flaky on shared CI. Production builds
/// pay zero overhead because the counter is gated behind
/// `#[cfg(any(test, feature = "_test-counter"))]`. The Python side
/// gates its assertion the same way.
#[cfg(any(test, feature = "_test-counter"))]
static PANOPTIC_MATCHING_PASS_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

#[cfg(any(test, feature = "_test-counter"))]
fn inc_panoptic_matching_count() {
    PANOPTIC_MATCHING_PASS_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(not(any(test, feature = "_test-counter")))]
fn inc_panoptic_matching_count() {}

/// Reset the panoptic matching-pass counter and return the previous
/// value. Test-only.
#[cfg(any(test, feature = "_test-counter"))]
#[pyfunction]
pub(crate) fn _test_reset_panoptic_matching_count() -> u64 {
    PANOPTIC_MATCHING_PASS_COUNT.swap(0, std::sync::atomic::Ordering::Relaxed)
}

/// Read the current panoptic matching-pass counter without resetting.
/// Test-only.
#[cfg(any(test, feature = "_test-counter"))]
#[pyfunction]
pub(crate) fn _test_read_panoptic_matching_count() -> u64 {
    PANOPTIC_MATCHING_PASS_COUNT.load(std::sync::atomic::Ordering::Relaxed)
}

/// Slice metric row carried over the FFI boundary from the C3
/// partition orchestrator. Kept private — the Python lane reads the
/// `slices` Arrow batch on [`PyPartitionedPanopticReport`].
struct PanopticSliceMetrics {
    axis: String,
    value: String,
    n_images: u64,
    n_detections: u64,
    pq: f64,
    sq: f64,
    rq: f64,
}

/// Result of an ADR-0046 C3 partitioned panoptic eval.
///
/// `overall` is bit-identical to a non-partitioned
/// `evaluate_panoptic` over the same handles — the ADR-0046
/// load-bearing parity contract. `slices_capsule()` exposes the
/// per-`(axis, value)` cell metrics as an Arrow `RecordBatch` for
/// zero-copy hand-off to polars / pandas / pyarrow.
#[pyclass(module = "vernier._core", name = "PartitionedPanopticReport", frozen)]
pub(crate) struct PyPartitionedPanopticReport {
    summary: PanopticSummary,
    overall_n_images: u64,
    overall_n_detections: u64,
    slice_rows: Vec<PanopticSliceRow>,
}

#[pymethods]
impl PyPartitionedPanopticReport {
    /// Bit-identical to a non-partitioned `evaluate_panoptic` over the
    /// same handles.
    #[getter]
    fn overall(&self) -> PyPanopticSummary {
        PyPanopticSummary {
            inner: self.summary.clone(),
        }
    }

    /// Dataset image count behind `overall`.
    #[getter]
    fn overall_n_images(&self) -> u64 {
        self.overall_n_images
    }

    /// DT segment count behind `overall`.
    #[getter]
    fn overall_n_detections(&self) -> u64 {
        self.overall_n_detections
    }

    /// Number of `(axis, value)` cells in the partition.
    #[getter]
    fn n_slices(&self) -> usize {
        self.slice_rows.len()
    }

    /// Arrow `RecordBatch` of per-slice rows. Schema matches the C1
    /// path's `slices_batch_panoptic` output verbatim, so the Python
    /// wrapper's DataFrame coercion is identical across the C1 / C3
    /// transition. Built fresh per call (cheap — slice count is
    /// SLICES_CAP-bounded at 256 max).
    fn slices_capsule(&self) -> PyResult<ArrowRecordBatchPy> {
        let batch = slices_record_batch_panoptic(&self.slice_rows)
            .map_err(|e| PyValueError::new_err(format!("arrow build failed: {e}")))?;
        Ok(wrap_batch(batch))
    }

    fn __repr__(&self) -> String {
        format!(
            "PartitionedPanopticReport(overall_n_images={}, overall_n_detections={}, n_slices={})",
            self.overall_n_images,
            self.overall_n_detections,
            self.slice_rows.len(),
        )
    }
}

use crate::partition_py::warn_about_manifest;

/// C3 partitioned panoptic eval (ADR-0046 §"Performance").
///
/// Runs `evaluate_per_image` exactly **once** to retain per-image
/// per-category accumulator deltas, then folds + summarizes those
/// deltas under (a) no filter for `overall` and (b) each slice's
/// image-id set for the per-slice rows. The matching pass is never
/// re-run per slice — the load-bearing C3 axiom. Empty slices
/// (legal in the partition spec for `__unassigned__` buckets) emit a
/// zero-valued row instead of routing through the kernel's empty-set
/// rejection (same convention as the C1 fallback path).
#[pyfunction]
#[pyo3(signature = (
    gt,
    dt,
    parity_mode,
    things_stuff_split,
    boundary,
    dilation_ratio,
    manifest,
    cross_axes = None,
    key_kind = "image_id",
    *,
    num_threads = None,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_panoptic_partitioned(
    py: Python<'_>,
    gt: &PyPanopticDataset,
    dt: &PyPanopticPredictions,
    parity_mode: &str,
    things_stuff_split: bool,
    boundary: bool,
    dilation_ratio: f64,
    manifest: &Bound<'_, PyAny>,
    cross_axes: Option<Vec<Vec<String>>>,
    key_kind: &str,
    num_threads: Option<usize>,
) -> PyResult<PyPartitionedPanopticReport> {
    let mode = parse_panoptic_parity_mode(parity_mode)?;
    let boundary_cfg = boundary_cfg_from_ffi(boundary, dilation_ratio, mode)?;
    let thread_policy = threads::resolve_threads(py, num_threads);

    let manifest_bytes = manifest_to_canonical_json(py, manifest, key_kind)?;
    let cross = cross_axes.unwrap_or_default();

    // Build image_id -> idx map in id-ascending order (the canonical
    // I-axis order used elsewhere). The spec builder requires it to
    // resolve `image_indices`; the C3 partition path itself filters
    // on `image_ids` (the manifest's primary key) directly, but the
    // spec builder shares the resolution with the instance lane.
    //
    // `vernier_panoptic::ImageId = i64` and
    // `vernier_core::dataset::ImageId` is the i64-wrapper struct; we
    // lift through the wrapper for the spec builder and convert back
    // when materializing the per-slice id sets below.
    let mut sorted_image_ids: Vec<ImageId> = gt.inner.images.keys().copied().collect();
    sorted_image_ids.sort_unstable();
    let image_id_to_idx: HashMap<vernier_core::dataset::ImageId, usize> = sorted_image_ids
        .iter()
        .enumerate()
        .map(|(i, id)| (vernier_core::dataset::ImageId(*id), i))
        .collect();

    let (spec, warnings) = partition_spec_from_manifest(&manifest_bytes, &image_id_to_idx, &cross)
        .map_err(|e| PyValueError::new_err(format!("manifest resolution failed: {e}")))?;
    warn_about_manifest(py, &warnings)?;

    // Pre-compute per-slice (image_ids, n_segments) on the FFI thread
    // — the data is on the Python side anyway via the bound handles.
    let total_segments: u64 = dt
        .inner
        .images
        .values()
        .map(|e| e.segments.len() as u64)
        .sum();
    let n_images_overall: u64 = gt.inner.images.len() as u64;
    let mut slice_inputs: Vec<(String, String, HashSet<ImageId>, u64, u64)> =
        Vec::with_capacity(spec.slices.len());
    for sl in &spec.slices {
        // `Slice::image_ids` holds `vernier_core::dataset::ImageId`
        // (i64-wrapper). Convert into the panoptic-side ImageId alias.
        let panoptic_ids: HashSet<ImageId> = sl.image_ids.iter().map(|id| id.0).collect();
        let n_images = panoptic_ids.len() as u64;
        let n_segments: u64 = dt
            .inner
            .images
            .iter()
            .filter(|(id, _)| panoptic_ids.contains(id))
            .map(|(_, e)| e.segments.len() as u64)
            .sum();
        slice_inputs.push((
            sl.axis.clone(),
            sl.value.clone(),
            panoptic_ids,
            n_images,
            n_segments,
        ));
    }

    let gt_arc = Arc::clone(&gt.inner);
    let dt_arc = Arc::clone(&dt.inner);
    let categories = gt_arc.categories.clone();

    type PerImageVec = Vec<(ImageId, HashMap<CategoryId, PqStat>)>;
    type C3Out = (PanopticSummary, Vec<PanopticSliceMetrics>);

    let (overall_summary, slice_metrics) = py
        .detach(move || -> Result<C3Out, PanopticError> {
            inc_panoptic_matching_count();
            let opts = EvaluateOptions {
                pq_iou_threshold: None,
                categories_override: None,
                summarize: SummarizeOptions::default(),
                boundary: boundary_cfg,
            };
            // ADR-0047: route the matching pass through the parallel
            // sibling when `num_threads > 1`; otherwise stay on the
            // sequential path. The per-image deltas are sorted by
            // image_id in both cases, so the downstream fold + per-
            // slice folds are bit-equal across thread counts.
            let per_image: PerImageVec = match thread_policy.thread_count() {
                None => evaluate_per_image(&gt_arc, &dt_arc, mode, &opts)?,
                Some(n) => {
                    let pool = threads::build_scoped_pool(n).map_err(|detail| {
                        PanopticError::Partial(vernier_partial::PartialError::Format {
                            kind: vernier_partial::PartialFormatErrorKind::Internal { detail },
                        })
                    })?;
                    pool.install(|| evaluate_per_image_parallel(&gt_arc, &dt_arc, mode, &opts))?
                }
            };

            // Overall — un-filtered fold + summarize.
            let acc_all = fold_per_image(&per_image, None);
            let overall = summarize_from_acc_with_options(
                acc_all,
                &categories,
                mode,
                things_stuff_split,
                &SummarizeOptions::default(),
            )?;

            // Per-slice — N folds under each slice's image-id filter.
            // Empty slices (legal for __unassigned__ buckets) emit a
            // zero-valued row instead of routing through the kernel's
            // empty-filter rejection.
            let mut metrics: Vec<PanopticSliceMetrics> = Vec::with_capacity(slice_inputs.len());
            for (axis, value, ids, n_images, n_segments) in slice_inputs {
                if ids.is_empty() {
                    metrics.push(PanopticSliceMetrics {
                        axis,
                        value,
                        n_images: 0,
                        n_detections: 0,
                        pq: 0.0,
                        sq: 0.0,
                        rq: 0.0,
                    });
                    continue;
                }
                let acc = fold_per_image(&per_image, Some(&ids));
                let summary = summarize_from_acc_with_options(
                    acc,
                    &categories,
                    mode,
                    things_stuff_split,
                    &SummarizeOptions::default(),
                )?;
                metrics.push(PanopticSliceMetrics {
                    axis,
                    value,
                    n_images,
                    n_detections: n_segments,
                    pq: summary.pq,
                    sq: summary.sq,
                    rq: summary.rq,
                });
            }
            Ok((overall, metrics))
        })
        .map_err(|e| panoptic_error_to_pyerr(py, e))?;

    let slice_rows: Vec<PanopticSliceRow> = slice_metrics
        .into_iter()
        .map(|m| PanopticSliceRow {
            axis: m.axis,
            value: m.value,
            n_images: m.n_images,
            n_detections: m.n_detections,
            pq: m.pq,
            sq: m.sq,
            rq: m.rq,
        })
        .collect();
    Ok(PyPartitionedPanopticReport {
        summary: overall_summary,
        overall_n_images: n_images_overall,
        overall_n_detections: total_segments,
        slice_rows,
    })
}

// ---------------------------------------------------------------------------
// Error mapping
// ---------------------------------------------------------------------------

/// Map a [`PanopticError`] to a Python exception. The `Partial`
/// variant routes through [`crate::partial_error_to_pyerr`] so the
/// five distributed-eval exception classes are shared with the
/// instance and semantic paradigms (`vernier.panoptic.PartialFormatMismatch
/// is vernier.instance.PartialFormatMismatch`). Other variants
/// surface as `PyValueError` with the structured fields embedded in
/// the message body so the parity harness can lift them
/// programmatically.
fn panoptic_error_to_pyerr(py: Python<'_>, e: PanopticError) -> PyErr {
    match e {
        PanopticError::Partial(inner) => return crate::partial_error_to_pyerr(py, &inner),
        PanopticError::DuplicateImageId { image_id } => {
            return PyValueError::new_err(format!(
                "duplicate panoptic image_id={image_id} in streaming evaluator"
            ));
        }
        _ => {}
    }
    match e {
        PanopticError::ShapeMismatch {
            image_id,
            gt_shape,
            dt_shape,
        } => PyValueError::new_err(format!(
            "panoptic shape mismatch on image_id={image_id}: gt={gt_shape:?}, dt={dt_shape:?}"
        )),
        PanopticError::UnknownPredSegmentId { image_id, segment_id } => PyValueError::new_err(
            format!("unknown panoptic prediction segment id {segment_id} on image_id={image_id}"),
        ),
        PanopticError::MissingPredSegmentInPng {
            image_id,
            segment_id,
        } => PyValueError::new_err(format!(
            "panoptic prediction segment id {segment_id} declared in segments_info missing from PNG (image_id={image_id})"
        )),
        PanopticError::DuplicateSegmentId {
            image_id,
            segment_id,
            side,
        } => PyValueError::new_err(format!(
            "duplicate panoptic segment id {segment_id} on image_id={image_id} (side={side})"
        )),
        PanopticError::MissingPanopticImage { image_id, path } => {
            PyValueError::new_err(format!("missing panoptic PNG for image_id={image_id}: {path}"))
        }
        PanopticError::MissingPredictionsForImage { image_id } => {
            PyValueError::new_err(format!("missing panoptic prediction for image_id={image_id}"))
        }
        PanopticError::NonRgbPng { image_id, mode } => PyValueError::new_err(format!(
            "panoptic PNG for image_id={image_id} is not RGB (mode={mode})"
        )),
        PanopticError::EmptyCategoryFilter { context } => PyValueError::new_err(format!(
            "panoptic category filter for {context} is empty (W6 strict)"
        )),
        PanopticError::Json(e) => PyValueError::new_err(format!("panoptic JSON parse error: {e}")),
        PanopticError::Png(s) => PyValueError::new_err(format!("panoptic PNG decode error: {s}")),
        PanopticError::InvalidInput { detail } => {
            PyValueError::new_err(format!("invalid panoptic input: {detail}"))
        }
        // PanopticError is non_exhaustive; future variants surface
        // as a generic ValueError with the Display message.
        other => PyValueError::new_err(format!("{other}")),
    }
}

// ---------------------------------------------------------------------------
// Streaming evaluator (ADR-0028 §"Streaming"; ADR-0032 distributed merge)
// ---------------------------------------------------------------------------

// Local copy of the parser: ADR-0025 firewalls `vernier-panoptic` from
// `vernier-core` (no edge in either direction), so `ParityMode` here is
// `vernier_panoptic::ParityMode` — a distinct nominal type from the
// canonical `vernier_core::ParityMode` that `crate::parse_parity_mode`
// returns. Sharing the parser would force adapter code that breaks the
// firewall.
fn parse_panoptic_parity_mode(s: &str) -> PyResult<ParityMode> {
    match s {
        "strict" => Ok(ParityMode::Strict),
        "corrected" => Ok(ParityMode::Corrected),
        other => Err(PyValueError::new_err(format!(
            "unknown panoptic parity_mode {other:?}; expected 'strict' or 'corrected'"
        ))),
    }
}

/// Validate `(boundary, dilation_ratio)` from the FFI and lift into an
/// `Option<BoundaryConfig>`. Centralizes the keyword-pair contract
/// shared by all four panoptic entry points (evaluate, to_partial,
/// merge_partials, BackgroundPanopticEvaluator.__init__).
fn boundary_cfg_from_ffi(
    boundary: bool,
    dilation_ratio: f64,
    parity_mode: ParityMode,
) -> PyResult<Option<BoundaryConfig>> {
    if !boundary {
        return Ok(None);
    }
    if !dilation_ratio.is_finite() || dilation_ratio <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "dilation_ratio must be a positive finite float when boundary=True, got {dilation_ratio}"
        )));
    }
    Ok(Some(BoundaryConfig {
        dilation_ratio,
        parity_mode,
    }))
}

/// Build an [`ImageEntry`] from per-image FFI inputs: a 2-D uint32
/// label map ndarray plus the JSON segments_info bytes for that one
/// image. Centralizes the FFI boundary conversion that
/// [`PyPanopticDataset::from_arrays`] does in bulk; the streaming
/// evaluator's `update` path uses this to consume one image at a
/// time.
fn build_image_entry<'py>(
    py: Python<'py>,
    image_id: ImageId,
    label_map: &PyReadonlyArray2<'py, u32>,
    segments_info_bytes: &[u8],
    side: &'static str,
) -> PyResult<ImageEntry> {
    let view = label_map.as_array();
    let (h, w) = (view.shape()[0] as u32, view.shape()[1] as u32);
    let buf: Vec<u32> = view.iter().copied().collect();
    let segs: Vec<SegmentInfoJson> = serde_json::from_slice(segments_info_bytes).map_err(|e| {
        panoptic_error_to_pyerr(
            py,
            PanopticError::InvalidInput {
                detail: format!("segments_info JSON for image_id={image_id} is invalid: {e}"),
            },
        )
    })?;
    let segments: Vec<SegmentInfo> = segs
        .into_iter()
        .map(|s| SegmentInfo {
            id: s.id,
            category_id: s.category_id,
            iscrowd: parse_bool_or_int(&s.iscrowd),
            area: s.area,
        })
        .collect();
    let mut entry = ImageEntry::from_components(image_id, h, w, buf, segments, side)
        .map_err(|e| panoptic_error_to_pyerr(py, e))?;
    if side == "dt" {
        // S3: pred area is always overwritten from the PNG marginal,
        // matching the batch `PyPanopticPredictions::from_arrays` path.
        // Without this, the streaming / background submit() path would
        // silently use the JSON `area` field — currently OK on COCO
        // because panopticapi's writers ship `area == PNG marginal`,
        // but a latent spec gap.
        entry.recompute_areas_from_png();
    }
    Ok(entry)
}

/// Build an `ImageEntry` directly from a panoptic PNG byte blob —
/// fuses libpng decode, the RGB→id pass, and (DT side) the S3 area
/// recompute + S1/S11 validation in a single walk over the decoded
/// pixels. The companion to [`build_image_entry`] on the
/// `submit`/`update` side; both produce equivalent `ImageEntry`s,
/// this one bypasses the Pillow → numpy → uint32 round-trip the
/// Python wrapper would otherwise drive on the main thread.
/// Parse the `segments_info` JSON only — used by `submit_png` to
/// surface schema errors synchronously on the FFI thread while
/// deferring the PNG decode itself to the worker pool inside
/// `apply_update_parallel`.
fn parse_segments_info(
    py: Python<'_>,
    image_id: ImageId,
    segments_info_bytes: &[u8],
) -> PyResult<Vec<SegmentInfo>> {
    let segs: Vec<SegmentInfoJson> = serde_json::from_slice(segments_info_bytes).map_err(|e| {
        panoptic_error_to_pyerr(
            py,
            PanopticError::InvalidInput {
                detail: format!("segments_info JSON for image_id={image_id} is invalid: {e}"),
            },
        )
    })?;
    Ok(segs
        .into_iter()
        .map(|s| SegmentInfo {
            id: s.id,
            category_id: s.category_id,
            iscrowd: parse_bool_or_int(&s.iscrowd),
            area: s.area,
        })
        .collect())
}

/// Unpack one 5-tuple `(image_id, gt_label_map, gt_segments_info,
/// dt_label_map, dt_segments_info)` from the Python side into an
/// owned `(ImageId, gt ImageEntry, dt ImageEntry)`. Shared between
/// the sequential and parallel paths of [`evaluate_panoptic_to_partial`]
/// — the unpack is identical, only the downstream dispatch differs.
fn panoptic_unpack_image_tuple_array<'py>(
    py: Python<'py>,
    item: &Bound<'py, PyAny>,
) -> PyResult<(ImageId, ImageEntry, ImageEntry)> {
    let tup = item.cast::<pyo3::types::PyTuple>().map_err(|_| {
        PyValueError::new_err(
            "evaluate_panoptic_to_partial expects a list of 5-tuples \
             (image_id, gt_label_map, gt_segments_info, dt_label_map, dt_segments_info)",
        )
    })?;
    if tup.len() != 5 {
        return Err(PyValueError::new_err(format!(
            "evaluate_panoptic_to_partial expects 5-tuples; got {}-tuple",
            tup.len()
        )));
    }
    let image_id: ImageId = tup.get_item(0)?.extract()?;
    let gt_label_map: PyReadonlyArray2<'py, u32> = tup.get_item(1)?.extract()?;
    let gt_segs_bytes: Vec<u8> = tup.get_item(2)?.extract()?;
    let dt_label_map: PyReadonlyArray2<'py, u32> = tup.get_item(3)?.extract()?;
    let dt_segs_bytes: Vec<u8> = tup.get_item(4)?.extract()?;
    let gt_entry = build_image_entry(py, image_id, &gt_label_map, &gt_segs_bytes, "gt")?;
    let dt_entry = build_image_entry(py, image_id, &dt_label_map, &dt_segs_bytes, "dt")?;
    Ok((image_id, gt_entry, dt_entry))
}

/// One-shot per-rank streaming submit + serialize partial (ADR-0035).
///
/// Functionally equivalent to constructing a `StreamingPanopticEvaluator`,
/// calling `update` per image, then `finalize_to_partial`. Each
/// per-image entry in ``images`` is a 5-tuple
/// ``(image_id, gt_label_map, gt_segments_info, dt_label_map, dt_segments_info)``
/// matching the streaming substrate's `update` signature.
#[pyfunction]
#[pyo3(signature = (
    images,
    categories,
    parity_mode,
    rank_id,
    *,
    things_stuff_split = true,
    retain_per_image_deltas = false,
    boundary = false,
    dilation_ratio = BOUNDARY_PANOPTIC_DILATION_RATIO_DEFAULT,
    num_threads = None,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_panoptic_to_partial<'py>(
    py: Python<'py>,
    images: &Bound<'py, PyList>,
    categories: &[u8],
    parity_mode: &str,
    rank_id: u32,
    things_stuff_split: bool,
    retain_per_image_deltas: bool,
    boundary: bool,
    dilation_ratio: f64,
    num_threads: Option<usize>,
) -> PyResult<Bound<'py, PyBytes>> {
    let mode = parse_panoptic_parity_mode(parity_mode)?;
    let cats = parse_categories(categories).map_err(|e| panoptic_error_to_pyerr(py, e))?;
    let boundary_cfg = boundary_cfg_from_ffi(boundary, dilation_ratio, mode)?;
    let thread_policy = threads::resolve_threads(py, num_threads);
    let mut ev =
        StreamingPanopticEvaluator::new(cats, mode, things_stuff_split, retain_per_image_deltas)
            .with_rank(rank_id)
            .map_err(|e| panoptic_error_to_pyerr(py, e))?;
    if let Some(cfg) = boundary_cfg {
        ev = ev
            .with_boundary(cfg)
            .map_err(|e| panoptic_error_to_pyerr(py, e))?;
    }

    match thread_policy.thread_count() {
        None => {
            // Sequential streaming path: process per image so we never
            // hold the full label-map corpus in memory at once (PR #187
            // streaming-runner property; the eager Vec<ImageEntry>
            // form regressed from ~120 MiB to ~12 GiB on COCO panoptic
            // val).
            for item in images.iter() {
                let (image_id, gt_entry, dt_entry) = panoptic_unpack_image_tuple_array(py, &item)?;
                py.detach(|| ev.update(image_id, &gt_entry, &dt_entry))
                    .map_err(|e| panoptic_error_to_pyerr(py, e))?;
            }
        }
        Some(n) => {
            // Parallel path: build the full per-image owned vector
            // under GIL (validation + numpy conversion happens here),
            // then dispatch the matching pass under a scoped pool.
            // The memory regression PR #187 guarded against is for the
            // sequential path's "submit eagerly" mode; the parallel
            // path explicitly opts in to "hold the batch", paying the
            // RAM cost in exchange for the throughput.
            let mut batch: Vec<(ImageId, ImageEntry, ImageEntry)> =
                Vec::with_capacity(images.len());
            for item in images.iter() {
                let triple = panoptic_unpack_image_tuple_array(py, &item)?;
                batch.push(triple);
            }
            let pool = threads::build_scoped_pool(n).map_err(|detail| {
                PyValueError::new_err(format!("rayon pool build failed: {detail}"))
            })?;
            py.detach(|| pool.install(|| ev.update_parsed_parallel(batch)))
                .map_err(|e| panoptic_error_to_pyerr(py, e))?;
        }
    }

    let bytes = py
        .detach(move || ev.finalize_to_partial())
        .map_err(|e| panoptic_error_to_pyerr(py, e))?;
    Ok(PyBytes::new(py, &bytes))
}

/// Merge per-rank partials into a final summary (ADR-0035).
#[pyfunction]
#[pyo3(signature = (
    categories,
    partials,
    parity_mode,
    *,
    things_stuff_split = true,
    retain_per_image_deltas = false,
    boundary = false,
    dilation_ratio = BOUNDARY_PANOPTIC_DILATION_RATIO_DEFAULT,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn merge_panoptic_partials<'py>(
    py: Python<'py>,
    categories: &[u8],
    partials: &Bound<'py, PyList>,
    parity_mode: &str,
    things_stuff_split: bool,
    retain_per_image_deltas: bool,
    boundary: bool,
    dilation_ratio: f64,
) -> PyResult<PyPanopticSummary> {
    let mode = parse_panoptic_parity_mode(parity_mode)?;
    let cats = parse_categories(categories).map_err(|e| panoptic_error_to_pyerr(py, e))?;
    let boundary_cfg = boundary_cfg_from_ffi(boundary, dilation_ratio, mode)?;
    let owned: Vec<Vec<u8>> = partials
        .iter()
        .map(|item| {
            item.cast::<PyBytes>()
                .map_err(|_| {
                    PyValueError::new_err("merge_panoptic_partials expects a list of bytes objects")
                })
                .map(|b| b.as_bytes().to_vec())
        })
        .collect::<PyResult<_>>()?;
    let summary = py
        .detach(move || -> Result<PanopticSummary, PanopticError> {
            let slices: Vec<&[u8]> = owned.iter().map(|v| v.as_slice()).collect();
            let merged = StreamingPanopticEvaluator::from_partials_with_boundary(
                cats,
                mode,
                things_stuff_split,
                retain_per_image_deltas,
                boundary_cfg,
                &slices,
            )?;
            merged.finalize()
        })
        .map_err(|e| panoptic_error_to_pyerr(py, e))?;
    Ok(PyPanopticSummary { inner: summary })
}

// ---------------------------------------------------------------------------
// Background panoptic evaluator (ADR-0014 + ADR-0032).
// ---------------------------------------------------------------------------

/// Per-image payload sent over the worker channel. Two variants
/// trade off main-thread cost vs decoded-payload size:
///
/// - [`PanopticUpdate::Decoded`] — used by [`submit`]. The FFI thread
///   has already decoded the user-supplied label-map arrays into
///   [`ImageEntry`]s; the worker just folds them into the kernel.
///   ~5–10 MB per item on COCO val2017 (decoded `u32` label map +
///   side tables).
/// - [`PanopticUpdate::RawPng`] — used by [`submit_png`]. The FFI
///   thread parses the small `segments_info` JSON synchronously (so
///   schema errors surface on the submit call) and ships the raw PNG
///   bytes through the channel without decoding. The worker decodes
///   PNGs inside its rayon pool (when `num_threads > 1`) so decode
///   cost itself parallelises. ~5 KB per item on COCO val2017.
///
/// The variant choice has no observable effect on the kernel — it is
/// purely an internal placement of where decode runs. Single-threaded
/// workers decode `RawPng` items in the same worker thread that would
/// have run them under [`submit`].
pub(crate) enum PanopticUpdate {
    Decoded {
        image_id: ImageId,
        gt: ImageEntry,
        dt: ImageEntry,
    },
    RawPng {
        image_id: ImageId,
        gt_bytes: Vec<u8>,
        gt_segments: Vec<SegmentInfo>,
        dt_bytes: Vec<u8>,
        dt_segments: Vec<SegmentInfo>,
    },
}

impl BackgroundCapable for StreamingPanopticEvaluator {
    type Update = PanopticUpdate;
    type Summary = PanopticSummary;
    type Error = PanopticError;

    fn apply_update(&mut self, u: PanopticUpdate) -> Result<(), PanopticError> {
        match u {
            PanopticUpdate::Decoded { image_id, gt, dt } => self.update(image_id, &gt, &dt),
            PanopticUpdate::RawPng {
                image_id,
                gt_bytes,
                gt_segments,
                dt_bytes,
                dt_segments,
            } => {
                let gt = decode_panoptic_png(image_id, &gt_bytes, gt_segments, "gt")?;
                let dt = decode_panoptic_png(image_id, &dt_bytes, dt_segments, "dt")?;
                self.update(image_id, &gt, &dt)
            }
        }
    }

    fn apply_update_parallel(&mut self, batch: Vec<PanopticUpdate>) -> Result<(), PanopticError> {
        use rayon::prelude::*;
        // Decode any `RawPng` payloads in parallel inside the ambient
        // pool, then hand the unified `(ImageId, ImageEntry,
        // ImageEntry)` batch to the existing strict-mode parallel
        // streaming kernel. The two passes share the same pool — the
        // outer `par_iter` here completes before
        // `update_parsed_parallel` starts its own `par_iter`, so
        // pool-thread oversubscription doesn't occur.
        let parsed: Vec<(ImageId, ImageEntry, ImageEntry)> = batch
            .into_par_iter()
            .map(|u| match u {
                PanopticUpdate::Decoded { image_id, gt, dt } => Ok((image_id, gt, dt)),
                PanopticUpdate::RawPng {
                    image_id,
                    gt_bytes,
                    gt_segments,
                    dt_bytes,
                    dt_segments,
                } => {
                    let gt = decode_panoptic_png(image_id, &gt_bytes, gt_segments, "gt")?;
                    let dt = decode_panoptic_png(image_id, &dt_bytes, dt_segments, "dt")?;
                    Ok((image_id, gt, dt))
                }
            })
            .collect::<Result<_, PanopticError>>()?;
        StreamingPanopticEvaluator::update_parsed_parallel(self, parsed)
    }

    fn finalize(self) -> Result<PanopticSummary, PanopticError> {
        StreamingPanopticEvaluator::finalize(self)
    }

    fn finalize_to_partial(self) -> Result<Vec<u8>, PanopticError> {
        StreamingPanopticEvaluator::finalize_to_partial(self)
    }

    fn images_seen(&self) -> usize {
        self.n_images()
    }

    fn worker_disconnected() -> PanopticError {
        PanopticError::Partial(PartialError::Format {
            kind: vernier_partial::PartialFormatErrorKind::Internal {
                detail: "background panoptic worker is no longer reachable".to_string(),
            },
        })
    }

    fn already_finalized() -> PanopticError {
        PanopticError::Partial(PartialError::Format {
            kind: vernier_partial::PartialFormatErrorKind::Internal {
                detail: "BackgroundPanopticEvaluator has already been finalized".to_string(),
            },
        })
    }
}

/// Background panoptic-quality evaluator (ADR-0014 + ADR-0032).
///
/// Wraps a single dedicated worker thread that owns the underlying
/// [`StreamingPanopticEvaluator`]. `submit(image_id, gt_label_map,
/// gt_segments_info, dt_label_map, dt_segments_info)` posts one image;
/// `snapshot` / `finalize` block on a worker reply. Mirrors the
/// instance and semantic background surfaces.
///
/// `retain_per_image_deltas=True` opts into the strict-mode bit-
/// equality property at merge time (ADR-0032 §"Determinism") at the
/// cost of ~2× streaming memory, same as the sibling streaming
/// evaluator.
#[pyclass(module = "vernier._core", name = "BackgroundPanopticEvaluator")]
pub(crate) struct PyBackgroundPanopticEvaluator {
    lifecycle: Mutex<BackgroundLifecycle<StreamingPanopticEvaluator>>,
    n_categories: usize,
    /// Threading policy resolved at construction time, mirroring the
    /// pool the worker thread builds from `BackgroundConfig.num_threads`.
    /// `Some(_)` ⇒ `submit_png` defers libpng decode to the worker
    /// pool (it parallelises there); `None` ⇒ `submit_png` decodes
    /// inline, preserving the pre-Stage-A producer/consumer overlap
    /// shape byte-for-byte for single-threaded callers.
    num_threads: Option<std::num::NonZeroUsize>,
}

impl PyBackgroundPanopticEvaluator {
    fn lock_lifecycle(
        &self,
    ) -> PyResult<std::sync::MutexGuard<'_, BackgroundLifecycle<StreamingPanopticEvaluator>>> {
        self.lifecycle.lock().map_err(|_| {
            PyRuntimeError::new_err("BackgroundPanopticEvaluator state mutex poisoned")
        })
    }
}

#[pymethods]
impl PyBackgroundPanopticEvaluator {
    #[new]
    #[pyo3(signature = (
        categories,
        parity_mode,
        *,
        things_stuff_split = true,
        retain_per_image_deltas = false,
        rank_id = None,
        queue_capacity = 8,
        worker_affinity = None,
        worker_nice = 5,
        shutdown_timeout_seconds = 5.0,
        boundary = false,
        dilation_ratio = BOUNDARY_PANOPTIC_DILATION_RATIO_DEFAULT,
        num_threads = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        categories: &[u8],
        parity_mode: &str,
        things_stuff_split: bool,
        retain_per_image_deltas: bool,
        rank_id: Option<u32>,
        queue_capacity: usize,
        worker_affinity: Option<usize>,
        worker_nice: i32,
        shutdown_timeout_seconds: f64,
        boundary: bool,
        dilation_ratio: f64,
        num_threads: Option<usize>,
    ) -> PyResult<Self> {
        let mode = parse_panoptic_parity_mode(parity_mode)?;
        let cats = parse_categories(categories).map_err(|e| panoptic_error_to_pyerr(py, e))?;
        let n_categories = cats.len();
        let boundary_cfg = boundary_cfg_from_ffi(boundary, dilation_ratio, mode)?;
        let mut inner = StreamingPanopticEvaluator::new(
            cats,
            mode,
            things_stuff_split,
            retain_per_image_deltas,
        );
        if let Some(rid) = rank_id {
            inner = inner
                .with_rank(rid)
                .map_err(|e| panoptic_error_to_pyerr(py, e))?;
        }
        if let Some(cfg) = boundary_cfg {
            inner = inner
                .with_boundary(cfg)
                .map_err(|e| panoptic_error_to_pyerr(py, e))?;
        }
        // Resolve `num_threads` through the shared FFI policy (env-var
        // override + re-entry warning + `0` → auto), mirroring the
        // instance batch entries; the resulting `ThreadPolicy`'s
        // thread count drives both the worker's pool build and the
        // `submit_png` decode-deferral choice.
        let thread_policy = crate::threads::resolve_threads(py, num_threads);
        let num_threads_nz = thread_policy.thread_count();
        let config = BackgroundConfig {
            queue_capacity,
            worker_affinity,
            worker_nice,
            shutdown_timeout: validate_shutdown_timeout(shutdown_timeout_seconds)?,
            num_threads: num_threads_nz,
        };
        let core = BackgroundCore::spawn(inner, config).map_err(|e| {
            PyRuntimeError::new_err(format!("failed to spawn background worker: {e}"))
        })?;

        let this = Self {
            lifecycle: Mutex::new(BackgroundLifecycle::new(core)),
            n_categories,
            num_threads: num_threads_nz,
        };
        poll_scheduling_warning(py, "BackgroundPanopticEvaluator", || {
            Ok(this.lock_lifecycle()?.take_scheduling_outcome())
        })?;
        Ok(this)
    }

    /// Number of categories in the taxonomy.
    #[getter]
    fn n_categories(&self) -> usize {
        self.n_categories
    }

    /// Submit one image's GT/DT pair to the worker. `timeout` mirrors
    /// the instance / semantic background surfaces.
    #[pyo3(signature = (
        image_id,
        gt_label_map,
        gt_segments_info,
        dt_label_map,
        dt_segments_info,
        *,
        timeout = None,
    ))]
    #[allow(
        clippy::needless_pass_by_value,
        clippy::too_many_arguments,
        reason = "PyO3 requires `PyReadonlyArray2` by value as a pyfunction argument; \
                  the rest mirror the streaming evaluator's update signature"
    )]
    fn submit<'py>(
        &self,
        py: Python<'py>,
        image_id: i64,
        gt_label_map: PyReadonlyArray2<'py, u32>,
        gt_segments_info: &[u8],
        dt_label_map: PyReadonlyArray2<'py, u32>,
        dt_segments_info: &[u8],
        timeout: Option<f64>,
    ) -> PyResult<()> {
        // Build the two ImageEntry on the FFI thread (still under
        // GIL) so JSON parsing + segment-id validation are surfaced
        // synchronously instead of as stashed worker errors. The
        // user-supplied label-map ndarrays are already decoded, so
        // there's no work to defer — wrap in the `Decoded` variant.
        let gt = build_image_entry(py, image_id, &gt_label_map, gt_segments_info, "gt")?;
        let dt = build_image_entry(py, image_id, &dt_label_map, dt_segments_info, "dt")?;
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

        let payload = PanopticUpdate::Decoded { image_id, gt, dt };
        let lifecycle = &self.lifecycle;
        let result = py.detach(move || -> Result<(), SubmitError<PanopticError>> {
            let guard = lifecycle.lock().map_err(|_| {
                SubmitError::Eval(StreamingPanopticEvaluator::worker_disconnected())
            })?;
            let core = guard.active().map_err(SubmitError::Eval)?;
            match timeout_dur {
                None => core.submit_blocking(payload).map_err(SubmitError::Eval),
                Some(t) => core.submit_timeout(payload, t),
            }
        });
        result.map_err(|e| match e {
            SubmitError::Eval(inner) => panoptic_error_to_pyerr(py, inner),
            SubmitError::Full(full) => queue_full_to_pyerr(py, full),
        })
    }

    /// Variant of [`Self::submit`] that takes panoptic PNG byte blobs
    /// instead of pre-decoded uint32 ndarrays. Fuses libpng decode +
    /// RGB→id + (DT side) S3 area marginals + S1/S11 validation in
    /// one Rust pass; skips the Pillow → numpy → uint32 round-trip
    /// the Python wrapper would otherwise drive on the main thread.
    /// Strictly equivalent to
    /// `submit(decode_label_map_png(p), ...)` on the result; the diff
    /// is wall-time, not correctness.
    #[pyo3(signature = (
        image_id,
        gt_png_bytes,
        gt_segments_info,
        dt_png_bytes,
        dt_segments_info,
        *,
        timeout = None,
    ))]
    #[allow(
        clippy::too_many_arguments,
        reason = "mirrors the streaming evaluator's update_png signature"
    )]
    fn submit_png(
        &self,
        py: Python<'_>,
        image_id: i64,
        gt_png_bytes: &[u8],
        gt_segments_info: &[u8],
        dt_png_bytes: &[u8],
        dt_segments_info: &[u8],
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

        // Parse `segments_info` synchronously on both code paths so
        // schema errors surface on the submit call. The only
        // threaded/sequential split is whether libpng decode runs
        // now (sequential: producer/consumer overlap against the
        // worker stays exactly today's behaviour) or inside the
        // worker pool (threaded: libpng parallelises across workers).
        let gt_segments = parse_segments_info(py, image_id, gt_segments_info)?;
        let dt_segments = parse_segments_info(py, image_id, dt_segments_info)?;
        let payload = if self.num_threads.is_some() {
            PanopticUpdate::RawPng {
                image_id,
                gt_bytes: gt_png_bytes.to_vec(),
                gt_segments,
                dt_bytes: dt_png_bytes.to_vec(),
                dt_segments,
            }
        } else {
            let gt = decode_panoptic_png(image_id, gt_png_bytes, gt_segments, "gt")
                .map_err(|e| panoptic_error_to_pyerr(py, e))?;
            let dt = decode_panoptic_png(image_id, dt_png_bytes, dt_segments, "dt")
                .map_err(|e| panoptic_error_to_pyerr(py, e))?;
            PanopticUpdate::Decoded { image_id, gt, dt }
        };
        let lifecycle = &self.lifecycle;
        let result = py.detach(move || -> Result<(), SubmitError<PanopticError>> {
            let guard = lifecycle.lock().map_err(|_| {
                SubmitError::Eval(StreamingPanopticEvaluator::worker_disconnected())
            })?;
            let core = guard.active().map_err(SubmitError::Eval)?;
            match timeout_dur {
                None => core.submit_blocking(payload).map_err(SubmitError::Eval),
                Some(t) => core.submit_timeout(payload, t),
            }
        });
        result.map_err(|e| match e {
            SubmitError::Eval(inner) => panoptic_error_to_pyerr(py, inner),
            SubmitError::Full(full) => queue_full_to_pyerr(py, full),
        })
    }

    /// Drain the queue, finalize the evaluator, and join the worker.
    fn finalize(&self, py: Python<'_>) -> PyResult<PyPanopticSummary> {
        let lifecycle = &self.lifecycle;
        let summary = py
            .detach(|| {
                let mut guard = lifecycle
                    .lock()
                    .map_err(|_| StreamingPanopticEvaluator::worker_disconnected())?;
                guard.take_and_finalize()
            })
            .map_err(|e| panoptic_error_to_pyerr(py, e))?;
        Ok(PyPanopticSummary { inner: summary })
    }

    /// ADR-0032 / ADR-0035: drain, serialize the final state, and
    /// shut the worker down.
    fn finalize_to_partial<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let lifecycle = &self.lifecycle;
        let blob = py
            .detach(|| {
                let mut guard = lifecycle
                    .lock()
                    .map_err(|_| StreamingPanopticEvaluator::worker_disconnected())?;
                guard.take_and_finalize_to_partial()
            })
            .map_err(|e| panoptic_error_to_pyerr(py, e))?;
        Ok(PyBytes::new(py, &blob))
    }

    /// Context-manager entry.
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Context-manager exit: best-effort shutdown.
    #[pyo3(signature = (_exc_type=None, _exc=None, _tb=None))]
    fn __exit__(
        &self,
        py: Python<'_>,
        _exc_type: Option<Py<PyAny>>,
        _exc: Option<Py<PyAny>>,
        _tb: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        let lifecycle = &self.lifecycle;
        py.detach(|| {
            if let Ok(mut guard) = lifecycle.lock() {
                guard.shutdown();
            }
        });
        Ok(())
    }

    fn __del__(&self, py: Python<'_>) {
        let lifecycle = &self.lifecycle;
        py.detach(|| {
            if let Ok(mut guard) = lifecycle.lock() {
                guard.shutdown();
            }
        });
    }

    /// Mirror of the underlying evaluator's `n_images`. Advisory.
    #[getter]
    fn n_images(&self) -> PyResult<usize> {
        Ok(self
            .lock_lifecycle()?
            .active()
            .map(BackgroundCore::images_seen)
            .unwrap_or(0))
    }

    /// Approximate count of `Update` messages waiting in the channel.
    #[getter]
    fn queue_depth(&self) -> PyResult<usize> {
        Ok(self
            .lock_lifecycle()?
            .active()
            .map(BackgroundCore::queue_depth)
            .unwrap_or(0))
    }

    fn __repr__(&self) -> String {
        format!(
            "BackgroundPanopticEvaluator(n_categories={})",
            self.n_categories
        )
    }
}

/// Register panoptic FFI symbols on the `vernier._core` module. Called
/// from the `_core` `#[pymodule]` factory in `lib.rs`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPanopticDataset>()?;
    m.add_class::<PyPanopticPredictions>()?;
    m.add_class::<PyPanopticSummary>()?;
    m.add_class::<PyClassPanopticStats>()?;
    m.add_class::<PyGroupPanopticStats>()?;
    m.add_class::<PyBackgroundPanopticEvaluator>()?;
    m.add_class::<PyPartitionedPanopticReport>()?;
    m.add_function(wrap_pyfunction!(evaluate_panoptic, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_panoptic_to_partial, m)?)?;
    m.add_function(wrap_pyfunction!(merge_panoptic_partials, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_panoptic_partitioned, m)?)?;
    #[cfg(any(test, feature = "_test-counter"))]
    {
        m.add_function(wrap_pyfunction!(_test_reset_panoptic_matching_count, m)?)?;
        m.add_function(wrap_pyfunction!(_test_read_panoptic_matching_count, m)?)?;
    }
    Ok(())
}
