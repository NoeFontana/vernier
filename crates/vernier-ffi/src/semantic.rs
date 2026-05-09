//! PyO3 bindings for [`vernier_semantic`] (ADR-0028).
//!
//! Surface:
//! - [`PyConfusionMatrix`] / [`PyClassSemanticStats`] /
//!   [`PySemanticSummary`] — read-only wrappers over the result types.
//! - [`evaluate_semantic_from_arrays`] — free pyfunction that runs the
//!   confusion-matrix kernel + summarize pass under `py.detach`
//!   (ADR-0006). Takes a Python dict of `(image_id -> uint32 (H, W)
//!   ndarray)` per side, the evaluator config (n_classes,
//!   ignore_label, label_remap), and a parity_mode string.
//!
//! Mirrors the panoptic FFI shape (`crates/vernier-ffi/src/panoptic.rs`)
//! with two simplifications:
//!
//! - The semantic dataset doesn't need a parsed-once `PySemanticDataset`
//!   handle (the kernel walks the pixel pairs linearly; nothing to
//!   pre-compute and cache the way panoptic's segment-id validation
//!   does).
//! - Inputs arrive as plain uint32 ndarrays — no JSON metadata side
//!   channel. Cityscapes / ADE20K / Pascal-VOC presets bake their
//!   ignore-label and class-count conventions on the Python side
//!   (PR-B5) and pass plain primitives across the FFI boundary.
//!
//! `from_files` (PNG decode) and `from_binary_masks` (per-class mask
//! merging) are documented follow-ups landing in PR-B5 alongside the
//! preset constructors that drive them.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

use numpy::PyReadonlyArray2;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyDictMethods, PyList};

use vernier_partial::PartialError;
use vernier_semantic::kernel::accumulate_confusion;
use vernier_semantic::{
    summarize, ClassSemanticStats, ConfusionMatrix, ParityMode, SemanticError, SemanticSummary,
    StreamingSemanticEvaluator,
};

use crate::background::BackgroundConfig;
use crate::background_streaming::{
    BackgroundCapable, BackgroundCore, BackgroundLifecycle, SubmitError,
};
use crate::numpy_utils::{parse_uint32_label_maps, ImageId, LabelMap};
use crate::{poll_scheduling_warning, queue_full_to_pyerr, validate_shutdown_timeout};

fn parse_label_remap(remap: Option<&Bound<'_, PyDict>>) -> PyResult<Option<HashMap<u32, u32>>> {
    match remap {
        None => Ok(None),
        Some(d) => {
            let mut out: HashMap<u32, u32> = HashMap::with_capacity(d.len());
            for (k, v) in d.iter() {
                let from: u32 = k.extract().map_err(|e| {
                    PyValueError::new_err(format!("label_remap key must be uint32: {e}"))
                })?;
                let to: u32 = v.extract().map_err(|e| {
                    PyValueError::new_err(format!("label_remap value must be uint32: {e}"))
                })?;
                out.insert(from, to);
            }
            Ok(Some(out))
        }
    }
}

fn parse_parity_mode(s: &str) -> PyResult<ParityMode> {
    match s {
        "strict" => Ok(ParityMode::Strict),
        "corrected" => Ok(ParityMode::Corrected),
        other => Err(PyValueError::new_err(format!(
            "unknown semantic parity_mode {other:?}; expected 'strict' or 'corrected'"
        ))),
    }
}

/// Apply a label remap in place to one image's flat DT pixel buffer
/// (quirk **AK2**). Pixels with values not in `remap` are left
/// untouched. Pixels remapped to `>= n_classes` (e.g., to the ignore
/// label sentinel of 255 on a 19-class evaluator) are silently
/// dropped by the kernel via the existing AI4 strict-MS path.
fn apply_label_remap(buf: &mut [u32], remap: &HashMap<u32, u32>) {
    for v in buf.iter_mut() {
        if let Some(&new) = remap.get(v) {
            *v = new;
        }
    }
}

/// Per-class semantic-segmentation row exposed to Python.
///
/// Mirrors [`vernier_semantic::ClassSemanticStats`] one-to-one. The
/// NaN-vs-0.0 disposition for zero-support classes (quirk **AL2**) is
/// already baked in by the time the row reaches Python; the Python
/// side reads the values as-is.
#[pyclass(
    module = "vernier._core",
    name = "ClassSemanticStats",
    frozen,
    skip_from_py_object
)]
#[derive(Clone, Copy)]
pub(crate) struct PyClassSemanticStats {
    inner: ClassSemanticStats,
}

#[pymethods]
impl PyClassSemanticStats {
    #[getter]
    fn class_id(&self) -> u32 {
        self.inner.class_id
    }
    #[getter]
    fn iou(&self) -> f64 {
        self.inner.iou
    }
    #[getter]
    fn accuracy(&self) -> f64 {
        self.inner.accuracy
    }
    #[getter]
    fn precision(&self) -> f64 {
        self.inner.precision
    }
    #[getter]
    fn n_gt_pixels(&self) -> u64 {
        self.inner.n_gt_pixels
    }
    #[getter]
    fn n_dt_pixels(&self) -> u64 {
        self.inner.n_dt_pixels
    }

    fn __repr__(&self) -> String {
        format!(
            "ClassSemanticStats(class_id={}, iou={:.4}, accuracy={:.4}, precision={:.4}, \
             n_gt_pixels={}, n_dt_pixels={})",
            self.inner.class_id,
            self.inner.iou,
            self.inner.accuracy,
            self.inner.precision,
            self.inner.n_gt_pixels,
            self.inner.n_dt_pixels,
        )
    }
}

/// `(N, N)` confusion-matrix view exposed to Python.
///
/// The flat `Vec<u64>` storage on the Rust side is exposed as a 2-D
/// `numpy.ndarray` via [`PyConfusionMatrix::counts`]. ADR-0028 §F1
/// promotes the matrix to a first-class output: downstream calibration
/// / error-decomposition / model-diff tools consume it directly.
#[pyclass(module = "vernier._core", name = "ConfusionMatrix", frozen)]
pub(crate) struct PyConfusionMatrix {
    inner: ConfusionMatrix,
}

#[pymethods]
impl PyConfusionMatrix {
    /// Number of evaluation classes; the matrix is `(n_classes, n_classes)`.
    #[getter]
    fn n_classes(&self) -> u32 {
        self.inner.n_classes()
    }

    /// Total pixel count across all cells. Equals
    /// `sum(counts)` and is useful for sanity checks.
    #[getter]
    fn total(&self) -> u64 {
        self.inner.counts().iter().sum()
    }

    /// Trace (number of correct-class pixels). Equals
    /// `sum(diag(counts))` and is the numerator of pixel accuracy.
    #[getter]
    fn trace(&self) -> u64 {
        let n = self.inner.n_classes() as usize;
        let cs = self.inner.counts();
        (0..n).map(|c| cs[c * n + c]).sum()
    }

    /// `counts[g, d]` lookup. Returns the integer pixel count for the
    /// given (gt_class, pred_class) cell. Bounds-checked.
    fn get(&self, g: u32, d: u32) -> PyResult<u64> {
        let n = self.inner.n_classes();
        if g >= n || d >= n {
            return Err(PyValueError::new_err(format!(
                "ConfusionMatrix.get index out of range: ({g}, {d}) for n_classes={n}"
            )));
        }
        Ok(self.inner.get(g, d))
    }

    /// `(N, N)` numpy view of the confusion matrix as a fresh
    /// `numpy.uint64` array. The buffer is materialized on each call
    /// (cheap for typical N ≤ 150) so the FFI boundary doesn't have
    /// to manage a long-lived borrow into the Rust-side `Vec<u64>`.
    fn counts<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray2<u64>> {
        use numpy::PyArray2;
        let n = self.inner.n_classes() as usize;
        let flat = self.inner.counts().to_vec();
        let arr = PyArray2::<u64>::from_vec2(
            py,
            &flat.chunks(n).map(<[u64]>::to_vec).collect::<Vec<_>>(),
        )
        // `from_vec2` only fails if the row lengths differ; our
        // chunked-by-n flat buffer has uniform rows by construction.
        // Surface a typed error rather than panic to honor the
        // workspace clippy deny list (no `unwrap`).
        .unwrap_or_else(|_| PyArray2::<u64>::zeros(py, (n, n), false));
        arr
    }

    fn __repr__(&self) -> String {
        format!(
            "ConfusionMatrix(n_classes={}, total={}, trace={})",
            self.inner.n_classes(),
            self.inner.counts().iter().sum::<u64>(),
            {
                let n = self.inner.n_classes() as usize;
                let cs = self.inner.counts();
                (0..n).map(|c| cs[c * n + c]).sum::<u64>()
            }
        )
    }
}

/// Top-level semantic-evaluation result. Read via field accessors.
///
/// Sibling to [`crate::PySummary`] (instance) and
/// [`crate::panoptic::PyPanopticSummary`] (panoptic) per ADR-0028
/// §"Public Python surface".
#[pyclass(module = "vernier._core", name = "SemanticSummary", frozen)]
pub(crate) struct PySemanticSummary {
    inner: SemanticSummary,
}

#[pymethods]
impl PySemanticSummary {
    #[getter]
    fn miou(&self) -> f64 {
        self.inner.miou
    }
    #[getter]
    fn fwiou(&self) -> f64 {
        self.inner.fwiou
    }
    #[getter]
    fn pixel_accuracy(&self) -> f64 {
        self.inner.pixel_accuracy
    }
    #[getter]
    fn mean_accuracy(&self) -> f64 {
        self.inner.mean_accuracy
    }

    /// Per-class rows keyed by class id. Returns a Python `dict`
    /// (constructed fresh on each call from the underlying BTreeMap).
    fn per_class<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (cls, row) in &self.inner.per_class {
            dict.set_item(*cls, PyClassSemanticStats { inner: *row })?;
        }
        Ok(dict)
    }

    /// The accumulated `(N, N)` confusion matrix as a Python wrapper.
    /// Constructed fresh on each access from a clone of the underlying
    /// matrix (the matrix is small — at most 150² = 22500 cells —
    /// so the clone is cheap).
    #[getter]
    fn confusion_matrix(&self) -> PyConfusionMatrix {
        PyConfusionMatrix {
            inner: self.inner.confusion_matrix.clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "SemanticSummary(miou={:.4}, fwiou={:.4}, pixel_accuracy={:.4}, mean_accuracy={:.4})",
            self.inner.miou, self.inner.fwiou, self.inner.pixel_accuracy, self.inner.mean_accuracy,
        )
    }
}

/// Map a [`SemanticError`] to a Python exception. The `Partial`
/// variant routes through [`crate::partial_error_to_pyerr`] so the
/// five distributed-eval exception classes are shared with the
/// instance paradigm (`vernier.semantic.PartialFormatMismatch is
/// vernier.instance.PartialFormatMismatch`). Other variants surface
/// as `PyValueError` with the structured fields embedded in the
/// message body for the parity harness to lift programmatically.
fn semantic_error_to_pyerr(py: Python<'_>, e: &SemanticError) -> PyErr {
    match e {
        SemanticError::Partial(inner) => crate::partial_error_to_pyerr(py, inner),
        other => PyValueError::new_err(format!("{other}")),
    }
}

/// Run the full semantic-segmentation evaluation (kernel + summarize)
/// against pre-decoded uint32 label maps. Drops the GIL via
/// `py.detach` for the duration of the kernel walk + summarize pass
/// (ADR-0006).
///
/// `gt_label_maps` and `dt_label_maps` are dicts mapping image id
/// (int) to a 2-D `numpy.ndarray` of dtype `uint32` whose pixel values
/// are class ids in `[0, n_classes) ∪ {ignore_label}` (GT) and
/// `[0, n_classes)` (DT).
///
/// `n_classes` is the evaluation class count. `ignore_label`, when
/// present, masks pixels with `gt == ignore_label` from the histogram
/// (quirk **AJ2**). `label_remap`, when present, applies to DT pixels
/// before bincount (quirk **AK2**). `parity_mode` selects the NaN
/// disposition for zero-support per-class entries (quirk **AL2**).
#[pyfunction]
#[pyo3(signature = (
    gt_label_maps,
    dt_label_maps,
    n_classes,
    parity_mode,
    *,
    ignore_label = None,
    label_remap = None,
))]
pub(crate) fn evaluate_semantic_from_arrays<'py>(
    py: Python<'py>,
    gt_label_maps: &Bound<'py, PyDict>,
    dt_label_maps: &Bound<'py, PyDict>,
    n_classes: u32,
    parity_mode: &str,
    ignore_label: Option<u32>,
    label_remap: Option<&Bound<'py, PyDict>>,
) -> PyResult<PySemanticSummary> {
    if n_classes == 0 {
        return Err(PyValueError::new_err(
            "semantic evaluator requires n_classes >= 1",
        ));
    }
    let mode = parse_parity_mode(parity_mode)?;
    let mut gt_maps = parse_uint32_label_maps(gt_label_maps, "semantic gt")?;
    let mut dt_maps = parse_uint32_label_maps(dt_label_maps, "semantic dt")?;
    let remap = parse_label_remap(label_remap)?;

    // Validate that every GT image has a matching DT image. Quirk
    // AM1 — strict against MS / CS / PA. Surface the missing image id
    // as a typed error.
    for image_id in gt_maps.keys() {
        if !dt_maps.contains_key(image_id) {
            return Err(semantic_error_to_pyerr(
                py,
                &SemanticError::MissingPrediction {
                    image_id: *image_id,
                },
            ));
        }
    }

    // Apply label_remap to DT buffers up-front (AK2). Pre-applying at
    // FFI parse time avoids a per-pixel dict lookup in the hot kernel
    // loop.
    if let Some(remap) = &remap {
        for (_, _, buf) in dt_maps.values_mut() {
            apply_label_remap(buf, remap);
        }
    }

    if gt_maps.is_empty() {
        return Err(semantic_error_to_pyerr(py, &SemanticError::EmptyDataset));
    }

    // Build the iteration plan on the Python thread (the data we move
    // into `py.detach` is plain `Vec<u32>` slices and a HashMap of
    // image ids — no PyO3 types cross the boundary). Image-id
    // iteration order is sorted for determinism (quirk AM5).
    let mut image_ids: Vec<ImageId> = gt_maps.keys().copied().collect();
    image_ids.sort_unstable();

    // Eagerly extract the (gt_buf, dt_buf, gt_shape, dt_shape) tuple
    // per image so the GIL-free closure has only owned data.
    let mut work: Vec<(ImageId, LabelMap, LabelMap)> = Vec::with_capacity(image_ids.len());
    for image_id in &image_ids {
        let gt = gt_maps.remove(image_id).ok_or_else(|| {
            // Unreachable in practice: image_id came from gt_maps.keys() above.
            PyValueError::new_err(format!(
                "internal: missing gt label_map for image_id={image_id}"
            ))
        })?;
        let dt = dt_maps.remove(image_id).ok_or_else(|| {
            // Unreachable: validated above.
            PyValueError::new_err(format!(
                "internal: missing dt label_map for image_id={image_id}"
            ))
        })?;
        if gt.0 != dt.0 || gt.1 != dt.1 {
            return Err(semantic_error_to_pyerr(
                py,
                &SemanticError::ShapeMismatch {
                    image_id: *image_id,
                    gt_shape: (gt.0, gt.1),
                    dt_shape: (dt.0, dt.1),
                },
            ));
        }
        work.push((*image_id, gt, dt));
    }

    let summary = py.detach(move || {
        let mut confusion = ConfusionMatrix::zeros(n_classes);
        for (_, (_, _, gt_buf), (_, _, dt_buf)) in &work {
            accumulate_confusion(gt_buf, dt_buf, ignore_label, &mut confusion);
        }
        summarize(confusion, mode)
    });

    Ok(PySemanticSummary { inner: summary })
}

/// One-shot per-rank streaming submit + serialize partial (ADR-0035).
///
/// Functionally equivalent to constructing a `StreamingSemanticEvaluator`,
/// calling `update` per image in sorted `image_id` order, then
/// `finalize_to_partial`. Exposed as a single pyfunction so the streaming
/// pyclass stays Rust-internal.
#[pyfunction]
#[pyo3(signature = (
    gt_label_maps,
    dt_label_maps,
    n_classes,
    parity_mode,
    rank_id,
    *,
    ignore_label = None,
))]
pub(crate) fn evaluate_semantic_to_partial<'py>(
    py: Python<'py>,
    gt_label_maps: &Bound<'py, PyDict>,
    dt_label_maps: &Bound<'py, PyDict>,
    n_classes: u32,
    parity_mode: &str,
    rank_id: u32,
    ignore_label: Option<u32>,
) -> PyResult<Bound<'py, PyBytes>> {
    if n_classes == 0 {
        return Err(PyValueError::new_err(
            "evaluate_semantic_to_partial requires n_classes >= 1",
        ));
    }
    let mode = parse_parity_mode(parity_mode)?;
    let mut gt_maps = parse_uint32_label_maps(gt_label_maps, "semantic gt")?;
    let mut dt_maps = parse_uint32_label_maps(dt_label_maps, "semantic dt")?;

    for image_id in gt_maps.keys() {
        if !dt_maps.contains_key(image_id) {
            return Err(semantic_error_to_pyerr(
                py,
                &SemanticError::MissingPrediction {
                    image_id: *image_id,
                },
            ));
        }
    }

    let mut image_ids: Vec<ImageId> = gt_maps.keys().copied().collect();
    image_ids.sort_unstable();

    let mut work: Vec<(ImageId, LabelMap, LabelMap)> = Vec::with_capacity(image_ids.len());
    for image_id in &image_ids {
        let gt = gt_maps.remove(image_id).ok_or_else(|| {
            PyValueError::new_err(format!(
                "internal: missing gt label_map for image_id={image_id}"
            ))
        })?;
        let dt = dt_maps.remove(image_id).ok_or_else(|| {
            PyValueError::new_err(format!(
                "internal: missing dt label_map for image_id={image_id}"
            ))
        })?;
        if gt.0 != dt.0 || gt.1 != dt.1 {
            return Err(semantic_error_to_pyerr(
                py,
                &SemanticError::ShapeMismatch {
                    image_id: *image_id,
                    gt_shape: (gt.0, gt.1),
                    dt_shape: (dt.0, dt.1),
                },
            ));
        }
        work.push((*image_id, gt, dt));
    }

    let bytes = py
        .detach(move || -> Result<Vec<u8>, SemanticError> {
            let mut ev = StreamingSemanticEvaluator::new(n_classes, ignore_label, mode)
                .with_rank(rank_id)?;
            for (image_id, gt, dt) in &work {
                ev.update(*image_id, &gt.2, &dt.2)?;
            }
            ev.finalize_to_partial()
        })
        .map_err(|e| semantic_error_to_pyerr(py, &e))?;
    Ok(PyBytes::new(py, &bytes))
}

/// Merge per-rank partials into a final summary (ADR-0035).
#[pyfunction]
#[pyo3(signature = (n_classes, partials, parity_mode, *, ignore_label = None))]
pub(crate) fn merge_semantic_partials<'py>(
    py: Python<'py>,
    n_classes: u32,
    partials: &Bound<'py, PyList>,
    parity_mode: &str,
    ignore_label: Option<u32>,
) -> PyResult<PySemanticSummary> {
    if n_classes == 0 {
        return Err(PyValueError::new_err(
            "merge_semantic_partials requires n_classes >= 1",
        ));
    }
    let mode = parse_parity_mode(parity_mode)?;
    let owned: Vec<Vec<u8>> = partials
        .iter()
        .map(|item| {
            item.cast::<PyBytes>()
                .map_err(|_| {
                    PyValueError::new_err("merge_semantic_partials expects a list of bytes objects")
                })
                .map(|b| b.as_bytes().to_vec())
        })
        .collect::<PyResult<_>>()?;
    let summary = py
        .detach(move || -> Result<SemanticSummary, SemanticError> {
            let slices: Vec<&[u8]> = owned.iter().map(|v| v.as_slice()).collect();
            let merged =
                StreamingSemanticEvaluator::from_partials(n_classes, ignore_label, mode, &slices)?;
            Ok(StreamingSemanticEvaluator::finalize(merged))
        })
        .map_err(|e| semantic_error_to_pyerr(py, &e))?;
    Ok(PySemanticSummary { inner: summary })
}

// ---------------------------------------------------------------------------
// Background semantic evaluator (ADR-0014 + ADR-0032).
//
// The `BackgroundCapable` impl is the only paradigm-side seam needed
// by the generic `BackgroundCore<E>` worker; everything else
// (channel, atomics, scheduling) is shared with the panoptic side.
// ---------------------------------------------------------------------------

/// Per-image payload carried over the worker channel. Owned data —
/// the FFI thread builds this from the user's numpy arrays before
/// dropping the GIL.
pub(crate) struct SemanticUpdate {
    image_id: ImageId,
    gt: Vec<u32>,
    dt: Vec<u32>,
}

impl BackgroundCapable for StreamingSemanticEvaluator {
    type Update = SemanticUpdate;
    type Summary = SemanticSummary;
    type Error = SemanticError;

    fn apply_update(&mut self, u: SemanticUpdate) -> Result<(), SemanticError> {
        self.update(u.image_id, &u.gt, &u.dt)
    }

    fn finalize(self) -> Result<SemanticSummary, SemanticError> {
        Ok(StreamingSemanticEvaluator::finalize(self))
    }

    fn finalize_to_partial(self) -> Result<Vec<u8>, SemanticError> {
        StreamingSemanticEvaluator::finalize_to_partial(self)
    }

    fn images_seen(&self) -> usize {
        self.n_images()
    }

    fn worker_disconnected() -> SemanticError {
        SemanticError::Partial(PartialError::Format {
            kind: vernier_partial::PartialFormatErrorKind::Internal {
                detail: "background semantic worker is no longer reachable".to_string(),
            },
        })
    }

    fn already_finalized() -> SemanticError {
        SemanticError::Partial(PartialError::Format {
            kind: vernier_partial::PartialFormatErrorKind::Internal {
                detail: "BackgroundSemanticEvaluator has already been finalized".to_string(),
            },
        })
    }
}

/// Background semantic-segmentation evaluator (ADR-0014 + ADR-0032).
///
/// Wraps a single dedicated worker thread that owns the underlying
/// [`StreamingSemanticEvaluator`]. `submit(image_id, gt, dt)` posts
/// one image; `snapshot()` / `finalize()` block on a worker reply.
///
/// Mirrors the panoptic and instance background surfaces: same
/// constructor knobs (`queue_capacity`, `worker_affinity`,
/// `worker_nice`, `shutdown_timeout_seconds`), same context-manager
/// lifecycle, same `to_partial` / `finalize_to_partial` for
/// distributed-eval gather (ADR-0032).
#[pyclass(module = "vernier._core", name = "BackgroundSemanticEvaluator")]
pub(crate) struct PyBackgroundSemanticEvaluator {
    lifecycle: Mutex<BackgroundLifecycle<StreamingSemanticEvaluator>>,
    n_classes: u32,
}

impl PyBackgroundSemanticEvaluator {
    fn lock_lifecycle(
        &self,
    ) -> PyResult<std::sync::MutexGuard<'_, BackgroundLifecycle<StreamingSemanticEvaluator>>> {
        self.lifecycle.lock().map_err(|_| {
            PyRuntimeError::new_err("BackgroundSemanticEvaluator state mutex poisoned")
        })
    }
}

#[pymethods]
impl PyBackgroundSemanticEvaluator {
    #[new]
    #[pyo3(signature = (
        n_classes,
        parity_mode,
        *,
        ignore_label = None,
        rank_id = None,
        queue_capacity = 8,
        worker_affinity = None,
        worker_nice = 5,
        shutdown_timeout_seconds = 5.0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        n_classes: u32,
        parity_mode: &str,
        ignore_label: Option<u32>,
        rank_id: Option<u32>,
        queue_capacity: usize,
        worker_affinity: Option<usize>,
        worker_nice: i32,
        shutdown_timeout_seconds: f64,
    ) -> PyResult<Self> {
        if n_classes == 0 {
            return Err(PyValueError::new_err(
                "BackgroundSemanticEvaluator requires n_classes >= 1",
            ));
        }
        let mode = parse_parity_mode(parity_mode)?;
        let mut inner = StreamingSemanticEvaluator::new(n_classes, ignore_label, mode);
        if let Some(rid) = rank_id {
            inner = inner
                .with_rank(rid)
                .map_err(|e| semantic_error_to_pyerr(py, &e))?;
        }
        let config = BackgroundConfig {
            queue_capacity,
            worker_affinity,
            worker_nice,
            shutdown_timeout: validate_shutdown_timeout(shutdown_timeout_seconds)?,
        };
        let core = BackgroundCore::spawn(inner, config).map_err(|e| {
            PyRuntimeError::new_err(format!("failed to spawn background worker: {e}"))
        })?;

        let this = Self {
            lifecycle: Mutex::new(BackgroundLifecycle::new(core)),
            n_classes,
        };
        poll_scheduling_warning(py, "BackgroundSemanticEvaluator", || {
            Ok(this.lock_lifecycle()?.take_scheduling_outcome())
        })?;
        Ok(this)
    }

    /// Number of evaluation classes (constant for the lifetime of the
    /// evaluator).
    #[getter]
    fn n_classes(&self) -> u32 {
        self.n_classes
    }

    /// Submit one image's `(gt, dt)` label-map pair to the worker.
    /// `timeout` mirrors the instance background:
    ///
    /// - `None` (default) → block until a slot is free
    /// - `0.0` → single non-blocking attempt; raise `QueueFullError`
    ///   if the queue is full
    /// - `t > 0.0` → wait up to `t` seconds; raise `QueueFullError`
    ///   on timeout
    #[pyo3(signature = (image_id, gt, dt, *, timeout = None))]
    #[allow(
        clippy::needless_pass_by_value,
        reason = "PyO3 requires `PyReadonlyArray2` by value as a pyfunction argument; \
                  the borrow lives only for the call's duration"
    )]
    fn submit<'py>(
        &self,
        py: Python<'py>,
        image_id: i64,
        gt: PyReadonlyArray2<'py, u32>,
        dt: PyReadonlyArray2<'py, u32>,
        timeout: Option<f64>,
    ) -> PyResult<()> {
        let gt_view = gt.as_array();
        let dt_view = dt.as_array();
        let gt_shape = gt_view.shape();
        let dt_shape = dt_view.shape();
        if gt_shape != dt_shape {
            return Err(semantic_error_to_pyerr(
                py,
                &SemanticError::ShapeMismatch {
                    image_id,
                    gt_shape: (gt_shape[0] as u32, gt_shape[1] as u32),
                    dt_shape: (dt_shape[0] as u32, dt_shape[1] as u32),
                },
            ));
        }
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

        let payload = SemanticUpdate {
            image_id,
            gt: gt_view.iter().copied().collect(),
            dt: dt_view.iter().copied().collect(),
        };

        let lifecycle = &self.lifecycle;
        let result = py.detach(move || -> Result<(), SubmitError<SemanticError>> {
            let guard = lifecycle.lock().map_err(|_| {
                SubmitError::Eval(StreamingSemanticEvaluator::worker_disconnected())
            })?;
            let core = guard.active().map_err(SubmitError::Eval)?;
            match timeout_dur {
                None => core.submit_blocking(payload).map_err(SubmitError::Eval),
                Some(t) => core.submit_timeout(payload, t),
            }
        });
        result.map_err(|e| match e {
            SubmitError::Eval(inner) => semantic_error_to_pyerr(py, &inner),
            SubmitError::Full(full) => queue_full_to_pyerr(py, full),
        })
    }

    /// Drain the queue, finalize the evaluator, and join the worker.
    fn finalize(&self, py: Python<'_>) -> PyResult<PySemanticSummary> {
        let lifecycle = &self.lifecycle;
        let summary = py
            .detach(|| {
                let mut guard = lifecycle
                    .lock()
                    .map_err(|_| StreamingSemanticEvaluator::worker_disconnected())?;
                guard.take_and_finalize()
            })
            .map_err(|e| semantic_error_to_pyerr(py, &e))?;
        Ok(PySemanticSummary { inner: summary })
    }

    /// ADR-0032 / ADR-0035: drain, serialize the final state, and
    /// shut the worker down.
    fn finalize_to_partial<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let lifecycle = &self.lifecycle;
        let blob = py
            .detach(|| {
                let mut guard = lifecycle
                    .lock()
                    .map_err(|_| StreamingSemanticEvaluator::worker_disconnected())?;
                guard.take_and_finalize_to_partial()
            })
            .map_err(|e| semantic_error_to_pyerr(py, &e))?;
        Ok(PyBytes::new(py, &blob))
    }

    /// Context-manager entry. Returns `self` so `with ev as e:`
    /// binds the constructed instance.
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

    /// Best-effort cleanup if the user lets the wrapper go out of
    /// scope without explicit `finalize` / `__exit__`. Silences all
    /// errors — raising from `__del__` is invisible to the caller
    /// anyway.
    fn __del__(&self, py: Python<'_>) {
        let lifecycle = &self.lifecycle;
        py.detach(|| {
            if let Ok(mut guard) = lifecycle.lock() {
                guard.shutdown();
            }
        });
    }

    /// Mirror of the underlying evaluator's `n_images`. Advisory —
    /// updated by the worker after each successful submit.
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
        format!("BackgroundSemanticEvaluator(n_classes={})", self.n_classes)
    }
}

/// Register semantic FFI symbols on the `vernier._core` module. Called
/// from the `_core` `#[pymodule]` factory in `lib.rs`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyConfusionMatrix>()?;
    m.add_class::<PyClassSemanticStats>()?;
    m.add_class::<PySemanticSummary>()?;
    m.add_class::<PyBackgroundSemanticEvaluator>()?;
    m.add_function(wrap_pyfunction!(evaluate_semantic_from_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_semantic_to_partial, m)?)?;
    m.add_function(wrap_pyfunction!(merge_semantic_partials, m)?)?;
    Ok(())
}
