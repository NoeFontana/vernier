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

use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};

use vernier_semantic::{
    accumulate_confusion, summarize, ClassSemanticStats, ConfusionMatrix, ParityMode,
    SemanticError, SemanticSummary, StreamingSemanticEvaluator,
};

/// Per-image label-map shape `(height, width, flat_pixels)`. Aliased
/// to keep [`parse_label_maps`]' return type readable.
type LabelMap = (u32, u32, Vec<u32>);

/// Image id type. Mirrors [`vernier_semantic::error::ImageId`] (`i64`)
/// so the FFI surface uses one integer width across the project.
type ImageId = i64;

fn parse_label_maps<'py>(
    label_maps: &Bound<'py, PyDict>,
    side: &'static str,
) -> PyResult<HashMap<ImageId, LabelMap>> {
    let mut out: HashMap<ImageId, LabelMap> = HashMap::with_capacity(label_maps.len());
    for (key, value) in label_maps.iter() {
        let image_id: ImageId = key.extract().map_err(|e| {
            PyValueError::new_err(format!(
                "semantic {side} label_maps dict key must be an integer image id: {e}"
            ))
        })?;
        let arr: PyReadonlyArray2<u32> = value.extract().map_err(|e| {
            PyValueError::new_err(format!(
                "semantic {side} label_maps[{image_id}] must be a 2-D uint32 ndarray: {e}"
            ))
        })?;
        let view = arr.as_array();
        let (h, w) = (view.shape()[0], view.shape()[1]);
        if h > u32::MAX as usize || w > u32::MAX as usize {
            return Err(PyValueError::new_err(format!(
                "semantic {side} label_maps[{image_id}] shape ({h}, {w}) exceeds u32 bounds"
            )));
        }
        // Materialize to a flat Vec<u32>. The kernel walks pixel pairs
        // linearly; row-major flatten is a single allocation.
        let buf: Vec<u32> = view.iter().copied().collect();
        if out.insert(image_id, (h as u32, w as u32, buf)).is_some() {
            // PyDict can't have duplicate keys, but extracting via
            // `iter()` over the dict could in principle surface the
            // same key twice for non-canonical Python mappings. The
            // typed variant on the Rust side is DuplicateImageId; the
            // FFI surfaces it as PyValueError so the user's catch
            // site is consistent across the whole semantic surface.
            return Err(PyValueError::new_err(format!(
                "semantic {side} label_maps has duplicate image_id={image_id}"
            )));
        }
    }
    Ok(out)
}

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

/// Map a [`SemanticError`] to a Python `ValueError` with a structured
/// message. Mirrors
/// [`crate::panoptic::panoptic_error_to_pyerr`](super::panoptic): the
/// structured fields go into the message body so the parity harness
/// can lift them programmatically.
fn semantic_error_to_pyerr(e: &SemanticError) -> PyErr {
    PyValueError::new_err(format!("{e}"))
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
    let mut gt_maps = parse_label_maps(gt_label_maps, "gt")?;
    let mut dt_maps = parse_label_maps(dt_label_maps, "dt")?;
    let remap = parse_label_remap(label_remap)?;

    // Validate that every GT image has a matching DT image. Quirk
    // AM1 — strict against MS / CS / PA. Surface the missing image id
    // as a typed error.
    for image_id in gt_maps.keys() {
        if !dt_maps.contains_key(image_id) {
            return Err(semantic_error_to_pyerr(&SemanticError::MissingPrediction {
                image_id: *image_id,
            }));
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
        return Err(semantic_error_to_pyerr(&SemanticError::EmptyDataset));
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
            return Err(semantic_error_to_pyerr(&SemanticError::ShapeMismatch {
                image_id: *image_id,
                gt_shape: (gt.0, gt.1),
                dt_shape: (dt.0, dt.1),
            }));
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

/// Streaming semantic-segmentation evaluator (ADR-0028 §"Streaming").
///
/// Wraps [`vernier_semantic::StreamingSemanticEvaluator`] for the
/// Python side. Construct, call `update(image_id, gt, dt)` per
/// image, then `snapshot()` (non-consuming) or `finalize()`
/// (consuming) to read the [`PySemanticSummary`].
///
/// Concurrency: this pyclass is **not** `frozen` because `update`
/// mutates internal state. Single-threaded per evaluator; callers
/// needing concurrent updates use a pool-of-evaluators pattern,
/// mirroring the instance `BackgroundEvaluator`.
#[pyclass(module = "vernier._core", name = "StreamingSemanticEvaluator")]
pub(crate) struct PyStreamingSemanticEvaluator {
    inner: StreamingSemanticEvaluator,
}

#[pymethods]
impl PyStreamingSemanticEvaluator {
    #[new]
    #[pyo3(signature = (n_classes, parity_mode, *, ignore_label = None))]
    fn new(n_classes: u32, parity_mode: &str, ignore_label: Option<u32>) -> PyResult<Self> {
        if n_classes == 0 {
            return Err(PyValueError::new_err(
                "StreamingSemanticEvaluator requires n_classes >= 1",
            ));
        }
        let mode = parse_parity_mode(parity_mode)?;
        Ok(Self {
            inner: StreamingSemanticEvaluator::new(n_classes, ignore_label, mode),
        })
    }

    /// Number of `update` calls accepted so far.
    #[getter]
    fn n_images(&self) -> usize {
        self.inner.n_images()
    }

    /// Number of evaluation classes.
    #[getter]
    fn n_classes(&self) -> u32 {
        self.inner.n_classes()
    }

    /// Fold one image's `(gt, dt)` label-map pair into the running
    /// confusion matrix. Drops the GIL via `py.detach` (ADR-0006)
    /// for the duration of the kernel walk.
    #[allow(
        clippy::needless_pass_by_value,
        reason = "PyO3 requires `PyReadonlyArray2` by value as a pyfunction argument; \
                  the borrow lives only for the call's duration"
    )]
    fn update<'py>(
        &mut self,
        py: Python<'py>,
        image_id: i64,
        gt: PyReadonlyArray2<'py, u32>,
        dt: PyReadonlyArray2<'py, u32>,
    ) -> PyResult<()> {
        let gt_view = gt.as_array();
        let dt_view = dt.as_array();
        let gt_shape = gt_view.shape();
        let dt_shape = dt_view.shape();
        if gt_shape != dt_shape {
            return Err(semantic_error_to_pyerr(&SemanticError::ShapeMismatch {
                image_id,
                gt_shape: (gt_shape[0] as u32, gt_shape[1] as u32),
                dt_shape: (dt_shape[0] as u32, dt_shape[1] as u32),
            }));
        }
        let gt_buf: Vec<u32> = gt_view.iter().copied().collect();
        let dt_buf: Vec<u32> = dt_view.iter().copied().collect();
        py.detach(move || self.inner.update(image_id, &gt_buf, &dt_buf))
            .map_err(|e| semantic_error_to_pyerr(&e))?;
        Ok(())
    }

    /// Compute the [`PySemanticSummary`] from the current state
    /// without consuming the evaluator.
    fn snapshot(&self, py: Python<'_>) -> PySemanticSummary {
        let summary = py.detach(|| self.inner.snapshot());
        PySemanticSummary { inner: summary }
    }

    /// Consume the evaluator's internal state and produce the final
    /// [`PySemanticSummary`]. After this call the evaluator object
    /// is in a reset state with the same shape but zero
    /// accumulation; further `update` / `snapshot` calls operate on
    /// the reset state. Users who want strict consume-once semantics
    /// should drop the Python reference instead.
    fn finalize(&mut self, py: Python<'_>) -> PySemanticSummary {
        let placeholder =
            StreamingSemanticEvaluator::new(self.inner.n_classes(), None, ParityMode::default());
        let consumed = std::mem::replace(&mut self.inner, placeholder);
        let summary = py.detach(move || consumed.finalize());
        PySemanticSummary { inner: summary }
    }

    fn __repr__(&self) -> String {
        format!(
            "StreamingSemanticEvaluator(n_classes={}, n_images={})",
            self.inner.n_classes(),
            self.inner.n_images(),
        )
    }
}

/// Register semantic FFI symbols on the `vernier._core` module. Called
/// from the `_core` `#[pymodule]` factory in `lib.rs`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyConfusionMatrix>()?;
    m.add_class::<PyClassSemanticStats>()?;
    m.add_class::<PySemanticSummary>()?;
    m.add_class::<PyStreamingSemanticEvaluator>()?;
    m.add_function(wrap_pyfunction!(evaluate_semantic_from_arrays, m)?)?;
    Ok(())
}
