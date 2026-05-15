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

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use std::time::Duration;

use numpy::PyReadonlyArray2;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyDictMethods, PyList};

use vernier_partial::PartialError;
use vernier_semantic::decode::{decode_grayscale8, evaluate_from_pngs};
use vernier_semantic::kernel::accumulate_confusion;
use vernier_semantic::{
    summarize_with_options, ClassSemanticStats, ConfusionMatrix, GroupSemanticStats, SemanticError,
    SemanticSummary, StreamingSemanticEvaluator, SummarizeOptions,
};

use crate::arrow_helpers::{wrap_batch, ArrowRecordBatchPy};
use crate::background::BackgroundConfig;
use crate::background_streaming::{
    BackgroundCapable, BackgroundCore, BackgroundLifecycle, SubmitError,
};
use crate::manifest_py::manifest_to_canonical_json;
use crate::numpy_utils::ImageId;
use crate::tables::{slices_record_batch_semantic, SemanticSliceRow};
use crate::{poll_scheduling_warning, queue_full_to_pyerr, validate_shutdown_timeout};
use vernier_core::manifest::partition_spec_from_manifest;

/// Per-image label-map buffer extracted from a numpy ndarray. Lets the
/// semantic FFI accept `uint8` / `uint16` / `uint32` natively (ADR-0037);
/// the kernel walks at native width via the [`ClassId`] trait.
///
/// [`ClassId`]: vernier_semantic::kernel::ClassId
pub(crate) enum SemanticPixelBuf {
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
}

/// Extract a 2-D `uint8` / `uint16` / `uint32` ndarray into a typed
/// pixel buffer. Returns the buffer plus `(height, width)`.
fn extract_label_map(
    arr: &Bound<'_, PyAny>,
    context: &str,
    image_id: ImageId,
) -> PyResult<(SemanticPixelBuf, (u32, u32))> {
    if let Ok(a) = arr.extract::<PyReadonlyArray2<u32>>() {
        let view = a.as_array();
        let (h, w) = check_shape(image_id, context, view.shape())?;
        return Ok((
            SemanticPixelBuf::U32(view.iter().copied().collect()),
            (h, w),
        ));
    }
    if let Ok(a) = arr.extract::<PyReadonlyArray2<u16>>() {
        let view = a.as_array();
        let (h, w) = check_shape(image_id, context, view.shape())?;
        return Ok((
            SemanticPixelBuf::U16(view.iter().copied().collect()),
            (h, w),
        ));
    }
    if let Ok(a) = arr.extract::<PyReadonlyArray2<u8>>() {
        let view = a.as_array();
        let (h, w) = check_shape(image_id, context, view.shape())?;
        return Ok((SemanticPixelBuf::U8(view.iter().copied().collect()), (h, w)));
    }
    Err(PyValueError::new_err(format!(
        "{context} label_maps[{image_id}] must be a 2-D uint8/uint16/uint32 ndarray"
    )))
}

fn check_shape(image_id: ImageId, context: &str, shape: &[usize]) -> PyResult<(u32, u32)> {
    let (h, w) = (shape[0], shape[1]);
    if h > u32::MAX as usize || w > u32::MAX as usize {
        return Err(PyValueError::new_err(format!(
            "{context} label_maps[{image_id}] shape ({h}, {w}) exceeds u32 bounds"
        )));
    }
    Ok((h as u32, w as u32))
}

/// `(height, width, decoded_pixels)` triple for one image's label map.
type SemanticLabelMap = (u32, u32, SemanticPixelBuf);

fn parse_semantic_label_maps(
    label_maps: &Bound<'_, PyDict>,
    context: &str,
) -> PyResult<HashMap<ImageId, SemanticLabelMap>> {
    let mut out: HashMap<ImageId, SemanticLabelMap> = HashMap::with_capacity(label_maps.len());
    for (key, value) in label_maps.iter() {
        let image_id: ImageId = key.extract().map_err(|e| {
            PyValueError::new_err(format!(
                "{context} label_maps dict key must be an integer image id: {e}"
            ))
        })?;
        let (buf, (h, w)) = extract_label_map(&value, context, image_id)?;
        if out.insert(image_id, (h, w, buf)).is_some() {
            return Err(PyValueError::new_err(format!(
                "{context} label_maps has duplicate image_id={image_id}"
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

/// Per-group rollup row (ADR-0041).
///
/// Mirrors [`vernier_semantic::GroupSemanticStats`] one-to-one. Built
/// only when the caller passes `class_grouping=` to
/// `evaluate_semantic_from_arrays`; reachable from Python via
/// `SemanticSummary.per_group`.
#[pyclass(
    module = "vernier._core",
    name = "GroupSemanticStats",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyGroupSemanticStats {
    inner: GroupSemanticStats,
}

#[pymethods]
impl PyGroupSemanticStats {
    #[getter]
    fn label(&self) -> &str {
        &self.inner.label
    }
    #[getter]
    fn member_class_ids(&self) -> Vec<u32> {
        self.inner.member_class_ids.clone()
    }
    #[getter]
    fn miou(&self) -> f64 {
        self.inner.miou
    }
    #[getter]
    fn mean_accuracy(&self) -> f64 {
        self.inner.mean_accuracy
    }
    #[getter]
    fn pixel_accuracy(&self) -> f64 {
        self.inner.pixel_accuracy
    }
    #[getter]
    fn fwiou(&self) -> f64 {
        self.inner.fwiou
    }

    fn __repr__(&self) -> String {
        format!(
            "GroupSemanticStats(label={:?}, miou={:.4}, mean_accuracy={:.4}, pixel_accuracy={:.4}, fwiou={:.4})",
            self.inner.label,
            self.inner.miou,
            self.inner.mean_accuracy,
            self.inner.pixel_accuracy,
            self.inner.fwiou,
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

    /// Per-group rollup keyed by group label (ADR-0041). Empty when
    /// the evaluator was run without `class_grouping`. Returns a fresh
    /// dict on each call.
    fn per_group<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (label, row) in &self.inner.per_group {
            dict.set_item(label, PyGroupSemanticStats { inner: row.clone() })?;
        }
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        format!(
            "SemanticSummary(miou={:.4}, fwiou={:.4}, pixel_accuracy={:.4}, mean_accuracy={:.4})",
            self.inner.miou, self.inner.fwiou, self.inner.pixel_accuracy, self.inner.mean_accuracy,
        )
    }
}

impl PySemanticSummary {
    pub(crate) fn summary_ref(&self) -> &SemanticSummary {
        &self.inner
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
    class_filter = None,
    class_grouping = None,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_semantic_from_arrays<'py>(
    py: Python<'py>,
    gt_label_maps: &Bound<'py, PyDict>,
    dt_label_maps: &Bound<'py, PyDict>,
    n_classes: u32,
    parity_mode: &str,
    ignore_label: Option<u32>,
    label_remap: Option<&Bound<'py, PyDict>>,
    class_filter: Option<Vec<u32>>,
    class_grouping: Option<Vec<(String, Vec<u32>)>>,
) -> PyResult<PySemanticSummary> {
    if n_classes == 0 {
        return Err(PyValueError::new_err(
            "semantic evaluator requires n_classes >= 1",
        ));
    }
    let mode = crate::parse_parity_mode(parity_mode)?;
    let mut gt_maps = parse_semantic_label_maps(gt_label_maps, "semantic gt")?;
    let mut dt_maps = parse_semantic_label_maps(dt_label_maps, "semantic dt")?;
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
    // loop. Remap output values are u32 — buffers narrower than u32
    // promote here so the remap range can hit any class id.
    if let Some(remap) = &remap {
        for (_, _, buf) in dt_maps.values_mut() {
            *buf = apply_remap_promote_u32(buf, remap);
        }
    }

    if gt_maps.is_empty() {
        return Err(semantic_error_to_pyerr(py, &SemanticError::EmptyDataset));
    }

    let mut image_ids: Vec<ImageId> = gt_maps.keys().copied().collect();
    image_ids.sort_unstable();

    let mut work: Vec<(ImageId, SemanticLabelMap, SemanticLabelMap)> =
        Vec::with_capacity(image_ids.len());
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

    let summary = py.detach(move || {
        let mut confusion = ConfusionMatrix::zeros(n_classes);
        for (_, (_, _, gt_buf), (_, _, dt_buf)) in &work {
            fold_pair_buf(gt_buf, dt_buf, ignore_label, &mut confusion);
        }
        let options = SummarizeOptions {
            class_filter: class_filter.as_deref(),
            class_groups: class_grouping.as_deref(),
        };
        summarize_with_options(confusion, mode, &options)
    });

    Ok(PySemanticSummary { inner: summary })
}

// ---------------------------------------------------------------------------
// Partitioned semantic eval (ADR-0046 C3).
//
// One pass over the input images builds a `Vec<(image_id,
// ConfusionMatrix)>` of per-image deltas; the partition orchestrator
// then sums only the matrices for slice images and summarizes from
// that fold. Confusion-matrix sums are u64-additive, so the fold is
// bit-identical to a fresh kernel pass over the slice (no f64
// non-associativity to worry about, unlike the panoptic path).
// ---------------------------------------------------------------------------

/// Test-only call counter for the semantic per-image fold pass.
/// Symmetric to the panoptic counter; the Python perf test asserts
/// the per-image fold runs **exactly once** regardless of slice count.
#[cfg(any(test, feature = "_test-counter"))]
static SEMANTIC_FOLD_PASS_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

#[cfg(any(test, feature = "_test-counter"))]
fn inc_semantic_fold_count() {
    SEMANTIC_FOLD_PASS_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(not(any(test, feature = "_test-counter")))]
fn inc_semantic_fold_count() {}

#[cfg(any(test, feature = "_test-counter"))]
#[pyfunction]
pub(crate) fn _test_reset_semantic_fold_count() -> u64 {
    SEMANTIC_FOLD_PASS_COUNT.swap(0, std::sync::atomic::Ordering::Relaxed)
}

#[cfg(any(test, feature = "_test-counter"))]
#[pyfunction]
pub(crate) fn _test_read_semantic_fold_count() -> u64 {
    SEMANTIC_FOLD_PASS_COUNT.load(std::sync::atomic::Ordering::Relaxed)
}

/// Result of an ADR-0046 C3 partitioned semantic eval.
///
/// `overall` is bit-identical to a non-partitioned
/// `evaluate_semantic_from_arrays` over the same inputs.
#[pyclass(module = "vernier._core", name = "PartitionedSemanticReport", frozen)]
pub(crate) struct PyPartitionedSemanticReport {
    summary: SemanticSummary,
    overall_n_images: u64,
    slice_rows: Vec<SemanticSliceRow>,
}

#[pymethods]
impl PyPartitionedSemanticReport {
    /// Bit-identical to a non-partitioned `evaluate_semantic_from_arrays`
    /// over the same inputs.
    #[getter]
    fn overall(&self) -> PySemanticSummary {
        PySemanticSummary {
            inner: self.summary.clone(),
        }
    }

    /// Image count behind `overall`.
    #[getter]
    fn overall_n_images(&self) -> u64 {
        self.overall_n_images
    }

    /// Always `0` — semantic has no detection notion; the column is
    /// shape-parity with panoptic / instance.
    #[getter]
    fn overall_n_detections(&self) -> u64 {
        0
    }

    /// Number of `(axis, value)` cells in the partition.
    #[getter]
    fn n_slices(&self) -> usize {
        self.slice_rows.len()
    }

    /// Arrow `RecordBatch` of per-slice rows. Built fresh per call.
    fn slices_capsule(&self) -> PyResult<ArrowRecordBatchPy> {
        let batch = slices_record_batch_semantic(&self.slice_rows)
            .map_err(|e| PyValueError::new_err(format!("arrow build failed: {e}")))?;
        Ok(wrap_batch(batch))
    }

    fn __repr__(&self) -> String {
        format!(
            "PartitionedSemanticReport(overall_n_images={}, n_slices={})",
            self.overall_n_images,
            self.slice_rows.len(),
        )
    }
}

fn warn_about_manifest_sem(
    py: Python<'_>,
    warnings: &[vernier_core::manifest::ManifestWarning],
) -> PyResult<()> {
    if warnings.is_empty() {
        return Ok(());
    }
    let warnings_mod = py.import("warnings")?;
    for w in warnings {
        let msg = match w {
            vernier_core::manifest::ManifestWarning::UnknownKey { key } => {
                format!("manifest key {key:?} is not present in the dataset; skipping")
            }
        };
        warnings_mod.call_method1("warn", (msg,))?;
    }
    Ok(())
}

/// C3 partitioned semantic eval (ADR-0046 §"Performance").
///
/// Folds each image's `(gt, dt)` pair into a per-image confusion
/// matrix exactly **once**, then aggregates + summarizes those
/// matrices under (a) no filter for `overall` and (b) each slice's
/// image-id set for the per-slice rows. The kernel is never re-run
/// per slice — the load-bearing C3 axiom.
#[pyfunction]
#[pyo3(signature = (
    gt_label_maps,
    dt_label_maps,
    n_classes,
    parity_mode,
    manifest,
    *,
    ignore_label = None,
    label_remap = None,
    class_filter = None,
    class_grouping = None,
    cross_axes = None,
    key_kind = "image_id",
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_semantic_partitioned<'py>(
    py: Python<'py>,
    gt_label_maps: &Bound<'py, PyDict>,
    dt_label_maps: &Bound<'py, PyDict>,
    n_classes: u32,
    parity_mode: &str,
    manifest: &Bound<'py, PyAny>,
    ignore_label: Option<u32>,
    label_remap: Option<&Bound<'py, PyDict>>,
    class_filter: Option<Vec<u32>>,
    class_grouping: Option<Vec<(String, Vec<u32>)>>,
    cross_axes: Option<Vec<Vec<String>>>,
    key_kind: &str,
) -> PyResult<PyPartitionedSemanticReport> {
    if n_classes == 0 {
        return Err(PyValueError::new_err(
            "semantic evaluator requires n_classes >= 1",
        ));
    }
    let mode = crate::parse_parity_mode(parity_mode)?;
    let mut gt_maps = parse_semantic_label_maps(gt_label_maps, "semantic gt")?;
    let mut dt_maps = parse_semantic_label_maps(dt_label_maps, "semantic dt")?;
    let remap = parse_label_remap(label_remap)?;

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

    if let Some(remap) = &remap {
        for (_, _, buf) in dt_maps.values_mut() {
            *buf = apply_remap_promote_u32(buf, remap);
        }
    }

    if gt_maps.is_empty() {
        return Err(semantic_error_to_pyerr(py, &SemanticError::EmptyDataset));
    }

    let mut image_ids: Vec<ImageId> = gt_maps.keys().copied().collect();
    image_ids.sort_unstable();

    let manifest_bytes = manifest_to_canonical_json(py, manifest, key_kind)?;
    let cross = cross_axes.unwrap_or_default();

    let image_id_to_idx: HashMap<vernier_core::dataset::ImageId, usize> = image_ids
        .iter()
        .enumerate()
        .map(|(i, id)| (vernier_core::dataset::ImageId(*id), i))
        .collect();
    let (spec, warnings) = partition_spec_from_manifest(&manifest_bytes, &image_id_to_idx, &cross)
        .map_err(|e| PyValueError::new_err(format!("manifest resolution failed: {e}")))?;
    warn_about_manifest_sem(py, &warnings)?;

    // Pre-compute per-slice image-id filter sets on the FFI thread.
    let mut slice_inputs: Vec<(String, String, HashSet<ImageId>, u64)> =
        Vec::with_capacity(spec.slices.len());
    for sl in &spec.slices {
        let sem_ids: HashSet<ImageId> = sl.image_ids.iter().map(|id| id.0).collect();
        let n_images = sem_ids.len() as u64;
        slice_inputs.push((sl.axis.clone(), sl.value.clone(), sem_ids, n_images));
    }

    let n_images_overall = image_ids.len() as u64;

    // Build the per-image work vec exactly like the un-partitioned
    // path so the kernel walk is identical (matching pass parity).
    let mut work: Vec<(ImageId, SemanticLabelMap, SemanticLabelMap)> =
        Vec::with_capacity(image_ids.len());
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

    struct SemSliceMetrics {
        axis: String,
        value: String,
        n_images: u64,
        miou: f64,
        fwiou: f64,
        pixel_accuracy: f64,
        mean_accuracy: f64,
    }
    type C3Out = (SemanticSummary, Vec<SemSliceMetrics>);

    let (overall_summary, slice_metrics) = py.detach(move || -> C3Out {
        inc_semantic_fold_count();
        // Per-image confusion matrices. Built once; folded under
        // different filters at summarize time (C3).
        let mut per_image: Vec<(ImageId, ConfusionMatrix)> = Vec::with_capacity(work.len());
        for (image_id, (_, _, gt_buf), (_, _, dt_buf)) in &work {
            let mut cm = ConfusionMatrix::zeros(n_classes);
            fold_pair_buf(gt_buf, dt_buf, ignore_label, &mut cm);
            per_image.push((*image_id, cm));
        }

        let options = SummarizeOptions {
            class_filter: class_filter.as_deref(),
            class_groups: class_grouping.as_deref(),
        };

        // Overall — un-filtered sum.
        let mut overall_cm = ConfusionMatrix::zeros(n_classes);
        for (_, cm) in &per_image {
            overall_cm.add_assign_unchecked(cm);
        }
        let overall = summarize_with_options(overall_cm, mode, &options);

        // Per-slice — sum only the matrices for the slice's images.
        let mut metrics: Vec<SemSliceMetrics> = Vec::with_capacity(slice_inputs.len());
        for (axis, value, ids, n_images) in slice_inputs {
            if ids.is_empty() {
                metrics.push(SemSliceMetrics {
                    axis,
                    value,
                    n_images: 0,
                    miou: 0.0,
                    fwiou: 0.0,
                    pixel_accuracy: 0.0,
                    mean_accuracy: 0.0,
                });
                continue;
            }
            let mut slice_cm = ConfusionMatrix::zeros(n_classes);
            for (image_id, cm) in &per_image {
                if ids.contains(image_id) {
                    slice_cm.add_assign_unchecked(cm);
                }
            }
            let summary = summarize_with_options(slice_cm, mode, &options);
            metrics.push(SemSliceMetrics {
                axis,
                value,
                n_images,
                miou: summary.miou,
                fwiou: summary.fwiou,
                pixel_accuracy: summary.pixel_accuracy,
                mean_accuracy: summary.mean_accuracy,
            });
        }
        (overall, metrics)
    });

    let slice_rows: Vec<SemanticSliceRow> = slice_metrics
        .into_iter()
        .map(|m| SemanticSliceRow {
            axis: m.axis,
            value: m.value,
            n_images: m.n_images,
            n_detections: 0,
            miou: m.miou,
            fwiou: m.fwiou,
            pixel_accuracy: m.pixel_accuracy,
            mean_accuracy: m.mean_accuracy,
        })
        .collect();
    Ok(PyPartitionedSemanticReport {
        summary: overall_summary,
        overall_n_images: n_images_overall,
        slice_rows,
    })
}

/// Dispatch [`accumulate_confusion`] over the runtime dtype of a
/// `(gt, dt)` [`SemanticPixelBuf`] pair. Buffers can be a mix of
/// widths (e.g., `u8` GT with `u32` remapped DT after `apply_remap`).
fn fold_pair_buf(
    gt: &SemanticPixelBuf,
    dt: &SemanticPixelBuf,
    ignore_label: Option<u32>,
    confusion: &mut ConfusionMatrix,
) {
    use SemanticPixelBuf::{U16, U32, U8};
    match (gt, dt) {
        (U8(g), U8(d)) => accumulate_confusion(g, d, ignore_label, confusion),
        (U16(g), U16(d)) => accumulate_confusion(g, d, ignore_label, confusion),
        (U32(g), U32(d)) => accumulate_confusion(g, d, ignore_label, confusion),
        // Mixed-width pairs widen the narrower side to u32. Cheap
        // (one cast pass per image) and only fires if a caller hands
        // us mismatched dtypes — uncommon enough to warrant the cast.
        (g, d) => {
            let g32 = pixels_to_u32(g);
            let d32 = pixels_to_u32(d);
            accumulate_confusion(&g32, &d32, ignore_label, confusion);
        }
    }
}

fn pixels_to_u32(buf: &SemanticPixelBuf) -> Vec<u32> {
    match buf {
        SemanticPixelBuf::U8(v) => v.iter().map(|&x| u32::from(x)).collect(),
        SemanticPixelBuf::U16(v) => v.iter().map(|&x| u32::from(x)).collect(),
        SemanticPixelBuf::U32(v) => v.clone(),
    }
}

fn apply_remap_promote_u32(buf: &SemanticPixelBuf, remap: &HashMap<u32, u32>) -> SemanticPixelBuf {
    let promoted = pixels_to_u32(buf);
    let mut out = promoted;
    for v in out.iter_mut() {
        if let Some(&new) = remap.get(v) {
            *v = new;
        }
    }
    SemanticPixelBuf::U32(out)
}

/// Build a [`SemanticUpdate`] payload, choosing the variant by the
/// matching dtype of the input pair. Mixed-width pairs widen to u32.
fn build_payload(image_id: ImageId, gt: SemanticPixelBuf, dt: SemanticPixelBuf) -> SemanticUpdate {
    use SemanticPixelBuf::{U16, U32, U8};
    match (gt, dt) {
        (U8(gt), U8(dt)) => SemanticUpdate::U8 { image_id, gt, dt },
        (U16(gt), U16(dt)) => SemanticUpdate::U16 { image_id, gt, dt },
        (U32(gt), U32(dt)) => SemanticUpdate::U32 { image_id, gt, dt },
        (g, d) => SemanticUpdate::U32 {
            image_id,
            gt: pixels_to_u32(&g),
            dt: pixels_to_u32(&d),
        },
    }
}

/// Run the semantic-segmentation evaluation directly against PNG label
/// maps on disk (ADR-0037). The fused libpng decode + confusion-matrix
/// fold runs inside `py.detach`, so the GIL is released for the whole
/// batch and only one `(gt, dt)` pair of decoded buffers is in flight
/// at a time.
///
/// Format contract: 8-bit grayscale PNGs only. RGB / paletted /
/// 16-bit grayscale are rejected with `UnsupportedPngFormat`. Callers
/// with wider class-id ranges should use `evaluate_semantic_from_arrays`
/// with `np.uint16` / `np.uint32` ndarrays instead.
#[pyfunction]
#[pyo3(signature = (
    gt_paths,
    dt_paths,
    n_classes,
    parity_mode,
    *,
    ignore_label = None,
))]
pub(crate) fn evaluate_semantic_from_pngs<'py>(
    py: Python<'py>,
    gt_paths: &Bound<'py, PyDict>,
    dt_paths: &Bound<'py, PyDict>,
    n_classes: u32,
    parity_mode: &str,
    ignore_label: Option<u32>,
) -> PyResult<PySemanticSummary> {
    if n_classes == 0 {
        return Err(PyValueError::new_err(
            "evaluate_semantic_from_pngs requires n_classes >= 1",
        ));
    }
    let mode = crate::parse_parity_mode(parity_mode)?;

    let gt_pairs = parse_path_dict(gt_paths, "semantic gt")?;
    let dt_map = parse_path_dict_to_hashmap(dt_paths, "semantic dt")?;

    let summary = py
        .detach(move || -> Result<SemanticSummary, SemanticError> {
            evaluate_from_pngs(&gt_pairs, &dt_map, n_classes, ignore_label, mode)
        })
        .map_err(|e| semantic_error_to_pyerr(py, &e))?;
    Ok(PySemanticSummary { inner: summary })
}

/// Extract a `dict[image_id: int, path: str | os.PathLike]` into a
/// sorted `Vec<(image_id, PathBuf)>` (iteration order is deterministic
/// per quirk **AM5**).
fn parse_path_dict(
    d: &Bound<'_, PyDict>,
    label: &str,
) -> PyResult<Vec<(ImageId, std::path::PathBuf)>> {
    let mut out: Vec<(ImageId, std::path::PathBuf)> = Vec::with_capacity(d.len());
    for (k, v) in d.iter() {
        let image_id: ImageId = k.extract().map_err(|e| {
            PyValueError::new_err(format!("{label}: image_id keys must be int: {e}"))
        })?;
        let path: std::path::PathBuf = v.extract().map_err(|e| {
            PyValueError::new_err(format!("{label}: values must be str or os.PathLike: {e}"))
        })?;
        out.push((image_id, path));
    }
    out.sort_unstable_by_key(|(iid, _)| *iid);
    Ok(out)
}

/// Same shape as [`parse_path_dict`] but lands in a [`HashMap`] for
/// O(1) prediction lookup keyed by image_id.
fn parse_path_dict_to_hashmap(
    d: &Bound<'_, PyDict>,
    label: &str,
) -> PyResult<HashMap<ImageId, std::path::PathBuf>> {
    let mut out: HashMap<ImageId, std::path::PathBuf> = HashMap::with_capacity(d.len());
    for (k, v) in d.iter() {
        let image_id: ImageId = k.extract().map_err(|e| {
            PyValueError::new_err(format!("{label}: image_id keys must be int: {e}"))
        })?;
        let path: std::path::PathBuf = v.extract().map_err(|e| {
            PyValueError::new_err(format!("{label}: values must be str or os.PathLike: {e}"))
        })?;
        out.insert(image_id, path);
    }
    Ok(out)
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
    let mode = crate::parse_parity_mode(parity_mode)?;
    let mut gt_maps = parse_semantic_label_maps(gt_label_maps, "semantic gt")?;
    let mut dt_maps = parse_semantic_label_maps(dt_label_maps, "semantic dt")?;

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

    // Sorted iteration for determinism (quirk AM5).
    let mut image_ids: Vec<ImageId> = gt_maps.keys().copied().collect();
    image_ids.sort_unstable();

    // Construct the evaluator up front so we can stream per-image
    // updates inside the loop — only one decoded image-pair lives in
    // memory at a time on this path (vs an eager Vec over the whole
    // corpus, ~4 GiB on ADE20K val).
    let mut ev = StreamingSemanticEvaluator::new(n_classes, ignore_label, mode)
        .with_rank(rank_id)
        .map_err(|e| semantic_error_to_pyerr(py, &e))?;
    for image_id in &image_ids {
        let (gh, gw, gt_buf) = gt_maps.remove(image_id).ok_or_else(|| {
            PyValueError::new_err(format!(
                "internal: missing gt label_map for image_id={image_id}"
            ))
        })?;
        let (dh, dw, dt_buf) = dt_maps.remove(image_id).ok_or_else(|| {
            PyValueError::new_err(format!(
                "internal: missing dt label_map for image_id={image_id}"
            ))
        })?;
        if (gh, gw) != (dh, dw) {
            return Err(semantic_error_to_pyerr(
                py,
                &SemanticError::ShapeMismatch {
                    image_id: *image_id,
                    gt_shape: (gh, gw),
                    dt_shape: (dh, dw),
                },
            ));
        }
        let image_id = *image_id;
        py.detach(|| update_streaming(&mut ev, image_id, &gt_buf, &dt_buf))
            .map_err(|e| semantic_error_to_pyerr(py, &e))?;
    }

    let bytes = py
        .detach(move || ev.finalize_to_partial())
        .map_err(|e| semantic_error_to_pyerr(py, &e))?;
    Ok(PyBytes::new(py, &bytes))
}

/// Dispatch [`StreamingSemanticEvaluator::update`] over the runtime
/// dtype of a `(gt, dt)` [`SemanticPixelBuf`] pair. Same widening
/// fallback as [`fold_pair_buf`] for mixed-width inputs.
fn update_streaming(
    ev: &mut StreamingSemanticEvaluator,
    image_id: ImageId,
    gt: &SemanticPixelBuf,
    dt: &SemanticPixelBuf,
) -> Result<(), SemanticError> {
    use SemanticPixelBuf::{U16, U32, U8};
    match (gt, dt) {
        (U8(g), U8(d)) => ev.update(image_id, g, d),
        (U16(g), U16(d)) => ev.update(image_id, g, d),
        (U32(g), U32(d)) => ev.update(image_id, g, d),
        (g, d) => {
            let g32 = pixels_to_u32(g);
            let d32 = pixels_to_u32(d);
            ev.update(image_id, &g32, &d32)
        }
    }
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
    let mode = crate::parse_parity_mode(parity_mode)?;
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
/// the FFI thread builds this from the user's numpy arrays (or
/// PNG-decoded `u8` buffer) before dropping the GIL. ADR-0037: kernel
/// walks at native dtype, so the channel carries the natural width.
pub(crate) enum SemanticUpdate {
    U32 {
        image_id: ImageId,
        gt: Vec<u32>,
        dt: Vec<u32>,
    },
    U16 {
        image_id: ImageId,
        gt: Vec<u16>,
        dt: Vec<u16>,
    },
    U8 {
        image_id: ImageId,
        gt: Vec<u8>,
        dt: Vec<u8>,
    },
}

impl BackgroundCapable for StreamingSemanticEvaluator {
    type Update = SemanticUpdate;
    type Summary = SemanticSummary;
    type Error = SemanticError;

    fn apply_update(&mut self, u: SemanticUpdate) -> Result<(), SemanticError> {
        match u {
            SemanticUpdate::U32 { image_id, gt, dt } => self.update(image_id, &gt, &dt),
            SemanticUpdate::U16 { image_id, gt, dt } => self.update(image_id, &gt, &dt),
            SemanticUpdate::U8 { image_id, gt, dt } => self.update(image_id, &gt, &dt),
        }
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
        let mode = crate::parse_parity_mode(parity_mode)?;
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
    /// Accepts `uint8` / `uint16` / `uint32` 2-D ndarrays (ADR-0037);
    /// the worker walks at native dtype without an upcast. `timeout`
    /// mirrors the instance background:
    ///
    /// - `None` (default) → block until a slot is free
    /// - `0.0` → single non-blocking attempt; raise `QueueFullError`
    ///   if the queue is full
    /// - `t > 0.0` → wait up to `t` seconds; raise `QueueFullError`
    ///   on timeout
    #[pyo3(signature = (image_id, gt, dt, *, timeout = None))]
    fn submit(
        &self,
        py: Python<'_>,
        image_id: i64,
        gt: &Bound<'_, PyAny>,
        dt: &Bound<'_, PyAny>,
        timeout: Option<f64>,
    ) -> PyResult<()> {
        let (gt_buf, gt_dims) = extract_label_map(gt, "BackgroundSemanticEvaluator gt", image_id)?;
        let (dt_buf, dt_dims) = extract_label_map(dt, "BackgroundSemanticEvaluator dt", image_id)?;
        if gt_dims != dt_dims {
            return Err(semantic_error_to_pyerr(
                py,
                &SemanticError::ShapeMismatch {
                    image_id,
                    gt_shape: gt_dims,
                    dt_shape: dt_dims,
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

        let payload = build_payload(image_id, gt_buf, dt_buf);

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

    /// Submit one image's `(gt_png_bytes, dt_png_bytes)` 8-bit grayscale
    /// PNG pair to the worker (ADR-0037). Decodes synchronously on the
    /// FFI thread (under `py.detach`) and sends the native-width `u8`
    /// label maps across the channel; the worker folds at native width
    /// without a 4× upcast.
    ///
    /// Strictly equivalent to
    /// ``submit(image_id, decode_label_map_png(gt_path).astype(uint32),
    /// decode_label_map_png(dt_path).astype(uint32), ...)``; the diff
    /// is wall-time, not correctness.
    ///
    /// Format contract: 8-bit grayscale only. `timeout` mirrors `submit`.
    #[pyo3(signature = (image_id, gt_png_bytes, dt_png_bytes, *, timeout = None))]
    fn submit_png(
        &self,
        py: Python<'_>,
        image_id: i64,
        gt_png_bytes: &[u8],
        dt_png_bytes: &[u8],
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

        // Decode + send happen GIL-released so the user's main thread
        // is free to fan out the next image.
        let owned_gt_bytes = gt_png_bytes.to_vec();
        let owned_dt_bytes = dt_png_bytes.to_vec();
        let lifecycle = &self.lifecycle;
        let result = py.detach(move || -> Result<(), SubmitError<SemanticError>> {
            let (gt_buf, gt_dims) =
                decode_grayscale8(image_id, &owned_gt_bytes).map_err(SubmitError::Eval)?;
            let (dt_buf, dt_dims) =
                decode_grayscale8(image_id, &owned_dt_bytes).map_err(SubmitError::Eval)?;
            if gt_dims != dt_dims {
                return Err(SubmitError::Eval(SemanticError::ShapeMismatch {
                    image_id,
                    gt_shape: gt_dims,
                    dt_shape: dt_dims,
                }));
            }
            let payload = SemanticUpdate::U8 {
                image_id,
                gt: gt_buf,
                dt: dt_buf,
            };
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
    m.add_class::<PyGroupSemanticStats>()?;
    m.add_class::<PySemanticSummary>()?;
    m.add_class::<PyBackgroundSemanticEvaluator>()?;
    m.add_class::<PyPartitionedSemanticReport>()?;
    m.add_function(wrap_pyfunction!(evaluate_semantic_from_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_semantic_from_pngs, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_semantic_to_partial, m)?)?;
    m.add_function(wrap_pyfunction!(merge_semantic_partials, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_semantic_partitioned, m)?)?;
    #[cfg(any(test, feature = "_test-counter"))]
    {
        m.add_function(wrap_pyfunction!(_test_reset_semantic_fold_count, m)?)?;
        m.add_function(wrap_pyfunction!(_test_read_semantic_fold_count, m)?)?;
    }
    Ok(())
}
