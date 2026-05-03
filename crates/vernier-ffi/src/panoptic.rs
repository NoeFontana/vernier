//! PyO3 bindings for [`vernier_panoptic`] (ADR-0025).
//!
//! Surface:
//! - [`PyPanopticDataset`] / [`PyPanopticPredictions`] — frozen
//!   parsed-once handles (mirrors the [`crate::dataset::PyDataset`]
//!   pattern). `from_arrays` builds a handle from a Python dict of
//!   uint32 ndarrays + JSON-string segments_info; `from_files` reads
//!   PNG paths + a JSON file (PR-4b — Python-side helper for now).
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
use std::sync::Arc;

use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};
use serde::Deserialize;

use vernier_panoptic::{
    evaluate, CategoryId, CategoryMeta, ClassPanopticStats, ImageEntry, ImageId, PanopticDataset,
    PanopticError, PanopticPredictions, PanopticSummary, ParityMode, SegmentInfo,
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
    isthing: bool,
}

/// Convert the `iscrowd` field to bool. panopticapi accepts `int` or
/// bool (quirk **S5**); we normalize to bool.
fn parse_iscrowd(v: &serde_json::Value) -> bool {
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
                    iscrowd: parse_iscrowd(&s.iscrowd),
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
                    isthing: c.isthing,
                },
            )
        })
        .collect())
}

// ---------------------------------------------------------------------------
// Label-map extraction: `Bound<PyDict>` -> `HashMap<ImageId, LabelMap>`.
// Each value must be a 2-D contiguous uint32 array (panoptic-encoded
// segment ids, post-rgb2id). Shape is read from the array.
// ---------------------------------------------------------------------------

/// `(height, width, flat_pixels)` triple — the post-decode shape the
/// kernel consumes. Aliased to keep [`parse_label_maps`]' return type
/// readable.
type LabelMap = (u32, u32, Vec<u32>);

fn parse_label_maps<'py>(label_maps: &Bound<'py, PyDict>) -> PyResult<HashMap<ImageId, LabelMap>> {
    let mut out: HashMap<ImageId, LabelMap> = HashMap::with_capacity(label_maps.len());
    for (key, value) in label_maps.iter() {
        let image_id: ImageId = key.extract().map_err(|e| {
            PyValueError::new_err(format!(
                "panoptic label_maps dict key must be an integer image id: {e}"
            ))
        })?;
        let arr: PyReadonlyArray2<u32> = value.extract().map_err(|e| {
            PyValueError::new_err(format!(
                "panoptic label_maps[{image_id}] must be a 2-D uint32 ndarray: {e}"
            ))
        })?;
        let view = arr.as_array();
        let (h, w) = (view.shape()[0], view.shape()[1]);
        if h > u32::MAX as usize || w > u32::MAX as usize {
            return Err(PyValueError::new_err(format!(
                "panoptic label_maps[{image_id}] shape ({h}, {w}) exceeds u32 bounds"
            )));
        }
        // Materialize to a flat Vec<u32>. The kernel walks pixel pairs
        // linearly; row-major flatten is a single allocation.
        let buf: Vec<u32> = view.iter().copied().collect();
        out.insert(image_id, (h as u32, w as u32, buf));
    }
    Ok(out)
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
        label_maps: &Bound<'py, PyDict>,
        segments_info: &[u8],
        categories: &[u8],
    ) -> PyResult<Self> {
        let mut maps = parse_label_maps(label_maps)?;
        let segs = parse_segments_map(segments_info).map_err(panoptic_error_to_pyerr)?;
        let cats = parse_categories(categories).map_err(panoptic_error_to_pyerr)?;

        let mut images: HashMap<ImageId, ImageEntry> = HashMap::with_capacity(maps.len());
        for (image_id, segments) in segs {
            let (h, w, label_map) = maps.remove(&image_id).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "panoptic gt: segments_info has image_id={image_id} but label_maps does not"
                ))
            })?;
            let entry = ImageEntry::from_components(image_id, h, w, label_map, segments, "gt")
                .map_err(panoptic_error_to_pyerr)?;
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
    fn from_arrays<'py>(label_maps: &Bound<'py, PyDict>, segments_info: &[u8]) -> PyResult<Self> {
        let mut maps = parse_label_maps(label_maps)?;
        let segs = parse_segments_map(segments_info).map_err(panoptic_error_to_pyerr)?;

        let mut images: HashMap<ImageId, ImageEntry> = HashMap::with_capacity(maps.len());
        for (image_id, segments) in segs {
            let (h, w, label_map) = maps.remove(&image_id).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "panoptic dt: segments_info has image_id={image_id} but label_maps does not"
                ))
            })?;
            let mut entry = ImageEntry::from_components(image_id, h, w, label_map, segments, "dt")
                .map_err(panoptic_error_to_pyerr)?;
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

    fn __repr__(&self) -> String {
        format!(
            "ClassPanopticStats(pq={:.4}, sq={:.4}, rq={:.4}, n_tp={}, n_fp={}, n_fn={})",
            self.inner.pq,
            self.inner.sq,
            self.inner.rq,
            self.inner.n_tp,
            self.inner.n_fp,
            self.inner.n_fn,
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

    /// `dict[str, dict | None]` matching panopticapi's `pq_compute`
    /// return shape exactly when `strict=True` (the count fields are
    /// dropped from per_class to match W8). Useful for round-tripping
    /// with downstream tools that expect the upstream dict layout.
    #[pyo3(signature = (*, strict=false))]
    fn to_dict<'py>(&self, py: Python<'py>, strict: bool) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);

        let bucket =
            |label: &str, pq: f64, sq: f64, rq: f64, n: usize| -> PyResult<Bound<'py, PyDict>> {
                let d = PyDict::new(py);
                d.set_item("pq", pq)?;
                d.set_item("sq", sq)?;
                d.set_item("rq", rq)?;
                d.set_item("n", n)?;
                let _ = label;
                Ok(d)
            };

        out.set_item(
            "All",
            bucket(
                "All",
                self.inner.pq,
                self.inner.sq,
                self.inner.rq,
                self.inner.n,
            )?,
        )?;
        match (
            self.inner.pq_things,
            self.inner.sq_things,
            self.inner.rq_things,
            self.inner.n_things,
        ) {
            (Some(p), Some(s), Some(r), Some(n)) => {
                out.set_item("Things", bucket("Things", p, s, r, n)?)?;
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
                out.set_item("Stuff", bucket("Stuff", p, s, r, n)?)?;
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

// ---------------------------------------------------------------------------
// Free pyfunctions
// ---------------------------------------------------------------------------

/// Run the full panoptic evaluation (kernel + attribute + summarize)
/// against pre-built [`PyPanopticDataset`] / [`PyPanopticPredictions`].
/// Drops the GIL via `py.detach` for the duration of the kernel walk
/// (ADR-0006).
#[pyfunction]
#[pyo3(signature = (gt, dt, parity_mode, things_stuff_split=true))]
pub(crate) fn evaluate_panoptic(
    py: Python<'_>,
    gt: &PyPanopticDataset,
    dt: &PyPanopticPredictions,
    parity_mode: &str,
    things_stuff_split: bool,
) -> PyResult<PyPanopticSummary> {
    let mode = match parity_mode {
        "strict" | "Strict" => ParityMode::Strict,
        "corrected" | "Corrected" => ParityMode::Corrected,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown panoptic parity_mode {other:?}; expected 'strict' or 'corrected'"
            )))
        }
    };
    let gt_arc = Arc::clone(&gt.inner);
    let dt_arc = Arc::clone(&dt.inner);
    let summary = py
        .detach(move || evaluate(&gt_arc, &dt_arc, mode, things_stuff_split))
        .map_err(panoptic_error_to_pyerr)?;
    Ok(PyPanopticSummary { inner: summary })
}

// ---------------------------------------------------------------------------
// Error mapping
// ---------------------------------------------------------------------------

/// Map a [`PanopticError`] to a Python `ValueError` with a structured
/// message. Mirrors the LVIS pattern in
/// `crates/vernier-ffi/src/dataset.rs:lvis_error_to_pyerr`: the
/// upstream-style `EvalError -> PyValueError` shim discards the
/// structured fields, so for the panoptic variants we keep them
/// explicit in the message so users (and the parity harness) can
/// lift them programmatically.
fn panoptic_error_to_pyerr(e: PanopticError) -> PyErr {
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

/// Register panoptic FFI symbols on the `vernier._core` module. Called
/// from the `_core` `#[pymodule]` factory in `lib.rs`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPanopticDataset>()?;
    m.add_class::<PyPanopticPredictions>()?;
    m.add_class::<PyPanopticSummary>()?;
    m.add_class::<PyClassPanopticStats>()?;
    m.add_function(wrap_pyfunction!(evaluate_panoptic, m)?)?;
    Ok(())
}
