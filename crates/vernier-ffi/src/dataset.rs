//! Parsed-once GT handle exposed to Python as `vernier._core.Dataset`
//! (ADR-0020).
//!
//! The bytes-path `evaluate` calls keep their current shape; this
//! module adds an alternate entry where the GT JSON is parsed once and
//! reused. The handle owns the per-kernel caches that already exist
//! in [`vernier_core`] — [`BoundaryGtCache`] and [`SegmGtCache`] — so
//! repeated `evaluate` calls against the same `Dataset` skip the
//! per-annotation derivations the kernels would otherwise rebuild.
//!
//! Bbox and Keypoints kernels have no GT-side derivation cache today.
//! For those, the dataset path still saves the GT-JSON parse but is
//! otherwise identical to the bytes path.

use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyFrozenSet, PyTuple};

use vernier_core::dataset::{CategoryId, ImageId};
use vernier_core::similarity::{BoundaryGtCache, SegmGtCache};
use vernier_core::{CocoDataset, EvalDataset, EvalError};

use crate::parse_gt;

/// Parsed-once COCO ground-truth dataset.
///
/// Construct with [`PyDataset::from_json`]; pass to the
/// `evaluate_*_summary_with_dataset` family. Reusing the same instance
/// across `evaluate` calls reuses the GT-side derivations the cached
/// kernels populate on first use (per ADR-0020). The handle is frozen
/// — its identity *is* the cache key.
#[pyclass(module = "vernier._core", name = "CocoDataset", frozen)]
pub(crate) struct PyDataset {
    inner: Arc<CocoDataset>,
    boundary_cache: Arc<BoundaryGtCache>,
    segm_cache: Arc<SegmGtCache>,
}

/// `Send`-able snapshot of a [`PyDataset`]'s contents — three `Arc`
/// clones bundled for the per-call hand-off to `py.detach`. Adding a
/// future cache slot (OKS visibility, segm GT counts) means one extra
/// field here, not another `run_pipeline_*` argument.
#[derive(Clone)]
pub(crate) struct DatasetSnapshot {
    pub(crate) gt: Arc<CocoDataset>,
    pub(crate) boundary_cache: Arc<BoundaryGtCache>,
    pub(crate) segm_cache: Arc<SegmGtCache>,
}

impl DatasetSnapshot {
    /// Wrap a freshly parsed [`CocoDataset`] in a snapshot with empty
    /// per-kernel caches. Used by `evaluate_*_grid` (the JSON-bytes
    /// path) so the grid can later hand back a [`PyDataset`] without
    /// re-parsing.
    pub(crate) fn from_parsed(gt: CocoDataset) -> Self {
        Self {
            gt: Arc::new(gt),
            boundary_cache: Arc::new(BoundaryGtCache::new()),
            segm_cache: Arc::new(SegmGtCache::new()),
        }
    }

    /// Consume the snapshot and surface its three components for a
    /// caller that owns the streaming substrate. The streaming
    /// evaluator takes a `CocoDataset` by value, so we unwrap the
    /// `Arc` (cheap when our `Arc` is the sole holder — bytes path)
    /// or clone the inner value (dataset-handle path, where the
    /// caller-owned `CocoDataset` keeps the original alive).
    pub(crate) fn into_parts(self) -> (CocoDataset, Arc<BoundaryGtCache>, Arc<SegmGtCache>) {
        (
            Arc::unwrap_or_clone(self.gt),
            self.boundary_cache,
            self.segm_cache,
        )
    }
}

impl DatasetSnapshot {
    pub(crate) fn caches(&self) -> DatasetCaches<'_> {
        DatasetCaches {
            boundary: &self.boundary_cache,
            segm: &self.segm_cache,
        }
    }
}

/// Borrow view over the per-kernel GT-side caches (ADR-0020). Threads
/// through `run_pipeline_with_dataset` and `EvalIouType::run_cached`
/// so the dispatch signature stays a single argument as more cache
/// slots get added. `Copy` because all fields are shared references.
#[derive(Clone, Copy)]
pub(crate) struct DatasetCaches<'a> {
    pub(crate) boundary: &'a BoundaryGtCache,
    pub(crate) segm: &'a SegmGtCache,
}

impl PyDataset {
    /// Capture the dataset's GT and per-kernel caches as `Arc` clones,
    /// so the resulting [`DatasetSnapshot`] can be moved into a
    /// `py.detach` closure while the originals stay live on this
    /// handle.
    pub(crate) fn snapshot(&self) -> DatasetSnapshot {
        DatasetSnapshot {
            gt: Arc::clone(&self.inner),
            boundary_cache: Arc::clone(&self.boundary_cache),
            segm_cache: Arc::clone(&self.segm_cache),
        }
    }

    /// Crate-internal `Arc` clone of the GT dataset. Used by the
    /// result-tables FFI module so the table builder closure can move
    /// the dataset into `py.detach` without re-parsing the JSON.
    pub(crate) fn dataset_ref(&self) -> Arc<vernier_core::CocoDataset> {
        Arc::clone(&self.inner)
    }

    /// Reconstruct a handle from a [`DatasetSnapshot`] (three `Arc`
    /// clones; no parsing). Used by `PyEvalGrid::dataset` so the
    /// `tables=` path doesn't re-parse GT JSON the grid already
    /// produced.
    pub(crate) fn from_snapshot(s: DatasetSnapshot) -> Self {
        Self {
            inner: s.gt,
            boundary_cache: s.boundary_cache,
            segm_cache: s.segm_cache,
        }
    }
}

#[pymethods]
impl PyDataset {
    /// Parses a COCO ground-truth JSON payload into a reusable
    /// [`Dataset`] handle. Raises `ValueError` on malformed JSON.
    #[staticmethod]
    fn from_json(gt_json: &Bound<'_, PyBytes>) -> PyResult<Self> {
        let gt = parse_gt(gt_json.as_bytes())?;
        Ok(Self {
            inner: Arc::new(gt),
            boundary_cache: Arc::new(BoundaryGtCache::new()),
            segm_cache: Arc::new(SegmGtCache::new()),
        })
    }

    /// Parses an LVIS v1 ground-truth JSON payload into a reusable
    /// [`Dataset`] handle. The handle exposes the federated metadata
    /// (`pos_category_ids`, `neg_category_ids`,
    /// `not_exhaustive_category_ids`, `category_frequency`) the
    /// orchestrator reads to apply LVIS evaluation semantics
    /// (ADR-0026).
    ///
    /// Raises `ValueError` on malformed JSON, on the disjointness
    /// violations of quirk **AA7** (a category in both `not_exhaustive`
    /// and `neg`, or a `neg` category that has GT on the same image),
    /// or on missing `frequency` tags (quirk **AB6**).
    ///
    /// Migration guide for users coming from `lvis-api`:
    /// `docs/explanation/lvis-migration.md`. Lead with the silent-
    /// federated-semantics gotcha — loading LVIS-shaped JSON via
    /// `Dataset.from_json` (the COCO loader) silently drops the
    /// federated extras and produces systematically lower AP under
    /// COCO semantics.
    #[staticmethod]
    fn from_lvis_json(gt_json: &Bound<'_, PyBytes>) -> PyResult<Self> {
        let gt =
            CocoDataset::from_lvis_json_bytes(gt_json.as_bytes()).map_err(lvis_error_to_pyerr)?;
        Ok(Self {
            inner: Arc::new(gt),
            boundary_cache: Arc::new(BoundaryGtCache::new()),
            segm_cache: Arc::new(SegmGtCache::new()),
        })
    }

    /// Per-image positive-category set (quirk **AA1**, derived from
    /// GTs at load). `None` when this dataset was loaded via
    /// [`Self::from_json`] (COCO path) rather than
    /// [`Self::from_lvis_json`].
    #[getter]
    fn pos_category_ids<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        federated_image_map_to_dict(py, self.inner.pos_category_ids())
    }

    /// Per-image negative-category set (quirk **AA2**). `None` when
    /// this dataset is not federated.
    #[getter]
    fn neg_category_ids<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        federated_image_map_to_dict(py, self.inner.neg_category_ids())
    }

    /// Per-image not-exhaustive-category set (quirk **AA3**). `None`
    /// when this dataset is not federated.
    #[getter]
    fn not_exhaustive_category_ids<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        federated_image_map_to_dict(py, self.inner.not_exhaustive_category_ids())
    }

    /// Per-category frequency tag as the LVIS single-letter form
    /// (`"r"` / `"c"` / `"f"`; quirk **AB1**). `None` when this
    /// dataset is not federated.
    #[getter]
    fn category_frequency<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(map) = self.inner.category_frequency() else {
            return Ok(None);
        };
        let dict = PyDict::new(py);
        for (cat_id, freq) in map {
            dict.set_item(cat_id.0, freq.as_letter())?;
        }
        Ok(Some(dict))
    }

    /// `True` when this dataset carries LVIS federated metadata —
    /// equivalent to `pos_category_ids is not None`. Cheap shortcut
    /// for orchestration code that gates behaviour on the federated
    /// flag.
    #[getter]
    fn is_federated(&self) -> bool {
        self.inner.is_federated()
    }

    /// Number of GT annotations carried by the dataset.
    #[getter]
    fn num_annotations(&self) -> usize {
        self.inner.annotations().len()
    }

    /// Number of images.
    #[getter]
    fn num_images(&self) -> usize {
        self.inner.images().len()
    }

    /// Number of categories.
    #[getter]
    fn num_categories(&self) -> usize {
        self.inner.categories().len()
    }

    /// Drops every cached GT-side derivation. Reset point for users
    /// who want to free memory between long-lived training cycles
    /// without dropping the dataset itself.
    fn clear_cache(&self) {
        self.boundary_cache.clear();
        self.segm_cache.clear();
    }

    /// Observability-only: count of GT annotations whose boundary
    /// band is currently cached (ADR-0020). Useful for debugging or
    /// tests that need to assert cache reuse; not a stable contract,
    /// and the value can change shape as new cache slots are added.
    #[getter]
    fn boundary_cache_len(&self) -> usize {
        self.boundary_cache.len()
    }

    /// Observability-only: count of GT annotations whose segm
    /// bbox+area derivation is currently cached (ADR-0020). Same
    /// caveats as [`Self::boundary_cache_len`].
    #[getter]
    fn segm_cache_len(&self) -> usize {
        self.segm_cache.len()
    }

    fn __repr__(&self) -> String {
        let federated = if self.inner.is_federated() {
            ", federated=True"
        } else {
            ""
        };
        format!(
            "CocoDataset(images={}, annotations={}, categories={}{federated})",
            self.inner.images().len(),
            self.inner.annotations().len(),
            self.inner.categories().len(),
        )
    }
}

/// Convert an `Option<&HashMap<ImageId, HashSet<CategoryId>>>` (the
/// shape of `pos` / `neg` / `not_exhaustive` on the dataset) to an
/// optional Python `dict[int, frozenset[int]]`. Returns `None`
/// (Python `None`) when the map is absent — the caller treats this
/// as "dataset is COCO-flat, not federated".
fn federated_image_map_to_dict<'py>(
    py: Python<'py>,
    map: Option<&std::collections::HashMap<ImageId, std::collections::HashSet<CategoryId>>>,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let Some(map) = map else {
        return Ok(None);
    };
    let dict = PyDict::new(py);
    for (image_id, cats) in map {
        let ids: Vec<i64> = cats.iter().map(|c| c.0).collect();
        let py_ids = PyTuple::new(py, ids)?;
        let frozen = PyFrozenSet::new(py, py_ids.iter())?;
        dict.set_item(image_id.0, frozen)?;
    }
    Ok(Some(dict))
}

/// Map an [`EvalError`] from the LVIS loader to a Python `ValueError`
/// with a structured message. The plain `EvalError -> PyValueError`
/// shim used by other paths discards the structured fields; for the
/// LVIS variants we keep the field information explicit so users (and
/// the migration guide) can lift it programmatically.
fn lvis_error_to_pyerr(e: EvalError) -> PyErr {
    match e {
        EvalError::LvisFederatedConflict {
            image_id,
            category_id,
            detail,
        } => PyValueError::new_err(format!(
            "lvis federated conflict on image_id={image_id}, category_id={category_id}: {detail}"
        )),
        EvalError::MissingFrequency { category_ids } => PyValueError::new_err(format!(
            "lvis dataset is missing `frequency` on {} categories: {category_ids:?}",
            category_ids.len()
        )),
        other => PyValueError::new_err(format!("{other}")),
    }
}
