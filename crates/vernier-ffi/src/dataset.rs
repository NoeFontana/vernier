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
use pyo3::types::PyBytes;

use vernier_core::{BoundaryGtCache, CocoDataset, EvalDataset, SegmGtCache};

/// Parsed-once COCO ground-truth dataset.
///
/// Construct with [`PyDataset::from_json`]; pass to the
/// `evaluate_*_summary_with_dataset` family. Reusing the same instance
/// across `evaluate` calls reuses the GT-side derivations the cached
/// kernels populate on first use (per ADR-0020). The handle is frozen
/// — its identity *is* the cache key.
#[pyclass(module = "vernier._core", name = "Dataset", frozen)]
pub(crate) struct PyDataset {
    inner: Arc<CocoDataset>,
    boundary_cache: Arc<BoundaryGtCache>,
    segm_cache: Arc<SegmGtCache>,
}

/// `Send`-able snapshot of a [`PyDataset`]'s contents — three `Arc`
/// clones bundled for the per-call hand-off to `py.detach`. Adding a
/// future cache slot (OKS visibility, segm GT counts) means one extra
/// field here, not another `run_pipeline_*` argument.
pub(crate) struct DatasetSnapshot {
    pub(crate) gt: Arc<CocoDataset>,
    pub(crate) boundary_cache: Arc<BoundaryGtCache>,
    pub(crate) segm_cache: Arc<SegmGtCache>,
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
}

#[pymethods]
impl PyDataset {
    /// Parses a COCO ground-truth JSON payload into a reusable
    /// [`Dataset`] handle. Raises `ValueError` on malformed JSON.
    #[staticmethod]
    fn from_json(gt_json: &Bound<'_, PyBytes>) -> PyResult<Self> {
        let gt = CocoDataset::from_json_bytes(gt_json.as_bytes())
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;
        Ok(Self {
            inner: Arc::new(gt),
            boundary_cache: Arc::new(BoundaryGtCache::new()),
            segm_cache: Arc::new(SegmGtCache::new()),
        })
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

    fn __repr__(&self) -> String {
        format!(
            "Dataset(images={}, annotations={}, categories={})",
            self.inner.images().len(),
            self.inner.annotations().len(),
            self.inner.categories().len(),
        )
    }
}
