//! Python binding for [`vernier_core::breakdown::Breakdown`].
//!
//! ADR-0039 lifts the Rust-only `Breakdown` to the Python surface. The
//! first consumer is `vernier.instance.Evaluator.area_ranges` (ADR-0040);
//! semantic and panoptic class grouping (ADRs 0041 / 0042) will add a
//! parallel `from_class_groups(...)` factory in their respective phases
//! when class-id-keyed storage lands on the Rust side.
//!
//! The PyO3 surface is intentionally lean: a `from_ranges(...)`
//! classmethod for f64-keyed area-style breakdowns, plus read-only
//! getters mirroring the Rust accessors. Construction validates inputs
//! and raises `ValueError` on degenerate shape (empty buckets, NaN /
//! infinite bounds, `lo > hi`, duplicate labels). The closed-on-both-ends
//! inclusion semantics from ADR-0016 (quirk D6) carry over verbatim —
//! the Python type does not expose a `Range`-style alternative.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;
use std::collections::HashSet;

use vernier_core::breakdown::{Breakdown, Bucket};

/// Python wrapper around [`vernier_core::breakdown::Breakdown`].
#[pyclass(module = "vernier._core", name = "Breakdown", frozen, eq)]
#[pyo3(skip_from_py_object)]
#[derive(Debug, Clone, PartialEq)]
pub struct PyBreakdown {
    pub(crate) inner: Breakdown,
}

impl PyBreakdown {
    /// Construct from a Rust `Breakdown`. Used by sentinel-resolution
    /// paths that materialize the kernel-canonical layout.
    pub fn from_inner(inner: Breakdown) -> Self {
        Self { inner }
    }

    /// Borrow the underlying Rust `Breakdown`.
    pub fn inner(&self) -> &Breakdown {
        &self.inner
    }
}

#[pymethods]
impl PyBreakdown {
    /// Construct from f64-keyed buckets.
    ///
    /// `buckets` is a sequence of `(label, lo, hi)` triples, one per
    /// bucket. `[lo, hi]` is closed on both ends per ADR-0016 (quirk
    /// D6); an annotation whose key sits exactly on a boundary lands in
    /// both adjacent buckets.
    ///
    /// Raises `ValueError` on:
    ///
    /// - empty `buckets`;
    /// - NaN or infinite `lo` / `hi`;
    /// - `lo < 0`;
    /// - `lo > hi`;
    /// - duplicate bucket labels.
    #[classmethod]
    fn from_ranges(
        _cls: &Bound<'_, PyType>,
        axis: String,
        buckets: Vec<(String, f64, f64)>,
    ) -> PyResult<Self> {
        if buckets.is_empty() {
            return Err(PyValueError::new_err(
                "Breakdown.from_ranges: buckets must be non-empty",
            ));
        }
        let mut seen_labels: HashSet<String> = HashSet::with_capacity(buckets.len());
        let mut converted: Vec<Bucket> = Vec::with_capacity(buckets.len());
        for (idx, (label, lo, hi)) in buckets.into_iter().enumerate() {
            if !lo.is_finite() {
                return Err(PyValueError::new_err(format!(
                    "Breakdown.from_ranges: bucket[{idx}] lo must be finite, got {lo}"
                )));
            }
            if !hi.is_finite() {
                return Err(PyValueError::new_err(format!(
                    "Breakdown.from_ranges: bucket[{idx}] hi must be finite, got {hi}"
                )));
            }
            if lo < 0.0 {
                return Err(PyValueError::new_err(format!(
                    "Breakdown.from_ranges: bucket[{idx}] lo must be >= 0, got {lo}"
                )));
            }
            if lo > hi {
                return Err(PyValueError::new_err(format!(
                    "Breakdown.from_ranges: bucket[{idx}] requires lo <= hi, got lo={lo}, hi={hi}"
                )));
            }
            if !seen_labels.insert(label.clone()) {
                return Err(PyValueError::new_err(format!(
                    "Breakdown.from_ranges: duplicate bucket label {label:?}"
                )));
            }
            converted.push(Bucket::new(idx, label, lo, hi));
        }
        Ok(Self {
            inner: Breakdown::new(axis, converted),
        })
    }

    /// Axis name (e.g., `"area"`).
    #[getter]
    fn axis(&self) -> String {
        self.inner.axis().to_string()
    }

    /// Buckets as a list of `(label, lo, hi)` triples in construction
    /// order.
    #[getter]
    fn buckets(&self) -> Vec<(String, f64, f64)> {
        self.inner
            .buckets()
            .iter()
            .map(|b| (b.label.to_string(), b.lo, b.hi))
            .collect()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        let n = self.inner.len();
        let axis = self.inner.axis();
        format!("Breakdown(axis={axis:?}, len={n})")
    }
}
