//! Python binding for [`vernier_core::breakdown::Breakdown`] and
//! [`vernier_core::breakdown::ClassGroupBreakdown`].
//!
//! ADR-0039 lifts the Rust-only breakdown machinery to the Python
//! surface. The wrapper unifies two flavors under one Python type:
//!
//! - **Range** (PR #217 / ADR-0040) — the `from_ranges(...)` factory
//!   for f64-keyed area-style buckets. First consumer is
//!   `vernier.instance.Evaluator.area_ranges`.
//! - **Class groups** (ADR-0041 / ADR-0042) — the
//!   `from_class_groups(...)` factory for class-id partitions. First
//!   consumers are `vernier.semantic.Evaluator.class_grouping` and
//!   `vernier.panoptic.Evaluator.class_grouping`.
//!
//! The two flavors are kept structurally distinct on the Rust side —
//! `Breakdown` (range) and `ClassGroupBreakdown` (class groups) are
//! sibling structs in `vernier-core` — but funnel through one Python
//! type so callers see a single `Breakdown` import. Variant
//! discrimination is exposed via the `kind` property.
//!
//! Construction validates inputs and raises `ValueError` on degenerate
//! shape (empty buckets, NaN / infinite bounds, `lo > hi`, duplicate
//! labels, partition violations). The closed-on-both-ends inclusion
//! semantics from ADR-0016 (quirk D6) carry over verbatim for ranges;
//! class groups enforce strict partition (no class id in two groups).

use pyo3::exceptions::{PyAttributeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyType;
use std::collections::HashSet;

use vernier_core::breakdown::{Breakdown, Bucket, ClassGroup, ClassGroupBreakdown};

/// Internal storage variant for [`PyBreakdown`].
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum BreakdownInner {
    /// Range-keyed (f64 buckets) — instance area ranges.
    Range(Breakdown),
    /// Class-id-keyed (named groups partitioning class ids) —
    /// semantic / panoptic class groupings.
    ClassGroups(ClassGroupBreakdown),
}

/// Python wrapper around [`Breakdown`] / [`ClassGroupBreakdown`].
#[pyclass(module = "vernier._core", name = "Breakdown", frozen, eq)]
#[pyo3(skip_from_py_object)]
#[derive(Debug, Clone, PartialEq)]
pub struct PyBreakdown {
    pub(crate) inner: BreakdownInner,
}

impl PyBreakdown {
    /// Construct from a Rust range `Breakdown`. Used by sentinel-
    /// resolution paths that materialize the kernel-canonical layout.
    pub fn from_inner(inner: Breakdown) -> Self {
        Self {
            inner: BreakdownInner::Range(inner),
        }
    }

    /// Construct from a Rust `ClassGroupBreakdown`.
    pub fn from_inner_class_groups(inner: ClassGroupBreakdown) -> Self {
        Self {
            inner: BreakdownInner::ClassGroups(inner),
        }
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
            inner: BreakdownInner::Range(Breakdown::new(axis, converted)),
        })
    }

    /// Construct from class-id-keyed groups.
    ///
    /// `groups` is a sequence of `(label, class_ids)` pairs, one per
    /// group. Group order on input determines the group axis index
    /// (first pair is index 0). Strict partition discipline is enforced
    /// — no class id may appear in two groups.
    ///
    /// Raises `ValueError` on:
    ///
    /// - empty `groups`;
    /// - any group with empty `class_ids`;
    /// - duplicate group labels;
    /// - the same class id appearing in more than one group.
    #[classmethod]
    fn from_class_groups(
        _cls: &Bound<'_, PyType>,
        axis: String,
        groups: Vec<(String, Vec<u32>)>,
    ) -> PyResult<Self> {
        if groups.is_empty() {
            return Err(PyValueError::new_err(
                "Breakdown.from_class_groups: groups must be non-empty",
            ));
        }
        let mut seen_labels: HashSet<String> = HashSet::with_capacity(groups.len());
        let mut seen_ids: HashSet<u32> = HashSet::new();
        let mut converted: Vec<ClassGroup> = Vec::with_capacity(groups.len());
        for (idx, (label, class_ids)) in groups.into_iter().enumerate() {
            if class_ids.is_empty() {
                return Err(PyValueError::new_err(format!(
                    "Breakdown.from_class_groups: group[{idx}] {label:?} has empty class_ids"
                )));
            }
            if !seen_labels.insert(label.clone()) {
                return Err(PyValueError::new_err(format!(
                    "Breakdown.from_class_groups: duplicate group label {label:?}"
                )));
            }
            for &cid in &class_ids {
                if !seen_ids.insert(cid) {
                    return Err(PyValueError::new_err(format!(
                        "Breakdown.from_class_groups: class id {cid} appears in multiple \
                         groups (partition discipline)"
                    )));
                }
            }
            converted.push(ClassGroup::new(idx, label, class_ids));
        }
        Ok(Self {
            inner: BreakdownInner::ClassGroups(ClassGroupBreakdown::new(axis, converted)),
        })
    }

    /// Axis name (e.g., `"area"`, `"vehicle_taxonomy"`).
    #[getter]
    fn axis(&self) -> String {
        match &self.inner {
            BreakdownInner::Range(b) => b.axis().to_string(),
            BreakdownInner::ClassGroups(g) => g.axis().to_string(),
        }
    }

    /// Variant discriminator: `"range"` for `from_ranges`-constructed
    /// breakdowns, `"class_groups"` for `from_class_groups`-constructed
    /// ones. Use this to dispatch in validators that accept a
    /// `Breakdown` of a specific shape.
    #[getter]
    fn kind(&self) -> &'static str {
        match &self.inner {
            BreakdownInner::Range(_) => "range",
            BreakdownInner::ClassGroups(_) => "class_groups",
        }
    }

    /// Range buckets as a list of `(label, lo, hi)` triples in
    /// construction order.
    ///
    /// Raises `AttributeError` if this Breakdown was built via
    /// `from_class_groups`. Use `class_groups` instead.
    #[getter]
    fn buckets(&self) -> PyResult<Vec<(String, f64, f64)>> {
        match &self.inner {
            BreakdownInner::Range(b) => Ok(b
                .buckets()
                .iter()
                .map(|b| (b.label.to_string(), b.lo, b.hi))
                .collect()),
            BreakdownInner::ClassGroups(_) => Err(PyAttributeError::new_err(
                "Breakdown.buckets is only available on range breakdowns; \
                 this breakdown was built via from_class_groups — use \
                 .class_groups instead",
            )),
        }
    }

    /// Class-id groups as a list of `(label, class_ids)` pairs in
    /// construction order.
    ///
    /// Raises `AttributeError` if this Breakdown was built via
    /// `from_ranges`. Use `buckets` instead.
    #[getter]
    fn class_groups(&self) -> PyResult<Vec<(String, Vec<u32>)>> {
        match &self.inner {
            BreakdownInner::Range(_) => Err(PyAttributeError::new_err(
                "Breakdown.class_groups is only available on class-group \
                 breakdowns; this breakdown was built via from_ranges — \
                 use .buckets instead",
            )),
            BreakdownInner::ClassGroups(g) => Ok(g
                .groups()
                .iter()
                .map(|g| (g.label.to_string(), g.class_ids().to_vec()))
                .collect()),
        }
    }

    fn __len__(&self) -> usize {
        match &self.inner {
            BreakdownInner::Range(b) => b.len(),
            BreakdownInner::ClassGroups(g) => g.len(),
        }
    }

    fn __repr__(&self) -> String {
        let (kind, axis, n) = match &self.inner {
            BreakdownInner::Range(b) => ("range", b.axis(), b.len()),
            BreakdownInner::ClassGroups(g) => ("class_groups", g.axis(), g.len()),
        };
        format!("Breakdown(kind={kind:?}, axis={axis:?}, len={n})")
    }
}
