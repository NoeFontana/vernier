//! User-facing LRP parameters.
//!
//! [`LrpParams`] is the inputs bundle for the per-kernel
//! `optimal_lrp_*` entry points in [`super`]. Mirrors the borrow-shape
//! of [`crate::EvaluateParams`]: the tau-grid slice lives on the caller
//! and is borrowed for the duration of the call so the same canonical
//! `0.01`-step grid can be reused across many invocations without
//! reallocating.
//!
//! Defaults are not provided through `Default::default()`: per
//! ADR-0044, `tp_threshold` and `tau_grid` are kernel-dependent and
//! resolved by [`super::defaults::defaults_for`] rather than by a
//! struct-level zero.

use crate::evaluate::AreaRange;

/// Configuration for one LRP error-decomposition call.
///
/// Per ADR-0044 the thresholds are per-kernel; the defaults the paper
/// picks are `tp_threshold = 0.5` for every kernel, and the tau grid
/// is `0.01` step over `[0.0, 1.0]` (101 points inclusive of both
/// endpoints).
#[derive(Debug, Clone, Copy)]
pub struct LrpParams<'a> {
    /// TP IoU threshold — matched (DT, GT) pairs with IoU >= this
    /// value are TPs. The Oksuz TPAMI 2021 paper's recommended
    /// operating point and the same `0.5` reading AP@0.5 carries.
    pub tp_threshold: f64,

    /// Confidence-threshold grid scanned for the argmin. Per ADR-0044
    /// the canonical default is the 101-point grid `0.00, 0.01, ...,
    /// 1.00`. Caller supplies the slice so the same canonical table is
    /// reused across many invocations without reallocating.
    pub tau_grid: &'a [f64],

    /// Per-image cap on detections kept at matching time (score-desc,
    /// stable mergesort tie-break per quirk **A1**). Mirrors
    /// [`crate::EvaluateParams::max_dets_per_image`].
    pub max_dets_per_image: usize,

    /// Quirk **L4**: when `false`, every category is collapsed onto a
    /// single bucket. Mirrors
    /// [`crate::EvaluateParams::use_cats`].
    pub use_cats: bool,

    /// IoU thresholds for the *upstream* evaluator. LRP only consumes
    /// the retained per-cell IoU matrices and the per-cell metadata,
    /// not the matching-engine outputs at any specific threshold —
    /// the tau search is over confidence, not IoU. We still need to
    /// pass an iou-thresholds slice to the evaluator entry point;
    /// `[tp_threshold]` is the minimal correct choice.
    pub iou_thresholds: &'a [f64],

    /// Area ranges threaded through to the upstream evaluator. LRP is
    /// area-agnostic (the paper reports per-class numbers, not
    /// per-area-bucket), so only the `all` bucket is consumed
    /// downstream; the rest are inert but the evaluator pipeline
    /// expects a populated slice. Use [`AreaRange::coco_default`].
    pub area_ranges: &'a [AreaRange],
}
