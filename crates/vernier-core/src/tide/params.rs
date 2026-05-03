//! User-facing TIDE parameters.
//!
//! [`TideParams`] is the inputs bundle for the eight-pass orchestrator
//! [`super::error_decomposition_bbox`]. Mirrors the borrow-shape of
//! [`crate::EvaluateParams`]: the IoU / recall / area-range slices live
//! on the caller and are borrowed for the duration of the call so the
//! same canonical [`crate::iou_thresholds`] / [`crate::recall_thresholds`]
//! / [`crate::AreaRange::coco_default`] tables can be reused across many
//! TIDE invocations without reallocating.
//!
//! Defaults are not provided through `Default::default()`: per ADR-0022,
//! `(t_f, t_b)` are kernel-dependent and resolved by future
//! `tide::defaults_for(kernel)` rather than by a struct-level zero.

use crate::evaluate::AreaRange;

/// Configuration for one TIDE error-decomposition call.
///
/// Per ADR-0022 the thresholds are per-kernel; the bbox defaults the
/// paper picks are `t_f = 0.5`, `t_b = 0.1`. Other kernels resolve
/// their own values through a future `tide::defaults_for` helper.
#[derive(Debug, Clone, Copy)]
pub struct TideParams<'a> {
    /// Foreground / match threshold. Detections with same-class IoU
    /// `>= t_f` against an unmatched GT are TPs.
    pub t_f: f64,
    /// Background threshold. Detections with `max(iou_same, iou_cross)
    /// < t_b` are Bkg false positives.
    pub t_b: f64,
    /// Per-image cap on detections kept at matching time (score-desc,
    /// stable mergesort tie-break per quirk **A1**). Mirrors
    /// [`crate::EvaluateParams::max_dets_per_image`].
    pub max_dets_per_image: usize,
    /// Quirk **L4**: when `false`, every category is collapsed onto a
    /// single bucket and cross-class attribution is degenerate (Cls and
    /// Both bins are structurally empty).
    pub use_cats: bool,
    /// IoU thresholds for the AP ladder used by every accumulate pass.
    /// Use [`crate::iou_thresholds`] for the canonical 10-point COCO
    /// ladder.
    pub iou_thresholds: &'a [f64],
    /// Recall thresholds for AP integration. Use
    /// [`crate::recall_thresholds`] for the canonical 101-point grid.
    pub recall_thresholds: &'a [f64],
    /// Area ranges used by the accumulate passes. The orchestrator
    /// reads the `all` bucket only (TIDE binning is area-agnostic);
    /// the other ranges flow through so the underlying
    /// [`crate::evaluate_with`] call can populate the cells the
    /// summarizer expects. Use [`AreaRange::coco_default`] for the
    /// canonical four-bucket grid.
    pub area_ranges: &'a [AreaRange],
}
