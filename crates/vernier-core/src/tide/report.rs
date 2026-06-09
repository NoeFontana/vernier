//! TIDE report and configuration types.
//!
//! [`TideReport`] is the per-bin ΔmAP output the eight-pass
//! orchestration (Week 2) produces; [`TideConfig`] is the resolved
//! configuration the call ran under, recorded alongside so a screenshot
//! of a number can be re-derived from the report alone (per ADR-0022).
//!
//! These are storage types only — population happens in the rewrite
//! layer. Neither carries a `Default` impl: `TideReport` has no
//! meaningful zero (the per-bin deltas are call outputs, not initial
//! state), and `TideConfig`'s defaults are per-kernel and resolved by
//! a future `tide::defaults_for` helper, not by `Default::default()`.

use std::collections::HashMap;

use crate::tide::bins::TideErrorBin;

/// Closed set of kernels TIDE supports today (ADR-0021/0022).
///
/// Recorded on every [`TideConfig`] so reports self-describe their
/// kernel without leaking the concrete [`crate::similarity::Similarity`]
/// implementor type. The `as_str` projection gives the canonical
/// lowercase identifier the FFI surface and the numpy oracle both use
/// (`"bbox"` / `"segm"` / `"boundary"`); pin this representation so a
/// screenshot of `config.kernel` survives the round-trip through any
/// downstream tooling.
///
/// Keypoints (OKS) is intentionally absent per ADR-0024.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelMarker {
    /// Bounding-box IoU (ADR-0008).
    Bbox,
    /// Segmentation-mask IoU (ADR-0009).
    Segm,
    /// Boundary IoU (ADR-0010); pair the marker with the call's
    /// `dilation_ratio` if you need to cross-reference the kernel
    /// configuration end-to-end.
    Boundary,
}

impl KernelMarker {
    /// Canonical lowercase name; pinned so it stays bit-stable across
    /// the FFI dict surface and the oracle's `config.kernel` field.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bbox => "bbox",
            Self::Segm => "segm",
            Self::Boundary => "boundary",
        }
    }
}

/// Resolved TIDE configuration for one call.
///
/// Per ADR-0022, the `(t_f, t_b)` thresholds are per-kernel and the
/// resolved values land here so every report self-describes. The
/// [`KernelMarker`] tags the kernel a downstream consumer can group on
/// without reaching into the concrete `Similarity` implementor.
#[derive(Debug, Clone, PartialEq)]
pub struct TideConfig {
    /// Foreground / match threshold: at-or-above is a TP, unmatched
    /// detections are FP candidates. Defaults to `0.5` everywhere per
    /// ADR-0022.
    pub t_f: f64,
    /// Background threshold: IoU `< t_b` against every GT means
    /// "background". Per-kernel default per ADR-0022.
    pub t_b: f64,
    /// Kernel this config was resolved for. Closed set per
    /// ADR-0021/0024 (no keypoints).
    pub kernel: KernelMarker,
    /// Optional cap on per-detection cross-class IoU storage (per
    /// ADR-0023, the `cross_class_topk` knob). `None` = full
    /// materialize, the default.
    pub cross_class_topk: Option<usize>,
}

/// Output of a TIDE pass: per-bin ΔmAP plus the configuration the call
/// ran under.
///
/// `delta_per_bin` carries one entry per [`TideErrorBin`] populated by
/// the rewrite layer; absent bins (e.g. structurally-zero Cls/Both
/// bins on a single-class workload) are simply missing from the map
/// rather than recorded as `0.0`. The all-FPs-removed sanity total
/// (the paper's "perfect rejection" benchmark — what mAP would be if
/// every FP were correctly rejected) is recorded separately as a
/// reference number. Note: `sum(delta_per_bin) <= delta_all_fp` does
/// NOT hold in general — `cls`, `loc`, and `both` reclassify FPs into
/// TPs (lifting both precision AND recall, while `delta_all_fp` only
/// drops FPs), so they can individually or in sum exceed `delta_all_fp`
/// on real data. The only universal upper bound is
/// `baseline_map + delta_per_bin[k] <= 1.0` (AP is in [0, 1]).
///
/// No `Default` impl: a default-constructed `TideReport` would have
/// no meaningful semantics — every field is a call output, not
/// initial state.
#[derive(Debug, Clone)]
pub struct TideReport {
    /// Baseline mAP — the headline number before any bin-specific
    /// correction is applied. The per-bin deltas are subtractions
    /// against this.
    pub baseline_map: f64,
    /// Per-bin ΔmAP. A bin's value is the mAP increase the call would
    /// achieve if every detection assigned to that bin were corrected.
    /// Bins not populated by the rewrite layer (e.g. Cls/Both on a
    /// single-class workload, where they are structurally zero) are
    /// absent from the map.
    pub delta_per_bin: HashMap<TideErrorBin, f64>,
    /// Reference total: ΔmAP from the all-FPs-removed pass.
    ///
    /// Useful as the paper's "perfect rejection" benchmark. Note: this
    /// is NOT a tight upper bound on the per-bin deltas. `cls` / `loc`
    /// / `both` reclassify FPs into TPs (lifting both precision AND
    /// recall), while `all_fp` only drops FPs (precision only) — so
    /// the reclassifying bins can individually exceed `delta_all_fp`
    /// on real data. Empirically observed on DETR-R50:
    /// `delta[loc] ≈ 0.076` vs `delta_all_fp ≈ 0.053`. The only
    /// universal upper bound is `baseline_map + delta <= 1.0`.
    pub delta_all_fp: f64,
    /// Resolved configuration this report was produced under (per
    /// ADR-0022).
    pub config: TideConfig,
}
