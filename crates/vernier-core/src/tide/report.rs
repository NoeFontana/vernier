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

/// Resolved TIDE configuration for one call.
///
/// Per ADR-0022, the `(t_f, t_b)` thresholds are per-kernel and the
/// resolved values land here so every report self-describes. The
/// kernel marker (a string identifier — e.g. `"bbox"`, `"segm"`,
/// `"boundary"`) lets downstream tooling group reports across kernels
/// without reaching into the kernel type itself.
#[derive(Debug, Clone, PartialEq)]
pub struct TideConfig {
    /// Foreground / match threshold: at-or-above is a TP, unmatched
    /// detections are FP candidates. Defaults to `0.5` everywhere per
    /// ADR-0022.
    pub t_f: f64,
    /// Background threshold: IoU `< t_b` against every GT means
    /// "background". Per-kernel default per ADR-0022.
    pub t_b: f64,
    /// Identifier of the kernel this config was resolved for — e.g.
    /// `"bbox"`, `"segm"`, `"boundary"`. Free-form by design; the
    /// rewrite-layer entry point that produces this report names its
    /// own kernel.
    pub kernel: String,
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
/// (the paper's "perfect rejection" upper bound — what mAP would be
/// if every FP were correctly rejected) is recorded separately so the
/// caller can sanity-check `sum(delta_per_bin) <= delta_all_fp`.
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
    /// Sanity total: ΔmAP from the all-FPs-removed pass. Useful as
    /// the paper's "perfect rejection" upper bound; the per-bin
    /// deltas should sum to at most this value.
    pub delta_all_fp: f64,
    /// Resolved configuration this report was produced under (per
    /// ADR-0022).
    pub config: TideConfig,
}
