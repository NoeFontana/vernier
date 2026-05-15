//! Parity-mode flag and pinned numerical constants.
//!
//! This module is the single home for every numerical constant and
//! algorithmic primitive that vernier needs to reproduce bit-exactly to
//! match `pycocotools` 2.0.11. Each item is doc-tagged with the quirk ID
//! from `docs/engineering/pycocotools-quirks.md` it corresponds to, and
//! with the ADR that ratifies its choice.
//!
//! The whole file is load-bearing for the parity contract from ADR-0002.
//! Changes here ripple through every algorithm crate and require their
//! own ADR.

use std::cmp::Ordering;
use std::sync::OnceLock;

/// Parity mode (per ADR-0002, amended 2026-05-10).
///
/// Picks which disposition vernier honors for each row of the
/// pycocotools quirks survey. `Strict` is the canonical migration path
/// for users with downstream tooling calibrated to pycocotools' exact
/// numerical behavior; `Corrected` is the default for net-new users.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParityMode {
    /// Reproduce every pycocotools behavior bit-exactly, including known
    /// bugs (D1's overwritten ignore field, H2's silent merge, …). The
    /// default when migrating from pycocotools.
    Strict,
    /// Apply opinionated fixes for behaviors classified `corrected` in
    /// the disposition table. Default for net-new users; opt-out via
    /// `Strict`.
    #[default]
    Corrected,
}

/// Substitute for `numpy.spacing(1)` in pycocotools' precision/recall
/// arithmetic. (Quirk **C8** — strict.) On every supported platform
/// `f64::EPSILON == 2.220446049250313e-16`, identical to `np.spacing(1)`
/// to all bits.
pub const PARITY_EPS: f64 = f64::EPSILON;

/// Tolerance fudge in the matching loop's initial best-IoU seed. Used as
/// `min(threshold, 1.0 - IOU_BOUNDARY_EPS)` so a detection with IoU
/// exactly at the threshold still matches. (Quirk **B1** — strict.)
pub const IOU_BOUNDARY_EPS: f64 = 1e-10;

/// The 10 IoU thresholds at which COCO eval reports AP. Built via the
/// same `linspace(0.5, 0.95, 10)` formula pycocotools uses (not
/// `arange`, which accumulates float error). (Quirk **L1** — strict.)
pub fn iou_thresholds() -> &'static [f64] {
    static IOU_THRESHOLDS: OnceLock<Vec<f64>> = OnceLock::new();
    IOU_THRESHOLDS.get_or_init(|| linspace(0.5, 0.95, 10))
}

/// The 101 recall thresholds used for AP integration. Built via
/// `linspace(0.0, 1.0, 101)`. (Quirk **L2**, **C1** — strict.)
pub fn recall_thresholds() -> &'static [f64] {
    static RECALL_THRESHOLDS: OnceLock<Vec<f64>> = OnceLock::new();
    RECALL_THRESHOLDS.get_or_init(|| linspace(0.0, 1.0, 101))
}

/// Quirk **P1** — strict. numpy quantile method `'linear'` for
/// calibration bin-edges (ADR-0018; see
/// `docs/engineering/calibration-quirks.md`).
///
/// Pinned as a string for documentation parity with numpy's `method=`
/// kwarg; the actual computation is implemented in the crate-private
/// `quantile_linear`. Recorded here so any drift away from the pinned
/// method shows up as a constant-rename in code review.
pub const CALIBRATION_QUANTILE_METHOD: &str = "linear";

/// Linear-interpolation quantile, bit-equivalent to
/// `numpy.quantile(values, q, method='linear')` (Quirk **P1** — strict).
///
/// Used by the calibration summarizer (ADR-0018) to derive bin-edges
/// from the score distribution. The caller is responsible for sorting
/// `sorted_values` ascending; a debug-only assertion guards the
/// invariant.
///
/// # Algorithm
///
/// For each `q` in `qs`, compute `pos = q * (n - 1)`, then
/// `lo = floor(pos)`, `hi = ceil(pos)`, `frac = pos - lo`. The result
/// is `values[lo] + (values[hi] - values[lo]) * frac`. This matches
/// numpy's `'linear'` interpolation rule precisely.
///
/// # Edge cases
///
/// - Empty `sorted_values` returns an empty output (no quantiles to
///   interpolate). Callers must guard against this case explicitly if
///   it carries domain meaning.
/// - `qs` values outside `[0, 1]` are not validated; the caller is
///   expected to pass values produced by `linspace(0.0, 1.0, n+1)` or
///   similar.
pub(crate) fn quantile_linear(sorted_values: &[f64], qs: &[f64]) -> Vec<f64> {
    if sorted_values.is_empty() {
        return Vec::new();
    }
    debug_assert!(
        sorted_values.windows(2).all(|w| w[0] <= w[1]),
        "quantile_linear: sorted_values must be ascending"
    );
    let n = sorted_values.len();
    if n == 1 {
        let only = sorted_values[0];
        return qs.iter().map(|_| only).collect();
    }
    let last = n - 1;
    let mut out = Vec::with_capacity(qs.len());
    for &q in qs {
        let pos = q * (last as f64);
        let lo_idx = pos.floor() as usize;
        let hi_idx = pos.ceil() as usize;
        // Clamp defensively in case of floating drift; with q in [0,1]
        // the indices are already within bounds, but the clamp keeps
        // the function total without panicking on out-of-domain inputs.
        let lo_idx = lo_idx.min(last);
        let hi_idx = hi_idx.min(last);
        let frac = pos - (lo_idx as f64);
        let lo = sorted_values[lo_idx];
        let hi = sorted_values[hi_idx];
        out.push(lo + (hi - lo) * frac);
    }
    out
}

/// Stable score-descending argsort. (Quirk **A1** — strict.)
///
/// Mirrors `np.argsort(-scores, kind='mergesort')`: the returned
/// permutation indexes `scores` such that `scores[perm[0]] >=
/// scores[perm[1]] >= ...`, with ties resolved to the input order.
/// Used by the matching engine for per-image DT ordering and by the
/// accumulator for the merged-stream re-sort across images.
pub fn argsort_score_desc(scores: &[f64]) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..scores.len()).collect();
    perm.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(Ordering::Equal));
    perm
}

/// Reproduces `numpy.linspace(start, stop, num, endpoint=True)`.
///
/// Numpy's algorithm computes `step = (stop - start) / (num - 1)` once
/// (in f64) and emits `start + i * step` for `i in 0..num`, with the
/// final element snapped to `stop` exactly. This implementation follows
/// the same shape, so the resulting array is bit-equal to numpy's
/// across the platforms we target.
pub(crate) fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    if num == 0 {
        return Vec::new();
    }
    if num == 1 {
        return vec![start];
    }
    let last = num - 1;
    let step = (stop - start) / (last as f64);
    let mut out = Vec::with_capacity(num);
    for i in 0..last {
        out.push(start + (i as f64) * step);
    }
    out.push(stop);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parity_eps_matches_numpy_spacing_1() {
        // np.spacing(1) == 2.220446049250313e-16 on every platform we
        // support; equal to f64::EPSILON to all bits.
        assert_eq!(PARITY_EPS, 2.220446049250313e-16);
    }

    #[test]
    fn iou_boundary_eps_is_1e_neg_10() {
        // pycocotools/cocoeval.py:276 — `min(t, 1 - 1e-10)` seeds the
        // initial best-IoU so detections at exactly the threshold match.
        assert_eq!(IOU_BOUNDARY_EPS, 1e-10);
    }

    #[test]
    fn iou_thresholds_match_numpy_linspace() {
        // Ground truth captured from `numpy.linspace(0.5, 0.95, 10)`
        // (NumPy 2.0). Index 8 is `0.8999999999999999` — one ulp below
        // 0.9 — and is what every pycocotools install on every supported
        // platform actually receives. Pinning the bit pattern is the
        // entire point of quirk **L1**.
        let expected_bits: [u64; 10] = [
            4602678819172646912, // 0.5
            4603129179135383962, // 0.55
            4603579539098121011, // 0.6
            4604029899060858061, // 0.65
            4604480259023595110, // 0.7
            4604930618986332160, // 0.75
            4605380978949069210, // 0.8
            4605831338911806259, // 0.85
            4606281698874543308, // 0.8999999999999999
            4606732058837280358, // 0.95 (snapped)
        ];
        let got = iou_thresholds();
        assert_eq!(got.len(), expected_bits.len());
        for (i, (g, e)) in got.iter().zip(expected_bits.iter()).enumerate() {
            assert_eq!(
                g.to_bits(),
                *e,
                "iouThr[{i}] differs: got bits {} ({:e})",
                g.to_bits(),
                g
            );
        }
    }

    #[test]
    fn recall_thresholds_have_101_points_endpoints_pinned() {
        let r = recall_thresholds();
        assert_eq!(r.len(), 101);
        assert_eq!(r[0], 0.0);
        assert_eq!(r[100], 1.0);
        // Midpoint check — bit-equal to numpy's linspace.
        assert_eq!(r[50].to_bits(), 0.5_f64.to_bits());
    }

    #[test]
    fn linspace_handles_degenerate_sizes() {
        assert!(linspace(0.0, 1.0, 0).is_empty());
        assert_eq!(linspace(0.5, 0.5, 1), vec![0.5]);
    }

    #[test]
    fn parity_mode_default_is_corrected() {
        // ADR-0002: corrected is the default for net-new users; strict
        // is the migration mode.
        assert_eq!(ParityMode::default(), ParityMode::Corrected);
    }

    #[test]
    fn calibration_quantile_method_pinned_to_linear() {
        // ADR-0018 / Quirk P1: bin-edges use numpy method='linear'.
        assert_eq!(CALIBRATION_QUANTILE_METHOD, "linear");
    }

    #[test]
    fn quantile_linear_matches_numpy_on_arange() {
        // Reference values from numpy:
        //   import numpy as np
        //   v = np.arange(11, dtype=float)  # 0..10
        //   np.quantile(v, [0.0, 0.25, 0.5, 0.75, 1.0], method='linear')
        //   -> array([ 0. ,  2.5,  5. ,  7.5, 10. ])
        let v: Vec<f64> = (0..=10).map(|x| x as f64).collect();
        let got = quantile_linear(&v, &[0.0, 0.25, 0.5, 0.75, 1.0]);
        assert_eq!(got, vec![0.0, 2.5, 5.0, 7.5, 10.0]);
    }

    #[test]
    fn quantile_linear_interpolates_two_points() {
        // np.quantile([0.0, 1.0], [0.0, 0.5, 1.0], method='linear')
        //   -> array([0. , 0.5, 1. ])
        let got = quantile_linear(&[0.0, 1.0], &[0.0, 0.5, 1.0]);
        assert_eq!(got, vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn quantile_linear_single_element_replicates() {
        // Single-element input → that element for every q.
        let got = quantile_linear(&[0.42], &[0.0, 0.3, 1.0]);
        assert_eq!(got, vec![0.42, 0.42, 0.42]);
    }

    #[test]
    fn quantile_linear_empty_input_is_empty() {
        // Empty input is a no-op: the caller is expected to short-circuit
        // before constructing bin-edges.
        let got = quantile_linear(&[], &[0.0, 0.5, 1.0]);
        assert!(got.is_empty());
    }

    #[test]
    fn quantile_linear_endpoints_pinned() {
        // q=0 → first, q=1 → last, regardless of distribution shape.
        let v = [0.1, 0.2, 0.8, 0.9];
        let got = quantile_linear(&v, &[0.0, 1.0]);
        assert_eq!(got, vec![0.1, 0.9]);
    }
}
