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

use std::sync::OnceLock;

/// Three-tier parity mode (per ADR-0002).
///
/// Picks which disposition vernier honors for each row of the
/// pycocotools quirks survey. `Strict` is the canonical migration path
/// for users with downstream tooling calibrated to pycocotools' exact
/// numerical behavior; `Corrected` is the default for net-new users.
///
/// `Aligned` does not appear here because aligned-tier changes are, by
/// definition, output-equivalent to strict — they ship in every mode.
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
/// arithmetic. (Quirk **C8** — aligned.) On every supported platform
/// `f64::EPSILON == 2.220446049250313e-16`, identical to `np.spacing(1)`
/// to all bits.
pub const PARITY_EPS: f64 = f64::EPSILON;

/// Tolerance fudge in the matching loop's initial best-IoU seed. Used as
/// `min(threshold, 1.0 - IOU_BOUNDARY_EPS)` so a detection with IoU
/// exactly at the threshold still matches. (Quirk **B1** — strict.)
pub const IOU_BOUNDARY_EPS: f64 = 1e-10;

/// Divide-by-zero guard added to OKS area before the per-keypoint
/// distance is normalized by it. (Quirk **F2** — aligned.)
pub const OKS_AREA_EPS: f64 = f64::EPSILON;

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

/// Reproduces `numpy.linspace(start, stop, num, endpoint=True)`.
///
/// Numpy's algorithm computes `step = (stop - start) / (num - 1)` once
/// (in f64) and emits `start + i * step` for `i in 0..num`, with the
/// final element snapped to `stop` exactly. This implementation follows
/// the same shape, so the resulting array is bit-equal to numpy's
/// across the platforms we target.
fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
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
}
