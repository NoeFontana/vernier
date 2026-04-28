//! Pinned numerical constants for the boundary-IoU subsystem.
//!
//! Parallel to [`crate::parity`]: same role (single home for parity-
//! contract knobs), different oracle. The pycocotools and
//! boundary-IoU lifecycles are intentionally decoupled per ADR-0010,
//! so their constants live in separate modules even though both share
//! the global [`crate::parity::ParityMode`] enum.
//!
//! Each item is doc-tagged with the quirk ID from
//! `docs/engineering/boundary-iou-quirks.md` it corresponds to.
//! Changes here are an ADR-level decision (ADR-0010 §"Performance
//! baseline" lists the perf-budget constants; behavioral constants
//! follow the same governance).

/// Default Chebyshev-ball dilation ratio for boundary IoU. Pinned at
/// `0.02` per Cheng et al. 2021 and quirk **M4** of the boundary-IoU
/// quirks survey. The LVIS variant exposes `0.008`; vernier surfaces
/// the value as a public field on `BoundaryIou` rather than hardcoding
/// it at the call site (M4 disposition `corrected` for the API).
pub const BOUNDARY_DILATION_RATIO_DEFAULT: f64 = 0.02;

/// IoU-equality tolerance applied under [`crate::parity::ParityMode::Aligned`]
/// when comparing vernier's boundary-IoU output against the vendored
/// oracle. Stricter checks live in `Strict` mode (bit-equal). The
/// `1e-9` value is the same magnitude pycocotools' `np.testing` defaults
/// use for IoU comparisons; pinned here so the boundary parity harness
/// has a single knob to tune as the oracle stabilises.
pub const BOUNDARY_PARITY_EPS: f64 = 1e-9;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dilation_ratio_default_is_cheng_2021_value() {
        // Cheng, Girshick, Dollár, Berg, Kirillov (CVPR 2021): the
        // paper recommends `dilation_ratio = 0.02`. The bowenc0221
        // reference hardcodes the same value at `boundary_utils.py:9`.
        assert_eq!(BOUNDARY_DILATION_RATIO_DEFAULT, 0.02);
    }

    #[test]
    fn parity_eps_is_1e_neg_9() {
        assert_eq!(BOUNDARY_PARITY_EPS, 1e-9);
    }
}
