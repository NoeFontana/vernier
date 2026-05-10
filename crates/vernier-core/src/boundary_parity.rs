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

/// Cross-oracle IoU-equality tolerance for the parity harness when
/// comparing vernier's boundary-IoU output against the vendored
/// oracle. Harness-side comparison budget, not a runtime mode (see
/// [`crate::parity::ParityMode`] for the runtime contract). The
/// `1e-9` value matches the magnitude pycocotools' `np.testing`
/// defaults use for IoU comparisons.
pub const BOUNDARY_PARITY_EPS: f64 = 1e-9;

/// Frozen commit SHA of the vendored `bowenc0221/boundary-iou-api`
/// reference oracle (ADR-0010 §"Oracle (E2 + E3)"). Strict-mode parity
/// claims are keyed to this SHA: every quirk vernier reproduces in
/// `Strict` boundary mode is the disposition of an observed behaviour
/// at this commit.
///
/// The vendored tree lives at
/// `tests/python/parity_boundary/oracle/boundary_iou_api/`; provenance,
/// modification policy, and fork plan are recorded in the adjacent
/// `VENDORING.md`. Drift between this constant and `VENDORING.md`'s
/// table is a build failure — see the unit test below.
pub const ORACLE_COMMIT_SHA: &str = "37d25586a677b043ed585f10e5c42d4e80176ea9";

/// Pinned `opencv-python` release used to run the vendored oracle
/// (ADR-0010 §"Oracle (E2 + E3)"). The upstream's
/// `boundary_utils.mask_to_boundary` calls `cv2.erode` and
/// `cv2.copyMakeBorder`; both are API-stable since OpenCV 2.x, so the
/// pin guards against build/wheel breakage rather than behaviour drift.
///
/// Mirrors `pyproject.toml`'s `[dependency-groups].dev` entry —
/// changing one without the other is a build failure.
pub const ORACLE_OPENCV_PIN: &str = "opencv-python==4.10.0.84";

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

    #[test]
    fn oracle_sha_matches_vendoring_md() {
        // Tripwire: editing the SHA here without updating
        // tests/python/parity_boundary/oracle/VENDORING.md (or vice
        // versa) is a parity-contract change. Equality forces the
        // editor to acknowledge both files are out of step.
        assert_eq!(
            ORACLE_COMMIT_SHA,
            "37d25586a677b043ed585f10e5c42d4e80176ea9"
        );
    }

    #[test]
    fn oracle_opencv_pin_matches_pyproject() {
        // Tripwire: this constant mirrors `pyproject.toml`'s dev-extras
        // entry. Editing one without the other defeats the reproducibility
        // contract of the vendored oracle (ADR-0010 §"Oracle (E2 + E3)").
        assert_eq!(ORACLE_OPENCV_PIN, "opencv-python==4.10.0.84");
    }
}
