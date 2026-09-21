//! Oracle constants, pinned with their provenance.
//!
//! Every literal here is a fact about somebody else's code, not a
//! tunable of ours. Each one carries the upstream file and the symbol it
//! comes from, and the tests at the bottom of this module read
//! `tests/python/parity_obb/oracle/VENDORING.md` at compile time and
//! assert the two agree. Editing one without the other is a build
//! failure — which is the point: a parity constant that can drift
//! silently from its record is not pinned, it is merely written down.
//!
//! See ADR-0063 §"Oracles" for why the `D2` claim is keyed to a
//! `(commit, platform)` pair and not to the commit alone.

/// Verbatim copy of `VENDORING.md`, pulled in so the tripwires below
/// check the real file rather than a second transcription of it.
#[cfg(test)]
const VENDORING_MD: &str = include_str!("../../../tests/python/parity_obb/oracle/VENDORING.md");

// ---------------------------------------------------------------------
// D2 — detectron2
// ---------------------------------------------------------------------

/// Pinned detectron2 commit for the `RotatedBox` strict claim.
pub const D2_COMMIT_SHA: &str = "a25898a09d6ee232767647e92c6177fb1c642369";

/// SHA-256 of the vendored `box_iou_rotated_utils.h`.
pub const D2_UTILS_SHA256: &str =
    "8540cf5a4652ce8c1b3e9f98aa277c0920c7778f4356545b73df70547b684a0b";

/// SHA-256 of the vendored `rotated_coco_evaluation.py`.
pub const D2_EVAL_SHA256: &str = "55816d89c6958b45bea5ecbf6d6e5e007ca5db84e07a7dc1e4a50762e3b1e691";

/// detectron2's degrees-to-radians constant, **verbatim**.
///
/// `box_iou_rotated_utils.h::get_rotated_vertices` writes
/// `double theta = box.a * 0.01745329251;` with the literal spelled out
/// and a comment claiming it is `M_PI / 180.`. It is not: the true value
/// is `0.017453292519943295`, so the literal is short by about
/// `9.94e-12`, a relative error near `5.7e-10`. At `theta = 90` that
/// misses a right angle by roughly `5.1e-8` degrees — invisible in the
/// f32 result, but load-bearing for bit-exactness, which is why the
/// replica must use this value and never
/// [`crate::convention::DEG_TO_RAD`] (quirk **OB4**).
pub const D2_DEG_TO_RAD: f64 = 0.017_453_292_51;

/// `get_intersection_points`'s relaxation, `double EPS = 1e-5`.
///
/// Applied to the segment parameters `t1`, `t2` and to the
/// vertex-in-rectangle projections. Note the type: `EPS` is a `double`
/// while the quantities compared against it are `float`, so every one of
/// those comparisons promotes to f64 (quirk **OB6**).
pub const D2_EPS: f64 = 1e-5;

/// Parallel-line guard in `get_intersection_points`:
/// `if (fabs(det) <= 1e-14) continue;`.
pub const D2_DET_EPS: f64 = 1e-14;

/// Angular-tie tolerance in the hull sort: `crossProduct < -1e-6` and
/// `fabs(crossProduct) < 1e-6`.
pub const D2_CROSS_EPS: f64 = 1e-6;

/// Coincident-point tolerance in hull step 4: `dist[k] > 1e-8`.
pub const D2_DIST_EPS: f64 = 1e-8;

/// Degenerate-box guard in `single_box_iou_rotated`:
/// `if (area1 < 1e-14 || area2 < 1e-14) return 0.f;`.
pub const D2_AREA_EPS: f64 = 1e-14;

/// Intersection-point buffer size, `Point<T> intersectPts[24]`.
///
/// The bound is `4 * 4 + 4 + 4 = 24`: every edge pair can contribute one
/// point, and each rectangle can contribute all four of its own
/// vertices.
pub const D2_MAX_POINTS: usize = 24;

// ---------------------------------------------------------------------
// DK — DOTA_devkit
// ---------------------------------------------------------------------

/// Pinned DOTA_devkit commit for the `Quad` strict claim.
pub const DK_COMMIT_SHA: &str = "d3f8da45d4091b1dab37d9fbe4d6e6a50928e410";

/// SHA-256 of upstream `polyiou.cpp` at [`DK_COMMIT_SHA`].
///
/// The file itself is **not** vendored: DOTA_devkit states no license
/// anywhere in the repository, so there is no right to redistribute it.
/// The hash is what makes the replica's description of the algorithm
/// checkable against the real thing by anyone who fetches it.
pub const DK_POLYIOU_SHA256: &str =
    "ffbe0459419f962ce1695cd4c49beacb97b95ca42381f244da91f5b56dcb301a";

/// SHA-256 of upstream `dota_evaluation_task1.py` at [`DK_COMMIT_SHA`] —
/// the script that supplies the horizontal-box prefilter.
pub const DK_TASK1_SHA256: &str =
    "c334f2986ba83e368f13f1e36bc93cba88f728ac24e61049ac8a3545c59e346b";

/// `polyiou.cpp`'s global tolerance, `const double eps = 1E-8`.
///
/// Used only through `sig(d) = (d > eps) - (d < -eps)`, which is an
/// *absolute* sign test on a cross product — so its effective angular
/// resolution scales with the square of the coordinate magnitude. At
/// DOTA-scale coordinates that makes it effectively exact-zero; near the
/// origin it is a real tolerance. Reproduced as-is (quirk **OB6**).
pub const DK_EPS: f64 = 1e-8;

/// `polyiou.cpp`'s `#define maxn 51`, the polygon scratch size.
pub const DK_MAXN: usize = 51;

#[cfg(test)]
mod tests {
    use super::*;

    /// Assert `needle` appears in the vendoring record.
    fn recorded(needle: &str) -> bool {
        VENDORING_MD.contains(needle)
    }

    #[test]
    fn d2_provenance_matches_vendoring_md() {
        assert!(recorded(D2_COMMIT_SHA), "D2_COMMIT_SHA not in VENDORING.md");
        assert!(recorded(D2_UTILS_SHA256), "D2_UTILS_SHA256 not recorded");
        assert!(recorded(D2_EVAL_SHA256), "D2_EVAL_SHA256 not recorded");
    }

    #[test]
    fn dk_provenance_matches_vendoring_md() {
        assert!(recorded(DK_COMMIT_SHA), "DK_COMMIT_SHA not in VENDORING.md");
        assert!(
            recorded(DK_POLYIOU_SHA256),
            "DK_POLYIOU_SHA256 not recorded"
        );
        assert!(recorded(DK_TASK1_SHA256), "DK_TASK1_SHA256 not recorded");
    }

    /// The DK oracle must never be redistributed from this repository.
    ///
    /// The upstream carries no license, so the record has to say so in
    /// as many words. If someone later vendors the files and rewrites
    /// this paragraph, this test fails and the ADR-level conversation
    /// happens before the bytes land.
    #[test]
    fn dk_is_recorded_as_unlicensed() {
        assert!(
            recorded("DOTA_devkit carries no license"),
            "VENDORING.md must state the DK licensing position"
        );
    }

    #[test]
    fn d2_deg_to_rad_is_the_truncated_literal_not_pi_over_180() {
        let honest = core::f64::consts::PI / 180.0;
        assert_ne!(
            D2_DEG_TO_RAD, honest,
            "the whole point of this constant is that it is *not* pi/180"
        );
        // Short by ~9.94e-12: a 12-significant-digit truncation.
        let delta = honest - D2_DEG_TO_RAD;
        assert!(delta > 9.9e-12 && delta < 1.0e-11, "delta = {delta}");
    }

    #[test]
    fn d2_tolerances_are_the_upstream_literals() {
        assert_eq!(D2_EPS, 1e-5);
        assert_eq!(D2_DET_EPS, 1e-14);
        assert_eq!(D2_CROSS_EPS, 1e-6);
        assert_eq!(D2_DIST_EPS, 1e-8);
        assert_eq!(D2_AREA_EPS, 1e-14);
        assert_eq!(D2_MAX_POINTS, 24);
    }

    #[test]
    fn dk_tolerances_are_the_upstream_literals() {
        assert_eq!(DK_EPS, 1e-8);
        assert_eq!(DK_MAXN, 51);
    }
}
