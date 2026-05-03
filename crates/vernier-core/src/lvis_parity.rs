//! Pinned numerical constants for the LVIS evaluation subsystem.
//!
//! Parallel to [`crate::parity`] (pycocotools) and
//! [`crate::boundary_parity`] (boundary-IoU): same role (single home for
//! parity-contract knobs), different oracle. Each subsystem's lifecycle
//! is intentionally decoupled per ADR-0026 §"Parity strategy", so
//! constants live in separate modules even though all three share the
//! global [`crate::parity::ParityMode`] enum.
//!
//! Each item is doc-tagged with the quirk ID from the ADR-0026
//! appendix it corresponds to. Changes here are an ADR-level decision;
//! the canonical decision record is ADR-0026 itself.
//!
//! The vendored oracle lives at
//! `tests/python/parity_lvis/oracle/lvis_api/`; provenance, modification
//! policy, and fork plan are recorded in the adjacent `VENDORING.md`.
//! Drift between any constant here and `VENDORING.md` is a build
//! failure — see the unit tests below.

/// Default per-image `max_dets` for LVIS evaluation. Pinned at `300` per
/// the upstream `Params` constructor at `eval.py:528` and the
/// `LVISResults` constructor's `max_dets` argument default at
/// `results.py:10`. Quirk **AC1**.
///
/// vernier exposes this as a constant rather than an implicit
/// per-dataset sentinel: per ADR-0026 decision driver F2, users supply
/// `max_dets=(300,)` to the `Evaluator` explicitly when evaluating LVIS
/// data. See `docs/explanation/lvis-migration.md`.
pub const LVIS_DEFAULT_MAX_DETS: usize = 300;

/// Default Chebyshev-ball dilation ratio for boundary IoU on LVIS. The
/// upstream `lvis-api` does not implement boundary IoU (it raises at
/// `eval.py:25-26` for any `iou_type` other than `bbox`/`segm`); the
/// LVIS-specific value of `0.008` originates in the `bowenc0221` fork
/// and is captured by quirk **Q1** of the boundary-IoU survey
/// (boundary-iou-quirks.md). Quirk **AE3** of ADR-0026 disposes the
/// API surface as **corrected**: vernier surfaces the value as a public
/// field on `BoundaryIou` rather than hardcoding it at the call site,
/// so LVIS users pass `Boundary(dilation_ratio=0.008)` explicitly.
///
/// The value here mirrors the literal `0.008` in the migration guide
/// and tutorials; centralizing it lets a future LVIS challenge-year
/// variant change the ratio without scattering edits across the
/// codebase.
pub const LVIS_BOUNDARY_DILATION_RATIO_DEFAULT: f64 = 0.008;

/// IoU/AP-equality tolerance applied under
/// [`crate::parity::ParityMode::Aligned`] when comparing vernier's LVIS
/// output against the vendored oracle. Stricter checks live in `Strict`
/// mode (bit-equal). The `1e-9` value matches
/// [`crate::boundary_parity::BOUNDARY_PARITY_EPS`]; pinned here so the
/// LVIS parity harness has a single knob to tune as the oracle
/// stabilises.
pub const LVIS_PARITY_EPS: f64 = 1e-9;

/// Pinned PyPI release of the vendored `lvis-dataset/lvis-api`
/// reference oracle. Strict-mode parity claims for LVIS are keyed to
/// this version: every quirk vernier reproduces in `Strict` LVIS mode
/// is the disposition of an observed behaviour at this release.
///
/// The vendored tree lives at
/// `tests/python/parity_lvis/oracle/lvis_api/`; provenance,
/// modification policy, and fork plan are recorded in the adjacent
/// `VENDORING.md`. Drift between this constant and `VENDORING.md` is a
/// build failure — see the unit test below.
pub const ORACLE_LVIS_VERSION: &str = "0.5.3";

/// Frozen commit SHA of the vendored `lvis-dataset/lvis-api` reference
/// oracle, corresponding to the PyPI [`ORACLE_LVIS_VERSION`] release
/// point. The PyPI sdist for `lvis==0.5.3` ships no LICENSE; the
/// canonical BSD-2-Clause notice and the source tree are sourced from
/// the upstream GitHub repo at this SHA, where all six `lvis/*.py`
/// files are byte-equal to the sdist payload (see `VENDORING.md`
/// §"Provenance — byte-equality").
pub const ORACLE_LVIS_COMMIT_SHA: &str = "031ac21f939bcb5f1ca8de2ab8704082e101ff9b";

/// Pinned `pycocotools` release the vendored `lvis` oracle runs against.
/// `lvis==0.5.3` declares no `pycocotools` constraint in its package
/// metadata but imports `pycocotools.mask` at runtime (`lvis/lvis.py:15`)
/// for ground-truth mask decoding. vernier already pins
/// `pycocotools==2.0.11` (ADR-0002) for the pycocotools parity oracle;
/// because the LVIS oracle declares no constraint, the two co-exist
/// cleanly. Quirk **AH6** disposed as **informational**.
///
/// Mirrors `pyproject.toml`'s `[dependency-groups].dev` entry —
/// changing one without the other is a build failure. Drift between
/// this constant and the boundary-IoU oracle's pycocotools assumption
/// is a parity-contract change requiring an ADR.
pub const ORACLE_PYCOCOTOOLS_PIN: &str = "pycocotools==2.0.11";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_max_dets_is_300() {
        // Tripwire: lvis-api `Params` constructor pins `max_dets = 300`
        // at eval.py:528 (quirk AC1). The ADR-0026 default plan F2 is
        // keyed to this value; changing it without an ADR breaks the
        // strict-mode parity claim.
        assert_eq!(LVIS_DEFAULT_MAX_DETS, 300);
    }

    #[test]
    fn boundary_dilation_ratio_default_is_lvis_value() {
        // The bowenc0221 fork's `LVISEval` exposes `dilation_ratio` and
        // the LVIS convention is `0.008` (boundary-iou-quirks.md Q1,
        // ADR-0026 AE3). Different from the COCO/non-LVIS default of
        // `0.02` (BOUNDARY_DILATION_RATIO_DEFAULT in boundary_parity.rs).
        assert_eq!(LVIS_BOUNDARY_DILATION_RATIO_DEFAULT, 0.008);
    }

    #[test]
    fn parity_eps_matches_boundary_magnitude() {
        // Same magnitude as the boundary-IoU oracle's aligned-mode
        // tolerance.
        assert_eq!(LVIS_PARITY_EPS, 1e-9);
    }

    #[test]
    fn oracle_version_matches_vendoring_md() {
        // Tripwire: editing this constant without updating
        // tests/python/parity_lvis/oracle/VENDORING.md (or vice versa)
        // is a parity-contract change. Equality forces the editor to
        // acknowledge both files are out of step.
        assert_eq!(ORACLE_LVIS_VERSION, "0.5.3");
    }

    #[test]
    fn oracle_sha_matches_vendoring_md() {
        // Tripwire: as above for the pinned commit. The SHA points at
        // the upstream GitHub release commit (031ac21f) whose tree is
        // byte-equal to the PyPI lvis-0.5.3 sdist (verified via the
        // hash table in VENDORING.md §"Provenance — byte-equality").
        assert_eq!(
            ORACLE_LVIS_COMMIT_SHA,
            "031ac21f939bcb5f1ca8de2ab8704082e101ff9b"
        );
    }

    #[test]
    fn oracle_pycocotools_pin_matches_pyproject() {
        // Tripwire: this constant mirrors `pyproject.toml`'s
        // `[dependency-groups].dev` entry for the pycocotools oracle
        // (ADR-0002). The LVIS oracle's runtime requires the same pin;
        // editing one without the other defeats the reproducibility
        // contract for LVIS strict-mode parity (ADR-0026 §"Parity
        // strategy", quirk AH6).
        assert_eq!(ORACLE_PYCOCOTOOLS_PIN, "pycocotools==2.0.11");
    }
}
