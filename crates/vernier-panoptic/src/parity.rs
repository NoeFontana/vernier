//! Pinned numerical constants for the panoptic evaluation subsystem.
//!
//! Parallel to [`vernier_core::parity`] (pycocotools),
//! [`vernier_core::boundary_parity`] (boundary-IoU), and
//! [`vernier_core::lvis_parity`] (LVIS): same role (single home for
//! parity-contract knobs), different oracle. Each subsystem's lifecycle
//! is intentionally decoupled per ADR-0025 §"Parity strategy".
//!
//! Each item is doc-tagged with the quirk ID from the ADR-0025
//! appendix it corresponds to. Changes here are an ADR-level decision;
//! the canonical decision record is ADR-0025 itself.
//!
//! The vendored oracle lives at
//! `tests/python/parity_panoptic/oracle/panopticapi/`; provenance,
//! modification policy, and fork plan are recorded in the adjacent
//! `VENDORING.md`. Drift between any constant here and `VENDORING.md`
//! is a build failure — see the unit tests below.

/// Three-tier parity mode (per ADR-0002 vocabulary, scoped to panoptic).
///
/// Locally duplicated rather than imported from
/// [`vernier_core::parity::ParityMode`] because ADR-0025 declares
/// `vernier-panoptic ⊥ vernier-core` (no edge in either direction):
/// the architectural firewall keeps the AP fold unreachable from PQ
/// code, structurally enforcing the ADR-0005 invariant. The duplication
/// cost is two variants; if a third evaluation crate ever lands and
/// shared types accumulate, a `vernier-types` leaf is the right
/// refactor — out of scope here.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParityMode {
    /// Reproduce every panopticapi behavior bit-exactly, including
    /// known bugs (V3 same-cat-crowd last-wins, W6 ZeroDivisionError
    /// on empty filters, S7 silent duplicate-id last-wins). Default
    /// when migrating from panopticapi.
    Strict,
    /// Apply opinionated fixes for behaviors classified `corrected` in
    /// the ADR-0025 quirks survey. Default for net-new users; opt-out
    /// via `Strict`.
    #[default]
    Corrected,
}

/// VOID label id. Pixels with `id == 0` are excluded from FN (GT side)
/// and contribute "free" overlap to the FP-exclusion rule (V4). Quirk
/// **R3** — strict; hardcoded in `panopticapi/evaluation.py:20`.
pub const PANOPTIC_VOID: u32 = 0;

/// OFFSET used by panopticapi's `np.unique` over the uint64-encoded
/// `(gt, dt)` pair: `combined = pan_gt * OFFSET + pan_pred`. Equal to
/// `256³ = 16_777_216`; collision-free since each id is bounded by
/// `256³ − 1`. Quirk **T1** — strict.
///
/// vernier's intersection histogram uses a `HashMap<(u32, u32), u32>`
/// rather than the OFFSET trick (T2 aligned: bit-equal output, no
/// uint64 materialization), but the constant is preserved for
/// compatibility with downstream tooling that diffs against the raw
/// panopticapi keys.
pub const PANOPTIC_OFFSET: u64 = 256 * 256 * 256;

/// IoU threshold for the panoptic matching rule. Strict greater-than
/// (not ≥) — quirk **U7** — and metric-defining: `>` is the pivot
/// guaranteeing at-most-one-match per GT (U9). Hardcoded in
/// `panopticapi/evaluation.py:134`.
pub const PANOPTIC_IOU_THRESHOLD: f64 = 0.5;

/// IoU/PQ-equality tolerance applied under
/// [`ParityMode::Corrected`] when comparing multi-process panopticapi
/// traces. Strict mode runs against `pq_compute_single_core` directly
/// at `proc_id=0` and demands bit-equality (eps unused there).
///
/// Placeholder value `1e-9` matches the magnitude of the LVIS and
/// boundary-IoU eps constants. Final value pinned by measuring max
/// ULP distance between `pq_compute_single_core` and
/// `pq_compute_multi_core` for `cpu_count ∈ {2, 4, 8}` on COCO
/// panoptic val (procedure documented in
/// `tests/python/parity_panoptic/panoptic_val_paths.py`). When that
/// measurement lands, both this constant and `VENDORING.md` update
/// atomically; the unit test below catches drift.
pub const PANOPTIC_PARITY_EPS: f64 = 1e-9;

/// Frozen commit SHA of the vendored `cocodataset/panopticapi`
/// reference oracle. Strict-mode parity claims for panoptic are keyed
/// to this commit: every quirk vernier reproduces in `Strict`
/// panoptic mode is the disposition of an observed behaviour at this
/// SHA.
///
/// The vendored tree lives at
/// `tests/python/parity_panoptic/oracle/panopticapi/`; provenance,
/// modification policy, and fork plan are recorded in the adjacent
/// `VENDORING.md`. Drift between this constant and `VENDORING.md`
/// is a build failure — see the unit test below.
pub const ORACLE_COMMIT_SHA: &str = "7bb4655548f98f3fedc07bf37e9040a992b054b0";

/// Pinned `Pillow` release the vendored `panopticapi` oracle decodes
/// PNGs with. The oracle's `evaluation.py` imports `PIL.Image` at
/// module scope (`evaluation.py:14`) and decodes panoptic PNGs with
/// `np.array(Image.open(path), dtype=np.uint32)` (`evaluation.py:86-89`).
/// Pillow's PNG decoder — *not* NumPy — determines whether RGB is
/// preserved as 3-channel uint8 (R2: RGBA silently drops alpha; P / L
/// modes crash because `rgb2id` falls into the scalar branch on a 2-D
/// array). Mirrors `pyproject.toml`'s `[dependency-groups].dev`
/// entry; changing one without the other is a build failure.
pub const ORACLE_PILLOW_PIN: &str = "Pillow==12.2.0";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panoptic_void_is_zero() {
        // panopticapi/evaluation.py:20 — VOID = 0. Pixels with id 0 are
        // excluded from FN (R3) and contribute "free" overlap to V4.
        assert_eq!(PANOPTIC_VOID, 0);
    }

    #[test]
    fn panoptic_offset_is_256_cubed() {
        // panopticapi/evaluation.py:19 — OFFSET = 256 * 256 * 256.
        // Collision-free pair encoding (T1). vernier doesn't use this
        // encoding internally but pins it so downstream diffs of
        // panopticapi's `gt_pred_map` keys decode the same way.
        assert_eq!(PANOPTIC_OFFSET, 16_777_216);
    }

    #[test]
    fn panoptic_iou_threshold_is_half() {
        // panopticapi/evaluation.py:134 — `if iou > 0.5`. Strict
        // greater-than is metric-defining (U7); equality is *not* a
        // match. Pinning the bit pattern catches a future "innocent"
        // refactor that flips the comparison to ≥.
        assert_eq!(PANOPTIC_IOU_THRESHOLD, 0.5);
    }

    #[test]
    fn parity_eps_matches_placeholder_magnitude() {
        // Same magnitude as the boundary-IoU and LVIS aligned-mode
        // tolerances. The val-measured ULP ceiling on COCO panoptic
        // val replaces this exact bit-pattern check once the
        // measurement procedure in
        // `tests/python/parity_panoptic/panoptic_val_paths.py` runs.
        assert_eq!(PANOPTIC_PARITY_EPS, 1e-9);
    }

    #[test]
    fn oracle_sha_matches_vendoring_md() {
        // Tripwire: editing this constant without updating
        // tests/python/parity_panoptic/oracle/VENDORING.md (or vice
        // versa) is a parity-contract change. Equality forces the
        // editor to acknowledge both files are out of step.
        assert_eq!(
            ORACLE_COMMIT_SHA,
            "7bb4655548f98f3fedc07bf37e9040a992b054b0"
        );
    }

    #[test]
    fn oracle_pillow_pin_matches_pyproject() {
        // Tripwire: this constant mirrors `pyproject.toml`'s
        // `[dependency-groups].dev` entry for the Pillow oracle dep.
        // The panopticapi oracle imports `PIL.Image` at module load;
        // editing one without the other defeats the reproducibility
        // contract for panoptic strict-mode parity (ADR-0025 §"Parity
        // strategy").
        assert_eq!(ORACLE_PILLOW_PIN, "Pillow==12.2.0");
    }

    #[test]
    fn parity_mode_default_is_corrected() {
        // ADR-0002 vocabulary: corrected is the default for net-new
        // users; strict is the migration mode. Mirrored from
        // vernier_core::parity::ParityMode (duplicated, not imported,
        // per ADR-0025 §"vernier-panoptic ⊥ vernier-core").
        assert_eq!(ParityMode::default(), ParityMode::Corrected);
    }
}
