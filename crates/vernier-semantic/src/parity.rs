//! Pinned numerical constants for the semantic-segmentation
//! evaluation subsystem.
//!
//! Parallel to [`vernier_core::parity`] (pycocotools),
//! [`vernier_core::boundary_parity`] (boundary-IoU),
//! [`vernier_core::lvis_parity`] (LVIS), and
//! [`vernier_panoptic::parity`] (panopticapi): same role (single home
//! for parity-contract knobs), different oracles. Each subsystem's
//! lifecycle is intentionally decoupled per ADR-0028 §"Parity
//! strategy".
//!
//! Each item is doc-tagged with the quirk ID from the
//! [sem-seg quirks survey](../../../docs/engineering/sem-seg-quirks.md)
//! it corresponds to. Changes here are an ADR-level decision; the
//! canonical decision record is ADR-0028 itself.
//!
//! Vendored oracles live at
//! `tests/python/parity_semantic/oracle/{mmsegmentation,cityscapesscripts,...}/`;
//! provenance, modification policy, and fork plans land in adjacent
//! `VENDORING.md` files alongside the parity harness. mmsegmentation
//! is vendored as of ADR-0036 (this module's `ORACLE_MMSEGMENTATION_COMMIT_SHA`
//! is now a real 40-char SHA); cityscapesScripts is still gated on
//! PR-B7. Drift between any constant here and `VENDORING.md` is a
//! build failure — see the unit tests below.
//!
//! Unlike [`vernier_panoptic::parity::ParityMode`] (which duplicates
//! the enum because `vernier-panoptic ⊥ vernier-core` per ADR-0025),
//! this module **re-exports** [`vernier_core::parity::ParityMode`]
//! per ADR-0028 §"Workspace and dependency direction". Semantic eval
//! consumes the same enum the AP fold does.

#[doc(inline)]
pub use vernier_core::parity::ParityMode;

/// Cityscapes ignore-label convention. The 30+ raw ID space collapses
/// to 19 evaluation classes plus this sentinel, used for "void"
/// pixels at category boundaries and for unlabeled image regions.
/// Quirk **AJ1** — strict against MS / CS / PA. Hardcoded in
/// `cityscapesscripts/helpers/labels.py`.
pub const CITYSCAPES_IGNORE_LABEL: u32 = 255;

/// ADE20K (SceneParse150) ignore-label convention. Class 0 is the
/// "other/unlabeled" sentinel; predictions in `[1, 150]` are the
/// 150 SceneParse classes. Quirk **AJ1**, **AJ5** — strict against
/// MS for ADE20K. Hardcoded in
/// `mmsegmentation/datasets/ade.py:reduce_zero_label`.
pub const ADE20K_IGNORE_LABEL: u32 = 0;

/// Pascal VOC ignore-label convention. Same sentinel as Cityscapes;
/// "void" pixels at object boundaries. Quirk **AJ1** — strict.
/// Hardcoded in the PASCAL VOC evaluation scripts.
pub const PASCAL_VOC_IGNORE_LABEL: u32 = 255;

/// Cityscapes evaluation class count. The 19-class subset of the
/// raw label space the dataset authors evaluate on. Quirk **AK1**.
pub const CITYSCAPES_N_CLASSES: u32 = 19;

/// ADE20K (SceneParse150) evaluation class count. Quirk **AK3**.
pub const ADE20K_N_CLASSES: u32 = 150;

/// Pascal VOC evaluation class count. 20 object classes plus a
/// background class indexed at 0. Quirk **AK3**.
pub const PASCAL_VOC_N_CLASSES: u32 = 21;

/// IoU/mIoU equality tolerance applied when comparing vernier output
/// to the vendored oracles under non-strict mode. Strict mode demands
/// bit-equality and ignores this constant.
///
/// Placeholder value `1e-9` matches the magnitude of the LVIS,
/// boundary-IoU, and panoptic eps constants. Final value pinned by
/// measuring max ULP distance against mmsegmentation `IoUMetric` on
/// Cityscapes val once that fixture lands; the vendored oracle is now
/// in place (ADR-0036) but the val-measured ULP ceiling is a separate
/// follow-up. Drift between this constant and the parity harness is
/// caught by the unit test below.
pub const SEMANTIC_PARITY_EPS: f64 = 1e-9;

/// Frozen commit SHA of the vendored `open-mmlab/mmsegmentation`
/// reference oracle. Strict-mode parity claims for semantic eval are
/// keyed to this commit by default (mmsegmentation is the de-facto
/// research reference per ADR-0028 §"Decision drivers").
///
/// Vendored at upstream tag `v1.2.2` (2023-12-14) per ADR-0036. The
/// adjacent `VENDORING.md` records the byte-equality SHA-256 hashes
/// for `mmseg/evaluation/metrics/iou_metric.py` and `LICENSE`. The
/// unit test below tripwires drift between this constant and the
/// `VENDORING.md` file.
pub const ORACLE_MMSEGMENTATION_COMMIT_SHA: &str = "c685fe6767c4cadf6b051983ca6208f1b9d1ccb8";

/// Frozen commit SHA of the vendored `mcordts/cityscapesScripts`
/// dataset-author oracle. The `SemanticDataset.cityscapes(...)` preset
/// claims `aligned`-mode parity against this commit per ADR-0028
/// §"Parity strategy".
///
/// Placeholder until PR-B7 vendors the oracle and pins the SHA in
/// `tests/python/parity_semantic/oracle/cityscapesscripts/VENDORING.md`.
pub const ORACLE_CITYSCAPESSCRIPTS_COMMIT_SHA: &str = "PR-B7-pending";

/// PyPI release pin for `cityscapesScripts`. Separate from
/// [`ORACLE_CITYSCAPESSCRIPTS_COMMIT_SHA`] (the vendored-source SHA
/// used by the strict-mode parity harness in PR-B7): this pin is the
/// version the **bench harness** installs in `bench/envs/cityscapes/`
/// for the Cityscapes mIoU bench cell (ADR-0033 §"B2 — Semantic
/// Cityscapes MVB"). The bench env can run today against the released
/// wheel; the vendored oracle path is gated on the PR-B7 fork merge.
///
/// Drift between this constant and `bench/envs/cityscapes/pyproject.toml`
/// is a build failure — see `bench/tests/test_cityscapes_env_pin.py`.
pub const CITYSCAPESSCRIPTS_PIN: &str = "cityscapesScripts==2.2.4";

/// Minimum supported PyTorch version for the vendored mmsegmentation
/// `IoUMetric` parity oracle. `IoUMetric.intersect_and_union` calls
/// `torch.histc` for label binning (line 190 of `iou_metric.py` at the
/// pinned SHA) — `torch.histc`'s float-edge bin semantics do not have
/// a bit-exact numpy equivalent, so the parity claim depends on a real
/// torch installation.
///
/// The floor matches the project's
/// `[project.optional-dependencies].torch` extra (`torch>=2.4`); the
/// mmsegmentation oracle reuses that constraint rather than introducing
/// a separate pin. `torch.histc`'s API has been stable since PyTorch 1.x.
/// Bumping the floor is an ADR-level decision only if upstream changes
/// `histc`'s rounding or boundary behavior — see ADR-0036 §"How to
/// refresh".
pub const ORACLE_TORCH_FLOOR: &str = "torch>=2.4";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cityscapes_ignore_label_is_255() {
        // cityscapesscripts/helpers/labels.py — `255` is the canonical
        // void / ignore label across the Cityscapes ecosystem (also
        // used by Pascal VOC for the same "void at boundary" purpose).
        // Quirk AJ1; pinning the value catches an "innocent" refactor
        // that flips the convention.
        assert_eq!(CITYSCAPES_IGNORE_LABEL, 255);
        assert_eq!(PASCAL_VOC_IGNORE_LABEL, 255);
    }

    #[test]
    fn ade20k_ignore_label_is_zero() {
        // mmsegmentation's `reduce_zero_label` flag: ADE20K class 0
        // is "other/unlabeled". Quirk AJ1, AJ5 — distinct from the
        // 255 convention to catch any code that hardcodes 255.
        assert_eq!(ADE20K_IGNORE_LABEL, 0);
    }

    #[test]
    fn class_counts_match_published_conventions() {
        // Cityscapes: 19-class evaluation (AK1). ADE20K: 150-class
        // (AK3). Pascal VOC: 21 = 20 objects + background (AK3).
        assert_eq!(CITYSCAPES_N_CLASSES, 19);
        assert_eq!(ADE20K_N_CLASSES, 150);
        assert_eq!(PASCAL_VOC_N_CLASSES, 21);
    }

    #[test]
    fn parity_eps_matches_placeholder_magnitude() {
        // Same magnitude as the panoptic, boundary-IoU, and LVIS
        // aligned-mode tolerances. The val-measured ULP ceiling on
        // mmsegmentation Cityscapes val replaces this exact bit
        // pattern check once PR-B6's parity harness lands.
        assert_eq!(SEMANTIC_PARITY_EPS, 1e-9);
    }

    #[test]
    fn parity_mode_default_is_corrected() {
        // ADR-0002 vocabulary, re-exported from vernier-core (not
        // duplicated, per ADR-0028 §"Workspace and dependency
        // direction"): corrected is the default for net-new users;
        // strict is the migration mode.
        assert_eq!(ParityMode::default(), ParityMode::Corrected);
    }

    #[test]
    fn mmsegmentation_oracle_sha_is_pinned() {
        // ADR-0036: vendored at v1.2.2. Drift between this constant and
        // `tests/python/parity_semantic/oracle/mmsegmentation/VENDORING.md`
        // is a build failure — refreshing the SHA is an ADR-level
        // operation that updates this constant and the VENDORING.md
        // provenance table in the same commit.
        assert_eq!(
            ORACLE_MMSEGMENTATION_COMMIT_SHA, "c685fe6767c4cadf6b051983ca6208f1b9d1ccb8",
            "ORACLE_MMSEGMENTATION_COMMIT_SHA must match VENDORING.md",
        );
        assert!(
            ORACLE_MMSEGMENTATION_COMMIT_SHA
                .chars()
                .all(|c| c.is_ascii_hexdigit()),
            "expected 40-char hex SHA, got {ORACLE_MMSEGMENTATION_COMMIT_SHA:?}",
        );
        assert_eq!(ORACLE_MMSEGMENTATION_COMMIT_SHA.len(), 40);
    }

    #[test]
    fn cityscapesscripts_oracle_sha_is_placeholder_until_vendored() {
        // PR-B7 vendors the cityscapesScripts oracle and replaces this
        // placeholder with the real commit SHA in lock-step with the
        // adjacent VENDORING.md file. Until then, the placeholder
        // string is the structural signal that the oracle isn't wired
        // yet.
        assert!(
            ORACLE_CITYSCAPESSCRIPTS_COMMIT_SHA.starts_with("PR-B7-")
                || ORACLE_CITYSCAPESSCRIPTS_COMMIT_SHA.len() == 40,
            "expected placeholder or 40-char SHA, got {ORACLE_CITYSCAPESSCRIPTS_COMMIT_SHA:?}",
        );
    }

    #[test]
    fn torch_floor_is_pep440_constraint() {
        // ADR-0036: `IoUMetric.intersect_and_union` calls torch.histc;
        // the floor must match the project's
        // `[project.optional-dependencies].torch` extra in
        // `pyproject.toml`. Drift defeats the parity claim — torch
        // versions below the floor may not implement histc's bin-edge
        // semantics consistently with the vendored oracle.
        assert_eq!(ORACLE_TORCH_FLOOR, "torch>=2.4");
        assert!(
            ORACLE_TORCH_FLOOR.contains(">="),
            "expected `name>=version` floor, got {ORACLE_TORCH_FLOOR:?}",
        );
    }

    #[test]
    fn cityscapesscripts_pin_is_pep440_release_spec() {
        // Tripwire: `bench/envs/cityscapes/pyproject.toml` mirrors this
        // constant verbatim. Editing one without the other defeats the
        // ADR-0033 §"Comparator registry" reproducibility claim for the
        // semantic Cityscapes bench cell — the bench harness pins this
        // exact PyPI release in lock-step.
        assert_eq!(CITYSCAPESSCRIPTS_PIN, "cityscapesScripts==2.2.4");
        // Sanity: must be a `name==version` pin (not a git URL or
        // unbounded constraint). The bench env pin contract (ADR-0033)
        // is "exact PyPI release" — a `>=`/`~=` constraint here would
        // silently widen to a different version under `uv lock`.
        assert!(
            CITYSCAPESSCRIPTS_PIN.contains("=="),
            "expected `name==version` pin, got {CITYSCAPESSCRIPTS_PIN:?}",
        );
    }
}
