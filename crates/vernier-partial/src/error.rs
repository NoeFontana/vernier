//! Typed errors for the partial wire format and merge surface
//! (ADR-0031, generalized in ADR-0032).
//!
//! These variants live in `vernier-partial` so all three paradigm
//! crates (`vernier-core`, `vernier-semantic`, `vernier-panoptic`)
//! share one vocabulary. Each paradigm's top-level error type wraps
//! [`PartialError`] via a single `From` arm so the FFI can map the
//! five Python exception classes uniformly.
//!
//! The [`PartialFormatErrorKind::tag`] strings are part of the public
//! Python surface — tests assert against them — and must stay stable
//! across format-version bumps.

use thiserror::Error;

/// Top-level error returned by the encode/decode and merge surface in
/// this crate. Each variant maps 1:1 to one of the five Python
/// exception classes (`PartialFormatMismatch`, `PartialDatasetMismatch`,
/// `PartialParamsMismatch`, `PartialPartitionOverlap`,
/// `PartialRankCollision`).
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PartialError {
    /// Framing or structural check tripped. See
    /// [`PartialFormatErrorKind`] for which one.
    #[error("partial wire format rejected: {kind}")]
    Format {
        /// Sub-discriminator naming the specific check.
        kind: PartialFormatErrorKind,
    },

    /// Partial's `dataset_hash` does not match the receiving rank's
    /// live dataset. Sampler / config bug; refusing protects the merge
    /// from silently producing un-reproducible numbers.
    #[error("partial dataset_hash mismatch: expected {expected:02x?}, got {actual:02x?}")]
    DatasetMismatch {
        /// Receiving rank's `dataset_hash`.
        expected: [u8; 32],
        /// Partial's declared `dataset_hash`.
        actual: [u8; 32],
    },

    /// Partial's `params_hash` does not match the receiving rank's.
    /// Means params (max_dets / iou_thresholds / use_cats / …) diverged.
    #[error("partial params_hash mismatch: expected {expected:02x?}, got {actual:02x?}")]
    ParamsMismatch {
        /// Receiving rank's `params_hash`.
        expected: [u8; 32],
        /// Partial's declared `params_hash`.
        actual: [u8; 32],
    },

    /// Two partials cover the same `image_id` — the disjoint-partition
    /// rule is violated. Names both rank ids and the colliding image
    /// so the user can fix their `DistributedSampler`.
    #[error("partials cover image_id={image_id} on both rank {rank_a} and rank {rank_b}")]
    PartitionOverlap {
        /// Lower rank id (`min(a, b)`) — sorted for determinism.
        rank_a: u32,
        /// Higher rank id.
        rank_b: u32,
        /// Image id that appeared in both partials' `seen_images`.
        image_id: i64,
    },

    /// Two strict-mode partials declare the same `rank_id`. Strict
    /// merge requires distinct ids so the cross-rank tiebreak gives a
    /// total order. Corrected mode tolerates collisions.
    #[error("partials share rank_id={rank_id} in strict mode")]
    RankCollision {
        /// The duplicated rank id.
        rank_id: u32,
    },
}

/// Sub-discriminator for [`PartialError::Format`]. The validation
/// pipeline runs cheapest-first, so the variant also indicates how
/// far the validator got before tripping.
///
/// **Tag stability is part of the public Python surface.** The
/// snake_case [`Self::tag`] strings appear on the FFI exception's
/// `kind` attribute and are asserted against in the parity test
/// suite. Renames require a coordinated stub + test update.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PartialFormatErrorKind {
    /// Length too short to contain even the framing header.
    #[error("partial too short: observed {observed} bytes, minimum {minimum}")]
    TooShort {
        /// Observed length.
        observed: usize,
        /// Minimum required.
        minimum: usize,
    },
    /// Magic-bytes prefix did not match `b"VRPS"`.
    #[error("wrong magic bytes: expected \"VRPS\", got {found:02x?}")]
    WrongMagic {
        /// First four bytes found.
        found: [u8; 4],
    },
    /// `format_version` byte did not match the receiving rank's
    /// compiled-in [`crate::envelope::FORMAT_VERSION`].
    #[error("wrong format_version: expected {expected}, got {found}")]
    WrongVersion {
        /// Compiled-in version.
        expected: u8,
        /// Found in the partial.
        found: u8,
    },
    /// CRC32 footer did not match. Truncation, transport corruption,
    /// or hand-crafted payload.
    #[error("crc32 footer mismatch")]
    Crc,
    /// `paradigm_kind` byte did not match. Means the partial belongs
    /// to a different paradigm (e.g., a semantic partial loaded by
    /// an instance evaluator).
    #[error("paradigm_kind mismatch: expected {expected}, got {found}")]
    ParadigmMismatch {
        /// Receiving evaluator's paradigm.
        expected: u8,
        /// Partial's declared paradigm.
        found: u8,
    },
    /// `discriminator` did not match within a paradigm. For instance:
    /// merging a bbox partial into a segm evaluator. Each paradigm
    /// defines its own discriminator semantics; the value is opaque
    /// to this crate.
    ///
    /// The tag `"kernel_mismatch"` is preserved for backward
    /// compatibility with ADR-0031 v1 instance partials.
    #[error("kernel_kind mismatch: expected discriminant {expected}, got {found}")]
    KernelMismatch {
        /// Receiving evaluator's discriminator.
        expected: u32,
        /// Partial's declared discriminator.
        found: u32,
    },
    /// Header `shape_fingerprint` did not match. Each paradigm
    /// interprets the four `u32` slots: instance uses
    /// `(n_categories, n_area_ranges, n_images, retain_iou)`,
    /// semantic uses `(n_classes, 0, n_images, 0)`, panoptic uses
    /// `(n_categories, 0, n_images, things_stuff_split)`.
    ///
    /// The tag `"grid_mismatch"` is preserved for backward
    /// compatibility with ADR-0031 v1 instance partials.
    #[error("shape fingerprint mismatch: {detail}")]
    GridMismatch {
        /// Free-form detail naming which slot mismatched.
        detail: String,
    },
    /// `parity_mode` byte did not match.
    #[error("parity_mode mismatch: expected discriminant {expected}, got {found}")]
    ParityMismatch {
        /// Receiving evaluator's parity mode discriminant.
        expected: u8,
        /// Partial's declared parity mode discriminant.
        found: u8,
    },
    /// rkyv archive validation refused the payload. Pointer offsets
    /// out of range, bad enum tag, etc.
    #[error("rkyv archive validation failed: {detail}")]
    RkyvDecode {
        /// rkyv's diagnostic message.
        detail: String,
    },
}

impl PartialFormatErrorKind {
    /// Stable snake_case identifier for this variant. Surfaced on the
    /// FFI exception's `kind` attribute. Adding a variant requires
    /// adding a tag here — the exhaustive match makes the omission a
    /// compile error.
    pub fn tag(&self) -> &'static str {
        match self {
            Self::TooShort { .. } => "too_short",
            Self::WrongMagic { .. } => "wrong_magic",
            Self::WrongVersion { .. } => "wrong_version",
            Self::Crc => "crc",
            Self::ParadigmMismatch { .. } => "paradigm_mismatch",
            Self::KernelMismatch { .. } => "kernel_mismatch",
            Self::GridMismatch { .. } => "grid_mismatch",
            Self::ParityMismatch { .. } => "parity_mismatch",
            Self::RkyvDecode { .. } => "rkyv_decode",
        }
    }
}
