//! Trait surface for paradigm-specific partial bodies.
//!
//! `vernier-partial` owns the wire envelope (magic, version, CRC,
//! header) and the partition + rank-collision policy; each paradigm
//! crate owns its body type and its merge accumulator. The traits
//! here are the seam.
//!
//! Encode and decode are routed through [`crate::envelope::encode`]
//! and [`crate::envelope::decode`] respectively, both of which take
//! the body archive bytes as already-rkyv-archived `&[u8]` so this
//! crate never touches paradigm-specific rkyv generics.

/// Paradigm discriminator carried in the wire envelope header.
///
/// Stable u8 values: load-bearing across the format-version boundary
/// because cross-paradigm partials are rejected by integer compare on
/// this byte (see [`crate::error::PartialFormatErrorKind::ParadigmMismatch`]).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParadigmKind {
    /// Instance detection (bbox / segm / boundary / keypoints) —
    /// the AP fold from `vernier-core`.
    Instance = 0,
    /// Semantic segmentation — confusion-matrix fold from
    /// `vernier-semantic`.
    Semantic = 1,
    /// Panoptic quality — per-category PqStat fold from
    /// `vernier-panoptic`.
    Panoptic = 2,
}

impl ParadigmKind {
    /// Map a wire byte back to the typed enum. Returns `None` if the
    /// byte is unknown — the receiver surfaces this as
    /// [`PartialError::Format`] with kind
    /// [`crate::error::PartialFormatErrorKind::ParadigmMismatch`].
    pub fn from_u8(b: u8) -> Option<Self> {
        match b {
            0 => Some(Self::Instance),
            1 => Some(Self::Semantic),
            2 => Some(Self::Panoptic),
            _ => None,
        }
    }

    /// The wire byte for this paradigm.
    pub const fn as_u8(self) -> u8 {
        self as u8
    }
}

/// Receiving rank's expectation of what a partial should look like.
/// Each paradigm builds one from its own (dataset, params, parity_mode,
/// evaluator config) and passes it to [`crate::envelope::decode`].
///
/// Validation is paradigm-agnostic byte compare except for
/// `shape_fingerprint`, where the receiver supplies its own values
/// and the four-slot interpretation is paradigm-specific (instance:
/// `(n_categories, n_area_ranges, n_images, retain_iou)`; semantic:
/// `(n_classes, 0, n_images, 0)`; panoptic: `(n_categories, 0,
/// n_images, things_stuff_split)`).
#[derive(Debug, Clone)]
pub struct PartialExpectation {
    /// Receiving evaluator's paradigm.
    pub paradigm: ParadigmKind,
    /// Paradigm-defined sub-kind (instance: `KernelKind` discriminant;
    /// semantic / panoptic: `0`).
    pub discriminator: u32,
    /// `ParityMode` discriminant. Strict and corrected cells differ
    /// by construction so a mismatched mode means the partial cannot
    /// be merged.
    pub parity_mode: u8,
    /// Receiving rank's `dataset_hash`. Partial must match exactly.
    pub dataset_hash: [u8; 32],
    /// Receiving rank's `params_hash`. Partial must match exactly.
    pub params_hash: [u8; 32],
    /// Paradigm-specific four-slot fingerprint.
    pub shape_fingerprint: [u32; 4],
    /// Whether the receiving evaluator is in strict mode. Affects the
    /// rank-collision check (corrected mode tolerates duplicate ranks;
    /// strict refuses).
    pub strict_mode: bool,
}

/// Marker trait declaring the paradigm a partial body type belongs to.
/// Each paradigm crate implements this on its `WirePartialBody` (or
/// equivalent) wire-form struct.
///
/// The trait does not carry rkyv bounds because each paradigm is
/// responsible for its own body archive — `vernier-partial` only
/// transports already-archived body bytes.
pub trait Partial {
    /// Which paradigm this partial body belongs to. Carried in the
    /// wire envelope header for cross-paradigm rejection.
    const PARADIGM: ParadigmKind;
}
