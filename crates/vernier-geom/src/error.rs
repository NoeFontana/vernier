//! Error type for oriented-box and polygon geometry.

use thiserror::Error;

/// Errors raised while validating or preparing oriented geometry.
///
/// Per ADR-0063 the ingestion layer is strict where `pycocotools`-style
/// sentinels would be silent: non-finite coordinates and zero-area
/// quads are typed errors in *both* parity modes, because there is no
/// oracle value to reproduce — DK's zero-area path evaluates `0/0` and
/// returns `NaN`, which the matching engine's finite-matrix contract
/// forbids (quirk **OB7**).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum GeomError {
    /// A coordinate, extent or angle was `NaN` or infinite.
    ///
    /// The kernel contract in ADR-0005 requires a finite similarity
    /// matrix; a non-finite input can only produce a non-finite entry,
    /// so it is rejected at ingestion rather than propagated.
    #[error("non-finite {field} in {shape} geometry (value index {index})")]
    NonFinite {
        /// Which geometry family the offending value belongs to.
        shape: ShapeKind,
        /// Which slot of the tuple was non-finite.
        field: &'static str,
        /// Index into the raw `[f64; 5]` / `[f64; 8]` payload.
        index: usize,
    },

    /// A quad encloses zero signed area. Quirk **OB7** (`corrected` in
    /// both modes): DK computes `inter / (|A1| + |A2| - inter)` and
    /// returns `NaN` here.
    #[error("degenerate quad: signed area is zero (collinear or duplicated vertices)")]
    ZeroAreaQuad,

    /// A quad's edges cross. Quirk **OB16** (`corrected`): DK's
    /// triangle-fan decomposition silently returns a
    /// cancellation-dependent value for a self-intersecting polygon, so
    /// the canonical kernel refuses it.
    #[error("self-intersecting quad: edges {a} and {b} cross")]
    SelfIntersectingQuad {
        /// First crossing edge index, `0..4`.
        a: usize,
        /// Second crossing edge index, `0..4`.
        b: usize,
    },
}

/// Which geometry family a [`GeomError`] refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeKind {
    /// Center-based rotated box, `[cx, cy, w, h, theta]`.
    RotatedBox,
    /// Four vertices, `[x0, y0, x1, y1, x2, y2, x3, y3]`.
    Quad,
}

impl core::fmt::Display for ShapeKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::RotatedBox => f.write_str("rotated-box"),
            Self::Quad => f.write_str("quad"),
        }
    }
}
