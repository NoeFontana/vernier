//! Oriented-box and polygon geometry for the vernier evaluation library.
//!
//! Per ADR-0063 this is a pure-Rust leaf crate: no Python dependency, no
//! dependency on `vernier-core`, and nothing here knows what a detection
//! or a category is. It answers one question — *how much do these two
//! shapes overlap* — and ADR-0063 has it answer that question in more
//! than one voice.
//!
//! This is the first of them: [`kernel`], the **canonical** f64 kernel
//! that backs `parity_mode="corrected"`. The two op-exact oracle
//! replicas that back `parity_mode="strict"` — detectron2's f32
//! rotated-box IoU and DOTA_devkit's f64 polygon IoU — land in the
//! follow-up as `replica::d2` and `replica::dk`.
//!
//! The split exists because the two oracles disagree with each other
//! and with exact geometry, and both disagreements are load-bearing.
//! `strict` reproduces an oracle bit for bit; `corrected` computes the
//! answer. ADR-0008 rejected exactly this kind of parity-mode branch for
//! the bbox kernel, where the alternative only bought throughput — here
//! the canonical kernel has a correctness mandate, since D2 emits IoU
//! outside `[0, 1]` and runs its geometry in f32 while DK returns
//! non-zero residues on disjoint pairs.
//!
//! # Conventions are required, never inferred
//!
//! An oriented box means nothing without an angle unit and a rotation
//! direction, and getting either wrong produces plausible numbers rather
//! than an error. [`Convention`] therefore has no `Default`, and every
//! entry point takes one. See [`convention`] for the quotient structure
//! that makes `le90` / `le135` / `oc` a non-question.
//!
//! # Worked example
//!
//! ```
//! use vernier_geom::{Convention, Denominator, PreparedRBox, RotatedBox};
//!
//! // detectron2's convention: degrees, counter-clockwise on screen.
//! let conv = Convention::D2;
//! let gt = PreparedRBox::new(
//!     RotatedBox { cx: 0.0, cy: 0.0, w: 4.0, h: 2.0, theta: 0.0 },
//!     conv,
//! );
//! let dt = PreparedRBox::new(
//!     RotatedBox { cx: 1.0, cy: 0.0, w: 4.0, h: 2.0, theta: 0.0 },
//!     conv,
//! );
//!
//! // Intersection 6, union 10.
//! let iou = vernier_geom::rbox_iou(&gt, &dt, conv, Denominator::Union);
//! assert_eq!(iou, 0.6);
//!
//! // A box against itself is exactly 1.0, not 0.9999999999999999.
//! assert_eq!(vernier_geom::rbox_iou(&gt, &gt, conv, Denominator::Union), 1.0);
//! ```
//!
//! # Numerical policy
//!
//! - The canonical kernel is f64 end-to-end and clips in the ground
//!   truth's own frame, so its error is bounded by the pair's aspect
//!   ratio rather than by the image coordinate magnitude.
//! - No FMA anywhere. Rust does not contract implicitly, and nothing
//!   here asks for a fused multiply-add.
//! - Results are bit-identical across dispatch targets and thread
//!   counts (ADR-0047, "one wheel, one behavior"): the only SIMD in the
//!   crate is the broad phase, which vectorizes *across* pairs and never
//!   reassociates a reduction within one.
//!
//! See `docs/engineering/obb-quirks.md` for the `OB` rows this crate
//! dispositions.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod angle;
pub mod broad;
pub mod calipers;
pub mod clip;
pub mod convention;
pub mod error;
pub mod kernel;
pub mod prepared;
pub mod quad;

pub use angle::{angle_error_deg, NEAR_SQUARE_TAU};
pub use calipers::min_area_rect;
pub use convention::{AngleUnit, Convention, RotatedBox, Rotation};
pub use error::{GeomError, ShapeKind};
pub use kernel::{quad_iou, rbox_iou, Denominator};
pub use prepared::{Aabb, PreparedRBox};
pub use quad::{PreparedQuad, Quad};

/// Library version string, lockstep with every crate in the workspace.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_set() {
        assert!(!VERSION.is_empty());
    }
}
