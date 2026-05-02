//! Similarity computation between ground-truth and detection annotations.
//!
//! Per ADR-0005, the [`Similarity`] trait is the single extension point
//! for new IoU types. Each impl owns its own configuration (sigmas,
//! crowd asymmetry, prefilter thresholds, …) and the matching engine
//! reads only the trait — never a concrete IoU type.
//!
//! Per ADR-0004 (and ADR-0008 for bbox), kernels pick an internal numeric
//! type tuned to the algorithm's error budget — bbox IoU is `f64`
//! end-to-end; future segm and OKS kernels follow the f32-internal /
//! f64-boundary policy from ADR-0004. Every impl writes the boundary
//! matrix as `f64`.
//!
//! Per ADR-0003, inner loops are wrapped in `pulp::Arch::dispatch` so
//! the same source compiles to AVX2 / AVX-512 / NEON variants picked at
//! process start; no nightly required.

use ndarray::ArrayViewMut2;

use crate::error::EvalError;

pub mod bbox;
pub mod boundary;
pub mod oks;
pub mod segm;

pub use bbox::{BboxAnn, BboxIou};
pub use boundary::{BoundaryGtCache, BoundaryIou};
pub(crate) use boundary::boundary_iou_compute;
pub use oks::{OksAnn, OksSimilarity, COCO_PERSON_SIGMAS};
pub use segm::{SegmAnn, SegmGtCache, SegmIou};
pub(crate) use segm::{segm_iou_compute, SegmComputeScratch};

/// Computes a pairwise similarity matrix for one image's GT × DT pairs.
///
/// The associated [`Self::Annotation`] type is the per-impl annotation
/// shape. For bbox ([`BboxIou`]) it carries the bbox plus the `is_crowd`
/// flag — the E1 asymmetry lives in the impl, not in matching, so new
/// IoU types (segm, OKS, …) plug in without touching the matching
/// engine.
pub trait Similarity: Send + Sync {
    /// Per-impl annotation shape. The matching engine sees only the
    /// matrix produced by [`Self::compute`]; the annotation type stays
    /// inside the impl plus its caller.
    type Annotation;

    /// Writes the gt × dt similarity matrix into `out`.
    ///
    /// Rows index ground-truth annotations; columns index detections.
    /// Out-of-band signals (impossible match, dimension issue) are
    /// reported via [`EvalError`]; impls do not panic. "No overlap" is
    /// expressed as `0.0`, never as a sentinel — quirk **I2**
    /// `corrected`, replacing pycocotools' `-1` return on RLE dim
    /// mismatch.
    fn compute(
        &self,
        gts: &[Self::Annotation],
        dts: &[Self::Annotation],
        out: &mut ArrayViewMut2<'_, f64>,
    ) -> Result<(), EvalError>;
}
