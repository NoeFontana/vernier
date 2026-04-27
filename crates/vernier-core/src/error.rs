//! Typed errors for the evaluator.
//!
//! Per the workspace clippy lints, we forbid `panic!`, `unwrap`, and
//! `expect` in non-test code. Every fallible operation in `vernier-core`
//! returns `Result<_, EvalError>`, including `Similarity::compute`
//! (per ADR-0005).

use thiserror::Error;
use vernier_mask::MaskError;

/// Unified error type for evaluation paths.
///
/// Variants are kept coarse on purpose: each one corresponds to a class
/// of failure a caller can plausibly recover from or report distinctly.
/// We add new variants as they're needed, rather than enumerating every
/// possible cause up front.
#[derive(Debug, Error)]
pub enum EvalError {
    /// Two annotations or two RLEs disagree on dimensions in a way that
    /// makes the operation undefined. Replaces the `-1` sentinel
    /// pycocotools' `rleIou` returns on dimension mismatch (quirk
    /// **I2**, dispositioned `corrected` per ADR-0002).
    #[error("dimension mismatch: {detail}")]
    DimensionMismatch {
        /// Free-form detail string for the operator that detected the
        /// mismatch; carries the offending dimensions.
        detail: String,
    },

    /// Annotation could not be parsed from JSON, or referenced an
    /// `image_id` / `category_id` that the dataset does not contain.
    /// Quirk **J5** in pycocotools is the matching enforcement on
    /// `loadRes`.
    #[error("invalid annotation: {detail}")]
    InvalidAnnotation {
        /// Free-form detail string identifying the offending field.
        detail: String,
    },

    /// JSON deserialization failed before any vernier-side validation.
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),

    /// Mask-side operation failed (codec decode, polygon rasterization,
    /// merge dimension mismatch). Propagated from `vernier-mask` per
    /// ADR-0009's one-way dependency.
    #[error("mask: {0}")]
    Mask(#[from] MaskError),

    /// Numeric input was not finite (NaN or infinity reached an
    /// arithmetic that cannot tolerate it). Used at boundaries where
    /// we receive scores or coordinates from external code.
    #[error("non-finite value in {context}")]
    NonFinite {
        /// Where the non-finite value was encountered.
        context: &'static str,
    },

    /// Caller-supplied evaluation parameters are inconsistent with the
    /// data they're being applied to (e.g., a maxDet value that the
    /// accumulator never saw, an IoU threshold absent from the
    /// ladder). Distinct from `InvalidAnnotation`, which is for
    /// dataset-side data errors.
    #[error("invalid config: {detail}")]
    InvalidConfig {
        /// Free-form detail string identifying the offending parameter.
        detail: String,
    },
}
