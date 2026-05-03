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

    /// Streaming evaluator memory budget exceeded. Carries a breakdown of
    /// where bytes are spent so the user can pick a remediation (shard,
    /// shrink iou_thresholds, raise budget).
    #[error("memory budget exceeded: used {used_bytes} / budget {budget_bytes} bytes")]
    OutOfBudget {
        /// Total bytes the evaluator was holding when it tripped the budget.
        used_bytes: usize,
        /// Configured budget cap.
        budget_bytes: usize,
        /// Stable keys: `"cells_store"`, `"scores"`, `"match_flags"`. The
        /// schema is future-additive — consumers must tolerate extra keys.
        breakdown: std::collections::HashMap<&'static str, usize>,
    },

    /// Feature wired but not yet implemented in v0. Used by the streaming
    /// evaluator's `checkpoint`/`restore` pair, deferred per the user's
    /// scope decision; future ADR re-introduces the implementation.
    #[error("not implemented: {feature}")]
    NotImplemented {
        /// Stable identifier of the unimplemented feature, e.g.
        /// `"StreamingEvaluator::checkpoint"`.
        feature: &'static str,
    },

    /// `per_pair` row count exceeded the configured cap (ADR-0019
    /// `TablesConfig::per_pair_max_rows`). Carries the observed count
    /// at the moment the cap was tripped and the cap value, so callers
    /// can decide whether to raise the cap or constrain the workload.
    #[error("per_pair table exceeded cap: would emit at least {observed} rows, cap {cap}")]
    PerPairOverflow {
        /// Best-effort lower bound on the row count at the moment the
        /// cap was tripped. The check is per-cell so the actual final
        /// count may be larger; this is the value that triggered the
        /// abort.
        observed: usize,
        /// `TablesConfig::per_pair_max_rows` value the caller (or
        /// default) configured.
        cap: usize,
    },

    /// LVIS federated metadata violates the disjointness invariant
    /// for one `(image, category)` cell: the category appears in both
    /// `not_exhaustive_category_ids` and `neg_category_ids` (or is
    /// listed in `neg_category_ids` while a GT of that category exists,
    /// which would put it implicitly in `pos`). Quirk **AA7** of
    /// ADR-0026, dispositioned `corrected`: lvis-api silently picks
    /// `not_exhaustive` on overlap; vernier rejects at load.
    #[error("lvis federated conflict on image_id={image_id}, category_id={category_id}: {detail}")]
    LvisFederatedConflict {
        /// Offending image id.
        image_id: i64,
        /// Offending category id.
        category_id: i64,
        /// Free-form detail string identifying which constraint failed
        /// (e.g., `"category in both not_exhaustive and neg"`).
        detail: &'static str,
    },

    /// LVIS dataset is missing the `frequency` field on one or more
    /// categories. Quirk **AB6** of ADR-0026, dispositioned `corrected`:
    /// lvis-api raises `KeyError` mid-eval on the first miss; vernier
    /// raises at load with the full list of offending categories so
    /// the failure is debuggable in one shot.
    ///
    /// The `category_ids` list is sorted ascending for stable error
    /// messages.
    #[error(
        "lvis dataset is missing `frequency` on {} categories: {category_ids:?}",
        category_ids.len()
    )]
    MissingFrequency {
        /// Sorted list of category ids that lacked a `frequency` value.
        category_ids: Vec<i64>,
    },
}
