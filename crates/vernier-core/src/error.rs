//! Typed errors for the evaluator.
//!
//! Per the workspace clippy lints, we forbid `panic!`, `unwrap`, and
//! `expect` in non-test code. Every fallible operation in `vernier-core`
//! returns `Result<_, EvalError>`, including `Similarity::compute`
//! (per ADR-0005).

use thiserror::Error;
use vernier_mask::MaskError;
use vernier_partial::PartialError;

// Re-export the shared sub-discriminator under its existing path so
// callers (FFI, tests) keep using `EvalError::PartialFormatMismatch
// { kind: PartialFormatErrorKind }` unchanged after ADR-0032's move
// of the framing logic into the leaf crate.
pub use vernier_partial::PartialFormatErrorKind;

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

    /// Partial wire-format header / framing rejected by
    /// [`crate::distributed::validate_partial`] (ADR-0031). The `kind`
    /// names which structural check tripped — magic, version, CRC,
    /// kernel discriminator, grid dims, or rkyv archive validation.
    #[error("partial wire format rejected: {kind}")]
    PartialFormatMismatch {
        /// Which framing or structural check failed. See
        /// [`PartialFormatErrorKind`].
        kind: PartialFormatErrorKind,
    },

    /// One or more partials carry a `dataset_hash` that doesn't match
    /// the live dataset's. Means the partial was computed against a
    /// different GT than the receiving rank loaded — almost always a
    /// sampler / config bug; refusing protects the merge result from
    /// the head-rank's perspective. ADR-0031 §"Validation order" #6.
    #[error("partial dataset_hash mismatch: expected {expected:02x?}, got {actual:02x?}")]
    PartialDatasetMismatch {
        /// Receiving rank's `dataset_hash` (what the partial was
        /// expected to be computed against).
        expected: [u8; 32],
        /// Partial's declared `dataset_hash` (what was actually used).
        actual: [u8; 32],
    },

    /// One or more partials carry a `params_hash` that doesn't match
    /// the receiving rank's. Means the partial was produced with
    /// different `iou_thresholds` / `max_dets` / `use_cats` / etc. and
    /// the merged result would not equal a batch run. ADR-0031
    /// §"Validation order" #7.
    #[error("partial params_hash mismatch: expected {expected:02x?}, got {actual:02x?}")]
    PartialParamsMismatch {
        /// Receiving rank's `params_hash`.
        expected: [u8; 32],
        /// Partial's declared `params_hash`.
        actual: [u8; 32],
    },

    /// Two partials cover the same `image_id` — the disjoint-partition
    /// rule (ADR-0031 §"Axis D" D1) is violated. Almost always a
    /// `DistributedSampler` misconfiguration where two ranks evaluated
    /// the same image. The error names both rank ids and the colliding
    /// image so the user can fix their sampler.
    #[error("partials cover image_id={image_id} on both rank {rank_a} and rank {rank_b}")]
    PartialPartitionOverlap {
        /// Lower rank id involved in the collision (sorted for
        /// determinism — `min(a, b)`).
        rank_a: u32,
        /// Higher rank id involved in the collision.
        rank_b: u32,
        /// Image id that appeared in both partials' `seen_images`.
        image_id: i64,
    },

    /// Two strict-mode partials declare the same `rank_id`. ADR-0031
    /// §"Axis C" C2: strict-mode merge requires distinct rank ids so
    /// the future `(score, rank_id, local_position)` tiebreak gives a
    /// total order. Corrected mode tolerates collisions.
    #[error("partials share rank_id={rank_id} in strict mode")]
    PartialRankCollision {
        /// The duplicated rank id.
        rank_id: u32,
    },
}

/// Translate a leaf-crate [`PartialError`] into the equivalent
/// [`EvalError`] variant. Centralizes the variant-name mapping
/// (`Format` ↔ `PartialFormatMismatch` etc.) so call sites use `?` to
/// propagate.
impl From<PartialError> for EvalError {
    fn from(err: PartialError) -> Self {
        match err {
            PartialError::Format { kind } => EvalError::PartialFormatMismatch { kind },
            PartialError::DatasetMismatch { expected, actual } => {
                EvalError::PartialDatasetMismatch { expected, actual }
            }
            PartialError::ParamsMismatch { expected, actual } => {
                EvalError::PartialParamsMismatch { expected, actual }
            }
            PartialError::PartitionOverlap {
                rank_a,
                rank_b,
                image_id,
            } => EvalError::PartialPartitionOverlap {
                rank_a,
                rank_b,
                image_id,
            },
            PartialError::RankCollision { rank_id } => EvalError::PartialRankCollision { rank_id },
        }
    }
}
