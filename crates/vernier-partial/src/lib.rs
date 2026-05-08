//! Shared partial wire format and merge policy for vernier's
//! distributed evaluation surface (ADR-0031, generalized in
//! ADR-0032).
//!
//! `vernier-partial` is a leaf crate. The three paradigm crates
//! (`vernier-core`, `vernier-semantic`, `vernier-panoptic`) each
//! depend on this one and supply their own paradigm-specific wire
//! body type plus merge accumulator that embeds
//! [`merge::BaseMergeAccumulator`]. `vernier-partial` does not depend
//! on any of them — keeping the dep DAG flat is what makes the
//! cross-paradigm refactor possible without circular dependencies.
//!
//! ## What lives here
//!
//! - The wire envelope ([`envelope::WireEnvelopeHeader`]), magic +
//!   version constants ([`envelope::MAGIC`],
//!   [`envelope::FORMAT_VERSION`]), framing helpers
//!   ([`envelope::encode`], [`envelope::with_validated_envelope`]).
//! - The five typed errors mapped 1:1 to Python exceptions
//!   ([`error::PartialError`]).
//! - The [`traits::Partial`], [`traits::ParadigmKind`], and
//!   [`traits::PartialExpectation`] surface paradigm crates implement
//!   against.
//! - The paradigm-agnostic merge policy
//!   ([`merge::BaseMergeAccumulator`]).
//!
//! ## What does *not* live here
//!
//! - Paradigm-specific body types (instance per-image cells,
//!   semantic confusion matrix, panoptic per-category PqStats).
//!   Each paradigm owns its body archive and decode.
//! - Dataset / params hashing. Each paradigm hashes against its own
//!   canonical form. The 32-byte hashes carried in the envelope
//!   header are opaque to this crate.
//! - The streaming evaluator types themselves. `vernier-partial`
//!   knows nothing about evaluator state; it only validates the
//!   envelope and accumulates partition / rank-collision policy.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod envelope;
pub mod error;
pub mod merge;
pub mod traits;

// Each item lives at exactly one path — its home module. Adding a
// re-export here widens the headline; treat it as a deliberate
// decision, not a default for new pub items.
pub use envelope::{encode, with_validated_envelope, WireEnvelopeHeader, FORMAT_VERSION, MAGIC};
pub use error::{PartialError, PartialFormatErrorKind};
pub use merge::RankId;
