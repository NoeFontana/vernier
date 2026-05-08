//! Semantic-segmentation evaluation for vernier (ADR-0028).
//!
//! This crate is a sibling to [`vernier-core`] (instance evaluation,
//! AP fold) and [`vernier-panoptic`] (panoptic-quality). Unlike
//! `vernier-panoptic`, which is structurally independent of
//! `vernier-core` per ADR-0025, this crate **depends on
//! `vernier-core`** for [`vernier_core::parity::ParityMode`], the
//! [`Breakdown`](vernier_core::breakdown::Breakdown) axis, the
//! streaming-evaluator interface (ADR-0013), and the result-tables
//! interface (ADR-0019). The reason is concrete reuse: those
//! abstractions are useful enough for semantic eval that duplicating
//! them would be silly. ADR-0028 §"Workspace and dependency direction"
//! ratifies the asymmetry.
//!
//! The architectural firewall enforced by ADR-0005 still holds: this
//! crate has no edge to `matching.rs`, `accumulate.rs`, or the
//! `Similarity` trait — the AP fold is unreachable from semantic-mIoU
//! code by construction. The kernel here is a per-image confusion
//! matrix accumulator, not a per-detection matching loop.
//!
//! The public Rust surface is the source of truth for vernier's
//! semantic-mIoU semantics. The [`vernier-ffi`] crate exposes it to
//! Python as `vernier.semantic.Evaluator` and friends.
//!
//! See [ADR-0028](../../docs/adr/0028-sem-seg.md) for the design
//! decisions that shaped this crate, and the
//! [sem-seg quirks survey](../../docs/engineering/sem-seg-quirks.md)
//! for the `(quirk_id, oracle) → mode` disposition table this crate
//! implements against. The vendored oracles
//! ([`open-mmlab/mmsegmentation`], [`mcordts/cityscapesScripts`], and
//! the Pascal VOC / ADE20K reference scripts) ship under
//! `tests/python/parity_semantic/oracle/` in subsequent PRs.
//!
//! [`open-mmlab/mmsegmentation`]: https://github.com/open-mmlab/mmsegmentation
//! [`mcordts/cityscapesScripts`]: https://github.com/mcordts/cityscapesScripts

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod distributed;
pub mod error;
pub mod kernel;
pub mod parity;
pub mod stream;
pub mod summarize;

// Each item lives at exactly one path — its home module.
// `ConfusionMatrix` is at the root as a first-class output of the
// semantic paradigm (ADR-0028 §F1), on equal footing with
// `SemanticSummary`.
pub use error::SemanticError;
pub use kernel::ConfusionMatrix;
pub use parity::ParityMode;
pub use stream::StreamingSemanticEvaluator;
pub use summarize::{summarize, ClassSemanticStats, SemanticSummary};

/// Library version string. Useful for parity tracing in fixtures and
/// for debugging mismatches between Rust and Python sides of the FFI
/// boundary. Mirrors [`vernier_core::VERSION`] and
/// [`vernier_panoptic::VERSION`] in shape.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_set() {
        assert!(!VERSION.is_empty());
    }
}
