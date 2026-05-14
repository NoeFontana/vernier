//! Panoptic-quality (PQ) evaluation for vernier (ADR-0025).
//!
//! This crate is a sibling to `vernier-core`: both depend on
//! `vernier-mask` as a leaf, neither depends on the other. The
//! architectural firewall enforces the ADR-0005 invariant — PQ does
//! not share the AP fold (different matching rule, different data
//! model), so the AP-side `matching.rs` / `accumulate.rs` / `Similarity`
//! trait cannot be edited from here even by accident.
//!
//! The public Rust surface is the source of truth for vernier's PQ
//! semantics. The `vernier-ffi` crate exposes it to Python as
//! `vernier.PanopticEvaluator` and friends.
//!
//! See [ADR-0025](../../docs/adr/0025-panoptic-api.md) for the design
//! decisions that shaped this crate, and `tests/python/parity_panoptic/`
//! for the strict-mode parity harness against the vendored
//! [`cocodataset/panopticapi`] oracle.
//!
//! [`cocodataset/panopticapi`]: https://github.com/cocodataset/panopticapi

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod attribute;
pub mod boundary;
pub mod dataset;
pub mod decode;
pub mod distributed;
pub mod error;
pub mod kernel;
pub mod parity;
pub mod stream;
pub mod summarize;
pub mod tables;

// Each item lives at exactly one path — its home module. Adding a
// re-export here widens the headline; treat it as a deliberate
// decision, not a default for new pub items.
pub use boundary::{BoundaryConfig, BoundaryScratch, BOUNDARY_PANOPTIC_DILATION_RATIO_DEFAULT};
pub use dataset::{PanopticDataset, PanopticPredictions};
pub use error::PanopticError;
pub use parity::{ParityMode, BOUNDARY_PANOPTIC_ORACLE_COMMIT_SHA};
pub use summarize::{
    evaluate, evaluate_with_options, ClassPanopticStats, EvaluateOptions, GroupPanopticStats,
    PanopticSummary, SummarizeOptions,
};

/// Library version string. Useful for parity tracing in fixtures and
/// for debugging mismatches between Rust and Python sides of the FFI
/// boundary. Mirrors `vernier_core::VERSION` in shape.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_set() {
        assert!(!VERSION.is_empty());
    }
}
