//! Panoptic-quality (PQ) evaluation for vernier (ADR-0025).
//!
//! This crate is a sibling to [`vernier-core`]: both depend on
//! [`vernier-mask`] as a leaf, neither depends on the other. The
//! architectural firewall enforces the ADR-0005 invariant — PQ does
//! not share the AP fold (different matching rule, different data
//! model), so the AP-side `matching.rs` / `accumulate.rs` / `Similarity`
//! trait cannot be edited from here even by accident.
//!
//! The public Rust surface is the source of truth for vernier's PQ
//! semantics. The [`vernier-ffi`] crate exposes it to Python as
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

pub mod error;
pub mod parity;

pub use error::PanopticError;
pub use parity::{
    ParityMode, ORACLE_COMMIT_SHA, ORACLE_PILLOW_PIN, PANOPTIC_IOU_THRESHOLD, PANOPTIC_OFFSET,
    PANOPTIC_PARITY_EPS, PANOPTIC_VOID,
};

/// Library version string. Useful for parity tracing in fixtures and
/// for debugging mismatches between Rust and Python sides of the FFI
/// boundary. Mirrors [`vernier_core::VERSION`] in shape.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_set() {
        assert!(!VERSION.is_empty());
    }
}
