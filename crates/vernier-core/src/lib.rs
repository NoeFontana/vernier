//! Pure-Rust core for vernier.
//!
//! This crate has **no Python dependencies** and is usable directly from Rust
//! binaries, CLI tools, and embedded contexts (e.g., ROS2 nodes).
//!
//! By design, the public API of this crate is the source of truth for vernier's
//! evaluation semantics. The [`vernier-ffi`] crate is a thin data-conversion
//! layer over this one; if you find yourself adding logic to `vernier-ffi`
//! rather than here, that's a code smell worth resolving.
//!
//! See the project's `docs/explanation/` for the architecture overview and
//! `docs/adr/` for the design decisions that shaped this crate.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod dataset;
pub mod error;
pub mod parity;

pub use dataset::{
    AnnId, Annotation, AnnotationIter, Bbox, CategoryId, CategoryMeta, CocoAnnotation, CocoDataset,
    CocoJson, EvalDataset, ImageId, ImageMeta,
};
pub use error::EvalError;
pub use parity::{
    iou_thresholds, recall_thresholds, ParityMode, IOU_BOUNDARY_EPS, OKS_AREA_EPS, PARITY_EPS,
};

/// Library version string. Useful for parity tracing in fixtures and for
/// debugging mismatches between Rust and Python sides of the FFI boundary.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_set() {
        assert!(!VERSION.is_empty());
    }
}
