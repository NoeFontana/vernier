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

pub mod accumulate;
pub mod dataset;
pub mod error;
pub mod evaluate;
pub mod matching;
pub mod parity;
pub mod segmentation;
pub mod similarity;
pub mod summarize;

pub use accumulate::{accumulate, AccumulateParams, Accumulated, PerImageEval};
pub use dataset::{
    AnnId, Annotation, AnnotationIter, Bbox, CategoryId, CategoryMeta, CocoAnnotation, CocoDataset,
    CocoDetection, CocoDetections, CocoJson, DetectionInput, EvalDataset, ImageId, ImageMeta,
};
pub use error::EvalError;
pub use evaluate::{
    evaluate_bbox, evaluate_segm, evaluate_with, AreaRange, EvalGrid, EvalImageMeta, EvalKernel,
    EvaluateParams, COLLAPSED_CATEGORY_SENTINEL,
};
pub use matching::{match_image, MatchResult};
pub use parity::{
    iou_thresholds, recall_thresholds, ParityMode, IOU_BOUNDARY_EPS, OKS_AREA_EPS, PARITY_EPS,
};
pub use segmentation::{Segmentation, SegmentationRle, SegmentationRleCounts};
pub use similarity::{BboxAnn, BboxIou, SegmAnn, SegmIou, Similarity};
pub use summarize::{
    summarize_detection, summarize_with, AreaRng, MaxDetSelector, Metric, StatLine, StatRequest,
    Summary,
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
