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
pub mod boundary_parity;
pub mod breakdown;
pub mod dataset;
pub mod error;
pub mod evaluate;
pub mod matching;
pub mod parity;
pub mod segmentation;
pub mod similarity;
pub mod stream;
pub mod summarize;
pub mod tables;
pub mod tide;

pub use accumulate::{accumulate, sort_max_dets, AccumulateParams, Accumulated, PerImageEval};
pub use boundary_parity::{
    BOUNDARY_DILATION_RATIO_DEFAULT, BOUNDARY_PARITY_EPS, ORACLE_COMMIT_SHA, ORACLE_OPENCV_PIN,
};
pub use breakdown::{Breakdown, Bucket};
pub use dataset::{
    AnnId, Annotation, AnnotationIter, Bbox, CategoryId, CategoryMeta, CocoAnnotation, CocoDataset,
    CocoDetection, CocoDetections, CocoJson, DetectionInput, EvalDataset, ImageId, ImageMeta,
};
pub use error::EvalError;
pub use evaluate::{
    evaluate_bbox, evaluate_boundary, evaluate_boundary_cached, evaluate_keypoints, evaluate_segm,
    evaluate_segm_cached, evaluate_with, evaluate_with_retention, AreaRange, EvalGrid,
    EvalImageMeta, EvalKernel, EvaluateParams, OwnedEvaluateParams, COLLAPSED_CATEGORY_SENTINEL,
};
pub use matching::{match_image, MatchResult};
pub use parity::{iou_thresholds, recall_thresholds, ParityMode, IOU_BOUNDARY_EPS, PARITY_EPS};
pub use segmentation::{Segmentation, SegmentationRle, SegmentationRleCounts};
pub use similarity::{
    BboxAnn, BboxIou, BoundaryGtCache, BoundaryIou, OksAnn, OksSimilarity, SegmAnn, SegmGtCache,
    SegmIou, Similarity, COCO_PERSON_SIGMAS,
};
pub use stream::{
    EvalGridMeta, MemoryBudget, ParsedDetections, PerImageEvalStore, StreamingEvaluator,
    UpdateReport,
};
pub use summarize::{
    summarize_detection, summarize_with, AreaRng, MaxDetSelector, Metric, StatLine, StatRequest,
    Summary,
};
pub use tables::{
    aggregate_per_class_support, build_per_class, build_per_detection, build_per_image,
    build_per_pair, build_tables, BboxColumns, CrossClassIous, MatchStatus, PerClassSupport,
    PerClassTable, PerDetectionTable, PerImageTable, PerPairTable, RetainedIous, Tables,
    TablesConfig, TablesRequest,
};
pub use tide::{
    apply_fix, assign_bins, compute_cross_class_ious, error_decomposition_bbox, BinAssignment,
    DtBin, DtBinLabel, FixKind, TideConfig, TideErrorBin, TideParams, TideReport,
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
