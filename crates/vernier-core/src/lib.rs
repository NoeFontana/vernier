//! Pure-Rust core for vernier.
//!
//! This crate has **no Python dependencies** and is usable directly from Rust
//! binaries, CLI tools, and embedded contexts.
//!
//! By design, the public API of this crate is the source of truth for vernier's
//! evaluation semantics. The `vernier-ffi` crate is a thin data-conversion
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
pub mod calibration;
pub mod dataset;
pub mod distributed;
pub mod error;
pub mod evaluate;
pub mod evaluate_parallel;
pub mod lrp;
pub mod lvis_parity;
pub mod manifest;
pub mod manifest_csv;
pub mod matching;
pub mod parity;
pub mod partition;
pub mod segmentation;
pub mod similarity;
pub mod stream;
pub mod summarize;
pub mod tables;
pub mod tide;

// Each item lives at exactly one path — its home module. Adding a
// re-export here widens the headline; treat it as a deliberate
// decision keyed to the README minimal-usage example, not a default
// for new pub items.
pub use accumulate::Accumulated;
pub use dataset::{CocoDataset, CocoDetections, EvalDataset};
pub use error::EvalError;
pub use evaluate::{
    evaluate_bbox, evaluate_boundary, evaluate_keypoints, evaluate_segm, evaluate_with, AreaRange,
    EvalGrid, EvaluateParams,
};
pub use evaluate_parallel::{
    evaluate_bbox_parallel, evaluate_boundary_cached_parallel, evaluate_boundary_parallel,
    evaluate_keypoints_parallel, evaluate_segm_cached_parallel, evaluate_segm_parallel,
    evaluate_with_parallel,
};
pub use parity::ParityMode;
pub use summarize::Summary;

/// Library version string. Useful for parity tracing in fixtures and for
/// debugging mismatches between Rust and Python sides of the FFI boundary.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Stage-0 instrumentation hook for the bbox-IoU optimization plan.
///
/// Writes the `(kind, g, d, wall_ns)` records accumulated across every
/// `BboxIou::compute` and `BboxIou::compute_overlap_mask` call to
/// `path` as CSV (header `kind,g,d,wall_ns`; `kind` is the variant
/// label `FullIou` or `OverlapMask`), then clears the in-process
/// buffer. Returns the number of records written.
///
/// Only present when the crate is compiled with `--features
/// bench-histogram`. Bench harness builds set this; the shipped wheel
/// never does, so production runs carry zero recording overhead.
#[cfg(feature = "bench-histogram")]
pub fn dump_bbox_iou_histogram_csv(path: &std::path::Path) -> std::io::Result<usize> {
    similarity::bbox::histogram::dump_csv(path)
}

/// `(par_iter_ns, serial_post_ns, n_calls)` for [`evaluate_with_parallel`]
/// since last read, then resets. Gated on `bench-timings`.
#[cfg(feature = "bench-timings")]
pub fn read_and_reset_evaluate_parallel_timings() -> (u64, u64, u64) {
    evaluate_parallel::timings::read_and_reset()
}

/// `build_gt_anns` + `build_dt_anns` call count since last read,
/// then resets. Gated on `bench-timings`.
#[cfg(feature = "bench-timings")]
pub fn read_and_reset_build_anns_count() -> u64 {
    evaluate::build_anns_count::read_and_reset()
}

/// `(gt_parse_ns, gt_from_parts_ns, dt_parse_ns, dt_from_inputs_ns)`
/// for the COCO loaders since last read, then resets. Gated on `bench-timings`.
#[cfg(feature = "bench-timings")]
pub fn read_and_reset_dataset_timings() -> (u64, u64, u64, u64) {
    dataset::dataset_timings::read_and_reset()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_set() {
        assert!(!VERSION.is_empty());
    }
}
