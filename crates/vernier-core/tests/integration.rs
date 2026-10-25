//! Consolidated integration test suite for `vernier-core`.
//!
//! Combining integration test files into a single binary avoids compiling
//! and linking duplicate copies of large dependencies (`arrow`, `serde`,
//! `rkyv`, `rayon`, `proptest`), significantly reducing link times and
//! disk usage while preserving full test isolation under `cargo-nextest`.

#[path = "it/common/mod.rs"]
mod common;

#[path = "it/confusion_matrix.rs"]
mod confusion_matrix;

#[path = "it/fuzz_regressions.rs"]
mod fuzz_regressions;

#[path = "it/partition_overall_parity.rs"]
mod partition_overall_parity;

#[path = "it/tide_fp_iou_histogram.rs"]
mod tide_fp_iou_histogram;

#[path = "it/tide_oracle_parity.rs"]
mod tide_oracle_parity;
