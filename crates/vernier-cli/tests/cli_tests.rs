//! Consolidated integration test suite for `vernier-cli`.
//!
//! Combining CLI integration tests into a single binary avoids linking duplicate
//! copies of heavy dependencies, significantly speeding up build times and reducing
//! target directory bloat while retaining granular test reporting in `cargo-nextest`.

#[path = "it/aggregate.rs"]
mod aggregate;

#[path = "it/eval.rs"]
mod eval;

#[path = "it/eval_manifest.rs"]
mod eval_manifest;
