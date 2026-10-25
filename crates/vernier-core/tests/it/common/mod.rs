//! Shared helpers for integration tests in `crates/vernier-core/tests/`.
//!
//! Cargo treats top-level files in `tests/` as separate binaries, so
//! per-test boilerplate (path resolution, fixture loading) duplicates
//! by default. This module collects the helpers that have grown copies
//! across `tide_oracle_parity.rs`, `tide_fp_iou_histogram.rs`, and
//! `confusion_matrix.rs`, all of which read fixtures from the Python
//! tree at `tests/python/oracle/tide/`. Renaming or relocating that
//! tree breaks every consumer here in lockstep — see
//! `tests/python/oracle/tide/fixtures/README.md` for the signpost.
//!
//! `mod common;` lives in each test binary that needs these helpers.
//! The `dead_code` allow is needed because not every binary uses every
//! helper (cargo lints unused mod items per binary).

#![allow(dead_code)]

use std::path::PathBuf;

/// Resolve a fixture file path under `tests/python/oracle/tide/fixtures/<name>/<file>`.
///
/// Tests live at `crates/vernier-core/tests/`; the oracle fixtures
/// are at the workspace root, two directories up.
pub(crate) fn fixture_path(name: &str, file: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("../../tests/python/oracle/tide/fixtures");
    p.push(name);
    p.push(file);
    p
}

/// Resolve an expected-output path under `tests/python/oracle/tide/expected/<name>.json`.
///
/// Same JSON files are consumed by `tests/python/oracle/tide/test_oracle.py`;
/// see that file's `_load_expected` helper for the Python side.
pub(crate) fn expected_path(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("../../tests/python/oracle/tide/expected");
    p.push(format!("{name}.json"));
    p
}
