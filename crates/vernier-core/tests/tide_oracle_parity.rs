//! TIDE Rust ↔ numpy-oracle parity (bbox, single-`t_f`).
//!
//! Per ADR-0021, the numpy oracle at `tests/python/oracle/tide/oracle.py`
//! is the spec for `vernier::error_decomposition_bbox`. This integration
//! test mirrors the hand-computed assertions from
//! `tests/python/oracle/tide/test_oracle.py` against the Rust
//! implementation: every per-bin ΔmAP and the all-FP-removed sanity
//! delta must match the values pinned in the oracle test file within
//! `1e-9`.
//!
//! The Python parity test (Agent B's deliverable) closes the other half
//! of the contract: it runs the oracle and Rust on the same fixtures
//! end-to-end and asserts they agree. This file is the Rust-side
//! replica that catches a regression without needing the Python harness
//! to be set up.

// Integration tests live outside `lib.rs`, so the workspace lints
// (which deny `panic`/`unwrap`/`expect` in production) need a per-file
// allow. Tests are explicitly exempted in `lib.rs`'s
// `#[cfg_attr(test, allow(...))]` for inline tests; the same exemption
// here keeps the integration test surface honest.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::PathBuf;

use vernier_core::{
    error_decomposition_bbox, iou_thresholds, recall_thresholds, AreaRange, CocoDataset,
    CocoDetections, ParityMode, TideErrorBin, TideParams,
};

const PARITY_TOL: f64 = 1e-9;

fn fixture_path(name: &str, file: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // tests live at `crates/vernier-core/tests/`; the oracle fixtures are
    // at the workspace root under `tests/python/oracle/tide/fixtures/`.
    p.push("../../tests/python/oracle/tide/fixtures");
    p.push(name);
    p.push(file);
    p
}

fn load_fixture(name: &str) -> (CocoDataset, CocoDetections) {
    let gt_bytes = std::fs::read(fixture_path(name, "gt.json"))
        .unwrap_or_else(|e| panic!("failed to read gt.json for fixture {name}: {e}"));
    let dt_bytes = std::fs::read(fixture_path(name, "dt.json"))
        .unwrap_or_else(|e| panic!("failed to read dt.json for fixture {name}: {e}"));
    let gt = CocoDataset::from_json_bytes(&gt_bytes)
        .unwrap_or_else(|e| panic!("failed to parse gt.json for fixture {name}: {e}"));
    let dt = CocoDetections::from_json_bytes(&dt_bytes)
        .unwrap_or_else(|e| panic!("failed to parse dt.json for fixture {name}: {e}"));
    (gt, dt)
}

fn run_tide(name: &str) -> vernier_core::TideReport {
    let (gt, dt) = load_fixture(name);
    let area_ranges = AreaRange::coco_default();
    let params = TideParams {
        t_f: 0.5,
        t_b: 0.1,
        max_dets_per_image: 100,
        use_cats: true,
        iou_thresholds: iou_thresholds(),
        recall_thresholds: recall_thresholds(),
        area_ranges: &area_ranges,
    };
    error_decomposition_bbox(&gt, &dt, params, ParityMode::Strict)
        .unwrap_or_else(|e| panic!("error_decomposition_bbox failed on fixture {name}: {e}"))
}

fn delta_or_zero(report: &vernier_core::TideReport, bin: TideErrorBin) -> f64 {
    report.delta_per_bin.get(&bin).copied().unwrap_or(0.0)
}

fn assert_close(actual: f64, expected: f64, label: &str) {
    let diff = (actual - expected).abs();
    assert!(
        diff < PARITY_TOL,
        "{label}: expected {expected} got {actual} (diff {diff} > tol {PARITY_TOL})"
    );
}

#[test]
fn all_perfect_baseline_one_no_deltas() {
    // From `test_oracle.py::test_all_perfect_baseline_one_and_no_deltas`:
    // baseline = 1.0, every delta = 0.
    let r = run_tide("all_perfect");
    assert_close(r.baseline_map, 1.0, "baseline_map");
    for bin in [
        TideErrorBin::Cls,
        TideErrorBin::Loc,
        TideErrorBin::Both,
        TideErrorBin::Dupe,
        TideErrorBin::Bkg,
        TideErrorBin::Missed,
    ] {
        assert_close(delta_or_zero(&r, bin), 0.0, &format!("delta[{bin:?}]"));
    }
    assert_close(r.delta_all_fp, 0.0, "delta_all_fp");
}

#[test]
fn all_bkg_isolates_bkg_bin() {
    // From `test_oracle.py::test_all_bkg_isolates_bkg_bin`:
    // baseline = 0.5, delta_bkg = 0.5, every other = 0.
    let r = run_tide("all_bkg");
    assert_close(r.baseline_map, 0.5, "baseline_map");
    assert_close(delta_or_zero(&r, TideErrorBin::Bkg), 0.5, "delta[bkg]");
    for bin in [
        TideErrorBin::Cls,
        TideErrorBin::Loc,
        TideErrorBin::Both,
        TideErrorBin::Dupe,
        TideErrorBin::Missed,
    ] {
        assert_close(delta_or_zero(&r, bin), 0.0, &format!("delta[{bin:?}]"));
    }
    assert_close(r.delta_all_fp, 0.5, "delta_all_fp");
}

#[test]
fn all_cls_isolates_cls_bin() {
    // From `test_oracle.py::test_all_cls_isolates_cls_bin`:
    // baseline = 0, delta_cls = 1.0, others = 0,
    // delta_all_fp_removed = 0 (loose-bound, removing FPs leaves zero
    // detections).
    let r = run_tide("all_cls");
    assert_close(r.baseline_map, 0.0, "baseline_map");
    assert_close(delta_or_zero(&r, TideErrorBin::Cls), 1.0, "delta[cls]");
    for bin in [
        TideErrorBin::Loc,
        TideErrorBin::Both,
        TideErrorBin::Dupe,
        TideErrorBin::Bkg,
        TideErrorBin::Missed,
    ] {
        assert_close(delta_or_zero(&r, bin), 0.0, &format!("delta[{bin:?}]"));
    }
    assert_close(r.delta_all_fp, 0.0, "delta_all_fp");
}

#[test]
fn all_loc_isolates_loc_bin() {
    // From `test_oracle.py::test_all_loc_isolates_loc_bin`:
    // baseline = 0, delta_loc = 1.0, others = 0, delta_all_fp = 0.
    let r = run_tide("all_loc");
    assert_close(r.baseline_map, 0.0, "baseline_map");
    assert_close(delta_or_zero(&r, TideErrorBin::Loc), 1.0, "delta[loc]");
    for bin in [
        TideErrorBin::Cls,
        TideErrorBin::Both,
        TideErrorBin::Dupe,
        TideErrorBin::Bkg,
        TideErrorBin::Missed,
    ] {
        assert_close(delta_or_zero(&r, bin), 0.0, &format!("delta[{bin:?}]"));
    }
    assert_close(r.delta_all_fp, 0.0, "delta_all_fp");
}

#[test]
fn all_dupe_isolates_dupe_bin() {
    // From `test_oracle.py::test_all_dupe_isolates_dupe_bin`:
    // baseline = 76/101, delta_dupe = 25/101, others = 0,
    // delta_all_fp_removed = delta_dupe.
    let r = run_tide("all_dupe");
    let expected_baseline = 76.0_f64 / 101.0_f64;
    let expected_dupe = 25.0_f64 / 101.0_f64;
    assert_close(r.baseline_map, expected_baseline, "baseline_map");
    assert_close(
        delta_or_zero(&r, TideErrorBin::Dupe),
        expected_dupe,
        "delta[dupe]",
    );
    for bin in [
        TideErrorBin::Cls,
        TideErrorBin::Loc,
        TideErrorBin::Both,
        TideErrorBin::Bkg,
        TideErrorBin::Missed,
    ] {
        assert_close(delta_or_zero(&r, bin), 0.0, &format!("delta[{bin:?}]"));
    }
    assert_close(r.delta_all_fp, expected_dupe, "delta_all_fp");
}

#[test]
fn with_ignore_does_not_bin_crowd_matched_dts() {
    // From `test_oracle.py::test_with_ignore_does_not_bin_crowd_matched_dts`:
    // baseline = 0.5, delta_bkg = 0.5, others = 0, delta_all_fp = 0.5.
    let r = run_tide("with_ignore");
    assert_close(r.baseline_map, 0.5, "baseline_map");
    assert_close(delta_or_zero(&r, TideErrorBin::Bkg), 0.5, "delta[bkg]");
    for bin in [
        TideErrorBin::Cls,
        TideErrorBin::Loc,
        TideErrorBin::Both,
        TideErrorBin::Dupe,
        TideErrorBin::Missed,
    ] {
        assert_close(delta_or_zero(&r, bin), 0.0, &format!("delta[{bin:?}]"));
    }
    assert_close(r.delta_all_fp, 0.5, "delta_all_fp");
}

#[test]
fn loc_vs_both_priority_loc_wins_when_same_class_gt_is_closer() {
    // From `test_oracle.py::test_loc_vs_both_priority_loc_wins_...`:
    // baseline = 0, delta_loc = 0.5, others = 0, delta_all_fp = 0.
    let r = run_tide("loc_vs_both_priority");
    assert_close(r.baseline_map, 0.0, "baseline_map");
    assert_close(delta_or_zero(&r, TideErrorBin::Loc), 0.5, "delta[loc]");
    for bin in [
        TideErrorBin::Cls,
        TideErrorBin::Both,
        TideErrorBin::Dupe,
        TideErrorBin::Bkg,
        TideErrorBin::Missed,
    ] {
        assert_close(delta_or_zero(&r, bin), 0.0, &format!("delta[{bin:?}]"));
    }
    assert_close(r.delta_all_fp, 0.0, "delta_all_fp");
}

#[test]
fn config_carries_resolved_thresholds() {
    let r = run_tide("all_perfect");
    assert_eq!(r.config.t_f, 0.5);
    assert_eq!(r.config.t_b, 0.1);
    assert_eq!(r.config.kernel, "bbox");
    assert!(r.config.cross_class_topk.is_none());
}
