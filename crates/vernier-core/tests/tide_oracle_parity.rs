//! TIDE Rust ↔ numpy-oracle parity (bbox, single-`t_f`).
//!
//! Per ADR-0021, the numpy oracle at `tests/python/oracle/tide/oracle.py`
//! is the spec for `vernier::error_decomposition_bbox`. This integration
//! test loads per-fixture expected outputs from
//! `tests/python/oracle/tide/expected/<name>.json` — the same JSON
//! `tests/python/oracle/tide/test_oracle.py` reads — and asserts the
//! Rust implementation matches those values within `1e-9` ΔmAP. Both
//! sides reading the same file is what keeps Rust and Python from
//! drifting; the *why* behind each pinned value lives in the Python
//! test docstrings (re-derive from there, not from the JSON).
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

use std::collections::HashMap;

use serde::Deserialize;
use vernier_core::{
    error_decomposition_bbox, error_decomposition_segm, iou_thresholds, recall_thresholds,
    AreaRange, CocoDataset, CocoDetections, ParityMode, TideErrorBin, TideParams, TideReport,
};

mod common;
use common::{expected_path, fixture_path};

const PARITY_TOL: f64 = 1e-9;

#[derive(Deserialize)]
struct ExpectedReport {
    baseline_map: f64,
    deltas: HashMap<String, f64>,
    // The Python `error_decomposition` returns this under
    // `delta_all_fp_removed`; the JSON key matches the Python public API.
    // The Rust `TideReport` struct (`vernier_core`) shortens it to
    // `delta_all_fp`; we keep that field name local and rename only on
    // the wire so neither side has to learn the other's spelling.
    #[serde(rename = "delta_all_fp_removed")]
    delta_all_fp: f64,
}

fn load_expected(name: &str) -> ExpectedReport {
    let bytes = std::fs::read(expected_path(name))
        .unwrap_or_else(|e| panic!("failed to read expected/{name}.json: {e}"));
    serde_json::from_slice(&bytes)
        .unwrap_or_else(|e| panic!("failed to parse expected/{name}.json: {e}"))
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

fn run_tide(name: &str) -> TideReport {
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

/// Run the segm TIDE wrapper on a fixture. Mirrors [`run_tide`] but
/// pins the [`vernier_core::similarity::SegmIou`] kernel via the
/// `error_decomposition_segm` entry point. Fixtures need to carry
/// `segmentation` on every GT (and either `segmentation` or — under
/// strict parity, via the J2 path — a bbox-only DT entry).
fn run_tide_segm(name: &str) -> TideReport {
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
    error_decomposition_segm(&gt, &dt, params, ParityMode::Strict)
        .unwrap_or_else(|e| panic!("error_decomposition_segm failed on fixture {name}: {e}"))
}

fn delta_or_zero(report: &TideReport, bin: TideErrorBin) -> f64 {
    report.delta_per_bin.get(&bin).copied().unwrap_or(0.0)
}

fn bin_key(bin: TideErrorBin) -> &'static str {
    match bin {
        TideErrorBin::Cls => "cls",
        TideErrorBin::Loc => "loc",
        TideErrorBin::Both => "both",
        TideErrorBin::Dupe => "dupe",
        TideErrorBin::Bkg => "bkg",
        TideErrorBin::Missed => "missed",
    }
}

fn assert_close(actual: f64, expected: f64, label: &str) {
    let diff = (actual - expected).abs();
    assert!(
        diff < PARITY_TOL,
        "{label}: expected {expected} got {actual} (diff {diff} > tol {PARITY_TOL})"
    );
}

/// Assert the report matches every pinned value in `expected`: the
/// baseline mAP, every per-bin Δ, and the all-FPs-removed sanity
/// delta. Bins absent from `expected.deltas` are treated as zero so
/// the JSON only needs to enumerate the bins a fixture actually pins.
fn assert_report_matches(actual: &TideReport, expected: &ExpectedReport) {
    assert_close(actual.baseline_map, expected.baseline_map, "baseline_map");
    for bin in [
        TideErrorBin::Cls,
        TideErrorBin::Loc,
        TideErrorBin::Both,
        TideErrorBin::Dupe,
        TideErrorBin::Bkg,
        TideErrorBin::Missed,
    ] {
        let key = bin_key(bin);
        let want = expected.deltas.get(key).copied().unwrap_or(0.0);
        assert_close(delta_or_zero(actual, bin), want, &format!("delta[{key}]"));
    }
    assert_close(actual.delta_all_fp, expected.delta_all_fp, "delta_all_fp");
}

#[test]
fn all_perfect_baseline_one_no_deltas() {
    assert_report_matches(&run_tide("all_perfect"), &load_expected("all_perfect"));
}

#[test]
fn all_bkg_isolates_bkg_bin() {
    assert_report_matches(&run_tide("all_bkg"), &load_expected("all_bkg"));
}

#[test]
fn all_cls_isolates_cls_bin() {
    assert_report_matches(&run_tide("all_cls"), &load_expected("all_cls"));
}

#[test]
fn all_loc_isolates_loc_bin() {
    assert_report_matches(&run_tide("all_loc"), &load_expected("all_loc"));
}

#[test]
fn all_dupe_isolates_dupe_bin() {
    assert_report_matches(&run_tide("all_dupe"), &load_expected("all_dupe"));
}

#[test]
fn with_ignore_does_not_bin_crowd_matched_dts() {
    assert_report_matches(&run_tide("with_ignore"), &load_expected("with_ignore"));
}

#[test]
fn loc_vs_both_priority_loc_wins_when_same_class_gt_is_closer() {
    assert_report_matches(
        &run_tide("loc_vs_both_priority"),
        &load_expected("loc_vs_both_priority"),
    );
}

#[test]
fn config_carries_resolved_thresholds() {
    let r = run_tide("all_perfect");
    assert_eq!(r.config.t_f, 0.5);
    assert_eq!(r.config.t_b, 0.1);
    assert_eq!(r.config.kernel, vernier_core::KernelMarker::Bbox);
    assert!(r.config.cross_class_topk.is_none());
}

// -- segm kernel parity tests (Week 3) -----------------------------------
//
// Each segm fixture's expected ΔmAP is derived in the matching
// `tests/python/oracle/tide/test_oracle.py::test_segm_*` docstring. The
// rectangle polygons used in the segm fixtures rasterize to exactly the
// same pixel set as the bbox covers — both `Rle::from_polygon`
// (Rust side) and `_rasterize_polygon_axis_aligned` (oracle side) take
// integer-aligned vertex coords and emit identical masks — so the IoU
// values feeding the matching engine match the bbox case bit-for-bit.

#[test]
fn segm_all_perfect_baseline_one_no_deltas() {
    assert_report_matches(
        &run_tide_segm("segm_all_perfect"),
        &load_expected("segm_all_perfect"),
    );
}

#[test]
fn segm_all_loc_isolates_loc_bin() {
    assert_report_matches(
        &run_tide_segm("segm_all_loc"),
        &load_expected("segm_all_loc"),
    );
}

#[test]
fn segm_all_cls_isolates_cls_bin() {
    assert_report_matches(
        &run_tide_segm("segm_all_cls"),
        &load_expected("segm_all_cls"),
    );
}

#[test]
fn segm_config_carries_resolved_thresholds() {
    let r = run_tide_segm("segm_all_perfect");
    assert_eq!(r.config.t_f, 0.5);
    assert_eq!(r.config.t_b, 0.1);
    assert_eq!(r.config.kernel, vernier_core::KernelMarker::Segm);
    assert!(r.config.cross_class_topk.is_none());
}
