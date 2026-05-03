//! FP-IoU histogram extractor smoke (ADR-0022 t_b ratification).
//!
//! Sister to `tide_oracle_parity.rs`. The histogram extractor reuses
//! the same bin-assignment machinery the oracle parity tests already
//! cover, so this file's job is narrow: confirm the surface contract
//! (parallel arrays, kernel marker, sanity counts) on a handful of
//! fixtures with known FP shape.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::PathBuf;

use vernier_core::tide::{
    compute_fp_iou_histogram_bbox, compute_fp_iou_histogram_segm, KernelMarker,
};
use vernier_core::{
    iou_thresholds, recall_thresholds, AreaRange, CocoDataset, CocoDetections, ParityMode,
    TideParams,
};

fn fixture_path(name: &str, file: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("../../tests/python/oracle/tide/fixtures");
    p.push(name);
    p.push(file);
    p
}

fn load_fixture(name: &str) -> (CocoDataset, CocoDetections) {
    let gt_bytes = std::fs::read(fixture_path(name, "gt.json"))
        .unwrap_or_else(|e| panic!("read gt.json for {name}: {e}"));
    let dt_bytes = std::fs::read(fixture_path(name, "dt.json"))
        .unwrap_or_else(|e| panic!("read dt.json for {name}: {e}"));
    let gt = CocoDataset::from_json_bytes(&gt_bytes).expect("parse gt");
    let dt = CocoDetections::from_json_bytes(&dt_bytes).expect("parse dt");
    (gt, dt)
}

fn make_params<'a>(area_ranges: &'a [AreaRange]) -> TideParams<'a> {
    TideParams {
        t_f: 0.5,
        t_b: 0.1,
        max_dets_per_image: 100,
        use_cats: true,
        iou_thresholds: iou_thresholds(),
        recall_thresholds: recall_thresholds(),
        area_ranges,
    }
}

#[test]
fn all_perfect_yields_zero_fps() {
    let (gt, dt) = load_fixture("all_perfect");
    let area_ranges = AreaRange::coco_default();
    let h = compute_fp_iou_histogram_bbox(&gt, &dt, make_params(&area_ranges), ParityMode::Strict)
        .unwrap();
    assert_eq!(h.kernel, KernelMarker::Bbox);
    assert!((h.t_f - 0.5).abs() < 1e-12);
    assert_eq!(h.n_fps, 0);
    assert_eq!(h.iou_same.len(), 0);
    assert_eq!(h.iou_cross.len(), 0);
}

#[test]
fn all_bkg_iou_same_is_zero_for_every_fp() {
    let (gt, dt) = load_fixture("all_bkg");
    let area_ranges = AreaRange::coco_default();
    let h = compute_fp_iou_histogram_bbox(&gt, &dt, make_params(&area_ranges), ParityMode::Strict)
        .unwrap();
    assert!(h.n_fps > 0, "all_bkg fixture should produce FPs");
    assert_eq!(h.iou_same.len(), h.n_fps);
    assert_eq!(h.iou_cross.len(), h.n_fps);
    for &v in &h.iou_same {
        assert!(v < 0.1, "expected Bkg-binned DTs' iou_same < t_b=0.1, got {v}");
    }
}

#[test]
fn all_loc_iou_same_in_loc_band() {
    let (gt, dt) = load_fixture("all_loc");
    let area_ranges = AreaRange::coco_default();
    let h = compute_fp_iou_histogram_bbox(&gt, &dt, make_params(&area_ranges), ParityMode::Strict)
        .unwrap();
    assert!(h.n_fps > 0);
    // The fixture is constructed so every FP is a Loc — same-class IoU
    // in [t_b, t_f). Allow boundary inclusivity.
    for &v in &h.iou_same {
        assert!(
            (0.1..0.5).contains(&v),
            "expected Loc-binned DTs' iou_same ∈ [0.1, 0.5), got {v}"
        );
    }
}

#[test]
fn segm_kernel_marker_is_segm() {
    let (gt, dt) = load_fixture("segm_all_perfect");
    let area_ranges = AreaRange::coco_default();
    let h = compute_fp_iou_histogram_segm(&gt, &dt, make_params(&area_ranges), ParityMode::Strict)
        .unwrap();
    assert_eq!(h.kernel, KernelMarker::Segm);
    assert_eq!(h.n_fps, 0);
}

#[test]
fn n_total_dts_at_least_n_fps() {
    let (gt, dt) = load_fixture("all_bkg");
    let area_ranges = AreaRange::coco_default();
    let h = compute_fp_iou_histogram_bbox(&gt, &dt, make_params(&area_ranges), ParityMode::Strict)
        .unwrap();
    assert!(h.n_total_dts >= h.n_fps);
}
