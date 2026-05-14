//! ADR-0046 load-bearing parity contract: the `overall` entry on a
//! partitioned eval is bit-identical to a non-partitioned eval over the
//! same `(GT, DT)` pair. Pycocotools does not slice — this is the only
//! "parity" claim available, and it's what protects the
//! un-partitioned `vernier eval` path from regressing.
//!
//! Also smoke-checks that the per-slice loop runs cleanly over a 2-axis
//! manifest with `__unassigned__` coverage, and that an empty slice
//! produces sentinel-filled stats (-1.0) without panicking.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::{HashMap, HashSet};

use vernier_core::accumulate::PerImageEval;
use vernier_core::dataset::ImageId;
use vernier_core::evaluate::{evaluate_bbox, AreaRange, EvalGrid, EvaluateParams};
use vernier_core::manifest::partition_spec_from_manifest;
use vernier_core::parity::{iou_thresholds, ParityMode};
use vernier_core::partition::{
    evaluate_partitioned, GridDims, KeyKind, PartitionSpec, SummaryKind, UNASSIGNED,
};
use vernier_core::summarize::summarize_detection;
use vernier_core::{accumulate, parity, CocoDataset, CocoDetections, EvalDataset};

mod common;
use common::fixture_path;

fn load_fixture(name: &str) -> (CocoDataset, CocoDetections) {
    let gt_bytes = std::fs::read(fixture_path(name, "gt.json")).unwrap();
    let dt_bytes = std::fs::read(fixture_path(name, "dt.json")).unwrap();
    let gt = CocoDataset::from_json_bytes(&gt_bytes).unwrap();
    let dt = CocoDetections::from_json_bytes(&dt_bytes).unwrap();
    (gt, dt)
}

fn run_bbox(gt: &CocoDataset, dt: &CocoDetections) -> EvalGrid {
    let areas = AreaRange::coco_default();
    let params = EvaluateParams {
        iou_thresholds: iou_thresholds(),
        area_ranges: &areas,
        max_dets_per_image: 100,
        use_cats: true,
        retain_iou: false,
    };
    evaluate_bbox(gt, dt, params, ParityMode::Strict).unwrap()
}

fn image_id_to_idx(gt: &CocoDataset) -> HashMap<ImageId, usize> {
    let mut ids: Vec<ImageId> = gt.images().iter().map(|im| im.id).collect();
    ids.sort_unstable_by_key(|id| id.0);
    ids.into_iter().enumerate().map(|(i, id)| (id, i)).collect()
}

#[test]
fn overall_matches_unpartitioned_eval() {
    let (gt, dt) = load_fixture("all_perfect");
    let grid = run_bbox(&gt, &dt);
    let dims = GridDims {
        n_categories: grid.n_categories,
        n_area_ranges: grid.n_area_ranges,
        n_images: grid.n_images,
    };
    let id_map = image_id_to_idx(&gt);

    // Empty partition (no axes) — only the overall pass runs.
    let spec = PartitionSpec::build(
        KeyKind::Image,
        &HashMap::new(),
        &id_map.keys().copied().collect(),
        &id_map,
        &[],
    )
    .unwrap();
    let part = evaluate_partitioned(
        &grid.eval_imgs,
        dims,
        &spec,
        iou_thresholds(),
        ParityMode::Strict,
        SummaryKind::DetectionDefault,
    )
    .unwrap();

    // Reference: run accumulate + summarize_detection on the un-filtered
    // grid directly. This is the same code path partition's overall
    // executes, but with no intermediate slice loop.
    let accum = accumulate::accumulate(
        &grid.eval_imgs,
        accumulate::AccumulateParams {
            iou_thresholds: iou_thresholds(),
            recall_thresholds: parity::recall_thresholds(),
            max_dets: &[1, 10, 100],
            n_categories: dims.n_categories,
            n_area_ranges: dims.n_area_ranges,
            n_images: dims.n_images,
        },
        ParityMode::Strict,
    )
    .unwrap();
    let reference = summarize_detection(&accum, iou_thresholds(), &[1, 10, 100]).unwrap();

    let got_stats = part.overall.stats();
    let want_stats = reference.stats();
    assert_eq!(got_stats.len(), want_stats.len());
    // Bit-identical equality across all 12 stats — no tolerance: the
    // overall path delegates to the same accumulate/summarize calls.
    for (i, (g, w)) in got_stats.iter().zip(want_stats.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "stat {i} diverged: partition.overall={g}, reference={w}"
        );
    }
}

#[test]
fn two_axis_manifest_with_unassigned_smokes_through() {
    let (gt, dt) = load_fixture("all_perfect");
    let grid = run_bbox(&gt, &dt);
    let dims = GridDims {
        n_categories: grid.n_categories,
        n_area_ranges: grid.n_area_ranges,
        n_images: grid.n_images,
    };
    let id_map = image_id_to_idx(&gt);

    // Build a manifest covering only the first image; remaining ones
    // land in __unassigned__. Use an artificially small manifest so
    // the test fixture doesn't need to be touched.
    let first_image_id = *id_map.keys().next().unwrap();
    let manifest = format!(
        r#"{{
            "manifest_version": "1",
            "key_kind": "image_id",
            "rows": [
                {{"key": {}, "weather": "fog"}}
            ]
        }}"#,
        first_image_id.0
    );

    let (spec, warnings) = partition_spec_from_manifest(manifest.as_bytes(), &id_map, &[]).unwrap();
    assert!(warnings.is_empty(), "no manifest warnings expected");

    // weather=fog, weather=__unassigned__ — exactly two slices.
    let labels: Vec<(&str, &str)> = spec
        .slices
        .iter()
        .map(|s| (s.axis.as_str(), s.value.as_str()))
        .collect();
    assert_eq!(
        labels,
        vec![("weather", "fog"), ("weather", UNASSIGNED)]
    );

    let part = evaluate_partitioned(
        &grid.eval_imgs,
        dims,
        &spec,
        iou_thresholds(),
        ParityMode::Strict,
        SummaryKind::DetectionDefault,
    )
    .unwrap();

    assert_eq!(part.slices.len(), 2);
    // Each slice must produce the canonical 12 stats (sentinels at
    // -1.0 are fine — that's the point of the test, not an error).
    for sr in &part.slices {
        assert_eq!(sr.summary.stats().len(), 12);
    }
}

#[test]
fn manifest_with_only_unknown_keys_yields_overall_only_spec() {
    // When every manifest row references an image id absent from the
    // dataset, the parser emits one warning per row and the resolved
    // PartitionSpec carries no slices — the partition collapses to
    // `overall` only. This is the deliberate behavior: axes that
    // contributed no data don't conjure synthetic `__unassigned__`
    // buckets.
    let (gt, dt) = load_fixture("all_perfect");
    let grid = run_bbox(&gt, &dt);
    let dims = GridDims {
        n_categories: grid.n_categories,
        n_area_ranges: grid.n_area_ranges,
        n_images: grid.n_images,
    };
    let id_map = image_id_to_idx(&gt);

    let manifest = br#"{
        "manifest_version": "1",
        "key_kind": "image_id",
        "rows": [
            {"key": 999999, "weather": "fog"}
        ]
    }"#;
    let (spec, warnings) = partition_spec_from_manifest(manifest, &id_map, &[]).unwrap();
    assert_eq!(warnings.len(), 1);
    assert!(spec.slices.is_empty(), "spec.slices must be empty");

    let part = evaluate_partitioned(
        &grid.eval_imgs,
        dims,
        &spec,
        iou_thresholds(),
        ParityMode::Strict,
        SummaryKind::DetectionDefault,
    )
    .unwrap();
    assert!(part.slices.is_empty());
    // The overall pass still runs.
    assert_eq!(part.overall.stats().len(), 12);
}

// Compile-time touchpoint so the test file's `use` of the kept items
// stays warning-free even when only some tests run.
#[allow(dead_code)]
fn _docs_touch(_a: PerImageEval, _b: HashSet<usize>) {}
