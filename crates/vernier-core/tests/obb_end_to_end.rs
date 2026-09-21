#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::print_stderr
)]

//! End-to-end oriented-box evaluation through the unmodified ADR-0005
//! spine (ADR-0063 M4).
//!
//! The architectural claim this file defends is negative: *nothing in
//! `matching.rs`, `accumulate.rs` or `summarize.rs` changed*. Two new
//! kernels produce a dense f64 matrix, the spine consumes it, and AP
//! comes out. If oriented boxes had needed a spine edit, that would be a
//! separate ADR — so a test that runs the whole fold on oriented
//! geometry is the cheapest way to keep the claim honest.

use vernier_core::accumulate::{accumulate, AccumulateParams};
use vernier_core::dataset::{
    AnnId, CategoryId, CocoDataset, CocoDetections, DetectionArea, DetectionInput, ImageId,
};
use vernier_core::evaluate::{
    evaluate_quad, evaluate_rotated_box, AreaRange, EvalGrid, EvaluateParams,
};
use vernier_core::parity::{iou_thresholds, recall_thresholds, ParityMode};
use vernier_core::summarize::summarize_detection;
use vernier_geom::{AngleUnit, Convention, Rotation};

/// The canonical four-bucket COCO area grid. The 12-stat detection plan
/// indexes all four, so an oriented-box grid has to carry them too —
/// which is itself part of the claim: the summarizer is unchanged.
fn areas() -> [AreaRange; 4] {
    AreaRange::coco_default()
}

const CONV: Convention = Convention::D2;

fn gt_json(records: &str) -> CocoDataset {
    let json = format!(
        r#"{{"images": [{{"id": 1, "width": 1024, "height": 1024}}],
             "categories": [{{"id": 1, "name": "ship"}}],
             "annotations": [{records}]}}"#
    );
    CocoDataset::from_json_bytes(json.as_bytes()).expect("gt parses")
}

/// `rbox` -> the axis-aligned `bbox` a COCO record must still carry, as
/// the tight envelope. The Python and CLI ingest paths do this for the
/// user; here it is spelled out so the test reads as data.
fn envelope(rbox: [f64; 5]) -> [f64; 4] {
    let b = vernier_geom::RotatedBox::from_slice(&rbox).expect("finite");
    let (ex, ey) = b.aabb_half_extents(CONV);
    [b.cx - ex, b.cy - ey, 2.0 * ex, 2.0 * ey]
}

fn gt_record(id: i64, rbox: [f64; 5]) -> String {
    let e = envelope(rbox);
    let area = rbox[2] * rbox[3];
    format!(
        r#"{{"id": {id}, "image_id": 1, "category_id": 1, "iscrowd": 0,
             "area": {area}, "bbox": [{}, {}, {}, {}], "rbox": [{}, {}, {}, {}, {}]}}"#,
        e[0], e[1], e[2], e[3], rbox[0], rbox[1], rbox[2], rbox[3], rbox[4]
    )
}

/// A detection with **no** supplied `area`, so the `DetectionArea`
/// rule under test is the only thing that can set it.
fn dt_no_area(id: i64, score: f64, rbox: [f64; 5]) -> DetectionInput {
    DetectionInput {
        area: None,
        ..dt(id, score, rbox)
    }
}

fn dt(id: i64, score: f64, rbox: [f64; 5]) -> DetectionInput {
    let e = envelope(rbox);
    DetectionInput {
        id: Some(AnnId(id)),
        image_id: ImageId(1),
        category_id: CategoryId(1),
        score,
        bbox: e.into(),
        // Quirk OB13: the detection's area comes from the *oriented*
        // geometry, matching `loadRes`'s `bb[2] * bb[3]` on a length-5
        // bbox. Supplying it keeps the area bucket in step with the
        // kernel instead of with the envelope.
        area: Some(rbox[2] * rbox[3]),
        segmentation: None,
        keypoints: None,
        num_keypoints: None,
        rbox: Some(rbox),
        quad: None,
    }
}

fn params<'a>(thresholds: &'a [f64], areas: &'a [AreaRange]) -> EvaluateParams<'a> {
    EvaluateParams {
        iou_thresholds: thresholds,
        area_ranges: areas,
        max_dets_per_image: 100,
        use_cats: true,
        retain_iou: false,
        retain_meta: false,
    }
}

/// `AP @ [0.50:0.95]` — the first line of the canonical 12-stat plan.
fn ap(grid: &EvalGrid, thresholds: &[f64]) -> f64 {
    let acc = accumulate(
        &grid.eval_imgs,
        AccumulateParams {
            iou_thresholds: thresholds,
            recall_thresholds: recall_thresholds(),
            max_dets: &[1, 10, 100],
            n_categories: grid.n_categories,
            n_area_ranges: grid.n_area_ranges,
            n_images: grid.n_images,
        },
        ParityMode::Corrected,
    )
    .expect("accumulate");
    summarize_detection(&acc, thresholds, &[1, 10, 100], ParityMode::Corrected)
        .expect("summarize")
        .stats()[0]
}

/// Number of `(threshold 0, detection)` slots that matched a ground
/// truth, across the whole grid. Reads the spine's own output rather
/// than a summary statistic, which is what makes it usable with a
/// one-rung ladder.
fn matches_at_first_rung(grid: &EvalGrid) -> usize {
    grid.eval_imgs
        .iter()
        .flatten()
        .map(|cell| cell.dt_matched.row(0).iter().filter(|&&m| m).count())
        .sum()
}

#[test]
fn perfect_detections_score_one() {
    let area_grid = areas();
    let gt = gt_json(&format!(
        "{}, {}",
        gt_record(1, [100.0, 100.0, 60.0, 20.0, 30.0]),
        gt_record(2, [400.0, 300.0, 80.0, 25.0, -15.0])
    ));
    let dts = CocoDetections::from_inputs_with_area(
        vec![
            dt(1, 0.9, [100.0, 100.0, 60.0, 20.0, 30.0]),
            dt(2, 0.8, [400.0, 300.0, 80.0, 25.0, -15.0]),
        ],
        DetectionArea::Supplied,
    )
    .expect("detections");

    for mode in [ParityMode::Strict, ParityMode::Corrected] {
        let grid =
            evaluate_rotated_box(&gt, &dts, params(iou_thresholds(), &area_grid), mode, CONV)
                .expect("evaluate");
        let ap = ap(&grid, iou_thresholds());
        assert!((ap - 1.0).abs() < 1e-12, "{mode:?} AP = {ap}");
    }
}

#[test]
fn a_ninety_degree_orientation_error_destroys_ap() {
    let area_grid = areas();
    // Same center, same extents, rotated a quarter turn: a long thin
    // ship predicted across its own beam. The IoU of two congruent
    // rectangles at 90 degrees is `2*w*h / (w^2 + h^2)`, which for
    // 60 x 20 is 0.3 — below every COCO threshold.
    let gt = gt_json(&gt_record(1, [100.0, 100.0, 60.0, 20.0, 0.0]));
    let dts = CocoDetections::from_inputs_with_area(
        vec![dt(1, 0.9, [100.0, 100.0, 60.0, 20.0, 90.0])],
        DetectionArea::Supplied,
    )
    .expect("detections");
    let grid = evaluate_rotated_box(
        &gt,
        &dts,
        params(iou_thresholds(), &area_grid),
        ParityMode::Corrected,
        CONV,
    )
    .expect("evaluate");
    let ap = ap(&grid, iou_thresholds());
    assert_eq!(ap, 0.0, "a 90-degree error must not score");
}

#[test]
fn the_t1_ladder_recovers_a_pair_f64_comparison_would_drop() {
    let area_grid = areas();
    // ADR-0063 axis T1, demonstrated rather than asserted.
    //
    // detectron2's `computeIoU` hands `evaluateImg` a torch **f32**
    // tensor, and torch treats the `np.float64` threshold as a weak
    // scalar, so `ious[d, g] < iou` runs in f32 against `f32(t)`. At
    // `t = 0.7` that is `0.699999988...`, and a detection whose IoU is
    // exactly that value therefore *matches* under detectron2 while a
    // faithful f64 comparison drops it.
    //
    // Such a pair exists and this test finds it: IoU falls monotonically
    // with the offset, so bisect the f32 offset until the strict kernel
    // returns exactly `f32(0.7)`.
    let target = f64::from(0.7_f32);
    let gt_box = [0.0, 0.0, 100.0, 40.0, 0.0];
    let strict_iou =
        |dx: f32| vernier_geom::replica::d2::iou(&[f64::from(dx), 0.0, 100.0, 40.0, 0.0], &gt_box);
    let (mut lo, mut hi) = (0.0_f32.to_bits(), 30.0_f32.to_bits());
    while lo + 1 < hi {
        let mid = lo + (hi - lo) / 2;
        if strict_iou(f32::from_bits(mid)) > target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let dx = f32::from_bits(hi);
    assert_eq!(
        strict_iou(dx),
        target,
        "bisection failed to land on f32(0.7); dx = {dx:?}"
    );
    assert!(
        target < 0.7,
        "f32(0.7) must round down, else there is nothing to show"
    );

    let gt = gt_json(&gt_record(1, gt_box));
    let dts = CocoDetections::from_inputs_with_area(
        vec![dt(1, 0.9, [f64::from(dx), 0.0, 100.0, 40.0, 0.0])],
        DetectionArea::Supplied,
    )
    .expect("detections");

    // One rung, at 0.7.
    let ladder = [0.7_f64];

    // The shipped path projects the ladder and keeps the match, exactly
    // as RotatedCOCOeval does.
    let projected = evaluate_rotated_box(
        &gt,
        &dts,
        params(&ladder, &area_grid),
        ParityMode::Strict,
        CONV,
    )
    .expect("evaluate");
    assert_eq!(
        matches_at_first_rung(&projected),
        4,
        "T1 ladder should keep the match in every area bucket"
    );

    // The counterfactual, stated against the numbers rather than
    // against a second evaluator: the matrix entry is *below* the
    // declared 0.7, so a faithful f64 comparison drops this pair. That
    // is the divergence T1 exists to close, worth about `2^-24` of the
    // IoU axis per threshold — negligible per pair, not negligible
    // over DOTA-v2.
    assert!(
        strict_iou(dx) < ladder[0],
        "an f64 comparison against the declared threshold must drop the pair"
    );

    // And the regression test for the hole this once had. The
    // projection used to live on `evaluate_rotated_box`, so every other
    // way of reaching the same kernel — the generic funnel, the
    // parallel pass, the LRP decompose pass, the partitioned entry
    // points — compared an f32-valued matrix against an unprojected f64
    // threshold. It now lives on `EvalKernel::project_thresholds` and
    // is applied by the funnel, so the generic path agrees.
    let generic = vernier_core::evaluate::evaluate_with(
        &gt,
        &dts,
        params(&ladder, &area_grid),
        ParityMode::Strict,
        &vernier_core::similarity::RotatedBoxIou::new(CONV, ParityMode::Strict),
    )
    .expect("evaluate");
    assert_eq!(
        matches_at_first_rung(&generic),
        4,
        "the generic funnel must apply the same ladder the kernel entry point does"
    );

    let parallel = vernier_core::evaluate_parallel::evaluate_rotated_box_parallel(
        &gt,
        &dts,
        params(&ladder, &area_grid),
        ParityMode::Strict,
        CONV,
    )
    .expect("evaluate");
    assert_eq!(
        matches_at_first_rung(&parallel),
        4,
        "the parallel pass must apply the same ladder the sequential one does"
    );
}

/// The LRP path compares the retained IoU matrices against
/// `tp_threshold` directly, which the funnel's `iou_thresholds`
/// projection does not reach — so `tp_threshold` gets the T1 projection
/// of its own. Under the strict rotated-box replica a pair sitting
/// exactly on `f32(0.7)` is a TP for detectron2, and has to be one
/// here.
///
/// Observing that takes a little care, because LRP is *tie-neutral* at
/// exactly the threshold: promoting a would-be FN into a TP whose
/// normalized localization error is exactly `1.0` adds `1` to the
/// numerator and removes `1` from it, leaving the score untouched. The
/// difference shows up when the boundary detection would otherwise be
/// a **false positive** — then the denominator `n_tp + n_fp + n_fn`
/// grows too. So: two ground truths, and the boundary detection scored
/// *above* the clean one, which puts it inside the optimal tau.
#[test]
fn the_t1_ladder_reaches_the_lrp_threshold_too() {
    let area_grid = areas();
    let gt_box = [0.0, 0.0, 100.0, 40.0, 0.0];
    let strict_iou =
        |dx: f32| vernier_geom::replica::d2::iou(&[f64::from(dx), 0.0, 100.0, 40.0, 0.0], &gt_box);
    let target = f64::from(0.7_f32);
    let (mut lo, mut hi) = (0.0_f32.to_bits(), 30.0_f32.to_bits());
    while lo + 1 < hi {
        let mid = lo + (hi - lo) / 2;
        if strict_iou(f32::from_bits(mid)) > target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let dx = f32::from_bits(hi);
    assert_eq!(strict_iou(dx), target);

    // GT 1 is the boundary pair; GT 2 is matched exactly, so it
    // contributes no localization error and the arithmetic below stays
    // readable.
    let clean = [500.0, 500.0, 100.0, 40.0, 0.0];
    let gt = gt_json(&format!(
        "{}, {}",
        gt_record(1, gt_box),
        gt_record(2, clean)
    ));
    let dts = CocoDetections::from_inputs_with_area(
        vec![
            dt(1, 0.9, [f64::from(dx), 0.0, 100.0, 40.0, 0.0]),
            dt(2, 0.8, clean),
        ],
        DetectionArea::Supplied,
    )
    .expect("detections");

    let tau: Vec<f64> = (0..=100).map(|i| f64::from(i) / 100.0).collect();
    let ladder = [0.7_f64];
    let report = vernier_core::lrp::optimal_lrp_rotated_box(
        &gt,
        &dts,
        vernier_core::lrp::LrpParams {
            tp_threshold: 0.7,
            tau_grid: &tau,
            max_dets_per_image: 100,
            use_cats: true,
            iou_thresholds: &ladder,
            area_ranges: &area_grid,
        },
        ParityMode::Strict,
        CONV,
    )
    .expect("lrp");

    let cls = report
        .per_class
        .first()
        .expect("one category in the fixture");
    // Two TPs, no FP, no FN: `sum_loc / (1 - t') = 1` from the boundary
    // pair alone, over a denominator of 2.
    let loc = cls
        .olrp_loc
        .expect("the pair at f32(0.7) is a TP for detectron2, so it must be one here");
    assert!((0.0..=1.0).contains(&loc), "oLRP_Loc out of range: {loc}");
    let olrp = cls.olrp.expect("a class with positive GTs has an oLRP");
    assert!(
        (olrp - 0.5).abs() < 1e-9,
        "expected oLRP = 0.5 with the boundary pair matched; without the \
         projection it is a false positive and the score is 2/3, got {olrp}"
    );
    // The label keeps what the caller declared, not what was compared.
    assert_eq!(report.config.tp_threshold, 0.7);
}

#[test]
fn a_missing_rbox_field_is_a_typed_error_not_a_silent_zero() {
    let area_grid = areas();
    let json = r#"{"images": [{"id": 1, "width": 64, "height": 64}],
                   "categories": [{"id": 1, "name": "ship"}],
                   "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                                    "iscrowd": 0, "area": 16.0,
                                    "bbox": [0.0, 0.0, 4.0, 4.0]}]}"#;
    let gt = CocoDataset::from_json_bytes(json.as_bytes()).expect("gt parses");
    let dts = CocoDetections::from_inputs_with_area(
        vec![dt(1, 0.9, [2.0, 2.0, 4.0, 4.0, 0.0])],
        DetectionArea::Supplied,
    )
    .expect("detections");
    let err = evaluate_rotated_box(
        &gt,
        &dts,
        params(iou_thresholds(), &area_grid),
        ParityMode::Corrected,
        CONV,
    )
    .expect_err("a GT with no rbox must fail loudly");
    let msg = err.to_string();
    assert!(msg.contains("rbox"), "{msg}");
    assert!(
        msg.contains("length-5"),
        "the message must warn about the len-5 trap: {msg}"
    );
}

#[test]
fn the_quad_kernel_runs_the_same_fold() {
    let area_grid = areas();
    let corners = |rbox: [f64; 5]| {
        vernier_geom::RotatedBox::from_slice(&rbox)
            .expect("finite")
            .corners(CONV)
    };
    let g = [100.0, 100.0, 60.0, 20.0, 30.0];
    let q = corners(g);
    let e = envelope(g);
    let json = format!(
        r#"{{"images": [{{"id": 1, "width": 1024, "height": 1024}}],
             "categories": [{{"id": 1, "name": "ship"}}],
             "annotations": [{{"id": 1, "image_id": 1, "category_id": 1, "iscrowd": 0,
                               "area": 1200.0, "bbox": [{}, {}, {}, {}],
                               "quad": [{}, {}, {}, {}, {}, {}, {}, {}]}}]}}"#,
        e[0], e[1], e[2], e[3], q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7]
    );
    let gt = CocoDataset::from_json_bytes(json.as_bytes()).expect("gt parses");

    let mut d = dt(1, 0.9, g);
    d.rbox = None;
    d.quad = Some(q);
    let dts = CocoDetections::from_inputs_with_area(vec![d], DetectionArea::Supplied)
        .expect("detections");

    for mode in [ParityMode::Strict, ParityMode::Corrected] {
        let grid =
            evaluate_quad(&gt, &dts, params(iou_thresholds(), &area_grid), mode).expect("evaluate");
        let ap = ap(&grid, iou_thresholds());
        assert!((ap - 1.0).abs() < 1e-9, "{mode:?} AP = {ap}");
    }
}

#[test]
fn radians_and_degrees_describe_the_same_evaluation() {
    let area_grid = areas();
    let deg = Convention::new(AngleUnit::Deg, Rotation::ScreenCcw);
    let rad = Convention::new(AngleUnit::Rad, Rotation::ScreenCcw);

    let run = |conv: Convention, theta_g: f64, theta_d: f64| {
        let g = [100.0, 100.0, 60.0, 20.0, theta_g];
        let gt = gt_json(&gt_record(1, g));
        let dts = CocoDetections::from_inputs_with_area(
            vec![dt(1, 0.9, [104.0, 98.0, 58.0, 22.0, theta_d])],
            DetectionArea::Supplied,
        )
        .expect("detections");
        let grid = evaluate_rotated_box(
            &gt,
            &dts,
            params(iou_thresholds(), &area_grid),
            ParityMode::Corrected,
            conv,
        )
        .expect("evaluate");
        ap(&grid, iou_thresholds())
    };

    let a = run(deg, 30.0, 25.0);
    let b = run(rad, 30.0_f64.to_radians(), 25.0_f64.to_radians());
    assert!((a - b).abs() < 1e-12, "{a} vs {b}");
    assert!(
        a > 0.0 && a < 1.0,
        "the pair should be a partial match: {a}"
    );
}

/// Quirk **OB13**. `loadRes` sets `area = bb[2] * bb[3]`, and on the
/// length-5 `bbox` detectron2 writes that is the *oriented* `w * h`.
/// vernier keeps `bbox` as the axis-aligned envelope, so it has to
/// derive the area from the geometry instead — otherwise a rotated
/// object lands in a different COCO area bucket than the kernel that
/// scored it.
///
/// A 60 x 15 box at 45 degrees is the cleanest witness: oriented area
/// 900 is *small* (`< 32^2`), while its envelope is 53.03 on a side,
/// which is 2812 and *medium*.
#[test]
fn the_dt_area_comes_from_the_oriented_geometry_not_the_envelope() {
    let rbox = [500.0, 500.0, 60.0, 15.0, 45.0];
    let env = envelope(rbox);
    let envelope_area = env[2] * env[3];
    assert!(
        envelope_area > 1024.0 && rbox[2] * rbox[3] < 1024.0,
        "fixture must straddle the small/medium boundary: oriented {} vs envelope {envelope_area}",
        rbox[2] * rbox[3]
    );

    let oriented = CocoDetections::from_inputs_with_area(
        vec![dt_no_area(1, 0.9, rbox)],
        DetectionArea::Oriented,
    )
    .expect("detections");
    assert_eq!(oriented.detections()[0].area, 900.0);

    // The old behavior, still reachable and still correct for the
    // axis-aligned kernels: the envelope.
    let from_bbox = CocoDetections::from_inputs_with_area(
        vec![dt_no_area(1, 0.9, rbox)],
        DetectionArea::FromBbox,
    )
    .expect("detections");
    assert_eq!(from_bbox.detections()[0].area, envelope_area);

    // Quads take the enclosed area, and it is the *same* number the IoU
    // denominator uses — `PreparedQuad::area`, not a second shoelace
    // that could differ in the last place.
    let quad = vernier_geom::RotatedBox::from_slice(&rbox)
        .expect("finite")
        .corners(CONV);
    let mut input = dt_no_area(2, 0.9, rbox);
    input.rbox = None;
    input.quad = Some(quad);
    let quads = CocoDetections::from_inputs_with_area(vec![input], DetectionArea::Oriented)
        .expect("detections");
    let prepared = vernier_geom::PreparedQuad::new(&quad, false).expect("valid quad");
    assert_eq!(quads.detections()[0].area, prepared.area);
    assert!((quads.detections()[0].area - 900.0).abs() < 1e-9);
}
