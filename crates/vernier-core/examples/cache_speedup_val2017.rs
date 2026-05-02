//! End-to-end measurement of [`BoundaryGtCache`] benefit on the COCO
//! val2017 perfect-segm workload.
//!
//! Models the training-loop validation pattern: same GT, fresh DT
//! each pass. The bench harness in `bench/` runs single-shot evaluates
//! per impl and so doesn't surface what the cache buys; this example
//! does two passes back-to-back in one process and reports both.
//!
//! Inputs (same conventions as the bench harness):
//! - `VERNIER_COCO_GT_PATH` → GT JSON (falls back to
//!   `~/.cache/vernier-bench/coco_val2017/instances_val2017.json`)
//! - `VERNIER_COCO_DT_SEGM_PATH` → DT JSON (falls back to
//!   `<repo>/.cache/coco-val2017/perfect_dt_segm.json`)
//!
//! Run:
//! ```sh
//! cargo run --release --example cache_speedup_val2017 -p vernier-core
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::print_stdout)]

use std::env;
use std::path::PathBuf;
use std::time::Instant;

use vernier_core::{
    evaluate_boundary, evaluate_boundary_cached, iou_thresholds, AreaRange, BoundaryGtCache,
    CocoDataset, CocoDetections, EvalDataset, EvaluateParams, ParityMode,
    BOUNDARY_DILATION_RATIO_DEFAULT,
};

fn gt_path() -> PathBuf {
    if let Ok(env) = env::var("VERNIER_COCO_GT_PATH") {
        return PathBuf::from(env);
    }
    let home = env::var("HOME").expect("HOME not set");
    PathBuf::from(home).join(".cache/vernier-bench/coco_val2017/instances_val2017.json")
}

fn dt_path() -> PathBuf {
    if let Ok(env) = env::var("VERNIER_COCO_DT_SEGM_PATH") {
        return PathBuf::from(env);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../.cache/coco-val2017/perfect_dt_segm.json")
}

fn time_ms<F: FnOnce() -> R, R>(f: F) -> (R, f64) {
    let t = Instant::now();
    let r = f();
    let ms = t.elapsed().as_secs_f64() * 1000.0;
    (r, ms)
}

fn main() {
    let gt_p = gt_path();
    let dt_p = dt_path();
    println!("GT: {}", gt_p.display());
    println!("DT: {}", dt_p.display());

    let (gt_bytes, load_gt_ms) = time_ms(|| std::fs::read(&gt_p).expect("read GT"));
    let (dt_bytes, load_dt_ms) = time_ms(|| std::fs::read(&dt_p).expect("read DT"));
    println!(
        "Loaded {:.1} MiB GT in {:.0} ms, {:.1} MiB DT in {:.0} ms",
        gt_bytes.len() as f64 / 1_048_576.0,
        load_gt_ms,
        dt_bytes.len() as f64 / 1_048_576.0,
        load_dt_ms,
    );

    let (gt, parse_gt_ms) =
        time_ms(|| CocoDataset::from_json_bytes(&gt_bytes).expect("parse GT"));
    let (dt, parse_dt_ms) =
        time_ms(|| CocoDetections::from_json_bytes(&dt_bytes).expect("parse DT"));
    println!("Parsed GT in {parse_gt_ms:.0} ms, DT in {parse_dt_ms:.0} ms");
    println!(
        "GT: {} images, {} categories; DT: {} detections",
        gt.images().len(),
        gt.categories().len(),
        dt.detections().len(),
    );

    let area = AreaRange::coco_default();
    let params = EvaluateParams {
        iou_thresholds: iou_thresholds(),
        area_ranges: &area,
        max_dets_per_image: 100,
        use_cats: true,
    };
    let ratio = BOUNDARY_DILATION_RATIO_DEFAULT;

    println!("\n=== uncached: two back-to-back evaluate_boundary calls ===");
    let (_, u1_ms) = time_ms(|| {
        evaluate_boundary(&gt, &dt, params, ParityMode::Strict, ratio).unwrap()
    });
    println!("call 1: {u1_ms:>9.0} ms");
    let (_, u2_ms) = time_ms(|| {
        evaluate_boundary(&gt, &dt, params, ParityMode::Strict, ratio).unwrap()
    });
    println!("call 2: {u2_ms:>9.0} ms");

    println!("\n=== cached: two back-to-back evaluate_boundary_cached calls (shared cache) ===");
    let cache = BoundaryGtCache::new();
    let (_, c1_ms) = time_ms(|| {
        evaluate_boundary_cached(&gt, &dt, params, ParityMode::Strict, ratio, &cache).unwrap()
    });
    println!(
        "call 1 (cold cache, populates): {c1_ms:>9.0} ms ({} entries)",
        cache.len()
    );
    let (_, c2_ms) = time_ms(|| {
        evaluate_boundary_cached(&gt, &dt, params, ParityMode::Strict, ratio, &cache).unwrap()
    });
    println!("call 2 (warm cache, hits):      {c2_ms:>9.0} ms");

    let speedup = u2_ms / c2_ms;
    let saved_ms = u2_ms - c2_ms;
    println!("\n=== summary ===");
    println!(
        "uncached pass-2 wall  : {u2_ms:>9.0} ms\n\
         cached   pass-2 wall  : {c2_ms:>9.0} ms\n\
         saved per warm call   : {saved_ms:>9.0} ms ({speedup:.2}× faster)"
    );
}
