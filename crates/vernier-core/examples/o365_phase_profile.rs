//! Phase-by-phase profile of the Objects365 bbox cell.
//!
//! The bench harness reports one `total` per impl; this example splits
//! that total into GT parse / DT parse / match / accumulate so a perf
//! push can target the phase that actually dominates at 1.24 M boxes.
//!
//! Inputs (same conventions as the bench harness):
//! - `VERNIER_OBJECTS365_GT_PATH` → GT JSON (falls back to the bench cache)
//! - `VERNIER_OBJECTS365_DT_PATH` → DT JSON (falls back to the bench cache)
//!
//! Run:
//! ```sh
//! cargo run --release --example o365_phase_profile -p vernier-core
//! VERNIER_PROFILE_THREADS=8 cargo run --release --example o365_phase_profile -p vernier-core
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::print_stdout)]

use std::env;
use std::path::PathBuf;
use std::time::Instant;

use vernier_core::accumulate::{accumulate, AccumulateParams};
use vernier_core::parity::{iou_thresholds, recall_thresholds};
use vernier_core::similarity::BboxIou;
use vernier_core::{
    evaluate_bbox, evaluate_with_parallel, AreaRange, CocoDataset, CocoDetections, EvalDataset,
    EvaluateParams, ParityMode,
};

fn path_from(env_var: &str, fallback: &str) -> PathBuf {
    if let Ok(p) = env::var(env_var) {
        return PathBuf::from(p);
    }
    let home = env::var("HOME").expect("HOME not set");
    PathBuf::from(home).join(fallback)
}

fn time_ms<F: FnOnce() -> R, R>(f: F) -> (R, f64) {
    let t = Instant::now();
    let r = f();
    (r, t.elapsed().as_secs_f64() * 1000.0)
}

fn main() {
    let gt_p = path_from(
        "VERNIER_OBJECTS365_GT_PATH",
        ".cache/vernier-bench/objects365_val/zhiyuan_objv2_val.json",
    );
    let dt_p = path_from(
        "VERNIER_OBJECTS365_DT_PATH",
        ".cache/vernier-bench/jittered/objects365_val_jittered_seed0_v2.json",
    );
    if let Ok(n) = env::var("VERNIER_PROFILE_THREADS") {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n.parse().expect("VERNIER_PROFILE_THREADS must be an int"))
            .build_global()
            .expect("build rayon pool");
        println!("rayon threads: {n}");
    }

    let (gt_bytes, read_gt_ms) = time_ms(|| std::fs::read(&gt_p).expect("read GT"));
    let (dt_bytes, read_dt_ms) = time_ms(|| std::fs::read(&dt_p).expect("read DT"));
    let (gt, parse_gt_ms) = time_ms(|| CocoDataset::from_json_bytes(&gt_bytes).expect("parse GT"));
    let (dt, parse_dt_ms) =
        time_ms(|| CocoDetections::from_json_bytes(&dt_bytes).expect("parse DT"));

    let area = AreaRange::coco_default();
    let params = EvaluateParams {
        iou_thresholds: iou_thresholds(),
        area_ranges: &area,
        max_dets_per_image: 100,
        use_cats: true,
        retain_iou: false,
    };
    let (grid, match_ms) =
        time_ms(|| evaluate_bbox(&gt, &dt, params, ParityMode::Strict).expect("evaluate"));

    // What the dense (K x A x I) grid costs before any IoU is computed:
    // 116.8M slots on O365, of which ~2% are ever occupied.
    let n_slots = gt.categories().len() * area.len() * gt.images().len();
    let (grid_alloc, alloc_ms) = time_ms(|| vec![0_u64; n_slots * 2]);
    println!(
        "dense grid slots  {n_slots:>9}  bare alloc+touch {alloc_ms:>7.0} ms ({:.1} MiB of pointers)",
        (n_slots * 16) as f64 / 1_048_576.0
    );
    drop(grid_alloc);

    let (_pgrid, par_ms) = time_ms(|| {
        evaluate_with_parallel(&gt, &dt, params, ParityMode::Strict, &BboxIou).expect("parallel")
    });
    println!("match parallel    {par_ms:>9.0} ms (vs {match_ms:.0} ms serial)");
    #[cfg(feature = "bench-timings")]
    {
        let (par_ns, post_ns, calls) = vernier_core::read_and_reset_evaluate_parallel_timings();
        println!(
            "  par_iter region {:>9.0} ms | post-pass (fill) {:>7.0} ms | calls {calls}",
            par_ns as f64 / 1e6,
            post_ns as f64 / 1e6
        );
    }

    let max_dets = [1_usize, 10, 100];
    let acc_params = AccumulateParams {
        iou_thresholds: iou_thresholds(),
        recall_thresholds: recall_thresholds(),
        max_dets: &max_dets,
        n_categories: gt.categories().len(),
        n_area_ranges: area.len(),
        n_images: gt.images().len(),
    };
    let (_acc, accumulate_ms) = time_ms(|| {
        accumulate(&grid.eval_imgs, acc_params, ParityMode::Strict).expect("accumulate")
    });

    let total = read_gt_ms + read_dt_ms + parse_gt_ms + parse_dt_ms + match_ms + accumulate_ms;
    let pct = |ms: f64| 100.0 * ms / total;
    println!(
        "{} images · {} categories · {} detections",
        gt.images().len(),
        gt.categories().len(),
        dt.detections().len()
    );
    for (name, ms) in [
        ("read GT+DT", read_gt_ms + read_dt_ms),
        ("parse GT", parse_gt_ms),
        ("parse DT", parse_dt_ms),
        ("match (evaluate)", match_ms),
        ("accumulate", accumulate_ms),
    ] {
        println!("{name:<18} {ms:>9.0} ms  {:>5.1}%", pct(ms));
    }
    println!("{:<18} {total:>9.0} ms", "TOTAL");
}
