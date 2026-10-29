//! Parallel sibling of [`crate::evaluate::evaluate_with`] (ADR-0047).
//!
//! Each category's occupied images are cut into image-ascending chunks,
//! dispatched via `par_iter` against a per-worker `CellScratch`, and
//! collected in task order — category-major, image-ascending — then
//! concatenated per category.
//!
//! Strict-mode bit-equality across thread counts holds by construction:
//! both paths share `GridPlan` and the per-cell body,
//! the matching kernel is pure on owned data, and concatenating the
//! chunks in order reproduces the sequential pass's packed arena exactly
//! (metadata and the retained-IoU map are keyed, so their order is
//! immaterial). The parity harness asserts this along the `num_threads`
//! axis.
//!
//! The caller must `install` a `rayon::ThreadPool` around the call;
//! this module uses `par_iter` against the ambient pool.

use std::collections::HashMap;

use ndarray::Array2;
use rayon::prelude::*;

use crate::accumulate::{PackedCategory, PackedEvalGrid};
use crate::dataset::{CocoDataset, CocoDetections};
use crate::error::EvalError;
use crate::evaluate::{
    evaluate_category_slice, CellScratch, EvalGrid, EvalImageMeta, EvalKernel, EvaluateParams,
    GridPlan, KernelScratch, PackedPass,
};
use crate::matching::MatchScratch;
use crate::parity::ParityMode;

/// Wall-time split for [`evaluate_with_parallel`], gated on `bench-timings`.
#[cfg(feature = "bench-timings")]
pub(crate) mod timings {
    use crate::bench_counters::BenchCounterSet;

    pub(super) const PAR_ITER_NS: usize = 0;
    pub(super) const POST_PASS_NS: usize = 1;
    pub(super) const N_CALLS: usize = 2;

    pub(super) static COUNTERS: BenchCounterSet<3> = BenchCounterSet::new();

    pub(crate) fn read_and_reset() -> (u64, u64, u64) {
        let [p, s, n] = COUNTERS.read_and_reset();
        (p, s, n)
    }
}

/// Parallel sibling of [`crate::evaluate::evaluate_with`]. Caller
/// `install`s a `rayon::ThreadPool` around the call.
///
/// # Errors
///
/// Propagates the first [`EvalError`] from any per-cell call.
pub fn evaluate_with_parallel<K: EvalKernel>(
    gt: &CocoDataset,
    dt: &CocoDetections,
    params: EvaluateParams<'_>,
    parity_mode: ParityMode,
    kernel: &K,
) -> Result<EvalGrid, EvalError> {
    evaluate_subset_with_parallel(gt, dt, params, parity_mode, kernel, None)
        .map(PackedPass::into_grid)
}

/// Subset-scoped sibling of [`evaluate_with_parallel`] (ADR-0066).
pub(crate) fn evaluate_subset_with_parallel<K: EvalKernel>(
    gt: &CocoDataset,
    dt: &CocoDetections,
    params: EvaluateParams<'_>,
    parity_mode: ParityMode,
    kernel: &K,
    image_subset: Option<&[usize]>,
) -> Result<PackedPass, EvalError> {
    /// Upper bound on a task's width: wide enough to amortize the chunk
    /// concat, narrow enough that one heavy category cannot straggle.
    const MAX_CHUNK_IMAGES: usize = 512;
    /// Tasks per worker the width aims for, so rayon can rebalance.
    const TASKS_PER_THREAD: usize = 4;

    #[cfg(feature = "bench-timings")]
    let t_par = std::time::Instant::now();

    let plan = GridPlan::new(gt, dt, params, parity_mode, kernel, image_subset)?;
    let n_i = plan.images.len();
    let n_a = params.area_ranges.len();
    let n_t = params.iou_thresholds.len();
    let n_k = plan.category_buckets.len();
    let builds_meta = params.builds_meta();

    // Partition each category's occupied-image slice into fixed-width
    // tasks so heavy categories (`person`) and `use_cats=false` (`n_k = 1`)
    // fan out across all workers without tail stragglers. The width
    // shrinks below `MAX_CHUNK_IMAGES` when the grid holds too few cells
    // to give every worker `TASKS_PER_THREAD` tasks: a few-category,
    // few-image grid (one class, dense cells) would otherwise be a single
    // task and run serially. `chunk_tasks` is category-major and
    // image-ascending within a category, so the in-order `collect` below
    // is already canonical.
    let n_cells: usize = (0..n_k).map(|k| plan.occupied.row(k).len()).sum();
    let chunk_images = n_cells
        .div_ceil(rayon::current_num_threads() * TASKS_PER_THREAD)
        .clamp(1, MAX_CHUNK_IMAGES);
    let mut chunk_tasks: Vec<(usize, &[u32])> = Vec::new();
    for k in 0..n_k {
        for chunk in plan.occupied.row(k).chunks(chunk_images) {
            chunk_tasks.push((k, chunk));
        }
    }

    struct ChunkOutput {
        k: usize,
        packed: PackedCategory,
        meta: Vec<(usize, Box<EvalImageMeta>)>,
        retained: Vec<(usize, Array2<f64>)>,
    }

    let chunk_outputs: Vec<ChunkOutput> = chunk_tasks
        .into_par_iter()
        .map_init(
            || {
                (
                    CellScratch::new(),
                    KernelScratch::<K::Annotation>::default(),
                    MatchScratch::default(),
                )
            },
            |(scratch, kernel_scratch, match_scratch), (k, img_chunk)| {
                let mut meta = Vec::new();
                let mut retained = Vec::new();
                let packed = evaluate_category_slice(
                    scratch,
                    kernel_scratch,
                    match_scratch,
                    plan.category_buckets[k],
                    img_chunk,
                    &plan,
                    gt,
                    dt,
                    params,
                    parity_mode,
                    kernel,
                    builds_meta.then_some(&mut meta),
                    params.retain_iou.then_some(&mut retained),
                )?;
                Ok::<ChunkOutput, EvalError>(ChunkOutput {
                    k,
                    packed,
                    meta,
                    retained,
                })
            },
        )
        .collect::<Result<Vec<_>, _>>()?;

    #[cfg(feature = "bench-timings")]
    let par_ns = u64::try_from(t_par.elapsed().as_nanos()).unwrap_or(u64::MAX);

    #[cfg(feature = "bench-timings")]
    let t_post = std::time::Instant::now();

    let mut per_cat_chunks: Vec<Vec<PackedCategory>> = (0..n_k).map(|_| Vec::new()).collect();
    let mut meta = builds_meta.then(Vec::new);
    let mut retained = params.retain_iou.then(HashMap::new);
    for out in chunk_outputs {
        per_cat_chunks[out.k].push(out.packed);
        if let Some(meta) = meta.as_mut() {
            meta.extend(
                out.meta
                    .into_iter()
                    .map(|(cat_offset, m)| ((out.k, cat_offset / n_i, cat_offset % n_i), m)),
            );
        }
        if let Some(map) = retained.as_mut() {
            map.extend(out.retained.into_iter().map(|(i, mat)| ((out.k, i), mat)));
        }
    }

    let categories: Vec<PackedCategory> = per_cat_chunks
        .into_par_iter()
        .map(|chunks| PackedCategory::concat_chunks(chunks, n_a, n_t))
        .collect::<Result<_, _>>()?;

    let pass = PackedPass {
        packed: PackedEvalGrid {
            categories,
            n_categories: n_k,
            n_area_ranges: n_a,
            n_images: n_i,
            n_t,
        },
        meta,
        retained,
    };

    #[cfg(feature = "bench-timings")]
    {
        let post_ns = u64::try_from(t_post.elapsed().as_nanos()).unwrap_or(u64::MAX);
        timings::COUNTERS.add(timings::PAR_ITER_NS, par_ns);
        timings::COUNTERS.add(timings::POST_PASS_NS, post_ns);
        timings::COUNTERS.bump(timings::N_CALLS);
    }
    Ok(pass)
}

// ----- Per-paradigm parallel sibling wrappers -------------------------
//
// One `pub fn` per `crate::evaluate::evaluate_*` entry; the FFI routes
// between sequential and parallel without holding the kernel type.

use crate::evaluate::{boundary_kernel, segm_kernel};
use crate::similarity::{BboxIou, BoundaryGtCache, OksSimilarity, SegmGtCache};

/// Parallel sibling of [`crate::evaluate::evaluate_bbox`].
///
/// # Errors
/// Propagates [`EvalError`] from the underlying kernel and matching
/// calls.
pub fn evaluate_bbox_parallel(
    gt: &CocoDataset,
    dt: &CocoDetections,
    params: EvaluateParams<'_>,
    parity_mode: ParityMode,
) -> Result<EvalGrid, EvalError> {
    evaluate_with_parallel(gt, dt, params, parity_mode, &BboxIou)
}

/// Parallel sibling of [`crate::evaluate::evaluate_segm`].
///
/// # Errors
/// Propagates [`EvalError`] from the underlying kernel and matching
/// calls.
pub fn evaluate_segm_parallel(
    gt: &CocoDataset,
    dt: &CocoDetections,
    params: EvaluateParams<'_>,
    parity_mode: ParityMode,
) -> Result<EvalGrid, EvalError> {
    evaluate_with_parallel(gt, dt, params, parity_mode, &segm_kernel(None))
}

/// Parallel sibling of [`crate::evaluate::evaluate_segm_cached`].
///
/// # Errors
/// Propagates [`EvalError`] from the underlying kernel and matching
/// calls.
pub fn evaluate_segm_cached_parallel(
    gt: &CocoDataset,
    dt: &CocoDetections,
    params: EvaluateParams<'_>,
    parity_mode: ParityMode,
    cache: &SegmGtCache,
) -> Result<EvalGrid, EvalError> {
    evaluate_with_parallel(gt, dt, params, parity_mode, &segm_kernel(Some(cache)))
}

/// Parallel sibling of [`crate::evaluate::evaluate_boundary`].
///
/// Uncached path: the cell loop derives GT bands inline. Callers that
/// run repeated evaluations against the same GT should construct a
/// [`BoundaryGtCache`] and use [`evaluate_boundary_cached_parallel`]
/// instead (per ADR-0020); the first call populates the cache, every
/// subsequent call reuses the bands.
///
/// # Errors
/// Propagates [`EvalError`] from the underlying kernel and matching
/// calls.
pub fn evaluate_boundary_parallel(
    gt: &CocoDataset,
    dt: &CocoDetections,
    params: EvaluateParams<'_>,
    parity_mode: ParityMode,
    dilation_ratio: f64,
) -> Result<EvalGrid, EvalError> {
    evaluate_with_parallel(
        gt,
        dt,
        params,
        parity_mode,
        &boundary_kernel(dilation_ratio, None),
    )
}

/// Parallel sibling of [`crate::evaluate::evaluate_boundary_cached`].
///
/// # Errors
/// Propagates [`EvalError`] from the underlying kernel and matching
/// calls.
pub fn evaluate_boundary_cached_parallel(
    gt: &CocoDataset,
    dt: &CocoDetections,
    params: EvaluateParams<'_>,
    parity_mode: ParityMode,
    dilation_ratio: f64,
    cache: &BoundaryGtCache,
) -> Result<EvalGrid, EvalError> {
    cache.align_ratio(dilation_ratio);
    evaluate_with_parallel(
        gt,
        dt,
        params,
        parity_mode,
        &boundary_kernel(dilation_ratio, Some(cache)),
    )
}

/// Parallel sibling of [`crate::evaluate::evaluate_keypoints`].
///
/// # Errors
/// Propagates [`EvalError`] from the underlying kernel and matching
/// calls.
pub fn evaluate_keypoints_parallel(
    gt: &CocoDataset,
    dt: &CocoDetections,
    params: EvaluateParams<'_>,
    parity_mode: ParityMode,
    sigmas: HashMap<i64, Vec<f64>>,
) -> Result<EvalGrid, EvalError> {
    evaluate_with_parallel(gt, dt, params, parity_mode, &OksSimilarity::new(sigmas))
}

#[cfg(test)]
mod tests {
    //! Rust-side smoke: bit-equal output to `evaluate_with` on a
    //! small synthetic bbox fixture. The load-bearing parity assertion
    //! lives in `tests/python/parity/` (cross-thread bit-equality
    //! across every paradigm and real COCO/LVIS fixtures).
    use super::*;
    use crate::accumulate::{accumulate, AccumulateParams, PerImageEval};
    use crate::dataset::{
        AnnId, Bbox, CategoryId, CategoryMeta, CocoAnnotation, DetectionInput, ImageId, ImageMeta,
    };
    use crate::evaluate::{evaluate_with, AreaRange};
    use crate::parity::{iou_thresholds, recall_thresholds};
    use crate::similarity::BboxIou;

    fn img(id: i64, w: u32, h: u32) -> ImageMeta {
        ImageMeta {
            id: ImageId(id),
            width: w,
            height: h,
            file_name: None,
        }
    }

    fn cat(id: i64, name: &str) -> CategoryMeta {
        CategoryMeta {
            id: CategoryId(id),
            name: name.into(),
            supercategory: None,
        }
    }

    fn ann(id: i64, image: i64, cat_id: i64, bbox: (f64, f64, f64, f64)) -> CocoAnnotation {
        CocoAnnotation {
            id: AnnId(id),
            image_id: ImageId(image),
            category_id: CategoryId(cat_id),
            area: bbox.2 * bbox.3,
            is_crowd: false,
            ignore_flag: None,
            bbox: Bbox {
                x: bbox.0,
                y: bbox.1,
                w: bbox.2,
                h: bbox.3,
            },
            segmentation: None,
            keypoints: None,
            num_keypoints: None,
        }
    }

    fn dt_input(image: i64, cat_id: i64, score: f64, bbox: (f64, f64, f64, f64)) -> DetectionInput {
        DetectionInput {
            id: None,
            image_id: ImageId(image),
            category_id: CategoryId(cat_id),
            score,
            bbox: Bbox {
                x: bbox.0,
                y: bbox.1,
                w: bbox.2,
                h: bbox.3,
            },
            area: None,
            segmentation: None,
            keypoints: None,
            num_keypoints: None,
        }
    }

    fn tiny_grid_inputs() -> (CocoDataset, CocoDetections) {
        let images = vec![img(1, 100, 100), img(2, 100, 100), img(3, 100, 100)];
        let cats = vec![cat(1, "a"), cat(2, "b")];
        let mut anns = Vec::new();
        let mut next_id = 1_i64;
        for &im_id in &[1_i64, 2, 3] {
            for &cat_id in &[1_i64, 2] {
                anns.push(ann(next_id, im_id, cat_id, (10.0, 10.0, 20.0, 20.0)));
                next_id += 1;
            }
        }
        let gt = CocoDataset::from_parts(images, anns, cats).expect("dataset");
        let dts = CocoDetections::from_inputs(vec![
            dt_input(1, 1, 0.9, (12.0, 12.0, 18.0, 18.0)),
            dt_input(2, 1, 0.8, (11.0, 11.0, 20.0, 20.0)),
            dt_input(3, 2, 0.7, (15.0, 15.0, 10.0, 10.0)),
        ])
        .expect("detections");
        (gt, dts)
    }

    /// A subset pass (the streaming `update`) is the full pass restricted
    /// to the subset's images: identical cells there, `None` elsewhere —
    /// sequential and parallel, with an unsorted, repeated, out-of-range
    /// subset.
    #[test]
    fn subset_pass_is_the_full_pass_restricted_to_its_images() {
        let (gt, dt) = tiny_grid_inputs();
        let area = AreaRange::coco_default();
        let params = EvaluateParams {
            iou_thresholds: iou_thresholds(),
            area_ranges: &area,
            max_dets_per_image: 100,
            use_cats: true,
            retain_iou: true,
            retain_meta: false,
        };
        let full = evaluate_with(&gt, &dt, params, ParityMode::Strict, &BboxIou).expect("full");
        let subset = [2_usize, 0, 2, 99];
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .expect("pool");
        let seq = crate::evaluate::evaluate_subset_with(
            &gt,
            &dt,
            params,
            ParityMode::Strict,
            &BboxIou,
            Some(&subset),
        )
        .expect("seq subset")
        .into_grid();
        let par = pool
            .install(|| {
                evaluate_subset_with_parallel(
                    &gt,
                    &dt,
                    params,
                    ParityMode::Strict,
                    &BboxIou,
                    Some(&subset),
                )
            })
            .expect("par subset")
            .into_grid();
        let n_i = full.n_images;
        let full_cells = full.eval_imgs();
        for grid in [&seq, &par] {
            let cells = grid.eval_imgs();
            assert_eq!(cells.len(), full_cells.len());
            for (flat, (got, want)) in cells.iter().zip(full_cells.iter()).enumerate() {
                let in_subset = subset.contains(&(flat % n_i));
                match (got, want, in_subset) {
                    (None, _, false) | (None, None, true) => {}
                    (Some(g), Some(w), true) => {
                        assert_eq!(g.dt_scores, w.dt_scores, "flat={flat}");
                        assert_eq!(g.dt_matched, w.dt_matched, "flat={flat}");
                        assert_eq!(g.dt_ignore, w.dt_ignore, "flat={flat}");
                        assert_eq!(g.gt_ignore, w.gt_ignore, "flat={flat}");
                    }
                    _ => panic!("presence mismatch at flat={flat} (in subset: {in_subset})"),
                }
            }
        }
        let full_ious = full.retained_ious.as_ref().expect("full retains");
        for grid in [&seq, &par] {
            let ious = grid.retained_ious.as_ref().expect("subset retains");
            let mut got: Vec<(usize, usize)> = ious.iter().map(|(k, i, _)| (k, i)).collect();
            got.sort_unstable();
            let mut want: Vec<(usize, usize)> = full_ious
                .iter()
                .map(|(k, i, _)| (k, i))
                .filter(|&(_, i)| subset.contains(&i))
                .collect();
            want.sort_unstable();
            assert_eq!(got, want, "retained IoU keys are the subset's");
            for &(k, i) in &got {
                assert_eq!(ious.get(k, i), full_ious.get(k, i), "(k={k}, i={i})");
            }
        }
        assert!(full_cells
            .iter()
            .enumerate()
            .any(|(f, c)| c.is_some() && f % n_i == 1));
    }

    #[test]
    fn parallel_eval_imgs_bit_equals_sequential_bbox() {
        let (gt, dt) = tiny_grid_inputs();
        let area = AreaRange::coco_default();
        let params = EvaluateParams {
            iou_thresholds: iou_thresholds(),
            area_ranges: &area,
            max_dets_per_image: 100,
            use_cats: true,
            retain_iou: false,
            retain_meta: false,
        };
        let kernel = BboxIou;

        let seq = evaluate_with(&gt, &dt, params, ParityMode::Strict, &kernel).expect("seq");

        // Run under a pool of 4 threads to force the par_iter to fan
        // out; the result must be byte-equal to the sequential one.
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("pool");
        let par = pool
            .install(|| evaluate_with_parallel(&gt, &dt, params, ParityMode::Strict, &kernel))
            .expect("par");

        assert_eq!(seq.n_categories, par.n_categories);
        assert_eq!(seq.n_area_ranges, par.n_area_ranges);
        assert_eq!(seq.n_images, par.n_images);
        let (seq_cells, par_cells) = (seq.eval_imgs(), par.eval_imgs());
        assert_eq!(seq_cells.len(), par_cells.len());
        for (idx, (s, p)) in seq_cells.iter().zip(par_cells.iter()).enumerate() {
            match (s, p) {
                (None, None) => {}
                (Some(s), Some(p)) => {
                    assert_eq!(s.dt_scores, p.dt_scores, "dt_scores mismatch at flat={idx}");
                    assert_eq!(s.gt_ignore, p.gt_ignore, "gt_ignore mismatch at flat={idx}");
                    assert_eq!(
                        s.dt_matched, p.dt_matched,
                        "dt_matched mismatch at flat={idx}"
                    );
                    assert_eq!(s.dt_ignore, p.dt_ignore, "dt_ignore mismatch at flat={idx}");
                }
                _ => panic!("Some/None mismatch at flat={idx}"),
            }
        }
    }

    #[test]
    fn parallel_summary_bit_equals_sequential_across_thread_counts() {
        let (gt, dt) = tiny_grid_inputs();
        let area = AreaRange::coco_default();
        let params = EvaluateParams {
            iou_thresholds: iou_thresholds(),
            area_ranges: &area,
            max_dets_per_image: 100,
            use_cats: true,
            retain_iou: false,
            retain_meta: false,
        };
        let kernel = BboxIou;

        let seq = evaluate_with(&gt, &dt, params, ParityMode::Strict, &kernel).expect("seq");

        let acc_params = AccumulateParams {
            iou_thresholds: iou_thresholds(),
            recall_thresholds: recall_thresholds(),
            max_dets: &[1, 10, 100],
            n_categories: seq.n_categories,
            n_area_ranges: seq.n_area_ranges,
            n_images: seq.n_images,
        };
        let acc_seq =
            accumulate(&seq.eval_imgs(), acc_params, ParityMode::Strict).expect("acc seq");

        // Sweep across a few thread counts to catch any race or order-
        // dependence the par_iter scheduler might introduce.
        for n_threads in [2usize, 3, 4, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n_threads)
                .build()
                .expect("pool");
            let par = pool
                .install(|| evaluate_with_parallel(&gt, &dt, params, ParityMode::Strict, &kernel))
                .expect("par");
            let acc_par =
                accumulate(&par.eval_imgs(), acc_params, ParityMode::Strict).expect("acc par");
            assert_eq!(
                acc_seq.precision, acc_par.precision,
                "precision mismatch at n_threads={n_threads}"
            );
            assert_eq!(
                acc_seq.recall, acc_par.recall,
                "recall mismatch at n_threads={n_threads}"
            );
            assert_eq!(
                acc_seq.scores, acc_par.scores,
                "scores mismatch at n_threads={n_threads}"
            );
        }
    }

    #[test]
    fn f6_keypoints_use_cats_false_blackout_holds_in_parallel() {
        // Quirk F6 (corrected, ADR-0058): the strict-mode OKS blackout
        // under `useCats=0` is applied in the per-cell body, so it must
        // fire identically here and in `evaluate_with`. Assert both
        // directions against the sequential orchestrator.
        use std::collections::HashMap;

        use crate::similarity::OksSimilarity;

        let kps: Vec<f64> = (0..17)
            .flat_map(|i| [10.0 + f64::from(i), 10.0 + f64::from(i) * 2.0, 2.0])
            .collect();
        let images = vec![img(1, 100, 100)];
        let cats = vec![cat(1, "person"), cat(2, "animal")];
        let mut gt_ann = ann(1, 1, 1, (0.0, 0.0, 40.0, 80.0));
        gt_ann.area = 3200.0;
        gt_ann.keypoints = Some(kps.clone());
        gt_ann.num_keypoints = Some(17);
        let gt = CocoDataset::from_parts(images, vec![gt_ann], cats).expect("dataset");
        // DT carries the *other* category — only the `useCats=0`
        // collapse can bring it into the same cell as the GT.
        let mut dt_in = dt_input(1, 2, 0.99, (0.0, 0.0, 40.0, 80.0));
        dt_in.keypoints = Some(kps);
        dt_in.num_keypoints = Some(17);
        let dts = CocoDetections::from_inputs(vec![dt_in]).expect("detections");

        let area = AreaRange::keypoints_default();
        let params = EvaluateParams {
            iou_thresholds: iou_thresholds(),
            area_ranges: &area,
            max_dets_per_image: 20,
            use_cats: false,
            retain_iou: false,
            retain_meta: false,
        };
        let kernel = OksSimilarity::new(HashMap::new());
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("pool");

        for (mode, want_match) in [(ParityMode::Strict, false), (ParityMode::Corrected, true)] {
            let seq = evaluate_with(&gt, &dts, params, mode, &kernel).expect("seq");
            let par = pool
                .install(|| evaluate_with_parallel(&gt, &dts, params, mode, &kernel))
                .expect("par");
            let seq_cell = seq.cell(0, 0, 0).expect("seq cell");
            let par_cell = par.cell(0, 0, 0).expect("par cell");
            assert_eq!(
                seq_cell.dt_matched, par_cell.dt_matched,
                "parallel diverges from sequential in {mode:?} mode"
            );
            assert_eq!(
                par_cell.dt_matched.iter().all(|&m| m),
                want_match,
                "unexpected match outcome in {mode:?} mode"
            );
        }
    }

    /// `cell` / `category_cells` on a packed grid unpack exactly what the
    /// dense view holds, and agree with the same accessors on a grid
    /// assembled from that dense view.
    #[test]
    fn cell_accessors_agree_with_the_dense_view() {
        let (gt, dt) = tiny_grid_inputs();
        let area = AreaRange::coco_default();
        let params = EvaluateParams {
            iou_thresholds: iou_thresholds(),
            area_ranges: &area,
            max_dets_per_image: 100,
            use_cats: true,
            retain_iou: false,
            retain_meta: false,
        };
        let packed = evaluate_with(&gt, &dt, params, ParityMode::Strict, &BboxIou).expect("seq");
        let dense: Vec<Option<Box<PerImageEval>>> = packed.eval_imgs().into_owned();
        let (n_k, n_a, n_i) = (packed.n_categories, packed.n_area_ranges, packed.n_images);
        let dense_grid =
            EvalGrid::from_eval_imgs(dense.clone(), Vec::new(), false, n_k, n_a, n_i, None);
        assert!(
            dense.iter().any(Option::is_some),
            "fixture must populate cells"
        );

        let same = |x: &PerImageEval, y: &PerImageEval| {
            x.dt_scores == y.dt_scores
                && x.gt_ignore == y.gt_ignore
                && x.dt_matched == y.dt_matched
                && x.dt_ignore == y.dt_ignore
        };
        for grid in [&packed, &dense_grid] {
            for k in 0..n_k {
                for a in 0..n_a {
                    let expected: Vec<(usize, &PerImageEval)> = (0..n_i)
                        .filter_map(|i| dense[(k * n_a + a) * n_i + i].as_deref().map(|c| (i, c)))
                        .collect();
                    let walked: Vec<_> = grid.category_cells(k, a).collect();
                    assert_eq!(walked.len(), expected.len(), "(k={k}, a={a})");
                    for ((i, cell), (ei, ecell)) in walked.iter().zip(&expected) {
                        assert_eq!(i, ei);
                        assert!(same(cell, ecell), "category_cells (k={k}, a={a}, i={i})");
                    }
                    for i in 0..n_i {
                        match (
                            grid.cell(k, a, i),
                            dense[(k * n_a + a) * n_i + i].as_deref(),
                        ) {
                            (None, None) => {}
                            (Some(c), Some(e)) => {
                                assert!(same(&c, e), "cell (k={k}, a={a}, i={i})")
                            }
                            _ => panic!("presence mismatch at (k={k}, a={a}, i={i})"),
                        }
                    }
                }
            }
            assert!(grid.cell(n_k, 0, 0).is_none());
            assert!(grid.cell(0, n_a, 0).is_none());
            assert!(grid.cell(0, 0, n_i).is_none());
            assert_eq!(grid.category_cells(n_k, 0).count(), 0);
            assert_eq!(grid.category_cells(0, n_a).count(), 0);
        }

        // Both `accumulate_grid*` (the packed sweep) equal `accumulate`
        // over the dense view.
        let ladder = crate::accumulate::GridAccumulateParams {
            iou_thresholds: iou_thresholds(),
            recall_thresholds: recall_thresholds(),
            max_dets: &[1, 10, 100],
        };
        let via_slice =
            accumulate(&dense, ladder.for_grid(&packed), ParityMode::Strict).expect("slice");
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .expect("pool");
        for grid in [&packed, &dense_grid] {
            for via_grid in [
                crate::accumulate::accumulate_grid(grid, ladder, ParityMode::Strict).expect("grid"),
                pool.install(|| {
                    crate::accumulate::accumulate_grid_parallel(grid, ladder, ParityMode::Strict)
                })
                .expect("grid parallel"),
            ] {
                assert_eq!(via_grid.precision, via_slice.precision);
                assert_eq!(via_grid.recall, via_slice.recall);
                assert_eq!(via_grid.scores, via_slice.scores);
            }
        }

        // A hand-built grid whose cells do not cover its declared axes
        // walks as empty rather than panicking, like `cell()`.
        let short = EvalGrid::from_eval_imgs(vec![None; 3], Vec::new(), false, 2, 1, 3, None);
        assert_eq!(short.category_cells(1, 0).count(), 0);
        assert!(short.cell(1, 0, 0).is_none());
    }

    /// More than 16 IoU thresholds take the general `bool` flag layout
    /// instead of the 16-bit words: the empty-GT fill, the matched-cell
    /// gather, the parallel chunk concat and the accumulator sweep all
    /// have a separate arm for it. Pinned against `accumulate` over the
    /// dense view, sequential and parallel.
    #[test]
    fn general_flag_layout_agrees_with_the_dense_view() {
        let (gt, dt) = tiny_grid_inputs();
        let area = AreaRange::coco_default();
        let thresholds: Vec<f64> = (0..20).map(|t| 0.05 + 0.0475 * f64::from(t)).collect();
        let params = EvaluateParams {
            iou_thresholds: &thresholds,
            area_ranges: &area,
            max_dets_per_image: 100,
            use_cats: true,
            retain_iou: false,
            retain_meta: false,
        };
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .expect("pool");
        let seq = evaluate_with(&gt, &dt, params, ParityMode::Strict, &BboxIou).expect("seq");
        let par = pool
            .install(|| evaluate_with_parallel(&gt, &dt, params, ParityMode::Strict, &BboxIou))
            .expect("par");
        assert_eq!(seq.packed().map(|p| p.n_t), Some(20), "general layout");
        let dense: Vec<Option<Box<PerImageEval>>> = seq.eval_imgs().into_owned();
        assert!(
            dense.iter().any(Option::is_some),
            "fixture must populate cells"
        );
        assert_eq!(format!("{dense:?}"), format!("{:?}", par.eval_imgs()));

        let ladder = crate::accumulate::GridAccumulateParams {
            iou_thresholds: &thresholds,
            recall_thresholds: recall_thresholds(),
            max_dets: &[1, 10, 100],
        };
        let via_slice =
            accumulate(&dense, ladder.for_grid(&seq), ParityMode::Strict).expect("slice");
        for grid in [&seq, &par] {
            for via_grid in [
                crate::accumulate::accumulate_grid(grid, ladder, ParityMode::Strict).expect("grid"),
                pool.install(|| {
                    crate::accumulate::accumulate_grid_parallel(grid, ladder, ParityMode::Strict)
                })
                .expect("grid parallel"),
            ] {
                assert_eq!(via_grid.precision, via_slice.precision);
                assert_eq!(via_grid.recall, via_slice.recall);
                assert_eq!(via_grid.scores, via_slice.scores);
            }
        }
    }
}
