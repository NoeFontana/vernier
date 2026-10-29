//! Parallel sibling of [`crate::evaluate::evaluate_with`] (ADR-0047).
//!
//! The outer `(k, i)` cell loop is dispatched via `par_iter` against a
//! per-worker `CellScratch`; the calling thread folds the results
//! into pre-sized output `Vec`s at deterministic flat indices.
//!
//! Strict-mode bit-equality across thread counts holds by construction:
//! output writes go to deterministic `k*n_a*n_i + a*n_i + i` slots,
//! the matching kernel is pure on owned data, `accumulate()` runs
//! serially on the dense output, and the optional retained-IoU map is
//! built in canonical `(k, i)` order. The parity harness asserts this
//! property along the `num_threads` axis.
//!
//! The caller must `install` a `rayon::ThreadPool` around the call;
//! this module uses `par_iter` against the ambient pool.

use std::collections::{HashMap, HashSet};

use ndarray::Array2;
use rayon::prelude::*;

use crate::accumulate::{PackedCategory, PackedEvalGrid};
use crate::cell_occupancy;
use crate::dataset::{CategoryId, CocoDataset, CocoDetections, EvalDataset, ImageId, ImageMeta};
use crate::error::EvalError;
use crate::evaluate::{
    evaluate_category_slice, strict_oks_use_cats_blackout, CellScratch, EvalGrid, EvalImageMeta,
    EvalKernel, EvaluateParams, KernelScratch,
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
}

/// Subset-scoped sibling of [`evaluate_with_parallel`] (ADR-0066).
pub(crate) fn evaluate_subset_with_parallel<K: EvalKernel>(
    gt: &CocoDataset,
    dt: &CocoDetections,
    params: EvaluateParams<'_>,
    parity_mode: ParityMode,
    kernel: &K,
    image_subset: Option<&[usize]>,
) -> Result<EvalGrid, EvalError> {
    const CHUNK_IMAGES: usize = 512;

    let mut images: Vec<&ImageMeta> = gt.images().iter().collect();
    images.sort_unstable_by_key(|im| im.id.0);
    let n_i = images.len();
    let n_a = params.area_ranges.len();
    let n_t = params.iou_thresholds.len();

    let category_buckets: Vec<Option<CategoryId>> = if params.use_cats {
        let mut cats: Vec<_> = gt.categories().iter().map(|c| c.id).collect();
        cats.sort_unstable_by_key(|id| id.0);
        cats.into_iter().map(Some).collect()
    } else {
        vec![None]
    };
    let n_k = category_buckets.len();
    let builds_meta = params.builds_meta();

    let federated_per_image: Vec<Option<(&HashSet<CategoryId>, &HashSet<CategoryId>)>> =
        match (params.use_cats, gt.federated()) {
            (true, Some(fed)) => images
                .iter()
                .map(|im| {
                    let neg = fed.neg_category_ids.get(&im.id)?;
                    let nel = fed.not_exhaustive_category_ids.get(&im.id)?;
                    Some((neg, nel))
                })
                .collect(),
            _ => Vec::new(),
        };

    let strict_lvis_zero_area_filter =
        matches!(parity_mode, ParityMode::Strict) && gt.federated().is_some();
    let oks_use_cats_blackout = strict_oks_use_cats_blackout(kernel, params.use_cats, parity_mode);

    let gt_anns = gt.annotations();

    #[cfg(feature = "bench-timings")]
    let t_par = std::time::Instant::now();

    let image_ids: Vec<ImageId> = images.iter().map(|im| im.id).collect();
    let bucket_index = cell_occupancy::BucketIndex::new(&category_buckets);
    let occupied =
        cell_occupancy::by_category_subset(&image_ids, image_subset, n_k, gt, dt, &bucket_index);

    // Partition each category's occupied-image slice into `CHUNK_IMAGES`-wide
    // tasks so heavy categories (`person`) and `use_cats=false` (`n_k = 1`)
    // fan out across all workers without tail stragglers, while sparse
    // categories (`<= CHUNK_IMAGES`) produce a single zero-copy chunk.
    let mut chunk_tasks: Vec<(usize, &[u32])> = Vec::new();
    let mut cat_chunk_offsets: Vec<usize> = Vec::with_capacity(n_k + 1);
    cat_chunk_offsets.push(0);
    for k in 0..n_k {
        for chunk in occupied.row(k).chunks(CHUNK_IMAGES) {
            chunk_tasks.push((k, chunk));
        }
        cat_chunk_offsets.push(chunk_tasks.len());
    }

    struct ChunkOutput {
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
                    category_buckets[k],
                    img_chunk,
                    &images,
                    gt,
                    gt_anns,
                    dt,
                    params,
                    parity_mode,
                    kernel,
                    &federated_per_image,
                    strict_lvis_zero_area_filter,
                    oks_use_cats_blackout,
                    if builds_meta { Some(&mut meta) } else { None },
                    if params.retain_iou {
                        Some(&mut retained)
                    } else {
                        None
                    },
                )?;
                Ok::<ChunkOutput, EvalError>(ChunkOutput {
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

    let mut chunk_iter = chunk_outputs.into_iter();
    let mut per_cat_packed: Vec<Vec<PackedCategory>> = Vec::with_capacity(n_k);
    let mut per_cat_meta: Vec<Vec<(usize, Box<EvalImageMeta>)>> = if builds_meta {
        Vec::with_capacity(n_k)
    } else {
        Vec::new()
    };
    let mut retained_ious_map: Option<HashMap<(usize, usize), Array2<f64>>> = if params.retain_iou {
        Some(HashMap::new())
    } else {
        None
    };

    for k in 0..n_k {
        let count = cat_chunk_offsets[k + 1] - cat_chunk_offsets[k];
        let mut packed_chunks = Vec::with_capacity(count);
        let mut meta_cells = Vec::new();
        for _ in 0..count {
            if let Some(out) = chunk_iter.next() {
                packed_chunks.push(out.packed);
                if builds_meta {
                    meta_cells.extend(out.meta);
                }
                if let Some(map) = retained_ious_map.as_mut() {
                    // Chunks within category `k` are in strictly ascending `i`
                    // order, matching sequential insertion order directly.
                    for (i, mat) in out.retained {
                        map.insert((k, i), mat);
                    }
                }
            }
        }
        per_cat_packed.push(packed_chunks);
        if builds_meta {
            per_cat_meta.push(meta_cells);
        }
    }

    let packed_categories: Vec<PackedCategory> = per_cat_packed
        .into_par_iter()
        .map(|chunks| PackedCategory::concat_chunks(chunks, n_a, n_t))
        .collect();

    let mut eval_imgs_meta: Vec<Option<Box<EvalImageMeta>>> = Vec::new();
    if builds_meta {
        let chunk_len = n_a * n_i;
        let total_slots = n_k * chunk_len;
        eval_imgs_meta = vec![None; total_slots];
        if chunk_len > 0 {
            eval_imgs_meta
                .par_chunks_mut(chunk_len)
                .zip(per_cat_meta.into_par_iter())
                .for_each(|(meta_chunk, cells)| {
                    for (offset, meta) in cells {
                        meta_chunk[offset] = Some(meta);
                    }
                });
        }
    }

    let packed = PackedEvalGrid {
        categories: packed_categories,
        n_categories: n_k,
        n_area_ranges: n_a,
        n_images: n_i,
        n_t,
    };
    let grid = EvalGrid::from_packed(
        packed,
        eval_imgs_meta,
        builds_meta,
        retained_ious_map.map(crate::tables::RetainedIous::from_map),
    );

    #[cfg(feature = "bench-timings")]
    {
        let post_ns = u64::try_from(t_post.elapsed().as_nanos()).unwrap_or(u64::MAX);
        timings::COUNTERS.add(timings::PAR_ITER_NS, par_ns);
        timings::COUNTERS.add(timings::POST_PASS_NS, post_ns);
        timings::COUNTERS.bump(timings::N_CALLS);
    }
    Ok(grid)
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
    use crate::accumulate::{accumulate, AccumulateParams};
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
        assert_eq!(seq.eval_imgs().len(), par.eval_imgs().len());
        for (idx, (s, p)) in seq
            .eval_imgs()
            .iter()
            .zip(par.eval_imgs().iter())
            .enumerate()
        {
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
        let acc_seq = crate::accumulate::accumulate_grid(&seq, acc_params, ParityMode::Strict)
            .expect("acc seq");
        let acc_dense =
            accumulate(seq.eval_imgs(), acc_params, ParityMode::Strict).expect("acc dense");
        assert_eq!(acc_seq.precision, acc_dense.precision);
        assert_eq!(acc_seq.recall, acc_dense.recall);
        assert_eq!(acc_seq.scores, acc_dense.scores);

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
            let acc_par = pool
                .install(|| {
                    crate::accumulate::accumulate_grid_parallel(
                        &par,
                        acc_params,
                        ParityMode::Strict,
                    )
                })
                .expect("acc par");
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
}
