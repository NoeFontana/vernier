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

use ndarray::{Array2, ArrayView2, ArrayViewMut2};
use rayon::prelude::*;

use crate::accumulate::PerImageEval;
use crate::dataset::{CategoryId, CocoDataset, CocoDetections, EvalDataset, ImageMeta};
use crate::error::EvalError;
use crate::evaluate::{
    dt_top_indices_for_cell_into, evaluate_cell, gt_indices_for_cell, raw_dt_indices_for_cell,
    CellBuffers, CellScratch, EvalGrid, EvalImageMeta, EvalKernel, EvaluateParams, KernelScratch,
    COLLAPSED_CATEGORY_SENTINEL,
};
use crate::parity::ParityMode;

/// Wall-time split for [`evaluate_with_parallel`], gated on `bench-timings`.
#[cfg(feature = "bench-timings")]
pub(crate) mod timings {
    use std::sync::atomic::{AtomicU64, Ordering};

    pub(super) static PAR_ITER_NS: AtomicU64 = AtomicU64::new(0);
    pub(super) static SERIAL_POST_NS: AtomicU64 = AtomicU64::new(0);
    pub(super) static N_CALLS: AtomicU64 = AtomicU64::new(0);

    pub(crate) fn read_and_reset() -> (u64, u64, u64) {
        let p = PAR_ITER_NS.swap(0, Ordering::Relaxed);
        let s = SERIAL_POST_NS.swap(0, Ordering::Relaxed);
        let n = N_CALLS.swap(0, Ordering::Relaxed);
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
    let mut images: Vec<&ImageMeta> = gt.images().iter().collect();
    images.sort_unstable_by_key(|im| im.id.0);
    let n_i = images.len();
    let n_a = params.area_ranges.len();

    let category_buckets: Vec<Option<CategoryId>> = if params.use_cats {
        let mut cats: Vec<_> = gt.categories().iter().map(|c| c.id).collect();
        cats.sort_unstable_by_key(|id| id.0);
        cats.into_iter().map(Some).collect()
    } else {
        vec![None]
    };
    let n_k = category_buckets.len();

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

    let gt_anns = gt.annotations();
    let dt_anns = dt.detections();

    // Image-major working buffer: per-image rayon tasks own a
    // contiguous `n_k * n_a`-slot chunk via `par_chunks_mut`; transpose
    // to the canonical `(k, a, i)` layout once after the parallel region.
    #[cfg(feature = "bench-timings")]
    let t_par = std::time::Instant::now();

    let total_slots = n_k * n_a * n_i;
    let chunk_len = n_k * n_a;
    let mut working_eval: Vec<Option<Box<PerImageEval>>> = vec![None; total_slots];
    let mut working_meta: Vec<Option<Box<EvalImageMeta>>> = vec![None; total_slots];

    let retained_per_image: Vec<Vec<(usize, Array2<f64>)>> = working_eval
        .par_chunks_mut(chunk_len)
        .zip(working_meta.par_chunks_mut(chunk_len))
        .enumerate()
        .map_init(
            || {
                (
                    CellScratch::new(),
                    KernelScratch::<K::Annotation>::default(),
                )
            },
            |(scratch, kernel_scratch), (i, (eval_chunk, meta_chunk))| {
                let mut per_image_retained: Vec<(usize, Array2<f64>)> = Vec::new();
                for k in 0..n_k {
                    let eval_slice = &mut eval_chunk[k * n_a..(k + 1) * n_a];
                    let meta_slice = &mut meta_chunk[k * n_a..(k + 1) * n_a];
                    process_one_cell_into(
                        scratch,
                        kernel_scratch,
                        k,
                        i,
                        &images,
                        &category_buckets,
                        gt,
                        gt_anns,
                        dt,
                        dt_anns,
                        params,
                        parity_mode,
                        kernel,
                        &federated_per_image,
                        strict_lvis_zero_area_filter,
                        eval_slice,
                        meta_slice,
                        &mut per_image_retained,
                    )?;
                }
                Ok::<Vec<(usize, Array2<f64>)>, EvalError>(per_image_retained)
            },
        )
        .collect::<Result<Vec<_>, _>>()?;

    #[cfg(feature = "bench-timings")]
    let par_ns = u64::try_from(t_par.elapsed().as_nanos()).unwrap_or(u64::MAX);

    #[cfg(feature = "bench-timings")]
    let t_post = std::time::Instant::now();

    // Transpose image-major `(i, k, a)` → canonical `(k, a, i)`.
    let mut eval_imgs: Vec<Option<Box<PerImageEval>>> = vec![None; total_slots];
    let mut eval_imgs_meta: Vec<Option<Box<EvalImageMeta>>> = vec![None; total_slots];
    for i in 0..n_i {
        let base = i * chunk_len;
        for k in 0..n_k {
            for a in 0..n_a {
                let src = base + k * n_a + a;
                let dst = k * n_a * n_i + a * n_i + i;
                eval_imgs[dst] = working_eval[src].take();
                eval_imgs_meta[dst] = working_meta[src].take();
            }
        }
    }
    drop(working_eval);
    drop(working_meta);

    let mut retained_pairs: Vec<((usize, usize), Array2<f64>)> = Vec::new();
    for (i, per_image) in retained_per_image.into_iter().enumerate() {
        for (k, iou) in per_image {
            retained_pairs.push(((k, i), iou));
        }
    }

    // Sort by `(k, i)` before HashMap insertion so the final map is
    // built in the same order the sequential path would have used —
    // independent of par_iter completion order.
    let retained_ious_map: Option<HashMap<(usize, usize), Array2<f64>>> = if params.retain_iou {
        retained_pairs.sort_by_key(|&(key, _)| key);
        Some(retained_pairs.into_iter().collect())
    } else {
        None
    };

    let grid = EvalGrid {
        eval_imgs,
        eval_imgs_meta,
        n_categories: n_k,
        n_area_ranges: n_a,
        n_images: n_i,
        retained_ious: retained_ious_map.map(crate::tables::RetainedIous::from_map),
    };
    #[cfg(feature = "bench-timings")]
    {
        use std::sync::atomic::Ordering;
        let post_ns = u64::try_from(t_post.elapsed().as_nanos()).unwrap_or(u64::MAX);
        timings::PAR_ITER_NS.fetch_add(par_ns, Ordering::Relaxed);
        timings::SERIAL_POST_NS.fetch_add(post_ns, Ordering::Relaxed);
        timings::N_CALLS.fetch_add(1, Ordering::Relaxed);
    }
    Ok(grid)
}

/// Per-cell body. Writes per-area-range outputs directly into
/// `eval_slice` / `meta_slice` (length = `params.area_ranges.len()`);
/// pushes retained IoU into `retained_out` when enabled.
#[allow(clippy::too_many_arguments)]
fn process_one_cell_into<K: EvalKernel>(
    scratch: &mut CellScratch,
    kernel_scratch: &mut KernelScratch<K::Annotation>,
    k: usize,
    i: usize,
    images: &[&ImageMeta],
    category_buckets: &[Option<CategoryId>],
    gt: &CocoDataset,
    gt_anns: &[crate::dataset::CocoAnnotation],
    dt: &CocoDetections,
    dt_anns: &[crate::dataset::CocoDetection],
    params: EvaluateParams<'_>,
    parity_mode: ParityMode,
    kernel: &K,
    federated_per_image: &[Option<(&HashSet<CategoryId>, &HashSet<CategoryId>)>],
    strict_lvis_zero_area_filter: bool,
    eval_slice: &mut [Option<Box<PerImageEval>>],
    meta_slice: &mut [Option<Box<EvalImageMeta>>],
    retained_out: &mut Vec<(usize, Array2<f64>)>,
) -> Result<(), EvalError> {
    let cat = category_buckets[k];
    let category_id = cat.map_or(COLLAPSED_CATEGORY_SENTINEL, |c| c.0);
    let image = images[i];
    let image_id = image.id;

    let gt_indices_raw = gt_indices_for_cell(gt, image_id, cat);
    let gt_indices_buf: Vec<usize>;
    let gt_indices: &[usize] =
        if strict_lvis_zero_area_filter && gt_indices_raw.iter().any(|&j| gt_anns[j].area <= 0.0) {
            gt_indices_buf = gt_indices_raw
                .iter()
                .copied()
                .filter(|&j| gt_anns[j].area > 0.0)
                .collect();
            &gt_indices_buf
        } else {
            gt_indices_raw
        };
    let raw_dt_indices = raw_dt_indices_for_cell(dt, image_id, cat);
    if gt_indices.is_empty() && raw_dt_indices.is_empty() {
        return Ok(());
    }

    // AA4 cell-skip + AA3 `not_exhaustive` flag; `federated_per_image`
    // is `Some` exactly when federated semantics apply.
    let mut not_exhaustive_for_cell = false;
    if let (Some(c), Some(Some((neg_set, nel_set)))) = (cat, federated_per_image.get(i)) {
        if gt_indices.is_empty() && !neg_set.contains(&c) {
            return Ok(());
        }
        not_exhaustive_for_cell = nel_set.contains(&c);
    }

    dt_top_indices_for_cell_into(
        &mut scratch.dt_indices,
        &mut scratch.dt_score_buf,
        &mut scratch.dt_perm_buf,
        dt_anns,
        raw_dt_indices,
        params.max_dets_per_image,
    );

    scratch.gt_areas.clear();
    scratch
        .gt_areas
        .extend(gt_indices.iter().map(|&j| gt_anns[j].area));
    scratch.gt_iscrowd.clear();
    scratch
        .gt_iscrowd
        .extend(gt_indices.iter().map(|&j| gt_anns[j].is_crowd));
    scratch.gt_base_ignore.clear();
    scratch.gt_base_ignore.extend(
        gt_indices.iter().map(|&j| {
            gt_anns[j].effective_ignore(parity_mode) || kernel.extra_gt_ignore(&gt_anns[j])
        }),
    );
    scratch.gt_ids.clear();
    scratch
        .gt_ids
        .extend(gt_indices.iter().map(|&j| gt_anns[j].id.0));
    scratch.dt_areas.clear();
    scratch
        .dt_areas
        .extend(scratch.dt_indices.iter().map(|&j| dt_anns[j].area));
    scratch.dt_scores.clear();
    scratch
        .dt_scores
        .extend(scratch.dt_indices.iter().map(|&j| dt_anns[j].score));
    scratch.dt_ids.clear();
    scratch
        .dt_ids
        .extend(scratch.dt_indices.iter().map(|&j| dt_anns[j].id.0));

    #[cfg(feature = "bench-timings")]
    {
        crate::evaluate::build_anns_count::bump();
        crate::evaluate::build_anns_count::bump();
    }
    kernel.build_gt_anns_into(&mut kernel_scratch.gt, gt_anns, gt_indices, image)?;
    kernel.build_dt_anns_into(
        &mut kernel_scratch.dt,
        dt_anns,
        &scratch.dt_indices,
        image,
        parity_mode,
    )?;

    let g = kernel_scratch.gt.len();
    let d = kernel_scratch.dt.len();
    scratch.iou_buf.clear();
    scratch.iou_buf.resize(g * d, 0.0);
    if g > 0 && d > 0 {
        let mut iou_view =
            ArrayViewMut2::from_shape((g, d), &mut scratch.iou_buf[..]).map_err(|e| {
                EvalError::DimensionMismatch {
                    detail: format!("iou scratch view: {e}"),
                }
            })?;
        kernel.compute(&kernel_scratch.gt, &kernel_scratch.dt, &mut iou_view)?;
    }

    let iou_view = ArrayView2::from_shape((g, d), &scratch.iou_buf[..]).map_err(|e| {
        EvalError::DimensionMismatch {
            detail: format!("iou scratch view: {e}"),
        }
    })?;
    let buffers = CellBuffers {
        image_id: image_id.0,
        category_id,
        max_det: params.max_dets_per_image,
        gt_areas: &scratch.gt_areas,
        gt_iscrowd: &scratch.gt_iscrowd,
        gt_base_ignore: &scratch.gt_base_ignore,
        gt_ids: &scratch.gt_ids,
        dt_areas: &scratch.dt_areas,
        dt_scores: &scratch.dt_scores,
        dt_ids: &scratch.dt_ids,
        iou: iou_view,
        not_exhaustive: not_exhaustive_for_cell,
    };

    for (a, area) in params.area_ranges.iter().enumerate() {
        let (cell, meta) = evaluate_cell(
            &mut scratch.gt_ignore_buf,
            &buffers,
            area,
            params.iou_thresholds,
            parity_mode,
        )?;
        eval_slice[a] = Some(Box::new(cell));
        meta_slice[a] = Some(Box::new(meta));
    }

    if params.retain_iou {
        let cloned = Array2::from_shape_vec((g, d), scratch.iou_buf.clone()).map_err(|e| {
            EvalError::DimensionMismatch {
                detail: format!("retained iou clone: {e}"),
            }
        })?;
        retained_out.push((k, cloned));
    }

    Ok(())
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
        assert_eq!(seq.eval_imgs.len(), par.eval_imgs.len());
        for (idx, (s, p)) in seq.eval_imgs.iter().zip(par.eval_imgs.iter()).enumerate() {
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
        let acc_seq = accumulate(&seq.eval_imgs, acc_params, ParityMode::Strict).expect("acc seq");

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
                accumulate(&par.eval_imgs, acc_params, ParityMode::Strict).expect("acc par");
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
}
