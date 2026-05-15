//! Per-class matching and LRP additive decomposition.
//!
//! For one class this module:
//!
//! 1. Walks the per-`(category, image)` retained IoU matrices, doing
//!    its own greedy matching at `tp_threshold` (not the matching
//!    engine's threshold ladder — LRP only ever runs at a single
//!    IoU floor; the tau search is over confidence, not IoU).
//! 2. Concatenates the per-image `(dt_score, dt_matched, dt_ignore,
//!    dt_iou)` arrays into per-class arrays in the score order of
//!    the per-cell matching pass.
//! 3. Filters out ignore detections (matched-to-crowd or
//!    matched-to-ignore-GT) before handing the arrays to
//!    `super::tau_search::search_tau`.
//! 4. Decomposes the tau-search result into the three additive
//!    components per the paper's eq. 10:
//!
//!    - `oLRP_Loc = sum_TP*(1 - IoU) / NTP* / (1 - tp_threshold)`
//!    - `oLRP_FP  = NFP* / (NTP* + NFP*)`
//!    - `oLRP_FN  = NFN* / (NTP* + NFN*)`
//!
//! The matching loop is a literal transcription of the oracle's
//! `_match_per_class` (in `tests/python/oracle/lrp/oracle.py`):
//! score-descending traversal, prefer non-crowd GTs first, fall back
//! to crowd GTs (matched→ignore), one GT consumed per non-crowd
//! match.
//!
//! ## What this is *not*
//!
//! This module does NOT reuse `crate::matching::match_image`. The
//! AP-side matcher operates over an IoU-threshold ladder and produces
//! a `(T, D)` matched-array shape that the LRP tau search would have
//! to flatten back to a single threshold. Doing our own single-
//! threshold pass is both simpler (the loop is ~20 lines) and more
//! faithful to the oracle the metric is validated against.

use std::collections::HashSet;

use ndarray::ArrayView2;

use crate::dataset::{Annotation, CategoryId, CocoDataset, CocoDetections, EvalDataset, ImageMeta};
use crate::error::EvalError;
use crate::evaluate::{evaluate_with, EvalGrid, EvalKernel, EvaluateParams};
use crate::parity::ParityMode;
use crate::tables::RetainedIous;

use super::params::LrpParams;
use super::tau_search::{search_tau, TauSearchResult};

/// One class's contribution to the LRP report.
///
/// Mirrors [`super::LrpPerClass`] but in the internal coordinate
/// system: the orchestrator at module-top translates the
/// `category_index` back to the COCO `category_id`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PerClassDecomposition {
    /// Position of this class in the K-axis (id-ascending order). The
    /// orchestrator turns this into a `category_id`.
    pub category_index: usize,
    /// Per-class oLRP, or `1.0` for an all-FN class. `None` flags a
    /// class with no positive (non-crowd / non-ignore) GTs — the
    /// orchestrator emits `NaN` to the user.
    pub olrp: Option<f64>,
    /// Per-class `oLRP_Loc`. `None` when the class has no TPs at any
    /// tau, or no positive GTs.
    pub olrp_loc: Option<f64>,
    /// Per-class `oLRP_FP`. `None` when the class has no TPs at any
    /// tau and no FPs either, or no positive GTs.
    pub olrp_fp: Option<f64>,
    /// Per-class `oLRP_FN`. `None` when the class has no positive
    /// GTs.
    pub olrp_fn: Option<f64>,
    /// Per-class optimal tau. `None` when the class has no TPs at any
    /// tau, or no positive GTs.
    pub tau: Option<f64>,
}

/// Run the per-class matching + tau search + decomposition for one
/// category index `k` on a pre-computed [`crate::evaluate::EvalGrid`]
/// retaining IoUs.
///
/// `retained` carries the per-`(k, i)` IoU matrices keyed in the same
/// coordinate system the [`crate::evaluate::EvalGrid`] uses. `meta`
/// is the per-cell metadata (gt ids, dt ids, dt scores) parallel to
/// the IoU matrix columns. The matching pass reads `is_crowd` and the
/// effective ignore flag from the original [`CocoDataset`] —
/// retained_ious does not carry these, by design (it is geometry
/// only).
///
/// `image_mask` opts into ADR-0046 partitioned LRP: when `Some(mask)`,
/// only images `i` where `mask[i]` is `true` contribute to the per-
/// class arrays and `n_pos_gt` count. The matching pass itself is not
/// re-run — partitioning is a filter on the post-match decompose
/// walk, exactly the C3 axiom AP partitioning honours.
///
/// Note: `image_mask` is a dense `Vec<bool>` of length `n_images`,
/// materialised once per slice by [`decompose_all_classes`]. A
/// `HashSet` lookup inside the `n_cats × n_slices × n_images` inner
/// loop showed ~30–75 s of probe overhead per partitioned LRP call at
/// LVIS scale (1203 × 5k × 256); the dense mask reduces that to a
/// branch-predictable array read.
#[allow(clippy::too_many_arguments)]
pub(crate) fn decompose_class(
    gt: &CocoDataset,
    k: usize,
    category_id: Option<CategoryId>,
    n_images: usize,
    image_order: &[&ImageMeta],
    retained: &RetainedIous,
    grid: &crate::evaluate::EvalGrid,
    parity_mode: ParityMode,
    params: &LrpParams<'_>,
    image_mask: Option<&[bool]>,
) -> Result<PerClassDecomposition, EvalError> {
    // Per-class concatenated arrays. We size to a per-class upper bound
    // (max_dets_per_image * n_images) so the inner loop never
    // reallocates.
    let cap = params.max_dets_per_image.saturating_mul(n_images);
    let mut dt_score: Vec<f64> = Vec::with_capacity(cap);
    let mut dt_matched: Vec<bool> = Vec::with_capacity(cap);
    let mut dt_iou: Vec<f64> = Vec::with_capacity(cap);
    let mut n_pos_gt: u64 = 0;

    // Scratch buffers reused across every cell in this class. Avoids
    // 3 fresh Vec allocations per (image, category) cell — at
    // 5000 images x 80 cats that's ~1.2M allocations skipped per call.
    let mut gt_crowd: Vec<bool> = Vec::new();
    let mut gt_ignore_mask: Vec<bool> = Vec::new();
    let mut gt_taken: Vec<bool> = Vec::new();

    // A4-only ordering: per-class GTs are filtered + sorted by the
    // dataset-side gather. We just need the count of non-crowd /
    // non-ignore GTs and the crowd flags per cell.
    let gt_anns = gt.annotations();

    for (i, image) in image_order.iter().enumerate() {
        // ADR-0046 partitioned LRP: skip cells whose image index is
        // not in the slice. Filter applies uniformly to the n_pos_gt
        // count and the per-cell matching emission below — both must
        // be restricted to the slice so the per-class arrays and the
        // denominator are consistent.
        if let Some(mask) = image_mask {
            if !mask[i] {
                continue;
            }
        }
        let cell = match grid.cell(k, 0, i) {
            Some(c) => c,
            None => continue,
        };
        // cell_meta presence is a precondition for cell presence; skip
        // defensively so a streaming-evaluator desync can never panic
        // the LRP pass.
        if grid.cell_meta(k, 0, i).is_none() {
            continue;
        }

        // GT count and crowd flags. The orchestrator already filtered
        // ignore-GTs out of the matching, but for the n_pos_gt total
        // we need to count non-ignore, non-crowd GTs that were
        // actually in this (image, category) cell. PerImageEval's
        // gt_ignore array is in the matching engine's *sorted* order
        // (A4: ignore-asc); EvalImageMeta.gt_ids is in the *same*
        // sorted order (the matching engine permutes GTs by ignore
        // before packing). For each GT id in the meta, look up the
        // annotation to get its crowd flag — the matching engine's
        // ignore flag already folds crowd in (effective_ignore =
        // is_crowd in strict, or honors user's ignore in corrected).
        let image_id = image.id;
        let gt_indices = match category_id {
            Some(c) => gt.ann_indices_for(image_id, c),
            None => gt.ann_indices_for_image(image_id),
        };
        // Count "positive" GTs the LRP oracle counts: not-crowd,
        // not-ignore. The matching engine ignores crowd-and-ignore as
        // "ignore" via `effective_ignore`; we mirror that here.
        for &j in gt_indices {
            let ann = &gt_anns[j];
            if !ann.is_crowd() && !ann.effective_ignore(parity_mode) {
                n_pos_gt += 1;
            }
        }

        // IoU matrix for this cell. The matrix is shape (G, D) where
        // rows are GT in the matching engine's sorted order, cols are
        // DT in score-desc order. `dt_scores` is in the same DT
        // order; the GT id at sorted-row `r` is `meta.gt_ids[r]`.
        // Borrow (not pop) so the retained_ious store can serve a
        // second decompose pass — the ADR-0046 partitioned LRP path
        // re-walks the grid once per slice, all sharing one matching
        // pass (C3).
        let iou_view = match retained.get(k, i) {
            Some(v) => v,
            None => continue, // No DTs and/or no GTs in this cell.
        };

        run_cell_matching(
            &iou_view,
            cell,
            gt_indices,
            gt_anns,
            parity_mode,
            params.tp_threshold,
            &mut dt_score,
            &mut dt_matched,
            &mut dt_iou,
            &mut gt_crowd,
            &mut gt_ignore_mask,
            &mut gt_taken,
        );
    }

    if n_pos_gt == 0 {
        // No positive GTs in this class. Flag with NaN.
        return Ok(PerClassDecomposition {
            category_index: k,
            olrp: None,
            olrp_loc: None,
            olrp_fp: None,
            olrp_fn: None,
            tau: None,
        });
    }

    // Drop ignore detections before the tau search. The matching
    // pass above never emits them — matched-to-crowd / matched-to-
    // ignore-GT detections are skipped, so the arrays already contain
    // only TP/FP candidates.

    let search = search_tau(
        &dt_score,
        &dt_matched,
        &dt_iou,
        n_pos_gt,
        params.tp_threshold,
        params.tau_grid,
    );
    let result = match search {
        Some(r) => r,
        None => {
            return Ok(PerClassDecomposition {
                category_index: k,
                olrp: None,
                olrp_loc: None,
                olrp_fp: None,
                olrp_fn: None,
                tau: None,
            });
        }
    };

    Ok(decompose_at(result, params, k))
}

/// Decompose the tau-search result into the four reported values per
/// the paper's eq. 10.
fn decompose_at(
    result: TauSearchResult,
    params: &LrpParams<'_>,
    category_index: usize,
) -> PerClassDecomposition {
    let n_tp = result.stats.n_tp;
    let n_fp = result.stats.n_fp;
    let n_fn = result.stats.n_fn;
    let sum_loc = result.stats.sum_loc;
    let one_minus_tau_tp = if params.tp_threshold >= 1.0 {
        1.0
    } else {
        1.0 - params.tp_threshold
    };

    let tau = params.tau_grid.get(result.star).copied();

    if n_tp > 0 {
        let n_tp_f = n_tp as f64;
        let loc = (sum_loc / n_tp_f) / one_minus_tau_tp;
        let fp = if n_tp + n_fp > 0 {
            (n_fp as f64) / ((n_tp + n_fp) as f64)
        } else {
            0.0
        };
        let fn_rate = if n_tp + n_fn > 0 {
            (n_fn as f64) / ((n_tp + n_fn) as f64)
        } else {
            0.0
        };
        PerClassDecomposition {
            category_index,
            olrp: Some(result.lrp),
            olrp_loc: Some(loc),
            olrp_fp: Some(fp),
            olrp_fn: Some(fn_rate),
            tau,
        }
    } else {
        // Degenerate: no TPs at the optimal tau. Per oracle, loc /
        // fp_rate are NaN-undefined, fn_rate is 1.0 (all positives
        // missed), tau is unreported. olrp itself is well-defined
        // (= 1.0 in the all-FN case).
        PerClassDecomposition {
            category_index,
            olrp: Some(result.lrp),
            olrp_loc: None,
            olrp_fp: if n_fp > 0 { None } else { Some(0.0) },
            olrp_fn: Some(1.0),
            tau: None,
        }
    }
}

/// Greedy matching for one `(image, category)` cell using the
/// retained IoU matrix.
///
/// Score-descending traversal of detections, preferring non-crowd
/// non-ignore GTs first, falling back to crowd-or-ignore GTs (matched
/// → drop). The oracle's algorithm verbatim.
///
/// Output: appends one entry per detection to `dt_score`,
/// `dt_matched`, `dt_iou`. Ignore detections (matched to crowd or to
/// an ignore-GT) are *dropped* from the arrays — they don't count as
/// TP or FP in the LRP scan and adding them only to filter them
/// downstream would cost memory.
#[allow(clippy::too_many_arguments)]
fn run_cell_matching(
    iou_mat: &ArrayView2<'_, f64>,
    cell: &crate::accumulate::PerImageEval,
    gt_indices: &[usize],
    gt_anns: &[crate::dataset::CocoAnnotation],
    parity_mode: ParityMode,
    tp_threshold: f64,
    dt_score: &mut Vec<f64>,
    dt_matched: &mut Vec<bool>,
    dt_iou: &mut Vec<f64>,
    gt_crowd: &mut Vec<bool>,
    gt_ignore_mask: &mut Vec<bool>,
    gt_taken: &mut Vec<bool>,
) {
    // The IoU matrix has shape (G, D) where G is the un-sorted gather
    // order (gt_indices) and D is the score-desc sorted DT order.
    // EvalImageMeta.gt_ids is in the matching engine's *sorted*
    // (ignore-asc) order; we walk gt_indices directly (the IoU
    // matrix's row axis) and pull crowd/ignore per row.
    let g = iou_mat.nrows();
    let d = iou_mat.ncols();

    // Refresh the three scratch buffers in place — caller owns the
    // backing allocations, we reuse them across all cells in the class.
    gt_crowd.clear();
    gt_ignore_mask.clear();
    for &j in gt_indices {
        let a = &gt_anns[j];
        gt_crowd.push(a.is_crowd());
        gt_ignore_mask.push(a.effective_ignore(parity_mode));
    }
    debug_assert_eq!(gt_crowd.len(), g);
    debug_assert_eq!(gt_ignore_mask.len(), g);

    gt_taken.clear();
    gt_taken.resize(g, false);

    // DT scores in the column-axis order (= score-desc, the same as
    // PerImageEval.dt_scores).
    let scores = &cell.dt_scores;
    debug_assert_eq!(scores.len(), d);

    // Walk detections in score-desc order (= column order).
    for k in 0..d {
        // First pass: best non-crowd, non-ignore GT >= tp_threshold.
        let mut best_iou = tp_threshold;
        let mut best_g: Option<usize> = None;
        for j in 0..g {
            if gt_taken[j] {
                continue;
            }
            if gt_crowd[j] || gt_ignore_mask[j] {
                continue;
            }
            let v = iou_mat[(j, k)];
            if v >= best_iou {
                best_iou = v;
                best_g = Some(j);
            }
        }
        if let Some(j) = best_g {
            gt_taken[j] = true;
            dt_score.push(scores[k]);
            dt_matched.push(true);
            dt_iou.push(best_iou);
            continue;
        }
        // Second pass: crowd/ignore GTs (matched → drop, do not emit
        // an entry; these never count as TP or FP).
        let mut best_iou_alt = tp_threshold;
        let mut hit_ignore = false;
        for j in 0..g {
            if !(gt_crowd[j] || gt_ignore_mask[j]) {
                continue;
            }
            let v = iou_mat[(j, k)];
            if v >= best_iou_alt {
                best_iou_alt = v;
                hit_ignore = true;
            }
        }
        if hit_ignore {
            // Dropped — matched to crowd/ignore. No emit.
            continue;
        }
        // Unmatched detection. Emit as FP candidate.
        dt_score.push(scores[k]);
        dt_matched.push(false);
        dt_iou.push(0.0);
    }
}

/// Pre-computed state shared across one (overall + N slices)
/// partitioned LRP pass.
///
/// Built by [`prepare_lrp_pass`] in one matching pass (the C3 axiom of
/// ADR-0046) and re-used by [`decompose_all_classes`] for every
/// `image_filter` value: `None` for the overall report and
/// `Some(&slice.image_indices)` for each slice. The matching engine
/// is invoked exactly once per partitioned LRP call regardless of how
/// many slices the user requested.
pub(crate) struct LrpPassContext<'gt> {
    /// Borrowed GT dataset; used for the post-match annotation
    /// lookups (crowd/ignore flags + n_pos_gt counting).
    pub(crate) gt: &'gt CocoDataset,
    /// Per-`(category_index, image_index)` IoU matrices retained from
    /// the matching pass.
    pub(crate) retained: RetainedIous,
    /// Owned `EvalGrid` post-`retained_ious.take()`; carries the
    /// `eval_imgs` / `eval_imgs_meta` slabs the decompose pass reads.
    pub(crate) grid: EvalGrid,
    /// Image-metadata views in the id-ascending order the matching
    /// pass's I-axis uses. Indexing this slice by `i` reproduces the
    /// grid's `(k, a, i)` coordinate.
    pub(crate) image_order: Vec<&'gt ImageMeta>,
    /// Category id bucket per K-axis position; `None` collapses to a
    /// single class-agnostic bucket when `use_cats = false`.
    pub(crate) category_buckets: Vec<Option<CategoryId>>,
}

/// Run the matching pass once and surface the per-class decompose
/// context. Invoked once per `optimal_lrp_*` call — including the
/// partitioned path, where it serves N+1 decompose walks (overall
/// plus one per slice) from a single matching pass.
pub(crate) fn prepare_lrp_pass<'gt, K: EvalKernel>(
    gt: &'gt CocoDataset,
    dt: &CocoDetections,
    kernel: &K,
    params: &LrpParams<'_>,
    parity_mode: ParityMode,
) -> Result<LrpPassContext<'gt>, EvalError> {
    // Force `retain_iou` on so the per-cell matrices land on
    // `EvalGrid.retained_ious`. Per ADR-0043 this is engine-internal
    // — the LRP user never sees the flag.
    let eval_params = EvaluateParams {
        iou_thresholds: params.iou_thresholds,
        area_ranges: params.area_ranges,
        max_dets_per_image: params.max_dets_per_image,
        use_cats: params.use_cats,
        retain_iou: true,
    };
    let mut grid = evaluate_with(gt, dt, eval_params, parity_mode, kernel)?;

    // Lift the retained-IoU store out of the grid so the context owns
    // it. The store is borrowed (not consumed) by each decompose pass
    // — the partition path runs N+1 passes against the same store.
    let retained = grid
        .retained_ious
        .take()
        .ok_or_else(|| EvalError::InvalidConfig {
            detail: "lrp: evaluate_with returned no retained_ious despite retain_iou=true".into(),
        })?;

    // Reconstruct the same image / category orderings the
    // evaluator's K and I axes use, so the (k, i) keys on retained
    // line up with what we walk here.
    let mut images: Vec<&ImageMeta> = gt.images().iter().collect();
    images.sort_unstable_by_key(|im| im.id.0);

    let category_buckets: Vec<Option<CategoryId>> = if params.use_cats {
        let mut cats: Vec<_> = gt.categories().iter().map(|c| c.id).collect();
        cats.sort_unstable_by_key(|c| c.0);
        cats.into_iter().map(Some).collect()
    } else {
        vec![None]
    };
    debug_assert_eq!(category_buckets.len(), grid.n_categories);

    Ok(LrpPassContext {
        gt,
        retained,
        grid,
        image_order: images,
        category_buckets,
    })
}

/// Walk the per-class decompositions over an already-prepared
/// [`LrpPassContext`], optionally filtered to a slice's image set.
///
/// `image_filter == None` reproduces the un-partitioned LRP shape;
/// `image_filter == Some(&set)` is the ADR-0046 partitioned LRP per-
/// slice walk.
pub(crate) fn decompose_all_classes(
    ctx: &LrpPassContext<'_>,
    parity_mode: ParityMode,
    params: &LrpParams<'_>,
    image_filter: Option<&HashSet<usize>>,
) -> Result<Vec<PerClassDecomposition>, EvalError> {
    let n_images = ctx.image_order.len();

    // Materialise the (sparse) HashSet filter into a dense Vec<bool>
    // once per pass. The per-class loop visits every image and the
    // HashSet probe in the inner loop dominates at LVIS scale; a
    // contiguous Vec is branch-predictable and ~5–10× faster on the
    // hot path (`decompose_class` docstring).
    let mask_storage: Option<Vec<bool>> = image_filter.map(|set| {
        let mut m = vec![false; n_images];
        for &i in set {
            if i < n_images {
                m[i] = true;
            }
        }
        m
    });
    let image_mask = mask_storage.as_deref();

    let mut out: Vec<PerClassDecomposition> = Vec::with_capacity(ctx.category_buckets.len());
    for (k, cat) in ctx.category_buckets.iter().enumerate() {
        let d = decompose_class(
            ctx.gt,
            k,
            *cat,
            n_images,
            &ctx.image_order,
            &ctx.retained,
            &ctx.grid,
            parity_mode,
            params,
            image_mask,
        )?;
        out.push(d);
    }
    Ok(out)
}
