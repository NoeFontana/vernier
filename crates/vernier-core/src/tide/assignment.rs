//! Bin assignment for TIDE error decomposition.
//!
//! Walks every detection that survives the per-image `max_dets` cap and
//! assigns it to one of the six TIDE bins (Cls / Loc / Both / Dupe / Bkg
//! plus the `Tp`/`Ignore` non-FP labels). Walks every non-ignore GT and
//! flags those without a same-class match at `t_f` as Missed.
//!
//! The algorithm mirrors `tests/python/oracle/tide/oracle.py::
//! _attribute_bins` exactly — that file is the spec per ADR-0021 and
//! the Rust output is correct iff `|delta_rust − delta_oracle| < 1e-9`
//! per bin per fixture (see `tests/tide_oracle_parity.rs`).
//!
//! ## Inputs
//!
//! - `gt` / `dt` are the source dataset and detection list.
//! - `cross_class` carries the un-class-filtered per-image IoU matrices
//!   from the orchestrator-level side pass (ADR-0023). Rows index DTs in
//!   the same per-image score-desc order [`crate::evaluate::
//!   dt_top_indices_for_cell`] uses; columns index GTs in dataset
//!   insertion order. We read it for `iou_same` / `iou_cross` to
//!   sidestep recomputing the kernel.
//! - `params` supplies `t_f` / `t_b` / `max_dets_per_image` / `use_cats`.
//!
//! ## Output
//!
//! [`BinAssignment`] carries a per-`(image_id, dt_input_idx)` label
//! plus the per-bin `target_gt_local_idx` the rewrite layer needs:
//! the wrong-class GT for Cls, the same-class GT for Loc. For Dupe /
//! Bkg / Both / Missed the rewrite layer needs no extra payload (drop
//! the DT, or flip the GT's ignore flag).

use std::collections::HashMap;

use ndarray::Array2;

use crate::dataset::{
    CategoryId, CocoAnnotation, CocoDataset, CocoDetection, CocoDetections, EvalDataset, ImageId,
};
use crate::error::EvalError;
use crate::evaluate::dt_top_indices_for_cell;
use crate::matching::{match_image, MatchResult};
use crate::parity::ParityMode;
use crate::tables::CrossClassIous;

use super::params::TideParams;

/// One detection's TIDE label at `t_f`, plus the rewrite-layer target
/// (when the bin's correction needs one).
///
/// `target_gt_local_idx` indexes into the **per-image** GT list in
/// dataset insertion order — the same axis [`CrossClassIous::gt_classes`]
/// uses as columns. Its meaning depends on `bin`:
///
/// - `Cls`  — index of the wrong-class GT to relabel onto.
/// - `Loc`  — index of the same-class GT to snap the bbox to.
/// - any other bin — meaningless (`-1`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DtBinLabel {
    /// The TIDE bin (or non-FP label).
    pub bin: DtBin,
    /// Per-image local GT index used by the `Cls` / `Loc` corrections;
    /// `-1` for bins that need no target.
    pub target_gt_local_idx: i32,
}

/// Per-detection TIDE label, including the two non-FP labels.
///
/// The TP and Ignore labels are not in [`super::TideErrorBin`] (which
/// only enumerates the six error bins) — they live here because the
/// bin-assignment loop needs to know that "this DT was a true positive,
/// no rewrite needed" or "this DT matched only an ignore-GT, not
/// counted as an FP". See `oracle.py:466-471`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DtBin {
    /// True positive: matched a non-ignore GT at `t_f`.
    Tp,
    /// Matched only an ignore-GT (e.g. `iscrowd=1`) at `t_f`. Excluded
    /// from FP accounting.
    Ignore,
    /// Cls error — wrong-class GT overlaps at IoU `>= t_f`.
    Cls,
    /// Loc error — same-class GT overlaps at IoU `∈ [t_b, t_f)`, and
    /// the same-class IoU is at least the cross-class IoU.
    Loc,
    /// Both error — wrong-class GT overlaps at IoU `∈ [t_b, t_f)`,
    /// and the same-class IoU did not reach `t_b` (or was lower).
    Both,
    /// Dupe error — same-class GT overlaps at IoU `>= t_f` but a
    /// higher-scoring same-class DT already claimed it.
    Dupe,
    /// Bkg error — best IoU against any GT is `< t_b`.
    Bkg,
}

/// Bin-assignment output for one TIDE call.
///
/// Two flat maps keyed by `(image_id, dt_input_idx)` and `(image_id,
/// gt_input_idx)` respectively. The `dt_input_idx` is the index into
/// the input [`CocoDetections`] (the auto-incrementing position the
/// detection originally occupied in the input list); the
/// `gt_input_idx` is the index into the input [`CocoDataset`]
/// annotations (also dataset insertion order).
///
/// Both indices are stable across the rewrite layer's per-bin calls:
/// the rewrite rebuilds detections preserving these positions so the
/// targets stay valid.
#[derive(Debug, Default, Clone)]
pub struct BinAssignment {
    /// `(image_id, dt_input_idx)` → bin label. DTs evicted by the
    /// per-image `max_dets` cap are absent (mirrors the oracle's
    /// `attribution.get(d.dt_idx)` returning `None` for evicted DTs).
    pub dt_labels: HashMap<(i64, usize), DtBinLabel>,
    /// `(image_id, gt_input_idx)` for non-ignore GTs unmatched by any
    /// same-class DT at `t_f`. The Missed correction marks these as
    /// `ignore=true` in the rewrite step.
    pub missed_gts: Vec<(i64, usize)>,
}

/// Walk every image and assign TIDE bins per the oracle's algorithm.
///
/// ## Algorithm
///
/// For each image:
///
/// 1. Apply the per-image score-desc `max_dets_per_image` cap (quirk
///    **A1**) to the DTs. Evicted DTs are not labelled.
/// 2. For each category present on the image, run a greedy match
///    (matching engine, non-cross-class) at `t_f` against that
///    category's same-class same-image GTs. Track `gt_taken_by` and
///    per-DT `matched` / `ignore` status at `t_f`.
/// 3. For each surviving DT, look up its `iou_same` and `iou_cross`
///    from the cross-class side-pass storage and apply the priority
///    decision (`oracle.py:496-531`):
///    - `Tp` if matched at `t_f` to a non-ignore GT.
///    - `Ignore` if matched only to an ignore-GT.
///    - else FP, with the priority chain
///      `Dupe → Cls → Loc → Both → Bkg`.
/// 4. For each non-ignore GT, mark it Missed iff no same-class DT
///    matched it at `t_f` (per `oracle.py:533-547`).
///
/// # Errors
///
/// Propagates [`EvalError`] from [`match_image`] (only on dimension
/// mismatch — kernel work is already done by the time we get here).
pub fn assign_bins(
    gt: &CocoDataset,
    dt: &CocoDetections,
    cross_class: &CrossClassIous,
    params: &TideParams<'_>,
) -> Result<BinAssignment, EvalError> {
    let mut images: Vec<&crate::dataset::ImageMeta> = gt.images().iter().collect();
    images.sort_unstable_by_key(|im| im.id.0);

    let mut out = BinAssignment::default();
    let gt_anns = gt.annotations();
    let dt_anns = dt.detections();

    // Map (image_id, gt_input_idx) → per-image local column index used
    // by CrossClassIous. Same ordering used by the side pass:
    // `gt.ann_indices_for_image(image_id)` (dataset insertion order
    // for that image).
    for (image_idx, image) in images.iter().enumerate() {
        assign_bins_for_image(
            image_idx,
            image.id,
            gt,
            dt,
            cross_class,
            params,
            gt_anns,
            dt_anns,
            &mut out,
        )?;
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn assign_bins_for_image(
    image_idx: usize,
    image_id: ImageId,
    gt: &CocoDataset,
    dt: &CocoDetections,
    cross_class: &CrossClassIous,
    params: &TideParams<'_>,
    gt_anns: &[CocoAnnotation],
    dt_anns: &[CocoDetection],
    out: &mut BinAssignment,
) -> Result<(), EvalError> {
    // Per-image GT list in the same column ordering CrossClassIous uses.
    // `compute_cross_class_ious` calls `gt.ann_indices_for_image(image_id)`,
    // which returns indices in dataset-insertion order (HashMap insertion
    // order is irrelevant — the by_image map's Vec preserves the order
    // the dataset was constructed in).
    let gt_local_indices: &[usize] = gt.ann_indices_for_image(image_id);
    // DT list capped + sorted in score-desc order, again matching the
    // side pass's ordering for row indexing.
    let dt_local_indices = dt_top_indices_for_cell(dt, image_id, None, params.max_dets_per_image);

    if gt_local_indices.is_empty() && dt_local_indices.is_empty() {
        return Ok(());
    }

    let cross = cross_class.get(image_idx);

    // For each DT in the cap-applied list, its row index in the cross
    // matrix is its position in `dt_local_indices` (the side pass walks
    // the same `dt_top_indices_for_cell` output).
    // For each GT in `gt_local_indices`, its column index in the cross
    // matrix is its position in `gt_local_indices`.

    // 1. Per-class same-class greedy match at t_f. Track per-DT match
    //    status and per-GT-local-column "taken_by" map.
    //    The categories we iterate over are the ids actually present on
    //    the image (matches the oracle's `cats_in_image` set).
    let mut per_dt_matched: HashMap<usize, bool> = HashMap::new();
    let mut per_dt_ignore: HashMap<usize, bool> = HashMap::new();
    // `gt_taken_by[gt_local_col_idx] = dt_input_idx` — same-class match
    // took this GT at t_f. Used for Missed attribution and not for
    // Dupe (Dupe is geometric: iou_same >= t_f).
    let mut gt_taken_by: HashMap<usize, usize> = HashMap::new();

    let cats_in_image: Vec<CategoryId> = if params.use_cats {
        let mut cats: Vec<CategoryId> = gt_local_indices
            .iter()
            .map(|&j| gt_anns[j].category_id)
            .chain(dt_local_indices.iter().map(|&j| dt_anns[j].category_id))
            .collect();
        cats.sort_unstable_by_key(|c| c.0);
        cats.dedup();
        cats
    } else {
        // L4: collapse — single virtual category.
        vec![CategoryId(crate::evaluate::COLLAPSED_CATEGORY_SENTINEL)]
    };

    for cat in cats_in_image {
        same_class_match_one_category(
            &gt_local_indices_with_pos(gt_local_indices, gt_anns, cat, params.use_cats),
            &dt_local_indices_with_pos(&dt_local_indices, dt_anns, cat, params.use_cats),
            gt_anns,
            dt_anns,
            params,
            &mut per_dt_matched,
            &mut per_dt_ignore,
            &mut gt_taken_by,
        )?;
    }

    // 2. Per-DT bin label using the cross-class side-pass IoU.
    for (row_idx, &dt_input_idx) in dt_local_indices.iter().enumerate() {
        let dt = &dt_anns[dt_input_idx];
        let key = (image_id.0, dt_input_idx);

        if per_dt_ignore.get(&dt_input_idx).copied().unwrap_or(false) {
            out.dt_labels.insert(
                key,
                DtBinLabel {
                    bin: DtBin::Ignore,
                    target_gt_local_idx: -1,
                },
            );
            continue;
        }
        if per_dt_matched.get(&dt_input_idx).copied().unwrap_or(false) {
            out.dt_labels.insert(
                key,
                DtBinLabel {
                    bin: DtBin::Tp,
                    target_gt_local_idx: -1,
                },
            );
            continue;
        }

        // FP: compute iou_same / iou_cross from the side pass.
        let (iou_same, best_same_col, iou_cross, best_cross_col) = best_same_and_cross(
            row_idx,
            dt.category_id,
            cross,
            gt_local_indices,
            gt_anns,
            params.use_cats,
        );

        let label = pick_bin(
            iou_same,
            best_same_col,
            iou_cross,
            best_cross_col,
            params.t_f,
            params.t_b,
        );
        out.dt_labels.insert(key, label);
    }

    // 3. Missed: non-ignore GTs not in `gt_taken_by`.
    for (col_idx, &gt_input_idx) in gt_local_indices.iter().enumerate() {
        let g = &gt_anns[gt_input_idx];
        // Use the same effective_ignore semantics the matching path uses
        // (D1) — Strict + Corrected both fold iscrowd into ignore.
        if g.is_crowd || g.ignore_flag.unwrap_or(false) {
            continue;
        }
        if gt_taken_by.contains_key(&col_idx) {
            continue;
        }
        out.missed_gts.push((image_id.0, gt_input_idx));
    }

    Ok(())
}

/// Build a `(local_col_idx, gt_input_idx)` list for one category, where
/// `local_col_idx` matches the cross-class column ordering.
fn gt_local_indices_with_pos(
    gt_local_indices: &[usize],
    gt_anns: &[CocoAnnotation],
    cat: CategoryId,
    use_cats: bool,
) -> Vec<(usize, usize)> {
    gt_local_indices
        .iter()
        .enumerate()
        .filter(|&(_, &gi)| !use_cats || gt_anns[gi].category_id == cat)
        .map(|(col, &gi)| (col, gi))
        .collect()
}

/// Build a `(row_idx, dt_input_idx)` list for one category. `row_idx`
/// matches the cross-class row ordering.
fn dt_local_indices_with_pos(
    dt_local_indices: &[usize],
    dt_anns: &[CocoDetection],
    cat: CategoryId,
    use_cats: bool,
) -> Vec<(usize, usize)> {
    dt_local_indices
        .iter()
        .enumerate()
        .filter(|&(_, &di)| !use_cats || dt_anns[di].category_id == cat)
        .map(|(row, &di)| (row, di))
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn same_class_match_one_category(
    gts_in_cat: &[(usize, usize)], // (col_idx in cross matrix, gt_input_idx)
    dts_in_cat: &[(usize, usize)], // (row_idx in cross matrix, dt_input_idx)
    gt_anns: &[CocoAnnotation],
    dt_anns: &[CocoDetection],
    params: &TideParams<'_>,
    per_dt_matched: &mut HashMap<usize, bool>,
    per_dt_ignore: &mut HashMap<usize, bool>,
    gt_taken_by: &mut HashMap<usize, usize>,
) -> Result<(), EvalError> {
    if dts_in_cat.is_empty() {
        return Ok(());
    }
    let n_g = gts_in_cat.len();
    let n_d = dts_in_cat.len();

    // Build same-class IoU matrix by computing afresh via the bbox
    // kernel. Rebuilding here (rather than reading from CrossClassIous's
    // submatrix) keeps the assignment module free of an axis-orientation
    // mistake — the cross-class storage is `(D, G)` and the matching
    // engine needs `(G, D)`, so a sub-slice would have to be transposed
    // anyway. Bbox IoU is cheap and the alternative slicing is trickier
    // to get right.
    let mut iou = Array2::<f64>::zeros((n_g, n_d));
    if n_g > 0 {
        for (gi_local, &(_, gi)) in gts_in_cat.iter().enumerate() {
            let g_box = gt_anns[gi].bbox;
            for (di_local, &(_, di)) in dts_in_cat.iter().enumerate() {
                let d_box = dt_anns[di].bbox;
                iou[(gi_local, di_local)] = bbox_iou_pair(g_box, d_box);
            }
        }
    }

    let gt_ignore: Vec<bool> = gts_in_cat
        .iter()
        .map(|&(_, gi)| {
            let g = &gt_anns[gi];
            // Mirror the oracle's "iscrowd OR ignore" — see oracle.py:
            // `gt_ignore_k = np.array([g.iscrowd or g.ignore for g in gts_k])`.
            // The matching engine reads the same flag.
            g.is_crowd || g.ignore_flag.unwrap_or(false)
        })
        .collect();
    let gt_iscrowd: Vec<bool> = gts_in_cat
        .iter()
        .map(|&(_, gi)| gt_anns[gi].is_crowd)
        .collect();
    let dt_scores: Vec<f64> = dts_in_cat
        .iter()
        .map(|&(_, di)| dt_anns[di].score)
        .collect();

    let single_threshold = [params.t_f];
    let MatchResult {
        dt_perm,
        gt_perm,
        dt_matches: dt_matches_pos,
        gt_matches: gt_matches_pos,
        dt_ignore,
    } = match_image(
        iou.view(),
        &gt_ignore,
        &gt_iscrowd,
        &dt_scores,
        &single_threshold,
        ParityMode::Strict,
    )?;

    // Record per-DT matched / ignore at t_f. Permutations are over the
    // dts_in_cat slot ordering — map back to the global dt_input_idx.
    for (sorted_d, &orig_d) in dt_perm.iter().enumerate() {
        let (_row_idx, dt_input_idx) = dts_in_cat[orig_d];
        let matched = dt_matches_pos[(0, sorted_d)] >= 0;
        let is_ignore = dt_ignore[(0, sorted_d)];
        per_dt_matched.insert(dt_input_idx, matched);
        per_dt_ignore.insert(dt_input_idx, is_ignore);
    }
    // Record per-GT taken_by for Missed attribution. Note: the
    // matching engine returns gt_matches in the gt_perm order; the
    // oracle's gt_matched_by uses the original GT order. Map perm →
    // gts_in_cat[orig_g] → cross-matrix column index.
    for (sorted_g, &orig_g) in gt_perm.iter().enumerate() {
        let dt_pos = gt_matches_pos[(0, sorted_g)];
        if dt_pos < 0 {
            continue;
        }
        // Skip ignore-GTs: pycocotools' matching can mark an ignore-GT
        // as matched but the oracle's `gt_matched_by` only records
        // non-ignore matches (see `_greedy_match`, `gt_matched_by`
        // stays -1 for ignore matches per oracle.py:300-304).
        if gt_ignore[orig_g] {
            continue;
        }
        let (col_idx, _gt_input_idx) = gts_in_cat[orig_g];
        let dt_orig = dt_perm[dt_pos as usize];
        let (_row_idx, dt_input_idx) = dts_in_cat[dt_orig];
        gt_taken_by.insert(col_idx, dt_input_idx);
    }
    Ok(())
}

/// Pure axis-aligned bbox IoU on COCO `[x, y, w, h]`. Mirrors
/// `oracle.py::bbox_iou` for one pair.
fn bbox_iou_pair(g: crate::dataset::Bbox, d: crate::dataset::Bbox) -> f64 {
    let g_x2 = g.x + g.w;
    let g_y2 = g.y + g.h;
    let d_x2 = d.x + d.w;
    let d_y2 = d.y + d.h;
    let inter_w = (g_x2.min(d_x2) - g.x.max(d.x)).max(0.0);
    let inter_h = (g_y2.min(d_y2) - g.y.max(d.y)).max(0.0);
    let inter = inter_w * inter_h;
    let union = g.w * g.h + d.w * d.h - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// Pull `iou_same` / `iou_cross` for one DT row out of the cross-class
/// IoU matrix. Returns the best IoU and the per-image-local GT column
/// index for each side, or `(0.0, -1, 0.0, -1)` when the image has no
/// GTs (or the matrix is absent).
///
/// The cross-class side pass already labels each row/column with the
/// category index, but we read the GT category from `gt_anns` directly
/// because it costs nothing here and keeps the side-pass parallel
/// vectors out of the hot path's mental model.
fn best_same_and_cross(
    row_idx: usize,
    dt_cat: CategoryId,
    cross: Option<ndarray::ArrayView2<'_, f64>>,
    gt_local_indices: &[usize],
    gt_anns: &[CocoAnnotation],
    use_cats: bool,
) -> (f64, i32, f64, i32) {
    // No GT data on this image → both sides are zero (Bkg territory).
    let cross = match cross {
        Some(m) => m,
        None => return (0.0, -1, 0.0, -1),
    };
    if cross.ncols() == 0 {
        return (0.0, -1, 0.0, -1);
    }

    let mut iou_same = 0.0_f64;
    let mut best_same: i32 = -1;
    let mut iou_cross = 0.0_f64;
    let mut best_cross: i32 = -1;

    for (col, &gt_input_idx) in gt_local_indices.iter().enumerate() {
        let v = cross[(row_idx, col)];
        let g_cat = gt_anns[gt_input_idx].category_id;
        let same_class = !use_cats || g_cat == dt_cat;
        // Strict `>` mirrors the oracle (`if ious[g_local] > iou_same`)
        // — the first column wins ties, matching the oracle's iteration.
        if same_class {
            if v > iou_same {
                iou_same = v;
                best_same = col as i32;
            }
        } else if v > iou_cross {
            iou_cross = v;
            best_cross = col as i32;
        }
    }
    (iou_same, best_same, iou_cross, best_cross)
}

/// Apply the priority chain from `oracle.py:496-531`. Returns the bin
/// label and the rewrite-layer target.
fn pick_bin(
    iou_same: f64,
    best_same_col: i32,
    iou_cross: f64,
    best_cross_col: i32,
    t_f: f64,
    t_b: f64,
) -> DtBinLabel {
    if iou_same >= t_f {
        return DtBinLabel {
            bin: DtBin::Dupe,
            // The rewrite drops Dupe DTs; target unused but recorded
            // for symmetry with the oracle's _BinAttribution shape.
            target_gt_local_idx: best_same_col,
        };
    }
    if iou_cross >= t_f {
        return DtBinLabel {
            bin: DtBin::Cls,
            target_gt_local_idx: best_cross_col,
        };
    }
    if iou_same >= t_b && iou_same >= iou_cross {
        return DtBinLabel {
            bin: DtBin::Loc,
            target_gt_local_idx: best_same_col,
        };
    }
    if iou_cross >= t_b {
        return DtBinLabel {
            bin: DtBin::Both,
            target_gt_local_idx: best_cross_col,
        };
    }
    DtBinLabel {
        bin: DtBin::Bkg,
        target_gt_local_idx: -1,
    }
}
