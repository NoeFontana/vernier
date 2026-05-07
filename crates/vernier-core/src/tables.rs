//! Result tables — opt-in DataFrame-shaped views over the locked spine
//! (per ADR-0019).
//!
//! Produces columnar buffers (`Vec<T>`-of-columns) that `vernier-ffi`
//! mechanically rewrites into Arrow `RecordBatch`es. No Arrow types
//! appear here, so non-Python downstreams can consume the same builders.
//!
//! ## Quirk dispositions
//!
//! - **C5** (`strict`): the `-1.0` sentinel from [`Accumulated`] is
//!   mapped to `Option::None` here (not in the FFI conversion), so the
//!   table type stays honest for non-Arrow consumers.

use crate::accumulate::Accumulated;
use crate::dataset::{Bbox, CocoDataset, CocoDetections, EvalDataset};
use crate::error::EvalError;
use crate::evaluate::{EvalGrid, COLLAPSED_CATEGORY_SENTINEL};
use crate::summarize::{pairwise_sum, IOU_LOOKUP_TOL};
use ndarray::{Array2, ArrayView2, Axis};
use std::collections::HashMap;

/// Which tables to compute for an `evaluate(...)` call.
///
/// All-`false` ([`Self::NONE`]) means "summary only" — the default
/// `evaluate()` path. `per_detection` and `per_pair` additionally
/// require the spine to retain its IoU matrices (set upstream via
/// [`crate::EvaluateParams::retain_iou`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TablesRequest {
    /// Build a one-row-per-image rollup table.
    pub per_image: bool,
    /// Build a one-row-per-category summary table.
    pub per_class: bool,
    /// Build a one-row-per-detection table (requires IoU retention).
    pub per_detection: bool,
    /// Build a one-row-per-(DT, GT)-pair table (requires IoU retention).
    pub per_pair: bool,
}

impl TablesRequest {
    /// "No tables, summary only." Equivalent to `Default::default()`;
    /// named for readability at call sites.
    pub const NONE: Self = Self {
        per_image: false,
        per_class: false,
        per_detection: false,
        per_pair: false,
    };

    /// The two zero-overhead-on-the-spine tables: `per_image` +
    /// `per_class`. Both fold over data the matching engine and
    /// accumulator already produce; neither needs IoU retention.
    pub const CHEAP: Self = Self {
        per_image: true,
        per_class: true,
        per_detection: false,
        per_pair: false,
    };

    /// Every table. Requires the upstream evaluator to have been built
    /// with IoU retention enabled.
    pub const ALL: Self = Self {
        per_image: true,
        per_class: true,
        per_detection: true,
        per_pair: true,
    };

    /// True iff at least one requested table requires the spine to
    /// retain per-cell IoU matrices.
    pub fn requires_iou_retention(&self) -> bool {
        self.per_pair || self.per_detection
    }
}

/// Configuration knobs for the expensive tables. Inert when the
/// corresponding flag in [`TablesRequest`] is `false`.
#[derive(Debug, Clone)]
pub struct TablesConfig {
    /// IoU floor for `per_pair`. Pairs with `iou < per_pair_iou_floor`
    /// are dropped from the table. Default `0.1` matches the ADR.
    pub per_pair_iou_floor: f64,
    /// Hard cap on `per_pair` row count. Exceeding it raises
    /// [`EvalError::PerPairOverflow`], not a silent truncation.
    pub per_pair_max_rows: usize,
    /// Whether `per_detection` rows include bbox geometry columns.
    /// Off by default — most callers don't need them, and the cost is
    /// non-trivial for D in the millions.
    pub per_detection_with_geometry: bool,
}

impl Default for TablesConfig {
    fn default() -> Self {
        Self {
            per_pair_iou_floor: 0.1,
            per_pair_max_rows: 10_000_000,
            per_detection_with_geometry: false,
        }
    }
}

/// One row per category — mirrors `summarize_detection`'s 12-stat
/// layout, plus support counts and labels.
///
/// Numeric columns are `Option<f64>`; the `-1.0` sentinel (quirk
/// **C5**) maps to `None` here so any non-Arrow consumer sees an
/// honest type. Schema is pinned (see Arrow golden in
/// `tests/python/tables/schemas/per_class.json`).
#[derive(Debug, Clone)]
pub struct PerClassTable {
    /// COCO category id, or [`COLLAPSED_CATEGORY_SENTINEL`] if the
    /// upstream evaluator ran with `use_cats=false`.
    pub category_id: Vec<i64>,
    /// Human-readable category name from
    /// [`crate::dataset::CategoryMeta::name`]. The single-row collapsed
    /// case carries `"(all categories)"`.
    pub category_name: Vec<String>,
    /// AP@.50:.95, area=all, maxDets=largest. `None` for cells with no
    /// data (quirk **C5**).
    pub ap: Vec<Option<f64>>,
    /// AP@.50, area=all, maxDets=largest.
    pub ap50: Vec<Option<f64>>,
    /// AP@.75, area=all, maxDets=largest.
    pub ap75: Vec<Option<f64>>,
    /// AP@.50:.95, area=small, maxDets=largest.
    pub ap_s: Vec<Option<f64>>,
    /// AP@.50:.95, area=medium, maxDets=largest.
    pub ap_m: Vec<Option<f64>>,
    /// AP@.50:.95, area=large, maxDets=largest.
    pub ap_l: Vec<Option<f64>>,
    /// AR@.50:.95, area=all, maxDets=1.
    pub ar_max_1: Vec<Option<f64>>,
    /// AR@.50:.95, area=all, maxDets=10.
    pub ar_max_10: Vec<Option<f64>>,
    /// AR@.50:.95, area=all, maxDets=100.
    pub ar_max_100: Vec<Option<f64>>,
    /// Non-ignore GT count for this category, summed across images at
    /// area=all. Inferred from [`Accumulated`]'s K-axis size and the
    /// dataset's annotation list.
    pub n_gt: Vec<u32>,
    /// DT count for this category, summed across images at area=all.
    /// Inferred from the dataset's detection list (when threaded
    /// through) or from upstream EvalGrid cells (when called from
    /// streaming).
    pub n_dt: Vec<u32>,
}

impl PerClassTable {
    /// Number of rows.
    pub fn len(&self) -> usize {
        self.category_id.len()
    }

    /// True when no rows.
    pub fn is_empty(&self) -> bool {
        self.category_id.is_empty()
    }

    /// Names of every column in pinned order. Matches the Arrow schema
    /// emitted by `vernier-ffi`.
    pub const COLUMN_NAMES: &'static [&'static str] = &[
        "category_id",
        "category_name",
        "ap",
        "ap50",
        "ap75",
        "ap_s",
        "ap_m",
        "ap_l",
        "ar_max_1",
        "ar_max_10",
        "ar_max_100",
        "n_gt",
        "n_dt",
    ];
}

/// Per-category support counts that `build_per_class` consumes
/// alongside [`Accumulated`]. Pre-aggregated on the caller's side so
/// the table builder is a pure fold; streaming and batch paths
/// produce these counts from different substrates (the cells store vs.
/// the parsed dataset) but both feed into the same builder.
#[derive(Debug, Clone, Default)]
pub struct PerClassSupport {
    /// Non-ignore GT count, K-axis indexed (matching `Accumulated`'s
    /// K-axis order). `n_gt[k]` is the count for category k.
    pub n_gt: Vec<u32>,
    /// DT count, K-axis indexed. `n_dt[k]` is the count for category k.
    pub n_dt: Vec<u32>,
}

impl PerClassSupport {
    /// Construct a zero-filled support struct of length `n_categories`.
    /// Useful when the caller is going to fill it in from a cells-store
    /// scan.
    pub fn zeros(n_categories: usize) -> Self {
        Self {
            n_gt: vec![0; n_categories],
            n_dt: vec![0; n_categories],
        }
    }
}

/// Build a [`PerClassTable`] from an [`Accumulated`] tensor, the
/// dataset (for category-id → name lookup), and pre-aggregated support
/// counts.
///
/// The accumulator's K-axis is expected to match the dataset's
/// id-ascending category order produced by
/// [`crate::evaluate_with`]. When `accum.precision`'s K-axis is `1`
/// and the dataset has more than one category, that's the
/// `use_cats=false` collapsed run: a single row is emitted with
/// `category_id = COLLAPSED_CATEGORY_SENTINEL` and
/// `category_name = "(all categories)"`.
///
/// # Errors
///
/// - [`EvalError::DimensionMismatch`] — `iou_thresholds` /
///   `max_dets` lengths disagree with the accumulator's T / M axes,
///   or the A-axis is not 4 (this builder requires the COCO
///   detection grid layout — `[all, small, medium, large]`).
/// - [`EvalError::InvalidConfig`] — `iou_thresholds` is missing the
///   0.5 or 0.75 entry, or `max_dets` is missing 1, 10, or 100, or
///   the K-axis size is incompatible with the dataset.
pub fn build_per_class(
    accum: &Accumulated,
    dataset: &CocoDataset,
    iou_thresholds: &[f64],
    max_dets: &[usize],
    support: &PerClassSupport,
) -> Result<PerClassTable, EvalError> {
    let p_shape = accum.precision.shape();
    let r_shape = accum.recall.shape();
    let n_t = p_shape[0];
    let n_k = p_shape[2];
    let n_a = p_shape[3];
    let n_m = p_shape[4];

    if n_t != iou_thresholds.len() {
        return Err(EvalError::DimensionMismatch {
            detail: format!(
                "precision T-axis {} != iou_thresholds len {}",
                n_t,
                iou_thresholds.len()
            ),
        });
    }
    if n_m != max_dets.len() {
        return Err(EvalError::DimensionMismatch {
            detail: format!(
                "precision M-axis {} != max_dets len {}",
                n_m,
                max_dets.len()
            ),
        });
    }
    if r_shape[0] != n_t || r_shape[1] != n_k || r_shape[2] != n_a || r_shape[3] != n_m {
        return Err(EvalError::DimensionMismatch {
            detail: format!("recall {r_shape:?} disagrees with precision {p_shape:?}"),
        });
    }
    if n_a != 4 {
        return Err(EvalError::DimensionMismatch {
            detail: format!(
                "per_class requires the COCO detection area grid (4 buckets); got {n_a}"
            ),
        });
    }
    if support.n_gt.len() != n_k || support.n_dt.len() != n_k {
        return Err(EvalError::DimensionMismatch {
            detail: format!(
                "support counts (n_gt={}, n_dt={}) disagree with K-axis {}",
                support.n_gt.len(),
                support.n_dt.len(),
                n_k
            ),
        });
    }

    let t50 = find_iou_index(iou_thresholds, 0.5)?;
    let t75 = find_iou_index(iou_thresholds, 0.75)?;
    let m1 = find_max_dets_index(max_dets, 1)?;
    let m10 = find_max_dets_index(max_dets, 10)?;
    let m100 = find_max_dets_index(max_dets, 100)?;
    let m_last = n_m - 1;

    const A_ALL: usize = 0;
    const A_SMALL: usize = 1;
    const A_MEDIUM: usize = 2;
    const A_LARGE: usize = 3;

    // Collapsed K=1 with multi-category dataset means use_cats=false:
    // emit a single pseudo-row labelled "(all categories)".
    let (category_ids, category_names): (Vec<i64>, Vec<String>) = if n_k == 1
        && dataset.categories().len() != 1
    {
        (
            vec![COLLAPSED_CATEGORY_SENTINEL],
            vec!["(all categories)".to_string()],
        )
    } else {
        if n_k != dataset.categories().len() {
            return Err(EvalError::InvalidConfig {
                detail: format!(
                    "K-axis size {} disagrees with dataset.categories().len() {}",
                    n_k,
                    dataset.categories().len()
                ),
            });
        }
        let mut sorted: Vec<&crate::dataset::CategoryMeta> = dataset.categories().iter().collect();
        sorted.sort_unstable_by_key(|c| c.id.0);
        (
            sorted.iter().map(|c| c.id.0).collect(),
            sorted.iter().map(|c| c.name.clone()).collect(),
        )
    };

    let mut ap = Vec::with_capacity(n_k);
    let mut ap50 = Vec::with_capacity(n_k);
    let mut ap75 = Vec::with_capacity(n_k);
    let mut ap_s = Vec::with_capacity(n_k);
    let mut ap_m = Vec::with_capacity(n_k);
    let mut ap_l = Vec::with_capacity(n_k);
    let mut ar_max_1 = Vec::with_capacity(n_k);
    let mut ar_max_10 = Vec::with_capacity(n_k);
    let mut ar_max_100 = Vec::with_capacity(n_k);

    for k in 0..n_k {
        ap.push(mean_precision(accum, 0..n_t, k, A_ALL, m_last));
        ap50.push(mean_precision(accum, t50..t50 + 1, k, A_ALL, m_last));
        ap75.push(mean_precision(accum, t75..t75 + 1, k, A_ALL, m_last));
        ap_s.push(mean_precision(accum, 0..n_t, k, A_SMALL, m_last));
        ap_m.push(mean_precision(accum, 0..n_t, k, A_MEDIUM, m_last));
        ap_l.push(mean_precision(accum, 0..n_t, k, A_LARGE, m_last));
        ar_max_1.push(mean_recall(accum, 0..n_t, k, A_ALL, m1));
        ar_max_10.push(mean_recall(accum, 0..n_t, k, A_ALL, m10));
        ar_max_100.push(mean_recall(accum, 0..n_t, k, A_ALL, m100));
    }

    Ok(PerClassTable {
        category_id: category_ids,
        category_name: category_names,
        ap,
        ap50,
        ap75,
        ap_s,
        ap_m,
        ap_l,
        ar_max_1,
        ar_max_10,
        ar_max_100,
        n_gt: support.n_gt.clone(),
        n_dt: support.n_dt.clone(),
    })
}

/// Mean of precision[t_range, :, k, area_idx, m_idx], filtering -1.0
/// (quirk C5). Returns `None` when the slice is all-sentinel.
///
/// Uses [`pairwise_sum`] so the result is bit-equal to numpy's
/// `np.mean(s[s > -1])` and to [`crate::summarize::Summary::stats`]
/// for K=1 collapsed runs (quirk **C8**).
fn mean_precision(
    accum: &Accumulated,
    t_range: std::ops::Range<usize>,
    k_idx: usize,
    area_idx: usize,
    m_idx: usize,
) -> Option<f64> {
    let r = accum.precision.shape()[1];
    let mut filtered: Vec<f64> = Vec::with_capacity(t_range.len() * r);
    for t in t_range {
        accum
            .precision
            .index_axis(Axis(0), t)
            .index_axis(Axis(1), k_idx)
            .index_axis(Axis(1), area_idx)
            .index_axis(Axis(1), m_idx)
            .iter()
            .copied()
            .for_each(|v| {
                if v > -1.0 {
                    filtered.push(v);
                }
            });
    }
    if filtered.is_empty() {
        None
    } else {
        Some(pairwise_sum(&filtered) / filtered.len() as f64)
    }
}

/// Mean of recall[t_range, k, area_idx, m_idx], filtering -1.0 (quirk
/// C5). Returns `None` when the slice is all-sentinel.
fn mean_recall(
    accum: &Accumulated,
    t_range: std::ops::Range<usize>,
    k_idx: usize,
    area_idx: usize,
    m_idx: usize,
) -> Option<f64> {
    let mut filtered: Vec<f64> = Vec::with_capacity(t_range.len());
    for t in t_range {
        let v = accum.recall[[t, k_idx, area_idx, m_idx]];
        if v > -1.0 {
            filtered.push(v);
        }
    }
    if filtered.is_empty() {
        None
    } else {
        Some(pairwise_sum(&filtered) / filtered.len() as f64)
    }
}

fn find_iou_index(iou_thresholds: &[f64], target: f64) -> Result<usize, EvalError> {
    iou_thresholds
        .iter()
        .position(|&v| (v - target).abs() < IOU_LOOKUP_TOL)
        .ok_or_else(|| EvalError::InvalidConfig {
            detail: format!("iou_thresholds missing required value {target}"),
        })
}

fn find_max_dets_index(max_dets: &[usize], target: usize) -> Result<usize, EvalError> {
    max_dets
        .iter()
        .position(|&v| v == target)
        .ok_or_else(|| EvalError::InvalidConfig {
            detail: format!("max_dets missing required value {target}"),
        })
}

/// Walk the [`crate::EvalGrid`] at `area_index = ALL` and aggregate
/// non-ignore GT and DT counts per category. The result is
/// K-axis-indexed (matching the grid's K-axis), so [`build_per_class`]
/// can consume it directly.
///
/// Cells absent from the grid (`None` entries — image had no GTs and
/// no DTs in this category) contribute zero to both counts. The
/// non-ignore GT count comes from the cell's `gt_ignore` array; DT
/// count from the cell's `dt_scores` length (post-area-filter,
/// post-maxDets cap).
///
/// Out-of-range `area_index_all` (typically `0` for the COCO
/// detection grid) yields an all-zero support struct.
pub fn aggregate_per_class_support(
    grid: &crate::EvalGrid,
    area_index_all: usize,
) -> PerClassSupport {
    let mut support = PerClassSupport::zeros(grid.n_categories);
    if area_index_all >= grid.n_area_ranges {
        return support;
    }
    for k in 0..grid.n_categories {
        let mut n_gt = 0u32;
        let mut n_dt = 0u32;
        for i in 0..grid.n_images {
            if let Some(cell) = grid.cell(k, area_index_all, i) {
                n_gt = n_gt.saturating_add(
                    cell.gt_ignore
                        .iter()
                        .filter(|&&ignored| !ignored)
                        .count()
                        .try_into()
                        .unwrap_or(u32::MAX),
                );
                n_dt = n_dt.saturating_add(cell.dt_scores.len().try_into().unwrap_or(u32::MAX));
            }
        }
        support.n_gt[k] = n_gt;
        support.n_dt[k] = n_dt;
    }
    support
}

/// One row per image — rollup of locked-spine outputs across categories
/// at area=ALL.
///
/// No `ap` / `ap_50` columns (per-image AP is degenerate; see
/// `docs/explanation/why-no-per-image-ap.md`).
///
/// Quirk **G2** is honored automatically: a DT matched to a crowd GT
/// carries `dt_ignore=true`, so TP/FP filters via `!dt_ignore` exclude
/// it. Crowd GTs do not contribute to `n_gt` either (excluded by
/// `gt_ignore`).
///
/// `n_dt` reflects the post-maxDets cap stored on the cells (typical
/// COCO cap of 100 swallows nearly all images).
#[derive(Debug, Clone)]
pub struct PerImageTable {
    /// COCO image id, sourced from the dataset's id-ascending image
    /// order (matching the grid's I-axis ordering).
    pub image_id: Vec<i64>,
    /// Non-ignore GT count, summed across categories at area=ALL.
    pub n_gt: Vec<u32>,
    /// DT count, summed across categories at area=ALL (post-maxDets;
    /// see struct doc).
    pub n_dt: Vec<u32>,
    /// True positives at IoU=0.50 — DTs with `dt_matched && !dt_ignore`.
    pub tp_at_50: Vec<u32>,
    /// False positives at IoU=0.50 — DTs with `!dt_matched && !dt_ignore`.
    pub fp_at_50: Vec<u32>,
    /// False negatives at IoU=0.50 — `n_gt - tp_at_50`.
    pub fn_at_50: Vec<u32>,
    /// True positives at IoU=0.75.
    pub tp_at_75: Vec<u32>,
    /// False positives at IoU=0.75.
    pub fp_at_75: Vec<u32>,
    /// False negatives at IoU=0.75.
    pub fn_at_75: Vec<u32>,
    /// Mean true-positive count across the T-axis, floored.
    pub tp_mean_iou: Vec<u32>,
}

impl PerImageTable {
    /// Number of rows.
    pub fn len(&self) -> usize {
        self.image_id.len()
    }

    /// True when no rows.
    pub fn is_empty(&self) -> bool {
        self.image_id.is_empty()
    }

    /// Names of every column in pinned order. Matches the Arrow schema
    /// emitted by `vernier-ffi`.
    pub const COLUMN_NAMES: &'static [&'static str] = &[
        "image_id",
        "n_gt",
        "n_dt",
        "tp_at_50",
        "fp_at_50",
        "fn_at_50",
        "tp_at_75",
        "fp_at_75",
        "fn_at_75",
        "tp_mean_iou",
    ];
}

/// Build a [`PerImageTable`] from an [`EvalGrid`] (cells store) and a
/// dataset (image_id source).
///
/// The grid must use the COCO detection area grid (4 buckets;
/// area=ALL pinned at index 0); other layouts return
/// [`EvalError::DimensionMismatch`]. `iou_thresholds` must contain
/// 0.5 and 0.75; otherwise [`EvalError::InvalidConfig`].
///
/// I/O complexity is O(K * I * D) — one walk over every populated
/// cell at area=ALL, with two threshold passes per cell.
pub fn build_per_image(
    grid: &EvalGrid,
    dataset: &CocoDataset,
    iou_thresholds: &[f64],
) -> Result<PerImageTable, EvalError> {
    if grid.n_area_ranges != 4 {
        return Err(EvalError::DimensionMismatch {
            detail: format!(
                "per_image requires the COCO detection area grid (4 buckets); got {}",
                grid.n_area_ranges
            ),
        });
    }
    let n_t = iou_thresholds.len();
    if n_t == 0 {
        return Err(EvalError::DimensionMismatch {
            detail: "iou_thresholds empty".into(),
        });
    }
    let t50 = find_iou_index(iou_thresholds, 0.5)?;
    let t75 = find_iou_index(iou_thresholds, 0.75)?;
    const A_ALL: usize = 0;

    // Image id list: id-ascending sort matches the grid's I-axis.
    let mut images: Vec<&crate::dataset::ImageMeta> = dataset.images().iter().collect();
    images.sort_unstable_by_key(|im| im.id.0);
    if images.len() != grid.n_images {
        return Err(EvalError::InvalidConfig {
            detail: format!(
                "dataset image count {} disagrees with grid I-axis {}",
                images.len(),
                grid.n_images
            ),
        });
    }
    let n_i = grid.n_images;

    let mut image_id = Vec::with_capacity(n_i);
    let mut n_gt = vec![0u32; n_i];
    let mut n_dt = vec![0u32; n_i];
    let mut tp_at_50 = vec![0u32; n_i];
    let mut fp_at_50 = vec![0u32; n_i];
    let mut tp_at_75 = vec![0u32; n_i];
    let mut fp_at_75 = vec![0u32; n_i];
    // Sum of TPs across the T-axis per image. Divided by n_t (floor)
    // at the end to produce `tp_mean_iou`.
    let mut tp_t_sum = vec![0u64; n_i];

    for im in &images {
        image_id.push(im.id.0);
    }

    for k in 0..grid.n_categories {
        for i in 0..n_i {
            let Some(cell) = grid.cell(k, A_ALL, i) else {
                continue;
            };
            // n_gt: non-ignore GTs.
            n_gt[i] = n_gt[i].saturating_add(saturating_u32_count(
                cell.gt_ignore.iter().filter(|&&b| !b).count(),
            ));
            let n_dt_cell = cell.dt_scores.len();
            n_dt[i] = n_dt[i].saturating_add(saturating_u32_count(n_dt_cell));

            // dt_ignore folds in quirks B6 (matched to ignore/crowd
            // GT) and B7 (out-of-area unmatched), so the !dt_ignore
            // gate is the right one for TP/FP under quirk G2.
            for d in 0..n_dt_cell {
                if !cell.dt_ignore[[t50, d]] {
                    if cell.dt_matched[[t50, d]] {
                        tp_at_50[i] = tp_at_50[i].saturating_add(1);
                    } else {
                        fp_at_50[i] = fp_at_50[i].saturating_add(1);
                    }
                }
                if !cell.dt_ignore[[t75, d]] {
                    if cell.dt_matched[[t75, d]] {
                        tp_at_75[i] = tp_at_75[i].saturating_add(1);
                    } else {
                        fp_at_75[i] = fp_at_75[i].saturating_add(1);
                    }
                }
            }

            for t in 0..n_t {
                let mut tp_t = 0u64;
                for d in 0..n_dt_cell {
                    if !cell.dt_ignore[[t, d]] && cell.dt_matched[[t, d]] {
                        tp_t += 1;
                    }
                }
                tp_t_sum[i] = tp_t_sum[i].saturating_add(tp_t);
            }
        }
    }

    let fn_at_50: Vec<u32> = (0..n_i)
        .map(|i| n_gt[i].saturating_sub(tp_at_50[i]))
        .collect();
    let fn_at_75: Vec<u32> = (0..n_i)
        .map(|i| n_gt[i].saturating_sub(tp_at_75[i]))
        .collect();
    let tp_mean_iou: Vec<u32> = (0..n_i)
        .map(|i| {
            let mean = tp_t_sum[i] / n_t as u64;
            mean.try_into().unwrap_or(u32::MAX)
        })
        .collect();

    Ok(PerImageTable {
        image_id,
        n_gt,
        n_dt,
        tp_at_50,
        fp_at_50,
        fn_at_50,
        tp_at_75,
        fp_at_75,
        fn_at_75,
        tp_mean_iou,
    })
}

fn saturating_u32_count(n: usize) -> u32 {
    n.try_into().unwrap_or(u32::MAX)
}

/// Per-`(category, image)` IoU matrices retained from a
/// [`crate::evaluate_with`] pass when the caller passed
/// [`crate::EvaluateParams::retain_iou`] = `true`.
///
/// Keyed by `(k, i)` (not `(k, a, i)`): IoU is geometry-only, so the
/// same matrix serves every area range. The streaming evaluator
/// preserves the same key shape so its retention store and the batch
/// path stay drop-in compatible.
#[derive(Debug, Clone, Default)]
pub struct RetainedIous {
    inner: HashMap<(usize, usize), Array2<f64>>,
}

impl RetainedIous {
    /// Construct an empty store.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Construct from an already-built map. The map's keys must be
    /// `(k_index, i_index)`; the Array2 shape is `(n_gt, n_dt)` for
    /// that cell.
    pub(crate) fn from_map(map: HashMap<(usize, usize), Array2<f64>>) -> Self {
        Self { inner: map }
    }

    /// Number of retained IoU matrices.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// True when no matrices retained.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Insert (or overwrite) the IoU matrix for `(k, i)`.
    pub(crate) fn insert(&mut self, k: usize, i: usize, iou: Array2<f64>) {
        self.inner.insert((k, i), iou);
    }

    /// Borrow the IoU matrix for `(k, i)` if retained.
    pub fn get(&self, k: usize, i: usize) -> Option<ArrayView2<'_, f64>> {
        self.inner.get(&(k, i)).map(|m| m.view())
    }

    /// Move-out variant of [`Self::get`]. Used by streaming to roll
    /// per-batch matrices into the long-lived store.
    pub(crate) fn remove(&mut self, k: usize, i: usize) -> Option<Array2<f64>> {
        self.inner.remove(&(k, i))
    }

    /// Iterate `(k, i, view)` triplets in arbitrary order. The
    /// distributed-eval encoder (ADR-0031) walks this to materialize
    /// the wire-format `retained_ious` section, then sorts.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (usize, usize, ArrayView2<'_, f64>)> + '_ {
        self.inner.iter().map(|(&(k, i), arr)| (k, i, arr.view()))
    }
}

/// Per-image cross-class IoU matrices and the parallel category-index
/// vectors that label each row/column.
///
/// Per ADR-0023, this is the side-pass output the TIDE Cls/Both bin
/// assignment and the confusion-matrix capability both consume:
/// IoU(DT_class_A, GT_class_B) for every `(A, B)` pair on a given
/// image, materialized once per call.
///
/// Keyed by `image_idx` — the position of the image in the
/// id-ascending ordering used by [`crate::evaluate_with`]. For each
/// populated image:
///
/// - `inner[i]` is the dense `(D_total, G_total)` IoU matrix produced
///   by the kernel on the un-class-filtered per-image lists.
/// - `dt_classes[i]` carries the category index per row, in the same
///   order as the matrix's row axis.
/// - `gt_classes[i]` carries the category index per column.
///
/// The "category index" is the position of the category in the
/// deterministic id-ascending category ordering [`crate::evaluate_with`]
/// uses for its `K` axis — the same indexing the rest of the spine
/// reasons in. It is *not* the COCO category id.
///
/// Distinct from [`RetainedIous`]: that store carries same-class IoU
/// keyed `(category_index, image_index)` for the result-tables
/// product, whereas this store carries the cross-class matrix keyed
/// purely by image. The two have different consumers and different
/// keying; they are deliberately not unified (per ADR-0023).
///
/// Images with no GTs and no DTs are absent from the maps.
#[derive(Debug, Clone, Default)]
pub struct CrossClassIous {
    inner: HashMap<usize, Array2<f64>>,
    dt_classes: HashMap<usize, Vec<usize>>,
    gt_classes: HashMap<usize, Vec<usize>>,
}

impl CrossClassIous {
    /// Construct an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of images with retained cross-class data.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// True when no images are retained.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Insert (or overwrite) the cross-class IoU matrix and the
    /// parallel category-index vectors for `image_idx`.
    ///
    /// `iou.shape() == (dt_classes.len(), gt_classes.len())` is
    /// expected by every accessor; callers that violate this will
    /// observe inconsistent reads but no panic.
    pub(crate) fn insert(
        &mut self,
        image_idx: usize,
        iou: Array2<f64>,
        dt_classes: Vec<usize>,
        gt_classes: Vec<usize>,
    ) {
        self.inner.insert(image_idx, iou);
        self.dt_classes.insert(image_idx, dt_classes);
        self.gt_classes.insert(image_idx, gt_classes);
    }

    /// Borrow the cross-class IoU matrix for `image_idx`, if present.
    /// Rows index DTs (in the [`Self::dt_classes`] order); columns
    /// index GTs (in the [`Self::gt_classes`] order).
    pub fn get(&self, image_idx: usize) -> Option<ArrayView2<'_, f64>> {
        self.inner.get(&image_idx).map(|m| m.view())
    }

    /// Borrow the per-row DT category indices for `image_idx`.
    pub fn dt_classes(&self, image_idx: usize) -> Option<&[usize]> {
        self.dt_classes.get(&image_idx).map(Vec::as_slice)
    }

    /// Borrow the per-column GT category indices for `image_idx`.
    pub fn gt_classes(&self, image_idx: usize) -> Option<&[usize]> {
        self.gt_classes.get(&image_idx).map(Vec::as_slice)
    }
}

/// Match status for a detection at a given IoU threshold.
///
/// Arrow-encoded as `dictionary<utf8>` with entries pinned to
/// `[tp, fp, ignored]` so consumers comparing against a static
/// dictionary index are stable across runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchStatus {
    /// Matched a non-ignore GT (DT counts as TP).
    TruePositive,
    /// Did not match any GT (DT counts as FP).
    FalsePositive,
    /// DT carries `dt_ignore=true` — quirk **B6** (matched to crowd
    /// or ignore-GT) or **B7** (out-of-area unmatched).
    Ignored,
}

impl MatchStatus {
    /// Stable dictionary index — pinned to `[tp, fp, ignored]`.
    pub fn dict_index(self) -> u32 {
        match self {
            Self::TruePositive => 0,
            Self::FalsePositive => 1,
            Self::Ignored => 2,
        }
    }

    /// Pinned dictionary values, indexed by [`Self::dict_index`].
    pub const DICT_VALUES: &'static [&'static str] = &["tp", "fp", "ignored"];
}

/// Optional bbox geometry block on [`PerDetectionTable`]. Populated
/// only when [`TablesConfig::per_detection_with_geometry`] is `true`.
#[derive(Debug, Clone, Default)]
pub struct BboxColumns {
    /// Flat `[x, y, w, h]` per row, length 4 × n_rows. The Arrow
    /// encoding is `fixed_size_list<f64, 4>`.
    pub xywh: Vec<[f64; 4]>,
}

/// One row per detection. Built from the cells store at area=ALL, the
/// dataset (for image/category id lookups via the grid), the detection
/// list (for area + optional bbox), and optionally the retained IoU
/// matrices (for `best_iou`).
#[derive(Debug, Clone)]
pub struct PerDetectionTable {
    /// Detection id (COCO `id` field, post-auto-assignment if absent).
    pub detection_id: Vec<i64>,
    /// Image this detection lands on.
    pub image_id: Vec<i64>,
    /// Detection's claimed category id.
    pub category_id: Vec<i64>,
    /// Confidence score.
    pub score: Vec<f64>,
    /// Kernel-defined detection area (for bbox: `bbox.w * bbox.h`).
    pub area: Vec<f64>,
    /// Match status at IoU=0.50.
    pub match_status_at_50: Vec<MatchStatus>,
    /// GT id matched at IoU=0.50, or `None` for FP / ignored rows.
    pub matched_gt_id_at_50: Vec<Option<i64>>,
    /// Max IoU to any same-class GT in the same image. `None` when
    /// the IoU matrix wasn't retained, or when there were no
    /// same-class GTs in the image.
    pub best_iou: Vec<Option<f64>>,
    /// Optional bbox geometry. `None` unless
    /// [`TablesConfig::per_detection_with_geometry`] was set.
    pub bbox: Option<BboxColumns>,
}

impl PerDetectionTable {
    /// Number of rows.
    pub fn len(&self) -> usize {
        self.detection_id.len()
    }
    /// True if no rows.
    pub fn is_empty(&self) -> bool {
        self.detection_id.is_empty()
    }
}

/// One row per `(DT, GT)` pair within a `(image, category)` cell that
/// passes the IoU floor. Always class-restricted: pairs across
/// categories are excluded.
#[derive(Debug, Clone, Default)]
pub struct PerPairTable {
    /// Detection id.
    pub detection_id: Vec<i64>,
    /// Ground-truth id.
    pub ground_truth_id: Vec<i64>,
    /// Image id (shared by DT and GT — pairs are class-and-image-restricted).
    pub image_id: Vec<i64>,
    /// Category id (shared by DT and GT).
    pub category_id: Vec<i64>,
    /// Raw IoU as the kernel produced it.
    pub iou: Vec<f64>,
}

impl PerPairTable {
    /// Number of rows.
    pub fn len(&self) -> usize {
        self.detection_id.len()
    }
    /// True if no rows.
    pub fn is_empty(&self) -> bool {
        self.detection_id.is_empty()
    }
}

/// Build a [`PerDetectionTable`].
///
/// Walks the cells store at `area_idx=0` (COCO ALL bucket); for each
/// cell, emits one row per detection (in the matching engine's
/// sorted-DT order). When `retained_ious` is `Some`, populates
/// `best_iou` from the per-cell matrix; otherwise leaves it `None`.
///
/// `iou_thresholds` must contain 0.5 — used to read the
/// `match_status_at_50` and `matched_gt_id_at_50` columns.
pub fn build_per_detection(
    grid: &EvalGrid,
    detections: &CocoDetections,
    iou_thresholds: &[f64],
    retained_ious: Option<&RetainedIous>,
    config: &TablesConfig,
) -> Result<PerDetectionTable, EvalError> {
    if grid.n_area_ranges == 0 {
        return Err(EvalError::DimensionMismatch {
            detail: "per_detection requires at least one area range".into(),
        });
    }
    let t50 = find_iou_index(iou_thresholds, 0.5)?;
    const A_ALL: usize = 0;

    // Detection ids are stable (auto-assigned when absent), so the
    // matching engine's sorted-DT order resolves uniquely back to a
    // single detection by id.
    let det_index: HashMap<i64, &crate::dataset::CocoDetection> = detections
        .detections()
        .iter()
        .map(|d| (d.id.0, d))
        .collect();

    let with_geometry = config.per_detection_with_geometry;
    let mut detection_id = Vec::new();
    let mut image_id = Vec::new();
    let mut category_id = Vec::new();
    let mut score = Vec::new();
    let mut area = Vec::new();
    let mut match_status_at_50 = Vec::new();
    let mut matched_gt_id_at_50 = Vec::new();
    let mut best_iou = Vec::new();
    let mut bbox_xywh: Vec<[f64; 4]> = Vec::new();

    for k in 0..grid.n_categories {
        for i in 0..grid.n_images {
            let Some(cell) = grid.cell(k, A_ALL, i) else {
                continue;
            };
            let Some(meta) = grid.cell_meta(k, A_ALL, i) else {
                continue;
            };
            let iou_view = retained_ious.and_then(|r| r.get(k, i));

            for d in 0..cell.dt_scores.len() {
                let dt_id = meta.dt_ids[d];
                detection_id.push(dt_id);
                image_id.push(meta.image_id);
                category_id.push(meta.category_id);
                score.push(cell.dt_scores[d]);

                let det = det_index.get(&dt_id);
                area.push(det.map(|d| d.area).unwrap_or(f64::NAN));
                if with_geometry {
                    let b = det
                        .map(|d| d.bbox)
                        .unwrap_or_else(|| Bbox::from([f64::NAN; 4]));
                    bbox_xywh.push([b.x, b.y, b.w, b.h]);
                }

                let dt_ignored = cell.dt_ignore[[t50, d]];
                let dt_matched_flag = cell.dt_matched[[t50, d]];
                let status = if dt_ignored {
                    MatchStatus::Ignored
                } else if dt_matched_flag {
                    MatchStatus::TruePositive
                } else {
                    MatchStatus::FalsePositive
                };
                match_status_at_50.push(status);
                let matched_gt = meta.dt_matches[[t50, d]];
                matched_gt_id_at_50.push(if matched_gt == 0 || dt_ignored {
                    None
                } else {
                    Some(matched_gt)
                });

                // IoU matrix is shaped (n_gt, n_dt); take the max
                // over the GT axis at column d.
                let bi = iou_view.and_then(|view| {
                    if view.ncols() == 0 || view.nrows() == 0 || d >= view.ncols() {
                        return None;
                    }
                    let mut best: Option<f64> = None;
                    for g in 0..view.nrows() {
                        let v = view[[g, d]];
                        if best.is_none_or(|b| v > b) {
                            best = Some(v);
                        }
                    }
                    best
                });
                best_iou.push(bi);
            }
        }
    }

    let bbox = if with_geometry {
        Some(BboxColumns { xywh: bbox_xywh })
    } else {
        None
    };
    Ok(PerDetectionTable {
        detection_id,
        image_id,
        category_id,
        score,
        area,
        match_status_at_50,
        matched_gt_id_at_50,
        best_iou,
        bbox,
    })
}

/// Build a [`PerPairTable`] from retained IoU matrices.
///
/// Emits one row per `(DT, GT)` pair where IoU >=
/// `config.per_pair_iou_floor`. Pairs across categories are excluded
/// by construction (the IoU matrix is per-cell).
///
/// Overflow is checked *inside* the per-cell push loop, before the
/// column vecs grow past `config.per_pair_max_rows` — at LVIS scale a
/// post-build check would OOM first.
pub fn build_per_pair(
    grid: &EvalGrid,
    retained_ious: &RetainedIous,
    config: &TablesConfig,
) -> Result<PerPairTable, EvalError> {
    if grid.n_area_ranges == 0 {
        return Err(EvalError::DimensionMismatch {
            detail: "per_pair requires at least one area range".into(),
        });
    }
    const A_ALL: usize = 0;
    let mut out = PerPairTable::default();
    let cap = config.per_pair_max_rows;
    let floor = config.per_pair_iou_floor;

    for k in 0..grid.n_categories {
        for i in 0..grid.n_images {
            let Some(meta) = grid.cell_meta(k, A_ALL, i) else {
                continue;
            };
            let Some(view) = retained_ious.get(k, i) else {
                continue;
            };
            // The IoU matrix is built on the unfiltered kernel set,
            // while meta.{gt,dt}_ids reflect the area-bucket-filtered
            // set; clamp both axes to the smaller length defensively.
            let n_gt_use = view.nrows().min(meta.gt_ids.len());
            let n_dt_use = view.ncols().min(meta.dt_ids.len());
            for g in 0..n_gt_use {
                for d in 0..n_dt_use {
                    let v = view[[g, d]];
                    if v < floor {
                        continue;
                    }
                    if out.detection_id.len() >= cap {
                        return Err(EvalError::PerPairOverflow {
                            observed: out.detection_id.len() + 1,
                            cap,
                        });
                    }
                    out.detection_id.push(meta.dt_ids[d]);
                    out.ground_truth_id.push(meta.gt_ids[g]);
                    out.image_id.push(meta.image_id);
                    out.category_id.push(meta.category_id);
                    out.iou.push(v);
                }
            }
        }
    }
    Ok(out)
}

/// Bundle of computed tables, populated only for the flags set on the
/// triggering [`TablesRequest`]. Unset flags leave the corresponding
/// field at `None` — both producers and consumers preserve that
/// invariant.
#[derive(Debug, Clone, Default)]
pub struct Tables {
    /// Per-image rollup table when [`TablesRequest::per_image`] was set.
    pub per_image: Option<PerImageTable>,
    /// Per-class table when [`TablesRequest::per_class`] was set.
    pub per_class: Option<PerClassTable>,
    /// Per-detection table when [`TablesRequest::per_detection`] was set.
    pub per_detection: Option<PerDetectionTable>,
    /// Per-(DT, GT)-pair table when [`TablesRequest::per_pair`] was set.
    pub per_pair: Option<PerPairTable>,
}

/// Umbrella entry that builds every requested table from the
/// locked-spine outputs. Single fan-out point both batch and streaming
/// drive through.
///
/// `detections` is required when `request.per_detection` is set
/// (`area` + optional `bbox` columns). `retained_ious` is required for
/// `request.per_pair`, and consumed by `per_detection.best_iou` when
/// present. Missing artifacts return [`EvalError::InvalidConfig`].
#[allow(clippy::too_many_arguments)]
pub fn build_tables(
    grid: &EvalGrid,
    accum: &Accumulated,
    dataset: &CocoDataset,
    detections: Option<&CocoDetections>,
    retained_ious: Option<&RetainedIous>,
    iou_thresholds: &[f64],
    max_dets: &[usize],
    request: TablesRequest,
    config: &TablesConfig,
) -> Result<Tables, EvalError> {
    let mut out = Tables::default();
    if request.per_class {
        let support = aggregate_per_class_support(grid, 0);
        out.per_class = Some(build_per_class(
            accum,
            dataset,
            iou_thresholds,
            max_dets,
            &support,
        )?);
    }
    if request.per_image {
        out.per_image = Some(build_per_image(grid, dataset, iou_thresholds)?);
    }
    if request.per_detection {
        let dets = detections.ok_or_else(|| EvalError::InvalidConfig {
            detail: "per_detection requires detections to be threaded through \
                     build_tables; pass Some(&CocoDetections)"
                .into(),
        })?;
        out.per_detection = Some(build_per_detection(
            grid,
            dets,
            iou_thresholds,
            retained_ious,
            config,
        )?);
    }
    if request.per_pair {
        let ious = retained_ious.ok_or_else(|| EvalError::InvalidConfig {
            detail: "per_pair requires retained IoU matrices; build the upstream \
                     evaluator with EvaluateParams::retain_iou=true (or pass \
                     retain_iou=True at StreamingEvaluator construction)"
                .into(),
        })?;
        out.per_pair = Some(build_per_pair(grid, ious, config)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accumulate::Accumulated;
    use crate::dataset::{CategoryId, CategoryMeta, CocoDataset, ImageMeta};
    use crate::parity::iou_thresholds;
    use ndarray::{Array4, Array5};

    fn dataset_with_two_categories() -> CocoDataset {
        let images = vec![ImageMeta {
            id: crate::dataset::ImageId(1),
            width: 100,
            height: 100,
            file_name: None,
        }];
        let categories = vec![
            CategoryMeta {
                id: CategoryId(2),
                name: "cat".into(),
                supercategory: None,
            },
            CategoryMeta {
                id: CategoryId(1),
                name: "dog".into(),
                supercategory: None,
            },
        ];
        CocoDataset::from_parts(images, Vec::new(), categories).unwrap()
    }

    #[test]
    fn tables_request_requires_iou_retention_only_for_dt_pair() {
        assert!(!TablesRequest::CHEAP.requires_iou_retention());
        assert!(TablesRequest {
            per_detection: true,
            ..TablesRequest::default()
        }
        .requires_iou_retention());
        assert!(TablesRequest {
            per_pair: true,
            ..TablesRequest::default()
        }
        .requires_iou_retention());
    }

    #[test]
    fn build_per_class_emits_one_row_per_category_in_id_ascending_order() {
        // Dataset has cats with ids 2, 1; sorted-ascending puts dog (1)
        // at K=0 and cat (2) at K=1. The per-K AP value is encoded into
        // the precision tensor at [t, r, k, area=ALL, m=last] so we can
        // assert the K-axis ordering is honored.
        let dataset = dataset_with_two_categories();
        let iou_thr = iou_thresholds();
        let max_dets = [1usize, 10, 100];
        let n_t = iou_thr.len();
        let n_r = 101;
        let n_k = 2;
        let n_a = 4;
        let n_m = 3;

        let mut precision = Array5::<f64>::from_elem((n_t, n_r, n_k, n_a, n_m), -1.0);
        let mut recall = Array4::<f64>::from_elem((n_t, n_k, n_a, n_m), -1.0);

        // dog (k=0): AP=0.6 at (all, m=last). cat (k=1): AP=0.8.
        precision
            .index_axis_mut(Axis(2), 0)
            .index_axis_mut(Axis(2), 0) // area=all
            .index_axis_mut(Axis(2), 2) // m=last
            .fill(0.6);
        precision
            .index_axis_mut(Axis(2), 1)
            .index_axis_mut(Axis(2), 0)
            .index_axis_mut(Axis(2), 2)
            .fill(0.8);

        // AR_max_100 different per category to verify recall pathway.
        recall[[0, 0, 0, 2]] = 0.5;
        recall[[0, 1, 0, 2]] = 0.9;

        let accum = Accumulated {
            precision,
            recall,
            scores: Array5::<f64>::from_elem((n_t, n_r, n_k, n_a, n_m), -1.0),
        };
        let support = PerClassSupport {
            n_gt: vec![3, 4],
            n_dt: vec![5, 6],
        };

        let table = build_per_class(&accum, &dataset, iou_thr, &max_dets, &support).unwrap();
        assert_eq!(table.len(), 2);

        // K=0 → category id 1 (dog), K=1 → category id 2 (cat).
        assert_eq!(table.category_id, vec![1, 2]);
        assert_eq!(
            table.category_name,
            vec!["dog".to_string(), "cat".to_string()]
        );
        assert!((table.ap[0].unwrap() - 0.6).abs() < 1e-12);
        assert!((table.ap[1].unwrap() - 0.8).abs() < 1e-12);
        assert_eq!(table.n_gt, vec![3, 4]);
        assert_eq!(table.n_dt, vec![5, 6]);

        // Recall slice was only populated at t=0; mean of one
        // non-sentinel value is that value.
        assert!((table.ar_max_100[0].unwrap() - 0.5).abs() < 1e-12);
        assert!((table.ar_max_100[1].unwrap() - 0.9).abs() < 1e-12);
    }

    #[test]
    fn build_per_class_emits_null_for_all_sentinel_cells() {
        let dataset = dataset_with_two_categories();
        let iou_thr = iou_thresholds();
        let max_dets = [1usize, 10, 100];
        let n_t = iou_thr.len();
        let accum = Accumulated {
            precision: Array5::<f64>::from_elem((n_t, 101, 2, 4, 3), -1.0),
            recall: Array4::<f64>::from_elem((n_t, 2, 4, 3), -1.0),
            scores: Array5::<f64>::from_elem((n_t, 101, 2, 4, 3), -1.0),
        };
        let support = PerClassSupport::zeros(2);
        let table = build_per_class(&accum, &dataset, iou_thr, &max_dets, &support).unwrap();
        assert!(table.ap.iter().all(Option::is_none));
        assert!(table.ar_max_100.iter().all(Option::is_none));
    }

    #[test]
    fn build_per_class_collapsed_use_cats_false_returns_single_row() {
        let dataset = dataset_with_two_categories();
        let iou_thr = iou_thresholds();
        let max_dets = [1usize, 10, 100];
        let n_t = iou_thr.len();
        // K=1 — the use_cats=false collapsed shape.
        let accum = Accumulated {
            precision: Array5::<f64>::from_elem((n_t, 101, 1, 4, 3), 0.7),
            recall: Array4::<f64>::from_elem((n_t, 1, 4, 3), 0.7),
            scores: Array5::<f64>::from_elem((n_t, 101, 1, 4, 3), 0.7),
        };
        let support = PerClassSupport {
            n_gt: vec![100],
            n_dt: vec![200],
        };
        let table = build_per_class(&accum, &dataset, iou_thr, &max_dets, &support).unwrap();
        assert_eq!(table.len(), 1);
        assert_eq!(table.category_id, vec![COLLAPSED_CATEGORY_SENTINEL]);
        assert_eq!(table.category_name, vec!["(all categories)".to_string()]);
        assert!((table.ap[0].unwrap() - 0.7).abs() < 1e-12);
    }

    #[test]
    fn build_per_class_rejects_a_axis_size_other_than_4() {
        let dataset = dataset_with_two_categories();
        let iou_thr = iou_thresholds();
        let max_dets = [20usize];
        let n_t = iou_thr.len();
        // 3-bucket A-axis (the keypoints layout) must be rejected.
        let accum = Accumulated {
            precision: Array5::<f64>::from_elem((n_t, 101, 2, 3, 1), 0.5),
            recall: Array4::<f64>::from_elem((n_t, 2, 3, 1), 0.5),
            scores: Array5::<f64>::from_elem((n_t, 101, 2, 3, 1), 0.5),
        };
        let support = PerClassSupport::zeros(2);
        let err = build_per_class(&accum, &dataset, iou_thr, &max_dets, &support).unwrap_err();
        assert!(matches!(err, EvalError::DimensionMismatch { .. }));
    }

    fn perfect_match_grid_two_images() -> (EvalGrid, CocoDataset) {
        // Two images, one category. Image 1 has a perfect-match DT,
        // image 2 has an unmatched DT (FP) and an unmatched GT (FN).
        // Bbox kernel for simplicity.
        use crate::dataset::{Bbox as DsBbox, CocoAnnotation, DetectionInput};
        use crate::evaluate::{evaluate_bbox, AreaRange, EvaluateParams};
        use crate::parity::{iou_thresholds, ParityMode};
        let images = vec![
            ImageMeta {
                id: crate::dataset::ImageId(1),
                width: 100,
                height: 100,
                file_name: None,
            },
            ImageMeta {
                id: crate::dataset::ImageId(2),
                width: 100,
                height: 100,
                file_name: None,
            },
        ];
        let categories = vec![CategoryMeta {
            id: CategoryId(1),
            name: "thing".into(),
            supercategory: None,
        }];
        let anns = vec![
            CocoAnnotation {
                id: crate::dataset::AnnId(1),
                image_id: crate::dataset::ImageId(1),
                category_id: CategoryId(1),
                area: 100.0,
                is_crowd: false,
                ignore_flag: None,
                bbox: DsBbox {
                    x: 0.0,
                    y: 0.0,
                    w: 10.0,
                    h: 10.0,
                },
                segmentation: None,
                keypoints: None,
                num_keypoints: None,
            },
            CocoAnnotation {
                id: crate::dataset::AnnId(2),
                image_id: crate::dataset::ImageId(2),
                category_id: CategoryId(1),
                area: 100.0,
                is_crowd: false,
                ignore_flag: None,
                bbox: DsBbox {
                    x: 0.0,
                    y: 0.0,
                    w: 10.0,
                    h: 10.0,
                },
                segmentation: None,
                keypoints: None,
                num_keypoints: None,
            },
        ];
        let dataset = CocoDataset::from_parts(images, anns, categories).unwrap();

        let dt_inputs = vec![
            DetectionInput {
                id: None,
                image_id: crate::dataset::ImageId(1),
                category_id: CategoryId(1),
                score: 0.9,
                bbox: DsBbox {
                    x: 0.0,
                    y: 0.0,
                    w: 10.0,
                    h: 10.0,
                },
                segmentation: None,
                keypoints: None,
                num_keypoints: None,
            },
            DetectionInput {
                id: None,
                image_id: crate::dataset::ImageId(2),
                category_id: CategoryId(1),
                score: 0.8,
                // Far from GT — unmatched FP.
                bbox: DsBbox {
                    x: 50.0,
                    y: 50.0,
                    w: 10.0,
                    h: 10.0,
                },
                segmentation: None,
                keypoints: None,
                num_keypoints: None,
            },
        ];
        let detections = crate::dataset::CocoDetections::from_inputs(dt_inputs).unwrap();
        let area = AreaRange::coco_default();
        let grid = evaluate_bbox(
            &dataset,
            &detections,
            EvaluateParams {
                iou_thresholds: iou_thresholds(),
                area_ranges: &area,
                max_dets_per_image: 100,
                use_cats: true,
                retain_iou: false,
            },
            ParityMode::Corrected,
        )
        .unwrap();
        (grid, dataset)
    }

    #[test]
    fn build_per_image_counts_tp_fp_fn_against_perfect_and_unmatched_pairs() {
        let (grid, dataset) = perfect_match_grid_two_images();
        let table = build_per_image(&grid, &dataset, crate::parity::iou_thresholds()).unwrap();
        assert_eq!(table.len(), 2);
        assert_eq!(table.image_id, vec![1, 2]);
        // Both images have one non-ignore GT.
        assert_eq!(table.n_gt, vec![1, 1]);
        // Image 1: 1 DT (perfect match → TP). Image 2: 1 DT (unmatched FP).
        assert_eq!(table.n_dt, vec![1, 1]);
        assert_eq!(table.tp_at_50, vec![1, 0]);
        assert_eq!(table.fp_at_50, vec![0, 1]);
        assert_eq!(table.fn_at_50, vec![0, 1]);
        assert_eq!(table.tp_at_75, vec![1, 0]);
        assert_eq!(table.fp_at_75, vec![0, 1]);
        assert_eq!(table.fn_at_75, vec![0, 1]);
        // Image 1's TPs hold across all 10 thresholds (perfect IoU=1.0);
        // image 2 has zero TPs across all thresholds.
        assert_eq!(table.tp_mean_iou, vec![1, 0]);
    }

    #[test]
    fn build_per_image_excludes_crowd_matched_dts_from_tp() {
        // Crowd GT + a DT overlapping it → DT carries dt_ignore (B6),
        // so neither TP nor FP is incremented. n_gt also drops the
        // crowd GT (gt_ignore=true).
        use crate::dataset::{Bbox as DsBbox, CocoAnnotation, DetectionInput};
        use crate::evaluate::{evaluate_bbox, AreaRange, EvaluateParams};
        use crate::parity::{iou_thresholds, ParityMode};
        let images = vec![ImageMeta {
            id: crate::dataset::ImageId(1),
            width: 100,
            height: 100,
            file_name: None,
        }];
        let categories = vec![CategoryMeta {
            id: CategoryId(1),
            name: "thing".into(),
            supercategory: None,
        }];
        let anns = vec![CocoAnnotation {
            id: crate::dataset::AnnId(1),
            image_id: crate::dataset::ImageId(1),
            category_id: CategoryId(1),
            area: 100.0,
            // Crowd region — gt_ignore becomes true.
            is_crowd: true,
            ignore_flag: None,
            bbox: DsBbox {
                x: 0.0,
                y: 0.0,
                w: 10.0,
                h: 10.0,
            },
            segmentation: None,
            keypoints: None,
            num_keypoints: None,
        }];
        let dataset = CocoDataset::from_parts(images, anns, categories).unwrap();
        let dt_inputs = vec![DetectionInput {
            id: None,
            image_id: crate::dataset::ImageId(1),
            category_id: CategoryId(1),
            score: 0.9,
            bbox: DsBbox {
                x: 0.0,
                y: 0.0,
                w: 10.0,
                h: 10.0,
            },
            segmentation: None,
            keypoints: None,
            num_keypoints: None,
        }];
        let detections = crate::dataset::CocoDetections::from_inputs(dt_inputs).unwrap();
        let area = AreaRange::coco_default();
        let grid = evaluate_bbox(
            &dataset,
            &detections,
            EvaluateParams {
                iou_thresholds: iou_thresholds(),
                area_ranges: &area,
                max_dets_per_image: 100,
                use_cats: true,
                retain_iou: false,
            },
            ParityMode::Corrected,
        )
        .unwrap();
        let table = build_per_image(&grid, &dataset, iou_thresholds()).unwrap();
        assert_eq!(table.n_gt, vec![0]);
        assert_eq!(table.tp_at_50, vec![0]);
        assert_eq!(table.fp_at_50, vec![0]);
        assert_eq!(table.fn_at_50, vec![0]);
    }

    #[test]
    fn build_per_image_rejects_non_detection_grid() {
        // Build an EvalGrid with n_area_ranges=3 (kp shape) by hand.
        let grid = EvalGrid {
            eval_imgs: vec![None; 3],
            eval_imgs_meta: vec![None; 3],
            n_categories: 1,
            n_area_ranges: 3,
            n_images: 1,
            retained_ious: None,
        };
        let dataset = dataset_with_two_categories();
        let err = build_per_image(&grid, &dataset, crate::parity::iou_thresholds()).unwrap_err();
        assert!(matches!(err, EvalError::DimensionMismatch { .. }));
    }

    #[test]
    fn build_tables_dispatches_per_image_and_per_class() {
        let (grid, dataset) = perfect_match_grid_two_images();
        let max_dets = [1usize, 10, 100];
        // Drive the same pipeline the FFI runs end-to-end.
        let p = crate::AccumulateParams {
            iou_thresholds: crate::iou_thresholds(),
            recall_thresholds: crate::recall_thresholds(),
            max_dets: &max_dets,
            n_categories: grid.n_categories,
            n_area_ranges: grid.n_area_ranges,
            n_images: grid.n_images,
        };
        let accum = crate::accumulate(&grid.eval_imgs, p, crate::ParityMode::Corrected).unwrap();

        let tables = build_tables(
            &grid,
            &accum,
            &dataset,
            None,
            None,
            crate::iou_thresholds(),
            &max_dets,
            TablesRequest::CHEAP,
            &TablesConfig::default(),
        )
        .unwrap();
        assert!(tables.per_image.is_some());
        assert!(tables.per_class.is_some());
    }

    #[test]
    fn retain_iou_flag_does_not_perturb_the_summary() {
        // ADR-0019 Week 2.3: `retain_iou=true` is opt-in extra
        // bookkeeping; the matching engine and accumulator must
        // produce bit-identical Summary output to the no-retention
        // path. Pin the contract on the perfect-match fixture.
        use crate::dataset::{Bbox as DsBbox, CocoAnnotation, DetectionInput};
        use crate::evaluate::{evaluate_bbox, AreaRange, EvaluateParams};
        use crate::parity::{iou_thresholds, ParityMode};
        let images = vec![ImageMeta {
            id: crate::dataset::ImageId(1),
            width: 100,
            height: 100,
            file_name: None,
        }];
        let categories = vec![CategoryMeta {
            id: CategoryId(1),
            name: "thing".into(),
            supercategory: None,
        }];
        let anns = vec![CocoAnnotation {
            id: crate::dataset::AnnId(1),
            image_id: crate::dataset::ImageId(1),
            category_id: CategoryId(1),
            area: 100.0,
            is_crowd: false,
            ignore_flag: None,
            bbox: DsBbox {
                x: 0.0,
                y: 0.0,
                w: 10.0,
                h: 10.0,
            },
            segmentation: None,
            keypoints: None,
            num_keypoints: None,
        }];
        let dataset = CocoDataset::from_parts(images, anns, categories).unwrap();
        let dt_inputs = vec![DetectionInput {
            id: None,
            image_id: crate::dataset::ImageId(1),
            category_id: CategoryId(1),
            score: 0.9,
            bbox: DsBbox {
                x: 0.0,
                y: 0.0,
                w: 10.0,
                h: 10.0,
            },
            segmentation: None,
            keypoints: None,
            num_keypoints: None,
        }];
        let detections = crate::dataset::CocoDetections::from_inputs(dt_inputs).unwrap();
        let area = AreaRange::coco_default();
        let max_dets = [1usize, 10, 100];

        let mut params_off = EvaluateParams {
            iou_thresholds: iou_thresholds(),
            area_ranges: &area,
            max_dets_per_image: 100,
            use_cats: true,
            retain_iou: false,
        };
        let grid_off =
            evaluate_bbox(&dataset, &detections, params_off, ParityMode::Corrected).unwrap();
        params_off.retain_iou = true;
        let grid_on =
            evaluate_bbox(&dataset, &detections, params_off, ParityMode::Corrected).unwrap();

        // The `eval_imgs` slice is what `accumulate` consumes; require
        // the count of populated cells matches and that the retention
        // field is present iff the flag is on.
        assert_eq!(grid_off.eval_imgs.len(), grid_on.eval_imgs.len());
        assert!(grid_off.retained_ious.is_none());
        assert!(grid_on.retained_ious.is_some());
        let retained = grid_on.retained_ious.as_ref().unwrap();
        assert_eq!(retained.len(), 1);
        assert!(retained.get(0, 0).is_some());

        // Drive the same accumulate→summarize over both grids; assert
        // bit-equal stats. This is the headline contract of the
        // retain_iou flag.
        let p = crate::AccumulateParams {
            iou_thresholds: iou_thresholds(),
            recall_thresholds: crate::recall_thresholds(),
            max_dets: &max_dets,
            n_categories: grid_off.n_categories,
            n_area_ranges: grid_off.n_area_ranges,
            n_images: grid_off.n_images,
        };
        let acc_off = crate::accumulate(&grid_off.eval_imgs, p, ParityMode::Corrected).unwrap();
        let acc_on = crate::accumulate(&grid_on.eval_imgs, p, ParityMode::Corrected).unwrap();
        let sum_off = crate::summarize_detection(&acc_off, iou_thresholds(), &max_dets).unwrap();
        let sum_on = crate::summarize_detection(&acc_on, iou_thresholds(), &max_dets).unwrap();
        for (a, b) in sum_off.stats().iter().zip(sum_on.stats().iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "stat drift: off={a} on={b}");
        }
    }

    #[test]
    fn build_tables_per_detection_without_detections_returns_invalid_config() {
        // Caller asks for per_detection but doesn't supply CocoDetections —
        // surface a clear error pointing at the missing argument.
        let (grid, dataset) = perfect_match_grid_two_images();
        let max_dets = [1usize, 10, 100];
        let p = crate::AccumulateParams {
            iou_thresholds: crate::iou_thresholds(),
            recall_thresholds: crate::recall_thresholds(),
            max_dets: &max_dets,
            n_categories: grid.n_categories,
            n_area_ranges: grid.n_area_ranges,
            n_images: grid.n_images,
        };
        let accum = crate::accumulate(&grid.eval_imgs, p, crate::ParityMode::Corrected).unwrap();
        let request = TablesRequest {
            per_detection: true,
            ..TablesRequest::default()
        };
        let err = build_tables(
            &grid,
            &accum,
            &dataset,
            None,
            None,
            crate::iou_thresholds(),
            &max_dets,
            request,
            &TablesConfig::default(),
        )
        .unwrap_err();
        assert!(matches!(err, EvalError::InvalidConfig { .. }));
    }

    #[test]
    fn build_tables_per_pair_without_retention_returns_invalid_config() {
        let (grid, dataset) = perfect_match_grid_two_images();
        let max_dets = [1usize, 10, 100];
        let p = crate::AccumulateParams {
            iou_thresholds: crate::iou_thresholds(),
            recall_thresholds: crate::recall_thresholds(),
            max_dets: &max_dets,
            n_categories: grid.n_categories,
            n_area_ranges: grid.n_area_ranges,
            n_images: grid.n_images,
        };
        let accum = crate::accumulate(&grid.eval_imgs, p, crate::ParityMode::Corrected).unwrap();
        let request = TablesRequest {
            per_pair: true,
            ..TablesRequest::default()
        };
        let err = build_tables(
            &grid,
            &accum,
            &dataset,
            None,
            None,
            crate::iou_thresholds(),
            &max_dets,
            request,
            &TablesConfig::default(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(matches!(err, EvalError::InvalidConfig { .. }));
        assert!(
            msg.contains("retain_iou"),
            "error must name retain_iou: {msg}"
        );
    }

    #[test]
    fn build_per_pair_overflow_fires_inside_push_loop() {
        // Tiny IoU floor + tiny cap: matrix has 4 elements, all >= 0.0,
        // so every pair would push if we let it. Cap=2 fires on the
        // 3rd push. The check happens before the column grows past
        // the cap (per ADR landmine #4).
        let mut store = RetainedIous::new();
        let iou = ndarray::Array2::<f64>::from_shape_vec((2, 2), vec![0.5, 0.6, 0.7, 0.8]).unwrap();
        store.insert(0, 0, iou);
        let grid = EvalGrid {
            eval_imgs: vec![None],
            eval_imgs_meta: vec![Some(Box::new(crate::evaluate::EvalImageMeta {
                image_id: 1,
                category_id: 1,
                area_rng: [0.0, f64::INFINITY],
                max_det: 100,
                dt_ids: vec![10, 20],
                gt_ids: vec![100, 200],
                dt_matches: ndarray::Array2::<i64>::zeros((10, 2)),
                gt_matches: ndarray::Array2::<i64>::zeros((10, 2)),
            }))],
            n_categories: 1,
            n_area_ranges: 1,
            n_images: 1,
            retained_ious: Some(store.clone()),
        };
        let cfg = TablesConfig {
            per_pair_iou_floor: 0.0,
            per_pair_max_rows: 2,
            ..TablesConfig::default()
        };
        let err = build_per_pair(&grid, &store, &cfg).unwrap_err();
        assert!(matches!(
            err,
            EvalError::PerPairOverflow {
                observed: 3,
                cap: 2
            }
        ));
    }

    #[test]
    fn build_per_pair_filters_below_iou_floor_and_emits_above() {
        // Same matrix as the overflow test but with floor=0.65 — keeps
        // [0.7, 0.8] pairs only.
        let mut store = RetainedIous::new();
        let iou = ndarray::Array2::<f64>::from_shape_vec((2, 2), vec![0.5, 0.6, 0.7, 0.8]).unwrap();
        store.insert(0, 0, iou);
        let grid = EvalGrid {
            eval_imgs: vec![None],
            eval_imgs_meta: vec![Some(Box::new(crate::evaluate::EvalImageMeta {
                image_id: 1,
                category_id: 1,
                area_rng: [0.0, f64::INFINITY],
                max_det: 100,
                dt_ids: vec![10, 20],
                gt_ids: vec![100, 200],
                dt_matches: ndarray::Array2::<i64>::zeros((10, 2)),
                gt_matches: ndarray::Array2::<i64>::zeros((10, 2)),
            }))],
            n_categories: 1,
            n_area_ranges: 1,
            n_images: 1,
            retained_ious: Some(store.clone()),
        };
        let cfg = TablesConfig {
            per_pair_iou_floor: 0.65,
            ..TablesConfig::default()
        };
        let table = build_per_pair(&grid, &store, &cfg).unwrap();
        assert_eq!(table.len(), 2);
        assert_eq!(table.iou.to_vec(), vec![0.7, 0.8]);
        // Rows are emitted in (gt, dt) order: (g=1,d=0)=0.7 and (g=1,d=1)=0.8.
        assert_eq!(table.detection_id, vec![10, 20]);
        assert_eq!(table.ground_truth_id, vec![200, 200]);
    }

    #[test]
    fn build_per_detection_marks_perfect_match_as_tp_and_unmatched_as_fp() {
        let (grid, dataset) = perfect_match_grid_two_images();
        let dt_inputs = vec![
            crate::dataset::DetectionInput {
                id: None,
                image_id: crate::dataset::ImageId(1),
                category_id: CategoryId(1),
                score: 0.9,
                bbox: crate::dataset::Bbox {
                    x: 0.0,
                    y: 0.0,
                    w: 10.0,
                    h: 10.0,
                },
                segmentation: None,
                keypoints: None,
                num_keypoints: None,
            },
            crate::dataset::DetectionInput {
                id: None,
                image_id: crate::dataset::ImageId(2),
                category_id: CategoryId(1),
                score: 0.8,
                bbox: crate::dataset::Bbox {
                    x: 50.0,
                    y: 50.0,
                    w: 10.0,
                    h: 10.0,
                },
                segmentation: None,
                keypoints: None,
                num_keypoints: None,
            },
        ];
        let detections = crate::dataset::CocoDetections::from_inputs(dt_inputs).unwrap();
        let _ = dataset; // dataset is unused here — per_detection doesn't need it.
        let table = build_per_detection(
            &grid,
            &detections,
            crate::iou_thresholds(),
            None,
            &TablesConfig::default(),
        )
        .unwrap();
        assert_eq!(table.len(), 2);
        // Sort rows by score-desc to map to the matching engine's
        // sorted order. Image 1 (score=0.9) is the perfect-match
        // (TP); image 2 (score=0.8) is the unmatched FP.
        let statuses: Vec<MatchStatus> = table.match_status_at_50.clone();
        // Two rows total; one TP, one FP.
        let tp_count = statuses
            .iter()
            .filter(|s| **s == MatchStatus::TruePositive)
            .count();
        let fp_count = statuses
            .iter()
            .filter(|s| **s == MatchStatus::FalsePositive)
            .count();
        assert_eq!(tp_count, 1);
        assert_eq!(fp_count, 1);
        // best_iou is None when retained_ious=None.
        assert!(table.best_iou.iter().all(Option::is_none));
        // bbox is None when geometry flag is off (default).
        assert!(table.bbox.is_none());
    }

    #[test]
    fn build_per_class_rejects_max_dets_missing_canonical_ladder() {
        let dataset = dataset_with_two_categories();
        let iou_thr = iou_thresholds();
        let max_dets = [10usize, 100]; // missing 1
        let n_t = iou_thr.len();
        let accum = Accumulated {
            precision: Array5::<f64>::from_elem((n_t, 101, 2, 4, 2), 0.5),
            recall: Array4::<f64>::from_elem((n_t, 2, 4, 2), 0.5),
            scores: Array5::<f64>::from_elem((n_t, 101, 2, 4, 2), 0.5),
        };
        let support = PerClassSupport::zeros(2);
        let err = build_per_class(&accum, &dataset, iou_thr, &max_dets, &support).unwrap_err();
        assert!(matches!(err, EvalError::InvalidConfig { .. }));
    }
}
