//! Per-category aggregation and the [`PanopticSummary`] public type.
//!
//! Folds the per-image [`PqStat`] outputs from [`crate::attribute`] into
//! per-category [`ClassPanopticStats`], then applies the W1–W8 quirks
//! to produce the final unweighted means + things/stuff buckets. The
//! one place "innocent refactor" produces wrong numbers is W7 —
//! global SQ is the **mean of per-category SQs**, *not*
//! `total_iou / total_TP`. The two are not equal on long-tailed
//! datasets.
//!
//! Strict-vs-corrected divergence in this module:
//! - **W6 corrected** (default) returns zeros when the per-category
//!   filter is empty; **strict** raises [`PanopticError::EmptyCategoryFilter`]
//!   to match panopticapi's `ZeroDivisionError` shape.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::attribute::{attribute_image, PqStat};
use crate::boundary::{BoundaryConfig, BoundaryScratch};
use crate::dataset::{
    CategoryId, CategoryMeta, ImageEntry, ImageId, PanopticDataset, PanopticPredictions,
};
use crate::error::PanopticError;
use crate::kernel::{pq_image_at_threshold, pq_image_at_threshold_with_boundary};
use crate::parity::ParityMode;
use crate::parity::PANOPTIC_IOU_THRESHOLD;

/// Per-class PQ row. Strict superset of panopticapi's `{pq, sq, rq}`
/// shape (quirk **W8**); the count fields are vernier-only and the
/// FFI's `to_dict(strict=True)` shim drops them to match the upstream
/// dict shape exactly.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ClassPanopticStats {
    /// Panoptic Quality. Computed directly via W1
    /// `PQ_c = sum_iou_c / (TP_c + 0.5*FP_c + 0.5*FN_c)`, **not**
    /// `SQ_c * RQ_c` — those are algebraically equal but f64
    /// non-associative; bit-equality with panopticapi requires the
    /// direct form.
    pub pq: f64,
    /// Segmentation Quality. `sum_iou_c / TP_c`, or `0.0` when
    /// `TP_c == 0`.
    pub sq: f64,
    /// Recognition Quality. `TP_c / (TP_c + 0.5*FP_c + 0.5*FN_c)`,
    /// or `0.0` when the denominator is zero.
    pub rq: f64,
    /// Raw TP count (vernier-only, not in panopticapi's per_class
    /// output per W8).
    pub n_tp: u64,
    /// Raw FP count (vernier-only).
    pub n_fp: u64,
    /// Raw FN count (vernier-only).
    pub n_fn: u64,
    /// Sum of IoU across the TP segments for this category.
    pub iou_sum: f64,
}

/// Per-group panoptic rollup (ADR-0042).
///
/// Built when [`SummarizeOptions::class_groups`] is set: each group
/// fold averages PQ / SQ / RQ over its member category ids using the
/// quirk **W6** convention (skip categories with `n_tp + n_fp + n_fn
/// == 0`). The rollup is a post-summarize aggregation over the
/// per-class table; the kernel and accumulator are unaffected.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupPanopticStats {
    /// Group label, as passed in by the caller.
    pub label: String,
    /// Category ids that compose this group, sorted ascending.
    pub member_category_ids: Vec<CategoryId>,
    /// PQ averaged over the group's contributing members.
    pub pq: f64,
    /// SQ averaged over the group's contributing members.
    pub sq: f64,
    /// RQ averaged over the group's contributing members.
    pub rq: f64,
    /// Number of contributing categories in the group (W5 / W6).
    pub n: usize,
}

/// Optional axes for [`summarize_from_acc_with_options`] (ADR-0042).
///
/// `category_filter` restricts the global PQ/SQ/RQ averages to the
/// supplied category subset; the per-class breakdown remains
/// complete. `class_groups` populates [`PanopticSummary::per_group`]
/// with one entry per `(label, ids)` partition. Both default to
/// `None`; the unparametrized [`summarize_from_acc`] uses defaults.
#[derive(Debug, Clone, Default)]
pub struct SummarizeOptions<'a> {
    /// Subset of category ids that contribute to the global PQ / SQ /
    /// RQ averages and to the things / stuff buckets. `None` means
    /// "every category contributes" (the panopticapi default).
    pub category_filter: Option<&'a [u32]>,
    /// Class-id partitions for the per-group rollup.
    pub class_groups: Option<&'a [(String, Vec<u32>)]>,
}

/// Top-level panoptic evaluation result.
///
/// `pq_things` / `pq_stuff` (and the SQ/RQ variants) are `None` when
/// `things_stuff_split=false` was passed to [`evaluate`]. `n` /
/// `n_things` / `n_stuff` carry the count of contributing categories
/// per W5 — useful for diagnosing all-zero rows on long-tailed
/// datasets.
#[derive(Debug, Clone, PartialEq)]
pub struct PanopticSummary {
    /// Global PQ — unweighted mean over the present-categories
    /// subset (W3).
    pub pq: f64,
    /// Global SQ — unweighted mean of per-category SQs over the
    /// present subset (W7; **not** pooled `total_iou / total_TP`).
    pub sq: f64,
    /// Global RQ — unweighted mean over the present subset.
    pub rq: f64,
    /// PQ over the things subset (W4); `None` when split is disabled.
    pub pq_things: Option<f64>,
    /// SQ over the things subset; `None` when split is disabled.
    pub sq_things: Option<f64>,
    /// RQ over the things subset; `None` when split is disabled.
    pub rq_things: Option<f64>,
    /// PQ over the stuff subset (W4); `None` when split is disabled.
    pub pq_stuff: Option<f64>,
    /// SQ over the stuff subset; `None` when split is disabled.
    pub sq_stuff: Option<f64>,
    /// RQ over the stuff subset; `None` when split is disabled.
    pub rq_stuff: Option<f64>,
    /// Per-category rows, keyed by COCO category id. `BTreeMap` for
    /// deterministic iteration order — important when the FFI
    /// serializes to a Python dict and the user round-trips it.
    pub per_class: BTreeMap<CategoryId, ClassPanopticStats>,
    /// Number of categories that contributed to the global mean (W5).
    /// Equal to `per_class.iter().filter(|s| s.n_tp + s.n_fp + s.n_fn > 0).count()`.
    pub n: usize,
    /// Number of contributing things categories (`None` when split
    /// is disabled).
    pub n_things: Option<usize>,
    /// Number of contributing stuff categories (`None` when split
    /// is disabled).
    pub n_stuff: Option<usize>,
    /// Per-group rollup (ADR-0042). Populated only when the caller
    /// passes [`SummarizeOptions::class_groups`]; empty for the
    /// canonical no-grouping path. Sorted by label via `BTreeMap`.
    pub per_group: BTreeMap<String, GroupPanopticStats>,
}

/// Convert one accumulated [`PqStat`] into a [`ClassPanopticStats`]
/// row. Quirk **W1** ratios; the per-image fold has already been
/// done by `attribute_image` upstream.
fn class_stats(stat: PqStat) -> ClassPanopticStats {
    let tp = stat.n_tp as f64;
    let fp = stat.n_fp as f64;
    let fn_count = stat.n_fn as f64;
    let denom = tp + 0.5 * (fp + fn_count);
    let pq = if denom == 0.0 {
        0.0
    } else {
        stat.sum_iou / denom
    };
    let sq = if stat.n_tp == 0 {
        0.0
    } else {
        stat.sum_iou / tp
    };
    let rq = if denom == 0.0 { 0.0 } else { tp / denom };
    ClassPanopticStats {
        pq,
        sq,
        rq,
        n_tp: stat.n_tp,
        n_fp: stat.n_fp,
        n_fn: stat.n_fn,
        iou_sum: stat.sum_iou,
    }
}

/// Unweighted mean over the W2-filtered subset (categories with
/// `n_tp + n_fp + n_fn > 0`). Returns `(pq, sq, rq, n)`. Quirk **W6**
/// wraps this: strict mode raises on `n == 0`, corrected returns zeros.
fn average(rows: impl Iterator<Item = ClassPanopticStats>) -> (f64, f64, f64, usize) {
    let mut pq_sum = 0.0;
    let mut sq_sum = 0.0;
    let mut rq_sum = 0.0;
    let mut n = 0usize;
    for r in rows {
        if r.n_tp + r.n_fp + r.n_fn == 0 {
            continue;
        }
        pq_sum += r.pq;
        sq_sum += r.sq;
        rq_sum += r.rq;
        n += 1;
    }
    if n == 0 {
        (0.0, 0.0, 0.0, 0)
    } else {
        let nf = n as f64;
        (pq_sum / nf, sq_sum / nf, rq_sum / nf, n)
    }
}

/// Apply quirk **W6** to the (possibly empty) average result. In
/// `Strict` mode an empty filter raises; in `Corrected` mode the
/// zero-tuple from [`average`] is returned as-is.
fn finalize_average(
    raw: (f64, f64, f64, usize),
    mode: ParityMode,
    context: &'static str,
) -> Result<(f64, f64, f64, usize), PanopticError> {
    let (pq, sq, rq, n) = raw;
    if n == 0 && mode == ParityMode::Strict {
        return Err(PanopticError::EmptyCategoryFilter { context });
    }
    Ok((pq, sq, rq, n))
}

/// Top-level panoptic evaluation orchestrator (single-threaded per
/// ADR-0006 + quirk **X1** corrected: bypasses panopticapi's
/// multiprocessing pool entirely).
///
/// Returns a [`PanopticSummary`] with global, things, and stuff
/// buckets (the latter two `None` when `things_stuff_split=false`).
///
/// Predictions must cover every GT image (quirk **Y4**); a missing
/// `image_id` raises [`PanopticError::MissingPredictionsForImage`].
/// Pred-only images are silently ignored, mirroring upstream
/// (quirk **Y5**, `evaluation.py:213-216`).
pub fn evaluate(
    gt: &PanopticDataset,
    dt: &PanopticPredictions,
    mode: ParityMode,
    things_stuff_split: bool,
) -> Result<PanopticSummary, PanopticError> {
    evaluate_with_options(
        gt,
        dt,
        mode,
        things_stuff_split,
        &EvaluateOptions::default(),
    )
}

/// Optional ADR-0042 axes for [`evaluate_with_options`].
///
/// `pq_iou_threshold` overrides the canonical 0.5 matching gate.
/// `categories_override` replaces the dataset's `isthing` flags
/// (used by `stuff_thing_partition` to override the dataset-derived
/// classification). The summarize-time axes (`category_filter`,
/// `class_groups`) live on [`SummarizeOptions`] and flow through
/// `summarize`.
#[derive(Default)]
pub struct EvaluateOptions<'a> {
    /// Custom IoU threshold for the matching gate (`iou >
    /// threshold`). `None` falls back to the canonical
    /// [`crate::parity::PANOPTIC_IOU_THRESHOLD`] = 0.5.
    pub pq_iou_threshold: Option<f64>,
    /// Replacement categories map. When `Some`, the things / stuff
    /// split keys off this rather than `gt.categories`. Used to
    /// honor a user-supplied `stuff_thing_partition`.
    pub categories_override: Option<&'a HashMap<CategoryId, CategoryMeta>>,
    /// Summarize-time options (filter + grouping).
    pub summarize: SummarizeOptions<'a>,
    /// Boundary-PQ configuration (ADR-0025 Z1 amendment). When
    /// `Some`, the per-image kernel composes
    /// `min(mask_iou, boundary_iou)`; `None` runs instance PQ.
    pub boundary: Option<BoundaryConfig>,
}

/// Like [`evaluate`] but with optional ADR-0042 axes
/// ([`EvaluateOptions`]). The default-options form is bit-identical
/// to `evaluate(...)`.
pub fn evaluate_with_options(
    gt: &PanopticDataset,
    dt: &PanopticPredictions,
    mode: ParityMode,
    things_stuff_split: bool,
    options: &EvaluateOptions<'_>,
) -> Result<PanopticSummary, PanopticError> {
    let per_image = evaluate_per_image(gt, dt, mode, options)?;
    let categories = options.categories_override.unwrap_or(&gt.categories);
    let acc = fold_per_image(&per_image, None);
    summarize_from_acc_with_options(
        acc,
        categories,
        mode,
        things_stuff_split,
        &options.summarize,
    )
}

/// Per-image accumulator delta produced by [`evaluate_per_image`]:
/// the per-category [`PqStat`] map attributed to one image. The
/// partition orchestrator stores a sorted `Vec<PerImageAccum>` once
/// and folds it under different image-id filters per slice
/// (ADR-0046 C3).
pub type PerImageAccum = (ImageId, HashMap<CategoryId, PqStat>);

/// Run the panoptic per-image matching + attribution pass and return
/// the per-image `(image_id, per_category PqStat)` deltas in image-id
/// order. The C3 partitioned path (ADR-0046) consumes this exactly
/// once and then folds + summarizes per slice.
///
/// Storage is `Vec<PerImageAccum>` — a small per-image map, not a
/// per-class tensor. At LVIS-class scale (1203 categories x thousands
/// of images) the total memory stays in the "small map per image"
/// regime because each image only touches the categories actually
/// present in its GT or DT.
///
/// Determinism: images are walked in image-id order, matching the
/// non-partitioned [`evaluate_with_options`]; the slot order in the
/// returned vec is therefore canonical and a downstream filter
/// preserves it.
pub fn evaluate_per_image(
    gt: &PanopticDataset,
    dt: &PanopticPredictions,
    mode: ParityMode,
    options: &EvaluateOptions<'_>,
) -> Result<Vec<PerImageAccum>, PanopticError> {
    // Sort references in image-id order so the f64 summation across
    // images is deterministic (matches panopticapi's annotation-list
    // iteration which is JSON order). The downstream summation is
    // non-associative, so non-determinism would leak into the
    // strict-mode parity claim.
    let mut sorted_gt: Vec<(&ImageId, &ImageEntry)> = gt.images.iter().collect();
    sorted_gt.sort_unstable_by_key(|(id, _)| *id);
    let threshold = options.pq_iou_threshold.unwrap_or(PANOPTIC_IOU_THRESHOLD);
    let categories = options.categories_override.unwrap_or(&gt.categories);
    let max_category_id: u32 = categories
        .keys()
        .copied()
        .filter_map(|id| u32::try_from(id).ok())
        .max()
        .unwrap_or(0);
    let mut gt_scratch = BoundaryScratch::new();
    let mut dt_scratch = BoundaryScratch::new();
    let mut out: Vec<(ImageId, HashMap<CategoryId, PqStat>)> = Vec::with_capacity(sorted_gt.len());
    for (image_id, gt_entry) in sorted_gt {
        let dt_entry =
            dt.images
                .get(image_id)
                .ok_or(PanopticError::MissingPredictionsForImage {
                    image_id: *image_id,
                })?;
        let report = match options.boundary {
            None => pq_image_at_threshold(*image_id, gt_entry, dt_entry, threshold)?,
            Some(cfg) => pq_image_at_threshold_with_boundary(
                *image_id,
                gt_entry,
                dt_entry,
                threshold,
                cfg,
                max_category_id,
                &mut gt_scratch,
                &mut dt_scratch,
            )?,
        };
        let per_image = attribute_image(gt_entry, dt_entry, &report, mode);
        out.push((*image_id, per_image));
    }
    Ok(out)
}

/// Fold the per-image accumulator deltas produced by
/// [`evaluate_per_image`] into a single per-category map, optionally
/// restricted to an image-id filter set (the ADR-0046 C3 mechanism).
///
/// Passing `image_filter = None` reproduces the un-partitioned fold;
/// passing `Some(&set)` aggregates only the deltas whose `image_id`
/// is in `set` — the load-bearing partition operation. The walk
/// preserves image-id order, so the f64 sums match the canonical
/// (non-partitioned) summation under that filter.
pub fn fold_per_image(
    per_image: &[PerImageAccum],
    image_filter: Option<&HashSet<ImageId>>,
) -> HashMap<CategoryId, PqStat> {
    let mut acc: HashMap<CategoryId, PqStat> = HashMap::new();
    for (image_id, deltas) in per_image {
        if let Some(set) = image_filter {
            if !set.contains(image_id) {
                continue;
            }
        }
        for (cat, stat) in deltas {
            acc.entry(*cat).or_default().add_assign(stat);
        }
    }
    acc
}

/// Build a [`PanopticSummary`] from an already-aggregated per-category
/// [`PqStat`] map. The streaming and distributed-merge paths land
/// here after their own per-image fold (see
/// [`crate::stream::StreamingPanopticEvaluator`] / [`crate::distributed`]).
///
/// `categories` supplies the things/stuff partition for the W4 split.
/// `mode` controls W6 (strict raises on empty filter; corrected
/// returns zeros). `things_stuff_split=false` returns `None` for the
/// per-bucket fields.
pub fn summarize_from_acc(
    acc: HashMap<CategoryId, PqStat>,
    categories: &HashMap<CategoryId, CategoryMeta>,
    mode: ParityMode,
    things_stuff_split: bool,
) -> Result<PanopticSummary, PanopticError> {
    summarize_from_acc_with_options(
        acc,
        categories,
        mode,
        things_stuff_split,
        &SummarizeOptions::default(),
    )
}

/// Like [`summarize_from_acc`] but with optional [`SummarizeOptions`]
/// for the ADR-0042 `category_filter` / `class_grouping` axes.
///
/// `category_filter`, when set, restricts the global PQ/SQ/RQ averages
/// (and the things / stuff buckets when `things_stuff_split=true`) to
/// the supplied category subset. The per-class breakdown remains
/// complete. `class_groups` populates
/// [`PanopticSummary::per_group`] with per-partition rollups.
///
/// Both axes are post-summarize aggregations: the per-category
/// accumulator is class-keyed, so distributed-eval ranks accumulate
/// identically regardless of filter / grouping choice.
pub fn summarize_from_acc_with_options(
    acc: HashMap<CategoryId, PqStat>,
    categories: &HashMap<CategoryId, CategoryMeta>,
    mode: ParityMode,
    things_stuff_split: bool,
    options: &SummarizeOptions<'_>,
) -> Result<PanopticSummary, PanopticError> {
    // Per-class summary, sorted by category id (BTreeMap).
    let per_class: BTreeMap<CategoryId, ClassPanopticStats> = acc
        .into_iter()
        .map(|(cat, stat)| (cat, class_stats(stat)))
        .collect();

    // The user-facing CategoryId space is u32-keyed (ADR-0042); the
    // kernel's CategoryId is an i64 alias. Filter / grouping conversion
    // happens once here so the inner loops stay typed.
    let filter_set: Option<HashSet<CategoryId>> = options
        .category_filter
        .map(|f| f.iter().map(|c| CategoryId::from(*c)).collect());

    // Global means — restricted to category_filter if provided.
    let (pq, sq, rq, n) = match &filter_set {
        None => finalize_average(average(per_class.values().copied()), mode, "all")?,
        Some(filter_ids) => finalize_average(
            average(
                per_class
                    .iter()
                    .filter_map(|(cat, s)| filter_ids.contains(cat).then_some(*s)),
            ),
            mode,
            "all",
        )?,
    };

    // Things/stuff (W4) — applied within the filter scope when one is set.
    let (pq_things, sq_things, rq_things, n_things, pq_stuff, sq_stuff, rq_stuff, n_stuff) =
        if things_stuff_split {
            let in_filter = |cat: &CategoryId| filter_set.as_ref().is_none_or(|f| f.contains(cat));
            let things = average(per_class.iter().filter_map(|(cat, s)| {
                categories
                    .get(cat)
                    .filter(|m| m.isthing && in_filter(cat))
                    .map(|_| *s)
            }));
            let stuff = average(per_class.iter().filter_map(|(cat, s)| {
                categories
                    .get(cat)
                    .filter(|m| !m.isthing && in_filter(cat))
                    .map(|_| *s)
            }));
            // Things/stuff buckets are independent; an empty things
            // bucket on a stuff-only dataset is a real downstream
            // surface and should not poison the all-bucket result.
            // Strict mode still raises (matches panopticapi's
            // zero-division).
            let (pq_t, sq_t, rq_t, n_t) = finalize_average(things, mode, "things")?;
            let (pq_s, sq_s, rq_s, n_s) = finalize_average(stuff, mode, "stuff")?;
            (
                Some(pq_t),
                Some(sq_t),
                Some(rq_t),
                Some(n_t),
                Some(pq_s),
                Some(sq_s),
                Some(rq_s),
                Some(n_s),
            )
        } else {
            (None, None, None, None, None, None, None, None)
        };

    let per_group = options
        .class_groups
        .map(|groups| build_per_group(groups, &per_class))
        .unwrap_or_default();

    Ok(PanopticSummary {
        pq,
        sq,
        rq,
        pq_things,
        sq_things,
        rq_things,
        pq_stuff,
        sq_stuff,
        rq_stuff,
        per_class,
        n,
        n_things,
        n_stuff,
        per_group,
    })
}

/// Roll up per-group panoptic stats. Mirrors the `average()` reduction
/// applied to a category subset; categories absent from `per_class`
/// are skipped (no contribution; not an error).
fn build_per_group(
    groups: &[(String, Vec<u32>)],
    per_class: &BTreeMap<CategoryId, ClassPanopticStats>,
) -> BTreeMap<String, GroupPanopticStats> {
    let mut out = BTreeMap::new();
    for (label, ids) in groups {
        let id_set: HashSet<CategoryId> = ids.iter().map(|c| CategoryId::from(*c)).collect();
        let (pq, sq, rq, n) = average(
            per_class
                .iter()
                .filter_map(|(cat, s)| id_set.contains(cat).then_some(*s)),
        );
        let mut members: Vec<CategoryId> = id_set.into_iter().collect();
        members.sort_unstable();
        out.insert(
            label.clone(),
            GroupPanopticStats {
                label: label.clone(),
                member_category_ids: members,
                pq,
                sq,
                rq,
                n,
            },
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{CategoryMeta, ImageEntry, SegmentInfo};
    use rustc_hash::FxHashMap;
    use std::collections::HashMap;

    fn entry(
        height: u32,
        width: u32,
        label_map: Vec<u32>,
        segments: &[(u32, CategoryId, bool, u64)],
    ) -> ImageEntry {
        let mut map = FxHashMap::default();
        for &(id, category_id, iscrowd, area) in segments {
            map.insert(
                id,
                SegmentInfo {
                    id,
                    category_id,
                    iscrowd,
                    area,
                },
            );
        }
        ImageEntry {
            height,
            width,
            label_map,
            segments: map,
        }
    }

    #[test]
    fn class_stats_w1_direct_form() {
        // 3 TP with sum_iou=2.4, 1 FP, 1 FN.
        // denom = 3 + 0.5*1 + 0.5*1 = 4
        // PQ = 2.4 / 4 = 0.6 (exact in f64)
        // SQ = 2.4 / 3 (one ulp below 0.8 in f64; W1 demands the
        //              direct form, so we don't post-round)
        // RQ = 3 / 4 = 0.75 (exact)
        let stat = PqStat {
            sum_iou: 2.4,
            n_tp: 3,
            n_fp: 1,
            n_fn: 1,
        };
        let row = class_stats(stat);
        assert_eq!(row.pq, 0.6);
        assert_eq!(row.sq, 2.4 / 3.0);
        assert_eq!(row.rq, 0.75);
        assert_eq!(row.n_tp, 3);
        assert_eq!(row.n_fp, 1);
        assert_eq!(row.n_fn, 1);
        assert_eq!(row.iou_sum, 2.4);
    }

    #[test]
    fn class_stats_zero_tp_returns_zero_sq() {
        // No TP, just FN. denom = 0.5, but SQ has its own zero guard.
        let stat = PqStat {
            sum_iou: 0.0,
            n_tp: 0,
            n_fp: 0,
            n_fn: 1,
        };
        let row = class_stats(stat);
        assert_eq!(row.pq, 0.0);
        assert_eq!(row.sq, 0.0);
        assert_eq!(row.rq, 0.0);
    }

    #[test]
    fn average_excludes_zero_rows_w2() {
        // Three categories: two real, one all-zero. Only the two
        // real rows contribute to the mean.
        let real_a = ClassPanopticStats {
            pq: 0.6,
            sq: 0.8,
            rq: 0.75,
            n_tp: 3,
            n_fp: 1,
            n_fn: 1,
            iou_sum: 2.4,
        };
        let real_b = ClassPanopticStats {
            pq: 0.4,
            sq: 0.5,
            rq: 0.8,
            n_tp: 2,
            n_fp: 1,
            n_fn: 0,
            iou_sum: 1.0,
        };
        let zero = ClassPanopticStats::default();
        let (pq, sq, rq, n) = average([real_a, real_b, zero].into_iter());
        assert_eq!(n, 2);
        assert_eq!(pq, 0.5);
        assert_eq!(sq, 0.65);
        assert_eq!(rq, 0.775);
    }

    #[test]
    fn aggregate_w7_mean_not_pooled() {
        // W7 regression: global SQ is the unweighted *mean* of
        // per-category SQs, not the pooled `total_iou / total_TP`.
        // The two diverge on long-tailed data; this test pins the
        // divergence so a future "innocent" refactor that swaps to
        // pooling fails CI.
        //
        // Direct PqStat construction (no kernel pipeline) keeps this
        // test focused on the aggregation arithmetic:
        //   cat 100: sum_iou = 1.6, n_tp = 2  -> SQ_100 = 0.8
        //   cat 200: sum_iou = 0.6, n_tp = 1  -> SQ_200 = 0.6
        // mean(SQ_c) = (0.8 + 0.6) / 2 = 0.7
        // pooled total_iou / total_TP = 2.2 / 3 ≈ 0.7333
        let row_100 = class_stats(PqStat {
            sum_iou: 1.6,
            n_tp: 2,
            n_fp: 0,
            n_fn: 0,
        });
        let row_200 = class_stats(PqStat {
            sum_iou: 0.6,
            n_tp: 1,
            n_fp: 0,
            n_fn: 0,
        });
        let (_pq, sq, _rq, n) = average([row_100, row_200].into_iter());
        assert_eq!(n, 2);
        assert!((sq - 0.7).abs() < 1e-12);
        // Confirm divergence from pooled: |mean - pooled| > 1%.
        let pooled = 2.2_f64 / 3.0;
        assert!((sq - pooled).abs() > 0.01);
    }

    #[test]
    fn evaluate_pipeline_perfect_match_with_things_stuff_split() {
        // 1x10 image, two GT segments perfectly covered by two DT
        // segments. cat 100 is thing, cat 200 is stuff. Both pairs
        // produce iou=1.0 → SQ=1.0, RQ=1.0, PQ=1.0 per class.
        let gt = entry(
            1,
            10,
            vec![1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            &[(1, 100, false, 5), (2, 200, false, 5)],
        );
        let dt = entry(
            1,
            10,
            vec![10, 10, 10, 10, 10, 11, 11, 11, 11, 11],
            &[(10, 100, false, 5), (11, 200, false, 5)],
        );

        let mut gt_images = HashMap::new();
        gt_images.insert(1i64, gt);
        let mut categories = HashMap::new();
        categories.insert(
            100,
            CategoryMeta {
                id: 100,
                isthing: true,
            },
        );
        categories.insert(
            200,
            CategoryMeta {
                id: 200,
                isthing: false,
            },
        );
        let gt_dataset = PanopticDataset::from_components(gt_images, categories);

        let mut dt_images = HashMap::new();
        dt_images.insert(1i64, dt);
        let dt_predictions = PanopticPredictions::from_components(dt_images);

        let summary = evaluate(&gt_dataset, &dt_predictions, ParityMode::Corrected, true).unwrap();

        // Per-class: both are 1.0 across the board.
        assert_eq!(summary.per_class[&100].pq, 1.0);
        assert_eq!(summary.per_class[&100].sq, 1.0);
        assert_eq!(summary.per_class[&100].rq, 1.0);
        assert_eq!(summary.per_class[&200].pq, 1.0);
        assert_eq!(summary.per_class[&200].sq, 1.0);
        assert_eq!(summary.per_class[&200].rq, 1.0);

        // Global + bucket means: all 1.0.
        assert_eq!(summary.pq, 1.0);
        assert_eq!(summary.sq, 1.0);
        assert_eq!(summary.rq, 1.0);
        assert_eq!(summary.pq_things, Some(1.0));
        assert_eq!(summary.pq_stuff, Some(1.0));
        assert_eq!(summary.n, 2);
        assert_eq!(summary.n_things, Some(1));
        assert_eq!(summary.n_stuff, Some(1));
    }

    #[test]
    fn evaluate_pipeline_no_things_in_split() {
        // Stuff-only dataset; things bucket should still be Some
        // (returns 0.0, n_things=0) under Corrected.
        let gt = entry(1, 4, vec![1, 1, 1, 1], &[(1, 200, false, 4)]);
        let dt = entry(1, 4, vec![10, 10, 10, 10], &[(10, 200, false, 4)]);
        let mut gt_images = HashMap::new();
        gt_images.insert(1i64, gt);
        let mut categories = HashMap::new();
        categories.insert(
            200,
            CategoryMeta {
                id: 200,
                isthing: false,
            },
        );
        let gt_dataset = PanopticDataset::from_components(gt_images, categories);

        let mut dt_images = HashMap::new();
        dt_images.insert(1i64, dt);
        let dt_predictions = PanopticPredictions::from_components(dt_images);

        let summary = evaluate(&gt_dataset, &dt_predictions, ParityMode::Corrected, true).unwrap();
        // Things bucket is Some(0.0), n_things=0 (W6 corrected).
        assert_eq!(summary.pq_things, Some(0.0));
        assert_eq!(summary.n_things, Some(0));
        // Stuff bucket has the cat 200 row.
        assert_eq!(summary.pq_stuff, Some(1.0));
        assert_eq!(summary.n_stuff, Some(1));
    }

    #[test]
    fn empty_dataset_corrected_returns_zeros() {
        let gt = PanopticDataset::from_components(HashMap::new(), HashMap::new());
        let dt = PanopticPredictions::from_components(HashMap::new());
        let summary = evaluate(&gt, &dt, ParityMode::Corrected, false).unwrap();
        assert_eq!(summary.pq, 0.0);
        assert_eq!(summary.sq, 0.0);
        assert_eq!(summary.rq, 0.0);
        assert_eq!(summary.n, 0);
    }

    #[test]
    fn empty_dataset_strict_raises_w6() {
        let gt = PanopticDataset::from_components(HashMap::new(), HashMap::new());
        let dt = PanopticPredictions::from_components(HashMap::new());
        let err = evaluate(&gt, &dt, ParityMode::Strict, false).unwrap_err();
        match err {
            PanopticError::EmptyCategoryFilter { context } => assert_eq!(context, "all"),
            other => panic!("expected EmptyCategoryFilter, got {other:?}"),
        }
    }

    #[test]
    fn missing_pred_image_returns_typed_error() {
        let gt_entry = entry(1, 1, vec![1], &[(1, 100, false, 1)]);
        let mut images = HashMap::new();
        images.insert(7i64, gt_entry);
        let mut categories = HashMap::new();
        categories.insert(
            100,
            CategoryMeta {
                id: 100,
                isthing: true,
            },
        );
        let gt = PanopticDataset::from_components(images, categories);
        let dt = PanopticPredictions::from_components(HashMap::new());

        let err = evaluate(&gt, &dt, ParityMode::Corrected, false).unwrap_err();
        match err {
            PanopticError::MissingPredictionsForImage { image_id } => assert_eq!(image_id, 7),
            other => panic!("expected MissingPredictionsForImage, got {other:?}"),
        }
    }

    #[test]
    fn pred_only_image_silently_ignored_y5() {
        // GT has no images; DT has one. Y5: pred-only is silently
        // ignored (no error). Result is the empty-dataset case.
        let gt = PanopticDataset::from_components(HashMap::new(), HashMap::new());
        let dt_entry = entry(1, 1, vec![10], &[(10, 100, false, 1)]);
        let mut dt_images = HashMap::new();
        dt_images.insert(1i64, dt_entry);
        let dt = PanopticPredictions::from_components(dt_images);
        let summary = evaluate(&gt, &dt, ParityMode::Corrected, false).unwrap();
        assert_eq!(summary.n, 0);
    }

    #[test]
    fn evaluate_per_image_then_full_fold_matches_evaluate() {
        // ADR-0046 C3: running `evaluate_per_image` then folding the
        // result (no filter) must produce a summary bit-identical to
        // `evaluate(...)` — this is the load-bearing parity contract
        // that lets the partition orchestrator return the un-partitioned
        // `overall` from the per-image deltas instead of a second
        // matching pass.
        let gt = entry(
            1,
            10,
            vec![1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            &[(1, 100, false, 5), (2, 200, false, 5)],
        );
        let dt = entry(
            1,
            10,
            vec![10, 10, 10, 10, 10, 11, 11, 11, 11, 11],
            &[(10, 100, false, 5), (11, 200, false, 5)],
        );
        let mut gt_images = HashMap::new();
        gt_images.insert(1i64, gt);
        let mut dt_images = HashMap::new();
        dt_images.insert(1i64, dt);
        let mut categories = HashMap::new();
        categories.insert(
            100,
            CategoryMeta {
                id: 100,
                isthing: true,
            },
        );
        categories.insert(
            200,
            CategoryMeta {
                id: 200,
                isthing: false,
            },
        );
        let gt_dataset = PanopticDataset::from_components(gt_images, categories.clone());
        let dt_predictions = PanopticPredictions::from_components(dt_images);

        let direct = evaluate(&gt_dataset, &dt_predictions, ParityMode::Corrected, true).unwrap();

        let opts = EvaluateOptions::default();
        let per_image =
            evaluate_per_image(&gt_dataset, &dt_predictions, ParityMode::Corrected, &opts).unwrap();
        let acc = fold_per_image(&per_image, None);
        let from_deltas =
            summarize_from_acc(acc, &categories, ParityMode::Corrected, true).unwrap();

        assert_eq!(direct.pq.to_bits(), from_deltas.pq.to_bits());
        assert_eq!(direct.sq.to_bits(), from_deltas.sq.to_bits());
        assert_eq!(direct.rq.to_bits(), from_deltas.rq.to_bits());
        assert_eq!(direct.pq_things, from_deltas.pq_things);
        assert_eq!(direct.pq_stuff, from_deltas.pq_stuff);
    }

    #[test]
    fn fold_per_image_with_image_filter_restricts_aggregate() {
        // Two images: image 1 has a perfect TP for cat 100; image 2
        // has only an FN for cat 100. Filtering to {image 1} produces
        // PQ=1; filtering to {image 2} produces PQ=0. Filtering to
        // both produces the canonical merged result. Pins the C3
        // image-id filter semantic.
        let gt1 = entry(1, 4, vec![1, 1, 1, 1], &[(1, 100, false, 4)]);
        let dt1 = entry(1, 4, vec![10, 10, 10, 10], &[(10, 100, false, 4)]);
        let gt2 = entry(1, 4, vec![2, 2, 2, 2], &[(2, 100, false, 4)]);
        let dt2 = entry(1, 4, vec![0, 0, 0, 0], &[]);

        let mut gt_images = HashMap::new();
        gt_images.insert(1i64, gt1);
        gt_images.insert(2i64, gt2);
        let mut dt_images = HashMap::new();
        dt_images.insert(1i64, dt1);
        dt_images.insert(2i64, dt2);
        let mut categories = HashMap::new();
        categories.insert(
            100,
            CategoryMeta {
                id: 100,
                isthing: true,
            },
        );
        let gt_dataset = PanopticDataset::from_components(gt_images, categories.clone());
        let dt_predictions = PanopticPredictions::from_components(dt_images);

        let opts = EvaluateOptions::default();
        let per_image =
            evaluate_per_image(&gt_dataset, &dt_predictions, ParityMode::Corrected, &opts).unwrap();

        let only_one: HashSet<ImageId> = HashSet::from([1]);
        let acc_one = fold_per_image(&per_image, Some(&only_one));
        let s_one = summarize_from_acc(acc_one, &categories, ParityMode::Corrected, false).unwrap();
        assert_eq!(s_one.pq, 1.0);

        let only_two: HashSet<ImageId> = HashSet::from([2]);
        let acc_two = fold_per_image(&per_image, Some(&only_two));
        let s_two = summarize_from_acc(acc_two, &categories, ParityMode::Corrected, false).unwrap();
        assert_eq!(s_two.pq, 0.0);

        // Both: matches the un-filtered call.
        let both: HashSet<ImageId> = HashSet::from([1, 2]);
        let acc_both = fold_per_image(&per_image, Some(&both));
        let s_both =
            summarize_from_acc(acc_both, &categories, ParityMode::Corrected, false).unwrap();
        let s_full = evaluate(&gt_dataset, &dt_predictions, ParityMode::Corrected, false).unwrap();
        assert_eq!(s_both.pq.to_bits(), s_full.pq.to_bits());
    }
}
