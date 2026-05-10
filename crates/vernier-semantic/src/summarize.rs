//! Per-class aggregation and the [`SemanticSummary`] public type.
//!
//! Folds the global [`ConfusionMatrix`] (built by repeated
//! [`crate::kernel::accumulate_confusion`] calls into a single shared
//! matrix) into the seven headline outputs ADR-0028 commits to:
//!
//! - **mIoU** — unweighted mean of per-class IoU (quirk **AL3**).
//! - **FWIoU** — frequency-weighted IoU (quirk **AL4**).
//! - **pixel accuracy** — `trace / total` (quirk **AL5**).
//! - **mean accuracy** — unweighted mean of per-class recall (quirk
//!   **AL6**).
//! - **per-class IoU** — `TP_c / (TP_c + FP_c + FN_c)` (quirk **AL1**).
//! - **per-class accuracy** — `TP_c / (TP_c + FN_c)` (quirk **AL6**).
//! - the **confusion matrix** itself, as a first-class output (quirk
//!   **AL8**, ADR-0028 §F1).
//!
//! Per-class precision (`TP_c / (TP_c + FP_c)`) is included alongside
//! IoU and accuracy in [`ClassSemanticStats`]; the four-tuple
//! `(IoU, accuracy, precision, support)` is the canonical research
//! report shape (mmsegmentation `IoUMetric`).
//!
//! ## NaN handling, by parity mode (quirk **AL2**)
//!
//! For a class with `TP_c + FP_c + FN_c == 0` (never seen, never
//! predicted), per-class IoU is undefined. Oracles disagree:
//! mmsegmentation returns `nan`; cityscapesScripts returns `0.0`; the
//! ADE20K reference returns `nan`.
//!
//! - [`ParityMode::Strict`]: per-class IoU is `f64::NAN`. Matches
//!   mmsegmentation's `np.nanmean` semantics.
//! - [`ParityMode::Corrected`]: per-class IoU is `0.0` for
//!   zero-support classes. Matches cityscapesScripts' shape and
//!   avoids surprising downstream consumers that don't handle NaN.
//!
//! Either way, the **mean** (mIoU and mAcc) excludes zero-support
//! classes from the average per quirk **AL3** — the same convention
//! as panopticapi (W2) and LVIS (AB3). A class never seen
//! contributes neither to the mean nor to its denominator.
//!
//! [`ParityMode::Strict`]: vernier_core::parity::ParityMode::Strict
//! [`ParityMode::Corrected`]: vernier_core::parity::ParityMode::Corrected

use std::collections::BTreeMap;

use vernier_core::parity::ParityMode;

use crate::kernel::ConfusionMatrix;

/// Per-class semantic-segmentation row.
///
/// `class_id` is the original class id (post-`label_remap` if a remap
/// was applied), preserved so callers can correlate against the
/// dataset's category list. The four metric fields share the same
/// NaN-vs-0.0 disposition rule as the per-class IoU on
/// [`SemanticSummary`]: under [`ParityMode::Strict`] they are
/// `f64::NAN` for zero-support classes; under
/// [`ParityMode::Corrected`] they are `0.0`.
///
/// [`ParityMode::Strict`]: vernier_core::parity::ParityMode::Strict
/// [`ParityMode::Corrected`]: vernier_core::parity::ParityMode::Corrected
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClassSemanticStats {
    /// Class id this row corresponds to.
    pub class_id: u32,
    /// Intersection-over-union: `TP / (TP + FP + FN)`. Quirk **AL1**.
    pub iou: f64,
    /// Per-class recall: `TP / (TP + FN)`. Quirk **AL6**.
    pub accuracy: f64,
    /// Per-class precision: `TP / (TP + FP)`. Reported alongside
    /// recall to round out the per-class research report shape.
    pub precision: f64,
    /// Number of pixels labeled this class in GT (= `TP + FN`).
    pub n_gt_pixels: u64,
    /// Number of pixels predicted this class in DT (= `TP + FP`).
    pub n_dt_pixels: u64,
}

/// Per-group semantic-segmentation rollup (ADR-0041).
///
/// Built when [`SummarizeOptions::class_groups`] is set: each group
/// fold reduces over its member class ids only. The rollup mirrors
/// the headline scalars in shape so per-group output is comparable
/// across groups.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupSemanticStats {
    /// Group label, as passed in by the caller (typically a taxonomy
    /// name like `"vehicles"` / `"surfaces"`).
    pub label: String,
    /// Class ids that compose this group, sorted ascending.
    pub member_class_ids: Vec<u32>,
    /// Mean IoU restricted to this group's classes (skipping
    /// zero-support members per quirk **AL3**).
    pub miou: f64,
    /// Mean per-class recall restricted to this group's classes
    /// (zero-support skipped, same as the headline `mean_accuracy`).
    pub mean_accuracy: f64,
    /// Pixel accuracy over this group's classes only —
    /// `sum_c TP_c / sum_c (TP_c + FN_c)` for `c` in members.
    pub pixel_accuracy: f64,
    /// Frequency-weighted IoU over the group's classes.
    pub fwiou: f64,
}

/// Optional axes for [`summarize_with_options`] (ADR-0041).
///
/// Both fields default to `None`; the unparametrized
/// [`summarize`] is the `Default` shape and produces the headline
/// scalars over every class in the confusion matrix.
#[derive(Debug, Clone, Default)]
pub struct SummarizeOptions<'a> {
    /// Subset of class ids that contribute to the headline scalars
    /// (`miou` / `fwiou` / `mean_accuracy` / `pixel_accuracy`). The
    /// per-class breakdown remains complete regardless of this
    /// filter; only the reductions are restricted.
    ///
    /// `None` means "every class contributes" (the COCO default).
    pub class_filter: Option<&'a [u32]>,
    /// Class-id partitions to roll up into per-group stats. Each
    /// `(label, class_ids)` pair contributes one
    /// [`GroupSemanticStats`] entry on
    /// [`SemanticSummary::per_group`].
    ///
    /// `None` leaves `per_group` empty.
    pub class_groups: Option<&'a [(String, Vec<u32>)]>,
}

/// Top-level semantic-evaluation result.
///
/// Sibling to [`vernier_core::summary::Summary`] (instance) and
/// [`vernier_panoptic::summarize::PanopticSummary`] (panoptic) per
/// ADR-0028 §"Public Python surface". Carries the four headline
/// scalars on top, the per-class breakdown, and the confusion matrix
/// itself as a first-class output (downstream tools — calibration,
/// error decomposition, model-diff — consume the matrix directly).
#[derive(Debug, Clone, PartialEq)]
pub struct SemanticSummary {
    /// Mean intersection-over-union, ignore-aware. Unweighted mean of
    /// per-class IoU over classes with non-zero support
    /// (`TP + FP + FN > 0`). Quirk **AL3**.
    pub miou: f64,
    /// Frequency-weighted IoU. `sum_c (freq_c * IoU_c)` where
    /// `freq_c = (TP_c + FN_c) / total_evaluated_pixels`. Weighting
    /// reflects class prevalence in GT. Quirk **AL4**.
    pub fwiou: f64,
    /// Pixel accuracy. `trace(confusion) / sum(confusion)` — total
    /// correct pixels over total evaluated pixels (ignore label
    /// already excluded). Quirk **AL5**.
    pub pixel_accuracy: f64,
    /// Mean per-class accuracy (recall). Unweighted mean over classes
    /// with non-zero GT support (`TP + FN > 0`). Quirk **AL6**.
    pub mean_accuracy: f64,
    /// Per-class breakdown, keyed by class id. `BTreeMap` for
    /// deterministic iteration order (quirk **AL7**: per-class output
    /// order is the construction-order of the categories list, which
    /// equals natural class-id order on every preset).
    pub per_class: BTreeMap<u32, ClassSemanticStats>,
    /// The accumulated confusion matrix. ADR-0028 §F1 promotes this
    /// to a first-class output: downstream calibration / error
    /// decomposition / model-diff tools consume it directly without
    /// re-running the kernel.
    pub confusion_matrix: ConfusionMatrix,
    /// Per-group rollup (ADR-0041). Populated only when the caller
    /// passes [`SummarizeOptions::class_groups`]; empty for the
    /// canonical no-grouping path. Keyed by group label in
    /// construction order via the `BTreeMap`'s sorted iteration —
    /// stable across runs for any given grouping definition.
    pub per_group: BTreeMap<String, GroupSemanticStats>,
}

/// Derive a [`SemanticSummary`] from a fully-accumulated
/// [`ConfusionMatrix`].
///
/// `parity_mode` selects the NaN-vs-0.0 disposition for zero-support
/// per-class entries (quirk **AL2**). The mean reductions (mIoU,
/// mAcc) skip zero-support classes regardless of mode (quirk
/// **AL3**), so the parity_mode choice is observable on the
/// `per_class` rows but does not change the headline scalars.
///
/// `confusion` is consumed (moved into the result's
/// `confusion_matrix` field). If the caller needs to keep a copy,
/// clone before calling.
pub fn summarize(confusion: ConfusionMatrix, parity_mode: ParityMode) -> SemanticSummary {
    summarize_with_options(confusion, parity_mode, &SummarizeOptions::default())
}

/// Like [`summarize`] but with optional [`SummarizeOptions`] for the
/// ADR-0041 `class_filter` / `class_grouping` axes.
///
/// Filtering restricts the headline scalars (mIoU / FWIoU / pixel
/// accuracy / mean accuracy) to the supplied class subset; the
/// per-class breakdown remains complete. Grouping populates
/// [`SemanticSummary::per_group`] with one entry per `(label, ids)`
/// partition.
///
/// Both axes are post-summarize aggregations: the confusion matrix
/// is class-keyed, so distributed-eval ranks accumulate identically
/// regardless of filter / grouping choice. Ranks may diverge on
/// these without affecting `from_partials_to_summary`.
pub fn summarize_with_options(
    confusion: ConfusionMatrix,
    parity_mode: ParityMode,
    options: &SummarizeOptions<'_>,
) -> SemanticSummary {
    let n_classes = confusion.n_classes();
    let n = n_classes as usize;
    let counts = confusion.counts();

    // Row sums = TP_c + FN_c (number of GT pixels of class c).
    // Column sums = TP_c + FP_c (number of DT pixels of class c).
    // Diagonal  = TP_c.
    let mut row_sum = vec![0u64; n];
    let mut col_sum = vec![0u64; n];
    let mut diag = vec![0u64; n];
    for g in 0..n {
        for d in 0..n {
            let cell = counts[g * n + d];
            row_sum[g] += cell;
            col_sum[d] += cell;
            if g == d {
                diag[g] = cell;
            }
        }
    }

    let total_evaluated: u64 = row_sum.iter().sum();
    let trace: u64 = diag.iter().sum();

    // Pixel accuracy: trace / total, defined as 0.0 on the empty
    // dataset (matches cityscapesScripts' ZeroDivisionError-avoidance
    // shape; vernier surfaces an EmptyDataset error upstream of this
    // function so this branch is mostly defensive).
    let pixel_accuracy = if total_evaluated == 0 {
        0.0
    } else {
        (trace as f64) / (total_evaluated as f64)
    };

    // Per-class rows + accumulators for the means.
    let mut per_class = BTreeMap::new();
    let mut iou_sum = 0.0f64;
    let mut iou_n = 0usize;
    let mut acc_sum = 0.0f64;
    let mut acc_n = 0usize;
    let mut fwiou = 0.0f64;

    for c in 0..n {
        let tp = diag[c];
        let n_gt = row_sum[c];
        let n_dt = col_sum[c];
        let fp = n_dt - tp;
        let fn_ = n_gt - tp;

        let iou_denom = tp + fp + fn_;
        let iou = if iou_denom > 0 {
            (tp as f64) / (iou_denom as f64)
        } else {
            zero_support_value(parity_mode)
        };

        let acc_denom = tp + fn_;
        let acc = if acc_denom > 0 {
            (tp as f64) / (acc_denom as f64)
        } else {
            zero_support_value(parity_mode)
        };

        // Precision is reported for completeness; same NaN/0.0 rule.
        let prec_denom = tp + fp;
        let precision = if prec_denom > 0 {
            (tp as f64) / (prec_denom as f64)
        } else {
            zero_support_value(parity_mode)
        };

        if iou_denom > 0 {
            iou_sum += iou;
            iou_n += 1;
        }
        if acc_denom > 0 {
            acc_sum += acc;
            acc_n += 1;
        }
        // FWIoU weights by GT frequency. Classes with no GT
        // contribute nothing (freq_c = 0); classes with no DT but
        // non-zero GT still contribute (IoU_c = 0 because TP=0).
        if total_evaluated > 0 && n_gt > 0 {
            let freq = (n_gt as f64) / (total_evaluated as f64);
            // Use the IoU value already computed (which is 0.0 when
            // TP=0 and FP+FN > 0 — well-defined regardless of mode).
            fwiou += freq * iou;
        }

        per_class.insert(
            c as u32,
            ClassSemanticStats {
                class_id: c as u32,
                iou,
                accuracy: acc,
                precision,
                n_gt_pixels: n_gt,
                n_dt_pixels: n_dt,
            },
        );
    }

    // mIoU / mAcc skip zero-support classes regardless of mode (AL3).
    // Empty-dataset case (no class with support) → 0.0 to avoid NaN
    // in downstream means; vernier rejects empty datasets upstream so
    // this branch is defensive.
    let (miou, mean_accuracy, fwiou, pixel_accuracy) = match options.class_filter {
        None => {
            let mu = if iou_n > 0 {
                iou_sum / iou_n as f64
            } else {
                0.0
            };
            let ma = if acc_n > 0 {
                acc_sum / acc_n as f64
            } else {
                0.0
            };
            (mu, ma, fwiou, pixel_accuracy)
        }
        Some(filter) => filtered_headlines(filter, n, &diag, &row_sum, &col_sum),
    };

    let per_group = options
        .class_groups
        .map(|groups| build_per_group(groups, n, &diag, &row_sum, &col_sum))
        .unwrap_or_default();

    SemanticSummary {
        miou,
        fwiou,
        pixel_accuracy,
        mean_accuracy,
        per_class,
        confusion_matrix: confusion,
        per_group,
    }
}

/// Compute the four headline scalars over a subset of class ids.
/// Out-of-range or duplicate ids in `filter` are skipped silently
/// (the FFI layer validates against `n_classes` before this call).
fn filtered_headlines(
    filter: &[u32],
    n: usize,
    diag: &[u64],
    row_sum: &[u64],
    col_sum: &[u64],
) -> (f64, f64, f64, f64) {
    let mut iou_sum = 0.0f64;
    let mut iou_n = 0usize;
    let mut acc_sum = 0.0f64;
    let mut acc_n = 0usize;
    let mut tp_total: u64 = 0;
    let mut row_total: u64 = 0;
    for &cid in filter {
        let c = cid as usize;
        if c >= n {
            continue;
        }
        let tp = diag[c];
        let n_gt = row_sum[c];
        let n_dt = col_sum[c];
        let fp = n_dt - tp;
        let fn_ = n_gt - tp;
        let iou_denom = tp + fp + fn_;
        if iou_denom > 0 {
            iou_sum += (tp as f64) / (iou_denom as f64);
            iou_n += 1;
        }
        let acc_denom = tp + fn_;
        if acc_denom > 0 {
            acc_sum += (tp as f64) / (acc_denom as f64);
            acc_n += 1;
        }
        tp_total += tp;
        row_total += n_gt;
    }
    let miou = if iou_n > 0 {
        iou_sum / iou_n as f64
    } else {
        0.0
    };
    let mean_accuracy = if acc_n > 0 {
        acc_sum / acc_n as f64
    } else {
        0.0
    };
    let pixel_accuracy = if row_total > 0 {
        (tp_total as f64) / (row_total as f64)
    } else {
        0.0
    };
    // FWIoU restricted to the filter subset uses each class's
    // freq within the filter (n_gt_c / row_total) so the weights
    // sum to 1 and the result is interpretable as "FWIoU within
    // the filter scope" rather than "FWIoU on a fragment of the
    // full dataset".
    let mut fwiou = 0.0f64;
    if row_total > 0 {
        for &cid in filter {
            let c = cid as usize;
            if c >= n {
                continue;
            }
            let tp = diag[c];
            let n_gt = row_sum[c];
            let n_dt = col_sum[c];
            let fp = n_dt - tp;
            let fn_ = n_gt - tp;
            let iou_denom = tp + fp + fn_;
            if n_gt > 0 && iou_denom > 0 {
                let iou = (tp as f64) / (iou_denom as f64);
                fwiou += (n_gt as f64 / row_total as f64) * iou;
            }
        }
    }
    (miou, mean_accuracy, fwiou, pixel_accuracy)
}

/// Roll up per-group stats over the supplied class-id partitions.
fn build_per_group(
    groups: &[(String, Vec<u32>)],
    n: usize,
    diag: &[u64],
    row_sum: &[u64],
    col_sum: &[u64],
) -> BTreeMap<String, GroupSemanticStats> {
    let mut out = BTreeMap::new();
    for (label, ids) in groups {
        let (miou, mean_accuracy, fwiou, pixel_accuracy) =
            filtered_headlines(ids, n, diag, row_sum, col_sum);
        let mut members: Vec<u32> = ids.iter().copied().filter(|&c| (c as usize) < n).collect();
        members.sort_unstable();
        members.dedup();
        out.insert(
            label.clone(),
            GroupSemanticStats {
                label: label.clone(),
                member_class_ids: members,
                miou,
                mean_accuracy,
                pixel_accuracy,
                fwiou,
            },
        );
    }
    out
}

/// Per-class metric value when `TP + FP + FN == 0` (or the analogous
/// zero-denominator case for accuracy / precision). Quirk **AL2**.
fn zero_support_value(parity_mode: ParityMode) -> f64 {
    match parity_mode {
        ParityMode::Strict => f64::NAN,
        ParityMode::Corrected => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::accumulate_confusion;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn perfect_match_yields_unit_metrics() {
        let gt = vec![0u32, 1, 2, 0, 1, 2];
        let dt = gt.clone();
        let mut cm = ConfusionMatrix::zeros(3);
        accumulate_confusion(&gt, &dt, None, &mut cm);

        let s = summarize(cm, ParityMode::Corrected);
        assert!(approx_eq(s.miou, 1.0, 0.0));
        assert!(approx_eq(s.fwiou, 1.0, 0.0));
        assert!(approx_eq(s.pixel_accuracy, 1.0, 0.0));
        assert!(approx_eq(s.mean_accuracy, 1.0, 0.0));
        for stats in s.per_class.values() {
            assert!(approx_eq(stats.iou, 1.0, 0.0));
            assert!(approx_eq(stats.accuracy, 1.0, 0.0));
            assert!(approx_eq(stats.precision, 1.0, 0.0));
        }
    }

    #[test]
    fn one_off_diagonal_pixel_drives_metric_drop() {
        // 2 classes, 4 pixels: 3 perfect, 1 mistake (gt=0,dt=1).
        // class 0: TP=1, FP=0, FN=1, IoU = 1/(1+0+1) = 0.5
        // class 1: TP=2, FP=1, FN=0, IoU = 2/(2+1+0) = 2/3
        // mIoU = (0.5 + 2/3) / 2 ≈ 0.5833
        let gt = vec![0u32, 0, 1, 1];
        let dt = vec![0u32, 1, 1, 1];
        let mut cm = ConfusionMatrix::zeros(2);
        accumulate_confusion(&gt, &dt, None, &mut cm);

        let s = summarize(cm, ParityMode::Corrected);
        let expected_miou = (0.5 + 2.0 / 3.0) / 2.0;
        assert!(
            approx_eq(s.miou, expected_miou, 1e-12),
            "miou={} expected={expected_miou}",
            s.miou
        );

        // pAcc = 3/4 = 0.75 (3 correct out of 4).
        assert!(approx_eq(s.pixel_accuracy, 0.75, 0.0));

        // mAcc: per-class recall = (1/(1+1)=0.5, 2/(2+0)=1.0); mean = 0.75.
        assert!(approx_eq(s.mean_accuracy, 0.75, 1e-12));

        // FWIoU: freq_0 = 2/4, freq_1 = 2/4; FWIoU = 0.5*0.5 + 0.5*2/3 = 0.5833...
        let expected_fwiou = 0.5 * 0.5 + 0.5 * (2.0 / 3.0);
        assert!(approx_eq(s.fwiou, expected_fwiou, 1e-12));
    }

    #[test]
    fn ignore_label_excluded_from_totals() {
        // gt=[0, 255, 1, 1]; dt=[0, 0, 1, 1]; ignore=255.
        // After mask: gt=[0,1,1], dt=[0,1,1]. Three pixels, all
        // diagonal → mIoU=1.0, pAcc=1.0.
        let gt = vec![0u32, 255, 1, 1];
        let dt = vec![0u32, 0, 1, 1];
        let mut cm = ConfusionMatrix::zeros(2);
        accumulate_confusion(&gt, &dt, Some(255), &mut cm);

        let s = summarize(cm, ParityMode::Corrected);
        assert!(approx_eq(s.miou, 1.0, 0.0));
        assert!(approx_eq(s.pixel_accuracy, 1.0, 0.0));

        // Confusion matrix carries 3 pixels total (the ignore was excluded).
        let total: u64 = s.confusion_matrix.counts().iter().sum();
        assert_eq!(total, 3);
    }

    #[test]
    fn zero_support_class_excluded_from_means_strict_yields_nan() {
        // 3 classes; class 2 is never in GT and never predicted.
        // Class 0 perfect (1 pixel); class 1 perfect (1 pixel).
        // mIoU should be 1.0 (mean over the two supported classes,
        // class 2 skipped by AL3).
        // Per-class IoU for class 2: NaN under Strict, 0.0 under
        // Corrected (AL2).
        let gt = vec![0u32, 1];
        let dt = vec![0u32, 1];
        let mut cm = ConfusionMatrix::zeros(3);
        accumulate_confusion(&gt, &dt, None, &mut cm);

        let s_strict = summarize(cm.clone(), ParityMode::Strict);
        assert!(approx_eq(s_strict.miou, 1.0, 0.0));
        assert!(s_strict.per_class[&2].iou.is_nan());
        assert!(s_strict.per_class[&2].accuracy.is_nan());

        let s_corr = summarize(cm, ParityMode::Corrected);
        assert!(approx_eq(s_corr.miou, 1.0, 0.0));
        assert!(approx_eq(s_corr.per_class[&2].iou, 0.0, 0.0));
        assert!(approx_eq(s_corr.per_class[&2].accuracy, 0.0, 0.0));
    }

    #[test]
    fn class_with_no_dt_but_nonzero_gt_contributes_zero_iou() {
        // class 0: 2 GT pixels, both predicted as 1. TP=0, FP=0, FN=2.
        // class 1: TP=0 too — both predictions land in class 1 row's
        //   off-diagonal; col-sum for class 1 is 2, row-sum is 0;
        //   FN_1 = 0, FP_1 = 2. IoU_1 = 0 / (0 + 2 + 0) = 0.
        // Both classes have IoU = 0 (well-defined, included in mean).
        // mIoU = 0.
        let gt = vec![0u32, 0];
        let dt = vec![1u32, 1];
        let mut cm = ConfusionMatrix::zeros(2);
        accumulate_confusion(&gt, &dt, None, &mut cm);

        let s = summarize(cm, ParityMode::Corrected);
        assert!(approx_eq(s.miou, 0.0, 0.0));
        assert!(approx_eq(s.pixel_accuracy, 0.0, 0.0));
    }

    #[test]
    fn empty_confusion_matrix_returns_zeros_not_nan() {
        // Defensive: vernier surfaces EmptyDataset upstream of
        // summarize, but if it ever leaks through, the headline
        // scalars are 0.0 (not NaN) so downstream means stay sane.
        let cm = ConfusionMatrix::zeros(3);
        let s = summarize(cm, ParityMode::Corrected);
        assert!(approx_eq(s.miou, 0.0, 0.0));
        assert!(approx_eq(s.fwiou, 0.0, 0.0));
        assert!(approx_eq(s.pixel_accuracy, 0.0, 0.0));
        assert!(approx_eq(s.mean_accuracy, 0.0, 0.0));
    }

    #[test]
    fn per_class_iteration_is_class_id_sorted() {
        // BTreeMap insertion order is the class-id sort. Pin it to
        // catch a future swap to HashMap that would silently
        // reintroduce non-determinism (quirk AL7).
        let gt = vec![0u32, 1, 2];
        let dt = vec![0u32, 1, 2];
        let mut cm = ConfusionMatrix::zeros(5);
        accumulate_confusion(&gt, &dt, None, &mut cm);

        let s = summarize(cm, ParityMode::Corrected);
        let ids: Vec<u32> = s.per_class.keys().copied().collect();
        assert_eq!(ids, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn deterministic_across_release_and_debug() {
        // The kernel and summarize are integer/f64 deterministic.
        // Two identical runs produce bit-equal results in the same
        // build mode; cross-mode bit-equality is asserted by the
        // u64-only kernel + f64 deterministic-rounding summarize.
        let gt = vec![0u32, 1, 1, 2, 2, 2];
        let dt = vec![0u32, 1, 0, 2, 1, 2];
        let mut cm1 = ConfusionMatrix::zeros(3);
        let mut cm2 = ConfusionMatrix::zeros(3);
        accumulate_confusion(&gt, &dt, None, &mut cm1);
        accumulate_confusion(&gt, &dt, None, &mut cm2);
        let s1 = summarize(cm1, ParityMode::Strict);
        let s2 = summarize(cm2, ParityMode::Strict);
        assert_eq!(s1.miou.to_bits(), s2.miou.to_bits());
        assert_eq!(s1.fwiou.to_bits(), s2.fwiou.to_bits());
        assert_eq!(s1.pixel_accuracy.to_bits(), s2.pixel_accuracy.to_bits());
        assert_eq!(s1.mean_accuracy.to_bits(), s2.mean_accuracy.to_bits());
    }
}
