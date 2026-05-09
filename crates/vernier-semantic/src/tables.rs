//! Per-class semantic-segmentation result table.
//!
//! Columnar fold over [`SemanticSummary::per_class`] plus the
//! [`ConfusionMatrix`] diagonal, in mmseg / ADE20K column order
//! (TP / FP / FN pixels per class).

use crate::kernel::ConfusionMatrix;
use crate::summarize::SemanticSummary;

/// Per-class semantic table. One row per class id present in the
/// summary's `per_class` map, in `class_id` order.
///
/// Column shape is pinned by the Arrow schema golden in
/// `tests/python/tables/semantic/schemas/per_class.json`. Edits are
/// deliberate.
///
/// `n_gt_pixels` and `n_dt_pixels` are the row and column sums of the
/// confusion matrix; `tp_pixels` is the diagonal entry; `fp_pixels`
/// and `fn_pixels` are derived as `n_dt - tp` and `n_gt - tp`. All
/// nine columns target the canonical mmseg / ADE20K research-report
/// shape.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PerClassTable {
    /// Class id (post-`label_remap` if a remap was applied), as `i64`
    /// for Arrow-Int64 schema parity with the panoptic and instance
    /// per_class tables.
    pub category_id: Vec<i64>,
    /// Per-class IoU. `0.0` or `NaN` for zero-support classes
    /// depending on parity mode (quirk **AL2**).
    pub iou: Vec<f64>,
    /// Per-class recall: `TP / (TP + FN)`.
    pub accuracy: Vec<f64>,
    /// Per-class precision: `TP / (TP + FP)`.
    pub precision: Vec<f64>,
    /// Number of GT pixels of this class (= `TP + FN`).
    pub n_gt_pixels: Vec<u64>,
    /// Number of DT pixels of this class (= `TP + FP`).
    pub n_dt_pixels: Vec<u64>,
    /// True-positive pixels (confusion matrix diagonal at this class).
    pub tp_pixels: Vec<u64>,
    /// False-positive pixels (`n_dt_pixels - tp_pixels`).
    pub fp_pixels: Vec<u64>,
    /// False-negative pixels (`n_gt_pixels - tp_pixels`).
    pub fn_pixels: Vec<u64>,
}

/// Build a [`PerClassTable`] from a summary and its confusion matrix.
/// Allocates nine `Vec`s of length `summary.per_class.len()`; one u64
/// indexed lookup per class for `tp_pixels`, then derives FP / FN.
pub fn build_per_class(summary: &SemanticSummary, confusion: &ConfusionMatrix) -> PerClassTable {
    let n = summary.per_class.len();
    let mut table = PerClassTable {
        category_id: Vec::with_capacity(n),
        iou: Vec::with_capacity(n),
        accuracy: Vec::with_capacity(n),
        precision: Vec::with_capacity(n),
        n_gt_pixels: Vec::with_capacity(n),
        n_dt_pixels: Vec::with_capacity(n),
        tp_pixels: Vec::with_capacity(n),
        fp_pixels: Vec::with_capacity(n),
        fn_pixels: Vec::with_capacity(n),
    };
    for (class_id, row) in summary.per_class.iter() {
        let cid = *class_id;
        let tp = confusion.get(cid, cid);
        // FP / FN derive from the row-sum / col-sum identities the
        // summary already used to populate n_gt_pixels / n_dt_pixels.
        let fp = row.n_dt_pixels.saturating_sub(tp);
        let fn_ = row.n_gt_pixels.saturating_sub(tp);
        table.category_id.push(cid as i64);
        table.iou.push(row.iou);
        table.accuracy.push(row.accuracy);
        table.precision.push(row.precision);
        table.n_gt_pixels.push(row.n_gt_pixels);
        table.n_dt_pixels.push(row.n_dt_pixels);
        table.tp_pixels.push(tp);
        table.fp_pixels.push(fp);
        table.fn_pixels.push(fn_);
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::ConfusionMatrix;
    use crate::summarize::summarize;
    use vernier_core::parity::ParityMode;

    #[test]
    fn build_emits_one_row_per_class_in_btree_order() {
        // 3 classes, perfect prediction on a 4x1 image where pixels go
        // [0, 1, 1, 2] for both GT and DT.
        // diag = [1, 2, 1], row_sum = [1, 2, 1], col_sum = [1, 2, 1].
        let mut m = ConfusionMatrix::zeros(3);
        let counts = m.counts_mut();
        // counts[g * n + d]; flat indices for the (0,0), (1,1), (2,2)
        // diagonal of a 3-class matrix.
        counts[0] = 1; // class 0 TP
        counts[4] = 2; // class 1 TP
        counts[8] = 1; // class 2 TP
        let summary = summarize(m, ParityMode::Corrected);
        let table = build_per_class(&summary, &summary.confusion_matrix);
        assert_eq!(table.category_id, vec![0, 1, 2]);
        assert_eq!(table.iou, vec![1.0, 1.0, 1.0]);
        assert_eq!(table.accuracy, vec![1.0, 1.0, 1.0]);
        assert_eq!(table.precision, vec![1.0, 1.0, 1.0]);
        assert_eq!(table.n_gt_pixels, vec![1, 2, 1]);
        assert_eq!(table.n_dt_pixels, vec![1, 2, 1]);
        assert_eq!(table.tp_pixels, vec![1, 2, 1]);
        assert_eq!(table.fp_pixels, vec![0, 0, 0]);
        assert_eq!(table.fn_pixels, vec![0, 0, 0]);
    }

    #[test]
    fn build_reports_fp_and_fn_from_off_diagonal() {
        // 2 classes, GT = [0, 0, 1, 1], DT = [0, 1, 1, 1]. Confusion:
        //   counts[0,0] = 1, counts[0,1] = 1, counts[1,1] = 2.
        // diag = [1, 2], row_sum (n_gt) = [2, 2], col_sum (n_dt) = [1, 3].
        // FP_0 = 0, FP_1 = 1; FN_0 = 1, FN_1 = 0.
        let mut m = ConfusionMatrix::zeros(2);
        let counts = m.counts_mut();
        // (g=0, d=0), (g=0, d=1), (g=1, d=1) on a 2-class matrix.
        counts[0] = 1;
        counts[1] = 1;
        counts[3] = 2;
        let summary = summarize(m, ParityMode::Corrected);
        let table = build_per_class(&summary, &summary.confusion_matrix);
        assert_eq!(table.tp_pixels, vec![1, 2]);
        assert_eq!(table.fp_pixels, vec![0, 1]);
        assert_eq!(table.fn_pixels, vec![1, 0]);
        assert_eq!(table.n_gt_pixels, vec![2, 2]);
        assert_eq!(table.n_dt_pixels, vec![1, 3]);
    }

    #[test]
    fn empty_summary_yields_empty_table() {
        // Zero-class confusion is rejected upstream by the kernel; use
        // the smallest non-trivial one with all zeros.
        let m = ConfusionMatrix::zeros(2);
        let summary = summarize(m, ParityMode::Corrected);
        let table = build_per_class(&summary, &summary.confusion_matrix);
        // Every class is present with zero support; FP / FN are zero.
        assert_eq!(table.category_id, vec![0, 1]);
        assert_eq!(table.tp_pixels, vec![0, 0]);
        assert_eq!(table.fp_pixels, vec![0, 0]);
        assert_eq!(table.fn_pixels, vec![0, 0]);
    }
}
