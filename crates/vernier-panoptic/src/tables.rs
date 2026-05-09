//! Per-class panoptic-quality result table.
//!
//! Columnar fold over [`PanopticSummary::per_class`] in `BTreeMap`
//! (category-id) order, ready for the FFI layer to wrap as Arrow
//! columns.

use crate::dataset::CategoryId;
use crate::summarize::PanopticSummary;

/// Per-class panoptic table. One row per category that appears in the
/// summary's `per_class` map, in `CategoryId` order.
///
/// Column shape is pinned by the Arrow schema golden in
/// `tests/python/tables/panoptic/schemas/per_class.json`. Edits are
/// deliberate.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PerClassTable {
    /// COCO category id.
    pub category_id: Vec<i64>,
    /// PQ per category (W1 direct form).
    pub pq: Vec<f64>,
    /// SQ per category. `0.0` when `n_tp == 0`.
    pub sq: Vec<f64>,
    /// RQ per category. `0.0` when the W1 denominator is zero.
    pub rq: Vec<f64>,
    /// TP segment count.
    pub n_tp: Vec<u64>,
    /// FP segment count.
    pub n_fp: Vec<u64>,
    /// FN segment count.
    pub n_fn: Vec<u64>,
    /// Sum of IoU across the TP segments.
    pub iou_sum: Vec<f64>,
}

/// Build a [`PerClassTable`] from a summary.
pub fn build_per_class(summary: &PanopticSummary) -> PerClassTable {
    let n = summary.per_class.len();
    let mut table = PerClassTable {
        category_id: Vec::with_capacity(n),
        pq: Vec::with_capacity(n),
        sq: Vec::with_capacity(n),
        rq: Vec::with_capacity(n),
        n_tp: Vec::with_capacity(n),
        n_fp: Vec::with_capacity(n),
        n_fn: Vec::with_capacity(n),
        iou_sum: Vec::with_capacity(n),
    };
    for (cat, row) in summary.per_class.iter() {
        let cat_id: CategoryId = *cat;
        table.category_id.push(cat_id);
        table.pq.push(row.pq);
        table.sq.push(row.sq);
        table.rq.push(row.rq);
        table.n_tp.push(row.n_tp);
        table.n_fp.push(row.n_fp);
        table.n_fn.push(row.n_fn);
        table.iou_sum.push(row.iou_sum);
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::summarize::ClassPanopticStats;
    use std::collections::BTreeMap;

    fn summary_with_rows(rows: Vec<(CategoryId, ClassPanopticStats)>) -> PanopticSummary {
        let mut per_class = BTreeMap::new();
        for (cat, row) in rows {
            per_class.insert(cat, row);
        }
        PanopticSummary {
            pq: 0.0,
            sq: 0.0,
            rq: 0.0,
            pq_things: None,
            sq_things: None,
            rq_things: None,
            pq_stuff: None,
            sq_stuff: None,
            rq_stuff: None,
            per_class,
            n: 0,
            n_things: None,
            n_stuff: None,
        }
    }

    #[test]
    fn build_emits_one_row_per_category_in_btree_order() {
        let summary = summary_with_rows(vec![
            (
                200,
                ClassPanopticStats {
                    pq: 0.4,
                    sq: 0.5,
                    rq: 0.8,
                    n_tp: 2,
                    n_fp: 1,
                    n_fn: 0,
                    iou_sum: 1.0,
                },
            ),
            (
                100,
                ClassPanopticStats {
                    pq: 0.6,
                    sq: 0.8,
                    rq: 0.75,
                    n_tp: 3,
                    n_fp: 1,
                    n_fn: 1,
                    iou_sum: 2.4,
                },
            ),
        ]);
        let table = build_per_class(&summary);
        // BTreeMap order — 100 before 200.
        assert_eq!(table.category_id, vec![100, 200]);
        assert_eq!(table.pq, vec![0.6, 0.4]);
        assert_eq!(table.sq, vec![0.8, 0.5]);
        assert_eq!(table.rq, vec![0.75, 0.8]);
        assert_eq!(table.n_tp, vec![3, 2]);
        assert_eq!(table.n_fp, vec![1, 1]);
        assert_eq!(table.n_fn, vec![1, 0]);
        assert_eq!(table.iou_sum, vec![2.4, 1.0]);
    }

    #[test]
    fn empty_summary_yields_empty_table() {
        let summary = summary_with_rows(vec![]);
        let table = build_per_class(&summary);
        assert!(table.category_id.is_empty());
        assert!(table.iou_sum.is_empty());
    }
}
