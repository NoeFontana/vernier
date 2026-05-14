//! Per-class confidence-threshold search.
//!
//! Given the four parallel per-detection arrays
//! `(dt_score, dt_matched, dt_ignore, dt_iou)` for one class plus the
//! count of non-crowd / non-ignore GTs, `search_tau` sweeps the
//! tau-grid and returns the threshold that minimises LRP, along with
//! the active-set statistics at that threshold (so the caller can
//! decompose the metric without re-evaluating).
//!
//! The algorithm is a literal transcription of the oracle at
//! `tests/python/oracle/lrp/oracle.py::_lrp_per_class` — the Rust
//! implementation's correctness contract per ADR-0043 is that this
//! function agrees with the oracle to `1e-9` per fixture per class.
//!
//! Tie-break: when two grid values produce the same LRP, the **larger
//! tau** wins (per ADR-0043). The deployment reading is "fewer FPs at
//! the operating point"; that is the more conservative choice and
//! matches what a practitioner who saw "either threshold gives the
//! same score" would default to.

/// Per-class active-set statistics at one tau value.
///
/// Pre-multiplied by `1 / (1 - tp_threshold)` is the `loc`-term
/// contribution; the [`super::decompose`] layer turns these into the
/// three additive components and the per-class oLRP.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct TauStats {
    /// Number of active TPs at this tau.
    pub n_tp: u64,
    /// Number of active FPs (active detections that did not match a
    /// non-crowd GT).
    pub n_fp: u64,
    /// Number of non-crowd GTs not matched by an active detection.
    pub n_fn: u64,
    /// Sum of `1 - IoU` over active TPs.
    pub sum_loc: f64,
}

/// Result of the per-class tau search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct TauSearchResult {
    /// Optimal tau on the grid — the index into the input grid slice.
    pub star: usize,
    /// LRP value at the optimal tau.
    pub lrp: f64,
    /// Active-set statistics at the optimal tau (carried so
    /// [`super::decompose`] can derive the three additive components
    /// without re-scanning).
    pub stats: TauStats,
}

/// Sweep `tau_grid` and return the LRP-minimising index plus the
/// active-set statistics at that index.
///
/// Inputs are the four parallel per-detection arrays produced by
/// per-class matching ([`super::decompose::match_per_class`]) plus the
/// count of non-crowd / non-ignore GTs. The four arrays must all have
/// the same length `N`; ignored detections (matched-to-crowd) must
/// have been filtered out by the caller — they never count as TP or
/// FP and the LRP formula does not see them.
///
/// `tp_threshold` is the IoU floor that defined what a TP is during
/// matching; this function only uses it to compute the localization
/// weight `1 / (1 - tp_threshold)`. The actual TP / FP / FN counts
/// are derived from the `dt_matched` flags the caller computed at the
/// IoU floor.
///
/// Tie-break: larger tau wins (per ADR-0043). Equivalent to
/// `len - 1 - argmin(lrp[::-1])`.
///
/// Returns `None` when there are no non-crowd GTs (`n_pos_gt == 0`) —
/// the per-class oLRP is undefined and the caller emits a flagged
/// `NaN` row. This is distinct from the "all-FN" case (positive GTs
/// exist but no TPs at any tau), which is a valid `oLRP = 1.0`
/// outcome that this function returns normally.
///
/// # Panics
///
/// Panics if `dt_score.len() != dt_matched.len() != dt_iou.len()` or
/// if `tau_grid` is empty. Both are caller invariants — the public
/// surface in [`super`] validates them before calling this function.
pub(crate) fn search_tau(
    dt_score: &[f64],
    dt_matched: &[bool],
    dt_iou: &[f64],
    n_pos_gt: u64,
    tp_threshold: f64,
    tau_grid: &[f64],
) -> Option<TauSearchResult> {
    if n_pos_gt == 0 {
        return None;
    }
    debug_assert_eq!(dt_score.len(), dt_matched.len());
    debug_assert_eq!(dt_score.len(), dt_iou.len());
    debug_assert!(!tau_grid.is_empty());

    // Treat tau_tp == 1.0 as a sum-of-zeros collapse so the loc-error
    // weight stays defined. Mirrors the oracle's guard.
    let one_minus_tau_tp = if tp_threshold >= 1.0 {
        1.0
    } else {
        1.0 - tp_threshold
    };

    // Sort a permutation of detection indices by score-descending.
    // Per-image dt_scores are score-desc but the per-class concatenation
    // interleaves images, so we sort once here (O(N log N)) to enable
    // the incremental sweep below. NaN scores fall to the back via
    // Ordering::Equal — they're never excluded but also never the min.
    let n = dt_score.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        dt_score[b]
            .partial_cmp(&dt_score[a])
            .unwrap_or(core::cmp::Ordering::Equal)
    });

    // Walk tau from largest to smallest. As tau decreases, more
    // detections become "active" (`dt_score >= tau`). With `order`
    // score-desc, a single forward pointer through `order` suffices —
    // any detection already active stays active for every smaller tau.
    // Result: O(N + T) sweep vs the naive O(T * N).
    let mut n_tp: u64 = 0;
    let mut n_fp: u64 = 0;
    let mut sum_loc: f64 = 0.0;
    let mut cursor: usize = 0;

    let mut best_lrp = f64::INFINITY;
    let mut best_idx: usize = 0;
    let mut best_stats = TauStats {
        n_tp: 0,
        n_fp: 0,
        n_fn: n_pos_gt,
        sum_loc: 0.0,
    };

    for idx in (0..tau_grid.len()).rev() {
        let tau = tau_grid[idx];
        while cursor < n {
            let det = order[cursor];
            if dt_score[det] < tau {
                break;
            }
            if dt_matched[det] {
                n_tp += 1;
                sum_loc += 1.0 - dt_iou[det];
            } else {
                n_fp += 1;
            }
            cursor += 1;
        }
        // Saturating sub guards the (impossible-by-construction) case
        // where greedy matching produced more TPs than positive GTs.
        let n_fn = n_pos_gt.saturating_sub(n_tp);
        let denom = (n_tp + n_fp + n_fn) as f64;
        let lrp = if denom == 0.0 {
            // Unreachable here: n_pos_gt > 0 implies denom > 0 once
            // tau drops below all DT scores (n_fn carries the GTs).
            // Mirror the oracle's `0.0` convention defensively.
            0.0
        } else {
            (sum_loc / one_minus_tau_tp + (n_fp + n_fn) as f64) / denom
        };
        // Reverse scan: the first minimum encountered is the largest
        // tau achieving it (the ADR-0043 tie-break). Strict `<` keeps
        // that first-encountered candidate.
        if lrp < best_lrp {
            best_lrp = lrp;
            best_idx = idx;
            best_stats = TauStats {
                n_tp,
                n_fp,
                n_fn,
                sum_loc,
            };
        }
    }

    Some(TauSearchResult {
        star: best_idx,
        lrp: best_lrp,
        stats: best_stats,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_class_returns_none() {
        let r = search_tau(&[], &[], &[], 0, 0.5, &[0.0, 0.5, 1.0]);
        assert!(r.is_none());
    }

    #[test]
    fn perfect_match_class_zero_lrp() {
        // One detection, perfectly matched (IoU = 1.0), one positive GT.
        let r = search_tau(&[0.9], &[true], &[1.0], 1, 0.5, &[0.0, 0.5, 0.9, 1.0])
            .expect("non-empty class");
        // At tau = 0.0, n_tp=1, n_fp=0, n_fn=0, sum_loc=0 -> LRP=0.
        // At tau = 0.9, same. At tau = 1.0, active drops to 0 ->
        // n_tp=0, n_fp=0, n_fn=1 -> LRP=(0 + 0 + 1)/1 = 1.0.
        // Larger-tau tie-break: 0.9 is the last 0.0 LRP. Confirmed.
        assert!(r.lrp.abs() < 1e-12);
        assert_eq!(r.stats.n_tp, 1);
        assert_eq!(r.stats.n_fn, 0);
    }

    #[test]
    fn argmin_ties_take_larger_tau() {
        // Two detections, both matched at IoU=1.0, one positive GT.
        // Detections with scores 0.4 and 0.8. n_pos_gt = 1 BUT both
        // are matched. Mirror the oracle: greedy matching makes only
        // one a TP. Synthesize: dt0 matched=true, dt1 matched=false.
        // At tau=0.0 both active: 1 TP + 1 FP + 0 FN -> denom=2,
        // LRP=(0/0.5 + 1 + 0)/2 = 0.5.
        // At tau=0.5: only dt1 active (score 0.8): 0 TP + 1 FP +
        // 1 FN -> LRP=(0 + 1 + 1)/2 = 1.0.
        // At tau=0.9 nothing active: 0+0+1, LRP=1.0.
        // Min is 0.5 at tau=0.0.
        // Now add a tied case: dt0 score=0.0 (TP), dt1 score=0.5
        // (FP). At tau=0.0: 1TP+1FP+0FN denom=2 -> LRP=0.5. At
        // tau=0.5: 0TP+1FP+1FN denom=2 -> LRP=1.0. At tau=0.6:
        // 0+0+1 -> LRP=1.0. No tie. Need a real tie scenario:
        // one TP at score 0.5, no other dts, n_pos_gt=1. At
        // tau=0.0..0.5 inclusive: 1+0+0=1 -> LRP=0. At tau>0.5:
        // 0+0+1 -> LRP=1. Tie at LRP=0 across tau=[0, 0.5]. The
        // oracle picks the largest tau on tie -> tau=0.5.
        let grid: Vec<f64> = (0..=10).map(|i| f64::from(i) / 10.0).collect();
        let r = search_tau(&[0.5], &[true], &[1.0], 1, 0.5, &grid).expect("class");
        // tau=0.5 is grid[5].
        assert_eq!(r.star, 5);
        assert!(r.lrp.abs() < 1e-12);
    }

    #[test]
    fn all_fp_class_lrp_one() {
        // One detection, not matched. One positive GT.
        // tau=0.0: 0 TP + 1 FP + 1 FN -> denom=2, LRP=1.0.
        // tau>0.5: 0+0+1 -> denom=1, LRP=1.0.
        // Tie everywhere → larger tau wins → tau=1.0.
        let grid: Vec<f64> = (0..=10).map(|i| f64::from(i) / 10.0).collect();
        let r = search_tau(&[0.5], &[false], &[0.0], 1, 0.5, &grid).expect("class");
        assert!((r.lrp - 1.0).abs() < 1e-12);
        // The trailing entries past tau > 0.5 all give LRP=1.0;
        // larger-tau wins → grid[10]=1.0.
        assert_eq!(r.star, grid.len() - 1);
    }
}
