//! Confusion matrix integration tests against the existing TIDE
//! fixtures (`all_perfect`, `all_cls`, `all_bkg`, `with_ignore`).
//!
//! Each test hand-computes the expected `(gt_class, dt_class) -> count`
//! map for one fixture and asserts the Rust implementation agrees. The
//! fixtures here are reused from `tests/python/oracle/tide/fixtures/`
//! — the same shape that powers the TIDE oracle parity test — so any
//! drift in fixture content is caught across both capabilities.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::HashMap;

use vernier_core::similarity::BboxIou;
use vernier_core::tide::{compute_confusion_matrix, ConfusionMatrixCounts};
use vernier_core::{CocoDataset, CocoDetections, ParityMode};

mod common;
use common::fixture_path;

fn load_fixture(name: &str) -> (CocoDataset, CocoDetections) {
    let gt_bytes = std::fs::read(fixture_path(name, "gt.json"))
        .unwrap_or_else(|e| panic!("failed to read gt.json for fixture {name}: {e}"));
    let dt_bytes = std::fs::read(fixture_path(name, "dt.json"))
        .unwrap_or_else(|e| panic!("failed to read dt.json for fixture {name}: {e}"));
    let gt = CocoDataset::from_json_bytes(&gt_bytes)
        .unwrap_or_else(|e| panic!("failed to parse gt.json for fixture {name}: {e}"));
    let dt = CocoDetections::from_json_bytes(&dt_bytes)
        .unwrap_or_else(|e| panic!("failed to parse dt.json for fixture {name}: {e}"));
    (gt, dt)
}

fn run_cm(name: &str) -> ConfusionMatrixCounts {
    let (gt, dt) = load_fixture(name);
    compute_confusion_matrix(&gt, &dt, &BboxIou, 0.5, 100, ParityMode::Strict)
        .unwrap_or_else(|e| panic!("compute_confusion_matrix failed on fixture {name}: {e}"))
}

/// Local lookup: matrix index for a COCO category id. Mirrors the
/// public helper in `vernier_core::tide::confusion` but kept inline so
/// the test file is self-contained.
fn idx_of(cm: &ConfusionMatrixCounts, cat_id: i64) -> Option<usize> {
    cm.category_ids.iter().position(|&id| id == cat_id)
}

/// Build the expected counts map from raw COCO ids and assert it
/// matches the Rust output cell-for-cell. `expected` is a list of
/// `(gt_id, dt_id, count)` where `None` is the FP/Missed sentinel.
fn assert_counts(
    cm: &ConfusionMatrixCounts,
    expected: &[(Option<i64>, Option<i64>, u64)],
    name: &str,
) {
    let mut want: HashMap<(Option<usize>, Option<usize>), u64> = HashMap::new();
    for &(g_id, d_id, c) in expected {
        let g_idx = g_id.map(|id| {
            idx_of(cm, id).unwrap_or_else(|| panic!("[{name}] gt category id {id} not in matrix"))
        });
        let d_idx = d_id.map(|id| {
            idx_of(cm, id).unwrap_or_else(|| panic!("[{name}] dt category id {id} not in matrix"))
        });
        want.insert((g_idx, d_idx), c);
    }
    assert_eq!(
        cm.counts, want,
        "[{name}] confusion matrix counts mismatch:\n  got: {:?}\n  want: {:?}",
        cm.counts, want
    );
}

#[test]
fn all_perfect_diagonal_only() {
    // Two GTs (cat 1, cat 2), two DTs perfectly aligned with their
    // same-class GT. Every count lands on the diagonal.
    let cm = run_cm("all_perfect");
    assert_counts(
        &cm,
        &[(Some(1), Some(1), 1), (Some(2), Some(2), 1)],
        "all_perfect",
    );
}

#[test]
fn all_cls_off_diagonal_only() {
    // Two GTs, two DTs at the right boxes but with the wrong class.
    // Pair (gt=1, dt=2) at GT 1's location; pair (gt=2, dt=1) at GT
    // 2's location.
    let cm = run_cm("all_cls");
    assert_counts(
        &cm,
        &[(Some(1), Some(2), 1), (Some(2), Some(1), 1)],
        "all_cls",
    );
}

#[test]
fn all_bkg_fp_row_and_missed_column() {
    // Four DTs: two background (high score, no overlap), two
    // covering DTs (lower score, but the score ordering doesn't
    // matter — argmax picks the highest-IoU GT). Both background
    // DTs land in the FP row; the covering DTs land on the diagonal.
    // No missed-GT counts (every GT is covered by the lower-score
    // covering DT).
    let cm = run_cm("all_bkg");
    assert_counts(
        &cm,
        &[
            (None, Some(1), 1),
            (None, Some(2), 1),
            (Some(1), Some(1), 1),
            (Some(2), Some(2), 1),
        ],
        "all_bkg",
    );
}

#[test]
fn with_ignore_crowd_gt_excluded_from_missed() {
    // Image 1: one iscrowd GT (cat 1). DTs:
    //   - DT 0: covers crowd at score 0.6 → dropped (no FP, no TP).
    //   - DT 1: pure FP at score 0.9 → counts in (None, cat 1).
    // Image 2: one regular GT. DT 2: covers it → diagonal (cat 1, cat 1).
    let cm = run_cm("with_ignore");
    assert_counts(
        &cm,
        &[(None, Some(1), 1), (Some(1), Some(1), 1)],
        "with_ignore",
    );
}
