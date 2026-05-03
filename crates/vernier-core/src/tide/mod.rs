//! TIDE error decomposition (Bolya et al., 2020).
//!
//! Decomposes the headline ΔmAP into six bins — Cls / Loc / Both / Dupe /
//! Bkg / Missed — by running corrected accumulations per bin and
//! subtracting from baseline. The bin definitions and the algorithmic
//! contract live in the ADRs that govern this module:
//!
//! - **ADR-0021** — TIDE numpy oracle as the correctness model. The Rust
//!   implementation's correctness contract is `|delta_rust − delta_oracle|
//!   < 1e-9` per bin per fixture; this module is correct iff it agrees
//!   with the oracle.
//! - **ADR-0022** — Per-kernel `(t_f, t_b)` thresholds. Defaults live
//!   alongside the algorithm rather than as Python-side constants; the
//!   resolved thresholds are recorded on every [`report::TideReport`].
//! - **ADR-0023** — Cross-class IoU as an orchestrator-level side pass.
//!   The matching engine (ADR-0005) is left untouched; cross-class
//!   overlaps are gathered by a separate pass through the same
//!   [`crate::similarity::Similarity`] kernel and stored in
//!   [`crate::tables::CrossClassIous`].
//! - **ADR-0024** — TIDE on keypoints (OKS) is deferred to a later
//!   release; this module does not ship a keypoints branch.
//!
//! ## Module layout
//!
//! - [`bins`] — the [`bins::TideErrorBin`] enum naming the six bins.
//! - [`cross_class`] — the side-pass driver
//!   [`cross_class::compute_cross_class_ious`] that populates the
//!   [`crate::tables::CrossClassIous`] storage from a dataset and
//!   detection list. The storage type itself lives next to
//!   [`crate::tables::RetainedIous`] in [`crate::tables`].
//! - [`confusion`] — sibling capability
//!   [`confusion::compute_confusion_matrix`] that consumes the same
//!   cross-class side-pass output to count `(gt_class, dt_class)`
//!   pairs across the dataset. Funded by the same pass as
//!   [`error_decomposition_with`] — one walk, two consumers.
//! - [`params`] — [`params::TideParams`], the inputs bundle for
//!   [`error_decomposition_bbox`].
//! - [`assignment`] — bin-assignment driver
//!   [`assignment::assign_bins`] mirroring `oracle.py::_attribute_bins`.
//! - [`rewrite`] — per-bin correction layer
//!   [`rewrite::apply_fix`] mirroring `oracle.py::_apply_fix`.
//! - [`report`] — [`report::TideReport`] (the per-bin ΔmAP output) and
//!   [`report::TideConfig`] (the resolved thresholds + kernel marker
//!   recorded alongside, per ADR-0022).
//!
//! ## Status
//!
//! Bbox TIDE end-to-end (Week 2) plus segm TIDE end-to-end (Week 3).
//! Boundary kernel and the `mode="per_threshold"` variant remain out
//! of scope for this PR.

pub mod assignment;
pub mod bins;
pub mod confusion;
pub mod cross_class;
pub mod params;
pub mod report;
pub mod rewrite;

pub use assignment::{assign_bins, BinAssignment, DtBin, DtBinLabel};
pub use bins::TideErrorBin;
pub use confusion::{compute_confusion_matrix, ConfusionMatrixCounts};
pub use cross_class::compute_cross_class_ious;
pub use params::TideParams;
pub use report::{KernelMarker, TideConfig, TideReport};
pub use rewrite::{apply_fix, FixKind};

use std::collections::HashMap;

use crate::accumulate::{accumulate, sort_max_dets, AccumulateParams};
use crate::dataset::{CocoDataset, CocoDetections};
use crate::error::EvalError;
use crate::evaluate::{evaluate_with_retention, EvalKernel, EvaluateParams};
use crate::parity::ParityMode;
use crate::similarity::{BboxIou, BoundaryIou, SegmIou};

/// End-to-end TIDE error decomposition over an arbitrary
/// [`EvalKernel`].
///
/// Per ADR-0021, this is the kernel-generic entry point: bbox / segm /
/// boundary all delegate here with their own kernel instance. The
/// algorithm is identical across kernels — only the IoU definition
/// changes — so the per-kernel `error_decomposition_*` wrappers exist
/// to pin a defensible kernel-name string into the [`TideConfig`] and
/// keep the discriminated entry-point list short for FFI registration.
///
/// 1. Runs [`evaluate_with_retention`] to produce the standard
///    [`crate::EvalGrid`] *and* the cross-class IoU side-pass output
///    in one call.
/// 2. Computes baseline mAP via the local [`compute_map`] helper
///    (oracle-faithful semantics; intentionally not [`crate::summarize_with`]
///    — see the helper's doc for the divergence).
/// 3. Walks bin assignment ([`assign_bins`]) using the cross-class
///    storage for `iou_same` / `iou_cross`.
/// 4. For each of the six TIDE bins, builds a corrected
///    `(CocoDataset, CocoDetections)` via [`apply_fix`], re-runs
///    `evaluate_with` (with the same kernel) + the local mAP helper,
///    and records `delta_bin = corrected_map - baseline_map`. Bins
///    with no DT/GT to correct are absent from the output map (the
///    FFI layer fills missing keys with `0.0`).
/// 5. Runs the all-FP-removed sanity pass and records its delta.
///
/// Eight `evaluate_with` passes total (one baseline + six bins + one
/// sanity). The re-detect approach (rebuild detections per bin) is
/// slower than the cell-rewrite-in-place optimization the ADR sketches
/// for Week 5 but is correct-by-construction against the numpy oracle.
///
/// `kernel_marker` is recorded verbatim on [`TideConfig::kernel`]; the
/// per-kernel wrappers below pin the canonical [`KernelMarker`] variant
/// alongside the concrete kernel implementor.
///
/// # Errors
///
/// Propagates [`EvalError`] from the underlying evaluation,
/// accumulate, summarize, and rewrite calls.
pub fn error_decomposition_with<K: EvalKernel>(
    gt: &CocoDataset,
    dt: &CocoDetections,
    kernel: &K,
    kernel_marker: KernelMarker,
    params: TideParams<'_>,
    parity_mode: ParityMode,
) -> Result<TideReport, EvalError> {
    let eval_params = EvaluateParams {
        iou_thresholds: params.iou_thresholds,
        area_ranges: params.area_ranges,
        max_dets_per_image: params.max_dets_per_image,
        use_cats: params.use_cats,
        retain_iou: false,
    };

    // 1. Baseline pass + cross-class side pass in one call.
    let (grid, cross_class) = evaluate_with_retention(gt, dt, eval_params, parity_mode, kernel)?;
    let baseline_map = compute_map(&grid, params.iou_thresholds, params.max_dets_per_image)?;

    // 2. Bin assignment.
    let assignment = assign_bins(gt, dt, &cross_class, &params)?;

    // 3. Per-bin re-evaluation. Skip bins that have no work to do (no
    //    DTs assigned to them) so the output map matches the oracle's
    //    "absent bin" convention from `TideReport.delta_per_bin`'s docs.
    let mut delta_per_bin: HashMap<TideErrorBin, f64> = HashMap::new();
    for (bin, fix) in [
        (TideErrorBin::Cls, FixKind::Cls),
        (TideErrorBin::Loc, FixKind::Loc),
        (TideErrorBin::Both, FixKind::Both),
        (TideErrorBin::Dupe, FixKind::Dupe),
        (TideErrorBin::Bkg, FixKind::Bkg),
        (TideErrorBin::Missed, FixKind::Missed),
    ] {
        if !bin_has_work(&assignment, bin) {
            continue;
        }
        let delta = run_fix_pass(
            gt,
            dt,
            kernel,
            &assignment,
            fix,
            &params,
            parity_mode,
            baseline_map,
        )?;
        delta_per_bin.insert(bin, delta);
    }

    // 4. All-FP-removed sanity pass — always runs even if every bin
    //    is empty, so the report carries a defensible upper-bound
    //    number for the caller's own reasoning.
    let delta_all_fp = run_fix_pass(
        gt,
        dt,
        kernel,
        &assignment,
        FixKind::AllFp,
        &params,
        parity_mode,
        baseline_map,
    )?;

    let config = TideConfig {
        t_f: params.t_f,
        t_b: params.t_b,
        kernel: kernel_marker,
        cross_class_topk: None,
    };
    Ok(TideReport {
        baseline_map,
        delta_per_bin,
        delta_all_fp,
        config,
    })
}

/// End-to-end bbox TIDE error decomposition.
///
/// Thin wrapper over [`error_decomposition_with`] that pins the
/// [`BboxIou`] kernel and the canonical `"bbox"` kernel-name string.
/// See the generic entry point's doc for the full algorithm.
///
/// # Errors
///
/// Propagates [`EvalError`] from the underlying evaluation,
/// accumulate, summarize, and rewrite calls.
pub fn error_decomposition_bbox(
    gt: &CocoDataset,
    dt: &CocoDetections,
    params: TideParams<'_>,
    parity_mode: ParityMode,
) -> Result<TideReport, EvalError> {
    error_decomposition_with(gt, dt, &BboxIou, KernelMarker::Bbox, params, parity_mode)
}

/// End-to-end segm TIDE error decomposition.
///
/// Thin wrapper over [`error_decomposition_with`] that pins the
/// [`SegmIou`] kernel and the canonical `"segm"` kernel-name string.
/// See the generic entry point's doc for the full algorithm.
///
/// # Errors
///
/// Propagates [`EvalError`] from the underlying evaluation,
/// accumulate, summarize, and rewrite calls.
pub fn error_decomposition_segm(
    gt: &CocoDataset,
    dt: &CocoDetections,
    params: TideParams<'_>,
    parity_mode: ParityMode,
) -> Result<TideReport, EvalError> {
    error_decomposition_with(gt, dt, &SegmIou, KernelMarker::Segm, params, parity_mode)
}

/// End-to-end boundary-segm TIDE error decomposition.
///
/// Thin wrapper over [`error_decomposition_with`] that pins the
/// [`BoundaryIou`] kernel (configured with the caller-supplied
/// `dilation_ratio`) and the canonical `"boundary"` kernel-name
/// string. See the generic entry point's doc for the full algorithm;
/// ADR-0010 for the boundary kernel's geometry; ADR-0022 for the
/// per-kernel `(t_f, t_b)` defaults (the boundary row carries the
/// tentative `t_b = 0.05` default at `dilation_ratio = 0.02`).
///
/// `dilation_ratio` is taken as a direct argument rather than a knob
/// on [`TideParams`] so the kernel-name → kernel-config coupling lives
/// in this wrapper instead of leaking into the kernel-generic
/// `TideParams`. The cached-eval path
/// ([`crate::evaluate_boundary_cached`]) is intentionally not used
/// here: TIDE re-evaluates the same dataset eight times under one
/// process; the un-cached kernel re-derives bands per call but avoids
/// threading a [`crate::similarity::BoundaryGtCache`] through the
/// per-bin rewrite passes. A cached variant is a Week-5 perf
/// follow-up.
///
/// # Errors
///
/// Propagates [`EvalError`] from the underlying evaluation,
/// accumulate, summarize, and rewrite calls.
pub fn error_decomposition_boundary(
    gt: &CocoDataset,
    dt: &CocoDetections,
    params: TideParams<'_>,
    parity_mode: ParityMode,
    dilation_ratio: f64,
) -> Result<TideReport, EvalError> {
    let kernel = BoundaryIou { dilation_ratio };
    error_decomposition_with(gt, dt, &kernel, KernelMarker::Boundary, params, parity_mode)
}

fn bin_has_work(assignment: &BinAssignment, bin: TideErrorBin) -> bool {
    match bin {
        TideErrorBin::Missed => !assignment.missed_gts.is_empty(),
        TideErrorBin::Cls => assignment.dt_labels.values().any(|l| l.bin == DtBin::Cls),
        TideErrorBin::Loc => assignment.dt_labels.values().any(|l| l.bin == DtBin::Loc),
        TideErrorBin::Both => assignment.dt_labels.values().any(|l| l.bin == DtBin::Both),
        TideErrorBin::Dupe => assignment.dt_labels.values().any(|l| l.bin == DtBin::Dupe),
        TideErrorBin::Bkg => assignment.dt_labels.values().any(|l| l.bin == DtBin::Bkg),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_fix_pass<K: EvalKernel>(
    gt: &CocoDataset,
    dt: &CocoDetections,
    kernel: &K,
    assignment: &BinAssignment,
    fix: FixKind,
    params: &TideParams<'_>,
    parity_mode: ParityMode,
    baseline_map: f64,
) -> Result<f64, EvalError> {
    let (corrected_gt, corrected_dt) = apply_fix(gt, dt, assignment, fix)?;
    let eval_params = EvaluateParams {
        iou_thresholds: params.iou_thresholds,
        area_ranges: params.area_ranges,
        max_dets_per_image: params.max_dets_per_image,
        use_cats: params.use_cats,
        retain_iou: false,
    };
    // Re-evaluate. Use `evaluate_with` (not the retention variant) —
    // we only need the EvalGrid for the corrected mAP, not the cross-
    // class side pass. The kernel is whatever the caller passed in:
    // bbox / segm / boundary all flow through this same pass.
    let grid = crate::evaluate::evaluate_with(
        &corrected_gt,
        &corrected_dt,
        eval_params,
        parity_mode,
        kernel,
    )?;
    let corrected_map = compute_map(&grid, params.iou_thresholds, params.max_dets_per_image)?;
    Ok(corrected_map - baseline_map)
}

/// Compute mAP from an [`crate::EvalGrid`] using the **oracle's**
/// per-cell AP convention (mirrors `oracle.py::_compute_map`).
///
/// For each `(category, threshold)` cell at `area=all`, `max_dets=
/// largest`:
///
/// - if the cell has no non-ignore GTs (`npig == 0`) → the cell's AP
///   is `-1` (filtered, does not contribute to the mean);
/// - else if the cell has no DTs but has GTs → AP is `0.0` (recall
///   stays at 0, precision lane is uniformly zero);
/// - else → AP is the standard 101-point recall-sampled mean of the
///   precision envelope.
///
/// The accumulator's filled `recall` and `precision` arrays let us
/// distinguish the two empty-cell cases without re-running the cell
/// loop: when `npig == 0`, both `recall` and `precision` stay at the
/// `-1` sentinel; when `npig > 0` but `n_dts == 0`, `recall` is set
/// to `0.0` while `precision` stays at `-1` (see
/// `accumulate::accumulate_cell`'s `n_d == 0` early-exit).
///
/// This is intentionally **not** a call to [`crate::summarize_with`].
/// pycocotools — and therefore the standard summarizer — filters every
/// `-1` precision sample out of the mean, which gives the wrong answer
/// for the second case (cells with GTs but no DTs should contribute a
/// real `0.0` to mAP, not be silently dropped). Per ADR-0021's note on
/// mAP semantics, the oracle is the spec; this helper computes the
/// oracle's mean directly rather than touching `summarize.rs`.
///
/// When every cell is empty (`-1`), the helper returns `0.0` —
/// matches `oracle.py::_compute_map`'s `if not ap_values: return 0.0`
/// early-exit so the per-bin delta arithmetic stays defined.
fn compute_map(
    grid: &crate::EvalGrid,
    iou_thresholds: &[f64],
    max_dets_per_image: usize,
) -> Result<f64, EvalError> {
    // Single-rung max_dets ladder — matches the oracle, which has only
    // one cap. Sorting is a no-op for a one-entry slice; we still call
    // `sort_max_dets` to keep the discipline ADR-0002 / quirk **A2**
    // requires at the param-construction boundary.
    let mut max_dets = vec![max_dets_per_image];
    sort_max_dets(&mut max_dets);
    let acc = accumulate(
        &grid.eval_imgs,
        AccumulateParams {
            iou_thresholds,
            recall_thresholds: crate::recall_thresholds(),
            max_dets: &max_dets,
            n_categories: grid.n_categories,
            n_area_ranges: grid.n_area_ranges,
            n_images: grid.n_images,
        },
        ParityMode::Strict,
    )?;

    let n_t = acc.precision.shape()[0];
    let n_r = acc.precision.shape()[1];
    let n_k = acc.precision.shape()[2];
    let area_idx = 0usize; // `all` bucket — index 0 in `AreaRange::coco_default`.
    let m_idx = max_dets.len() - 1;

    let mut ap_values: Vec<f64> = Vec::with_capacity(n_t * n_k);
    for t in 0..n_t {
        for k in 0..n_k {
            // Recall is `-1` when npig == 0 → cell is empty, filter out.
            // Recall is `0` (set by the `n_d == 0` branch) when there
            // are GTs but no DTs → AP = 0.0.
            // Otherwise: mean over the R recall-thresholds of the
            // precision envelope, matching the oracle's
            // `sampled.mean()` step.
            let recall_v = acc.recall[(t, k, area_idx, m_idx)];
            if recall_v < 0.0 {
                continue; // npig == 0 — sentinel, filter.
            }
            // Walk the R recall-axis directly via indexing — the
            // multi-step `index_axis` chain returns temporaries that
            // get dropped before they can be borrowed.
            let mut any_nonneg = false;
            let mut s = 0.0_f64;
            for r in 0..n_r {
                let v = acc.precision[(t, r, k, area_idx, m_idx)];
                if v >= 0.0 {
                    any_nonneg = true;
                    s += v;
                }
            }
            if any_nonneg {
                // Accumulator either fills every R slot (DTs case)
                // or leaves them all at -1 (no-DTs case). The
                // any_nonneg check means we're in the filled branch
                // — sum / R is the canonical 101-point AP.
                ap_values.push(s / (n_r as f64));
            } else {
                // npig > 0 but n_dts == 0 → AP = 0.0 (oracle).
                ap_values.push(0.0);
            }
        }
    }

    if ap_values.is_empty() {
        return Ok(0.0);
    }
    let total: f64 = ap_values.iter().sum();
    Ok(total / (ap_values.len() as f64))
}
