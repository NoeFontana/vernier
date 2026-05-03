//! Per-bin correction layer (re-detect approach).
//!
//! Each TIDE bin's correction is realized by rebuilding a corrected
//! `(CocoDataset, CocoDetections)` pair and running the standard
//! evaluation pipeline (`evaluate_with` + `accumulate` +
//! `summarize_detection`) on it. The mAP on the corrected inputs minus
//! the baseline mAP is the per-bin ΔmAP this PR ships.
//!
//! This is the "re-detect" path called out in the Week-2 plan: slower
//! than the cell-rewrite-in-place optimization (which is a Week-5 perf
//! follow-up) but correct-by-construction against the ADR-0021 numpy
//! oracle. Eight `evaluate_with` passes per [`super::error_decomposition_bbox`]
//! call (baseline + six bins + all-FP-removed sanity).
//!
//! ## Per-bin recipes (mirroring `oracle.py::_apply_fix`)
//!
//! - **Cls** — for each Cls-binned DT, relabel its `category_id` to the
//!   wrong-class GT's class. The matching pipeline naturally routes
//!   the relabeled DT into the new cell.
//! - **Loc** — for each Loc-binned DT, snap its bbox onto the same-
//!   class target GT's bbox so IoU=1.0 at every threshold.
//! - **Both / Dupe / Bkg** — drop the DT.
//! - **Missed** — set `ignore=true` on the missed GTs (oracle uses
//!   `ignore`; vernier's GT type carries `ignore_flag: Option<bool>`,
//!   and `effective_ignore` resolves to `iscrowd || ignore` under
//!   either parity mode, so flipping `ignore_flag` is sufficient).
//! - **all_fp** (sanity) — drop every FP-binned DT (any of cls / loc /
//!   both / dupe / bkg) at `t_f` simultaneously.

use crate::dataset::{
    CocoAnnotation, CocoDataset, CocoDetection, CocoDetections, DetectionInput, EvalDataset,
};
use crate::error::EvalError;

use super::assignment::{BinAssignment, DtBin};

/// Which correction to apply when calling [`apply_fix`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixKind {
    /// Cls — relabel each Cls-binned DT to the wrong-class GT's class.
    Cls,
    /// Loc — snap each Loc-binned DT's bbox to its same-class target.
    Loc,
    /// Both — drop the DT.
    Both,
    /// Dupe — drop the DT.
    Dupe,
    /// Bkg — drop the DT.
    Bkg,
    /// Missed — flip `ignore=true` on every Missed GT.
    Missed,
    /// All-FP-removed sanity pass — drop every DT whose bin is one of
    /// the five FP bins.
    AllFp,
}

/// Build a corrected `(CocoDataset, CocoDetections)` pair for one bin
/// fix.
///
/// The bin labels come from [`BinAssignment`]; the source dataset and
/// detections are the originals the assignment was computed from.
///
/// # Errors
///
/// Propagates [`EvalError`] from [`CocoDataset::from_parts`] and
/// [`CocoDetections::from_inputs`] when the rebuild fails. The Cls /
/// Loc paths look up GT targets from the assignment's
/// `target_gt_local_idx`; an out-of-range target is a logic bug in
/// [`super::assignment`] and surfaces here as an
/// [`EvalError::InvalidAnnotation`] with the offending image.
pub fn apply_fix(
    gt: &CocoDataset,
    dt: &CocoDetections,
    assignment: &BinAssignment,
    fix: FixKind,
) -> Result<(CocoDataset, CocoDetections), EvalError> {
    // ---- GT side ----
    let mut gts: Vec<CocoAnnotation> = gt.annotations().to_vec();
    if matches!(fix, FixKind::Missed) {
        let missed_set: std::collections::HashSet<(i64, usize)> =
            assignment.missed_gts.iter().copied().collect();
        for (gt_input_idx, ann) in gts.iter_mut().enumerate() {
            if missed_set.contains(&(ann.image_id.0, gt_input_idx)) {
                ann.ignore_flag = Some(true);
            }
        }
    }
    let new_gt = CocoDataset::from_parts(gt.images().to_vec(), gts, gt.categories().to_vec())?;

    // ---- DT side ----
    let mut new_dts: Vec<DetectionInput> = Vec::with_capacity(dt.detections().len());
    let original_anns = gt.annotations();
    for (dt_input_idx, det) in dt.detections().iter().enumerate() {
        let key = (det.image_id.0, dt_input_idx);
        let label = assignment.dt_labels.get(&key).copied();

        // Build the GT-input-index list for this image so we can
        // resolve `target_gt_local_idx` (which is a column index into
        // the per-image GT list in dataset insertion order — the same
        // axis CrossClassIous uses).
        let resolve_target = |target: i32| -> Result<&CocoAnnotation, EvalError> {
            let local_indices = new_gt.ann_indices_for_image(det.image_id);
            let target_usize =
                usize::try_from(target).map_err(|_| EvalError::InvalidAnnotation {
                    detail: format!(
                        "rewrite: invalid target_gt_local_idx={target} for DT id={} on image {}",
                        det.id.0, det.image_id.0
                    ),
                })?;
            local_indices
                .get(target_usize)
                .map(|&j| &original_anns[j])
                .ok_or_else(|| EvalError::InvalidAnnotation {
                    detail: format!(
                        "rewrite: target_gt_local_idx={target} out of range \
                         for image {} (have {} GTs)",
                        det.image_id.0,
                        local_indices.len()
                    ),
                })
        };

        match (fix, label) {
            // ALL_FP: drop every FP, keep TPs and Ignore-matched DTs and
            // DTs evicted by the cap (label = None) — same shape as the
            // oracle.
            (FixKind::AllFp, Some(lbl))
                if matches!(
                    lbl.bin,
                    DtBin::Cls | DtBin::Loc | DtBin::Both | DtBin::Dupe | DtBin::Bkg
                ) =>
            {
                continue;
            }
            // CLS: relabel Cls-binned DTs to the wrong-class target GT's category.
            (FixKind::Cls, Some(lbl)) if lbl.bin == DtBin::Cls => {
                let target = resolve_target(lbl.target_gt_local_idx)?;
                new_dts.push(DetectionInput {
                    id: Some(det.id),
                    image_id: det.image_id,
                    category_id: target.category_id,
                    score: det.score,
                    bbox: det.bbox,
                    segmentation: det.segmentation.clone(),
                    keypoints: det.keypoints.clone(),
                    num_keypoints: det.num_keypoints,
                });
            }
            // LOC: snap Loc-binned DT's bbox to the same-class target GT.
            (FixKind::Loc, Some(lbl)) if lbl.bin == DtBin::Loc => {
                let target = resolve_target(lbl.target_gt_local_idx)?;
                new_dts.push(DetectionInput {
                    id: Some(det.id),
                    image_id: det.image_id,
                    category_id: det.category_id,
                    score: det.score,
                    bbox: target.bbox,
                    segmentation: det.segmentation.clone(),
                    keypoints: det.keypoints.clone(),
                    num_keypoints: det.num_keypoints,
                });
            }
            // BOTH / DUPE / BKG: drop the DT.
            (FixKind::Both, Some(lbl)) if lbl.bin == DtBin::Both => continue,
            (FixKind::Dupe, Some(lbl)) if lbl.bin == DtBin::Dupe => continue,
            (FixKind::Bkg, Some(lbl)) if lbl.bin == DtBin::Bkg => continue,
            // Every other case: pass the DT through unchanged. Includes
            // - DTs evicted by max_dets cap (label = None) → keep them
            //   (mirrors oracle: `attribution.get(d.dt_idx) is None →
            //   pass through`).
            // - DTs whose bin doesn't match the current fix.
            // - Missed fix on the DT side: untouched (the GT side
            //   handled it above).
            _ => {
                new_dts.push(passthrough_input(det));
            }
        }
    }

    let new_dt = CocoDetections::from_inputs(new_dts)?;
    Ok((new_gt, new_dt))
}

/// Rebuild a [`DetectionInput`] from an already-resolved
/// [`CocoDetection`], preserving its id and bbox verbatim. Used in the
/// rewrite layer's pass-through branch so the corrected detections list
/// keeps the original input ordering and ids — which matters for the
/// downstream `evaluate_with` cell ordering only insofar as scores
/// determine the in-cell sort, but ids being preserved makes the
/// rewrite output debug-printable side-by-side with the input.
fn passthrough_input(det: &CocoDetection) -> DetectionInput {
    DetectionInput {
        id: Some(det.id),
        image_id: det.image_id,
        category_id: det.category_id,
        score: det.score,
        bbox: det.bbox,
        segmentation: det.segmentation.clone(),
        keypoints: det.keypoints.clone(),
        num_keypoints: det.num_keypoints,
    }
}
