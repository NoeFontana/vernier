//! FP-IoU histogram extractor for ADR-0022 `t_b` ratification.
//!
//! For every detection that bin assignment classifies as a false
//! positive (Cls / Loc / Both / Dupe / Bkg — i.e. anything that's not
//! TP and not Ignore), record the best same-class IoU and the best
//! cross-class IoU at the time of the bin pick. Python-side analysis
//! consumes the resulting `(iou_same, iou_cross)` arrays to compute
//! the "bin-as-Bkg fraction at candidate `t_b`" curve the ADR's
//! decision gate calls for.
//!
//! The implementation is a thin shim over the regular TIDE pipeline:
//! it runs [`evaluate_with_retention`] for the cross-class side pass
//! (ADR-0023), runs [`assign_bins`] (ADR-0021), then walks the
//! resulting `dt_labels` and copies out the IoU values that bin pick
//! already computed. No re-walk through the kernel; the iou values
//! ride along on [`DtBinLabel`] for free.

use crate::dataset::{CocoDataset, CocoDetections};
use crate::error::EvalError;
use crate::evaluate::EvalKernel;
use crate::parity::ParityMode;
use crate::similarity::{BboxIou, BoundaryIou, SegmIou};

use super::assignment::{assign_bins, DtBin};
use super::cross_class::compute_cross_class_ious;
use super::params::TideParams;
use super::report::KernelMarker;

/// Output of [`compute_fp_iou_histogram_with`].
///
/// `iou_same[i]` and `iou_cross[i]` are the best same-class and
/// best-cross-class IoUs for the `i`-th false-positive detection
/// (Cls / Loc / Both / Dupe / Bkg). Order is undefined — Python-side
/// analysis just bins them.
///
/// `n_total_dts` is the total number of detections that survived the
/// per-image `max_dets` cap (TP + Ignore + every FP variant). `n_fps`
/// is `iou_same.len()` (== `iou_cross.len()`); both fields are
/// surfaced explicitly so the caller can compute the FP fraction
/// without re-summing.
#[derive(Debug, Clone)]
pub struct FpIouHistogram {
    /// Best same-class IoU per FP detection, in dt_labels iteration order.
    pub iou_same: Vec<f64>,
    /// Best cross-class IoU per FP detection, parallel to `iou_same`.
    pub iou_cross: Vec<f64>,
    /// Kernel that produced the IoUs (mirrors [`super::TideConfig::kernel`]).
    pub kernel: KernelMarker,
    /// `t_f` used to identify TP/Ignore. Cross-checks the analysis
    /// script's assumed value (ADR-0022 standard is `0.5`).
    pub t_f: f64,
    /// Total surviving detections (TP + Ignore + every FP).
    pub n_total_dts: usize,
    /// FP count — equal to `iou_same.len()` and `iou_cross.len()`.
    pub n_fps: usize,
}

/// Generic FP-IoU histogram extractor. Per-kernel wrappers below pin
/// the kernel implementor and the [`KernelMarker`] string.
///
/// `t_b` on `params` is unused — bin-as-Bkg is decided by Python-side
/// analysis from the histogram, not pinned in the extractor — but it
/// rides along on [`TideParams`] anyway so callers can reuse the same
/// param-construction code as `error_decomposition_*`.
///
/// # Errors
///
/// Propagates [`EvalError`] from the underlying evaluation and bin
/// assignment passes.
pub fn compute_fp_iou_histogram_with<K: EvalKernel>(
    gt: &CocoDataset,
    dt: &CocoDetections,
    kernel: &K,
    kernel_marker: KernelMarker,
    params: TideParams<'_>,
    parity_mode: ParityMode,
) -> Result<FpIouHistogram, EvalError> {
    // Skip the full accumulator pass — `error_decomposition_with` calls
    // `evaluate_with_retention` because it needs the EvalGrid for mAP;
    // the histogram only needs the cross-class IoUs, so call the
    // side-pass driver directly. Saves the n_t × n_k × n_a accumulator
    // walk on every call.
    let cross_class =
        compute_cross_class_ious(gt, dt, kernel, parity_mode, params.max_dets_per_image)?;
    let assignment = assign_bins(gt, dt, &cross_class, &params)?;

    let mut iou_same = Vec::new();
    let mut iou_cross = Vec::new();
    let n_total_dts = assignment.dt_labels.len();

    for label in assignment.dt_labels.values() {
        match label.bin {
            DtBin::Tp | DtBin::Ignore => continue,
            DtBin::Cls | DtBin::Loc | DtBin::Both | DtBin::Dupe | DtBin::Bkg => {
                iou_same.push(label.iou_same);
                iou_cross.push(label.iou_cross);
            }
        }
    }

    let n_fps = iou_same.len();
    Ok(FpIouHistogram {
        iou_same,
        iou_cross,
        kernel: kernel_marker,
        t_f: params.t_f,
        n_total_dts,
        n_fps,
    })
}

/// Bbox-kernel FP-IoU histogram. Thin wrapper over
/// [`compute_fp_iou_histogram_with`].
///
/// # Errors
///
/// Propagates [`EvalError`] from the underlying passes.
pub fn compute_fp_iou_histogram_bbox(
    gt: &CocoDataset,
    dt: &CocoDetections,
    params: TideParams<'_>,
    parity_mode: ParityMode,
) -> Result<FpIouHistogram, EvalError> {
    compute_fp_iou_histogram_with(gt, dt, &BboxIou, KernelMarker::Bbox, params, parity_mode)
}

/// Segm-kernel FP-IoU histogram.
///
/// # Errors
///
/// Propagates [`EvalError`] from the underlying passes.
pub fn compute_fp_iou_histogram_segm(
    gt: &CocoDataset,
    dt: &CocoDetections,
    params: TideParams<'_>,
    parity_mode: ParityMode,
) -> Result<FpIouHistogram, EvalError> {
    compute_fp_iou_histogram_with(gt, dt, &SegmIou, KernelMarker::Segm, params, parity_mode)
}

/// Boundary-kernel FP-IoU histogram. `dilation_ratio` configures the
/// boundary band thickness (ADR-0010); ADR-0022 uses `0.02` as the
/// standard.
///
/// # Errors
///
/// Propagates [`EvalError`] from the underlying passes.
pub fn compute_fp_iou_histogram_boundary(
    gt: &CocoDataset,
    dt: &CocoDetections,
    params: TideParams<'_>,
    parity_mode: ParityMode,
    dilation_ratio: f64,
) -> Result<FpIouHistogram, EvalError> {
    let kernel = BoundaryIou { dilation_ratio };
    compute_fp_iou_histogram_with(
        gt,
        dt,
        &kernel,
        KernelMarker::Boundary,
        params,
        parity_mode,
    )
}
