//! LRP / oLRP error decomposition (Oksuz et al., ECCV 2018; TPAMI 2021).
//!
//! Decomposes a model's detection performance into a single number
//! (oLRP) plus three additive components — `oLRP_Loc` / `oLRP_FP` /
//! `oLRP_FN` — minimised over a per-class confidence threshold `tau`.
//! The metric ships the *(number, threshold)* pair as the headline
//! deliverable because `tau` is the deployable cutoff a practitioner
//! would set on the model to get the reported behavior.
//!
//! The semantics, oracle, namespace, and the kemaloksuz tripwire are
//! governed by:
//!
//! - **ADR-0043** — numpy oracle as the correctness model. The Rust
//!   implementation is correct iff it agrees with
//!   `tests/python/oracle/lrp/oracle.py` within `1e-9` per component
//!   per class per fixture.
//! - **ADR-0044** — Per-kernel `(tp_threshold, tau_grid)` defaults.
//!   `tp_threshold = 0.5` for every kernel; tau grid `0.01` step
//!   over `[0.0, 1.0]`.
//! - **ADR-0045** — LRP on keypoints (OKS) ships as a first-class
//!   kernel, in contrast to TIDE-on-OKS (ADR-0024 deferral). The
//!   structural objections that justified that deferral do not
//!   transfer to LRP.
//!
//! ## Module layout
//!
//! - [`params`] — [`LrpParams`], the inputs bundle.
//! - [`defaults`] — [`defaults::tp_threshold_for`] and
//!   [`defaults::default_tau_grid`] resolve per-kernel defaults
//!   per ADR-0044.
//! - [`tau_search`] — per-class confidence sweep.
//! - [`decompose`] — per-class greedy matching + additive
//!   decomposition.

pub mod decompose;
pub mod defaults;
pub mod params;
pub mod tau_search;

pub use defaults::{default_tau_grid, tp_threshold_for};
pub use params::LrpParams;

use crate::dataset::{CocoDataset, CocoDetections, EvalDataset};
use crate::error::EvalError;
use crate::evaluate::COLLAPSED_CATEGORY_SENTINEL;
use crate::parity::ParityMode;
use crate::similarity::{BboxIou, BoundaryIou, OksSimilarity, SegmIou};

use std::collections::{HashMap, HashSet};

/// Closed set of kernels LRP supports today (per ADR-0043 +
/// ADR-0045). Mirrors [`crate::tide::report::KernelMarker`] but adds
/// the [`Keypoints`](Self::Keypoints) variant TIDE intentionally omits
/// per ADR-0024.
///
/// `as_str` projection gives the canonical lowercase identifier the
/// FFI surface and the numpy oracle both use (`"bbox"` / `"segm"` /
/// `"boundary"` / `"keypoints"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LrpKernelMarker {
    /// Bounding-box IoU (ADR-0008).
    Bbox,
    /// Segmentation-mask IoU (ADR-0009).
    Segm,
    /// Boundary IoU (ADR-0010).
    Boundary,
    /// OKS (Object Keypoint Similarity, ADR-0012).
    Keypoints,
}

impl LrpKernelMarker {
    /// Canonical lowercase name; pinned so it stays bit-stable across
    /// the FFI dict surface and the oracle's `config.kernel` field.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bbox => "bbox",
            Self::Segm => "segm",
            Self::Boundary => "boundary",
            Self::Keypoints => "keypoints",
        }
    }
}

/// Per-class entry on an [`LrpReport`].
///
/// Per ADR-0043, classes with no positive GTs report `tau = None,
/// olrp = None` and are excluded from the headline mean. Classes
/// where no detection ever surfaces a TP at any tau ("all-FN") report
/// `tau = None, olrp = Some(1.0)` — the paper's worst-case lower
/// bound — and are *included* in the headline mean (an all-FN class
/// is a real class with a real cost).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LrpPerClass {
    /// COCO category id; [`COLLAPSED_CATEGORY_SENTINEL`] when the
    /// caller passed `use_cats=false`.
    pub category_id: i64,
    /// Per-class oLRP. `None` for classes with no positive GTs (a
    /// class that exists only as a column on the detection side).
    pub olrp: Option<f64>,
    /// Per-class `oLRP_Loc`. `None` when the class has no TPs at any
    /// tau, or no positive GTs.
    pub olrp_loc: Option<f64>,
    /// Per-class `oLRP_FP`. `None` when the class has no TPs at any
    /// tau and no FPs either, or no positive GTs.
    pub olrp_fp: Option<f64>,
    /// Per-class `oLRP_FN`. `None` when the class has no positive
    /// GTs.
    pub olrp_fn: Option<f64>,
    /// Optimal tau (confidence threshold) the class's oLRP was
    /// achieved at. `None` for all-FN classes (no TP at any tau) and
    /// classes with no positive GTs.
    pub tau: Option<f64>,
}

/// Resolved LRP configuration recorded on every [`LrpReport`].
///
/// Per ADR-0044, the `(tp_threshold, tau_grid_len)` resolved values
/// land here so every report self-describes; screenshots of a number
/// are re-derivable from the report alone.
#[derive(Debug, Clone, PartialEq)]
pub struct LrpConfig {
    /// IoU/OKS floor above which a matched pair counts as a TP.
    pub tp_threshold: f64,
    /// Tau grid length used for the search. The grid itself is
    /// borrowed by the call; storing the length here keeps the
    /// `LrpConfig` value-typed without forcing a clone of the grid.
    pub tau_grid_len: usize,
    /// Kernel this report was produced under.
    pub kernel: LrpKernelMarker,
}

/// Output of an LRP pass: aggregated headline numbers, per-class
/// breakdown, and the resolved configuration.
///
/// Per ADR-0043, the headline numbers are unweighted means over
/// classes with at least one positive GT. Classes with `olrp = None`
/// (no positive GTs) are excluded; all-FN classes (`olrp = Some(1.0),
/// tau = None`) ARE included — the worst-case is a real result, not
/// missing data.
#[derive(Debug, Clone)]
pub struct LrpReport {
    /// Mean of `per_class[k].olrp` across classes with at least one
    /// positive GT. `0.0` when no such class exists (vacuous —
    /// matches the oracle's empty-mean convention).
    pub olrp: f64,
    /// Mean of `per_class[k].olrp_loc` across classes with at least
    /// one TP at the optimal tau.
    pub olrp_loc: f64,
    /// Mean of `per_class[k].olrp_fp` across classes with at least
    /// one TP at the optimal tau (the same denominator as
    /// `olrp_loc`).
    pub olrp_fp: f64,
    /// Mean of `per_class[k].olrp_fn` across classes with at least
    /// one positive GT. Note: this denominator differs from `olrp_loc`
    /// / `olrp_fp` — `oLRP_FN` is well-defined for all-FN classes
    /// (the FN rate at tau is 1.0 for those classes) so they
    /// contribute to the mean.
    pub olrp_fn: f64,
    /// Per-class breakdown, one entry per class — in the same
    /// id-ascending order [`crate::evaluate_with`]'s K axis uses.
    pub per_class: Vec<LrpPerClass>,
    /// Number of classes with no positive GTs (excluded from the
    /// headline means). Useful for sanity-checking a report from a
    /// federated dataset.
    pub n_empty_classes: u32,
    /// Resolved configuration this report was produced under (per
    /// ADR-0044).
    pub config: LrpConfig,
}

/// End-to-end LRP over an arbitrary [`crate::evaluate::EvalKernel`].
///
/// Per ADR-0043, this is the kernel-generic entry point: bbox / segm
/// / boundary / keypoints all delegate here with their own kernel
/// instance. The algorithm is identical across kernels — only the
/// similarity definition changes — so the per-kernel
/// `optimal_lrp_*` wrappers exist to pin a defensible kernel-name
/// string into the [`LrpConfig`] and keep the discriminated entry-
/// point list short for FFI registration.
///
/// 1. Runs the per-image evaluation pass with `retain_iou=true` so
///    the resulting [`crate::evaluate::EvalGrid`] carries the
///    per-`(category, image)` IoU matrices on
///    [`crate::evaluate::EvalGrid::retained_ious`].
/// 2. For each category index `k` walks the retained IoU matrices,
///    runs the per-cell greedy matching at `params.tp_threshold` (the
///    oracle's `_match_per_class` algorithm), and concatenates the
///    per-image `(score, matched, iou)` triplets into class-level
///    arrays.
/// 3. Sweeps `params.tau_grid` to find the LRP-minimising tau per
///    class (tie-break: larger tau wins, per ADR-0043).
/// 4. Decomposes the result into `oLRP_Loc` / `oLRP_FP` / `oLRP_FN`
///    via paper eq. 10, packages everything into [`LrpReport`].
///
/// # Errors
///
/// Propagates [`EvalError`] from the underlying evaluation pass.
/// Returns [`EvalError::InvalidConfig`] if `tau_grid` is empty.
pub fn optimal_lrp_with<K: crate::evaluate::EvalKernel>(
    gt: &CocoDataset,
    dt: &CocoDetections,
    kernel: &K,
    kernel_marker: LrpKernelMarker,
    params: LrpParams<'_>,
    parity_mode: ParityMode,
) -> Result<LrpReport, EvalError> {
    validate_params(&params)?;
    let ctx = decompose::prepare_lrp_pass(gt, dt, kernel, &params, parity_mode)?;
    let decompositions = decompose::decompose_all_classes(&ctx, parity_mode, &params, None)?;
    Ok(build_report(
        gt,
        &decompositions,
        params.use_cats,
        params.tp_threshold,
        params.tau_grid.len(),
        kernel_marker,
    ))
}

/// End-to-end LRP for a kernel + partition spec (ADR-0046).
///
/// Runs the matching engine **once** (the C3 axiom) and then runs the
/// post-match decompose pipeline `1 + slices.len()` times: once for
/// the overall report, then once per slice filtering the decompose
/// walk to that slice's image indices. The matching pass is never
/// re-invoked per slice — partitioned LRP enjoys the same "1× match
/// + N cheap aggregate passes" performance shape as partitioned AP.
///
/// `image_filters` is the parallel vector of `image_indices` sets,
/// one per slice in the caller's intended output order. The returned
/// vector pairs them: index `0` is the overall report; indices `1..=N`
/// are the slice reports in the same order as `image_filters`.
///
/// # Errors
///
/// Propagates [`EvalError`] from the underlying matching / decompose
/// passes.
pub fn optimal_lrp_with_partitioned<K: crate::evaluate::EvalKernel>(
    gt: &CocoDataset,
    dt: &CocoDetections,
    kernel: &K,
    kernel_marker: LrpKernelMarker,
    params: LrpParams<'_>,
    parity_mode: ParityMode,
    image_filters: &[HashSet<usize>],
) -> Result<Vec<LrpReport>, EvalError> {
    validate_params(&params)?;
    let ctx = decompose::prepare_lrp_pass(gt, dt, kernel, &params, parity_mode)?;
    let mut reports: Vec<LrpReport> = Vec::with_capacity(image_filters.len() + 1);

    let overall = decompose::decompose_all_classes(&ctx, parity_mode, &params, None)?;
    reports.push(build_report(
        gt,
        &overall,
        params.use_cats,
        params.tp_threshold,
        params.tau_grid.len(),
        kernel_marker,
    ));

    for filter in image_filters {
        let sliced = decompose::decompose_all_classes(&ctx, parity_mode, &params, Some(filter))?;
        reports.push(build_report(
            gt,
            &sliced,
            params.use_cats,
            params.tp_threshold,
            params.tau_grid.len(),
            kernel_marker,
        ));
    }
    Ok(reports)
}

/// Translate a vector of per-class decompositions into a public
/// [`LrpReport`], plus aggregate the headline numbers.
///
/// Shared between [`optimal_lrp_with`] (single-pass) and
/// [`optimal_lrp_with_partitioned`] (one call per `(overall, slice...)`
/// decompose walk).
fn build_report(
    gt: &CocoDataset,
    decompositions: &[decompose::PerClassDecomposition],
    use_cats: bool,
    tp_threshold: f64,
    tau_grid_len: usize,
    kernel_marker: LrpKernelMarker,
) -> LrpReport {
    let cat_id_by_index = build_category_id_lookup(gt, use_cats);
    let mut per_class: Vec<LrpPerClass> = Vec::with_capacity(decompositions.len());
    for d in decompositions {
        let category_id = cat_id_by_index
            .get(&d.category_index)
            .copied()
            .unwrap_or(COLLAPSED_CATEGORY_SENTINEL);
        per_class.push(LrpPerClass {
            category_id,
            olrp: d.olrp,
            olrp_loc: d.olrp_loc,
            olrp_fp: d.olrp_fp,
            olrp_fn: d.olrp_fn,
            tau: d.tau,
        });
    }
    let (olrp, olrp_loc, olrp_fp, olrp_fn, n_empty) = aggregate(&per_class);
    LrpReport {
        olrp,
        olrp_loc,
        olrp_fp,
        olrp_fn,
        per_class,
        n_empty_classes: n_empty,
        config: LrpConfig {
            tp_threshold,
            tau_grid_len,
            kernel: kernel_marker,
        },
    }
}

/// End-to-end LRP for the bbox kernel.
///
/// Thin wrapper over [`optimal_lrp_with`] that pins the [`BboxIou`]
/// kernel and the canonical `"bbox"` kernel-name string.
///
/// # Errors
///
/// Propagates [`EvalError`] from the underlying evaluation pass.
pub fn optimal_lrp_bbox(
    gt: &CocoDataset,
    dt: &CocoDetections,
    params: LrpParams<'_>,
    parity_mode: ParityMode,
) -> Result<LrpReport, EvalError> {
    optimal_lrp_with(gt, dt, &BboxIou, LrpKernelMarker::Bbox, params, parity_mode)
}

/// End-to-end LRP for the segm (mask) kernel.
///
/// # Errors
///
/// Propagates [`EvalError`] from the underlying evaluation pass.
pub fn optimal_lrp_segm(
    gt: &CocoDataset,
    dt: &CocoDetections,
    params: LrpParams<'_>,
    parity_mode: ParityMode,
) -> Result<LrpReport, EvalError> {
    optimal_lrp_with(gt, dt, &SegmIou, LrpKernelMarker::Segm, params, parity_mode)
}

/// End-to-end LRP for the boundary-segm kernel.
///
/// `dilation_ratio` configures the boundary band thickness (ADR-0010
/// default `0.02` for COCO, `0.008` for LVIS).
///
/// # Errors
///
/// Propagates [`EvalError`] from the underlying evaluation pass.
pub fn optimal_lrp_boundary(
    gt: &CocoDataset,
    dt: &CocoDetections,
    params: LrpParams<'_>,
    parity_mode: ParityMode,
    dilation_ratio: f64,
) -> Result<LrpReport, EvalError> {
    let kernel = BoundaryIou { dilation_ratio };
    optimal_lrp_with(
        gt,
        dt,
        &kernel,
        LrpKernelMarker::Boundary,
        params,
        parity_mode,
    )
}

/// End-to-end LRP for the keypoints (OKS) kernel.
///
/// `sigmas` is the per-category sigma override map consumed by
/// [`OksSimilarity::new`]; an empty map means "use the COCO-person
/// 17-sigma table for every category" (quirk **F1**, `corrected`).
///
/// Per ADR-0045 LRP-on-OKS ships in 0.5.x. The structural objections
/// that justified the TIDE-on-OKS deferral (ADR-0024) do not transfer
/// — LRP has no cross-class bins, and `oLRP_Loc = 1 - mean(OKS on TPs)`
/// is well-defined on OKS workloads.
///
/// # Errors
///
/// Propagates [`EvalError`] from the underlying evaluation pass.
pub fn optimal_lrp_keypoints(
    gt: &CocoDataset,
    dt: &CocoDetections,
    params: LrpParams<'_>,
    parity_mode: ParityMode,
    sigmas: HashMap<i64, Vec<f64>>,
) -> Result<LrpReport, EvalError> {
    let kernel = OksSimilarity::new(sigmas);
    optimal_lrp_with(
        gt,
        dt,
        &kernel,
        LrpKernelMarker::Keypoints,
        params,
        parity_mode,
    )
}

/// Partitioned-LRP entry point for the bbox kernel (ADR-0046).
///
/// Thin wrapper over [`optimal_lrp_with_partitioned`] pinning the
/// [`BboxIou`] kernel.
///
/// # Errors
///
/// Propagates [`EvalError`] from the underlying evaluation pass.
pub fn optimal_lrp_bbox_partitioned(
    gt: &CocoDataset,
    dt: &CocoDetections,
    params: LrpParams<'_>,
    parity_mode: ParityMode,
    image_filters: &[HashSet<usize>],
) -> Result<Vec<LrpReport>, EvalError> {
    optimal_lrp_with_partitioned(
        gt,
        dt,
        &BboxIou,
        LrpKernelMarker::Bbox,
        params,
        parity_mode,
        image_filters,
    )
}

/// Partitioned-LRP entry point for the segm (mask) kernel.
///
/// # Errors
///
/// Propagates [`EvalError`] from the underlying evaluation pass.
pub fn optimal_lrp_segm_partitioned(
    gt: &CocoDataset,
    dt: &CocoDetections,
    params: LrpParams<'_>,
    parity_mode: ParityMode,
    image_filters: &[HashSet<usize>],
) -> Result<Vec<LrpReport>, EvalError> {
    optimal_lrp_with_partitioned(
        gt,
        dt,
        &SegmIou,
        LrpKernelMarker::Segm,
        params,
        parity_mode,
        image_filters,
    )
}

/// Partitioned-LRP entry point for the boundary-segm kernel.
///
/// # Errors
///
/// Propagates [`EvalError`] from the underlying evaluation pass.
pub fn optimal_lrp_boundary_partitioned(
    gt: &CocoDataset,
    dt: &CocoDetections,
    params: LrpParams<'_>,
    parity_mode: ParityMode,
    dilation_ratio: f64,
    image_filters: &[HashSet<usize>],
) -> Result<Vec<LrpReport>, EvalError> {
    let kernel = BoundaryIou { dilation_ratio };
    optimal_lrp_with_partitioned(
        gt,
        dt,
        &kernel,
        LrpKernelMarker::Boundary,
        params,
        parity_mode,
        image_filters,
    )
}

/// Partitioned-LRP entry point for the keypoints (OKS) kernel.
///
/// # Errors
///
/// Propagates [`EvalError`] from the underlying evaluation pass.
pub fn optimal_lrp_keypoints_partitioned(
    gt: &CocoDataset,
    dt: &CocoDetections,
    params: LrpParams<'_>,
    parity_mode: ParityMode,
    sigmas: HashMap<i64, Vec<f64>>,
    image_filters: &[HashSet<usize>],
) -> Result<Vec<LrpReport>, EvalError> {
    let kernel = OksSimilarity::new(sigmas);
    optimal_lrp_with_partitioned(
        gt,
        dt,
        &kernel,
        LrpKernelMarker::Keypoints,
        params,
        parity_mode,
        image_filters,
    )
}

fn validate_params(params: &LrpParams<'_>) -> Result<(), EvalError> {
    if params.tau_grid.is_empty() {
        return Err(EvalError::InvalidConfig {
            detail: "lrp: tau_grid must contain at least one value".into(),
        });
    }
    if !params.tp_threshold.is_finite() {
        return Err(EvalError::InvalidConfig {
            detail: format!(
                "lrp: tp_threshold must be finite; got {}",
                params.tp_threshold
            ),
        });
    }
    if !(0.0..=1.0).contains(&params.tp_threshold) {
        return Err(EvalError::InvalidConfig {
            detail: format!(
                "lrp: tp_threshold must lie in [0.0, 1.0]; got {}",
                params.tp_threshold
            ),
        });
    }
    Ok(())
}

fn build_category_id_lookup(gt: &CocoDataset, use_cats: bool) -> HashMap<usize, i64> {
    let mut out = HashMap::new();
    if use_cats {
        let mut cats: Vec<_> = gt.categories().iter().map(|c| c.id).collect();
        cats.sort_unstable_by_key(|c| c.0);
        for (idx, id) in cats.into_iter().enumerate() {
            out.insert(idx, id.0);
        }
    } else {
        // L4 collapse: single bucket carries every category.
        out.insert(0, COLLAPSED_CATEGORY_SENTINEL);
    }
    out
}

/// Aggregate per-class entries into the headline numbers.
///
/// Per ADR-0043:
///
/// - `olrp` mean is over classes with `olrp.is_some()` (positive GTs
///   exist). All-FN classes contribute (their `olrp = 1.0`).
/// - `olrp_loc`/`olrp_fp` means are over classes with TPs at the
///   optimal tau (where the components are defined).
/// - `olrp_fn` mean uses the same denominator as `olrp` — all-FN
///   classes contribute their `fn_rate = 1.0`.
fn aggregate(per_class: &[LrpPerClass]) -> (f64, f64, f64, f64, u32) {
    let mut olrp_sum = 0.0_f64;
    let mut olrp_n = 0_u64;
    let mut loc_sum = 0.0_f64;
    let mut loc_n = 0_u64;
    let mut fp_sum = 0.0_f64;
    let mut fp_n = 0_u64;
    let mut fn_sum = 0.0_f64;
    let mut fn_n = 0_u64;
    let mut empty: u32 = 0;

    for entry in per_class {
        if let Some(o) = entry.olrp {
            olrp_sum += o;
            olrp_n += 1;
        } else {
            empty = empty.saturating_add(1);
        }
        if let Some(v) = entry.olrp_loc {
            loc_sum += v;
            loc_n += 1;
        }
        if let Some(v) = entry.olrp_fp {
            fp_sum += v;
            fp_n += 1;
        }
        if let Some(v) = entry.olrp_fn {
            fn_sum += v;
            fn_n += 1;
        }
    }

    let mean = |s: f64, n: u64| if n == 0 { 0.0 } else { s / (n as f64) };
    (
        mean(olrp_sum, olrp_n),
        mean(loc_sum, loc_n),
        mean(fp_sum, fp_n),
        mean(fn_sum, fn_n),
        empty,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{
        Bbox, CategoryMeta, CocoAnnotation, CocoDataset, CocoDetection, CocoDetections, ImageMeta,
    };

    fn cat(id: i64) -> CategoryMeta {
        CategoryMeta {
            id: crate::dataset::CategoryId(id),
            name: format!("cat_{id}"),
            supercategory: None,
        }
    }

    fn img(id: i64) -> ImageMeta {
        ImageMeta {
            id: crate::dataset::ImageId(id),
            width: 100,
            height: 100,
            file_name: None,
        }
    }

    fn gt_ann(id: i64, image_id: i64, category_id: i64, bbox: Bbox) -> CocoAnnotation {
        CocoAnnotation {
            id: crate::dataset::AnnId(id),
            image_id: crate::dataset::ImageId(image_id),
            category_id: crate::dataset::CategoryId(category_id),
            area: bbox.w * bbox.h,
            is_crowd: false,
            ignore_flag: None,
            bbox,
            segmentation: None,
            keypoints: None,
            num_keypoints: None,
        }
    }

    fn dt_ann(id: i64, image_id: i64, category_id: i64, bbox: Bbox, score: f64) -> CocoDetection {
        CocoDetection {
            id: crate::dataset::AnnId(id),
            image_id: crate::dataset::ImageId(image_id),
            category_id: crate::dataset::CategoryId(category_id),
            score,
            bbox,
            area: bbox.w * bbox.h,
            segmentation: None,
            keypoints: None,
            num_keypoints: None,
        }
    }

    fn bbox(x: f64, y: f64, w: f64, h: f64) -> Bbox {
        Bbox { x, y, w, h }
    }

    fn build_perfect_dataset() -> (CocoDataset, CocoDetections) {
        let gt = CocoDataset::from_parts(
            vec![img(1)],
            vec![gt_ann(1, 1, 1, bbox(0.0, 0.0, 10.0, 10.0))],
            vec![cat(1)],
        )
        .expect("gt build");
        let dt =
            CocoDetections::from_records(vec![dt_ann(1, 1, 1, bbox(0.0, 0.0, 10.0, 10.0), 0.9)]);
        (gt, dt)
    }

    fn build_all_fp_dataset() -> (CocoDataset, CocoDetections) {
        // GT and DT do not overlap.
        let gt = CocoDataset::from_parts(
            vec![img(1)],
            vec![gt_ann(1, 1, 1, bbox(0.0, 0.0, 10.0, 10.0))],
            vec![cat(1)],
        )
        .expect("gt build");
        let dt =
            CocoDetections::from_records(vec![dt_ann(1, 1, 1, bbox(50.0, 50.0, 10.0, 10.0), 0.9)]);
        (gt, dt)
    }

    fn default_params<'a>(
        iou_thresholds: &'a [f64],
        area_ranges: &'a [crate::evaluate::AreaRange],
        tau_grid: &'a [f64],
    ) -> LrpParams<'a> {
        LrpParams {
            tp_threshold: 0.5,
            tau_grid,
            max_dets_per_image: 100,
            use_cats: true,
            iou_thresholds,
            area_ranges,
        }
    }

    #[test]
    fn perfect_match_olrp_zero() {
        let (gt, dt) = build_perfect_dataset();
        let iou_thr = [0.5];
        let area = crate::evaluate::AreaRange::coco_default();
        let tau_grid = default_tau_grid();
        let params = default_params(&iou_thr, &area, tau_grid);
        let report = optimal_lrp_bbox(&gt, &dt, params, ParityMode::Corrected).expect("eval");
        assert!(report.olrp.abs() < 1e-9, "olrp = {}", report.olrp);
        assert!(report.olrp_loc.abs() < 1e-9);
        assert!(report.olrp_fp.abs() < 1e-9);
        assert!(report.olrp_fn.abs() < 1e-9);
        assert_eq!(report.per_class.len(), 1);
        let cls = report.per_class[0];
        assert_eq!(cls.category_id, 1);
        assert_eq!(cls.olrp, Some(0.0));
        assert!(cls.tau.is_some());
    }

    #[test]
    fn all_fp_class_olrp_one() {
        let (gt, dt) = build_all_fp_dataset();
        let iou_thr = [0.5];
        let area = crate::evaluate::AreaRange::coco_default();
        let tau_grid = default_tau_grid();
        let params = default_params(&iou_thr, &area, tau_grid);
        let report = optimal_lrp_bbox(&gt, &dt, params, ParityMode::Corrected).expect("eval");
        assert!((report.olrp - 1.0).abs() < 1e-9, "olrp = {}", report.olrp);
        // All-FN class: no TPs at any tau. Headline olrp_fn = 1.0.
        assert!((report.olrp_fn - 1.0).abs() < 1e-9);
        let cls = report.per_class[0];
        assert_eq!(cls.olrp, Some(1.0));
        assert!(cls.tau.is_none());
        assert_eq!(cls.olrp_loc, None);
    }

    #[test]
    fn partitioned_overall_matches_unpartitioned() {
        // ADR-0046 load-bearing parity claim: index 0 of the
        // partitioned reports vector must be bit-identical to the
        // un-partitioned LRP report (the matching pass runs once
        // and the slice filter is identity over the empty-set filter).
        let (gt, dt) = build_perfect_dataset();
        let iou_thr = [0.5];
        let area = crate::evaluate::AreaRange::coco_default();
        let tau_grid = default_tau_grid();
        let params = default_params(&iou_thr, &area, tau_grid);
        let baseline =
            optimal_lrp_bbox(&gt, &dt, params, ParityMode::Corrected).expect("baseline eval");
        let part = optimal_lrp_bbox_partitioned(
            &gt,
            &dt,
            params,
            ParityMode::Corrected,
            &[HashSet::from([0])],
        )
        .expect("partitioned eval");
        assert_eq!(part.len(), 2);
        // Overall must match un-partitioned bit-identically.
        assert_eq!(part[0].olrp, baseline.olrp);
        assert_eq!(part[0].olrp_loc, baseline.olrp_loc);
        assert_eq!(part[0].olrp_fp, baseline.olrp_fp);
        assert_eq!(part[0].olrp_fn, baseline.olrp_fn);
        assert_eq!(part[0].per_class.len(), baseline.per_class.len());
        for (p, b) in part[0].per_class.iter().zip(baseline.per_class.iter()) {
            assert_eq!(p.olrp, b.olrp);
            assert_eq!(p.olrp_loc, b.olrp_loc);
        }
        // With a filter selecting the only image, the slice report
        // should also match overall.
        assert_eq!(part[1].olrp, baseline.olrp);
    }

    #[test]
    fn partitioned_empty_filter_yields_empty_class() {
        // No images in the filter → every class is "no positive GTs"
        // → olrp/loc/fp/fn = None for each per-class entry.
        let (gt, dt) = build_perfect_dataset();
        let iou_thr = [0.5];
        let area = crate::evaluate::AreaRange::coco_default();
        let tau_grid = default_tau_grid();
        let params = default_params(&iou_thr, &area, tau_grid);
        let part = optimal_lrp_bbox_partitioned(
            &gt,
            &dt,
            params,
            ParityMode::Corrected,
            &[HashSet::new()],
        )
        .expect("partitioned eval");
        assert_eq!(part.len(), 2);
        // The empty-filter slice carries no positive GTs (the only
        // image is filtered out), so the per-class entry is the
        // "no positive GTs" shape: every value is None.
        let cls = part[1].per_class[0];
        assert_eq!(cls.olrp, None);
        assert_eq!(cls.olrp_loc, None);
        assert_eq!(cls.olrp_fp, None);
        assert_eq!(cls.olrp_fn, None);
        assert_eq!(part[1].n_empty_classes, 1);
    }

    #[test]
    fn tau_search_argmin_tie_picks_larger_tau() {
        // One perfectly matched detection at score 0.5. The LRP
        // value is 0 across the grid prefix [0.00..0.50] inclusive
        // and 1.0 past that. Tie-break per ADR-0043: larger tau
        // wins, so tau = 0.50.
        let gt = CocoDataset::from_parts(
            vec![img(1)],
            vec![gt_ann(1, 1, 1, bbox(0.0, 0.0, 10.0, 10.0))],
            vec![cat(1)],
        )
        .expect("gt build");
        let dt =
            CocoDetections::from_records(vec![dt_ann(1, 1, 1, bbox(0.0, 0.0, 10.0, 10.0), 0.5)]);
        let iou_thr = [0.5];
        let area = crate::evaluate::AreaRange::coco_default();
        let tau_grid = default_tau_grid();
        let params = default_params(&iou_thr, &area, tau_grid);
        let report = optimal_lrp_bbox(&gt, &dt, params, ParityMode::Corrected).expect("eval");
        let cls = report.per_class[0];
        assert_eq!(cls.olrp, Some(0.0));
        // 101-point grid starts at 0.0; index 50 = tau=0.50.
        let tau = cls.tau.expect("tau set on TP class");
        assert!(
            (tau - 0.50).abs() < 1e-9,
            "argmin-tie should pick larger tau on ties; got {tau}"
        );
    }
}
