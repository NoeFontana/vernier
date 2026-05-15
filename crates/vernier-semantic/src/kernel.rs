//! Per-image confusion-matrix accumulation kernel.
//!
//! The simplest kernel in vernier (per ADR-0028 §"Algorithm"): one
//! pass over flattened `(H, W)` GT and DT label maps, indexing a
//! `(n_classes, n_classes)` matrix of `u64` cell counts. No matching
//! loop, no IoU computation, no per-detection scoring — this isn't
//! the AP fold.
//!
//! Confusion matrices are additively aggregable across images, which
//! makes streaming evaluation a thin wrapper (ADR-0013 reuse): the
//! per-image accumulator folds straight into a global matrix.
//! [`crate::summarize::summarize`] derives the seven headline metrics
//! from the global matrix at finalize time.
//!
//! **Numerical layout.** Cell counts are `u64`. Worst case fits
//! comfortably (Cityscapes train set: 5000 images × 2048×1024 ≈
//! 1.05e10 pixels per class; full ImageNet-style scale stays under
//! `u64::MAX = 1.8e19` with seven orders of magnitude of headroom).
//! Per-class IoU and the four headline scalars compute in `f64`
//! end-to-end at finalize time per ADR-0008. No SIMD on the v1 path
//! per ADR-0028 §"Numerical layout" — the kernel is integer-bound
//! and dominated by memory bandwidth, not arithmetic throughput.
//!
//! **Input contract.** GT and DT slices must have equal length and
//! both must be flattened `(H, W)` row-major. Pixel values must be in
//! `[0, n_classes)` ∪ `{ignore_label}`. Out-of-range / shape-mismatch
//! validation lives upstream at the dataset-constructor boundary
//! (PR-B5's `SemanticPredictions::from_arrays`); the kernel uses
//! `debug_assert!` to catch contract violations in test builds and
//! relies on the upstream validator in release.

use crate::error::ImageId;
use crate::error::SemanticError;

/// Per-image / per-dataset confusion matrix.
///
/// Row index is the GT class (`g`); column index is the prediction
/// class (`d`). `counts[g, d]` is the number of pixels labeled `g` in
/// GT and predicted `d` by DT, accumulated over every image folded
/// into this matrix.
///
/// Storage is a flat `Vec<u64>` of length `n_classes * n_classes`,
/// row-major. The flat shape avoids an `ndarray` dep (the matrix is
/// small — 150² = 22500 cells = 180 KB at the largest realistic
/// configuration, ADE20K) and keeps the FFI conversion trivial: the
/// numpy `(N, N)` view points straight at this buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfusionMatrix {
    n_classes: u32,
    counts: Vec<u64>,
}

impl ConfusionMatrix {
    /// Construct an `(n_classes, n_classes)` zero matrix.
    pub fn zeros(n_classes: u32) -> Self {
        let n = n_classes as usize;
        Self {
            n_classes,
            counts: vec![0; n * n],
        }
    }

    /// Number of evaluation classes.
    pub const fn n_classes(&self) -> u32 {
        self.n_classes
    }

    /// Read-only access to the row-major `(n_classes, n_classes)`
    /// flat buffer. The FFI layer borrows this as a numpy `(N, N)`
    /// view.
    pub fn counts(&self) -> &[u64] {
        &self.counts
    }

    /// Mutable access to the flat buffer for in-place folding (e.g.,
    /// element-wise add of a per-image matrix into a global one).
    pub(crate) fn counts_mut(&mut self) -> &mut [u64] {
        &mut self.counts
    }

    /// `counts[g, d]` lookup. Bounds-checked in debug; tight loops
    /// should index the flat slice directly via [`counts`].
    ///
    /// [`counts`]: Self::counts
    pub fn get(&self, g: u32, d: u32) -> u64 {
        let n = self.n_classes as usize;
        debug_assert!((g as usize) < n && (d as usize) < n);
        self.counts[g as usize * n + d as usize]
    }

    /// Element-wise addition of `other` into `self` **without** the
    /// typed shape-mismatch error path, as a fast path for the C3
    /// partitioned summarize loop (ADR-0046). Both matrices must
    /// share the same `n_classes` by construction — caller invariant,
    /// debug-asserted.
    ///
    /// Callers that need the typed `SemanticError::ShapeMismatch`
    /// error path should stay on [`Self::add_assign`]; this entry
    /// point is for the internal partition orchestrator where the
    /// per-image matrices are constructed against the same
    /// `n_classes` as the target.
    pub fn add_assign_unchecked(&mut self, other: &Self) {
        debug_assert_eq!(
            self.n_classes, other.n_classes,
            "add_assign_unchecked contract: matrices must share n_classes",
        );
        for (lhs, rhs) in self.counts.iter_mut().zip(other.counts.iter()) {
            *lhs += rhs;
        }
    }

    /// Element-wise addition of `other` into `self`. Both matrices
    /// must share the same `n_classes`. Used by the streaming
    /// evaluator (ADR-0013) and by the multi-image accumulator.
    pub fn add_assign(&mut self, other: &Self) -> Result<(), SemanticError> {
        if self.n_classes != other.n_classes {
            // The streaming evaluator constructs both matrices with
            // the same n_classes by construction, so this branch is
            // a sanity check rather than an expected error path.
            // Surface it as ShapeMismatch with a synthetic image_id
            // so the caller has a typed variant to handle.
            return Err(SemanticError::ShapeMismatch {
                image_id: ImageId::default(),
                gt_shape: (self.n_classes, self.n_classes),
                dt_shape: (other.n_classes, other.n_classes),
            });
        }
        for (lhs, rhs) in self.counts.iter_mut().zip(other.counts.iter()) {
            *lhs += rhs;
        }
        Ok(())
    }
}

/// Class-id types accepted by [`accumulate_confusion`] (ADR-0037).
///
/// Implementations exist for `u8`, `u16`, and `u32` — the natural
/// dtypes for class-id label maps (PNG-decoded inputs are typically
/// `u8`; sparse-class workloads like Cityscapes 30+ stay in `u16`;
/// `u32` remains the canonical wire format for the FFI array path
/// and for the partials format).
///
/// Class ids widen to `u32` inside the kernel for comparison against
/// `ignore_label` and indexing the confusion matrix; the cast is free
/// on smaller types and lets the kernel walk a `(H, W)` `u8` buffer
/// without a 4× upcast at the FFI boundary.
pub trait ClassId: Copy {
    /// Widen the class id to `u32` for ignore-label comparison and
    /// confusion-matrix indexing.
    fn as_u32(self) -> u32;
}

impl ClassId for u8 {
    #[inline(always)]
    fn as_u32(self) -> u32 {
        u32::from(self)
    }
}

impl ClassId for u16 {
    #[inline(always)]
    fn as_u32(self) -> u32 {
        u32::from(self)
    }
}

impl ClassId for u32 {
    #[inline(always)]
    fn as_u32(self) -> u32 {
        self
    }
}

/// Fold one image's `(gt, dt)` label-map pair into a confusion matrix.
///
/// `gt` and `dt` are flattened row-major `(H, W)` slices of equal
/// length, generic over [`ClassId`] (`u8` / `u16` / `u32`).
/// `n_classes` is the evaluation class count (taken from
/// `confusion.n_classes()`). `ignore_label`, when present, masks out
/// pixels with `gt == ignore_label` from the histogram (per quirk
/// **AJ2**); the prediction value at those pixels is ignored.
///
/// Pixels where `gt` is in `[0, n_classes)` and `dt` is in
/// `[0, n_classes)` increment `confusion.counts[gt, dt]` by 1.
///
/// Pixels where `dt >= n_classes` are silently skipped on this hot
/// path. The corrected-default behavior under
/// [`ParityMode::Corrected`] is to reject out-of-range predictions at
/// the dataset-constructor boundary (quirk **AI4**); under
/// [`ParityMode::Strict`] mmsegmentation truncates over-class entries
/// (matched by the silent skip here). Either way the kernel never
/// writes to an out-of-bounds confusion cell.
///
/// Pixels where `gt >= n_classes` and `gt != ignore_label` are a
/// **dataset-side validation failure** (the dataset constructor
/// guarantees GT in `[0, n_classes) ∪ {ignore_label}`) and are
/// `debug_assert!`-ed in test builds; in release they are silently
/// skipped to avoid an out-of-bounds write.
///
/// [`ParityMode::Corrected`]: crate::parity::ParityMode::Corrected
/// [`ParityMode::Strict`]: crate::parity::ParityMode::Strict
pub fn accumulate_confusion<T: ClassId>(
    gt: &[T],
    dt: &[T],
    ignore_label: Option<u32>,
    confusion: &mut ConfusionMatrix,
) {
    debug_assert_eq!(
        gt.len(),
        dt.len(),
        "kernel contract: gt and dt slices must share length",
    );
    let n = confusion.n_classes as usize;
    let counts = confusion.counts_mut();
    for (&g, &d) in gt.iter().zip(dt.iter()) {
        let g = g.as_u32();
        let d = d.as_u32();
        if Some(g) == ignore_label {
            continue;
        }
        let g = g as usize;
        let d = d as usize;
        // Branch #1: gt out of range. Dataset validator should have
        // caught this; silent-skip in release per the doc-comment
        // contract.
        if g >= n {
            debug_assert!(
                false,
                "kernel contract: gt class out of range: {g} >= n_classes={n} \
                 (validate at SemanticDataset::from_arrays / from_files)",
            );
            continue;
        }
        // Branch #2: dt out of range. Quirk AI4 silent-skip path
        // matching mmsegmentation's truncation behavior. The
        // corrected-default rejects upstream so this branch only
        // fires under ParityMode::Strict.
        if d >= n {
            continue;
        }
        counts[g * n + d] += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_constructs_n_by_n_matrix() {
        let cm = ConfusionMatrix::zeros(3);
        assert_eq!(cm.n_classes(), 3);
        assert_eq!(cm.counts().len(), 9);
        assert!(cm.counts().iter().all(|&c| c == 0));
    }

    #[test]
    fn perfect_diagonal_match() {
        // GT == DT pointwise; every pixel lands on the diagonal.
        let gt = [0u32, 1, 2, 0, 1, 2];
        let dt = [0u32, 1, 2, 0, 1, 2];
        let mut cm = ConfusionMatrix::zeros(3);
        accumulate_confusion(&gt, &dt, None, &mut cm);
        assert_eq!(cm.get(0, 0), 2);
        assert_eq!(cm.get(1, 1), 2);
        assert_eq!(cm.get(2, 2), 2);
        // Off-diagonal cells stay zero.
        for g in 0..3u32 {
            for d in 0..3u32 {
                if g != d {
                    assert_eq!(cm.get(g, d), 0, "off-diagonal ({g},{d}) should be 0");
                }
            }
        }
    }

    #[test]
    fn off_diagonal_pixel_lands_in_correct_cell() {
        // One pixel with gt=0, dt=1 → counts[0, 1] += 1.
        let gt = [0u32];
        let dt = [1u32];
        let mut cm = ConfusionMatrix::zeros(3);
        accumulate_confusion(&gt, &dt, None, &mut cm);
        assert_eq!(cm.get(0, 1), 1);
        assert_eq!(cm.get(0, 0), 0);
        assert_eq!(cm.get(1, 0), 0);
    }

    #[test]
    fn ignore_label_excludes_pixels_from_both_axes() {
        // Pixels where gt == ignore_label are dropped entirely (quirk
        // AJ2). The dt value at those pixels does not contribute to
        // any cell.
        let gt = [0u32, 255, 1, 255];
        let dt = [0u32, 0, 1, 2];
        let mut cm = ConfusionMatrix::zeros(3);
        accumulate_confusion(&gt, &dt, Some(255), &mut cm);
        // Only the two non-ignore pixels (gt=0,dt=0 and gt=1,dt=1)
        // contribute. The ignore pixels (255,0) and (255,2) drop.
        assert_eq!(cm.get(0, 0), 1);
        assert_eq!(cm.get(1, 1), 1);
        let total: u64 = cm.counts().iter().sum();
        assert_eq!(total, 2, "ignore pixels must not contribute to any cell");
    }

    #[test]
    fn out_of_range_dt_silent_skip() {
        // dt value >= n_classes is dropped silently (AI4 strict-MS
        // path). The kernel never writes out of bounds.
        let gt = [0u32, 1];
        let dt = [0u32, 99]; // 99 >= 3
        let mut cm = ConfusionMatrix::zeros(3);
        accumulate_confusion(&gt, &dt, None, &mut cm);
        assert_eq!(cm.get(0, 0), 1, "in-range pixel still counts");
        // Out-of-range dt does not increment any cell on row gt=1.
        for d in 0..3u32 {
            assert_eq!(cm.get(1, d), 0, "row 1 must be empty (dt was out of range)");
        }
    }

    #[test]
    fn ade20k_style_zero_ignore_label() {
        // ADE20K uses ignore_label=0 (quirk AJ5). Pixels with gt=0
        // are excluded; pixels with gt in [1, 150] are counted.
        let gt = [0u32, 0, 1, 2, 1];
        let dt = [99u32, 99, 1, 2, 0];
        let mut cm = ConfusionMatrix::zeros(150);
        accumulate_confusion(&gt, &dt, Some(0), &mut cm);
        // Only the gt=1,2 pixels contribute. dt=99 / dt=0 in those
        // positions are written verbatim (no special handling for dt
        // matching the ignore label per AI3 silent-drop).
        assert_eq!(cm.get(1, 1), 1);
        assert_eq!(cm.get(2, 2), 1);
        assert_eq!(cm.get(1, 0), 1);
        let total: u64 = cm.counts().iter().sum();
        assert_eq!(total, 3, "two ignore pixels excluded, three counted");
    }

    #[test]
    fn add_assign_folds_two_images() {
        // Streaming-style accumulation: two per-image confusion
        // matrices sum element-wise into a global one.
        let mut img1 = ConfusionMatrix::zeros(2);
        accumulate_confusion(&[0u32, 1], &[0u32, 1], None, &mut img1);

        let mut img2 = ConfusionMatrix::zeros(2);
        accumulate_confusion(&[0u32, 0, 1], &[0u32, 1, 1], None, &mut img2);

        let mut global = ConfusionMatrix::zeros(2);
        global.add_assign(&img1).unwrap();
        global.add_assign(&img2).unwrap();

        // img1: counts[0,0]=1, counts[1,1]=1
        // img2: counts[0,0]=1, counts[0,1]=1, counts[1,1]=1
        // sum:   counts[0,0]=2, counts[0,1]=1, counts[1,1]=2
        assert_eq!(global.get(0, 0), 2);
        assert_eq!(global.get(0, 1), 1);
        assert_eq!(global.get(1, 0), 0);
        assert_eq!(global.get(1, 1), 2);
    }

    #[test]
    fn add_assign_rejects_shape_mismatch() {
        let mut a = ConfusionMatrix::zeros(2);
        let b = ConfusionMatrix::zeros(3);
        let err = a.add_assign(&b).unwrap_err();
        assert!(matches!(err, SemanticError::ShapeMismatch { .. }));
    }

    #[test]
    fn add_assign_unchecked_equals_add_assign_on_matching_shape() {
        // ADR-0046 C3 partitioned summarize uses the unchecked path
        // for the hot per-slice fold; pin that it produces the same
        // matrix as the typed-error variant on matching shapes.
        let mut a_checked = ConfusionMatrix::zeros(3);
        accumulate_confusion(&[0u32, 1, 2], &[0u32, 1, 2], None, &mut a_checked);
        let mut a_unchecked = a_checked.clone();
        let mut b = ConfusionMatrix::zeros(3);
        accumulate_confusion(&[1u32, 1, 2], &[1u32, 0, 2], None, &mut b);

        let mut checked_only = a_checked.clone();
        checked_only.add_assign(&b).unwrap();
        a_unchecked.add_assign_unchecked(&b);
        assert_eq!(checked_only, a_unchecked);
    }

    #[test]
    fn u8_path_matches_u32_for_identical_values() {
        // ADR-0037: the u8 monomorphization produces a bit-identical
        // confusion matrix to the u32 path when the values fit. PNG
        // label maps are u8 by nature; this is the load-bearing
        // equivalence for the fused-decode FFI surface.
        let gt = [0u8, 1, 2, 0, 255, 1, 2];
        let dt = [0u8, 1, 0, 0, 7, 1, 2];
        let ignore = Some(255u32);
        let n = 3u32;

        let mut cm_u8 = ConfusionMatrix::zeros(n);
        accumulate_confusion(&gt, &dt, ignore, &mut cm_u8);

        let gt_u32: Vec<u32> = gt.iter().map(|&x| u32::from(x)).collect();
        let dt_u32: Vec<u32> = dt.iter().map(|&x| u32::from(x)).collect();
        let mut cm_u32 = ConfusionMatrix::zeros(n);
        accumulate_confusion(&gt_u32, &dt_u32, ignore, &mut cm_u32);

        assert_eq!(
            cm_u8, cm_u32,
            "u8 and u32 paths must produce identical matrices"
        );
    }

    #[test]
    fn u16_path_matches_u32_for_identical_values() {
        // Cityscapes 19-class evaluation with ignore_label=255 — u16
        // is the natural width when raw labels span 0..34 before the
        // trainId remap, and ignore_label=255 is too large to fit in
        // a generic-over-T parameter narrower than u16.
        let gt: Vec<u16> = vec![0, 1, 18, 255, 5, 255, 12, 0];
        let dt: Vec<u16> = vec![0, 1, 18, 7, 6, 0, 18, 999];
        let ignore = Some(255u32);
        let n = 19u32;

        let mut cm_u16 = ConfusionMatrix::zeros(n);
        accumulate_confusion(&gt, &dt, ignore, &mut cm_u16);

        let gt_u32: Vec<u32> = gt.iter().map(|&x| u32::from(x)).collect();
        let dt_u32: Vec<u32> = dt.iter().map(|&x| u32::from(x)).collect();
        let mut cm_u32 = ConfusionMatrix::zeros(n);
        accumulate_confusion(&gt_u32, &dt_u32, ignore, &mut cm_u32);

        assert_eq!(
            cm_u16, cm_u32,
            "u16 and u32 paths must produce identical matrices"
        );
    }

    #[test]
    fn deterministic_across_pixel_orderings() {
        // The histogram fold is order-independent (commutative
        // addition); the same pixel multiset produces the same
        // confusion matrix regardless of input order. Pinning this
        // pre-empts an "innocent" parallel rewrite that batches
        // pixels and changes ordering.
        let pixels: Vec<(u32, u32)> = vec![
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
            (1, 1),
            (0, 0),
            (2, 2),
            (2, 0),
        ];
        let mut shuffled = pixels.clone();
        shuffled.reverse();

        let unflatten = |pairs: &[(u32, u32)]| -> (Vec<u32>, Vec<u32>) {
            let gt = pairs.iter().map(|&(g, _)| g).collect();
            let dt = pairs.iter().map(|&(_, d)| d).collect();
            (gt, dt)
        };

        let (gt1, dt1) = unflatten(&pixels);
        let (gt2, dt2) = unflatten(&shuffled);

        let mut a = ConfusionMatrix::zeros(3);
        let mut b = ConfusionMatrix::zeros(3);
        accumulate_confusion(&gt1, &dt1, None, &mut a);
        accumulate_confusion(&gt2, &dt2, None, &mut b);
        assert_eq!(a, b, "histogram must be order-independent");
    }
}
