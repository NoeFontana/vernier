//! Streaming semantic-segmentation evaluator (ADR-0028 §"Streaming").
//!
//! Confusion matrices are additively aggregable across images, which
//! makes streaming evaluation a thin orchestration layer over
//! [`crate::kernel::accumulate_confusion`]: the per-instance
//! [`StreamingSemanticEvaluator`] holds a single
//! [`ConfusionMatrix`] and folds each `update(gt, dt)` call into it
//! via the same kernel the batch path uses.
//!
//! Unlike [`vernier_core::stream::StreamingEvaluator`] (which is
//! generic over the AP-fold `EvalKernel` and orchestrates a per-image
//! cell store), this evaluator is a flat `O(n_classes²)` accumulator.
//! There is no "running snapshot" / "fast snapshot" distinction (per
//! ADR-0013 §"Fast snapshot mode") — `snapshot()` is constant-time
//! relative to image count: it's a fold over the fixed-size
//! `(n_classes, n_classes)` matrix, not over a per-image cell store.
//!
//! API shape mirrors the [instance streaming evaluator's]
//! `new` / `update` / `snapshot` / `finalize` lifecycle so the
//! `vernier.semantic` Python surface aligns with `vernier.instance`.
//!
//! [instance streaming evaluator's]: vernier_core::stream::StreamingEvaluator

use crate::error::{ImageId, SemanticError};
use crate::kernel::{accumulate_confusion, ConfusionMatrix};
use crate::parity::ParityMode;
use crate::summarize::{summarize, SemanticSummary};

/// Streaming semantic-segmentation evaluator.
///
/// Construct via [`StreamingSemanticEvaluator::new`], feed per-image
/// `(gt, dt)` slices via [`update`](Self::update), read intermediate
/// state via [`snapshot`](Self::snapshot), and produce the final
/// [`SemanticSummary`] via [`finalize`](Self::finalize).
///
/// Concurrency: this type is not `Sync`; callers in async / threaded
/// contexts wrap it in a `Mutex` (the FFI layer does this for the
/// Python `BackgroundEvaluator` shape if and when it lands for
/// semantic).
#[derive(Debug, Clone)]
pub struct StreamingSemanticEvaluator {
    confusion: ConfusionMatrix,
    ignore_label: Option<u32>,
    parity_mode: ParityMode,
    n_images: usize,
}

impl StreamingSemanticEvaluator {
    /// Build a new evaluator. `n_classes` is the evaluation class
    /// count; `ignore_label`, when present, masks pixels with
    /// `gt == ignore_label` from the histogram (quirk **AJ2**).
    /// `parity_mode` selects the NaN-vs-0.0 disposition for
    /// zero-support per-class entries (quirk **AL2**).
    ///
    /// `n_classes == 0` is rejected at the dataset-constructor
    /// boundary (Python `Dataset.__post_init__`) so this constructor
    /// trusts its input. Callers building the streaming evaluator
    /// directly from Rust should validate `n_classes >= 1` upstream.
    pub fn new(n_classes: u32, ignore_label: Option<u32>, parity_mode: ParityMode) -> Self {
        Self {
            confusion: ConfusionMatrix::zeros(n_classes),
            ignore_label,
            parity_mode,
            n_images: 0,
        }
    }

    /// Number of `update` calls accepted so far. Useful for progress
    /// reporting and for confirming the streaming traversal saw every
    /// expected image.
    pub const fn n_images(&self) -> usize {
        self.n_images
    }

    /// Number of evaluation classes.
    pub const fn n_classes(&self) -> u32 {
        self.confusion.n_classes()
    }

    /// Borrow the in-progress confusion matrix. Useful for diagnostic
    /// inspection; callers wanting a snapshot of the metrics should
    /// use [`snapshot`](Self::snapshot) instead.
    pub const fn confusion(&self) -> &ConfusionMatrix {
        &self.confusion
    }

    /// Fold one image's `(gt, dt)` label-map pair into the running
    /// confusion matrix. `image_id` is recorded for error attribution
    /// only — the kernel does not key by image id.
    ///
    /// Returns [`SemanticError::ShapeMismatch`] if the GT and DT
    /// slices have different lengths (the streaming evaluator is
    /// the load-bearing place for this check; the kernel itself
    /// `debug_assert!`s but does not return).
    pub fn update(
        &mut self,
        image_id: ImageId,
        gt: &[u32],
        dt: &[u32],
    ) -> Result<(), SemanticError> {
        if gt.len() != dt.len() {
            // Surface as ShapeMismatch with synthetic (h, w) shapes.
            // The streaming API takes flat slices so we report length
            // pairs rather than 2-D shapes; downstream consumers can
            // still attribute via `image_id`.
            return Err(SemanticError::ShapeMismatch {
                image_id,
                gt_shape: (1, gt.len() as u32),
                dt_shape: (1, dt.len() as u32),
            });
        }
        accumulate_confusion(gt, dt, self.ignore_label, &mut self.confusion);
        self.n_images += 1;
        Ok(())
    }

    /// Compute the [`SemanticSummary`] from the current state without
    /// consuming the evaluator. Clones the in-progress confusion
    /// matrix; callers who don't need to keep updating after the
    /// snapshot should prefer [`finalize`](Self::finalize) which
    /// avoids the clone.
    pub fn snapshot(&self) -> SemanticSummary {
        summarize(self.confusion.clone(), self.parity_mode)
    }

    /// Consume the evaluator and produce the final
    /// [`SemanticSummary`]. Equivalent to [`snapshot`](Self::snapshot)
    /// but transfers ownership of the confusion matrix into the
    /// summary (no clone).
    pub fn finalize(self) -> SemanticSummary {
        summarize(self.confusion, self.parity_mode)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn streaming_finalize_equals_batch() -> Result<(), SemanticError> {
        // Two images, both perfect match. Streaming finalize must
        // bit-equal the batch summarize over the same images.
        let mut ev = StreamingSemanticEvaluator::new(3, None, ParityMode::Corrected);
        ev.update(1, &[0, 1, 2], &[0, 1, 2])?;
        ev.update(2, &[0, 0, 1, 2], &[0, 0, 1, 2])?;
        let stream_summary = ev.finalize();

        let mut batch_cm = ConfusionMatrix::zeros(3);
        accumulate_confusion(&[0, 1, 2], &[0, 1, 2], None, &mut batch_cm);
        accumulate_confusion(&[0, 0, 1, 2], &[0, 0, 1, 2], None, &mut batch_cm);
        let batch_summary = summarize(batch_cm, ParityMode::Corrected);

        assert_eq!(stream_summary.miou.to_bits(), batch_summary.miou.to_bits());
        assert_eq!(
            stream_summary.fwiou.to_bits(),
            batch_summary.fwiou.to_bits()
        );
        assert_eq!(
            stream_summary.pixel_accuracy.to_bits(),
            batch_summary.pixel_accuracy.to_bits(),
        );
        Ok(())
    }

    #[test]
    fn streaming_snapshot_equals_finalize_when_idempotent() -> Result<(), SemanticError> {
        let mut ev = StreamingSemanticEvaluator::new(2, None, ParityMode::Corrected);
        ev.update(1, &[0, 1, 0, 1], &[0, 1, 1, 1])?;
        let snap = ev.snapshot();
        let fin = ev.finalize();
        assert_eq!(snap.miou.to_bits(), fin.miou.to_bits());
        Ok(())
    }

    #[test]
    fn snapshot_does_not_consume_evaluator() -> Result<(), SemanticError> {
        let mut ev = StreamingSemanticEvaluator::new(2, None, ParityMode::Corrected);
        ev.update(1, &[0], &[0])?;
        let _ = ev.snapshot();
        // Evaluator is still usable.
        ev.update(2, &[1], &[1])?;
        assert_eq!(ev.n_images(), 2);
        Ok(())
    }

    #[test]
    fn shape_mismatch_returns_typed_error() {
        let mut ev = StreamingSemanticEvaluator::new(2, None, ParityMode::Corrected);
        let err = ev.update(7, &[0, 1], &[0]).unwrap_err();
        assert!(matches!(
            err,
            SemanticError::ShapeMismatch { image_id: 7, .. }
        ));
        // Failed update does not increment n_images.
        assert_eq!(ev.n_images(), 0);
    }

    #[test]
    fn ignore_label_propagates_through_streaming() -> Result<(), SemanticError> {
        let mut ev = StreamingSemanticEvaluator::new(2, Some(255), ParityMode::Corrected);
        ev.update(1, &[0, 255, 1], &[0, 99, 1])?;
        let summary = ev.finalize();
        // Two non-ignore pixels, both diagonal → mIoU = 1.0.
        assert!(approx_eq(summary.miou, 1.0, 0.0));
        assert_eq!(summary.confusion_matrix.counts().iter().sum::<u64>(), 2);
        Ok(())
    }

    #[test]
    fn empty_evaluator_returns_zeros_not_nan() {
        let ev = StreamingSemanticEvaluator::new(3, None, ParityMode::Corrected);
        let summary = ev.finalize();
        assert!(approx_eq(summary.miou, 0.0, 0.0));
        assert_eq!(summary.confusion_matrix.counts().iter().sum::<u64>(), 0);
    }

    #[test]
    fn n_classes_accessor_matches_construction() {
        let ev = StreamingSemanticEvaluator::new(19, Some(255), ParityMode::Strict);
        assert_eq!(ev.n_classes(), 19);
        assert_eq!(ev.n_images(), 0);
    }
}
