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

use std::collections::HashSet;

use rayon::prelude::*;
use vernier_partial::RankId;

use crate::distributed::{encode, semantic_expectation, EncodeInput, SemanticMergeAccumulator};
use crate::error::{ImageId, SemanticError};
use crate::kernel;
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
    /// Image ids passed to [`Self::update`]. Used by the partition-
    /// disjointness check at merge time (ADR-0032). Memory cost is
    /// 8 bytes per image; negligible at any realistic scale.
    seen_images: HashSet<i64>,
    /// Optional rank identifier carried in the partial wire header.
    /// `None` for single-rank flows; `Some(_)` is required for
    /// strict-mode merge across ranks.
    rank_id: Option<RankId>,
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
            seen_images: HashSet::new(),
            rank_id: None,
        }
    }

    /// Set the rank identifier carried in the partial wire header.
    /// Required (non-`None`) for strict-mode cross-rank merge per
    /// ADR-0032. Calling this after the first
    /// [`update`](Self::update) is a programming error — the rank
    /// id is a construction-time property.
    pub fn with_rank(mut self, rank_id: RankId) -> Result<Self, SemanticError> {
        if self.n_images > 0 {
            return Err(SemanticError::Partial(
                vernier_partial::PartialError::Format {
                    kind: vernier_partial::PartialFormatErrorKind::RkyvDecode {
                        detail: "with_rank must be called before the first update()".to_string(),
                    },
                },
            ));
        }
        self.rank_id = Some(rank_id);
        Ok(self)
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
    /// Generic over [`kernel::ClassId`] (`u8` / `u16` / `u32`) per
    /// ADR-0037: the streaming path can accept native-width PNG-decoded
    /// buffers without a 4× upcast at the boundary.
    ///
    /// Returns [`SemanticError::ShapeMismatch`] if the GT and DT
    /// slices have different lengths (the streaming evaluator is
    /// the load-bearing place for this check; the kernel itself
    /// `debug_assert!`s but does not return).
    pub fn update<T: kernel::ClassId>(
        &mut self,
        image_id: ImageId,
        gt: &[T],
        dt: &[T],
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
        self.seen_images.insert(image_id);
        Ok(())
    }

    /// Parallel sibling of [`Self::update`] over a slice of per-image
    /// `(image_id, gt, dt)` triples (ADR-0047). Each rayon worker
    /// accumulates a thread-local confusion matrix; the u64-additive
    /// reduction is order-independent, so the result is bit-equal to
    /// the sequential path across every thread count.
    ///
    /// Caller `install`s a `rayon::ThreadPool` around the call; no
    /// pool is built here. Empty slice is a no-op.
    ///
    /// # Errors
    /// [`SemanticError::ShapeMismatch`] from the first length-mismatched
    /// `(gt, dt)` pair (par_iter short-circuits).
    pub fn update_parsed_parallel<T: kernel::ClassId + Send + Sync>(
        &mut self,
        items: &[(ImageId, &[T], &[T])],
    ) -> Result<(), SemanticError> {
        if items.is_empty() {
            return Ok(());
        }
        // Validate lengths up-front in parallel; the per-thread fold
        // assumes `gt.len() == dt.len()` and otherwise we'd debug-assert
        // in the kernel. par_iter short-circuits on the first error.
        items
            .par_iter()
            .try_for_each(|(image_id, gt, dt)| -> Result<(), SemanticError> {
                if gt.len() != dt.len() {
                    return Err(SemanticError::ShapeMismatch {
                        image_id: *image_id,
                        gt_shape: (1, gt.len() as u32),
                        dt_shape: (1, dt.len() as u32),
                    });
                }
                Ok(())
            })?;

        let n_classes = self.confusion.n_classes();
        let ignore_label = self.ignore_label;

        // Tree-reduction: each task starts from a per-thread zero
        // matrix, folds in its image, and reduces pairwise. The fold
        // is u64-additive, so the reduction order is irrelevant —
        // strict-mode bit-equality holds for every thread count.
        let summed = items
            .par_iter()
            .fold(
                || ConfusionMatrix::zeros(n_classes),
                |mut acc, (_, gt, dt)| {
                    accumulate_confusion(gt, dt, ignore_label, &mut acc);
                    acc
                },
            )
            .reduce(
                || ConfusionMatrix::zeros(n_classes),
                |mut a, b| {
                    a.add_assign_unchecked(&b);
                    a
                },
            );

        self.confusion.add_assign_unchecked(&summed);
        self.n_images += items.len();
        for (image_id, _, _) in items {
            self.seen_images.insert(*image_id);
        }
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

    fn encode_input(&self) -> EncodeInput<'_> {
        EncodeInput {
            confusion: &self.confusion,
            ignore_label: self.ignore_label,
            parity_mode: self.parity_mode,
            rank_id: self.rank_id,
            n_images: self.n_images as u32,
            seen_images: &self.seen_images,
        }
    }

    /// Serialize the current evaluator state to an opaque byte blob
    /// (ADR-0032). The evaluator stays usable for further `update`
    /// calls; for the consuming form use [`Self::finalize_to_partial`].
    pub fn snapshot_to_partial(&self) -> Result<Vec<u8>, SemanticError> {
        Ok(encode(&self.encode_input())?)
    }

    /// Consuming variant of [`Self::snapshot_to_partial`] — the rank-
    /// local final state in a distributed-eval gather (ADR-0032).
    pub fn finalize_to_partial(self) -> Result<Vec<u8>, SemanticError> {
        Ok(encode(&self.encode_input())?)
    }

    /// Construct an evaluator equivalent to a batch run over the
    /// union of all partials' submitted images (ADR-0032).
    ///
    /// All partials must share `n_classes`, `ignore_label`,
    /// `parity_mode`. In strict mode every partial must declare a
    /// distinct `rank_id`. Image-id sets across partials must be
    /// disjoint. Confusion matrices sum element-wise — strict-mode
    /// merge is unconditionally bit-equal to a batch run over the
    /// union (no `(score, rank_id, local_position)` tiebreak needed;
    /// semantic doesn't fold detections by score).
    pub fn from_partials(
        n_classes: u32,
        ignore_label: Option<u32>,
        parity_mode: ParityMode,
        partials: &[&[u8]],
    ) -> Result<Self, SemanticError> {
        let mut ev = Self::new(n_classes, ignore_label, parity_mode);
        let strict = parity_mode == ParityMode::Strict;
        let mut acc = SemanticMergeAccumulator::new(n_classes, strict);
        let exp = semantic_expectation(n_classes, ignore_label, parity_mode);
        for bytes in partials {
            vernier_partial::with_validated_envelope(bytes, &exp, |view| acc.ingest(&view))?;
        }
        ev.confusion = acc.confusion;
        ev.n_images = acc.n_images as usize;
        ev.seen_images = acc.base.image_ids().collect();
        Ok(ev)
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
        ev.update(1, &[0u32, 1, 2], &[0u32, 1, 2])?;
        ev.update(2, &[0u32, 0, 1, 2], &[0u32, 0, 1, 2])?;
        let stream_summary = ev.finalize();

        let mut batch_cm = ConfusionMatrix::zeros(3);
        accumulate_confusion(&[0u32, 1, 2], &[0u32, 1, 2], None, &mut batch_cm);
        accumulate_confusion(&[0u32, 0, 1, 2], &[0u32, 0, 1, 2], None, &mut batch_cm);
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
        ev.update(1, &[0u32, 1, 0, 1], &[0u32, 1, 1, 1])?;
        let snap = ev.snapshot();
        let fin = ev.finalize();
        assert_eq!(snap.miou.to_bits(), fin.miou.to_bits());
        Ok(())
    }

    #[test]
    fn snapshot_does_not_consume_evaluator() -> Result<(), SemanticError> {
        let mut ev = StreamingSemanticEvaluator::new(2, None, ParityMode::Corrected);
        ev.update(1, &[0u32], &[0u32])?;
        let _ = ev.snapshot();
        // Evaluator is still usable.
        ev.update(2, &[1u32], &[1u32])?;
        assert_eq!(ev.n_images(), 2);
        Ok(())
    }

    #[test]
    fn shape_mismatch_returns_typed_error() {
        let mut ev = StreamingSemanticEvaluator::new(2, None, ParityMode::Corrected);
        let err = ev.update(7, &[0u32, 1], &[0u32]).unwrap_err();
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
        ev.update(1, &[0u32, 255, 1], &[0u32, 99, 1])?;
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

    #[test]
    fn parallel_update_bit_equals_sequential() -> Result<(), SemanticError> {
        // u64-additive: parallel and sequential folds over the same
        // batch must produce identical confusion matrices.
        let pairs: Vec<(ImageId, Vec<u32>, Vec<u32>)> = vec![
            (1, vec![0u32, 1, 2, 0], vec![0u32, 1, 2, 1]),
            (2, vec![1u32, 1, 0, 2, 2], vec![1u32, 0, 0, 2, 1]),
            (3, vec![0u32, 0, 0, 0], vec![0u32, 1, 0, 0]),
            (4, vec![2u32, 2, 2, 2], vec![2u32, 2, 2, 0]),
        ];

        let mut seq = StreamingSemanticEvaluator::new(3, None, ParityMode::Strict);
        for (iid, g, d) in &pairs {
            seq.update(*iid, g, d)?;
        }

        let mut par = StreamingSemanticEvaluator::new(3, None, ParityMode::Strict);
        let items: Vec<(ImageId, &[u32], &[u32])> = pairs
            .iter()
            .map(|(iid, g, d)| (*iid, g.as_slice(), d.as_slice()))
            .collect();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("pool");
        pool.install(|| par.update_parsed_parallel(&items))?;

        assert_eq!(seq.confusion().counts(), par.confusion().counts());
        assert_eq!(seq.n_images(), par.n_images());
        Ok(())
    }

    #[test]
    fn parallel_update_propagates_shape_mismatch() {
        let mut par = StreamingSemanticEvaluator::new(3, None, ParityMode::Strict);
        let g = vec![0u32, 1];
        let d = vec![0u32];
        let items: Vec<(ImageId, &[u32], &[u32])> = vec![(7, g.as_slice(), d.as_slice())];
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .expect("pool");
        let err = pool
            .install(|| par.update_parsed_parallel(&items))
            .unwrap_err();
        assert!(matches!(
            err,
            SemanticError::ShapeMismatch { image_id: 7, .. }
        ));
        // Failed validation does not mutate state.
        assert_eq!(par.n_images(), 0);
    }

    #[test]
    fn parallel_update_empty_is_noop() -> Result<(), SemanticError> {
        let mut par = StreamingSemanticEvaluator::new(3, None, ParityMode::Strict);
        let items: Vec<(ImageId, &[u32], &[u32])> = Vec::new();
        par.update_parsed_parallel(&items)?;
        assert_eq!(par.n_images(), 0);
        Ok(())
    }
}
