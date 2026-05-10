//! Streaming panoptic-quality evaluator (ADR-0032).
//!
//! Per-image PqStat folds are commutative + associative when summed
//! by integer counters (`n_tp`, `n_fp`, `n_fn`); the `sum_iou` field
//! is f64 and therefore *not* associative. This evaluator gives users
//! two paths via the `retain_per_image_deltas` constructor flag:
//!
//! - **Default (off)** — only `acc: HashMap<CategoryId, PqStat>` is
//!   kept. Cheap streaming + cheap merge, but corrected-mode merge
//!   stays within ADR-0004's 4-ULP envelope rather than bit-equaling
//!   batch.
//! - **`retain_per_image_deltas=true`** — additionally store the
//!   per-image `(image_id, per_category_map)` so `from_partials` can
//!   re-sort by `image_id` across ranks and re-sum, matching
//!   [`crate::summarize::evaluate`]'s sorted iteration. Strict-mode
//!   merge then **bit-equals** a batch run over the union.
//!
//! The mid-stream `snapshot()` / `finalize()` always summarizes from
//! the per-image-sorted fold (the `acc` field updated in image-id
//! order on `update`), so single-rank determinism does not depend on
//! the flag.

use std::collections::{HashMap, HashSet};

use vernier_partial::RankId;

use crate::attribute::{attribute_image, PqStat};
use crate::dataset::{CategoryId, CategoryMeta, ImageEntry, ImageId};
use crate::distributed::{encode, panoptic_expectation, EncodeInput, PanopticMergeAccumulator};
use crate::error::PanopticError;
use crate::kernel::pq_image_with_id;
use crate::parity::ParityMode;
use crate::summarize::{summarize_from_acc, PanopticSummary};

/// Streaming panoptic-quality evaluator. Mirrors
/// `vernier_core::stream::StreamingEvaluator`'s lifecycle for the
/// PQ paradigm: construct, [`update`](Self::update) per image,
/// [`snapshot`](Self::snapshot) (non-consuming) or
/// [`finalize`](Self::finalize) (consuming) to read the
/// [`PanopticSummary`].
#[derive(Debug, Clone)]
pub struct StreamingPanopticEvaluator {
    /// Per-category PqStat fold, updated as images stream in.
    acc: HashMap<CategoryId, PqStat>,
    /// Categories taxonomy (cross-rank invariant). Carried for the
    /// W4 things/stuff split at finalize time.
    categories: HashMap<CategoryId, CategoryMeta>,
    /// Optional per-image deltas — populated iff
    /// `retain_per_image_deltas=true` at construction. Holds one
    /// `(image_id, per_category_map)` entry per `update` call,
    /// kept in image-id order (the same order downstream merge
    /// will re-sort to for strict-mode bit-equality).
    per_image: Option<Vec<(ImageId, HashMap<CategoryId, PqStat>)>>,
    /// Image ids passed to [`Self::update`]. Used by the partition-
    /// disjointness check at merge time.
    seen_images: HashSet<ImageId>,
    parity_mode: ParityMode,
    things_stuff_split: bool,
    /// When `true`, [`Self::update`] also pushes onto `per_image`
    /// and emitted partials carry `per_image_deltas` for cross-
    /// rank strict-mode bit-equality merge.
    retain_per_image_deltas: bool,
    rank_id: Option<RankId>,
}

impl StreamingPanopticEvaluator {
    /// Build a new evaluator. `categories` is the (id → metadata)
    /// taxonomy; `parity_mode` selects strict-vs-corrected per
    /// quirks V3 / W6; `things_stuff_split` enables the per-bucket
    /// summary; `retain_per_image_deltas` enables strict-mode bit-
    /// equality across ranks at the cost of ~2× streaming memory.
    ///
    /// `retain_per_image_deltas=true` is only beneficial in strict
    /// mode: corrected-mode merge sums `per_category` directly and
    /// never consults the deltas, so the extra memory is wasted on
    /// the corrected path.
    pub fn new(
        categories: HashMap<CategoryId, CategoryMeta>,
        parity_mode: ParityMode,
        things_stuff_split: bool,
        retain_per_image_deltas: bool,
    ) -> Self {
        Self {
            acc: HashMap::new(),
            categories,
            per_image: if retain_per_image_deltas {
                Some(Vec::new())
            } else {
                None
            },
            seen_images: HashSet::new(),
            parity_mode,
            things_stuff_split,
            retain_per_image_deltas,
            rank_id: None,
        }
    }

    /// Set the rank identifier carried in the partial wire header
    /// (ADR-0032). Required for strict-mode cross-rank merge. Calling
    /// this after the first [`update`](Self::update) is a programming
    /// error.
    pub fn with_rank(mut self, rank_id: RankId) -> Result<Self, PanopticError> {
        if !self.seen_images.is_empty() {
            return Err(PanopticError::Partial(
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

    /// Number of `update` calls accepted so far.
    pub fn n_images(&self) -> usize {
        self.seen_images.len()
    }

    /// Number of categories in the taxonomy.
    pub fn n_categories(&self) -> usize {
        self.categories.len()
    }

    /// Borrow the categories taxonomy.
    pub const fn categories(&self) -> &HashMap<CategoryId, CategoryMeta> {
        &self.categories
    }

    /// Fold one image's GT/DT pair into the running PqStat
    /// accumulator. Routes through the existing `pq_image_with_id`
    /// + `attribute_image` kernels (untouched per ADR-0005).
    ///
    /// `image_id` must not have been passed to a previous `update`
    /// on this evaluator — the duplicate would produce a partial
    /// that's invalid for distributed merge (the partition rule
    /// fires across ranks, not within one).
    pub fn update(
        &mut self,
        image_id: ImageId,
        gt: &ImageEntry,
        dt: &ImageEntry,
    ) -> Result<(), PanopticError> {
        if !self.seen_images.insert(image_id) {
            return Err(PanopticError::DuplicateImageId { image_id });
        }
        let report = pq_image_with_id(image_id, gt, dt)?;
        let per_image = attribute_image(gt, dt, &report, self.parity_mode);

        // Fold into the per-category accumulator. The `acc` is the
        // cheap path used by `snapshot()` / `finalize()`.
        for (cat, stat) in &per_image {
            self.acc.entry(*cat).or_default().add_assign(stat);
        }

        if let Some(deltas) = self.per_image.as_mut() {
            deltas.push((image_id, per_image));
        }
        Ok(())
    }

    /// Compute a [`PanopticSummary`] from the current state without
    /// consuming the evaluator. Clones the per-category accumulator;
    /// callers who don't need to keep updating after the snapshot
    /// should prefer [`finalize`](Self::finalize) which avoids the
    /// clone.
    pub fn snapshot(&self) -> Result<PanopticSummary, PanopticError> {
        summarize_from_acc(
            self.acc.clone(),
            &self.categories,
            self.parity_mode,
            self.things_stuff_split,
        )
    }

    /// Consume the evaluator and produce the final
    /// [`PanopticSummary`].
    pub fn finalize(self) -> Result<PanopticSummary, PanopticError> {
        summarize_from_acc(
            self.acc,
            &self.categories,
            self.parity_mode,
            self.things_stuff_split,
        )
    }

    fn encode_input(&self) -> EncodeInput<'_> {
        EncodeInput {
            categories: &self.categories,
            acc: &self.acc,
            per_image: self.per_image.as_deref(),
            seen_images: &self.seen_images,
            parity_mode: self.parity_mode,
            things_stuff_split: self.things_stuff_split,
            retain_per_image_deltas: self.retain_per_image_deltas,
            rank_id: self.rank_id,
            n_images: self.seen_images.len() as u32,
        }
    }

    /// Serialize the current state to an opaque byte blob (ADR-0032).
    /// Non-consuming. The body carries `per_image_deltas` iff
    /// `retain_per_image_deltas=true` was set at construction.
    pub fn snapshot_to_partial(&self) -> Result<Vec<u8>, PanopticError> {
        Ok(encode(&self.encode_input())?)
    }

    /// Consuming variant of [`Self::snapshot_to_partial`].
    pub fn finalize_to_partial(self) -> Result<Vec<u8>, PanopticError> {
        Ok(encode(&self.encode_input())?)
    }

    /// Construct an evaluator equivalent to a batch run over the
    /// union of all partials' submitted images (ADR-0032).
    ///
    /// All partials must share `categories`, `parity_mode`,
    /// `things_stuff_split`, and `retain_per_image_deltas`. In strict
    /// mode every partial must declare a distinct `rank_id`. Image-id
    /// sets across partials must be disjoint.
    ///
    /// **Determinism:** with `retain_per_image_deltas=true`, strict-
    /// mode merge re-sorts all per-image deltas by `image_id` and
    /// re-sums in that order, matching [`crate::summarize::evaluate`]'s
    /// sorted iteration. The result is bit-equal to a batch run over
    /// the union. With `retain_per_image_deltas=false`, the merge
    /// adds per-rank `acc` directly; f64 non-associativity allows up
    /// to 4-ULP wobble vs batch (ADR-0004 envelope).
    ///
    /// **Returned evaluator is summary-ready, not re-mergeable:** the
    /// per-image deltas are folded into `acc` and discarded. Calling
    /// [`Self::finalize_to_partial`] on the returned value emits a
    /// partial *without* deltas, so it cannot be re-merged in strict
    /// mode. Callers needing a re-mergeable artifact should retain
    /// the original partials.
    pub fn from_partials(
        categories: HashMap<CategoryId, CategoryMeta>,
        parity_mode: ParityMode,
        things_stuff_split: bool,
        retain_per_image_deltas: bool,
        partials: &[&[u8]],
    ) -> Result<Self, PanopticError> {
        let strict = parity_mode == ParityMode::Strict;
        let exp = panoptic_expectation(
            &categories,
            parity_mode,
            things_stuff_split,
            retain_per_image_deltas,
        );
        let mut acc = PanopticMergeAccumulator::new(strict, retain_per_image_deltas);
        for bytes in partials {
            vernier_partial::with_validated_envelope(bytes, &exp, |view| acc.ingest(&view))?;
        }
        if retain_per_image_deltas {
            // Strict-mode bit-equality path: fold per-image deltas
            // in image-id order to match summarize.rs:188.
            acc.finalize_strict();
        }
        let mut ev = Self::new(
            categories,
            parity_mode,
            things_stuff_split,
            retain_per_image_deltas,
        );
        ev.acc = acc.acc;
        ev.seen_images = acc.base.image_ids().collect();
        // Per-image deltas are not carried into the merged evaluator
        // — `from_partials` returns a "summary-ready" evaluator, not
        // one set up for further re-merge. Re-merging would require
        // re-running the kernel on the original GT/DT, which the
        // partials don't carry.
        ev.per_image = None;
        ev.retain_per_image_deltas = false;
        Ok(ev)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::SegmentInfo;

    fn three_class_categories() -> HashMap<CategoryId, CategoryMeta> {
        let mut m = HashMap::new();
        m.insert(
            1,
            CategoryMeta {
                id: 1,
                isthing: true,
            },
        );
        m.insert(
            2,
            CategoryMeta {
                id: 2,
                isthing: false,
            },
        );
        m.insert(
            3,
            CategoryMeta {
                id: 3,
                isthing: true,
            },
        );
        m
    }

    fn perfect_match_image(image_id: ImageId) -> (ImageEntry, ImageEntry) {
        // Tiny 2x2 image with two segments (id 1 → cat 1, id 2 → cat 2).
        let _ = image_id;
        let label_map = vec![1, 1, 2, 2];
        let segments = vec![
            SegmentInfo {
                id: 1,
                category_id: 1,
                iscrowd: false,
                area: 2,
            },
            SegmentInfo {
                id: 2,
                category_id: 2,
                iscrowd: false,
                area: 2,
            },
        ];
        let gt =
            ImageEntry::from_components(image_id, 2, 2, label_map.clone(), segments.clone(), "gt")
                .expect("test fixture: gt");
        let dt = ImageEntry::from_components(image_id, 2, 2, label_map, segments, "dt")
            .expect("test fixture: dt");
        (gt, dt)
    }

    #[test]
    fn streaming_finalize_equals_batch() {
        // Two perfect-match images → mIoU should be 1.0 across both
        // categories, batch and streaming agree.
        let cats = three_class_categories();
        let mut ev = StreamingPanopticEvaluator::new(cats, ParityMode::Corrected, false, false);
        for img in [10, 20] {
            let (gt, dt) = perfect_match_image(img);
            ev.update(img, &gt, &dt).unwrap();
        }
        let summary = ev.finalize().unwrap();
        // Categories 1 and 2 contributed; category 3 had no support.
        assert_eq!(summary.n, 2);
        // Perfect match: pq == 1.0 over the two contributing categories.
        assert_eq!(summary.pq, 1.0);
    }

    #[test]
    fn from_partials_corrected_matches_batch_within_envelope() {
        let cats = three_class_categories();
        let mk_partial = |rank: u32, image_ids: &[ImageId]| -> Vec<u8> {
            let mut ev =
                StreamingPanopticEvaluator::new(cats.clone(), ParityMode::Corrected, false, false)
                    .with_rank(rank)
                    .unwrap();
            for &img in image_ids {
                let (gt, dt) = perfect_match_image(img);
                ev.update(img, &gt, &dt).unwrap();
            }
            ev.finalize_to_partial().unwrap()
        };

        let p0 = mk_partial(0, &[10, 11]);
        let p1 = mk_partial(1, &[20, 21]);
        let merged = StreamingPanopticEvaluator::from_partials(
            cats,
            ParityMode::Corrected,
            false,
            false,
            &[&p0, &p1],
        )
        .unwrap();
        let summary = merged.finalize().unwrap();
        // All four images perfect-match → pq=1.0 exactly (integer-only counters drive this).
        assert_eq!(summary.pq, 1.0);
        assert_eq!(summary.n, 2);
    }

    #[test]
    fn duplicate_image_id_within_one_evaluator_rejected() {
        let cats = three_class_categories();
        let mut ev = StreamingPanopticEvaluator::new(cats, ParityMode::Corrected, false, false);
        let (gt, dt) = perfect_match_image(7);
        ev.update(7, &gt, &dt).unwrap();
        let err = ev.update(7, &gt, &dt).unwrap_err();
        assert!(matches!(
            err,
            PanopticError::DuplicateImageId { image_id: 7 }
        ));
    }

    #[test]
    fn with_rank_after_update_rejected() {
        let cats = three_class_categories();
        let mut ev = StreamingPanopticEvaluator::new(cats, ParityMode::Corrected, false, false);
        let (gt, dt) = perfect_match_image(1);
        ev.update(1, &gt, &dt).unwrap();
        let err = ev.with_rank(0).unwrap_err();
        assert!(matches!(err, PanopticError::Partial(_)));
    }
}
