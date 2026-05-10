//! Distributed semantic-segmentation merge (ADR-0032).
//!
//! Each rank evaluates its slice of images via
//! [`crate::stream::StreamingSemanticEvaluator`], emits a partial via
//! [`crate::stream::StreamingSemanticEvaluator::finalize_to_partial`]
//! (or `snapshot_to_partial`), and the head rank reconstructs an
//! evaluator equivalent to a batch run over the union via
//! [`crate::stream::StreamingSemanticEvaluator::from_partials`].
//!
//! The wire envelope (magic, version, CRC, header validation) lives
//! in [`vernier_partial`]. This module owns:
//!
//! - `WireSemanticBody` (private) — the rkyv-archivable confusion
//!   matrix + per-image metadata.
//! - `SemanticMergeAccumulator` (private) — the merge fold. Confusion-matrix
//!   sum is integer-additive (commutative + associative), so strict-
//!   mode merge is **unconditionally bit-equal** to a batch run over
//!   the union (no `(score, rank_id, local_position)` tiebreak
//!   needed; semantic doesn't fold detections by score).
//! - [`semantic_dataset_hash`] / [`semantic_params_hash`] — paradigm-
//!   specific canonical fingerprints.

use rkyv::rancor::Error as RkyvError;
use vernier_partial::envelope::ValidatedView;
use vernier_partial::merge::BaseMergeAccumulator;
use vernier_partial::traits::{ParadigmKind, Partial, PartialExpectation};
use vernier_partial::{PartialError, PartialFormatErrorKind, WireEnvelopeHeader};

use crate::kernel::ConfusionMatrix;
use crate::parity::ParityMode;

/// `parity_mode = Strict` discriminant carried in the wire header.
pub(crate) const PARITY_STRICT: u8 = 0;
/// `parity_mode = Corrected` discriminant.
pub(crate) const PARITY_CORRECTED: u8 = 1;

/// Sub-kind discriminator for the semantic paradigm. Always `0` —
/// semantic has only one kernel family today. Reserved value space
/// for future variants (boundary semantic, hierarchical mIoU).
const SEMANTIC_DISCRIMINATOR: u32 = 0;

pub(crate) fn encode_parity_mode(m: ParityMode) -> u8 {
    match m {
        ParityMode::Strict => PARITY_STRICT,
        ParityMode::Corrected => PARITY_CORRECTED,
    }
}

/// BLAKE3 fingerprint over `(n_classes, ignore_label)`. Cross-rank
/// merge requires every partial to declare the same dataset shape,
/// otherwise the per-rank confusion matrices have incompatible
/// dimensions.
pub fn semantic_dataset_hash(n_classes: u32, ignore_label: Option<u32>) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(&n_classes.to_le_bytes());
    match ignore_label {
        Some(label) => {
            h.update(&[1u8]);
            h.update(&label.to_le_bytes());
        }
        None => {
            h.update(&[0u8]);
        }
    }
    *h.finalize().as_bytes()
}

/// BLAKE3 fingerprint over `parity_mode`. The streaming surface
/// rejects `label_remap` per ADR-0028 §"Streaming"; if/when streaming
/// gains remap support, the hash widens and [`vernier_partial::FORMAT_VERSION`]
/// bumps so old partials are refused at decode.
pub fn semantic_params_hash(parity_mode: ParityMode) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(&[encode_parity_mode(parity_mode)]);
    *h.finalize().as_bytes()
}

// ===========================================================================
// Wire body
// ===========================================================================

/// Semantic-paradigm body. The confusion matrix is the entire merge
/// state — semantic has no per-image cells, no detection list, no
/// retained IoUs. `seen_images` is recorded only for the disjoint-
/// partition check (the kernel itself never keys by image_id).
///
/// Field order is wire-format load-bearing (rkyv archived layout
/// follows declaration order). Add new fields at the end and bump
/// [`vernier_partial::FORMAT_VERSION`].
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub(crate) struct WireSemanticBody {
    /// `(n_classes, n_classes)` row-major confusion matrix counts.
    confusion_counts: Vec<u64>,
    /// Number of `update()` calls this rank made.
    n_images: u32,
    /// Sorted image ids this rank evaluated. Used by the partition-
    /// disjointness check.
    seen_images: Vec<i64>,
}

impl Partial for WireSemanticBody {
    const PARADIGM: ParadigmKind = ParadigmKind::Semantic;
}

// ===========================================================================
// Encode
// ===========================================================================

/// Inputs to [`encode`]. The streaming evaluator lifts its state
/// into this struct; we do the rkyv archive + framing.
pub(crate) struct EncodeInput<'a> {
    pub(crate) confusion: &'a ConfusionMatrix,
    pub(crate) ignore_label: Option<u32>,
    pub(crate) parity_mode: ParityMode,
    pub(crate) rank_id: Option<u32>,
    pub(crate) n_images: u32,
    pub(crate) seen_images: &'a std::collections::HashSet<i64>,
}

pub(crate) fn build_header(input: &EncodeInput<'_>) -> WireEnvelopeHeader {
    WireEnvelopeHeader {
        paradigm_kind: ParadigmKind::Semantic.as_u8(),
        discriminator: SEMANTIC_DISCRIMINATOR,
        parity_mode: encode_parity_mode(input.parity_mode),
        rank_id: input.rank_id,
        dataset_hash: semantic_dataset_hash(input.confusion.n_classes(), input.ignore_label),
        params_hash: semantic_params_hash(input.parity_mode),
        // Slot 0 is the cross-rank invariant (n_classes); the others
        // are unused. Per-rank `n_images` lives in the body, not the
        // header, because cross-rank ranks evaluate disjoint image
        // slices and therefore have different counts.
        shape_fingerprint: [input.confusion.n_classes(), 0, 0, 0],
    }
}

pub(crate) fn build_body(input: &EncodeInput<'_>) -> WireSemanticBody {
    let mut sorted_images: Vec<i64> = input.seen_images.iter().copied().collect();
    sorted_images.sort_unstable();
    WireSemanticBody {
        confusion_counts: input.confusion.counts().to_vec(),
        n_images: input.n_images,
        seen_images: sorted_images,
    }
}

/// Serialize streaming-semantic state to a partial byte blob.
pub(crate) fn encode(input: &EncodeInput<'_>) -> Result<Vec<u8>, PartialError> {
    let header = build_header(input);
    let body = build_body(input);
    let body_archive = rkyv::to_bytes::<RkyvError>(&body).map_err(|e| PartialError::Format {
        kind: PartialFormatErrorKind::RkyvDecode {
            detail: format!("rkyv::to_bytes(body) failed: {e}"),
        },
    })?;
    vernier_partial::encode(&header, &body_archive)
}

// ===========================================================================
// Expectation builder
// ===========================================================================

/// Build the [`PartialExpectation`] the receiving rank passes to
/// [`vernier_partial::with_validated_envelope`] for semantic partials.
pub(crate) fn semantic_expectation(
    n_classes: u32,
    ignore_label: Option<u32>,
    parity_mode: ParityMode,
) -> PartialExpectation {
    PartialExpectation {
        paradigm: ParadigmKind::Semantic,
        discriminator: SEMANTIC_DISCRIMINATOR,
        parity_mode: encode_parity_mode(parity_mode),
        dataset_hash: semantic_dataset_hash(n_classes, ignore_label),
        params_hash: semantic_params_hash(parity_mode),
        shape_fingerprint: [n_classes, 0, 0, 0],
        strict_mode: parity_mode == ParityMode::Strict,
    }
}

// ===========================================================================
// Merge accumulator
// ===========================================================================

/// Accumulator [`crate::stream::StreamingSemanticEvaluator::from_partials`]
/// folds into. Embeds [`BaseMergeAccumulator`] for the partition +
/// rank-collision policy and adds a single
/// [`ConfusionMatrix`] for the actual fold.
pub(crate) struct SemanticMergeAccumulator {
    pub(crate) base: BaseMergeAccumulator,
    pub(crate) confusion: ConfusionMatrix,
    pub(crate) n_images: u32,
}

impl SemanticMergeAccumulator {
    pub(crate) fn new(n_classes: u32, strict: bool) -> Self {
        Self {
            base: BaseMergeAccumulator::new(strict),
            confusion: ConfusionMatrix::zeros(n_classes),
            n_images: 0,
        }
    }

    /// Drain one validated envelope view into this accumulator.
    pub(crate) fn ingest(&mut self, view: &ValidatedView<'_>) -> Result<(), PartialError> {
        let rank_id = vernier_partial::envelope::rank_id_from_archive(view.header);
        self.base.ingest_rank_id(rank_id)?;

        let mut aligned: rkyv::util::AlignedVec<16> =
            rkyv::util::AlignedVec::with_capacity(view.body_archive.len());
        aligned.extend_from_slice(view.body_archive);
        let archived =
            rkyv::access::<ArchivedWireSemanticBody, RkyvError>(&aligned).map_err(|e| {
                PartialError::Format {
                    kind: PartialFormatErrorKind::RkyvDecode {
                        detail: format!("rkyv::access(body) failed: {e}"),
                    },
                }
            })?;

        self.base
            .ingest_image_ids(rank_id, archived.seen_images.iter().map(|v| v.to_native()))?;

        let n = self.confusion.n_classes() as usize;
        let expected_len = n.saturating_mul(n);
        if archived.confusion_counts.len() != expected_len {
            return Err(PartialError::Format {
                kind: PartialFormatErrorKind::RkyvDecode {
                    detail: format!(
                        "confusion_counts length {} != n_classes^2 = {expected_len}",
                        archived.confusion_counts.len()
                    ),
                },
            });
        }
        let dst = self.confusion.counts_mut();
        for (i, archived_count) in archived.confusion_counts.iter().enumerate() {
            dst[i] = dst[i].saturating_add(archived_count.to_native());
        }
        self.n_images = self.n_images.saturating_add(archived.n_images.to_native());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn fake_input(rank_id: Option<u32>) -> (ConfusionMatrix, HashSet<i64>) {
        let mut cm = ConfusionMatrix::zeros(3);
        // Pre-populate with deterministic counts so the round-trip
        // exercises non-zero data.
        let dst = cm.counts_mut();
        for (i, slot) in dst.iter_mut().enumerate() {
            *slot = (i as u64) + (rank_id.unwrap_or(0) as u64) * 100;
        }
        let mut seen = HashSet::new();
        for img in 0..5 {
            seen.insert(img + (rank_id.unwrap_or(0) as i64) * 1000);
        }
        (cm, seen)
    }

    #[test]
    fn round_trip_via_envelope() {
        let (cm, seen) = fake_input(Some(0));
        let input = EncodeInput {
            confusion: &cm,
            ignore_label: None,
            parity_mode: ParityMode::Corrected,
            rank_id: Some(0),
            n_images: 5,
            seen_images: &seen,
        };
        let bytes = encode(&input).unwrap();
        let exp = semantic_expectation(3, None, ParityMode::Corrected);
        let mut acc = SemanticMergeAccumulator::new(3, false);
        vernier_partial::with_validated_envelope(&bytes, &exp, |view| acc.ingest(&view)).unwrap();
        assert_eq!(acc.confusion.counts(), cm.counts());
        assert_eq!(acc.n_images, 5);
    }

    #[test]
    fn merge_two_partials_sums_confusion_matrices() {
        let (cm0, seen0) = fake_input(Some(0));
        let (cm1, seen1) = fake_input(Some(1));
        let bytes0 = encode(&EncodeInput {
            confusion: &cm0,
            ignore_label: None,
            parity_mode: ParityMode::Corrected,
            rank_id: Some(0),
            n_images: 5,
            seen_images: &seen0,
        })
        .unwrap();
        let bytes1 = encode(&EncodeInput {
            confusion: &cm1,
            ignore_label: None,
            parity_mode: ParityMode::Corrected,
            rank_id: Some(1),
            n_images: 5,
            seen_images: &seen1,
        })
        .unwrap();
        let exp = semantic_expectation(3, None, ParityMode::Corrected);
        let mut acc = SemanticMergeAccumulator::new(3, false);
        vernier_partial::with_validated_envelope(&bytes0, &exp, |view| acc.ingest(&view)).unwrap();
        vernier_partial::with_validated_envelope(&bytes1, &exp, |view| acc.ingest(&view)).unwrap();

        let expected: Vec<u64> = cm0
            .counts()
            .iter()
            .zip(cm1.counts())
            .map(|(a, b)| a + b)
            .collect();
        assert_eq!(acc.confusion.counts(), expected.as_slice());
        assert_eq!(acc.n_images, 10);
    }

    #[test]
    fn dataset_hash_changes_with_n_classes() {
        assert_ne!(
            semantic_dataset_hash(3, None),
            semantic_dataset_hash(4, None)
        );
    }

    #[test]
    fn dataset_hash_distinguishes_some_none_ignore_label() {
        // (3, Some(0)) and (3, None) must hash differently — Some(0)
        // is a real ignore label (ADE20K), not the same as no
        // ignore at all.
        assert_ne!(
            semantic_dataset_hash(3, Some(0)),
            semantic_dataset_hash(3, None)
        );
    }

    #[test]
    fn params_hash_changes_with_parity_mode() {
        assert_ne!(
            semantic_params_hash(ParityMode::Strict),
            semantic_params_hash(ParityMode::Corrected)
        );
    }

    #[test]
    fn rejects_partition_overlap() {
        let (cm, seen) = fake_input(Some(0));
        let input = EncodeInput {
            confusion: &cm,
            ignore_label: None,
            parity_mode: ParityMode::Corrected,
            rank_id: Some(0),
            n_images: 5,
            seen_images: &seen,
        };
        let bytes = encode(&input).unwrap();
        let exp = semantic_expectation(3, None, ParityMode::Corrected);
        let mut acc = SemanticMergeAccumulator::new(3, false);
        vernier_partial::with_validated_envelope(&bytes, &exp, |view| acc.ingest(&view)).unwrap();
        // Same image_ids — partition rule fires.
        let err = vernier_partial::with_validated_envelope(&bytes, &exp, |view| acc.ingest(&view))
            .unwrap_err();
        assert!(matches!(err, PartialError::PartitionOverlap { .. }));
    }
}
