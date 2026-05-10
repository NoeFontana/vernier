//! Distributed panoptic-quality merge (ADR-0032).
//!
//! Each rank evaluates its slice of images via
//! [`crate::stream::StreamingPanopticEvaluator`], emits a partial via
//! `finalize_to_partial`, and the head rank reconstructs an evaluator
//! equivalent to a batch run over the union via `from_partials`.
//!
//! The wire envelope (magic, version, CRC, header validation) lives
//! in [`vernier_partial`]. This module owns:
//!
//! - `WirePanopticBody` (private) — the rkyv-archivable per-category
//!   PqStat accumulator plus the optional per-image deltas (when
//!   `retain_per_image_deltas=True`) needed to reconstruct strict-
//!   mode bit-equality across f64 non-associativity.
//! - `PanopticMergeAccumulator` (private) — two paths:
//!   - **Corrected mode**: sum `per_category` across ranks. Fast, low
//!     memory, but **not bit-equal** to a batch run because f64
//!     addition isn't associative; the 4-ULP envelope (ADR-0004)
//!     bounds the wobble.
//!   - **Strict mode** (with `retain_per_image_deltas=True`):
//!     re-sort all `(image_id, per_image_pqmap)` tuples by image_id
//!     across ranks and re-sum in that order, matching
//!     [`crate::summarize::evaluate`]'s sorted iteration. Bit-equal
//!     to batch.
//! - [`panoptic_dataset_hash`] / [`panoptic_params_hash`] — paradigm-
//!   specific canonical fingerprints.

use std::collections::HashMap;

use rkyv::rancor::Error as RkyvError;
use vernier_partial::envelope::ValidatedView;
use vernier_partial::merge::BaseMergeAccumulator;
use vernier_partial::traits::{ParadigmKind, Partial, PartialExpectation};
use vernier_partial::{PartialError, PartialFormatErrorKind, WireEnvelopeHeader};

use crate::attribute::PqStat;
use crate::dataset::{CategoryId, CategoryMeta, ImageId};
use crate::parity::ParityMode;

// ===========================================================================
// Header field encoders + paradigm constants
// ===========================================================================

pub(crate) const PARITY_STRICT: u8 = 0;
pub(crate) const PARITY_CORRECTED: u8 = 1;

/// Sub-kind discriminator for the panoptic paradigm. Always `0` for
/// instance-based PQ; reserved value space for boundary-PQ (Q3 / Z2
/// follow-up ADR).
const PANOPTIC_DISCRIMINATOR: u32 = 0;

pub(crate) fn encode_parity_mode(m: ParityMode) -> u8 {
    match m {
        ParityMode::Strict => PARITY_STRICT,
        ParityMode::Corrected => PARITY_CORRECTED,
    }
}

// ===========================================================================
// Hashes
// ===========================================================================

/// BLAKE3 fingerprint over the categories taxonomy. The cross-rank
/// invariant for panoptic is the `(category_id, isthing)` set: every
/// rank evaluates against the same taxonomy, otherwise per-category
/// PqStat aggregation produces values that don't correspond to any
/// single dataset.
pub fn panoptic_dataset_hash(categories: &HashMap<CategoryId, CategoryMeta>) -> [u8; 32] {
    let mut sorted: Vec<(CategoryId, bool)> =
        categories.iter().map(|(&id, m)| (id, m.isthing)).collect();
    sorted.sort_unstable_by_key(|(id, _)| *id);

    let mut h = blake3::Hasher::new();
    h.update(&(sorted.len() as u32).to_le_bytes());
    for (id, isthing) in sorted {
        h.update(&id.to_le_bytes());
        h.update(&[u8::from(isthing)]);
    }
    *h.finalize().as_bytes()
}

/// BLAKE3 fingerprint over `(parity_mode, things_stuff_split,
/// retain_per_image_deltas)`. All three control how the merge fold
/// interprets the body: parity_mode picks the V3 storage shape (per
/// `attribute_image`), things_stuff_split decides whether the summary
/// has per-bucket means, and retain_per_image_deltas decides whether
/// strict-mode merge has the data it needs to re-sum in image-id
/// order.
pub fn panoptic_params_hash(
    parity_mode: ParityMode,
    things_stuff_split: bool,
    retain_per_image_deltas: bool,
) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(&[encode_parity_mode(parity_mode)]);
    h.update(&[u8::from(things_stuff_split)]);
    h.update(&[u8::from(retain_per_image_deltas)]);
    *h.finalize().as_bytes()
}

// ===========================================================================
// Wire body
// ===========================================================================

/// rkyv mirror of [`PqStat`]. The runtime type stays free of rkyv
/// derives per the ADR-0005 spirit (kernel types are pure Rust);
/// conversion happens at the encode/decode boundary.
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize, Debug, Clone, Copy)]
pub(crate) struct WirePqStat {
    sum_iou: f64,
    n_tp: u64,
    n_fp: u64,
    n_fn: u64,
}

impl From<PqStat> for WirePqStat {
    fn from(p: PqStat) -> Self {
        Self {
            sum_iou: p.sum_iou,
            n_tp: p.n_tp,
            n_fp: p.n_fp,
            n_fn: p.n_fn,
        }
    }
}

/// Per-image delta entry: one image's contribution as a per-category
/// `PqStat` map. Carried only when `retain_per_image_deltas=True`
/// (strict-mode bit-equality property).
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub(crate) struct WirePerImagePqMap {
    image_id: i64,
    /// Sorted by `category_id` for deterministic encoding.
    entries: Vec<(i64, WirePqStat)>,
}

/// Panoptic-paradigm body. Carries:
///
/// - `per_category`: sum of all per-image PqStats this rank produced,
///   sorted by `category_id`. Corrected-mode merge sums these
///   directly across ranks.
/// - `per_image_deltas`: optional list of `(image_id, per_category_map)`
///   sorted by `image_id`. Required for strict-mode merge; absent in
///   corrected mode by default.
/// - `seen_images`: sorted image ids for the partition-disjointness
///   check.
/// - `n_images`: rank-local image count (for diagnostics).
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub(crate) struct WirePanopticBody {
    per_category: Vec<(i64, WirePqStat)>,
    per_image_deltas: Option<Vec<WirePerImagePqMap>>,
    seen_images: Vec<i64>,
    n_images: u32,
}

impl Partial for WirePanopticBody {
    const PARADIGM: ParadigmKind = ParadigmKind::Panoptic;
}

// ===========================================================================
// Encode
// ===========================================================================

/// Inputs to [`encode`]. The streaming evaluator lifts its state
/// into this struct.
pub(crate) struct EncodeInput<'a> {
    pub(crate) categories: &'a HashMap<CategoryId, CategoryMeta>,
    pub(crate) acc: &'a HashMap<CategoryId, PqStat>,
    pub(crate) per_image: Option<&'a [(ImageId, HashMap<CategoryId, PqStat>)]>,
    pub(crate) seen_images: &'a std::collections::HashSet<ImageId>,
    pub(crate) parity_mode: ParityMode,
    pub(crate) things_stuff_split: bool,
    pub(crate) retain_per_image_deltas: bool,
    pub(crate) rank_id: Option<u32>,
    pub(crate) n_images: u32,
}

pub(crate) fn build_header(input: &EncodeInput<'_>) -> WireEnvelopeHeader {
    WireEnvelopeHeader {
        paradigm_kind: ParadigmKind::Panoptic.as_u8(),
        discriminator: PANOPTIC_DISCRIMINATOR,
        parity_mode: encode_parity_mode(input.parity_mode),
        rank_id: input.rank_id,
        dataset_hash: panoptic_dataset_hash(input.categories),
        params_hash: panoptic_params_hash(
            input.parity_mode,
            input.things_stuff_split,
            input.retain_per_image_deltas,
        ),
        // Slot 0 is `n_categories` — a cross-rank invariant. The
        // remaining slots are unused; per-rank `n_images` lives in
        // the body because ranks evaluate disjoint image slices.
        shape_fingerprint: [input.categories.len() as u32, 0, 0, 0],
    }
}

fn pack_per_category(acc: &HashMap<CategoryId, PqStat>) -> Vec<(i64, WirePqStat)> {
    let mut out: Vec<(i64, WirePqStat)> = acc.iter().map(|(&id, &s)| (id, s.into())).collect();
    out.sort_unstable_by_key(|(id, _)| *id);
    out
}

fn pack_per_image(per_image: &[(ImageId, HashMap<CategoryId, PqStat>)]) -> Vec<WirePerImagePqMap> {
    let mut out: Vec<WirePerImagePqMap> = per_image
        .iter()
        .map(|(image_id, entries)| {
            let mut sorted: Vec<(i64, WirePqStat)> =
                entries.iter().map(|(&cat, &s)| (cat, s.into())).collect();
            sorted.sort_unstable_by_key(|(id, _)| *id);
            WirePerImagePqMap {
                image_id: *image_id,
                entries: sorted,
            }
        })
        .collect();
    out.sort_unstable_by_key(|m| m.image_id);
    out
}

pub(crate) fn build_body(input: &EncodeInput<'_>) -> WirePanopticBody {
    let mut sorted_images: Vec<i64> = input.seen_images.iter().copied().collect();
    sorted_images.sort_unstable();
    WirePanopticBody {
        per_category: pack_per_category(input.acc),
        per_image_deltas: input.per_image.map(pack_per_image),
        seen_images: sorted_images,
        n_images: input.n_images,
    }
}

/// Serialize streaming-panoptic state to a partial byte blob.
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
/// [`vernier_partial::with_validated_envelope`] for panoptic partials.
pub(crate) fn panoptic_expectation(
    categories: &HashMap<CategoryId, CategoryMeta>,
    parity_mode: ParityMode,
    things_stuff_split: bool,
    retain_per_image_deltas: bool,
) -> PartialExpectation {
    PartialExpectation {
        paradigm: ParadigmKind::Panoptic,
        discriminator: PANOPTIC_DISCRIMINATOR,
        parity_mode: encode_parity_mode(parity_mode),
        dataset_hash: panoptic_dataset_hash(categories),
        params_hash: panoptic_params_hash(parity_mode, things_stuff_split, retain_per_image_deltas),
        shape_fingerprint: [categories.len() as u32, 0, 0, 0],
        strict_mode: parity_mode == ParityMode::Strict,
    }
}

// ===========================================================================
// Merge accumulator
// ===========================================================================

/// Per-image delta carried across the merge accumulator's strict-
/// mode path. Owned (not archive-borrowed) because we collect deltas
/// from every partial then sort the union.
type OwnedPerImage = (ImageId, HashMap<CategoryId, PqStat>);

/// Accumulator [`crate::stream::StreamingPanopticEvaluator::from_partials`]
/// folds into. Embeds [`BaseMergeAccumulator`] for partition + rank-
/// collision policy. Two paths:
///
/// - Corrected: `acc` accumulates the per-category sum directly from
///   each partial's `per_category`.
/// - Strict: `per_image_deltas` collects every rank's per-image
///   maps; `finalize` sorts by image_id and sums into the final
///   `acc` matching [`crate::summarize::evaluate`]'s order.
pub(crate) struct PanopticMergeAccumulator {
    pub(crate) base: BaseMergeAccumulator,
    /// Final aggregated per-category state. In corrected mode this
    /// is updated as partials arrive; in strict mode it stays empty
    /// until [`Self::finalize_strict`] runs.
    pub(crate) acc: HashMap<CategoryId, PqStat>,
    /// Strict-mode per-image deltas, collected across ranks. Only
    /// populated when the receiving evaluator declared
    /// `retain_per_image_deltas=true`.
    pub(crate) per_image_deltas: Vec<OwnedPerImage>,
    pub(crate) n_images: u32,
    /// Whether the receiver expects per-image deltas. `true` =
    /// strict-mode bit-equality path; `false` = corrected-mode
    /// direct-sum path.
    retain_deltas: bool,
}

impl PanopticMergeAccumulator {
    pub(crate) fn new(strict: bool, retain_deltas: bool) -> Self {
        Self {
            base: BaseMergeAccumulator::new(strict),
            acc: HashMap::new(),
            per_image_deltas: Vec::new(),
            n_images: 0,
            retain_deltas,
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
            rkyv::access::<ArchivedWirePanopticBody, RkyvError>(&aligned).map_err(|e| {
                PartialError::Format {
                    kind: PartialFormatErrorKind::RkyvDecode {
                        detail: format!("rkyv::access(body) failed: {e}"),
                    },
                }
            })?;

        self.base
            .ingest_image_ids(rank_id, archived.seen_images.iter().map(|v| v.to_native()))?;

        self.n_images = self.n_images.saturating_add(archived.n_images.to_native());

        if self.retain_deltas {
            // Strict-mode path: expect per-image deltas on every
            // partial; collect into the buffer for sorted re-sum at
            // finalize time.
            let Some(deltas) = archived.per_image_deltas.as_ref() else {
                return Err(PartialError::Format {
                    kind: PartialFormatErrorKind::RkyvDecode {
                        detail: "strict-mode panoptic merge expected per_image_deltas; partial \
                                 was emitted with retain_per_image_deltas=false"
                            .to_string(),
                    },
                });
            };
            for delta in deltas.iter() {
                let image_id = delta.image_id.to_native();
                let mut entries: HashMap<CategoryId, PqStat> = HashMap::new();
                for entry in delta.entries.iter() {
                    let cat = entry.0.to_native();
                    let stat = unpack_pqstat(&entry.1);
                    entries.insert(cat, stat);
                }
                self.per_image_deltas.push((image_id, entries));
            }
        } else {
            // Corrected-mode path: sum per_category directly.
            for entry in archived.per_category.iter() {
                let cat = entry.0.to_native();
                let stat = unpack_pqstat(&entry.1);
                self.acc.entry(cat).or_default().add_assign(&stat);
            }
        }
        Ok(())
    }

    /// Sort the strict-mode per-image deltas by image_id and fold
    /// them into `acc`. Matches [`crate::summarize::evaluate`]'s
    /// sorted iteration so f64 sums align bit-for-bit with batch.
    /// Idempotent: the accumulator's other state (rank ids, image
    /// owners) is not consulted past this point.
    pub(crate) fn finalize_strict(&mut self) {
        self.per_image_deltas.sort_by_key(|(image_id, _)| *image_id);
        for (_, per_image) in self.per_image_deltas.drain(..) {
            for (cat, stat) in per_image {
                self.acc.entry(cat).or_default().add_assign(&stat);
            }
        }
    }
}

fn unpack_pqstat(archived: &ArchivedWirePqStat) -> PqStat {
    PqStat {
        sum_iou: archived.sum_iou.to_native(),
        n_tp: archived.n_tp.to_native(),
        n_fp: archived.n_fp.to_native(),
        n_fn: archived.n_fn.to_native(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn fake_categories() -> HashMap<CategoryId, CategoryMeta> {
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

    fn fake_acc(seed: u64) -> HashMap<CategoryId, PqStat> {
        let mut m = HashMap::new();
        m.insert(
            1,
            PqStat {
                sum_iou: 1.0 + seed as f64 * 0.1,
                n_tp: 2 + seed,
                n_fp: 1,
                n_fn: 0,
            },
        );
        m.insert(
            2,
            PqStat {
                sum_iou: 0.5 + seed as f64 * 0.05,
                n_tp: 1,
                n_fp: 0,
                n_fn: 1 + seed,
            },
        );
        m
    }

    #[test]
    fn dataset_hash_changes_with_categories() {
        let cats = fake_categories();
        let mut more_cats = cats.clone();
        more_cats.insert(
            4,
            CategoryMeta {
                id: 4,
                isthing: false,
            },
        );
        assert_ne!(
            panoptic_dataset_hash(&cats),
            panoptic_dataset_hash(&more_cats)
        );
    }

    #[test]
    fn dataset_hash_changes_with_isthing() {
        let cats = fake_categories();
        let mut flipped = cats.clone();
        flipped.insert(
            1,
            CategoryMeta {
                id: 1,
                isthing: false,
            },
        );
        assert_ne!(
            panoptic_dataset_hash(&cats),
            panoptic_dataset_hash(&flipped)
        );
    }

    #[test]
    fn params_hash_distinguishes_all_three_axes() {
        let baseline = panoptic_params_hash(ParityMode::Corrected, false, false);
        assert_ne!(
            baseline,
            panoptic_params_hash(ParityMode::Strict, false, false)
        );
        assert_ne!(
            baseline,
            panoptic_params_hash(ParityMode::Corrected, true, false)
        );
        assert_ne!(
            baseline,
            panoptic_params_hash(ParityMode::Corrected, false, true)
        );
    }

    #[test]
    fn corrected_round_trip_via_envelope() {
        let cats = fake_categories();
        let acc = fake_acc(0);
        let mut seen = HashSet::new();
        for img in 0..3 {
            seen.insert(img);
        }
        let bytes = encode(&EncodeInput {
            categories: &cats,
            acc: &acc,
            per_image: None,
            seen_images: &seen,
            parity_mode: ParityMode::Corrected,
            things_stuff_split: false,
            retain_per_image_deltas: false,
            rank_id: Some(0),
            n_images: 3,
        })
        .unwrap();

        let exp = panoptic_expectation(&cats, ParityMode::Corrected, false, false);
        let mut merge = PanopticMergeAccumulator::new(false, false);
        vernier_partial::with_validated_envelope(&bytes, &exp, |view| merge.ingest(&view)).unwrap();
        assert_eq!(merge.n_images, 3);
        assert_eq!(merge.acc.len(), 2);
        assert_eq!(merge.acc[&1].n_tp, acc[&1].n_tp);
    }

    #[test]
    fn corrected_merge_two_partials_sums_per_category() {
        let cats = fake_categories();
        let acc0 = fake_acc(0);
        let acc1 = fake_acc(1);
        let mk_bytes = |acc: &HashMap<CategoryId, PqStat>, rank: u32, imgs: [i64; 2]| {
            let mut seen = HashSet::new();
            seen.insert(imgs[0]);
            seen.insert(imgs[1]);
            encode(&EncodeInput {
                categories: &cats,
                acc,
                per_image: None,
                seen_images: &seen,
                parity_mode: ParityMode::Corrected,
                things_stuff_split: false,
                retain_per_image_deltas: false,
                rank_id: Some(rank),
                n_images: 2,
            })
            .unwrap()
        };
        let b0 = mk_bytes(&acc0, 0, [0, 1]);
        let b1 = mk_bytes(&acc1, 1, [2, 3]);

        let exp = panoptic_expectation(&cats, ParityMode::Corrected, false, false);
        let mut merge = PanopticMergeAccumulator::new(false, false);
        vernier_partial::with_validated_envelope(&b0, &exp, |view| merge.ingest(&view)).unwrap();
        vernier_partial::with_validated_envelope(&b1, &exp, |view| merge.ingest(&view)).unwrap();

        assert_eq!(merge.n_images, 4);
        assert_eq!(merge.acc[&1].n_tp, acc0[&1].n_tp + acc1[&1].n_tp);
        assert_eq!(merge.acc[&1].n_fp, acc0[&1].n_fp + acc1[&1].n_fp);
        assert_eq!(merge.acc[&2].n_fn, acc0[&2].n_fn + acc1[&2].n_fn);
    }

    #[test]
    fn strict_without_deltas_rejected() {
        // Encode WITHOUT per_image_deltas, then try to merge in strict
        // mode (which expects deltas). Should raise typed error.
        let cats = fake_categories();
        let acc = fake_acc(0);
        let seen = HashSet::new();
        let bytes = encode(&EncodeInput {
            categories: &cats,
            acc: &acc,
            per_image: None,
            seen_images: &seen,
            parity_mode: ParityMode::Strict,
            things_stuff_split: false,
            retain_per_image_deltas: false,
            rank_id: Some(0),
            n_images: 0,
        })
        .unwrap();

        // The receiving expectation declares retain_per_image_deltas=true
        // — params_hash will mismatch first.
        let exp = panoptic_expectation(&cats, ParityMode::Strict, false, true);
        let mut merge = PanopticMergeAccumulator::new(true, true);
        let err =
            vernier_partial::with_validated_envelope(&bytes, &exp, |view| merge.ingest(&view))
                .unwrap_err();
        // Either ParamsMismatch (because retain flag is in params hash)
        // or RkyvDecode if hashes happened to align — params hash is the
        // load-bearing check.
        assert!(matches!(err, PartialError::ParamsMismatch { .. }));
    }
}
