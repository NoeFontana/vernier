//! Instance-paradigm body for the partial wire format and merge
//! orchestration (ADR-0031, generalized in ADR-0032).
//!
//! Per ADR-0032, the wire envelope (magic, version, CRC, header
//! fields, partition + rank-collision policy, the five typed errors)
//! lives in the leaf crate [`vernier_partial`]. This module owns the
//! *instance-paradigm* body — the per-`(k, a, i)` cells, meta_cells,
//! retained_ious, and dets_seen produced by the AP fold — plus the
//! pack/unpack helpers that translate between the spine's runtime
//! state ([`PerImageEval`], [`EvalImageMeta`], …) and the rkyv
//! archived form on the wire.
//!
//! Encoding is one rkyv archive of [`WireInstanceBody`] handed to
//! [`vernier_partial::encode`] alongside a header built via
//! [`build_header`]. Decoding routes through
//! [`vernier_partial::with_validated_envelope`] which validates
//! framing + paradigm + dataset/params/parity hashes and then hands
//! the body archive bytes to [`InstanceMergeAccumulator::ingest`]
//! to fold into the merged state.
//!
//! ## Determinism
//!
//! The cells / meta_cells / retained_ious stores in the runtime are
//! `HashMap`s; their iteration order is not stable. The wire form
//! sorts every collection by key before archiving so two encodings
//! of equal state produce byte-identical blobs. This matters for
//! [`crate::StreamingEvaluator::checkpoint`] /
//! [`crate::StreamingEvaluator::restore`] round-trip equality, and
//! for cross-rank reproducibility under the strict-tier `(rank_id,
//! local_position)` ordering reserved in ADR-0031.

use std::collections::HashMap;

use ndarray::Array2;
use rkyv::rancor::Error as RkyvError;
use vernier_partial::{
    BaseMergeAccumulator, ParadigmKind, Partial, PartialError, PartialExpectation, ValidatedView,
    WireEnvelopeHeader,
};

use crate::accumulate::PerImageEval;
use crate::dataset::{AnnId, Bbox, CategoryId, CocoDataset, CocoDetection, ImageId};
use crate::error::PartialFormatErrorKind;
use crate::evaluate::{EvalImageMeta, EvalKernel, OwnedEvaluateParams};
use crate::parity::ParityMode;
use crate::segmentation::{Segmentation, SegmentationRle, SegmentationRleCounts};
use crate::tables::RetainedIous;
use crate::EvalError;

// Re-export `RankId` under its existing path so callers
// (FFI, stream.rs) can keep using `crate::distributed::RankId`.
pub use vernier_partial::RankId;

// ===========================================================================
// Wire-form shadow types
// ===========================================================================
//
// All collections are `Vec` (sorted by key where applicable) — never
// `HashMap` — so the archived bytes are deterministic. Array2 fields
// flatten to `(shape: [u32; 2], data: Vec<T>)` so rkyv can archive
// them without needing a `with` adapter on `ndarray`.
//
// **Field order on every `#[derive(rkyv::*)]` struct below is
// wire-format load-bearing.** rkyv's archived layout follows the
// declaration order; reordering or inserting a field changes the
// byte layout and silently breaks compatibility with already-emitted
// partials. Add new fields at the end and bump [`FORMAT_VERSION`]
// when doing so. The receiver-side validator refuses mismatched
// versions, but only if we remember to bump.

#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize, PartialEq, Eq, PartialOrd, Ord)]
struct WireGridKey {
    k: u32,
    a: u32,
    i: u32,
}

#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
struct WirePerImageEval {
    dt_scores: Vec<f64>,
    dt_matched_shape: [u32; 2],
    dt_matched_data: Vec<u8>,
    dt_ignore_shape: [u32; 2],
    dt_ignore_data: Vec<u8>,
    gt_ignore: Vec<u8>,
}

#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
struct WireEvalImageMeta {
    image_id: i64,
    category_id: i64,
    area_rng: [f64; 2],
    max_det: u64,
    dt_ids: Vec<i64>,
    gt_ids: Vec<i64>,
    dt_matches_shape: [u32; 2],
    dt_matches_data: Vec<i64>,
    gt_matches_shape: [u32; 2],
    gt_matches_data: Vec<i64>,
}

#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
struct WireRetainedIousEntry {
    k: u32,
    i: u32,
    shape: [u32; 2],
    data: Vec<f64>,
}

#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
struct WireBbox {
    x: f64,
    y: f64,
    w: f64,
    h: f64,
}

#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
enum WireSegmentationCounts {
    Compressed(String),
    Uncompressed(Vec<u32>),
}

#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
struct WireSegmentationRle {
    h: u32,
    w: u32,
    counts: WireSegmentationCounts,
}

#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
enum WireSegmentation {
    Polygons(Vec<Vec<f64>>),
    Rle(WireSegmentationRle),
}

#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
struct WireCocoDetection {
    id: i64,
    image_id: i64,
    category_id: i64,
    score: f64,
    bbox: WireBbox,
    area: f64,
    segmentation: Option<WireSegmentation>,
    keypoints: Option<Vec<f64>>,
    num_keypoints: Option<u32>,
}

/// Instance-paradigm body (the per-`(k, a, i)` AP-fold cells). Slots
/// into the partial wire format alongside the
/// [`vernier_partial::WireEnvelopeHeader`] that frames it.
///
/// Fields are private — the encode + decode entry points
/// ([`build_body`], [`InstanceMergeAccumulator::ingest`]) are the only
/// supported access path, which keeps the wire-format invariants
/// (sorted vec, packed array shapes) impossible to violate from
/// outside this module.
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub(crate) struct WireInstanceBody {
    n_detections: u64,
    next_dt_id: i64,
    seen_images: Vec<i64>,
    cells: Vec<(WireGridKey, WirePerImageEval)>,
    meta_cells: Option<Vec<(WireGridKey, WireEvalImageMeta)>>,
    retained_ious: Option<Vec<WireRetainedIousEntry>>,
    dets_seen: Option<Vec<WireCocoDetection>>,
}

impl Partial for WireInstanceBody {
    const PARADIGM: ParadigmKind = ParadigmKind::Instance;
}

// ===========================================================================
// Parity mode encoding
// ===========================================================================
//
// `ParityMode` is paradigm-defined; the wire envelope carries it as a
// `u8` discriminant. Stable values (`Strict=0, Corrected=1`) are
// load-bearing across the format-version boundary.

pub(crate) const PARITY_STRICT: u8 = 0;
pub(crate) const PARITY_CORRECTED: u8 = 1;

pub(crate) fn encode_parity_mode(m: ParityMode) -> u8 {
    match m {
        ParityMode::Strict => PARITY_STRICT,
        ParityMode::Corrected => PARITY_CORRECTED,
    }
}

/// Flatten an `Array2<E>` into a `(shape, Vec<W>)` pair via `convert`.
/// `W` is the wire element type (`u8` for bools, `T` itself for
/// numeric types). Used by every pack helper below.
fn pack_array<E: Copy, W>(arr: &Array2<E>, convert: impl Fn(E) -> W) -> ([u32; 2], Vec<W>) {
    let (rows, cols) = arr.dim();
    (
        [rows as u32, cols as u32],
        arr.iter().copied().map(convert).collect(),
    )
}

/// Inverse of [`pack_array`]: reconstruct `Array2<E>` from a wire
/// `(shape, &[W])` pair. `convert` maps each wire element to the
/// runtime element type. Returns the typed
/// [`PartialFormatErrorKind::RkyvDecode`] on shape/data mismatch so
/// the framing-validator's vocabulary is preserved.
fn unpack_array<W: Copy, E>(
    shape: [u32; 2],
    data: &[W],
    field: &'static str,
    convert: impl Fn(W) -> E,
) -> Result<Array2<E>, PartialError> {
    let rows = shape[0] as usize;
    let cols = shape[1] as usize;
    if data.len() != rows.saturating_mul(cols) {
        return Err(PartialError::Format {
            kind: PartialFormatErrorKind::RkyvDecode {
                detail: format!(
                    "{field} shape {rows}x{cols} doesn't match data len {}",
                    data.len()
                ),
            },
        });
    }
    let owned: Vec<E> = data.iter().copied().map(convert).collect();
    Array2::from_shape_vec((rows, cols), owned).map_err(|e| PartialError::Format {
        kind: PartialFormatErrorKind::RkyvDecode {
            detail: format!("{field} from_shape_vec: {e}"),
        },
    })
}

fn pack_per_image_eval(p: &PerImageEval) -> WirePerImageEval {
    let (dt_matched_shape, dt_matched_data) = pack_array(&p.dt_matched, u8::from);
    let (dt_ignore_shape, dt_ignore_data) = pack_array(&p.dt_ignore, u8::from);
    WirePerImageEval {
        dt_scores: p.dt_scores.clone(),
        dt_matched_shape,
        dt_matched_data,
        dt_ignore_shape,
        dt_ignore_data,
        gt_ignore: p.gt_ignore.iter().map(|&b| u8::from(b)).collect(),
    }
}

fn pack_eval_image_meta(m: &EvalImageMeta) -> WireEvalImageMeta {
    let (dt_matches_shape, dt_matches_data) = pack_array(&m.dt_matches, |v| v);
    let (gt_matches_shape, gt_matches_data) = pack_array(&m.gt_matches, |v| v);
    WireEvalImageMeta {
        image_id: m.image_id,
        category_id: m.category_id,
        area_rng: m.area_rng,
        max_det: m.max_det as u64,
        dt_ids: m.dt_ids.clone(),
        gt_ids: m.gt_ids.clone(),
        dt_matches_shape,
        dt_matches_data,
        gt_matches_shape,
        gt_matches_data,
    }
}

fn pack_segmentation(seg: &Segmentation) -> WireSegmentation {
    match seg {
        Segmentation::Polygons(polys) => WireSegmentation::Polygons(polys.clone()),
        Segmentation::Rle(rle) => WireSegmentation::Rle(WireSegmentationRle {
            h: rle.size[0],
            w: rle.size[1],
            counts: match &rle.counts {
                SegmentationRleCounts::Compressed(s) => {
                    WireSegmentationCounts::Compressed(s.clone())
                }
                SegmentationRleCounts::Uncompressed(c) => {
                    WireSegmentationCounts::Uncompressed(c.clone())
                }
            },
        }),
    }
}

fn pack_coco_detection(d: &CocoDetection) -> WireCocoDetection {
    WireCocoDetection {
        id: d.id.0,
        image_id: d.image_id.0,
        category_id: d.category_id.0,
        score: d.score,
        bbox: WireBbox {
            x: d.bbox.x,
            y: d.bbox.y,
            w: d.bbox.w,
            h: d.bbox.h,
        },
        area: d.area,
        segmentation: d.segmentation.as_ref().map(pack_segmentation),
        keypoints: d.keypoints.clone(),
        num_keypoints: d.num_keypoints,
    }
}

// ===========================================================================
// Decoding side: archived view -> spine
// ===========================================================================

fn unpack_segmentation(archived: &ArchivedWireSegmentation) -> Result<Segmentation, PartialError> {
    match archived {
        ArchivedWireSegmentation::Polygons(polys) => {
            let owned: Vec<Vec<f64>> = polys
                .iter()
                .map(|p| p.iter().map(|v| v.to_native()).collect())
                .collect();
            Ok(Segmentation::Polygons(owned))
        }
        ArchivedWireSegmentation::Rle(rle) => {
            let counts = match &rle.counts {
                ArchivedWireSegmentationCounts::Compressed(s) => {
                    SegmentationRleCounts::Compressed(s.as_str().to_string())
                }
                ArchivedWireSegmentationCounts::Uncompressed(c) => {
                    SegmentationRleCounts::Uncompressed(c.iter().map(|v| v.to_native()).collect())
                }
            };
            Ok(Segmentation::Rle(SegmentationRle {
                size: [rle.h.to_native(), rle.w.to_native()],
                counts,
            }))
        }
    }
}

fn unpack_coco_detection(
    archived: &ArchivedWireCocoDetection,
) -> Result<CocoDetection, PartialError> {
    let segmentation = match archived.segmentation.as_ref() {
        Some(s) => Some(unpack_segmentation(s)?),
        None => None,
    };
    let keypoints = archived
        .keypoints
        .as_ref()
        .map(|kps| kps.iter().map(|v| v.to_native()).collect());
    let num_keypoints = archived.num_keypoints.as_ref().map(|v| v.to_native());
    Ok(CocoDetection {
        id: AnnId(archived.id.to_native()),
        image_id: ImageId(archived.image_id.to_native()),
        category_id: CategoryId(archived.category_id.to_native()),
        score: archived.score.to_native(),
        bbox: Bbox {
            x: archived.bbox.x.to_native(),
            y: archived.bbox.y.to_native(),
            w: archived.bbox.w.to_native(),
            h: archived.bbox.h.to_native(),
        },
        area: archived.area.to_native(),
        segmentation,
        keypoints,
        num_keypoints,
    })
}

fn unpack_per_image_eval(
    archived: &ArchivedWirePerImageEval,
) -> Result<PerImageEval, PartialError> {
    let dt_scores: Vec<f64> = archived.dt_scores.iter().map(|v| v.to_native()).collect();
    let dt_matched_shape = [
        archived.dt_matched_shape[0].to_native(),
        archived.dt_matched_shape[1].to_native(),
    ];
    let dt_matched_data: Vec<u8> = archived.dt_matched_data.iter().copied().collect();
    let dt_matched = unpack_array(dt_matched_shape, &dt_matched_data, "dt_matched", |v| v != 0)?;
    let dt_ignore_shape = [
        archived.dt_ignore_shape[0].to_native(),
        archived.dt_ignore_shape[1].to_native(),
    ];
    let dt_ignore_data: Vec<u8> = archived.dt_ignore_data.iter().copied().collect();
    let dt_ignore = unpack_array(dt_ignore_shape, &dt_ignore_data, "dt_ignore", |v| v != 0)?;
    let gt_ignore: Vec<bool> = archived.gt_ignore.iter().map(|&v| v != 0).collect();
    Ok(PerImageEval {
        dt_scores,
        dt_matched,
        dt_ignore,
        gt_ignore,
    })
}

fn unpack_eval_image_meta(
    archived: &ArchivedWireEvalImageMeta,
) -> Result<EvalImageMeta, PartialError> {
    let dt_matches_shape = [
        archived.dt_matches_shape[0].to_native(),
        archived.dt_matches_shape[1].to_native(),
    ];
    let dt_matches = unpack_array(
        dt_matches_shape,
        archived.dt_matches_data.as_slice(),
        "dt_matches",
        |v| v.to_native(),
    )?;
    let gt_matches_shape = [
        archived.gt_matches_shape[0].to_native(),
        archived.gt_matches_shape[1].to_native(),
    ];
    let gt_matches = unpack_array(
        gt_matches_shape,
        archived.gt_matches_data.as_slice(),
        "gt_matches",
        |v| v.to_native(),
    )?;
    Ok(EvalImageMeta {
        image_id: archived.image_id.to_native(),
        category_id: archived.category_id.to_native(),
        area_rng: [
            archived.area_rng[0].to_native(),
            archived.area_rng[1].to_native(),
        ],
        max_det: archived.max_det.to_native() as usize,
        dt_ids: archived.dt_ids.iter().map(|v| v.to_native()).collect(),
        gt_ids: archived.gt_ids.iter().map(|v| v.to_native()).collect(),
        dt_matches,
        gt_matches,
    })
}

// ===========================================================================
// Encode entry point (spine -> bytes)
// ===========================================================================

/// Inputs to [`encode`]. The streaming evaluator lifts the relevant
/// spine state into this struct; we do the rkyv archive + framing.
pub(crate) struct EncodeInput<'a, K: EvalKernel> {
    pub dataset: &'a CocoDataset,
    pub kernel: &'a K,
    pub params: &'a OwnedEvaluateParams,
    pub parity_mode: ParityMode,
    pub rank_id: Option<RankId>,
    pub n_categories: u32,
    pub n_area_ranges: u32,
    pub n_images: u32,
    pub n_detections: u64,
    pub next_dt_id: i64,
    pub seen_images: &'a std::collections::HashSet<i64>,
    pub cells: &'a HashMap<(usize, usize, usize), PerImageEval>,
    pub meta_cells: Option<&'a HashMap<(usize, usize, usize), EvalImageMeta>>,
    pub retained_ious: Option<&'a RetainedIous>,
    pub dets_seen: Option<&'a [CocoDetection]>,
    pub retain_iou: bool,
}

/// Build the paradigm-agnostic envelope header from instance-side
/// evaluator state. The shape fingerprint slots are
/// `(n_categories, n_area_ranges, n_images, retain_iou as u32)`.
pub(crate) fn build_header<K: EvalKernel>(
    input: &EncodeInput<'_, K>,
) -> Result<WireEnvelopeHeader, EvalError> {
    Ok(WireEnvelopeHeader {
        paradigm_kind: ParadigmKind::Instance.as_u8(),
        discriminator: input.kernel.kind().discriminator(),
        parity_mode: encode_parity_mode(input.parity_mode),
        rank_id: input.rank_id,
        dataset_hash: input.dataset.dataset_hash(),
        params_hash: input.params.params_hash()?,
        shape_fingerprint: [
            input.n_categories,
            input.n_area_ranges,
            input.n_images,
            u32::from(input.retain_iou),
        ],
    })
}

/// Lift the spine state in `input` into the rkyv-archivable
/// [`WireInstanceBody`]. All collections are sorted by key so the
/// archive bytes are deterministic — load-bearing for the round-trip
/// equality property and for cross-rank reproducibility under the
/// strict tier (ADR-0031 §"Determinism").
pub(crate) fn build_body<K: EvalKernel>(input: &EncodeInput<'_, K>) -> WireInstanceBody {
    let mut sorted_images: Vec<i64> = input.seen_images.iter().copied().collect();
    sorted_images.sort_unstable();

    let mut cells: Vec<(WireGridKey, WirePerImageEval)> = input
        .cells
        .iter()
        .map(|(&(k, a, i), v)| {
            (
                WireGridKey {
                    k: k as u32,
                    a: a as u32,
                    i: i as u32,
                },
                pack_per_image_eval(v),
            )
        })
        .collect();
    cells.sort_unstable_by(|a, b| a.0.cmp(&b.0));

    let meta_cells = input.meta_cells.map(|map| {
        let mut v: Vec<(WireGridKey, WireEvalImageMeta)> = map
            .iter()
            .map(|(&(k, a, i), m)| {
                (
                    WireGridKey {
                        k: k as u32,
                        a: a as u32,
                        i: i as u32,
                    },
                    pack_eval_image_meta(m),
                )
            })
            .collect();
        v.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        v
    });

    let retained_ious = input.retained_ious.map(retained_ious_to_wire);

    let dets_seen = input
        .dets_seen
        .map(|slice| slice.iter().map(pack_coco_detection).collect());

    WireInstanceBody {
        n_detections: input.n_detections,
        next_dt_id: input.next_dt_id,
        seen_images: sorted_images,
        cells,
        meta_cells,
        retained_ious,
        dets_seen,
    }
}

/// Serialize streaming evaluator state to a partial byte blob.
pub(crate) fn encode<K: EvalKernel>(input: &EncodeInput<'_, K>) -> Result<Vec<u8>, EvalError> {
    let header = build_header(input)?;
    let body = build_body(input);
    let body_archive =
        rkyv::to_bytes::<RkyvError>(&body).map_err(|e| EvalError::PartialFormatMismatch {
            kind: PartialFormatErrorKind::RkyvDecode {
                detail: format!("rkyv::to_bytes(body) failed: {e}"),
            },
        })?;
    Ok(vernier_partial::encode(&header, &body_archive)?)
}

fn retained_ious_to_wire(r: &RetainedIous) -> Vec<WireRetainedIousEntry> {
    // RetainedIous exposes len/get/iter via its public API, but the
    // private inner HashMap is what we need to walk for encoding.
    // We walk via the public iteration helper below.
    let mut out: Vec<WireRetainedIousEntry> = retained_ious_iter(r)
        .map(|(k, i, arr)| {
            let (shape, data) = pack_array(&arr, |v| v);
            WireRetainedIousEntry {
                k: k as u32,
                i: i as u32,
                shape,
                data,
            }
        })
        .collect();
    out.sort_unstable_by(|a, b| (a.k, a.i).cmp(&(b.k, b.i)));
    out
}

fn retained_ious_iter(r: &RetainedIous) -> impl Iterator<Item = (usize, usize, Array2<f64>)> + '_ {
    r.iter().map(|(k, i, view)| (k, i, view.to_owned()))
}

// ===========================================================================
// Cross-partial walks: drain archived bodies into a merge accumulator
// ===========================================================================
//
// Framing + header validation lives in `vernier_partial`; this section
// owns just the instance-paradigm body fold. The merge entry point
// for the streaming evaluator lifts each body's archived view via
// [`vernier_partial::with_validated_envelope`] and hands it to
// [`InstanceMergeAccumulator::ingest`].

/// Build the [`PartialExpectation`] the receiving rank passes to
/// [`vernier_partial::with_validated_envelope`] for instance partials.
pub(crate) fn instance_expectation<K: EvalKernel>(
    dataset: &CocoDataset,
    kernel: &K,
    params: &OwnedEvaluateParams,
    parity_mode: ParityMode,
    n_categories: u32,
    n_area_ranges: u32,
    n_images: u32,
) -> Result<PartialExpectation, EvalError> {
    Ok(PartialExpectation {
        paradigm: ParadigmKind::Instance,
        discriminator: kernel.kind().discriminator(),
        parity_mode: encode_parity_mode(parity_mode),
        dataset_hash: dataset.dataset_hash(),
        params_hash: params.params_hash()?,
        shape_fingerprint: [
            n_categories,
            n_area_ranges,
            n_images,
            u32::from(params.retain_iou),
        ],
        strict_mode: parity_mode == ParityMode::Strict,
    })
}

/// Accumulator that [`crate::StreamingEvaluator::from_partials`] folds into.
/// Owns the merged instance-paradigm state and embeds a
/// [`BaseMergeAccumulator`] for the partition + rank-collision policy.
pub(crate) struct InstanceMergeAccumulator {
    /// Paradigm-agnostic policy state (image-id ownership, rank ids).
    pub base: BaseMergeAccumulator,
    pub n_detections: usize,
    pub next_dt_id: i64,
    pub cells: HashMap<(usize, usize, usize), PerImageEval>,
    pub meta_cells: HashMap<(usize, usize, usize), EvalImageMeta>,
    pub retained_ious_map: HashMap<(usize, usize), Array2<f64>>,
    pub dets_seen: Vec<CocoDetection>,
    pub retain_iou: bool,
}

impl InstanceMergeAccumulator {
    pub(crate) fn new(strict: bool) -> Self {
        Self {
            base: BaseMergeAccumulator::new(strict),
            n_detections: 0,
            next_dt_id: 1,
            cells: HashMap::new(),
            meta_cells: HashMap::new(),
            retained_ious_map: HashMap::new(),
            dets_seen: Vec::new(),
            retain_iou: false,
        }
    }

    pub(crate) fn set_retain_iou(&mut self, retain_iou: bool) {
        self.retain_iou = retain_iou;
    }

    /// Drain one validated envelope view into this accumulator. The
    /// view's `body_archive` slice is copied into an aligned buffer
    /// then accessed as [`ArchivedWireInstanceBody`].
    ///
    /// Returns errors for partition overlap (ADR-0031 D1) and
    /// strict-mode rank collisions. The outer caller propagates via
    /// `?` through the `From<PartialError> for EvalError` impl.
    pub(crate) fn ingest(&mut self, view: &ValidatedView<'_>) -> Result<(), PartialError> {
        let rank_id = vernier_partial::rank_id_from_archive(view.header);
        self.base.ingest_rank_id(rank_id)?;

        let mut aligned: rkyv::util::AlignedVec<16> =
            rkyv::util::AlignedVec::with_capacity(view.body_archive.len());
        aligned.extend_from_slice(view.body_archive);
        let archived =
            rkyv::access::<ArchivedWireInstanceBody, RkyvError>(&aligned).map_err(|e| {
                PartialError::Format {
                    kind: PartialFormatErrorKind::RkyvDecode {
                        detail: format!("rkyv::access(body) failed: {e}"),
                    },
                }
            })?;

        // Image-id disjointness — the partition rule.
        self.base
            .ingest_image_ids(rank_id, archived.seen_images.iter().map(|v| v.to_native()))?;

        self.n_detections += archived.n_detections.to_native() as usize;
        let candidate_next = archived.next_dt_id.to_native();
        if candidate_next > self.next_dt_id {
            self.next_dt_id = candidate_next;
        }

        for entry in archived.cells.iter() {
            let key = &entry.0;
            let value = &entry.1;
            let triple = (
                key.k.to_native() as usize,
                key.a.to_native() as usize,
                key.i.to_native() as usize,
            );
            let p = unpack_per_image_eval(value)?;
            self.cells.insert(triple, p);
        }

        if let Some(metas) = archived.meta_cells.as_ref() {
            for entry in metas.iter() {
                let key = &entry.0;
                let value = &entry.1;
                let triple = (
                    key.k.to_native() as usize,
                    key.a.to_native() as usize,
                    key.i.to_native() as usize,
                );
                let m = unpack_eval_image_meta(value)?;
                self.meta_cells.insert(triple, m);
            }
        }

        if let Some(ious) = archived.retained_ious.as_ref() {
            for entry in ious.iter() {
                let shape = [entry.shape[0].to_native(), entry.shape[1].to_native()];
                let arr = unpack_array(shape, entry.data.as_slice(), "retained_iou", |v| {
                    v.to_native()
                })?;
                self.retained_ious_map.insert(
                    (entry.k.to_native() as usize, entry.i.to_native() as usize),
                    arr,
                );
            }
        }

        if let Some(dets) = archived.dets_seen.as_ref() {
            for d in dets.iter() {
                self.dets_seen.push(unpack_coco_detection(d)?);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instance_body_round_trips_through_envelope() {
        // Encode an empty body, ship it through the envelope, decode.
        // Pins the contract that vernier-partial encodes any
        // body-archive bytes opaquely and the instance-side decode
        // reads them back identically.
        let body = WireInstanceBody {
            n_detections: 0,
            next_dt_id: 1,
            seen_images: vec![],
            cells: vec![],
            meta_cells: None,
            retained_ious: None,
            dets_seen: None,
        };
        let body_archive = rkyv::to_bytes::<RkyvError>(&body).unwrap();
        let header = WireEnvelopeHeader {
            paradigm_kind: ParadigmKind::Instance.as_u8(),
            discriminator: 0,
            parity_mode: PARITY_CORRECTED,
            rank_id: None,
            dataset_hash: [0xAB; 32],
            params_hash: [0xCD; 32],
            shape_fingerprint: [80, 4, 5000, 0],
        };
        let bytes = vernier_partial::encode(&header, &body_archive).unwrap();

        let exp = PartialExpectation {
            paradigm: ParadigmKind::Instance,
            discriminator: 0,
            parity_mode: PARITY_CORRECTED,
            dataset_hash: [0xAB; 32],
            params_hash: [0xCD; 32],
            shape_fingerprint: [80, 4, 5000, 0],
            strict_mode: false,
        };
        let mut acc = InstanceMergeAccumulator::new(false);
        vernier_partial::with_validated_envelope(&bytes, &exp, |view| acc.ingest(&view)).unwrap();
        assert_eq!(acc.n_detections, 0);
    }
}
