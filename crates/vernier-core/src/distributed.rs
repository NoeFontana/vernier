//! Distributed evaluation: partial wire format and merge orchestration.
//!
//! Per ADR-0031, every rank in a multi-process eval runs its own
//! [`StreamingEvaluator`], serializes its post-`update` state to a
//! byte blob via [`StreamingEvaluator::finalize_to_partial`] (or
//! [`StreamingEvaluator::snapshot_to_partial`] mid-stream), then the
//! head rank reconstructs an evaluator equivalent to a batch run over
//! the union of all partials' submitted detections via
//! [`StreamingEvaluator::from_partials`].
//!
//! This module is the wire format. It defines:
//!
//! - The [`RankId`] type alias and the wire-format constants
//!   ([`MAGIC`], [`FORMAT_VERSION`]) that frame every blob.
//! - Private wire-form shadow structs that mirror the spine state
//!   ([`crate::PerImageEval`], [`crate::EvalImageMeta`], etc.) but
//!   with `rkyv::Archive` / `rkyv::Serialize` / `rkyv::Deserialize`
//!   derives. The spine types themselves stay free of rkyv per
//!   ADR-0005 — conversion happens at the encode / decode boundary.
//! - The [`encode`] / [`decode`] entry points and the cheapest-first
//!   [`with_validated_partial`] pipeline that returns one of the
//!   [`crate::EvalError::PartialFormatMismatch`] /
//!   [`crate::EvalError::PartialDatasetMismatch`] /
//!   [`crate::EvalError::PartialParamsMismatch`] variants on each
//!   class of failure.
//!
//! ## Wire framing
//!
//! Every partial is laid out as:
//!
//! ```text
//! [ 4 bytes : MAGIC = b"VRPS" ]
//! [ 1 byte  : FORMAT_VERSION  ]
//! [ N bytes : rkyv archive of WirePartial ]
//! [ 4 bytes : CRC32(IEEE) over the preceding (5 + N) bytes ]
//! ```
//!
//! Magic + version sit *outside* the rkyv archive on purpose: those
//! checks are cheap and must succeed before we try to read the
//! archive (which would fail with an opaque error on a stale-format
//! payload otherwise). The CRC catches transport corruption and
//! truncation that the rkyv archive validator would not catch
//! cleanly.
//!
//! ## Determinism
//!
//! The cells / meta_cells / retained_ious stores in the runtime are
//! `HashMap`s; their iteration order is not stable. The wire form
//! sorts every collection by key before archiving so two encodings
//! of equal state produce byte-identical blobs. This matters for
//! [`StreamingEvaluator::checkpoint`] / [`crate::StreamingEvaluator::restore`]
//! round-trip equality, and for cross-rank reproducibility under the
//! strict-tier `(rank_id, local_position)` ordering reserved in
//! ADR-0031.

use std::collections::HashMap;

use ndarray::Array2;
use rkyv::rancor::Error as RkyvError;

use crate::accumulate::PerImageEval;
use crate::dataset::{AnnId, Bbox, CategoryId, CocoDataset, CocoDetection, ImageId};
use crate::error::PartialFormatErrorKind;
use crate::evaluate::{EvalImageMeta, EvalKernel, KernelKind, OwnedEvaluateParams};
use crate::parity::ParityMode;
use crate::segmentation::{Segmentation, SegmentationRle, SegmentationRleCounts};
use crate::tables::RetainedIous;
use crate::EvalError;

// ===========================================================================
// Public surface
// ===========================================================================

/// Identifier for one rank in a multi-process eval. Strict-mode merge
/// uses `(rank_id, local_position)` as the global stream-order
/// tiebreak; corrected mode ignores it.
pub type RankId = u32;

/// Wire-format magic: ASCII `"VRPS"` (vernier partial state). Every
/// valid partial starts with these four bytes.
pub const MAGIC: [u8; 4] = *b"VRPS";

/// Wire-format version. Bumped on any breaking change to the
/// archived layout. Old versions are refused at decode with a typed
/// [`PartialFormatErrorKind::WrongVersion`].
pub const FORMAT_VERSION: u8 = 1;

/// Minimum bytes a partial must carry to even attempt parsing:
/// 4 magic + 1 version + 4 CRC.
const MIN_PARTIAL_BYTES: usize = MAGIC.len() + 1 + 4;

// ===========================================================================
// Wire-form shadow types
// ===========================================================================
//
// All collections are `Vec` (sorted by key where applicable) — never
// `HashMap` — so the archived bytes are deterministic. Array2 fields
// flatten to `(shape: [u32; 2], data: Vec<T>)` so rkyv can archive
// them without needing a `with` adapter on `ndarray`.

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

#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub(crate) struct WirePartialHeader {
    parity_mode: u8,
    kernel_kind: u8,
    retain_iou: u8,
    rank_id: Option<u32>,
    n_categories: u32,
    n_area_ranges: u32,
    n_images: u32,
    dataset_hash: [u8; 32],
    params_hash: [u8; 32],
}

#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub(crate) struct WirePartialBody {
    n_detections: u64,
    next_dt_id: i64,
    seen_images: Vec<i64>,
    cells: Vec<(WireGridKey, WirePerImageEval)>,
    meta_cells: Option<Vec<(WireGridKey, WireEvalImageMeta)>>,
    retained_ious: Option<Vec<WireRetainedIousEntry>>,
    dets_seen: Option<Vec<WireCocoDetection>>,
}

#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub(crate) struct WirePartial {
    pub(crate) header: WirePartialHeader,
    pub(crate) body: WirePartialBody,
}

// ===========================================================================
// Encoding side: spine -> wire
// ===========================================================================

const PARITY_STRICT: u8 = 0;
const PARITY_CORRECTED: u8 = 1;

const KERNEL_BBOX: u8 = 0;
const KERNEL_SEGM: u8 = 1;
const KERNEL_BOUNDARY: u8 = 2;
const KERNEL_KEYPOINTS: u8 = 3;

fn encode_parity_mode(m: ParityMode) -> u8 {
    match m {
        ParityMode::Strict => PARITY_STRICT,
        ParityMode::Corrected => PARITY_CORRECTED,
    }
}

fn decode_parity_mode(b: u8) -> Option<ParityMode> {
    match b {
        PARITY_STRICT => Some(ParityMode::Strict),
        PARITY_CORRECTED => Some(ParityMode::Corrected),
        _ => None,
    }
}

fn encode_kernel_kind(k: KernelKind) -> u8 {
    match k {
        KernelKind::Bbox => KERNEL_BBOX,
        KernelKind::Segm => KERNEL_SEGM,
        KernelKind::Boundary => KERNEL_BOUNDARY,
        KernelKind::Keypoints => KERNEL_KEYPOINTS,
    }
}

fn pack_bool_array(arr: &Array2<bool>) -> ([u32; 2], Vec<u8>) {
    let (rows, cols) = arr.dim();
    let shape = [rows as u32, cols as u32];
    let data: Vec<u8> = arr.iter().map(|&b| u8::from(b)).collect();
    (shape, data)
}

fn unpack_bool_array(
    shape: [u32; 2],
    data: &[u8],
    field: &'static str,
) -> Result<Array2<bool>, EvalError> {
    let rows = shape[0] as usize;
    let cols = shape[1] as usize;
    if data.len() != rows.saturating_mul(cols) {
        return Err(EvalError::PartialFormatMismatch {
            kind: PartialFormatErrorKind::RkyvDecode {
                detail: format!(
                    "{field} shape {rows}x{cols} doesn't match data len {}",
                    data.len()
                ),
            },
        });
    }
    let bools: Vec<bool> = data.iter().map(|&v| v != 0).collect();
    Array2::from_shape_vec((rows, cols), bools).map_err(|e| EvalError::PartialFormatMismatch {
        kind: PartialFormatErrorKind::RkyvDecode {
            detail: format!("{field} from_shape_vec: {e}"),
        },
    })
}

fn pack_i64_array(arr: &Array2<i64>) -> ([u32; 2], Vec<i64>) {
    let (rows, cols) = arr.dim();
    ([rows as u32, cols as u32], arr.iter().copied().collect())
}

fn unpack_i64_array(
    shape: [u32; 2],
    data: Vec<i64>,
    field: &'static str,
) -> Result<Array2<i64>, EvalError> {
    let rows = shape[0] as usize;
    let cols = shape[1] as usize;
    if data.len() != rows.saturating_mul(cols) {
        return Err(EvalError::PartialFormatMismatch {
            kind: PartialFormatErrorKind::RkyvDecode {
                detail: format!(
                    "{field} shape {rows}x{cols} doesn't match data len {}",
                    data.len()
                ),
            },
        });
    }
    Array2::from_shape_vec((rows, cols), data).map_err(|e| EvalError::PartialFormatMismatch {
        kind: PartialFormatErrorKind::RkyvDecode {
            detail: format!("{field} from_shape_vec: {e}"),
        },
    })
}

fn pack_f64_array(arr: &Array2<f64>) -> ([u32; 2], Vec<f64>) {
    let (rows, cols) = arr.dim();
    ([rows as u32, cols as u32], arr.iter().copied().collect())
}

fn pack_per_image_eval(p: &PerImageEval) -> WirePerImageEval {
    let (dt_matched_shape, dt_matched_data) = pack_bool_array(&p.dt_matched);
    let (dt_ignore_shape, dt_ignore_data) = pack_bool_array(&p.dt_ignore);
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
    let (dt_matches_shape, dt_matches_data) = pack_i64_array(&m.dt_matches);
    let (gt_matches_shape, gt_matches_data) = pack_i64_array(&m.gt_matches);
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

fn unpack_segmentation(archived: &ArchivedWireSegmentation) -> Result<Segmentation, EvalError> {
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

fn unpack_coco_detection(archived: &ArchivedWireCocoDetection) -> Result<CocoDetection, EvalError> {
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

fn unpack_per_image_eval(archived: &ArchivedWirePerImageEval) -> Result<PerImageEval, EvalError> {
    let dt_scores: Vec<f64> = archived.dt_scores.iter().map(|v| v.to_native()).collect();
    let dt_matched_shape = [
        archived.dt_matched_shape[0].to_native(),
        archived.dt_matched_shape[1].to_native(),
    ];
    let dt_matched_data: Vec<u8> = archived.dt_matched_data.iter().copied().collect();
    let dt_matched = unpack_bool_array(dt_matched_shape, &dt_matched_data, "dt_matched")?;
    let dt_ignore_shape = [
        archived.dt_ignore_shape[0].to_native(),
        archived.dt_ignore_shape[1].to_native(),
    ];
    let dt_ignore_data: Vec<u8> = archived.dt_ignore_data.iter().copied().collect();
    let dt_ignore = unpack_bool_array(dt_ignore_shape, &dt_ignore_data, "dt_ignore")?;
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
) -> Result<EvalImageMeta, EvalError> {
    let dt_matches_data: Vec<i64> = archived
        .dt_matches_data
        .iter()
        .map(|v| v.to_native())
        .collect();
    let dt_matches_shape = [
        archived.dt_matches_shape[0].to_native(),
        archived.dt_matches_shape[1].to_native(),
    ];
    let dt_matches = unpack_i64_array(dt_matches_shape, dt_matches_data, "dt_matches")?;
    let gt_matches_data: Vec<i64> = archived
        .gt_matches_data
        .iter()
        .map(|v| v.to_native())
        .collect();
    let gt_matches_shape = [
        archived.gt_matches_shape[0].to_native(),
        archived.gt_matches_shape[1].to_native(),
    ];
    let gt_matches = unpack_i64_array(gt_matches_shape, gt_matches_data, "gt_matches")?;
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

/// Serialize streaming evaluator state to a partial byte blob.
pub(crate) fn encode<K: EvalKernel>(input: &EncodeInput<'_, K>) -> Result<Vec<u8>, EvalError> {
    let header = WirePartialHeader {
        parity_mode: encode_parity_mode(input.parity_mode),
        kernel_kind: encode_kernel_kind(input.kernel.kind()),
        retain_iou: u8::from(input.retain_iou),
        rank_id: input.rank_id,
        n_categories: input.n_categories,
        n_area_ranges: input.n_area_ranges,
        n_images: input.n_images,
        dataset_hash: input.dataset.dataset_hash(),
        params_hash: input.params.params_hash()?,
    };

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

    let body = WirePartialBody {
        n_detections: input.n_detections,
        next_dt_id: input.next_dt_id,
        seen_images: sorted_images,
        cells,
        meta_cells,
        retained_ious,
        dets_seen,
    };

    let partial = WirePartial { header, body };

    let archive_bytes =
        rkyv::to_bytes::<RkyvError>(&partial).map_err(|e| EvalError::PartialFormatMismatch {
            kind: PartialFormatErrorKind::RkyvDecode {
                detail: format!("rkyv::to_bytes failed: {e}"),
            },
        })?;

    let mut out = Vec::with_capacity(MAGIC.len() + 1 + archive_bytes.len() + 4);
    out.extend_from_slice(&MAGIC);
    out.push(FORMAT_VERSION);
    out.extend_from_slice(&archive_bytes);
    let crc = crc32fast::hash(&out);
    out.extend_from_slice(&crc.to_le_bytes());
    Ok(out)
}

fn retained_ious_to_wire(r: &RetainedIous) -> Vec<WireRetainedIousEntry> {
    // RetainedIous exposes len/get/iter via its public API, but the
    // private inner HashMap is what we need to walk for encoding.
    // We walk via the public iteration helper below.
    let mut out: Vec<WireRetainedIousEntry> = retained_ious_iter(r)
        .map(|(k, i, arr)| {
            let (shape, data) = pack_f64_array(&arr);
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
// Decode entry point (bytes -> archived view + validation)
// ===========================================================================

/// Validate framing (length, magic, version, CRC) and return the
/// rkyv archive bytes (the payload between header and CRC footer).
fn validate_framing(bytes: &[u8]) -> Result<&[u8], EvalError> {
    if bytes.len() < MIN_PARTIAL_BYTES {
        return Err(EvalError::PartialFormatMismatch {
            kind: PartialFormatErrorKind::TooShort {
                observed: bytes.len(),
                minimum: MIN_PARTIAL_BYTES,
            },
        });
    }
    let magic: [u8; 4] = bytes[..4]
        .try_into()
        .map_err(|_| EvalError::PartialFormatMismatch {
            kind: PartialFormatErrorKind::TooShort {
                observed: bytes.len(),
                minimum: MIN_PARTIAL_BYTES,
            },
        })?;
    if magic != MAGIC {
        return Err(EvalError::PartialFormatMismatch {
            kind: PartialFormatErrorKind::WrongMagic { found: magic },
        });
    }
    let version = bytes[4];
    if version != FORMAT_VERSION {
        return Err(EvalError::PartialFormatMismatch {
            kind: PartialFormatErrorKind::WrongVersion {
                expected: FORMAT_VERSION,
                found: version,
            },
        });
    }
    let split = bytes.len() - 4;
    let stored_crc = u32::from_le_bytes(bytes[split..].try_into().map_err(|_| {
        EvalError::PartialFormatMismatch {
            kind: PartialFormatErrorKind::Crc,
        }
    })?);
    let actual_crc = crc32fast::hash(&bytes[..split]);
    if stored_crc != actual_crc {
        return Err(EvalError::PartialFormatMismatch {
            kind: PartialFormatErrorKind::Crc,
        });
    }
    Ok(&bytes[5..split])
}

/// Expectation passed to [`with_validated_partial`] so each
/// header-field check can name its own specific error.
pub(crate) struct PartialExpectation<'a> {
    pub parity_mode: ParityMode,
    pub kernel_kind: KernelKind,
    pub retain_iou: bool,
    pub n_categories: u32,
    pub n_area_ranges: u32,
    pub n_images: u32,
    pub dataset_hash: [u8; 32],
    pub params_hash: [u8; 32],
    /// Used by callers that want to build error messages without
    /// recomputing the live dataset's hash.
    pub _phantom: std::marker::PhantomData<&'a ()>,
}

/// Validate a partial blob's framing + header fields against the
/// receiving rank's live config and run a callback on the validated
/// archived body.
///
/// Why callback-based: the rkyv archived view requires 8-byte
/// alignment, but the input `&[u8]` slice (as produced by user
/// transport) is not guaranteed to be aligned. We copy the archive
/// bytes into an [`rkyv::util::AlignedVec`] before [`rkyv::access`]
/// so the pointer reads inside the archive are well-aligned. The
/// AlignedVec is the borrow source for the archived view, so the
/// callback runs while it's still in scope.
pub(crate) fn with_validated_partial<R>(
    bytes: &[u8],
    expected: &PartialExpectation<'_>,
    body: impl FnOnce(&ArchivedWirePartial) -> Result<R, EvalError>,
) -> Result<R, EvalError> {
    let archive_bytes = validate_framing(bytes)?;
    // 16-byte alignment covers every primitive rkyv writes on x86_64
    // and aarch64. (rkyv defaults to 16 on these targets.)
    let mut aligned: rkyv::util::AlignedVec<16> =
        rkyv::util::AlignedVec::with_capacity(archive_bytes.len());
    aligned.extend_from_slice(archive_bytes);
    let archived = rkyv::access::<ArchivedWirePartial, RkyvError>(&aligned).map_err(|e| {
        EvalError::PartialFormatMismatch {
            kind: PartialFormatErrorKind::RkyvDecode {
                detail: format!("rkyv::access failed: {e}"),
            },
        }
    })?;
    validate_header_fields(archived, expected)?;
    body(archived)
}

fn validate_header_fields(
    archived: &ArchivedWirePartial,
    expected: &PartialExpectation<'_>,
) -> Result<(), EvalError> {
    let h = &archived.header;
    let kernel_kind = h.kernel_kind;
    if kernel_kind != encode_kernel_kind(expected.kernel_kind) {
        return Err(EvalError::PartialFormatMismatch {
            kind: PartialFormatErrorKind::KernelMismatch {
                expected: encode_kernel_kind(expected.kernel_kind),
                found: kernel_kind,
            },
        });
    }
    let parity_mode = h.parity_mode;
    let parity = decode_parity_mode(parity_mode).ok_or(EvalError::PartialFormatMismatch {
        kind: PartialFormatErrorKind::RkyvDecode {
            detail: format!("unknown parity_mode discriminant: {parity_mode}"),
        },
    })?;
    if parity != expected.parity_mode {
        return Err(EvalError::PartialFormatMismatch {
            kind: PartialFormatErrorKind::ParityMismatch {
                expected: expected.parity_mode,
                found: parity,
            },
        });
    }
    let retain_iou = h.retain_iou != 0;
    if retain_iou != expected.retain_iou {
        return Err(EvalError::PartialFormatMismatch {
            kind: PartialFormatErrorKind::RetainIouMismatch {
                expected: expected.retain_iou,
                found: retain_iou,
            },
        });
    }
    let nc = h.n_categories.to_native();
    let na = h.n_area_ranges.to_native();
    let ni = h.n_images.to_native();
    if nc != expected.n_categories || na != expected.n_area_ranges || ni != expected.n_images {
        return Err(EvalError::PartialFormatMismatch {
            kind: PartialFormatErrorKind::GridMismatch {
                detail: format!(
                    "expected ({}/{}/{}), got ({nc}/{na}/{ni})",
                    expected.n_categories, expected.n_area_ranges, expected.n_images
                ),
            },
        });
    }
    let actual_dataset_hash: [u8; 32] = h.dataset_hash;
    if actual_dataset_hash != expected.dataset_hash {
        return Err(EvalError::PartialDatasetMismatch {
            expected: expected.dataset_hash,
            actual: actual_dataset_hash,
        });
    }
    let actual_params_hash: [u8; 32] = h.params_hash;
    if actual_params_hash != expected.params_hash {
        return Err(EvalError::PartialParamsMismatch {
            expected: expected.params_hash,
            actual: actual_params_hash,
        });
    }
    Ok(())
}

// ===========================================================================
// Cross-partial walks: drain archived bodies into a merge accumulator
// ===========================================================================

/// Accumulator that [`StreamingEvaluator::from_partials`] folds into.
/// Owns the merged state and accepts archived bodies one at a time.
pub(crate) struct MergeAccumulator {
    pub n_detections: usize,
    pub next_dt_id: i64,
    pub seen_images: std::collections::HashSet<i64>,
    pub seen_image_to_rank: HashMap<i64, RankId>,
    pub seen_rank_ids: std::collections::HashSet<RankId>,
    pub cells: HashMap<(usize, usize, usize), PerImageEval>,
    pub meta_cells: HashMap<(usize, usize, usize), EvalImageMeta>,
    pub retained_ious_map: HashMap<(usize, usize), Array2<f64>>,
    pub dets_seen: Vec<CocoDetection>,
    pub retain_iou: bool,
    pub strict: bool,
}

impl MergeAccumulator {
    pub(crate) fn new(strict: bool) -> Self {
        Self {
            n_detections: 0,
            next_dt_id: 1,
            seen_images: std::collections::HashSet::new(),
            seen_image_to_rank: HashMap::new(),
            seen_rank_ids: std::collections::HashSet::new(),
            cells: HashMap::new(),
            meta_cells: HashMap::new(),
            retained_ious_map: HashMap::new(),
            dets_seen: Vec::new(),
            retain_iou: false,
            strict,
        }
    }

    pub(crate) fn set_retain_iou(&mut self, retain_iou: bool) {
        self.retain_iou = retain_iou;
    }

    /// Drain one archived partial into this accumulator.
    ///
    /// Returns errors for partition overlap (ADR-0031 D1) and strict-
    /// mode rank collisions (D8).
    pub(crate) fn ingest(&mut self, archived: &ArchivedWirePartial) -> Result<(), EvalError> {
        // Strict-mode rank-id distinctness.
        let rank_id = archived.header.rank_id.as_ref().map(|v| v.to_native());
        if self.strict {
            if let Some(rid) = rank_id {
                if !self.seen_rank_ids.insert(rid) {
                    return Err(EvalError::PartialRankCollision { rank_id: rid });
                }
            }
        }

        // Image-id disjointness — the partition rule. Records every
        // image_id with the rank that owns it; on collision, errors
        // with both rank ids.
        for img in archived.body.seen_images.iter() {
            let id = img.to_native();
            let owner = rank_id.unwrap_or(u32::MAX);
            if let Some(&prev) = self.seen_image_to_rank.get(&id) {
                let (a, b) = if prev <= owner {
                    (prev, owner)
                } else {
                    (owner, prev)
                };
                return Err(EvalError::PartialPartitionOverlap {
                    rank_a: a,
                    rank_b: b,
                    image_id: id,
                });
            }
            self.seen_image_to_rank.insert(id, owner);
            self.seen_images.insert(id);
        }

        self.n_detections += archived.body.n_detections.to_native() as usize;
        let candidate_next = archived.body.next_dt_id.to_native();
        if candidate_next > self.next_dt_id {
            self.next_dt_id = candidate_next;
        }

        for entry in archived.body.cells.iter() {
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

        if let Some(metas) = archived.body.meta_cells.as_ref() {
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

        if let Some(ious) = archived.body.retained_ious.as_ref() {
            for entry in ious.iter() {
                let shape = [entry.shape[0].to_native(), entry.shape[1].to_native()];
                let data: Vec<f64> = entry.data.iter().map(|v| v.to_native()).collect();
                let arr = pack_f64_array_unpack(shape, data, "retained_iou")?;
                self.retained_ious_map.insert(
                    (entry.k.to_native() as usize, entry.i.to_native() as usize),
                    arr,
                );
            }
        }

        if let Some(dets) = archived.body.dets_seen.as_ref() {
            for d in dets.iter() {
                self.dets_seen.push(unpack_coco_detection(d)?);
            }
        }

        Ok(())
    }
}

fn pack_f64_array_unpack(
    shape: [u32; 2],
    data: Vec<f64>,
    field: &'static str,
) -> Result<Array2<f64>, EvalError> {
    let rows = shape[0] as usize;
    let cols = shape[1] as usize;
    if data.len() != rows.saturating_mul(cols) {
        return Err(EvalError::PartialFormatMismatch {
            kind: PartialFormatErrorKind::RkyvDecode {
                detail: format!(
                    "{field} shape {rows}x{cols} doesn't match data len {}",
                    data.len()
                ),
            },
        });
    }
    Array2::from_shape_vec((rows, cols), data).map_err(|e| EvalError::PartialFormatMismatch {
        kind: PartialFormatErrorKind::RkyvDecode {
            detail: format!("{field} from_shape_vec: {e}"),
        },
    })
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_framing_rejects_too_short() {
        let bytes = b"VRP";
        let err = validate_framing(bytes).unwrap_err();
        assert!(matches!(
            err,
            EvalError::PartialFormatMismatch {
                kind: PartialFormatErrorKind::TooShort { .. }
            }
        ));
    }

    #[test]
    fn validate_framing_rejects_wrong_magic() {
        let mut bytes = vec![0u8; MIN_PARTIAL_BYTES];
        bytes[0..4].copy_from_slice(b"FAKE");
        let err = validate_framing(&bytes).unwrap_err();
        assert!(matches!(
            err,
            EvalError::PartialFormatMismatch {
                kind: PartialFormatErrorKind::WrongMagic { .. }
            }
        ));
    }

    #[test]
    fn validate_framing_rejects_wrong_version() {
        let mut bytes = vec![0u8; MIN_PARTIAL_BYTES];
        bytes[0..4].copy_from_slice(&MAGIC);
        bytes[4] = 99;
        // CRC will also mismatch but version check is earlier.
        let err = validate_framing(&bytes).unwrap_err();
        assert!(matches!(
            err,
            EvalError::PartialFormatMismatch {
                kind: PartialFormatErrorKind::WrongVersion { .. }
            }
        ));
    }

    #[test]
    fn validate_framing_rejects_bad_crc() {
        let mut bytes = vec![0u8; MIN_PARTIAL_BYTES];
        bytes[0..4].copy_from_slice(&MAGIC);
        bytes[4] = FORMAT_VERSION;
        // Last 4 bytes are a wrong CRC (zero).
        let err = validate_framing(&bytes).unwrap_err();
        assert!(matches!(
            err,
            EvalError::PartialFormatMismatch {
                kind: PartialFormatErrorKind::Crc
            }
        ));
    }

    #[test]
    fn validate_framing_accepts_round_trip() {
        // Hand-build a valid framing around an empty payload.
        let payload: &[u8] = &[];
        let mut bytes = Vec::with_capacity(MIN_PARTIAL_BYTES);
        bytes.extend_from_slice(&MAGIC);
        bytes.push(FORMAT_VERSION);
        bytes.extend_from_slice(payload);
        let crc = crc32fast::hash(&bytes);
        bytes.extend_from_slice(&crc.to_le_bytes());
        let extracted = validate_framing(&bytes).unwrap();
        assert!(extracted.is_empty());
    }
}
