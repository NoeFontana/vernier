//! Native PNG decode + RGB→id fold + S3 area marginal in one pass.
//!
//! Replaces the Pillow → numpy.array(uint32) → `R + 256·G + 256²·B`
//! pipeline that the bench harness drives on the main thread. The
//! Python pipeline materializes (1) a uint8 RGB ndarray, (2) a uint32
//! RGB ndarray (via `np.array(img, dtype=np.uint32)`), and (3) the
//! uint32 id label map (via the three-channel arithmetic). The Rust
//! pipeline reads decoder bytes into a single uint8 buffer and emits
//! the uint32 label map in one walk over those bytes — saves two
//! buffer allocations and the Python interpreter dispatch on the
//! per-pixel arithmetic.
//!
//! On the DT side the same walk fuses the **S3** marginal-area
//! recompute (panopticapi `evaluation.py:102` overwrites JSON `area`
//! with the PNG marginal) and the **S1** unknown-id check (`KeyError`
//! at `evaluation.py:96-101`) and the **S11** declared-but-absent
//! check. GT side just decodes — areas come from JSON per quirk
//! **S4**, and PNG-id-not-in-segments_info is silently ignored per
//! quirk **S8**.

use std::cell::RefCell;
use std::io::Cursor;

use rustc_hash::FxHashMap;

use crate::dataset::{ImageEntry, ImageId, SegmentInfo};
use crate::error::PanopticError;
use crate::parity::PANOPTIC_VOID;

/// Cap on the dense `id → segment_index` lookup table size before
/// falling back to a hashmap. 1 M entries × 4 bytes = 4 MiB scratch
/// per image — well within streaming-runner RSS limits, comfortably
/// covers every realistic panoptic dataset (COCO ids fit in ≤6 digits;
/// the encoded space is 256³ but no real dataset uses more than a
/// small fraction).
const DENSE_LOOKUP_MAX_ID: u32 = 1_000_000;

thread_local! {
    /// Per-thread reusable buffer for the DT-side `id → segment_index`
    /// dense lookup. Sized at most `DENSE_LOOKUP_MAX_ID + 1` (~4 MiB);
    /// in practice COCO panoptic images cap out around max_id ≈ 200K
    /// (~800 KB). Reusing the allocation across calls saves the
    /// `Vec::resize(_, u32::MAX)` zero pass per call (~50 µs at memory
    /// bandwidth limits) — the per-call cost otherwise sits on the
    /// hot path.
    ///
    /// Thread-local is the right scope: `submit_png` and `update_png`
    /// are called from the FFI thread; the kernel worker thread (in
    /// the Background path) consumes the resulting `ImageEntry` but
    /// does not call `decode_panoptic_png`. Each thread gets its own
    /// buffer, no synchronization needed.
    static DT_LOOKUP_SCRATCH: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
}

/// Decode a panoptic-encoded RGB PNG and build an [`ImageEntry`] in a
/// single pass. `segments_list` carries the segments_info JSON for
/// this image (already parsed by the FFI layer). `side` is `"gt"` or
/// `"dt"`; the DT path fuses the S3 area recompute + S1/S11
/// validation, the GT path does neither.
///
/// Errors:
/// - [`PanopticError::Png`] — `png` crate decode failure (corrupt or
///   truncated bytes, unsupported chunk).
/// - [`PanopticError::NonRgbPng`] — input is not 8-bit RGB (R2).
/// - [`PanopticError::DuplicateSegmentId`] — segments_info has two
///   entries with the same id (S7, corrected mode).
/// - [`PanopticError::UnknownPredSegmentId`] — DT-side only; PNG
///   carries an id not in segments_info (S1).
/// - [`PanopticError::MissingPredSegmentInPng`] — DT-side only;
///   segments_info declares an id absent from the PNG (S11).
pub fn decode_panoptic_png(
    image_id: ImageId,
    bytes: &[u8],
    segments_list: Vec<SegmentInfo>,
    side: &'static str,
) -> Result<ImageEntry, PanopticError> {
    if side == "dt" {
        decode_dt(image_id, bytes, segments_list)
    } else {
        decode_gt(image_id, bytes, segments_list)
    }
}

/// Validated 8-bit-RGB row-streaming reader plus image dimensions.
struct PngStream<'a> {
    height: u32,
    width: u32,
    reader: png::Reader<Cursor<&'a [u8]>>,
}

/// Initialize a row-streaming PNG reader for an 8-bit RGB panoptic
/// image. Returns the reader plus the validated dimensions. Other
/// color types (Grayscale / Indexed / GrayscaleAlpha / RGBA) are
/// rejected at the FFI boundary per quirk **R2**.
fn init_reader(image_id: ImageId, bytes: &[u8]) -> Result<PngStream<'_>, PanopticError> {
    let decoder = png::Decoder::new(Cursor::new(bytes));
    let reader = decoder
        .read_info()
        .map_err(|e| PanopticError::Png(format!("image_id={image_id}: read_info: {e}")))?;
    let info = reader.info();
    let color_type = info.color_type;
    let bit_depth = info.bit_depth;
    if color_type != png::ColorType::Rgb || bit_depth != png::BitDepth::Eight {
        return Err(PanopticError::NonRgbPng {
            image_id,
            mode: color_type_label(color_type, bit_depth),
        });
    }
    let height = info.height;
    let width = info.width;
    Ok(PngStream {
        height,
        width,
        reader,
    })
}

/// GT-side decode: RGB→id straight into `label_map`, row by row,
/// without ever materializing the full `3 * H * W` RGB buffer. No
/// segment-id lookup, no per-pixel area accumulation, no S1/S11
/// validation — quirk **S8** is "GT JSON-extras silently kept;
/// matching iterates the histogram, not the dict", and quirk **S4**
/// is "GT areas come from JSON, not from the PNG marginal." So the
/// GT inner loop is just `id = R | G<<8 | B<<16; push`.
///
/// The S7 duplicate-id check still runs against `segments_list` so
/// the corrected disposition rejects malformed input — that walk is
/// over the (small) segments JSON, not the (large) pixel buffer.
fn decode_gt(
    image_id: ImageId,
    bytes: &[u8],
    segments_list: Vec<SegmentInfo>,
) -> Result<ImageEntry, PanopticError> {
    let PngStream {
        height,
        width,
        mut reader,
    } = init_reader(image_id, bytes)?;
    let n_pixels = (height as usize) * (width as usize);
    let mut label_map: Vec<u32> = Vec::with_capacity(n_pixels);

    while let Some(row) = reader
        .next_row()
        .map_err(|e| PanopticError::Png(format!("image_id={image_id}: next_row: {e}")))?
    {
        pack_rgb_row(row.data(), &mut label_map);
    }

    let mut segments: FxHashMap<u32, SegmentInfo> =
        FxHashMap::with_capacity_and_hasher(segments_list.len(), Default::default());
    for seg in segments_list {
        if segments.insert(seg.id, seg).is_some() {
            return Err(PanopticError::DuplicateSegmentId {
                image_id,
                segment_id: seg.id,
                side: "gt",
            });
        }
    }

    Ok(ImageEntry {
        height,
        width,
        label_map,
        segments,
    })
}

/// Pack `row_bytes` (length `3*w`, RGB triples) into `w` `u32` ids and
/// append to `out`. Each id is `R | G<<8 | B<<16` per the panoptic
/// PNG convention. Delegates to [`crate::pixel_pack::pack_rgb_row`]
/// for the SIMD-dispatched body.
#[inline]
fn pack_rgb_row(row_bytes: &[u8], out: &mut Vec<u32>) {
    let start = out.len();
    out.resize(start + row_bytes.len() / 3, 0);
    crate::pixel_pack::pack_rgb_row(row_bytes, &mut out[start..]);
}

/// DT-side decode: the full validation + S3 area marginal pass, also
/// row-streaming. Per pixel we (a) push id, (b) skip VOID, (c) look
/// up in the segment remap, (d) bump per-segment area + mark seen.
/// After the walk we check S11 (every declared id was seen) and
/// overwrite each segment's `area` with the PNG marginal (S3).
///
/// Two fully-inlined inner-loop variants dispatch on `max_id`:
/// - **Dense** (`max_id <= DENSE_LOOKUP_MAX_ID`): direct `Vec<u32>`
///   lookup, branch-free on the per-pixel hot path. The common case
///   for COCO panoptic.
/// - **Sparse** (rare): `FxHashMap` lookup for datasets whose encoded
///   id space exceeds the dense cap.
///
/// Hoisting the dispatch out of the inner loop lets LLVM specialize
/// each variant; the enum dispatch otherwise sits in the per-pixel
/// path.
fn decode_dt(
    image_id: ImageId,
    bytes: &[u8],
    segments_list: Vec<SegmentInfo>,
) -> Result<ImageEntry, PanopticError> {
    let PngStream {
        height,
        width,
        mut reader,
    } = init_reader(image_id, bytes)?;
    let n_pixels = (height as usize) * (width as usize);
    let n_segments = segments_list.len();
    let max_id = segments_list.iter().map(|s| s.id).max().unwrap_or(0);
    let mut areas: Vec<u64> = vec![0; n_segments];
    let mut seen: Vec<bool> = vec![false; n_segments];
    let mut label_map: Vec<u32> = Vec::with_capacity(n_pixels);

    if max_id <= DENSE_LOOKUP_MAX_ID {
        // Build the dense `id → idx` table from the thread-local
        // scratch. Duplicate-id check is the only S7 source on this
        // path.
        let cap_needed = (max_id as usize) + 1;
        DT_LOOKUP_SCRATCH.with(|cell| -> Result<(), PanopticError> {
            let mut table = cell.borrow_mut();
            table.clear();
            table.resize(cap_needed, u32::MAX);
            for (idx, seg) in segments_list.iter().enumerate() {
                let slot = &mut table[seg.id as usize];
                if *slot != u32::MAX {
                    return Err(PanopticError::DuplicateSegmentId {
                        image_id,
                        segment_id: seg.id,
                        side: "dt",
                    });
                }
                *slot = idx as u32;
            }
            while let Some(row) = reader
                .next_row()
                .map_err(|e| PanopticError::Png(format!("image_id={image_id}: next_row: {e}")))?
            {
                for chunk in row.data().chunks_exact(3) {
                    let id =
                        (chunk[0] as u32) | ((chunk[1] as u32) << 8) | ((chunk[2] as u32) << 16);
                    label_map.push(id);
                    if id == PANOPTIC_VOID {
                        continue;
                    }
                    let i = id as usize;
                    if i < table.len() {
                        let v = table[i];
                        if v != u32::MAX {
                            areas[v as usize] += 1;
                            seen[v as usize] = true;
                            continue;
                        }
                    }
                    return Err(PanopticError::UnknownPredSegmentId {
                        image_id,
                        segment_id: id,
                    });
                }
            }
            Ok(())
        })?;
    } else {
        let mut map: FxHashMap<u32, u32> =
            FxHashMap::with_capacity_and_hasher(n_segments, Default::default());
        for (idx, seg) in segments_list.iter().enumerate() {
            if map.insert(seg.id, idx as u32).is_some() {
                return Err(PanopticError::DuplicateSegmentId {
                    image_id,
                    segment_id: seg.id,
                    side: "dt",
                });
            }
        }
        while let Some(row) = reader
            .next_row()
            .map_err(|e| PanopticError::Png(format!("image_id={image_id}: next_row: {e}")))?
        {
            for chunk in row.data().chunks_exact(3) {
                let id = (chunk[0] as u32) | ((chunk[1] as u32) << 8) | ((chunk[2] as u32) << 16);
                label_map.push(id);
                if id == PANOPTIC_VOID {
                    continue;
                }
                match map.get(&id) {
                    Some(&v) => {
                        areas[v as usize] += 1;
                        seen[v as usize] = true;
                    }
                    None => {
                        return Err(PanopticError::UnknownPredSegmentId {
                            image_id,
                            segment_id: id,
                        });
                    }
                }
            }
        }
    }

    for (idx, seg) in segments_list.iter().enumerate() {
        if !seen[idx] {
            return Err(PanopticError::MissingPredSegmentInPng {
                image_id,
                segment_id: seg.id,
            });
        }
    }

    let mut segments: FxHashMap<u32, SegmentInfo> =
        FxHashMap::with_capacity_and_hasher(n_segments, Default::default());
    for (idx, mut seg) in segments_list.into_iter().enumerate() {
        seg.area = areas[idx];
        segments.insert(seg.id, seg);
    }

    Ok(ImageEntry {
        height,
        width,
        label_map,
        segments,
    })
}

fn color_type_label(c: png::ColorType, d: png::BitDepth) -> &'static str {
    match (c, d) {
        (png::ColorType::Rgb, png::BitDepth::Sixteen) => "RGB16",
        (png::ColorType::Rgba, png::BitDepth::Eight) => "RGBA8",
        (png::ColorType::Rgba, png::BitDepth::Sixteen) => "RGBA16",
        (png::ColorType::Grayscale, _) => "Grayscale",
        (png::ColorType::GrayscaleAlpha, _) => "GrayscaleAlpha",
        (png::ColorType::Indexed, _) => "Indexed",
        _ => "Other",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::CategoryId;

    /// Encode a `(H, W) u32` panoptic label map as a PNG-bytes blob via
    /// the `png` crate's encoder. Produces 8-bit RGB, the same shape
    /// panopticapi consumes.
    fn encode_label_map_to_png(height: u32, width: u32, label_map: &[u32]) -> Vec<u8> {
        let mut rgb = Vec::with_capacity((height as usize) * (width as usize) * 3);
        for &id in label_map {
            rgb.push((id & 0xff) as u8);
            rgb.push(((id >> 8) & 0xff) as u8);
            rgb.push(((id >> 16) & 0xff) as u8);
        }
        let mut out = Vec::new();
        {
            let mut enc = png::Encoder::new(&mut out, width, height);
            enc.set_color(png::ColorType::Rgb);
            enc.set_depth(png::BitDepth::Eight);
            let mut writer = enc.write_header().unwrap();
            writer.write_image_data(&rgb).unwrap();
        }
        out
    }

    fn seg(id: u32, category_id: CategoryId, iscrowd: bool, area: u64) -> SegmentInfo {
        SegmentInfo {
            id,
            category_id,
            iscrowd,
            area,
        }
    }

    #[test]
    fn round_trip_gt_keeps_json_areas() {
        // 2x3 image: id=1 covers 4 pixels, id=2 covers 1, id=0 (VOID)
        // covers 1. GT side: areas come from JSON; PNG marginals are
        // ignored. We pass a "lying" JSON area to confirm the GT path
        // does NOT overwrite it.
        let h = 2u32;
        let w = 3u32;
        let lm = vec![1u32, 1, 2, 1, 0, 1];
        let png_bytes = encode_label_map_to_png(h, w, &lm);
        let segments = vec![
            seg(1, 100, false, 99_999), // lying area
            seg(2, 200, false, 12_345), // lying area
        ];
        let entry = decode_panoptic_png(7, &png_bytes, segments, "gt").unwrap();
        assert_eq!(entry.height, h);
        assert_eq!(entry.width, w);
        assert_eq!(entry.label_map, lm);
        // GT side keeps JSON areas untouched (S4).
        assert_eq!(entry.segments[&1].area, 99_999);
        assert_eq!(entry.segments[&2].area, 12_345);
    }

    #[test]
    fn round_trip_dt_overwrites_with_png_marginals() {
        // Same fixture; DT side recomputes areas from PNG (S3).
        let h = 2u32;
        let w = 3u32;
        let lm = vec![1u32, 1, 2, 1, 0, 1];
        let png_bytes = encode_label_map_to_png(h, w, &lm);
        let segments = vec![
            seg(1, 100, false, 99_999), // ignored on DT
            seg(2, 200, false, 12_345), // ignored on DT
        ];
        let entry = decode_panoptic_png(7, &png_bytes, segments, "dt").unwrap();
        assert_eq!(entry.label_map, lm);
        assert_eq!(entry.segments[&1].area, 4); // 4 pixels
        assert_eq!(entry.segments[&2].area, 1); // 1 pixel
    }

    #[test]
    fn dt_rejects_unknown_segment_in_png_s1() {
        // PNG carries id=99 but segments_info only has id=1.
        let h = 1u32;
        let w = 4u32;
        let lm = vec![1u32, 1, 99, 1];
        let png_bytes = encode_label_map_to_png(h, w, &lm);
        let segments = vec![seg(1, 100, false, 0)];
        let err = decode_panoptic_png(42, &png_bytes, segments, "dt").unwrap_err();
        match err {
            PanopticError::UnknownPredSegmentId {
                image_id,
                segment_id,
            } => {
                assert_eq!(image_id, 42);
                assert_eq!(segment_id, 99);
            }
            other => panic!("expected UnknownPredSegmentId, got {other:?}"),
        }
    }

    #[test]
    fn dt_rejects_missing_segment_in_png_s11() {
        // segments_info declares id=2 but PNG never carries it.
        let h = 1u32;
        let w = 4u32;
        let lm = vec![1u32, 1, 1, 1];
        let png_bytes = encode_label_map_to_png(h, w, &lm);
        let segments = vec![seg(1, 100, false, 0), seg(2, 200, false, 0)];
        let err = decode_panoptic_png(42, &png_bytes, segments, "dt").unwrap_err();
        match err {
            PanopticError::MissingPredSegmentInPng {
                image_id,
                segment_id,
            } => {
                assert_eq!(image_id, 42);
                assert_eq!(segment_id, 2);
            }
            other => panic!("expected MissingPredSegmentInPng, got {other:?}"),
        }
    }

    #[test]
    fn gt_side_silently_ignores_unknown_png_ids_s8() {
        // PNG carries id=99 but segments_info only has id=1. GT side
        // accepts (S8: matching iterates histogram, not dict).
        let h = 1u32;
        let w = 4u32;
        let lm = vec![1u32, 1, 99, 1];
        let png_bytes = encode_label_map_to_png(h, w, &lm);
        let segments = vec![seg(1, 100, false, 7)];
        let entry = decode_panoptic_png(42, &png_bytes, segments, "gt").unwrap();
        assert_eq!(entry.label_map, lm);
        assert_eq!(entry.segments[&1].area, 7); // JSON area kept
    }

    #[test]
    fn rejects_duplicate_segment_id_s7() {
        let h = 1u32;
        let w = 1u32;
        let lm = vec![1u32];
        let png_bytes = encode_label_map_to_png(h, w, &lm);
        let segments = vec![seg(1, 100, false, 0), seg(1, 200, false, 0)];
        let err = decode_panoptic_png(7, &png_bytes, segments, "gt").unwrap_err();
        match err {
            PanopticError::DuplicateSegmentId {
                image_id,
                segment_id,
                side,
            } => {
                assert_eq!(image_id, 7);
                assert_eq!(segment_id, 1);
                assert_eq!(side, "gt");
            }
            other => panic!("expected DuplicateSegmentId, got {other:?}"),
        }
    }

    #[test]
    fn rejects_non_rgb_png_r2() {
        // Encode an 8-bit grayscale PNG and attempt to decode.
        let h = 2u32;
        let w = 2u32;
        let mut out = Vec::new();
        {
            let mut enc = png::Encoder::new(&mut out, w, h);
            enc.set_color(png::ColorType::Grayscale);
            enc.set_depth(png::BitDepth::Eight);
            let mut writer = enc.write_header().unwrap();
            writer.write_image_data(&[0u8, 1, 2, 3]).unwrap();
        }
        let err = decode_panoptic_png(0, &out, vec![], "gt").unwrap_err();
        match err {
            PanopticError::NonRgbPng { image_id, mode } => {
                assert_eq!(image_id, 0);
                assert_eq!(mode, "Grayscale");
            }
            other => panic!("expected NonRgbPng, got {other:?}"),
        }
    }

    #[test]
    fn sparse_lookup_path_handles_high_ids() {
        // Force the Sparse path with an id above DENSE_LOOKUP_MAX_ID.
        // Encode a 1x2 image with that high id at one pixel, VOID at
        // the other. DT side, area = 1 expected.
        let high_id: u32 = DENSE_LOOKUP_MAX_ID + 1;
        let h = 1u32;
        let w = 2u32;
        let lm = vec![high_id, 0u32];
        let png_bytes = encode_label_map_to_png(h, w, &lm);
        let segments = vec![seg(high_id, 100, false, 0)];
        let entry = decode_panoptic_png(0, &png_bytes, segments, "dt").unwrap();
        assert_eq!(entry.label_map, lm);
        assert_eq!(entry.segments[&high_id].area, 1);
    }
}
