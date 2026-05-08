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
    let (height, width, rgb) = decode_rgb_8bit(image_id, bytes)?;
    let n_pixels = (height as usize) * (width as usize);

    // Build the per-image segment lookup. Direct Vec<u32> when the
    // raw id range fits the DENSE_LOOKUP_MAX_ID cap; FxHashMap fallback
    // otherwise. The Vec path is the typical case (COCO panoptic ids
    // stay well under 1 M); the hashmap fallback handles datasets that
    // pack high bits of the encoded id space.
    let n_segments = segments_list.len();
    let max_id = segments_list.iter().map(|s| s.id).max().unwrap_or(0);
    let mut lookup = SegmentLookup::build(image_id, &segments_list, max_id, side)?;
    let mut areas: Vec<u64> = vec![0; n_segments];
    let mut seen: Vec<bool> = vec![false; n_segments];

    let mut label_map: Vec<u32> = Vec::with_capacity(n_pixels);
    let dt = side == "dt";

    // Inner loop: read 3 bytes per pixel, encode as id, fold into the
    // label map and (DT side) into the per-segment area + seen
    // bitmap. The hot path is straight-line code; the per-pixel
    // branches are well-predicted (VOID hits are rare in non-trivial
    // segmentations, and the DT-side validation either always hits or
    // always misses for a given image's id space).
    let mut chunks = rgb.chunks_exact(3);
    for chunk in &mut chunks {
        // SAFETY of indexing: `chunks_exact(3)` yields a `&[u8]` of
        // length exactly 3, so [0]/[1]/[2] are bounds-checked but
        // collapse to a single bounds check the compiler can hoist.
        let id = (chunk[0] as u32) | ((chunk[1] as u32) << 8) | ((chunk[2] as u32) << 16);
        label_map.push(id);
        if id == PANOPTIC_VOID {
            continue;
        }
        match lookup.index_of(id) {
            Some(idx) => {
                areas[idx] += 1;
                seen[idx] = true;
            }
            None => {
                if dt {
                    return Err(PanopticError::UnknownPredSegmentId {
                        image_id,
                        segment_id: id,
                    });
                }
                // GT: ignore unknown ids per S8 (segments_info may
                // declare more ids than appear in the PNG; the kernel
                // iterates the histogram, not the dict).
            }
        }
    }
    debug_assert!(chunks.remainder().is_empty());

    if dt {
        for (idx, seg) in segments_list.iter().enumerate() {
            if !seen[idx] {
                return Err(PanopticError::MissingPredSegmentInPng {
                    image_id,
                    segment_id: seg.id,
                });
            }
        }
    }

    // Materialize the segments map. DT side overwrites area with the
    // PNG marginal (S3); GT side keeps the JSON value (S4).
    let mut segments: FxHashMap<u32, SegmentInfo> =
        FxHashMap::with_capacity_and_hasher(n_segments, Default::default());
    for (idx, mut seg) in segments_list.into_iter().enumerate() {
        if dt {
            seg.area = areas[idx];
        }
        // Duplicate id was already rejected by `SegmentLookup::build`.
        segments.insert(seg.id, seg);
    }

    Ok(ImageEntry {
        height,
        width,
        label_map,
        segments,
    })
}

/// Read the PNG header, verify 8-bit RGB, and decode into a flat
/// `Vec<u8>` of length `3 * height * width`. Other color types
/// (Grayscale / Indexed / GrayscaleAlpha / RGBA) are rejected at the
/// FFI boundary per quirk **R2** (panopticapi silently drops alpha on
/// RGBA via `[:,:,0..2]` indexing and crashes on P/L; vernier rejects).
fn decode_rgb_8bit(image_id: ImageId, bytes: &[u8]) -> Result<(u32, u32, Vec<u8>), PanopticError> {
    let decoder = png::Decoder::new(Cursor::new(bytes));
    let mut reader = decoder
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
    let buf_size = reader.output_buffer_size().ok_or_else(|| {
        PanopticError::Png(format!(
            "image_id={image_id}: output_buffer_size unavailable"
        ))
    })?;
    let mut buf = vec![0u8; buf_size];
    let out = reader
        .next_frame(&mut buf)
        .map_err(|e| PanopticError::Png(format!("image_id={image_id}: next_frame: {e}")))?;
    // `next_frame` writes the raw image bytes; for 8-bit RGB this is
    // exactly `3 * width * height` bytes contiguous, no padding.
    let (h, w) = (out.height, out.width);
    buf.truncate((h as usize) * (w as usize) * 3);
    Ok((h, w, buf))
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

/// Per-image segment-id → segment-index map. Two backends:
/// - `Dense`: `Vec<u32>` of size `max_id + 1`, sentinel `u32::MAX` for
///   absent. O(1) read, no hashing. Fast path.
/// - `Sparse`: `FxHashMap<u32, u32>` for datasets whose encoded id
///   space exceeds [`DENSE_LOOKUP_MAX_ID`]. Constant-time read with a
///   small constant factor; happens only for adversarial inputs.
enum SegmentLookup {
    Dense { table: Vec<u32> },
    Sparse { map: FxHashMap<u32, u32> },
}

impl SegmentLookup {
    fn build(
        image_id: ImageId,
        segments: &[SegmentInfo],
        max_id: u32,
        side: &'static str,
    ) -> Result<Self, PanopticError> {
        if max_id <= DENSE_LOOKUP_MAX_ID {
            let mut table = vec![u32::MAX; (max_id as usize) + 1];
            for (idx, seg) in segments.iter().enumerate() {
                let slot = &mut table[seg.id as usize];
                if *slot != u32::MAX {
                    return Err(PanopticError::DuplicateSegmentId {
                        image_id,
                        segment_id: seg.id,
                        side,
                    });
                }
                *slot = idx as u32;
            }
            Ok(SegmentLookup::Dense { table })
        } else {
            let mut map: FxHashMap<u32, u32> =
                FxHashMap::with_capacity_and_hasher(segments.len(), Default::default());
            for (idx, seg) in segments.iter().enumerate() {
                if map.insert(seg.id, idx as u32).is_some() {
                    return Err(PanopticError::DuplicateSegmentId {
                        image_id,
                        segment_id: seg.id,
                        side,
                    });
                }
            }
            Ok(SegmentLookup::Sparse { map })
        }
    }

    #[inline(always)]
    fn index_of(&mut self, id: u32) -> Option<usize> {
        match self {
            SegmentLookup::Dense { table } => {
                let i = id as usize;
                if i < table.len() {
                    let v = table[i];
                    if v != u32::MAX {
                        return Some(v as usize);
                    }
                }
                None
            }
            SegmentLookup::Sparse { map } => map.get(&id).map(|&v| v as usize),
        }
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
