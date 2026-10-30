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
use vernier_mask::label_png::{decode_label_png, LabelColor, LabelRows};

use crate::dataset::{ImageEntry, ImageId, SegmentInfo};
use crate::error::PanopticError;
use crate::parity::PANOPTIC_VOID;
use crate::runs::runs;

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

/// Row-streaming reader for an 8-bit RGB panoptic PNG, plus its
/// validated dimensions.
///
/// Label maps go through `vernier_mask::label_png`, which unfilters a
/// piecewise-constant image about twice as fast as a general decoder and
/// takes every PNG a label-map writer produces. Anything it refuses — a
/// layout it does not handle, or a damaged file — goes to the `png` crate,
/// which also reports every decode error. When the fast path accepts a file
/// it has already validated all of it, so its rows cannot fail.
struct PngStream<'a> {
    height: u32,
    width: u32,
    rows: RowSource<'a>,
}

enum RowSource<'a> {
    Label(LabelRows<'a>),
    General(Box<png::Reader<Cursor<&'a [u8]>>>),
}

impl PngStream<'_> {
    fn next_row(&mut self, image_id: ImageId) -> Result<Option<&[u8]>, PanopticError> {
        match &mut self.rows {
            RowSource::Label(rows) => Ok(rows.next()),
            RowSource::General(reader) => reader
                .next_row()
                .map(|row| row.map(|r| r.data()))
                .map_err(|e| PanopticError::Png(format!("image_id={image_id}: next_row: {e}"))),
        }
    }
}

thread_local! {
    /// The fast path's image buffer (`H × (1 + 3W)` bytes), reused across
    /// the images a thread decodes.
    static INFLATE_SCRATCH: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

/// Runs `f` over an 8-bit RGB panoptic PNG's row stream: the label-map fast
/// path, decoded into this thread's scratch buffer, when it takes the file;
/// the `png` crate otherwise.
fn with_png<T>(
    image_id: ImageId,
    bytes: &[u8],
    f: impl FnOnce(PngStream<'_>) -> Result<T, PanopticError>,
) -> Result<T, PanopticError> {
    // The buffer keeps the largest map the thread has decoded: less than
    // the `4 × W × H` label map each decode allocates anyway, and a dataset
    // of large maps would otherwise reallocate and fault it in per image.
    INFLATE_SCRATCH.with_borrow_mut(|scratch| {
        match decode_label_png(bytes, LabelColor::Rgb8, scratch) {
            Some(png) => f(PngStream {
                height: png.height(),
                width: png.width(),
                rows: RowSource::Label(png.into_rows()),
            }),
            None => init_reader(image_id, bytes).and_then(f),
        }
    })
}

/// Initialize the `png` crate's row-streaming reader for an 8-bit RGB
/// panoptic image. Returns the reader plus the validated dimensions. Other
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
        rows: RowSource::General(Box::new(reader)),
    })
}

/// GT-side decode: RGB→id straight into `label_map`, row by row. The
/// fast path holds the inflated image (`H × (1 + 3W)` bytes) in the
/// thread's scratch buffer; the `png` crate fallback streams. No
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
    let (height, width, label_map) = with_png(image_id, bytes, |mut png| {
        let (height, width) = (png.height, png.width);
        let n_pixels = (height as usize) * (width as usize);
        let mut label_map: Vec<u32> = Vec::with_capacity(n_pixels);

        while let Some(row) = png.next_row(image_id)? {
            pack_rgb_row(row, &mut label_map);
        }
        Ok((height, width, label_map))
    })?;

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

/// DT-side decode: the full validation + S3 area marginal pass, row by
/// row over the same reader as the GT side. Each row is packed to ids,
/// then walked run by run (see `runs`): a run of one id is skipped if
/// VOID, looked up in the segment remap unless it repeats the last id, and
/// adds its length to that segment's area. After the walk we check S11
/// (every declared id was seen) and overwrite each segment's `area` with
/// the PNG marginal (S3).
///
/// Lookups are per change of id, not per pixel, so a hash map serves every
/// id space, with no dense table to size.
fn decode_dt(
    image_id: ImageId,
    bytes: &[u8],
    segments_list: Vec<SegmentInfo>,
) -> Result<ImageEntry, PanopticError> {
    let n_segments = segments_list.len();
    let (height, width, label_map, areas, seen) = with_png(image_id, bytes, |mut png| {
        let (height, width) = (png.height, png.width);
        let n_pixels = (height as usize) * (width as usize);
        let mut areas: Vec<u64> = vec![0; n_segments];
        let mut seen: Vec<bool> = vec![false; n_segments];
        let mut label_map: Vec<u32> = Vec::with_capacity(n_pixels);

        let mut index: FxHashMap<u32, usize> =
            FxHashMap::with_capacity_and_hasher(n_segments, Default::default());
        for (idx, seg) in segments_list.iter().enumerate() {
            if index.insert(seg.id, idx).is_some() {
                return Err(PanopticError::DuplicateSegmentId {
                    image_id,
                    segment_id: seg.id,
                    side: "dt",
                });
            }
        }
        // The last id looked up, and its segment index: a noisy map's short
        // runs mostly repeat it.
        let mut last = (PANOPTIC_VOID, 0usize);
        while let Some(row) = png.next_row(image_id)? {
            let start = label_map.len();
            pack_rgb_row(row, &mut label_map);
            for (id, len) in runs(&label_map[start..]) {
                if id == PANOPTIC_VOID {
                    continue;
                }
                if id != last.0 {
                    let Some(&idx) = index.get(&id) else {
                        return Err(PanopticError::UnknownPredSegmentId {
                            image_id,
                            segment_id: id,
                        });
                    };
                    last = (id, idx);
                }
                areas[last.1] += len as u64;
                seen[last.1] = true;
            }
        }
        Ok((height, width, label_map, areas, seen))
    })?;

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
    fn high_ids_decode() {
        // Ids span the whole 24-bit RGB space. Encode a 1x2 image with
        // the top id at one pixel, VOID at the other. DT side, area = 1
        // expected.
        let high_id: u32 = 0x00FF_FFFF;
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
