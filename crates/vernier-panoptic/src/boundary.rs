//! Boundary panoptic-quality map construction (ADR-0025 Z1/Z2 amendment).
//!
//! Builds the `pan_boundary` label map that the second confusion-histogram
//! scan in [`crate::kernel`] reads from. Mirrors the upstream
//! `boundary_iou/coco_panoptic_api/evaluation.py:105-127` boundary loop:
//! for each segment, erode the mask by `dilation_pixels(h, w, dilation_ratio)`
//! Chebyshev pixels, paint the eroded boundary band back as `segment.id`,
//! wipe the interior to a synthetic `BOUNDARY_ID` sentinel that does not
//! collide with any real segment id (so the boundary confusion scan never
//! picks up interior intersections — only band-vs-band).
//!
//! Two parity modes (ADR-0025 §Z1):
//!
//! - [`build_boundary_map_strict`] — JSON-order, in-place mutation:
//!   later segments' bands lose pixels stolen by earlier segments'
//!   bands. Bit-exact upstream reproduction.
//! - [`build_boundary_map_corrected`] — snapshot-based, sorted by
//!   segment id: deterministic regardless of JSON order. Equal to
//!   strict when no two segments' bands overlap.
//!
//! FN/FP attribution ([`crate::attribute`]) is unchanged across both
//! modes — boundary state is consumed only inside the matching step's
//! `min(mask_iou, boundary_iou)` composition.

use std::mem;

use rustc_hash::FxHashMap;
use vernier_mask::ops::{dilation_pixels, erode_bbox_into_scratch, ErodeScratch};

use crate::dataset::{ImageEntry, SegmentInfo};
use crate::parity::{ParityMode, PANOPTIC_VOID};

/// Default Chebyshev-ball dilation ratio for boundary panoptic quality.
///
/// Pinned at `0.02` per Cheng et al. 2021 and the M4 row of the
/// boundary-IoU quirks survey; mirrors
/// `vernier_core::boundary_parity::BOUNDARY_DILATION_RATIO_DEFAULT`.
/// Duplicated here rather than imported because vernier-panoptic has
/// no edge to vernier-core (ADR-0025).
pub const BOUNDARY_PANOPTIC_DILATION_RATIO_DEFAULT: f64 = 0.02;

/// Boundary-PQ configuration threaded through the per-image kernel and
/// the streaming evaluator.
#[derive(Debug, Clone, Copy)]
pub struct BoundaryConfig {
    /// Chebyshev-ball dilation ratio. `0.02` is the default.
    pub dilation_ratio: f64,
    /// Strict mirrors upstream's in-place mutation + JSON-order
    /// dependence; corrected is deterministic snapshot-based.
    pub parity_mode: ParityMode,
}

impl Default for BoundaryConfig {
    fn default() -> Self {
        Self {
            dilation_ratio: BOUNDARY_PANOPTIC_DILATION_RATIO_DEFAULT,
            parity_mode: ParityMode::Corrected,
        }
    }
}

/// Output of one boundary-map construction pass.
///
/// `ids` is the per-pixel boundary label map (length `h * w`, same
/// shape as the input `label_map`). Pixels carry either `0` (VOID),
/// `boundary_id` (interior — sentinel for "this pixel belongs to a
/// segment but is not on its boundary band"), or a real segment id
/// (pixel sits on that segment's band).
///
/// `boundary_areas` maps each segment id to its band's pixel count.
///
/// `boundary_id` is the value used for interior pixels. The kernel
/// reads this back to avoid treating interior intersections as
/// band-vs-band matches.
#[derive(Debug)]
pub struct BoundaryMap {
    /// Per-pixel boundary label map.
    pub ids: Vec<u32>,
    /// `segment_id -> band_area`. Segments with empty bands map to `0`.
    pub boundary_areas: FxHashMap<u32, u64>,
    /// Sentinel id used to mark interior (non-band) pixels of segments.
    pub boundary_id: u32,
}

/// Reusable scratch for boundary-map construction. Holds the per-call
/// growable buffers so the streaming evaluator amortizes allocations
/// across many `update` calls.
#[derive(Default, Debug)]
pub struct BoundaryScratch {
    /// Erosion scratch buffers (from vernier-mask). Owns the
    /// bbox-shape raster and eroded buffers fed to / read from
    /// [`erode_bbox_into_scratch`].
    pub erode: ErodeScratch,
    /// Output `ids` buffer; ownership is moved into the returned
    /// [`BoundaryMap`] on each call, then replenished on the next.
    ids: Vec<u32>,
    /// Read-only snapshot of the input map (corrected mode only).
    snapshot: Vec<u32>,
    /// Sorted segment ids (corrected-mode iteration order).
    sorted_ids: Vec<u32>,
    /// Per-segment `(min_x, min_y, bw, bh)` and pixel count from one
    /// O(h*w) pre-pass over the original `pan`.
    bboxes: FxHashMap<u32, ([u32; 4], u64)>,
    /// Per-segment band-area accumulator; moved into the returned map.
    boundary_areas: FxHashMap<u32, u64>,
}

impl BoundaryScratch {
    /// Fresh scratch with no allocations.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Pick a `boundary_id` that collides with neither any segment id nor
/// any category id. Upstream uses `max(category_id) + 1`; we widen to
/// `max(max_category_id, max_segment_id, max_pan_value) + 1` so
/// synthetic fixtures with seg_id > max_category_id cannot collide
/// (COCO panoptic in practice has seg_id ≪ 16M and category_id < 200,
/// so this matches upstream there bit-exactly).
fn choose_boundary_id(pan: &[u32], max_category_id: u32, segments_info: &[SegmentInfo]) -> u32 {
    let max_seg = segments_info.iter().map(|s| s.id).max().unwrap_or(0);
    let max_pan = pan.iter().copied().max().unwrap_or(0);
    max_category_id.max(max_seg).max(max_pan).saturating_add(1)
}

#[derive(Clone, Copy)]
enum BuildMode {
    /// Read each segment's pixels from the (mutated) output `ids`,
    /// wipe interior to `boundary_id` before painting the band.
    /// Mirrors upstream's JSON-order in-place mutation.
    Strict,
    /// Read from an immutable snapshot; output `ids` is pre-filled
    /// with `boundary_id`/`VOID` so painting just the band suffices.
    Corrected,
}

/// Strict boundary-map construction (mirrors upstream
/// `evaluation.py:105-127`). Iterates `segments_info` in
/// caller-supplied order and mutates the boundary map in place: reads
/// at iteration `k` see writes from iterations `< k`.
pub fn build_boundary_map_strict(
    pan: &[u32],
    h: u32,
    w: u32,
    segments_info: &[SegmentInfo],
    max_category_id: u32,
    dilation_ratio: f64,
    scratch: &mut BoundaryScratch,
) -> BoundaryMap {
    build_inner(
        pan,
        h,
        w,
        segments_info,
        max_category_id,
        dilation_ratio,
        scratch,
        BuildMode::Strict,
    )
}

/// Corrected boundary-map construction. Reads from an immutable
/// snapshot of `pan` and iterates segments sorted by id — output is
/// independent of caller-supplied order. Bands may overlap on output
/// (later-iterated segment wins for shared pixels) but each segment's
/// `boundary_area` is its own band count, unadjusted for overlap.
/// Equal to [`build_boundary_map_strict`] when no bands overlap.
pub fn build_boundary_map_corrected(
    pan: &[u32],
    h: u32,
    w: u32,
    segments_info: &[SegmentInfo],
    max_category_id: u32,
    dilation_ratio: f64,
    scratch: &mut BoundaryScratch,
) -> BoundaryMap {
    build_inner(
        pan,
        h,
        w,
        segments_info,
        max_category_id,
        dilation_ratio,
        scratch,
        BuildMode::Corrected,
    )
}

/// Top-level dispatch: build the boundary map under the configured
/// parity mode. Convenience wrapper for the kernel's per-image step.
pub fn build_boundary_map(
    entry: &ImageEntry,
    segments_info: &[SegmentInfo],
    max_category_id: u32,
    config: BoundaryConfig,
    scratch: &mut BoundaryScratch,
) -> BoundaryMap {
    let mode = match config.parity_mode {
        ParityMode::Strict => BuildMode::Strict,
        ParityMode::Corrected => BuildMode::Corrected,
    };
    build_inner(
        &entry.label_map,
        entry.height,
        entry.width,
        segments_info,
        max_category_id,
        config.dilation_ratio,
        scratch,
        mode,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_inner(
    pan: &[u32],
    h: u32,
    w: u32,
    segments_info: &[SegmentInfo],
    max_category_id: u32,
    dilation_ratio: f64,
    scratch: &mut BoundaryScratch,
    mode: BuildMode,
) -> BoundaryMap {
    debug_assert_eq!(pan.len(), (h as usize) * (w as usize));

    let boundary_id = choose_boundary_id(pan, max_category_id, segments_info);
    let d = dilation_pixels(h, w, dilation_ratio) as usize;

    // Single O(h*w) pre-pass for all segment bboxes against the original
    // `pan`. Strict mode mutates the output `ids` only to repaint
    // existing segment pixels (never introduces new ones), so the
    // original bbox is a sufficient bounding region for the live pixels.
    scratch.bboxes.clear();
    compute_segment_bboxes(pan, h, w, segments_info, &mut scratch.bboxes);

    // Prepare the output `ids` buffer.
    scratch.ids.clear();
    scratch.ids.reserve(pan.len());
    match mode {
        BuildMode::Strict => scratch.ids.extend_from_slice(pan),
        BuildMode::Corrected => {
            scratch.snapshot.clear();
            scratch.snapshot.extend_from_slice(pan);
            scratch.ids.extend(pan.iter().map(|&px| {
                if px == PANOPTIC_VOID {
                    PANOPTIC_VOID
                } else {
                    boundary_id
                }
            }));
        }
    }

    scratch.boundary_areas.clear();

    // Build the iteration order. For corrected mode we sort segment
    // ids ascending; for strict we use caller order. `mem::take` lets
    // us hold the sorted vector while mutating other scratch fields
    // inside the loop.
    let mut iter_ids: Vec<u32> = mem::take(&mut scratch.sorted_ids);
    iter_ids.clear();
    iter_ids.extend(segments_info.iter().map(|s| s.id));
    if matches!(mode, BuildMode::Corrected) {
        iter_ids.sort_unstable();
    }

    for &seg_id in &iter_ids {
        let band_area = match scratch.bboxes.get(&seg_id) {
            Some(&(bbox, fg_count)) if fg_count > 0 => {
                let source: &[u32] = match mode {
                    BuildMode::Strict => &scratch.ids,
                    BuildMode::Corrected => &scratch.snapshot,
                };
                fill_raster_bbox(
                    scratch.erode.raster_bbox_mut(),
                    source,
                    h as usize,
                    bbox,
                    seg_id,
                );
                erode_bbox_into_scratch(&mut scratch.erode, bbox[2] as usize, bbox[3] as usize, d);
                paint_band(
                    &mut scratch.ids,
                    h as usize,
                    bbox,
                    seg_id,
                    boundary_id,
                    mode,
                    &scratch.erode,
                )
            }
            _ => 0,
        };
        scratch.boundary_areas.insert(seg_id, band_area);
    }

    // Restore the sorted_ids vector for the next call.
    scratch.sorted_ids = iter_ids;

    BoundaryMap {
        ids: mem::take(&mut scratch.ids),
        boundary_areas: mem::take(&mut scratch.boundary_areas),
        boundary_id,
    }
}

/// Compute every segment's `(min_x, min_y, bw, bh, pixel_count)` in
/// one column-major pass over `pan`. Segments absent from `pan` retain
/// the seed `pixel_count == 0` and the caller short-circuits.
fn compute_segment_bboxes(
    pan: &[u32],
    h: u32,
    w: u32,
    segments_info: &[SegmentInfo],
    out: &mut FxHashMap<u32, ([u32; 4], u64)>,
) {
    if h == 0 || w == 0 || segments_info.is_empty() {
        return;
    }
    out.reserve(segments_info.len());
    for seg in segments_info {
        // Seed with (max_x, max_y) sentinels in slots 2/3 so we can
        // accumulate max-by-self-update before converting to (bw, bh).
        out.insert(seg.id, ([u32::MAX, u32::MAX, 0, 0], 0));
    }
    let h_usize = h as usize;
    for x in 0..(w as usize) {
        let col_base = x * h_usize;
        for y in 0..h_usize {
            let Some(entry) = out.get_mut(&pan[col_base + y]) else {
                continue;
            };
            let (bbox, count) = entry;
            *count += 1;
            let xu = x as u32;
            let yu = y as u32;
            if xu < bbox[0] {
                bbox[0] = xu;
            }
            if yu < bbox[1] {
                bbox[1] = yu;
            }
            if xu > bbox[2] {
                bbox[2] = xu;
            }
            if yu > bbox[3] {
                bbox[3] = yu;
            }
        }
    }
    for (bbox, count) in out.values_mut() {
        if *count == 0 {
            *bbox = [0, 0, 0, 0];
        } else {
            bbox[2] = bbox[2] - bbox[0] + 1;
            bbox[3] = bbox[3] - bbox[1] + 1;
        }
    }
}

/// Walk the raster/eroded bbox buffers and write band pixels onto
/// `ids`. In strict mode, every original-mask pixel gets `boundary_id`
/// first (interior wipe), then band pixels overwrite to `seg_id` —
/// matching upstream's `pan[mask] = BOUNDARY_ID; pan[band] = el.id`.
/// In corrected mode, `ids` was pre-filled with `boundary_id` over the
/// foreground; only band pixels need writing.
fn paint_band(
    ids: &mut [u32],
    h: usize,
    bbox: [u32; 4],
    seg_id: u32,
    boundary_id: u32,
    mode: BuildMode,
    erode: &ErodeScratch,
) -> u64 {
    let bx = bbox[0] as usize;
    let by = bbox[1] as usize;
    let bw = bbox[2] as usize;
    let bh = bbox[3] as usize;
    let raster = erode.raster_bbox();
    let eroded = erode.eroded_bbox();
    let mut band_area: u64 = 0;
    for col in 0..bw {
        let raster_col = col * bh;
        let dst_col_base = (bx + col) * h + by;
        for row in 0..bh {
            let raster_idx = raster_col + row;
            let m = raster[raster_idx];
            if m == 0 {
                continue;
            }
            let dst = dst_col_base + row;
            if matches!(mode, BuildMode::Strict) {
                ids[dst] = boundary_id;
            }
            if m != eroded[raster_idx] {
                ids[dst] = seg_id;
                band_area += 1;
            }
        }
    }
    band_area
}

/// Fill a `bw * bh` column-major byte raster: `1` where the source
/// `pan` map equals `seg_id` inside `bbox`, `0` elsewhere.
fn fill_raster_bbox(raster: &mut Vec<u8>, pan: &[u32], h: usize, bbox: [u32; 4], seg_id: u32) {
    let bx = bbox[0] as usize;
    let by = bbox[1] as usize;
    let bw = bbox[2] as usize;
    let bh = bbox[3] as usize;
    raster.clear();
    raster.resize(bw * bh, 0);
    for col in 0..bw {
        let src_col_base = (bx + col) * h + by;
        let dst_col_base = col * bh;
        for row in 0..bh {
            if pan[src_col_base + row] == seg_id {
                raster[dst_col_base + row] = 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{CategoryId, SegmentInfo};

    fn seg(id: u32, category_id: CategoryId) -> SegmentInfo {
        SegmentInfo {
            id,
            category_id,
            iscrowd: false,
            area: 0,
        }
    }

    #[test]
    fn empty_map_produces_no_areas() {
        let pan: Vec<u32> = Vec::new();
        let segs: Vec<SegmentInfo> = Vec::new();
        let mut scratch = BoundaryScratch::new();
        let m = build_boundary_map_strict(&pan, 0, 0, &segs, 100, 0.02, &mut scratch);
        assert!(m.ids.is_empty());
        assert!(m.boundary_areas.is_empty());
    }

    #[test]
    fn boundary_id_avoids_segment_collision() {
        let pan = vec![1_000_000u32; 9];
        let segs = vec![seg(1_000_000, 5)];
        let mut scratch = BoundaryScratch::new();
        let m = build_boundary_map_strict(&pan, 3, 3, &segs, 100, 0.02, &mut scratch);
        assert!(m.boundary_id > 1_000_000);
    }

    #[test]
    fn full_image_single_segment_has_band_subset_of_mask() {
        let pan = vec![1u32; 25];
        let segs = vec![seg(1, 100)];
        let mut scratch = BoundaryScratch::new();
        let m = build_boundary_map_strict(&pan, 5, 5, &segs, 200, 0.5, &mut scratch);
        let band_count_in_ids = m.ids.iter().filter(|&&v| v == 1).count() as u64;
        assert_eq!(band_count_in_ids, m.boundary_areas[&1]);
        let interior_count = m.ids.iter().filter(|&&v| v == m.boundary_id).count() as u64;
        assert!(interior_count + m.boundary_areas[&1] == 25);
    }

    #[test]
    fn corrected_and_strict_match_when_bands_disjoint() {
        let mut pan = vec![0u32; 36];
        for x in 0..2 {
            for y in 0..2 {
                pan[x * 6 + y] = 1;
            }
        }
        for x in 4..6 {
            for y in 4..6 {
                pan[x * 6 + y] = 2;
            }
        }
        let segs = vec![seg(1, 100), seg(2, 100)];
        let mut s1 = BoundaryScratch::new();
        let mut s2 = BoundaryScratch::new();
        let strict = build_boundary_map_strict(&pan, 6, 6, &segs, 200, 0.5, &mut s1);
        let corrected = build_boundary_map_corrected(&pan, 6, 6, &segs, 200, 0.5, &mut s2);
        assert_eq!(strict.boundary_areas[&1], corrected.boundary_areas[&1]);
        assert_eq!(strict.boundary_areas[&2], corrected.boundary_areas[&2]);
    }

    #[test]
    fn segment_absent_from_pan_yields_empty_band() {
        // S8 quirk: GT segment declared in JSON but missing from PNG.
        let pan = vec![1u32; 9];
        let segs = vec![seg(99, 100)];
        let mut scratch = BoundaryScratch::new();
        let m = build_boundary_map_strict(&pan, 3, 3, &segs, 200, 0.02, &mut scratch);
        assert_eq!(m.boundary_areas[&99], 0);
    }

    #[test]
    fn config_default_is_corrected() {
        let cfg = BoundaryConfig::default();
        assert_eq!(cfg.dilation_ratio, 0.02);
        assert_eq!(cfg.parity_mode, ParityMode::Corrected);
    }

    #[test]
    fn strict_in_place_mutation_visible_in_subsequent_segments() {
        let h = 4u32;
        let w = 4u32;
        let mut pan = vec![0u32; (h * w) as usize];
        for x in 0..(w as usize) {
            for y in 0..2 {
                pan[x * (h as usize) + y] = 1;
            }
            for y in 2..4 {
                pan[x * (h as usize) + y] = 2;
            }
        }
        let segs = vec![seg(1, 100), seg(2, 100)];
        let mut s = BoundaryScratch::new();
        let m = build_boundary_map_strict(&pan, h, w, &segs, 200, 0.5, &mut s);
        assert!(m.ids.contains(&1));
        assert!(m.ids.contains(&2));
        assert!(m.boundary_areas.contains_key(&1));
        assert!(m.boundary_areas.contains_key(&2));
    }

    #[test]
    fn corrected_uses_sorted_segment_id_iteration() {
        let pan = vec![1u32, 1, 2, 2];
        let segs_fwd = vec![seg(1, 100), seg(2, 100)];
        let segs_rev = vec![seg(2, 100), seg(1, 100)];
        let mut s_fwd = BoundaryScratch::new();
        let mut s_rev = BoundaryScratch::new();
        let m_fwd = build_boundary_map_corrected(&pan, 2, 2, &segs_fwd, 200, 0.5, &mut s_fwd);
        let m_rev = build_boundary_map_corrected(&pan, 2, 2, &segs_rev, 200, 0.5, &mut s_rev);
        assert_eq!(m_fwd.ids, m_rev.ids);
        assert_eq!(m_fwd.boundary_areas, m_rev.boundary_areas);
    }

    #[test]
    fn scratch_reuse_across_calls_preserves_output() {
        // Run the same fixture twice through the same scratch — the
        // second result must match the first bit-exactly, otherwise
        // the scratch buffers aren't being cleared correctly.
        let pan = vec![1u32; 25];
        let segs = vec![seg(1, 100)];
        let mut scratch = BoundaryScratch::new();
        let m1 = build_boundary_map_strict(&pan, 5, 5, &segs, 200, 0.5, &mut scratch);
        let (m1_ids, m1_areas, m1_bid) = (m1.ids, m1.boundary_areas, m1.boundary_id);
        let m2 = build_boundary_map_strict(&pan, 5, 5, &segs, 200, 0.5, &mut scratch);
        assert_eq!(m1_ids, m2.ids);
        assert_eq!(m1_areas, m2.boundary_areas);
        assert_eq!(m1_bid, m2.boundary_id);
    }
}
