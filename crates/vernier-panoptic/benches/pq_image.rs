//! Per-image PQ kernel microbench.
//!
//! Targets [`pq_image_with_id`] (`kernel.rs:89`) — the per-image
//! intersection-histogram + matching loop the streaming runner calls
//! N times per cell. The val2017 perfect-DT cell spends ~51 s on the
//! `stream_pq` stage across 5000 images, so even a few-microsecond
//! per-image win compounds into seconds at the cell level.
//!
//! Two scenarios mirror the COCO panoptic val workload shape:
//!
//! - **`coco_like`** — 480×640 = 307,200 px, 50 GT + 50 DT segments
//!   per image, perfect match (every GT pixel matches its DT
//!   counterpart). This is the val2017_perfect cell shape.
//! - **`coco_jittered`** — same image / segment counts, but ~20% of
//!   GT pixels are reassigned to a random other DT segment id. Mimics
//!   realistic detector predictions where pairs don't line up cleanly.
//!
//! Run with `cargo bench -p vernier-panoptic --bench pq_image` (or
//! `just bench`).

#![allow(clippy::unwrap_used)]

use divan::{black_box, Bencher};
use rustc_hash::FxHashMap;
use vernier_panoptic::dataset::{ImageEntry, SegmentInfo};
use vernier_panoptic::pq_image_with_id;

fn main() {
    divan::main();
}

#[derive(Clone, Copy)]
struct Scenario {
    height: u32,
    width: u32,
    n_segments: u32,
    /// 0 = perfect match; >0 = fraction of GT pixels reassigned to a
    /// different DT segment id (in basis points, so 2000 = 20%).
    jitter_bp: u32,
}

const COCO_LIKE: Scenario = Scenario {
    height: 480,
    width: 640,
    n_segments: 50,
    jitter_bp: 0,
};

const COCO_JITTERED: Scenario = Scenario {
    height: 480,
    width: 640,
    n_segments: 50,
    jitter_bp: 2000,
};

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// Build a `(label_map, segments)` pair where each pixel is assigned
/// to one of `n_segments` segment ids in column-major stripes (kept
/// simple — the kernel doesn't care about spatial layout, only the
/// per-pixel id distribution).
fn build_image(s: Scenario, seed: u64) -> (Vec<u32>, FxHashMap<u32, SegmentInfo>) {
    let n_pixels = (s.height as usize) * (s.width as usize);
    let mut label_map = Vec::with_capacity(n_pixels);
    let stripe = (n_pixels / s.n_segments as usize).max(1);
    for px_idx in 0..n_pixels {
        // Segment ids start at 1 (0 is VOID per panoptic convention).
        let seg_id = ((px_idx / stripe).min(s.n_segments as usize - 1) as u32) + 1;
        label_map.push(seg_id);
    }
    let mut segments =
        FxHashMap::with_capacity_and_hasher(s.n_segments as usize, Default::default());
    let mut state = seed;
    for seg_id in 1..=s.n_segments {
        // Half things, half stuff; alternate categories so the matching
        // loop's category check exercises both equality and inequality.
        let category_id = ((xorshift(&mut state) as usize) % 16) as i64 + 1;
        let area = stripe as u64;
        segments.insert(
            seg_id,
            SegmentInfo {
                id: seg_id,
                category_id,
                iscrowd: false,
                area,
            },
        );
    }
    (label_map, segments)
}

/// Apply jitter: randomly reassign `jitter_bp / 10000` of GT pixels to
/// a different (non-zero) segment id.
fn jitter_label_map(label_map: &mut [u32], n_segments: u32, jitter_bp: u32, seed: u64) {
    if jitter_bp == 0 {
        return;
    }
    let mut state = seed;
    for px in label_map.iter_mut() {
        if (xorshift(&mut state) as u32) % 10000 < jitter_bp {
            // Bump to a different segment id (mod n_segments + 1, never 0).
            let bump = ((xorshift(&mut state) as u32) % (n_segments - 1)) + 1;
            *px = ((*px - 1 + bump) % n_segments) + 1;
        }
    }
}

fn build_pair(s: Scenario) -> (ImageEntry, ImageEntry) {
    let (gt_lm, gt_segs) = build_image(s, 0xdead_beef_cafe_babe);
    let mut dt_lm = gt_lm.clone();
    jitter_label_map(&mut dt_lm, s.n_segments, s.jitter_bp, 0x1234_5678_90ab_cdef);
    let (_, dt_segs) = build_image(s, 0xdead_beef_cafe_babe); // same segments as GT
    let gt = ImageEntry {
        height: s.height,
        width: s.width,
        label_map: gt_lm,
        segments: gt_segs,
    };
    let dt = ImageEntry {
        height: s.height,
        width: s.width,
        label_map: dt_lm,
        segments: dt_segs,
    };
    (gt, dt)
}

fn run(bencher: Bencher, s: Scenario) {
    let (gt, dt) = build_pair(s);
    bencher.bench_local(|| pq_image_with_id(black_box(0), black_box(&gt), black_box(&dt)).unwrap());
}

#[divan::bench]
fn coco_like(bencher: Bencher) {
    run(bencher, COCO_LIKE);
}

#[divan::bench]
fn coco_jittered(bencher: Bencher) {
    run(bencher, COCO_JITTERED);
}

/// `ImageEntry::from_components` walks every DT pixel for the
/// known-segment-id check. With 5000 images × 300k pixels = 1.5B
/// HashMap lookups, the hasher choice dominates the streaming
/// runner's per-image cost.
#[divan::bench]
fn from_components_dt(bencher: Bencher) {
    let s = COCO_LIKE;
    let (label_map, segments) = build_image(s, 0xdead_beef_cafe_babe);
    let segments_list: Vec<SegmentInfo> = segments.values().copied().collect();
    bencher.bench_local(|| {
        ImageEntry::from_components(
            black_box(0),
            s.height,
            s.width,
            black_box(label_map.clone()),
            black_box(segments_list.clone()),
            "dt",
        )
        .unwrap()
    });
}
