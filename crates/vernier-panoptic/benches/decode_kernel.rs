//! Microbenchmarks for `decode_panoptic_png`.
//!
//! Run with `cargo bench -p vernier-panoptic --bench decode_kernel`.
//!
//! Measures the PNG decode + RGB→id pack + (DT side) S3 area marginal
//! kernel — the dominant per-image CPU cost on
//! `coco_panoptic_val2017_perfect` (see the bench-investigation
//! notes accompanying ADR-0025). Each call corresponds to one
//! `submit_png` half-pair on the streaming runner.
//!
//! Shapes follow COCO val2017 panoptic geometry: 480×640 images,
//! 50 segments per image, segment ids RGB-packed above the dense
//! lookup cap so the DT path exercises its sparse FxHashMap branch
//! (same as the production hot path).

#![allow(clippy::unwrap_used)]

use divan::{black_box, Bencher};
use vernier_panoptic::dataset::SegmentInfo;
use vernier_panoptic::decode::decode_panoptic_png;

fn main() {
    divan::main();
}

const HEIGHT: u32 = 480;
const WIDTH: u32 = 640;
const N_SEGMENTS: u32 = 50;

/// Build a synthetic `(H, W)` panoptic label map with `n_segments`
/// horizontal stripes. Segment ids stay small (1..=N_SEGMENTS) so the
/// DT decoder takes its `Dense(Vec<u32>)` lookup branch — the
/// pixel-pack hotloop is identical to the sparse branch (`R|G<<8|B<<16
/// → push → lookup`), so isolating it on the dense path is sufficient
/// to measure the SIMD opportunity targeted by this round.
fn build_label_map() -> Vec<u32> {
    let n_pixels = (HEIGHT as usize) * (WIDTH as usize);
    let stripe = (n_pixels / N_SEGMENTS as usize).max(1);
    (0..n_pixels)
        .map(|i| ((i / stripe).min(N_SEGMENTS as usize - 1) as u32) + 1)
        .collect()
}

/// Encode a `(H, W) u32` panoptic label map as the 8-bit RGB PNG that
/// `decode_panoptic_png` consumes.
fn encode_png(label_map: &[u32]) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(label_map.len() * 3);
    for &id in label_map {
        rgb.push((id & 0xff) as u8);
        rgb.push(((id >> 8) & 0xff) as u8);
        rgb.push(((id >> 16) & 0xff) as u8);
    }
    let mut out = Vec::new();
    let mut enc = png::Encoder::new(&mut out, WIDTH, HEIGHT);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    let mut writer = enc.write_header().unwrap();
    writer.write_image_data(&rgb).unwrap();
    drop(writer);
    out
}

fn build_segments() -> Vec<SegmentInfo> {
    (1..=N_SEGMENTS)
        .map(|id| SegmentInfo {
            id,
            category_id: (id as i64 % 16) + 1,
            iscrowd: false,
            area: 0,
        })
        .collect()
}

#[divan::bench]
fn decode_gt(bencher: Bencher) {
    let png_bytes = encode_png(&build_label_map());
    let segments = build_segments();
    bencher.bench_local(|| {
        decode_panoptic_png(
            black_box(0),
            black_box(&png_bytes),
            black_box(segments.clone()),
            "gt",
        )
        .unwrap()
    });
}

#[divan::bench]
fn decode_dt(bencher: Bencher) {
    let png_bytes = encode_png(&build_label_map());
    let segments = build_segments();
    bencher.bench_local(|| {
        decode_panoptic_png(
            black_box(0),
            black_box(&png_bytes),
            black_box(segments.clone()),
            "dt",
        )
        .unwrap()
    });
}
