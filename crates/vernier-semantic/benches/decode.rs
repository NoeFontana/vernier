//! Microbenchmarks for the PNG decode path.
//!
//! Run with `cargo bench -p vernier-semantic --bench decode`.
//!
//! Measures `decode_grayscale8_into` on a synthetic 640×480 8-bit
//! grayscale PNG that matches val2017's dominant geometry. The PNG is
//! encoded once at bench start (out of the timed loop). Two arms:
//!
//! * `into_reuse` — pre-warmed reusable buffer (the steady state
//!   that [`vernier_semantic::decode::evaluate_from_pngs`] enters
//!   after the first image of the val2017 fold loop).
//! * `into_fresh` — fresh empty `Vec<u8>` per iteration (the
//!   pre-buffer-pool allocation pattern; documents the per-decode
//!   savings the pool delivers).

#![allow(clippy::unwrap_used)]

use divan::{black_box, Bencher};
use png::Encoder;
use vernier_semantic::decode::decode_grayscale8_into;

fn main() {
    divan::main();
}

const HEIGHT: u32 = 480;
const WIDTH: u32 = 640;

/// Build a synthetic 8-bit grayscale PNG matching val2017's dominant
/// `(H, W) = (480, 640)` geometry. The label-map content is a simple
/// 50-stripe pattern — encoded size lands in the same 5-10 KB range
/// as a real val2017 semantic GT, so libpng's inflate sees a
/// comparable workload.
fn build_sample_png() -> Vec<u8> {
    let n_pixels = (HEIGHT as usize) * (WIDTH as usize);
    let stripe = n_pixels / 50;
    let pixels: Vec<u8> = (0..n_pixels)
        .map(|i| ((i / stripe).min(49) as u8) % 133)
        .collect();
    let mut buf = Vec::new();
    {
        let mut enc = Encoder::new(&mut buf, WIDTH, HEIGHT);
        enc.set_color(png::ColorType::Grayscale);
        enc.set_depth(png::BitDepth::Eight);
        let mut writer = enc.write_header().unwrap();
        writer.write_image_data(&pixels).unwrap();
        writer.finish().unwrap();
    }
    buf
}

#[divan::bench]
fn into_reuse(bencher: Bencher) {
    let png_bytes = build_sample_png();
    // Pre-warm the buffer so the bench measures the steady-state
    // path (no allocator + memset on the resize call).
    let mut buf: Vec<u8> = Vec::new();
    decode_grayscale8_into(0, &png_bytes, &mut buf).unwrap();

    bencher.bench_local(|| {
        decode_grayscale8_into(black_box(0), black_box(&png_bytes), black_box(&mut buf)).unwrap()
    });
}

#[divan::bench]
fn into_fresh(bencher: Bencher) {
    let png_bytes = build_sample_png();
    bencher
        .with_inputs(Vec::<u8>::new)
        .bench_local_values(|mut buf| {
            decode_grayscale8_into(black_box(0), black_box(&png_bytes), black_box(&mut buf))
                .unwrap();
            buf
        });
}
