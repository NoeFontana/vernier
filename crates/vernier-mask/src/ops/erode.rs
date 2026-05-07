//! Erosion of an RLE binary mask by a Chebyshev (L-infinity) ball.
//!
//! [`erode_chebyshev_ball`] is the erosion operator used by the
//! boundary-IoU metric (Cheng et al. 2021). See the boundary-IoU
//! quirks survey for the full algorithmic contract: rows **M1**, **M5**
//! (kernel shape and size), **N1**–**N2** (1-pixel zero pad applied
//! once before erosion, stripped after), **N3** (pad value is hard
//! zero), **N4** (single-pass output bit-equal to iterative on integer
//! binary input).
//!
//! Per ADR-0010 §"Algorithm specification (A2)" the implementation is
//! separable — a 1D row pass (along x) followed by a 1D column pass
//! (along y), each sliding a window of size `2d+1`. On integer binary
//! input this is bit-equal to iterative 3×3 erosion applied `d`
//! times: the (3×3 all-ones)^d structuring element under iterated
//! dilation equals the (2d+1)-square all-ones kernel, and on a
//! min-filter erosion is the dual. Equivalence is asserted by a
//! property test against an iterative reference kept in this file's
//! `tests` module.
//!
//! Reads outside the once-padded `(h+2) × (w+2)` image return `1`,
//! matching `cv2.erode` default `BORDER_CONSTANT` for binary input.
//! The only zeros that contribute to the sliding min are the explicit
//! 1-pixel zero ring; this is what makes the single-pass form
//! bit-equal to the iterative reference at the `(d-1)` ring near the
//! image edge for `d ≥ 2`.

use crate::error::MaskError;
use crate::rle::Rle;

/// Reusable scratch space for the `_into` erosion variants.
///
/// Holds growable buffers that get sized to the per-call shape on
/// each invocation. Reusing one instance across many calls amortizes
/// the per-mask allocations away — particularly useful in the
/// boundary-IoU dataset-wide pass, which decodes / erodes /
/// XOR-encodes ~36k masks per `evaluate_boundary` on val2017.
///
/// Construct with [`ErodeScratch::new`] (zero-allocation start).
/// Buffers grow as `clear() + resize()` on each call, so a sequence
/// of similarly-shaped masks pays zero allocations after the first.
#[derive(Default)]
pub struct ErodeScratch {
    pub(crate) padded: Vec<u8>,
    pub(crate) row_scratch: Vec<u8>,
    pub(crate) row_in: Vec<u8>,
    pub(crate) row_out: Vec<u8>,
    pub(crate) raster: Vec<u8>,
    pub(crate) eroded: Vec<u8>,
    pub(crate) min_temp: Vec<u8>,
    // u64-packed buffers for the chunked row pass: each `u64` element
    // carries 8 byte rows, so AND on the u64 lane is byte-wise AND on
    // 8 rows in parallel.
    pub(crate) row_in_u64: Vec<u64>,
    pub(crate) row_out_u64: Vec<u64>,
    pub(crate) min_temp_u64: Vec<u64>,
}

impl ErodeScratch {
    /// Creates an empty scratch instance with no buffer allocations.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Erodes an RLE binary mask by a `(2 * radius_pixels + 1)`-square
/// L-infinity (Chebyshev) ball structuring element.
///
/// `radius_pixels = 0` is the identity. Empty masks (`h == 0` or
/// `w == 0`) round-trip unchanged. Per quirks **N1**–**N2** the mask
/// is padded with a 1-pixel zero border before erosion, then the pad
/// is stripped after — so border-touching foreground pixels are
/// eroded against the zero pad, which is the metric-defining
/// behaviour rather than an implementation quirk.
pub fn erode_chebyshev_ball(rle: &Rle, radius_pixels: u32) -> Result<Rle, MaskError> {
    let mut scratch = ErodeScratch::new();
    erode_chebyshev_ball_into(rle, radius_pixels, &mut scratch)
}

/// `_into` variant of [`erode_chebyshev_ball`] reusing a caller-owned
/// [`ErodeScratch`]. Same semantics; lets hot-path callers (e.g. the
/// boundary-IoU dataset-wide pass) amortize per-mask allocations.
pub fn erode_chebyshev_ball_into(
    rle: &Rle,
    radius_pixels: u32,
    scratch: &mut ErodeScratch,
) -> Result<Rle, MaskError> {
    if rle.h == 0 || rle.w == 0 || radius_pixels == 0 {
        return Ok(rle.clone());
    }
    let h = rle.h as usize;
    let w = rle.w as usize;
    let d = radius_pixels as usize;
    rle.to_raster_bytes_into(&mut scratch.raster);
    erode_raster_into_scratch(scratch, h, w, d);
    Rle::from_raster_bytes(&scratch.eroded, rle.h, rle.w)
}

/// Pad → row pass → column pass → strip pad on a column-major byte
/// raster. Test-only wrapper so the proptest in this file can compare
/// against the iterative reference without going through RLE
/// encode/decode.
#[cfg(test)]
fn erode_raster_chebyshev(raster: &[u8], h: usize, w: usize, d: usize) -> Vec<u8> {
    let mut scratch = ErodeScratch::new();
    scratch.raster.clear();
    scratch.raster.extend_from_slice(raster);
    erode_raster_into_scratch(&mut scratch, h, w, d);
    scratch.eroded
}

/// Pad → row pass → column pass → strip pad on a column-major byte
/// raster. Reads input from `scratch.raster` (caller writes the raster
/// in via [`Rle::to_raster_bytes_into`] or `Vec::extend_from_slice`)
/// and writes the eroded interior to `scratch.eroded`.
pub(crate) fn erode_raster_into_scratch(scratch: &mut ErodeScratch, h: usize, w: usize, d: usize) {
    let ph = h + 2;
    let pw = w + 2;
    let pad_size = ph * pw;

    // The ring of `padded` must be zero (N1); the interior is fully
    // overwritten in step 1, so a clear-and-resize-with-zero is the
    // simplest way to guarantee a clean ring across reuses.
    scratch.padded.clear();
    scratch.padded.resize(pad_size, 0);

    // The remaining buffers are fully overwritten on each call; the
    // resize-with-0 just sizes them — the value is irrelevant.
    scratch.row_scratch.clear();
    scratch.row_scratch.resize(pad_size, 0);
    scratch.row_in.clear();
    scratch.row_in.resize(pw, 0);
    scratch.row_out.clear();
    scratch.row_out.resize(pw, 0);
    scratch.eroded.clear();
    scratch.eroded.resize(h * w, 0);
    // Two `max(ph, pw)`-byte ping-pong halves for the sparse-table
    // build inside `min_filter_binary`. Contents are dead on entry
    // (overwritten before read), so grow-only — no re-zeroing on the
    // hot path.
    let scan_needed = 2 * ph.max(pw);
    if scratch.min_temp.len() < scan_needed {
        scratch.min_temp.resize(scan_needed, 0);
    }
    // u64-packed row-pass buffers: each `u64` carries 8 row bytes for
    // a single x, so the bulk row pass processes 8 rows in parallel
    // with contiguous u64 loads. Sized only when the bulk path runs
    // (`ph >= 8`); otherwise the scalar path covers everything.
    let row_chunks = ph / 8;
    if row_chunks > 0 {
        if scratch.row_in_u64.len() < pw {
            scratch.row_in_u64.resize(pw, 0);
        }
        if scratch.row_out_u64.len() < pw {
            scratch.row_out_u64.resize(pw, 0);
        }
        if scratch.min_temp_u64.len() < 2 * pw {
            scratch.min_temp_u64.resize(2 * pw, 0);
        }
    }

    let ErodeScratch {
        padded,
        row_scratch,
        row_in,
        row_out,
        raster,
        eroded,
        min_temp,
        row_in_u64,
        row_out_u64,
        min_temp_u64,
    } = scratch;

    // 1. Pad to (h+2, w+2) column-major with a 1-pixel zero ring (N1).
    //    Each source column is contiguous (column-major), so the per-x
    //    body is a `copy_from_slice` between aligned `h`-byte slices.
    for x in 0..w {
        let src_start = x * h;
        let dst_start = (x + 1) * ph + 1;
        padded[dst_start..dst_start + h].copy_from_slice(&raster[src_start..src_start + h]);
    }

    // 2. Row pass: column-major means x-stride = ph (non-contiguous),
    //    so the scalar form gathers byte-by-byte for each y. The bulk
    //    form below packs 8 rows per x into a u64 and runs one min
    //    filter for the chunk — `(2d+1)`-way binary min reduces to AND,
    //    and a u64 AND is a byte-wise AND across 8 rows in parallel.
    let bulk_end = row_chunks * 8;
    for c in 0..row_chunks {
        let y_chunk = c * 8;
        for (x, slot) in row_in_u64[..pw].iter_mut().enumerate() {
            let base = x * ph + y_chunk;
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(&padded[base..base + 8]);
            *slot = u64::from_ne_bytes(bytes);
        }
        min_filter_binary_u64(&row_in_u64[..pw], d, &mut row_out_u64[..pw], min_temp_u64);
        for (x, &v) in row_out_u64[..pw].iter().enumerate() {
            let base = x * ph + y_chunk;
            row_scratch[base..base + 8].copy_from_slice(&v.to_ne_bytes());
        }
    }
    // Tail rows that didn't fit a full u64 chunk (`ph % 8`). On val2017
    // image dims this is at most 7 of ~480 rows; for `ph < 8` images
    // (e.g. some property-test cases) the bulk loop is skipped entirely
    // and this scalar path covers everything.
    for y in bulk_end..ph {
        for x in 0..pw {
            row_in[x] = padded[x * ph + y];
        }
        min_filter_binary(row_in, d, row_out, min_temp);
        for x in 0..pw {
            row_scratch[x * ph + y] = row_out[x];
        }
    }

    // 3. Column pass: y-stride = 1 means each column is a contiguous
    //    `ph`-element slice. `padded` is unused after the row pass, so
    //    reuse it as the column-pass output buffer.
    for x in 0..pw {
        let col = x * ph;
        let col_in = &row_scratch[col..col + ph];
        let col_out = &mut padded[col..col + ph];
        min_filter_binary(col_in, d, col_out, min_temp);
    }

    // 4. Strip the pad: copy interior (1..=w) × (1..=h) back to (h, w)
    //    column-major (N2).
    for x in 0..w {
        let src_start = (x + 1) * ph + 1;
        let dst_start = x * h;
        eroded[dst_start..dst_start + h].copy_from_slice(&padded[src_start..src_start + h]);
    }
}

/// `dst[i] = a[i] & b[i]` over equal-length slices. Written as a
/// `zip` chain so LLVM lowers it to 16/32-byte SSE2/AVX2 byte-AND
/// lanes — the only autovec-shaped primitive `min_filter_binary`
/// needs.
#[inline]
fn and_into(dst: &mut [u8], a: &[u8], b: &[u8]) {
    debug_assert_eq!(dst.len(), a.len());
    debug_assert_eq!(dst.len(), b.len());
    for ((d, &x), &y) in dst.iter_mut().zip(a.iter()).zip(b.iter()) {
        *d = x & y;
    }
}

/// 1D sliding-window minimum on a binary `{0, 1}` byte sequence with
/// kernel size `2 * d + 1`. Reads outside the sequence return `1` (so
/// they don't pull the min below) — matches `cv2.erode`'s
/// `BORDER_CONSTANT` default for binary input.
///
/// On binary input `min` reduces to `&`. The interior is computed by
/// a sparse-table build over `temp`: at level `l ∈ 1..=k` we have
/// `temp_l[i] = AND input[i .. i + 2^l]`, computed as
/// `temp_l[i] = temp_{l-1}[i] & temp_{l-1}[i + 2^(l-1)]`. With
/// `k = floor(log2(2d + 1))` this is two non-overlapping slice
/// reads + one slice write per level — the autovec shape. The
/// interior output is then
/// `output[j] = temp_k[j - d] & temp_k[j + d + 1 - 2^k]`, again two
/// contiguous slice ANDs. Edge regions (`[0, d)` and `[n - d, n)`)
/// straddle OOB and are clipped scalarly.
///
/// `temp` is caller-provided scratch of length `≥ 2 * n`: two
/// `n`-byte ping-pong halves for the level build.
fn min_filter_binary(input: &[u8], d: usize, output: &mut [u8], temp: &mut [u8]) {
    let n = input.len();
    debug_assert_eq!(n, output.len());
    debug_assert!(temp.len() >= 2 * n);
    if n == 0 {
        return;
    }
    if d == 0 {
        output.copy_from_slice(input);
        return;
    }

    // Window-AND on the clipped range `[max(0, j-d), min(n, j+d+1))`.
    // Used for the `n <= 2d` degenerate case and for the two edge
    // regions of the standard path.
    let clip_and = |j: usize| -> u8 {
        let lo = j.saturating_sub(d);
        let hi = (j + d + 1).min(n);
        input[lo..hi].iter().fold(1_u8, |acc, &v| acc & v)
    };

    if n <= 2 * d {
        for (j, out) in output.iter_mut().enumerate() {
            *out = clip_and(j);
        }
        return;
    }

    for (j, out) in (0..d).zip(output[..d].iter_mut()) {
        *out = clip_and(j);
    }
    for (j, out) in (n - d..n).zip(output[n - d..].iter_mut()) {
        *out = clip_and(j);
    }

    // Sparse-table build. `win >= 3` here (`d >= 1` and the
    // `n <= 2d` branch already returned), so `leading_zeros`
    // doesn't observe the all-zero edge.
    let win = 2 * d + 1;
    let k = (usize::BITS - 1 - win.leading_zeros()) as usize;
    let pow_k = 1_usize << k;

    let (mut src, mut dst_buf) = temp.split_at_mut(n);

    // Level 1 reads from `input`, skipping the otherwise-required
    // copy of `input` into `src` before the loop. After this block,
    // `src` holds level-1 data (or level-0 if k == 0, which doesn't
    // happen here since win >= 3).
    {
        let stride = 1_usize;
        let len = n - 2 + 1;
        and_into(&mut src[..len], &input[..len], &input[stride..stride + len]);
    }
    for l in 2..=k {
        let stride = 1_usize << (l - 1);
        let len = n - (1_usize << l) + 1;
        and_into(&mut dst_buf[..len], &src[..len], &src[stride..stride + len]);
        std::mem::swap(&mut src, &mut dst_buf);
    }

    // `output[j + d] = src[j] & src[j + win - 2^k]` for the interior.
    let interior_len = n - 2 * d;
    let hi_off = win - pow_k;
    and_into(
        &mut output[d..d + interior_len],
        &src[..interior_len],
        &src[hi_off..hi_off + interior_len],
    );
}

/// OOB constant for the u64-packed row pass: every byte = 1, so a
/// per-byte AND with this leaves the corresponding row's binary value
/// unchanged — matching the scalar `clip_and` initial accumulator.
const ONES_U64: u64 = 0x0101_0101_0101_0101;

/// `dst[i] = a[i] & b[i]` over equal-length slices of `u64`. Same
/// shape as [`and_into`] for the byte-packed row pass: each `u64`
/// lane holds 8 row bytes, and a u64 AND is byte-wise AND across
/// those 8 rows in parallel.
#[inline]
fn and_into_u64(dst: &mut [u64], a: &[u64], b: &[u64]) {
    debug_assert_eq!(dst.len(), a.len());
    debug_assert_eq!(dst.len(), b.len());
    for ((d, &x), &y) in dst.iter_mut().zip(a.iter()).zip(b.iter()) {
        *d = x & y;
    }
}

/// `u64`-packed variant of [`min_filter_binary`].
///
/// Each input element is a `u64` carrying 8 byte rows packed via
/// `u64::from_ne_bytes`; the algorithm and edge handling are identical
/// to the scalar form, with byte-AND replaced by `u64`-AND (correct
/// because the bytes are `{0, 1}` and AND is bitwise). OOB reads
/// return [`ONES_U64`] — per-byte `1`, which is what the scalar form
/// uses.
///
/// Used by [`erode_raster_into_scratch`]'s row pass to process 8 rows
/// per pass over a column-major raster: 8 consecutive y values at a
/// fixed x are 8 contiguous bytes, i.e. a single `u64` load. This
/// removes the per-row strided gather/scatter into byte-wide `row_in`
/// / `row_out` buffers and replaces it with one row-pass per 8-row
/// chunk over `u64` buffers — the same sparse-table levels and edge
/// passes, but 1/8 the loop iterations through the row dimension.
fn min_filter_binary_u64(input: &[u64], d: usize, output: &mut [u64], temp: &mut [u64]) {
    let n = input.len();
    debug_assert_eq!(n, output.len());
    debug_assert!(temp.len() >= 2 * n);
    if n == 0 {
        return;
    }
    if d == 0 {
        output.copy_from_slice(input);
        return;
    }

    let clip_and = |j: usize| -> u64 {
        let lo = j.saturating_sub(d);
        let hi = (j + d + 1).min(n);
        input[lo..hi].iter().fold(ONES_U64, |acc, &v| acc & v)
    };

    if n <= 2 * d {
        for (j, out) in output.iter_mut().enumerate() {
            *out = clip_and(j);
        }
        return;
    }

    for (j, out) in (0..d).zip(output[..d].iter_mut()) {
        *out = clip_and(j);
    }
    for (j, out) in (n - d..n).zip(output[n - d..].iter_mut()) {
        *out = clip_and(j);
    }

    let win = 2 * d + 1;
    let k = (usize::BITS - 1 - win.leading_zeros()) as usize;
    let pow_k = 1_usize << k;

    let (mut src, mut dst_buf) = temp.split_at_mut(n);

    {
        let stride = 1_usize;
        let len = n - 2 + 1;
        and_into_u64(&mut src[..len], &input[..len], &input[stride..stride + len]);
    }
    for l in 2..=k {
        let stride = 1_usize << (l - 1);
        let len = n - (1_usize << l) + 1;
        and_into_u64(&mut dst_buf[..len], &src[..len], &src[stride..stride + len]);
        std::mem::swap(&mut src, &mut dst_buf);
    }

    let interior_len = n - 2 * d;
    let hi_off = win - pow_k;
    and_into_u64(
        &mut output[d..d + interior_len],
        &src[..interior_len],
        &src[hi_off..hi_off + interior_len],
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// Iterative reference: pad once with a 1-pixel zero ring, apply a
    /// 3×3 min-filter `d` times with OOB-fills-with-1, strip the pad.
    /// This is what `cv2.erode(padded, np.ones((3,3)), iterations=d)`
    /// computes on a binary `np.uint8` mask under the OpenCV pin range
    /// covered by quirk **M6**.
    fn erode_iterative_reference(raster: &[u8], h: usize, w: usize, d: usize) -> Vec<u8> {
        if h == 0 || w == 0 || d == 0 {
            return raster.to_vec();
        }
        let ph = h + 2;
        let pw = w + 2;
        let mut buf = vec![0u8; ph * pw];
        for x in 0..w {
            for y in 0..h {
                buf[(x + 1) * ph + (y + 1)] = raster[x * h + y];
            }
        }
        let mut scratch = vec![0u8; ph * pw];
        for _ in 0..d {
            for x in 0..pw {
                for y in 0..ph {
                    let mut m: u8 = 1;
                    for dx in -1i32..=1 {
                        for dy in -1i32..=1 {
                            let nx = x as i32 + dx;
                            let ny = y as i32 + dy;
                            let v = if nx >= 0 && nx < pw as i32 && ny >= 0 && ny < ph as i32 {
                                buf[(nx as usize) * ph + (ny as usize)]
                            } else {
                                1
                            };
                            if v < m {
                                m = v;
                            }
                        }
                    }
                    scratch[x * ph + y] = m;
                }
            }
            std::mem::swap(&mut buf, &mut scratch);
        }
        let mut out = vec![0u8; h * w];
        for x in 0..w {
            for y in 0..h {
                out[x * h + y] = buf[(x + 1) * ph + (y + 1)];
            }
        }
        out
    }

    fn rle(h: u32, w: u32, counts: Vec<u32>) -> Rle {
        Rle { h, w, counts }
    }

    #[test]
    fn empty_mask_round_trips_unchanged() {
        let r = rle(0, 0, vec![]);
        assert_eq!(erode_chebyshev_ball(&r, 5).unwrap(), r);
    }

    #[test]
    fn radius_zero_is_identity() {
        let r = rle(2, 2, vec![1, 2, 1]);
        assert_eq!(erode_chebyshev_ball(&r, 0).unwrap(), r);
    }

    #[test]
    fn full_3x3_erodes_to_single_center_pixel_at_d1() {
        // 3×3 fully-foreground mask. With the zero pad (N1) the kernel
        // sees zeros around the perimeter, so erosion by the 3×3
        // Chebyshev ball (d=1) leaves only the center pixel: M5
        // (square not cross) is what makes this a single pixel rather
        // than a + shape.
        let r = rle(3, 3, vec![0, 9]);
        let eroded = erode_chebyshev_ball(&r, 1).unwrap();
        // Center pixel at (x=1, y=1) → flat idx 1*3 + 1 = 4. RLE
        // canonicalises as [bg=4, fg=1, bg=4].
        assert_eq!(eroded.counts, vec![4, 1, 4]);
    }

    #[test]
    fn full_5x5_eroded_by_d1_yields_3x3_interior_block() {
        // 5×5 fully-fg, padded → eroded by d=1 → interior 3×3 block
        // (1..=3, 1..=3) of foreground = 9 fg pixels. Frames the
        // metric: the boundary band of a fully-fg image is a frame of
        // width d (P3).
        let r = rle(5, 5, vec![0, 25]);
        let eroded = erode_chebyshev_ball(&r, 1).unwrap();
        let fg_count: u32 = eroded.to_raster_bytes().iter().map(|&b| u32::from(b)).sum();
        assert_eq!(fg_count, 9);
    }

    #[test]
    fn kernel_larger_than_image_erodes_everything() {
        // Quirk P1: when (2d+1) > min(h, w), erosion of fully-fg
        // yields fully-zero — the structuring element cannot fit
        // inside the foreground anywhere.
        let r = rle(3, 3, vec![0, 9]);
        let eroded = erode_chebyshev_ball(&r, 10).unwrap();
        assert_eq!(eroded.area(), 0);
    }

    #[test]
    fn empty_foreground_erosion_is_empty() {
        // P2: erosion of an empty mask is empty.
        let r = rle(4, 4, vec![16]);
        let eroded = erode_chebyshev_ball(&r, 2).unwrap();
        assert_eq!(eroded.area(), 0);
        assert_eq!(eroded.h, 4);
        assert_eq!(eroded.w, 4);
    }

    #[test]
    fn single_pixel_mask_erodes_to_zero_at_d1() {
        // P4: a single-pixel mask + d=1 erosion → empty. The pad pulls
        // every neighbour to zero, so the min over the 3×3 window is 0.
        let mut raster = vec![0u8; 9];
        raster[4] = 1; // (x=1, y=1) flat idx = 4
        let r = Rle::from_raster_bytes(&raster, 3, 3).unwrap();
        let eroded = erode_chebyshev_ball(&r, 1).unwrap();
        assert_eq!(eroded.area(), 0);
    }

    #[test]
    fn van_herk_matches_iterative_on_hand_traced_fixture() {
        // 4×5 mask with a non-trivial pattern, exercising columns
        // partially eroded plus columns fully zeroed plus a column
        // straddling the kernel boundary.
        let raster: Vec<u8> = vec![
            // x=0: column-major, h=4 rows.
            0, 1, 0, 0, // (0,0..4)
            1, 1, 1, 1, // (1,0..4)
            1, 1, 1, 1, // (2,0..4)
            0, 0, 0, 0, // (3,0..4)
            1, 1, 0, 0, // (4,0..4)
        ];
        for d in 1..=3 {
            let single_pass = erode_raster_chebyshev(&raster, 4, 5, d);
            let iterative = erode_iterative_reference(&raster, 4, 5, d);
            assert_eq!(single_pass, iterative, "d={d}");
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 512, ..ProptestConfig::default() })]
        #[test]
        fn single_pass_matches_iterative_on_random_binary_masks(
            (h, w, d, raster) in (1usize..=8, 1usize..=8, 1usize..=4)
                .prop_flat_map(|(h, w, d)| {
                    let len = h * w;
                    (
                        Just(h),
                        Just(w),
                        Just(d),
                        proptest::collection::vec(0u8..=1, len..=len),
                    )
                }),
        ) {
            let single_pass = erode_raster_chebyshev(&raster, h, w, d);
            let iterative = erode_iterative_reference(&raster, h, w, d);
            prop_assert_eq!(single_pass, iterative);
        }

        // Sized to span every `ph % 8` residue (h ∈ 6..=22 → ph ∈ 8..=24)
        // so the row pass exercises both the bulk u64 chunks and the
        // scalar tail across all chunk counts up to 3.
        #[test]
        fn single_pass_matches_iterative_across_u64_chunk_boundaries(
            (h, w, d, raster) in (6usize..=22, 6usize..=22, 1usize..=5)
                .prop_flat_map(|(h, w, d)| {
                    let len = h * w;
                    (
                        Just(h),
                        Just(w),
                        Just(d),
                        proptest::collection::vec(0u8..=1, len..=len),
                    )
                }),
        ) {
            let single_pass = erode_raster_chebyshev(&raster, h, w, d);
            let iterative = erode_iterative_reference(&raster, h, w, d);
            prop_assert_eq!(single_pass, iterative);
        }

        #[test]
        fn erosion_is_subset_of_input(
            bytes in proptest::collection::vec(0u8..=1, 0..120),
            d in 1u32..=4,
        ) {
            // For any mask M and any radius r, erode(M, r) ⊆ M
            // because erosion is a min-filter. This is the property
            // the boundary-band XOR step (N5) relies on: the band is
            // M XOR erode(M) only because erode(M) ⊆ M.
            let len = bytes.len() as u32;
            let r = Rle::from_raster_bytes(&bytes, 1, len)?;
            let eroded = erode_chebyshev_ball(&r, d)?;
            let m = r.to_raster_bytes();
            let e = eroded.to_raster_bytes();
            for (mi, ei) in m.iter().zip(&e) {
                prop_assert!(*ei <= *mi);
            }
        }
    }
}
