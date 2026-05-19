//! SIMD-accelerated RGB → `u32` pack used by the panoptic PNG decoder.
//!
//! Each output `u32` is `R | G<<8 | B<<16` per the panoptic PNG
//! convention (the same `R + 256·G + 256²·B` formula panopticapi uses
//! in its `evaluation.py` decode).
//!
//! ## Unsafe policy
//!
//! The host crate is `#![deny(unsafe_code)]` at root and every other
//! module is unsafe-free. The single unsafe site — the SSSE3
//! `pshufb`-based [`pack_rgb_row_ssse3`] — is marked with
//! `#[allow(unsafe_code)]`. The unsafe is necessary because
//! `std::arch` intrinsics under `#[target_feature]` are `unsafe fn`
//! (caller must guarantee feature availability;
//! `is_x86_feature_detected!` is the runtime guard). The body uses
//! `_mm_loadu_si128` / `_mm_shuffle_epi8` / `_mm_storeu_si128` over
//! pointer arithmetic bounds-checked at the dispatch site via
//! `simd_pixels`.

/// Pack `row_bytes` (length `3 * dst.len()`, RGB triples) into `dst`
/// as `u32` ids. Each output id is `R | G<<8 | B<<16` per the
/// panoptic PNG convention; the high byte is always zero.
///
/// Dispatch picks the SSSE3 `pshufb` fast path when available on
/// `x86_64`; falls back to a scalar loop otherwise. The dispatch
/// itself is a single feature-detect on first call, cached
/// process-wide.
///
/// # Panics
/// Debug-only: panics when `row_bytes.len() != 3 * dst.len()`.
#[inline]
pub(crate) fn pack_rgb_row(row_bytes: &[u8], dst: &mut [u32]) {
    debug_assert_eq!(row_bytes.len(), dst.len() * 3);
    #[cfg(target_arch = "x86_64")]
    {
        // `is_x86_feature_detected!` caches the result via std's
        // internal `Once`; calls past the first are a relaxed atomic
        // load + branch — cheap enough to leave on the hot path.
        if std::arch::is_x86_feature_detected!("ssse3") {
            return pack_rgb_row_ssse3_dispatch(row_bytes, dst);
        }
    }
    pack_rgb_row_scalar(row_bytes, dst);
}

/// Scalar reference implementation. Exposed so tests + downstream
/// fuzz harnesses can pin SIMD/scalar bit-equality.
#[inline]
pub(crate) fn pack_rgb_row_scalar(row_bytes: &[u8], dst: &mut [u32]) {
    debug_assert_eq!(row_bytes.len(), dst.len() * 3);
    for (i, slot) in dst.iter_mut().enumerate() {
        let off = i * 3;
        *slot = u32::from(row_bytes[off])
            | (u32::from(row_bytes[off + 1]) << 8)
            | (u32::from(row_bytes[off + 2]) << 16);
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn pack_rgb_row_ssse3_dispatch(row_bytes: &[u8], dst: &mut [u32]) {
    let n = dst.len();
    // The SIMD loop loads 16 bytes per 4-pixel iteration starting at
    // `i * 3`. To keep every load in-bounds we need
    // `i * 3 + 16 <= 3 * n`, i.e. `i <= n - 16/3`. Snap down so
    // `simd_pixels` is a multiple of 4 and the last SIMD iteration
    // never overruns. The scalar tail handles the remaining ≤ 9
    // pixels.
    let simd_pixels = if n >= 6 { ((n - 6) / 4) * 4 + 4 } else { 0 };
    // SAFETY: SSSE3 was feature-checked at the public-API call site,
    // and `simd_pixels` is bounded so every 16-byte load reads
    // in-bounds (`i * 3 + 16 <= 3 * n` for all
    // `i ∈ [0, simd_pixels)` stepping by 4) and every 16-byte store
    // fits inside `dst[0..simd_pixels]`.
    #[allow(unsafe_code)]
    unsafe {
        pack_rgb_row_ssse3(row_bytes, dst, simd_pixels);
    }
    for (slot, chunk) in dst[simd_pixels..]
        .iter_mut()
        .zip(row_bytes[simd_pixels * 3..].chunks_exact(3))
    {
        *slot = u32::from(chunk[0]) | (u32::from(chunk[1]) << 8) | (u32::from(chunk[2]) << 16);
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
#[target_feature(enable = "ssse3")]
unsafe fn pack_rgb_row_ssse3(row_bytes: &[u8], dst: &mut [u32], simd_pixels: usize) {
    use std::arch::x86_64::{
        __m128i, _mm_loadu_si128, _mm_setr_epi8, _mm_shuffle_epi8, _mm_storeu_si128,
    };
    // `pshufb` control mask: byte `i` of the output is `input[mask[i]]`,
    // or `0x00` when `mask[i] >= 0x80` (sign bit set). Mapping
    // `[R0 G0 B0 R1 G1 B1 R2 G2 B2 R3 G3 B3 _ _ _ _]` →
    // `[R0 G0 B0 0 R1 G1 B1 0 R2 G2 B2 0 R3 G3 B3 0]` builds 4
    // contiguous `u32` ids in one `pshufb`.
    let mask = _mm_setr_epi8(0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1, 9, 10, 11, -1);
    let row_ptr = row_bytes.as_ptr();
    let dst_ptr = dst.as_mut_ptr();
    let mut i = 0;
    // Caller's `simd_pixels` bound guarantees `i * 3 + 16 <= row_bytes.len()`
    // and `i + 4 <= dst.len()` for every iteration, so the 16-byte
    // load and the 16-byte store both sit fully in-bounds.
    while i < simd_pixels {
        let chunk = _mm_loadu_si128(row_ptr.add(i * 3) as *const __m128i);
        let packed = _mm_shuffle_epi8(chunk, mask);
        _mm_storeu_si128(dst_ptr.add(i) as *mut __m128i, packed);
        i += 4;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn random_rgb(n: usize, seed: u64) -> Vec<u8> {
        let mut state = seed;
        (0..n * 3)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as u8
            })
            .collect()
    }

    #[test]
    fn empty_is_noop() {
        let mut dst = Vec::<u32>::new();
        pack_rgb_row(&[], &mut dst);
        assert!(dst.is_empty());
    }

    #[test]
    fn matches_scalar_across_sizes() {
        // Cover small sizes (forces every-pixel scalar tail), sizes
        // near SIMD-block boundaries (4, 8, 12), the worst-case
        // overrun-prevention boundary (5, 6, 7, 9), and a typical
        // PNG row width (640).
        for &n in &[0usize, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 17, 100, 640] {
            let bytes = random_rgb(n, 0xdead_beef_cafe_babe);
            let mut a = vec![0u32; n];
            let mut b = vec![0u32; n];
            pack_rgb_row(&bytes, &mut a);
            pack_rgb_row_scalar(&bytes, &mut b);
            assert_eq!(a, b, "mismatch at n={n}");
        }
    }

    proptest! {
        #[test]
        fn pack_matches_scalar_random(n in 0usize..=300) {
            let bytes = random_rgb(n, 42);
            let mut simd_out = vec![0u32; n];
            let mut scalar_out = vec![0u32; n];
            pack_rgb_row(&bytes, &mut simd_out);
            pack_rgb_row_scalar(&bytes, &mut scalar_out);
            prop_assert_eq!(simd_out, scalar_out);
        }
    }
}
