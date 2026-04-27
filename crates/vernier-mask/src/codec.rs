//! 6-bit char string codec for COCO RLE counts.
//!
//! Mirrors `rleToString` (`mc:237-250`) and `rleFrString`
//! (`mc:252-267`) from `pycocotools-2.0.11/common/maskApi.c`.
//!
//! Wire format (quirks **G1**, **G2**, **G3**, **K3**):
//! - 5 data bits per char in the low nibble + bit 4 (sign).
//! - Bit 5 (`0x20`) is the continuation flag: more chars to follow.
//! - Bit 4 (`0x10`) is the sign bit, valid only on the last char of
//!   a run (continuation cleared). Triggers two's-complement sign
//!   extension on decode.
//! - Each char `c ∈ [0x00, 0x3F]` is shifted into the printable
//!   range by `c += 0x30` on emit (and reversed on decode), giving
//!   a wire alphabet of `[0x30, 0x6F]` (`'0'..='o'`).
//! - From index `i > 2` (zero-indexed: starting at the 4th run),
//!   the encoder writes `cnts[i] - cnts[i-2]`; the decoder undoes
//!   this with `x += cnts[m-2]` after decoding.
//!
//! The C `long` is at least 32 bits and on every modern platform
//! 64; we use `i64` directly. The C left-shift of a negative value
//! (`-1 << 5*k`) is implementation-defined; we use the explicit
//! Rust formulation `!0_i64 << shift` which is the same bit pattern
//! in two's complement (quirk **G3** disposition: `aligned`).

use crate::error::MaskError;

const SHIFT_CHUNK: u32 = 5;
const CONT_MASK: i64 = 0x20;
const SIGN_MASK: i64 = 0x10;
const DATA_MASK: i64 = 0x1F;
const ASCII_OFFSET: i64 = 0x30;
const WIRE_MIN: u8 = 0x30;
const WIRE_MAX: u8 = 0x6F;

/// Maximum chars per run. A `u32` plus sign bit fits in 33 bits;
/// at 5 data bits per char that's 7 chars (35 bits). 13 leaves
/// generous headroom and prevents `5 * k` from approaching the
/// 64-bit shift boundary on malformed input.
const MAX_CHARS_PER_RUN: u32 = 13;

/// Encodes a slice of run lengths into the COCO 6-bit char string.
///
/// Output bytes are in `b'0'..=b'o'`. Round-trips through
/// [`decode_counts`] to the exact input.
pub fn encode_counts(counts: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(counts.len() * 2);
    for (i, &c) in counts.iter().enumerate() {
        let mut x = c as i64;
        if i > 2 {
            x -= counts[i - 2] as i64;
        }
        let mut more = true;
        while more {
            let chunk = x & DATA_MASK;
            x >>= SHIFT_CHUNK;
            more = if chunk & SIGN_MASK != 0 {
                x != -1
            } else {
                x != 0
            };
            let byte_val = chunk | if more { CONT_MASK } else { 0 };
            out.push((byte_val + ASCII_OFFSET) as u8);
        }
    }
    out
}

/// Decodes a COCO 6-bit char string into a vector of run lengths.
///
/// Errors on bytes outside `[0x30, 0x6F]`, on a string that ends
/// mid-run (continuation bit set on the final byte), or on a
/// decoded run length that does not fit in `u32`.
pub fn decode_counts(s: &[u8]) -> Result<Vec<u32>, MaskError> {
    let mut counts = Vec::with_capacity(s.len() / 2 + 1);
    let mut p = 0usize;
    while p < s.len() {
        let mut x: i64 = 0;
        let mut k: u32 = 0;
        loop {
            if p >= s.len() {
                return Err(MaskError::MalformedRle {
                    reason: "string ended mid-run (continuation bit set on final byte)",
                });
            }
            let byte = s[p];
            if !(WIRE_MIN..=WIRE_MAX).contains(&byte) {
                return Err(MaskError::MalformedRle {
                    reason: "byte outside legal [0x30, 0x6F] range",
                });
            }
            if k >= MAX_CHARS_PER_RUN {
                return Err(MaskError::MalformedRle {
                    reason: "run encoding exceeds maximum char count",
                });
            }
            let c = (byte as i64) - ASCII_OFFSET;
            x |= (c & DATA_MASK) << (SHIFT_CHUNK * k);
            let more = c & CONT_MASK != 0;
            p += 1;
            k += 1;
            if !more {
                if c & SIGN_MASK != 0 {
                    x |= !0_i64 << (SHIFT_CHUNK * k);
                }
                break;
            }
        }
        if counts.len() > 2 {
            x += counts[counts.len() - 2] as i64;
        }
        if !(0..=i64::from(u32::MAX)).contains(&x) {
            return Err(MaskError::MalformedRle {
                reason: "decoded run length outside u32 range",
            });
        }
        counts.push(x as u32);
    }
    Ok(counts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn empty_round_trip() {
        assert_eq!(encode_counts(&[]), b"");
        assert_eq!(decode_counts(b"").unwrap(), Vec::<u32>::new());
    }

    #[test]
    fn small_single_char_runs() {
        // Each value < 16 fits in one char with no sign bit set.
        // Values 0..=9 map to ASCII '0'..='9'.
        assert_eq!(encode_counts(&[0]), b"0");
        assert_eq!(encode_counts(&[5]), b"5");
        assert_eq!(decode_counts(b"0").unwrap(), vec![0]);
        assert_eq!(decode_counts(b"5").unwrap(), vec![5]);
    }

    #[test]
    fn value_16_needs_two_chars_for_sign_disambiguation() {
        // 16 has bit 4 set in its low 5 bits; the encoder cannot
        // emit it as one char without it being mistaken for a
        // sign-extended -16, so a continuation char follows.
        assert_eq!(encode_counts(&[16]), b"`0");
        assert_eq!(decode_counts(b"`0").unwrap(), vec![16]);
    }

    #[test]
    fn differential_kicks_in_at_index_three() {
        // First three runs encoded as-is; from i>2 the encoder
        // writes counts[i] - counts[i-2].
        let counts = vec![1, 2, 3, 4, 5];
        let encoded = encode_counts(&counts);
        assert_eq!(encoded, b"12322");
        assert_eq!(decode_counts(&encoded).unwrap(), counts);
    }

    #[test]
    fn negative_differential_uses_sign_extension() {
        // counts[3] - counts[1] = 1 - 2 = -1, encoded as 'O'
        // (0x4F = 0x30 + 0x1F, sign bit set, no continuation).
        let counts = vec![1, 2, 3, 1];
        let encoded = encode_counts(&counts);
        assert_eq!(encoded, b"123O");
        assert_eq!(decode_counts(&encoded).unwrap(), counts);
    }

    #[test]
    fn rejects_out_of_range_byte() {
        assert!(matches!(
            decode_counts(b"5\x20"),
            Err(MaskError::MalformedRle { .. })
        ));
        assert!(matches!(
            decode_counts(b"5p"),
            Err(MaskError::MalformedRle { .. })
        ));
    }

    #[test]
    fn rejects_truncated_run() {
        // Continuation bit set ('`' = 0x60 = continuation) but
        // string ends before the next char arrives.
        assert!(matches!(
            decode_counts(b"`"),
            Err(MaskError::MalformedRle { .. })
        ));
    }

    proptest! {
        #[test]
        fn encode_decode_round_trip(counts in proptest::collection::vec(any::<u32>(), 0..200)) {
            let encoded = encode_counts(&counts);
            let decoded = decode_counts(&encoded)?;
            prop_assert_eq!(decoded, counts);
        }

        #[test]
        fn encoded_bytes_in_legal_range(counts in proptest::collection::vec(any::<u32>(), 0..200)) {
            for byte in encode_counts(&counts) {
                prop_assert!((WIRE_MIN..=WIRE_MAX).contains(&byte));
            }
        }
    }
}
