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
//! in two's complement (quirk **G3** disposition: `strict`).

use std::cell::RefCell;
use std::sync::Arc;

use crate::error::{MalformedRleReason, MaskError};

const SHIFT_CHUNK: u32 = 5;
const CONT_MASK: i64 = 0x20;
const SIGN_MASK: i64 = 0x10;
const DATA_MASK: i64 = 0x1F;
const ASCII_OFFSET: i64 = 0x30;
const WIRE_MIN: u8 = 0x30;
const WIRE_MAX: u8 = 0x6F;

/// Tight algebraic bound: a `u32` plus a sign-disambiguation bit
/// fits in 33 bits, which at 5 data bits per char is 7 chars.
/// Inputs that produce more chars cannot represent a valid `u32`
/// run length and are rejected to bound work on malformed input.
const MAX_CHARS_PER_RUN: u32 = 7;

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
    let mut counts = Vec::with_capacity(s.len());
    if decode_fast(s, &mut counts) {
        Ok(counts)
    } else {
        decode_checked(s)
    }
}

thread_local! {
    /// [`decode_counts_shared`]'s decode buffer, reused across calls so a
    /// mask costs one exact allocation (the shared buffer) instead of a
    /// growable one plus a copy.
    static DECODE_SCRATCH: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
}

/// [`decode_counts`] into the shared buffer [`crate::Rle`] keeps.
pub(crate) fn decode_counts_shared(s: &[u8]) -> Result<Arc<[u32]>, MaskError> {
    DECODE_SCRATCH.with_borrow_mut(|scratch| {
        scratch.clear();
        if decode_fast(s, scratch) {
            Ok(Arc::from(scratch.as_slice()))
        } else {
            decode_checked(s).map(Arc::from)
        }
    })
}

/// Decodes a well-formed string onto `counts` and returns `true`; `false`
/// for anything else, which [`decode_checked`] then reports with its
/// canonical error.
///
/// Runs `i > 2` carry the difference to the run two back, which is small:
/// on RF-DETR masks 74% of runs are one char, 24.5% two, 1.4% longer. The
/// one- and two-char cases take a select-based path — `byte - 0x30 <
/// 0x20` both rejects anything below the wire alphabet and says the run
/// ends there, and a 5-bit value sign-extends with two shifts. Longer
/// runs take the general loop.
fn decode_fast(s: &[u8], counts: &mut Vec<u32>) -> bool {
    decode_fast_inner(s, counts).is_some()
}

fn decode_fast_inner(s: &[u8], counts: &mut Vec<u32>) -> Option<()> {
    counts.reserve(s.len());
    let mut i = 0usize;
    while i < s.len() {
        // Quirk G3: the sign bit of a run's last char extends.
        let sext5 = |c: u8| (i64::from(c) << 59) >> 59;
        let c0 = s[i].wrapping_sub(WIRE_MIN);
        let c1 = s.get(i + 1).map_or(u8::MAX, |b| b.wrapping_sub(WIRE_MIN));
        let one = c0 < CONT_MASK as u8;
        let mut x = if one || (c0 <= (DATA_MASK | CONT_MASK) as u8 && c1 < CONT_MASK as u8) {
            // A one- or two-char run (99% of them on real masks): both
            // values are computed and one selected, so the length — 1 or 2
            // about three to one, unpredictably — never steers a branch.
            let two = i64::from(c0) & DATA_MASK | sext5(c1) << SHIFT_CHUNK;
            i += if one { 1 } else { 2 };
            if one {
                sext5(c0)
            } else {
                two
            }
        } else {
            let mut x = 0_i64;
            let mut shift = 0_u32;
            loop {
                let c = i64::from(s.get(i)?.wrapping_sub(WIRE_MIN));
                if c > DATA_MASK | CONT_MASK || shift >= SHIFT_CHUNK * MAX_CHARS_PER_RUN {
                    return None;
                }
                i += 1;
                x |= (c & DATA_MASK) << shift;
                shift += SHIFT_CHUNK;
                if c & CONT_MASK == 0 {
                    if c & SIGN_MASK != 0 {
                        x |= !0_i64 << shift;
                    }
                    break x;
                }
            }
        };
        if counts.len() > 2 {
            x += i64::from(counts[counts.len() - 2]);
        }
        counts.push(u32::try_from(x).ok()?);
    }
    Some(())
}

/// The checked byte-at-a-time decode: every condition tested where it
/// arises, so a malformed string reports the error the format defines.
fn decode_checked(s: &[u8]) -> Result<Vec<u32>, MaskError> {
    let mut counts = Vec::with_capacity(s.len() / 2 + 1);
    let mut iter = s.iter().copied();
    while let Some(mut byte) = iter.next() {
        let mut x: i64 = 0;
        let mut k: u32 = 0;
        loop {
            if !(WIRE_MIN..=WIRE_MAX).contains(&byte) {
                return Err(MaskError::MalformedRle(MalformedRleReason::ByteOutOfRange));
            }
            if k >= MAX_CHARS_PER_RUN {
                return Err(MaskError::MalformedRle(
                    MalformedRleReason::ChunkLimitExceeded,
                ));
            }
            // byte is provably in [0x30, 0x6F], so c lands in [0, 0x3F].
            let c = (byte as i64) - ASCII_OFFSET;
            x |= (c & DATA_MASK) << (SHIFT_CHUNK * k);
            let more = c & CONT_MASK != 0;
            k += 1;
            if !more {
                if c & SIGN_MASK != 0 {
                    x |= !0_i64 << (SHIFT_CHUNK * k);
                }
                break;
            }
            byte = iter
                .next()
                .ok_or(MaskError::MalformedRle(MalformedRleReason::TruncatedString))?;
        }
        if counts.len() > 2 {
            x += counts[counts.len() - 2] as i64;
        }
        if !(0..=i64::from(u32::MAX)).contains(&x) {
            return Err(MaskError::MalformedRle(MalformedRleReason::U32Overflow));
        }
        counts.push(x as u32);
    }
    Ok(counts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// The fast path decodes exactly what the checked decoder does and
        /// errs exactly where it errs — the same reason — on any byte string:
        /// round-tripped counts (one-, two- and long-char runs), near-wire
        /// noise (out-of-alphabet bytes, over-long and truncated runs, `u32`
        /// overflow), and the empty string; and the shared-buffer decode
        /// [`crate::Rle`] uses agrees with both.
        #[test]
        fn fast_decode_agrees_with_the_checked_decoder(
            counts in proptest::collection::vec(prop_oneof![0_u32..40, any::<u32>()], 0..60),
            noise in proptest::collection::vec(0x2e_u8..0x72, 0..80),
            splice in any::<bool>(),
        ) {
            let mut wire = encode_counts(&counts);
            if splice {
                wire.extend_from_slice(&noise);
            }
            for s in [&wire[..], &noise[..]] {
                let checked = format!("{:?}", decode_checked(s));
                prop_assert_eq!(format!("{:?}", decode_counts(s)), checked.clone());
                prop_assert_eq!(
                    format!("{:?}", decode_counts_shared(s).map(|a| a.to_vec())),
                    checked
                );
            }
            prop_assert_eq!(decode_counts(&encode_counts(&counts)).unwrap(), counts);
        }
    }

    /// A run past seven chars is refused on the fast path too, even when
    /// its value would fit — `P` is a continuation char carrying zero.
    #[test]
    fn an_over_long_run_is_refused_even_when_its_value_fits() {
        for wire in [&b"PPPPPPP0"[..], &b"0PPPPPPPP0"[..]] {
            assert!(matches!(
                decode_counts(wire),
                Err(MaskError::MalformedRle(
                    MalformedRleReason::ChunkLimitExceeded
                ))
            ));
        }
        assert_eq!(decode_counts(b"PPPPPP0").unwrap(), vec![0]);
    }

    #[test]
    fn empty_round_trip() {
        assert_eq!(encode_counts(&[]), b"");
        assert_eq!(decode_counts(b"").unwrap(), Vec::<u32>::new());
    }

    #[test]
    fn small_single_char_runs() {
        assert_eq!(encode_counts(&[0]), b"0");
        assert_eq!(encode_counts(&[5]), b"5");
        assert_eq!(decode_counts(b"0").unwrap(), vec![0]);
        assert_eq!(decode_counts(b"5").unwrap(), vec![5]);
    }

    #[test]
    fn value_16_needs_two_chars_for_sign_disambiguation() {
        assert_eq!(encode_counts(&[16]), b"`0");
        assert_eq!(decode_counts(b"`0").unwrap(), vec![16]);
    }

    #[test]
    fn differential_kicks_in_at_index_three() {
        let counts = vec![1, 2, 3, 4, 5];
        let encoded = encode_counts(&counts);
        assert_eq!(encoded, b"12322");
        assert_eq!(decode_counts(&encoded).unwrap(), counts);
    }

    #[test]
    fn negative_differential_uses_sign_extension() {
        // counts[3] - counts[1] = -1, encoded as 'O' = 0x4F (sign bit set, no continuation).
        let counts = vec![1, 2, 3, 1];
        let encoded = encode_counts(&counts);
        assert_eq!(encoded, b"123O");
        assert_eq!(decode_counts(&encoded).unwrap(), counts);
    }

    #[test]
    fn rejects_out_of_range_byte() {
        assert_eq!(
            decode_counts(b"5\x20").unwrap_err().to_string(),
            "malformed RLE counts string: byte outside legal [0x30, 0x6F] range",
        );
        assert!(matches!(
            decode_counts(b"5p"),
            Err(MaskError::MalformedRle(MalformedRleReason::ByteOutOfRange))
        ));
    }

    #[test]
    fn rejects_truncated_run() {
        // '`' = 0x60 sets the continuation bit but no follow-up byte arrives.
        assert!(matches!(
            decode_counts(b"`"),
            Err(MaskError::MalformedRle(MalformedRleReason::TruncatedString))
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
