#![no_main]
#![allow(clippy::unwrap_used, clippy::panic)]

//! Fuzz `vernier_mask::codec::decode_counts`.
//!
//! Entry point verified at `crates/vernier-mask/src/codec.rs:73`;
//! re-exported as `vernier_mask::decode_counts`.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = vernier_mask::decode_counts(data);
});
