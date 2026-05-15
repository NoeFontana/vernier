#![no_main]
#![allow(clippy::unwrap_used, clippy::panic)]

//! Fuzz `vernier_core::dataset::CocoDetections::from_json_bytes`.
//!
//! Entry point verified at `crates/vernier-core/src/dataset.rs:1212`.
//! `from_inputs` is the typed constructor it wraps; the bytes-in path
//! is the closer surface for a raw-byte fuzzer.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = vernier_core::CocoDetections::from_json_bytes(data);
});
