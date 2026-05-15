#![no_main]
#![allow(clippy::unwrap_used, clippy::panic)]

//! Fuzz the untagged `Segmentation` enum deserializer.
//!
//! `Segmentation` is defined at
//! `crates/vernier-core/src/segmentation.rs:49` and re-exported via
//! the module path. It's an untagged enum (polygons vs RLE), so the
//! fuzzer exercises serde's discrimination logic as well as each
//! variant's body.

use libfuzzer_sys::fuzz_target;
use vernier_core::segmentation::Segmentation;

fuzz_target!(|data: &[u8]| {
    let _ = serde_json::from_slice::<Segmentation>(data);
});
