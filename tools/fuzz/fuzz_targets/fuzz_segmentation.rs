#![no_main]
#![allow(clippy::unwrap_used, clippy::panic)]

//! Fuzz the `Segmentation` deserializer.
//!
//! `vernier_core::segmentation::Segmentation` reads by JSON shape through
//! hand-written visitors (an array is polygons, an object is RLE; counts
//! are a string or a run-length array), so the fuzzer exercises that
//! dispatch as well as each variant's body. It must never panic, only
//! accept or refuse.

use libfuzzer_sys::fuzz_target;
use vernier_core::segmentation::Segmentation;

fuzz_target!(|data: &[u8]| {
    let _ = serde_json::from_slice::<Segmentation>(data);
});
