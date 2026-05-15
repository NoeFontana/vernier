#![no_main]
#![allow(clippy::unwrap_used, clippy::panic)]

//! Fuzz `vernier_core::dataset::CocoDataset::from_json_bytes`.
//!
//! Entry point verified at `crates/vernier-core/src/dataset.rs:484`.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = vernier_core::CocoDataset::from_json_bytes(data);
});
