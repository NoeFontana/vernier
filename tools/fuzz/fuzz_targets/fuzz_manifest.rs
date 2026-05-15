#![no_main]
#![allow(clippy::unwrap_used, clippy::panic)]

//! Fuzz `vernier_core::manifest::parse_manifest`.
//!
//! Entry point verified at `crates/vernier-core/src/manifest.rs:122`.
//! The function needs `known_image_ids` and `known_labels` sets to
//! validate cross-references; we pass empty sets so the fuzzer is
//! probing the JSON/CSV parsing path itself rather than any specific
//! reference-integrity rule.

use libfuzzer_sys::fuzz_target;
use std::collections::HashSet;

fuzz_target!(|data: &[u8]| {
    let images = HashSet::new();
    let labels = HashSet::new();
    let _ = vernier_core::manifest::parse_manifest(data, &images, &labels);
});
