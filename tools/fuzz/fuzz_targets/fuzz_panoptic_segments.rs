#![no_main]
#![allow(clippy::unwrap_used, clippy::panic)]

//! Raw serde shape for the panoptic segments JSON (see README).

use libfuzzer_sys::fuzz_target;
use std::collections::HashMap;

fuzz_target!(|data: &[u8]| {
    let _ = serde_json::from_slice::<HashMap<String, Vec<serde_json::Value>>>(data);
});
