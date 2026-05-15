//! Regression gate for the `tools/fuzz/` harnesses.
//!
//! For each minimised crash input checked into
//! `tools/fuzz/regressions/<target_name>/*.bin`, re-run the matching
//! entry point and assert it returns (doesn't panic). If the
//! regressions directory doesn't exist, or any target subdirectory is
//! missing, the test trivially passes — this keeps the gate quiet
//! until real reproducers are checked in.

use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::PathBuf;

use vernier_core::segmentation::Segmentation;
use vernier_core::{CocoDataset, CocoDetections};

/// Resolve `tools/fuzz/regressions/` relative to this crate's manifest.
/// Returns `None` if `CARGO_MANIFEST_DIR` is unset; callers treat that as
/// "no corpus to replay" so the test stays quiet outside of cargo.
fn regressions_root() -> Option<PathBuf> {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").ok()?;
    Some(
        PathBuf::from(manifest_dir)
            .join("..")
            .join("..")
            .join("tools")
            .join("fuzz")
            .join("regressions"),
    )
}

/// Collect every `*.bin` file under `<root>/<target>/`. Returns an
/// empty vector if the directory does not exist.
fn corpus_for(target: &str) -> Vec<PathBuf> {
    let Some(dir) = regressions_root().map(|r| r.join(target)) else {
        return Vec::new();
    };
    let Ok(entries) = fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "bin") {
            out.push(path);
        }
    }
    out.sort();
    out
}

/// Replay each input against `f`. `f` must not panic on any input.
/// Files that fail to read are skipped with no assertion — the gate
/// only fires on actual panics inside `f`.
fn replay<F: Fn(&[u8])>(target: &str, f: F) {
    for path in corpus_for(target) {
        if let Ok(bytes) = fs::read(&path) {
            f(&bytes);
        }
    }
}

#[test]
fn fuzz_coco_dataset_regressions() {
    replay("fuzz_coco_dataset", |data| {
        let _ = CocoDataset::from_json_bytes(data);
    });
}

#[test]
fn fuzz_coco_detections_regressions() {
    replay("fuzz_coco_detections", |data| {
        let _ = CocoDetections::from_json_bytes(data);
    });
}

#[test]
fn fuzz_manifest_regressions() {
    let images = HashSet::new();
    let labels = HashSet::new();
    replay("fuzz_manifest", |data| {
        let _ = vernier_core::manifest::parse_manifest(data, &images, &labels);
    });
}

#[test]
fn fuzz_rle_counts_regressions() {
    replay("fuzz_rle_counts", |data| {
        let _ = vernier_mask::decode_counts(data);
    });
}

#[test]
fn fuzz_segmentation_regressions() {
    replay("fuzz_segmentation", |data| {
        let _ = serde_json::from_slice::<Segmentation>(data);
    });
}

#[test]
fn fuzz_panoptic_segments_regressions() {
    use std::collections::HashMap;
    replay("fuzz_panoptic_segments", |data| {
        let _ = serde_json::from_slice::<HashMap<String, Vec<serde_json::Value>>>(data);
    });
}
