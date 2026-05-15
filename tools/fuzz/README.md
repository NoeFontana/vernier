# vernier-fuzz

libFuzzer harnesses for vernier's untrusted-input parsers, driven by
[`cargo-fuzz`](https://rust-fuzz.github.io/book/cargo-fuzz.html). This
crate is intentionally outside the main Cargo workspace so its nightly
toolchain and libFuzzer link flags don't bleed into the rest of the
repo.

## Running

From the workspace root (cargo-fuzz walks up to the nearest Cargo.toml
and expects `fuzz/` next to it, so `--fuzz-dir` is mandatory here):

```bash
cargo +nightly fuzz build --fuzz-dir tools/fuzz
cargo +nightly fuzz run --fuzz-dir tools/fuzz <target> -- -max_total_time=120
```

| Target | Probes |
| --- | --- |
| `fuzz_coco_dataset` | `CocoDataset::from_json_bytes` |
| `fuzz_coco_detections` | `CocoDetections::from_json_bytes` |
| `fuzz_panoptic_segments` | Raw serde shape `HashMap<String, Vec<Value>>` — `parse_segments_map` lives in the PyO3-linked `vernier-ffi` and can't link into a `no_main` binary; swap this body when a public `vernier-core` loader lands |
| `fuzz_manifest` | `parse_manifest` |
| `fuzz_rle_counts` | `decode_counts` — COCO 6-bit RLE wire decoder |
| `fuzz_segmentation` | Untagged `Segmentation` enum |

## Regression corpus

Minimised reproducers go in `tools/fuzz/regressions/<target>/*.bin`
and replay on every `cargo test` via
`crates/vernier-core/tests/fuzz_regressions.rs`.
