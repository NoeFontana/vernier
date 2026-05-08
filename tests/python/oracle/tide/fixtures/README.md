# TIDE oracle fixtures

Each subdirectory holds one `gt.json` + `dt.json` pair pinned by both
languages:

- Python — `tests/python/oracle/tide/test_oracle.py`
- Rust — `crates/vernier-core/tests/{tide_oracle_parity,tide_fp_iou_histogram,confusion_matrix}.rs`

The Rust tests read these files via the hardcoded relative path
`../../tests/python/oracle/tide/fixtures/` from each crate's
`CARGO_MANIFEST_DIR`. **Renaming or relocating any subdirectory here
silently breaks the Rust integration tests** — update the Rust
consumers in lockstep, or move the path resolution behind a shared
constant.

The matching expected outputs live alongside in `../expected/<name>.json`
and are likewise consumed by both languages (see `expected/README.md`).
