# vernier-cli

[![Crates.io](https://img.shields.io/crates/v/vernier-cli.svg)](https://crates.io/crates/vernier-cli)

Command-line driver for the [vernier](https://github.com/NoeFontana/vernier) evaluation library.

The `vernier` binary runs COCO-format evaluation (bbox, segm, boundary,
keypoints) without a Python interpreter. CI pipelines, robotics replay
pipelines, and shell-driven non-regression checks are the audience —
[ADR-0015](https://github.com/NoeFontana/vernier/blob/main/docs/adr/0015-vernier-cli.md)
captures the design.

## Installation

Pre-built binaries (recommended for CI):

```sh
# cargo binstall — downloads the right binary for your host triple
cargo binstall vernier-cli

# Or curl-pipe-bash from the GitHub Release
curl -L https://github.com/NoeFontana/vernier/releases/latest/download/vernier-installer.sh | bash
```

From source:

```sh
cargo install vernier-cli
```

Pre-built binaries ship for `x86_64-unknown-linux-gnu`,
`aarch64-unknown-linux-gnu`, `aarch64-apple-darwin`, and
`x86_64-pc-windows-msvc`.

## Usage

```sh
# Strict-mode parity with pycocotools' summarize() stdout
vernier eval --gt gt.json --dt dt.json --iou-type bbox

# Multi-emit: one eval, multiple outputs
vernier eval --gt gt.json --dt dt.json --iou-type segm \
    --emit text \
    --emit json=result.json
```

Output is byte-deterministic across runs of the same input — no timestamps, no
host metadata, sorted JSON keys, atomic file writes. CI archives diff cleanly
and regression-tracking buckets work out of the box.

The full surface (`--parity-mode`, `--max-dets`, `--use-cats`,
`--dilation-ratio`, `--sigmas`, `--emit FMT[=PATH]`, `--quiet`) is documented in
[ADR-0015 §"Surface"](https://github.com/NoeFontana/vernier/blob/main/docs/adr/0015-vernier-cli.md#surface).

## Exit codes

| Code | Meaning |
|------|---------|
| 0    | Eval ran, summary written |
| 1    | Eval ran but failed (parity-mode constraint, malformed JSON, kernel error) |
| 2    | Argument parse / validation failure |

## Architectural notes

`vernier-cli` is a thin parse-and-format layer over `vernier-core`. It links
neither `vernier-ffi` nor PyO3 — pure-Rust dep tree per
[ADR-0015](https://github.com/NoeFontana/vernier/blob/main/docs/adr/0015-vernier-cli.md).
The binary's stability commitments (flag surface, JSON schema version, output
determinism) are pinned from its first ship under the 0.0.x release line.

## License

Dual-licensed under MIT or Apache-2.0, at your option.
