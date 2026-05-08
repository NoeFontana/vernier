# vernier-semantic

[![Crates.io](https://img.shields.io/crates/v/vernier-semantic.svg)](https://crates.io/crates/vernier-semantic)
[![Docs.rs](https://docs.rs/vernier-semantic/badge.svg)](https://docs.rs/vernier-semantic)

Semantic-segmentation evaluation (mIoU + per-class IoU + pixel accuracy) for the
[vernier](https://github.com/NoeFontana/vernier) library
([ADR-0028](https://github.com/NoeFontana/vernier/blob/main/docs/adr/0028-sem-seg.md)).

This crate is a sibling to `vernier-core` (instance AP) and `vernier-panoptic`
(PQ). Unlike `vernier-panoptic`, it depends on `vernier-core` for shared
abstractions: `ParityMode` ([ADR-0002][adr2]), `Breakdown` ([ADR-0016][adr16]),
the streaming-evaluator interface ([ADR-0013][adr13]), and the result-tables
interface ([ADR-0019][adr19]). The asymmetry is concrete reuse, not
philosophical isolation —
[ADR-0028 §"Workspace and dependency direction"](https://github.com/NoeFontana/vernier/blob/main/docs/adr/0028-sem-seg.md)
ratifies it.

The architectural firewall enforced by ADR-0005 still holds: this crate has no
edge to `matching.rs`, `accumulate.rs`, or the `Similarity` trait — the AP fold
is unreachable from semantic-mIoU code by construction. The kernel here is a
per-image confusion matrix accumulator, not a per-detection matching loop.

## Parity oracles

vernier-semantic targets three reference implementations under the three-tier
parity model:

- [open-mmlab/mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
  (ADE20K, Pascal VOC, Cityscapes presets)
- [mcordts/cityscapesScripts](https://github.com/mcordts/cityscapesScripts)
- The Pascal VOC and ADE20K reference scripts

The full quirk disposition table lives in
[`docs/engineering/sem-seg-quirks.md`](https://github.com/NoeFontana/vernier/blob/main/docs/engineering/sem-seg-quirks.md).
Strict-mode oracle vendoring is in flight (PR-B6/B7/B8) — until those land,
parity is asserted against in-tree synthetic fixtures.

## Installation

```toml
[dependencies]
vernier-semantic = "0.0.1"
```

Stable Rust only (MSRV in `rust-toolchain.toml`, currently 1.83).

## Minimal usage

```rust
use vernier_semantic::{accumulate_confusion, ConfusionMatrix};

let mut cm = ConfusionMatrix::new(/* n_classes */ 21, /* ignore_label */ Some(255));
accumulate_confusion(&mut cm, &gt_mask, &dt_mask)?;
let summary = cm.summarize();
println!("mIoU = {:.4}", summary.miou);
```

## License

Dual-licensed under MIT or Apache-2.0, at your option.

[adr2]: https://github.com/NoeFontana/vernier/blob/main/docs/adr/0002-three-tier-parity-model.md
[adr13]: https://github.com/NoeFontana/vernier/blob/main/docs/adr/0013-streaming-evaluator.md
[adr16]: https://github.com/NoeFontana/vernier/blob/main/docs/adr/0016-generalized-breakdown-axis.md
[adr19]: https://github.com/NoeFontana/vernier/blob/main/docs/adr/0019-result-tables.md
