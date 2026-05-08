# vernier-partial

[![Crates.io](https://img.shields.io/crates/v/vernier-partial.svg)](https://crates.io/crates/vernier-partial)
[![Docs.rs](https://docs.rs/vernier-partial/badge.svg)](https://docs.rs/vernier-partial)

Shared wire envelope and partition policy for
[vernier](https://github.com/NoeFontana/vernier)'s distributed-evaluation
partials
([ADR-0031](https://github.com/NoeFontana/vernier/blob/main/docs/adr/0031-dist-eval.md),
generalized in
[ADR-0032](https://github.com/NoeFontana/vernier/blob/main/docs/adr/0032-dist-eval-paradigms.md)).

This crate is a leaf: every paradigm crate (`vernier-core` for instance AP,
`vernier-semantic` for mIoU, `vernier-panoptic` for PQ) depends on it for the
wire format; nothing depends back. Keeping the dep DAG flat is what makes the
cross-paradigm refactor work without circular dependencies.

## What lives here

- The wire envelope (`WireEnvelopeHeader`), magic + version constants
  (`MAGIC`, `FORMAT_VERSION`), and framing helpers (`encode`,
  `with_validated_envelope`).
- The five typed errors (`PartialError::*`) mapped 1:1 to the Python
  exception hierarchy.
- The `Partial`, `ParadigmKind`, and `PartialExpectation` traits that
  paradigm crates implement.
- The paradigm-agnostic merge policy (`BaseMergeAccumulator`).

## What does *not* live here

- Paradigm-specific body types (per-image cells, confusion matrices,
  `PqStat` aggregations). Each paradigm owns its body archive and decode.
- Dataset / params hashing. Each paradigm hashes its own canonical form;
  the 32-byte hashes carried in the envelope are opaque to this crate.
- Streaming evaluator state. `vernier-partial` validates envelopes; it
  knows nothing about evaluator internals.

## Wire format

The envelope is a `rkyv` archive prefixed with a magic number and a
`FORMAT_VERSION` field, followed by a paradigm-specific body archive, and
terminated by a CRC32 footer that catches transport corruption rkyv's archive
validator misses. Cross-paradigm merges are structurally rejected: a partial
encoded as instance cannot be merged into a panoptic accumulator without
typed-error feedback.

`FORMAT_VERSION` is currently `2` — the 1→2 hard break landed alongside
ADR-0032's cross-paradigm generalization.

## Installation

Most users won't depend on `vernier-partial` directly — it's pulled in
transitively by `vernier-core`, `vernier-semantic`, and `vernier-panoptic`.
Direct use is for tooling that produces or consumes the wire format outside
the paradigm crates (custom shard runners, debugging tools).

```toml
[dependencies]
vernier-partial = "0.0.1"
```

Stable Rust only (MSRV in `rust-toolchain.toml`, currently 1.83).

## License

Dual-licensed under MIT or Apache-2.0, at your option.
