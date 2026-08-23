# Contributing to vernier

Welcome. This document covers the basics; see `docs/engineering/` for the
detailed standards once they're written.

## Repository layout

```
crates/
  vernier/           facade crate (ADR-0048); re-exports the five library crates, no logic
  vernier-core/      pure Rust evaluation logic; no Python dependency
  vernier-mask/      pure Rust COCO RLE codec, polygon rasterizer, mask ops (ADR-0009)
  vernier-ffi/       PyO3 bindings; data conversion only, no business logic
  vernier-cli/       `vernier` binary — workspace member per ADR-0015
  vernier-panoptic/  PQ evaluator (ADR-0025); sibling to vernier-core
  vernier-semantic/  mIoU evaluator (ADR-0028); sibling to vernier-core
  vernier-partial/   distributed-eval wire envelope (ADR-0031, ADR-0032)
python/
  vernier/           thin Python wrapper; the user-facing API lives here
docs/
  adr/               Architecture Decision Records
  ...                Diátaxis-organized documentation
tests/
  rust/              Rust integration tests
  python/            Python tests against the FFI boundary
```

## Development quickstart

Prerequisites:

- [Rust stable](https://rustup.rs/) (`rustc >= 1.83`, pinned in `rust-toolchain.toml`)
- [uv](https://docs.astral.sh/uv/) for the Python toolchain (Python `>= 3.10`)
- [just](https://github.com/casey/just) for task running
- [`cargo-nextest`](https://nexte.st/) for the Rust test runner
- [`cargo-deny`](https://github.com/EmbarkStudios/cargo-deny) for `just audit`
- [`cargo-hack`](https://github.com/taiki-e/cargo-hack) for the facade
  feature-powerset check in `just lint`

```bash
# One-time setup
just bootstrap

# Iterate
just develop      # fast incremental rebuild
just test         # Rust + Python tests
just lint         # clippy + ruff + pyright (read-only, CI-equivalent)
just fmt          # auto-format everything
just audit        # cargo-deny check
```

## ADR-driven workflow

Significant changes — anything that affects the public API, the FFI boundary,
the parity contract, the data model, or the build/distribution story — start
with an ADR draft, not a PR. The lifecycle:

1. **Propose.** Copy `docs/adr/template.md` to `docs/adr/NNNN-short-title.md`
   with the next available number. Fill in *Context*, *Decision*, and
   *Consequences*. Status starts as `proposed`.
2. **Review.** Open a PR containing only the ADR. Discuss in review until
   consensus is reached.
3. **Accept.** Status changes to `accepted` and the ADR is merged. ADRs are
   immutable from this point — supersede with a later ADR rather than
   editing accepted ones.
4. **Implement (Red).** Write a failing test in either Rust or Python that
   captures the agreed behavior.
5. **Implement (Green).** Minimum code to pass the test.
6. **Refactor & benchmark.** Clean up. For hot paths, verify with `divan`
   (Rust) or `pytest-benchmark` (Python) that performance has not regressed.

For small, mechanical changes (typo fixes, dependency bumps, internal
refactors with no API impact), skip the ADR and go straight to a PR.

### Vendoring third-party code

Adding a vendored reference (test-only) is an ADR-level decision. See
[`docs/engineering/vendoring.md`](docs/engineering/vendoring.md) for
the policy, layout convention, and the `VENDORING.md` template.

## Local checks before opening a PR

```bash
just lint    # clippy, ruff, pyright (must pass)
just test    # nextest + pytest (must pass)
just audit   # cargo-deny (must pass)
```

CI runs the same gates; running them locally first saves a round trip.

## Code style

- **Rust:** `rustfmt` defaults, `clippy` clean (workspace lints in
  `Cargo.toml` deny `unwrap_used`, `expect_used`, `panic`, `todo`,
  `unimplemented`, `dbg_macro`).
- **Python:** `ruff format` defaults, `pyright` strict on `python/vernier/`.
- **Commits:** conventional commits style is encouraged but not enforced.

## Documentation

The docs site is built with mkdocs-material and gated in CI (ADR-0027).

- **Every PR that changes a public Python symbol updates the docstring in the same commit.** Auto-generated reference (`mkdocstrings`) keeps the rendered API reference true to code; the reviewer's job is to bounce PRs that skip this step.
- Run `just build-docs` and `just lint-docs` before opening a PR that touches `docs/` or `python/vernier/` — those mirror the CI gates.
- Run `just serve-docs` for live-reload local preview at `http://127.0.0.1:8000`.
- Migration guides live in `docs/migrate/`; Diátaxis quadrant pages in `tutorials/`, `how-to/`, `reference/`, `explanation/`.

## License

By contributing, you agree that your contributions will be dual-licensed
under MIT and Apache 2.0.
