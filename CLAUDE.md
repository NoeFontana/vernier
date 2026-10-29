# CLAUDE.md

Project guidance for Claude Code. Domain memory lives at `~/.claude/projects/-home-dev-src-vernier/memory/`.

## Project shape

Mixed Rust/Python monorepo. Build via `maturin` + `uv`; tasks via `just`.

- `crates/vernier/` — facade crate (ADR-0048). `src/lib.rs` only: whole-crate `pub use` aliases for the five library crates, plus rustdoc. **No logic, ever** — if a module appears here, the firewall has been breached. `#[cfg(feature = ...)]` may gate a `pub use` line and nothing else.
- `crates/vernier-core/` — pure Rust eval logic. No Python deps. `#![forbid(unsafe_code)]`. Source of truth for semantics.
- `crates/vernier-mask/` — COCO RLE codec, polygon rasterizer, mask ops (ADR-0009). Leaf crate (no reverse dep on core). No Python deps. `#![forbid(unsafe_code)]`.
- `crates/vernier-ffi/` — PyO3 bindings → `vernier._core`. Data conversion only; logic belongs in core. `publish = false`.
- `python/vernier/` — user-facing Python wrapper. `pyright` strict.
- `docs/adr/` — MADR-format ADRs. Significant changes start as a `proposed` ADR, not a PR. Immutable once `accepted`; supersede with a new ADR.
- `docs/changelog/` — frozen per-release history, one file per version. Do not search, sweep, or edit it. `CHANGELOG.md` keeps `[Unreleased]` plus the most recently released version; older entries move here.

## Commands (use `just`)

- `just setup-host` — once per machine: sccache + clang/mold in `~/.cargo/config.toml`
- `just bootstrap` — first thing in a fresh clone **or new worktree** (~40 s with sccache): `uv sync --all-extras` + pin pyo3's interpreter in a git-ignored `.cargo/config.toml`
- `just data-pull` / `data-push` / `data-status` — generated bench predictions via the private HF dataset pinned in `tools/dev/bench-data.json`. Pull before re-running inference; after a populate, push and commit the manifest
- `just develop` — rebuild the extension if Rust changed; rarely needed, since `uv run` does it itself (`pyext` profile: optimized, no LTO)
- `just build` — release wheel into `target/wheels/`
- `just check` — ultra-fast static analysis (workspace compile check + ruff + pyright, ~1s)
- `just test-changed` — only the suites the branch's diff can affect (prints why); the default inner loop
- `just test` / `test-rust` / `test-py` / `test-parity` — CI's set or a subset (pytest deselects `slow` by default via `addopts`)
- `just test-fast` — all Rust tests + Python minus parity
- `just test-slow` — the `slow` tier (see the recipe before running it)
- Test workers default to 3 per run (parallel agents share the machine); override `PYTEST_XDIST_AUTO_NUM_WORKERS` / `NEXTEST_TEST_THREADS` in a git-ignored `.env`
- `just test-crate <name>` — test a specific workspace crate (e.g. `core`, `cli`, `mask`)
- `just lint` — clippy + ruff + pyright + powerset + wasm (CI-equivalent, read-only)
- `just lint-fast` — fast local lint loop (clippy + ruff + pyright)
- `just fmt` — cargo fmt + ruff format + ruff check --fix
- `just audit` — `cargo deny check`
- `just check-features` — `cargo hack --feature-powerset check -p vernier` (ADR-0048 facade gate; runs as part of `lint-rust`)
- `just prune` — prune compiled target artifacts without destroying `.venv`
- `just gc` — drop stale incremental caches in every worktree's `target/`
- `just clean` — nuke `target/`, `.venv`, built `_core*.so`

Single tests (bare `cargo` and `just` build identically, so mixing them costs no rebuild):

```bash
cargo nextest run -p vernier-core <test_name>
uv run pytest tests/python/parity/test_parity.py::test_perfect_match_baseline_ap
uv run pytest -m parity                  # all parity tests
uv run pytest -m ""                      # include slow (pytest skips it by default)
```

Prereqs: Rust stable (pinned in `rust-toolchain.toml`), `uv`, `just`, `cargo-nextest`, `cargo-deny`, `cargo-hack`. `rustc >= 1.83`, Python >= 3.10. Optional, used by `just setup-host`: `sccache`, `clang` + `mold`.

## Parity contract — read before changing eval logic

- `pycocotools==2.0.11` pinned **exactly** in `pyproject.toml` — the reference oracle. Bumping is ADR-level.
- Each quirk gets one disposition in `docs/engineering/pycocotools-quirks.md`: **strict** (bit-exact, default) / **corrected** (opt-in fix; default diverges). Two tiers, not three — ADR-0002's 2026-05-10 amendment folded the old `aligned` tier into `strict` (every row tagged `aligned` was already asserted bit-equal). Do not reintroduce `aligned` in new guidance; reviving it needs its own ADR.
- Parity harness: `tests/python/parity/harness.py`. `_run_vernier` is the candidate side and calls the real evaluator; `_run_pycocotools` is the oracle. `_run_vernier` is the single swap point.

## Code style — non-obvious bits

- Workspace clippy denies `unwrap_used`, `expect_used`, `panic`, `todo`, `unimplemented`, `dbg_macro`. `print_stdout` / `print_stderr` warn. Workspace rust lints deny `unused_must_use`; warn `missing_docs`, `unreachable_pub`.
- `vernier-core` `#![forbid(unsafe_code)]`. `vernier-ffi` may carry audited unsafe (DLPack); each crate states policy in `lib.rs`.
- Stable Rust only. Workspace deps pinned by minor version in the workspace table — bump there, not per-crate. PyO3 features pinned to `abi3-py310`.
- Python `ruff` includes PYI; stub edits sometimes need stripping docstrings / pass-bodies (PYI021/PYI048).
- `python/vernier/_core.pyi` is hand-written, never generated (no `pyo3-stub-gen`) — it encodes `Literal`/`TypedDict`/`NDArray` precision the Rust signatures don't carry. Edit it in the same commit as the FFI change; `tests/python/test_core_stub_conformance.py` enforces the shape. See `docs/engineering/python-type-stubs.md`.

## Non-trivial workflows

- **Quirk-driven test:** fixture in `tests/python/parity/fixtures/<name>/{gt,dt}.json`; add name to `ALL_FIXTURES` in `tests/python/parity/test_parity.py`. Cite quirks by ID (e.g. "B1", "D6") per the disposition table.
- **Claiming a registry name:** don't, ahead of time. Per ADR-0048, *a crate name is claimed by its first real release, and never before* — an anticipated crate is recorded in the ADR that anticipates it, and nothing is published to hold it. What is published, and under which credentials, is in `docs/engineering/registry-reservations.md`.
- **Committing:** conventional commits encouraged. `just lint && just test && just audit` mirrors CI.
