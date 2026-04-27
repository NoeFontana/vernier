# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project shape

Mixed Rust/Python monorepo. Build is glued by `maturin` + `uv`; tasks via `just`.

- `crates/vernier-core/` — pure Rust evaluation logic. **No Python deps.** Source of truth for semantics. `#![forbid(unsafe_code)]`.
- `crates/vernier-mask/` — pure Rust COCO RLE codec, polygon rasterizer, mask ops (per ADR-0009). Leaf crate: depended on by `vernier-core`, never the reverse. **No Python deps.** `#![forbid(unsafe_code)]`.
- `crates/vernier-ffi/` — PyO3 bindings, compiled as the `vernier._core` extension module. **Data conversion only, no business logic** — if logic creeps in here it belongs in `vernier-core`. Not published as a crate (`publish = false`); ships inside the wheel.
- `python/vernier/` — thin Python wrapper, the user-facing API. `pyright` runs `strict` on this directory.
- `tools/reservations/` — placeholder packages that hold names on crates.io / PyPI. Each has its own `[workspace]` table and is deliberately **outside** the Cargo workspace. Don't add it to workspace members and don't `just build` it. See `docs/engineering/registry-reservations.md`.
- `docs/adr/` — Architecture Decision Records (MADR format). Significant changes start as a `proposed` ADR, not a PR. ADRs are immutable once `accepted`; supersede with a new ADR rather than edit.

## Common commands (use `just`)

```bash
just bootstrap        # one-time: uv sync + maturin develop --release
just develop          # fast iterative rebuild (debug Rust)
just build            # release wheel into target/wheels/
just test             # Rust nextest + Python pytest
just test-rust        # cargo nextest run --workspace
just test-py          # uv run pytest
just test-parity      # only tests marked @pytest.mark.parity
just lint             # clippy + ruff + pyright (CI-equivalent, read-only)
just fmt              # cargo fmt + ruff format + ruff check --fix
just audit            # cargo deny check
just clean            # nuke target/, .venv, built _core*.so
```

Single-test invocations:

```bash
cargo nextest run -p vernier-core <test_name>
uv run pytest tests/python/parity/test_parity.py::test_perfect_match_baseline_ap
uv run pytest -m parity                  # all parity tests
uv run pytest -m "not slow"              # skip slow
```

Prereqs: Rust stable (pinned in `rust-toolchain.toml`), `uv`, `just`, `cargo-nextest`, `cargo-deny`. `rustc >= 1.83`, Python >= 3.10.

## Parity contract — read before changing eval logic

`pycocotools==2.0.11` is pinned **exactly** in `pyproject.toml` and is the reference oracle. Bumping it is an ADR-level decision, not a routine dep update — every quirk vernier reproduces is keyed to this version.

Each pycocotools quirk gets one of three dispositions, surveyed in `docs/engineering/pycocotools-quirks.md`:

- **strict** — reproduce bit-exactly. Default.
- **aligned** — match semantics; outputs match within a documented tolerance.
- **corrected** — opt-in opinionated fix; default behavior diverges and the divergence is documented.

The parity harness (`tests/python/parity/harness.py`) double-runs reference and candidate and diffs every intermediate. Today the candidate (`_run_vernier`) delegates to pycocotools — the suite is a tautology until the Rust evaluator lands. When that changes, `_run_vernier` is the single function to swap.

## Code style — non-obvious bits

- **Workspace clippy** denies: `unwrap_used`, `expect_used`, `panic`, `todo`, `unimplemented`, `dbg_macro`. `print_stdout`/`print_stderr` warn. Workspace rust lints deny `unused_must_use` and warn `missing_docs`, `unreachable_pub`. Don't introduce these in new code.
- `vernier-core` `#![forbid(unsafe_code)]`. `vernier-ffi` may have carefully-audited unsafe (e.g., DLPack interop); each crate states its policy in its `lib.rs`.
- Stable Rust only. Workspace dependencies (ndarray, pyo3, numpy, pyo3-stub-gen, thiserror) are pinned by minor version in the workspace table — bump there, not in individual crates. PyO3 features are pinned to `abi3-py310` for stable wheel ABI.
- Python: `ruff` lints include PYI (stub-file linting). Stub edits sometimes require stripping docstrings / pass-bodies (PYI021/PYI048).

## Non-trivial workflows

- **Adding a quirk-driven test:** put a fixture in `tests/python/parity/fixtures/<name>/{gt,dt}.json`, add the name to `ALL_FIXTURES` in `tests/python/parity/test_parity.py`. The disposition table in `docs/engineering/pycocotools-quirks.md` is the canonical index — cite quirks by ID (e.g., "B1", "D6").
- **Reserving a new registry name:** edit `tools/reservations/`, then `./tools/reservations/reserve.sh --publish` (crates.io) or trigger `pypi-reserve.yml` (PyPI). Don't expect this to interact with the workspace.
- **Committing:** conventional commits encouraged. `just lint && just test && just audit` mirrors CI.
