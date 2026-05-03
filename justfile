# vernier task runner. Run `just` to list all recipes.
# Requires: just, uv, cargo, cargo-nextest, cargo-deny.

set shell := ["bash", "-cu"]

# Default recipe: list available recipes.
default:
    @just --list

# ---------------------------------------------------------------------------
# Bootstrap & build
# ---------------------------------------------------------------------------

# Set up the dev environment from a fresh clone.
bootstrap:
    uv sync --all-extras
    uv run maturin develop --release

# Fast iteration build (debug Rust, slow Python).
develop:
    uv run maturin develop

# Optimized wheel into target/wheels/.
build:
    uv run maturin build --release

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# Run all tests (Rust + Python).
test: test-rust test-py

test-rust:
    uv run cargo nextest run --workspace

test-py:
    uv run pytest

# Run only parity tests (against pycocotools reference).
test-parity:
    uv run pytest -m parity

# Run LVIS parity tests against the vendored lvis-api reference oracle
# (ADR-0026). The oracle lives at tests/python/parity_lvis/oracle/lvis_api/;
# the parity harness is added in subsequent PRs of the LVIS rollout.
test-parity-lvis:
    uv run pytest -m parity_lvis

# Run the LVIS v1 val whole-dataset parity smoke. Requires
# VERNIER_LVIS_GT_PATH and VERNIER_LVIS_DT_PATH to point at the GT
# annotations and a detector predictions JSON. See
# `python -m lvis_val_cache` for the canonical setup.
test-parity-lvis-val:
    uv run pytest -m parity_lvis_val -v

# Run panoptic parity tests against the vendored panopticapi reference
# oracle (ADR-0025). The oracle lives at
# tests/python/parity_panoptic/oracle/panopticapi/.
test-parity-panoptic:
    uv run pytest -m parity_panoptic

# Run the COCO panoptic val2017 whole-dataset parity smoke. Requires
# VERNIER_PANOPTIC_GT_PATH, VERNIER_PANOPTIC_GT_PNG_DIR,
# VERNIER_PANOPTIC_DT_PATH, and VERNIER_PANOPTIC_DT_PNG_DIR. See
# `python -m panoptic_val_cache` for the canonical setup.
test-parity-panoptic-val:
    uv run pytest -m parity_panoptic_val -v

# Run the COCO val2017 whole-dataset parity smoke.
# Requires VERNIER_COCO_GT_PATH and VERNIER_COCO_DT_PATH to point at
# the GT annotations and a detector predictions JSON. See
# docs/engineering/coco-val-parity.md (and tools/fetch-coco-val.sh)
# for the canonical setup.
test-coco-val:
    uv run pytest -m coco_val -v

# ---------------------------------------------------------------------------
# Lint & format
# ---------------------------------------------------------------------------

# Run all linters (CI-equivalent, read-only).
lint: lint-rust lint-py

lint-rust:
    cargo fmt --all -- --check
    cargo clippy --workspace --all-targets -- -D warnings

lint-py:
    uv run ruff check .
    uv run ruff format --check .
    uv run pyright

# Auto-format Rust and Python (writes changes).
fmt:
    cargo fmt --all
    uv run ruff format .
    uv run ruff check --fix .

# ---------------------------------------------------------------------------
# Docs
# ---------------------------------------------------------------------------

# Build the docs site into site/ (strict — mirrors CI check).
build-docs:
    uv run --group docs mkdocs build --strict

# Serve docs with live-reload at http://127.0.0.1:8000.
serve-docs:
    uv run --group docs mkdocs serve

# Spell-check user-facing docs (excludes adr/ and engineering/ internal dirs).
lint-docs:
    uv run --group docs codespell docs/ --skip "docs/adr,docs/engineering"

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

# Run Rust microbenchmarks (divan). Dev-only — divan is a dev-dep and never
# linked into the production wheel or any non-bench target.
ubench:
    cargo bench --workspace

# Sync the local bench harness env and every per-impl env (ADR-0017).
bench-sync:
    uv sync --directory bench
    for env in bench/envs/*/; do uv sync --directory "$env"; done

# Run the bench harness's own pytest suite. Isolated from `just test-py`.
bench-test:
    uv run --directory bench pytest tests/

# Run the bench harness end-to-end. Forwards all args to `vernier-bench run`.
# Example: just bench-run --impl vernier --workload smoke --iou bbox
bench-run *ARGS:
    uv run --directory bench python -m bench run {{ARGS}}

# ---------------------------------------------------------------------------
# Audit & maintenance
# ---------------------------------------------------------------------------

# Security/license audit of Rust dependencies.
audit:
    cargo deny check

# Remove all build artifacts and the venv.
clean:
    cargo clean
    rm -rf .venv target/wheels
    find python -name "_core*.so" -delete -o -name "_core*.pyd" -delete

# Print toolchain versions (paste into bug reports).
versions:
    @echo "rustc:  $(rustc --version)"
    @echo "cargo:  $(cargo --version)"
    @echo "uv:     $(uv --version)"
    @echo "python: $(uv run python --version)"
