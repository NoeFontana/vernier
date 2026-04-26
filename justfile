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
