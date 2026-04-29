# Registry name reservations

This directory holds the throwaway packages we publish at v0.0.0 to claim
the names `vernier`, `vernier-core`, `vernier-mask` on crates.io and
`vernier` on PyPI. `vernier-cli` lived here until it was promoted to a
real workspace member at `crates/vernier-cli/` per ADR-0015.

These are **not** part of the workspace and are **not** built by `just
build`. Each crate skeleton is its own standalone package (note the empty
`[workspace]` table in each `Cargo.toml`), and the PyPI placeholder is its
own pyproject. They exist purely so we can squat the names defensively
before the real implementations ship.

## Layout

```
tools/reservations/
├── crates/
│   ├── vernier/        # crates.io: vernier (top-level umbrella name)
│   ├── vernier-core/   # crates.io: vernier-core (pure-Rust core)
│   └── vernier-mask/   # crates.io: vernier-mask (RLE/mask helpers)
└── pypi/
    └── vernier/        # PyPI: vernier (the eventual wheel)
```

`vernier-ffi` is **deliberately excluded** — it's an internal-only crate
that ships as the `vernier._core` extension module and will never be
published to crates.io as a standalone package.

## Why placeholders, not the real crates?

For `vernier-core` we already have a real crate at `crates/vernier-core/`,
but publishing it today would commit us to an API surface that doesn't
exist yet. The reservation skeleton publishes a deliberately empty v0.0.0
and lets the real crate evolve in-tree. The project ships under 0.0.x
patches (v0.0.1, v0.0.2, …) until the core and extended feature set is
complete; the move to a stable 0.1.0+ release line is a deliberate
later decision.

## Publishing

Use `tools/reservations/reserve.sh` — it dry-runs by default and only
uploads when invoked with `--publish`. See the script header for the
full command sequence.

After the names are reserved, this directory can either stay (as a record
of the placeholder shape) or be deleted; the on-registry artifacts are
what actually hold the names.
