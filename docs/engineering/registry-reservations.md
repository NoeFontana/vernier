# Registry name reservations

Working note recording the registry-name posture for vernier. Audience:
maintainers (i.e. me) when the names need to be re-published, transferred,
or yanked.

## What is reserved

| Registry  | Name            | Owner        | First version | Status                                                |
|-----------|-----------------|--------------|---------------|-------------------------------------------------------|
| crates.io | `vernier`       | `NoeFontana` | `0.0.0`       | placeholder                                           |
| crates.io | `vernier-core`  | `NoeFontana` | `0.0.0`       | placeholder                                           |
| crates.io | `vernier-mask`  | `NoeFontana` | `0.0.0`       | placeholder                                           |
| crates.io | `vernier-cli`   | `NoeFontana` | `0.0.0`       | promoted to workspace member per ADR-0015             |
| PyPI      | `vernier`       | `NoeFontana` | `0.0.0`       | placeholder                                           |

Each v0.0.0 artifact is a deliberately empty placeholder. The skeletons
live under `tools/reservations/` in this repo. They are **not** part of
the Cargo workspace (each has its own `[workspace]` table) and they are
**not** built by `just build` — they exist solely so the names exist
on-registry and nobody else can squat them.

## What is *not* reserved

- **`vernier-ffi`** — the PyO3 bindings crate. It ships as the
  `vernier._core` extension module inside the Python wheel and will never
  be a top-level crates.io publication.
- Speculative subcrate names (`vernier-keypoints`, `vernier-bench`, …) —
  reserve them lazily if and when the split actually happens. Squatting
  by speculation is how we end up with fifteen empty crates polluting the
  registry.

## Why placeholders, not the real crates

The real `vernier-core` already exists at `crates/vernier-core/` in the
workspace. Publishing it today would commit us to whatever
half-implemented public API happens to be there, and crates.io versions
are immutable. The placeholder skeleton publishes a deliberately empty
v0.0.0 and lets the real crate evolve in-tree under 0.0.x patches —
the project ships v0.0.1, v0.0.2, … until the core and extended feature
set is complete. Moving to a stable 0.1.0+ release line is a deliberate
later decision.

The placeholder for the umbrella `vernier` crate is similarly throwaway.
Eventually that name will host the user-facing crate; until we know what
shape that takes, the registry just needs a marker.

## Cosmetic gotcha: the v0.0.0 metadata

The first batch of crates was published before this document existed,
with `repository = https://github.com/vernier-rs/vernier` (an aspirational
org) baked in. crates.io does **not** allow editing or re-publishing a
version, only yanking — so v0.0.0 will keep that stale URL forever.

Action: nothing. The in-tree skeletons have been corrected to
`https://github.com/NoeFontana/vernier`, so 0.0.x patches (v0.0.1,
v0.0.2, …) will be right. Nobody reads the metadata of a v0.0.0
placeholder.

## Auth

### crates.io
API token, scoped to `publish-new` and `publish-update` for the four
reserved crate names. Either:

- `cargo login <token>` — persists to `~/.cargo/credentials.toml`.
- `CARGO_REGISTRY_TOKEN=<token> ./tools/reservations/reserve.sh --publish`
  — ephemeral, scoped to the one invocation.

There is **no automated crates.io publish**. crates.io versions are
irreversible; gating publishes behind a manual local command keeps the
mistake-blast-radius small.

### PyPI
Trusted Publisher (OIDC) configured at
<https://pypi.org/manage/account/publishing/>:

- PyPI project: `vernier`
- Owner: `NoeFontana`
- Repository: `vernier`
- Workflow: `pypi-reserve.yml` (for the placeholder) /
  `wheels.yml` (for the eventual real wheel — currently the OIDC publish
  step is stubbed out at the bottom of that file)
- Environment: none

No API token is stored anywhere. The GHA workflow's `id-token: write`
permission lets GitHub mint a short-lived OIDC token that PyPI matches
against the configured trusted publisher.

## Publishing the placeholders

```bash
# dry-run everything (default; safe)
./tools/reservations/reserve.sh

# publish all four crates to crates.io
cargo login <token>
./tools/reservations/reserve.sh --publish

# publish only one crate (useful for first-time validation)
./tools/reservations/reserve.sh --publish --only vernier-mask
```

For PyPI, push the workflow to `main`, then trigger
`PyPI Reserve` from the GitHub Actions UI (or `gh workflow run
pypi-reserve.yml`).

## Publishing the real crates

When the real implementation lands, the reservation skeletons can be
deleted from the tree (the on-registry artifacts are what hold the names).

The publish path then becomes:

- **crates.io**: `cargo publish` from each real crate (`crates/vernier-core/`,
  the eventual `crates/vernier/` umbrella, etc.). The first
  non-placeholder version must be `>= 0.0.1` since v0.0.0 is taken;
  the project ships under 0.0.x patches until the core and extended
  feature set is complete.
- **PyPI**: re-point the trusted publisher at `wheels.yml`, uncomment the
  release block at the bottom of that workflow, and tag the corresponding
  `v0.0.x` release. The maturin pipeline builds linux/macos/windows
  wheels and uploads them via OIDC.

The first move to `0.1.0+` is a deliberate, separate decision (likely
warrants its own ADR for stability commitments) — not a routine publish.

## Yank / transfer policy

If a name needs to move (org transfer, ownership change, hijacked
account):

- **crates.io** does not support project transfers. The accepted path is
  to add the new owner via `cargo owner --add <user>`, leaving the old
  owner in place (or removing them once the new owner can publish).
- **PyPI** supports collaborator additions and (with admin help) account
  ownership transfers; see PEP 541.

A yank only hides the version from `cargo`/`pip` resolution defaults —
the name stays reserved. That's exactly what we want if a placeholder
needs to retire without freeing up the squat.
