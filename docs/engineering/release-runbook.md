# Release runbook

How to ship a vernier patch to PyPI and crates.io. The pipeline is in
[`.github/workflows/wheels.yml`](../../.github/workflows/wheels.yml);
this document is the operator's view.

## Topology

A `v*` tag fans out into seven jobs in `wheels.yml`:

```
linux   ─┐
macos   ─┤
windows ─┤  ─► smoke ─┐
sdist   ─┤            ├─► publish-pypi      (env: pypi,      OIDC → PyPI)
         │            └─► publish-crates-io (env: crates-io, OIDC → crates.io)
verify-tag ──────────────► (gates both publish jobs)
```

Both publish jobs are gated by GitHub `environment:` rules — a required
reviewer must approve in the Actions UI before any bytes upload. PyPI
permits yank but never re-upload of the same version; crates.io permits
yank only as a deprecation signal. A mistaken publish costs a patch
bump, so the manual gate is a deliberate speed bump.

## One-time setup

Done once per repo, then the per-release flow takes over.

### 1. PyPI Trusted Publisher — **migrate before v0.0.1**

The `vernier` PyPI project already has a Trusted Publisher entry from
the 0.0.0 reservation, but it's pointed at the now-deleted
`pypi-reserve.yml` workflow with `Environment: none`. The first tag
push will fail OIDC matching unless this entry is updated **before**
pushing `v0.0.1`.

At <https://pypi.org/manage/project/vernier/settings/publishing/>,
edit the existing entry (or delete + re-add) so it reads:

- Owner: `NoeFontana`
- Repository: `vernier`
- Workflow: `wheels.yml`
- Environment: `pypi`

The environment field is what binds the publisher to the
`environment: pypi` gate in the `publish-pypi` job — without it, the
required-reviewer checkpoint is bypassed even if the workflow asks
for it.

### 2. crates.io Trusted Publishers (one entry per crate)

crates.io scopes Trusted Publishers per crate, not per repo. Each of
the three published crates needs its own configuration at
<https://crates.io/me/trusted-publishers>:

| Crate | Repository | Workflow | Environment |
|---|---|---|---|
| `vernier-mask` | `NoeFontana/vernier` | `wheels.yml` | `crates-io` |
| `vernier-core` | `NoeFontana/vernier` | `wheels.yml` | `crates-io` |
| `vernier-cli`  | `NoeFontana/vernier` | `wheels.yml` | `crates-io` |

`vernier-ffi` is `publish = false` (ships only inside the wheel) and
the top-level `vernier` crate name is held by the placeholder under
`tools/reservations/crates/vernier/` at 0.0.0 — neither needs a
publisher entry.

The first publish for a crate name has to be done manually with an API
token (Trusted Publishers can only update an existing crate, not
create a new one). For the v0.0.1 first release, see "First release
bootstrap" below.

### 3. GitHub repo environments

In `Settings → Environments`, create two environments:

- **`pypi`** — required reviewer: repo owner. No deployment branch
  restrictions (the `if: startsWith(github.ref, 'refs/tags/v')` on the
  job already constrains who can trigger it).
- **`crates-io`** — same.

The required-reviewer gate is the one human checkpoint between a
pushed tag and an irreversible publish.

## First release bootstrap (v0.0.1 only)

crates.io Trusted Publishers can only attach to an existing crate, so
the very first publish of `vernier-mask` / `vernier-core` /
`vernier-cli` has to use a one-shot API token. After v0.0.1 lands, the
trusted-publisher entries from "One-time setup §2" take over.

```sh
# 1. From https://crates.io/me, mint a token scoped to publish-new for
#    the three crates. Single-use; revoke after v0.0.1.

# 2. Verify the workspace builds cleanly from a fresh checkout.
git checkout v0.0.1
cargo publish -p vernier-mask --dry-run
cargo publish -p vernier-core --dry-run
cargo publish -p vernier-cli  --dry-run

# 3. Publish in dependency order. Modern cargo blocks until the index
#    propagates, so each line returns when the next is safe to start.
export CARGO_REGISTRY_TOKEN=<one-shot token>
cargo publish -p vernier-mask
cargo publish -p vernier-core
cargo publish -p vernier-cli
unset CARGO_REGISTRY_TOKEN

# 4. At https://crates.io/crates/<name>/settings, add the Trusted
#    Publisher entries from §2. Revoke the one-shot token.
```

PyPI v0.0.0 is already published via the now-deleted
`pypi-reserve.yml`; the v0.0.1 PyPI publish runs through the standard
Trusted Publisher path on tag push, **once §1 above has been done**.
The migration step is the only Python-side bootstrap; tag push handles
the rest.

## Per-release flow

Steady-state once the bootstrap above is done.

```sh
# 1. Open a release PR.
git switch -c chore/release-X.Y.Z
# Bump these five strings to X.Y.Z:
#   - Cargo.toml          [workspace.package].version
#   - pyproject.toml      [project].version
#   - crates/vernier-core/Cargo.toml      vernier-mask path-dep
#   - crates/vernier-cli/Cargo.toml       vernier-core path-dep
#   - crates/vernier-ffi/Cargo.toml       vernier-core path-dep
cargo update -p vernier-core -p vernier-mask -p vernier-cli -p vernier-ffi --workspace
uv lock
# Add a CHANGELOG.md entry under the new version heading.
just lint && just test && just audit
git commit -am "chore(release): bump workspace and wheel to X.Y.Z"
git push -u origin chore/release-X.Y.Z
gh pr create --title "chore(release): X.Y.Z"

# 2. Once CI is green and the PR is merged, tag from main.
git switch main && git pull --ff-only
git tag -s vX.Y.Z -m "Release X.Y.Z"
git push origin vX.Y.Z

# 3. The tag push triggers wheels.yml. Watch the run:
gh run watch --exit-status

# 4. When publish-pypi and publish-crates-io enter the "Waiting" state,
#    review the run summary at https://github.com/NoeFontana/vernier/actions
#    and approve each environment. The verify-tag job has already
#    confirmed the tag matches workspace + pyproject.

# 5. Smoke-verify the published artifacts.
pip install --no-cache-dir vernier==X.Y.Z
cargo install --version X.Y.Z vernier-cli  # or `cargo add vernier-core` in a scratch crate
```

## Rollback / failure modes

### `verify-tag` fails

Tag pushed without bumping `workspace.package.version` /
`project.version`. Delete the tag and re-cut after a proper bump PR:

```sh
git push --delete origin vX.Y.Z
git tag -d vX.Y.Z
```

### Wheels build but smoke fails

A platform-specific FFI / dynamic-linker breakage. The publish jobs
won't run (smoke is a `needs:` for both). Fix on a follow-up PR, bump
to X.Y.(Z+1), re-tag.

### `publish-pypi` fails after `publish-crates-io` succeeds (or vice versa)

The two publish jobs run in parallel; either can succeed
independently. PyPI and crates.io are not transactional with each
other.

- **PyPI publish failed** — yank crates.io (`cargo yank --version
  X.Y.Z vernier-mask` etc.), bump to X.Y.(Z+1), re-tag. The yanked
  crates.io version stays in the registry as a deprecation marker.
- **crates.io publish failed** — yank PyPI at
  <https://pypi.org/manage/project/vernier/release/X.Y.Z/>, bump to
  X.Y.(Z+1), re-tag.
- **Partial crates.io** (e.g. `vernier-mask` published, `vernier-core`
  failed) — yank `vernier-mask@X.Y.Z`, bump, re-tag. Don't try to
  resume from the middle of the dependency chain on the same version.

### Need to remove a release entirely

Both registries treat publishes as immutable. Yank is the only escape
hatch:

- PyPI: <https://pypi.org/manage/project/vernier/release/X.Y.Z/> →
  "Yank release". Existing pins keep working; new installs fail unless
  the version is requested explicitly.
- crates.io: `cargo yank --version X.Y.Z vernier-<name>`. Same
  semantics — pinned consumers unaffected, fresh resolves skip it.

In both cases the version number is then permanently spent. The next
release is X.Y.(Z+1), not a re-cut of X.Y.Z.

## See also

- [`docs/engineering/registry-reservations.md`](registry-reservations.md) — how the placeholder names (`vernier`, `vernier-cli`, `vernier-core`, `vernier-mask`) on crates.io and `vernier` on PyPI were claimed.
- [`tools/reservations/crates/vernier/README.md`](../../tools/reservations/crates/vernier/README.md) — why the top-level `vernier` Rust crate stays at 0.0.0 indefinitely.
- [`CHANGELOG.md`](../../CHANGELOG.md) — release history.
