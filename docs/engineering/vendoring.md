# Vendoring third-party code

This document covers both the **policy** (what we vendor and why) and
the **process** (how to add a new vendored reference). The user-facing
inventory of what is currently vendored lives at the repo root in
[`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md); this file is
the engineering counterpart.

## Policy

vernier vendors **test- and bench-only reference implementations**
when they are load-bearing for parity claims
(the `bowenc0221/boundary-iou-api` precedent — see ADR-0010 — and the
`pycocotools==2.0.11` parity oracle pinned in `pyproject.toml`),
or as comparator implementations for the bench harness
(the `faster-coco-eval` precedent — see ADR-0017), or for non-parity
sanity-check purposes (the future `tidecv` case anticipated in
ADR-0021).

Vendoring takes **two flavors** in this repo, and both count for the
purposes of [`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md)
and the ADR-level discipline below:

- **In-tree source vendoring** — verbatim copies of upstream source
  trees checked into the repo at a pinned commit SHA
  (e.g. `tests/python/parity_boundary/oracle/boundary_iou_api/`).
  The bytes live in our tree; we own the refresh.
- **Pinned-package envs** — third-party packages pinned at exact
  versions in a `pyproject.toml` + `uv.lock` pair, where the pin is
  itself a parity / comparator claim
  (e.g. `bench/envs/pycocotools/`, `bench/envs/faster-coco-eval/`).
  The bytes do not live in our tree; the **pin is the artifact**.

What this means concretely, regardless of flavor:

- Vendored code is **never imported by `python/vernier/` and never
  linked into a `crates/` target**. The published wheel does not
  contain vendored bytes (in-tree flavor) and does not depend on
  vendored packages (pinned-package flavor). `cargo deny`'s license
  check is unaffected because vendored code is Python.
- Vendored references are pinned at a commit SHA (in-tree) or an
  exact version (pinned-package), not at a branch or open range.
  The pin is itself a parity claim — every quirk vernier reproduces
  in strict mode is keyed to the exact bytes the pin selects.
- Vendoring a new reference is an **ADR-level decision** per ADR-0001.
  The ADR motivates the dependency, names the parity contract, and
  fixes the pin. Refreshing a pin is also ADR-level — see "Refresh
  procedure" below.

## Layout convention

### In-tree source vendoring

Path: `tests/python/<harness>/oracle/<package>/`. Required artifacts
alongside the source tree:

- `VENDORING.md` — provenance, license analysis, role declaration,
  inventory, runtime deps, fork plan, refresh procedure. Use the
  template below.
- The original `LICENSE` file preserved verbatim from upstream. Do
  not edit license files.
- The source tree itself, unmodified except for documented exceptions
  recorded in `VENDORING.md`'s "Modifications" field.

The layout mirrors what large Apache projects do for `NOTICE`
artifacts, simplified for a repo that does not redistribute the
vendored code in its build artifacts.

### Pinned-package envs

Path: `bench/envs/<package-name>/` (the bench-harness convention,
ADR-0017) or the root `pyproject.toml` (for parity oracles consumed
by `tests/python/parity/`). Required artifacts:

- `pyproject.toml` with the package pinned at an exact version
  (`pycocotools==2.0.11`) or a tightly-bounded range pinned by
  `uv.lock` (`faster-coco-eval>=1.6` resolves to a single version
  via the lockfile). The pin's rationale lives in a comment at the
  declaration site.
- `uv.lock` capturing the exact resolved version + SHA of the
  upstream wheel. The lockfile is the byte-stable artifact.
- A short top-level entry in
  [`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md) with the
  upstream URL, license, role, and the location of the pin.

Pinned-package envs do not need a `VENDORING.md` — the pin is the
documentation. The notices entry plus the in-comment rationale at
the pin site are sufficient.

## `VENDORING.md` template

Copy this skeleton when adding a new vendored reference. The
boundary-iou-api file at
[`tests/python/parity_boundary/oracle/VENDORING.md`](../../tests/python/parity_boundary/oracle/VENDORING.md)
is the worked example.

```markdown
# Vendored oracle: `<owner>/<repo>`

This directory contains a frozen, verbatim copy of the
[`<owner>/<repo>`](https://github.com/<owner>/<repo>) reference
implementation of <what it does> (paper / spec citation if applicable).

The oracle is consumed only by `tests/python/<harness>/`. It is not
imported by `python/vernier/`, `crates/`, or any code that ships in
the published wheel.

Per **ADR-NNNN §"<section>"** this is a vendored, version-pinned
dependency: <what parity claim the pin underwrites>.

## Provenance

| Field                | Value |
| -------------------- | ----- |
| Upstream repo        | <URL> |
| Upstream commit SHA  | `<full sha>` |
| Upstream commit date | <YYYY-MM-DD> |
| Upstream branch      | <branch at fetch time> |
| Vendored on          | <YYYY-MM-DD> |
| Vendored by          | @<github-handle> |
| Modifications        | **None.** | <!-- or document explicitly -->

The pinned constants live in `crates/vernier-core/src/<area>_parity.rs`
as `ORACLE_COMMIT_SHA` (and any per-kernel pins). A unit test in that
module asserts the SHA recorded here matches the constant — drift
between the two is a build failure.

## License

<License name>. <Copyright line, verbatim from upstream LICENSE.>
<Compatibility statement against vernier's MIT/Apache-2.0 dual license.>

## Inventory — what we vendored, and what we did not

<Vendored tree as a fenced block; Skipped paths as a table with
"why skipped" reasons.>

## Runtime dependency: <name>

<If the oracle pulls in a non-stdlib dep, document the version pin
and why; mirror the pin into the parity-module constant.>

## Fork plan

<What we do if the upstream goes unmaintained, gets a CVE, or breaks.
Decide in advance so it is not a panic decision later.>

## How to refresh

<Steps for moving to a different upstream commit. Refreshing a SHA is
itself an ADR-level operation, not a routine update.>
```

## Process for adding a new vendored reference

The first three steps differ by flavor; the last three are the same.

**In-tree source vendoring**:

1. **Draft an ADR** describing what reference is being vendored, why
   the parity / sanity-check claim needs it, and what SHA to pin.
2. **Vendor at the pinned SHA.** Use `gh api repos/<owner>/<repo>/...`
   or `git archive` to fetch the exact tree at the SHA — do not
   shallow-clone a branch. Preserve the upstream `LICENSE` verbatim.
3. **Write `VENDORING.md` from the template.** The provenance table
   is load-bearing; everything else fills in around it.

**Pinned-package envs**:

1. **Draft an ADR** describing what reference is being vendored, why
   the parity / comparator claim needs it, and what version to pin.
2. **Pin in `pyproject.toml`** (root for parity oracles, or a new
   `bench/envs/<name>/pyproject.toml` for bench comparators).
   In-comment rationale at the pin site is required.
3. **Refresh the lockfile** (`uv lock` in the relevant directory).
   The lockfile is the byte-stable artifact; commit it.

**Common to both** (steps 4–6):

4. **Add an entry to [`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md)**
   following the existing section shape.
5. **Tripwire (in-tree only).** Add a constant in
   `crates/vernier-core/src/<area>_parity.rs` (e.g.
   `ORACLE_COMMIT_SHA`) so any drift between the pin recorded in
   code and the pin recorded in `VENDORING.md` is a build failure.
   Pinned-package envs do not need this — the pin lives in
   `pyproject.toml` directly.
6. **Update tooling excludes** (in-tree only). `ruff` and `pyright`
   should not lint vendored bytes (the upstream's style is the
   upstream's problem). Add the path to the relevant excludes.
   Pinned-package envs are not linted (they live in `.venv/`).
7. **Run `just audit`.** `cargo deny` is unaffected by Python
   vendoring but the workflow is the same gate as any other PR.

## Refresh procedure

Refreshing a vendored pin is itself an **ADR-level operation**, not a
routine dependency bump.

**In-tree source vendoring** — refreshing a pinned SHA:

1. Open a `proposed` ADR titled `Refresh <oracle> to <short-sha>`
   describing what changed in upstream and why we are moving forward.
2. Diff the upstream range and reflect any behavior change in the
   relevant quirks survey (e.g.
   [`docs/engineering/boundary-iou-quirks.md`](boundary-iou-quirks.md))
   or in the ADR.
3. Re-fetch the vendored tree at the new SHA (verbatim, no edits).
4. Update the provenance table in `VENDORING.md` and the tripwire
   constant in `crates/vernier-core/src/<area>_parity.rs` in the same
   commit so the SHA pin and the in-code pin stay synchronized.
5. Re-run the parity harness on the full fixture corpus.
   Differential output is the regression signal.

**Pinned-package envs** — refreshing a version pin:

1. Open a `proposed` ADR titled `Refresh <package> to <version>`
   describing what changed upstream (release notes, behavior diffs)
   and why the new version is acceptable. For oracle pins (e.g.
   `pycocotools`), include the parity-impact analysis on the
   relevant quirks survey
   ([`docs/engineering/pycocotools-quirks.md`](pycocotools-quirks.md)
   for the canonical example).
2. Bump the pin in the relevant `pyproject.toml` and re-run `uv lock`
   in the same commit.
3. Re-run the parity harness or the bench-comparator suite on the
   relevant fixture corpus.

The "no modifications" invariant on in-tree vendored trees is
structural — it is what makes the SHA pin a parity claim rather than
a snapshot. If a fix is needed, it goes upstream-or-fork (see the
boundary-iou-api fork plan), not into the vendored tree. The
analogous invariant on pinned-package envs is that we never patch
the installed wheel; if upstream is broken, we pin a different
version (or fork the package via `[tool.uv.sources]` overrides) and
record the decision in an ADR.
