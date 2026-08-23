# Registry artifacts

Working note recording what vernier publishes, where, and under which
credentials. Audience: maintainers (i.e. me) when an artifact needs to be
published, re-pointed, yanked, or transferred.

> **Renamed in substance, not in path.** Until ADR-0048 this file was a
> *reservation* register — it tracked deliberately-empty placeholder
> packages that existed only to hold names. That practice is retired; the
> placeholders under `tools/reservations/` are deleted and this file now
> tracks published artifacts. The path is unchanged because ADR-0009,
> ADR-0015, ADR-0025, and ADR-0028 link to it and accepted ADRs are
> immutable.

## The rule

> **A crate name is claimed by its first real release, and never before.**
> If a design anticipates a crate, the name is recorded in the ADR that
> anticipates it. Nothing is published to hold it.

This replaces the placeholder practice wholesale (ADR-0048). The practice
was reported to crates.io on 2026-08-10 under the anti-squatting clause of
the registry policy — correctly: `vernier` sat at an empty v0.0.0 whose
README stated it contained no code, pointing at a repository
(`vernier-rs/vernier`) that never existed. Six sibling crates were shipping
real releases at the time; the contrast is what made the empty one look
like a squat. See ADR-0048 for the full reasoning.

Anticipated-but-undesigned crates are **not** pre-reserved. That applies
today to `vernier-assign` (the optimal-assignment work in design) and to
any future 3D-evaluation crate: they publish at their first real release,
following the `vernier-cli` lifecycle precedent in ADR-0015
§"Workspace integration", minus the reservation step. The residual risk —
a third party taking a prefixed name in the interim — is accepted: the
impact is a rename during design, which is cheap precisely because the
crate is unpublished.

## What is published

### crates.io — seven crates, lockstep-versioned

Every crate inherits `workspace.package.version`; a release publishes all
seven at the same version, in dependency order.

| Crate | Role | ADR |
|---|---|---|
| `vernier` | Facade — re-exports the five library crates under one dependency; no code of its own | ADR-0048 |
| `vernier-core` | Instance-paradigm evaluation (AP fold), streaming, distributed partials | — |
| `vernier-mask` | COCO RLE codec, polygon rasterizer, mask ops | ADR-0009 |
| `vernier-panoptic` | Panoptic quality (PQ / SQ / RQ) | ADR-0025 |
| `vernier-semantic` | Semantic segmentation (mIoU / FWIoU) | ADR-0028 |
| `vernier-partial` | Distributed-eval wire envelope shared by the three paradigms | ADR-0031, ADR-0032 |
| `vernier-cli` | The `vernier` binary | ADR-0015 |

Publish order (topological sort of the internal dep graph; the facade is
always last because it depends on everything):

```
vernier-mask → vernier-partial → vernier-core → {vernier-panoptic, vernier-semantic}
             → vernier-cli → vernier
```

`wheels.yml`'s `publish-crates-io` job encodes this order. See
[`release-runbook.md`](release-runbook.md).

### PyPI — one project

| Project | Contents |
|---|---|
| `vernier` | The wheel: `python/vernier/` plus the `vernier._core` extension module built from `vernier-ffi` |

### What is *not* published

- **`vernier-ffi`** — `publish = false`. It is the PyO3 binding layer and
  ships only inside the wheel as `vernier._core`; it will never be a
  top-level crates.io publication.
- **`bench/`** — the local bench harness (ADR-0017, ADR-0033) is a
  development tool, not a distributed artifact.

## Historical artifacts

| Registry | Name | Version | Disposition |
|---|---|---|---|
| crates.io | `vernier` | `0.0.0` | Empty placeholder. **Yank once `vernier@0.2.0` is live** — never before (see below). |
| crates.io | `vernier-core` / `vernier-mask` / `vernier-cli` | `0.0.0` | Empty placeholders, superseded by real releases from v0.0.1 onward. |
| PyPI | `vernier` | `0.0.0` | Empty placeholder, superseded by the v0.0.1 wheel. |

**Yank order for `vernier@0.0.0` is load-bearing.** Yank it *after* the
first real `vernier` version is on crates.io, not before. That first
version is `0.2.0`, published standalone rather than through a release
tag — the six leaf crates were already live at 0.2.0, so the facade
resolves entirely against the registry. See
[`release-runbook.md`](release-runbook.md) §"One-off: publishing the
facade ahead of a release". Yanking is not
deletion — the artifact and its stale `vernier-rs/vernier` repository URL
are permanent — it only removes the version from dependency resolution.
Yanking first would leave the name held by a yanked empty crate, which is
a worse posture than the one being fixed.

The stale `repository = https://github.com/vernier-rs/vernier` URL on the
0.0.0 artifacts requires no further action; crates.io does not permit
editing or re-publishing a version. It is worth understanding as the
proximate cause of the 2026-08-10 report rather than as a cosmetic
footnote: a reviewer who followed it found nothing, and correctly
concluded there was no development activity to weigh.

## Auth

### crates.io

Trusted Publisher (OIDC) per crate, wired through `wheels.yml` on `v*` tag
push. crates.io scopes publishers per crate, not per repo, so each of the
seven names needs its own entry at
<https://crates.io/me/trusted-publishers>. The per-crate table and the
one-time setup flow live in [`release-runbook.md`](release-runbook.md)
§"One-time setup".

No `CARGO_REGISTRY_TOKEN` is stored in GitHub secrets. A one-shot API
token is needed only for a crate's *first* publish, because Trusted
Publishers can attach to an existing crate but cannot create one. That
does not apply to `vernier`: the name is already owned (v0.0.0), so its
trusted-publisher entry is a `publish-update` scope, not `publish-new`.

### PyPI

Trusted Publisher (OIDC) configured at
<https://pypi.org/manage/account/publishing/>:

- PyPI project: `vernier`
- Owner: `NoeFontana`
- Repository: `vernier`
- Workflow: `wheels.yml`
- Environment: *(blank — see [`release-runbook.md`](release-runbook.md)
  §"One-time setup" for why, and for the flip to `pypi` when this repo
  goes public)*

No API token is stored anywhere. The workflow's `id-token: write`
permission lets GitHub mint a short-lived OIDC token that PyPI matches
against the configured trusted publisher.

## Yank / transfer policy

If a name needs to move (ownership change, hijacked account):

- **crates.io** does not support project transfers. The accepted path is
  to add the new owner via `cargo owner --add <user>`, leaving the old
  owner in place (or removing them once the new owner can publish).
- **PyPI** supports collaborator additions and (with admin help) account
  ownership transfers; see PEP 541.

A yank only hides the version from `cargo` / `pip` resolution defaults —
the name stays owned. Both registries treat publishes as immutable; a
yanked version number is permanently spent.

## See also

- [`../adr/0048-vernier-facade-crate.md`](../adr/0048-vernier-facade-crate.md)
  — the facade decision and the retirement of the reservation practice.
- [`release-runbook.md`](release-runbook.md) — the operator flow for a
  release, including per-crate Trusted Publisher setup.
