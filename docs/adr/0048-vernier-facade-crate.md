# ADR-0048: Promote `vernier` to a facade crate and retire speculative name reservations

- **Status:** accepted (implementation errata on the first-publish version — see §"Versioning and publish order")
- **Date:** 2026-08-22
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** all contributors
- **Supersedes:** `docs/engineering/registry-reservations.md` §"Why placeholders,
  not the real crates" and the `vernier` row of §"What is reserved"
- **Related:** ADR-0015 (CLI crate placement), ADR-0029 (per-paradigm namespace),
  ADR-0009 / ADR-0025 / ADR-0028 (sibling-crate firewall)

## Context and problem statement

On 2026-08-10 the crates.io support team notified us that the `vernier`
crate was reported under the anti-squatting clause of the registry
policy, which disallows a crate that exists only to reserve a name
without genuine functionality, purpose, or significant development
activity on the corresponding repository. We must respond by
**2026-08-24** or the crate is removed. The notice explicitly states
that the rest of the account's crates are unaffected.

That last sentence is accurate, and it is the shape of the problem.
Six crates — `vernier-core`, `vernier-mask`, `vernier-panoptic`,
`vernier-semantic`, `vernier-partial`, `vernier-cli` — carry real
releases through the 0.0.1 → 0.0.4 line and out to 0.1.0 (2026-05-19).
Every name reserved under `tools/reservations/` has been redeemed by a
real implementation **except one**: the top-level `vernier` slot, still
sitting at the v0.0.0 skeleton whose README states, in as many words,
that it contains no code and exposes no public API.

Two aggravating details make it the obvious report target:

1. **The v0.0.0 metadata points at a repository that does not exist.**
   The first reservation batch shipped with
   `repository = "https://github.com/vernier-rs/vernier"`, an
   aspirational org we never created. crates.io versions are immutable,
   so that URL is permanent on v0.0.0. Anyone triaging the report
   followed it, found nothing, and correctly concluded there was no
   development activity to weigh. The policy's third branch — activity
   on the corresponding repository — was true in fact and unreachable
   in evidence.
2. **Nothing else on the account looks like this.** A reviewer scanning
   the account sees six actively-released crates and one empty
   eighteen-month-old husk. That contrast reads as leftover squatting
   rather than as a staging artifact, because from the outside the two
   are indistinguishable.

So the defect is not that we squatted. It is that we left a marker
behind after the thing it marked had shipped, and the marker carries
no evidence of what it belongs to. The registry is enforcing a rule we
had already written for ourselves —
`docs/engineering/registry-reservations.md` says, of speculative names,
that squatting by speculation is how we end up with fifteen empty
crates polluting the registry. We applied that discipline to
`vernier-keypoints` and `vernier-bench` and failed to apply it to our
own top-level name.

Two questions are bundled here and this ADR answers both, because
answering only the first leaves the practice that produced the problem
intact:

- **What does the `vernier` crate become?** The name has been held
  since the beginning "for the eventual user-facing crate", with the
  shape deliberately undecided. It is now decided or relinquished;
  there is no third state.
- **Does the reservation practice survive?** `tools/reservations/`
  exists as tooling, is documented as project policy, and would produce
  the same defect again the next time we anticipate a crate — including
  for the optimal-assignment work now in design.

This ADR triggers ADR-0001 §"Affect the public API (Python, Rust, or
CLI)" (a new published Rust surface), §"Set a project-wide convention"
(the registry posture), and §"Add or remove a top-level dependency, a
build target, or a supported platform" (a seventh publishable crate in
the release matrix).

## Decision drivers

- **The deadline is external and non-negotiable.** 2026-08-24. Every
  option below is scored partly on what it lets us truthfully say in a
  reply sent this week, and the honest answer for any option that
  requires shipping code is that a merged ADR is the evidence, not the
  artifact — see §"Versioning and publish order" for why the facade
  physically cannot ship ahead of a full workspace release.
- **Name symmetry across the three surfaces.** The wheel is `vernier`
  on PyPI. The binary is `vernier` per ADR-0015. A third party owning
  `vernier` on crates.io would give this project two of its three
  user-facing names and hand a stranger the one that Rust users type.
  For a project whose adoption thesis is migration from
  pycocotools / faster-coco-eval / panopticapi / mmsegmentation, that
  is a permanent confusion surface on the entry path.
- **ADR-0015 Axis 1 is still correct.** The CLI does not move into a
  library crate. `clap` in the dependency tree of every downstream
  library consumer was rejected there and nothing has changed; the
  facade inherits that rejection rather than relitigating it.
- **ADR-0029's namespace shape should not be re-argued in Rust.** The
  Python surface is `vernier.{instance, panoptic, semantic}` because
  the paradigms are structurally separate. The Rust surface answers to
  the same constraint and should reach the same answer, so that a user
  moving between the two carries one mental model.
- **A facade must contain no logic.** CLAUDE.md holds `vernier-ffi` and
  `vernier-cli` to "no business logic in the binding / binary." A
  facade that curates, renames, wraps, or adapts is a fourth place
  where semantics can live, and it would be the least-tested one. The
  same firewall reasoning that produced ADR-0009 / ADR-0025 / ADR-0028
  applies: the facade is a directory, not a layer.
- **"One wheel, one behavior" constrains features, and does not forbid
  them.** That principle (ADR-0047, and the `parallel`-feature
  rejection before it) is about *numerical behavior* varying by build
  configuration, which destroys the parity contract. A Cargo feature
  that gates a `pub use` line changes what is *nameable*, not what is
  *computed*. The distinction is load-bearing and is pinned as an
  invariant below rather than left to judgement.
- **The sibling crates are a real escape hatch.** Unlike the wheel —
  where the user takes what we ship — a Rust consumer who wants only
  bbox AP can depend on `vernier-core` directly and always could. The
  facade is convenience for the common case, never a gate, which lowers
  the stakes on every ergonomic choice in it.
- **Registry hygiene is now enforced, not merely tasteful.** Any future
  design that anticipates a crate (`vernier-assign` for the optimal
  assignment work; a 3D evaluation crate) must not repeat this.

## Considered options

### Axis A — what the `vernier` name becomes

1. **Facade library crate.** A real workspace member at
   `crates/vernier/` that re-exports the published paradigm crates
   under one dependency, with no code of its own.
2. **Rename `vernier-cli` to `vernier`.** The binary's package takes
   the top-level name; `cargo install vernier` works; the library entry
   point stays `vernier-core`.
3. **Relinquish.** Let the crate be removed. `vernier-core` is the Rust
   entry point; the name goes back to the pool.
4. **Republish a fuller placeholder and argue repository activity.**
   Fix the metadata, write a real README pointing at the live repo,
   publish v0.0.1, and rely on the policy's development-activity branch.

### Axis B — feature posture of the facade

1. **Unconditional.** All five paradigm crates re-exported always.
2. **Additive, re-export-only features, all enabled by default.**
   `instance` and `mask` unconditional; `panoptic`, `semantic`,
   `partial` optional and default-on.
3. **Docs-only facade.** No dependencies; the crate is rustdoc that
   tells the reader which sibling crate to depend on.

## Decision outcome

Chosen: **A1 + B2** — a real facade crate at `crates/vernier/`,
lockstep-versioned with the workspace, with additive per-paradigm
features that gate re-export and nothing else.

A2 is rejected because it converts a naming problem into an API
problem: renaming a published package is a hard break for
`cargo install vernier-cli` and for `cargo binstall` asset mapping
(ADR-0034), and it leaves the Rust *library* entry point at
`vernier-core` forever, which is the ergonomic gap we are trying to
close. A3 trades a permanent, unrecoverable loss for a week of work.
A4 is the option this ADR exists to reject: it is the same placeholder
with better paperwork, it would leave `tools/reservations/` alive, and
it invites a second report from the next person who reads the README.

B3 is rejected for the same reason as A4 — a crate whose entire content
is a pointer to other crates is the empty crate the policy is about,
however well written.

### Crate layout

```
crates/vernier/
├── Cargo.toml       # package, deps, features
└── src/
    └── lib.rs       # re-exports and crate-level rustdoc. Nothing else.
```

There is no `src/` beyond `lib.rs` and there never will be. If a
module appears in this crate, the firewall has been breached and the
code belongs in a paradigm crate — the same rule CLAUDE.md applies to
`vernier-ffi` and `vernier-cli`, stated here so it is not inferred.

`vernier-ffi` is **not** re-exported. It is the PyO3 binding shipped as
the `vernier._core` extension module and is not published to crates.io;
that exclusion is unchanged from `docs/engineering/registry-reservations.md`
§"What is *not* reserved". `vernier-cli` is **not** re-exported: it is
a binary, and depending on it from a library would drag `clap` into
every consumer, which is exactly ADR-0015 Axis 1 Option 2.

### Namespace shape

```rust
pub use vernier_core as instance;
pub use vernier_mask as mask;

#[cfg(feature = "panoptic")]
pub use vernier_panoptic as panoptic;
#[cfg(feature = "semantic")]
pub use vernier_semantic as semantic;
#[cfg(feature = "partial")]
pub use vernier_partial as partial;

/// Facade version. Lockstep with every re-exported crate.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
```

Two properties are worth naming explicitly.

**Module-per-paradigm is forced, not chosen.** `vernier_core`,
`vernier_panoptic`, and `vernier_semantic` each export `ParityMode` and
each export `VERSION`. A flat glob re-export does not compile without
shadowing, and any resolution of the collision by renaming would be
editorial opinion — logic — in a crate that is not allowed to have any.
The type system arrives at the shape ADR-0029 chose for Python on
independent grounds, which is a pleasant confirmation rather than a
coincidence: the names collide because the paradigms are genuinely
distinct, which is the same fact ADR-0029 was reasoning about.

**Whole-crate aliasing, not curated re-export.** `pub use vernier_core
as instance;` makes the facade's public API *identically* the union of
the leaf crates' public APIs, by construction. It cannot drift, it needs
no maintenance when a leaf adds a type, and it gives the facade no
opportunity to develop an opinion. The alternative — `pub mod instance
{ pub use vernier_core::{Evaluator, ParityMode, ...}; }` — buys the
ability to hide leaf internals from facade users, at the cost of a
hand-maintained list that will silently fall behind and a second,
weaker definition of "the public API." We take the drift-proof option;
if a leaf crate exports something that should not be public, the fix
belongs in the leaf crate.

**No prelude.** A prelude for this crate could contain only module
names, because no type name in vernier is unambiguous across paradigms.
`use vernier::prelude::*;` would then be a synonym for
`use vernier::{instance, panoptic};` — a second way to say the same
thing, which is the kind of redundant surface ADR-0035 spent a release
removing. Rejected.

### Feature posture and the invariant that makes it safe

| Feature     | Default | Gates                         |
|-------------|---------|-------------------------------|
| *(none)*    | —       | `instance`, `mask` — always on |
| `panoptic`  | on      | `pub use vernier_panoptic`    |
| `semantic`  | on      | `pub use vernier_semantic`    |
| `partial`   | on      | `pub use vernier_partial`     |

`instance` and `mask` are unconditional rather than default-on
features. Two reasons: `vernier-core` already depends on `vernier-mask`
(ADR-0009 — the segm `Similarity` lives in core and consumes `Rle`), so
gating `mask` would save no compilation and only remove a name; and
making the instance paradigm optional would mean
`default-features = false` yields a crate with no public items. That is
the empty crate this ADR exists to eliminate, reconstructed as a build
configuration. **`vernier` is non-empty in every reachable feature
combination**, and that is a property we now care about for reasons
beyond taste.

> **Invariant (hard).** `#[cfg(feature = ...)]` may appear in
> `crates/vernier/src/lib.rs` on `pub use` lines and nowhere else in
> the crate. No feature of `vernier` may be forwarded to a paradigm
> crate, enable a paradigm crate's feature, or otherwise reach past the
> re-export. A feature of this crate changes what is *nameable*, never
> what is *computed*. Violating this reintroduces the behavior
> fragmentation ADR-0047 rejected, through a door we opened for
> ergonomics.

CI gate: `cargo hack --feature-powerset check -p vernier`. Eight
configurations on a crate with one source file; the cost is noise, and
it is the mechanical enforcement that keeps the invariant from decaying
into a comment.

### Versioning and publish order

`vernier` inherits `workspace.package.version` like every other member.
It first publishes at the workspace's next release version — not at
0.0.1, and not restarting a version line. crates.io accepts any version
above the existing 0.0.0.

> **Errata (implementation, 2026-08-23).** This section was drafted
> against the premise that 0.2.0 was the current *unpublished*
> development version. It is not: `v0.2.0` was tagged and published on
> 2026-06-11, and main has moved past it. Two corrections follow.
>
> 1. The facade first publishes at whatever the **next** release cut
>    is, not at 0.2.0. Nothing else in this ADR depends on the specific
>    number, and §"What this ADR explicitly does not decide" already
>    declines to schedule a release.
> 2. The claim that publishing `vernier` "requires publishing the
>    entire 0.2.0 workspace set" — and the first decision driver's
>    "the facade physically cannot ship ahead of a full workspace
>    release" — do not hold at 0.2.0, because that workspace set is
>    already on crates.io. The packaged facade builds *and* passes all
>    six doctests against the published 0.2.0 leaf crates, outside the
>    workspace.
>
>    **Decision taken (2026-08-23):** publish `vernier@0.2.0`
>    standalone, ahead of any release. This is not the rushed release
>    §"Versioning and publish order" rejects — no tag is cut, no wheel
>    ships, no other crate moves, and the version pairs exactly with
>    the live 0.2.0 leaf set. It resolves the registry complaint by
>    construction rather than by argument, which is what the
>    §"Consequences" *Positive* clause promised. The lockstep property
>    is unchanged for every future release, where the facade's version
>    pins will point at an as-yet-unpublished workspace set; the
>    standalone path is available only because 0.2.0 is already live.
>
>    Operator steps: `docs/engineering/release-runbook.md`
>    §"One-off: publishing the facade ahead of a release".

This has an operational consequence that decides the reply strategy.
The facade's path dependencies pin the workspace version, so the
resolved graph is only publishable as part of a full workspace release.
**Publishing `vernier` therefore requires publishing the whole
workspace set at that version** — the facade cannot ship ahead of a
release, by construction, and lockstep is not a policy choice here but
a property of the dependency graph.

We do **not** cut a rushed 0.2.0 to beat 2026-08-24. A release shipped
to satisfy a support ticket is a worse outcome than a support ticket.
The reply carries this ADR and its PR; the policy's development-activity
branch is satisfied by a merged architectural decision with a named
implementation plan, and that is a truthful representation of where the
project is.

Publish order on the release tag, dependency-first:

```
vernier-mask → vernier-core → {vernier-panoptic, vernier-semantic, vernier-partial}
             → vernier-cli → vernier
```

The facade is last. `docs/engineering/release-runbook.md` gains the row.
Trusted Publisher (OIDC) needs a one-time entry for `vernier` at
`crates.io/me/trusted-publishers` before the tag — note this is a
`publish-update` scope, not `publish-new`, since we already own the name.

### Disposition of the v0.0.0 artifact

Yank v0.0.0 **after** the first real `vernier` release is live, never
before. Yanking is not
deletion — the artifact and its stale `vernier-rs/vernier` URL are
permanent — but it removes the empty version from dependency
resolution. Yanking first would leave the name held by a yanked empty
crate, which is a worse posture to be holding while a support ticket
is open.

The stale URL on v0.0.0 requires no further action, consistent with
`registry-reservations.md` §"Cosmetic gotcha". It is now, however,
worth understanding as the proximate cause of the report rather than as
a cosmetic footnote, and the reply should say so.

### Retiring the reservation practice

`tools/reservations/` is deleted in full: the four crate skeletons, the
PyPI skeleton, and `reserve.sh`. Every name it holds has been redeemed
by a real release; the directory's stated purpose is complete and its
continued existence is an invitation to repeat the defect.

`docs/engineering/registry-reservations.md` is rewritten from a
reservation register into a **published-artifact register**: what is on
each registry, at what version, under which Trusted Publisher config.
The §"Why placeholders, not the real crates" rationale is deleted
rather than amended — it argued for a practice we are ending, and
leaving it in place as history would make it discoverable as guidance.

The replacement rule, project-wide:

> **A crate name is claimed by its first real release, and never before.**
> If a design anticipates a crate, the name is recorded in the ADR that
> anticipates it. Nothing is published to hold it.

### Registry posture for anticipated crates

This applies immediately to work in design. The optimal-assignment
crate (`vernier-assign`, per the in-flight matching-layer ADR) and any
future 3D evaluation crate are **not** pre-reserved. They publish at
their first real release, following the `vernier-cli` lifecycle
precedent in ADR-0015 §"Workspace integration" — reservation retired,
workspace member added, real version published — minus the reservation
step, which no longer exists.

The residual risk is that a third party takes `vernier-assign` in the
interim. We accept it. The probability is low for a prefixed name with
no independent meaning; the impact is a rename during design, which is
cheap precisely because the crate is unpublished; and the RFC 3463
discussion records that prefixed names with a clear association to an
established project are treated as legitimate rather than as squats
under the same policy. Trading a certain, recurring policy violation
for an unlikely rename is the correct side of that trade.

### Consequences

- **Positive.** The crates.io complaint is resolved by construction
  rather than by argument — there is no version of "is this genuine
  functionality?" that a facade with real re-exports and compiled
  doctests loses. Rust consumers get a single dependency and a module
  map that matches the Python one they may already know. The facade
  cannot drift from the leaf crates because it *is* the leaf crates.
  The practice that produced the problem is gone, and the rule
  replacing it is one sentence.
- **Negative.** The facade's public API is the union of five leaf APIs
  with no insulation, so **any** breaking change in any leaf crate is a
  breaking change in `vernier`. That is the price of zero editorial
  control and it is charged on every release. The release matrix grows
  to seven publishable crates and the runbook gains a strictly-ordered
  final step. A reader landing on docs.rs for `vernier` sees crate
  aliases and must click through to the leaf crates for real
  documentation; the crate-level rustdoc has to carry the orientation
  burden that a curated surface would have carried structurally.
  Deleting `tools/reservations/` forfeits the ability to hold a name
  under time pressure, which we may one day want.
- **Neutral.** `cargo add vernier` gets the library; `cargo install
  vernier-cli` gets the binary. That asymmetry is documented rather
  than fixed, and fixing it is out of reach for as long as ADR-0015
  Axis 1 stands. The paradigm crates remain independently published and
  independently depend-able; nothing about the facade deprecates them,
  and the narrow consumer's path is unchanged.

### Documentation obligation

The facade ships crate-level rustdoc carrying: the module map, the
"depend on a leaf crate if you want a narrower tree" guidance, and one
**compiled doctest per re-exported module**. The doctests are not
decoration — they are what makes `cargo test -p vernier` meaningful and
what makes the crate demonstrably non-empty to a reader and to a
reviewer. This is the ADR-0027 code-tested-docs discipline applied to
the one crate where the docs *are* the content.

## What this ADR explicitly does not decide

- **Whether `cargo install vernier` should ever work.** It would
  require the `[[bin]]` in the facade, which puts `clap` in every
  library consumer's tree — ADR-0015 Axis 1 Option 2, rejected there
  and not reopened here. Revisit only if Cargo's `required-features`
  semantics change such that a bin's dependencies can be excluded from
  a lib consumer's resolution.
- **Whether the facade ever gains a curated surface.** Whole-crate
  aliasing is the decision. If a leaf crate's public API turns out to
  need hiding, that is a defect in the leaf crate and gets fixed there.
  A later ADR may revisit; this one closes it.
- **The `vernier-assign` crate name, shape, or boundary.** Owned by the
  optimal-assignment ADR. This ADR only decides that it will not be
  pre-reserved.
- **3D evaluation crate naming and structure.** Not yet in design.
- **Any change to the published paradigm crates.** The facade adds a
  consumer; it does not touch `vernier-core`, `vernier-mask`,
  `vernier-panoptic`, `vernier-semantic`, or `vernier-partial`. The
  ADR-0005 firewall is untouched.
- **The 0.2.0 release contents or date.** This ADR adds a crate to
  whatever 0.2.0 turns out to be; it does not schedule it.
- **The text of the reply to crates.io.** Operational, not
  architectural. This ADR is what the reply cites.

## Pros and cons of the options

### Axis A

**A1 — facade crate (chosen)**

- 👍 Closes the policy question by construction, permanently.
- 👍 One dependency for the common Rust consumer; module map matches
  ADR-0029's Python surface.
- 👍 Drift-proof: the facade is definitionally the union of the leaves.
- 👍 Retains the name across all three registries.
- 👎 Seventh publishable crate; strictly-ordered release step.
- 👎 No semver insulation — every leaf break is a facade break.

**A2 — rename `vernier-cli` to `vernier`**

- 👍 `cargo install vernier` works; one fewer name to explain.
- 👎 Hard break for existing `cargo install vernier-cli` users and for
  `cargo binstall` asset mapping (ADR-0034).
- 👎 Leaves the Rust *library* entry at `vernier-core` permanently,
  which is the actual ergonomic gap.
- 👎 Does not retire the reservation practice.

**A3 — relinquish**

- 👍 Zero work. Honest about the crate's current state.
- 👎 Unrecoverable. A third party owning `vernier` on crates.io, next
  to our `vernier` wheel and our `vernier` binary, is a permanent
  confusion and impersonation surface on the project's entry path.
- 👎 Asymmetric trade: a week of work against a forever cost.

**A4 — better placeholder**

- 👍 Fits inside the deadline with certainty.
- 👎 Still an empty crate. The policy text is about functionality,
  purpose, *or* repository activity — we would be leaning entirely on
  the third branch while the artifact itself continues to advertise
  that it has no code.
- 👎 Leaves `tools/reservations/` alive and the defect reproducible.
- 👎 Invites a second report, from a stronger position for the reporter.

### Axis B

**B1 — unconditional**

- 👍 Simplest possible manifest; no feature matrix, no CI powerset.
- 👍 Zero risk of the cfg invariant decaying.
- 👎 A bbox-only consumer pays compile time for panoptic and semantic
  to get the convenient name, and their only remedy is to abandon the
  facade and switch dependency identity to `vernier-core`.

**B2 — additive re-export-only features (chosen)**

- 👍 The narrow consumer trims within the facade instead of leaving it.
- 👍 Additive and monotone: enabling a feature only adds names.
- 👍 Non-empty in every feature combination, by construction.
- 👎 Introduces Cargo features to a project that has been deliberately
  hostile to them. Mitigated by the hard invariant and the powerset
  gate, but the precedent needs watching — the next contributor who
  wants a feature will cite this ADR.
- 👎 Eight CI configurations, forever.

**B3 — docs-only facade**

- 👍 No dependency edges at all.
- 👎 It is an empty crate with a good README, which is the thing the
  registry reported us for.

## Links and references

- ADR-0001 — Record architecture decisions; gates significance and pins
  the "numbers assigned on merge" rule this ADR's number follows.
- ADR-0005 — `Similarity` trait + matching-engine API lock; untouched.
- ADR-0009 — `vernier-mask` crate split; source of the core → mask
  dependency that makes a `mask` feature pointless.
- ADR-0015 — `vernier-cli` as a workspace binary. Axis 1 (no bin in a
  library crate) is inherited unchanged; §"Workspace integration"
  is the lifecycle precedent for retiring a reservation.
- ADR-0025 / ADR-0028 — panoptic and semantic sibling crates; the
  paradigm separation the module map reflects.
- ADR-0027 — Documentation framework; the code-tested-docs discipline
  the facade's doctests answer to.
- ADR-0029 — Per-paradigm submodule namespace. The Rust module map
  mirrors it; the collision analysis shows why both arrive at the same
  shape independently.
- ADR-0034 — aarch64 release target; `cargo binstall` asset mapping,
  relevant to why A2 is a hard break.
- ADR-0035 — API surface consolidation; the redundant-surface
  discipline behind rejecting a prelude.
- ADR-0047 — Threading model; the "one wheel, one behavior" rejection
  of feature-gated behavior that the B2 invariant is scoped against.
- `docs/engineering/registry-reservations.md` — superseded in part;
  rewritten as a published-artifact register.
- `docs/engineering/release-runbook.md` — gains the facade publish step
  and the Trusted Publisher entry.
- crates.io policies — <https://doc.crates.io/policies.html>
- RFC 3463, crates.io policy update —
  <https://rust-lang.github.io/rfcs/3463-crates-io-policy-update.html>
- crates.io support notice, 2026-08-10; response due 2026-08-24.
