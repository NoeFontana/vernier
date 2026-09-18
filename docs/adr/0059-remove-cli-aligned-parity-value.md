# ADR-0059: Remove the `aligned` value from `vernier eval --parity-mode`

- **Status:** proposed
- **Date:** 2026-09-17
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** all contributors

## Context and problem statement

ADR-0002's [2026-05-10 amendment](0002-three-tier-parity-model.md#amendment-2026-05-10-collapse-aligned-into-strict)
collapsed the `aligned` disposition tier into `strict`. Every row the
quirks survey had tagged `aligned` was already asserted bit-equal
against pycocotools, so the tier carried no tolerance budget of its
own; the label was review-time annotation ("structurally different,
numerically identical") that belongs in a row's rationale rather than
in the disposition vocabulary.

The runtime followed that amendment everywhere except one place.
`vernier_core::parity::ParityMode` ships `Strict` and `Corrected`
only. The Python surface is `Literal["strict", "corrected"]` and
rejects `"aligned"` with `ValueError`, regression-pinned by
`tests/python/test_evaluator.py::test_evaluator_rejects_invalid_parity_mode`.
The single surviving place a user can still type the word is the CLI:
`crates/vernier-cli/src/cli.rs` keeps a `ParityModeArg::Aligned`
variant whose only behaviour is to collapse to `ParityMode::Strict` in
the `From` impl.

That leftover is worse than a no-op. It is a documented, tab-completed,
`--help`-listed value that implies a third evaluation mode exists;
`vernier eval --parity-mode aligned --emit json` then writes
`"parity_mode": "strict"` into the result document, so the flag a CI
job records is not the flag it passed. Every doc page that describes
the CLI has had to carry a sentence explaining that one of the values
it lists is a lie. ADR-0002's own follow-up list names this removal as
outstanding work, and ADR-0015 §"Output stability" requires a
follow-up ADR before any CLI flag value is removed. This is that ADR.

## Decision drivers

- **ADR-0015 §"Output stability".** Flag additions are additive across
  patches; flag *removals* need their own ADR. The CLI surface is a
  committed surface, so this cannot be done as drive-by cleanup.
- **ADR-0002's two-tier contract.** The documented vocabulary is
  `strict` / `corrected`. A user-typeable third value contradicts the
  contract the project asks reviewers to apply.
- **The alias is already lossy.** The JSON formatter emits the
  *kernel-resolved* mode (ADR-0015 §"Formatter: JSON"), so an
  `aligned` invocation cannot be recovered from its own output. Keeping
  a flag value that does not round-trip through the result document is
  a worse compatibility story than not having it.
- **Silence is the failure mode to avoid.** Whatever we do, a script
  that passes `aligned` today must not keep running while quietly
  meaning something the operator did not type, and must not crash with
  a panic or a stack trace.
- **Pre-1.0 breakage budget.** The project is on the 0.1.x line, where
  "minor bumps signal breakage" is the stated contract (CHANGELOG
  preamble). A breaking removal is affordable now and gets more
  expensive with every release that ships the alias.

## Considered options

1. **Remove the value outright.** `aligned` becomes an invalid value;
   the CLI exits with an argument error naming the two valid values.
2. **Hide it, keep it working.** `#[value(hide = true)]` — drops it
   from `--help` and completions while still accepting it silently.
3. **Deprecate loudly, remove in a later release.** Accept `aligned`,
   print a deprecation warning to stderr, remove after a release cycle.
4. **Leave it.** Keep the alias and the explanatory sentence on every
   page that documents the flag.

## Decision outcome

Chosen option: **Option 1 — remove the value outright**, accepting that
this is a breaking change to a committed CLI surface.

The variant is deleted from `ParityModeArg`, which leaves the enum
exactly mirroring `vernier_core::ParityMode`. `--parity-mode aligned`
is now a hard argument error, exit code 2 (clap's argument-error code,
the same code `--dilation-ratio` with a non-boundary kernel already
produces per ADR-0015 §"Surface"), so shell scripts keep their existing
"bad invocation" vs "eval failed" split.

The rejection does not rely on clap's generic `invalid value` text. A
value parser intercepts the retired spelling specifically and says what
happened and what to type instead:

```
error: invalid value 'aligned' for '--parity-mode <PARITY_MODE>': the
       `aligned` parity mode was removed (ADR-0059); ADR-0002 folded the
       `aligned` disposition tier into `strict`, so pass `strict` for the
       behaviour `aligned` used to select [possible values: strict, corrected]
```

Any other unknown value gets the same shape without the migration
sentence. `crates/vernier-cli/tests/eval.rs` pins both: that `aligned`
is rejected, that the message names `strict` as the replacement and
lists the valid values, and that the exit code is 2.

Options 2 and 3 were rejected for the same reason: both keep a value
that cannot round-trip through `--emit json`, so both prolong the state
where a CI job's recorded parity mode differs from its invocation.
Option 3 additionally spends a release cycle of documentation on a
value that has been a no-op alias since 2026-05-10 — the deprecation
window already happened, it just was not written down as one. Option 4
is the status quo the ADR-0002 follow-up list exists to close.

### Consequences

- **Positive.** The CLI's parity vocabulary is exactly ADR-0002's, with
  no explanatory footnote on `docs/reference/cli-output-schema.md`,
  `docs/how-to/cli-eval.md`, or the crate README. `ParityModeArg` and
  `ParityMode` are now one-to-one, so the `From` impl is total without a
  collapsing arm. The word `aligned` survives in the CLI crate only as
  the removal test's input.
- **Negative.** A breaking change: any pipeline pinning
  `--parity-mode aligned` fails at the next upgrade. The failure is
  loud, at argument-parse time, before any eval work, and the message
  names the replacement — but it is still a failure, and it lands in a
  patch-shaped change rather than at a version boundary the operator
  chose. The blast radius is bounded by the value having been an alias
  for `strict` since 2026-05-10: the fix is a one-word edit that cannot
  change any number the pipeline reports.
- **Neutral.** Exit code 2 and the "argument error" category are
  unchanged; only the set of accepted values shrinks. The JSON schema
  is untouched — `parity_mode` was already emitting the kernel-resolved
  value, so no result document changes shape.

## Pros and cons of the options

### Option 1 (chosen) — remove outright

- 👍 The CLI stops advertising a mode that does not exist.
- 👍 Fails at parse time with a message naming the replacement; no
  silent behaviour change is possible.
- 👍 `ParityModeArg` mirrors `ParityMode` one-to-one; the collapsing
  arm in `From` disappears.
- 👎 Breaks pinned invocations without a version boundary to hide
  behind.

### Option 2 — hide but keep accepting

- 👍 Zero breakage.
- 👎 Keeps the alias that does not round-trip through `--emit json`.
- 👎 A hidden-but-accepted value is worse than either honest choice:
  undiscoverable for new users, still live for old scripts, and
  invisible to the reader of `--help` trying to understand a colleague's
  pipeline.

### Option 3 — deprecate loudly, remove later

- 👍 Conventional; gives operators a cycle to migrate.
- 👎 The deprecation cycle already ran (2026-05-10 → now) without being
  labelled one.
- 👎 Needs a stderr warning path the CLI deliberately does not have
  (ADR-0015 §"Decision drivers" — no logger, no `tracing`).
- 👎 Defers the doc cleanup by a release while adding a second breaking
  change later.

### Option 4 — leave it

- 👍 No work, no breakage.
- 👎 Every page documenting the flag carries a footnote explaining that
  a listed value is not a mode.
- 👎 Leaves ADR-0002's follow-up open indefinitely and keeps `aligned`
  in the project's live vocabulary.

## Links and references

- [ADR-0002](0002-three-tier-parity-model.md) — parity model; the
  2026-05-10 amendment that folded `aligned` into `strict` and listed
  this removal as a follow-up.
- [ADR-0015](0015-vernier-cli.md) — `vernier-cli`; §"Surface" documents
  the flag, §"Output stability" is the rule requiring this ADR.
- `crates/vernier-cli/src/cli.rs` — `ParityModeArg` and its
  `From<ParityModeArg> for ParityMode` impl.
- `crates/vernier-cli/tests/eval.rs` — the rejection test.
- `docs/reference/cli-output-schema.md` — the `parity_mode` field.
- `CHANGELOG.md` `[Unreleased]` §"Removed".
