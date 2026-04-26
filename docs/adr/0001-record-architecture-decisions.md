# ADR-0001: Record architecture decisions

- **Status:** accepted
- **Date:** 2026-04-26
- **Deciders:** project leads
- **Consulted:** —
- **Informed:** all contributors

## Context and problem statement

vernier will accumulate a long tail of small but consequential design
decisions: which version of pycocotools serves as the parity oracle, where
the FFI boundary lives, how we handle floating-point determinism, when to
release the GIL, what counts as "extended" versus "minimal" API, and many
more we haven't anticipated yet.

If those decisions live only in PR descriptions, Slack threads, and
contributors' heads, three things go wrong as the project grows:

1. **Decisions get relitigated.** Six months later, someone proposes a change
   that violates an earlier decision but doesn't know the earlier decision
   exists or why it was made. The discussion is repeated from scratch,
   sometimes with a different outcome, often without learning from the
   original trade-offs.
2. **Newcomers can't catch up.** There's no good answer to "why is the project
   structured this way?" except reading three years of git history.
3. **Reasoning is lost.** The decision survives in the code but the *why*
   doesn't, so future maintainers can't tell which constraints still apply
   and which have evaporated.

This is a well-understood problem with a well-understood solution: keep a
journal of architecturally significant decisions in the repository, beside
the code they describe.

## Decision drivers

- Decisions must be discoverable by new contributors without tribal knowledge.
- The process must be lightweight enough that people actually use it; if it
  feels like writing a thesis, contributors will route around it.
- Decisions must be linkable from PRs, code comments, and other docs.
- Records must be immutable once accepted, so that historical reasoning isn't
  silently rewritten.

## Considered options

1. **Wiki / Notion / external doc.** Easy to write, but lives outside the
   code, drifts out of sync, and requires separate access management.
2. **Long-form design docs in `docs/`.** Better than a wiki, but tends to
   produce a small number of large documents that nobody updates rather than
   a steady stream of focused decisions.
3. **Architecture Decision Records (ADRs)** in `docs/adr/`, one Markdown
   file per decision, in a numbered sequence, immutable once accepted.

## Decision outcome

Chosen option: **ADRs in `docs/adr/`**, using the
[MADR](https://adr.github.io/madr/) format.

The format is described in `docs/adr/template.md`. The lifecycle is described
in `CONTRIBUTING.md`. In short: significant changes start as a `proposed` ADR
in a PR, are discussed, and become `accepted` on merge. Once accepted, an
ADR is not edited; if circumstances change, a later ADR supersedes it and
sets the older one's status to `superseded by ADR-NNNN`.

### What counts as "significant"

ADRs are required for changes that:

- Affect the public API (Python, Rust, or CLI).
- Cross the FFI boundary.
- Change the parity contract with the reference oracle.
- Change the data model, error model, or threading model.
- Add or remove a top-level dependency, a build target, or a supported
  platform.
- Set a project-wide convention (style, naming, layout).

ADRs are *not* required for typo fixes, dependency version bumps, internal
refactors with no API impact, or test additions.

### Numbering and naming

- ADRs are numbered sequentially starting from `0001`. Numbers are assigned
  on merge, not on draft, to avoid renumbering churn from concurrent PRs.
- Filenames are `NNNN-short-kebab-title.md`. The title is in imperative mood
  ("use DLPack for tensor FFI", not "DLPack-based tensor FFI").
- Statuses: `proposed`, `accepted`, `superseded by ADR-NNNN`, `deprecated`.

### Consequences

- **Positive.** Newcomers have a single place to read project history.
  Decisions accumulate context that compounds in value over time. PRs become
  shorter because the rationale lives in the ADR, not the commit message.
- **Negative.** Light overhead per significant change. Some contributors
  will resist writing prose; the burden falls on reviewers to ask for an ADR
  when one is missing.
- **Neutral.** ADRs are a discipline, not a tool. They work to the extent
  the team takes them seriously; they fail silently if treated as paperwork.

## Links and references

- Michael Nygard, [Documenting Architecture Decisions](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)
  (the original 2011 article).
- [adr.github.io](https://adr.github.io/) — community resources, tools, and
  format variants.
- [MADR](https://adr.github.io/madr/) — the specific Markdown format used here.
