# Architecture Decision Records

This directory holds vernier's Architecture Decision Records (ADRs): the
immutable record of *why* the project is built the way it is.

The format is [MADR](https://adr.github.io/madr/) (Markdown Architecture
Decision Records). To propose a new ADR:

1. Copy `template.md` to `NNNN-short-kebab-title.md` with the next available
   number.
2. Fill in *Context*, *Decision*, *Consequences*. Set status to `proposed`.
3. Open a PR. Discuss until consensus.
4. On merge, status becomes `accepted`. From this point the ADR is
   immutable — supersede it with a later ADR rather than editing it.

## Index

| #    | Title                               | Status   |
| ---- | ----------------------------------- | -------- |
| 0001 | Record architecture decisions       | accepted |

(Update this table as ADRs land. Eventually we may automate it from the
front-matter, but until there are enough ADRs to make that worthwhile, hand
maintenance is fine.)
