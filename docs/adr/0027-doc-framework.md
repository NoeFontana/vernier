# ADR-0027: Documentation framework — Diátaxis on `mkdocs-material`, code-tested, gated in CI

- **Status:** accepted
- **Date:** 2026-05-03
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** all contributors

## Context and problem statement

Through 0.0.1, every `docs/` user-facing directory is a stub
README. The ADRs are dense and the engineering notes are thorough,
but the layer a typical user reaches first — tutorials, how-to
guides, reference, explanation — does not exist. For a
parity-preserving evaluator targeting research and MLOps audiences,
that gap is the single largest blocker to adoption: the project's
quality is invisible because nobody can find their way into it.

The decision in front of us is the *framework*, not the content.
What information architecture does the docs site adopt, what
toolchain renders it, what discipline enforces that the docs stay
true to the code, and who owns ongoing maintenance? Each of those
sub-decisions is small individually; together they set the
discoverability and trust posture for every release after 0.1.0.

This ADR triggers ADR-0001 §"Set a project-wide convention" (the
content-discipline rules for Diátaxis quadrants and the page-length
cap apply to every contributor) and §"Add or remove a top-level
dependency" (mkdocs-material, mkdocstrings, mike, lychee, codespell,
pytest-markdown-docs land as dev-dependencies on the docs site).

## Decision drivers

- **The first-success window is short.** A user evaluating vernier
  against pycocotools / faster-coco-eval has a budget of perhaps
  ten minutes to get a number out. If the docs framework doesn't
  produce that experience, no other quality of the project recovers
  the lost user.
- **Migration is the highest-leverage adoption surface.** Four
  competing libraries (pycocotools, faster-coco-eval, panopticapi,
  lvis-api, mmsegmentation, cityscapesScripts) each represent a
  cohort of pre-trained users. Migration guides for each lower the
  switching cost from "afternoon project" to "afternoon
  search-and-replace." Burying them under a generic how-to umbrella
  loses leverage; surfacing them as their own top-level navigation
  category is the load-bearing IA decision.
- **Code-example rot is the failure mode that destroys trust.** A
  documentation site whose tutorial doesn't run is worse than no
  documentation site, because the user concludes the project is
  unmaintained. The framework needs an enforcement mechanism, not
  a guideline — every fenced code block in `docs/` runs as a test,
  full stop.
- **Auto-generated reference where possible.** Hand-written
  reference rots within two releases; generated reference rots
  only when the underlying code does, and the rot is then a CI
  failure rather than silent drift. Python API via mkdocstrings
  reading the existing pyright-checked docstrings, CLI via clap
  derive, ADR index via the existing front-matter — three sources
  of generated content, zero hand-maintained reference pages.
- **Trust signal density.** The quirks surveys, the parity
  contract, the bench harness output, and the ADR list are
  vernier's distinctive technical artifacts. The framework has to
  surface them at depth ≤ 2 from the landing page, not bury them
  in an "advanced" subsection.
- **No new top-level *runtime* dependencies.** Per ADR-0001. The
  docs toolchain lives in a separate `[dependency-groups].docs`
  group; nothing here propagates into `vernier-core` or the
  `vernier` Python package. The audience installs `vernier` from
  PyPI; they don't get mkdocs pulled in.
- **Ownership from day one.** ADR-0017 set the precedent that
  release-time artifacts (the bench harness output) are
  release-runbook responsibilities. Documentation completeness
  joins that list.

## Considered options

The framework has six orthogonal axes. Each is decided independently;
the chosen design is the combination of one option per axis.

### Axis A — Information architecture

1. **Strict Diátaxis** — four top-level directories
   (`tutorials/`, `how-to/`, `reference/`, `explanation/`),
   no exceptions.
2. **Diátaxis with a top-level Migrate section** — four
   Diátaxis directories plus a `migrate/` peer for migration
   guides; everything else fits the four quadrants.
3. **Custom IA** — user-journey-driven sections (Get started,
   Common tasks, Reference, Architecture).
4. **Single-page docs** — one README at scale.

### Axis B — Toolchain

1. **mkdocs-material** with the Python plugin ecosystem.
2. **Sphinx** with myst-parser (Markdown).
3. **mdBook** (Rust-native, matches workspace).
4. **Docusaurus** (React/Node).
5. **Custom Next.js / Astro site.**

### Axis C — Code-example discipline

1. **All fenced code blocks tested in CI** via
   `pytest-markdown-docs` or equivalent.
2. **Tutorials only** tested in CI; how-to and reference exempt.
3. **No automated testing** — convention only.

### Axis D — Reference generation

1. **Auto-generated everywhere possible** (Python via mkdocstrings,
   CLI via clap derive, ADR index from front-matter).
2. **Hand-written, with generated cross-references**
   (intersphinx-style links).
3. **Auto-generated only on the Rust side** (docs.rs); hand-write
   Python reference.

### Axis E — Versioning policy

1. **Per minor release** via `mike` plugin; patch releases
   overwrite within a minor.
2. **Per patch release** — every 0.0.x has its own URL.
3. **Latest only** — stable URL, latest content, no history.

### Axis F — Maintenance ownership

1. **Project lead through 0.1.x**; reviewer rotation when
   external doc PRs become regular (>5/month).
2. **Dedicated docs maintainer from day one** (separate
   CODEOWNERS entry).
3. **Round-robin** across all contributors.

## Decision outcome

Chosen: **A2 + B1 + C1 + D1 + E1 + F1.**

### Information architecture (A2)

The docs site adopts Diátaxis with one navigational bend: a
top-level `Migrate` section that pulls migration guides out of the
how-to pile because they are the highest-leverage adoption surface
and deserve their own entry point. Strict Diátaxis (A1) would bury
those guides three clicks deep; custom IA (A3) loses the
quadrant-discipline benefit that makes Diátaxis work. The IA tree
lands as:

```
docs/
├── index.md                          # landing — value prop, install, 60-second example
├── tutorials/                        # learning-oriented; cap at 3 for 0.1.0
│   ├── first-evaluation.md           # 5-minute COCO val2017 quickstart
│   └── training-loop.md              # StreamingEvaluator + W&B/TensorBoard
├── migrate/                          # task-oriented; first-class
│   ├── from-pycocotools.md
│   ├── from-faster-coco-eval.md
│   ├── from-panopticapi.md           # ships with ADR-0025 implementation
│   ├── from-lvis-api.md              # ships with ADR-0026 implementation
│   └── from-mmsegmentation.md        # ships with ADR-0027 implementation
├── how-to/                           # task-oriented; ~6 at 0.1.0, grows over time
│   ├── ci-quality-gate.md
│   ├── debug-with-tables.md
│   ├── boundary-iou.md
│   ├── keypoints-oks.md
│   ├── background-evaluator.md
│   └── custom-breakdowns.md
├── reference/                        # information-oriented; mostly auto-generated
│   ├── python-api/                   # mkdocstrings, generated from docstrings
│   ├── rust-api.md                   # link out to docs.rs/vernier-{core,mask,...}
│   ├── cli.md                        # generated from clap; flags + exit codes
│   ├── json-schema.md                # the v1 schema from ADR-0015
│   ├── parity-quirks/                # navigable browser over the quirks surveys
│   │   ├── pycocotools.md
│   │   ├── boundary-iou.md
│   │   ├── panopticapi.md
│   │   ├── lvis-api.md
│   │   └── semantic-segmentation.md
│   └── adr-index.md                  # generated from docs/adr/ front-matter
├── explanation/                      # understanding-oriented; ~4 at 0.1.0
│   ├── why-parity-matters.md
│   ├── architecture-overview.md
│   ├── choosing-an-evaluator.md
│   └── performance-philosophy.md
├── benchmarks.md                     # rendered from ADR-0017 harness JSON
└── compare.md                        # vernier vs pycocotools vs faster-coco-eval
```

`docs/adr/` and `docs/engineering/` are not user-facing; they
remain in-tree but aren't surfaced in the navigation. The
`reference/adr-index.md` page exposes ADR titles + status for
users who want to dig deeper — a navigation shortcut, not a
full re-render.

### Toolchain (B1)

mkdocs-material with the following plugin set:

- **mkdocstrings (Python handler)** — auto-generates Python API
  reference from docstrings. The docstrings already exist in
  `python/vernier/` and are pyright-checked; mkdocstrings just
  surfaces them.
- **mike** — versioned-docs deployment. Pinned by minor version;
  patch releases overwrite within a minor.
- **mkdocs-redirects** — preserves URLs across reorganization.
- **lychee** — link checker, runs in CI.
- **pytest-markdown-docs** (or equivalent) — runs every fenced
  Python code block as a test.
- **codespell** — typo and common-misspelling check on `docs/`.

Sphinx (B2) was rejected: heavyweight, RST-encumbered (myst-parser
papers over but doesn't fix), the Python audience for vernier is
already on mkdocs-material via Polars / Astral / pydantic. mdBook
(B3) was rejected: lacks the Python-side mkdocstrings integration
that auto-gens the API reference, would force splitting Python
and Rust docs across two sites, and the user-facing audience for
vernier is at least 80% Python. Docusaurus (B4) and custom
Next.js (B5) were rejected: introduce a Node.js dependency the
project does not otherwise have, and the build pipeline complexity
is unjustified for a documentation site.

The Rust API documentation ships at `docs.rs/vernier-core`,
`docs.rs/vernier-mask`, `docs.rs/vernier-panoptic` (when ADR-0025
lands), `docs.rs/vernier-semantic` (when ADR-0027 lands). The
vernier docs site links out to docs.rs rather than re-rendering —
docs.rs is the canonical home for Rust API reference, and
re-rendering is wasted work.

The benchmark page is rendered from the ADR-0017 harness's JSON
output. The release-runbook gains a step "verify benchmarks page is
current" (added to `docs/engineering/release-runbook.md` as part of
this ADR).

### Code-example discipline (C1)

Every fenced code block in `docs/` tagged `python` runs as a test.
The test corpus is part of the regular `pytest` invocation; a
tutorial-breaking PR fails CI. The mechanism is
`pytest-markdown-docs` (or equivalent — the choice of plugin is
implementation detail; what matters is the gate). The harness
treats:

- **Tutorials**: every code block runs end-to-end in sequence.
  State carries between blocks within a tutorial; a broken first
  block breaks all subsequent blocks.
- **How-to guides**: every code block runs in isolation. State
  does not carry; each block is its own test.
- **Reference**: code blocks in docstrings are tested via
  `pytest --doctest-modules` on `python/vernier/`. Reference
  pages render from those docstrings; if the docstring's example
  passes, the reference page's example passes.
- **Migration guides**: every "after" code block runs (the "after"
  is what the user will paste). "Before" blocks are illustrative
  pycocotools / lvis-api / mmsegmentation code; not tested by
  vernier's CI.

C2 (tutorials only) was rejected: how-to guides are the
most-clicked pages in adoption, and a broken how-to guide is a
broken adoption funnel. C3 (no automated testing) was rejected:
hand-discipline doesn't survive contact with a six-month
contributor turnover.

### Reference generation (D1)

Auto-generation everywhere it's mechanically possible:

- **Python API** — mkdocstrings reads the existing pyright-checked
  docstrings in `python/vernier/`. No hand-written API reference.
  Page structure mirrors the namespace structure from ADR-0028:
  `reference/python-api/instance.md`, `panoptic.md`, `semantic.md`,
  `summarize.md`, plus the top-level shared types.
- **CLI** — `vernier-cli` uses clap's derive macros; a build-time
  codegen step emits `reference/cli.md` from the parsed clap
  structures. The CLI flags and exit codes documentation is the
  generated artifact.
- **JSON schema** — hand-curated reference, because the schema
  itself is the artifact (ADR-0015 v1 contract). The page
  documents the schema; the schema documents what the CLI
  produces.
- **Quirks browser** — rendered from the existing markdown
  surveys with a small post-processor that adds anchors and
  cross-links across surveys. The post-processor lives at
  `tools/docs/render-quirks.py`; produces one page per survey
  with navigation across surveys.
- **ADR index** — generated from `docs/adr/*.md` front-matter
  (status, title, date). Sorts by number; renders the title and
  one-line summary.

D2 (hand-written with cross-references) is the alternative; it's
how Sphinx-based sites typically work. Rejected because hand-
written reference rots, and the rot is silent. D3 (Rust auto, Python
hand-written) is rejected because the Python API is the larger user
surface and auto-generation is cheaper than hand-maintenance.

### Versioning (E1)

`mike` versions docs per minor release. Patch releases (0.1.0 →
0.1.1) overwrite within the minor. Minor releases (0.1.x → 0.2.0)
spawn a new versioned URL. The "stable" alias points at the
latest tagged release; the "latest" alias points at HEAD.

E2 (per patch) is rejected: adds noise to the version selector
and creates a maintenance treadmill for back-porting fixes. E3
(latest only) is rejected: a user pinned to 0.1.5 can't read
0.1.5-era docs after 0.2.0 ships, which breaks the trust contract.

The release runbook gains one step: "tag this minor's docs in mike
during publish". Patch releases need no doc-version step.

### Maintenance ownership (F1)

For 0.1.0 and through the early patch releases, docs maintenance
sits with the project lead. Once external contributions on docs
become regular (>5 external doc PRs per month), a docs reviewer
rotation establishes and a separate `CODEOWNERS` entry for `docs/`
follows. Until then, docs reviews go through the same review
process as code.

The single most important ongoing discipline: **every PR that
changes a public symbol updates the docstring in the same PR**.
Auto-generated reference handles the rest. The reviewer's job is
to bounce PRs that violate this rule. CI catches the easy cases
(missing docstring, broken example); reviewers catch the hard
cases (the docstring exists but is now misleading).

F2 (dedicated maintainer) is rejected on cost grounds: the
project doesn't have a dedicated docs role, and pretending it
does creates a single point of failure when that person is
unavailable. F3 (round-robin) is rejected because it spreads the
context too thin; documentation requires a holistic view of the
site, and rotating reviewers don't develop that view fast enough
to catch IA drift.

### CI gates

Five gates land in `.github/workflows/`:

1. **Doc coverage** — `interrogate --fail-under 100` on
   `python/vernier/`. Rust side enforces the existing
   `missing_docs = "warn"` lint, promoted to `deny` for
   `vernier-core` and `vernier-mask` public modules.
2. **Code-example testing** — `pytest tests/python/docs/` runs
   every fenced Python block in `docs/`. Part of the regular
   `just test` and `just test-py` invocations.
3. **Link checking** — `lychee` runs on the rendered site for
   every PR that touches `docs/`. External links checked
   weekly in a scheduled job (external link checks are flaky on
   PR timescales).
4. **Build success** — `mkdocs build --strict` (warnings are
   errors). Broken cross-references, missing files, malformed
   YAML all fail the build.
5. **Spelling** — `codespell` against `docs/`. Domain-specific
   terms live in `.codespellignore`.

The gates compose with the existing `just lint` workflow; running
locally before opening a PR catches issues early.

### Initial-release content scope

The minimum docs for a credible 0.1.0 are listed below with rough
effort estimates. Items marked **blocking** must exist before the
release tag; items marked **followup** can land in a 0.1.x patch.

| # | Item | Effort | Status |
|---|---|---|---|
| 1 | `index.md` landing page | 1 day | blocking |
| 2 | Tutorial: first evaluation (COCO val2017) | 2 days | blocking |
| 3 | Tutorial: training-loop integration | 2 days | blocking |
| 4 | Migrate from pycocotools | 2 days | blocking |
| 5 | Migrate from faster-coco-eval | 1 day | blocking |
| 6 | Migrate from panopticapi | 2 days | followup (with ADR-0025) |
| 7 | Migrate from lvis-api | 1 day | followup (with ADR-0026) |
| 8 | Migrate from mmsegmentation | 1 day | followup (with ADR-0027) |
| 9 | How-to: CI quality gate via `vernier-cli` | 1 day | blocking |
| 10 | How-to: debug with per-image tables | 1 day | blocking |
| 11 | How-to: boundary IoU | 0.5 day | blocking |
| 12 | How-to: keypoints OKS | 0.5 day | blocking |
| 13 | How-to: BackgroundEvaluator | 1 day | blocking |
| 14 | How-to: custom breakdowns (ADR-0016) | 0.5 day | followup |
| 15 | Reference: Python API (auto-gen) | 1 day setup | blocking |
| 16 | Reference: CLI (codegen from clap) | 1 day setup | blocking |
| 17 | Reference: JSON schema | 0.5 day | blocking |
| 18 | Reference: parity quirks browser | 2 days setup | blocking |
| 19 | Reference: ADR index (auto-gen) | 0.5 day | blocking |
| 20 | Explanation: why parity matters | 1 day | blocking |
| 21 | Explanation: architecture overview | 1 day | blocking |
| 22 | Explanation: choosing an evaluator | 1 day | blocking |
| 23 | Explanation: performance philosophy | 0.5 day | followup |
| 24 | Comparison page | 2 days | blocking |
| 25 | Benchmarks page (rendered from ADR-0017) | 2 days setup | blocking |
| 26 | mkdocs-material site setup + CI | 2 days | blocking |

Total blocking effort: ~24 days. With one engineer at 50% on
docs, that's ~10 calendar weeks; with a focused two-week sprint
at 100%, ~12 days plus review cycles. Followup items add another
~5 days post-release.

## What this ADR explicitly does *not* decide

- **Specific tutorial / how-to content.** This ADR sets the
  framework, the CI gates, and the IA. Individual content lands
  PR-by-PR with code-example testing as the quality bar. Each
  blocking item from the table above is a sub-PR, not a
  re-litigation of the framework.
- **Marketing landing site separate from the docs.** The docs
  site *is* the landing site for now. A separate marketing page
  introduces a second voice and a second update lifecycle for
  no measurable adoption benefit at 0.1.0. Revisit if vernier
  acquires a corporate home that justifies one.
- **AI-generated content.** Hard no. The audience is technical
  and recognizes the smell; the trust cost outweighs the time
  savings. This applies to LLM-written tutorials, explanations,
  "improve clarity" passes, and chatbots. The docs are the
  answer; if the docs aren't, fix the docs.
- **Translations.** Single language (English) at 0.1.0. The
  audience reads English papers and code comments; localization
  is a 1.0+ conversation if it happens at all.
- **Video tutorials.** High production cost, high rot rate (every
  API change breaks them), narrow audience. Reconsider after a
  real user requests them.
- **In-browser interactive playground.** WASM-compiling vernier
  for in-browser execution is technically possible but a
  multi-week effort. Defer.
- **Discord / Slack community widget.** GitHub Discussions tab
  is sufficient until user count justifies a chat surface.
- **Telemetry on the docs site.** GitHub Pages defaults only.
  No third-party analytics. The audience reads docs in research
  and corporate environments where third-party telemetry triggers
  compliance review and burns goodwill for no measurable
  docs-quality gain.
- **Whether the docs site has a custom domain.** GitHub Pages
  default URL through the early patch releases. Custom domain
  follows naturally if vernier acquires a project home that
  justifies one.

## Consequences

- **Positive.** First-success time drops from "unknowable" to
  "sub-10-minute" because the tutorial path is short and tested.
  Migration guides convert pycocotools / faster-coco-eval /
  panopticapi / lvis-api / mmsegmentation users at search-and-
  replace effort instead of read-the-source effort. The quirks
  browser surfaces vernier's distinctive technical artifact at
  depth 2 from the landing; reviewers in technical-due-diligence
  conversations find what they need in 30 seconds. The CI gates
  prevent the failure mode where docs and code drift; auto-
  generation handles 80% of reference at zero ongoing cost. The
  trust signals (parity contract, benchmarks page, ADR list) are
  navigable. The framework scales to ADR-0028's per-paradigm
  namespace by mirroring the IA structure.
- **Negative.** ~24 days of blocking effort before 0.1.0 ships;
  this is not free, and competes with the implementation work for
  the deferred ADRs (0025 panoptic, 0026 LVIS, 0027 semantic).
  Code-example testing means every PR that touches a Python
  symbol now also touches the docs that demonstrate it; the
  cost is real but bounded. The plugin set
  (mkdocs-material, mkdocstrings, mike, lychee, codespell,
  pytest-markdown-docs) is one more thing to keep up to date —
  pinned in the `[dependency-groups].docs` group, but the
  group's own maintenance is now an ongoing task. The minor-
  version freeze under `mike` means a new minor release that
  ships with a docs error has it pinned for the lifetime of
  that minor; mitigated by the patch-release overwrite policy
  but worth flagging.
- **Neutral.** No new top-level runtime dependencies. The Rust
  side is unaffected (cargo doc + docs.rs already work). The
  CI surface grows by five gates; each is small and well-bounded.
  The release runbook gains two steps; small additions to a
  document that already lists a dozen.

## Pros and cons of the options

### A. Information architecture

- **A1 strict Diátaxis.** 👍 unambiguous quadrants. 👎 buries
  migration guides under how-to; loses the highest-leverage
  adoption surface.
- **A2 Diátaxis + Migrate (chosen).** 👍 surfaces migration as
  first-class; quadrant discipline preserved elsewhere. 👎
  navigational bend that strict Diátaxis advocates may
  question.
- **A3 custom IA.** 👍 maximum flexibility. 👎 loses the
  quadrant discipline that prevents tutorial / how-to /
  reference / explanation drift.
- **A4 single-page docs.** 👍 one URL. 👎 doesn't scale past a
  few hundred lines; defeats the auto-gen reference benefit.

### B. Toolchain

- **B1 mkdocs-material (chosen).** 👍 mature plugin ecosystem;
  Polars / Astral / pydantic precedent; mkdocstrings auto-gens
  Python API. 👎 plugin set is one more thing to maintain.
- **B2 Sphinx.** 👍 academic-Python heritage. 👎 RST-encumbered
  even with myst-parser; heavyweight; declining mindshare.
- **B3 mdBook.** 👍 Rust-native. 👎 lacks Python-side auto-gen;
  forces split sites.
- **B4 Docusaurus.** 👍 powerful. 👎 introduces Node.js dep.
- **B5 custom Next.js.** 👍 maximum control. 👎 wrong place to
  out-engineer a docs problem.

### C. Code-example testing

- **C1 all blocks tested (chosen).** 👍 zero rot. 👎 every
  doc-touching PR tests docs.
- **C2 tutorials only.** 👍 cheaper. 👎 broken how-to guides
  break adoption.
- **C3 convention only.** 👍 zero CI cost. 👎 doesn't survive
  contributor turnover.

### D. Reference generation

- **D1 auto-gen everywhere possible (chosen).** 👍 zero rot;
  reference is mechanically true to code. 👎 setup cost.
- **D2 hand-written + cross-refs.** 👍 cleaner prose. 👎 rots
  silently.
- **D3 Rust auto, Python hand.** 👍 docs.rs handles half. 👎
  Python is 80% of the user surface; auto-gen there has the
  most ROI.

### E. Versioning

- **E1 per minor (chosen).** 👍 clean URL story; right cadence
  for the project's 0.x velocity. 👎 minor with docs error stays
  pinned (mitigated by patch-overwrite policy).
- **E2 per patch.** 👍 every release pinned. 👎 noisy version
  selector.
- **E3 latest only.** 👍 simplest. 👎 breaks pinned-version
  users.

### F. Ownership

- **F1 lead through 0.1.x (chosen).** 👍 single point of
  decision; consistent voice. 👎 single point of failure when
  the lead is unavailable.
- **F2 dedicated maintainer.** 👍 clear ownership. 👎 the
  project doesn't have one to dedicate.
- **F3 round-robin.** 👍 spreads context. 👎 spreads it too
  thin; rotators don't develop the holistic view that catches
  IA drift.

## Links and references

- ADR-0001 — Record architecture decisions. The ADR index page
  in this framework is generated from `docs/adr/*.md`
  front-matter; the framework formalizes a consumption surface
  ADR-0001 implicitly created.
- ADR-0007 — `patch_pycocotools`. Migration-from-pycocotools
  guide is the load-bearing artifact; this framework is the
  venue.
- ADR-0015 — `vernier-cli`. The CLI reference is auto-generated
  from clap derive structures; ADR-0015's JSON schema is the
  hand-curated `reference/json-schema.md` page.
- ADR-0017 — Local bench harness. The benchmarks page is
  rendered from ADR-0017's JSON output; the release runbook
  gains "verify benchmarks page is current" as a step.
- ADR-0019 — Result tables. `EvalResult.per_image` /
  `per_class` / `per_detection` / `per_pair` are the load-
  bearing how-to-debug-with-tables guide subject.
- ADR-0025 — Panoptic-quality evaluation. Migration-from-
  panopticapi guide ships with the implementation; the
  panopticapi quirks survey renders into the quirks browser.
- ADR-0026 — LVIS federated evaluation. Migration-from-lvis-
  api guide ships with the implementation; the LVIS quirks
  survey renders into the quirks browser.
- Semantic segmentation (ADR TBD — not yet proposed). Migration-from-
  mmsegmentation guide ships with the implementation; the
  semantic-segmentation quirks survey renders into the quirks
  browser.
- Namespace restructure (ADR TBD — not yet proposed). The Python API
  reference's page structure mirrors `vernier.instance` /
  `vernier.panoptic` / `vernier.semantic`.
- `docs/engineering/release-runbook.md` — gains two
  documentation-related steps per the §"Toolchain" and
  §"Versioning" subsections.
- `CONTRIBUTING.md` — gains a §"Documentation" section pointing
  contributors at the framework's CI gates.
- [Diátaxis](https://diataxis.fr/) — the documentation
  framework this site follows.
- [mkdocs-material](https://squidfunk.github.io/mkdocs-material/)
  — the toolchain.
- [Polars docs](https://docs.pola.rs/) — closest peer
  (Rust+Python, high-performance, similar audience). Reference
  for tone and navigation density.
- [Astral docs](https://docs.astral.sh/) (uv, ruff) — reference
  for concise modern Python+Rust docs at the top of the field.
