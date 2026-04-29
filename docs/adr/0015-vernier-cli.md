# ADR-0015: Ship `vernier-cli` as a workspace binary that links `vernier-core` directly

- **Status:** proposed
- **Date:** 2026-04-29
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** all contributors

## Context and problem statement

Phases 1–3 have shipped the eval kernel for bbox, segm, boundary, and
keypoints, plus the `Evaluator` / `COCOeval` Python entry points and the
`patch_pycocotools` shim (ADR-0007). Today, the only way to invoke
vernier is from a Python interpreter: `import vernier`, instantiate
`Evaluator`, feed it bytes, read `Summary`. Every consumer of the
library — CI quality gates, robotics replay benches, Makefile-driven
batch jobs, the project's own non-regression harness — has to stand up
a Python environment to do so.

That assumption holds for the in-Python persona ADR-0007 was written
for. It breaks for three real consumers we already see:

1. **CI quality gates implemented in shell.** The standard recipe today
   is `python -c "from pycocotools.cocoeval import COCOeval; ..."`,
   either inlined in the workflow YAML or hidden in a `scripts/eval.sh`.
   It works; it is also a 30-line incantation that drags in numpy,
   COCOAPI's C extension build, and a virtualenv just to compute 12
   floats from two JSON files. The same persona also needs the *output
   side* of that recipe: a structured file the pipeline can parse with
   `jq`, share between jobs as an artifact, and store in a regression-
   tracking bucket. Today, those teams scrape the pycocotools `print()`
   output with a regex.
2. **Pipelines that route through `subprocess`.** Robotics replay
   pipelines, ROS launch files, and dataset-management tools call out
   to subprocesses; the natural surface for one of those is a CLI
   binary, not an in-process import. The current workaround is the
   same `python -c` recipe wrapped in a shell script.
3. **Maintainers' own COCO val2017 non-regression checks.** The whole-
   dataset parity smoke (`tests/python/parity/test_coco_val.py`) runs
   pycocotools and vernier in-process and diffs them. A second oracle
   that runs vernier from the binary the user actually installs would
   close the loop on "the wheel works in isolation, the binary works
   in isolation, both produce the same numbers" — a property today's
   harness asserts only implicitly via the wheel.

The reservation crate `vernier-cli` (`tools/reservations/crates/vernier-cli/`,
v0.0.0 on crates.io, README dated to a never-shipped v0.1.0) has been
holding the name in anticipation of this ADR. It contains no real
code; promoting it into the workspace as the v0.2.0 binary is the
mechanical part. The architectural decisions are:

- Where the binary lives in the workspace, what it depends on, and
  what it does *not* depend on.
- What flags it exposes, and how those flags map to the existing
  kernel-config types (ADR-0011) and parity modes (ADR-0002).
- What its output format guarantees are, in particular how strict
  mode reproduces pycocotools' `summarize()` stdout output (quirks
  L5 / L6 / L7) bit-for-bit.
- What the CLI's relationship is to the Python `patch_pycocotools`
  shim (ADR-0007), which already covers in-process drop-in migration.

The pre-existing alternative — a shell-script wrapper around
`python -m vernier` — is not zero-cost. A Python interpreter on
`PATH`, a virtualenv with the wheel installed, and the cold-start
cost of importing numpy and ndarray-on-PyO3 are real friction for the
non-Python persona this ADR addresses. The CLI exists to make that
persona a first-class user.

This ADR triggers ADR-0001 §"Affect the public API" (the CLI surface
itself), §"Add or remove a build target" (the binary), and §"Add or
remove a top-level dependency" (`clap`). It does **not** cross the
FFI boundary — the binary links `vernier-core` directly and never
touches PyO3 — which is itself a deliberate design choice the
*Considered options* section explores.

## Decision drivers

- **Pure-Rust path.** The CLI's value proposition is "run vernier
  without a Python interpreter." That rules out any architecture in
  which the CLI is a thin wrapper around the wheel.
- **ADR-0002 parity contract.** The CLI is the cleanest surface on
  which to assert byte-equality of the pycocotools summary output:
  shell stdout in, `diff` for verification. Strict-mode default.
  Other parity modes are explicit opt-in; they are not the binary's
  primary purpose.
- **ADR-0005 invariant.** The CLI cannot change `matching.rs` or
  `accumulate.rs`. It is an orchestration layer above the locked
  spine, identical in shape to how `Evaluator` orchestrates today.
- **ADR-0007 separation of concerns.** `patch_pycocotools` covers
  in-process migration. The CLI is the complement: out-of-process,
  shell-driven, no Python at all. The two surfaces do not overlap;
  the documentation positions them as siblings.
- **ADR-0011 kernel config.** The `--iou-type` flag maps directly to
  `IouKind` variants. The CLI does not invent a parallel string
  literal or enum; it parses into the existing type and dispatches
  through the existing kernel surface.
- **ADR-0012 keypoint defaults.** The kp `--max-dets` default is
  resolved via the kernel-canonical sentinel mechanism, not
  hardcoded in the CLI argument layer. A user who passes
  `--max-dets 1,10,100` for `--iou-type keypoints` gets that exact
  list (with the kp warnings that already fire in `vernier-core`);
  a user who omits `--max-dets` gets `[20]` for kp and `[1,10,100]`
  for det, exactly as the in-process API resolves.
- **ADR-0010 boundary disposition.** The `--iou-type boundary` path
  exposes `--dilation-ratio`, defaults to `0.02` (the bowenc0221
  reference value), and refuses to accept the flag when the IoU type
  is anything else.
- **Structured file output is a v0.2 commitment, not a follow-up.**
  CI pipelines need to parse, share, and store eval results — that
  rules out shipping with text-only at v0.2 and adding a structured
  format in a later release. Both text and JSON formatters ship at
  v0.2, and the surface lets a single invocation emit *multiple*
  formats from one eval run (see *Formatter abstraction* below). A
  team that lands on vernier today must not be forced to
  regex-scrape the text output to populate a results dashboard.
- **Decouple internal representation from formatting.** The eval
  pipeline produces one `Summary`; formatters consume it via a
  `Formatter` trait. New formatters are pure additions — one file,
  one registry entry, no kernel work. A single CLI invocation
  emitting two formats pays the eval cost once and the render cost
  per formatter; we never re-run eval to produce a second output
  shape.
- **No new top-level dep beyond `clap`.** The temptation to add
  `anyhow` for ergonomic errors, `colored` for nicer terminal
  output, `indicatif` for a progress bar, or `tracing` for
  structured logs is real. None of them earn their cost. The CLI
  uses `vernier-core`'s typed errors, prints to stderr without
  ANSI codes by default, has no progress bar (eval is fast enough
  that a 50k-image run completes in seconds), and has no logger.
- **No FFI dep.** The CLI links `vernier-core` and `vernier-mask`
  directly. `vernier-ffi` is not in its dep tree. This keeps the
  binary's compile time short, its size small, and its release
  cycle independent of PyO3 ABI churn.
- **Output stability.** The CLI is a *committed* surface starting at
  v0.2.0. Flag additions are SemVer-minor; flag removals or behavior
  changes are SemVer-major. The strict-mode text output is parity-
  pinned to pycocotools v2.0.11 — changing it requires the same ADR
  bar as bumping the pycocotools pin (ADR-0002 territory).

## Considered options

The CLI has three architectural axes worth considering separately.
The chosen design is one option per axis.

### Axis 1 — Binary location and dependency surface

1. **Workspace member at `crates/vernier-cli/`, links `vernier-core`
   directly.** Pure-Rust binary. No FFI, no Python. The reservation
   crate's `[workspace]` standalone manifest is replaced by a real
   workspace member.
2. **Binary target inside `vernier-core` (`[[bin]]` in
   `crates/vernier-core/Cargo.toml`).** Avoids a new crate. Forces
   every consumer of `vernier-core` (the FFI crate, downstream Rust
   users) to pull in `clap`. Pollutes the library's dep tree for a
   feature most of its users don't want.
3. **Python entry point (`python -m vernier eval`, exposed via
   `pyproject.toml`'s `[project.scripts] vernier-eval`).** Reuses
   the FFI plumbing. Drags a Python interpreter into every CLI
   invocation. Cold-start cost is dominated by `import numpy`. The
   non-Python persona this ADR exists for is not served.
4. **Standalone repo (`vernier-cli` as a separate git project).**
   Free of the workspace's lint and dep policies. Loses the lockstep
   release cadence with `vernier-core`; users who upgrade the wheel
   from 0.2.0 to 0.3.0 would have to remember to upgrade the CLI
   independently.

### Axis 2 — CLI structure (top-level vs. subcommand)

1. **Subcommand:** `vernier eval --gt ... --dt ... --iou-type ...`.
   Leaves room for future verbs (`vernier check`, `vernier diff`,
   `vernier reservations`) without breaking users' muscle memory.
2. **Flat:** `vernier --gt ... --dt ... --iou-type ...`. Shorter
   to type. Forecloses the verb extension point — adding a second
   verb later is a breaking change.

### Axis 3 — Output surface and default format

Both text and JSON formatters ship at v0.2; the axis covers (a) the
flag shape that selects them and (b) which formatter is the no-flag
default. Multi-emit (a single eval producing multiple output formats)
is required by the CI persona and is not negotiable; the question is
how the surface expresses it.

1. **Repeatable `--emit FMT[=PATH]` with text default.** A single
   flag covers single-emit (`vernier eval ...` → text on stdout)
   and multi-emit (`--emit text --emit json=result.json --emit
   junit=result.xml`) with one mental model. Default is text on
   stdout; the `python -c` drop-in story works without any flag
   tuning.
2. **`--format FMT` paired with `--output PATH`.** Familiar
   convention from `git`, `kubectl`, and others. Two flags, but
   forecloses multi-emit — a user who wants text-on-stdout *and*
   JSON-on-file has to invoke the CLI twice (paying eval cost
   twice) or build a follow-up `--emit` flag system later.
3. **`--format FMT,FMT` comma-separated with `--output PATH,PATH`
   parallel array.** Multi-emit in two flags. Order-coupling
   between the two flags is brittle and produces unhelpful errors
   ("--format had 3 entries, --output had 2"); using the same
   index in both arrays for "format → path" pairing is exactly the
   shape `--emit FMT=PATH` expresses without the parallel-array
   bookkeeping.
4. **JSON-by-default with `--emit text` opt-in.** Same flag system
   as option 1 with the default flipped. Trades the `python -c`
   drop-in story for a cleaner first impression for tooling, at
   the cost of surprising every COCO-summary-trained user.
5. **Both formats always to stdout/stderr.** Forces every consumer
   to filter; conflicts with the stdout-is-the-data convention;
   incompatible with `--quiet`. Rejected.

## Decision outcome

The chosen design is **Axis 1 option 1, Axis 2 option 1, Axis 3 option 1**:
a workspace member at `crates/vernier-cli/`, packaging the binary
`vernier`, structured as subcommands (`vernier eval ...` is the only
verb at v0.2; future verbs land additively), with text-by-default
output that reproduces pycocotools' `summarize()` stdout in strict
mode.

The reasoning per axis:

- **Axis 1 (workspace member).** The CLI's whole reason to exist is
  "skip the Python interpreter." A bin target inside `vernier-core`
  pollutes every downstream consumer's dep tree with `clap`; a
  Python entry point reintroduces the cold-start cost the binary
  was meant to avoid; a standalone repo loses the lockstep release
  with the kernel and is a maintenance tax for no clear benefit.
- **Axis 2 (subcommand).** A subcommand structure keeps the door
  open for `vernier check` (a fixture-loading sanity command),
  `vernier diff` (a parity differ between two `Summary` JSON
  outputs), or future streaming/background subcommands. The flat
  alternative would force every future verb to be either a flag
  (`--check`, `--diff`) or a breaking change. `vernier eval` is
  also the canonical incantation in the project plan and the
  forward roadmap (`docs/explanation/possible-extensions.md`'s
  CLI rows).
- **Axis 3 (Option 1: `--emit FMT[=PATH]` repeatable, text default).**
  One mental model covers both the simple case (no flags → text
  on stdout) and the CI multi-emit case (`--emit text --emit
  json=result.json` → eval runs once, both outputs land). The
  `Formatter` trait makes the multi-emit free at the kernel level
  (eval pays once, render pays per formatter). Text default
  preserves the strict-mode byte-equality story: `vernier eval ...
  > out.txt && diff out.txt cocoeval-out.txt` is the canonical
  parity test, not a special incantation. CI pipelines type
  `--emit json=result.json` once in their workflow YAML and never
  again. Option 2's `--format`/`--output` pairing was the prior
  draft of this ADR; it forecloses multi-emit and was rejected
  for that reason. Option 3's parallel-array shape is brittle.
  Option 4 (JSON default) trades the existing-user expectation
  for a marginally cleaner machine-tooling first impression — the
  COCO-summary muscle memory wins by an order of magnitude in
  user count.

### Workspace integration

The reservation crate at `tools/reservations/crates/vernier-cli/`
is *retired*: its `[workspace]` standalone manifest is removed and
its README is deleted. The new `crates/vernier-cli/` is a workspace
member, with `package.name = "vernier-cli"` and `[[bin]] name = "vernier"`.
Workspace `Cargo.toml`'s `members` list grows by one entry. The
reservation tracking doc (`docs/engineering/registry-reservations.md`)
gets a row update marking `vernier-cli` as "promoted to workspace
member at v0.2.0".

The retirement is a deliberate one-way door: once `vernier-cli` is
a workspace member, the placeholder slot on crates.io is consumed
by the real release. Future placeholder reservations under
`tools/reservations/` follow the same lifecycle.

### Crate layout

```
crates/vernier-cli/
├── Cargo.toml             # package, [[bin]], deps
├── src/
│   ├── main.rs            # entrypoint: parse args, dispatch
│   ├── cli.rs             # clap derive structs, validation
│   ├── commands/
│   │   ├── mod.rs
│   │   └── eval.rs        # vernier eval — the only verb at v0.2
│   ├── format/
│   │   ├── mod.rs         # Formatter trait, registry()
│   │   ├── text.rs        # impl Formatter for Text
│   │   └── json.rs        # impl Formatter for Json (schema-versioned)
│   └── error.rs           # CliError + From impls
└── tests/
    └── eval.rs            # assert_cmd integration tests
```

Three structural points:

- **No business logic in the binary.** `vernier-cli` is a thin
  argument-parsing and output-formatting layer over `vernier-core`.
  If logic creeps into `crates/vernier-cli/src/commands/eval.rs` —
  threshold computation, sigma resolution, dataset re-encoding —
  it belongs in `vernier-core`, exactly as `vernier-ffi` is held
  to the same line in CLAUDE.md.
- **Format modules own the public-output contract.** Each module
  implements the `Formatter` trait pinned in *Formatter abstraction*
  below. `format/text.rs` delegates to `Summary::pretty_lines()`
  (already in `vernier-core`) and adds the trailing newline.
  `format/json.rs` defines a stable schema (versioned per *Formatter:
  JSON* below) — this is *not* `serde_json::to_string(&summary)`
  because `Summary`'s internal layout is not a public API and
  reshaping it for users is the format module's job. New formatters
  drop into this directory as siblings; nothing else moves.
- **`tests/eval.rs` uses `assert_cmd`** to exercise the binary at
  the process boundary: argument validation, exit codes, stdout/
  stderr split, mutually-exclusive flag rejection. Numerical parity
  is asserted from the Python side (see *Parity harness* below) so
  the same fixtures cover both surfaces with one source of truth.

### Surface

```
vernier eval
    --gt PATH                          # required
    --dt PATH                          # required
    --iou-type {bbox,segm,boundary,keypoints}   # required, no default
    [--parity-mode {strict,aligned,corrected}]  # default: strict
    [--max-dets a,b,c]                 # default: kernel-canonical (ADR-0012)
    [--use-cats | --no-use-cats]       # default: --use-cats
    [--dilation-ratio FLOAT]           # default: 0.02; only with --iou-type boundary
    [--sigmas FILE]                    # only with --iou-type keypoints
    [--emit FMT[=PATH]]...             # repeatable; default: --emit text
    [--quiet]                          # suppress stderr progress messages
```

Five non-obvious points about this surface:

- **`--iou-type` is required, no default.** The four IoU types
  produce numerically distinct stats; defaulting to one would silently
  hide a missing flag in CI scripts and produce the wrong number with
  exit code 0. Forcing the explicit choice is the safe default.
  (`patch_pycocotools` does not face this issue because it inherits
  the `iou_type` from the user's `COCOeval(...)` call.)
- **`--parity-mode strict` default.** The CLI's role as a parity
  oracle ranks above its role as an opinionated fixer. Users who
  want corrected behavior (the in-process default) are presumed to
  be the in-process audience and have `Evaluator(..., parity_mode=
  "corrected")` available. Out-of-process users — CI gates, replay
  pipelines, comparison harnesses — almost universally want strict.
- **`--max-dets` resolution.** ADR-0012 made the kp default
  kernel-canonical. The CLI honors that: with `--iou-type keypoints`
  and no `--max-dets`, the value `[20]` is produced by `vernier-core`'s
  default-resolution path, not hardcoded in `cli.rs`. Symmetrically,
  `--iou-type bbox` with no `--max-dets` resolves to `[1, 10, 100]`.
  The ADR pin matters: future kernel-default changes propagate to
  the CLI without touching the argument layer.
- **`--dilation-ratio` and `--sigmas` are kind-coupled.** Passing
  `--dilation-ratio` with `--iou-type bbox` is rejected at parse
  time, not silently ignored. Same for `--sigmas` with non-keypoints
  IoU types. The validation lives in `cli.rs`'s `validate()` step,
  runs before any kernel call, and surfaces with exit code 2
  (clap's argument-error code) so shell scripts can distinguish
  "bad invocation" from "eval ran but failed."
- **`--quiet` is stderr-only.** Stdout always carries the summary.
  `--quiet` suppresses the (currently-empty, but reserved) stderr
  diagnostic stream. We do not ship a `--verbose` flag at v0.2;
  if structured logs become a real need, that lands behind a
  follow-up ADR adding `tracing` as a top-level dep.
- **`--emit FMT[=PATH]` is the only output flag and is
  repeatable.** A single invocation can emit text, JSON, and
  future formats from one eval run by listing `--emit` multiple
  times. `FMT` is a registered formatter name (`text`, `json` at
  v0.2). `=PATH` selects a destination; omitting it writes to
  stdout. At most one `--emit` may target stdout (otherwise the
  outputs would interleave in the byte stream); duplicate file
  paths are also rejected. Default if no `--emit` is provided:
  `--emit text` on stdout. The older `--format`/`--output`
  pairing was considered and rejected — see *Formatter
  abstraction* below.

### Formatter abstraction

The CLI runs eval exactly once per invocation and consumes the
resulting `Summary` from N formatters. The trait shape:

```rust
// crates/vernier-cli/src/format/mod.rs

pub trait Formatter: Send + Sync {
    /// Stable identifier exposed on the --emit flag and in tests.
    /// Lowercase, kebab-case, never renamed once shipped.
    const NAME: &'static str;

    /// Render the summary into the writer. Borrowing `&Summary`
    /// (not consuming) is what makes multi-emit free.
    fn render(
        &self,
        summary: &Summary,
        ctx: &FormatContext,
        out: &mut dyn io::Write,
    ) -> Result<(), CliError>;
}

pub struct FormatContext<'a> {
    pub iou_type: IouKind,
    pub parity_mode: ParityMode,
    pub max_dets: &'a [usize],
    pub use_cats: bool,
    // Format-specific knobs (e.g. JSON schema-version pin) live
    // on the formatter struct itself, not in the shared context.
}

pub fn registry() -> &'static [&'static dyn Formatter] {
    &[&format::text::Text, &format::json::Json]
}
```

Dispatch is a flat loop:

```rust
let summary = run_eval(&cfg)?;          // exactly once
for emit in cli.emits {
    let formatter = registry::lookup(&emit.format)?;
    let mut writer = open_writer(&emit.path)?;
    formatter.render(&summary, &ctx, &mut writer)?;
}
```

Three load-bearing properties of this shape:

- **Eval cost is amortized across all emits.** Adding a second
  `--emit` line is free at the kernel level — the matching engine,
  the accumulator, and the summarizer all run exactly once. Only
  the per-formatter render cost (JSON serialization, line
  formatting) scales with emit count, and that cost is
  small-constant compared to eval.
- **Adding a formatter is a one-file change.** A new format —
  NDJSON, JUnit, Parquet, the user's preferred internal shape —
  lands as a new module under `crates/vernier-cli/src/format/`,
  registered in `registry()`, exposed via the clap-derive enum
  `FormatName`. No edits to `eval.rs`, no plumbing through the
  argument layer, no kernel changes. The `Formatter` trait is
  what locks this property in.
- **The `Summary` type is the contract surface, not any
  particular wire format.** `Summary` lives in `vernier-core` and
  is shared with the in-process Python API. Formatter modules
  encode it for transport; the in-memory representation is what
  the rest of the codebase reasons about. We do not let JSON
  shape leak into `Summary`'s field layout, and we do not let
  pycocotools' text shape leak into `Summary` either — both are
  formatter-side concerns.

The earlier draft of this ADR proposed `--format FMT` paired with
`--output PATH`. That surface forecloses multi-emit: a user who
wants both text on stdout *and* JSON in a file has to invoke the
CLI twice and pay the eval cost twice. `--emit FMT[=PATH]` (which
this section pins) supports the single-output simple case
(`--emit json=result.json` or, equivalently, the no-flag default
`--emit text`) *and* the multi-output CI case (`--emit text --emit
json=result.json --emit junit=result.xml`) without separate
mental models. The cost is one slightly-novel flag instead of the
familiar `--format`/`--output` pair; the benefit is multi-emit
correctness by construction.

Per-formatter knobs (JSON schema version, future Parquet
compression codec, etc.) live on the formatter's struct via a
`#[command(flatten)]`-style sub-argument group on the clap parser
or via an opaque `--emit json[=PATH][,version=2]`-style key=value
suffix. The exact spelling is an implementation-PR question; the
ADR commits to per-formatter knobs being possible without
overloading global flags.

### Formatter: text (default)

The text format is byte-identical to `Summary::pretty_lines()`
joined by `'\n'` and terminated with `'\n'`. In strict mode, this
matches pycocotools' `summarize()` stdout output bit-for-bit
(modulo the trailing newline, which pycocotools also emits via
`print()`).

The fixture `tests/python/parity/test_cli.py::test_strict_text_matches_pycocotools_stdout`
asserts this byte-equality across `ALL_FIXTURES` by running the
CLI as a subprocess and capturing pycocotools' `summarize()`
stdout via `contextlib.redirect_stdout` (the harness already does
this). This pins quirks **L5 / L6 / L7** (the pycocotools-print
behavior) at the binary boundary, not just at the in-process
level.

### Formatter: JSON (`--emit json[=PATH]`)

The JSON schema is documented in
`docs/reference/cli-output-schema.md`. Rough shape:

```json
{
  "version": "1",
  "iou_type": "bbox",
  "parity_mode": "strict",
  "max_dets": [1, 10, 100],
  "use_cats": true,
  "lines": [
    {
      "metric": "AP",
      "iou_threshold": null,
      "iou_threshold_label": "0.50:0.95",
      "area": "all",
      "max_dets": 100,
      "value": 0.527
    },
    ...
  ],
  "stats": [0.527, 0.728, ...]
}
```

The `version` field is the *schema* version, not the vernier
version. v0.2's CLI ships `"version": "1"`. Schema changes that
add fields are backward-compatible (`"version": "1"` still parses
forward); schema changes that rename or remove fields bump to
`"version": "2"` and the CLI's `--format json` defaults stay on
`"version": "1"` until the next major release. The CLI exposes
`--json-schema-version 2` opt-in for users who want the new shape
before the major bump.

`stats` is duplicated alongside `lines` because tools that already
consume the pycocotools `stats` array (12-element det / 10-element
kp) get a one-line port: `summary["stats"][0]` is `AP`. We pay
the small redundancy to make migration trivial; we document the
relationship between `stats[i]` and `lines[i]` in
`docs/reference/coco-summary-stats.md`.

### Output determinism

The CI use case "store the eval result in a regression-tracking
bucket and `git diff` two runs" requires that the output file is
byte-deterministic across runs of the same input. The CLI commits
to:

- **Stable key order in JSON output.** Object keys are emitted in a
  fixed schema-defined order (the order shown in *Output: JSON
  format* above), not insertion order. The `lines` array is in
  plan order (the same order `Summary::pretty_lines()` produces),
  not sorted by metric name.
- **Pinned float formatting.** `value` fields render via Rust's
  default `{}` for `f64`, which is the round-trip-safe shortest
  representation; this is stable across Rust toolchain versions
  back to 1.83 (the workspace MSRV) and across platforms. The
  text format uses the existing `Summary::pretty_lines()` `{:0.3}`
  format, which is what pycocotools emits.
- **No timestamps.** The CLI does not embed a "generated at" field
  in the JSON output. The exact eval input (GT / DT / flags) is
  the identity of the result; a timestamp would silently change
  the bytes on every run and defeat archiving.
- **No environment leakage.** No host, user, working directory,
  or vernier-build-metadata fields. The version-of-vernier that
  produced the result is the contract surface (it lives in the
  schema as `"version": "1"`); the *commit* of vernier that
  produced it lives in the `cargo install`'s `Cargo.lock` /
  release tag, not in the output file.
- **Atomic file writes.** `--output PATH` writes to
  `PATH.tmp.<pid>`, fsyncs, and renames atomically. A reader
  running concurrently with the writer either sees the previous
  contents or the new contents in full, never a half-written
  file. This matters for CI pipelines where one job writes the
  result and another consumes it via a shared filesystem.
- **Parent-directory creation is opt-in.** `--output
  ./does/not/exist/result.json` fails with exit code 1 unless
  the user passes `--mkdir` (or its equivalent — final flag name
  TBD in the implementation PR). We do not silently create paths.

The result: byte-equal output for byte-equal input, across runs,
machines, and elapsed time — with one well-defined exception
(the schema version field, which only changes on schema
revisions). A user who pins `vernier-cli == 0.2.x` gets stable
output bytes for stable input bytes for the duration of the
0.2 series.

### Format alternatives considered (and deferred)

The "JSON or better" bar deserves a direct answer. The CLI ships
JSON at v0.2 and explicitly defers the following alternatives:

- **NDJSON / JSON Lines.** Right format for an event stream, wrong
  format for a single summary document. A vernier eval produces
  one summary per invocation; wrapping it in an NDJSON envelope
  costs `jq` users a `head -1` they shouldn't have to type. If
  the streaming evaluator (ADR-0013) ever grows a CLI front-end
  with mid-stream snapshots, NDJSON is the right format for *that*
  stream — and the schema's per-record shape is what
  `--format json` already produces, so the transition is a wrapper.
- **Parquet / Arrow.** Right format for *aggregating many runs*
  into a regression-tracking dataset. Wrong format for a single-
  summary CLI invocation: it pulls in `arrow` / `parquet` (large
  deps), produces a binary blob `jq` can't read, and the
  per-record overhead is wasteful for a 12-row table. The right
  shape is a follow-up tool (`vernier aggregate results-*.json
  --output history.parquet`), not the v0.2 CLI's primary output
  format. ADR-0001 §"Add or remove a top-level dependency" gates
  the `arrow` / `parquet` decision when that follow-up arrives.
- **JUnit XML / TAP.** Right format for *test reports* with a
  binary pass/fail axis; wrong format for a stats summary. CI
  systems consume JUnit XML to populate "X tests passed, Y
  failed" UIs. A COCO summary is a 12-row table of floats with
  no pass/fail semantics — fitting it into JUnit's
  `<testsuite>/<testcase>` shape would force every consumer to
  pick a threshold per metric and synthesize a status, which the
  CLI is not the right place to do. Users who want JUnit-flavored
  output for a CI dashboard build it on top of `--format json`
  with their own threshold policy.
- **YAML.** Marginally more human-readable than JSON, materially
  harder to parse from shell, and pulls in a YAML serde dep.
  Loses on every axis that matters for the CI persona.
- **TOML.** A config format. Not a fit for output.
- **MessagePack / CBOR.** Compact binary encodings of the same
  shape as JSON. Loses `jq`-style ad-hoc inspection, which is the
  CI persona's primary consumption mode. If someone has a real
  use case where the JSON size is the bottleneck, that's a
  follow-up flag — not a v0.2 default.

The pattern across these: each alternative has a *real* use case,
none of them are a better fit than JSON for the single-eval
single-document CLI shape this ADR ships. Aggregation tooling,
streaming front-ends, and binary-encoded outputs are follow-ups
that compose on top of the JSON surface; v0.2 commits to JSON as
the structured output, with the schema versioned to make later
breaking changes possible.

Critically, none of these follow-ups requires re-architecting the
CLI. Adding `--emit ndjson=stream.ndjson` or `--emit
junit=report.xml` is a new module under
`crates/vernier-cli/src/format/`, a one-line `registry()` entry,
and a clap-derive enum addition — nothing more. The eval pipeline
does not care which formatters consume its output. This is the
load-bearing property of the `Formatter` trait: format additions
are SemVer-minor, never SemVer-major, and never require an ADR
unless they pull in a new top-level dep (Parquet, MessagePack)
or change the kernel surface (NDJSON streaming).

### Exit codes

- **0** — eval ran, summary written.
- **1** — eval ran but failed (e.g., parity-mode constraint violation
  in `vernier-core`, JSON parse failure of GT/DT, dataset/detection
  schema mismatch, sigmas file invalid).
- **2** — argument parse / validation failure (clap's default).

The split matters for CI: `if vernier eval ...; then ...` distinguishes
"the binary was misused" (2, fix the script) from "the eval refused"
(1, look at the error message). Shell pipelines that pipe the JSON
output to `jq` get a clean signal: a 1-or-2 exit short-circuits the
pipeline, never producing stdout that `jq` would then misinterpret.

### Stdout / stderr split

- **stdout** carries the summary text or JSON, exclusively.
- **stderr** carries diagnostic messages — argument validation
  errors, eval errors with their typed message, the (currently-empty)
  progress stream.
- Errors that happen before any stdout output is produced exit
  cleanly with the appropriate code.
- Errors that happen mid-output (e.g., a stdout-write failure
  because the user piped to a closed FD) exit with 1 silently;
  the user already saw the broken pipe.

This is the standard Unix discipline. The CLI does not deviate.

### Distribution

- **crates.io.** `cargo install vernier-cli` is the supported install
  for users with a Rust toolchain. The package is published from the
  same workspace and tracks the workspace `version` field, so
  `vernier == 0.2.0` and `vernier-cli == 0.2.0` are released
  together.
- **Pre-built binaries.** GitHub Releases ship binaries for
  `x86_64-unknown-linux-gnu`, `aarch64-apple-darwin`, and
  `x86_64-pc-windows-msvc`. The release workflow extends the
  existing `cargo-dist`-style matrix (already used for the wheel
  build) — the addition is one job per target running
  `cargo build --release -p vernier-cli` and uploading the resulting
  binary as a release asset, with sha256sum and (later) a GPG
  signature.
- **`cargo binstall` works** because we publish releases with the
  binstall-conventional naming. No explicit binstall metadata is
  needed at v0.2; we add `[package.metadata.binstall]` if and only
  if the default heuristics fail on a target.
- **Homebrew, conda-forge, etc.** are out of scope for v0.2. The
  community can package the binary; we do not commit to maintaining
  those formulae.

### Parity harness

A new file `tests/python/parity/test_cli.py` adds:

- **`test_strict_text_matches_pycocotools_stdout`** — for every
  fixture in `ALL_FIXTURES`, invoke `vernier eval --iou-type X
  --gt fixture/gt.json --dt fixture/dt.json --parity-mode strict`
  as a subprocess, capture stdout (default emit is text). In the
  same test, run pycocotools' `COCOeval(gt, dt, iou).evaluate();
  .accumulate(); .summarize()` and capture its stdout via
  `contextlib.redirect_stdout`. Assert byte-equality.
- **`test_json_round_trip_matches_in_process_summary`** — for the
  same fixtures, invoke `--emit json=PATH` and load the result;
  assert that `summary["stats"]` matches
  `Evaluator(...).evaluate(...).stats` to bit-equality. Also
  assert `summary["version"] == "1"` and the `lines` array length
  matches the IoU type's plan length (12 for det, 10 for kp).
- **`test_multi_emit_runs_eval_once`** — invoke `--emit text
  --emit json=result.json` and assert: stdout matches the
  text-only invocation, `result.json` matches the json-only
  invocation, *and* the wall-clock cost is within a small margin
  of the single-emit cost (formatter render is cheap; the
  margin sanity-checks that we did not accidentally re-run the
  matcher). The margin is a coarse multiplier (e.g. 1.5×, not
  1.05×) because process-startup variance dominates at fixture
  scale; the load-bearing assertion is the byte-equality of both
  outputs against their single-emit references.
- **`test_emit_collision_on_stdout_rejected`** — invoke `--emit
  text --emit json` (both implicit-stdout); assert exit code 2
  and a typed error naming the collision.
- **`test_emit_duplicate_path_rejected`** — invoke `--emit
  text=out.txt --emit json=out.txt`; assert exit code 2 and a
  typed error.
- **`test_max_dets_default_resolves_via_kernel`** — invoke
  `--iou-type keypoints` without `--max-dets`; assert
  `summary["max_dets"] == [20]`. Symmetric for bbox/segm/boundary
  resolving to `[1, 10, 100]`. Pins ADR-0012's kernel-canonical
  default at the CLI surface.
- **`test_dilation_ratio_rejected_for_non_boundary`** — invoke
  `--iou-type bbox --dilation-ratio 0.02`; assert exit code 2 and
  a typed error on stderr.
- **`test_sigmas_rejected_for_non_keypoints`** — symmetric.
- **`test_unrecognized_iou_type_is_argument_error`** — invoke
  `--iou-type detection` (not a kind); assert exit code 2 and a
  clap parse error.
- **`test_json_output_is_byte_deterministic`** — run the CLI twice
  with `--format json --output result.json` against the same
  fixture; assert the two output files are byte-equal. Repeat
  across all four IoU types.
- **`test_output_path_writes_atomically`** — pre-populate
  `result.json` with sentinel bytes; run the CLI with
  `--output result.json`; assert the sentinel bytes were never
  visible to a concurrent reader (using a small file-watcher
  helper that records every byte sequence it observed).
- **`test_no_timestamp_in_json_output`** — sanity grep over the
  produced JSON for known timestamp patterns (ISO-8601, Unix
  epoch); assert none are present. Regression guard against an
  inadvertent `chrono::Utc::now()` showing up in a future PR.

Rust-side `tests/eval.rs` (in `crates/vernier-cli/tests/`) covers
the argument layer using `assert_cmd`:

- Mutually-exclusive flag combinations.
- Missing required flags (`--gt`, `--dt`, `--iou-type`).
- Unparseable values (`--max-dets foo`, `--dilation-ratio NaN`).
- `--output PATH` actually writes to the file rather than stdout.
- `--quiet` suppresses the stderr diagnostic stream.
- Exit-code mapping matches the contract above.

The two test layers are deliberately split: Python tests own
numerical parity, Rust tests own argument validation. Neither
duplicates the other.

### Versioning and stability commitments

- **`vernier eval` is committed surface at v0.2.0.** Flag additions
  are SemVer-minor; flag removals or default changes are SemVer-major.
  We do not introduce flags whose default we expect to change in a
  later release; if the future of a flag is unclear, it lives behind
  a `--unstable-X` prefix until the design firms up.
- **Strict-mode text output is parity-pinned to pycocotools v2.0.11.**
  Changing it requires the same bar as bumping the pycocotools pin
  (ADR-0002 territory: a new ADR with disposition update across
  every affected quirk).
- **JSON schema is versioned independently.** `version` field bumps
  on breaking changes. The CLI honors the historical schema until
  the next major release.

### What this ADR explicitly does *not* decide

- **A streaming subcommand.** `StreamingEvaluator` (ADR-0013) and
  `BackgroundEvaluator` (ADR-0014) are Python-only at v0.2. A
  `vernier eval --stream` flag (or a `vernier stream` subcommand)
  is a follow-up ADR; the chosen subcommand structure leaves the
  extension point open.
- **A `--threads N` flag.** ADR-0006 commits to single-threaded
  compute through Phase 5; the CLI honors that. If the threading
  model changes, the flag lands with the ADR that changes it.
- **Model evaluation, prediction, or training-loop integration.**
  The CLI ingests pre-existing GT/DT JSON files. It is not a
  prediction runner, not a confusion-matrix viewer, not a
  visualization tool. Those are different products.
- **Subcommands beyond `eval`.** `vernier check` (fixture sanity),
  `vernier diff` (Summary JSON differ), and `vernier reservations`
  (workspace tooling) are plausible future verbs. None of them is
  v0.2 work; the subcommand structure exists to keep them cheap to
  add later.
- **Aggregation across many runs.** Parquet / Arrow output for
  building a regression-tracking dataset is a separate tool
  (`vernier aggregate` or external) that consumes the v0.2 JSON
  output. Discussed in *Format alternatives considered*; not on
  the v0.2 menu.
- **NDJSON streaming output.** The right format for the streaming
  evaluator's CLI front-end (when ADR-0013 grows one) and not for
  the single-shot `vernier eval`. Discussed in *Format alternatives
  considered*; the v0.2 JSON shape is the per-snapshot record that
  a future NDJSON envelope will wrap.
- **JUnit XML / TAP output.** A pass/fail wrapper around the JSON
  summary requires a per-metric threshold policy that does not
  belong in the CLI. Users who want it build it on top of
  `--format json`.
- **Color, progress bars, structured logs.** No `colored`,
  `indicatif`, or `tracing` deps at v0.2. If user demand
  materializes, those are individual follow-up ADRs (ADR-0001
  §"Add or remove a top-level dependency").
- **Homebrew / conda-forge / OS package formulae.** Out of scope.
  Community-maintained packaging is welcome; we commit only to
  crates.io and GitHub Releases.

### Consequences

- **Positive.** Pure-Rust users get a parity gate without a Python
  install. CI scripts shrink from a `python -c` 30-liner to
  `vernier eval --gt foo.json --dt bar.json --iou-type bbox`.
  Pipelines that need to parse, share, or store the result do so
  via `--emit json=result.json` from day one — no regex-scraping
  the text output, no waiting for a follow-up release. A single
  invocation can emit text *and* JSON *and* future formats
  (`--emit text --emit json=result.json --emit junit=result.xml`)
  from one eval run; the `Formatter` trait makes that
  multi-emit free at the kernel level. Output is byte-deterministic
  across runs of the same input, so CI archives diff cleanly and
  regression-tracking buckets work out of the box. Strict-mode
  byte-equality with pycocotools' summary text is now asserted at
  the binary boundary, closing the parity loop more tightly than
  the wheel-only harness can. The reservation crate retires
  cleanly. ADR-0013/0014's streaming work does not need a CLI
  surface yet — the extension point pre-exists. The CLI's
  pure-Rust dep tree stays small, compiles fast, and is
  independent of PyO3 ABI churn. New output formats land
  additively as future follow-ups without new ADRs (unless they
  introduce a top-level dep), because the formatter trait is the
  extension seam.
- **Negative.** A new build target adds a maintenance line: the
  release matrix grows by three targets, binary signing becomes a
  topic the project has to take a position on, and `cargo install
  vernier-cli` becomes a support surface. `clap` is a top-level dep
  we don't get to walk back without a major-version flag-shape
  change. The CLI's flag surface is a public-API commitment from
  v0.2 forward; we lose the freedom to rename or drop flags
  without a release-engineering cost. The JSON schema is a second
  public-API commitment — once `"version": "1"` is in the wild, we
  cannot reshape it without bumping schema-version (and
  maintaining the older shape for the major-release window). The
  output-determinism guarantees (sorted keys, no timestamps,
  atomic writes) are durable contracts; relaxing any of them
  later breaks the regression-archive use case the v0.2 commitment
  is built around.
- **Neutral.** `vernier eval` and the in-process `Evaluator` cover
  overlapping territory for the in-Python persona — both produce a
  `Summary` from GT/DT. Documentation positions the CLI as the
  parity-gate / non-Python path and `Evaluator` as the in-Python
  path; users are told to pick whichever fits their environment,
  not both. The `--format json` schema is a new contract surface
  the project has to maintain alongside the in-process `Summary`
  shape; the schema-versioning rule keeps that cost bounded.

## Pros and cons of the options

### Axis 1 — Binary location

**Option 1 (chosen) — workspace member at `crates/vernier-cli/`**
- 👍 No FFI in the dep tree. Compile fast, ship small.
- 👍 Lockstep release with `vernier-core`; no version-skew between
  the wheel and the binary.
- 👍 Workspace lints, deps, and policies apply uniformly.
- 👎 New crate to maintain; release matrix grows.

**Option 2 — bin target inside `vernier-core`**
- 👍 Zero new crates.
- 👎 Forces every `vernier-core` consumer to pull in `clap`, even the
  FFI crate that has no use for it. Pollutes the library's dep tree
  for every consumer downstream of crates.io.

**Option 3 — Python entry point**
- 👍 Reuses the wheel's plumbing.
- 👎 Cold-start cost dominated by `import numpy`. The non-Python
  persona is not served. Defeats the ADR's reason to exist.

**Option 4 — standalone repo**
- 👍 No workspace coupling.
- 👎 Loses lockstep release with the kernel; users have to remember
  two version numbers; the CI matrix is owned by a second repo. No
  upside that compensates.

### Axis 2 — CLI structure

**Option 1 (chosen) — `vernier eval ...`**
- 👍 Verb-extensible without breaking changes.
- 👍 Matches the project plan and the forward roadmap.
- 👎 Slightly more to type than the flat alternative.

**Option 2 — flat `vernier --gt ...`**
- 👍 Marginally shorter.
- 👎 Adding a second verb later is a breaking change; we'd be
  forced to either add `--check`/`--diff` flags or release v1.0
  to introduce a real subcommand.

### Axis 3 — Output surface and default format

**Option 1 (chosen) — `--emit FMT[=PATH]` repeatable, text default**
- 👍 Single mental model spans simple-case and multi-emit case.
- 👍 Multi-emit pays the eval cost once, render cost per formatter.
- 👍 Text default preserves the pycocotools-summary parity test as
  the no-flag canonical incantation.
- 👍 The `Formatter` trait makes adding NDJSON / JUnit / Parquet
  later a one-file change with no kernel impact.
- 👎 `--emit FMT=PATH` is slightly novel compared to the
  `--format`/`--output` pair; users coming from `kubectl` or
  `git` need a moment to learn the shape.

**Option 2 — `--format FMT` paired with `--output PATH`**
- 👍 Familiar shape; zero learning curve.
- 👎 Forecloses multi-emit. A user who wants both text and JSON
  has to invoke twice and pay eval cost twice. Building a
  `--emit` flag system later would be a flag-shape break.

**Option 3 — `--format FMT,FMT` and `--output PATH,PATH` parallel arrays**
- 👍 Multi-emit in two flags.
- 👎 Order coupling between two arrays is brittle; produces
  unhelpful errors when lengths disagree. Just tag the format
  and path together (`FMT=PATH`) and skip the array bookkeeping.

**Option 4 — JSON default, text opt-in**
- 👍 Machine-friendly out of the box.
- 👎 First-time users get a JSON blob instead of the COCO table
  they expected. Loses the `python -c` drop-in story.

**Option 5 — both, on stdout/stderr**
- 👍 No flag tuning required.
- 👎 Conflicts with the stdout-is-the-data convention. Forces
  every consumer to filter. Mixed-output CLIs are notoriously
  hard to compose. Incompatible with `--quiet`.

## Links and references

- ADR-0001 — Record architecture decisions (§"Affect the public API",
  §"Add or remove a top-level dependency", §"Add or remove a build
  target"). The CLI hits all three.
- ADR-0002 — Three-tier parity model. The CLI's strict-mode default
  is what makes byte-equality with pycocotools' `summarize()` stdout
  the headline parity test.
- ADR-0005 — Lock the `Similarity` trait and matching-engine API for
  Phases 1–3. The CLI is an orchestration layer above the spine; it
  does not edit `matching.rs` or `accumulate.rs`.
- ADR-0006 — Threading model. The CLI honors the single-threaded
  compute commitment; `--threads` is not on the v0.2 menu.
- ADR-0007 — `patch_pycocotools` policy. The CLI is the complement:
  out-of-process, no Python interpreter, shell-driven. The two
  surfaces are siblings.
- ADR-0010 — Boundary IoU as an isolated subsystem. `--iou-type
  boundary` exposes `--dilation-ratio`; the default `0.02` matches
  bowenc0221.
- ADR-0011 — Discriminated kernel config. `--iou-type` parses into
  `IouKind` directly; no parallel CLI-side enum.
- ADR-0012 — OKS keypoints surface and kernel-canonical max-dets.
  The CLI's `--max-dets` default resolution defers to the kernel,
  inheriting the kp `[20]` / det `[1,10,100]` split without
  hardcoding it in the argument layer.
- `crates/vernier-core/src/summarize.rs` — `Summary::pretty_lines()`
  is the source of truth for the strict-mode text output.
- `tools/reservations/crates/vernier-cli/` — the placeholder crate
  this ADR retires.
- `docs/engineering/registry-reservations.md` — updated to reflect
  the promotion to workspace member.
- `docs/reference/coco-summary-stats.md` — referenced from the JSON
  schema doc to anchor the `stats[i]` ↔ `lines[i]` correspondence.
- `docs/reference/cli-output-schema.md` (new) — versioned JSON
  output schema. Created as part of the ADR's implementation PR.
