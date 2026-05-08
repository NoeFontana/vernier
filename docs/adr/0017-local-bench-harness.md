# ADR-0017: Local bench harness — subprocess-isolated, uv-managed, parity-coupled

- **Status:** accepted
- **Date:** 2026-05-01 (amended 2026-05-06 — workload realism: §"COCO val2017"
  expanded with mask-space jitter; §"Reference-model predictions" promoted from
  out-of-scope to a real subsection; §"Out of scope" line removed accordingly.
  Motivation in `docs/engineering/benchmarking/2026-05-snapshot.md` §"segm —
  perfect-match DT under-stresses matching"; further amended 2026-05-06 —
  ADR-0033 supersedes the streaming / `BackgroundEvaluator` carve-out under
  §"Out of scope".)
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** all contributors

## Context and problem statement

vernier's headline claim is "5–10× faster than pycocotools, faster than
faster-coco-eval, with bit-identical results to the reference oracle." Today
that claim has no supporting infrastructure: latency is reported anecdotally
from `time.perf_counter()` calls in scratch notebooks, and parity is
checked in unit tests that don't run on the same workloads as the timing
work. There is no artifact a contributor can produce on a laptop that
answers, for a given commit, "is vernier faster than the baselines on COCO
val2017, and did the speedup come at the cost of correctness?".

This ADR specifies the harness that produces that artifact. It is
explicitly local-only — there is no CI integration in scope here. Three
pressures shape the design.

1. **Three baselines, each with its own runtime.** vernier is benchmarked
   against pycocotools (the reference oracle), faster-coco-eval (the C++
   incumbent we want to beat), and boundary-iou-api (the only oracle for
   the boundary-IoU surface from ADR-0010). pycocotools and
   faster-coco-eval cannot both be installed in the same Python environment
   without `init_as_pycocotools` monkey-patching the `pycocotools` import
   namespace; boundary-iou-api is itself a pycocotools fork and conflicts
   on import. Whatever isolation strategy we pick has to survive contact
   with three packages that fight over the same module names.

2. **Latency without parity is a footgun.** Per ADR-0001's "Change the
   parity contract" trigger, anything that moves vernier's numerical
   outputs is significant. A "vernier got 12% faster!" report that
   silently broke ADR-0002's strict-tier bit-equality with pycocotools
   is the worst possible outcome — the kind of regression that ships
   before it's noticed. The harness has to make parity a side effect of
   every timing run, not a separate test pass that may or may not run.

3. **Local rigor is hard.** Dev machines have CPU governors that idle,
   thermal envelopes that sag under sustained load, background processes
   that steal cycles, and turbo boost that introduces non-stationary
   variance. A casual harness that prints "vernier: 4.2s" hides all of
   this and produces numbers that look authoritative but aren't. The
   harness has to distinguish "I'm iterating on a kernel and want a
   noisy single-shot answer in 30 seconds" from "I'm stamping a release
   benchmark that goes in the announcement post" without forcing the
   user to run two different tools.

The naive design — "a `bench/` directory with a single Python script that
runs each impl in-process N times and prints a table" — fails all three.
It can't isolate the baselines, has no plumbing for parity, and bakes one
rigor level into the script. We need a deliberate orchestration shape, and
this ADR defines it.

## Decision drivers

- **ADR-0001 triggers fired.** The harness affects the public CLI
  (`vernier-bench`), introduces a build target (`bench/` workspace),
  and adds a supported platform stance (Linux-only — see Axis G).
- **ADR-0002 parity contract.** The harness is the operational venue
  where the three-tier model is most visible: every release run is
  also a parity regression test on the strict tier (vs pycocotools)
  and the aligned tier (vs faster-coco-eval), plus a separate
  boundary parity check (vs boundary-iou-api, isolated per ADR-0010).
- **No new top-level *runtime* deps.** Per ADR-0001 §"Add or remove a
  top-level dependency". The harness sits in a separate `bench/`
  workspace with its own dependencies; nothing here propagates into
  `vernier-core` or the `vernier` Python package. uv, samply, py-spy,
  and perf are *operator* tools, not vernier deps.
- **No CI in scope.** The user explicitly defers CI integration to
  spare cost. Nothing here forecloses CI integration; the JSON result
  format and the exit-code contract are designed so a future GitHub
  Actions job is a wrapper, not a rewrite.
- **Reproducibility on a single machine over time.** The headline
  longitudinal use case is "did my commit make things faster or
  slower than yesterday's commit on the same machine." Cross-machine
  comparisons are out of scope; the result schema scopes results by
  machine fingerprint and we accept the consequence (you cannot
  meaningfully compare your laptop to your colleague's workstation).

## Considered options

The harness has seven orthogonal axes. Each axis is decided independently;
the chosen design is the combination of one option per axis. This shape
follows the precedent of ADR-0014, where multi-axis decomposition kept
the rationale honest.

### Axis A — Process isolation between implementations

1. **In-process via `importlib` reload tricks.** Run all impls in one
   Python process, manipulate `sys.modules` between runs to swap which
   `pycocotools` is active. Avoids subprocess overhead.
2. **Subprocess per implementation per rep.** Spawn a fresh Python
   interpreter for each `(impl, rep)` cell. Each interpreter sees one
   pycocotools-flavored package and nothing else.
3. **Separate Docker containers per impl.** Hermetic, also handles
   system library conflicts. Heavyweight per-run cost.

### Axis B — Environment management for the baselines

1. **System pip with extras.** `pip install pycocotools faster-coco-eval
   boundary-iou-api` in the same env, manage conflicts manually. Doesn't
   work — see Axis A1's failure mode.
2. **One uv-managed venv per impl, lockfiles checked in.** Four envs
   (`pycocotools`, `faster-coco-eval`, `boundary-iou-api`, `vernier`),
   each with its own `pyproject.toml` and `uv.lock`. Operator runs `uv
   sync` once.
3. **conda envs.** Same shape as B2 but conda. Heavier toolchain;
   solves problems we don't have (no non-PyPI C++ deps in scope).

### Axis C — Run mode separation (dev vs release)

1. **Two binaries** (`vernier-bench-dev`, `vernier-bench-release`).
   Different defaults baked in. Risks divergence as one binary's logic
   evolves and the other's doesn't.
2. **One binary, one `--mode {dev,release}` flag.** Mode selects a
   profile of `--reps`, warmup count, governor checks, IQR gate. The
   flag is the only difference; the code path is shared.
3. **One binary, fully-orthogonal flags** (no mode at all). User
   composes `--reps 10 --warmup 2 --check-governor`. Maximally
   flexible; no defaults to disagree about; users get it wrong by
   omission.

### Axis D — Parity check coupling

1. **Out-of-band.** Parity tested in `tests/`, latency tested in
   `bench/`, no overlap. Cheap; risks the regression-shipping failure
   mode described in §Context.
2. **Tensor dump + post-hoc compare.** Each runner writes its full
   `(T, R, K, A, M)` precision tensor to disk alongside its timing
   JSON. The orchestrator loads tensors and compares pairwise after
   the timing run completes. Parity is part of every release run.
3. **Summary-stat compare.** Each runner emits `{AP, AP50, AP75, ...}`
   floats; the orchestrator diffs them. Cheaper than D2; loses the
   ability to localize "where did they diverge" to a `(T, R, K, A,
   M)` index.

### Axis E — Latency measurement granularity

1. **End-to-end only.** One number per `(impl, workload, rep)`,
   measured by the orchestrator as parent-side wall-clock around the
   subprocess. No internal stage information.
2. **Stage-level only.** Each runner emits `load`, `evaluate`,
   `accumulate`, `summarize` timings, no end-to-end number. Stages
   don't align across impls — pycocotools' `loadRes` does work that
   vernier defers.
3. **Both, with notes.** End-to-end is the user-perceived figure;
   stage breakdown is best-effort with a `notes` field that documents
   what each impl puts in each stage.

### Axis F — Result storage format

1. **SQLite database.** Indexed query, transactional writes, single
   file. Adds a query language the operator has to learn for ad-hoc
   inspection.
2. **JSON files in a directory tree.** Self-describing, diff-friendly,
   `jq`-queryable. Schema evolution by `schema_version` field.
3. **Parquet.** Efficient for large longitudinal queries. Premature
   for our scale; not human-readable in failure mode.

### Axis G — Platform support

1. **Linux only.** `taskset`, `perf`, `cpupower`, `/sys/devices/...`
   for governor inspection — all assumed present. Tighter rigor mode.
2. **Linux + macOS.** Shim governor checks via `pmset`; replace
   `taskset` with macOS-native pinning (effectively impossible
   without `sudo`); replace `perf stat` with `xctrace` or skip.
3. **Cross-platform via abstraction layer.** Most general; most code;
   least value, since the team's bench machines are Linux.

## Decision outcome

The design is the combination **A2 + B2 + C2 + D2 + E3 + F2 + G1**:
subprocess-per-impl-per-rep, uv-managed envs with checked-in lockfiles,
single binary with `--mode`, post-hoc precision-tensor parity comparison,
end-to-end + stage timings, JSON result store, Linux-only.

Per axis:

- **A2 (subprocess per impl per rep).** The load-bearing decision. The
  three baselines structurally cannot coexist in one Python process —
  faster-coco-eval's `init_as_pycocotools` and boundary-iou-api's
  fork-of-pycocotools both expect to *be* `pycocotools` at import
  time. In-process reload tricks (A1) work in isolation but are
  fragile under refactoring and produce timing artifacts from cold
  caches in the freshly-loaded extension modules. Subprocess
  isolation also gives us peak-RSS for free via `os.wait4()`'s
  `ru_maxrss` field — no sampling, no instrumentation perturbing the
  measurement, no race. Spawn cost is ~100 ms per rep, well below
  the noise floor of any workload we benchmark.

- **B2 (uv envs, lockfiles checked in).** Four envs at
  `bench/envs/{pycocotools,faster-coco-eval,boundary-iou-api,vernier}/`,
  each with `pyproject.toml` and `uv.lock`. Operator runs `uv sync` in
  each on first use; subsequent runs are zero-cost. uv's resolver is
  fast enough that re-syncing on baseline version bumps is not a
  ceremony. The vernier env points at a locally-built wheel via a path
  dependency — `pyproject.toml`'s `[tool.uv.sources]` syntax handles
  this cleanly. conda (B3) solves problems we don't have; system pip
  (B1) doesn't work.

- **C2 (one binary, `--mode {dev,release,profile}`).** Two binaries
  drift; fully-orthogonal flags hide footguns in defaults.
  `--mode dev` is N=1, no warmup, no machine-state checks, target
  sub-30s on a laptop for the bbox cell. `--mode release` is N=10,
  two warmup reps discarded, randomized impl order across reps,
  governor check, IQR-relative-to-median gate at 5%. `--mode profile`
  is N=1 + tool-wrap (samply / py-spy / perf), parity skipped. Same
  binary, same orchestration, three sets of defaults.

- **D2 (tensor dump + post-hoc compare).** Each runner writes its
  full precision tensor as `.npy` and references it by relative path
  in the timing JSON. The orchestrator loads tensors after the timing
  run completes and runs the three-tier comparison from ADR-0002.
  Failure aborts in release mode (and writes a structured divergence
  report); failure warns in dev mode (so a kernel-in-progress doesn't
  block the inner loop). Summary-stat compare (D3) loses the ability
  to localize divergence to a `(T, R, K, A, M)` index, which is
  exactly the locator that turned an early-2026 strict-tier
  regression from a two-day debugging session into a 20-minute one.

- **E3 (end-to-end + best-effort stages).** End-to-end is the
  user-perceived number and the headline. Stage timings are
  best-effort with notes — pycocotools' `loadRes` decodes RLEs and
  builds dt-id mappings that vernier folds into the matching pass;
  declaring those "the same stage" would lie. Each runner declares
  what work happens in each stage via a `notes: [...]` array; the
  report renders those as footnotes on the comparison table. Don't
  paper over the misalignment.

- **F2 (JSON files in a tree, `schema_version` per file).** The
  result store is `bench/results/<git-sha>/<machine-fp>/<workload>/<iou>/<impl>.{json,npy}`.
  JSON is grep-friendly, diff-friendly, `jq`-queryable, and survives
  the operator deciding to delete a single result file by hand
  without tooling. Schema migrations live in
  `bench/harness/migrations/` and run lazily on read. The discipline
  is: never break old result files; always add fields, never
  repurpose them. SQLite is overkill at our expected scale (tens of
  thousands of result files over the project's lifetime, easily
  handled by directory walk + JSON parse).

- **G1 (Linux only).** The team's bench machines are Linux. macOS
  governor inspection requires `pmset` and produces a different set
  of states; CPU pinning without `sudo` is effectively impossible;
  `perf stat` has no macOS equivalent of equivalent fidelity. The
  10–15% extra code to support macOS buys us nothing the team needs
  this quarter, and writing it speculatively means it's untested
  when someone tries to use it. Linux-only also lets us assume
  `/proc/cpuinfo`, `/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor`,
  `taskset`, `numactl`, and a subset of `perf` exist. The harness
  emits a clear error if it's run on a non-Linux platform; we
  expand later if real demand appears.

### Repo layout

```
bench/
├── pyproject.toml                 # bench harness package, console script
├── uv.lock                        # locked harness deps (click, polars, etc.)
├── bench/
│   ├── __main__.py                # `python -m bench` -> CLI entry
│   ├── cli.py                     # click command tree
│   ├── envs/                      # per-impl uv envs (operator runs `uv sync`)
│   │   ├── pycocotools/{pyproject.toml, uv.lock}
│   │   ├── faster-coco-eval/{pyproject.toml, uv.lock}
│   │   ├── boundary-iou-api/{pyproject.toml, uv.lock}
│   │   └── vernier/{pyproject.toml, uv.lock}
│   ├── workloads/
│   │   ├── coco_val2017.py        # download + sha256-verified cache
│   │   ├── jittered_predictions.py# deterministic DT generator
│   │   └── synthetic.py           # parametric stress-test generator
│   ├── runners/                   # one per impl; identical CLI
│   │   ├── _protocol.py           # contract + JSON schema constants
│   │   ├── pycocotools_runner.py
│   │   ├── faster_coco_eval_runner.py
│   │   ├── boundary_iou_runner.py
│   │   └── vernier_runner.py
│   ├── harness/
│   │   ├── orchestrate.py         # subprocess fan-out, rusage capture
│   │   ├── parity.py              # 3-tier comparison, divergence report
│   │   ├── machine.py             # host fingerprint, governor checks
│   │   ├── stats.py               # median, IQR, outlier handling
│   │   ├── migrations/            # schema_version migrations (lazy)
│   │   └── timing.py              # stage timer protocol
│   └── reports/
│       ├── compare.py             # cross-impl + cross-commit
│       └── render.py              # markdown / plotly-svg
└── results/                       # gitignored except for a README
    └── <git-sha>/<machine-fp>/<workload>/<iou>/<impl>.{json,npy}
```

`bench/` is a Cargo-workspace-style sibling of `crates/` and
`python/`. It is not part of the published `vernier` Python package
and ships nothing to PyPI. The console-script entry is `vernier-bench`,
installed only into the harness's own venv.

### Runner contract

Every runner has the same CLI:

```
python -m bench.runners.<impl>_runner \
  --gt <path> \
  --dt <path> \
  --iou-type {bbox,segm,keypoints,boundary} \
  --output <result.json> \
  --tensor-output <precision.npy>
```

It produces a JSON file conforming to schema v1:

```json
{
  "schema_version": 1,
  "impl": "vernier",
  "impl_version": "0.4.2+abc123",
  "iou_type": "bbox",
  "workload_id": "coco_val2017_jittered_seed42",
  "stages": {
    "load":       {"wall_ns": 12300000, "notes": ["RLE decoded eagerly"]},
    "evaluate":   {"wall_ns": 870000000, "notes": []},
    "accumulate": {"wall_ns": 1200000, "notes": []},
    "summarize":  {"wall_ns": 9800, "notes": []},
    "total":      {"wall_ns": 884000000, "notes": []}
  },
  "summary_stats": {
    "AP": 0.421, "AP50": 0.612, "AP75": 0.453,
    "AP_small": 0.234, "AP_medium": 0.421, "AP_large": 0.567,
    "AR_1": 0.312, "AR_10": 0.481, "AR_100": 0.534
  },
  "tensor_path": "vernier.npy",
  "tensor_sha256": "8f3b...c2",
  "warnings": []
}
```

The orchestrator wraps the subprocess in `subprocess.Popen` +
`os.wait4()`, captures parent-side wall-clock as a sanity check
against the runner's self-reported `total` (assert agreement to
within 5 ms; a wider gap is a runner bug), reads `ru_maxrss` from
the rusage struct (Linux: KB; convert to bytes at capture time),
and merges everything into the per-rep result.

The runner contract is deliberately small. A new impl is added by
writing one ~150-line module conforming to this CLI. The orchestrator
discovers runners by directory listing of `bench/runners/*_runner.py`
— there is no plugin registration ceremony.

### Workloads

Three families, all reproducible.

**COCO val2017.** GT downloaded once to `~/.cache/vernier-bench/`,
sha256-verified against a hash pinned in `workloads/coco_val2017.py`.
Predictions generated by *jittering GT*. Bbox jitter is deterministic
Gaussian noise on `(x, y, w, h)` with a controlled FP/FN fraction and
confidence drawn from a beta biased toward correct-detection. **Segm
jitter is mask-space, not vertex-space:** for each non-crowd GT, the
generator decodes the polygon (or RLE) to a binary mask, applies
`scipy.ndimage.binary_dilation` and `binary_erosion` with iterations
drawn from a clipped Poisson, translates by integer pixels drawn from
a Gaussian, and re-encodes via pycocotools. Vertex-space jitter would
push self-intersecting polygons through pycocotools'  `rleFrPoly`
path (boundary-iou-quirks H3) — a parity disposition we keep out of
the test distribution. Mask-space dilate/erode also structurally
resembles real Mask R-CNN errors (fattened/thinned/translated) rather
than the unstructured shape vertex Gaussian noise produces. Mask
jitter draws come from an independent SeedSequence side stream so
v1→v2 of the workload is byte-identical on the bbox/score/FP fields
at every seed; only the segm field is new.

Encode/decode go through pycocotools, not vernier-mask, since using
the system under test as the codec for its own test data would be
circular. pycocotools is already pinned at `2.0.11` for the parity
oracle; reusing it for the workload generator costs a ~70 MB
transitive install on the runner envs (acceptable; runner envs never
import the workload code). The seed is part of the workload identity
(`workload_id` includes it), which makes a workload reproducible
across machines. Three reasons this is preferable to real-model
predictions as the *primary* workload (real predictions cover their
own ground in the next subsection):

1. Zero external dependency. No model checkpoint download, no
   inference dependency, no model-weights license question.
2. Tunable difficulty via the jitter parameters. Easy preset
   (most preds match well) for kernel-development inner loops;
   hard preset (low-IoU clusters) for stress-testing the matching
   logic. The hard preset is also the regime where vernier's
   advantage over pycocotools is largest, which we want to
   visualize in the headline plot.
3. **Free parity fuzzer.** Every new seed is a new strict-tier
   comparison against pycocotools across the full `(T, R, K, A, M)`
   accumulator. Adding seeds 43, 44, 45 to the release matrix
   accumulates strict-tier coverage on segm + boundary that we
   don't have time to write by hand.

**Synthetic stress-test.** Same generator parameterized by
`(n_images, n_categories, dt_per_image, gt_per_image, seed)`. The
release-mode matrix runs a ladder (10k / 50k / 100k images, fixed
categories=80, fixed dt_per_image=30) and fits scaling curves. This
is what tells us whether vernier's advantage holds at scale or
collapses past some inflection point — three point estimates from
val2017 / LVIS / a real model don't give you a curve. The synthetic
workload doubles as a free parity fuzzer: every parametric variation
is a new parity test, and varying the seed across release runs
accumulates strict-tier coverage we don't have time to write by hand.

**Reference-model predictions.** Real-model outputs as a workload
family, in addition to (not instead of) the jittered-GT workload.
Two sources, both materializing on disk before the bench harness
ever sees them:

1. **Mask R-CNN R50-FPN (Detectron2 model zoo).** Pre-computed on
   COCO val2017, hosted as a Hugging Face dataset blob, fetched by
   `tools/fetch-real-predictions.sh --maskrcnn` (pinned URL +
   SHA256 in `tools/real_predictions_cache/`). The bench harness
   has no inference dependency — predictions arrive as a JSON file.
2. **rf-detr (Nano + SegNano).** Inferred locally by the TIDE
   validation harness (`tests/python/integration/real_models/tide/`)
   which already depends on the heavy `[real-models]` extra (rfdetr,
   torch, supervision; ~5 GB on first install). The fetch script's
   `--rfdetr {nano,segnano}` flag shells into `uv run --extra
   real-models python -m
   tests.python.integration.real_models.tide._populate_cache`. Same
   cache as `pytest -m real_models`.

Both sources land in `platformdirs.user_cache_dir("vernier") /
"real-models"` (XDG-correct), shared with the TIDE harness.
`bench/bench/workloads/real_predictions.py` is a read-only adapter:
it never invokes the fetch tooling, never runs inference, never
downloads. A missing cache raises a `FileNotFoundError` pointing at
the right populator command. This keeps the bench env light (no
torch, no rfdetr, no detectron2) and surfaces a misconfigured
cache as a hard error rather than a silent zero-detection benchmark.

Workload identifiers encode the version pin so cross-snapshot
comparisons can't silently mix model versions:
`coco_val2017_maskrcnn_r50fpn_d2_v1`,
`coco_val2017_rfdetr_nano_v<rfdetr-pin>`,
`coco_val2017_rfdetr_segnano_v<rfdetr-pin>`. Bumping the Mask R-CNN
prediction blob is an ADR-level decision per the snapshot it
anchors. The rfdetr pin is owned by the TIDE harness's vendoring
policy.

The release-mode parity audit on these cells is the most aggressive
strict-tier check available — ~37 k Mask R-CNN detections including
the full FP tail will surface accumulator bugs the synthetic
workloads miss. Worth running once before any v1.0 cut regardless
of the perf claim.

The runtime matrix:

|                    | bbox | segm | keypoints | boundary |
|--------------------|:----:|:----:|:---------:|:--------:|
| pycocotools        |  ✓   |  ✓   |    ✓      |   —      |
| faster-coco-eval   |  ✓   |  ✓   |    ✓      |   —      |
| boundary-iou-api   |  —   |  —   |    —      |   ✓      |
| vernier            |  ✓   |  ✓   |    ✓      |   ✓      |

Empty cells are honest blanks in the report, not zeros. The boundary
column is parity-checked only against boundary-iou-api per ADR-0010's
isolation rule; the report renders boundary results in a separate
table to make the comparison set unambiguous.

### Run modes

`vernier-bench run --mode dev` is the inner-loop tool. One rep, no
warmup, no machine-state checks. Target: under 30s on a laptop for
COCO val2017 bbox across all four impls. Used while iterating on a
kernel; the operator wants a "did this commit break anything" answer
in the time it takes to read a Slack message.

`vernier-bench run --mode release` is the rigor mode. Two warmup reps
discarded (page cache, JIT, allocator state), 10 measurement reps,
**randomized impl order across reps** (so no impl systematically
benefits or suffers from thermal drift across the run), pre-flight
governor check (must be `performance` on every active core; aborts
if not, with the `cpupower` command to fix it), abort if the
relative IQR exceeds 5% of the median (signal the machine is too
noisy for the result to mean anything; tune the threshold per-machine
in a local config). Same JSON schema as dev mode, just more reps and
more aggregation metadata.

`vernier-bench run --mode profile --profiler {samply,py-spy,perf}`
wraps the runner subprocess in the chosen tool. N=1 by definition
(instrumentation perturbs measurement). Parity check is skipped
(same reason — profiling instrumentation can change which code paths
warm up and we don't want a profile run to fail-loud on a tolerance
that the instrumentation caused). Outputs go to
`bench/profiles/<git-sha>/<machine-fp>/<workload>/<iou>/<impl>/`
and are gitignored. The profiler choice matters: samply for
Rust-friendly stack sampling (vernier's hot loops), py-spy for
Python overhead in the FFI shim, perf for cycle/cache counters
when investigating microarchitectural questions.

The randomized-order trick is cheap and protects against the most
common dev-machine artifact: thermal drift over a 30-minute
benchmark run causes whichever impl runs first to look fastest. We
shuffle the run schedule once at the start; every rep is a draw
from a permutation of `(impl, workload, iou_type)`. The schedule is
deterministic given the run's seed (which we record), so a run can
be replayed identically.

### Parity coupling

After all runners complete for a given `(workload, iou_type, rep)`,
the orchestrator loads every precision tensor and runs three
comparisons:

- **Strict tier** (per ADR-0002): `np.array_equal(vernier, pycocotools)`.
  Required for vernier's drop-in claim. Failure aborts in release
  mode and writes a `divergence_report.json` to the result directory
  containing the divergent `(T, R, K, A, M)` index, the two values,
  and the first 16 bytes of each tensor's hash for triage. Failure
  in dev mode emits a warning and continues, so a kernel-in-progress
  doesn't block the inner loop on a parity gate the operator already
  knows is broken.
- **Aligned tier**: `np.allclose(vernier, faster_coco_eval, atol=4*np.finfo(np.float64).eps)`.
  faster-coco-eval is not bit-identical to pycocotools either —
  different float-summation order in their C++ accumulator — and it's
  worth knowing where on ADR-0004's tolerance ladder each impl sits.
  The 4-ULP bound is the same one the streaming evaluator uses in
  ADR-0013.
- **Boundary tier**: `np.allclose(vernier_boundary, boundary_iou_api, atol=4*ULP)`.
  Separate pair per ADR-0010, since the others don't compute
  boundary IoU. Same divergence-report shape on failure.

The divergence report is the operationally important output. "vernier
got 12% faster but pycocotools-strict-tier failed at index `(7, 0,
22, 1, 2)` with values `0.42312` vs `0.42313`" is a useful bug
report; "vernier got 12% faster but the test failed somewhere" is
not. Every release-mode run is a parity regression test by
construction.

### Result store and longitudinal queries

Path: `bench/results/<git-sha>/<machine-fp>/<workload>/<iou>/<impl>.{json,npy}`.

Machine fingerprint is the first 12 chars of `sha256(cpu_model +
n_cores + total_ram + os_release + glibc_version)`. Stable across
reboots; distinct between dev boxes; short enough to type in a path.
Computed by `bench/harness/machine.py` on every run; the same
function gates "is this result comparable to my previous results on
this machine?" — yes iff fingerprints match.

The result file holds per-rep timings, aggregated stats (median, IQR,
min, max for each stage and total), the implementation version, the
harness version, `schema_version: 1`, and a relative path to the
precision tensor (the tensor lives next to the JSON, not in some
content-addressable store — we don't dedupe across reps because
disagreement between reps' tensors would be its own bug).

Schema migrations live in `bench/harness/migrations/v{N}_to_v{N+1}.py`
and run lazily on read. The discipline is: never break old result
files; always add fields, never repurpose them. The first migration
will probably be adding cycle-count metrics from `perf stat` once we
need them; that change is a v1→v2 migration, not a schema rewrite.

`vernier-bench compare --base <sha> --head <sha>` cross-walks the
store and renders a markdown table of deltas (absolute + relative,
with sign-of-change colored). `vernier-bench report --since 30d`
is the longitudinal view, plotly-svg embedded into a markdown report
file. We do *not* build statistical-significance regression alerts
in v1 — wait until we have ≥30 commits of release-mode data on the
same machine to calibrate false-positive rates, or we'll train
ourselves to ignore alarms. The shape of "did vernier get faster
this quarter" is enough for the first six months.

### Linux-only justification

The harness emits a clear error if invoked on a non-Linux platform:

```
$ vernier-bench run --mode release
ERROR: vernier-bench requires Linux for rigor-mode execution.
       Detected: Darwin 23.4.0 (arm64).
       Reason: governor inspection, taskset, and perf stat have no
       portable equivalents on macOS that meet the harness's
       reproducibility requirements.
       To unblock dev-mode work on macOS, see ADR-0017 §"What this
       ADR explicitly does not decide".
```

Dev mode could in principle run on macOS (it doesn't depend on
governor inspection or `taskset`), but supporting it would mean
maintaining two `machine.py` paths and lying about the absence of
`perf` in the profile mode. The cleaner story is "Linux only,
period" until someone shows up with a workload that requires macOS.

### Test plan

The harness has its own test suite at `bench/tests/`. Highlights:

- **Subprocess isolation contract.** Spawn each baseline runner with a
  trivial workload; assert each completes without `ImportError` from
  pycocotools-namespace contention. The point is to catch a future
  refactor that breaks isolation.
- **Schema round-trip.** Write a v1 result file; read it through every
  migration; assert structural equality with a hand-written v_latest
  fixture. Catches migration bugs.
- **Parity divergence report shape.** Inject a deliberate
  off-by-one-ULP delta into a tensor pair; assert the divergence
  report contains the correct index and values.
- **Run schedule determinism.** Two `release` invocations with the
  same seed produce the same shuffle; two with different seeds don't.
- **Workload sha256 verification.** Corrupt the cached val2017 GT;
  assert the next run re-downloads and re-verifies.
- **Machine fingerprint stability.** Run the fingerprint computation
  100 times; assert the same value every time (catches a
  nondeterministic hash input).

Test execution time target: under 60s on a laptop for the full
harness suite. The harness is not the place to also stress-test
vernier-core; that lives in `crates/vernier-core/tests/`.

### What this ADR explicitly does *not* decide

- **CI integration.** Out of scope per the user's deferral. The
  result-file shape and exit-code contract are designed to be CI-
  friendly (machine-readable JSON, non-zero exit on parity failure
  in release mode) so a future GitHub Actions workflow is a wrapper,
  not a rewrite. A future ADR adds the workflow.
- **Cross-machine result aggregation.** The machine fingerprint
  scopes results to one host. Aggregating across machines requires
  either accepting that you're comparing apples to oranges, or
  building a per-machine normalization (ratio against a calibration
  workload). Both deserve a separate ADR; neither blocks v1.
- **Auto-regression alerts.** Statistical-significance regression
  detection requires ≥30 commits of data to calibrate; building it
  before then produces alarms tuned to nothing.
- **Streaming and BackgroundEvaluator surfaces.** Superseded by
  ADR-0033 (multi-paradigm bench harness extension). ADR-0033 lifts
  the streaming / `BackgroundEvaluator` deferral and folds those
  surfaces into the harness as paradigm-discriminated `Workload`
  variants (`StreamingWorkload`) and a paradigm-segmented result-store
  path. The runner protocol generalization predicted here lands in
  ADR-0033 §"Stage-name conventions per paradigm".
- **macOS support.** Out of scope per Axis G. A future ADR adds it
  if real demand appears, with a corresponding loss of rigor in the
  release-mode checks (or a more complex platform-shim layer).
- **Wheel-size or build-time benchmarks.** Latency only. vernier's
  wheel size is governed by a separate budget in
  `docs/engineering/release-checklist.md`.
- **GPU baselines.** `torchvision.ops.box_iou` and friends are GPU-
  side detection-utility benchmarks, not COCO-eval benchmarks. Out
  of scope.

### Consequences

- **Positive.** Every release-mode benchmark run is a parity
  regression test, and parity failures are localized to a tensor
  index rather than reported as "they differ". Subprocess isolation
  is the only design that makes the four impls coexist; the choice
  also yields free peak-RSS via `wait4(2)` rusage. uv-managed envs
  with checked-in lockfiles make first-time setup a `uv sync` per
  env (under a minute total) and re-runs zero-cost. The result store
  accumulates longitudinal data from day one; the schema migration
  discipline means we don't lose access to old results when the
  schema evolves. Linux-only lets the rigor mode actually be
  rigorous (governor checks, `taskset`, `perf`) instead of
  best-effort across platforms. The harness's own dependencies live
  in their own workspace and do not propagate into vernier — a user
  who installs `vernier` from PyPI does not get `polars` or `click`
  pulled in by transitivity.
- **Negative.** Subprocess spawn is ~100ms per rep, which doesn't
  matter for any individual workload but adds up to ~30 seconds of
  pure spawn overhead across a full release-mode matrix (4 impls × 4
  iou types × 10 reps × 2 workload families = 320 spawns). The
  baselines' env management is real work; faster-coco-eval's C++
  build occasionally breaks on new Python minors and the operator
  re-runs `uv sync` with a pinned older Python. Thermal drift on
  laptops is a real risk for the release mode; the IQR gate
  catches it but produces an aborted run rather than a result, which
  is the right behavior but is operationally annoying when running
  on a thermally-constrained MacBook (which we don't support, but
  the team will try). The full release-mode matrix runs in 30+
  minutes of sustained CPU; this is by design (rigor isn't free)
  but it's not a "run before lunch" tool. Linux-only forecloses
  any contributor on macOS from running release-mode benchmarks
  until a follow-up ADR; dev-mode also doesn't work on macOS by
  the strict reading of Axis G, which we may want to relax once the
  initial release lands.
- **Neutral.** The harness is a separate workspace; it does not
  ship with `vernier`. This is the right call but means contributors
  have to opt into bench tooling explicitly. The runner protocol is
  small and adding a new impl is ~150 lines of Python; this is good
  for extensibility but means the protocol's extension points
  (custom stages, custom per-impl metrics) become a future migration
  problem if we don't think them through up front. The result store
  is grep-friendly but produces tens of thousands of small files
  over the project's lifetime; backing it with SQLite later is a
  pure migration if we need it.

## Pros and cons of the options

### Axis A — Process isolation

**A2 (chosen) — subprocess per impl per rep**
- 👍 Isolates the three pycocotools-namespace-conflicting impls
  cleanly. No `sys.modules` manipulation. No surprise from one
  impl's import side effects leaking into another.
- 👍 Free peak-RSS from `wait4(2)`'s `ru_maxrss`. No sampling, no
  perturbation.
- 👍 Crash isolation: a segfault in one impl's C++ extension does
  not take down the harness.
- 👎 ~100 ms spawn per rep. Real but small.
- 👎 Each subprocess pays the JIT / extension-load cost on cold
  start. Mitigated by the warmup reps in release mode.

**A1 — in-process with `importlib` reload tricks**
- 👍 No spawn cost. Fastest possible inner loop.
- 👎 Fragile under refactoring; future contributors will not know
  the rules of which `sys.modules` keys to scrub.
- 👎 Cold-cache artifacts when reloaded extension modules re-decode
  RLE tables, etc.
- 👎 Crash in one impl takes down the run.

**A3 — Docker per impl**
- 👍 Hermetic to the level of system libraries.
- 👎 Container start/stop is seconds per rep. Order of magnitude
  worse than A2 for no benefit at our scale.
- 👎 Adds a hard dep on Docker for every contributor running
  benchmarks.

### Axis B — Environment management

**B2 (chosen) — uv envs, lockfiles checked in**
- 👍 `uv sync` is fast enough to be invisible. Lockfiles make first-
  time setup deterministic.
- 👍 Path dependency for vernier (`[tool.uv.sources]`) handles the
  locally-built-wheel case cleanly.
- 👎 Operator has to know to re-run `uv sync` after pulling a
  baseline-pin update. Mitigated by the harness checking lockfile
  hash on startup and warning on mismatch.

**B1 — system pip with extras**
- 👍 Zero ceremony.
- 👎 Doesn't work — pycocotools / faster-coco-eval / boundary-iou-api
  conflict at the namespace level.

**B3 — conda envs**
- 👍 Solves the C++ system-library problem if it existed.
- 👎 It doesn't exist for our baselines (all PyPI). conda is heavier
  toolchain for no incremental value.

### Axis C — Run mode separation

**C2 (chosen) — single binary, `--mode {dev,release,profile}`**
- 👍 One code path; defaults differ but logic is shared. No drift.
- 👍 Operator memorizes one command shape; mode is the variable.
- 👎 The mode flag couples three sets of defaults that may pull
  apart over time. Mitigated by the per-mode test in
  `bench/tests/test_modes.py` that asserts each mode's defaults
  match the documented profile.

**C1 — two binaries**
- 👍 Each binary is simpler in isolation.
- 👎 Logic drift; two test surfaces; two doc surfaces.

**C3 — fully orthogonal flags, no mode**
- 👍 Maximally flexible.
- 👎 Operators get rigor wrong by omission. The whole point of
  modes is encoding the right defaults for each use case.

### Axis D — Parity check coupling

**D2 (chosen) — tensor dump + post-hoc compare**
- 👍 Localizes divergence to a `(T, R, K, A, M)` index.
- 👍 Parity is part of every release run by construction.
- 👍 Tensors are cheap to dump (single `.npy` per runner per rep).
- 👎 Doubles result-store disk usage versus summary-stat-only.
  Acceptable; tensors compress well and are pruneable by age.

**D1 — out-of-band**
- 👍 Decouples concerns.
- 👎 Ships the regression-shipping failure mode described in
  §Context. Hard no.

**D3 — summary-stat compare**
- 👍 Tiny artifact size.
- 👎 No localization on failure. "AP differs by 0.0001" tells you
  nothing about which IoU threshold or which class.

### Axis E — Latency measurement granularity

**E3 (chosen) — end-to-end + best-effort stages**
- 👍 End-to-end is the headline; stages are diagnostic. Both.
- 👍 The `notes` field is honest about misalignment.
- 👎 Two numbers per impl per rep is more to render in the report.

**E1 — end-to-end only**
- 👍 Simplest report.
- 👎 No diagnosis when a regression appears. "It got slower" without
  "the slowdown is in `accumulate`" wastes hours per investigation.

**E2 — stages only**
- 👍 Forces stage-level rigor.
- 👎 Stages don't align across impls. The summed stages are not the
  user-perceived total when the impls do work in different stages.

### Axis F — Result storage

**F2 (chosen) — JSON tree, schema_version per file**
- 👍 Self-describing, grep-friendly, `jq`-queryable.
- 👍 Migration discipline is simple to enforce by code review.
- 👎 At 10⁵ result files, directory walks become slow. Premature to
  worry about; a later ADR migrates to SQLite if we hit it.

**F1 — SQLite**
- 👍 Indexed query.
- 👎 Adds a query language. Single corrupted file is harder to
  recover from than a single JSON file.

**F3 — Parquet**
- 👍 Efficient at scale.
- 👎 Not human-readable in failure mode.

### Axis G — Platform support

**G1 (chosen) — Linux only**
- 👍 `taskset`, `perf`, `cpupower`, `/sys/devices/system/cpu/...`
  all assumed present. Rigor mode is meaningfully rigorous.
- 👍 One platform path; less code; tested on the platform we use.
- 👎 Forecloses macOS contributors from running benchmarks. Mitigated
  by the clear error message and the deferred-ADR pointer.

**G2 — Linux + macOS**
- 👍 Wider contributor reach.
- 👎 governor inspection, CPU pinning, perf — all need shims. The
  shims are silently weaker than the Linux versions, so macOS
  results would be less rigorous in ways the operator wouldn't see.

**G3 — fully cross-platform abstraction**
- 👍 Most general.
- 👎 Most code, least value, none of the team's bench machines run
  Windows.

## Links and references

- ADR-0001 — Record architecture decisions (§"Affect the public API",
  §"Add or remove a top-level dependency", §"Add or remove a build
  target").
- ADR-0002 — Three-tier parity model. The strict and aligned tier
  names are reused by this harness's parity check; the boundary
  comparison sits alongside per ADR-0010 rather than as a third tier
  of ADR-0002's model.
- ADR-0004 — Numerical layout policy. The 4-ULP aligned tolerance is
  reused in the parity coupling.
- ADR-0010 — Boundary IoU isolation. The boundary tier of the parity
  check follows from this ADR's isolation rule.
- ADR-0013 — Streaming evaluator. The streaming surface, originally
  out of scope here, is folded into the harness by ADR-0033.
- ADR-0014 — `BackgroundEvaluator`. Same; ADR-0033 lifts the deferral.
- ADR-0033 — Multi-paradigm bench harness extension. Supersedes the
  §"Out of scope" streaming / `BackgroundEvaluator` carve-out and
  extends the harness across panoptic, semantic, and streaming.
- ADR-0015 — `vernier-cli` workspace binary. The bench harness sits
  in a sibling workspace and follows the same packaging pattern.
- `docs/explanation/possible-extensions.md` — the Phase 5 capability
  ranking, against which the headline benchmarks anchor vernier's
  positioning claim.
- [MADR](https://adr.github.io/madr/) — the format this ADR follows.
- [uv](https://docs.astral.sh/uv/) — the package manager used for
  per-impl environments.
- [samply](https://github.com/mstange/samply),
  [py-spy](https://github.com/benfred/py-spy),
  [perf](https://perf.wiki.kernel.org/) — the profilers wrapped by
  `--mode profile`.
