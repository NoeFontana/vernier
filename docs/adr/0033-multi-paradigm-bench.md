# ADR-0033: Extend the bench harness across paradigms (panoptic, semantic, streaming)

- **Status:** accepted
- **Date:** 2026-05-06 (accepted post-Stage-1 integration)
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** all contributors

## Context and problem statement

ADR-0017 specified a subprocess-isolated, parity-coupled local bench
harness for **detection only** — bbox / segm / keypoints / boundary cells
fanned out across pycocotools, faster-coco-eval, boundary-iou-api, and
vernier. Three other v1.0 surfaces ship without a quantitative loop
today: **panoptic** (ADR-0025), **semantic** (ADR-0028), and the
**streaming / distributed** family (ADR-0013, ADR-0014, ADR-0030,
ADR-0032). That is real perf debt: the rayon-style pattern of "the bench
cell sets the optimization target" only works if the cell exists. Three
paradigms with no harness coverage means three surfaces where every perf
claim has to be reasoned about anecdotally.

The existing harness is detection-shaped end-to-end: `Workload` is a
flat `@dataclass(frozen=True)` carrying `gt_path` / `dt_path`; the
result-store path has no paradigm segment; the parity comparator is
`np.array_equal` over a single precision tensor; and `IMPL_IOU_SUPPORT`
is hardcoded for four iou-types of one paradigm. Generalizing all of
that without breaking the live detection cells is the job this ADR
specifies.

The scope here is **Stage 1** — the harness extension plus the
Minimum-Viable-Bench cells for each paradigm. Stage 2 (per-surface
optimization passes that consume these cells as their inner loop) and
Stage 3 (real-prediction cells: Mask2Former for panoptic, OCRNet for
ADE20K) are framed at the end of this ADR but explicitly deferred.

The original Stage 1 plan included a Cityscapes B2 cell as the
semantic MVB. It was dropped post-Stage-1 because Cityscapes' license
restricts redistribution of derivative outputs (the public bench-result
tree is the wrong fit). The semantic paradigm's first concrete cell
moves to S3-B (ADE20K + mmseg).

## Decision drivers

- **Four paradigms, four oracle stacks.** Detection oracles are
  pycocotools / faster-coco-eval / boundary-iou-api. Panoptic oracle is
  `panopticapi.evaluation.pq_compute_single_core` (single-process per
  ADR-0025). Semantic oracle is `mmsegmentation.IoUMetric` for ADE20K
  (S3-B). Streaming has **no external oracle** — its parity contract
  is internal (batch and stream paths must produce the same
  `Summary.stats`).
- **Four fixture shapes.** Detection ships a `(gt.json, dt.json)` pair.
  Panoptic ships `(gt_png_dir, gt_json, dt_png_dir, dt_json,
  categories_json)`. Semantic ships `(gt_label_maps, dt_label_maps,
  n_classes, ignore_label, label_remap)`. Streaming reuses detection
  inputs but adds `iou_type` and a `chunk_schedule`. A single flat
  dataclass with nullable per-paradigm fields would surface every
  field as `None`-or-set everywhere it's read, defeating the point.
- **Per-paradigm parity tier, not one global tier.** Instance keeps the
  three-tier strict / aligned / boundary contract from ADR-0002.
  Panoptic is **strict** vs `pq_compute_single_core(proc_id=0, ...)`
  (pinned by SHA in `crates/vernier-panoptic/src/parity.rs`). Semantic
  ADE20K-vs-mmseg is **aligned** until PR-B6/7/8 vendors mmseg at a
  pinned SHA, at which point it can be re-graded to strict; the tier
  is a per-cell metadata flag, not a code change. Streaming is
  **bit-equal `Summary.stats`** between batch and stream — no oracle,
  but the discipline is just as tight.
- **No cross-paradigm comparison.** Per ADR-0032, a `WireEnvelopeBody`
  is a closed-world tagged union and merging across paradigms is
  structurally rejected. Comparing PQ to AP is a category error in
  exactly the same way: the comparator and the report renderer must
  refuse to do it. `vernier-bench compare` and `vernier-bench report`
  scope per-paradigm; an attempt to compare a PQ cell against an AP
  cell yields an error citing this ADR and ADR-0032.
- **Closed-world variant precedent.** ADR-0029 split the public Python
  surface into per-paradigm submodules (`vernier.panoptic`,
  `vernier.semantic`, etc.). ADR-0032 uses a closed-world
  `WireEnvelopeBody` enum so cross-paradigm merge is a compile-time
  rejection. The bench harness's `Workload` shape should follow the
  same pattern: a discriminated union whose discriminator is the
  paradigm name.

## Considered options

1. **Flat dataclass with `paradigm` enum + nullable per-paradigm
   fields.** One `Workload` carrying every paradigm's fields; the
   non-applicable ones default to `None`. Minimal diff; fast to type.
2. **Tagged union via Pydantic discriminated union.** One
   `Annotated[Union[InstanceWorkload, PanopticWorkload,
   SemanticWorkload, StreamingWorkload], Field(discriminator="paradigm")]`
   alias `Workload`. Each variant carries only its own fields.
3. **Separate harness per paradigm.** `vernier-bench-panoptic`,
   `vernier-bench-semantic`, etc. Each in its own subpackage with its
   own CLI, comparator, and result store.

## Decision outcome

Chosen option: **Option 2 — tagged union via Pydantic discriminated
union**, because it matches the closed-world precedent already set by
ADR-0029 (per-paradigm submodules) and ADR-0032 (`WireEnvelopeBody` as
a discriminated enum), and because Pydantic's discriminator validates
on parse so a malformed `Workload` JSON fails at load time with the
correct variant's error message rather than surfacing as a runtime
`AttributeError` deep inside a runner.

The harness extension lands as seven coupled changes:

1. **Workload tagged union** at `bench/bench/workloads/__init__.py`.
   Discriminator: `paradigm: Literal["instance", "panoptic", "semantic",
   "streaming"]`. Variants:
   - `InstanceWorkload(workload_id, gt_path, dt_path,
     supported_iou_types)` — current detection shape preserved.
   - `PanopticWorkload(workload_id, gt_png_dir, gt_json, dt_png_dir,
     dt_json, categories_json)`.
   - `SemanticWorkload(workload_id, gt_label_maps, dt_label_maps,
     n_classes, ignore_label, label_remap)`.
   - `StreamingWorkload(workload_id, gt_path, dt_path, iou_type,
     chunk_schedule)`.

2. **Paradigm-segmented result-store path.**
   `bench/results/<git_sha>/<machine_fp>/<paradigm>/<workload>/<metric>/<impl>.{json,npy}`.
   `<metric>` is the ex-`<iou>` slot, generalized: instance still uses
   bbox / segm / keypoints / boundary, panoptic uses `pq`, semantic
   uses `miou`, streaming uses `throughput` / `p99` / `rss` / etc. The
   migration from the existing v1 detection-only path is a separate
   PR-A-thick deliverable; A-thin's job is to emit the new path and
   tolerate v1 reads via the schema migration shim.

3. **Schema v2** at `bench/bench/harness/schema.py`. `schema_version`
   bumps from 1 → 2. Adds a required `paradigm: Paradigm` field, and
   generalizes the artifact-handling pair: `tensor_path: str` →
   `artifact_paths: dict[str, str]` and `tensor_sha256: str` →
   `artifact_sha256: dict[str, str]`. Detection runners populate
   `{"tensor": "vernier.npy"}` (and the matching sha map) so the
   detection cells stay shape-stable. Panoptic later populates
   `{"snapshot": "panoptic.json", "per_class": "per_class.npy"}`;
   streaming populates `{"summary": "stats.json", "rss_curve":
   "rss_curve.json"}`. A v1→v2 read-side compat shim lifts v1
   `tensor_path` / `tensor_sha256` into the new dicts and sets
   `paradigm="instance"`.

4. **Comparator registry** at `bench/bench/harness/parity.py`. The
   hardcoded `_TIER_PAIRS` dict is replaced by a registry keyed on
   `paradigm`. The instance comparator stays bit-identical to today
   (strict / aligned / boundary tiers from ADR-0002). The
   `ComparableArtifact` union (Pydantic, JSON-portable) admits four
   variants: `Tensor`, `PanopticSnapshot`, `ConfusionMatrix`,
   `StreamingPair`. Each carries a `to_canonical_form() -> dict` so the
   divergence report can render a stable snapshot of either side
   regardless of paradigm. B1/B2/B3 register the panoptic / semantic /
   streaming comparators; in A-thin they are stubs that raise
   `NotImplementedError` from `compare()`.

5. **CLI `--paradigm` flag** at `bench/bench/cli.py`.
   `vernier-bench run` gains `--paradigm
   {instance,panoptic,semantic,streaming,all}`. Auto-derived from the
   workload's discriminator when unambiguous (no workload name is
   reused across paradigms; explicit override is a future-proofing
   escape hatch). Paradigm/metric mismatches (e.g., `--paradigm
   semantic --iou bbox`) error at parse time.
   `vernier-bench compare` and `vernier-bench report` scope
   per-paradigm: one delta table per paradigm. A cross-paradigm
   comparison is rejected with a message citing ADR-0032.

6. **`IMPL_PARADIGM_SUPPORT` matrix** at `bench/bench/harness/matrix.py`.
   `IMPL_IOU_SUPPORT: dict[Impl, set[IouType]]` becomes
   `IMPL_PARADIGM_SUPPORT: dict[Paradigm, dict[Impl, set[Metric]]]`. The
   instance entry is populated; panoptic / semantic / streaming entries
   are empty for B1/B2/B3 to populate. Env discovery generalizes from
   impl-name (`bench/envs/<impl>/`) to env-name via a new
   `IMPL_TO_ENV_NAME: dict[Impl, str]` (defaults to identity for
   detection impls; `vernier_panoptic` and `panopticapi` will both map
   to `panopticapi` env, etc.).

7. **Stage-name conventions per paradigm.** Detection keeps `load /
   evaluate / accumulate / summarize / total`. Panoptic uses `load /
   match / pq / total`. Semantic uses `load / confusion / iou /
   total`. Streaming uses `load / chunk_<n> / finalize / total` plus a
   parallel `peak_rss_bytes` artifact. Stage keys remain open by
   convention so a runner can split a stage into sub-stages without a
   schema change.

### Consequences

- **Positive.** A single bench harness covers all four paradigms with
  one CLI, one result store, one comparator-registry pattern, one
  longitudinal report. The discriminated-union shape makes it a
  compile/parse-time error to mix paradigm fields incorrectly. The
  Stage 2 optimization passes (rayon AP fold, `pulp::Arch::dispatch`
  confusion-matrix bincount, panoptic PNG-decode hot path, DLPack tax
  close-out) consume these cells directly as optimization targets — no
  more anecdotal `time.perf_counter()` claims. The closed-world variant
  pattern (ADR-0029, ADR-0032) extends naturally; cross-paradigm
  comparison is a structural reject, mirroring ADR-0032's
  cross-paradigm merge reject.
- **Negative.** The flat-dataclass-to-tagged-union migration touches
  every detection runner and the orchestrator (the runners' fields
  move from `wl.gt_path` / `wl.dt_path` / `wl.supported_iou_types` to
  `InstanceWorkload`-only attributes). A-thick must run a one-shot
  result-store migration so the existing detection cells produce
  byte-equal content under the new path. Schema v2 forces a
  migration-shim implementation for v1 reads; A-thin lands the read-
  side shim with a unit test, A-thick lands the write-side migrator.
  ADE20K + mmseg env (~2 GB torch CPU) is a real disk + first-time
  `uv sync` cost; deferred to Stage 3 to avoid weighing down the
  Stage 1 first-time-setup budget.
- **Neutral.** The `IouType` literal stays for instance-paradigm
  callers (the matrix module re-exports it under the more general
  `Metric` alias for the schema). Result-tree depth grows by one
  segment; downstream readers (loaders, report renderers) expand their
  glob patterns by one level. The detection runners gain an `assert
  isinstance(wl, InstanceWorkload)` line at the top — a paradigm-
  specific entry point is the explicit precondition, not a runtime
  surprise.

### Deferred (explicitly not in scope here)

- **Stage 2 optimization passes.** rayon for AP fold,
  `pulp::Arch::dispatch` for confusion-matrix bincount, panoptic
  PNG-decode hot path, DLPack tax close-out. Each consumes its B-cell
  as the optimization target and lives in a surface-specific PR. The
  harness's `compare --base <sha> --head <sha>` is the inner loop for
  these; this ADR does not specify their sequencing.
- **Stage 3 real-prediction cells.** Mask2Former on COCO panoptic val
  and OCRNet on ADE20K val. Workloads ship as docstring TODOs with
  `_URL = None` cache stubs in
  `tools/real_predictions_cache/{panoptic,semantic}.py`; Stage 3 plugs
  in the pinned URLs.
- **mmseg env.** ~2 GB torch CPU. Deferred to S3-B; the semantic
  paradigm has no MVB cell at Stage 1 (Cityscapes was dropped per the
  license caveat).
- **Boundary-PQ (ADR-0025 Z1).** Conditional on upstream fork
  resolution. If resolved before v1.0, a panoptic boundary cell joins
  the snapshot; if not, the gap is documented.

## Pros and cons of the options

### Option 1 — flat dataclass with nullable per-paradigm fields

- 👍 Pros: Minimal diff; the existing `Workload(workload_id, gt_path,
  dt_path, supported_iou_types)` becomes
  `Workload(workload_id, paradigm, gt_path, dt_path, gt_png_dir,
  gt_json, ..., supported_iou_types)`. Every existing test that reads
  `wl.gt_path` keeps working unchanged for detection.
- 👎 Cons: Every reader has to remember which fields are populated for
  which paradigm. The "nullable everywhere" shape leaks into the
  schema and the comparator. No parse-time validation that a
  panoptic workload didn't accidentally set `dt_path`.
  Misuse failures surface as `AttributeError` deep inside a runner
  rather than at the CLI boundary.

### Option 2 — tagged union via Pydantic discriminated union (chosen)

- 👍 Pros: Each variant carries only its own fields. The discriminator
  validates at parse time. The runner subprocesses can `assert
  isinstance(wl, InstanceWorkload)` at the entry point and the type
  checker narrows from there. Mirrors ADR-0029's namespace split and
  ADR-0032's `WireEnvelopeBody` shape — the project already has a
  closed-world variant pattern. JSON-friendly serialization for
  embedding in `BenchResult`.
- 👎 Cons: The flat-dataclass-to-Pydantic-model migration is touched
  by every detection runner: `wl.gt_path` is now an
  `InstanceWorkload.gt_path` attribute access, gated by an
  `isinstance` check. The `_protocol.py:parse_runner_args()` signature
  is unchanged (the runner subprocess still takes `--gt` / `--dt`
  CLI args; the orchestrator unpacks the variant before invoking the
  runner), but the orchestrator's plumbing has to gain that
  unpacking step.

### Option 3 — separate harness per paradigm

- 👍 Pros: Each paradigm is fully isolated; no cross-paradigm coupling
  in the comparator or the result store. New paradigms add a new
  package without touching the existing ones.
- 👎 Cons: Four CLIs, four result stores, four longitudinal reports.
  Cross-paradigm comparison is *impossible* (not just structurally
  rejected — there's no shared infrastructure to do the
  rejection through). Re-implements 80% of the harness three times.
  Defeats the longitudinal-history-on-one-machine premise of ADR-0017.

## Links and references

- ADR-0017 (extended) — the detection bench harness this ADR generalizes.
  This ADR also amends ADR-0017 §"Out of scope" to remove the
  streaming / `BackgroundEvaluator` carve-out (this ADR supersedes it).
- ADR-0014 — `BackgroundEvaluator` instrumentation hook lands as the
  B5 follow-up runner; the latency-sample accumulator is a
  feature-gated extension, not a default code path.
- ADR-0025 — Panoptic API. Source of the panoptic parity contract:
  `pq_compute_single_core(proc_id=0)` strict, panopticapi SHA pinned
  in `crates/vernier-panoptic/src/parity.rs`. ADR-0025 §Z1
  (boundary-PQ) is the deferred-conditional cell.
- ADR-0028 — Semantic segmentation. Source of the semantic library
  design (per-paradigm submodule, three oracle stacks). ADE20K-vs-mmseg
  is the bench cell anchor (S3-B), aligned-tier pending PR-B6/7/8
  vendoring. The Cityscapes oracle path remains a valid library use
  case (users with Cityscapes installs of their own); only the
  redistributable-bench-cell story was dropped.
- ADR-0029 — Per-paradigm namespace restructure. Precedent for a
  closed-world variant per paradigm; this ADR's `Workload` discriminated
  union mirrors that shape.
- ADR-0030 — Buffer-protocol DLPack ingest. Source of the
  `coco_val2017_dlpack_vs_json` cell.
- ADR-0032 — Distributed evaluation across paradigms. Source of the
  cross-paradigm-merge structural reject pattern; this ADR mirrors it
  for cross-paradigm comparison. Streaming bit-equal `Summary.stats`
  between batch and stream is the in-paradigm parity contract.
- [MADR](https://adr.github.io/madr/) — the format this ADR follows.
