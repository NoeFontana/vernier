# ADR-0036: Vendor mmsegmentation `IoUMetric` standalone for semantic-segmentation parity

- **Status:** proposed
- **Date:** 2026-05-09
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** —

## Context and problem statement

Semantic-segmentation strict-mode parity has been infrastructure-ready
since ADR-0028 — `crates/vernier-semantic/src/parity.rs` reserves
`ORACLE_MMSEGMENTATION_COMMIT_SHA` with a `"PR-B6-pending"` placeholder
and a tripwire test enforcing the eventual flip — but no oracle has
been vendored. The blocker is `mmsegmentation`'s install footprint:
the package transitively pulls `mmcv`, `mmengine`, and PyTorch, ~3 GB
on a clean install, and would be a permanent CI tax on every contributor.
ADR-0033 cell **S3-B** (ADE20K semantic eval) sits at *aligned*-tier as
a result.

The semantic-quirks survey
([`docs/engineering/sem-seg-quirks.md`](../engineering/sem-seg-quirks.md))
flagged this as **AP5** — *informational*, with a "future cleanup
vendors only `IoUMetric` + dependencies (a slim subset, ~50 KB) if
PyTorch becomes a CI bottleneck" note — and as **open question 6**,
to be confirmed by reading `iou_metric.py` end-to-end.

A close reading of the file at upstream tag `v1.2.2`
(`c685fe6767c4cadf6b051983ca6208f1b9d1ccb8`, 2023-12-14, 286 lines)
shows:

- The numerics are concentrated in two static methods,
  `IoUMetric.intersect_and_union` (label binning + per-class
  histograms) and `IoUMetric.total_area_to_metrics` (mIoU / mAcc /
  mDice / mFscore aggregation). Together they are the parity surface.
- `intersect_and_union` calls `torch.histc(input.float(), bins=N,
  min=0, max=N-1)` to build the histograms. `torch.histc` does not
  have a bit-exact NumPy equivalent for the general case (its float-edge
  bin semantics differ from `np.bincount`'s integer-bin semantics);
  shimming it with NumPy would weaken the parity claim.
- All other external symbols — `mmengine.dist.is_main_process`,
  `mmengine.evaluator.BaseMetric`, `mmengine.logging.MMLogger`,
  `mmengine.logging.print_log`, `mmengine.utils.mkdir_or_exist`,
  `mmseg.registry.METRICS`, `prettytable.PrettyTable`, `PIL.Image` —
  are either decorators, abstract bases, log-formatting helpers, or
  `output_dir`-only paths. None are reached by the parity harness's
  call site (which invokes the static methods directly), and none
  affect the metric values returned.

So the right answer is *not* the user's first instinct ("pure NumPy on
confusion matrices, vendor 50 KB and ship") but a refinement: vendor
the file, stub the mmengine / mmseg.registry / prettytable surface,
keep `torch` as a real dependency. The result cuts the install
footprint from ~3 GB to a CPU-only torch wheel (~600 MB installed) and
removes mmcv / mmengine / mmsegmentation entirely from the test
environment.

## Decision drivers

- **Strict-mode parity-claim integrity.** The mmsegmentation pin must
  be bit-equal to upstream at a citable artifact (release tag), and
  the parity harness must call the same static methods upstream callers
  call. Shims that replace numerical calls (e.g. `torch.histc` →
  `np.bincount`) weaken the claim by one indirection.
- **CI cost.** Every CI run that touches the semantic parity job
  installs the oracle's runtime deps. The full mmsegmentation transitive
  is ~3 GB; the minimum-viable footprint must be much smaller.
- **Refresh discipline.** ADR-0001 makes vendoring an ADR-level
  decision and refreshes ADR-level too. The chosen approach must
  preserve `docs/engineering/vendoring.md`'s "no modifications to
  vendored bytes" invariant — the SHA pin is the parity claim.
- **Stub fragility.** Anything we substitute for upstream code is
  drift risk at refresh time; the stub surface should be as small as
  possible and exercised by a one-time `_verify_against_pip.py` check
  that asserts the vendored bytes + stubs reproduce a real
  `pip install mmsegmentation` exactly.

## Considered options

1. **Pinned-package env at `bench/envs/mmsegmentation/`.** Use
   `mmsegmentation==1.2.2` directly via uv. ~3 GB install, ~5–8 min
   first-time sync, permanent CI tax.
2. **Standalone `IoUMetric` vendor with stubs (chosen).** Vendor only
   `iou_metric.py` + `LICENSE` at the v1.2.2 SHA; provide a thin
   `mmengine` / `mmseg.registry` / `prettytable` stub layer; require
   `torch` as a real dep but no other heavy transitives. ~50 KB vendored
   + ~600 MB torch.
3. **Full source vendor.** Pull the entire mmsegmentation tree
   (configs, models, datasets, tools). Wildly out of scope; the parity
   contract needs one file.
4. **Stub torch as well.** Reimplement `torch.histc` on top of
   `np.bincount` in our shim layer. Avoids the ~600 MB torch dep but
   re-introduces a numerics shim between the oracle and the
   reference-truth artifact. Weakens the parity claim by one
   indirection.

## Decision outcome

Chosen option: **option 2 — standalone `IoUMetric` vendor with stubs**.

The vendored tree lives at
[`tests/python/parity_semantic/oracle/mmsegmentation/`](../../tests/python/parity_semantic/oracle/mmsegmentation/),
mirroring the layout of the boundary-IoU (ADR-0010), LVIS
(ADR-0026), and panoptic (ADR-0025) precedents. The pinned SHA flips
`ORACLE_MMSEGMENTATION_COMMIT_SHA` in
[`crates/vernier-semantic/src/parity.rs`](../../crates/vernier-semantic/src/parity.rs)
from `"PR-B6-pending"` to the real 40-char SHA; an additional
`ORACLE_TORCH_FLOOR` constant pins the runtime torch floor at the
project's existing `>= 2.4` constraint. The full provenance table,
byte-equality SHA-256 hashes, license analysis, and refresh procedure
are recorded in the adjacent
[`VENDORING.md`](../../tests/python/parity_semantic/oracle/mmsegmentation/VENDORING.md).

A one-time
[`_verify_against_pip.py`](../../tests/python/parity_semantic/oracle/mmsegmentation/_verify_against_pip.py)
script confirms the vendored bytes + stubs produce bit-identical
metrics to a real `pip install mmsegmentation==1.2.2`. It is run
locally during vendoring and at every SHA refresh (refresh procedure
step 7); it is **not** wired into CI, because the whole point of the
vendor is to avoid pulling the full mmsegmentation transitive.

### Consequences

- **Positive:**
  - Semantic strict-mode parity is now implementable (the oracle is
    importable and runs against a fixture). Per-quirk fixtures can
    follow without further infra work.
  - mmcv, mmengine, and the mmsegmentation package itself never enter
    the test environment. Net install savings: ~2.4 GB.
  - The vendoring discipline (no modifications to vendored bytes,
    SHA + VENDORING.md drift = build failure) extends to a fourth
    parity oracle without further policy changes.
- **Negative:**
  - `torch` becomes a hard dependency for `tests/python/parity_semantic/`.
    `pytest.importorskip("torch")` in the conftest gates the tree
    cleanly when torch is absent, mirroring the `real_models` extra
    pattern. CI for the semantic parity job pays the torch wheel
    install once per cache miss (~30 s).
  - The stub layer (`_mmengine_stub.py`, ~120 lines) is our code to
    maintain. If a future SHA refresh introduces new symbols, the
    stub extends in lock-step with the SHA bump. The
    `_verify_against_pip.py` check catches drift before merge.
- **Neutral:**
  - The bench cell **S3-B** (ADE20K semantic) can re-grade from
    *aligned* to *strict* once the per-quirk parity fixtures land.
    That re-grade is a follow-up commit, not gated on this ADR.
  - PR-B7 (cityscapesScripts) and PR-B8 (Pascal/ADE references) are
    untouched. Their placeholder SHAs stay `*-pending` until those
    oracles are vendored under their own ADRs.

## Pros and cons of the options

### Option 1: Pinned-package env

- 👍 Simplest implementation: a `pyproject.toml` with one pin and a
  short runner subprocess. No stubs, no shims, no hand-written code
  to maintain.
- 👍 Refresh is `uv lock` plus an ADR amendment.
- 👎 ~3 GB install, ~5–8 min first sync, every CI run that touches
  the semantic parity job pays this cost.
- 👎 Permanent contributor-onboarding tax.
- 👎 The full mmsegmentation install pulls `pycocotools` transitively,
  which would conflict with vernier's pinned `pycocotools==2.0.11`
  (ADR-0002) unless the env is fully isolated. Resolvable but adds
  config burden.

### Option 2: Standalone vendor with stubs (chosen)

- 👍 ~50 KB vendored + ~600 MB torch = ~80% reduction in install
  footprint vs. option 1.
- 👍 mmcv, mmengine, mmsegmentation absent from the test env. No
  pycocotools-version conflict.
- 👍 Existing `docs/engineering/vendoring.md` discipline applies
  unchanged. Tripwire test (`mmsegmentation_oracle_sha_is_pinned`)
  catches SHA / `VENDORING.md` drift at build time.
- 👎 Hand-written stubs (`_mmengine_stub.py`, ~120 lines) are our
  code to maintain.
- 👎 Refresh requires re-running `_verify_against_pip.py` locally
  (operator must spin up a separate venv with the real package).
  The procedure is documented in `VENDORING.md`'s refresh section.

### Option 3: Full source vendor

- 👍 No stubs.
- 👎 Pulls thousands of files outside the parity surface. Vastly
  out of scope.
- 👎 The unused subtrees (`mmseg/models/`, `mmseg/datasets/`,
  `mmseg/apis/`, `tools/`, `configs/`) carry their own license and
  attribution discipline; auditing them for the parity oracle role
  is wasted work.

### Option 4: Shim torch with `np.bincount`

- 👍 Avoids the ~600 MB torch dep.
- 👍 ~50 KB total install footprint (vendored bytes + small NumPy
  shim).
- 👎 Inserts a numerics shim between vernier and the parity
  artifact. The strict-mode claim becomes "vernier matches our
  numpy shim of upstream" rather than "vernier matches upstream".
  One indirection of drift risk for every refresh.
- 👎 `torch.histc`'s float-edge binning has subtle behavior
  (boundary inclusion, NaN handling, integer-vs-float input
  promotion) that is straightforward to get wrong at the shim
  layer. The `_verify_against_pip.py` check would catch breakage,
  but a "shim drift" failure mode would be added to refresh
  costs.

## Links and references

- **Related ADRs**:
  - ADR-0001 (record architecture decisions) — vendoring is ADR-level.
  - ADR-0002 (three-tier parity model) — `(quirk, oracle) → mode`
    keying.
  - ADR-0010 (boundary-IoU vendoring) — the discipline precedent.
  - ADR-0025 (panopticapi vendoring) — the worked example for
    `VENDORING.md` shape.
  - ADR-0026 (LVIS vendoring) — sibling oracle vendor.
  - ADR-0028 (semantic-segmentation parity strategy) — sets the
    multi-oracle keying that this ADR fills the first cell of.
  - ADR-0033 (multi-paradigm bench) — cell S3-B re-grades from
    *aligned* to *strict* on the back of this ADR.
- **External references**:
  - mmsegmentation v1.2.2 release: <https://github.com/open-mmlab/mmsegmentation/releases/tag/v1.2.2>
  - `mmseg/evaluation/metrics/iou_metric.py` at the pinned SHA:
    <https://github.com/open-mmlab/mmsegmentation/blob/c685fe6767c4cadf6b051983ca6208f1b9d1ccb8/mmseg/evaluation/metrics/iou_metric.py>
- **Quirks survey rows resolved**:
  [`docs/engineering/sem-seg-quirks.md`](../engineering/sem-seg-quirks.md)
  rows AP5 (informational → resolved) and open question 6
  (investigation closed: `IoUMetric` is *not* pure NumPy, but the
  bulk of the dep weight is still stubbable).
