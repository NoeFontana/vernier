# Migrating from `faster-coco-eval` to vernier

`faster-coco-eval` is a faster reimplementation of `pycocotools` that
preserves the `COCOeval` API. vernier and `faster-coco-eval` share a
goal — drop into pycocotools-shaped code — but differ on the parity
contract. This guide is for users who started on `faster-coco-eval`
and want vernier's auditable parity guarantees.

## TL;DR — what to change

```python
from vernier import COCOeval, patch_pycocotools          # shim path
from vernier.instance import Bbox, CocoDataset, Evaluator    # native path
```

| `faster-coco-eval` | vernier |
|---|---|
| `from faster_coco_eval import COCOeval_faster` | `from vernier import COCOeval` |
| `init_as_pycocotools()` (sys.modules patch) | `patch_pycocotools(parity_mode="strict")` (returns unpatch callable) |
| `COCOeval_faster(coco_gt, coco_dt, iouType="bbox").evaluate(); .accumulate(); .summarize()` | same call sequence on `vernier.COCOeval(...)` |
| `cocoEval.stats` | same `.stats` numpy array (12 entries, same order) |
| `cocoEval.stats_as_dict` | not exposed; use `summary.stats` (`list[float]`) on the native path |
| Per-image AP via `cocoEval.eval_imgs_dict` | not exposed by design — see [`why-no-per-image-ap.md`](../explanation/why-no-per-image-ap.md) |

Both projects monkey-patch `pycocotools.cocoeval.COCOeval`.
`faster-coco-eval`'s helper is named `init_as_pycocotools()`;
vernier's is `patch_pycocotools()` because the verb names the
mechanism (ADR-0007 §"Naming"). The behavior is the swap; the
difference is what semantics you get afterwards.

## Why migrate

`faster-coco-eval` is faster than pycocotools and silently corrects
several quirks. vernier takes a different stance: every quirk gets
an explicit disposition (`strict` / `aligned` / `corrected`) in
[`docs/engineering/pycocotools-quirks.md`](../engineering/pycocotools-quirks.md),
and the disposition is auditable per row. The trade-off:

- **`faster-coco-eval`** gets you faster numbers that *mostly* match
  pycocotools, with quirk fixes folded in opaquely.
- **vernier (strict mode)** gets you bit-exact pycocotools, with the
  exact divergences from raw pycocotools listed in the survey.
- **vernier (corrected mode)** gets you the documented fixes —
  declared per quirk, opt-in by mode, never silent.

If you have a downstream consumer that asserts numerical
reproducibility against a pinned pycocotools commit (CI quality
gate, paper reproduction, regulated environment), strict mode is
the auditable path.

A secondary lever: vernier exposes opt-in parallelism on every
public evaluate surface via `num_threads=N` (ADR-0047), with
strict-mode bit-equality preserved across thread counts. On val2017
boundary IoU at `num_threads=4`, vernier wall-time is ~17× lower
than faster-coco-eval (969 ms vs 17 200 ms — faster-coco-eval's
`boundary_cpu_count` does scale boundary, but the per-core wall
time is much higher). For non-boundary IoU types
(`bbox`, `segm`, `keypoints`), faster-coco-eval is single-threaded
regardless of how many cores you have; vernier scales those too.
See [`benchmarks.md`](../benchmarks.md) §"Threading scaling" for
the full table.

## Drop-in via `patch_pycocotools`

```python
from vernier import patch_pycocotools

unpatch = patch_pycocotools(parity_mode="strict")
try:
    # Same code that ran under faster-coco-eval's init_as_pycocotools()
    # runs unchanged here. The COCOeval symbol now points at vernier's
    # drop-in.
    from pycocotools.cocoeval import COCOeval
    cocoEval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    cocoEval.evaluate(); cocoEval.accumulate(); cocoEval.summarize()
finally:
    unpatch()
```

The context-manager form is `patched_pycocotools()` and nests
correctly (ADR-0007 §"Reentrancy"). For test setups, prefer the
context manager — the unwind is automatic on test failure.

## Pytest integration and import-order pitfall

`init_as_pycocotools()` and `patch_pycocotools()` share the same
mechanism (`sys.modules["pycocotools.cocoeval"].COCOeval` swap),
which means they share the same import-order pitfall: any module
that already imported `pycocotools.cocoeval.COCOeval` before the
patch fires keeps its original binding. The session-scoped
`conftest.py` snippet and the diagnosis recipe live in
[`from-pycocotools.md`](from-pycocotools.md#pytest-integration) —
the failure mode and fix are identical regardless of which tool
you're migrating from.

## Per-image AP

`faster-coco-eval` exposes per-image AP. vernier does not, by
design. PR curves from a single image are degenerate (one detection
gives a 0-or-1 precision step, not a curve to integrate). The
rationale and the polars recipe to reconstruct per-image diagnostics
from raw counts live at
[`why-no-per-image-ap.md`](../explanation/why-no-per-image-ap.md).
The replacement surface is the per-image / per-class / per-detection
/ per-pair tables documented at
[`how-to/result-tables.md`](../how-to/result-tables.md) — same
information, different shape.

## Parity gate recipe

A common faster-coco-eval workflow is "run both pycocotools and
faster-coco-eval, assert the stats match within tolerance." The
vernier equivalent is the parity smoke at
`tests/python/parity/test_parity.py`, which double-runs the
candidate (vernier) and reference (pycocotools) on a fixture corpus
and diffs every intermediate. For your own datasets, the shell
recipe at
[`how-to/cli-eval.md`](../how-to/cli-eval.md#parity-gate-against-pycocotools-in-a-shell)
is the production-shape gate: invoke vernier and pycocotools from
the CLI, diff the JSON outputs.

## See also

- [Migrating from `pycocotools`](from-pycocotools.md) — the upstream
  case, more detail on the shim, the COCOeval drop-in, and the
  sentinel mapping. If you patched pycocotools via
  `init_as_pycocotools()`, your migration path is structurally that
  guide.
- [ADR-0007](../adr/0007-patch-pycocotools-policy.md) — naming
  rationale (why `patch_pycocotools` and not
  `init_as_pycocotools`), reentrancy contract, parity-mode
  threading.
- [ADR-0002](../adr/0002-three-tier-parity-model.md) — strict / aligned /
  corrected parity tiers; what each tier promises and what it
  costs.
- [`docs/engineering/pycocotools-quirks.md`](../engineering/pycocotools-quirks.md)
  — the auditable disposition table for every quirk.
