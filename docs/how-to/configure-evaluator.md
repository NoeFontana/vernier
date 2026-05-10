# How to configure the evaluator

`vernier.instance.Evaluator` is a frozen dataclass; its fields
control which IoU kernel runs, which pycocotools quirks reproduce
bit-equally vs apply opt-in fixes, the per-image top-K ladder, and
two opt-ins (`use_cats`, `cast_inputs`). This page is the entry
point for "what should I tune?". The full signature is in
[`reference/python/instance.md`](../reference/python/instance.md);
the custom-grid axes (`iou_thresholds` / `recall_thresholds` /
`area_ranges`) get their own page —
[Custom evaluation grids](custom-evaluation-grids.md).

`Evaluator` is immutable per ADR-0006: build it once and call
`evaluate(...)` per dataset/detections pair. Use `with_options(...)`
to derive a copy with one field overridden.

## Choose the IoU kernel

| `iou=` | What it measures | Deep-dive |
|---|---|---|
| `Bbox()` | Axis-aligned bounding-box IoU (the COCO default) | — |
| `Segm()` | RLE / polygon mask IoU | — |
| `Boundary(dilation_ratio=0.02)` | IoU on the band around mask edges (ADR-0010) | [boundary-iou.md](boundary-iou.md) |
| `Keypoints(sigmas={...})` | OKS keypoint similarity (ADR-0012) | [keypoints-oks.md](keypoints-oks.md) |

Each variant carries its own kernel-specific parameters;
`Boundary` takes `dilation_ratio`, `Keypoints` takes per-category
`sigmas`. Switching kernels is a one-line change:

```python
from vernier.instance import Bbox, Boundary, Evaluator, Keypoints

Evaluator(iou=Bbox())
Evaluator(iou=Boundary(dilation_ratio=0.02))
Evaluator(iou=Keypoints(sigmas={1: (0.026, 0.025, 0.025, 0.035, 0.035)}))
```

## Choose the parity mode

`parity_mode` is either `"corrected"` (the default for new users)
or `"strict"`:

- **`"strict"`** — bit-equal to `pycocotools==2.0.11` on every
  quirk, including documented bugs. Pick this when migrating from
  pycocotools and your pipeline is calibrated to its output (model
  selection, published benchmarks, leaderboards).
- **`"corrected"`** — opt-in fixes for the quirks dispositioned
  `corrected` in
  [`engineering/pycocotools-quirks.md`](../engineering/pycocotools-quirks.md).
  Each fix is itemized so you know exactly when numbers diverge.

```python
Evaluator(iou=Bbox(), parity_mode="strict")
Evaluator(iou=Bbox(), parity_mode="corrected")
```

The full rationale lives in
[ADR-0002](../adr/0002-three-tier-parity-model.md).

## Set `max_dets`

`max_dets` is the per-image top-K detection ladder used to compute
the summary's `AR1 / AR10 / AR100` triple. Default is `None`,
which resolves to the kernel's canonical ladder:

| Kernel | Default ladder |
|---|---|
| `Bbox()` / `Segm()` / `Boundary()` | `(1, 10, 100)` |
| `Keypoints()` | `(20,)` |

Override with an explicit tuple:

```python
Evaluator(iou=Bbox(), max_dets=(1, 10, 100, 300))   # LVIS-flavored 300
Evaluator(iou=Bbox(), max_dets=(100,))              # single ladder rung
```

LVIS uses `(300,)` per quirk **AB4**; see
[Migrating from `lvis-api`](../migrate/from-lvis-api.md) for the
end-to-end recipe.

## `use_cats` — collapse categories

`use_cats=True` (default) evaluates per category and averages.
`use_cats=False` collapses every category onto a single virtual
class, mirroring pycocotools' `p.useCats=0` mode (quirk **L4**):

```python
Evaluator(iou=Bbox(), use_cats=False)
```

Use `False` for class-agnostic detection (e.g., open-vocabulary
localization folds where the model predicts boxes without
category labels).

## `cast_inputs` — accept fp32 / int32 arrays

`cast_inputs=False` (default) requires array-form `Detections` to
be fp64 / int64. `cast_inputs=True` opts into a one-shot
`f32→f64` / `i32→i64` promotion at the FFI boundary
(ADR-0030):

```python
Evaluator(iou=Bbox(), cast_inputs=True).evaluate(dataset, model_outputs)
```

The flag affects only the array-form ingest path — JSON-bytes
detections ignore it. Recipe in [array-ingest.md](array-ingest.md).

## Reuse the GT across calls

`Evaluator.evaluate(gt: bytes | CocoDataset, dt)` accepts a
parsed-once `CocoDataset` handle (ADR-0020) so the GT JSON is
parsed exactly once and per-kernel GT-side derivations cache
across calls. Boundary and segm benefit the most; bbox and
keypoints save only the parse:

```python
from pathlib import Path
from vernier.instance import Boundary, CocoDataset, Evaluator

gt = CocoDataset.from_json(Path("instances_val2017.json").read_bytes())
ev = Evaluator(iou=Boundary(), parity_mode="corrected")

# Each call reuses the cached GT bands; ~36k Chebyshev erodes paid once.
for epoch_dt_bytes in epoch_outputs:
    summary = ev.evaluate(gt, epoch_dt_bytes)
```

## In a training loop

`Evaluator.background(gt, ...)` lifts the same configuration onto a
worker thread so `submit(...)` does not block the trainer:

```python
with ev.background(gt) as bg:
    for images, _ in val_loader:
        bg.submit(json.dumps(model(images)).encode())
    summary = bg.finalize()
```

The worker-thread knobs (`queue_capacity`, `worker_affinity`,
`worker_nice`, `shutdown_timeout_seconds`, `memory_budget_bytes`,
`retain_iou`, `rank_id`, `record_latency_samples`) live on
`background(...)` rather than on `Evaluator` itself — they
configure deployment, not the matching kernel. Recipe:
[background-evaluator.md](background-evaluator.md).

## Across ranks

`Evaluator.evaluate_to_partial(gt, dt, *, rank_id)` per rank, then
`Evaluator.from_partials(gt, partials, **config)` on the head
rank, runs the same configuration across processes. `from_partials`
takes the same kwargs as `Evaluator(...)`; mismatches raise typed
`Partial*` errors. Recipe: [distributed-eval.md](distributed-eval.md).

## Custom IoU / recall / area grids

The `iou_thresholds`, `recall_thresholds`, and `area_ranges` fields
let you sweep IoU sensitivity, tune the AP integration density, or
define domain-specific area buckets. They have one footgun —
the canonical 12-stat summary plan cannot index into a
non-canonical grid — so they get their own page:
[Custom evaluation grids](custom-evaluation-grids.md).

## See also

- [`reference/python/instance.md`](../reference/python/instance.md)
  — the full `Evaluator` signature, auto-rendered from the source.
- [ADR-0002](../adr/0002-three-tier-parity-model.md) — strict /
  corrected parity contract.
- [ADR-0006](../adr/0006-threading-model.md) §"Frozen evaluator"
  — why `Evaluator` is immutable.
- [ADR-0011](../adr/0011-discriminated-kernel-config.md) — the
  `IouKind` discriminated union.
- [ADR-0012](../adr/0012-oks-keypoints-surface.md) — kernel-canonical
  `max_dets` ladder.
- [ADR-0020](../adr/0020-parsed-once-dataset-handle.md) — `CocoDataset`
  handle and the per-kernel GT cache.
- [ADR-0030](../adr/0030-buffer-protocol.md) — `cast_inputs` and
  array-form ingest.
