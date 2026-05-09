# Distributed evaluation across ranks

Per [ADR-0031](../adr/0031-dist-eval.md) (instance),
[ADR-0032](../adr/0032-dist-eval-paradigms.md) (semantic + panoptic), and
[ADR-0035](../adr/0035-api-surface-consolidation.md) (the public entry
class), every rank evaluates its own slice locally, gathers a small bytes
payload via the user's transport (`torch.distributed.all_gather_object`,
`mpi4py.comm.gather`, etc.), and the head rank reconstructs a
`Summary` equivalent to a batch run over the union of the partials.
The same idiom works in all three paradigms.

vernier ships a bytes interface; the transport is the user's problem (no
`import torch` inside vernier and no torch-version pin to chase). `rank_id`
is the user's responsibility — vernier doesn't try to discover it.

## Instance — bbox / segm / boundary / keypoints

```python
import torch.distributed as dist
from vernier.instance import Evaluator, Bbox

ev = Evaluator(iou=Bbox(), parity_mode="corrected")
partial = ev.evaluate_to_partial(
    gt_bytes, dt_for_this_rank, rank_id=dist.get_rank()
)  # bytes

gathered: list[bytes | None] = [None] * dist.get_world_size()
dist.all_gather_object(gathered, partial)

if dist.get_rank() == 0:
    summary = Evaluator.from_partials(
        gt_bytes, gathered, iou=Bbox(), parity_mode="corrected"
    )
    log_metrics(summary)
```

If your validation loop submits image-by-image (rather than handing a
single per-rank batch to `evaluate_to_partial`), use
`BackgroundEvaluator` and call `finalize_to_partial()` instead — the
gather + `Evaluator.from_partials(...)` step on the head rank is the
same.

## Semantic — mIoU / FWIoU / pAcc / mAcc

The headline difference for semantic: confusion-matrix sums are u64-
additive, so strict-mode merge is **unconditionally bit-equal** to a
batch run over the union — no tiebreak caveat:

```python
import torch.distributed as dist
import vernier.semantic as sem

rank_gt = sem.Dataset.from_arrays(rank_gt_maps, n_classes=19)  # Cityscapes
rank_dt = sem.Predictions.from_arrays(rank_dt_maps)

ev = sem.Evaluator(parity_mode="strict")
partial = ev.evaluate_to_partial(rank_gt, rank_dt, rank_id=dist.get_rank())

gathered: list[bytes | None] = [None] * dist.get_world_size()
dist.all_gather_object(gathered, partial)

if dist.get_rank() == 0:
    summary = sem.Evaluator.from_partials(
        n_classes=19, partials=gathered, parity_mode="strict",
    )
    log_metrics(summary)
```

## Panoptic — PQ

Panoptic has one additional knob — `retain_per_image_deltas` — that is
the *flagship* determinism control. Default `False` keeps single-rank
memory lean (per-category PqStat fold only). Set it to `True` on every
rank when you need strict-mode bit-equality across the merge boundary;
the merge accumulator re-sorts per-image deltas by `image_id` and
re-sums in batch order, recovering bit-equality despite f64 non-
associativity.

The panoptic `evaluate_to_partial` takes per-image tuples directly
(`PanopticDataset` does not yet expose per-image accessors — closing
that gap is a follow-up to ADR-0035):

```python
import torch.distributed as dist
import vernier.panoptic as pq

ev = pq.Evaluator(parity_mode="strict")

# Per-image tuples: (image_id, gt_label_map, gt_segments_info,
#                   dt_label_map, dt_segments_info)
images_for_this_rank = [...]

partial = ev.evaluate_to_partial(
    images_for_this_rank,
    categories=categories_json,
    rank_id=dist.get_rank(),
    retain_per_image_deltas=True,    # opt-in for deterministic CI gate
)

gathered: list[bytes | None] = [None] * dist.get_world_size()
dist.all_gather_object(gathered, partial)

if dist.get_rank() == 0:
    summary = pq.Evaluator.from_partials(
        categories_json,
        gathered,
        parity_mode="strict",
        retain_per_image_deltas=True,
    )
    log_metrics(pq=summary.pq, sq=summary.sq, rq=summary.rq)
```

Wire-size cost of `retain_per_image_deltas=True` is ~few hundred bytes per
image per rank: ~100 KB at Cityscapes val (500 images), ~1 MB at COCO
panoptic val (5k images). Corrected mode without deltas stays within
[ADR-0004](../adr/0004-numerical-layout-policy.md)'s 4-ULP envelope at zero
memory cost.

## Shared `Partial*` exception family

The five `Partial*` exception classes (`PartialFormatMismatch`,
`PartialDatasetMismatch`, `PartialParamsMismatch`,
`PartialPartitionOverlap`, `PartialRankCollision`) are
**paradigm-shared** — `vernier.instance.PartialDatasetMismatch is
vernier.semantic.PartialDatasetMismatch is
vernier.panoptic.PartialDatasetMismatch` holds, so a single top-level
handler catches the same condition across all three paradigms.

## See also

- [ADR-0031](../adr/0031-dist-eval.md) — instance streaming
  determinism + merge contract.
- [ADR-0032](../adr/0032-dist-eval-paradigms.md) — full determinism
  contract and validation surface across paradigms.
- [ADR-0035](../adr/0035-api-surface-consolidation.md) — why the
  entry class is `Evaluator` (not `StreamingEvaluator`).
