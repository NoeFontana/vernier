# Distributed evaluation across ranks

Per [ADR-0031](../adr/0031-dist-eval.md) (instance) and
[ADR-0032](../adr/0032-dist-eval-paradigms.md) (semantic + panoptic),
every rank evaluates its own slice locally, gathers a small bytes payload
via the user's transport (`torch.distributed.all_gather_object`,
`mpi4py.comm.gather`, etc.), and the head rank reconstructs an evaluator
equivalent to a batch run over the union of the partials. The same idiom
works in all three paradigms.

vernier ships a bytes interface; the transport is the user's problem (no
`import torch` inside vernier and no torch-version pin to chase). `rank_id`
is the user's responsibility — vernier doesn't try to discover it.
`from_partials()` returns an evaluator, not a `Summary`, so the same
instance can be checkpointed via `finalize_to_partial` or queried for
memory state; merge is a constructor, not a terminal operation.

## Instance — bbox / segm / boundary / keypoints

```python
import torch.distributed as dist
from vernier.instance import StreamingEvaluator

ev = StreamingEvaluator(gt_bytes, iou_type="bbox", rank_id=dist.get_rank())
for batch in val_loader_for_this_rank:
    ev.update(model(batch["images"]))

partial = ev.finalize_to_partial()  # bytes
gathered: list[bytes | None] = [None] * dist.get_world_size()
dist.all_gather_object(gathered, partial)

if dist.get_rank() == 0:
    merged = StreamingEvaluator.from_partials(
        gt_bytes, gathered, iou_type="bbox"
    )
    log_metrics(merged.finalize())
```

## Semantic — mIoU / FWIoU / pAcc / mAcc

`vernier.semantic.StreamingEvaluator` ships the same surface. The headline
difference: confusion-matrix sums are u64-additive, so strict-mode merge
is **unconditionally bit-equal** to a batch run over the union — no
`pytest.skip`, no tiebreak caveat:

```python
import torch.distributed as dist
import vernier.semantic as sem

ev = sem.StreamingEvaluator(
    n_classes=19,                    # Cityscapes
    parity_mode="strict",
    rank_id=dist.get_rank(),
)
for batch in val_loader_for_this_rank:
    for image_id, gt, dt in batch:
        ev.update(image_id, gt, dt)

partial = ev.finalize_to_partial()
gathered: list[bytes | None] = [None] * dist.get_world_size()
dist.all_gather_object(gathered, partial)

if dist.get_rank() == 0:
    merged = sem.StreamingEvaluator.from_partials(
        n_classes=19, partials=gathered, parity_mode="strict",
    )
    log_metrics(merged.finalize())
```

## Panoptic — PQ

`vernier.panoptic.StreamingEvaluator` ships the same surface plus one
additional knob — `retain_per_image_deltas` — that is the *flagship*
determinism control. Default `False` keeps single-rank streaming memory
lean (per-category PqStat fold only). Set it to `True` on every rank when
you need strict-mode bit-equality across the merge boundary; the merge
accumulator re-sorts per-image deltas by `image_id` and re-sums in batch
order, recovering bit-equality despite f64 non-associativity:

```python
import torch.distributed as dist
import vernier.panoptic as pq

ev = pq.StreamingEvaluator(
    categories_json,
    parity_mode="strict",
    retain_per_image_deltas=True,    # opt-in for deterministic CI gate
    rank_id=dist.get_rank(),
)
for batch in val_loader_for_this_rank:
    for image_id, gt_lm, gt_si, dt_lm, dt_si in batch:
        ev.update(image_id, gt_lm, gt_si, dt_lm, dt_si)

partial = ev.finalize_to_partial()
gathered: list[bytes | None] = [None] * dist.get_world_size()
dist.all_gather_object(gathered, partial)

if dist.get_rank() == 0:
    merged = pq.StreamingEvaluator.from_partials(
        categories_json,
        gathered,
        "strict",
        retain_per_image_deltas=True,
    )
    summary = merged.finalize()
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

- [ADR-0031](../adr/0031-dist-eval.md) — instance
  streaming determinism + merge contract.
- [ADR-0032](../adr/0032-dist-eval-paradigms.md) — full
  determinism contract and validation surface across paradigms.
