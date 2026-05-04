# How to submit detections as numpy arrays or DLPack tensors

`StreamingEvaluator.update(...)` and `BackgroundEvaluator.submit(...)`
accept detections in two forms (per ADR-0030):

- **JSON bytes** (legacy) — the COCO `loadRes` shape. One parser path,
  one `bytes.to_vec()` per call. Right when detections come from disk
  or a wire protocol.
- **`Detections` dicts** (new) — numpy arrays or any DLPack-CPU
  tensor (torch CPU, jax CPU, cupy host buffers). Skips
  `json.dumps` + `json.loads` on the hot path.

The matching kernel, accumulate, snapshot/finalize lifecycle, and
parity contract are identical across both ingest paths — they share
every line of code from `CocoDetections::from_inputs` onward.

## The training-loop case

Model outputs are already numpy arrays / torch tensors. Pass them
through directly:

```python
import numpy as np
from vernier.instance import StreamingEvaluator

gt_bytes = open("instances_val2017.json", "rb").read()
ev = StreamingEvaluator(gt_bytes, iou_type="bbox")

for images, image_ids in val_loader:
    out = model(images)  # boxes (N, 4), scores (N,), labels (N,)
    for image_id, b, s, l in zip(image_ids, out.boxes, out.scores, out.labels):
        ev.update({
            "image_id": int(image_id),
            "boxes": b.cpu().numpy().astype(np.float64),
            "scores": s.cpu().numpy().astype(np.float64),
            "labels": l.cpu().numpy().astype(np.int64),
        })

summary = ev.finalize()
```

`update` accepts a single per-image `Detections` dict (the natural
shape of a model forward) or a `Sequence[Detections]` for multi-image
batches in one call. Empty batches are valid; pass an empty list.

DLPack tensors flow through the same path:

```python
import torch
from vernier.instance import StreamingEvaluator

ev = StreamingEvaluator(gt_bytes, iou_type="bbox")
boxes = torch.zeros((128, 4), dtype=torch.float64)  # CPU tensor
scores = torch.zeros(128, dtype=torch.float64)
labels = torch.ones(128, dtype=torch.int64)

# numpy and torch CPU tensors both expose __dlpack__; the FFI screens
# device via __dlpack_device__ before reading the capsule.
ev.update({"image_id": 1, "boxes": boxes, "scores": scores, "labels": labels})
```

GPU-resident tensors are rejected explicitly — move to CPU first:

```python
ev.update({"image_id": 1, "boxes": boxes.cpu(), ...})
```

## Required dtypes and layout

The boundary contract (ADR-0030 §"Validation rules at the boundary"):

| Field | dtype | Layout |
|---|---|---|
| `boxes` | `float64` | `(N, 4)` C-contiguous, xywh |
| `scores` | `float64` | `(N,)` |
| `labels` | `int64` | `(N,)` |
| `rles[i].counts` | `uint32` | 1-D contiguous |
| `keypoints` | `float64` | `(N, K, 3)` C-contiguous |

`f32` is rejected, not silently promoted (per ADR-0004). For users
who want the convenience, opt in at construction:

```python
ev = StreamingEvaluator(gt_bytes, iou_type="bbox", cast_inputs=True)
```

`cast_inputs=True` silently runs `np.ascontiguousarray(arr, dtype=...)`
on each input and emits a one-shot `UserWarning` per evaluator. The
default (`False`) gives you the full ADR-0004 boundary check.

## Segmentation: array form requires uncompressed RLE

`iou_type="segm"` and `"boundary"` consume `rles: Sequence[RLE]`,
where each `RLE` is `{"counts": uint32 ndarray, "size": (h, w)}`.
Polygons and dense bitmasks are **not** accepted on the array path —
encode to RLE at the dataloader boundary instead:

```python
import numpy as np
from pycocotools import mask as pmask
from vernier.instance import StreamingEvaluator

gt_bytes = open("instances_val2017.json", "rb").read()
ev = StreamingEvaluator(gt_bytes, iou_type="segm")

for image_id, masks_hxw, boxes, scores, labels in batches:
    rles = []
    for m in masks_hxw:               # m: (H, W) bool
        compressed = pmask.encode(np.asfortranarray(m.astype(np.uint8)))
        # Convert to uncompressed counts:
        binary = np.asarray(pmask.decode(compressed))
        counts = _column_major_runs(binary)  # your own helper
        rles.append({
            "counts": np.asarray(counts, dtype=np.uint32),
            "size": (m.shape[0], m.shape[1]),
        })
    ev.update({
        "image_id": int(image_id),
        "boxes": boxes,
        "scores": scores,
        "labels": labels,
        "rles": rles,
    })
```

A future `vernier.mask.encode_batch` helper will package the encode
loop above (per ADR-0030 §"Future work").

## Keypoints

`iou_type="keypoints"` requires a `(N, K, 3)` float64 array of
`(x, y, v)` triplets:

```python
ev = StreamingEvaluator(gt_bytes, iou_type="keypoints")
ev.update({
    "image_id": 1,
    "boxes": np.asarray(...),    # (N, 4)
    "scores": np.asarray(...),   # (N,)
    "labels": np.asarray(...),   # (N,)
    "keypoints": np.asarray(...),  # (N, K, 3)
})
```

## Background evaluator

`BackgroundEvaluator.submit` accepts the same forms with the same
contract:

```python
from vernier.instance import BackgroundEvaluator

with BackgroundEvaluator(gt_bytes, iou_type="bbox") as bg:
    for image_id, boxes, scores, labels in val_loader:
        bg.submit({
            "image_id": int(image_id),
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
        })
    summary = bg.finalize()
```

## See also

- [Background evaluator](background-evaluator.md) — when to pick the
  background surface.
- [Boundary IoU](boundary-iou.md) — `dilation_ratio` is configured at
  evaluator construction; the array path consumes the same RLE shape.
- ADR-0030 — full validation table and the rationale for the
  CPU-only DLPack contract.
