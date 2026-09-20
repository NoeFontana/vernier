# Migrating from DOTA_devkit

vernier reproduces DOTA_devkit's polygon IoU **bit for bit** under
`parity_mode="strict"` — the kernel, the horizontal-box prefilter, the
in-place winding reversal, and the signed zeros.

It does **not** yet reproduce DOTA's headline metric. Read the last
section before you plan around this page.

## The short version

```python
from vernier.instance import Evaluator, Quad

evaluator = Evaluator(iou=Quad(), parity_mode="strict")
summary = evaluator.evaluate(gt_json_bytes, detections)
```

Every ground truth and detection carries a `quad` key of eight numbers:

```json
{"quad": [x0, y0, x1, y1, x2, y2, x3, y3], "bbox": [x, y, w, h], ...}
```

Vertex order is preserved exactly as you submit it. DOTA_devkit consumes
it that way — it only reverses the winding to normalize the signed area
— so `strict` does too.

## Converting a DOTA annotation file

DOTA's `.txt` format is one object per line:

```
x1 y1 x2 y2 x3 y3 x4 y4 category difficult
```

which becomes:

```python
def to_record(line, image_id, ann_id, cat_to_id):
    parts = line.split()
    quad = [float(v) for v in parts[:8]]
    xs, ys = quad[0::2], quad[1::2]
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": cat_to_id[parts[8]],
        "iscrowd": 0,
        "quad": quad,
        "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
        "area": shoelace(quad),
        # DOTA's `difficult` is not COCO's `iscrowd`. Map it to `ignore`
        # if you want the devkit's difficult-handling; leave it off
        # otherwise. They are different semantics and vernier will not
        # guess which you meant.
        "ignore": int(parts[9]) if len(parts) > 9 else 0,
    }
```

## What "bit for bit" covers

`parity_mode="strict"` reproduces, deliberately:

- **The origin-hinged triangle fan.** `polyiou` decomposes both polygons
  into triangles hinged on the *coordinate origin*, not on a vertex or a
  centroid, so its intermediates grow with the image coordinate. f64
  absorbs it; the shape of the computation is still part of the answer.
- **The `+1` horizontal-box gate.** `dota_evaluation_task1.py` only ever
  calls `iou_poly` for ground truths whose Pascal-VOC-style envelope
  (`max - min + 1`) overlaps the detection's. This is **not** an
  optimization: the fan returns small non-zero residues on
  envelope-overlapping but polygon-disjoint pairs, so skipping the gate
  would change the matrix. vernier applies the same gate.
- **In-place winding reversal**, which the caller then observes, because
  `iou_poly` re-reads both areas afterwards and the shoelace is summed
  in index order.
- **Signed zeros**, normalized to `+0.0` on the way out.

Evidence: `crates/vernier-geom/tests/dk_bridge.rs` — five million pairs
against the pinned `polyiou.cpp` compiled from source. Zero divergences.

## Where `corrected` differs

| | `strict` (DOTA_devkit) | `corrected` |
| --- | --- | --- |
| Zero-area quad | `0/0` → `NaN` | **typed error at ingestion, in both modes** |
| Self-intersecting quad | a cancellation-dependent number | typed error |
| Disjoint pairs | small non-zero residue possible | exactly `+0.0` |
| Frame | hinged at the image origin | the ground truth's own frame |

The zero-area case is the one row where `corrected` wins in *both*
modes: a `NaN` in the similarity matrix corrupts every match in the
cell, and there is no oracle value worth reproducing.

## The oracle is not vendored, and why

DOTA_devkit states no license — no `LICENSE` file, no header, no
statement in its README, and the GitHub API reports `"license": null`.
With no grant there is no right to redistribute, so this repository
contains none of its bytes. What it pins instead is the SHA-256 of each
file at a fixed commit.

To run the parity tests yourself:

```bash
uv run python tests/python/parity_obb/oracle/dota_devkit/fetch.py
```

That downloads the pinned files into a git-ignored cache, verifies each
hash, and builds the bridge harness. Tests that need it skip cleanly
when it is absent.

## What is not covered: DOTA's headline mAP

DOTA's task-1 evaluator does not use COCO matching. It uses Pascal VOC
matching, and the difference is structural rather than numerical:

- **VOC** takes the argmax over *all* ground truths and scores a false
  positive if that ground truth is already claimed.
- **COCO** falls back to the best *unclaimed* ground truth.
- A *difficult* argmax makes the detection ignored entirely.
- AP integrates over 11 points, or the all-point envelope, not COCO's
  101 recall thresholds.

That is an assignment policy plus a summarizer. It is orthogonal to
geometry, which is why ADR-0063 deliberately left it out: bolting a
second matching rule into a kernel ADR would edit the locked evaluation
spine. It lands with the assignment-axis ADR, alongside axis-aligned
Pascal VOC.

**Until then:** if you need the DOTA leaderboard number, stay on
DOTA_devkit. The kernel underneath is already at parity here, so the
move costs nothing when the protocol lands.

What vernier gives you today is COCO-protocol AP over DOTA geometry —
AP@[0.50:0.95] rather than mAP@0.50 — which is a stricter and more
informative number, and is what the rest of the detection world reports.
