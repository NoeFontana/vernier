#!/usr/bin/env python3
"""Materialize a 'perfect' COCO-style detections JSON from a GT JSON.

For every non-crowd GT annotation, emit a detection with the same
``image_id``, ``category_id``, and ``bbox``, scored 1.0. The result is
a deterministic predictions file usable as ``VERNIER_COCO_DT_PATH``
for the whole-dataset parity smoke (``just test-coco-val``).

This does not produce a numerically interesting AP curve — by
construction stats[0] == 1.0 — but it exercises the full pipeline at
real-world scale (5000 images / 80 categories / ~36k annotations on
COCO val2017). For non-trivial parity, point ``VERNIER_COCO_DT_PATH``
at a real detector's predictions JSON instead.

Usage:
    python tools/make-perfect-dt.py <gt.json> <dt.json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    gt_path = Path(sys.argv[1])
    dt_path = Path(sys.argv[2])

    with gt_path.open("rb") as f:
        gt = json.load(f)

    # Score = 1 - eps*idx so every detection has a unique, deterministic
    # score in input order. All-equal scores would force pycocotools'
    # stable mergesort and vernier's argsort to disagree on tie order
    # (quirk A1) and the smoke would re-test that single issue instead
    # of exercising the long tail of other quirks at scale. AP is still
    # trivially 1.0 because each detection has IoU=1 with exactly one
    # GT in its (image, category) cell.
    detections = [
        {
            "image_id": ann["image_id"],
            "category_id": ann["category_id"],
            "bbox": ann["bbox"],
            "score": 1.0 - 1e-9 * idx,
        }
        for idx, ann in enumerate(a for a in gt.get("annotations", []) if not a.get("iscrowd", 0))
    ]

    with dt_path.open("w") as f:
        json.dump(detections, f)

    print(f"wrote {len(detections)} detections → {dt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
