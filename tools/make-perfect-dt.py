#!/usr/bin/env python3
"""Materialize a 'perfect' COCO-style detections JSON from a GT JSON.

For every non-crowd GT annotation, emit a detection with the same
``image_id``, ``category_id``, and ``bbox``, scored 1.0. The result is
a deterministic predictions file usable as ``VERNIER_COCO_DT_PATH``
for the whole-dataset parity smoke (``just test-coco-val``).

With ``--segm``, also copy each GT's ``segmentation`` (polygon list or
RLE) and ``area`` onto the detection. The vernier segm path requires
``segmentation`` on every DT (quirk K3 disposition: ``corrected``);
the bbox-only output is unusable for ``iou_type="segm"``.

This does not produce a numerically interesting AP curve — by
construction stats[0] == 1.0 — but it exercises the full pipeline at
real-world scale (5000 images / 80 categories / ~36k annotations on
COCO val2017). For non-trivial parity, point ``VERNIER_COCO_DT_PATH``
at a real detector's predictions JSON instead.

Usage:
    python tools/make-perfect-dt.py [--segm] <gt.json> <dt.json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Materialize a 'perfect' COCO-style detections JSON from a GT JSON.",
    )
    parser.add_argument("gt_path", type=Path, help="path to COCO GT JSON")
    parser.add_argument("dt_path", type=Path, help="output detections JSON")
    parser.add_argument(
        "--segm",
        action="store_true",
        help="copy segmentation and area from GT onto each detection",
    )
    args = parser.parse_args()

    with args.gt_path.open("rb") as f:
        gt = json.load(f)

    if "annotations" not in gt:
        print(f"ERROR: {args.gt_path} has no 'annotations' key", file=sys.stderr)
        return 1

    non_crowd = [a for a in gt["annotations"] if not a.get("iscrowd", 0)]

    # Score = 1 - eps*idx so every detection has a unique, deterministic
    # score in input order. All-equal scores would force pycocotools'
    # stable mergesort and vernier's argsort to disagree on tie order
    # (quirk A1) and the smoke would re-test that single issue instead
    # of exercising the long tail of other quirks at scale. AP is still
    # trivially 1.0 because each detection has IoU=1 with exactly one
    # GT in its (image, category) cell.
    detections = []
    for idx, ann in enumerate(non_crowd):
        det = {
            "image_id": ann["image_id"],
            "category_id": ann["category_id"],
            "bbox": ann["bbox"],
            "score": 1.0 - 1e-9 * idx,
        }
        if args.segm:
            if "segmentation" not in ann:
                print(
                    f"ERROR: GT annotation id={ann.get('id')!r} missing 'segmentation'",
                    file=sys.stderr,
                )
                return 1
            det["segmentation"] = ann["segmentation"]
            det["area"] = ann["area"]
        detections.append(det)

    with args.dt_path.open("w") as f:
        json.dump(detections, f, separators=(",", ":"))

    print(f"wrote {len(detections)} detections → {args.dt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
