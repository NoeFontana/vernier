"""Whole-dataset parity smoke test against pycocotools on COCO val2017.

Five hand-crafted fixtures (``test_parity.py``) cover known quirks but
do not exercise the long tail of real-world inputs. This test runs the
full COCO val2017 ground truth against a published detector's
predictions and asserts every snapshot field — eval-image dicts,
precision/recall/scores tensors, the 12-element stats vector — matches
pycocotools bit-for-bit in strict parity mode.

Skipped by default: enabling it requires the COCO val2017 annotations
and a detector predictions JSON, neither of which can be committed
(license + size). Provide them via:

    VERNIER_COCO_GT_PATH  → path to instances_val2017.json
    VERNIER_COCO_DT_PATH  → path to detector predictions JSON

See ``docs/engineering/coco-val-parity.md`` for setup, including a
fetch helper that downloads the annotations to ``.cache/coco-val2017/``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from .harness import assert_snapshots_equal, snapshot

_GT_ENV = "VERNIER_COCO_GT_PATH"
_DT_ENV = "VERNIER_COCO_DT_PATH"


def _resolve(env: str) -> Path | None:
    value = os.environ.get(env)
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_file() else None


@pytest.mark.parity
@pytest.mark.coco_val
def test_coco_val2017_bbox_parity() -> None:
    gt = _resolve(_GT_ENV)
    dt = _resolve(_DT_ENV)
    if gt is None or dt is None:
        pytest.skip(
            f"set {_GT_ENV} and {_DT_ENV} to existing files to enable; "
            "see docs/engineering/coco-val-parity.md"
        )
    ref = snapshot("pycocotools", gt, dt, "bbox")
    cand = snapshot("vernier", gt, dt, "bbox")
    assert_snapshots_equal(ref, cand)
