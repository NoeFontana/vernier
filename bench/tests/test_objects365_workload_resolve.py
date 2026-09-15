"""Objects365 workload registry — name parsing, bbox-only support, and
the cache-key separation from the COCO / LVIS jittered DTs.

Offline by default: ``VERNIER_OBJECTS365_GT_PATH`` points the resolver at
a tiny inline GT. The real download is gated by
``VERNIER_BENCH_DOWNLOAD_TESTS=1``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from objects365_val_cache import GT_SHA256, file_sha256

from bench.harness.paths import REPO_ROOT
from bench.workloads import resolve

_DOWNLOAD_GATE = "VERNIER_BENCH_DOWNLOAD_TESTS"


def _o365_like_gt(path: Path) -> None:
    """One image, one crowd and two non-crowd boxes, O365's extra
    ``isfake`` / ``isreflected`` fields, and no ``segmentation``."""
    anns = [
        {"id": i, "image_id": 1, "category_id": 1, "bbox": [4 + i, 4, 8, 8], "area": 64.0,
         "iscrowd": int(i == 3), "isfake": 0, "isreflected": 0}
        for i in (1, 2, 3)
    ]  # fmt: skip
    path.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "width": 64, "height": 64, "file_name": "a.jpg", "url": ""}],
                "categories": [{"id": 1, "name": "Person"}],
                "annotations": anns,
                "licenses": [],
            }
        )
    )


def test_objects365_jittered_resolves_bbox_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VERNIER_BENCH_CACHE", str(tmp_path / "bench"))
    gt = tmp_path / "zhiyuan_objv2_val.json"
    _o365_like_gt(gt)
    monkeypatch.setenv("VERNIER_OBJECTS365_GT_PATH", str(gt))

    w = resolve("objects365_val_jittered_seed3", REPO_ROOT)
    assert w.workload_id == "objects365_val_jittered_seed3"
    assert w.paradigm == "instance"
    assert w.gt_path == gt
    assert w.supported_iou_types == frozenset({"bbox"})
    assert "objects365_val_jittered_seed3" in w.dt_path.name
    dets = json.loads(w.dt_path.read_text())
    assert dets, "jitter generator emitted no detections"
    assert all("segmentation" not in d for d in dets)
    assert all(d["image_id"] == 1 and len(d["bbox"]) == 4 for d in dets)


def test_objects365_malformed_seed_falls_through() -> None:
    with pytest.raises(ValueError, match="unknown workload"):
        resolve("objects365_val_jittered_seedx", REPO_ROOT)


@pytest.mark.skipif(
    os.environ.get(_DOWNLOAD_GATE) != "1",
    reason=f"set {_DOWNLOAD_GATE}=1 to exercise the Objects365 annotation fetch.",
)
def test_objects365_gt_download_matches_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VERNIER_BENCH_CACHE", str(tmp_path))
    monkeypatch.delenv("VERNIER_OBJECTS365_GT_PATH", raising=False)
    from bench.workloads import objects365_val

    gt = objects365_val.gt_path()
    assert file_sha256(gt) == GT_SHA256
