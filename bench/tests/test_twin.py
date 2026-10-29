"""Synthetic twins (ADR-0067): exact cost totals, determinism, no source text."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from bench.workloads import twin


def _source(tmp_path: Path) -> tuple[Path, Path]:
    """A small COCO-shaped GT/DT pair with polygons, crowd and text fields."""
    rng = np.random.default_rng(7)
    images, anns, dts = [], [], []
    for i in range(40):
        images.append(
            {"id": 1000 + i, "width": 320, "height": 240, "file_name": f"secret_{i:04d}.jpg"}
        )
        for _ in range(int(rng.integers(0, 6))):
            x, y = rng.uniform(0, 200), rng.uniform(0, 150)
            w, h = rng.uniform(8, 100), rng.uniform(8, 80)
            crowd = rng.random() < 0.1
            ann: dict[str, Any] = {
                "id": 50_000 + len(anns),
                "image_id": 1000 + i,
                "category_id": int(rng.integers(1, 6)),
                "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                "area": round(w * h * 0.7, 2),
                "iscrowd": int(crowd),
                "segmentation": (
                    {"counts": [int(x) * 240, 40, 76_800 - int(x) * 240 - 40], "size": [240, 320]}
                    if crowd
                    else [[x, y, x + w, y, x + w, y + h, x, y + h]]
                ),
            }
            anns.append(ann)
            for _ in range(int(rng.integers(0, 3))):
                dts.append(
                    {
                        "image_id": 1000 + i,
                        "category_id": ann["category_id"],
                        "bbox": [x + rng.normal(0, 3), y, w, h],
                        "score": float(rng.random()),
                    }
                )
    categories = [{"id": c, "name": f"name{c}"} for c in range(1, 6)]
    gt_path, dt_path = tmp_path / "gt.json", tmp_path / "dt.json"
    gt_path.write_text(
        json.dumps({"images": images, "annotations": anns, "categories": categories})
    )
    dt_path.write_text(json.dumps(dts))
    return gt_path, dt_path


def _write(tmp_path: Path, gt: dict[str, Any], dts: list[dict[str, Any]]) -> tuple[Path, Path]:
    gt_path, dt_path = tmp_path / "twin_gt.json", tmp_path / "twin_dt.json"
    gt_path.write_text(json.dumps(gt))
    dt_path.write_text(json.dumps(dts))
    return gt_path, dt_path


def test_twin_reproduces_cost_totals_exactly(tmp_path: Path) -> None:
    profile = twin.extract(*_source(tmp_path), masks=True, source="test")
    again = twin.extract(*_write(tmp_path, *twin.generate(profile, seed=0)), masks=True, source="t")
    for key in ("gt", "dt", "cells", "iou_pairs", "crowd"):
        assert again["totals"][key] == profile["totals"][key], key


def test_twin_is_deterministic_per_seed(tmp_path: Path) -> None:
    profile = twin.extract(*_source(tmp_path), masks=False, source="test")
    assert json.dumps(twin.generate(profile, seed=3)) == json.dumps(twin.generate(profile, seed=3))
    assert json.dumps(twin.generate(profile, seed=3)) != json.dumps(twin.generate(profile, seed=4))


def test_profile_keeps_no_source_text(tmp_path: Path) -> None:
    profile = twin.extract(*_source(tmp_path), masks=True, source="test")
    dumped = json.dumps(profile)
    assert "secret_" not in dumped
    assert "name1" not in dumped
