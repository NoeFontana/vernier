"""Vernier runner end-to-end smoke against a tiny inline LVIS-shaped
fixture. We don't depend on the lvis-val cache here — that gate is
covered in :mod:`test_lvis_workload_resolve`. The point is to confirm
the runner subprocess accepts an LVIS-shaped GT and produces a
schema-conformant output without exploding on federated metadata.

The fixture has 5 images, 3 categories, ~10 annotations. LVIS-specific
fields (``not_exhaustive_category_ids``, ``neg_category_ids``,
category ``frequency`` letters) are present on the GT JSON; the
COCO-shape evaluate path on this runner ignores them (per ADR-0026
the federated semantics layer in over the AP-fold core, and the
default runner takes the COCO surface — informational divergence is
expected on whole-LVIS comparisons but a tiny fixture's headline AP is
still well-defined).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from bench.harness.matrix import runner_module, uv_run_argv, uv_run_env
from bench.harness.paths import BENCH_ROOT
from tests.conftest import skip_if_no_env


def _lvis_inline_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Hand-rolled 5-image LVIS-shaped GT + perfect-DT. The LVIS schema
    extension is the per-image ``neg_category_ids`` / ``not_exhaustive``
    list and the per-category ``frequency`` letter; COCO-side evaluators
    ignore those, which is all the smoke needs."""
    images = [
        {
            "id": i,
            "width": 64,
            "height": 64,
            "file_name": f"{i:04d}.jpg",
            "coco_url": f"http://example/{i:04d}.jpg",
            "neg_category_ids": [],
            "not_exhaustive_category_ids": [],
        }
        for i in range(5)
    ]
    categories = [
        {"id": 1, "name": "obj_a", "frequency": "f", "image_count": 5, "instance_count": 5},
        {"id": 2, "name": "obj_b", "frequency": "c", "image_count": 3, "instance_count": 3},
        {"id": 3, "name": "obj_c", "frequency": "r", "image_count": 1, "instance_count": 1},
    ]
    annotations: list[dict[str, object]] = []
    ann_id = 1
    for img_idx in range(5):
        for cat_id in (1,) if img_idx >= 3 else (1, 2) if img_idx >= 1 else (1, 2, 3):
            x = 4 + cat_id * 4
            y = 4 + cat_id * 4
            sz = 16
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_idx,
                    "category_id": cat_id,
                    "bbox": [float(x), float(y), float(sz), float(sz)],
                    "area": float(sz * sz),
                    "iscrowd": 0,
                    "segmentation": [[x, y, x + sz, y, x + sz, y + sz, x, y + sz]],
                }
            )
            ann_id += 1

    gt = tmp_path / "lvis_v1_val_inline.json"
    gt.write_text(
        json.dumps({"images": images, "categories": categories, "annotations": annotations})
    )

    # GT-as-DT: every annotation re-emitted with score 1.0. Headline
    # AP must be 1.0 — anything else flags a runner regression.
    detections = [
        {
            "image_id": int(a["image_id"]),
            "category_id": int(a["category_id"]),
            "bbox": list(a["bbox"]),
            "score": 1.0,
            "segmentation": list(a["segmentation"]),
        }
        for a in annotations
    ]
    dt = tmp_path / "lvis_perfect_dt.json"
    dt.write_text(json.dumps(detections))
    return gt, dt


def test_vernier_runner_lvis_inline_smoke(tmp_path: Path) -> None:
    """The vernier runner must accept an LVIS-shaped GT through the
    standard COCO surface and produce a schema-valid JSON output. We
    assert AP=1.0 on perfect-DT; that wouldn't pin federated semantics
    but does confirm the bbox surface is reachable end-to-end."""
    skip_if_no_env("vernier")

    gt, dt = _lvis_inline_fixture(tmp_path)
    output = tmp_path / "vernier.json"
    tensor_output = tmp_path / "vernier.npy"

    cmd = uv_run_argv(
        BENCH_ROOT,
        "vernier",
        "-m",
        runner_module("vernier"),
        "--gt",
        str(gt),
        "--dt",
        str(dt),
        "--iou-type",
        "bbox",
        "--workload-id",
        "lvis_inline_smoke",
        "--output",
        str(output),
        "--tensor-output",
        str(tensor_output),
    )
    proc = subprocess.run(
        cmd, env=uv_run_env(BENCH_ROOT, "vernier"), check=False, capture_output=True
    )
    assert proc.returncode == 0, (
        f"vernier runner exited {proc.returncode}\n"
        f"stdout:\n{proc.stdout.decode(errors='replace')}\n"
        f"stderr:\n{proc.stderr.decode(errors='replace')}\n"
    )
    payload = json.loads(output.read_text())
    assert payload["impl"] == "vernier"
    assert payload["iou_type"] == "bbox"
    # Perfect-DT — every detection lines up with a GT, so AP = 1.0.
    assert payload["summary_stats"]["AP"] >= 0.999

    tensor = np.load(tensor_output)
    # Same (T, R, K, A, M) shape as COCO — LVIS-specific axis layout
    # only kicks in on the LVIS surface; this smoke uses the COCO surface.
    assert tensor.ndim == 5
