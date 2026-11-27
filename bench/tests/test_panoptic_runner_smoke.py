"""Smoke test for the ``hotcoco_panoptic`` runner vs ``vernier_panoptic``."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from bench.harness.matrix import runner_module, uv_run_argv, uv_run_env
from bench.harness.parity import PANOPTIC_PARITY_EPS, PanopticSnapshot, get_comparator
from bench.harness.paths import BENCH_ROOT
from tests.conftest import skip_if_no_env


def _rgb_png(path: Path, segment_id: int, height: int = 32, width: int = 32) -> None:
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[..., 0] = segment_id & 0xFF
    arr[..., 1] = (segment_id >> 8) & 0xFF
    arr[..., 2] = (segment_id >> 16) & 0xFF
    Image.fromarray(arr, mode="RGB").save(path)


def _panoptic_inline_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, Path]:
    gt_png_dir = tmp_path / "gt_png"
    dt_png_dir = tmp_path / "dt_png"
    gt_png_dir.mkdir()
    dt_png_dir.mkdir()

    # Include one thing (id=1), one stuff (id=2), and one unused category (id=3)
    # to verify unevaluated categories are omitted from per_class.
    categories = [
        {"id": 1, "name": "person", "supercategory": "person", "isthing": 1},
        {"id": 2, "name": "sky", "supercategory": "sky", "isthing": 0},
        {"id": 3, "name": "unused", "supercategory": "other", "isthing": 1},
    ]
    categories_json = tmp_path / "categories.json"
    categories_json.write_text(
        json.dumps([{**c, "isthing": bool(c["isthing"])} for c in categories])
    )

    _rgb_png(gt_png_dir / "0001.png", 101)
    _rgb_png(dt_png_dir / "0001.png", 101)
    _rgb_png(gt_png_dir / "0002.png", 202)
    _rgb_png(dt_png_dir / "0002.png", 202)

    images = [
        {"id": 1, "width": 32, "height": 32, "file_name": "0001.jpg"},
        {"id": 2, "width": 32, "height": 32, "file_name": "0002.jpg"},
    ]
    annotations = [
        {
            "image_id": 1,
            "file_name": "0001.png",
            "segments_info": [{"id": 101, "category_id": 1, "area": 1024, "iscrowd": 0}],
        },
        {
            "image_id": 2,
            "file_name": "0002.png",
            "segments_info": [{"id": 202, "category_id": 2, "area": 1024, "iscrowd": 0}],
        },
    ]

    gt_json = tmp_path / "gt.json"
    gt_json.write_text(
        json.dumps({"images": images, "categories": categories, "annotations": annotations})
    )
    dt_json = tmp_path / "dt.json"
    dt_json.write_text(json.dumps({"annotations": annotations}))
    return gt_png_dir, gt_json, dt_png_dir, dt_json, categories_json


def _run_panoptic_runner(
    impl: str,
    gt_png_dir: Path,
    gt_json: Path,
    dt_png_dir: Path,
    dt_json: Path,
    categories_json: Path,
    out_dir: Path,
) -> tuple[dict[str, Any], PanopticSnapshot, np.ndarray]:
    output = out_dir / f"{impl}.json"
    snapshot_output = out_dir / f"{impl}.snapshot.json"
    per_class_output = out_dir / f"{impl}_per_class.npy"
    cmd = uv_run_argv(
        BENCH_ROOT,
        impl,
        "-m",
        runner_module(impl),
        "--gt-png-dir",
        str(gt_png_dir),
        "--gt-json",
        str(gt_json),
        "--dt-png-dir",
        str(dt_png_dir),
        "--dt-json",
        str(dt_json),
        "--categories-json",
        str(categories_json),
        "--workload-id",
        "panoptic_inline_smoke",
        "--paradigm",
        "panoptic",
        "--output",
        str(output),
        "--snapshot-output",
        str(snapshot_output),
        "--per-class-output",
        str(per_class_output),
    )
    proc = subprocess.run(cmd, env=uv_run_env(BENCH_ROOT, impl), check=False, capture_output=True)
    assert proc.returncode == 0, (
        f"{impl} runner exited {proc.returncode}\n"
        f"stdout:\n{proc.stdout.decode(errors='replace')}\n"
        f"stderr:\n{proc.stderr.decode(errors='replace')}\n"
    )
    return (
        json.loads(output.read_text()),
        PanopticSnapshot.model_validate_json(snapshot_output.read_bytes()),
        np.load(per_class_output),
    )


def test_hotcoco_panoptic_runner_inline_smoke(tmp_path: Path) -> None:
    skip_if_no_env("hotcoco")
    skip_if_no_env("vernier_panoptic")

    gt_png, gt_json, dt_png, dt_json, cats_json = _panoptic_inline_fixture(tmp_path)
    payload, snap, per_class = _run_panoptic_runner(
        "hotcoco_panoptic", gt_png, gt_json, dt_png, dt_json, cats_json, tmp_path
    )
    assert payload["impl"] == "hotcoco_panoptic"
    assert payload["paradigm"] == "panoptic"
    assert abs(snap.pq - 1.0) <= PANOPTIC_PARITY_EPS
    assert set(snap.per_class) == {"1", "2"}
    assert per_class.shape == (2, 3)

    _, ref_snap, ref_per_class = _run_panoptic_runner(
        "vernier_panoptic", gt_png, gt_json, dt_png, dt_json, cats_json, tmp_path
    )
    assert per_class.shape == ref_per_class.shape
    report = get_comparator("panoptic").compare(
        workload_id="panoptic_inline_smoke",
        iou_type="pq",
        impl_outputs={"vernier_panoptic": ref_snap, "hotcoco_panoptic": snap},
    )
    assert report.passed, report.model_dump_json(indent=2)
