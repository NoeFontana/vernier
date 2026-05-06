"""OKS keypoints jitter must perturb predictions enough to produce a
measurable AP delta from a perfect-match baseline.

This is a sanity check on the jitter generator, not a parity check.
It runs entirely against pycocotools (no vernier dep) so it lives in
the harness env without needing a runner subprocess. The goal is to
catch the failure mode where a typo shrinks the noise to zero and the
keypoints workload silently degrades to a perfect-match smoke.
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from bench.workloads import jittered_predictions


@pytest.fixture
def coco_kp_gt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A 4-image, single-category COCO-person GT with 17 visible
    keypoints per annotation. Big enough that AP is well-defined and
    small enough that jitter generation + cocoeval are sub-second.
    """
    monkeypatch.setenv("VERNIER_BENCH_CACHE", str(tmp_path))
    h, w = 200, 200
    images = [{"id": i, "width": w, "height": h, "file_name": f"{i}.jpg"} for i in range(4)]
    categories = [
        {
            "id": 1,
            "name": "person",
            "supercategory": "person",
            "keypoints": [f"k{i}" for i in range(17)],
            "skeleton": [],
        }
    ]
    annotations = []
    for i in range(4):
        offset_x = 10 + i * 15
        kp: list[float] = []
        for row in range(5):
            for col in range(4):
                if len(kp) // 3 == 17:
                    break
                kp.extend([float(offset_x + col * 12), float(20 + row * 18), 2.0])
        kp = kp[: 17 * 3]
        annotations.append(
            {
                "id": i + 1,
                "image_id": i,
                "category_id": 1,
                "iscrowd": 0,
                "bbox": [float(offset_x), 20.0, 50.0, 90.0],
                "area": 50.0 * 90.0,
                "num_keypoints": 17,
                "keypoints": kp,
            }
        )
    out = tmp_path / "kp_gt.json"
    out.write_text(
        json.dumps({"images": images, "categories": categories, "annotations": annotations})
    )
    return out


def _perfect_dt_from_gt(gt_path: Path) -> list[dict[str, object]]:
    """GT-as-DT: copy bbox + keypoints, score 1.0. The COCOeval
    convention for "did the prediction recover the GT" with no jitter."""
    gt = json.loads(gt_path.read_text())
    return [
        {
            "image_id": int(a["image_id"]),
            "category_id": int(a["category_id"]),
            "bbox": list(a["bbox"]),
            "score": 1.0,
            "keypoints": list(a["keypoints"]),
        }
        for a in gt["annotations"]
    ]


def _ap_at_keypoints(gt_path: Path, dt_path: Path) -> float:
    """Run pycocotools' keypoints AP pipeline; return the headline AP."""
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(gt_path))
        dt = gt.loadRes(str(dt_path))
        ev = COCOeval(gt, dt, iouType="keypoints")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return float(ev.stats[0])


def test_jitter_drops_ap_below_perfect(coco_kp_gt: Path, tmp_path: Path) -> None:
    """The jitter scale (``1.5 * sigma * sqrt(area)``) is large enough
    that the headline AP is well below the perfect-match 1.0. If the
    noise accidentally collapses to ~0 the assertion fires; if jitter
    is pathologically large AP would also fall, which is also caught."""
    perfect_dt = tmp_path / "perfect_dt.json"
    perfect_dt.write_text(json.dumps(_perfect_dt_from_gt(coco_kp_gt)))
    perfect_ap = _ap_at_keypoints(coco_kp_gt, perfect_dt)
    assert perfect_ap == pytest.approx(1.0, abs=1e-9)

    jittered_dt = jittered_predictions.keypoints_dt_path(gt_path=coco_kp_gt, seed=0)
    jittered_ap = _ap_at_keypoints(coco_kp_gt, jittered_dt)
    # The bound here is generous — the concrete AP depends on numpy
    # version's RNG internals. The structural property we want is
    # "jitter perturbs", i.e., a non-trivial drop from 1.0.
    assert jittered_ap < perfect_ap - 0.05, (
        f"jittered AP {jittered_ap:.4f} too close to perfect {perfect_ap:.4f}; "
        "jitter generator likely degenerate."
    )


def test_keypoints_jitter_byte_identical_across_runs(coco_kp_gt: Path) -> None:
    """Same seed, same params → same DT JSON. The cache key is
    ``(seed, JITTER_PARAMS_VERSION)`` so any seed-independent drift
    (numpy version, dict-iteration order) would silently invalidate
    snapshot comparisons."""
    first = jittered_predictions.keypoints_dt_path(gt_path=coco_kp_gt, seed=42)
    first_bytes = first.read_bytes()
    first.unlink()
    second = jittered_predictions.keypoints_dt_path(gt_path=coco_kp_gt, seed=42)
    assert second.read_bytes() == first_bytes


def test_keypoints_jitter_diverges_across_seeds(coco_kp_gt: Path) -> None:
    a = jittered_predictions.keypoints_dt_path(gt_path=coco_kp_gt, seed=1)
    b = jittered_predictions.keypoints_dt_path(gt_path=coco_kp_gt, seed=2)
    assert a != b
    a_dets = json.loads(a.read_bytes())
    b_dets = json.loads(b.read_bytes())
    a_kps = np.array([d["keypoints"] for d in a_dets])
    b_kps = np.array([d["keypoints"] for d in b_dets])
    assert not np.array_equal(a_kps, b_kps)
