"""Zero-overhead microbenchmark for the default ``tables=None`` path.

Pins the contract that ``Evaluator().evaluate(gt, dt)`` (default) runs
within ``1 + tolerance`` of the underlying ``evaluate_bbox_summary``
FFI on the same fixture. ``min``-reduced over N samples; xfails if the
baseline is too small for stable timing on the runner.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

import vernier.panoptic as vp
import vernier.semantic as vs
from vernier._core import (
    evaluate_bbox_summary,
    evaluate_panoptic,
    evaluate_semantic_from_arrays,
)
from vernier.instance import Evaluator


# 16 images x 4 categories — heavy enough for the timer to see the
# Python wrapper overhead, light enough not to slow CI.
def _build_workload() -> tuple[bytes, bytes]:
    images = [{"id": i, "width": 200, "height": 200} for i in range(1, 17)]
    annotations = []
    detections = []
    aid = 1
    for img in images:
        for cat in range(1, 5):
            x = (aid % 5) * 30
            y = ((aid // 5) % 5) * 30
            annotations.append(
                {
                    "id": aid,
                    "image_id": img["id"],
                    "category_id": cat,
                    "bbox": [x, y, 20, 20],
                    "area": 400,
                    "iscrowd": 0,
                }
            )
            detections.append(
                {
                    "image_id": img["id"],
                    "category_id": cat,
                    "score": 0.5 + (aid % 50) * 0.01,
                    # Slight DT jitter so matching does work, not just trivial overlap.
                    "bbox": [x + 1, y + 1, 20, 20],
                }
            )
            aid += 1
    gt = json.dumps(
        {
            "images": images,
            "annotations": annotations,
            "categories": [{"id": c, "name": f"cat{c}"} for c in range(1, 5)],
        }
    ).encode()
    dt = json.dumps(detections).encode()
    return gt, dt


_GT, _DT = _build_workload()
_MAX_DETS = [1, 10, 100]
_PARITY = "corrected"
_USE_CATS = True

# 5% — tight enough to catch a meaningful regression but wide enough
# to survive Python interpreter variance on small fixtures.
_RATIO_TOLERANCE = 1.05
_N_SAMPLES = 50
_MIN_BASELINE_NS = 100_000  # 0.1 ms — below this, timing is too noisy


def _bench(call) -> int:
    start = time.perf_counter_ns()
    call()
    return time.perf_counter_ns() - start


def test_evaluator_evaluate_default_path_within_tolerance_of_direct_ffi() -> None:
    """``Evaluator().evaluate(gt, dt)`` (default: tables=None) wall
    time / direct ``evaluate_bbox_summary`` ≤ tolerance, ``min``-reduced."""

    # Warm up — JIT-style first-call costs (lazy class init, allocator
    # priming) shouldn't bias the comparison.
    for _ in range(5):
        evaluate_bbox_summary(_GT, _DT, _PARITY, _MAX_DETS, _USE_CATS)
        Evaluator().evaluate(_GT, _DT)

    direct_samples = [
        _bench(lambda: evaluate_bbox_summary(_GT, _DT, _PARITY, _MAX_DETS, _USE_CATS))
        for _ in range(_N_SAMPLES)
    ]
    wrapped_samples = [_bench(lambda: Evaluator().evaluate(_GT, _DT)) for _ in range(_N_SAMPLES)]

    direct_min = min(direct_samples)
    wrapped_min = min(wrapped_samples)

    if direct_min < _MIN_BASELINE_NS:
        pytest.xfail(
            f"baseline {direct_min} ns < {_MIN_BASELINE_NS} ns — fixture too "
            f"small for stable ratio timing on this runner"
        )

    ratio = wrapped_min / direct_min
    assert ratio <= _RATIO_TOLERANCE, (
        f"Evaluator().evaluate default path is {ratio:.3f}x direct FFI "
        f"(direct={direct_min} ns, wrapped={wrapped_min} ns) — exceeds "
        f"tolerance {_RATIO_TOLERANCE:.3f}x"
    )


def _build_panoptic_workload() -> tuple[vp.Dataset, vp.Predictions, str]:
    """8-image panoptic workload, 32x32 per image with two segments
    each. Sized so both the kernel walk and the per-image attribute
    fold push past the 100 us noise floor."""
    label_maps_gt: dict[int, np.ndarray] = {}
    label_maps_dt: dict[int, np.ndarray] = {}
    segs_gt: dict[str, list[dict]] = {}
    segs_dt: dict[str, list[dict]] = {}
    for img_id in range(1, 9):
        gt = np.zeros((32, 32), dtype=np.uint32)
        gt[:, :16] = 1
        gt[:, 16:] = 2
        dt = np.zeros((32, 32), dtype=np.uint32)
        dt[:, :16] = 10
        dt[:, 16:] = 11
        label_maps_gt[img_id] = gt
        label_maps_dt[img_id] = dt
        segs_gt[str(img_id)] = [
            {"id": 1, "category_id": 100, "iscrowd": False, "area": 512},
            {"id": 2, "category_id": 200, "iscrowd": False, "area": 512},
        ]
        segs_dt[str(img_id)] = [
            {"id": 10, "category_id": 100, "iscrowd": False, "area": 512},
            {"id": 11, "category_id": 200, "iscrowd": False, "area": 512},
        ]
    cats = json.dumps([{"id": 100, "isthing": True}, {"id": 200, "isthing": False}]).encode()
    gt = vp.Dataset.from_arrays(label_maps_gt, json.dumps(segs_gt).encode(), cats)
    dt = vp.Predictions.from_arrays(label_maps_dt, json.dumps(segs_dt).encode())
    return gt, dt, "corrected"


def test_panoptic_evaluate_default_path_within_tolerance_of_direct_ffi() -> None:
    """``vp.Evaluator().evaluate(gt, dt)`` (no ``tables=``) wall time
    vs. the direct ``evaluate_panoptic`` FFI. Pins the ADR-0038
    zero-overhead invariant on the panoptic paradigm."""
    gt, dt, parity = _build_panoptic_workload()

    for _ in range(5):
        evaluate_panoptic(gt, dt, parity, True)
        vp.Evaluator().evaluate(gt, dt)

    direct_samples = [
        _bench(lambda: evaluate_panoptic(gt, dt, parity, True)) for _ in range(_N_SAMPLES)
    ]
    wrapped_samples = [_bench(lambda: vp.Evaluator().evaluate(gt, dt)) for _ in range(_N_SAMPLES)]

    direct_min = min(direct_samples)
    wrapped_min = min(wrapped_samples)

    if direct_min < _MIN_BASELINE_NS:
        pytest.xfail(
            f"panoptic baseline {direct_min} ns < {_MIN_BASELINE_NS} ns — "
            f"fixture too small for stable ratio timing on this runner"
        )

    ratio = wrapped_min / direct_min
    assert ratio <= _RATIO_TOLERANCE, (
        f"vp.Evaluator().evaluate default path is {ratio:.3f}x direct FFI "
        f"(direct={direct_min} ns, wrapped={wrapped_min} ns) — exceeds "
        f"tolerance {_RATIO_TOLERANCE:.3f}x"
    )


def _build_semantic_workload() -> tuple[vs.Dataset, vs.Predictions, str]:
    """8-image semantic workload, 4 classes, 32x32 per image. Same
    shape rationale as the panoptic workload above."""
    label_maps_gt: dict[int, np.ndarray] = {}
    label_maps_dt: dict[int, np.ndarray] = {}
    rng = np.random.default_rng(seed=42)
    for img_id in range(1, 9):
        gt = rng.integers(0, 4, size=(32, 32), dtype=np.uint32)
        # DT is GT with ~20% of pixels perturbed to a neighbor class.
        dt = gt.copy()
        mask = rng.random(size=gt.shape) < 0.2
        dt[mask] = (dt[mask] + 1) % 4
        label_maps_gt[img_id] = gt
        label_maps_dt[img_id] = dt
    return (
        vs.Dataset.from_arrays(label_maps_gt, n_classes=4),
        vs.Predictions.from_arrays(label_maps_dt),
        "corrected",
    )


def test_semantic_evaluate_default_path_within_tolerance_of_direct_ffi() -> None:
    """``vs.Evaluator().evaluate(gt, dt)`` (no ``tables=``) wall time
    vs. the direct ``evaluate_semantic_from_arrays`` FFI. Pins the
    ADR-0038 zero-overhead invariant on the semantic paradigm and
    implicitly guards ADR-0037's fused decode+fold contract."""
    gt, dt, parity = _build_semantic_workload()

    def _direct() -> object:
        return evaluate_semantic_from_arrays(
            dict(gt.label_maps),
            dict(dt.label_maps),
            n_classes=gt.n_classes,
            parity_mode=parity,
            ignore_label=gt.ignore_label,
            label_remap=None,
        )

    for _ in range(5):
        _direct()
        vs.Evaluator().evaluate(gt, dt)

    direct_samples = [_bench(_direct) for _ in range(_N_SAMPLES)]
    wrapped_samples = [_bench(lambda: vs.Evaluator().evaluate(gt, dt)) for _ in range(_N_SAMPLES)]

    direct_min = min(direct_samples)
    wrapped_min = min(wrapped_samples)

    if direct_min < _MIN_BASELINE_NS:
        pytest.xfail(
            f"semantic baseline {direct_min} ns < {_MIN_BASELINE_NS} ns — "
            f"fixture too small for stable ratio timing on this runner"
        )

    ratio = wrapped_min / direct_min
    assert ratio <= _RATIO_TOLERANCE, (
        f"vs.Evaluator().evaluate default path is {ratio:.3f}x direct FFI "
        f"(direct={direct_min} ns, wrapped={wrapped_min} ns) — exceeds "
        f"tolerance {_RATIO_TOLERANCE:.3f}x"
    )
