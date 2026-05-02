"""Zero-overhead microbenchmark for the default ``tables=None`` path.

Pins the contract that ``Evaluator().evaluate(gt, dt)`` (default) runs
within ``1 + tolerance`` of the underlying ``evaluate_bbox_summary``
FFI on the same fixture. ``min``-reduced over N samples; xfails if the
baseline is too small for stable timing on the runner.
"""

from __future__ import annotations

import json
import time

import pytest

from vernier import Evaluator
from vernier._core import evaluate_bbox_summary


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
