"""Zero-overhead microbenchmark for the default ``tables=None`` path.

Pins the contract that the pure-Python ``Evaluator().evaluate`` wrapper
adds no material per-call cost over the FFI entry point it dispatches
to, on every paradigm that has one.

The measurement is a wall-clock comparison on a shared CI runner, so the
estimator is built to survive scheduler noise:

* **Three interleaved streams, ``min``-reduced.** Each round times the
  direct FFI, the wrapper, and the direct FFI *a second time*, rotating
  which goes first. Timing noise is one-sided — it can only make a
  sample slower — so the per-stream minimum is the best available
  estimate of true cost. Interleaving matters as much as the ``min``:
  measuring 40 direct calls and *then* 40 wrapped calls lets a
  contention episode that straddles only the second block inflate one
  side of the ratio wholesale.
* **A null control.** The second direct stream measures exactly the same
  work as the first, so ``|first - second|`` is this runner's own
  measurement noise, observed during this run. The failure threshold is
  scaled off it — bounded above, so a noisy runner widens the threshold
  but can never widen it past what a real regression costs — which is
  what makes one threshold work on a quiet laptop and a contended
  2-core runner alike.
* **Both an absolute and a relative arm.** The wrapper's real cost is a
  fixed handful of microseconds of Python, while scheduler noise is also
  absolute — at a ~200 us baseline a single 20 us hiccup is a 10% ratio
  swing. So the gate fails only when the ratio exceeds the tolerance
  *and* the absolute gap exceeds the noise-derived slack. A regression
  that doubles the wrapper trips both by an order of magnitude; a
  hiccup trips neither.
* **GC held off during the window.** Collection is one-sided noise that
  lands preferentially on the allocating (wrapper) stream.
* **Re-measurement before failing.** A regression reproduces; a
  contention episode does not. The gate fails only if every attempt
  trips it, which costs nothing on the happy path and widens no
  threshold.

The fixtures are sized so the baseline lands in the low hundreds of
microseconds on a dev box. That is deliberate: the wrapper's cost is
fixed, so on a fixture small enough to run in ~20 us it is worth ~5% of
the baseline all by itself and no ratio tolerance can separate signal
from fixed cost.
"""

from __future__ import annotations

import gc
import json
import time
from collections.abc import Callable

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

# 5% — the wrapper's measured cost is ~1-3 us against baselines in the
# 150-300 us band (well under 2%), so this leaves better than a 2x
# margin over the real signal while staying far below the 2x a genuine
# regression would produce.
_RATIO_TOLERANCE = 1.05
_N_SAMPLES = 40
_MIN_BASELINE_NS = 100_000  # 0.1 ms — below this, timing is too noisy

# Absolute arm. The ratio alone is not a usable gate at these baselines:
# a single scheduler hiccup is tens of microseconds regardless of how
# long the call takes. Slack is the larger of a fixed floor and a
# multiple of the noise the null control actually observed this run.
#
# The floor is set from measured noise, not from what makes the test
# pass: on a loaded 8-core box, `min`-of-40 estimates of *identical*
# work drift by up to ~20 us, and the CI false positive this gate was
# rewritten for was a 32 us gap on a ~280 us baseline. 25 us covers
# that, and is still an order of magnitude below the ~150-300 us gap a
# doubled wrapper produces.
_ABS_SLACK_FLOOR_NS = 25_000
_NOISE_SLACK_FACTOR = 3

# ...and a ceiling on that slack, because the noise term is otherwise
# unbounded: one load spike big enough to put `3 * noise` past the whole
# baseline would let a wrapper that *doubled* the cost sail through. The
# slack may never reach a quarter of the baseline, so a regression that
# costs anything like a full extra call is always out of reach of it.
# At the `_MIN_BASELINE_NS` floor this ceiling coincides exactly with
# `_ABS_SLACK_FLOOR_NS`, so the two bounds meet rather than cross.
# `test_gate_detects_a_doubled_wrapper` is what caught the missing cap.
_MAX_SLACK_FRACTION = 0.25

# A regression is reproducible; a contention episode is not. Re-measure
# before failing and require every attempt to trip the gate. This costs
# nothing on the happy path (the first attempt returns) and it does not
# widen the threshold by a nanosecond — a doubled wrapper trips all
# three attempts, every time.
_MAX_ATTEMPTS = 3


def _bench(call: Callable[[], object]) -> int:
    start = time.perf_counter_ns()
    call()
    return time.perf_counter_ns() - start


def _interleaved_minima(
    direct: Callable[[], object],
    wrapped: Callable[[], object],
) -> tuple[int, int, int]:
    """Time ``direct``, ``wrapped`` and ``direct`` again, interleaved.

    Returns ``(direct_min, wrapped_min, control_min)``. ``control_min``
    is a second, independent ``min``-reduced estimate of ``direct`` —
    the null measurement. Every stream gets the same sample count and
    the same share of the measurement window, so no stream is handed a
    better estimator than the others.
    """
    # Warm up — JIT-style first-call costs (lazy class init, allocator
    # priming) shouldn't bias the comparison.
    for _ in range(5):
        direct()
        wrapped()

    direct_samples: list[int] = []
    wrapped_samples: list[int] = []
    control_samples: list[int] = []
    # GC is one-sided noise and lands preferentially on the stream that
    # allocates most, which is the wrapper. Freeze it for the window.
    gc_was_enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        for i in range(_N_SAMPLES):
            # Rotate stream order so no stream permanently owns the
            # cache-warm or cache-cold slot in the round.
            if i % 3 == 0:
                direct_samples.append(_bench(direct))
                wrapped_samples.append(_bench(wrapped))
                control_samples.append(_bench(direct))
            elif i % 3 == 1:
                wrapped_samples.append(_bench(wrapped))
                control_samples.append(_bench(direct))
                direct_samples.append(_bench(direct))
            else:
                control_samples.append(_bench(direct))
                direct_samples.append(_bench(direct))
                wrapped_samples.append(_bench(wrapped))
    finally:
        if gc_was_enabled:
            gc.enable()

    return min(direct_samples), min(wrapped_samples), min(control_samples)


def _overhead_verdict(
    label: str,
    direct: Callable[[], object],
    wrapped: Callable[[], object],
) -> str | None:
    """One measurement. Returns ``None`` when the wrapper is within
    budget, else the explanation of why it is not."""
    direct_min, wrapped_min, control_min = _interleaved_minima(direct, wrapped)

    if direct_min < _MIN_BASELINE_NS:
        # `skip`, not `xfail`: the gate did not run, which is not the
        # same as "ran and was expected to fail". With the current
        # fixture sizes this branch should be unreachable — if it fires,
        # the fixture needs resizing, not the report ignoring.
        pytest.skip(
            f"{label} baseline {direct_min} ns < {_MIN_BASELINE_NS} ns — fixture "
            f"too small for stable ratio timing on this runner"
        )

    ratio = wrapped_min / direct_min
    delta_ns = wrapped_min - direct_min
    # Two `min`-reduced estimates of identical work: whatever separates
    # them is noise, by construction.
    noise_ns = abs(control_min - direct_min)
    slack_ns = min(
        max(_ABS_SLACK_FLOOR_NS, _NOISE_SLACK_FACTOR * noise_ns),
        int(_MAX_SLACK_FRACTION * direct_min),
    )

    if ratio <= _RATIO_TOLERANCE or delta_ns <= slack_ns:
        return None
    return (
        f"{label} default path is {ratio:.3f}x direct FFI and {delta_ns} ns "
        f"slower in absolute terms (direct={direct_min} ns, "
        f"wrapped={wrapped_min} ns) — exceeds both the {_RATIO_TOLERANCE:.3f}x "
        f"tolerance and the {slack_ns} ns noise slack "
        f"(control={control_min} ns, observed noise={noise_ns} ns)"
    )


def _assert_no_wrapper_overhead(
    label: str,
    direct: Callable[[], object],
    wrapped: Callable[[], object],
) -> None:
    """Fail iff the wrapper is both relatively *and* absolutely slower
    than the FFI it dispatches to, beyond this runner's own noise — and
    stays that way across every re-measurement."""
    verdicts: list[str] = []
    for _ in range(_MAX_ATTEMPTS):
        verdict = _overhead_verdict(label, direct, wrapped)
        if verdict is None:
            return
        verdicts.append(verdict)

    joined = "\n".join(f"  attempt {i + 1}: {v}" for i, v in enumerate(verdicts))
    raise AssertionError(f"{label} exceeded the overhead budget on every attempt:\n{joined}")


def test_evaluator_evaluate_default_path_within_tolerance_of_direct_ffi() -> None:
    """``Evaluator().evaluate(gt, dt)`` (default: tables=None) vs. the
    direct ``evaluate_bbox_summary`` FFI."""
    _assert_no_wrapper_overhead(
        "Evaluator().evaluate",
        lambda: evaluate_bbox_summary(_GT, _DT, _PARITY, _MAX_DETS, _USE_CATS),
        lambda: Evaluator().evaluate(_GT, _DT),
    )


def test_gate_detects_a_doubled_wrapper() -> None:
    """The gate itself must be able to fail.

    Guards against the whole file decaying into a no-op — which is
    exactly what the ``_MIN_BASELINE_NS`` xfail did to the panoptic and
    semantic variants while their fixtures were too small to clear it.
    A "wrapper" that runs the workload twice is the cheapest honest
    stand-in for a 2x regression.
    """

    def direct() -> object:
        return evaluate_bbox_summary(_GT, _DT, _PARITY, _MAX_DETS, _USE_CATS)

    def doubled() -> None:
        direct()
        direct()

    with pytest.raises(AssertionError, match="exceeds both"):
        _assert_no_wrapper_overhead("doubled-wrapper control", direct, doubled)


def _build_panoptic_workload() -> tuple[vp.Dataset, vp.Predictions, str]:
    """12-image panoptic workload, 128x128 per image with two segments
    each.

    Sized so the direct FFI lands around 200 us on a dev box. The old
    8x32x32 fixture ran in ~18 us, where the wrapper's fixed ~1 us of
    Python is 5% of the baseline all by itself — the ratio gate could
    not have passed on merit, and only ever "passed" by xfailing under
    ``_MIN_BASELINE_NS``.
    """
    label_maps_gt: dict[int, np.ndarray] = {}
    label_maps_dt: dict[int, np.ndarray] = {}
    segs_gt: dict[str, list[dict]] = {}
    segs_dt: dict[str, list[dict]] = {}
    side = 128
    half = side // 2
    area = half * side
    for img_id in range(1, 13):
        gt = np.zeros((side, side), dtype=np.uint32)
        gt[:, :half] = 1
        gt[:, half:] = 2
        dt = np.zeros((side, side), dtype=np.uint32)
        dt[:, :half] = 10
        dt[:, half:] = 11
        label_maps_gt[img_id] = gt
        label_maps_dt[img_id] = dt
        segs_gt[str(img_id)] = [
            {"id": 1, "category_id": 100, "iscrowd": False, "area": area},
            {"id": 2, "category_id": 200, "iscrowd": False, "area": area},
        ]
        segs_dt[str(img_id)] = [
            {"id": 10, "category_id": 100, "iscrowd": False, "area": area},
            {"id": 11, "category_id": 200, "iscrowd": False, "area": area},
        ]
    cats = json.dumps([{"id": 100, "isthing": True}, {"id": 200, "isthing": False}]).encode()
    gt = vp.Dataset.from_arrays(label_maps_gt, json.dumps(segs_gt).encode(), cats)
    dt = vp.Predictions.from_arrays(label_maps_dt, json.dumps(segs_dt).encode())
    return gt, dt, "corrected"


def test_panoptic_evaluate_default_path_within_tolerance_of_direct_ffi() -> None:
    """``vp.Evaluator().evaluate(gt, dt)`` (no ``tables=``) vs. the
    direct ``evaluate_panoptic`` FFI. Pins the ADR-0038 zero-overhead
    invariant on the panoptic paradigm."""
    gt, dt, parity = _build_panoptic_workload()
    _assert_no_wrapper_overhead(
        "vp.Evaluator().evaluate",
        lambda: evaluate_panoptic(gt, dt, parity, True),
        lambda: vp.Evaluator().evaluate(gt, dt),
    )


def _build_semantic_workload() -> tuple[vs.Dataset, vs.Predictions, str]:
    """16-image semantic workload, 4 classes, 64x64 per image. Same
    sizing rationale as the panoptic workload above (the old 8x32x32
    fixture ran in ~24 us and never cleared ``_MIN_BASELINE_NS``)."""
    label_maps_gt: dict[int, np.ndarray] = {}
    label_maps_dt: dict[int, np.ndarray] = {}
    rng = np.random.default_rng(seed=42)
    for img_id in range(1, 17):
        gt = rng.integers(0, 4, size=(64, 64), dtype=np.uint32)
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
    """``vs.Evaluator().evaluate(gt, dt)`` (no ``tables=``) vs. the
    direct ``evaluate_semantic_from_arrays`` FFI. Pins the ADR-0038
    zero-overhead invariant on the semantic paradigm and implicitly
    guards ADR-0037's fused decode+fold contract."""
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

    _assert_no_wrapper_overhead(
        "vs.Evaluator().evaluate",
        _direct,
        lambda: vs.Evaluator().evaluate(gt, dt),
    )
