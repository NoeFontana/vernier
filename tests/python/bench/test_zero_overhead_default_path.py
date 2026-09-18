"""Zero-overhead microbenchmark for the default ``tables=None`` path.

Pins the contract that the pure-Python ``Evaluator().evaluate`` wrapper
adds no material per-call cost over the FFI entry point it dispatches
to, on every paradigm that has one.

**What the gate asserts.** The wrapper's *absolute* added wall-clock per
call must stay under a fixed budget. It is deliberately not a ratio.
The wrapper is a fixed handful of microseconds of Python; that cost is
invariant under the Rust build profile, while the FFI baseline it would
be divided by is not. CI builds a **debug** wheel
(``.github/workflows/ci.yml``: ``maturin build`` with no ``--release``,
installed by the ``test-py`` job), where these fixtures clock ~9600 us
(instance), ~4000 us (panoptic) and ~1700 us (semantic) — 10-34x their
release figures. A percentage gate calibrated on release numbers
therefore means something 10-34x weaker in the job that actually runs
it, which is how this file previously ended up unable to see a
regression roughly 100x larger than the signal it exists to detect.
An absolute budget carries the same meaning into both profiles; where
the debug baseline is too large for the instrument to resolve it, a
stated precision floor — not a percentage tolerance — takes over. See
"Sensitivity" below for the exact numbers.

The measurement is a wall-clock comparison on a shared CI runner, so
the estimator is built to survive scheduler noise:

* **Three interleaved streams, ``min``-reduced.** Each round times the
  direct FFI, the wrapper, and the direct FFI *a second time*, rotating
  which goes first. Timing noise is one-sided — it can only make a
  sample slower — so the per-stream minimum is the best available
  estimate of true cost. Interleaving matters as much as the ``min``:
  measuring 42 direct calls and *then* 42 wrapped calls lets a
  contention episode that straddles only the second block inflate one
  side of the ratio wholesale.
* **A null control.** The second direct stream measures exactly the same
  work as the first, so ``|first - second|`` is this runner's own
  measurement noise, observed during this run. The slack is scaled off
  it — bounded above, so a noisy runner widens the slack but can never
  widen it past what a real regression costs — which is what makes one
  threshold work on a quiet laptop and a contended 2-core runner alike.
* **A rotation that balances predecessors, not just positions.** Only
  the wrapper allocates, so only the wrapper perturbs the *next* call's
  allocator and cache state. A rotation that merely balances each
  stream's position within the round does not balance what ran
  immediately before it, and the resulting bias points the permissive
  way on both terms at once (wrapper never preceded by itself ->
  ``wrapped_min`` low; control preceded by the wrapper twice as often
  as ``direct`` -> ``noise`` and hence the slack too wide). The round
  order here walks an Eulerian circuit over the six permutations, so
  across one full cycle every stream sits in every position exactly
  twice *and* is preceded by every stream (including itself) exactly
  twice. ``_N_SAMPLES`` is a multiple of 6 so the cycle always closes.
* **GC held off during the window.** Collection is one-sided noise that
  lands preferentially on the allocating (wrapper) stream.
* **Re-measurement before failing.** A regression reproduces; a
  contention episode does not. The gate fails only if every attempt
  trips it, which costs nothing on the happy path and widens no
  threshold.

**Sensitivity, stated honestly.** The gate fires when the wrapper's
added per-call cost exceeds

    max(25 us, 2% of the baseline, 3 x the noise this run measured)

capped at 25% of the baseline. The 25 us is the budget; the 2% is the
relative floor, which must sit above the wrapper's own proportional
cost (measured at ~1% — see ``_PRECISION_FLOOR_FRACTION``) or it fires
on correct code; the noise term reacts to this run's contention; the
cap keeps the noise term from ever reaching the size of a real
regression. Concretely, per paradigm:

=========  ================  =================  ================
paradigm   release baseline  detects (release)  detects (debug)
=========  ================  =================  ================
instance   ~280 us           25 us  (8.9%)      2% of baseline
panoptic   ~210 us           25 us  (11.9%)     2% of baseline
semantic   ~170 us           25 us  (14.7%)     2% of baseline
=========  ================  =================  ================

The debug column is a fraction rather than a number because CI's debug
baseline is not fixed: it has been measured between 7.8 ms and 15.5 ms
on the same fixture, and the floor tracks it.

Three things follow, and all three are worth saying out loud:

* **Release sensitivity is unchanged** by making the gate absolute. It
  was already the absolute arm that bound at every release baseline —
  25 us is 8.9-14.7% of them, so the old 5% ratio arm was never the
  operative condition and the "5% tolerance" was never the effective
  one. The effective tolerance was, and is, 25 us.
* **Debug sensitivity is 2.5x better than the 5% ratio** this gate
  replaced, and is bounded below by the wrapper's own ~1% cost: no
  floor beneath that can pass a correct wrapper, whatever it would do
  for sensitivity.
* The worst-case guarantee under arbitrarily bad contention is **25%
  of the baseline**, because that is where the noise term is capped.
  It is not, and never was, 5%.

The sensitivity left on the table is the debug wheel's doing, not the
estimator's: on a release wheel the 0.75% floor evaluates to 1-2 us and
the flat 25 us budget binds in CI too. Recovering it means running this
one file against a release build, which is a CI-structure decision (and
what ADR-0038:182's ``--mode release`` asks for) — flagged below, not
taken here.

The fixtures are sized so the *release* baseline lands in the low
hundreds of microseconds, which keeps the budget under the 25% cap in
both profiles, so the budget or the precision floor — never the cap —
is what normally binds.

**Known gap against ADR-0019 / ADR-0038 — flagged, not closed here.**
ADR-0019:672-677 requires wall-clock **and allocation count** within
**1%** of a **0.0.1-frozen** baseline, "on the same dedicated benchmark
runner the pulp-vs-scalar test uses"; ADR-0038:32/74/182 repeats the 1%
and adds ``val2017-jittered, --mode release``. Five separate
divergences, all live:

1. There is **no dedicated benchmark runner**. This file runs in the
   ordinary ``pytest -m "not slow"`` job on shared ``ubuntu-latest``.
2. There is **no ``pulp-vs-scalar`` job or test anywhere** in
   ``.github/workflows/`` or ``tests/`` — the runner the ADR points at
   does not exist and never did.
3. There is **no frozen 0.0.1 baseline**. This file compares
   wrapper-vs-FFI *within one build*, which cannot detect drift from a
   frozen baseline at all: a regression inside the Rust core moves both
   sides equally and is invisible here. "Measured wrapper cost is 1-3
   us, so <=0.8%, so the 1% contract is met in fact" is therefore not
   an equivalent claim — it is a different measurement.
4. **Allocation count is not measured at all**, in any profile.
5. The run is not ``val2017-jittered`` and not ``--mode release``.

Both ADRs are ``accepted`` and immutable, so this is recorded rather
than edited; closing it wants a superseding ADR, which is a decision
for a human, not this file.
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

# The gate proper, and the only condition that can fail the test: the
# wrapper may add at most this much wall-clock per call, in absolute
# nanoseconds. Absolute on purpose — see the module docstring. Measured
# wrapper cost is ~1-3 us in release and ~2-4 us in debug (it is the
# same Python either way), so 25 us is better than a 6x margin over the
# real signal while sitting far below the 150 us - 9.5 ms gap a doubled
# wrapper produces in any profile.
#
# It also clears observed measurement drift at release baselines: on a
# loaded 8-core box, `min`-of-42 estimates of *identical* work drift by
# up to ~20 us there, and the CI false positive this gate was rewritten
# for was a 32 us gap. Drift at larger baselines is the precision
# floor's job, not this number's.
_OVERHEAD_BUDGET_NS = 25_000

_N_SAMPLES = 42  # multiple of 6 — one whole rotation cycle, see below
_MIN_BASELINE_NS = 100_000  # 0.1 ms — below this, timing is too noisy

# Round orders, as indices into (direct, wrapped, control). This is an
# Eulerian circuit over the six permutations of three streams, chosen so
# that the last element of each round equals the first element of the
# next. Over one full cycle every stream occupies every position exactly
# twice AND is preceded by every stream — itself included — exactly
# twice, so neither the wrapper's own allocation nor anyone else's can
# bias one stream's minimum relative to another's. A plain 3-round
# rotation balances position but provably cannot balance predecessors:
# with it the wrapper is preceded by a pure-FFI call 3/3 rounds while
# the control is preceded by the wrapper 2/3, which biases `wrapped_min`
# low and `noise` high — both in the permissive direction.
_ROUND_ORDERS = (
    (0, 1, 2),
    (2, 0, 1),
    (1, 0, 2),
    (2, 1, 0),
    (0, 2, 1),
    (1, 2, 0),
)

# Relative floor. A `min`-of-42 estimate is not exact — its error is
# *multiplicative* — so at the multi-millisecond debug baseline CI
# measures, the 25 us budget above sits below what the instrument can
# resolve. This term is that floor.
#
# It was 0.75%, chosen against a stated "true wrapper cost of ~0.03%".
# That figure was wrong, and the error only surfaced when the relative
# term became the binding one. Measured directly, by timing the two
# arms and printing the ratio rather than inferring it from drift:
#
#     build      interpreter   baseline    wrapper overhead
#     release    3.14          0.294 ms    2.2-3.6 us   0.75-1.23 %
#     debug      3.10          7.79 ms     67-88 us     0.86-1.13 %
#     debug      3.10         15.47 ms    122-139 us    0.79-0.90 %
#
# The wrapper costs ~1 % of the baseline, *proportionally*, on every
# build and interpreter measured — it is not a fixed cost and it is not
# 0.03 %. A floor at 0.75 % therefore sits BELOW the thing it is meant
# to be a floor for, and fires on correct code the moment it binds.
# That is exactly what happened: the gate failed on two consecutive
# release PRs (#307 on 3.14's sibling leg, #308 on 3.10/numpy-1.26),
# both times with the null control tracking `direct` to within 0.01 %
# — so the instrument was fine and the budget was wrong.
#
# It never bound before because on release baselines the 25 us absolute
# term dominates: 25 us against a measured 3.6 us worst case is 7x
# margin. Only CI's debug wheel, whose baseline is 26-53x larger, moves
# the relative term into the binding position.
#
# 2 % is ~1.6x the worst true cost observed (1.23 %), the same safety
# ratio the original 0.75 % was reaching for against its (mistaken)
# input. It keeps the gate able to catch a wrapper that *doubles*:
# 2 x 1.23 % = 2.46 %, above the floor. And it leaves release
# sensitivity untouched, because 2 % of a 294 us baseline is 5.9 us and
# the 25 us term still binds there — which was the original design's
# stated intent.
_PRECISION_FLOOR_FRACTION = 0.02

# Noise term. Fixed and relative floors are both static; this one reacts
# to the contention this run actually saw, via the null control.
_NOISE_SLACK_FACTOR = 3

# ...and a ceiling on that slack, because the noise term is otherwise
# unbounded: one load spike big enough to put `3 * noise` past the whole
# baseline would let a wrapper that *doubled* the cost sail through. The
# slack may never reach a quarter of the baseline, so a regression that
# costs anything like a full extra call is always out of reach of it.
# At the `_MIN_BASELINE_NS` floor this ceiling coincides exactly with
# `_OVERHEAD_BUDGET_NS`, so the two bounds meet rather than cross.
# `test_gate_detects_a_doubled_wrapper` is what caught the missing cap.
_MAX_SLACK_FRACTION = 0.25

# A regression is reproducible; a contention episode is not. Re-measure
# before failing and require every attempt to trip the gate. This costs
# nothing on the happy path (the first attempt returns) and it does not
# widen the threshold by a nanosecond — a doubled wrapper trips all
# three attempts, every time. All-must-trip means detection probability
# is p**3 for a regression caught with probability p per attempt, which
# is why the quoted sensitivity is the *guaranteed* bound above, not a
# marginal one.
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
    the null measurement. Every stream gets the same sample count, the
    same share of the measurement window, the same distribution of
    positions within the round, and the same distribution of immediate
    predecessors, so no stream is handed a better estimator than the
    others and none of them is systematically warmed or perturbed by
    another. See ``_ROUND_ORDERS``.
    """
    # Warm up — JIT-style first-call costs (lazy class init, allocator
    # priming) shouldn't bias the comparison.
    for _ in range(5):
        direct()
        wrapped()

    streams: tuple[Callable[[], object], Callable[[], object], Callable[[], object]] = (
        direct,
        wrapped,
        direct,
    )
    samples: tuple[list[int], list[int], list[int]] = ([], [], [])
    # GC is one-sided noise and lands preferentially on the stream that
    # allocates most, which is the wrapper. Freeze it for the window.
    gc_was_enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        for i in range(_N_SAMPLES):
            for slot in _ROUND_ORDERS[i % len(_ROUND_ORDERS)]:
                samples[slot].append(_bench(streams[slot]))
    finally:
        if gc_was_enabled:
            gc.enable()

    direct_samples, wrapped_samples, control_samples = samples
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
        # `fail`, not `skip` and not `xfail`. A skip here would be
        # invisible in exactly the way the `xfail` this file replaced
        # was: `Skipped` derives from `BaseException`, so it sails
        # straight through the `pytest.raises(AssertionError)` in
        # `test_gate_detects_a_doubled_wrapper` and silences the canary
        # too — one bad constant and all four tests report `ssss`, which
        # reads as "fine" at a glance. With the current fixture sizes
        # this branch is unreachable in either build profile (the
        # smallest baseline is ~170 us in release, ~1700 us in debug);
        # if it ever fires, the fixture needs resizing, and that is a
        # human decision, not something to report and move on from.
        pytest.fail(
            f"{label} baseline {direct_min} ns < {_MIN_BASELINE_NS} ns — the "
            f"fixture is too small to time on this runner, so this gate is "
            f"not measuring anything. Resize the fixture; do not relax the "
            f"floor."
        )

    delta_ns = wrapped_min - direct_min
    # Reported, never asserted. The ratio is the wrong instrument here:
    # its numerator (Python) is build-profile-invariant and its
    # denominator (Rust) is not, so one percentage means a 10-34x
    # different absolute budget between the release wheel developers
    # build and the debug wheel CI installs. It stays in the message
    # because it is the number a human reads first.
    ratio = wrapped_min / direct_min
    # Two `min`-reduced estimates of identical work: whatever separates
    # them is noise, by construction.
    noise_ns = abs(control_min - direct_min)
    slack_ns = min(
        max(
            _OVERHEAD_BUDGET_NS,
            int(_PRECISION_FLOOR_FRACTION * direct_min),
            _NOISE_SLACK_FACTOR * noise_ns,
        ),
        int(_MAX_SLACK_FRACTION * direct_min),
    )

    if delta_ns <= slack_ns:
        return None
    return (
        f"{label} default path adds {delta_ns} ns per call over the direct "
        f"FFI — exceeds the {slack_ns} ns budget "
        f"(direct={direct_min} ns, wrapped={wrapped_min} ns, "
        f"control={control_min} ns, observed noise={noise_ns} ns, "
        f"ratio={ratio:.3f}x)"
    )


def _assert_no_wrapper_overhead(
    label: str,
    direct: Callable[[], object],
    wrapped: Callable[[], object],
) -> None:
    """Fail iff the wrapper adds more absolute wall-clock per call than
    ``_OVERHEAD_BUDGET_NS`` (widened only by this runner's own measured
    noise, and never past a quarter of the baseline) — and stays that
    way across every re-measurement."""
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


def test_round_orders_balance_position_and_predecessor() -> None:
    """``_ROUND_ORDERS`` must be balanced on both axes.

    The estimator's fairness claim is a property of this table, not of
    the timing code, so it is checked directly rather than asserted in
    a comment. Position balance alone is not enough: only the wrapper
    allocates, so an unbalanced *predecessor* distribution biases
    ``wrapped_min`` low and ``noise`` high, and both point the
    permissive way.
    """
    flat = [slot for order in _ROUND_ORDERS for slot in order]
    assert _N_SAMPLES % len(_ROUND_ORDERS) == 0, (
        "_N_SAMPLES must be a whole number of rotation cycles, else the "
        "balance below holds only on average"
    )

    for slot in (0, 1, 2):
        assert flat.count(slot) == len(_ROUND_ORDERS), "each stream once per round"

    positions = {slot: [0, 0, 0] for slot in (0, 1, 2)}
    for order in _ROUND_ORDERS:
        for position, slot in enumerate(order):
            positions[slot][position] += 1
    for slot, counts in positions.items():
        assert counts == [2, 2, 2], f"stream {slot} position counts {counts} unbalanced"

    # Cyclic, because the cycle repeats: the predecessor of the first
    # call of a cycle is the last call of the previous one.
    predecessors = {slot: [0, 0, 0] for slot in (0, 1, 2)}
    for index, slot in enumerate(flat):
        predecessors[slot][flat[index - 1]] += 1
    for slot, counts in predecessors.items():
        assert counts == [2, 2, 2], f"stream {slot} predecessor counts {counts} unbalanced"


def test_gate_detects_a_doubled_wrapper() -> None:
    """The gate itself must be able to fail.

    Guards against the whole file decaying into a no-op — which is
    exactly what the ``_MIN_BASELINE_NS`` xfail did to the panoptic and
    semantic variants while their fixtures were too small to clear it,
    and what a ``pytest.skip`` on that same branch would still do,
    since ``Skipped`` is a ``BaseException`` and would pass straight
    through the ``pytest.raises`` below. The branch calls
    ``pytest.fail``: whatever happens, this test cannot come back
    green-or-quiet without the gate having really fired.

    A "wrapper" that runs the workload twice is the cheapest honest
    stand-in for a 2x regression. That is ~2x the work of a normal
    variant times ``_MAX_ATTEMPTS`` (all attempts must trip before the
    gate fails), which makes this the most expensive test in the file
    by a wide margin — the bulk of the file's runtime is here.
    """

    def direct() -> object:
        return evaluate_bbox_summary(_GT, _DT, _PARITY, _MAX_DETS, _USE_CATS)

    def doubled() -> None:
        direct()
        direct()

    with pytest.raises(AssertionError, match="exceeds the"):
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
