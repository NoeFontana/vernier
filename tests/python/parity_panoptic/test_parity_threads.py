"""ADR-0047 cross-thread strict-mode bit-equality for the panoptic
paradigm.

The panoptic per-image ``PqStat`` fold is ``f64``-additive and **not**
associative across orderings (W7 — the global SQ is the unweighted
mean of per-category SQs, which sums into one f64 per category). Naive
parallel reduction across thread counts would drift.

This module pins two properties:

1. **Cross-thread bit-equality on every existing parity fixture.** For
   every ``num_threads in {None, 1, 2, 4, 8}`` and every
   :class:`PanopticSnapshot` field, the parallel result is byte-equal
   to the sequential one. Strict mode is the load-bearing assertion;
   corrected mode rides along on the same canonical image-id-sorted
   fold inside the kernel.

2. **The strict-mode forced-flag policy fires exactly once.** Calling
   :meth:`vernier.panoptic.Evaluator.evaluate` with
   ``parity_mode="strict"`` and ``num_threads >= 2`` flips
   ``retain_per_image_deltas`` to ``True`` (overriding the caller's
   ``False``) and emits a one-shot ``logging.info`` line. The override
   is logged once per process via the module-level sentinel; subsequent
   calls remain silent.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pytest

import vernier
import vernier.panoptic as pq

from .harness import (
    PanopticSnapshot,
    assert_snapshots_equal,
    snapshot,
    summary_to_snapshot,
)

#: Thread counts swept by the cross-thread bit-equality axis. Mirrors
#: the instance/semantic parity_threads tests; the panoptic batch path
#: shares the ``num_threads`` resolution + scoped-pool plumbing
#: (ADR-0047).
ADR_0047_PANOPTIC_THREAD_COUNTS: tuple[int | None, ...] = (None, 1, 2, 4, 8)


# ---------------------------------------------------------------------------
# Fixtures — small synthetic panoptic frames that exercise W1 (direct
# PQ form), W4 (things/stuff split), W7 (mean-vs-pooled SQ), U7 (IoU
# strict-greater), and V3 (multi-same-category crowd corrected fold).
# Each fixture mirrors an existing `test_parity_panoptic.py` shape so
# the cross-thread axis runs against the same shapes the cross-oracle
# axis runs against.
# ---------------------------------------------------------------------------


def _perfect_match_fixture() -> dict[str, Any]:
    return {
        "gt": {1: np.array([[1, 1, 1, 1, 1, 2, 2, 2, 2, 2]], dtype=np.uint32)},
        "dt": {1: np.array([[10, 10, 10, 10, 10, 11, 11, 11, 11, 11]], dtype=np.uint32)},
        "gt_segs": {
            1: [
                {"id": 1, "category_id": 100, "iscrowd": False, "area": 5},
                {"id": 2, "category_id": 200, "iscrowd": False, "area": 5},
            ]
        },
        "dt_segs": {
            1: [
                {"id": 10, "category_id": 100, "iscrowd": False, "area": 5},
                {"id": 11, "category_id": 200, "iscrowd": False, "area": 5},
            ]
        },
        "cats": [
            {"id": 100, "isthing": True},
            {"id": 200, "isthing": False},
        ],
    }


def _w7_long_tailed_fixture() -> dict[str, Any]:
    """Q5 / W7 from `test_parity_panoptic.py`. Long-tailed mix where
    the global SQ is sensitive to the per-category fold order (mean of
    per-category SQs); the per-image f64 IoU sum for cat 100 spans two
    images and therefore exercises the cross-thread fold-order axis."""
    gt = {
        1: np.array([[1] * 10], dtype=np.uint32),
        2: np.array([[1] * 10], dtype=np.uint32),
        3: np.array([[2] * 10], dtype=np.uint32),
    }
    dt = {
        1: np.array([[10] * 10], dtype=np.uint32),
        2: np.array([[10] * 7 + [0] * 3], dtype=np.uint32),
        3: np.array([[11] * 6 + [0] * 4], dtype=np.uint32),
    }
    return {
        "gt": gt,
        "dt": dt,
        "gt_segs": {
            1: [{"id": 1, "category_id": 100, "iscrowd": False, "area": 10}],
            2: [{"id": 1, "category_id": 100, "iscrowd": False, "area": 10}],
            3: [{"id": 2, "category_id": 200, "iscrowd": False, "area": 10}],
        },
        "dt_segs": {
            1: [{"id": 10, "category_id": 100, "iscrowd": False, "area": 10}],
            2: [{"id": 10, "category_id": 100, "iscrowd": False, "area": 7}],
            3: [{"id": 11, "category_id": 200, "iscrowd": False, "area": 6}],
        },
        "cats": [
            {"id": 100, "isthing": True},
            {"id": 200, "isthing": False},
        ],
    }


def _v3_multi_crowd_fixture() -> dict[str, Any]:
    """Q4 / V3 — multi same-category crowd. Strict-mode last-wins
    against vernier corrected sum-overlaps. The parallel path must
    bit-equal sequential under either mode."""
    return {
        "gt": {1: np.array([[1, 1, 2, 2, 2, 2, 3, 3, 3, 3]], dtype=np.uint32)},
        "dt": {1: np.array([[10] * 10], dtype=np.uint32)},
        "gt_segs": {
            1: [
                {"id": 1, "category_id": 100, "iscrowd": True, "area": 2},
                {"id": 2, "category_id": 100, "iscrowd": True, "area": 4},
                {"id": 3, "category_id": 999, "iscrowd": False, "area": 4},
            ]
        },
        "dt_segs": {1: [{"id": 10, "category_id": 100, "iscrowd": False, "area": 10}]},
        "cats": [
            {"id": 100, "isthing": True},
            {"id": 999, "isthing": False},
        ],
    }


_FIXTURES: dict[str, dict[str, Any]] = {
    "perfect_match": _perfect_match_fixture(),
    "w7_long_tailed": _w7_long_tailed_fixture(),
    "v3_multi_crowd": _v3_multi_crowd_fixture(),
}


# ---------------------------------------------------------------------------
# Cross-thread bit-equality — strict mode is the load-bearing
# assertion; corrected mode rides along on the same canonical image-
# id-sorted fold inside the kernel.
# ---------------------------------------------------------------------------


@pytest.mark.parity_panoptic
@pytest.mark.parity_threads
@pytest.mark.parametrize("fixture_name", sorted(_FIXTURES.keys()))
@pytest.mark.parametrize("parity_mode", ["strict", "corrected"])
@pytest.mark.parametrize("num_threads", ADR_0047_PANOPTIC_THREAD_COUNTS)
def test_panoptic_parity_across_thread_counts_bit_equal(
    fixture_name: str,
    parity_mode: vernier.ParityMode,
    num_threads: int | None,
) -> None:
    """ADR-0047 load-bearing parity assertion for the panoptic
    paradigm: every fixture, both parity modes, every thread count
    produces a :class:`PanopticSnapshot` bit-equal to the sequential
    (``num_threads=None``) baseline.

    Strict mode is bridged by the forced-flag policy
    (``retain_per_image_deltas=True`` re-sort + re-sum at finalize).
    Corrected mode is bit-equal by virtue of the kernel's canonical
    image-id-sorted per-image delta walk."""
    fx = _FIXTURES[fixture_name]
    baseline = snapshot(
        "vernier",
        fx["gt"],
        fx["gt_segs"],
        fx["dt"],
        fx["dt_segs"],
        fx["cats"],
        parity_mode=parity_mode,
        num_threads=None,
    )
    cand = snapshot(
        "vernier",
        fx["gt"],
        fx["gt_segs"],
        fx["dt"],
        fx["dt_segs"],
        fx["cats"],
        parity_mode=parity_mode,
        num_threads=num_threads,
    )
    assert_snapshots_equal(baseline, cand)


# ---------------------------------------------------------------------------
# Forced-flag policy — one-shot info log, override even when caller
# passed `retain_per_image_deltas=False`.
# ---------------------------------------------------------------------------


@pytest.mark.parity_panoptic
def test_panoptic_strict_forces_retain_per_image_deltas(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """ADR-0047 §"Panoptic" forced-flag policy.

    Construct a panoptic :class:`Evaluator` with ``parity_mode="strict"``
    and call :meth:`Evaluator.evaluate` with ``num_threads=4`` and
    ``retain_per_image_deltas=False``. Assert:

    (a) The result is bit-equal to a ``num_threads=None`` run — i.e.
        the strict-mode bit-equality property survives the parallel
        dispatch (the forced flag bridges the otherwise-non-associative
        per-image PqStat sum).
    (b) The info log fired exactly once during the parallel call. The
        sentinel is module-global; we reset it explicitly so the
        property is checked regardless of test execution order.
    """
    # Reset the one-shot sentinel so this test doesn't depend on
    # whether a previous test in the same process already triggered
    # the log.
    import vernier.panoptic as panoptic_mod

    panoptic_mod._FORCED_LOG_EMITTED = False

    fx = _perfect_match_fixture()

    # Build typed datasets via the harness path so this test stays
    # decoupled from the FFI's exact byte shapes.
    import json

    gt_segs_bytes = json.dumps({str(k): list(v) for k, v in fx["gt_segs"].items()}).encode()
    dt_segs_bytes = json.dumps({str(k): list(v) for k, v in fx["dt_segs"].items()}).encode()
    cats_bytes = json.dumps([dict(c) for c in fx["cats"]]).encode()
    gt = pq.Dataset.from_arrays({int(k): v for k, v in fx["gt"].items()}, gt_segs_bytes, cats_bytes)
    dt = pq.Predictions.from_arrays({int(k): v for k, v in fx["dt"].items()}, dt_segs_bytes)
    ev = pq.Evaluator(parity_mode="strict", things_stuff_split=True)

    # Baseline: no threads, no forced override; nothing to log.
    with caplog.at_level(logging.INFO, logger="vernier.panoptic"):
        baseline = ev.evaluate(gt, dt, num_threads=None, retain_per_image_deltas=False)
    baseline_msgs = [r.getMessage() for r in caplog.records if r.name == "vernier.panoptic"]
    # No forced-flag log on the sequential path (num_threads <= 1).
    assert not any("forcing retain_per_image_deltas" in m for m in baseline_msgs), (
        f"unexpected forced-flag log on sequential call: {baseline_msgs!r}"
    )

    caplog.clear()
    # Parallel call: trigger the policy override.
    with caplog.at_level(logging.INFO, logger="vernier.panoptic"):
        parallel = ev.evaluate(gt, dt, num_threads=4, retain_per_image_deltas=False)
    parallel_msgs = [
        r.getMessage()
        for r in caplog.records
        if r.name == "vernier.panoptic" and r.levelno == logging.INFO
    ]
    forced_msgs = [m for m in parallel_msgs if "forcing retain_per_image_deltas" in m]
    assert len(forced_msgs) == 1, (
        f"expected exactly one forced-flag info log, got {len(forced_msgs)}: {parallel_msgs!r}"
    )

    # Bit-equality across thread counts on every snapshot field.
    snap_seq: PanopticSnapshot = summary_to_snapshot(baseline)
    snap_par: PanopticSnapshot = summary_to_snapshot(parallel)
    assert_snapshots_equal(snap_seq, snap_par)

    # Second parallel call: the sentinel is one-shot, so no further
    # info log even though the policy still applies.
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="vernier.panoptic"):
        ev.evaluate(gt, dt, num_threads=4, retain_per_image_deltas=False)
    second_call_msgs = [
        m
        for r in caplog.records
        if r.name == "vernier.panoptic" and r.levelno == logging.INFO
        for m in [r.getMessage()]
        if "forcing retain_per_image_deltas" in m
    ]
    assert second_call_msgs == [], (
        f"second parallel call must not re-emit the forced-flag log, got {second_call_msgs!r}"
    )
