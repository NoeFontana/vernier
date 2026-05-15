"""Clean-room numpy reference for ADR-0018 detection-family calibration.

This module is the **oracle** for the calibration summarizer landed in
``crates/vernier-core/src/calibration.rs`` (Unit 1 of the ADR-0018
implementation plan). It mirrors the ADR-0010 isolation pattern used by
``tests/python/parity_boundary/numpy_reference.py``: a pure-numpy
implementation, self-contained, with no scipy / sklearn / statsmodels
dependency. ``sklearn.calibration.calibration_curve`` is a *sanity
cross-check* (semantics differ around bin edges); it is **not** used
here.

Implements ADR-0018 §"Decision outcome" → "DETR-aware defaults":

- Quantile binning by default (linear-method `np.quantile`, P1).
- ``min_score = 0.05`` cutoff (P3 — corrected vs pycocotools-free
  baseline).
- Wilson confidence intervals on per-bin accuracy (P4 — strict; explicit
  z=1.959963984540054, not ``statsmodels``).
- Macro per-class aggregation default (P6 — informational view; the
  per-class table itself is unweighted).

Risk-register mitigations baked into the algorithm:

- **R1 / P5** — quantile bin-edge degeneracy on small samples: detect
  duplicate edges, merge bins, surface ``effective_n_bins``.
- **R2** — Wilson CI on zero-count bins: emit ``np.nan`` for
  ``accuracy`` / ``ci_lo`` / ``ci_hi`` / ``mean_score`` / ``gap``.
- **R3 / P2** — ignore-region detections (``dt_ignore[t, d] == True``)
  drop out of the histogram entirely.

Numerical policy: ``f64`` end-to-end (ADR-0004). No ``float32`` casts.

Cell input shape mirrors ``crates/vernier-core/src/accumulate.rs:62-75``
(``PerImageEval``) so vernier-side harness code can marshal cells into
the oracle directly without a shape-conversion pass.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

# Wilson z-score for 95% CI. Pinned to the literal that the Rust kernel
# uses (quirk P4, strict). Do NOT round.
_WILSON_Z: float = 1.959963984540054

Binning = Literal["quantile", "equal_width"]
Confidence = Literal["wilson", "clopper_pearson"]
Aggregation = Literal["macro", "micro"]


@dataclass(frozen=True)
class PerImageCell:
    """Mirrors ``vernier-core``'s ``PerImageEval`` (accumulate.rs:62-75).

    The ``gt_ignore`` field on the Rust side is unused by calibration
    and is omitted here. The arrays must be ``np.float64`` (scores) and
    ``bool`` (matched / ignore); the oracle does not coerce dtypes — the
    caller is responsible for matching the Rust kernel's input dtype
    policy.
    """

    dt_scores: np.ndarray  # shape (D,), float64, sorted descending
    dt_matched: np.ndarray  # shape (T, D), bool
    dt_ignore: np.ndarray  # shape (T, D), bool


@dataclass(frozen=True)
class CalibrationParams:
    """Mirrors the Rust kernel's ``CalibrationParams`` (Unit 1 plan).

    Enums are kept stringly-typed on the Python side; the harness layer
    converts to the FFI's int/enum encoding before crossing the
    boundary. Defaults track ADR-0018 §"DETR-aware defaults".
    """

    iou_index: int = 0
    n_bins: int = 15
    binning: Binning = "quantile"
    min_score: float = 0.05
    confidence: Confidence = "wilson"
    per_class: bool = False
    per_class_aggregation: Aggregation = "macro"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _wilson_ci(
    correct: np.ndarray, n: np.ndarray, z: float = _WILSON_Z
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized Wilson interval. NaNs out where ``n == 0``.

    Both inputs broadcast; output dtype is ``np.float64``.
    """
    correct_f = correct.astype(np.float64, copy=False)
    n_f = n.astype(np.float64, copy=False)
    # Compute on a safe denominator, then mask out the n==0 entries.
    safe_n = np.where(n_f > 0.0, n_f, 1.0)
    phat = correct_f / safe_n
    z2 = z * z
    denom = 1.0 + z2 / safe_n
    center = (phat + z2 / (2.0 * safe_n)) / denom
    margin = (z / denom) * np.sqrt(phat * (1.0 - phat) / safe_n + z2 / (4.0 * safe_n * safe_n))
    lo = center - margin
    hi = center + margin
    mask = n_f > 0.0
    lo = np.where(mask, lo, np.nan)
    hi = np.where(mask, hi, np.nan)
    return lo, hi


def _flatten_cells(
    cells: list[PerImageCell],
    iou_index: int,
    min_score: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Pull ``(score, correct)`` pairs from a class's cells.

    Drops ignored detections (P2) and detections with
    ``score < min_score`` (P3). Returns ``(scores, correct)`` as
    ``float64`` and ``bool`` arrays respectively.
    """
    score_chunks: list[np.ndarray] = []
    correct_chunks: list[np.ndarray] = []
    for cell in cells:
        if cell.dt_scores.size == 0:
            continue
        if cell.dt_matched.ndim != 2 or cell.dt_ignore.ndim != 2:
            raise ValueError("dt_matched / dt_ignore must be 2D arrays")
        t_dim = cell.dt_matched.shape[0]
        if iou_index >= t_dim or iou_index < 0:
            raise ValueError(f"iou_index={iou_index} out of range for T={t_dim} on this cell")
        if cell.dt_matched.shape[1] != cell.dt_scores.shape[0]:
            raise ValueError("dt_matched D-axis must match dt_scores length")
        if cell.dt_ignore.shape != cell.dt_matched.shape:
            raise ValueError("dt_ignore shape must match dt_matched")
        scores = cell.dt_scores.astype(np.float64, copy=False)
        matched = cell.dt_matched[iou_index].astype(bool, copy=False)
        ignore = cell.dt_ignore[iou_index].astype(bool, copy=False)
        keep = (~ignore) & (scores >= min_score)
        if not keep.any():
            continue
        score_chunks.append(scores[keep])
        correct_chunks.append(matched[keep])
    if not score_chunks:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=bool),
        )
    return (
        np.concatenate(score_chunks).astype(np.float64, copy=False),
        np.concatenate(correct_chunks).astype(bool, copy=False),
    )


def _bin_edges(scores: np.ndarray, n_bins: int, binning: Binning) -> np.ndarray:
    """Construct bin edges per the binning strategy.

    For ``quantile``: ``np.quantile`` with ``method='linear'`` (P1), then
    deduplicate consecutive identical edges to mitigate R1 / P5.

    For ``equal_width``: ``np.linspace(min_score_present, 1.0, n+1)``.
    """
    if scores.size == 0:
        return np.empty(0, dtype=np.float64)
    if binning == "quantile":
        qs = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float64)
        raw_edges = np.quantile(scores, qs, method="linear").astype(np.float64, copy=False)
        # ``np.unique`` returns a sorted, deduped array. Quantile edges
        # are monotonically non-decreasing in q, so the ordering is
        # preserved. This is the R1 mitigation: duplicate quantile edges
        # collapse, ``effective_n_bins = len(unique) - 1``.
        edges = np.unique(raw_edges)
        return edges
    if binning == "equal_width":
        lo = float(scores.min())
        return np.linspace(lo, 1.0, n_bins + 1, dtype=np.float64)
    # Defensive runtime guard. Exhaustive over the ``Binning`` Literal so
    # pyright marks this as unreachable (informational, not an error).
    raise ValueError(f"unknown binning strategy: {binning!r}")


def _empty_reliability() -> dict[str, np.ndarray]:
    """Return a reliability table with zero rows."""
    return {
        "bin_id": np.empty(0, dtype=np.uint32),
        "score_lo": np.empty(0, dtype=np.float64),
        "score_hi": np.empty(0, dtype=np.float64),
        "mean_score": np.empty(0, dtype=np.float64),
        "accuracy": np.empty(0, dtype=np.float64),
        "count": np.empty(0, dtype=np.uint64),
        "gap": np.empty(0, dtype=np.float64),
        "ci_lo": np.empty(0, dtype=np.float64),
        "ci_hi": np.empty(0, dtype=np.float64),
    }


def _bin_one(
    scores: np.ndarray,
    correct: np.ndarray,
    edges: np.ndarray,
) -> tuple[dict[str, np.ndarray], float, float, int]:
    """Bin a flat ``(scores, correct)`` stream into a reliability table.

    Returns ``(reliability_dict, ece, mce, n_total)``. Empty inputs or
    single-edge inputs produce an empty table with NaN ECE/MCE.
    """
    n_total = int(scores.size)
    if n_total == 0 or edges.size < 2:
        return _empty_reliability(), float("nan"), float("nan"), n_total

    effective_n_bins = int(edges.size - 1)
    # ``side='right'`` then ``-1`` puts boundary-equal samples in the
    # *left* of the two adjacent bins (i.e. into [..., edges[i]]). Then
    # clip the rightmost edge into the last bin and any underflow from
    # the leftmost edge into bin 0. This matches the histogram
    # convention np.histogram uses for non-last bins, with a closed
    # final bin on the right.
    bin_idx = np.searchsorted(edges, scores, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, effective_n_bins - 1).astype(np.int64, copy=False)

    correct_f = correct.astype(np.float64, copy=False)
    count = np.bincount(bin_idx, minlength=effective_n_bins).astype(np.uint64, copy=False)
    sum_score = np.bincount(bin_idx, weights=scores, minlength=effective_n_bins).astype(
        np.float64, copy=False
    )
    sum_correct = np.bincount(bin_idx, weights=correct_f, minlength=effective_n_bins).astype(
        np.float64, copy=False
    )

    count_f = count.astype(np.float64, copy=False)
    safe_count = np.where(count_f > 0.0, count_f, 1.0)
    mean_score = np.where(count_f > 0.0, sum_score / safe_count, np.nan)
    accuracy = np.where(count_f > 0.0, sum_correct / safe_count, np.nan)
    gap = np.where(count_f > 0.0, accuracy - mean_score, np.nan)
    ci_lo, ci_hi = _wilson_ci(sum_correct, count_f)

    # ECE: weighted by count / N_total. R2: NaN bins (count == 0) drop
    # to zero weight via the (count > 0) mask. N_total is the explicit
    # non-filtered detection count, which equals count.sum() but we
    # keep both written out for clarity.
    nonempty = count_f > 0.0
    if not nonempty.any():
        ece = float("nan")
        mce = float("nan")
    else:
        weights = count_f / float(n_total)
        ece = float(np.sum(weights[nonempty] * np.abs(gap[nonempty])))
        mce = float(np.max(np.abs(gap[nonempty])))

    reliability = {
        "bin_id": np.arange(effective_n_bins, dtype=np.uint32),
        "score_lo": edges[:-1].astype(np.float64, copy=False),
        "score_hi": edges[1:].astype(np.float64, copy=False),
        "mean_score": mean_score.astype(np.float64, copy=False),
        "accuracy": accuracy.astype(np.float64, copy=False),
        "count": count,
        "gap": gap.astype(np.float64, copy=False),
        "ci_lo": ci_lo.astype(np.float64, copy=False),
        "ci_hi": ci_hi.astype(np.float64, copy=False),
    }
    return reliability, ece, mce, n_total


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def numpy_calibration(
    cells_by_class: dict[int, list[PerImageCell]],
    params: CalibrationParams,
) -> dict[str, object]:
    """Compute the calibration summary over a per-class cell store.

    See module docstring for the algorithm spec. The return shape
    mirrors the Rust kernel's ``CalibrationSummary`` so harness code can
    diff the two element-wise.

    Args:
        cells_by_class: mapping from ``class_id`` to its list of
            ``PerImageCell``. The marginal calibration is computed by
            concatenating across classes.
        params: calibration parameters; see ``CalibrationParams``.

    Raises:
        NotImplementedError: when ``params.confidence == "clopper_pearson"``
            (Phase-2; same surface as the Rust kernel).
        ValueError: when ``params.n_bins == 0`` or ``params.iou_index < 0``,
            or when any cell carries an inconsistent
            ``dt_scores`` / ``dt_matched`` / ``dt_ignore`` shape.
    """
    if params.confidence == "clopper_pearson":
        raise NotImplementedError("Clopper-Pearson CI not yet implemented; use Wilson")
    if params.n_bins == 0:
        raise ValueError("n_bins must be >= 1")
    if params.iou_index < 0:
        raise ValueError(f"iou_index must be >= 0, got {params.iou_index}")

    # 1) Flatten per-class, then concatenate to marginal.
    per_class_streams: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for class_id in sorted(cells_by_class.keys()):
        scores_k, correct_k = _flatten_cells(
            cells_by_class[class_id], params.iou_index, params.min_score
        )
        per_class_streams[class_id] = (scores_k, correct_k)

    if per_class_streams:
        marginal_scores = np.concatenate([s for s, _ in per_class_streams.values()]).astype(
            np.float64, copy=False
        )
        marginal_correct = np.concatenate([c for _, c in per_class_streams.values()]).astype(
            bool, copy=False
        )
    else:
        marginal_scores = np.empty(0, dtype=np.float64)
        marginal_correct = np.empty(0, dtype=bool)

    # 2) Bin-edge construction (marginal stream drives the edges; the
    # per-class binning *reuses* the marginal edges so per-class tables
    # are comparable bin-for-bin — the Rust kernel does the same).
    edges = _bin_edges(marginal_scores, params.n_bins, params.binning)
    effective_n_bins = int(max(0, edges.size - 1))

    reliability, ece, mce, n_total = _bin_one(marginal_scores, marginal_correct, edges)

    per_class_out: dict[str, np.ndarray] | None
    if params.per_class:
        class_ids: list[int] = []
        eces: list[float] = []
        mces: list[float] = []
        ns: list[int] = []
        for class_id, (scores_k, correct_k) in per_class_streams.items():
            _, ece_k, mce_k, n_k = _bin_one(scores_k, correct_k, edges)
            class_ids.append(int(class_id))
            eces.append(float(ece_k))
            mces.append(float(mce_k))
            ns.append(int(n_k))
        per_class_out = {
            "class_id": np.asarray(class_ids, dtype=np.uint32),
            "ece": np.asarray(eces, dtype=np.float64),
            "mce": np.asarray(mces, dtype=np.float64),
            "n": np.asarray(ns, dtype=np.uint64),
        }
    else:
        per_class_out = None

    return {
        "ece": ece,
        "mce": mce,
        "n_detections": n_total,
        "effective_n_bins": effective_n_bins,
        "reliability": reliability,
        "per_class": per_class_out,
    }


# ---------------------------------------------------------------------------
# Fixture-load helper (shared with seed.py / harness)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FixtureBundle:
    """Internal record returned by ``load_fixture``."""

    cells_by_class: dict[int, list[PerImageCell]] = field(default_factory=dict)
    n_iou_thresholds: int = 0


def load_fixture(fixture_dir: Path) -> _FixtureBundle:
    """Load a fixture's ``cells.json`` into ``PerImageCell`` records.

    The JSON format is the one written by ``fixtures/seed.py``:
    ``{ "n_iou_thresholds": int, "cells": [ {class_id, dt_scores, dt_matched, dt_ignore}, ... ] }``
    where ``dt_matched`` and ``dt_ignore`` are nested lists of shape
    ``[T][D]`` and ``dt_scores`` is a flat list of length ``D``.
    """
    path = fixture_dir / "cells.json"
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    n_t = int(payload["n_iou_thresholds"])
    cells_by_class: dict[int, list[PerImageCell]] = {}
    for raw in payload["cells"]:
        class_id = int(raw["class_id"])
        scores = np.asarray(raw["dt_scores"], dtype=np.float64)
        matched = np.asarray(raw["dt_matched"], dtype=bool)
        ignore = np.asarray(raw["dt_ignore"], dtype=bool)
        if matched.ndim == 1:
            matched = matched.reshape(1, -1)
        if ignore.ndim == 1:
            ignore = ignore.reshape(1, -1)
        cell = PerImageCell(dt_scores=scores, dt_matched=matched, dt_ignore=ignore)
        cells_by_class.setdefault(class_id, []).append(cell)
    return _FixtureBundle(cells_by_class=cells_by_class, n_iou_thresholds=n_t)


# ---------------------------------------------------------------------------
# Self-test (smoke check before Unit 4c wires the oracle to vernier)
# ---------------------------------------------------------------------------


def _self_test() -> int:
    """Load ``cal_perfect`` and assert ECE matches the expected scalar.

    ``cal_perfect`` is a clean monotone ramp of 10 scores per class
    across 3 classes (30 detections total), all matched at every
    threshold, all ``dt_ignore=False``. The marginal score stream after
    ``min_score=0.05`` filtering is the union of three identical ramps;
    every detection is correct, so for any bin layout
    ``accuracy == 1.0`` and ECE collapses to
    ``Σ (count_b / N) * (1 - mean_score_b)``, i.e. the score gap from
    1.0 weighted by bin mass.
    """
    fixture_dir = Path(__file__).parent / "fixtures" / "cal_perfect"
    bundle = load_fixture(fixture_dir)
    result = numpy_calibration(
        bundle.cells_by_class,
        CalibrationParams(iou_index=0, n_bins=15, binning="quantile"),
    )
    ece = float(result["ece"])  # type: ignore[arg-type]
    # Hand-computed expected: cal_perfect's scores are linspace(0.05, 1.0, 10)
    # per class, replicated across 3 classes. After min_score=0.05 (>=)
    # all 30 detections survive. With all correct=True, ECE equals
    # weighted mean of (1.0 - mean_score) per bin, which equals
    # (1.0 - overall_mean_score) = 1.0 - mean(linspace(0.05, 1.0, 10)).
    expected = 1.0 - float(np.mean(np.linspace(0.05, 1.0, 10, dtype=np.float64)))
    ok = abs(ece - expected) < 1e-12
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] cal_perfect ECE={ece:.12f} expected={expected:.12f}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())
