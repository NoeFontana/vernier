"""Deterministic fixture generator for the ADR-0018 calibration oracle.

Writes ``cells.json`` (and an optional ``meta.json`` sidecar carrying
``iou_type`` for the per-paradigm smoketests) under each fixture
directory. The JSON layout is the one consumed by
``numpy_oracle.load_fixture``:

::

    {
      "n_iou_thresholds": <T>,
      "cells": [
        {
          "class_id": <int>,
          "dt_scores":  [s0, s1, ...],          # length D, float64
          "dt_matched": [[bool, ...], ...],     # shape [T][D]
          "dt_ignore":  [[bool, ...], ...]      # shape [T][D]
        },
        ...
      ]
    }

One cell per ``(class, "image")`` is plenty for the oracle: the
calibration kernel folds over cells without re-sorting, so cell
granularity is observably equivalent to a single concatenated stream.
The harness (Unit 4c) will marshal vernier's real
``(K, A, I)``-shaped cell grid into the same JSON when it runs the
parity comparison.

Idempotence: every fixture is built from ``np.random.default_rng(seed)``
with a fixed seed per fixture. Re-running this script overwrites the
``cells.json`` with byte-identical content.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

# Canonical T axis size for COCO (0.5:0.05:0.95). The oracle's
# ``iou_index`` indexes into this T-axis; bbox/segm/keypoints all carry
# the same T=10 ladder.
_DEFAULT_T: int = 10


def _cell_json(
    class_id: int, scores: np.ndarray, matched_row: np.ndarray, ignore_row: np.ndarray, n_t: int
) -> dict[str, Any]:
    """Tile a single (T=1) match/ignore row across the T axis."""
    d = int(scores.size)
    matched = np.broadcast_to(matched_row.reshape(1, d), (n_t, d)).astype(bool, copy=True)
    ignore = np.broadcast_to(ignore_row.reshape(1, d), (n_t, d)).astype(bool, copy=True)
    return {
        "class_id": int(class_id),
        "dt_scores": [float(s) for s in scores.tolist()],
        "dt_matched": matched.tolist(),
        "dt_ignore": ignore.tolist(),
    }


def _write(
    fixture_dir: Path, cells: list[dict[str, Any]], n_t: int, meta: dict[str, Any] | None = None
) -> None:
    fixture_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"n_iou_thresholds": n_t, "cells": cells}
    out = fixture_dir / "cells.json"
    # ``sort_keys=True`` + fixed indent gives byte-identical output across
    # re-runs (the dict insertion order is already deterministic but
    # sorting cheaply pins it).
    with out.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, sort_keys=True, indent=2)
        fh.write("\n")
    if meta is not None:
        meta_path = fixture_dir / "meta.json"
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, sort_keys=True, indent=2)
            fh.write("\n")


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _build_cal_perfect(root: Path) -> None:
    """Three classes, 10 detections each, clean monotone score ramp,
    all matched at every threshold."""
    cells: list[dict[str, Any]] = []
    for class_id in (1, 2, 3):
        scores = np.linspace(0.05, 1.0, 10, dtype=np.float64)
        matched = np.ones(scores.size, dtype=bool)
        ignore = np.zeros(scores.size, dtype=bool)
        cells.append(_cell_json(class_id, scores, matched, ignore, _DEFAULT_T))
    _write(root / "cal_perfect", cells, _DEFAULT_T)


def _build_cal_overconfident(root: Path) -> None:
    """DETR-style bimodal scores: a no-object tail near 0.0-0.1 and a
    high-confidence cluster near 0.9-1.0 with ~50% accuracy."""
    rng = np.random.default_rng(seed=0xDE111)
    cells: list[dict[str, Any]] = []
    for class_id in (1, 2):
        # 30 low-score (no-object tail): scores in [0.0, 0.1], all wrong.
        # Below min_score=0.05 about half drop out — that's the point of
        # the fixture (exercises P3 alongside the bimodal shape).
        lo_scores = rng.uniform(0.0, 0.1, size=30).astype(np.float64)
        lo_matched = np.zeros(lo_scores.size, dtype=bool)
        # 30 high-score (deployment cluster): scores in [0.9, 1.0],
        # ~50% wrong (overconfident).
        hi_scores = rng.uniform(0.9, 1.0, size=30).astype(np.float64)
        hi_matched = rng.uniform(0.0, 1.0, size=30) < 0.5
        scores = np.concatenate([lo_scores, hi_scores])
        matched = np.concatenate([lo_matched, hi_matched])
        ignore = np.zeros(scores.size, dtype=bool)
        # Sort descending to mirror the Rust cell shape (dt_scores is
        # sorted-descending; see accumulate.rs:62).
        order = np.argsort(-scores, kind="stable")
        cells.append(_cell_json(class_id, scores[order], matched[order], ignore[order], _DEFAULT_T))
    _write(root / "cal_overconfident", cells, _DEFAULT_T)


def _build_cal_ignore_regions(root: Path) -> None:
    """Half the detections have ``dt_ignore[0, d] = True`` at the
    iou_index=0 threshold. The oracle must drop them from the
    histogram (P2 / R3)."""
    rng = np.random.default_rng(seed=0x16407E)
    cells: list[dict[str, Any]] = []
    for class_id in (1, 2):
        scores = np.sort(rng.uniform(0.1, 1.0, size=20).astype(np.float64))[::-1]
        matched = rng.uniform(0.0, 1.0, size=20) < 0.6
        ignore = np.zeros(scores.size, dtype=bool)
        # Flip every other detection's ignore bit on at iou_index=0.
        # The cell-builder tiles this row across T; for this fixture
        # the same ignore pattern holds at every T (the oracle reads
        # only iou_index=0 unless overridden).
        ignore[::2] = True
        cells.append(_cell_json(class_id, scores, matched, ignore, _DEFAULT_T))
    _write(root / "cal_ignore_regions", cells, _DEFAULT_T)


def _build_cal_segm_smoketest(root: Path) -> None:
    """Same data as cal_perfect, tagged ``iou_type='segm'`` in a
    sidecar. The cell shape is identical across bbox/segm/boundary
    (Shape 1) — this fixture asserts iou_type-genericity at the data
    level."""
    cells: list[dict[str, Any]] = []
    for class_id in (1, 2, 3):
        scores = np.linspace(0.05, 1.0, 10, dtype=np.float64)
        matched = np.ones(scores.size, dtype=bool)
        ignore = np.zeros(scores.size, dtype=bool)
        cells.append(_cell_json(class_id, scores, matched, ignore, _DEFAULT_T))
    _write(
        root / "cal_segm_smoketest",
        cells,
        _DEFAULT_T,
        meta={"iou_type": "segm"},
    )


def _build_cal_keypoints_smoketest(root: Path) -> None:
    """Same data as cal_overconfident, tagged ``iou_type='keypoints'``.

    Note: the canonical keypoints ``max_dets=[20]`` cap is applied by
    the upstream streaming evaluator before cells reach calibration —
    the oracle does *not* re-apply it. This fixture documents the
    expectation; harness wiring (Unit 4c) handles the cap end-to-end.
    """
    rng = np.random.default_rng(seed=0x6E1F0)
    cells: list[dict[str, Any]] = []
    for class_id in (1, 2):
        lo_scores = rng.uniform(0.0, 0.1, size=30).astype(np.float64)
        lo_matched = np.zeros(lo_scores.size, dtype=bool)
        hi_scores = rng.uniform(0.9, 1.0, size=30).astype(np.float64)
        hi_matched = rng.uniform(0.0, 1.0, size=30) < 0.5
        scores = np.concatenate([lo_scores, hi_scores])
        matched = np.concatenate([lo_matched, hi_matched])
        ignore = np.zeros(scores.size, dtype=bool)
        order = np.argsort(-scores, kind="stable")
        cells.append(_cell_json(class_id, scores[order], matched[order], ignore[order], _DEFAULT_T))
    _write(
        root / "cal_keypoints_smoketest",
        cells,
        _DEFAULT_T,
        meta={"iou_type": "keypoints", "max_dets": 20},
    )


def main() -> None:
    root = Path(__file__).parent
    _build_cal_perfect(root)
    _build_cal_overconfident(root)
    _build_cal_ignore_regions(root)
    _build_cal_segm_smoketest(root)
    _build_cal_keypoints_smoketest(root)
    print(f"Wrote 5 fixtures under {root}/")


if __name__ == "__main__":
    main()
