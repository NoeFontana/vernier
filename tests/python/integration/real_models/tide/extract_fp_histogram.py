"""Extract FP-IoU histograms from rf-detr predictions for ADR-0022 ratification.

Sister to ``run.py``: same inputs (cached predictions on COCO val2017),
different output. While ``run.py`` emits a TIDE report (per-bin ΔmAP),
this script emits the per-FP `(iou_same, iou_cross)` arrays plus a
sweep of "bin-as-Bkg fraction at candidate `t_b`" — the data the
ADR-0022 decision gate calls for.

Per-cell output:

- Per-axis 50-bin histograms on `[0.0, 1.0]` for `iou_same`,
  `iou_cross`, and `max(iou_same, iou_cross)`.
- A `bkg_fraction_at` curve: at each candidate `t_b` in
  `0, 0.01, 0.02, …, 0.50`, the fraction of FPs that would be binned
  as Bkg.

Usage::

    uv run --extra real-models python -m \\
        tests.python.integration.real_models.tide.extract_fp_histogram \\
        --model both --kernel all --output fp-histogram-report.json

The decision: if the `bkg_fraction_at` curve has a clear plateau near
the current `t_b` default (0.1 for segm, 0.05 for boundary), the
default is ratified. If it doesn't, the threshold needs adjustment.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

import vernier
from vernier._tide import KernelName

from ._cli_common import (
    KERNEL_FACTORIES,
    ModelName,
    emit,
    load_predictions,
    make_arg_parser,
    resolve_cells,
)

#: 50 equal-width bins on [0, 1] — fine enough to see structure, coarse
#: enough that single-detection noise doesn't dominate.
_HIST_BINS = 50

#: Candidate `t_b` values for the bkg-fraction sweep — every 0.01 from
#: 0 through 0.50 (inclusive, 51 points). Coarse enough that the JSON
#: stays readable; fine enough to spot a plateau near the current
#: defaults (0.1 / 0.05).
_T_B_CANDIDATES: np.ndarray = np.round(np.arange(51) * 0.01, 2)


def _histogram(values: np.ndarray, *, bins: int = _HIST_BINS) -> dict[str, list[float]]:
    """50-bin histogram on `[0, 1]` returned as JSON-friendly lists."""
    counts, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    return {"bin_edges": edges.tolist(), "counts": counts.tolist()}


def _bkg_fraction_curve(max_iou: np.ndarray) -> list[dict[str, float]]:
    """Sweep `t_b` and report the fraction of FPs that bin as Bkg.

    Sort once + ``searchsorted`` collapses 51 array-walks into one
    `O(n log n)` sort + 51 binary searches.
    """
    if max_iou.size == 0:
        return [{"t_b": float(t_b), "fraction": 0.0} for t_b in _T_B_CANDIDATES]
    sorted_max = np.sort(max_iou)
    counts = np.searchsorted(sorted_max, _T_B_CANDIDATES, side="left")
    fractions = counts / max_iou.size
    return [
        {"t_b": float(t_b), "fraction": float(f)}
        for t_b, f in zip(_T_B_CANDIDATES, fractions, strict=True)
    ]


def _run_cell(
    model_name: ModelName,
    kernel_name: KernelName,
    gt_bytes: bytes,
    predictions: bytes,
) -> dict[str, Any]:
    iou = KERNEL_FACTORIES[kernel_name]()
    t0 = time.perf_counter()
    h = vernier.fp_iou_histogram(gt_bytes, predictions, iou=iou)
    elapsed = time.perf_counter() - t0
    max_iou = np.maximum(h.iou_same, h.iou_cross)
    return {
        "model": model_name,
        "kernel": kernel_name,
        "t_f": h.t_f,
        "n_total_dts": h.n_total_dts,
        "n_fps": h.n_fps,
        "iou_same_histogram": _histogram(h.iou_same),
        "iou_cross_histogram": _histogram(h.iou_cross),
        "max_iou_histogram": _histogram(max_iou),
        "bkg_fraction_at": _bkg_fraction_curve(max_iou),
        "wall_clock_seconds": elapsed,
    }


def _print_summary(cells: list[dict[str, Any]]) -> None:
    """Compact per-cell summary on stderr."""
    print(file=sys.stderr)
    print(
        f"{'model':<10} {'kernel':<10} {'n_fps':>8} {'bkg@0.05':>10} "
        f"{'bkg@0.10':>10} {'bkg@0.20':>10} {'sec':>7}",
        file=sys.stderr,
    )
    for c in cells:
        sweep = {entry["t_b"]: entry["fraction"] for entry in c["bkg_fraction_at"]}
        print(
            f"{c['model']:<10} {c['kernel']:<10} {c['n_fps']:>8} "
            f"{sweep.get(0.05, float('nan')):>10.4f} "
            f"{sweep.get(0.10, float('nan')):>10.4f} "
            f"{sweep.get(0.20, float('nan')):>10.4f} "
            f"{c['wall_clock_seconds']:>7.2f}",
            file=sys.stderr,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_arg_parser(
        "rf-detr FP-IoU histogram extraction on COCO val2017 (ADR-0022).",
        kernel_help="vernier kernel to extract histograms from (default: all compatible)",
    )
    args = parser.parse_args(argv)
    cells = resolve_cells(args.model, args.kernel)
    if not cells:
        print(
            f"no compatible (model, kernel) pairs for --model={args.model} --kernel={args.kernel}",
            file=sys.stderr,
        )
        return 2

    gt_bytes, predictions = load_predictions(cells)

    results: list[dict[str, Any]] = []
    for model_name, kernel_name in cells:
        print(f"[{model_name}/{kernel_name}] extracting FP-IoU histogram...", file=sys.stderr)
        results.append(_run_cell(model_name, kernel_name, gt_bytes, predictions[model_name]))

    emit(
        {
            "rfdetr_version": "1.6.5.post0",
            "dataset": "coco-val2017",
            "schema_version": 1,
            "cells": results,
        },
        args.output,
    )
    _print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
