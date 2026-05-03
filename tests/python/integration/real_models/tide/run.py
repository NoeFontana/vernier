"""Ad-hoc CLI for the rf-detr TIDE validation run.

Sibling to the pytest harness in this directory: same machinery, a
different framing. The pytest tests gate structural invariants
(coherence, determinism); this script produces a *report* — a JSON
document with the per-bin ΔmAP values, baseline mAP, wall-clock and
peak-RSS measurements — that humans read after a 0.x.x TIDE bump to
sanity-check that real-model behavior didn't regress.

The script is **not** a regression gate. Hardware varies; bin-mass
distributions are model-dependent. The README documents the targets
qualitatively; the JSON output gives reviewers something concrete to
diff. ::

    uv run --extra real-models python -m \\
        tests.python.integration.real_models.tide.run \\
        --model both --kernel all --output validation-report.json

Skips inference when cached predictions exist (same cache as the
pytest harness via :mod:`._rfdetr_predict`); first run on a clean
machine takes ~30 minutes for SegNano on CPU.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Sequence
from typing import Any

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


def _peak_rss_mb() -> float:
    """Current process RSS in megabytes (psutil-backed).

    Returns ``-1.0`` if psutil isn't importable — the report still
    serializes; the metric is just absent. psutil is a transitive
    dependency of the ``real-models`` extra (via accelerate) so this
    branch only fires under unusual environments.
    """
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return -1.0


def _run_cell(
    model_name: ModelName,
    kernel_name: KernelName,
    gt_bytes: bytes,
    predictions: bytes,
) -> dict[str, Any]:
    iou = KERNEL_FACTORIES[kernel_name]()
    t0 = time.perf_counter()
    report = vernier.error_decomposition(gt_bytes, predictions, iou=iou)
    elapsed = time.perf_counter() - t0

    return {
        "model": model_name,
        "kernel": kernel_name,
        "baseline_map": report.baseline_map,
        "delta": dict(report.delta),
        "delta_all_fp_removed": report.delta_all_fp_removed,
        "config": {
            "t_f": report.config.t_f,
            "t_b": report.config.t_b,
            "kernel": report.config.kernel,
        },
        "wall_clock_seconds": elapsed,
        "peak_rss_mb_after_cell": _peak_rss_mb(),
    }


def _print_summary(cells: list[dict[str, Any]]) -> None:
    """Human-readable summary table to stderr (so stdout JSON stays clean)."""
    print(file=sys.stderr)
    print(
        f"{'model':<10} {'kernel':<10} {'baseline':>9} "
        f"{'+all_fp':>9} {'cls':>7} {'loc':>7} {'both':>7} "
        f"{'dupe':>7} {'bkg':>7} {'missed':>7} {'sec':>7}",
        file=sys.stderr,
    )
    for c in cells:
        d = c["delta"]
        print(
            f"{c['model']:<10} {c['kernel']:<10} "
            f"{c['baseline_map']:>9.4f} {c['delta_all_fp_removed']:>9.4f} "
            f"{d['cls']:>7.4f} {d['loc']:>7.4f} {d['both']:>7.4f} "
            f"{d['dupe']:>7.4f} {d['bkg']:>7.4f} {d['missed']:>7.4f} "
            f"{c['wall_clock_seconds']:>7.2f}",
            file=sys.stderr,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_arg_parser(
        "rf-detr TIDE validation run on COCO val2017",
        kernel_help="vernier kernel to decompose against (default: all compatible)",
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
        print(f"[{model_name}/{kernel_name}] running TIDE...", file=sys.stderr)
        results.append(_run_cell(model_name, kernel_name, gt_bytes, predictions[model_name]))

    emit(
        {
            "rfdetr_version": "1.6.5.post0",
            "dataset": "coco-val2017",
            "cells": results,
        },
        args.output,
    )
    _print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
