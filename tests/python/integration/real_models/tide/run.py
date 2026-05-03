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

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from tests.python.coco_val_paths import cache_dir as _coco_cache_dir

import vernier
from vernier import Bbox, Boundary, Segm
from vernier._tide import KernelName

from ._rfdetr_predict import (
    ModelName,
    cache_filename,
    predict_coco_val,
    predictions_cache_root,
)

ModelArg = Literal["nano", "segnano", "both"]
KernelArg = Literal["bbox", "segm", "boundary", "all"]

_MODEL_KERNELS: dict[ModelName, frozenset[KernelName]] = {
    "nano": frozenset({"bbox"}),
    "segnano": frozenset({"bbox", "segm", "boundary"}),
}

_KERNEL_FACTORIES: dict[KernelName, Any] = {
    "bbox": Bbox,
    "segm": Segm,
    "boundary": lambda: Boundary(dilation_ratio=0.02),
}


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="rf-detr TIDE validation run on COCO val2017",
    )
    parser.add_argument(
        "--model",
        choices=("nano", "segnano", "both"),
        default="both",
        help="rf-detr model class to run (default: both)",
    )
    parser.add_argument(
        "--kernel",
        choices=("bbox", "segm", "boundary", "all"),
        default="all",
        help="vernier kernel to decompose against (default: all compatible)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="path to write the JSON report (default: stdout)",
    )
    return parser.parse_args(argv)


def _resolve_cells(
    model_arg: ModelArg, kernel_arg: KernelArg
) -> list[tuple[ModelName, KernelName]]:
    """Expand the ``--model`` x ``--kernel`` cross product into compatible pairs.

    A request like ``--model nano --kernel boundary`` yields nothing
    (nano has no masks); we surface that as an empty cell list so the
    caller can exit cleanly with a useful error.
    """
    models: tuple[ModelName, ...] = ("nano", "segnano") if model_arg == "both" else (model_arg,)
    kernels: tuple[KernelName, ...] = (
        ("bbox", "segm", "boundary") if kernel_arg == "all" else (kernel_arg,)
    )
    return [(m, k) for m in models for k in kernels if k in _MODEL_KERNELS[m]]


def _coco_val_root() -> Path:
    """Locate the val2017 cache; abort cleanly if it's not populated.

    Mirrors :func:`tests.python.coco_val_paths.require_coco_val_root_with_images`
    but raises :class:`SystemExit` instead of calling ``pytest.skip``,
    since this is a CLI not a test fixture.
    """
    root = _coco_cache_dir()
    gt = root / "instances_val2017.json"
    images = root / "val2017"
    if not gt.is_file() or not images.is_dir():
        raise SystemExit(
            f"COCO val2017 not found at {root}: need both "
            f"instances_val2017.json and val2017/ images. "
            f"Run ./tools/fetch-coco-val.sh and place val2017/ images "
            f"alongside the GT JSON. Override the path with VERNIER_COCO_CACHE."
        )
    return root


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
    iou = _KERNEL_FACTORIES[kernel_name]()
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
    args = _parse_args(argv)
    cells = _resolve_cells(args.model, args.kernel)
    if not cells:
        print(
            f"no compatible (model, kernel) pairs for --model={args.model} --kernel={args.kernel}",
            file=sys.stderr,
        )
        return 2

    coco_root = _coco_val_root()
    gt_bytes = (coco_root / "instances_val2017.json").read_bytes()
    gt_dict = json.loads(gt_bytes)
    image_dir = coco_root / "val2017"
    cache_root = predictions_cache_root()

    predictions_by_model: dict[ModelName, bytes] = {}
    for model_name in {m for m, _ in cells}:
        print(f"[{model_name}] loading or generating predictions...", file=sys.stderr)
        predictions_by_model[model_name] = predict_coco_val(
            model_name=model_name,
            gt=gt_dict,
            image_dir=image_dir,
            cache_path=cache_root / cache_filename(model_name),
        )

    results: list[dict[str, Any]] = []
    for model_name, kernel_name in cells:
        print(f"[{model_name}/{kernel_name}] running TIDE...", file=sys.stderr)
        results.append(
            _run_cell(
                model_name,
                kernel_name,
                gt_bytes,
                predictions_by_model[model_name],
            )
        )

    payload = {
        "rfdetr_version": "1.6.5.post0",
        "dataset": "coco-val2017",
        "cells": results,
    }
    rendered = json.dumps(payload, indent=2)
    if args.output is not None:
        args.output.write_text(rendered)
    else:
        print(rendered)
    _print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
