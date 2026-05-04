"""Shared plumbing for the rf-detr-anchored TIDE CLIs.

The two CLI scripts in this directory — ``run.py`` (TIDE report) and
``extract_fp_histogram.py`` (ADR-0022 ratification data) — share
identical argparse, cell-resolution, and prediction-loading logic;
they only diverge in the per-cell computation and the report shape.
This module owns the shared parts; each CLI keeps its own ``_run_cell``
+ ``_print_summary`` + report-payload assembly.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from coco_val_cache import cache_root as _coco_cache_dir

from vernier._tide import KernelName
from vernier.instance import Bbox, Boundary, Segm

from ._rfdetr_predict import (
    ModelName,
    cache_filename,
    predict_coco_val,
    predictions_cache_root,
)

ModelArg = Literal["nano", "segnano", "both"]
KernelArg = Literal["bbox", "segm", "boundary", "all"]

#: Per-model kernel compatibility. RFDETRNano is bbox-only;
#: RFDETRSegNano produces masks usable by all three kernels.
MODEL_KERNELS: dict[ModelName, frozenset[KernelName]] = {
    "nano": frozenset({"bbox"}),
    "segnano": frozenset({"bbox", "segm", "boundary"}),
}

#: Kernel-name → kernel-instance factory. Boundary uses the canonical
#: ADR-0010 dilation ratio (0.02 for COCO).
KERNEL_FACTORIES: dict[KernelName, Any] = {
    "bbox": Bbox,
    "segm": Segm,
    "boundary": lambda: Boundary(dilation_ratio=0.02),
}


def make_arg_parser(description: str, *, kernel_help: str) -> argparse.ArgumentParser:
    """Build the standard `--model / --kernel / --output` parser shared
    by both CLIs. `kernel_help` differs per-CLI (one decomposes, the
    other extracts histograms) so it's a parameter."""
    parser = argparse.ArgumentParser(description=description)
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
        help=kernel_help,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="path to write the JSON report (default: stdout)",
    )
    return parser


def resolve_cells(model_arg: ModelArg, kernel_arg: KernelArg) -> list[tuple[ModelName, KernelName]]:
    """Expand the ``--model`` x ``--kernel`` cross product into compatible pairs.

    A request like ``--model nano --kernel boundary`` yields nothing
    (nano has no masks); we surface that as an empty cell list so the
    caller can exit cleanly with a useful error.
    """
    models: tuple[ModelName, ...] = ("nano", "segnano") if model_arg == "both" else (model_arg,)
    kernels: tuple[KernelName, ...] = (
        ("bbox", "segm", "boundary") if kernel_arg == "all" else (kernel_arg,)
    )
    return [(m, k) for m in models for k in kernels if k in MODEL_KERNELS[m]]


def coco_val_root() -> Path:
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
            f"Run `./tools/fetch-coco-val.sh --with-images` to populate "
            f"the cache. Override the path with VERNIER_COCO_CACHE."
        )
    return root


def load_predictions(
    cells: Sequence[tuple[ModelName, KernelName]],
) -> tuple[bytes, dict[ModelName, bytes]]:
    """Locate val2017, load GT bytes, and ensure predictions for every
    distinct model in ``cells``. Returns ``(gt_bytes, predictions_by_model)``.

    First-time inference is the cost driver (~30 min for SegNano on
    CPU); :func:`predict_coco_val` short-circuits on a cache hit so
    subsequent runs are seconds.
    """
    coco_root = coco_val_root()
    gt_bytes = (coco_root / "instances_val2017.json").read_bytes()
    gt_dict = json.loads(gt_bytes)
    image_dir = coco_root / "val2017"
    cache = predictions_cache_root()

    predictions: dict[ModelName, bytes] = {}
    for model_name in {m for m, _ in cells}:
        print(f"[{model_name}] loading or generating predictions...", file=sys.stderr)
        predictions[model_name] = predict_coco_val(
            model_name=model_name,
            gt=gt_dict,
            image_dir=image_dir,
            cache_path=cache / cache_filename(model_name),
        )
    return gt_bytes, predictions


def emit(payload: dict[str, Any], output: Path | None) -> None:
    """Write `payload` as pretty JSON to `output` (or stdout)."""
    rendered = json.dumps(payload, indent=2)
    if output is not None:
        output.write_text(rendered)
    else:
        print(rendered)
