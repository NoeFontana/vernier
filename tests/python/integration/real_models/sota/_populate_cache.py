"""CLI: populate the Hugging Face SOTA prediction cache.

The bench harness's ``coco_val2017_detr_r50_*`` workload (under
``bench/bench/workloads/real_predictions.py``) reads predictions from
disk only — it has no torch / transformers / huggingface_hub dep. This
script is the populator: it depends on the heavy ``[real-models]``
extra and is shelled into by ``tools/fetch-real-predictions.sh --detr``
(via ``real_predictions_cache.populate_detr_resnet50``).

Inference is the cost driver (~12-15h on an 8-core CPU for COCO
val2017 — 5000 images x ~9s/image); a cache hit is seconds. Same
cache as the pytest-driven SOTA harness — running ``pytest -m
real_models tests/python/integration/real_models/sota`` is the
test-time equivalent.

Usage::

    uv run --extra real-models python -m \\
        tests.python.integration.real_models.sota._populate_cache --detr
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from coco_val_cache import GT_FILENAME, IMAGES_DIRNAME
from coco_val_cache import cache_root as _coco_cache_root
from real_predictions_cache import detr_resnet50_cache_path

from ._detr_predict import predict_coco_val as _detr_predict_coco_val


def _coco_val_root() -> Path:
    """Locate the val2017 cache; abort cleanly if it's not populated.

    Mirrors ``tide/_cli_common.coco_val_root`` — copied rather than
    reused because this module sits parallel to ``tide/`` and shouldn't
    take a dep on rfdetr-specific plumbing.
    """
    root = _coco_cache_root()
    gt = root / GT_FILENAME
    images = root / IMAGES_DIRNAME
    if not gt.is_file() or not images.is_dir():
        raise SystemExit(
            f"COCO val2017 not found at {root}: need both "
            f"{GT_FILENAME} and {IMAGES_DIRNAME}/ images. "
            f"Run `./tools/fetch-coco-val.sh --with-images` to populate "
            f"the cache. Override the path with VERNIER_COCO_CACHE."
        )
    return root


def _populate_detr() -> Path:
    root = _coco_val_root()
    gt_dict = json.loads((root / GT_FILENAME).read_bytes())
    cache_path = detr_resnet50_cache_path()
    _detr_predict_coco_val(
        gt=gt_dict,
        image_dir=root / IMAGES_DIRNAME,
        cache_path=cache_path,
    )
    return cache_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tests.python.integration.real_models.sota._populate_cache",
        description="Run Hugging Face SOTA inference on COCO val2017, "
        "populate the prediction cache.",
    )
    parser.add_argument(
        "--detr",
        action="store_true",
        help="Run facebook/detr-resnet-50 (revision pinned by "
        "real_predictions_cache.DETR_RESNET50_REVISION) and write COCO "
        "results JSON to the cache.",
    )
    args = parser.parse_args(argv)

    if not args.detr:
        parser.error("at least one model flag is required (--detr)")

    path = _populate_detr()
    print(f"DETR-R50 predictions cached: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
