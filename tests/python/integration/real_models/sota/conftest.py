"""Fixtures for the Hugging Face SOTA validation harness.

Skips the entire module when the ``real-models`` extra is absent (it's
``transformers`` here, in contrast to ``tide/conftest.py`` which gates
on ``rfdetr``). When present, exposes a session-scoped fixture that
loads cached predictions or runs DETR-R50 inference once per session.

Inference is the cost driver — DETR-R50 on COCO val2017 takes ~12-15h
on an 8-core CPU first time (5000 images x ~9s/image; see
``_detr_predict.py`` for the per-image breakdown). The cache key is
``(model_name, hub_revision, dataset_id)`` (see
``real_predictions_cache.detr_resnet50_cache_filename``); subsequent
runs read from disk and skip the model entirely.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from real_predictions_cache import detr_resnet50_cache_path

# Probe every dep the SOTA harness actually loads at fixture-create time,
# not just `transformers` — a partial install (transformers present, timm
# or PIL missing) would otherwise ImportError mid-fixture instead of the
# clean skip the docstring promises.
_REAL_MODELS_REASON = "SOTA harness needs the `real-models` extra: `uv sync --extra real-models`"
for _mod in ("transformers", "torch", "huggingface_hub", "timm", "PIL"):
    pytest.importorskip(_mod, reason=_REAL_MODELS_REASON)


@pytest.fixture(scope="session")
def coco_gt_path(coco_val_root: Path) -> Path:
    return coco_val_root / "instances_val2017.json"


@pytest.fixture(scope="session")
def coco_gt_dict(coco_gt_path: Path) -> dict[str, Any]:
    return json.loads(coco_gt_path.read_bytes())


@pytest.fixture(scope="session")
def coco_val_image_dir(coco_val_root: Path) -> Path:
    return coco_val_root / "val2017"


@pytest.fixture(scope="session")
def detr_predictions_path(
    coco_gt_dict: dict[str, Any],
    coco_val_image_dir: Path,
) -> Path:
    """Run-or-load DETR-R50 predictions on COCO val2017.

    Returns the cache path (a JSON file in COCO ``loadRes`` format),
    not the bytes — tests pass this path straight into the parity
    harness, which reads it back through pycocotools' loader and
    vernier's JSON ingest. Re-using the path (vs. preloading bytes)
    keeps the two oracles' loaders independent.
    """
    from ._detr_predict import predict_coco_val

    cache = detr_resnet50_cache_path()
    if not cache.is_file():
        predict_coco_val(
            gt=coco_gt_dict,
            image_dir=coco_val_image_dir,
            cache_path=cache,
        )
    return cache
