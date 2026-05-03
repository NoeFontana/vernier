"""Fixtures for the rf-detr-anchored TIDE validation harness.

Skips the entire module when the ``real-models`` extra is absent. When
present, exposes a single ``predictions_for(model_name)`` factory that
loads cached predictions or runs inference once per session per model.

Inference is the cost driver — RFDETRSegNano on COCO val2017 takes
~30 minutes on CPU first time. The cache key is
``(model_name, rfdetr_version, dataset_id)`` (see
``_rfdetr_predict.cache_filename``); subsequent runs read from disk and
skip the model entirely.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from ._rfdetr_predict import ModelName, cache_filename, predict_coco_val

pytest.importorskip(
    "rfdetr",
    reason="real-model harness needs the `real-models` extra: `uv sync --extra real-models`",
)


@pytest.fixture(scope="session")
def coco_gt_bytes(coco_val_root: Path) -> bytes:
    return (coco_val_root / "instances_val2017.json").read_bytes()


@pytest.fixture(scope="session")
def coco_gt_dict(coco_gt_bytes: bytes) -> dict[str, Any]:
    return json.loads(coco_gt_bytes)


@pytest.fixture(scope="session")
def coco_val_image_dir(coco_val_root: Path) -> Path:
    return coco_val_root / "val2017"


@pytest.fixture(scope="session")
def predictions_for(
    coco_gt_dict: dict[str, Any],
    coco_val_image_dir: Path,
    predictions_cache_root: Path,
) -> Callable[[ModelName], bytes]:
    """Factory: ``model_name`` → COCO JSON predictions bytes.

    Memoizes per-session so multiple tests asking for the same model
    pay one disk read + one inference run at most. The disk cache
    in :mod:`._rfdetr_predict` survives across sessions; the in-memory
    memo here just dedups within the session.
    """
    memo: dict[ModelName, bytes] = {}

    def get(model_name: ModelName) -> bytes:
        if model_name not in memo:
            memo[model_name] = predict_coco_val(
                model_name=model_name,
                gt=coco_gt_dict,
                image_dir=coco_val_image_dir,
                cache_path=predictions_cache_root / cache_filename(model_name),
            )
        return memo[model_name]

    return get
