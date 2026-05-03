"""Shared fixtures for the real-model validation harnesses.

The ``real_models/`` subtree is the convention for end-to-end validation
runs that exercise vernier against actual model predictions on real
datasets — heavy, opt-in via ``@pytest.mark.real_models``, and gated on
both the relevant optional-dependency extra and the dataset cache.

This conftest sources the COCO val2017 layout from the same
``VERNIER_COCO_CACHE`` env var the parity smokes use (see
``tests/python/coco_val_paths.py``), extended with the additional
expectation that the image directory ``val2017/`` lives alongside
``instances_val2017.json``. Real-model harnesses need pixels to run
inference, while the existing parity smokes only consume cached
prediction JSONs — that's the only delta.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ...coco_val_paths import require_coco_val_root_with_images


@pytest.fixture(scope="session")
def coco_val_root() -> Path:
    return require_coco_val_root_with_images()


@pytest.fixture(scope="session")
def predictions_cache_root() -> Path:
    pytest.importorskip(
        "platformdirs",
        reason="real-model harness needs the `real-models` extra: `uv sync --extra real-models`",
    )
    from .tide._rfdetr_predict import predictions_cache_root as _root

    return _root()
