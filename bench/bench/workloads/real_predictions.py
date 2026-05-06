"""Disk-only adapter for real-model predictions cells.

Predictions come from two sources, both populating the same cache:

- **Mask R-CNN R50-FPN (Detectron2 model zoo)** — downloaded from a
  pinned Hugging Face URL via ``tools/fetch-real-predictions.sh
  --maskrcnn``. The bench harness reads the cached JSON without any
  inference dependency.
- **rf-detr (Nano / SegNano)** — inferred locally by the TIDE
  validation harness (which depends on the heavy ``real-models`` extra:
  torch, rfdetr, supervision). The bench harness reads the same JSON
  the TIDE conftest writes; running ``pytest -m real_models`` is
  enough to populate the cache.

The harness env stays light: this module imports nothing more than
``real_predictions_cache`` for path resolution.
"""

from __future__ import annotations

from pathlib import Path

from real_predictions_cache import (
    RFDETR_VERSION,
    RfdetrModelName,
    maskrcnn_cache_path,
    rfdetr_cache_path,
)

# Workload identifiers — kept as constants so the registry, tests, and
# CLI error messages don't drift.
MASKRCNN_R50FPN_WORKLOAD_ID = "coco_val2017_maskrcnn_r50fpn_d2_v1"
RFDETR_NANO_WORKLOAD_ID = f"coco_val2017_rfdetr_nano_v{RFDETR_VERSION}"
RFDETR_SEGNANO_WORKLOAD_ID = f"coco_val2017_rfdetr_segnano_v{RFDETR_VERSION}"


def maskrcnn_dt_path() -> Path:
    """Return the Mask R-CNN R50-FPN cached prediction JSON, or raise.

    The adapter is read-only — it never invokes the fetch tooling.
    Pointing users at ``tools/fetch-real-predictions.sh --maskrcnn``
    keeps the bench env free of HTTP / cryptography deps and surfaces
    a misconfigured cache as a hard error (rather than a silent zero-
    detection benchmark).
    """
    path = maskrcnn_cache_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Mask R-CNN R50-FPN prediction cache missing at {path}. "
            f"Run `./tools/fetch-real-predictions.sh --maskrcnn` to "
            f"download it (~250 MB, pinned URL + SHA256). The bench "
            f"adapter is read-only by design — it does not invoke the "
            f"fetcher implicitly."
        )
    return path


def rfdetr_dt_path(model_name: RfdetrModelName) -> Path:
    """Return the rf-detr cached prediction JSON for ``model_name``."""
    path = rfdetr_cache_path(model_name)
    if not path.is_file():
        raise FileNotFoundError(
            f"rf-detr ({model_name}) prediction cache missing at {path}. "
            f"Populate it by running `pytest -m real_models "
            f"tests/python/integration/real_models/tide/test_tide_real_models.py` "
            f"with the `real-models` extra installed (rfdetr, torch, "
            f"supervision; ~5 GB on first install). Once populated, the "
            f"bench harness reads the JSON without any inference dep."
        )
    return path
