"""Fixtures for the TIDE validation harness.

The harness exercises ``vernier.error_decomposition`` against real-model
predictions on COCO val2017. Two prediction sources land in the same
on-disk cache and are dispatched through one fixture factory:

- **rf-detr (Nano / SegNano)** — inferred locally by this directory's
  ``_rfdetr_predict`` module on the heavy ``real-models`` extra. The
  cache key is ``(model_name, rfdetr_version, dataset_id)``; bumping
  the rfdetr pin (an ADR-level operation) invalidates by construction.
- **DETR-R50** — inferred locally by the sibling SOTA harness
  (``tests/python/integration/real_models/sota/``). The TIDE cells
  reuse the same cache via :func:`real_predictions_cache.detr_resnet50_cache_path`;
  this directory does **not** own DETR inference. DETR has no segm
  output, so the matrix only exercises it on the bbox kernel.

Inference is the cost driver — RFDETRSegNano ~30 min, DETR-R50
~12-15 h on an 8-core CPU first time. Subsequent runs read from disk
and skip the model entirely. Per-model skip semantics:

- The rfdetr cells skip when ``rfdetr`` isn't importable (``real-models``
  extra missing) or when the dataset cache is unprovisioned.
- The DETR-R50 cell skips when its cache file isn't on disk — the
  TIDE harness does not invoke the SOTA populator to keep this tree's
  fixture cost bounded; if the cache is missing we point the user at
  the SOTA harness rather than spending ~12 h here.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import pytest

from ._rfdetr_predict import cache_filename, predict_coco_val

#: Model identifiers the TIDE matrix recognises. Superset of
#: :data:`._rfdetr_predict.ModelName` (which stays rfdetr-only by
#: design — that module owns rfdetr inference and nothing else). The
#: ``detr-r50`` literal points at the SOTA-harness-owned cache.
TideModelName = Literal["nano", "segnano", "detr-r50"]


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
) -> Callable[[TideModelName], bytes]:
    """Factory: ``model_name`` → COCO JSON predictions bytes.

    Dispatches by model id:

    - ``"nano"`` / ``"segnano"`` go through rf-detr inference (run-or-load).
      Skips when ``rfdetr`` is not importable.
    - ``"detr-r50"`` reads the SOTA-harness-owned cache. Skips when the
      cache file is absent — this directory does not own DETR
      inference, and shelling into a ~12 h populate from a test
      fixture would surprise both CI and local runs.

    Memoizes per-session so multiple tests asking for the same model
    pay one disk read + one inference run at most. The disk caches in
    :mod:`._rfdetr_predict` and :mod:`real_predictions_cache` survive
    across sessions; the in-memory memo here just dedups within the
    session.
    """
    memo: dict[TideModelName, bytes] = {}

    def get(model_name: TideModelName) -> bytes:
        if model_name in memo:
            return memo[model_name]
        if model_name == "detr-r50":
            payload = _load_detr_r50_cache_bytes()
        else:
            # Defer rfdetr import-check to call time so the DETR-R50
            # cell can run on a host without the rfdetr extra (the
            # cache may have been populated by the SOTA harness alone,
            # or fetched via ``./tools/fetch-real-predictions.sh --detr``).
            pytest.importorskip(
                "rfdetr",
                reason=(
                    "rf-detr TIDE cell needs the `real-models` extra: `uv sync --extra real-models`"
                ),
            )
            payload = predict_coco_val(
                model_name=model_name,
                gt=coco_gt_dict,
                image_dir=coco_val_image_dir,
                cache_path=predictions_cache_root / cache_filename(model_name),
            )
        memo[model_name] = payload
        return payload

    return get


def _load_detr_r50_cache_bytes() -> bytes:
    """Read the SOTA-harness-owned DETR-R50 prediction cache, or skip.

    The TIDE tree borrows the DETR-R50 predictions cache by reference
    (per ADR-0021's "oracle + real-model harness" split; PR #265 owns
    the inference contract). We deliberately do not invoke the SOTA
    populator here: a missing cache surfaces as a clean
    ``pytest.skip`` that points the user at the SOTA harness or the
    fetch shim, rather than spending ~12 h on a TIDE fixture.
    """
    from real_predictions_cache import detr_resnet50_cache_path

    cache_path = detr_resnet50_cache_path()
    if not cache_path.is_file():
        pytest.skip(
            f"DETR-R50 prediction cache missing at {cache_path}. "
            f"Populate via `./tools/fetch-real-predictions.sh --detr` or "
            f"`pytest -m real_models tests/python/integration/real_models/sota/"
            f"test_detr_real_models.py` (sibling SOTA harness owns the "
            f"inference path)."
        )
    return cache_path.read_bytes()
