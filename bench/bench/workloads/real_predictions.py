"""Disk-only adapter for real-model predictions cells.

Predictions come from four sources, all populating the same cache:

- **Mask R-CNN R50-FPN (Detectron2 model zoo)** — downloaded from a
  pinned Hugging Face URL via ``tools/fetch-real-predictions.sh
  --maskrcnn``. The bench harness reads the cached JSON without any
  inference dependency.
- **rf-detr (Nano / SegNano)** — inferred locally by the TIDE
  validation harness (which depends on the heavy ``real-models`` extra:
  torch, rfdetr, supervision). The bench harness reads the same JSON
  the TIDE conftest writes; running ``pytest -m real_models`` is
  enough to populate the cache.
- **Hugging Face SOTA (DETR-R50)** — inferred locally by the SOTA
  validation harness (``tests/python/integration/real_models/sota/``,
  same ``real-models`` extra: torch, transformers, huggingface_hub).
  Same cache contract: bench reads the JSON the SOTA conftest writes.
- **Hugging Face SOTA (Mask2Former Swin-T)** — two cells from one
  architecture family: a panoptic cell on COCO val2017 (PNG dir +
  panoptic_dt.json sidecar) and a semantic cell on ADE20K val (PNG
  dir of train-id label maps). Same SOTA harness + cache contract.
- **Hugging Face SOTA (ViTPose-base-simple)** — top-down keypoints
  on COCO val2017 (GT person boxes as input). Same SOTA harness +
  cache contract; emits a COCO keypoints results JSON.

The harness env stays light: this module imports nothing more than
``real_predictions_cache`` for path resolution.
"""

from __future__ import annotations

from pathlib import Path

from real_predictions_cache import (
    DETR_RESNET50_REVISION,
    LVIS_DETECTOR_REVISION,
    MASK2FORMER_ADE_REVISION,
    MASK2FORMER_PANOPTIC_REVISION,
    RFDETR_VERSION,
    VITPOSE_REVISION,
    RfdetrModelName,
    detr_resnet50_cache_path,
    lvis_detector_cache_path,
    mask2former_ade_cache_dir,
    mask2former_panoptic_cache_dir,
    mask2former_panoptic_dt_json_path,
    maskrcnn_cache_path,
    rfdetr_cache_path,
    vitpose_cache_path,
)

# Workload identifiers — kept as constants so the registry, tests, and
# CLI error messages don't drift.
MASKRCNN_R50FPN_WORKLOAD_ID = "coco_val2017_maskrcnn_r50fpn_d2_v1"
RFDETR_NANO_WORKLOAD_ID = f"coco_val2017_rfdetr_nano_v{RFDETR_VERSION}"
RFDETR_SEGNANO_WORKLOAD_ID = f"coco_val2017_rfdetr_segnano_v{RFDETR_VERSION}"
DETR_R50_WORKLOAD_ID = f"coco_val2017_detr_r50_v{DETR_RESNET50_REVISION[:7]}"
MASK2FORMER_PANOPTIC_WORKLOAD_ID = (
    f"coco_panoptic_val2017_mask2former_swin_t_v{MASK2FORMER_PANOPTIC_REVISION[:7]}"
)
MASK2FORMER_ADE_WORKLOAD_ID = f"ade20k_val_mask2former_swin_t_v{MASK2FORMER_ADE_REVISION[:7]}"
VITPOSE_WORKLOAD_ID = f"coco_val2017_vitpose_base_simple_v{VITPOSE_REVISION[:7]}"
LVIS_DETECTOR_WORKLOAD_ID = f"lvis_v1_val_deformable_detr_v{LVIS_DETECTOR_REVISION[:7]}"


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


def mask2former_panoptic_dt_paths() -> tuple[Path, Path]:
    """Return ``(dt_png_dir, dt_json_path)`` for the Mask2Former panoptic
    predictions cache.

    Two-piece return because panoptic results are not a single file:
    the PNG dir holds rgb2id-encoded segment maps (one per image),
    and the JSON sidecar holds the matching ``segments_info``. Both
    pieces share the same revision-pinned cache directory.

    Read-only adapter pattern (same as :func:`maskrcnn_dt_path`): a
    missing cache surfaces the populator hint rather than silently
    producing a degenerate benchmark. Mask2Former predictions are
    inferred locally by the SOTA harness; the bench env doesn't carry
    transformers / torch / huggingface_hub.
    """
    cache_dir = mask2former_panoptic_cache_dir()
    dt_json = mask2former_panoptic_dt_json_path()
    if not (cache_dir.is_dir() and dt_json.is_file()):
        raise FileNotFoundError(
            f"Mask2Former panoptic prediction cache missing at {cache_dir} "
            f"(expected dir + {dt_json.name}). Populate via "
            f"`./tools/fetch-real-predictions.sh --mask2former-panoptic` or "
            f"`pytest -m real_models tests/python/integration/real_models/sota/"
            f"test_mask2former_panoptic_real_models.py` with the `real-models` "
            f"extra installed (~20-25h on 8-core CPU first run). "
            f"MASK2FORMER_PANOPTIC_REVISION must also be pinned in source — see "
            f"tools/real_predictions_cache/real_predictions_cache/__init__.py."
        )
    return cache_dir, dt_json


def mask2former_ade_dt_path() -> Path:
    """Return the Mask2Former ADE-semantic predictions cache directory.

    Single-path return (unlike :func:`mask2former_panoptic_dt_paths`):
    semantic predictions are just per-image label-map PNGs, no JSON
    sidecar. Same read-only adapter pattern; same actionable error
    on a missing cache.
    """
    cache_dir = mask2former_ade_cache_dir()
    if not cache_dir.is_dir() or not any(cache_dir.iterdir()):
        raise FileNotFoundError(
            f"Mask2Former ADE-semantic prediction cache missing or empty at "
            f"{cache_dir}. Populate via "
            f"`./tools/fetch-real-predictions.sh --mask2former-ade` or "
            f"`pytest -m real_models tests/python/integration/real_models/sota/"
            f"test_mask2former_ade_real_models.py` with the `real-models` extra "
            f"and the ADE20K val cache provisioned (~3-4h on 8-core CPU first "
            f"run). MASK2FORMER_ADE_REVISION must also be pinned in source."
        )
    return cache_dir


def vitpose_dt_path() -> Path:
    """Return the ViTPose-base-simple cached keypoint prediction JSON.

    Same read-only adapter shape as :func:`detr_r50_dt_path`: if the
    cache is missing, point the user at the populator (either the
    pytest fixture or the ``tools/fetch-real-predictions.sh
    --vitpose`` shim that shells into the same ``real-models``
    extra). Keeping the bench env free of transformers / torch /
    huggingface_hub deps means an unpopulated cache surfaces as a
    hard error instead of a silent zero-keypoint benchmark.
    """
    path = vitpose_cache_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"ViTPose-base-simple prediction cache missing at {path}. "
            f"Populate it by running `pytest -m real_models "
            f"tests/python/integration/real_models/sota/test_vitpose_real_models.py` "
            f"with the `real-models` extra installed (torch, transformers, "
            f"huggingface_hub, timm; ~2-3h on an 8-core CPU for the first "
            f"run, seconds on cache hit). Alternatively, run "
            f"`./tools/fetch-real-predictions.sh --vitpose` for the same flow."
        )
    return path


def lvis_detector_dt_path() -> Path:
    """Return the LVIS detector cached prediction JSON.

    Same read-only adapter shape as :func:`detr_r50_dt_path`. The
    model is ``facebook/deformable-detr-box-supervised``; predictions
    are written by the SOTA harness's ``_lvis_detector_predict``
    module against LVIS v1 val. If the cache is missing, point the
    user at the populator (either the pytest fixture or the
    ``./tools/fetch-real-predictions.sh --lvis`` shim that shells
    into the same ``real-models`` extra). Keeping the bench env free
    of transformers / torch / huggingface_hub deps means an
    unpopulated cache surfaces as a hard error instead of a silent
    zero-detection benchmark.
    """
    path = lvis_detector_cache_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"LVIS detector prediction cache missing at {path}. "
            f"Populate it by running `pytest -m real_models "
            f"tests/python/integration/real_models/sota/test_lvis_real_models.py` "
            f"with the `real-models` extra installed (torch, transformers, "
            f"huggingface_hub, timm; ~48-72h on an 8-core CPU for the "
            f"first run, seconds on cache hit). Alternatively, run "
            f"`./tools/fetch-real-predictions.sh --lvis` for the same flow. "
            f"The LVIS v1 val cache must also be provisioned "
            f"(`python -m lvis_v1_val_cache fetch`)."
        )
    return path


def detr_r50_dt_path() -> Path:
    """Return the DETR-R50 cached prediction JSON.

    Same read-only adapter shape as :func:`maskrcnn_dt_path` /
    :func:`rfdetr_dt_path`: if the cache is missing, point the user at
    the populator (either the pytest fixture or the
    ``tools/fetch-real-predictions.sh --detr`` shim that shells into
    the same ``real-models`` extra). Keeping the bench env free of
    transformers / torch / huggingface_hub deps means an unpopulated
    cache surfaces as a hard error instead of a silent zero-detection
    benchmark.
    """
    path = detr_resnet50_cache_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"DETR-R50 prediction cache missing at {path}. "
            f"Populate it by running `pytest -m real_models "
            f"tests/python/integration/real_models/sota/test_detr_real_models.py` "
            f"with the `real-models` extra installed (torch, transformers, "
            f"huggingface_hub, timm; ~12-15h on an 8-core CPU for the "
            f"first run, seconds on cache hit). Alternatively, run "
            f"`./tools/fetch-real-predictions.sh --detr` for the same flow."
        )
    return path
