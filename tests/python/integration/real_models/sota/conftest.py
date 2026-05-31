"""Fixtures for the Hugging Face SOTA validation harness.

Skips the entire module when the ``real-models`` extra is absent (it's
``transformers`` here, in contrast to ``tide/conftest.py`` which gates
on ``rfdetr``). When present, exposes session-scoped fixtures that
load cached predictions or run inference once per session, one per
SOTA cell:

- :func:`detr_predictions_path` — DETR-R50 COCO val2017 (bbox).
- :func:`mask2former_panoptic_cache_paths` — Mask2Former Swin-T
  COCO panoptic val2017 (PNG dir + DT JSON sidecar).
- :func:`mask2former_ade_cache_paths` — Mask2Former Swin-T ADE20K
  val (semantic).

Inference is the cost driver — DETR-R50 ~12-15h, Mask2Former
panoptic ~20-25h, Mask2Former ADE ~3-4h on an 8-core CPU first time.
Subsequent runs read from disk and skip the model entirely.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from real_predictions_cache import (
    detr_resnet50_cache_path,
    mask2former_ade_cache_dir,
    mask2former_panoptic_cache_dir,
    mask2former_panoptic_dt_json_path,
)

# Probe every dep the SOTA harness actually loads at fixture-create time,
# not just `transformers` — a partial install (transformers present, timm
# or PIL missing) would otherwise ImportError mid-fixture instead of the
# clean skip the docstring promises.
_REAL_MODELS_REASON = "SOTA harness needs the `real-models` extra: `uv sync --extra real-models`"
for _mod in ("transformers", "torch", "huggingface_hub", "timm", "PIL"):
    pytest.importorskip(_mod, reason=_REAL_MODELS_REASON)

# The Mask2Former-ADE parity test pulls in
# ``tests.python.parity_semantic.harness``, which top-level-imports
# ``mmseg.evaluation.metrics.iou_metric.IoUMetric``. Pytest only fires
# the parity_semantic/conftest.py stub loader when collecting tests
# under that tree, so we replicate the install here — sibling-tree
# import requires the same stubs in sys.modules. Idempotent.
_PARITY_SEMANTIC_ORACLE = (
    Path(__file__).resolve().parents[3] / "parity_semantic" / "oracle" / "mmsegmentation"
)
if str(_PARITY_SEMANTIC_ORACLE) not in sys.path:
    sys.path.insert(0, str(_PARITY_SEMANTIC_ORACLE))
from _loader import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    install_stubs as _install_mmseg_stubs,
)

_install_mmseg_stubs()

# Same sibling-tree issue for the panoptic side: the parity_panoptic
# conftest adds the vendored ``panopticapi`` checkout to ``sys.path``,
# but only fires when collecting tests under that tree. Replicate
# here so the SOTA panoptic parity test can ``from panopticapi.evaluation
# import pq_compute_single_core`` directly.
_PARITY_PANOPTIC_ORACLE = (
    Path(__file__).resolve().parents[3] / "parity_panoptic" / "oracle" / "panopticapi"
)
if str(_PARITY_PANOPTIC_ORACLE) not in sys.path:
    sys.path.insert(0, str(_PARITY_PANOPTIC_ORACLE))


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


@pytest.fixture(scope="session")
def coco_panoptic_gt() -> tuple[Path, Path, dict[str, Any]]:
    """``(gt_json_path, gt_png_dir, gt_dict)`` for COCO panoptic val2017.

    Pulls from the :mod:`panoptic_val_cache` per-user cache. The fixture
    skips cleanly when the panoptic cache isn't provisioned —
    ``python -m panoptic_val_cache`` is a one-time setup the user runs
    independently of the SOTA harness.
    """
    from panoptic_val_cache import ensure_gt

    try:
        gt_json, gt_png_dir = ensure_gt()
    except (FileNotFoundError, RuntimeError) as e:
        pytest.skip(f"COCO panoptic val2017 cache not provisioned: {e}")
    gt_dict = json.loads(gt_json.read_bytes())
    return gt_json, gt_png_dir, gt_dict


@pytest.fixture(scope="session")
def mask2former_panoptic_cache_paths(
    coco_panoptic_gt: tuple[Path, Path, dict[str, Any]],
    coco_val_image_dir: Path,
) -> tuple[Path, Path]:
    """``(cache_dir, dt_json_path)`` for Mask2Former panoptic predictions.

    Skips cleanly when ``MASK2FORMER_PANOPTIC_REVISION`` is still the
    ``_UNPINNED_REVISION`` sentinel — the populator's preflight catches
    that, but at fixture time we surface a single-line "pin then run"
    message rather than the populator's longer one. Both are correct;
    the fixture-time skip avoids the model-load cost on a
    not-yet-configured machine.
    """
    from real_predictions_cache import (
        MASK2FORMER_PANOPTIC_MODEL_ID,
        MASK2FORMER_PANOPTIC_REVISION,
        _ensure_pinned_revision,
    )

    try:
        _ensure_pinned_revision(MASK2FORMER_PANOPTIC_REVISION, MASK2FORMER_PANOPTIC_MODEL_ID)
    except RuntimeError as e:
        pytest.skip(str(e))

    _, _, gt_dict = coco_panoptic_gt
    cache_dir = mask2former_panoptic_cache_dir()
    dt_json = mask2former_panoptic_dt_json_path()
    if not dt_json.is_file():
        from ._mask2former_panoptic_predict import predict_coco_panoptic_val

        predict_coco_panoptic_val(
            gt=gt_dict,
            image_dir=coco_val_image_dir,
            cache_dir=cache_dir,
            dt_json_path=dt_json,
        )
    return cache_dir, dt_json


@pytest.fixture(scope="session")
def ade20k_val_gt_dir() -> Path:
    """Materialized ADE20K val GT directory (train-id 0..149 + 255).

    Skips cleanly when the ADE20K cache isn't provisioned. The
    `ade20k_val_cache` module handles its own download flow; the
    fixture just bridges to the per-image label-map PNGs.
    """
    from ade20k_val_cache import ensure_gt

    try:
        gt_dir, _, _ = ensure_gt()
    except (FileNotFoundError, RuntimeError) as e:
        pytest.skip(f"ADE20K val cache not provisioned: {e}")
    return gt_dir


@pytest.fixture(scope="session")
def mask2former_ade_cache_paths(ade20k_val_gt_dir: Path) -> Path:
    """Cache dir for Mask2Former ADE-semantic predictions.

    Same pin-check + run-or-load shape as
    :func:`mask2former_panoptic_cache_paths`. The fixture relies on
    :func:`ade20k_val_gt_dir` (instead of taking a raw image-dir arg)
    so the populator and the parity test see the same materialized
    GT.
    """
    from ade20k_val_cache import scan_image_jpgs
    from real_predictions_cache import (
        MASK2FORMER_ADE_MODEL_ID,
        MASK2FORMER_ADE_REVISION,
        _ensure_pinned_revision,
    )

    try:
        _ensure_pinned_revision(MASK2FORMER_ADE_REVISION, MASK2FORMER_ADE_MODEL_ID)
    except RuntimeError as e:
        pytest.skip(str(e))

    # `ade20k_val_gt_dir` provisioned the cache; the images dir is its
    # sibling. Resolve via the canonical helper (not by walking up the
    # gt_dir path) so the convention stays in one place.
    from ade20k_val_cache import ensure_gt

    _, images_dir, _ = ensure_gt()
    image_paths = scan_image_jpgs(images_dir)

    cache_dir = mask2former_ade_cache_dir()
    if not all((cache_dir / f"{iid}.png").is_file() for iid in image_paths):
        from ._mask2former_ade_predict import predict_ade20k_val

        predict_ade20k_val(image_paths=image_paths, cache_dir=cache_dir)
    return cache_dir
