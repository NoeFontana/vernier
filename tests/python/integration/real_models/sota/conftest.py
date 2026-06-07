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
- :func:`vitpose_predictions_path` — ViTPose-base-simple COCO val2017
  (keypoints, top-down on GT person boxes).

Inference is the cost driver — DETR-R50 ~12-15h, Mask2Former
panoptic ~20-25h, Mask2Former ADE ~3-4h, ViTPose-base ~2-3h on an
8-core CPU first time. Subsequent runs read from disk and skip the
model entirely.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# Pin BLAS / OpenMP thread counts BEFORE any test imports torch (either
# transitively via transformers in this tree or directly via rfdetr in the
# sibling ``tide/`` tree). ``torch.set_num_threads(1)`` is documented as a
# no-op once the intra-op pool is initialised, so the env-time pin is the
# only reliable way to keep the cache key ``(model, revision, dataset)``
# host-independent across test orderings. ``setdefault`` so a deliberate
# parent-env override still wins.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import pytest
from real_predictions_cache import (
    detr_resnet50_cache_path,
    mask2former_ade_cache_dir,
    mask2former_panoptic_cache_dir,
    mask2former_panoptic_dt_json_path,
    vitpose_cache_path,
)

# Probe every dep the SOTA harness actually loads at fixture-create time,
# not just `transformers` — a partial install (transformers present, timm
# or PIL missing) would otherwise ImportError mid-fixture instead of the
# clean skip the docstring promises.
_REAL_MODELS_REASON = "SOTA harness needs the `real-models` extra: `uv sync --extra real-models`"
for _mod in ("transformers", "torch", "huggingface_hub", "timm", "PIL"):
    pytest.importorskip(_mod, reason=_REAL_MODELS_REASON)

# Sibling-tree oracle installation, guarded so a vendored-oracle path
# rename can never break collection for SOTA tests that don't depend on
# the renamed oracle (e.g. a broken parity_semantic path should not take
# the DETR cell down). Failures fall through to a fixture-time
# ``pytest.skip`` in the relevant per-cell fixtures rather than an
# import-time collection error; each install records its outcome so the
# right cell gets the right skip reason.
_SOTA_CONFTEST_DIR = Path(__file__).parent
# sota/ → real_models/ → integration/ → python/  (3 parents).
# `.resolve()` deliberately omitted: under git-worktree / symlinked
# checkouts, canonicalising the path would walk into a different
# repo tree than the developer is editing.
_TESTS_PYTHON_DIR = _SOTA_CONFTEST_DIR.parent.parent.parent

#: Set by the mmseg stub install path; ``None`` on success, an error
#: string on failure. Read by per-cell fixtures that need the oracle.
_MMSEG_STUB_INSTALL_ERROR: str | None = None

#: Set by the panopticapi sys.path insertion path; same shape.
_PANOPTICAPI_PATH_INSTALL_ERROR: str | None = None


def _install_mmseg_stubs_guarded() -> None:
    """Replicate the parity_semantic conftest's stub install.

    Pytest only fires the parity_semantic/conftest.py stub loader when
    collecting tests under that tree, so we replicate the install here
    — the Mask2Former-ADE parity test pulls in
    ``tests.python.parity_semantic.harness``, which top-level-imports
    ``mmseg.evaluation.metrics.iou_metric.IoUMetric`` and needs the
    stubs already in ``sys.modules``. Idempotent at the
    ``install_stubs`` call.
    """
    global _MMSEG_STUB_INSTALL_ERROR
    oracle = _TESTS_PYTHON_DIR / "parity_semantic" / "oracle" / "mmsegmentation"
    if not oracle.is_dir():
        _MMSEG_STUB_INSTALL_ERROR = f"vendored mmseg oracle missing at {oracle}"
        return
    if str(oracle) not in sys.path:
        sys.path.insert(0, str(oracle))
    try:
        from _loader import (  # pyright: ignore[reportMissingImports]
            install_stubs as _install,
        )

        _install()
    except ImportError as e:
        _MMSEG_STUB_INSTALL_ERROR = f"mmseg stub loader unavailable: {e}"


def _install_panopticapi_path_guarded() -> None:
    """Replicate the parity_panoptic conftest's sys.path insertion.

    The parity_panoptic conftest adds the vendored ``panopticapi``
    checkout to ``sys.path`` but only fires when collecting under that
    tree. Replicate here so the SOTA panoptic parity test can
    ``from panopticapi.evaluation import pq_compute_single_core``
    directly. Unlike the mmseg side this is a real vendored package,
    not a stub install.
    """
    global _PANOPTICAPI_PATH_INSTALL_ERROR
    oracle = _TESTS_PYTHON_DIR / "parity_panoptic" / "oracle" / "panopticapi"
    if not oracle.is_dir():
        _PANOPTICAPI_PATH_INSTALL_ERROR = f"vendored panopticapi missing at {oracle}"
        return
    if str(oracle) not in sys.path:
        sys.path.insert(0, str(oracle))


_install_mmseg_stubs_guarded()
_install_panopticapi_path_guarded()


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
    if _PANOPTICAPI_PATH_INSTALL_ERROR is not None:
        pytest.skip(_PANOPTICAPI_PATH_INSTALL_ERROR)
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
    if _MMSEG_STUB_INSTALL_ERROR is not None:
        pytest.skip(_MMSEG_STUB_INSTALL_ERROR)
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


@pytest.fixture(scope="session")
def coco_kp_gt_path(coco_val_root: Path) -> Path:
    """Path to the COCO val2017 keypoints GT JSON.

    Lives alongside the instances GT in the same val2017 cache root —
    the ``coco_val_cache`` module fetches both from the same upstream
    ``annotations_trainval2017.zip``. Eagerly materialises via
    :func:`coco_val_cache.ensure_kp_gt` so a user that already has the
    instances GT + images cache provisioned doesn't need to learn a
    new fetch flag for this one extra file; skips cleanly on any
    download / SHA-pin error.
    """
    from coco_val_cache import KP_GT_FILENAME, ensure_kp_gt

    path = coco_val_root / KP_GT_FILENAME
    if not path.is_file():
        try:
            ensure_kp_gt(cache=coco_val_root)
        except (RuntimeError, OSError) as e:
            pytest.skip(f"COCO val2017 keypoints GT unavailable at {path}: {e}")
    return path


@pytest.fixture(scope="session")
def coco_kp_gt_dict(coco_kp_gt_path: Path) -> dict[str, Any]:
    return json.loads(coco_kp_gt_path.read_bytes())


@pytest.fixture(scope="session")
def vitpose_predictions_path(
    coco_kp_gt_dict: dict[str, Any],
    coco_val_image_dir: Path,
) -> Path:
    """Run-or-load ViTPose-base-simple predictions on COCO val2017.

    Top-down predictor: iterates over GT person annotations (not
    images) because ViTPose needs a person-box crop. Skips cleanly
    when ``VITPOSE_REVISION`` is the ``_UNPINNED_REVISION`` sentinel
    — same shape gate as the Mask2Former fixtures, surfaced at
    fixture time so we don't pay the model-load cost on a
    not-yet-configured machine.

    Returns the cache path (a JSON file in COCO keypoints
    ``loadRes`` format), not the bytes — tests pass this path
    straight into the parity harness, which reads it back through
    pycocotools' loader (``iouType="keypoints"``) and vernier's JSON
    ingest. Re-using the path keeps the two oracles' loaders
    independent.
    """
    from real_predictions_cache import (
        VITPOSE_MODEL_ID,
        VITPOSE_REVISION,
        _ensure_pinned_revision,
    )

    try:
        _ensure_pinned_revision(VITPOSE_REVISION, VITPOSE_MODEL_ID)
    except RuntimeError as e:
        pytest.skip(str(e))

    cache = vitpose_cache_path()
    if not cache.is_file():
        from ._vitpose_predict import predict_coco_val

        # Mirror the panoptic / ADE fixtures' skip shape: when the
        # COCO val2017 images dir hasn't been provisioned the
        # populator raises FileNotFoundError on the first missing
        # JPEG; convert that to a clean pytest.skip pointing at
        # VERNIER_COCO_CACHE so the suite skips rather than
        # ERRORing on under-provisioned hosts.
        try:
            predict_coco_val(
                gt=coco_kp_gt_dict,
                image_dir=coco_val_image_dir,
                cache_path=cache,
            )
        except FileNotFoundError as e:
            pytest.skip(
                f"COCO val2017 images dir not provisioned under "
                f"VERNIER_COCO_CACHE (looked under {coco_val_image_dir}): {e}"
            )
    return cache
