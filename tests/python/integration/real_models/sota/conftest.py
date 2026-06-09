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

import contextlib
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
    lvis_detector_cache_path,
    mask2former_ade_cache_dir,
    mask2former_panoptic_cache_dir,
    mask2former_panoptic_dt_json_path,
    rfdetr_cache_path,
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

#: Set by the lvis-api sys.path insertion path; same shape.
_LVIS_API_PATH_INSTALL_ERROR: str | None = None

#: Set by the boundary_iou_api sys.path insertion path; same shape.
#: Read by the rfdetr-segnano boundary cell's predictions fixture so a
#: vendored-oracle path rename surfaces as a per-cell skip instead of
#: breaking SOTA collection wholesale.
_BOUNDARY_IOU_API_PATH_INSTALL_ERROR: str | None = None


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


def _install_lvis_api_path_guarded() -> None:
    """Replicate the parity_lvis conftest's sys.path insertion.

    The parity_lvis conftest adds the vendored ``lvis-api`` checkout
    to ``sys.path`` but only fires when collecting under that tree.
    Replicate here so the SOTA LVIS parity test can
    ``from lvis import LVIS, LVISEval, LVISResults`` directly. Same
    shape as :func:`_install_panopticapi_path_guarded`.

    Also re-binds ``np.float`` if missing — the vendored ``lvis-api``
    0.5.3 release predates NumPy 1.20's removal of the alias. The
    parity_lvis conftest does the same; replicated here for symmetry
    so the SOTA test isn't a sibling collection-order accident.
    """
    global _LVIS_API_PATH_INSTALL_ERROR
    oracle = _TESTS_PYTHON_DIR / "parity_lvis" / "oracle" / "lvis_api"
    if not oracle.is_dir():
        _LVIS_API_PATH_INSTALL_ERROR = f"vendored lvis-api missing at {oracle}"
        return
    if str(oracle) not in sys.path:
        sys.path.insert(0, str(oracle))
    import numpy as _np

    if not hasattr(_np, "float"):
        _np.float = float  # type: ignore[attr-defined]


#: Sentinel used to remember "no prior ``np.float`` attribute" so the
#: bail-out path can distinguish "we set it" from "it was already
#: there". A bare module-level object beats ``None`` because numpy
#: ships symbols that legitimately resolve to ``None`` (unlikely for
#: ``np.float``, but the contract is "any value we didn't pick").
_SENTINEL: object = object()


def _install_boundary_iou_api_path_guarded() -> None:
    """Replicate the parity_boundary conftest's sys.path insertion + shims.

    The parity_boundary conftest adds the vendored ``boundary_iou_api``
    checkout to ``sys.path`` AND installs three runtime shims:

    1. ``matplotlib`` stubs — ``coco.py`` does a top-level ``import
       matplotlib`` for visualization helpers we never call; pulling
       matplotlib into our test deps just for that is heavyweight.
    2. Single-process override for the multi-core boundary augmenter —
       the multiprocessing pool surfaces ``ConnectionResetError`` under
       our test harness on Python 3.14+; the single-core helper is
       functionally identical.
    3. ``np.float`` alias restore — the vendored ``cocoeval.py`` does
       ``.astype(dtype=np.float)``, removed in NumPy 1.20+.

    All three only fire when the parity_boundary conftest collects.
    When pytest is invoked against only the SOTA tree (e.g.
    ``pytest tests/python/integration/real_models/sota/``), the
    parity_boundary conftest never runs, so we replicate the shim
    install here. Each shim is idempotent — if the parity_boundary
    conftest DOES fire later (mixed pytest invocation), the second
    install no-ops cleanly.

    Failures (missing vendored tree, ImportError on the boundary
    package, any other error materialising the stub classes) record
    an error string; the per-cell fixture skips on it rather than
    breaking SOTA collection. On a bail, the matplotlib stubs and
    ``np.float`` shim are uninstalled — leaving them in
    ``sys.modules`` / on the ``numpy`` module would pollute the rest
    of the pytest process and silently change semantics for unrelated
    tests that import matplotlib or read ``np.float``.
    """
    global _BOUNDARY_IOU_API_PATH_INSTALL_ERROR
    oracle = _TESTS_PYTHON_DIR / "parity_boundary" / "oracle" / "boundary_iou_api"
    if not oracle.is_dir():
        _BOUNDARY_IOU_API_PATH_INSTALL_ERROR = f"vendored boundary_iou_api missing at {oracle}"
        return
    if str(oracle) not in sys.path:
        sys.path.insert(0, str(oracle))

    import types

    import numpy as _np

    # Capture the pre-install state so a bail can put it back exactly.
    # ``_installed_mpl_modules`` is the list of matplotlib-prefixed
    # ``sys.modules`` keys we created (so we only pop the ones we own;
    # an in-process matplotlib install we left alone stays alone).
    # ``_original_np_float`` is the previous ``np.float`` value (or
    # :data:`_SENTINEL` for "attribute was absent"), captured before
    # the shim install.
    _installed_mpl_modules: list[str] = []
    _original_np_float: object = getattr(_np, "float", _SENTINEL)

    def _undo_partial_install() -> None:
        """Pop the matplotlib stubs we installed and restore ``np.float``.

        Idempotent — called on the error paths after each
        :data:`_BOUNDARY_IOU_API_PATH_INSTALL_ERROR` set. Walk the
        list of keys WE inserted (not a blanket ``"matplotlib*"``
        prefix sweep) so a parallel install path that legitimately
        loaded matplotlib for some other reason isn't kneecapped.
        """
        for _key in _installed_mpl_modules:
            sys.modules.pop(_key, None)
        _installed_mpl_modules.clear()
        if _original_np_float is _SENTINEL:
            # We added ``np.float`` ourselves (or it didn't exist
            # pre-install); strip it. Numpy may bind some attributes
            # through a descriptor that refuses ``delattr``; suppress
            # rather than propagate — the calling code already wrote
            # the error string and the next collection cycle will
            # re-attempt the install cleanly.
            if hasattr(_np, "float"):
                with contextlib.suppress(AttributeError):
                    delattr(_np, "float")
        else:
            _np.float = _original_np_float  # type: ignore[attr-defined]

    # 1. matplotlib stubs (idempotent on ``"matplotlib" in sys.modules``).
    #    Catch ``(ImportError, AttributeError, TypeError)`` rather than
    #    just ``ImportError``: the ``type(_member, (), {})``-fabricated
    #    stub classes can raise non-Import errors at instantiation /
    #    attribute time, and any escape past this block leaks the
    #    half-installed stubs into the rest of the pytest process.
    try:
        if "matplotlib" not in sys.modules:
            for _name in ("matplotlib", "matplotlib.pyplot"):
                sys.modules[_name] = types.ModuleType(_name)
                _installed_mpl_modules.append(_name)
            for _sub, _member in (("collections", "PatchCollection"), ("patches", "Polygon")):
                _mod = types.ModuleType(f"matplotlib.{_sub}")
                setattr(_mod, _member, type(_member, (), {}))
                sys.modules[f"matplotlib.{_sub}"] = _mod
                _installed_mpl_modules.append(f"matplotlib.{_sub}")
    except (ImportError, AttributeError, TypeError) as e:
        _BOUNDARY_IOU_API_PATH_INSTALL_ERROR = (
            f"matplotlib stub install failed: {type(e).__name__}: {e}"
        )
        _undo_partial_install()
        return

    # 3 (installed BEFORE the boundary import below — ``cocoeval.py``
    #    is reached transitively from ``boundary_iou.utils.boundary_utils``
    #    on some import orderings and references ``np.float`` at
    #    module body parse time; installing it after the import is
    #    racy on a fresh interpreter).
    if not hasattr(_np, "float"):
        _np.float = float  # type: ignore[attr-defined]

    # 2. Single-process override on the multi-core augmenter. Touch the
    #    public symbol first so ``boundary_iou.utils.boundary_utils`` is
    #    in ``sys.modules``; then rebind the multi-core entry point to
    #    the single-core helper. If the import itself fails (vendored
    #    tree present but broken), record, undo the stub installs, and
    #    bail — the per-cell fixture will skip on the error.
    try:
        __import__("boundary_iou.utils.boundary_utils")
    except (ImportError, AttributeError, TypeError) as e:
        _BOUNDARY_IOU_API_PATH_INSTALL_ERROR = (
            f"boundary_iou import failed: {type(e).__name__}: {e}"
        )
        _undo_partial_install()
        return
    _utils_pkg = sys.modules["boundary_iou.utils"]
    _boundary_utils = sys.modules["boundary_iou.utils.boundary_utils"]

    def _single(
        annotations: list[Any],
        ann_to_mask: Any,
        dilation_ratio: float = 0.02,
    ) -> list[Any]:
        return _boundary_utils.augment_annotations_with_boundary_single_core(
            0, annotations, ann_to_mask, dilation_ratio
        )

    setattr(_boundary_utils, "augment_annotations_with_boundary_multi_core", _single)
    setattr(_utils_pkg, "augment_annotations_with_boundary_multi_core", _single)


_install_lvis_api_path_guarded()
_install_boundary_iou_api_path_guarded()


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
def rfdetr_segnano_predictions_path() -> Path:
    """Path to the rfdetr-segnano COCO val2017 predictions cache.

    READ-ONLY adapter (unlike :func:`detr_predictions_path`): this cell
    reuses the cache the TIDE harness populates — rfdetr ships as a
    pip package pinned by ``RFDETR_VERSION`` in source rather than a
    Hugging Face hub revision, and the existing TIDE inference run
    already produces RLE masks that the boundary kernel consumes
    directly. Re-running inference here would be a duplication of the
    same bytes-on-disk; the cache contract guarantees the same
    ``(model, version, dataset)`` tuple yields the same JSON.

    Skips cleanly when the rfdetr-segnano cache is not populated. The
    user provisions it by running the TIDE cell once
    (``pytest -m real_models tests/python/integration/real_models/tide/``)
    or by shelling into the bench-side populator
    (``./tools/fetch-real-predictions.sh --rfdetr segnano``).

    Also skips when the boundary_iou_api vendored oracle install
    failed — the boundary parity claim has no oracle without it.
    """
    if _BOUNDARY_IOU_API_PATH_INSTALL_ERROR is not None:
        pytest.skip(_BOUNDARY_IOU_API_PATH_INSTALL_ERROR)
    path = rfdetr_cache_path("segnano")
    if not path.is_file():
        pytest.skip(
            f"rfdetr-segnano predictions cache missing at {path}. Populate "
            f"by running the TIDE cell once: `pytest -m real_models "
            f"tests/python/integration/real_models/tide/` with the "
            f"`real-models` extra installed."
        )
    return path


@pytest.fixture(scope="session")
def lvis_v1_val_paths() -> tuple[Path, Path]:
    """``(gt_json_path, images_dir)`` for the LVIS v1 val cache.

    Pulls from the :mod:`lvis_v1_val_cache` provisioner. Skips
    cleanly when the cache isn't populated — ``python -m
    lvis_v1_val_cache fetch`` is a one-time setup the user runs
    independently of the SOTA harness (the download is ~190 MB GT
    zip + 778 MB images).
    """
    from lvis_v1_val_cache import GT_FILENAME, VAL_IMG_DIRNAME
    from lvis_v1_val_cache import cache_root as _lvis_cache_root

    root = _lvis_cache_root()
    gt = root / GT_FILENAME
    images = root / VAL_IMG_DIRNAME
    if not gt.is_file() or not images.exists():
        pytest.skip(
            f"LVIS v1 val cache not provisioned at {root}: "
            f"need {GT_FILENAME} and {VAL_IMG_DIRNAME}/. Run "
            f"`python -m lvis_v1_val_cache fetch` to populate."
        )
    return gt, images


@pytest.fixture(scope="session")
def lvis_detector_predictions_path(
    lvis_v1_val_paths: tuple[Path, Path],
) -> Path:
    """Run-or-load LVIS detector predictions on LVIS v1 val.

    Returns the cache path (an LVIS results JSON), not the bytes —
    same convention as :func:`detr_predictions_path`. The two
    oracles (``lvis-api``'s ``LVISResults`` constructor and
    vernier's federated grid) each read it back through their own
    loader.

    Skips cleanly when the LVIS detector revision is the
    :data:`_UNPINNED_REVISION` sentinel (a guard for the moment
    between scaffolding and SHA pinning) — same shape as the
    Mask2Former fixtures.
    """
    if _LVIS_API_PATH_INSTALL_ERROR is not None:
        pytest.skip(_LVIS_API_PATH_INSTALL_ERROR)
    from real_predictions_cache import (
        LVIS_DETECTOR_MODEL_ID,
        LVIS_DETECTOR_REVISION,
        _ensure_pinned_revision,
    )

    try:
        _ensure_pinned_revision(LVIS_DETECTOR_REVISION, LVIS_DETECTOR_MODEL_ID)
    except RuntimeError as e:
        pytest.skip(str(e))

    gt_path, images_dir = lvis_v1_val_paths
    cache = lvis_detector_cache_path()
    if not cache.is_file():
        from ._lvis_detector_predict import predict_lvis_val

        gt_dict = json.loads(gt_path.read_bytes())
        predict_lvis_val(
            gt=gt_dict,
            image_dir=images_dir,
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
