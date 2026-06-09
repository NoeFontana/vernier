"""Single source of truth for the real-model predictions cache.

Both Mask R-CNN (downloaded) and rf-detr (locally inferred via the TIDE
harness) land under the same per-user cache root so the bench adapter
can read either with one path-resolver. This module owns:

- :func:`cache_root` — XDG-correct cache directory.
- :func:`maskrcnn_cache_path` / :func:`rfdetr_cache_path` — stable
  filenames keyed on ``(model, version, dataset)``.
- :func:`ensure_maskrcnn` — atomic download + SHA256-verify.

The rf-detr inference path is owned by the TIDE harness (it depends on
the heavy ``real-models`` extra: torch, rfdetr, supervision). This
package just exposes the path it should write to so the bench adapter
and TIDE agree.

CLI entry point::

    python -m real_predictions_cache --maskrcnn
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import platformdirs
from coco_val_cache import _atomic_download, file_sha256

# Climb out of the ``real_predictions_cache/real_predictions_cache/``
# package nesting to reach repo root. Used by :func:`populate_rfdetr`
# to invoke the TIDE populator from the right cwd.
REPO_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# Mask R-CNN R50-FPN — Detectron2 model zoo (3x schedule, ``model_final_a3ec72``)
# ---------------------------------------------------------------------------

#: Identifier for our hosted prediction blob. Independent of the
#: Detectron2 model version: bumping this is a v1.0-level decision per
#: ``docs/engineering/benchmarking/`` snapshots, since the perf cells
#: are keyed on it.
MASKRCNN_BLOB_VERSION = "v1"

#: Download URL for the prediction blob. ``None`` until the upload to
#: ``NoeFontana/vernier-bench-predictions`` lands; :func:`ensure_maskrcnn`
#: errors loudly with an actionable message in the meantime so callers
#: don't silently succeed against an empty cache.
MASKRCNN_URL: str | None = None

#: SHA256 of the prediction-blob JSON. ``None`` paired with
#: :data:`MASKRCNN_URL`; fill both atomically when the upload lands.
MASKRCNN_SHA256: str | None = None

# ---------------------------------------------------------------------------
# rf-detr — pin matches the ``real-models`` extra in the root pyproject
# ---------------------------------------------------------------------------

RFDETR_VERSION = "1.6.5.post0"

#: rf-detr model variants the cache contract recognises. Mirrors the
#: TIDE harness's ``_rfdetr_predict.ModelName``; bench's
#: ``real_predictions.rfdetr_dt_path`` and :func:`populate_rfdetr` accept
#: only these.
RfdetrModelName = Literal["nano", "segnano"]

# ---------------------------------------------------------------------------
# Hugging Face SOTA harness — DETR-R50 (object detection)
# ---------------------------------------------------------------------------

#: Pinned ``facebook/detr-resnet-50`` commit on the Hugging Face hub.
#: The inference module loads weights with ``revision=DETR_RESNET50_REVISION``
#: so a weights bump on the hub can't silently shift cached predictions.
#: Bumping is an ADR-level decision (matches the rfdetr / Mask R-CNN
#: pin policy): perf snapshots and the bench's parity-on-real-data
#: claims are keyed on it.
DETR_RESNET50_REVISION = "1d5f47bd3bdd2c4bbfa585418ffe6da5028b4c0b"

# ---------------------------------------------------------------------------
# Hugging Face SOTA harness — Mask2Former Swin-Tiny COCO panoptic
# ---------------------------------------------------------------------------

#: Sentinel for an unpinned upstream revision. Callers MUST reject this
#: value at populate time — proceeding against ``"main"`` or any
#: mutable ref would defeat the cache-key invariant established by the
#: DETR-R50 follow-up (PR #265): the filename IS the integrity surface,
#: so any non-40-hex string means "do not populate, ask the user to pin".
_UNPINNED_REVISION = "TODO_PIN_VIA_HF_MODEL_INFO_SHA"


def _ensure_pinned_revision(revision: str, model_id: str) -> None:
    """Raise unless ``revision`` is a full 40-hex SHA.

    Discipline derived from PR #265: the cache filename embeds the
    revision and IS the data-integrity surface, so any non-SHA value
    (``"main"``, ``"latest"``, the :data:`_UNPINNED_REVISION` sentinel)
    must fail loudly rather than silently produce a cache file that
    re-resolves to a different upstream commit on a later populate.

    The error points at the one-liner the user runs to obtain the
    current SHA:

        python -c "from huggingface_hub import HfApi; print(HfApi().model_info('<model>').sha)"
    """
    if len(revision) == 40 and all(c in "0123456789abcdef" for c in revision):
        return
    one_liner = (
        f'python -c "from huggingface_hub import HfApi; '
        f"print(HfApi().model_info('{model_id}').sha)\""
    )
    raise RuntimeError(
        f"Hugging Face revision for {model_id!r} is unpinned (got {revision!r}). "
        f"Pin the constant in tools/real_predictions_cache/real_predictions_cache/__init__.py "
        f"to the current 40-hex commit SHA. Obtain it with: {one_liner}"
    )


MASK2FORMER_PANOPTIC_MODEL_ID = "facebook/mask2former-swin-tiny-coco-panoptic"
#: Pinned commit on the Hugging Face hub (resolved 2026-05-31). Same
#: bump-is-ADR-level policy as :data:`DETR_RESNET50_REVISION`. The
#: :data:`_UNPINNED_REVISION` sentinel is reserved for the moment
#: between cache-contract scaffolding and SHA pinning; once pinned it
#: never goes back to the sentinel without invalidating cached blobs.
MASK2FORMER_PANOPTIC_REVISION: str = "df6b1142ff50c3276559d9d78f35f6a579c75a77"

# ---------------------------------------------------------------------------
# Hugging Face SOTA harness — Mask2Former Swin-Tiny ADE20K semantic
# ---------------------------------------------------------------------------

MASK2FORMER_ADE_MODEL_ID = "facebook/mask2former-swin-tiny-ade-semantic"
#: Pinned commit on the Hugging Face hub (resolved 2026-05-31). Same
#: bump-is-ADR-level policy as :data:`MASK2FORMER_PANOPTIC_REVISION`.
MASK2FORMER_ADE_REVISION: str = "c8cf1b5e823aee214d937d0d001c1850ba44ef6a"

# ---------------------------------------------------------------------------
# Hugging Face SOTA harness — ViTPose-base-simple (keypoints, top-down)
# ---------------------------------------------------------------------------

VITPOSE_MODEL_ID = "usyd-community/vitpose-base-simple"
#: Pinned commit on the Hugging Face hub (resolved 2026-06-07). ViTPose
#: is a top-down pose estimator: it consumes person boxes (GT person
#: crops on this cell) and produces per-instance 17-keypoint vectors.
#: Same bump-is-ADR-level policy as the DETR / Mask2Former pins; the
#: cache filename embeds the full SHA so a weights bump on the hub
#: invalidates by construction.
VITPOSE_REVISION: str = "a93ac0c67e0b7e2c55287d21d4c460c8f3c54d45"

# ---------------------------------------------------------------------------
# Hugging Face SOTA harness — Deformable DETR R50 4x LVIS (box-supervised)
# ---------------------------------------------------------------------------

#: ``facebook/deformable-detr-box-supervised`` is the Apache-2.0,
#: transformers-compatible LVIS v1 detector closing the real-prediction
#: parity loop on the federated-evaluation paradigm (ADR-0026). The
#: checkpoint is "Box-Supervised_DeformDETR_R50_4x" from the original
#: Detic release, re-hosted on the Hugging Face hub with a transformers
#: ``DeformableDetrForObjectDetection`` config (300 queries, 1203
#: id2label entries covering the full LVIS v1 category set). Reports
#: 31.7 box mAP / 21.4 AP_r on LVIS v1 val.
#:
#: Discovery process (documented in ``docs/engineering/real-predictions-parity.md``):
#: queried for `lvis detr`, `deformable-detr lvis`, `co-detr lvis`,
#: `mm-grounding-dino lvis`, `glip lvis`, `owl-vit lvis`,
#: `lvis mask r-cnn huggingface`. Only ``facebook/deformable-detr-box-supervised``
#: returned an HF hub-hosted, transformers-loadable, LVIS v1-trained
#: detector whose id2label covers all 1203 categories. Downloads are
#: ~16/month (low — research checkpoint, not production), but the
#: alternative is the Detectron2 Mask R-CNN R50-FPN-LVIS fallback,
#: which would drag in a separate inference stack (Detectron2) just
#: for parity validation.
LVIS_DETECTOR_MODEL_ID = "facebook/deformable-detr-box-supervised"

#: Pinned commit on the Hugging Face hub (resolved 2026-06-07). Same
#: bump-is-ADR-level policy as :data:`DETR_RESNET50_REVISION`: the
#: cache filename embeds this SHA so a weights bump invalidates by
#: construction. Bumping is paired with re-running the SOTA parity
#: smoke and refreshing the headline numbers in
#: ``docs/engineering/real-predictions-parity.md``.
LVIS_DETECTOR_REVISION: str = "d7710a91d4e58c2fccd29f53e0ca350093b934f3"

_DATASET_ID_LVIS = "lvis-v1-val"

# ---------------------------------------------------------------------------
# Shared cache plumbing
# ---------------------------------------------------------------------------

CACHE_ENV = "VERNIER_REAL_PREDICTIONS_CACHE"
_DATASET_ID = "coco-val2017"
_DATASET_ID_ADE = "ade20k-val"


def cache_root(override: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the per-user real-models cache directory.

    Precedence: explicit ``override`` arg, then
    ``$VERNIER_REAL_PREDICTIONS_CACHE``, then
    ``platformdirs.user_cache_dir("vernier") / "real-models"``. The
    directory is *not* created here — the ``ensure_*`` helpers do.

    Convention is shared with
    ``tests/python/integration/real_models/tide/_rfdetr_predict.py`` so
    a single TIDE inference run populates the same cache the bench
    harness reads from.
    """
    if override is not None:
        return Path(override).expanduser()
    env = os.environ.get(CACHE_ENV)
    if env:
        return Path(env).expanduser()
    return Path(platformdirs.user_cache_dir("vernier")) / "real-models"


def maskrcnn_cache_filename() -> str:
    """Stable filename for the Mask R-CNN R50-FPN prediction blob."""
    return f"maskrcnn-r50fpn-d2-{MASKRCNN_BLOB_VERSION}-{_DATASET_ID}.json"


def maskrcnn_cache_path(*, cache: Path | None = None) -> Path:
    """Return the canonical cache location, with or without download."""
    return cache_root(cache) / maskrcnn_cache_filename()


def rfdetr_cache_filename(model_name: RfdetrModelName, *, version: str = RFDETR_VERSION) -> str:
    """Stable filename for cached rf-detr predictions.

    Mirrors the TIDE harness convention exactly (see
    ``tide/_rfdetr_predict.py::cache_filename``).
    """
    return f"rfdetr-{model_name}-{version}-{_DATASET_ID}.json"


def rfdetr_cache_path(
    model_name: RfdetrModelName,
    *,
    version: str = RFDETR_VERSION,
    cache: Path | None = None,
) -> Path:
    return cache_root(cache) / rfdetr_cache_filename(model_name, version=version)


def detr_resnet50_cache_filename(*, revision: str = DETR_RESNET50_REVISION) -> str:
    """Stable filename for ``facebook/detr-resnet-50`` predictions on
    COCO val2017.

    Embeds the FULL commit SHA so the filename IS the cache key — two
    future revisions sharing the first 7 hex chars can't collide on
    disk and silently serve stale predictions under a bumped pin. The
    user-facing workload ID (``coco_val2017_detr_r50_v<short>``) keeps
    the abbreviation for readability; that string is a label, not a
    data-integrity surface.
    """
    return f"detr-r50-{revision}-{_DATASET_ID}.json"


def detr_resnet50_cache_path(
    *,
    revision: str = DETR_RESNET50_REVISION,
    cache: Path | None = None,
) -> Path:
    """Return the canonical cache location for DETR-R50 predictions.

    Symmetric with :func:`rfdetr_cache_path`: the resolver doesn't
    populate the cache. :func:`populate_detr_resnet50` shells into the
    ``real-models`` extra to do the inference; the bench-side adapter
    (``bench/bench/workloads/real_predictions.py``) reads the resulting
    JSON without any inference dep of its own.
    """
    return cache_root(cache) / detr_resnet50_cache_filename(revision=revision)


def lvis_detector_cache_filename(*, revision: str = LVIS_DETECTOR_REVISION) -> str:
    """Stable filename for the LVIS detector's predictions on LVIS v1 val.

    The output shape is the COCO-detection results JSON shape (a list
    of ``{image_id, category_id, bbox: [x, y, w, h], score}`` records)
    so ``lvis-api``'s ``LVISResults`` constructor + vernier's federated
    bbox grid both ingest the same file. The dataset suffix
    (``lvis-v1-val``) disambiguates from the DETR-R50 cell, which
    shares the COCO results format but covers a different dataset
    (COCO val2017, 80 categories) and a different model.

    Embeds the FULL 40-hex SHA so future revisions sharing the first 7
    chars can't silently collide on disk under a bumped pin — same
    discipline as :func:`detr_resnet50_cache_filename`.
    """
    return f"deformable-detr-lvis-{revision}-{_DATASET_ID_LVIS}.json"


def lvis_detector_cache_path(
    *,
    revision: str = LVIS_DETECTOR_REVISION,
    cache: Path | None = None,
) -> Path:
    """Return the canonical cache location for LVIS detector predictions.

    Symmetric with :func:`detr_resnet50_cache_path`: read-only
    resolver; the bench-side adapter reads the JSON without any
    inference dep of its own. :func:`populate_lvis_detector` shells
    into the ``real-models`` extra to do the inference.
    """
    return cache_root(cache) / lvis_detector_cache_filename(revision=revision)


def mask2former_panoptic_cache_dirname(*, revision: str = MASK2FORMER_PANOPTIC_REVISION) -> str:
    """Stable directory name for Mask2Former panoptic predictions on
    COCO val2017.

    A panoptic cache is a directory (one RGB-encoded PNG per image +
    a single ``segments_info.json`` sidecar), not a single file —
    unlike DETR / rf-detr which fit in one COCO-detection-format JSON.
    The DETR-R50 lesson applies: embed the FULL 40-hex SHA so future
    revisions sharing the first 7 chars can't silently collide on
    disk under a bumped pin.
    """
    return f"mask2former-pan-swin-t-{revision}-{_DATASET_ID}"


def mask2former_panoptic_cache_dir(
    *,
    revision: str = MASK2FORMER_PANOPTIC_REVISION,
    cache: Path | None = None,
) -> Path:
    """Return the canonical cache *directory* for Mask2Former panoptic
    predictions. The directory contains per-image RGB PNGs (rgb2id
    encoded segment ids) and a top-level ``segments_info.json``."""
    return cache_root(cache) / mask2former_panoptic_cache_dirname(revision=revision)


def mask2former_panoptic_dt_json_path(
    *,
    revision: str = MASK2FORMER_PANOPTIC_REVISION,
    cache: Path | None = None,
) -> Path:
    """Path to the panoptic-DT JSON sidecar inside the cache dir.

    Shape: ``{"annotations": [{"image_id": int, "file_name": str,
    "segments_info": [{"id": int, "category_id": int, "area": int,
    ...}]}, ...]}``. Mirrors the COCO panoptic results format so
    ``panopticapi.evaluation.pq_compute_single_core`` and
    ``vernier.panoptic.Predictions.from_arrays`` can both consume the
    DT side without a second projection. ``category_id`` is the GT
    JSON's sparse COCO id (1..200 with gaps), not the model's
    contiguous 0..132 train-id space — see the inference module's
    class-mapping discussion.
    """
    return mask2former_panoptic_cache_dir(revision=revision, cache=cache) / "panoptic_dt.json"


def mask2former_ade_cache_dirname(*, revision: str = MASK2FORMER_ADE_REVISION) -> str:
    """Stable directory name for Mask2Former ADE-semantic predictions
    on ADE20K val (SceneParse150).

    A semantic cache is a directory of single-channel label-map PNGs
    (one per validation image, named ``<image_id>.png``). Same
    full-SHA convention as the panoptic variant — the filename IS the
    cache key.
    """
    return f"mask2former-ade-swin-t-{revision}-{_DATASET_ID_ADE}"


def mask2former_ade_cache_dir(
    *,
    revision: str = MASK2FORMER_ADE_REVISION,
    cache: Path | None = None,
) -> Path:
    """Return the canonical cache *directory* for Mask2Former ADE
    semantic predictions. The directory contains per-image label-map
    PNGs (uint8, train-id 0..149 + 255-ignore, mmseg convention)."""
    return cache_root(cache) / mask2former_ade_cache_dirname(revision=revision)


def vitpose_cache_filename(*, revision: str = VITPOSE_REVISION) -> str:
    """Stable filename for ViTPose-base-simple keypoint predictions on
    COCO val2017.

    Same full-40-hex-SHA convention as :func:`detr_resnet50_cache_filename`:
    the filename IS the cache key, so two future revisions sharing the
    first 7 hex chars can't collide on disk and silently serve stale
    keypoint vectors under a bumped pin. The cell is top-down (uses GT
    person bboxes as input crops), so the cache content depends on the
    GT JSON's person-box layout as well as the weights — both are
    pinned (GT via :data:`KP_GT_SHA256`, weights via this SHA).
    """
    return f"vitpose-base-simple-{revision}-{_DATASET_ID}.json"


def vitpose_cache_path(
    *,
    revision: str = VITPOSE_REVISION,
    cache: Path | None = None,
) -> Path:
    """Return the canonical cache location for ViTPose-base-simple
    keypoint predictions. The bench-side adapter
    (``bench/bench/workloads/real_predictions.py``) reads the JSON
    without any inference dep; the populator lives behind the
    ``[real-models]`` extra.
    """
    return cache_root(cache) / vitpose_cache_filename(revision=revision)


def ensure_maskrcnn(
    *,
    cache: Path | None = None,
    url: str | None = None,
    sha256: str | None = None,
) -> Path:
    """Return a verified path to the Mask R-CNN prediction blob, downloading if necessary.

    Idempotent: a cached file matching ``sha256`` short-circuits without
    network I/O. ``url`` and ``sha256`` default to module-level
    :data:`MASKRCNN_URL` / :data:`MASKRCNN_SHA256` — both are ``None``
    until the prediction blob is uploaded, which raises a clear
    ``RuntimeError`` rather than silently succeeding against an empty
    cache. Pass them explicitly for ad-hoc fetches.

    Raises ``RuntimeError`` on a post-download SHA mismatch; the
    caller should re-run, and if the mismatch persists open an issue.
    """
    final_url = url if url is not None else MASKRCNN_URL
    final_sha = sha256 if sha256 is not None else MASKRCNN_SHA256
    if final_url is None or final_sha is None:
        raise RuntimeError(
            "Mask R-CNN prediction blob URL/SHA256 not yet configured. "
            "Set MASKRCNN_URL and MASKRCNN_SHA256 in "
            "tools/real_predictions_cache/real_predictions_cache/__init__.py "
            "once the JSON is hosted on Hugging Face, or pass url=/sha256= "
            "explicitly to ensure_maskrcnn() for ad-hoc testing."
        )

    cache_dir = cache_root(cache)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / maskrcnn_cache_filename()

    if out.is_file() and file_sha256(out) == final_sha:
        return out

    out.unlink(missing_ok=True)
    _atomic_download(final_url, out)
    actual = file_sha256(out)
    if actual != final_sha:
        out.unlink(missing_ok=True)
        raise RuntimeError(
            f"Mask R-CNN prediction blob SHA256 mismatch: expected "
            f"{final_sha}, got {actual}. Either the upstream artifact "
            f"changed or the download was corrupted; re-run, and if "
            f"the mismatch persists open an issue."
        )
    return out


_RFDETR_POPULATOR_MODULE = "tests.python.integration.real_models.tide._populate_cache"
_DETR_R50_POPULATOR_MODULE = "tests.python.integration.real_models.sota._populate_cache"


def populate_rfdetr(model_name: RfdetrModelName) -> None:
    """Run rf-detr inference to populate the prediction cache.

    Shells into ``uv run --extra real-models python -m
    tests.python.integration.real_models.tide._populate_cache --model
    <model_name>`` so the heavy ``[real-models]`` extra (torch, rfdetr,
    supervision; ~5 GB on first install) lives outside this package's
    dep set. The TIDE module owns the inference path; this function is
    just the orchestrator.

    First run on a clean machine takes ~30 minutes per model on CPU; a
    cache hit is seconds. Cached output lands at
    :func:`rfdetr_cache_path`, which the bench adapter reads.
    """
    if model_name not in {"nano", "segnano"}:
        raise ValueError(f"unknown rf-detr model {model_name!r}; expected 'nano' or 'segnano'")
    cmd = [
        "uv",
        "run",
        "--extra",
        "real-models",
        "python",
        "-m",
        _RFDETR_POPULATOR_MODULE,
        "--model",
        model_name,
    ]
    print(
        f"Shelling into [real-models] extra for rf-detr {model_name} inference: {' '.join(cmd)}",
        file=sys.stderr,
    )
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


_MASK2FORMER_PANOPTIC_POPULATOR_FLAG = "--mask2former-panoptic"
_MASK2FORMER_ADE_POPULATOR_FLAG = "--mask2former-ade"
_VITPOSE_POPULATOR_FLAG = "--vitpose"


def populate_mask2former_panoptic() -> None:
    """Run Mask2Former Swin-Tiny inference on COCO val2017 to populate
    the panoptic prediction cache.

    Shells into ``uv run --extra real-models python -m
    tests.python.integration.real_models.sota._populate_cache
    --mask2former-panoptic`` — same shape as :func:`populate_detr_resnet50`.
    Validates the upstream revision is pinned before spawning the
    subprocess; an unpinned :data:`MASK2FORMER_PANOPTIC_REVISION`
    fails loudly here rather than later inside the inference process.

    First run on a clean machine takes ~20-25 hours on an 8-core CPU
    (Mask2Former Swin-T is heavier per image than DETR-R50). A cache
    hit is seconds. Cached output lands at
    :func:`mask2former_panoptic_cache_dir`.
    """
    _ensure_pinned_revision(MASK2FORMER_PANOPTIC_REVISION, MASK2FORMER_PANOPTIC_MODEL_ID)
    cmd = [
        "uv",
        "run",
        "--extra",
        "real-models",
        "python",
        "-m",
        _DETR_R50_POPULATOR_MODULE,
        _MASK2FORMER_PANOPTIC_POPULATOR_FLAG,
    ]
    print(
        f"Shelling into [real-models] extra for Mask2Former panoptic inference: {' '.join(cmd)}",
        file=sys.stderr,
    )
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def populate_mask2former_ade() -> None:
    """Run Mask2Former Swin-Tiny ADE-semantic inference on ADE20K val
    to populate the semantic prediction cache.

    Same shape as :func:`populate_mask2former_panoptic`. First run
    takes ~3-4 hours on an 8-core CPU (ADE20K val is 2000 images vs
    COCO's 5000, and the ADE checkpoint is the same Swin-T backbone).
    A cache hit is seconds. Cached output lands at
    :func:`mask2former_ade_cache_dir`.
    """
    _ensure_pinned_revision(MASK2FORMER_ADE_REVISION, MASK2FORMER_ADE_MODEL_ID)
    cmd = [
        "uv",
        "run",
        "--extra",
        "real-models",
        "python",
        "-m",
        _DETR_R50_POPULATOR_MODULE,
        _MASK2FORMER_ADE_POPULATOR_FLAG,
    ]
    print(
        f"Shelling into [real-models] extra for Mask2Former ADE inference: {' '.join(cmd)}",
        file=sys.stderr,
    )
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def populate_vitpose() -> None:
    """Run ViTPose-base-simple inference on COCO val2017 to populate
    the keypoint prediction cache.

    Shells into ``uv run --extra real-models python -m
    tests.python.integration.real_models.sota._populate_cache --vitpose``
    so the heavy ``[real-models]`` extra lives outside this package's
    dep set — same shape as :func:`populate_detr_resnet50`. Validates
    the upstream revision is pinned before spawning the subprocess; an
    unpinned :data:`VITPOSE_REVISION` fails loudly here rather than
    later inside the inference process.

    ViTPose is top-down: input is one person-box crop at a time, so the
    iteration count is the number of GT person annotations in
    val2017 (~11k boxes), not the 5000 images. First run on a clean
    machine takes ~2-3 hours on an 8-core CPU; a cache hit is seconds.
    Cached output lands at :func:`vitpose_cache_path`.
    """
    _ensure_pinned_revision(VITPOSE_REVISION, VITPOSE_MODEL_ID)
    cmd = [
        "uv",
        "run",
        "--extra",
        "real-models",
        "python",
        "-m",
        _DETR_R50_POPULATOR_MODULE,
        _VITPOSE_POPULATOR_FLAG,
    ]
    print(
        f"Shelling into [real-models] extra for ViTPose inference: {' '.join(cmd)}",
        file=sys.stderr,
    )
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


_LVIS_DETECTOR_POPULATOR_FLAG = "--lvis"


def populate_lvis_detector() -> None:
    """Run the LVIS detector on LVIS v1 val to populate the cache.

    Shells into ``uv run --extra real-models python -m
    tests.python.integration.real_models.sota._populate_cache
    --lvis-detector`` — same shape as :func:`populate_detr_resnet50`.
    Validates :data:`LVIS_DETECTOR_REVISION` is pinned before
    spawning the subprocess; an unpinned value fails loudly here
    rather than later inside the inference process.

    First run on a clean machine is the cost driver: LVIS v1 val has
    19,809 images at ~640x480 vs COCO's 5,000, and the Deformable-DETR
    forward is heavier per image than DETR-R50 (300-query head + 6
    decoder layers with deformable attention vs DETR's 100 / 6 with
    standard attention). Budget ~48-72 h on an 8-core CPU. A cache
    hit is seconds. Cached output lands at
    :func:`lvis_detector_cache_path`.
    """
    _ensure_pinned_revision(LVIS_DETECTOR_REVISION, LVIS_DETECTOR_MODEL_ID)
    cmd = [
        "uv",
        "run",
        "--extra",
        "real-models",
        "python",
        "-m",
        _DETR_R50_POPULATOR_MODULE,
        _LVIS_DETECTOR_POPULATOR_FLAG,
    ]
    print(
        f"Shelling into [real-models] extra for LVIS detector inference: {' '.join(cmd)}",
        file=sys.stderr,
    )
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def populate_detr_resnet50() -> None:
    """Run DETR-R50 inference on COCO val2017 to populate the cache.

    Shells into ``uv run --extra real-models python -m
    tests.python.integration.real_models.sota._populate_cache --detr``
    so the heavy ``[real-models]`` extra (torch, transformers,
    huggingface_hub) lives outside this package's dep set — same shape
    as :func:`populate_rfdetr`. The SOTA module owns the inference
    path; this function is just the orchestrator.

    First run on a clean machine takes ~12-15 hours on CPU (DETR-R50
    is ~9s per 640x480 image on an 8-core AMD EPYC-Milan; val2017 is
    5000 images). A cache hit is seconds. Cached output lands at
    :func:`detr_resnet50_cache_path`, which the bench adapter reads.
    """
    cmd = [
        "uv",
        "run",
        "--extra",
        "real-models",
        "python",
        "-m",
        _DETR_R50_POPULATOR_MODULE,
        "--detr",
    ]
    print(
        f"Shelling into [real-models] extra for DETR-R50 inference: {' '.join(cmd)}",
        file=sys.stderr,
    )
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m real_predictions_cache`` entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m real_predictions_cache",
        description=(
            "Populate the real-model predictions cache used by the bench "
            "harness (Mask R-CNN downloaded; rf-detr inferred on demand)."
        ),
    )
    parser.add_argument(
        "--maskrcnn",
        action="store_true",
        help="Download the Mask R-CNN R50-FPN (Detectron2 model zoo) "
        "prediction blob. Pinned URL + SHA256 in the package; "
        "ADR-level decision to bump.",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Override Mask R-CNN URL (for ad-hoc testing before the canonical upload lands).",
    )
    parser.add_argument(
        "--sha256",
        default=None,
        help="Override Mask R-CNN SHA256 to match --url.",
    )
    parser.add_argument(
        "--rfdetr",
        choices=["nano", "segnano"],
        default=None,
        help="Run rf-detr inference to populate the cache. Requires the "
        "`real-models` extra (torch, rfdetr, supervision). Note: this "
        "shells into the heavy extra; the bench env stays light.",
    )
    parser.add_argument(
        "--detr",
        action="store_true",
        help="Run facebook/detr-resnet-50 inference (Hugging Face SOTA "
        "harness) to populate the cache. Requires the `real-models` "
        "extra (torch, transformers, huggingface_hub, timm). ~12-15h "
        "on an 8-core CPU for COCO val2017. Same shell-into-the-extra "
        "shape as --rfdetr.",
    )
    parser.add_argument(
        "--mask2former-panoptic",
        action="store_true",
        help="Run facebook/mask2former-swin-tiny-coco-panoptic inference "
        "on COCO val2017 (panoptic segmentation). Requires the "
        "`real-models` extra. ~20-25h on 8-core CPU first run. "
        "MASK2FORMER_PANOPTIC_REVISION must be pinned in source.",
    )
    parser.add_argument(
        "--mask2former-ade",
        action="store_true",
        help="Run facebook/mask2former-swin-tiny-ade-semantic inference "
        "on ADE20K val (semantic segmentation). Requires the "
        "`real-models` extra + ADE20K val cache "
        "(`python -m ade20k_val_cache`). ~3-4h on 8-core CPU first "
        "run. MASK2FORMER_ADE_REVISION must be pinned in source.",
    )
    parser.add_argument(
        "--vitpose",
        action="store_true",
        help="Run usyd-community/vitpose-base-simple inference on COCO "
        "val2017 (keypoints, top-down: GT person boxes as input). "
        "Requires the `real-models` extra + COCO val2017 cache + "
        "keypoints GT (`./tools/fetch-coco-val.sh --with-images`). "
        "~2-3h on 8-core CPU first run. VITPOSE_REVISION must be "
        "pinned in source.",
    )
    parser.add_argument(
        "--lvis",
        action="store_true",
        dest="lvis_detector",
        help="Run facebook/deformable-detr-box-supervised on LVIS v1 val "
        "(bbox detection, 1203 categories, federated evaluation). "
        "Requires the `real-models` extra + LVIS v1 val cache "
        "(`python -m lvis_v1_val_cache fetch`). ~48-72h on 8-core CPU "
        "first run. LVIS_DETECTOR_REVISION must be pinned in source.",
    )
    args = parser.parse_args(argv)

    if not (
        args.maskrcnn
        or args.rfdetr
        or args.detr
        or args.mask2former_panoptic
        or args.mask2former_ade
        or args.vitpose
        or args.lvis_detector
    ):
        parser.error(
            "at least one of --maskrcnn / --rfdetr / --detr / "
            "--mask2former-panoptic / --mask2former-ade / "
            "--vitpose / --lvis is required"
        )

    if args.maskrcnn:
        path = ensure_maskrcnn(url=args.url, sha256=args.sha256)
        print(f"Mask R-CNN predictions ready: {path}")

    if args.rfdetr:
        populate_rfdetr(args.rfdetr)
        path = rfdetr_cache_path(args.rfdetr)
        print(f"rf-detr {args.rfdetr} predictions ready: {path}")

    if args.detr:
        populate_detr_resnet50()
        path = detr_resnet50_cache_path()
        print(f"DETR-R50 predictions ready: {path}")

    if args.mask2former_panoptic:
        populate_mask2former_panoptic()
        dir_path = mask2former_panoptic_cache_dir()
        print(f"Mask2Former panoptic predictions ready: {dir_path}")

    if args.mask2former_ade:
        populate_mask2former_ade()
        dir_path = mask2former_ade_cache_dir()
        print(f"Mask2Former ADE semantic predictions ready: {dir_path}")

    if args.vitpose:
        populate_vitpose()
        path = vitpose_cache_path()
        print(f"ViTPose-base-simple keypoint predictions ready: {path}")

    if args.lvis_detector:
        populate_lvis_detector()
        path = lvis_detector_cache_path()
        print(f"LVIS detector predictions ready: {path}")

    return 0
