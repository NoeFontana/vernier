"""Shared scaffolding for the Hugging Face SOTA real-prediction harness.

The three predictors (``_detr_predict``, ``_mask2former_panoptic_predict``,
``_mask2former_ade_predict``) share the same set of disciplines, paid
for once in PR #265's follow-up:

1. **Single-threaded inference** — both ``OMP_NUM_THREADS`` and
   ``MKL_NUM_THREADS`` are forced to ``"1"`` at import time AND
   ``torch.set_num_threads(1)`` is called before the first forward.
   The env-var path is the only one that reliably takes effect when
   torch's intra-op pool has already been initialised by an earlier
   test in the same pytest process; the in-process call is
   defence-in-depth. Cache keys are ``(model, revision, dataset)`` only,
   so without this pin two hosts on the same revision would populate
   bit-different bytes via summation-order drift in matmul reductions.
2. ``revision=`` pinned to the cache filename's SHA on both processor
   and model load; ``low_cpu_mem_usage=True`` on the model load alone
   (``AutoImageProcessor`` does not accept the kwarg) so the state dict
   isn't materialised twice during shard streaming.
3. Progress bars are opt-out (predictors default ``progress=True``);
   pass ``progress=False`` from parity smoke tests so the ``tqdm``
   import (lazy) and the bar wall don't show up under pytest capture.
4. Caches land via an atomic ``.part``→ rename so a SIGINT mid-write
   doesn't leave a half-truncated artefact (JSON sidecar OR per-image
   PNG) the next session mistakes for complete.
5. Class-name → GT-category-id joins are name-based with a loud-fail
   on any miss not listed in a per-cell documented drop set; the
   cache filename embeds the pinned SHA so a partial cache surfaces
   weeks later as a silent score regression rather than at populate
   time.

Each helper lives behind a lazy import where ``torch`` / ``transformers``
/ ``tqdm`` are the cost; this module imports nothing of the
``real-models`` extra at import time so it can sit beside lighter
modules without paying that bill.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, TypeVar

# Pin BLAS / OpenMP thread counts at import time. ``torch.set_num_threads``
# is documented to be a no-op once the intra-op pool has been initialised
# by an earlier ``import torch`` + parallel op in the same process (a
# common pattern when the rfdetr ``tide/`` cells run before the SOTA
# cells in the same pytest session). Setting the env vars before torch
# is imported is the only reliable way to keep the cache contract host-
# independent. ``setdefault`` so a deliberate user override (e.g. for a
# perf-sweep run that's NOT touching the cache) still wins.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def pin_inference_threads() -> None:
    """Pin PyTorch intra-op threads to 1.

    Defence-in-depth — the load-bearing pin is the import-time
    ``OMP_NUM_THREADS=1`` setting at the top of this module, which
    takes effect before torch's intra-op pool is initialised. This
    in-process call covers the case where ``OMP_NUM_THREADS`` was
    already exported in the parent env at something other than 1 (a
    deliberate override the env-time ``setdefault`` respects). Called
    by every predictor before the first forward pass.
    """
    import torch

    torch.set_num_threads(1)


def load_processor_and_model(
    model_id: str,
    revision: str,
    *,
    model_cls_name: str,
) -> tuple[Any, Any]:
    """Lazy ``transformers`` import + revision-pinned processor / model load.

    ``model_cls_name`` selects the ``AutoModelFor*`` class — DETR-R50
    uses ``AutoModelForObjectDetection``; Mask2Former (both panoptic
    and semantic) uses ``AutoModelForUniversalSegmentation``. Passed by
    name (not by class) so this module imports nothing of
    ``transformers`` at definition time.

    Pins ``torch.set_num_threads(1)`` before the model load so the
    cache contract holds; see :func:`pin_inference_threads`.
    ``low_cpu_mem_usage=True`` is only valid on
    ``AutoModelFor*.from_pretrained`` (``AutoImageProcessor`` does not
    accept the kwarg).

    Returns ``(processor, model)`` with ``model.eval()`` already
    applied.
    """
    import transformers as _tf

    pin_inference_threads()

    processor = _tf.AutoImageProcessor.from_pretrained(model_id, revision=revision)
    model_cls = getattr(_tf, model_cls_name)
    model = model_cls.from_pretrained(
        model_id,
        revision=revision,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return processor, model


_T = TypeVar("_T")


def tqdm_or_passthrough(
    items: Sequence[_T],
    *,
    desc: str,
    progress: bool,
) -> Iterator[_T]:
    """Yield from ``items`` with an optional ``tqdm`` bar.

    Lazy import: ``tqdm`` is in the ``real-models`` extra and we don't
    want to pay its import cost when running with ``progress=False``
    (e.g. inside parity smoke tests).

    Implemented as a generator that wraps ``tqdm`` in a context manager
    so an exception or SIGINT mid-iteration closes the bar deterministi-
    cally — without relying on tqdm's atexit hook, which doesn't fire
    on SIGKILL and leaves a half-rendered line on certain TTYs in
    subprocess-spawned populators.
    """
    if not progress:
        yield from items
        return
    from tqdm import tqdm

    with tqdm(total=len(items), desc=desc) as bar:
        for item in items:
            yield item
            bar.update(1)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Atomic write via ``<path>.part`` → ``rename(path)``.

    Protects against SIGINT mid-write leaving a half-truncated file the
    next session would mistake for complete. Mirrors
    ``coco_val_cache._atomic_download``'s convention. ``path``'s parent
    is created if absent.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    part.write_bytes(payload)
    part.replace(path)


def name_based_class_mapping(
    id2label: dict[int, str],
    gt_categories: Sequence[dict[str, Any]],
    *,
    dropped_names: frozenset[str] = frozenset(),
    na_marker: str | None = None,
    context: str,
) -> dict[int, int]:
    """Map model ``label_id`` (contiguous train-id space) to GT
    ``category_id`` (sparse upstream-id space) via name join.

    Loud-fail on any name that doesn't appear in the GT categories and
    isn't listed in ``dropped_names``. The cache filename embeds the
    pinned revision SHA, so a partial cache surfaces weeks later as a
    silent score regression — refusing to write at all is the only
    safe move when the join is ambiguous.

    ``na_marker`` is an optional sentinel string the model emits for
    "no class" slots (DETR-R50 uses ``"N/A"`` on its 91-slot 2014-class
    head); those are silently skipped. ``dropped_names`` is the
    documented per-cell allow-list of names that legitimately don't
    appear in the GT (e.g. the 11 COCO-91 → COCO-80 categories).

    ``context`` is the short human-readable identifier of the calling
    cell, embedded in the error message so a failure points at the
    exact (model, dataset) pair.
    """
    name_to_cat_id = {cat["name"]: int(cat["id"]) for cat in gt_categories}
    mapping: dict[int, int] = {}
    for label_id, name in id2label.items():
        if na_marker is not None and name == na_marker:
            continue
        cat_id = name_to_cat_id.get(name)
        if cat_id is None:
            if name in dropped_names:
                continue
            if dropped_names:
                drop_clause = f"isn't listed in the documented drop set ({sorted(dropped_names)})"
            else:
                # Cells that pass no drop set treat every model label as
                # must-resolve. Spell that out so a reader doesn't think
                # the remediation is to widen ``dropped_names``.
                drop_clause = (
                    "and this cell has no documented drop set — every "
                    "model label must resolve against the GT categories"
                )
            raise RuntimeError(
                f"{context}: label {label_id} ('{name}') has no matching "
                f"category in the GT JSON, and {drop_clause}. Refusing "
                f"to silently populate a partial cache under a pinned "
                f"revision SHA. GT category names: {sorted(name_to_cat_id)}"
            )
        mapping[label_id] = cat_id
    return mapping


def iter_image_records(
    gt_images: Sequence[dict[str, Any]],
    image_dir: Path,
    *,
    desc: str,
    progress: bool,
) -> Iterator[tuple[dict[str, Any], Path]]:
    """Yield ``(gt_image_record, on-disk-path)`` pairs.

    Centralises the "image file referenced by GT is missing" error so
    every predictor produces the same actionable message instead of
    drifting independently. Wraps in ``tqdm`` when ``progress=True``.
    """
    wrapped = tqdm_or_passthrough(gt_images, desc=desc, progress=progress)
    for img in wrapped:
        path = image_dir / img["file_name"]
        if not path.is_file():
            raise FileNotFoundError(
                f"image referenced by GT JSON missing on disk: {path}. "
                f"Re-run the dataset fetcher; the cache root must contain "
                f"the full images/ directory next to the GT JSON."
            )
        yield img, path


def make_detection_target_sizes(image_size_hw: tuple[int, int]) -> Any:
    """Return a 2D ``int64`` ``target_sizes`` tensor of shape (1, 2),
    suitable for transformers' object-detection ``post_process_*``.

    ``int64`` (not ``float64``) is load-bearing for the detection path:
    transformers builds ``scale_fct`` from this tensor and multiplies
    the float32 boxes against it. fp64 silently upcasts the box
    arithmetic, tying cached bytes to a transformers internal that
    can shift between minor versions.

    Note: this shape is detection-only. The segmentation post-process
    paths (``post_process_panoptic_segmentation`` /
    ``post_process_semantic_segmentation``) pass ``target_sizes[idx]``
    straight to ``torch.nn.functional.interpolate(size=...)``, which
    rejects a tensor argument in current PyTorch (raises
    ``TypeError: upsample_bilinear2d() received an invalid combination
    of arguments``). Segmentation callers pass a Python
    ``[(h, w)]`` list-of-tuples instead — see the call sites in the
    mask2former predictors.
    """
    import torch

    return torch.tensor([image_size_hw], dtype=torch.int64)
