"""rf-detr → COCO JSON adapter for the TIDE validation harness.

The :class:`rfdetr.RFDETRNano` and :class:`rfdetr.RFDETRSegNano` models
emit a :class:`supervision.Detections` object per image; vernier's
``error_decomposition`` wants a JSON byte payload in COCO's
``COCO.loadRes`` shape. This module is the single adapter — the harness
delegates here, the cache stores the COCO-shaped output, and the test
code never touches rfdetr's native types.

Cache discipline: predictions are keyed on ``(model_name,
model_version, dataset_id)``. Re-running the harness with the same pin
hits the cache and skips inference; bumping the rfdetr pin (an
ADR-level operation per the vendoring policy) invalidates the cache by
construction.

Version pinning is the SHA-pinning analog for vendored pip packages:
unlike the SOTA harness's Hugging Face cells (DETR-R50, Mask2Former
panoptic/ADE) which embed a 40-hex hub commit in the cache filename,
rfdetr ships as a pip package and ``RFDETR_VERSION`` (the package
pin) plays the same role. The cache filename embeds the version
string so any pip-side bump invalidates the cache by construction.

Thread-pin caveat: this module calls
:func:`_harness_common.pin_inference_threads` before the first
forward pass to keep newly-populated cache bytes host-independent
(matmul reduction order with intra-op threads is not deterministic
across NUMA topologies). The pin is cache-stable for NEW populates
only — any existing cache files predating this commit were produced
without the pin, and the bit-equality contract on the SOTA boundary
cell holds against whichever ordering those bytes capture (deleting
+ re-populating is the only way to re-tighten the seam for a host
that swapped CPU topology since the original populate).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

# Side-effect import: ``_harness_common`` sets ``OMP_NUM_THREADS=1`` /
# ``MKL_NUM_THREADS=1`` / ``OPENBLAS_NUM_THREADS=1`` at import time
# (via ``os.environ.setdefault``). Reaching for this at module top
# rather than inside the predictor body so the env-var pin lands
# before ``rfdetr`` (and its transitive ``torch`` import) materialises
# the intra-op thread pool; once initialised, ``torch.set_num_threads``
# is documented as a no-op. The in-process pin in
# :func:`_instantiate_model` is defence-in-depth for the case where
# the parent process exported a non-1 value the env-time
# ``setdefault`` deliberately respects.
from ..sota import _harness_common

if TYPE_CHECKING:
    import numpy as np
    from supervision import Detections


_RFDETR_VERSION = "1.6.5.post0"
_DATASET_ID = "coco-val2017"

#: Cache-blob content tag. Mirrors
#: :data:`real_predictions_cache._RFDETR_CACHE_BLOB_VERSION` exactly —
#: see that constant's docstring for the v1 → v2 rationale (the
#: thread-pin landing in :func:`_instantiate_model` makes ``v1`` bytes
#: host-dependent, ``v2`` bytes deterministic). Both filename builders
#: must agree byte-for-byte or the TIDE populator and the bench
#: adapter would point at different files; that's why this constant
#: is duplicated here rather than imported (this module pre-dates the
#: shared cache package and avoids the dep to stay light).
_RFDETR_CACHE_BLOB_VERSION = "v2"

#: rf-detr model variants the harness exercises. ``nano`` is the
#: bbox-only RFDETRNano; ``segnano`` is the instance-seg RFDETRSegNano
#: (which also produces masks usable by the boundary kernel).
ModelName = Literal["nano", "segnano"]


def cache_filename(model_name: ModelName) -> str:
    """Stable filename for cached predictions.

    Versioned + dataset-tagged so a pin bump or dataset swap can't
    silently reuse stale predictions. Also embeds
    :data:`_RFDETR_CACHE_BLOB_VERSION` (the cache-blob content tag) so
    a harness-side change that affects on-disk bytes (e.g. the
    thread-pin that bumped v1 → v2) forces a re-populate instead of
    silently serving stale bytes on hosts that already have a v1
    file.
    """
    return f"rfdetr-{model_name}-{_RFDETR_VERSION}-{_RFDETR_CACHE_BLOB_VERSION}-{_DATASET_ID}.json"


def predictions_cache_root() -> Path:
    """Per-user cache for model predictions (machine-local).

    Resolves via :func:`platformdirs.user_cache_dir` so the path is
    XDG-correct on Linux, ``~/Library/Caches/...`` on macOS, and
    ``%LOCALAPPDATA%\\...`` on Windows. Predictions are large and
    slow to recompute — keying them per ``(model, version, dataset)``
    lets a re-run of the harness skip inference entirely.
    """
    import platformdirs

    root = Path(platformdirs.user_cache_dir("vernier")) / "real-models"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _coco_class_mapping(gt: dict[str, Any], dense_class_names: list[str]) -> dict[int, int]:
    """Map rfdetr's dense ``class_id`` (0..79) to COCO's sparse
    ``category_id`` (1..90 with gaps).

    Resolved by name match against the GT JSON's ``categories`` list,
    not by sorted-position fallback: matching by index works on
    canonical COCO splits but breaks silently on any subset that drops
    a class. Name-match fails loudly, which is what we want.
    """
    name_to_cat_id = {cat["name"]: int(cat["id"]) for cat in gt["categories"]}
    mapping: dict[int, int] = {}
    for dense_idx, name in enumerate(dense_class_names):
        if name not in name_to_cat_id:
            raise RuntimeError(
                f"rfdetr COCO class {dense_idx} ('{name}') has no matching "
                f"category in the GT JSON; the harness can't map class ids. "
                f"GT category names: {sorted(name_to_cat_id)}"
            )
        mapping[dense_idx] = name_to_cat_id[name]
    return mapping


def _xyxy_to_xywh(xyxy: np.ndarray) -> list[float]:
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    return [x1, y1, x2 - x1, y2 - y1]


def _mask_to_rle(mask: np.ndarray) -> dict[str, Any]:
    """Boolean mask (H, W) → COCO RLE.

    ``np.asfortranarray(mask, dtype=np.uint8)`` does the dtype cast and
    Fortran-order enforcement in a single pass — the encoder needs both.
    The ``counts`` field comes back as bytes; decoding to ``ascii`` is
    required so the JSON layer doesn't choke on non-UTF-8 bytes.
    """
    import numpy as np
    from pycocotools import mask as mask_utils

    rle = mask_utils.encode(np.asfortranarray(mask, dtype=np.uint8))
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"size": [int(rle["size"][0]), int(rle["size"][1])], "counts": counts}


def _detections_to_records(
    detections: Detections,
    image_id: int,
    class_mapping: dict[int, int],
    *,
    include_masks: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    n = len(detections.xyxy)
    if n == 0:
        return records
    confidences = detections.confidence
    class_ids = detections.class_id
    masks = detections.mask if include_masks else None
    if confidences is None or class_ids is None:
        raise RuntimeError(
            "rfdetr returned a Detections object without confidence/class_id; "
            "the harness assumed a populated detection set"
        )
    if include_masks and masks is None:
        raise RuntimeError("expected segmentation masks on a seg model output, got None")

    for i in range(n):
        dense_class = int(class_ids[i])
        if dense_class not in class_mapping:
            continue
        rec: dict[str, Any] = {
            "image_id": int(image_id),
            "category_id": int(class_mapping[dense_class]),
            "bbox": _xyxy_to_xywh(detections.xyxy[i]),
            "score": float(confidences[i]),
        }
        if include_masks and masks is not None:
            rec["segmentation"] = _mask_to_rle(masks[i])
        records.append(rec)
    return records


def _instantiate_model(model_name: ModelName) -> tuple[Any, bool]:
    """Lazy rfdetr import + model instantiation. Returns ``(model, include_masks)``.

    Pins ``torch.set_num_threads(1)`` before instantiating the model
    so the cache contract holds against host topology changes — same
    discipline the SOTA harness's predictors enforce; see
    :func:`_harness_common.pin_inference_threads`. The pin is
    defence-in-depth on top of the import-time env-var pin in
    :mod:`_harness_common`; the env-var path is the only one that
    reliably wins once torch's intra-op pool is live.

    Device selection: rfdetr's ``config._detect_device()`` uses
    ``torch.accelerator.current_accelerator()`` which mis-reports
    ``cuda`` on CUDA-built PyTorch wheels even when no NVIDIA driver
    is present (driver check is deferred to first CUDA call, which
    then crashes inside ``predict()``). We explicitly probe
    ``torch.cuda.is_available()`` (which *does* try to load the
    driver) and pass ``device="cpu"`` when no GPU is reachable. On
    real CUDA hosts this is a no-op — the model still lands on GPU.
    """
    _harness_common.pin_inference_threads()

    import rfdetr
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_name == "nano":
        return rfdetr.RFDETRNano(device=device), False
    return rfdetr.RFDETRSegNano(device=device), True


def predict_coco_val(
    *,
    model_name: ModelName,
    gt: dict[str, Any],
    image_dir: Path,
    cache_path: Path,
    threshold: float = 0.05,
    progress: bool = True,
) -> bytes:
    """Run rfdetr inference on every image in ``gt['images']``, emit COCO JSON.

    Owns the cache contract end-to-end: a hit on ``cache_path`` returns
    the bytes without instantiating a model (which would download
    weights). On a miss, instantiates the model lazily, runs inference,
    writes the cache, and returns the bytes. Either way the bytes are
    in the same shape ``vernier.instance.error_decomposition`` consumes.

    ``threshold=0.05`` is deliberately permissive — TIDE rewards keeping
    low-confidence FPs visible (they populate the Bkg / Bkg+Cls bins);
    cutting them at 0.5 would distort the decomposition. The mAP
    accumulator sees the full PR curve regardless.
    """
    if cache_path.is_file():
        return cache_path.read_bytes()

    from rfdetr.assets.coco_classes import COCO_CLASSES

    model, include_masks = _instantiate_model(model_name)
    class_mapping = _coco_class_mapping(gt, list(COCO_CLASSES.values()))
    images: Iterable[dict[str, Any]] = gt["images"]

    if progress:
        from tqdm import tqdm

        images = tqdm(list(images), desc=f"rfdetr-{model_name} val2017")

    from PIL import Image

    records: list[dict[str, Any]] = []
    for img in images:
        image_path = image_dir / img["file_name"]
        if not image_path.is_file():
            raise FileNotFoundError(
                f"image referenced by GT JSON missing on disk: {image_path}. "
                f"Re-run the COCO val2017 fetcher; the cache root must contain "
                f"the full val2017/ directory next to instances_val2017.json."
            )
        # Force RGB before handing the image to ``model.predict``.
        # COCO val2017 contains a small number of grayscale (``"L"``)
        # and RGBA / CMYK images; rfdetr's ``predict()`` accepts a path
        # by ``Image.open()``-ing it directly, which preserves the
        # source mode and trips the model's 3-channel guard (see
        # ``rfdetr.detr.RFDETR.predict`` — ``if img.shape[0] != 3``).
        # Converting at the harness boundary keeps every val2017 image
        # going through the same canonicalisation and matches what the
        # other COCO-val SOTA cells do (HuggingFace image processors
        # ingest via ``Image.convert`` under the hood). ``convert("RGB")``
        # is a no-op on already-RGB images, so cache bytes stay
        # deterministic for the dominant case.
        with Image.open(image_path) as pil_img:
            rgb_img = pil_img.convert("RGB")
            detections = model.predict(rgb_img, threshold=threshold)
        records.extend(
            _detections_to_records(
                detections,
                image_id=int(img["id"]),
                class_mapping=class_mapping,
                include_masks=include_masks,
            )
        )

    payload = json.dumps(records).encode("utf-8")
    cache_path.write_bytes(payload)
    return payload
