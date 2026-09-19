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


#: rf-detr model variants the harness exercises. ``nano`` is the
#: bbox-only RFDETRNano; ``segnano`` is the instance-seg RFDETRSegNano
#: (which also produces masks usable by the boundary kernel).
ModelName = Literal["nano", "segnano"]

# Both helpers below delegate to ``real_predictions_cache``, which is
# the single owner of the rfdetr pin, the cache-blob version and the
# cache root. This module used to keep its own copy of all three and
# rebuild the filename itself, justified by a comment about avoiding
# the dep — stale since the package became a dev dependency that the
# four SOTA predictors and this directory's own conftest all import.
# The cost was concrete: the v2 → v3 bump had to be hand-applied in
# two files, and the only assertion pinning the spelling lives in
# ``bench/tests/``, which does not run in CI. A one-sided bump would
# therefore keep a corrupt cache live with a green suite — the exact
# failure the blob version exists to prevent.
#
# Imported inside the functions, not at module scope:
# ``real_predictions_cache`` imports ``platformdirs``, which ships in
# the ``real-models`` extra, and this module must stay importable
# without it so the mapping tests collect (and skip) cleanly on a
# host that has no extra installed.


def cache_filename(model_name: ModelName) -> str:
    """Stable filename for cached predictions.

    Versioned + dataset-tagged so a pin bump or dataset swap can't
    silently reuse stale predictions, and blob-versioned so a
    harness-side change that affects on-disk bytes forces a
    re-populate rather than serving stale ones.
    """
    from real_predictions_cache import rfdetr_cache_filename

    return rfdetr_cache_filename(model_name)


def predictions_cache_root() -> Path:
    """Per-user cache for model predictions (machine-local).

    Honours ``$VERNIER_REAL_PREDICTIONS_CACHE`` before falling back to
    the XDG location. The hand-rolled copy this replaces resolved
    ``platformdirs`` directly and ignored that variable, so on a host
    that set it the TIDE populator wrote to one root while the bench
    adapter read from another.
    """
    from real_predictions_cache import cache_root

    root = cache_root()
    root.mkdir(parents=True, exist_ok=True)
    return root


#: Class slot rfdetr's COCO checkpoints reserve for "no object".
#: ``class_embed.out_features`` is 91 on the COCO-pretrained
#: RFDETRNano / RFDETRSegNano: slots 1..90 are COCO's sparse category
#: ids and slot 0 is the only one left over. Detections carrying it
#: are skipped; every *other* unmapped id is a hard error.
_BACKGROUND_CLASS_ID = 0


def _coco_class_mapping(gt: dict[str, Any], model_classes: dict[int, str]) -> dict[int, int]:
    """Map rfdetr's ``class_id`` to COCO's ``category_id``.

    The name join is :func:`_harness_common.name_based_class_mapping`,
    the same helper the three SOTA predictors use — one discipline and
    one error message for "model label space vs GT category space"
    across the whole harness.

    The identity assertion on top is what the name join cannot give
    us. A name join is only as good as the *keys* it is handed, and
    ``rfdetr.assets.coco_classes.COCO_CLASSES`` is a
    ``{category_id: name}`` dict keyed by COCO's sparse ids (1..90
    with gaps) while the checkpoints emit that same sparse id as
    ``Detections.class_id`` (``class_embed.out_features == 91``: the
    90 category ids plus :data:`_BACKGROUND_CLASS_ID`). Hand the join
    a table re-keyed to a dense 0..79 index and it maps every label to
    a neighbouring category and reports success. That is exactly what
    shipped here, and what rfdetr's own ``predict()`` still does when
    it attaches ``data["class_name"]`` — so the two wrongs agreed and
    neither looked wrong from the outside.

    This cell is pinned to canonical COCO val2017, where the model's
    class space *is* the GT's category space, so identity is the
    correct answer and any departure from it means the two id spaces
    have come apart. A cell on a renumbered GT subset would need its
    own cache key and would drop this assertion, keeping the join.
    """
    mapping = _harness_common.name_based_class_mapping(
        model_classes, gt["categories"], context="rfdetr (coco-val2017)"
    )
    relabelled = {k: v for k, v in mapping.items() if k != v}
    if relabelled:
        first_id, first_cat = next(iter(sorted(relabelled.items())))
        raise RuntimeError(
            f"rfdetr class ids and COCO category ids have come apart: "
            f"{len(relabelled)} of {len(mapping)} labels resolve to a different "
            f"id than they carry (e.g. class {first_id} -> category {first_cat}). "
            f"On canonical COCO val2017 this mapping must be the identity; a "
            f"non-identity result means the class table was keyed on something "
            f"other than the model's own class-id space."
        )
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
        class_id = int(class_ids[i])
        if class_id == _BACKGROUND_CLASS_ID:
            continue
        if class_id not in class_mapping:
            # Loud, not `continue`: a partial cache under a pinned
            # filename surfaces weeks later as a silent score shift.
            raise RuntimeError(
                f"rfdetr emitted class_id {class_id}, which maps to no COCO "
                f"category (known ids: {min(class_mapping)}..{max(class_mapping)}). "
                f"Refusing to write a partial prediction cache."
            )
        rec: dict[str, Any] = {
            "image_id": int(image_id),
            "category_id": class_mapping[class_id],
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
    class_mapping = _coco_class_mapping(gt, COCO_CLASSES)
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
