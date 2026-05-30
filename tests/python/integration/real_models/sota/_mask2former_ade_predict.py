"""Mask2Former Swin-T → ADE20K semantic results adapter (SOTA harness).

``facebook/mask2former-swin-tiny-ade-semantic`` is the Apache-2.0,
CPU-runnable semantic-segmentation baseline next to the panoptic
cell. Loaded via ``transformers.AutoImageProcessor`` +
``AutoModelForUniversalSegmentation``; predictions land as one
single-channel uint8 label-map PNG per image in train-id space
(``0..149`` + ``255`` ignore), matching the GT directory the
:mod:`ade20k_val_cache` materializes.

Class space alignment is direct: Mask2Former ADE-semantic publishes
labels in mmsegmentation's ``reduce_zero_label=True`` 0..149 space,
which is the same train-id encoding our ADE20K cache writes. No
class-mapping step needed (contrast with the panoptic / DETR-R50
adapters that bridge a model train-id space to a GT sparse-id space).

Inference cost: ~5-7s per ADE20K val image on an 8-core AMD
EPYC-Milan; ADE20K val is 2000 images, so end-to-end is ~3-4h.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from real_predictions_cache import (
    MASK2FORMER_ADE_MODEL_ID,
    MASK2FORMER_ADE_REVISION,
)

_DATASET_LABEL = "ade20k-val"


def _instantiate_model(revision: str) -> tuple[Any, Any]:
    """Lazy transformers import + processor / model instantiation.

    Same pinning + threading discipline as the panoptic adapter:
    ``revision=`` ties weights to the cache filename's SHA;
    ``torch.set_num_threads(1)`` makes summation order
    host-independent so the cache key fully identifies the bytes.
    """
    import torch
    from transformers import AutoImageProcessor, AutoModelForUniversalSegmentation

    torch.set_num_threads(1)

    processor = AutoImageProcessor.from_pretrained(MASK2FORMER_ADE_MODEL_ID, revision=revision)
    model = AutoModelForUniversalSegmentation.from_pretrained(
        MASK2FORMER_ADE_MODEL_ID,
        revision=revision,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return processor, model


def _assert_ade_train_id_space(model: Any) -> None:
    """Guard that the loaded checkpoint's id2label is in the ADE20K
    150-class train-id space (0..149).

    Defensive: if a future revision ships a different label layout
    (e.g. 1..150 with 0=background, or some other class count), the
    converted GT and the model output would silently diverge. Better
    to fail at load time than write 2000 bit-wrong PNGs.
    """
    id2label = model.config.id2label
    n_labels = len(id2label)
    if n_labels != 150:
        raise RuntimeError(
            f"Mask2Former ADE checkpoint has {n_labels} labels; expected 150. "
            f"The cache contract assumes the mmseg `reduce_zero_label=True` "
            f"convention (0..149 train-id space). A different count means "
            f"the cache + GT class space need re-derivation."
        )
    keys = sorted(int(k) for k in id2label)
    if keys != list(range(150)):
        raise RuntimeError(
            f"Mask2Former ADE checkpoint's id2label keys are {keys[:3]}..{keys[-3:]}; "
            f"expected contiguous 0..149. mmseg `reduce_zero_label=True` is the "
            f"assumed training convention."
        )


def _label_map_for_image(
    *,
    image_size_hw: tuple[int, int],
    processor: Any,
    model: Any,
    image: Any,
    png_out_path: Path,
) -> None:
    """Run one forward pass + semantic post-process; write the per-image
    label-map PNG.

    ``post_process_semantic_segmentation`` returns an HxW int64 tensor
    of class predictions (the per-pixel argmax over the 150 ADE
    classes). We downcast to uint8 — train-id ``0..149`` fits — and
    write as a single-channel ``L``-mode PNG. ``decode_label_map_png``
    on the vernier side reads this back as a uint8 array.

    There is no ignore-label projection: Mask2Former argmaxes over
    150 classes, so every pixel gets a confident label. The GT's
    ``255`` ignore-label cells (upstream-zero pixels) are excluded
    from the IoUMetric reduction by mmseg's ignore-aware accumulator,
    not by the DT.
    """
    import numpy as np
    import torch
    from PIL import Image as PILImage

    inputs = processor(images=image, return_tensors="pt")
    with torch.inference_mode():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image_size_hw], dtype=torch.int64)
    seg_map = processor.post_process_semantic_segmentation(
        outputs,
        target_sizes=target_sizes,
    )[0]
    arr = seg_map.to(torch.int64).cpu().numpy()
    if arr.min() < 0 or arr.max() >= 150:
        raise RuntimeError(
            f"semantic prediction at {png_out_path.name} has class ids outside "
            f"[0, 149]: min={int(arr.min())}, max={int(arr.max())}. The cache "
            f"contract assumes train-id space; a shift would silently mis-score "
            f"against the converted GT."
        )
    PILImage.fromarray(arr.astype(np.uint8, copy=False), mode="L").save(png_out_path)


def predict_ade20k_val(
    *,
    image_paths: dict[int, Path],
    cache_dir: Path,
    revision: str = MASK2FORMER_ADE_REVISION,
    progress: bool = True,
) -> Path:
    """Run Mask2Former ADE inference on every image in ``image_paths``.

    Writes per-image PNGs into ``cache_dir``; idempotent at the
    per-image level (existing PNGs are skipped on a re-run, so a
    SIGINT mid-run resumes cheaply). Returns ``cache_dir`` once all
    expected PNGs are present.

    ``image_paths`` is the ``{image_id: Path}`` map produced by
    ``ade20k_val_cache.scan_image_jpgs()``. Inference walks it in
    sorted order for reproducibility (image ids ascending).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    sorted_ids: Sequence[int] = sorted(image_paths)

    # Short-circuit if every PNG is already present. The full-coverage
    # check matches the rfdetr / DETR-R50 atomic-completion contract;
    # we do it before instantiating the model so a hit avoids the
    # multi-hundred-MB weights download.
    if all((cache_dir / f"{iid}.png").is_file() for iid in sorted_ids):
        return cache_dir

    from PIL import Image

    processor, model = _instantiate_model(revision)
    _assert_ade_train_id_space(model)

    iterator: Iterable[int] = sorted_ids
    if progress:
        from tqdm import tqdm

        iterator = tqdm(sorted_ids, total=len(sorted_ids), desc=f"mask2former-ade {_DATASET_LABEL}")

    for image_id in iterator:
        png_out = cache_dir / f"{image_id}.png"
        if png_out.is_file():
            continue
        image_path = image_paths[image_id]
        with Image.open(image_path) as pil:
            pil_rgb = pil.convert("RGB")
            _label_map_for_image(
                image_size_hw=(pil_rgb.height, pil_rgb.width),
                processor=processor,
                model=model,
                image=pil_rgb,
                png_out_path=png_out,
            )

    return cache_dir
