"""CLI: populate the Hugging Face SOTA prediction cache.

The bench harness's ``coco_val2017_detr_r50_*`` /
``coco_val2017_mask2former_panoptic_*`` / ``ade20k_val_mask2former_*``
/ ``lvis_v1_val_deformable_detr_*`` workloads (under
``bench/bench/workloads/real_predictions.py``) read predictions from
disk only — no torch / transformers / huggingface_hub dep. This
script is the populator; it carries the heavy ``[real-models]``
extra and is shelled into by ``tools/fetch-real-predictions.sh``
(via the matching ``real_predictions_cache.populate_*`` helpers).

Four model flags, four cache layouts:

- ``--detr`` → one ``detr-r50-<sha>-coco-val2017.json`` (COCO
  detection results shape). ~12-15h on 8-core CPU.
- ``--mask2former-panoptic`` → a directory ``mask2former-pan-swin-t-
  <sha>-coco-val2017/`` with one rgb2id-encoded PNG per image +
  one ``panoptic_dt.json`` sidecar. ~20-25h on 8-core CPU.
- ``--mask2former-ade`` → a directory ``mask2former-ade-swin-t-
  <sha>-ade20k-val/`` with one single-channel label-map PNG per
  image (no JSON sidecar — semantic is just label maps). ~3-4h on
  8-core CPU.
- ``--vitpose`` → one ``vitpose-base-simple-<sha>-coco-val2017.json``
  (COCO keypoints results shape; top-down on GT person boxes).
  ~2-3h on 8-core CPU.
- ``--lvis`` → one ``deformable-detr-lvis-<sha>-lvis-v1-val.json``
  (LVIS results shape, same flat list as COCO detection). ~48-72h on
  8-core CPU (19,809 LVIS val images at ~10s/image for Deformable-
  DETR's 300-query / 6-decoder-layer forward).

Usage::

    uv run --extra real-models python -m \\
        tests.python.integration.real_models.sota._populate_cache --detr
    uv run --extra real-models python -m \\
        tests.python.integration.real_models.sota._populate_cache --mask2former-panoptic
    uv run --extra real-models python -m \\
        tests.python.integration.real_models.sota._populate_cache --mask2former-ade
    uv run --extra real-models python -m \\
        tests.python.integration.real_models.sota._populate_cache --vitpose
    uv run --extra real-models python -m \\
        tests.python.integration.real_models.sota._populate_cache --lvis
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from coco_val_cache import GT_FILENAME, IMAGES_DIRNAME, KP_GT_FILENAME, ensure_kp_gt
from coco_val_cache import cache_root as _coco_cache_root
from real_predictions_cache import (
    detr_resnet50_cache_path,
    lvis_detector_cache_path,
    mask2former_ade_cache_dir,
    mask2former_panoptic_cache_dir,
    mask2former_panoptic_dt_json_path,
    vitpose_cache_path,
)

from ._detr_predict import predict_coco_val as _detr_predict_coco_val

_PANOPTIC_GT_FILENAME = "panoptic_val2017.json"


def _coco_val_root() -> Path:
    """Locate the val2017 cache; abort cleanly if it's not populated.

    Mirrors ``tide/_cli_common.coco_val_root`` — copied rather than
    reused because this module sits parallel to ``tide/`` and shouldn't
    take a dep on rfdetr-specific plumbing.
    """
    root = _coco_cache_root()
    gt = root / GT_FILENAME
    images = root / IMAGES_DIRNAME
    if not gt.is_file() or not images.is_dir():
        raise SystemExit(
            f"COCO val2017 not found at {root}: need both "
            f"{GT_FILENAME} and {IMAGES_DIRNAME}/ images. "
            f"Run `./tools/fetch-coco-val.sh --with-images` to populate "
            f"the cache. Override the path with VERNIER_COCO_CACHE."
        )
    return root


def _populate_detr() -> Path:
    root = _coco_val_root()
    gt_dict = json.loads((root / GT_FILENAME).read_bytes())
    cache_path = detr_resnet50_cache_path()
    _detr_predict_coco_val(
        gt=gt_dict,
        image_dir=root / IMAGES_DIRNAME,
        cache_path=cache_path,
    )
    return cache_path


def _populate_mask2former_panoptic() -> Path:
    """Run Mask2Former panoptic inference on COCO val2017.

    Reads the COCO panoptic GT JSON (NOT the detection GT — the
    panoptic categories are a superset with stuff classes added).
    The panoptic GT lives in the :mod:`panoptic_val_cache` directory
    by convention; this populator pulls it from there so a clean
    machine that's only run ``./tools/fetch-coco-val.sh`` (detection
    flow) gets a clear "provision panoptic cache first" error.
    """
    from panoptic_val_cache import cache_root as _panoptic_cache_root

    root = _coco_val_root()  # for the images
    panoptic_root = _panoptic_cache_root()
    panoptic_gt = panoptic_root / _PANOPTIC_GT_FILENAME
    if not panoptic_gt.is_file():
        raise SystemExit(
            f"COCO panoptic val2017 GT not found at {panoptic_gt}. "
            f"Run `python -m panoptic_val_cache` to provision it "
            f"(~250MB JSON + ~3GB PNGs from images.cocodataset.org)."
        )
    gt_dict = json.loads(panoptic_gt.read_bytes())

    from ._mask2former_panoptic_predict import predict_coco_panoptic_val

    cache_dir = mask2former_panoptic_cache_dir()
    dt_json = mask2former_panoptic_dt_json_path()
    predict_coco_panoptic_val(
        gt=gt_dict,
        image_dir=root / IMAGES_DIRNAME,
        cache_dir=cache_dir,
        dt_json_path=dt_json,
    )
    return cache_dir


def _populate_vitpose() -> Path:
    """Run ViTPose-base-simple inference on COCO val2017 (keypoints).

    Top-down predictor: iterates over GT person annotations rather
    than over images, so the inner loop count is ~11k boxes (not
    5000 images). Ensures the keypoints GT is materialised under the
    val2017 cache root (same upstream zip as the instances GT) before
    handing off to the predictor.
    """
    root = _coco_val_root()
    # `ensure_kp_gt` is idempotent under the pinned SHA — cheap to
    # call even on a populated cache.
    ensure_kp_gt(cache=root)
    gt_dict = json.loads((root / KP_GT_FILENAME).read_bytes())

    from ._vitpose_predict import predict_coco_val as _vitpose_predict_coco_val

    cache_path = vitpose_cache_path()
    _vitpose_predict_coco_val(
        gt=gt_dict,
        image_dir=root / IMAGES_DIRNAME,
        cache_path=cache_path,
    )
    return cache_path


def _populate_lvis_detector() -> Path:
    """Run Deformable-DETR (LVIS box-supervised) inference on LVIS v1 val.

    Pulls the LVIS GT JSON + val2017 image directory from the
    :mod:`lvis_v1_val_cache` provisioner (which delegates the image
    side to :mod:`coco_val_cache` — LVIS v1 reuses COCO 2017 images).
    A missing cache surfaces the ``python -m lvis_v1_val_cache fetch``
    hint rather than half-running inference and writing a partial
    cache.
    """
    import os as _os

    from lvis_v1_val_cache import ensure_gt as _ensure_lvis_gt
    from lvis_v1_val_cache import ensure_images as _ensure_lvis_images

    gt_path = _ensure_lvis_gt()
    images_dir = _ensure_lvis_images()
    gt_dict = json.loads(gt_path.read_bytes())

    # Honour the same sub-sample knob the test side reads. Cuts the
    # 19,809-image full-corpus populate (~48-72 h CPU) down to a
    # representative prefix the harness can validate end-to-end in a
    # single session. The full populate (env unset, default sentinel
    # ``-1`` = no cap) remains supported and is what the published
    # headline numbers will be captured against.
    sample_env = _os.environ.get("VERNIER_LVIS_REAL_VAL_SAMPLE_IMAGES", "")
    try:
        sample_n = int(sample_env) if sample_env else -1
    except ValueError:
        sample_n = -1
    if sample_n > 0 and sample_n < len(gt_dict["images"]):
        kept = gt_dict["images"][:sample_n]
        kept_ids = {im["id"] for im in kept}
        gt_dict = {
            **gt_dict,
            "images": kept,
            "annotations": [a for a in gt_dict["annotations"] if a["image_id"] in kept_ids],
        }

    from ._lvis_detector_predict import predict_lvis_val

    cache_path = lvis_detector_cache_path()
    predict_lvis_val(
        gt=gt_dict,
        image_dir=images_dir,
        cache_path=cache_path,
    )
    return cache_path


def _populate_mask2former_ade() -> Path:
    """Run Mask2Former ADE semantic inference on ADE20K val.

    Reads the ADE20K val images from :mod:`ade20k_val_cache`. A
    missing cache surfaces the "provision via `python -m
    ade20k_val_cache`" message — same shape gate as the panoptic
    flow above.
    """
    from ade20k_val_cache import ensure_gt as _ensure_ade_gt
    from ade20k_val_cache import scan_image_jpgs

    _, images_dir, _ = _ensure_ade_gt()
    image_paths = scan_image_jpgs(images_dir)
    if not image_paths:
        raise SystemExit(
            f"ADE20K val images empty at {images_dir}. Re-run "
            f"`python -m ade20k_val_cache` to rematerialize."
        )

    from ._mask2former_ade_predict import predict_ade20k_val

    cache_dir = mask2former_ade_cache_dir()
    predict_ade20k_val(image_paths=image_paths, cache_dir=cache_dir)
    return cache_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tests.python.integration.real_models.sota._populate_cache",
        description="Run Hugging Face SOTA inference, populate the prediction cache.",
    )
    parser.add_argument(
        "--detr",
        action="store_true",
        help="Run facebook/detr-resnet-50 (revision pinned by "
        "real_predictions_cache.DETR_RESNET50_REVISION) and write COCO "
        "results JSON to the cache.",
    )
    parser.add_argument(
        "--mask2former-panoptic",
        action="store_true",
        help="Run facebook/mask2former-swin-tiny-coco-panoptic on COCO "
        "val2017 and write a per-image rgb2id PNG cache + "
        "panoptic_dt.json sidecar.",
    )
    parser.add_argument(
        "--mask2former-ade",
        action="store_true",
        help="Run facebook/mask2former-swin-tiny-ade-semantic on ADE20K "
        "val and write a per-image label-map PNG cache (train-id 0..149).",
    )
    parser.add_argument(
        "--vitpose",
        action="store_true",
        help="Run usyd-community/vitpose-base-simple on COCO val2017 "
        "(keypoints, top-down on GT person boxes) and write a COCO "
        "keypoints results JSON to the cache.",
    )
    parser.add_argument(
        "--lvis",
        action="store_true",
        dest="lvis_detector",
        help="Run facebook/deformable-detr-box-supervised on LVIS v1 val "
        "(revision pinned by LVIS_DETECTOR_REVISION) and write LVIS "
        "results JSON to the cache.",
    )
    args = parser.parse_args(argv)

    if not (
        args.detr
        or args.mask2former_panoptic
        or args.mask2former_ade
        or args.vitpose
        or args.lvis_detector
    ):
        parser.error(
            "at least one model flag is required: "
            "--detr / --mask2former-panoptic / --mask2former-ade / "
            "--vitpose / --lvis"
        )

    if args.detr:
        path = _populate_detr()
        print(f"DETR-R50 predictions cached: {path}")

    if args.mask2former_panoptic:
        path = _populate_mask2former_panoptic()
        print(f"Mask2Former panoptic predictions cached: {path}")

    if args.mask2former_ade:
        path = _populate_mask2former_ade()
        print(f"Mask2Former ADE-semantic predictions cached: {path}")

    if args.vitpose:
        path = _populate_vitpose()
        print(f"ViTPose-base-simple keypoint predictions cached: {path}")

    if args.lvis_detector:
        path = _populate_lvis_detector()
        print(f"LVIS detector predictions cached: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
