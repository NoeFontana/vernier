#!/usr/bin/env bash
# User-facing entry point for the real-model predictions cache.
#
# Logic lives in the `real_predictions_cache` Python package
# (`tools/real_predictions_cache/`), declared as a `[tool.uv.sources]`
# path dep of bench/pyproject.toml. This script is a thin shim so
# muscle memory aligns with `tools/fetch-coco-val.sh`; arguments
# forward to the Python CLI.
#
# Mask R-CNN predictions are fetched from a pinned Hugging Face URL
# (with SHA256 verification). rf-detr, DETR-R50, and Mask2Former
# (panoptic + ADE-semantic) predictions are inferred locally via the
# `real-models` extra (TIDE and SOTA harnesses) — see the package
# README. Available flags:
#   --maskrcnn               download Mask R-CNN R50-FPN predictions
#   --rfdetr {nano,segnano}  infer rf-detr predictions
#   --detr                   infer DETR-R50 (bbox) predictions
#   --mask2former-panoptic   infer Mask2Former Swin-T panoptic (COCO val2017)
#   --mask2former-ade        infer Mask2Former Swin-T ADE-semantic (ADE20K val)
set -euo pipefail
exec uv run python -m real_predictions_cache "$@"
