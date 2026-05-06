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
# (with SHA256 verification). rf-detr predictions are inferred locally
# via the `real-models` extra and TIDE harness — see the package
# README.
set -euo pipefail
exec uv run python -m real_predictions_cache "$@"
