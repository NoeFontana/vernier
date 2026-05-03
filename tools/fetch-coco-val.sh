#!/usr/bin/env bash
# User-facing entry point for COCO val2017 cache setup.
#
# Logic lives in the `coco_val_cache` Python package
# (`tools/coco_val_cache/`), declared as a `[tool.uv.sources]` path
# dep of both the root pyproject and bench/pyproject. This script is
# a thin shim so existing muscle memory ('./tools/fetch-coco-val.sh')
# still works; arguments forward to the Python CLI.
#
# COCO data is governed by the COCO terms of use; we never commit it.
set -euo pipefail
exec uv run python -m coco_val_cache "$@"
