"""CLI: populate the rf-detr prediction cache for a single model.

The bench harness's ``coco_val2017_rfdetr_*`` workloads (under
``bench/bench/workloads/real_predictions.py``) read predictions from
disk only — they have no torch / rfdetr / supervision dep. This script
is the populator: it depends on the heavy ``[real-models]`` extra and
is shelled into by ``tools/fetch-real-predictions.sh --rfdetr <model>``.

Inference is the cost driver (~30 minutes per SegNano on CPU); a cache
hit is seconds. Same cache as the pytest-driven TIDE harness — running
``pytest -m real_models`` populates it equivalently.

Usage::

    uv run --extra real-models python -m \\
        tests.python.integration.real_models.tide._populate_cache \\
        --model {nano|segnano}
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from ._cli_common import load_predictions
from ._rfdetr_predict import ModelName, cache_filename, predictions_cache_root


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tests.python.integration.real_models.tide._populate_cache",
        description="Run rf-detr inference on COCO val2017, populate the prediction cache.",
    )
    parser.add_argument(
        "--model",
        choices=("nano", "segnano"),
        required=True,
        help="rf-detr model class to populate the cache for",
    )
    args = parser.parse_args(argv)

    model: ModelName = args.model
    # ``load_predictions`` works on (model, kernel) cells; we synthesize a
    # one-cell list with bbox (the only kernel compatible with both nano
    # and segnano) just to satisfy the API. The kernel choice doesn't
    # affect inference — predictions cover all kernels the model emits.
    load_predictions([(model, "bbox")])
    cache_path = predictions_cache_root() / cache_filename(model)
    print(f"rf-detr {model} predictions cached: {cache_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
