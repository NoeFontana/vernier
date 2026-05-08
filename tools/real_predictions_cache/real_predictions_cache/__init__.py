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
# Shared cache plumbing
# ---------------------------------------------------------------------------

CACHE_ENV = "VERNIER_REAL_PREDICTIONS_CACHE"
_DATASET_ID = "coco-val2017"


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
    args = parser.parse_args(argv)

    if not (args.maskrcnn or args.rfdetr):
        parser.error("at least one of --maskrcnn / --rfdetr is required")

    if args.maskrcnn:
        path = ensure_maskrcnn(url=args.url, sha256=args.sha256)
        print(f"Mask R-CNN predictions ready: {path}")

    if args.rfdetr:
        populate_rfdetr(args.rfdetr)
        path = rfdetr_cache_path(args.rfdetr)
        print(f"rf-detr {args.rfdetr} predictions ready: {path}")

    return 0
