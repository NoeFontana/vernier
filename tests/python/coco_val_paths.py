"""Shared env-var contract + cache layout for the COCO val2017 parity smoke.

`tests/python/parity/test_coco_val.py` (bbox + segm) and
`tests/python/parity_boundary/test_coco_val.py` (boundary) both gate on
the same `VERNIER_COCO_*_PATH` env vars and the same cache layout —
the contract is single-sourced in `docs/engineering/coco-val-parity.md`.

Cache constants and the cache-root resolution come from the canonical
:mod:`coco_val_cache` package (a path-source dev dep at
``tools/coco_val_cache/``); this module adds only the test-side env-var
contract and the pytest skip-or-return helpers.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from coco_val_cache import GT_FILENAME, IMAGES_DIRNAME, cache_root

GT_ENV = "VERNIER_COCO_GT_PATH"
DT_ENV = "VERNIER_COCO_DT_PATH"
DT_SEGM_ENV = "VERNIER_COCO_DT_SEGM_PATH"
# Keypoints needs the kp-flavored GT (`person_keypoints_val2017.json`),
# distinct from the detection GT (`instances_val2017.json`) used by the
# bbox/segm/boundary tracks — so it gets its own GT env var alongside
# the predictions one.
GT_KEYPOINTS_ENV = "VERNIER_COCO_GT_KEYPOINTS_PATH"
DT_KEYPOINTS_ENV = "VERNIER_COCO_DT_KEYPOINTS_PATH"


def require_env_path(env: str) -> Path:
    value = os.environ.get(env)
    if not value:
        pytest.skip(f"{env} is unset; see docs/engineering/coco-val-parity.md")
    path = Path(value).expanduser()
    if not path.is_file():
        pytest.skip(f"{env}={value!r} does not point to a file")
    return path


def require_perfect_dt_artifacts(dt_filename: str) -> tuple[Path, Path]:
    """Locate the cached GT and a perfect-DT JSON, or skip the test.

    Both artifacts are produced by ``./tools/fetch-coco-val.sh``
    (a thin shim over the canonical :mod:`coco_val_cache` package).
    Skipping when either is missing keeps ``just test`` green on a
    clean checkout.
    """
    cache = cache_root()
    gt = cache / GT_FILENAME
    dt = cache / dt_filename
    if not gt.is_file() or not dt.is_file():
        pytest.skip(
            f"run ./tools/fetch-coco-val.sh to populate {cache}; "
            "see docs/engineering/coco-val-parity.md"
        )
    return gt, dt


def require_coco_val_root_with_images() -> Path:
    """Locate the val2017 cache containing both GT JSON *and* images,
    or skip the test.

    Distinct from :func:`require_perfect_dt_artifacts`: real-model
    harnesses need pixels (the ``val2017/`` image directory), not just
    cached prediction JSONs. ``./tools/fetch-coco-val.sh --with-images``
    populates both in one shot.
    """
    cache = cache_root()
    gt = cache / GT_FILENAME
    images = cache / IMAGES_DIRNAME
    if not gt.is_file() or not images.is_dir():
        pytest.skip(
            f"real-model harness needs both {gt} and {images}/ — run "
            f"`./tools/fetch-coco-val.sh --with-images` to populate the "
            f"cache; see docs/engineering/coco-val-parity.md"
        )
    return cache


