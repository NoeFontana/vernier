"""Whole-dataset parity smoke tests against pycocotools on COCO val2017.

Five hand-crafted fixtures (``test_parity.py``) cover known quirks but
do not exercise the long tail of real-world inputs. The tests here run
the full COCO val2017 ground truth and assert every snapshot field —
eval-image dicts, precision/recall/scores tensors, the 12-element stats
vector — matches pycocotools bit-for-bit in strict parity mode.

Skipped by default: enabling them requires the COCO val2017
annotations (license-restricted, ~245 MB) and a detections JSON.
Neither is committed. See ``docs/engineering/coco-val-parity.md`` for
the env-var contract and the ``tools/fetch-coco-val.sh`` helper that
populates ``.cache/coco-val2017/``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from .harness import assert_snapshots_equal, snapshot

_GT_ENV = "VERNIER_COCO_GT_PATH"
_DT_ENV = "VERNIER_COCO_DT_PATH"
_CACHE_ENV = "VERNIER_COCO_CACHE"

# Mirrors tools/fetch-coco-val.sh: same default and same VERNIER_COCO_CACHE
# override, so the perfect-DT test finds the artifacts the helper wrote.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CACHE_DIR = _REPO_ROOT / ".cache" / "coco-val2017"


def _cache_dir() -> Path:
    override = os.environ.get(_CACHE_ENV)
    return Path(override).expanduser() if override else _DEFAULT_CACHE_DIR


def _require_env_path(env: str) -> Path:
    value = os.environ.get(env)
    if not value:
        pytest.skip(f"{env} is unset; see docs/engineering/coco-val-parity.md")
    path = Path(value).expanduser()
    if not path.is_file():
        pytest.skip(f"{env}={value!r} does not point to a file")
    return path


def _assert_bbox_parity(gt: Path, dt: Path) -> None:
    ref = snapshot("pycocotools", gt, dt, "bbox")
    cand = snapshot("vernier", gt, dt, "bbox")
    assert_snapshots_equal(ref, cand)


@pytest.mark.parity
@pytest.mark.coco_val
def test_coco_val2017_bbox_parity() -> None:
    """Real detector predictions: bring your own JSON via env vars."""
    _assert_bbox_parity(_require_env_path(_GT_ENV), _require_env_path(_DT_ENV))


@pytest.mark.parity
@pytest.mark.coco_val
@pytest.mark.xfail(
    strict=False,
    reason=(
        "Phase 1 Week 5 ships the FFI surface; the matching engine "
        "(Week 3) and accumulator (Week 4) are still maturing. The "
        "perfect-DT smoke surfaces overlapping-GT tiebreak (A4) "
        "divergence on a handful of val2017 images. Tracked as the "
        "headline parity goal — turns green as those weeks land. See "
        "docs/engineering/coco-val-parity.md."
    ),
)
def test_coco_val2017_bbox_parity_perfect_dt() -> None:
    """Synthesised perfect-DT smoke: scale-only check (AP is trivially 1.0)."""
    cache = _cache_dir()
    gt = cache / "instances_val2017.json"
    dt = cache / "perfect_dt.json"
    if not gt.is_file() or not dt.is_file():
        pytest.skip(
            f"run ./tools/fetch-coco-val.sh to populate {cache}; "
            "see docs/engineering/coco-val-parity.md"
        )
    _assert_bbox_parity(gt, dt)
