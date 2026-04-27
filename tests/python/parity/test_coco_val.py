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

# Cache layout produced by tools/fetch-coco-val.sh; the perfect-DT smoke
# reads from here directly so it does not collide with the env-var test
# (which is for real detector predictions).
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CACHE_DIR = _REPO_ROOT / ".cache" / "coco-val2017"
_CACHED_GT = _CACHE_DIR / "instances_val2017.json"
_CACHED_PERFECT_DT = _CACHE_DIR / "perfect_dt.json"


def _resolve(env: str) -> Path | None:
    value = os.environ.get(env)
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_file() else None


@pytest.mark.parity
@pytest.mark.coco_val
def test_coco_val2017_bbox_parity() -> None:
    """Real detector predictions: bring your own JSON via env vars."""
    gt = _resolve(_GT_ENV)
    dt = _resolve(_DT_ENV)
    if gt is None or dt is None:
        pytest.skip(
            f"set {_GT_ENV} and {_DT_ENV} to existing files to enable; "
            "see docs/engineering/coco-val-parity.md"
        )
    ref = snapshot("pycocotools", gt, dt, "bbox")
    cand = snapshot("vernier", gt, dt, "bbox")
    assert_snapshots_equal(ref, cand)


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
    if not _CACHED_GT.is_file() or not _CACHED_PERFECT_DT.is_file():
        pytest.skip(
            "run ./tools/fetch-coco-val.sh to populate the cache; "
            "see docs/engineering/coco-val-parity.md"
        )
    ref = snapshot("pycocotools", _CACHED_GT, _CACHED_PERFECT_DT, "bbox")
    cand = snapshot("vernier", _CACHED_GT, _CACHED_PERFECT_DT, "bbox")
    assert_snapshots_equal(ref, cand)
