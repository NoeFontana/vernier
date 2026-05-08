"""End-to-end boundary-IoU parity tests against the bowenc0221 oracle.

Bit-equality is required across every intermediate of the COCOeval state
machine (per-image evaluation dicts, the dense precision/recall/scores
arrays, and the 12-element summary). The tests run on the same segm
fixture corpus as ``tests/python/parity/test_parity.py`` so a regression
in shared segm logic surfaces here too — boundary inherits all of
``vernier-core``'s segm parity surface plus the boundary-band composition.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ..parity.test_parity import SEGM_FIXTURES
from .e2e_harness import BoundaryEvalSnapshot, assert_snapshots_equal, snapshot

FIXTURES = Path(__file__).parent.parent / "parity" / "fixtures"

# Each fixture re-augments every annotation with cv2-based boundary
# masks (see `conftest.py:_force_single_process_boundary_augment` for
# why this is single-threaded under our harness). The corpus has 11
# segm fixtures; running them all is multi-minute, so most are gated
# behind `slow`. A small smoke trio stays in the default PR run to
# catch a "boundary entirely broken" regression — see
# BOUNDARY_FAST_SMOKE below.

# Reuse the segm corpus verbatim. Boundary IoU is defined for the same
# segmentation inputs and any divergence that breaks segm parity will
# also break boundary parity, so a single source of truth keeps the two
# corpora from drifting apart.
BOUNDARY_FIXTURES: list[str] = list(SEGM_FIXTURES)

# Smoke trio kept on the PR-time path: a perfect-match baseline (whose
# oracle augmentation is already cached for sibling tests below), a
# zero-overlap counterpart that the sanity-gate also exercises, and one
# boundary-specific edge case (`boundary_area_segm`) that catches band
# composition regressions invisible to plain segm parity.
BOUNDARY_FAST_SMOKE: frozenset[str] = frozenset(
    {"perfect_match_segm", "zero_overlap_segm", "boundary_area_segm"}
)


def _fixture_paths(name: str) -> tuple[Path, Path]:
    base = FIXTURES / name
    return base / "gt.json", base / "dt.json"


@pytest.fixture(scope="module")
def perfect_match_oracle() -> BoundaryEvalSnapshot:
    # The oracle's `createIndex` augments every annotation with a
    # cv2-based boundary mask; that's the heaviest cost in the file.
    # Multiple tests below consume the same `perfect_match_segm` oracle
    # snapshot, so cache it once per module rather than re-augmenting.
    gt, dt = _fixture_paths("perfect_match_segm")
    return snapshot("oracle", gt, dt)


@pytest.mark.parity_boundary
@pytest.mark.parametrize(
    "fixture",
    [
        pytest.param(
            name,
            marks=[] if name in BOUNDARY_FAST_SMOKE else [pytest.mark.slow],
        )
        for name in BOUNDARY_FIXTURES
    ],
)
def test_e2e_parity_against_oracle(fixture: str) -> None:
    gt, dt = _fixture_paths(fixture)
    ref = snapshot("oracle", gt, dt)
    cand = snapshot("vernier", gt, dt)
    assert_snapshots_equal(ref, cand)


@pytest.mark.parity_boundary
def test_harness_catches_real_differences(
    perfect_match_oracle: BoundaryEvalSnapshot,
) -> None:
    # Sanity gate: the comparator must reject genuinely different fixtures
    # or every real parity bug slips through silently.
    other = snapshot("oracle", *_fixture_paths("zero_overlap_segm"))
    with pytest.raises(AssertionError):
        assert_snapshots_equal(perfect_match_oracle, other)


@pytest.mark.parity_boundary
def test_perfect_match_baseline_ap(
    perfect_match_oracle: BoundaryEvalSnapshot,
) -> None:
    # Defends against both snapshot paths returning all-zeros and the
    # parametrized parity test still trivially passing.
    assert perfect_match_oracle.stats[0] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.slow
@pytest.mark.parity_boundary
def test_lvis_dilation_ratio_parity() -> None:
    # ADR-0010 §A2: LVIS uses `dilation_ratio = 0.008`. Exercising a
    # non-default ratio confirms the parameter actually threads through
    # both implementations rather than being ignored on one side.
    gt, dt = _fixture_paths("perfect_match_segm")
    ref = snapshot("oracle", gt, dt, dilation_ratio=0.008)
    cand = snapshot("vernier", gt, dt, dilation_ratio=0.008)
    assert_snapshots_equal(ref, cand)
