"""``BackgroundEvaluator`` accepts ``bytes | CocoDataset`` (ADR-0020).

Three behaviours covered:

1. **Numerical equivalence**: constructing from a `CocoDataset` produces
   the same `Summary.stats` as constructing from raw `gt_bytes`, on
   every IoU kernel. The cache changes how derivations are produced —
   never what they are.
2. **Cache reuse**: the same `CocoDataset` handle, fed into two
   consecutive `BackgroundEvaluator` cycles, populates its
   `boundary_cache_len` once and the second cycle observes the
   pre-populated count. The cache is what makes the per-epoch GT-band
   cost go from O(epochs) to O(1) (ADR-0020 §"Consequences").
3. **Type rejection**: passing anything other than `bytes` or
   `CocoDataset` to the constructor raises `TypeError` from the FFI.

The `Evaluator.background(gt)` factory shares the same code path under
the hood; we test both surfaces so a future refactor that splits them
doesn't silently regress.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

from vernier.instance import (
    BackgroundEvaluator,
    Bbox,
    Boundary,
    CocoDataset,
    Evaluator,
    IouKind,
    Keypoints,
    Segm,
)

IouType = Literal["bbox", "segm", "boundary", "keypoints"]

_KERNEL_FIXTURES: list[tuple[str, IouKind]] = [
    ("perfect_match", Bbox()),
    ("perfect_match_segm", Segm()),
    ("perfect_match_segm", Boundary()),
    ("keypoints_perfect_match", Keypoints()),
]


def _iou_type_of(iou: IouKind) -> IouType:
    match iou:
        case Bbox():
            return "bbox"
        case Segm():
            return "segm"
        case Boundary():
            return "boundary"
        case Keypoints():
            return "keypoints"
        case _:
            raise AssertionError(f"unknown iou {iou!r}")


@pytest.mark.parity
@pytest.mark.parametrize(("fixture", "iou"), _KERNEL_FIXTURES)
def test_background_with_dataset_matches_bytes(
    fixture: str, iou: IouKind, fixtures_dir: Path
) -> None:
    """Constructing from `CocoDataset` produces the same Summary as bytes."""
    iou_type = _iou_type_of(iou)
    gt_bytes = (fixtures_dir / fixture / "gt.json").read_bytes()
    dt_bytes = (fixtures_dir / fixture / "dt.json").read_bytes()

    bg_bytes = BackgroundEvaluator(gt_bytes, iou_type=iou_type, parity_mode="strict")
    bg_bytes.submit(dt_bytes)
    summary_bytes = bg_bytes.finalize()

    dataset = CocoDataset.from_json(gt_bytes)
    bg_dataset = BackgroundEvaluator(dataset, iou_type=iou_type, parity_mode="strict")
    bg_dataset.submit(dt_bytes)
    summary_dataset = bg_dataset.finalize()

    for i, (a, b) in enumerate(zip(summary_bytes.stats, summary_dataset.stats, strict=True)):
        assert b == pytest.approx(a, rel=0, abs=1e-12), (
            f"stat[{i}] diverged between bytes and CocoDataset paths: "
            f"bytes={a!r} dataset={b!r} (fixture={fixture}, iou={iou_type})"
        )


@pytest.mark.parity
@pytest.mark.parametrize(("fixture", "iou"), _KERNEL_FIXTURES)
def test_evaluator_background_factory_matches_direct(
    fixture: str, iou: IouKind, fixtures_dir: Path
) -> None:
    """`Evaluator.background(gt)` produces the same Summary as the
    direct `BackgroundEvaluator(...)` constructor with the same kernel
    config."""
    iou_type = _iou_type_of(iou)
    gt_bytes = (fixtures_dir / fixture / "gt.json").read_bytes()
    dt_bytes = (fixtures_dir / fixture / "dt.json").read_bytes()
    dataset = CocoDataset.from_json(gt_bytes)

    bg_direct = BackgroundEvaluator(dataset, iou_type=iou_type, parity_mode="strict")
    bg_direct.submit(dt_bytes)
    summary_direct = bg_direct.finalize()

    evaluator = Evaluator(iou=iou, parity_mode="strict")
    bg_factory = evaluator.background(dataset)
    bg_factory.submit(dt_bytes)
    summary_factory = bg_factory.finalize()

    for i, (a, b) in enumerate(zip(summary_direct.stats, summary_factory.stats, strict=True)):
        assert b == pytest.approx(a, rel=0, abs=1e-12), (
            f"stat[{i}] diverged: direct={a!r} factory={b!r} (fixture={fixture}, iou={iou_type})"
        )


@pytest.mark.parity
def test_boundary_cache_populated_by_background_dataset_path(fixtures_dir: Path) -> None:
    """Cache reuse is the headline benefit of ADR-0020. Building a
    `CocoDataset` and running a `BackgroundEvaluator` cycle with
    boundary IoU populates `boundary_cache_len`; running a second
    cycle leaves the count at the same level (no re-derivation).
    """
    gt_bytes = (fixtures_dir / "perfect_match_segm" / "gt.json").read_bytes()
    dt_bytes = (fixtures_dir / "perfect_match_segm" / "dt.json").read_bytes()
    dataset = CocoDataset.from_json(gt_bytes)

    assert dataset.boundary_cache_len == 0, (
        f"fresh CocoDataset should have an empty boundary cache; got {dataset.boundary_cache_len}"
    )

    bg1 = BackgroundEvaluator(dataset, iou_type="boundary", parity_mode="strict")
    bg1.submit(dt_bytes)
    bg1.finalize()
    populated = dataset.boundary_cache_len
    assert populated > 0, (
        "BackgroundEvaluator(CocoDataset, iou_type='boundary') should populate the "
        f"shared boundary cache; got {populated}"
    )

    # A second cycle must reuse the populated cache: the count never
    # exceeds the GT annotation count, and every entry from the first
    # cycle is reused (the cache is keyed by ann_id).
    bg2 = BackgroundEvaluator(dataset, iou_type="boundary", parity_mode="strict")
    bg2.submit(dt_bytes)
    bg2.finalize()
    assert dataset.boundary_cache_len == populated, (
        "second BackgroundEvaluator cycle should not re-derive any GT band; "
        f"cache went {populated} -> {dataset.boundary_cache_len}"
    )


@pytest.mark.parity
def test_segm_cache_populated_by_background_dataset_path(fixtures_dir: Path) -> None:
    """Same shape as the boundary cache test, on the segm path."""
    gt_bytes = (fixtures_dir / "perfect_match_segm" / "gt.json").read_bytes()
    dt_bytes = (fixtures_dir / "perfect_match_segm" / "dt.json").read_bytes()
    dataset = CocoDataset.from_json(gt_bytes)

    assert dataset.segm_cache_len == 0
    bg1 = BackgroundEvaluator(dataset, iou_type="segm", parity_mode="strict")
    bg1.submit(dt_bytes)
    bg1.finalize()
    populated = dataset.segm_cache_len
    assert populated > 0

    bg2 = BackgroundEvaluator(dataset, iou_type="segm", parity_mode="strict")
    bg2.submit(dt_bytes)
    bg2.finalize()
    assert dataset.segm_cache_len == populated


def test_background_rejects_invalid_gt_type() -> None:
    """Anything other than `bytes` or `CocoDataset` is a `TypeError`."""
    with pytest.raises(TypeError, match=r"bytes.*CocoDataset|CocoDataset.*bytes"):
        BackgroundEvaluator(123, iou_type="bbox")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"bytes.*CocoDataset|CocoDataset.*bytes"):
        BackgroundEvaluator("not bytes", iou_type="bbox")  # type: ignore[arg-type]
