"""Memory-budget plumbing for `StreamingEvaluator`.

Two contracts pinned here:

1. **Hard budget**: when a projected post-insert total would exceed
   `memory_budget_bytes`, `update()` raises `OutOfBudgetError`. The
   exception carries `used_bytes`, `budget_bytes`, and a `breakdown`
   dict whose keys cover at least `{"cells_store", "scores",
   "match_flags"}`. Evaluator state is unchanged on the failed call.

2. **Soft warning**: the first `update()` whose post-insert total
   crosses 80% of the budget emits a single `MemoryBudgetWarning`.
   Subsequent updates that remain over the soft threshold do NOT
   re-warn — the warning is one-shot per evaluator.
"""

from __future__ import annotations

import json
import warnings

import pytest

from vernier._impl import StreamingEvaluator
from vernier.instance import MemoryBudgetWarning, OutOfBudgetError


def _make_gt(n_images: int) -> bytes:
    """Synthesize a minimal GT JSON payload with `n_images` images and
    one annotation per image."""
    images = [
        {"id": i, "width": 100, "height": 100, "file_name": f"img{i}.jpg"}
        for i in range(1, n_images + 1)
    ]
    annotations = [
        {
            "id": i,
            "image_id": i,
            "category_id": 1,
            "bbox": [10.0, 10.0, 20.0, 20.0],
            "area": 400.0,
            "iscrowd": 0,
        }
        for i in range(1, n_images + 1)
    ]
    categories = [{"id": 1, "name": "thing", "supercategory": "stuff"}]
    payload = {"images": images, "annotations": annotations, "categories": categories}
    return json.dumps(payload).encode("utf-8")


def _make_dt_batch(image_id: int) -> bytes:
    """One detection on one image — cheap, predictable cell footprint."""
    return json.dumps(
        [
            {
                "image_id": image_id,
                "category_id": 1,
                "score": 0.9,
                "bbox": [10.0, 10.0, 20.0, 20.0],
            }
        ]
    ).encode("utf-8")


@pytest.mark.parity
def test_out_of_budget_error_carries_structured_attributes() -> None:
    # Choose a budget that allows the first batch but not the next few.
    # A single 1-DT batch on this synthetic GT consumes ~900 bytes (one
    # cell per (k=1, a=4, i)). Budget of 2000 → first update fits; the
    # second update overflows.
    gt = _make_gt(n_images=8)
    ev = StreamingEvaluator(gt, memory_budget_bytes=2000)

    # First update should succeed (~900 bytes).
    ev.update(_make_dt_batch(1))
    assert ev.memory_used_bytes <= 2000

    def submit_until_overflow() -> None:
        # Submit one batch per remaining image; second insert should
        # already overflow the 2000-byte cap.
        for image_id in range(2, 9):
            ev.update(_make_dt_batch(image_id))

    with pytest.raises(OutOfBudgetError) as exc_info:
        submit_until_overflow()

    exc = exc_info.value
    assert exc.budget_bytes == 2000
    assert exc.used_bytes > exc.budget_bytes
    assert {"cells_store", "scores", "match_flags"} <= set(exc.breakdown), (
        f"breakdown missing keys: got {set(exc.breakdown)}"
    )


@pytest.mark.parity
def test_memory_budget_warning_fires_exactly_once() -> None:
    # Each single-DT batch on this synthetic GT consumes 816 bytes
    # (one cell per image, A=4 area ranges). Pick `budget=10000` so:
    #   - 9 batches  -> 7344 bytes  (under 80% of 10000 = 8000)
    #   - 10 batches -> 8160 bytes  (crosses the soft threshold; warn fires)
    #   - 11 batches -> 8976 bytes  (still over soft, under hard; no re-warn)
    #   - 12 batches -> 9792 bytes  (still under 10000; clean re-warn check).
    gt = _make_gt(n_images=20)
    ev = StreamingEvaluator(gt, memory_budget_bytes=10000)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", MemoryBudgetWarning)
        for image_id in range(1, 11):
            ev.update(_make_dt_batch(image_id))

    matches = [w for w in caught if issubclass(w.category, MemoryBudgetWarning)]
    assert len(matches) == 1, (
        f"expected exactly one MemoryBudgetWarning across the first 10 updates, "
        f"got {len(matches)}: {[str(w.message) for w in matches]}"
    )

    # Subsequent updates that keep us over 80% must NOT re-warn — the
    # soft-warn is one-shot per evaluator.
    with warnings.catch_warnings(record=True) as caught_again:
        warnings.simplefilter("always", MemoryBudgetWarning)
        ev.update(_make_dt_batch(11))
        ev.update(_make_dt_batch(12))

    matches_again = [w for w in caught_again if issubclass(w.category, MemoryBudgetWarning)]
    assert matches_again == [], (
        f"expected zero further warnings, got {len(matches_again)}: "
        f"{[str(w.message) for w in matches_again]}"
    )
