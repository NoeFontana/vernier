"""Pin rf-detr's class-id semantics: sparse COCO ids, not dense 0..79.

Why this module exists: see the `v2` warning box in
``docs/engineering/real-predictions-parity.md``. The short version is
that nothing else in the suite can catch a systematic relabelling —
parity and coherence gates hold when both sides see the same wrong
labels.

These tests need no weights and no inference: they check the *mapping*,
which is where the bug lived. They run in about a second on a host with
the ``real-models`` extra and the COCO val2017 cache.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from ._rfdetr_predict import _BACKGROUND_CLASS_ID, _coco_class_mapping, _detections_to_records

pytestmark = pytest.mark.real_models


@pytest.fixture(scope="module")
def coco_classes() -> dict[int, str]:
    pytest.importorskip(
        "rfdetr",
        reason=(
            "rf-detr class-mapping tests need the `real-models` extra: "
            "`uv sync --extra real-models`"
        ),
    )
    from rfdetr.assets.coco_classes import COCO_CLASSES

    return COCO_CLASSES


def _fake_detections(class_ids: list[int]) -> Any:
    """Minimal stand-in carrying only the fields the record builder reads.

    Returns ``Any`` rather than casting to ``supervision.Detections``:
    the namespace genuinely is not one, and a cast asserting otherwise
    would be a false claim to keep a checker quiet.
    """
    import numpy as np

    n = len(class_ids)
    return SimpleNamespace(
        xyxy=np.tile([0.0, 0.0, 10.0, 10.0], (n, 1)),
        confidence=np.full(n, 0.9),
        class_id=np.array(class_ids),
        mask=None,
    )


def test_rfdetr_class_ids_are_sparse_coco_category_ids(
    coco_classes: dict[int, str], coco_gt_dict: dict[str, Any]
) -> None:
    """``COCO_CLASSES`` is keyed by category id, so the mapping is identity.

    The guard that matters is the second assertion: ids above 79 exist.
    A dense 0..79 reading cannot represent them, which is why the old
    code silently dropped every ``vase`` (86), ``clock`` (85),
    ``refrigerator`` (82), ``book`` (84) and six more — 10 of the 80
    COCO classes, 8.5% of all detections on val2017.
    """
    mapping = _coco_class_mapping(coco_gt_dict, coco_classes)

    gt_ids = {int(cat["id"]) for cat in coco_gt_dict["categories"]}
    assert mapping == {i: i for i in gt_ids}, "mapping is not the identity over the GT ids"
    assert sum(1 for k in mapping if k > 79) == 10, (
        "expected exactly 10 category ids above a dense-80 range; these are the "
        "ids a dense reading drops on the floor"
    )


def test_a_mismatched_class_id_space_is_rejected(
    coco_classes: dict[int, str], coco_gt_dict: dict[str, Any]
) -> None:
    """Two spellings of one failure: the model table and the GT disagree.

    The first is the exact bug — ``enumerate(COCO_CLASSES.values())``,
    which re-keys the table to a dense index. The second is its mirror,
    a GT renumbered to 0..79. Both resolve every *name*, so the name
    join reports success; the identity check is what fails, which is
    the whole reason it sits on top of the shared helper.
    """
    dense = dict(enumerate(coco_classes.values()))
    with pytest.raises(RuntimeError, match="identity"):
        _coco_class_mapping(coco_gt_dict, dense)

    renumbered = {"categories": [{"id": i, "name": n} for i, n in enumerate(coco_classes.values())]}
    with pytest.raises(RuntimeError, match="identity"):
        _coco_class_mapping(renumbered, coco_classes)


def test_unmappable_class_id_raises_instead_of_dropping_the_detection() -> None:
    """An id outside the mapping is a hard error, not a silent skip."""
    with pytest.raises(RuntimeError, match="maps to no COCO category"):
        _detections_to_records(
            _fake_detections([999]),
            image_id=1,
            class_mapping={1: 1},
            include_masks=False,
        )


def test_background_class_id_is_skipped_quietly() -> None:
    """Slot 0 is the checkpoint's no-object class, so it is not an error."""
    records = _detections_to_records(
        _fake_detections([_BACKGROUND_CLASS_ID, 1]),
        image_id=7,
        class_mapping={1: 1},
        include_masks=False,
    )
    assert [r["category_id"] for r in records] == [1]
