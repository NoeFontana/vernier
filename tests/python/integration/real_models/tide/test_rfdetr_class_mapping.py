"""Pin rf-detr's class-id semantics: sparse COCO ids, not dense 0..79.

This module exists because the harness shipped the opposite reading
for its whole life, and nothing else in the suite could notice. The
TIDE cells assert structural coherence and determinism; the boundary
cell asserts vernier/oracle parity. All of those hold just as well on
systematically relabelled detections — both sides of a parity test see
the same wrong labels — so the only thing that ever pointed at the bug
was an absolute metric nobody was gating:
``boundary AP@[.5:.95] = 0.0001``, which
``docs/engineering/real-predictions-parity.md`` rationalised as "low
by design".

These tests need no weights and no inference: they check the *mapping*,
which is where the bug lived. They run in about a second on a host with
the ``real-models`` extra and the COCO val2017 cache.
"""

from __future__ import annotations

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


def test_rfdetr_class_ids_are_sparse_coco_category_ids(
    coco_classes: dict[int, str], coco_gt_dict: dict[str, Any]
) -> None:
    """``COCO_CLASSES`` is keyed by category id, so the mapping is identity.

    The guard that matters is the last assertion: ids above 79 exist.
    A dense 0..79 reading cannot represent them, which is why the old
    code silently dropped every ``vase`` (86), ``clock`` (85),
    ``refrigerator`` (82), ``book`` (84) and six more — 10 of the 80
    COCO classes, 8.5% of all detections on val2017.
    """
    mapping = _coco_class_mapping(coco_gt_dict, coco_classes)

    gt_ids = {int(cat["id"]) for cat in coco_gt_dict["categories"]}
    assert set(mapping) == gt_ids, "mapping domain is not the GT category id set"
    assert all(k == v for k, v in mapping.items()), "mapping is not the identity"
    assert max(mapping) == 90, f"expected sparse ids up to 90, got max {max(mapping)}"
    assert len(mapping) == 80, f"expected 80 COCO classes, got {len(mapping)}"
    assert sum(1 for k in mapping if k > 79) == 10, (
        "expected exactly 10 category ids above a dense-80 range; these are the "
        "ids a dense reading drops on the floor"
    )


def test_dense_reading_of_coco_classes_is_rejected(
    coco_classes: dict[int, str], coco_gt_dict: dict[str, Any]
) -> None:
    """The exact bug, pinned: ``enumerate(COCO_CLASSES.values())``.

    This is what the harness did for its whole life. It type-checks,
    and every id it maps lands on a real COCO category — just the
    wrong one. Under the current signature it cannot be silent: the
    name check fails on the first entry whose dense index isn't also
    its category id.
    """
    dense = dict(enumerate(coco_classes.values()))
    with pytest.raises(RuntimeError, match="identity"):
        _coco_class_mapping(coco_gt_dict, dense)


def test_class_mapping_rejects_a_renumbered_gt(coco_classes: dict[int, str]) -> None:
    """A GT whose ids don't line up with rfdetr's must fail loudly.

    Identity is only correct because the checkpoint's class space *is*
    COCO's. A subset that renumbers its categories breaks that, and
    silently relabelling is the failure mode this whole module is
    about.
    """
    renumbered = {"categories": [{"id": i, "name": n} for i, n in enumerate(coco_classes.values())]}
    with pytest.raises(RuntimeError, match="identity"):
        _coco_class_mapping(renumbered, coco_classes)


def test_unmappable_class_id_raises_instead_of_dropping_the_detection() -> None:
    """An id outside the mapping is a hard error, not a silent skip.

    The cache filename embeds a pinned version and is the integrity
    surface (see ``_rfdetr_predict`` module docstring), so writing a
    cache that quietly omits detections is worse than not writing one.
    """
    pytest.importorskip("numpy")
    import numpy as np

    class _Detections:
        xyxy = np.array([[0.0, 0.0, 10.0, 10.0]])
        confidence = np.array([0.9])
        class_id = np.array([999])
        mask = None

    with pytest.raises(RuntimeError, match="maps to no COCO category"):
        _detections_to_records(
            _Detections(),  # type: ignore[arg-type]
            image_id=1,
            class_mapping={1: 1},
            include_masks=False,
        )


def test_background_class_id_is_skipped_quietly() -> None:
    """Slot 0 is the checkpoint's no-object class and is expected.

    ``class_embed.out_features == 91`` covers ids 0..90; COCO uses
    1..90, so 0 is the only slot left for background. Skipping it is
    correct, and distinguishing it from a genuinely unknown id is the
    reason :data:`_BACKGROUND_CLASS_ID` is named rather than inlined.
    """
    import numpy as np

    class _Detections:
        xyxy = np.array([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 5.0, 5.0]])
        confidence = np.array([0.9, 0.8])
        class_id = np.array([_BACKGROUND_CLASS_ID, 1])
        mask = None

    records = _detections_to_records(
        _Detections(),  # type: ignore[arg-type]
        image_id=7,
        class_mapping={1: 1},
        include_masks=False,
    )
    assert [r["category_id"] for r in records] == [1]
