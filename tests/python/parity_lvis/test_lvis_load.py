"""LVIS dataset loader smoke tests (PR-2 of the ADR-0026 rollout).

These exercise the public Python surface only — `Dataset.from_lvis_json`,
the federated accessors, and the `Frequency` enum. Cell-skip and
`dt_ignore` semantics land in PR-3 with their own parity harness; the
checks here pin the *shape* of the federated metadata as observed
through the FFI.

Quirk citations refer to the appendix of `docs/adr/0026-lvis-support.md`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from vernier import Frequency
from vernier.instance import CocoDataset

# A minimal valid LVIS GT: 2 images, 2 categories, 2 GTs. Image 1
# has cat 1 (and lists cat 2 in `neg`); image 2 has cat 2 (and flags
# itself non-exhaustive on cat 2). Reused across the positive-load
# tests; the negative tests build their own fixtures inline so the
# violated invariant is visible at the call site.
_LVIS_MIN_VALID: dict[str, object] = {
    "images": [
        {
            "id": 1,
            "width": 100,
            "height": 100,
            "neg_category_ids": [2],
            "not_exhaustive_category_ids": [],
        },
        {
            "id": 2,
            "width": 100,
            "height": 100,
            "neg_category_ids": [],
            "not_exhaustive_category_ids": [2],
        },
    ],
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [0, 0, 10, 10],
            "area": 100,
            "iscrowd": 0,
        },
        {
            "id": 2,
            "image_id": 2,
            "category_id": 2,
            "bbox": [0, 0, 20, 20],
            "area": 400,
            "iscrowd": 0,
        },
    ],
    "categories": [
        {"id": 1, "name": "a", "frequency": "f"},
        {"id": 2, "name": "b", "frequency": "r"},
    ],
}


def _gt_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload).encode("utf-8")


@pytest.mark.parity_lvis
def test_from_lvis_json_loads_minimal_valid_dataset() -> None:
    ds = Dataset.from_lvis_json(_gt_bytes(_LVIS_MIN_VALID))
    assert ds.num_images == 2
    assert ds.num_annotations == 2
    assert ds.num_categories == 2
    assert ds.is_federated is True


@pytest.mark.parity_lvis
def test_federated_accessors_match_aa1_aa2_aa3_ab1() -> None:
    ds = Dataset.from_lvis_json(_gt_bytes(_LVIS_MIN_VALID))
    pos = ds.pos_category_ids
    neg = ds.neg_category_ids
    nel = ds.not_exhaustive_category_ids
    freq = ds.category_frequency

    assert pos is not None
    assert neg is not None
    assert nel is not None
    assert freq is not None

    # AA1: pos derived from GTs, not from the JSON.
    assert pos[1] == frozenset({1})
    assert pos[2] == frozenset({2})

    # AA2: neg read verbatim.
    assert neg[1] == frozenset({2})
    assert neg[2] == frozenset()

    # AA3: not_exhaustive read verbatim.
    assert nel[1] == frozenset()
    assert nel[2] == frozenset({2})

    # AB1: frequency tags returned as the LVIS single-letter strings
    # — and the Frequency enum equates with them by virtue of the
    # `(str, Enum)` MRO.
    assert freq[1] == "f"
    assert freq[2] == "r"
    assert Frequency(freq[1]) is Frequency.FREQUENT
    assert Frequency(freq[2]) is Frequency.RARE


@pytest.mark.parity_lvis
def test_from_json_leaves_federated_accessors_none() -> None:
    # The COCO loader on the same payload silently drops the LVIS
    # extras (AG1) and leaves federated metadata absent — the
    # orchestrator falls back to COCO semantics on every cell.
    ds = Dataset.from_json(_gt_bytes(_LVIS_MIN_VALID))
    assert ds.is_federated is False
    assert ds.pos_category_ids is None
    assert ds.neg_category_ids is None
    assert ds.not_exhaustive_category_ids is None
    assert ds.category_frequency is None


@pytest.mark.parity_lvis
def test_aa7_pos_intersect_neg_raises() -> None:
    # Quirk AA7 (corrected): a category with GT on an image cannot
    # also appear in that image's `neg_category_ids`. ADR-0026
    # appendix question 3.
    bad = {
        "images": [
            {
                "id": 1,
                "width": 10,
                "height": 10,
                "neg_category_ids": [1],
                "not_exhaustive_category_ids": [],
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 5, 5],
                "area": 25,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "a", "frequency": "f"}],
    }
    with pytest.raises(ValueError, match="lvis federated conflict"):
        Dataset.from_lvis_json(_gt_bytes(bad))


@pytest.mark.parity_lvis
def test_aa7_not_exhaustive_outside_pos_raises() -> None:
    # Quirk AA7 (corrected): not_exhaustive ⊆ pos (by spec).
    bad = {
        "images": [
            {
                "id": 1,
                "width": 10,
                "height": 10,
                "neg_category_ids": [],
                "not_exhaustive_category_ids": [2],
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 5, 5],
                "area": 25,
                "iscrowd": 0,
            }
        ],
        "categories": [
            {"id": 1, "name": "a", "frequency": "f"},
            {"id": 2, "name": "b", "frequency": "r"},
        ],
    }
    with pytest.raises(ValueError, match="not_exhaustive"):
        Dataset.from_lvis_json(_gt_bytes(bad))


@pytest.mark.parity_lvis
def test_ab6_missing_frequency_collects_all_offenders() -> None:
    # Quirk AB6 (corrected): missing-on-some-categories must surface
    # the full sorted list, not just the first miss. ADR-0026
    # appendix question 4.
    bad = {
        "images": [
            {
                "id": 1,
                "width": 10,
                "height": 10,
                "neg_category_ids": [],
                "not_exhaustive_category_ids": [],
            }
        ],
        "annotations": [],
        "categories": [
            {"id": 7, "name": "g"},
            {"id": 3, "name": "c"},
        ],
    }
    with pytest.raises(ValueError, match=r"missing `frequency` on 2 categories: \[3, 7\]"):
        Dataset.from_lvis_json(_gt_bytes(bad))


@pytest.mark.parity_lvis
def test_lvis_oracle_loads_same_fixture() -> None:
    # Cross-check against the vendored oracle: the same JSON byte
    # payload that vernier accepts must also load cleanly under
    # `lvis-api`. The oracle expects the JSON on disk; we write a
    # tempfile and invoke `LVIS(path)`.
    oracle_path = (Path(__file__).parent / "oracle" / "lvis_api").resolve()
    import sys

    sys.path.insert(0, str(oracle_path))
    try:
        from lvis import LVIS  # type: ignore[import-not-found]
    finally:
        sys.path.remove(str(oracle_path))

    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(_LVIS_MIN_VALID, f)
        path = f.name
    try:
        oracle = LVIS(path)
        assert len(oracle.get_img_ids()) == 2
        assert len(oracle.get_cat_ids()) == 2
    finally:
        Path(path).unlink()


@pytest.mark.parity_lvis
def test_frequency_enum_is_str_compatible() -> None:
    # Python 3.10-compatible (str, Enum) MRO: Frequency values *are*
    # strings, so the round-trip through JSON and the equality with
    # `category_frequency` returns are byte-identical.
    assert Frequency.RARE == "r"
    assert Frequency.COMMON == "c"
    assert Frequency.FREQUENT == "f"
    assert json.loads(json.dumps(Frequency.RARE.value)) == "r"
