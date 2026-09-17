"""The three detection ingest routes are indistinguishable downstream.

vernier accepts detections as COCO result JSON bytes (the *file* route),
as a list of per-annotation result dicts (the *list* route, what
pycocotools- and TorchMetrics-style callers already hold), and as an
``(N, 7)`` float64 matrix (the *matrix* route). All three converge on one
``Vec<DetectionInput>`` before anything evaluates, so "nothing downstream
distinguishes them" is a property of the code — but a property nobody
checks is a property that rots. These tests check it.

The assertion is on ``EvalGrid.eval_imgs()``, not on the summary. The
summary would pass even if ids were permuted; ``eval_imgs`` carries
``dtIds``, ``dtScores``, ``dtMatches`` and ``dtIgnore`` per cell, so it
pins the *columns* — including the ``loadRes`` id assignment (quirk
**J1**), which is observable precisely because ``gtm[tind, m] = d['id']``
writes detection ids into the match arrays.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from vernier import _core

# --- shared fixture: one logical detection set, expressed three ways ---------

GT: dict[str, Any] = {
    "images": [
        {"id": 1, "width": 64, "height": 64},
        {"id": 2, "width": 64, "height": 64},
    ],
    "categories": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}],
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
            "image_id": 1,
            "category_id": 2,
            "bbox": [20, 20, 8, 8],
            "area": 64,
            "iscrowd": 0,
        },
        {
            "id": 3,
            "image_id": 2,
            "category_id": 1,
            "bbox": [5, 5, 30, 30],
            "area": 900,
            "iscrowd": 0,
        },
    ],
}

# Deliberately unsorted by score and interleaved across images so that any
# route that reorders — and therefore renumbers ids under J1 — is caught.
DETECTIONS: list[dict[str, Any]] = [
    {"image_id": 1, "category_id": 1, "bbox": [0.0, 0.0, 10.0, 10.0], "score": 0.90},
    {"image_id": 2, "category_id": 1, "bbox": [6.0, 6.0, 28.0, 28.0], "score": 0.40},
    {"image_id": 1, "category_id": 2, "bbox": [21.0, 19.0, 8.0, 8.0], "score": 0.75},
    {"image_id": 1, "category_id": 1, "bbox": [40.0, 40.0, 5.0, 5.0], "score": 0.55},
    {"image_id": 2, "category_id": 1, "bbox": [0.0, 0.0, 4.0, 4.0], "score": 0.99},
]

GT_BYTES = json.dumps(GT).encode()


def _as_bytes() -> bytes:
    """File route: exactly what a results JSON on disk would contain."""
    return json.dumps(DETECTIONS).encode()


def _as_dict_list() -> list[dict[str, Any]]:
    """List route: the per-annotation dicts, untouched."""
    return [dict(d) for d in DETECTIONS]


def _as_matrix() -> np.ndarray:
    """Matrix route: (N, 7) = image_id, x, y, w, h, score, category_id."""
    return np.array(
        [
            [
                float(d["image_id"]),
                d["bbox"][0],
                d["bbox"][1],
                d["bbox"][2],
                d["bbox"][3],
                d["score"],
                float(d["category_id"]),
            ]
            for d in DETECTIONS
        ],
        dtype=np.float64,
    )


def _grid(dt: Any) -> Any:
    return _core.evaluate_bbox_grid(GT_BYTES, dt, "strict", 100, True)


def _normalize(eval_imgs: list[Any]) -> list[dict[str, Any]]:
    """Render eval_imgs into plain comparable Python.

    Cells are keyed by (image_id, category_id, aRng, maxDet) and sorted,
    so the comparison does not depend on cell visiting order (ADR-0051)
    — only on the per-cell column contents.
    """
    out = []
    for cell in eval_imgs:
        if cell is None:
            continue
        rendered = {}
        for key, value in cell.items():
            rendered[key] = np.asarray(value).tolist() if isinstance(value, np.ndarray) else value
        out.append(rendered)
    out.sort(key=lambda c: (c["image_id"], c["category_id"], str(c["aRng"]), c["maxDet"]))
    return out


# --- the equivalence itself --------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [_as_dict_list, _as_matrix],
    ids=["list-route", "matrix-route"],
)
def test_route_matches_file_route_cell_for_cell(build: Any) -> None:
    """Every eval_imgs cell is identical to the file route's."""
    reference = _normalize(_grid(_as_bytes()).eval_imgs())
    candidate = _normalize(_grid(build()).eval_imgs())
    assert candidate == reference


@pytest.mark.parametrize(
    "build",
    [_as_dict_list, _as_matrix],
    ids=["list-route", "matrix-route"],
)
def test_route_matches_file_route_summary(build: Any) -> None:
    """The 12-stat summary agrees bit-for-bit."""
    reference = _grid(_as_bytes()).accumulate([1, 10, 100]).summarize().stats
    candidate = _grid(build()).accumulate([1, 10, 100]).summarize().stats
    assert candidate == reference


def _all_area_cells(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One (aRng, maxDet) slice: the widest area range at the widest maxDet.

    eval_imgs enumerates every (image, category, aRng, maxDet) cell, so
    the same detection appears once per area range. The id assertions
    below want each detection exactly once.
    """
    max_det = max(c["maxDet"] for c in cells)
    widest = max((c["aRng"] for c in cells), key=lambda r: r[1] - r[0])
    return [c for c in cells if c["maxDet"] == max_det and c["aRng"] == widest]


def test_detection_ids_are_assigned_by_input_position() -> None:
    """Quirk **J1**: ids are 1..N in *input* order, not score order.

    ``loadRes`` does ``ann['id'] = i + 1`` over the list as given. The
    highest-scoring detection here is the last one in input order, so a
    route that sorted before assigning would give it id 1 instead of 5.
    """
    cells = _all_area_cells(_normalize(_grid(_as_dict_list()).eval_imgs()))
    observed = sorted(i for cell in cells for i in cell["dtIds"])
    assert observed == [1, 2, 3, 4, 5]

    # Image 2 / category 1 holds input positions 1 and 4 -> ids 2 and 5.
    cell = next(c for c in cells if c["image_id"] == 2 and c["category_id"] == 1)
    assert sorted(cell["dtIds"]) == [2, 5]


def test_supplied_ids_are_preserved_on_the_list_route() -> None:
    """A supplied ``id`` survives — vernier's documented J1 position."""
    anns = _as_dict_list()
    for offset, ann in enumerate(anns):
        ann["id"] = 1000 + offset
    cells = _all_area_cells(_normalize(_grid(anns).eval_imgs()))
    observed = sorted(i for cell in cells for i in cell["dtIds"])
    assert observed == [1000, 1001, 1002, 1003, 1004]


def test_area_and_iscrowd_fields_are_ignored() -> None:
    """Quirks **J3** / **E2**+**J4**: area is derived, crowd is forced 0.

    A detection carrying a deliberately absurd ``area`` and ``iscrowd=1``
    must evaluate exactly as one carrying neither.
    """
    poisoned = _as_dict_list()
    for ann in poisoned:
        ann["area"] = 1e9
        ann["iscrowd"] = 1
    assert _normalize(_grid(poisoned).eval_imgs()) == _normalize(_grid(_as_dict_list()).eval_imgs())


# --- matrix route: validation is explicit, never silent ----------------------


def test_matrix_rejects_non_contiguous() -> None:
    strided = _as_matrix()[::2]
    assert not strided.flags["C_CONTIGUOUS"]
    with pytest.raises(TypeError, match="C-contiguous"):
        _grid(strided)


def test_matrix_rejects_float32() -> None:
    with pytest.raises((TypeError, ValueError), match=r"float64|f64|dtype"):
        _grid(_as_matrix().astype(np.float32))


def test_matrix_rejects_wrong_column_count() -> None:
    with pytest.raises(ValueError, match=r"\(N, 7\)"):
        _grid(np.ascontiguousarray(_as_matrix()[:, :6]))


def test_matrix_rejects_fractional_category_id() -> None:
    m = _as_matrix()
    m[2, 6] = 1.5
    with pytest.raises(ValueError, match=r"category_id.*not an integer"):
        _grid(m)


def test_matrix_rejects_fractional_image_id() -> None:
    m = _as_matrix()
    m[0, 0] = 1.25
    with pytest.raises(ValueError, match=r"image_id.*not an integer"):
        _grid(m)


def test_matrix_rejects_image_id_beyond_two_pow_53() -> None:
    m = _as_matrix()
    m[0, 0] = 2.0**60
    with pytest.raises(ValueError, match=r"2\^53"):
        _grid(m)


@pytest.mark.xfail(
    reason="A zero-row array reports non-C-contiguous strides on this branch. "
    "NumPy/torch flag semantics (a zero-element array is contiguous whatever "
    "its strides) are restored by commit 3ab5002 on fix/pycocotools-shim-parity, "
    "in dlpack::is_contiguous. This route needs no change of its own once that "
    "lands.",
    raises=TypeError,
    strict=True,
)
def test_matrix_accepts_an_empty_detection_set() -> None:
    empty = np.zeros((0, 7), dtype=np.float64)
    assert (
        _grid(empty).accumulate([1, 10, 100]).summarize().stats
        == _grid(b"[]").accumulate([1, 10, 100]).summarize().stats
    )


# --- the columnar Detections dict route is not re-routed ---------------------


def test_columnar_detections_dict_still_takes_the_array_route() -> None:
    """``boxes`` wins over ``bbox``: ADR-0030 payloads are untouched.

    The discriminator must not capture a columnar ``Detections`` dict
    just because some caller also put a ``category_id`` key on it.
    """
    columnar = {
        "image_id": 1,
        "boxes": np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float64),
        "scores": np.array([0.9], dtype=np.float64),
        "labels": np.array([1], dtype=np.int64),
        "category_id": 1,  # decoy
    }
    grid = _grid(columnar)
    cells = _normalize(grid.eval_imgs())
    assert any(cell["dtIds"] for cell in cells)
