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

The bbox fixture carries tied scores so the stable sort has a tie to
break; the segm / boundary / keypoints fixtures below carry the mask and
keypoint shapes the matrix route cannot express, and pin what a route
that carries *no* segmentation under ``iou_type='segm'`` means (quirk
**J2**).
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
    # Tied scores, in the same (image, category) cell as an earlier entry, so
    # the score sort has an actual tie to break. The sort is stable, so the
    # tie is broken by input position — which is also what J1 numbers by, so a
    # route that fed the sorter a different order would show up in `dtIds`.
    # 0.90 ties entry 0 (image 1, cat 1); 0.40 ties entry 1 (image 2, cat 1).
    {"image_id": 1, "category_id": 1, "bbox": [1.0, 1.0, 9.0, 9.0], "score": 0.90},
    {"image_id": 2, "category_id": 1, "bbox": [7.0, 7.0, 26.0, 26.0], "score": 0.40},
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
    return _core.evaluate_bbox_grid(GT_BYTES, dt, "strict", 100, True, retain_meta=True)


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
    highest-scoring detection here sits at input position 4, so a route
    that sorted before assigning would give it id 1 instead of 5.
    """
    cells = _all_area_cells(_normalize(_grid(_as_dict_list()).eval_imgs()))
    observed = sorted(i for cell in cells for i in cell["dtIds"])
    assert observed == list(range(1, len(DETECTIONS) + 1))

    # Image 2 / category 1 holds input positions 1, 4 and 6 -> ids 2, 5, 7.
    cell = next(c for c in cells if c["image_id"] == 2 and c["category_id"] == 1)
    assert sorted(cell["dtIds"]) == [2, 5, 7]


def test_supplied_ids_are_preserved_on_the_list_route() -> None:
    """A supplied ``id`` survives — vernier's documented J1 position."""
    anns = _as_dict_list()
    for offset, ann in enumerate(anns):
        ann["id"] = 1000 + offset
    cells = _all_area_cells(_normalize(_grid(anns).eval_imgs()))
    observed = sorted(i for cell in cells for i in cell["dtIds"])
    assert observed == [1000 + i for i in range(len(DETECTIONS))]


def test_iscrowd_is_ignored_on_the_list_route() -> None:
    """Quirks **E2**/**J4**: a detection is never a crowd, whatever it says.

    ``iscrowd`` is the one result-annotation field that is genuinely
    dropped: ``DetectionInput`` has nowhere to put it, so a detection
    carrying ``iscrowd=1`` must evaluate exactly as one carrying none.
    """
    poisoned = _as_dict_list()
    for ann in poisoned:
        ann["iscrowd"] = 1
    assert _normalize(_grid(poisoned).eval_imgs()) == _normalize(_grid(_as_dict_list()).eval_imgs())


def test_area_is_ignored_under_the_default_dt_area() -> None:
    """Quirk **J3**: under ``dt_area='bbox'`` the area comes from the box.

    A deliberately absurd ``area`` changes nothing, because the default
    derives the area rather than reading it — exactly as ``loadRes`` does.
    """
    poisoned = _as_dict_list()
    for ann in poisoned:
        ann["area"] = 1e9
    assert _normalize(_grid(poisoned).eval_imgs()) == _normalize(_grid(_as_dict_list()).eval_imgs())


def _supplied_area_grid(dt: Any) -> Any:
    return _core.evaluate_bbox_grid(
        GT_BYTES, dt, "strict", 100, True, dt_area="supplied", retain_meta=True
    )


def test_supplied_area_is_carried_on_the_list_route() -> None:
    """Quirk **J3**: under ``dt_area='supplied'`` the list route honours ``area``.

    ``area`` is therefore *carried*, not dropped. Dropping it would be
    invisible under the default and wrong here: the same payload as a
    results *file* buckets every detection into ``large``, and the list
    route has to do the same or the two routes are distinguishable.
    """
    dicts = _as_dict_list()
    for ann in dicts:
        ann["area"] = 1e9
    observed = _normalize(_supplied_area_grid(dicts).eval_imgs())
    assert observed == _normalize(_supplied_area_grid(json.dumps(dicts).encode()).eval_imgs())
    # ... and the supplied area is doing something, so the equality above
    # is not two routes agreeing on having ignored it.
    assert observed != _normalize(_grid(dicts).eval_imgs())


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


# --- a mixed list is pinpointed, whichever shape comes first -----------------


def _columnar_one() -> dict[str, Any]:
    return {
        "image_id": 1,
        "boxes": np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float64),
        "scores": np.array([0.9], dtype=np.float64),
        "labels": np.array([1], dtype=np.int64),
    }


def _result_one() -> dict[str, Any]:
    return {"image_id": 1, "category_id": 1, "bbox": [0.0, 0.0, 10.0, 10.0], "score": 0.9}


@pytest.mark.parametrize(
    "payload",
    [
        lambda: [_result_one(), _columnar_one()],
        lambda: [_columnar_one(), _result_one()],
    ],
    ids=["result-first", "columnar-first"],
)
def test_a_mixed_list_names_the_offending_index(payload: Any) -> None:
    """Quirk **J6**: entry 0 picks the *route*, and entry 1 is then located.

    Route selection reads the first entry (ADR-0057), so a heterogeneous
    list is a hard error either way. The error has to name the index in
    **both** directions — telling a caller that "detections" is missing a
    field is useless when "detections" is a 5000-entry list.
    """
    with pytest.raises(ValueError, match=r"detections\[1\]"):
        _grid(payload())


def test_a_single_dict_is_not_given_a_bogus_index() -> None:
    """The index is a locator, not decoration: a bare dict has no position."""
    with pytest.raises(ValueError, match=r"^detections: missing required field 'boxes'"):
        _grid({"image_id": 1, "scores": np.array([0.9], dtype=np.float64)})


# --- segm: the same equivalence, with masks ---------------------------------
#
# The matrix route is bbox-only by construction, so the segm equivalence is
# between the file route and the list route — in every `segmentation` shape a
# results file or an in-memory caller can hold. The bbox-only case, which is
# the one shape all three routes share, gets its own section below.

SEGM_GT: dict[str, Any] = {
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
            "segmentation": [[0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]],
        },
        {
            "id": 2,
            "image_id": 1,
            "category_id": 2,
            "bbox": [20, 20, 8, 8],
            "area": 64,
            "iscrowd": 0,
            "segmentation": [[20.0, 20.0, 28.0, 20.0, 28.0, 28.0, 20.0, 28.0]],
        },
        {
            "id": 3,
            "image_id": 2,
            "category_id": 1,
            "bbox": [5, 5, 30, 30],
            "area": 900,
            "iscrowd": 0,
            "segmentation": [[5.0, 5.0, 35.0, 5.0, 35.0, 35.0, 5.0, 35.0]],
        },
    ],
}
SEGM_GT_BYTES = json.dumps(SEGM_GT).encode()

# Same shape as DETECTIONS: unsorted by score, interleaved across images, and
# carrying a tie (the two 0.90 entries share image 1 / category 1).
SEGM_BOXES: list[dict[str, Any]] = [
    {"image_id": 1, "category_id": 1, "box": (0, 0, 10, 10), "score": 0.90},
    {"image_id": 2, "category_id": 1, "box": (6, 6, 28, 28), "score": 0.40},
    {"image_id": 1, "category_id": 2, "box": (21, 19, 8, 8), "score": 0.75},
    {"image_id": 2, "category_id": 1, "box": (0, 0, 4, 4), "score": 0.99},
    {"image_id": 1, "category_id": 1, "box": (1, 1, 9, 9), "score": 0.90},
]


def _rect_polygon(x: int, y: int, w: int, h: int) -> list[list[float]]:
    """The COCO polygon shape: one flat [x, y, ...] list, nested one level."""
    return [
        [
            float(x),
            float(y),
            float(x + w),
            float(y),
            float(x + w),
            float(y + h),
            float(x),
            float(y + h),
        ]
    ]


def _rect_mask(x: int, y: int, w: int, h: int) -> np.ndarray:
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[y : y + h, x : x + w] = 1
    return mask


def _uncompressed_counts(mask: np.ndarray) -> list[int]:
    """COCO column-major run lengths, first run always the zero run."""
    counts: list[int] = []
    run = 0
    previous = 0
    for value in mask.flatten(order="F"):
        current = 1 if value else 0
        if current == previous:
            run += 1
        else:
            counts.append(run)
            previous = current
            run = 1
    counts.append(run)
    return counts


def _segm_detections(shape: str) -> list[dict[str, Any]]:
    """The one logical mask set, in one of the accepted `segmentation` shapes.

    ``json`` is the pair of shapes a results *file* carries (and therefore
    the only two the file route can be handed); the rest are the in-memory
    shapes the list route additionally accepts.
    """
    out = []
    for det in SEGM_BOXES:
        x, y, w, h = det["box"]
        if shape == "polygons":
            segmentation: Any = _rect_polygon(x, y, w, h)
        elif shape == "counts_list":
            segmentation = {
                "counts": _uncompressed_counts(_rect_mask(x, y, w, h)),
                "size": [64, 64],
            }
        elif shape == "counts_str":
            segmentation = {
                "counts": _compressed_counts(_rect_mask(x, y, w, h)).decode("ascii"),
                "size": [64, 64],
            }
        elif shape == "counts_bytes":
            segmentation = {"counts": _compressed_counts(_rect_mask(x, y, w, h)), "size": [64, 64]}
        elif shape == "counts_uint32":
            segmentation = {
                "counts": np.array(_uncompressed_counts(_rect_mask(x, y, w, h)), dtype=np.uint32),
                "size": [64, 64],
            }
        elif shape == "bitmask":
            segmentation = _rect_mask(x, y, w, h)
        else:  # pragma: no cover - guards the parametrization
            raise AssertionError(shape)
        out.append(
            {
                "image_id": det["image_id"],
                "category_id": det["category_id"],
                "bbox": [float(x), float(y), float(w), float(h)],
                "score": det["score"],
                "segmentation": segmentation,
            }
        )
    return out


def _compressed_counts(mask: np.ndarray) -> bytes:
    """The COCO 6-bit string, via the pinned oracle (no encoder is public)."""
    from pycocotools import mask as mask_util

    encoded = mask_util.encode(np.asfortranarray(mask))
    counts = encoded["counts"]
    assert isinstance(counts, bytes)
    return counts


def _segm_grid(dt: Any, parity_mode: str = "strict") -> Any:
    return _core.evaluate_segm_grid(SEGM_GT_BYTES, dt, parity_mode, 100, True, retain_meta=True)


@pytest.mark.parametrize(
    "shape",
    ["polygons", "counts_list", "counts_str", "counts_bytes", "counts_uint32", "bitmask"],
)
def test_segm_list_route_matches_file_route_cell_for_cell(shape: str) -> None:
    """Every ``segmentation`` shape lands on the file route's numbers.

    The reference is the file route fed the *JSON* spelling of the same
    masks: polygons for the polygon shape, uncompressed counts for every
    RLE shape (a bitmask and a uint32 counts array are the same runs the
    file carries as a list of ints). A route that dropped, reordered or
    mis-decoded a mask moves a cell.
    """
    file_shape = "polygons" if shape == "polygons" else "counts_list"
    reference = _normalize(
        _segm_grid(json.dumps(_segm_detections(file_shape)).encode()).eval_imgs()
    )
    candidate = _normalize(_segm_grid(_segm_detections(shape)).eval_imgs())
    assert candidate == reference
    assert any(cell["dtIds"] for cell in candidate), "fixture evaluates nothing"


def test_segm_list_route_matches_file_route_summary() -> None:
    reference = (
        _segm_grid(json.dumps(_segm_detections("polygons")).encode())
        .accumulate([1, 10, 100])
        .summarize()
        .stats
    )
    candidate = _segm_grid(_segm_detections("polygons")).accumulate([1, 10, 100]).summarize().stats
    assert candidate == reference


def test_polygon_segmentation_is_accepted_on_the_list_route() -> None:
    """The reviewer's repro: a plain COCO polygon, not an RLE dict.

    ``DetectionInput.segmentation`` has always had a ``Polygons`` variant
    and ``from_json_bytes`` has always accepted it; before this test the
    list route raised ``TypeError`` on the same payload, which made the
    ADR-0057 equivalence claim false.
    """
    dt = [
        {
            "image_id": 1,
            "category_id": 1,
            "bbox": [0.0, 0.0, 10.0, 10.0],
            "score": 0.9,
            "segmentation": [[0.0, 0.0, 10.0, 0.0, 10.0, 10.0]],
        }
    ]
    reference = _normalize(_segm_grid(json.dumps(dt).encode()).eval_imgs())
    assert _normalize(_segm_grid(dt).eval_imgs()) == reference


def test_boundary_list_route_matches_file_route_cell_for_cell() -> None:
    """The boundary kernel reads the same masks through the same route."""

    def grid(dt: Any) -> Any:
        return _core.evaluate_boundary_grid(
            SEGM_GT_BYTES, dt, "strict", 100, True, 0.02, retain_meta=True
        )

    reference = _normalize(grid(json.dumps(_segm_detections("polygons")).encode()).eval_imgs())
    assert _normalize(grid(_segm_detections("polygons")).eval_imgs()) == reference


def test_bad_segmentation_error_names_the_segmentation_field() -> None:
    """Not ``rles`` — that is the other route's field name."""
    dt = _segm_detections("polygons")
    dt[1]["segmentation"] = 17
    with pytest.raises(TypeError, match=r"detections\[1\]\.segmentation"):
        _segm_grid(dt)


def test_polygons_are_still_refused_on_the_columnar_rles_field() -> None:
    """ADR-0030 is unchanged: ``rles`` is an array surface, not a file one."""
    columnar = {
        "image_id": 1,
        "boxes": np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float64),
        "scores": np.array([0.9], dtype=np.float64),
        "labels": np.array([1], dtype=np.int64),
        "rles": [[[0.0, 0.0, 10.0, 0.0, 10.0, 10.0]]],
    }
    with pytest.raises(TypeError, match=r"detections\.rles\[0\]"):
        _segm_grid(columnar)


@pytest.mark.parametrize("wrap", [bytearray, memoryview], ids=["bytearray", "memoryview"])
def test_buffer_counts_are_refused_instead_of_read_as_run_lengths(wrap: Any) -> None:
    """A wrong answer would be silent, so this one has to be an error.

    ``bytearray`` and ``memoryview`` over the *compressed* 6-bit string
    satisfy the sequence protocol and yield one ``int`` per byte, so the
    "uncompressed counts as a list of ints" branch would happily read
    each character as a run length and decode a completely different
    mask — with no exception anywhere. The two readings are
    indistinguishable from the object, so neither is guessed.
    """
    dt = _segm_detections("counts_bytes")
    dt[0]["segmentation"]["counts"] = wrap(dt[0]["segmentation"]["counts"])
    with pytest.raises(TypeError, match=r"detections\[0\]\.segmentation\.counts"):
        _segm_grid(dt)


def test_a_bad_segmentation_is_refused_on_the_bbox_grid_too() -> None:
    """``segmentation`` is read under ``iou_type='bbox'`` on purpose.

    The parsed mask is dead for *scoring* there — ``dt_area='mask'`` is
    refused outside the segm / boundary grids, and nothing else on the
    bbox path reads a detection's segmentation. It is **not** dead for
    *validation*: a results file whose ``segmentation`` is structurally
    wrong fails to load on the bbox grid, so skipping the read on the
    list route would make it accept payloads the file route rejects.
    That is precisely the route divergence ADR-0057 exists to avoid, so
    the read is not gated on ``iou_type``.
    """
    dicts = _as_dict_list()
    dicts[1]["segmentation"] = {"counts": [1, 2, 3], "size": [64]}
    with pytest.raises((TypeError, ValueError)):
        _grid(json.dumps(dicts).encode())
    with pytest.raises((TypeError, ValueError), match=r"detections\[1\]\.segmentation"):
        _grid(dicts)


# --- a route with no segmentation under segm means J2 -----------------------


def _bbox_only_under_segm() -> tuple[bytes, list[dict[str, Any]], np.ndarray]:
    dicts = [
        {
            "image_id": det["image_id"],
            "category_id": det["category_id"],
            "bbox": [float(v) for v in det["box"]],
            "score": det["score"],
        }
        for det in SEGM_BOXES
    ]
    matrix = np.array(
        [
            [
                float(d["image_id"]),
                *d["bbox"],
                d["score"],
                float(d["category_id"]),
            ]
            for d in dicts
        ],
        dtype=np.float64,
    )
    return json.dumps(dicts).encode(), dicts, matrix


def test_bbox_only_under_segm_is_j2_on_every_route() -> None:
    """**J2** (``strict``): a DT with no ``segmentation`` gets the bbox rectangle.

    A bbox-only results *file* has always evaluated this way under
    ``iou_type='segm'`` — pycocotools synthesizes
    ``[[x1,y1, x1,y2, x2,y2, x2,y1]]`` at ``coco.py:341`` and vernier
    reproduces it bit-for-bit. An ``(N, 7)`` matrix carries exactly what
    that file carries, and so does a result dict without the field, so
    all three land on the same numbers rather than on a route-specific
    guard. This is the deliberate reading of ADR-0057, not an accident.
    """
    as_bytes, as_dicts, as_matrix = _bbox_only_under_segm()
    reference = _normalize(_segm_grid(as_bytes).eval_imgs())
    assert any(cell["dtIds"] for cell in reference), "fixture evaluates nothing"
    assert _normalize(_segm_grid(as_dicts).eval_imgs()) == reference
    assert _normalize(_segm_grid(as_matrix).eval_imgs()) == reference


@pytest.mark.parametrize("route", [0, 1, 2], ids=["file", "list", "matrix"])
def test_bbox_only_under_segm_is_refused_in_corrected_mode(route: int) -> None:
    """**J2** (``corrected``): the same three routes refuse, identically.

    The refusal comes from core, which names the detection and its image
    — so the diagnostic does not depend on which route delivered it.
    """
    dt = _bbox_only_under_segm()[route]
    with pytest.raises(ValueError, match=r"has no `segmentation` field"):
        _segm_grid(dt, "corrected").accumulate([1, 10, 100])


def test_matrix_under_keypoints_is_refused() -> None:
    """A matrix carries no keypoints, and core says so rather than guessing."""
    _, _, as_matrix = _bbox_only_under_segm()
    with pytest.raises(ValueError, match=r"keypoints"):
        _core.evaluate_keypoints_grid(SEGM_GT_BYTES, as_matrix, "strict", 20, True, {})


# --- keypoints ---------------------------------------------------------------

KP_COORDS = [[float(4 * i), float(2 * i), 2.0] for i in range(17)]
KP_FLAT = [v for triplet in KP_COORDS for v in triplet]

KP_GT_BYTES = json.dumps(
    {
        "images": [{"id": 1, "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "person"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 40, 80],
                "area": 3200,
                "iscrowd": 0,
                "num_keypoints": 17,
                "keypoints": KP_FLAT,
            }
        ],
    }
).encode()

KP_DETECTIONS: list[dict[str, Any]] = [
    {
        "image_id": 1,
        "category_id": 1,
        "bbox": [0.0, 0.0, 40.0, 80.0],
        "score": 0.60,
        "keypoints": KP_FLAT,
    },
    {
        "image_id": 1,
        "category_id": 1,
        "bbox": [1.0, 1.0, 40.0, 80.0],
        "score": 0.99,
        "keypoints": [v + 1.0 for v in KP_FLAT],
        "num_keypoints": 17,
    },
]


def test_keypoints_list_route_matches_file_route_cell_for_cell() -> None:
    """``keypoints`` and ``num_keypoints`` survive the list route intact."""

    def grid(dt: Any) -> Any:
        return _core.evaluate_keypoints_grid(
            KP_GT_BYTES, dt, "strict", 20, True, {}, retain_meta=True
        )

    reference = _normalize(grid(json.dumps(KP_DETECTIONS).encode()).eval_imgs())
    assert any(cell["dtIds"] for cell in reference), "fixture evaluates nothing"
    assert _normalize(grid([dict(d) for d in KP_DETECTIONS]).eval_imgs()) == reference
    assert (
        grid([dict(d) for d in KP_DETECTIONS])
        .accumulate([20])
        .summarize([20], plan="keypoints")
        .stats
        == grid(json.dumps(KP_DETECTIONS).encode())
        .accumulate([20])
        .summarize([20], plan="keypoints")
        .stats
    )


# --- documented route asymmetries -------------------------------------------


def test_matrix_route_cannot_express_an_id_and_gets_the_j1_assignment() -> None:
    """The matrix has no id column, so every row takes its J1 position id.

    This is the one field the list route preserves and the matrix route
    cannot carry (ADR-0057 "Consequences"). It is not a divergence: a
    results *file* without ``id`` fields behaves the same way.
    """
    cells = _all_area_cells(_normalize(_grid(_as_matrix()).eval_imgs()))
    observed = sorted(i for cell in cells for i in cell["dtIds"])
    assert observed == list(range(1, len(DETECTIONS) + 1))

    # And a list carrying explicit ids is *not* renumbered — the asymmetry.
    anns = _as_dict_list()
    for offset, ann in enumerate(anns):
        ann["id"] = 500 + offset
    list_cells = _all_area_cells(_normalize(_grid(anns).eval_imgs()))
    assert sorted(i for cell in list_cells for i in cell["dtIds"]) == [
        500 + i for i in range(len(DETECTIONS))
    ]


def test_matrix_route_honours_cast_inputs() -> None:
    """``cast_inputs=True`` promotes the matrix, exactly as it does ``boxes``.

    Default-off is the ADR-0004 boundary; the opt-in is the documented
    way to ask for the copy, and it would be arbitrary for the one route
    that takes a single array to be the one route that ignores the flag.
    """
    # Typed `Any`: `DetectionsInput` spells the float64 requirement, and
    # this test is about what happens when a caller violates it anyway.
    as_f32: Any = _as_matrix().astype(np.float32)
    with pytest.raises(TypeError, match="float64"):
        _core.evaluate_bbox_grid(GT_BYTES, as_f32, "strict", 100, True)

    with pytest.warns(UserWarning, match="cast_inputs=True"):
        cast = _core.evaluate_bbox_grid(GT_BYTES, as_f32, "strict", 100, True, False, True)
    reference = _grid(_as_bytes()).accumulate([1, 10, 100]).summarize().stats
    assert cast.accumulate([1, 10, 100]).summarize().stats == reference


def test_non_contiguous_error_names_the_torch_fix_too() -> None:
    """Not every caller of this route holds a numpy array."""
    strided = _as_matrix()[::2]
    with pytest.raises(TypeError, match=r"contiguous\(\)"):
        _grid(strided)
