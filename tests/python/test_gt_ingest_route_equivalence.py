"""The two ground-truth ingest routes are indistinguishable downstream.

vernier accepts ground truth as COCO GT JSON bytes (the *file* route,
``CocoDataset.from_json``) and as columnar arrays (the *array* route,
``CocoDataset.from_arrays`` — ADR-0060). Both converge on
``CocoDataset::from_parts`` before anything evaluates, so "nothing
downstream distinguishes them" is a property of the code — but a property
nobody checks is a property that rots. These tests check it.

The assertion is on ``EvalGrid.eval_imgs()``, not on the summary. A
summary would pass even if ground truth were permuted: it folds every
cell into twelve numbers. ``eval_imgs`` carries ``gtIds``, ``gtIgnore``,
``gtMatches``, ``dtMatches`` and ``dtIgnore`` *per cell*, so it pins the
columns — including the GT id assignment (supplied, never assigned: the
mirror image of quirk **J1**) and the ascending-``_ignore`` GT sort
(quirk **A4**), both of which are observable only because ``gtIds`` and
``dtMatches`` write ground-truth ids into the output.

``test_negative_control_measures_the_fixture_s_power`` measures what this
fixture can actually catch, rather than asserting a number nobody
checked. It builds one deliberately-wrong dataset per bug class this
route can have and counts how many of the eleven per-cell columns move.
The headline result is in that test's docstring, and one row of it
governs the shape of this whole module: **a route that dropped the
``ignore`` column is invisible under ``strict`` — 0 of 11** — because
quirk **D1** overwrites the field there anyway. Every equivalence
assertion below is therefore parametrized over both parity modes.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from vernier import _core

# --- shared fixture: one logical GT document, expressed two ways -------------

# Three images, ten annotations, two categories. Deliberately:
#
#   * ids are non-sequential and *not* in ascending order, so a route that
#     assigned ids by position (what `loadRes` does to detections under
#     J1) would be caught in `gtIds`;
#   * annotations are interleaved across images and categories, so a
#     route that grouped or sorted them differently would be caught;
#   * `iscrowd=1` appears on an annotation that is *not* last, so the
#     ascending-`_ignore` sort (A4) has something to move;
#   * an explicit `ignore` field appears on an annotation whose `iscrowd`
#     disagrees with it, so D1's strict-vs-corrected split is live;
#   * areas straddle the 32**2 and 96**2 bucket edges, and at least one
#     `area` disagrees with `w * h`, so a route that *derived* GT area
#     instead of reading it verbatim would re-bucket that annotation.
GT: dict[str, Any] = {
    "images": [
        {"id": 7, "width": 128, "height": 128, "file_name": "seven.jpg"},
        {"id": 3, "width": 200, "height": 150, "file_name": "three.jpg"},
        {"id": 9, "width": 128, "height": 128},
    ],
    "categories": [
        {"id": 2, "name": "cat", "supercategory": "animal"},
        {"id": 5, "name": "dog"},
    ],
    "annotations": [
        # image 7 / cat 2 — a crowd GT placed FIRST in input order, in a
        # cell that also holds non-crowd GTs. Without it no cell mixes
        # ignore with non-ignore under `strict`, and the A4 sort has
        # nothing to move; with it, A4 must carry this entry to the tail.
        {
            "id": 399,
            "image_id": 7,
            "category_id": 2,
            "bbox": [0, 0, 100, 100],
            "area": 9000.0,
            "iscrowd": 1,
        },
        # image 7 — small, exact area
        {
            "id": 401,
            "image_id": 7,
            "category_id": 2,
            "bbox": [0, 0, 20, 20],
            "area": 400.0,
            "iscrowd": 0,
        },
        # image 3 — crowd, and NOT last in input order, so A4 must move it
        {
            "id": 88,
            "image_id": 3,
            "category_id": 2,
            "bbox": [10, 10, 60, 60],
            "area": 3600.0,
            "iscrowd": 1,
        },
        # image 7 — `area` deliberately disagrees with w*h *across a
        # bucket edge*: the box is 50x50 = 2500 (medium) but the mask's
        # area is 500 (small, under 32**2). A polygon GT's area is the
        # mask's, not the box's, so a route that derived it the way
        # quirk J3 derives a *detection's* would re-bucket this one —
        # and, measurably, nothing else in this fixture would catch that.
        {
            "id": 402,
            "image_id": 7,
            "category_id": 5,
            "bbox": [30, 30, 50, 50],
            "area": 500.0,
            "iscrowd": 0,
        },
        # image 9 — large
        {
            "id": 12,
            "image_id": 9,
            "category_id": 2,
            "bbox": [5, 5, 110, 110],
            "area": 12100.0,
            "iscrowd": 0,
        },
        # image 3 — explicit ignore=1 with iscrowd=0: D1 strict discards
        # the field (ignore := iscrowd = False), corrected honours it.
        {
            "id": 77,
            "image_id": 3,
            "category_id": 5,
            "bbox": [100, 20, 40, 40],
            "area": 1600.0,
            "iscrowd": 0,
            "ignore": 1,
        },
        # image 7 — second cat-2 GT on the same image, so a cell has two
        # ground truths and their relative order is observable.
        {
            "id": 400,
            "image_id": 7,
            "category_id": 2,
            "bbox": [60, 60, 25, 25],
            "area": 625.0,
            "iscrowd": 0,
        },
        # image 9 — crowd again, this time last
        {
            "id": 13,
            "image_id": 9,
            "category_id": 5,
            "bbox": [0, 0, 128, 128],
            "area": 16384.0,
            "iscrowd": 1,
        },
        # image 3 — explicit ignore=0 with iscrowd=1: the other side of
        # D1. Strict sets ignore := True; corrected honours the 0.
        {
            "id": 89,
            "image_id": 3,
            "category_id": 2,
            "bbox": [150, 100, 30, 30],
            "area": 900.0,
            "iscrowd": 1,
            "ignore": 0,
        },
        # image 7 — no `ignore` key at all, so the column must be able to
        # say "absent on this annotation and present on its neighbours".
        {
            "id": 403,
            "image_id": 7,
            "category_id": 5,
            "bbox": [90, 5, 15, 15],
            "area": 225.0,
            "iscrowd": 0,
        },
    ],
}

# Detections are held constant across every GT route: this module varies
# the GT ingest only. They are unsorted by score, interleaved across
# images, and carry tied scores inside one (image, category) cell.
DETECTIONS: list[dict[str, Any]] = [
    {"image_id": 7, "category_id": 2, "bbox": [0.0, 0.0, 20.0, 20.0], "score": 0.90},
    {"image_id": 3, "category_id": 2, "bbox": [12.0, 12.0, 58.0, 58.0], "score": 0.40},
    {"image_id": 9, "category_id": 2, "bbox": [6.0, 6.0, 108.0, 108.0], "score": 0.75},
    {"image_id": 7, "category_id": 5, "bbox": [31.0, 29.0, 50.0, 50.0], "score": 0.55},
    {"image_id": 3, "category_id": 5, "bbox": [101.0, 21.0, 39.0, 39.0], "score": 0.99},
    {"image_id": 7, "category_id": 2, "bbox": [61.0, 61.0, 24.0, 24.0], "score": 0.90},
    {"image_id": 9, "category_id": 5, "bbox": [2.0, 2.0, 120.0, 120.0], "score": 0.40},
    {"image_id": 7, "category_id": 5, "bbox": [89.0, 4.0, 16.0, 16.0], "score": 0.30},
    {"image_id": 3, "category_id": 2, "bbox": [151.0, 101.0, 29.0, 29.0], "score": 0.65},
]

DT_BYTES = json.dumps(DETECTIONS).encode()


def _from_arrays(images: Any, annotations: Any, categories: Any, **kw: Any) -> Any:
    """``CocoDataset.from_arrays`` with the argument types erased.

    The stub types the three sections as ``TypedDict``s, which is right
    for callers. Most tests here deliberately hand it payloads that are
    *not* well-typed — a column deleted, a dtype wrong, a sequence where
    an array belongs — because refusing those is the behaviour under
    test. Erasing here keeps that possible without weakening the shipped
    signature, the same way ``_grid(dt: Any)`` does on the detection side.
    """
    return _core.CocoDataset.from_arrays(images, annotations, categories, **kw)


def _gt_bytes(gt: dict[str, Any] | None = None) -> bytes:
    """File route: exactly what a GT JSON on disk would contain."""
    return json.dumps(gt if gt is not None else GT).encode()


def _image_columns(gt: dict[str, Any]) -> dict[str, Any]:
    imgs = gt["images"]
    return {
        "id": np.array([i["id"] for i in imgs], dtype=np.int64),
        "width": np.array([i["width"] for i in imgs], dtype=np.int64),
        "height": np.array([i["height"] for i in imgs], dtype=np.int64),
        "file_name": [i.get("file_name") for i in imgs],
    }


def _ann_columns(gt: dict[str, Any]) -> dict[str, Any]:
    """Annotation columns, with the ADR-0060 absent-is-negative encoding.

    ``ignore`` is int64 rather than uint8 precisely because this fixture
    mixes annotations that carry the field with annotations that do not,
    and ``-1`` is the only way a column can say "absent here".
    """
    anns = gt["annotations"]
    return {
        "id": np.array([a["id"] for a in anns], dtype=np.int64),
        "image_id": np.array([a["image_id"] for a in anns], dtype=np.int64),
        "category_id": np.array([a["category_id"] for a in anns], dtype=np.int64),
        "bbox": np.array([a["bbox"] for a in anns], dtype=np.float64),
        "area": np.array([a["area"] for a in anns], dtype=np.float64),
        "iscrowd": np.array([a["iscrowd"] for a in anns], dtype=np.uint8),
        "ignore": np.array([a.get("ignore", -1) for a in anns], dtype=np.int64),
    }


def _as_arrays(gt: dict[str, Any] | None = None) -> _core.CocoDataset:
    """Array route: the same document as columns, no JSON anywhere."""
    gt = gt if gt is not None else GT
    return _from_arrays(_image_columns(gt), _ann_columns(gt), gt["categories"])


def _as_json(gt: dict[str, Any] | None = None) -> _core.CocoDataset:
    return _core.CocoDataset.from_json(_gt_bytes(gt))


def _grid(gt: Any, parity_mode: str = "strict", dt: bytes = DT_BYTES) -> Any:
    """One bbox grid, from GT bytes or from a parsed-once handle.

    ``evaluate_bbox_grid`` takes GT JSON bytes; ``..._with_dataset`` takes
    the ADR-0020 handle, which is what both GT routes produce. Only bbox
    has a grid-taking ``_with_dataset`` entry point today, which is why
    the segm and keypoints comparisons below assert on summaries plus
    :attr:`CocoDataset.dataset_hash` rather than on cells.
    """
    if isinstance(gt, bytes):
        return _core.evaluate_bbox_grid(
            gt, dt, parity_mode=parity_mode, max_dets_per_image=100, use_cats=True, retain_meta=True
        )
    return _core.evaluate_bbox_grid(
        gt, dt, parity_mode=parity_mode, max_dets_per_image=100, use_cats=True, retain_meta=True
    )


#: Per-cell column names ``eval_imgs`` emits. Named here so the negative
#: control can report "k of CELL_COLUMNS diverged" against a number that
#: moves if the output shape does.
CELL_COLUMNS = (
    "aRng",
    "category_id",
    "dtIds",
    "dtIgnore",
    "dtMatches",
    "dtScores",
    "gtIds",
    "gtIgnore",
    "gtMatches",
    "image_id",
    "maxDet",
)


def _normalize(eval_imgs: list[Any]) -> list[dict[str, Any]]:
    """Render eval_imgs into plain comparable Python.

    Cells are sorted by (image_id, category_id, aRng, maxDet) so the
    comparison does not depend on cell visiting order (ADR-0051) — only
    on the per-cell column contents.
    """
    out = []
    for cell in eval_imgs:
        if cell is None:
            continue
        rendered = {
            key: (np.asarray(value).tolist() if isinstance(value, np.ndarray) else value)
            for key, value in cell.items()
        }
        out.append(rendered)
    out.sort(key=lambda c: (c["image_id"], c["category_id"], str(c["aRng"]), c["maxDet"]))
    return out


# --- the equivalence itself --------------------------------------------------


@pytest.mark.parametrize("parity_mode", ["strict", "corrected"])
def test_array_route_matches_file_route_cell_for_cell(parity_mode: str) -> None:
    """Every eval_imgs cell is identical to the file route's, in both modes.

    Both modes matter because D1's disposition differs between them: the
    fixture carries an explicit ``ignore`` that strict discards and
    corrected honours, so a route that dropped the field would pass under
    ``strict`` and fail under ``corrected``.
    """
    reference = _normalize(_grid(_gt_bytes(), parity_mode).eval_imgs())
    candidate = _normalize(_grid(_as_arrays(), parity_mode).eval_imgs())
    assert candidate == reference


@pytest.mark.parametrize("parity_mode", ["strict", "corrected"])
def test_array_route_matches_file_route_summary(parity_mode: str) -> None:
    """The 12-stat summary agrees bit-for-bit."""
    reference = _grid(_gt_bytes(), parity_mode).accumulate([1, 10, 100]).summarize().stats
    candidate = _grid(_as_arrays(), parity_mode).accumulate([1, 10, 100]).summarize().stats
    assert candidate == reference


def test_dataset_handle_agrees_with_raw_bytes() -> None:
    """The parsed-once handle (ADR-0020) built either way is equivalent."""
    reference = _normalize(_grid(_as_json()).eval_imgs())
    candidate = _normalize(_grid(_as_arrays()).eval_imgs())
    assert candidate == reference


def _all_area_cells(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One (aRng, maxDet) slice: the widest area range at the widest maxDet.

    ``eval_imgs`` enumerates every (image, category, aRng, maxDet) cell,
    and ``gtIgnore`` is *per cell*: pycocotools ORs the annotation's
    resolved ignore with "area outside this range"
    (``_ignore = gt['ignore'] or gt['area'] < aRng[0] or ...``). So an
    assertion about D1's resolution has to read the widest range, where
    the area term is always false and the flag is the annotation's own.
    """
    max_det = max(c["maxDet"] for c in cells)
    widest = max((c["aRng"] for c in cells), key=lambda r: r[1] - r[0])
    return [c for c in cells if c["maxDet"] == max_det and c["aRng"] == widest]


def _diverging_columns(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> set[str]:
    """Column names that differ in at least one cell between two renderings."""
    if len(left) != len(right):
        return set(CELL_COLUMNS)
    diverged: set[str] = set()
    for a, b in zip(left, right):
        for key in CELL_COLUMNS:
            if a.get(key) != b.get(key):
                diverged.add(key)
    return diverged


def _perturbations() -> dict[tuple[str, str], _core.CocoDataset]:
    """One deliberately-wrong dataset per realistic bug class of this route.

    Each is what the array route would produce if it got one thing wrong,
    built from the *same* logical document so the only difference is the
    bug.
    """
    out: dict[tuple[str, str], Any] = {}

    def add(name: str, mode: str, *, images: Any = None, anns: Any = None) -> None:
        out[(name, mode)] = _from_arrays(
            images if images is not None else _image_columns(GT),
            anns if anns is not None else _ann_columns(GT),
            GT["categories"],
        )

    # A route that reordered ground truth.
    add(
        "rotate-annotation-order",
        "strict",
        anns=_ann_columns({**GT, "annotations": GT["annotations"][1:] + GT["annotations"][:1]}),
    )
    # A route that assigned ids by position, the way loadRes does to
    # detections under J1, instead of reading the supplied ones.
    renumbered = _ann_columns(GT)
    renumbered["id"] = np.arange(1, len(GT["annotations"]) + 1, dtype=np.int64)
    add("renumber-gt-ids", "strict", anns=renumbered)
    # A route that dropped the `ignore` field.
    dropped = _ann_columns(GT)
    del dropped["ignore"]
    add("drop-ignore-column", "strict", anns=dropped)
    add("drop-ignore-column", "corrected", anns=dropped)
    # A route whose column could not say "absent", so it read absent as 0.
    zeroed = _ann_columns(GT)
    zeroed["ignore"] = np.where(zeroed["ignore"] < 0, 0, zeroed["ignore"])
    add("absent-ignore-read-as-zero", "strict", anns=zeroed)
    add("absent-ignore-read-as-zero", "corrected", anns=zeroed)
    # A route that dropped `iscrowd`.
    uncrowded = _ann_columns(GT)
    uncrowded["iscrowd"] = np.zeros_like(uncrowded["iscrowd"])
    add("zero-iscrowd-column", "strict", anns=uncrowded)
    # A route that *derived* GT area from the box, the way J3 derives a
    # detection's, instead of reading it verbatim.
    derived = _ann_columns(GT)
    derived["area"] = derived["bbox"][:, 2] * derived["bbox"][:, 3]
    add("derive-area-from-bbox", "strict", anns=derived)
    return out


#: Measured, not assumed: how many of the eleven per-cell ``eval_imgs``
#: columns move when the array route gets one thing wrong. Re-measure
#: with ``_diverging_columns`` if the fixture changes; do not adjust a
#: number to make a test pass.
MEASURED_NEGATIVE_CONTROL = {
    ("rotate-annotation-order", "strict"): {"dtMatches", "gtIds", "gtMatches"},
    ("renumber-gt-ids", "strict"): {"dtMatches", "gtIds"},
    ("drop-ignore-column", "strict"): set(),
    ("drop-ignore-column", "corrected"): {"dtIgnore", "gtIds", "gtIgnore", "gtMatches"},
    ("absent-ignore-read-as-zero", "strict"): set(),
    ("absent-ignore-read-as-zero", "corrected"): {
        "dtIgnore",
        "dtMatches",
        "gtIds",
        "gtIgnore",
        "gtMatches",
    },
    ("zero-iscrowd-column", "strict"): {
        "dtIgnore",
        "dtMatches",
        "gtIds",
        "gtIgnore",
        "gtMatches",
    },
    ("derive-area-from-bbox", "strict"): {"dtIgnore", "gtIds", "gtIgnore", "gtMatches"},
}


@pytest.mark.parametrize(("name", "mode"), sorted(MEASURED_NEGATIVE_CONTROL))
def test_negative_control_measures_the_fixture_s_power(name: str, mode: str) -> None:
    """How much of the output actually moves when the route is wrong.

    An equivalence test is only as good as its fixture's ability to fail,
    and a single number is a poor summary of that. This is the measured
    matrix: one deliberately-wrong dataset per bug class this route can
    have, against the count of ``eval_imgs`` columns that move.

    Measured on this fixture, out of eleven per-cell columns:

    ===========================  ==========  ==========
    perturbation                 strict      corrected
    ===========================  ==========  ==========
    zero the ``iscrowd`` column  **5**       --
    derive area from the bbox    **4**       --
    drop the ``ignore`` column   **0**       **4**
    absent ``ignore`` read as 0  **0**       **5**
    rotate annotation order      **3**       --
    renumber GT ids 1..N         **2**       --
    ===========================  ==========  ==========

    Two of those rows are the reason this test exists.

    **A route that dropped ``ignore`` entirely is invisible under
    ``strict`` — 0 of 11.** That is not a weak fixture; it is quirk
    **D1** working as specified: ``_prepare`` overwrites ``gt['ignore']``
    with ``iscrowd`` unconditionally, so under ``strict`` the field
    genuinely does not matter. An equivalence suite that ran only the
    default parity mode would therefore have given a dropped ``ignore``
    column a clean bill of health. Every equivalence assertion in this
    module is parametrized over both modes for exactly that reason.

    **The same is true of reading an absent ``ignore`` as 0** — the bug
    the negative-``ignore`` sentinel exists to make unrepresentable. It
    is silent under ``strict`` and moves four columns under
    ``corrected``.

    The four columns that never move under any perturbation are the cell
    keys (``aRng``, ``category_id``, ``image_id``, ``maxDet``), which
    address a cell rather than describe it; ``dtIds`` and ``dtScores``
    move only under perturbations that change matching, since the
    detections are held constant by this module by construction.
    """
    reference = _normalize(_grid(_gt_bytes(), mode).eval_imgs())
    perturbed = _normalize(_grid(_perturbations()[(name, mode)], mode).eval_imgs())
    diverged = _diverging_columns(reference, perturbed)
    assert diverged == MEASURED_NEGATIVE_CONTROL[(name, mode)], (
        f"{name}/{mode} moved {sorted(diverged)}; re-measure the table in this docstring "
        "rather than editing the expectation to match"
    )


def test_the_unperturbed_array_route_moves_nothing() -> None:
    """The control's control: without a bug, no column diverges at all."""
    for mode in ("strict", "corrected"):
        reference = _normalize(_grid(_gt_bytes(), mode).eval_imgs())
        assert (
            _diverging_columns(reference, _normalize(_grid(_as_arrays(), mode).eval_imgs()))
            == set()
        )


# --- the GT-specific quirks, one test each -----------------------------------


def test_gt_ids_are_supplied_never_assigned() -> None:
    """GT ids come from the payload — the mirror image of quirk J1.

    The fixture's ids (401, 88, 402, …) are non-sequential and unsorted,
    so a route that numbered them 1..N by position would be caught.
    """
    cells = _normalize(_grid(_as_arrays()).eval_imgs())
    seen = {gid for cell in cells for gid in cell["gtIds"]}
    assert seen == {a["id"] for a in GT["annotations"]}


def test_a4_sorts_gt_ascending_by_ignore_through_the_array_route() -> None:
    """Quirk **A4**: GTs are ordered non-ignore first, ignore last.

    Observable through ``gtIds`` paired with ``gtIgnore``: within a cell,
    every zero in ``gtIgnore`` precedes every one. The fixture puts a
    crowd GT (id 88) *before* a non-crowd one in input order, so the sort
    has real work to do.
    """
    cells = _normalize(_grid(_as_arrays()).eval_imgs())
    moved = 0
    for cell in cells:
        flags = cell["gtIgnore"]
        assert flags == sorted(flags), f"cell {cell['image_id']}/{cell['category_id']} unsorted"
        if flags and flags[0] != flags[-1]:
            moved += 1
    assert moved > 0, "fixture never exercises the A4 sort"


def _ignore_flags(grid: Any) -> dict[int, int]:
    """``{gt id: gtIgnore}`` over the widest area range, where the flag is
    the annotation's own resolved ignore and not the area-range term."""
    cells = _all_area_cells(_normalize(grid.eval_imgs()))
    return {gid: flag for cell in cells for gid, flag in zip(cell["gtIds"], cell["gtIgnore"])}


@pytest.mark.parametrize(
    ("parity_mode", "ann_id", "expected_ignore"),
    [
        # id 77: iscrowd=0, ignore=1. Strict discards the field and takes
        # iscrowd; corrected honours the explicit 1.
        ("strict", 77, 0),
        ("corrected", 77, 1),
        # id 89: iscrowd=1, ignore=0. Strict takes iscrowd; corrected
        # honours the explicit 0.
        ("strict", 89, 1),
        ("corrected", 89, 0),
    ],
)
def test_d1_ignore_overwrite_survives_the_array_route(
    parity_mode: str, ann_id: int, expected_ignore: int
) -> None:
    """Quirk **D1**, both dispositions, through the columnar ``ignore``.

    ``_prepare`` reads ``gt['ignore']`` and then unconditionally
    overwrites it with ``'iscrowd' in gt and gt['iscrowd']``. ``strict``
    replicates the overwrite; ``corrected`` honours the explicit field.
    Both must behave identically whether the field arrived as JSON or as
    an int64 column — which is only possible because the column can
    distinguish *absent* (``-1``) from *present and zero*.
    """
    flags = _ignore_flags(_grid(_as_arrays(), parity_mode))
    assert flags[ann_id] == expected_ignore
    # ... and agrees with the file route on the same document.
    assert flags == _ignore_flags(_grid(_gt_bytes(), parity_mode))


def test_ignore_column_absent_entirely_means_absent_everywhere() -> None:
    """Omitting the column is the same document as a GT with no ``ignore`` key."""
    stripped = {
        **GT,
        "annotations": [{k: v for k, v in a.items() if k != "ignore"} for a in GT["annotations"]],
    }
    columns = _ann_columns(stripped)
    del columns["ignore"]
    candidate = _from_arrays(_image_columns(stripped), columns, stripped["categories"])
    for mode in ("strict", "corrected"):
        reference = _normalize(_grid(_gt_bytes(stripped), mode).eval_imgs())
        assert _normalize(_grid(candidate, mode).eval_imgs()) == reference


def test_uint8_ignore_column_means_present_on_every_annotation() -> None:
    """A ``uint8`` ignore column has no negative, so every entry is present.

    That is the other half of the absent-versus-zero rule: a caller whose
    GT carries ``ignore`` on every annotation does not have to reach for
    the sentinel dtype.
    """
    filled = {
        **GT,
        "annotations": [{**a, "ignore": a.get("ignore", 0)} for a in GT["annotations"]],
    }
    columns = _ann_columns(filled)
    columns["ignore"] = np.array([a["ignore"] for a in filled["annotations"]], dtype=np.uint8)
    candidate = _from_arrays(_image_columns(filled), columns, filled["categories"])
    for mode in ("strict", "corrected"):
        reference = _normalize(_grid(_gt_bytes(filled), mode).eval_imgs())
        assert _normalize(_grid(candidate, mode).eval_imgs()) == reference


def test_e1_crowd_gt_keeps_its_ioa_denominator() -> None:
    """Quirk **E1**: a crowd GT scores ``intersect / area(dt)``.

    A route that dropped ``iscrowd`` would turn the crowd GTs into
    ordinary ones and change both the matches and the ignore flags. The
    check is that the array route agrees with the file route on a
    document whose crowd flags are load-bearing, and that flipping the
    column *does* move the answer — so the agreement is not vacuous.
    """
    reference = _normalize(_grid(_gt_bytes()).eval_imgs())
    assert _normalize(_grid(_as_arrays()).eval_imgs()) == reference

    columns = _ann_columns(GT)
    columns["iscrowd"] = np.zeros_like(columns["iscrowd"])
    flipped = _from_arrays(_image_columns(GT), columns, GT["categories"])
    assert _normalize(_grid(flipped).eval_imgs()) != reference


def test_gt_area_is_read_verbatim_not_derived() -> None:
    """GT ``area`` is the payload's, never the box's — unlike quirk **J3**.

    Annotation 402 carries ``area=1800`` against a 50x50 box (2500), which
    straddles nothing on its own but *does* change which area range the
    annotation falls in relative to a derived area. Scaling every area up
    by 100x must change the summary; if the route derived area from the
    bbox it could not.
    """
    reference = _grid(_as_arrays()).accumulate([1, 10, 100]).summarize().stats

    columns = _ann_columns(GT)
    columns["area"] = columns["area"] * 100.0
    rescaled = _from_arrays(_image_columns(GT), columns, GT["categories"])
    assert _grid(rescaled).accumulate([1, 10, 100]).summarize().stats != reference

    # And the verbatim value round-trips: the same scaling through JSON
    # produces the same numbers.
    scaled_doc = {
        **GT,
        "annotations": [{**a, "area": a["area"] * 100.0} for a in GT["annotations"]],
    }
    assert (
        _grid(rescaled).accumulate([1, 10, 100]).summarize().stats
        == _grid(_gt_bytes(scaled_doc)).accumulate([1, 10, 100]).summarize().stats
    )


def test_area_column_is_required() -> None:
    """There is no honest default for a GT area, so it is not optional."""
    columns = _ann_columns(GT)
    del columns["area"]
    with pytest.raises(ValueError, match=r"annotations: missing required column 'area'"):
        _from_arrays(_image_columns(GT), columns, GT["categories"])


def test_iscrowd_column_is_required() -> None:
    """``iscrowd`` drives D1 and E1; defaulting it would silently change AP."""
    columns = _ann_columns(GT)
    del columns["iscrowd"]
    with pytest.raises(ValueError, match=r"annotations: missing required column 'iscrowd'"):
        _from_arrays(_image_columns(GT), columns, GT["categories"])


def test_annotation_id_column_is_required() -> None:
    """GT ids are supplied, never assigned — so there is nothing to fall back to."""
    columns = _ann_columns(GT)
    del columns["id"]
    with pytest.raises(ValueError, match=r"annotations: missing required column 'id'"):
        _from_arrays(_image_columns(GT), columns, GT["categories"])


def test_image_counts_reach_the_handle() -> None:
    ds = _as_arrays()
    assert (ds.num_images, ds.num_annotations, ds.num_categories) == (3, 10, 2)


def test_image_dimensions_are_never_truncated() -> None:
    """A dimension outside u32 is refused, not wrapped."""
    images = _image_columns(GT)
    images["width"] = images["width"].copy()
    images["width"][1] = 2**33
    with pytest.raises(ValueError, match=r"images\[1\]\.width: .*out of range"):
        _from_arrays(images, _ann_columns(GT), GT["categories"])

    images["width"][1] = -1
    with pytest.raises(ValueError, match=r"images\[1\]\.width: .*out of range"):
        _from_arrays(images, _ann_columns(GT), GT["categories"])


def test_non_finite_geometry_is_refused() -> None:
    """JSON cannot carry NaN, so neither does this route.

    A NaN box is the failure this guard exists for: it compares false
    against every threshold, which historically read as a *perfect*
    similarity rather than an invalid one.
    """
    columns = _ann_columns(GT)
    columns["bbox"] = columns["bbox"].copy()
    columns["bbox"][2, 1] = np.nan
    with pytest.raises(ValueError, match=r"annotations\[2\]\.bbox\[1\]: expected a finite float"):
        _from_arrays(_image_columns(GT), columns, GT["categories"])

    columns = _ann_columns(GT)
    columns["area"] = columns["area"].copy()
    columns["area"][0] = np.inf
    with pytest.raises(ValueError, match=r"annotations\[0\]\.area: expected a finite float"):
        _from_arrays(_image_columns(GT), columns, GT["categories"])


def test_reference_integrity_is_the_shared_check() -> None:
    """An unknown image or category is refused by ``from_parts``, as on the file route."""
    columns = _ann_columns(GT)
    columns["image_id"] = columns["image_id"].copy()
    columns["image_id"][0] = 9999
    with pytest.raises(ValueError, match=r"unknown image_id=9999"):
        _from_arrays(_image_columns(GT), columns, GT["categories"])

    columns = _ann_columns(GT)
    columns["category_id"] = columns["category_id"].copy()
    columns["category_id"][3] = 4242
    with pytest.raises(ValueError, match=r"unknown category_id=4242"):
        _from_arrays(_image_columns(GT), columns, GT["categories"])


def test_column_lengths_must_agree() -> None:
    columns = _ann_columns(GT)
    columns["area"] = columns["area"][:-1]
    with pytest.raises(ValueError, match=r"annotations\.area: length 9 disagrees"):
        _from_arrays(_image_columns(GT), columns, GT["categories"])


def test_float32_columns_are_refused_by_default_and_cast_on_request() -> None:
    """ADR-0004's f64 boundary holds here, with ADR-0030's documented opt-in."""
    columns = _ann_columns(GT)
    columns["bbox"] = columns["bbox"].astype(np.float32)
    with pytest.raises(TypeError, match=r"annotations\.bbox"):
        _from_arrays(_image_columns(GT), columns, GT["categories"])

    with pytest.warns(UserWarning, match="cast_inputs"):
        cast = _from_arrays(_image_columns(GT), columns, GT["categories"], cast_inputs=True)
    assert cast.num_annotations == len(GT["annotations"])


def test_a_non_contiguous_column_is_refused_rather_than_copied() -> None:
    """A hidden copy is the cost this route exists to avoid, so it is named."""
    columns = _ann_columns(GT)
    wide = np.zeros((len(GT["annotations"]), 8), dtype=np.float64)
    wide[:, :4] = columns["bbox"]
    columns["bbox"] = wide[:, :4]
    assert not columns["bbox"].flags["C_CONTIGUOUS"]
    with pytest.raises((TypeError, ValueError), match=r"annotations\.bbox"):
        _from_arrays(_image_columns(GT), columns, GT["categories"])


def test_cast_inputs_reaches_the_flag_columns_too() -> None:
    """``cast_inputs`` is one switch, not one switch with two exceptions.

    ``iscrowd`` and ``ignore`` read through a dtype-dispatching reader
    rather than the single-dtype one every other column uses, so they
    are the two columns that can quietly fall outside the opt-in. A
    caller whose GT comes out of pandas or torch — where an ``int32``
    flag is ordinary — would then have to special-case exactly these
    two, on the one switch that exists so they need not.
    """
    for spelling in (
        np.array([a["iscrowd"] for a in GT["annotations"]], dtype=np.int32),
        [a["iscrowd"] for a in GT["annotations"]],
    ):
        columns = _ann_columns(GT)
        columns["iscrowd"] = spelling
        columns["ignore"] = [int(v) for v in columns["ignore"]]
        with pytest.warns(UserWarning, match="cast_inputs"):
            candidate = _from_arrays(
                _image_columns(GT), columns, GT["categories"], cast_inputs=True
            )
        assert candidate.dataset_hash == _as_json().dataset_hash


def test_flag_columns_hold_the_boundary_when_the_cast_is_not_requested() -> None:
    """The other half: on these columns the opt-in is still opt-*in*."""
    columns = _ann_columns(GT)
    columns["iscrowd"] = columns["iscrowd"].astype(np.int32)
    with pytest.raises(TypeError, match=r"annotations\.iscrowd: expected a 1-D bool"):
        _from_arrays(_image_columns(GT), columns, GT["categories"])


def test_a_flag_column_is_diagnosed_by_what_is_actually_wrong_with_it() -> None:
    """A structural rejection must not arrive dressed as a dtype rejection.

    ``iscrowd`` accepts three dtypes, which tempts the reader into
    trying each in turn and reporting whichever failed last. It would
    then answer a non-contiguous ``int64`` column — whose dtype is
    already right — with "expected bool, uint8 or int64", sending the
    caller to fix the one thing that was not broken. Dispatching from a
    single open is what keeps the diagnosis honest.
    """
    n = len(GT["annotations"])
    columns = _ann_columns(GT)
    wide = np.zeros((n, 2), dtype=np.int64)
    wide[:, 0] = columns["iscrowd"]
    columns["iscrowd"] = wide[:, 0]
    assert not columns["iscrowd"].flags["C_CONTIGUOUS"]
    with pytest.raises(TypeError, match=r"annotations\.iscrowd: array is not C-contiguous"):
        _from_arrays(_image_columns(GT), columns, GT["categories"])

    columns = _ann_columns(GT)
    columns["iscrowd"] = np.zeros((n, 2), dtype=np.int64)
    with pytest.raises(ValueError, match=r"annotations\.iscrowd: expected 1-D array, got 2-D"):
        _from_arrays(_image_columns(GT), columns, GT["categories"])


def test_a_negative_iscrowd_is_refused_rather_than_read_as_false() -> None:
    """ "Absent" is a meaning only an *optional* column has.

    ``ignore`` is optional per annotation, so a negative entry spells
    absent. ``iscrowd`` is required — the JSON route rejects anything
    but 0/1 there — so a negative entry spells nothing at all, and
    reading it as false would hand back a different crowd set: a wrong
    **E1** IoA denominator and, under ``strict``, a wrong **D1**
    ``_ignore``, with no diagnostic anywhere.
    """
    columns = _ann_columns(GT)
    crowds = columns["iscrowd"].astype(np.int64)
    crowds[2] = -1
    columns["iscrowd"] = crowds
    with pytest.raises(ValueError, match=r"annotations\[2\]\.iscrowd: -1 is not a valid flag"):
        _from_arrays(_image_columns(GT), columns, GT["categories"])


def test_a_negative_ignore_is_absent_where_a_zero_ignore_is_present() -> None:
    """The asymmetry the test above implies, asserted head-on.

    Under ``corrected``, **D1** lets an absent ``ignore`` fall back to
    ``iscrowd``; a present-and-zero one pins it false. On a crowd
    annotation the two spellings therefore disagree — which is the
    entire reason the sentinel encoding exists, and the reason
    ``iscrowd`` must not borrow it.
    """
    crowd_id = next(a["id"] for a in GT["annotations"] if a["iscrowd"] == 1)
    index = next(i for i, a in enumerate(GT["annotations"]) if a["id"] == crowd_id)

    absent = _ann_columns(GT)
    absent["ignore"] = np.full(len(GT["annotations"]), -1, dtype=np.int64)
    present = _ann_columns(GT)
    present["ignore"] = np.zeros(len(GT["annotations"]), dtype=np.int64)
    assert present["ignore"][index] == 0

    flags_absent = _ignore_flags(
        _grid(_from_arrays(_image_columns(GT), absent, GT["categories"]), "corrected")
    )
    flags_present = _ignore_flags(
        _grid(_from_arrays(_image_columns(GT), present, GT["categories"]), "corrected")
    )
    assert flags_absent[crowd_id] == 1
    assert flags_present[crowd_id] == 0


def test_lvis_federated_metadata_is_not_expressible_on_this_route() -> None:
    """The array route builds a COCO-flat dataset, and says so.

    ADR-0060 declines to give ``neg_category_ids`` /
    ``not_exhaustive_category_ids`` / per-category ``frequency`` a
    columnar spelling. The risk that decision carries is the one
    ``from_lvis_json``'s docstring already names — LVIS data loaded under
    COCO semantics scores systematically lower, silently — so the flag
    that distinguishes them is pinned here.
    """
    assert _as_arrays().is_federated is False


# --- payload-shape matrix: segmentation --------------------------------------

SEGM_GT: dict[str, Any] = {
    "images": [{"id": 1, "width": 32, "height": 32}],
    "categories": [{"id": 1, "name": "a"}],
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [2, 2, 8, 8],
            "area": 64.0,
            "iscrowd": 0,
        },
        {
            "id": 2,
            "image_id": 1,
            "category_id": 1,
            "bbox": [16, 16, 8, 8],
            "area": 64.0,
            "iscrowd": 0,
        },
    ],
}

#: The two annotations' masks, as plain rectangles.
_MASK_BOXES = [(2, 2, 8, 8), (16, 16, 8, 8)]


def _polygon(x: int, y: int, w: int, h: int) -> list[list[float]]:
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


def _bitmask(x: int, y: int, w: int, h: int) -> np.ndarray:
    m = np.zeros((32, 32), dtype=np.uint8)
    m[y : y + h, x : x + w] = 1
    return m


def _uncompressed_counts(mask: np.ndarray) -> list[int]:
    """Column-major run lengths, starting from a run of zeros."""
    counts: list[int] = []
    current = 0
    run = 0
    for value in mask.flatten(order="F"):
        if int(value) == current:
            run += 1
        else:
            counts.append(run)
            current = int(value)
            run = 1
    counts.append(run)
    return counts


def _compressed_counts(mask: np.ndarray) -> bytes:
    """The COCO 6-bit string, via the pinned oracle (no encoder is public)."""
    from pycocotools import mask as mask_util

    counts = mask_util.encode(np.asfortranarray(mask))["counts"]
    # pycocotools' stub widens this to `str | bytes`; on Python 3 it is
    # always `bytes` (quirk **K3** is about the *reading* side).
    assert isinstance(counts, bytes)
    return counts


SEGM_SHAPES = {
    # The three a GT JSON *file* can carry ...
    "polygon": lambda b: _polygon(*b),
    "counts_str": lambda b: {
        "counts": _compressed_counts(_bitmask(*b)).decode("ascii"),
        "size": [32, 32],
    },
    "counts_list": lambda b: {"counts": _uncompressed_counts(_bitmask(*b)), "size": [32, 32]},
    # ... plus the three in-memory spellings ADR-0030 added, which a
    # caller holding masks in memory has and a file never does.
    "counts_bytes": lambda b: {"counts": _compressed_counts(_bitmask(*b)), "size": [32, 32]},
    "counts_uint32": lambda b: {
        "counts": np.array(_uncompressed_counts(_bitmask(*b)), dtype=np.uint32),
        "size": [32, 32],
    },
    "bitmask": lambda b: _bitmask(*b),
}

#: The six accepted spellings collapse to **three** canonical forms, and
#: the value maps each spelling to the JSON-spellable member of its
#: group. `dataset_hash` retains the stored representation (a compressed
#: `counts` string and the identical mask as run lengths are the same
#: pixels but not the same canonical form), so hash equality is asserted
#: within a group; the *evaluated numbers* are asserted equal across all
#: six, because that is the property that must not depend on spelling.
_CANONICAL_GROUP = {
    "polygon": "polygon",
    "counts_str": "counts_str",
    "counts_bytes": "counts_str",
    "counts_list": "counts_list",
    "counts_uint32": "counts_list",
    "bitmask": "counts_list",
}


def _segm_dt() -> bytes:
    return json.dumps(
        [
            {
                "image_id": 1,
                "category_id": 1,
                "bbox": [2.0, 2.0, 8.0, 8.0],
                "score": 0.9,
                "segmentation": SEGM_SHAPES["counts_str"]((2, 2, 8, 8)),
            },
            {
                "image_id": 1,
                "category_id": 1,
                "bbox": [16.0, 16.0, 8.0, 8.0],
                "score": 0.5,
                "segmentation": SEGM_SHAPES["counts_str"]((16, 16, 8, 8)),
            },
        ]
    ).encode()


def _segm_stats(gt: Any) -> Any:
    if isinstance(gt, bytes):
        return _core.evaluate_segm_summary(
            gt, _segm_dt(), parity_mode="strict", max_dets=[1, 10, 100], use_cats=True
        ).stats
    return _core.evaluate_segm_summary(
        gt, _segm_dt(), parity_mode="strict", max_dets=[1, 10, 100], use_cats=True
    ).stats


@pytest.mark.parametrize("shape", sorted(SEGM_SHAPES))
def test_every_gt_segmentation_shape_is_accepted_and_agrees(shape: str) -> None:
    """The payload-shape matrix: each GT ``segmentation`` spelling.

    The JSON route accepts polygons (**K2**), a compressed ``counts``
    string (**K3**) and an uncompressed list of ints. The array column
    accepts those three *plus* the in-memory spellings ADR-0030 added —
    ``bytes`` counts and a 2-D bitmask — because a caller holding masks in
    memory has those and a file never does. It is a superset, and every
    member of it is compared against the file route fed the same logical
    masks.

    Two assertions, because only bbox has a grid-taking
    ``_with_dataset`` entry point:

    1. The 12-stat segm summary equals the file route's for **every**
       spelling. This is the property that must never depend on how the
       caller spelled the mask, and it is what catches a mask that
       decoded to the wrong pixels — the failure a ``bytearray`` of
       compressed counts once produced on the detection side.
    2. ``dataset_hash`` — byte equality of the canonical form, every
       ``counts`` byte included — equals the file route's for the
       spelling in the same canonical group. The six spellings collapse
       to three stored forms (polygon, compressed string, uncompressed
       runs); `dataset_hash` retains which one, by design, since it is
       the ADR-0031 wire-format identity rather than a pixel digest.
    """
    build = SEGM_SHAPES[shape]
    columns = {
        "id": np.array([1, 2], dtype=np.int64),
        "image_id": np.array([1, 1], dtype=np.int64),
        "category_id": np.array([1, 1], dtype=np.int64),
        "bbox": np.array([list(map(float, b)) for b in _MASK_BOXES], dtype=np.float64),
        "area": np.array([64.0, 64.0], dtype=np.float64),
        "iscrowd": np.zeros(2, dtype=np.uint8),
        "segmentation": [build(b) for b in _MASK_BOXES],
    }
    candidate = _from_arrays(
        {
            "id": np.array([1], dtype=np.int64),
            "width": np.array([32], dtype=np.int64),
            "height": np.array([32], dtype=np.int64),
        },
        columns,
        SEGM_GT["categories"],
    )

    # The file route's spelling of the same masks. For the two in-memory
    # shapes with no JSON form, the compressed string is the same mask.
    json_build = SEGM_SHAPES[_CANONICAL_GROUP[shape]]
    doc = {
        **SEGM_GT,
        "annotations": [
            {**a, "segmentation": json_build(b)}
            for a, b in zip(SEGM_GT["annotations"], _MASK_BOXES)
        ],
    }
    gt_bytes = json.dumps(doc).encode()
    assert _segm_stats(candidate) == _segm_stats(gt_bytes)
    assert candidate.dataset_hash == _core.CocoDataset.from_json(gt_bytes).dataset_hash


def test_segmentation_may_be_absent_on_some_annotations() -> None:
    """``None`` per entry is the column's spelling of a missing field.

    A GT file can omit ``segmentation`` on one annotation and carry it on
    the next; so can this column, because its entries are Python objects
    and Python has a null. (The fields that do *not* — ``ignore``,
    ``num_keypoints`` — use the negative sentinel instead.)
    """
    columns = {
        "id": np.array([1, 2], dtype=np.int64),
        "image_id": np.array([1, 1], dtype=np.int64),
        "category_id": np.array([1, 1], dtype=np.int64),
        "bbox": np.array([list(map(float, b)) for b in _MASK_BOXES], dtype=np.float64),
        "area": np.array([64.0, 64.0], dtype=np.float64),
        "iscrowd": np.zeros(2, dtype=np.uint8),
        "segmentation": [_polygon(*_MASK_BOXES[0]), None],
    }
    images = {
        "id": np.array([1], dtype=np.int64),
        "width": np.array([32], dtype=np.int64),
        "height": np.array([32], dtype=np.int64),
    }
    candidate = _from_arrays(images, columns, SEGM_GT["categories"])
    doc = {
        **SEGM_GT,
        "annotations": [
            {**SEGM_GT["annotations"][0], "segmentation": _polygon(*_MASK_BOXES[0])},
            SEGM_GT["annotations"][1],
        ],
    }
    gt_bytes = json.dumps(doc).encode()
    reference = _core.CocoDataset.from_json(gt_bytes)
    assert candidate.dataset_hash == reference.dataset_hash

    # And the routes agree on what such a document *does*: core refuses a
    # GT with no mask under `segm`, identically whichever route built it.
    # The agreement is the assertion — not that either one succeeds.
    with pytest.raises(ValueError, match=r"GT id=2 .* no `segmentation` field"):
        _segm_stats(candidate)
    with pytest.raises(ValueError, match=r"GT id=2 .* no `segmentation` field"):
        _segm_stats(gt_bytes)


def test_image_width_and_height_are_load_bearing_under_segm() -> None:
    """``width`` / ``height`` reach the polygon rasterizer, and are checked there.

    Bbox evaluation never reads image dimensions — measured: perturbing
    ``width`` moves 0 of 11 ``eval_imgs`` columns on the bbox grid. The
    kernels that *do* read them are segm and boundary, where the image
    size is the canvas a polygon is rasterized onto. So this is where the
    columns are pinned: the array route must agree with the file route,
    and a wrong dimension must change the answer.
    """
    columns = {
        "id": np.array([1, 2], dtype=np.int64),
        "image_id": np.array([1, 1], dtype=np.int64),
        "category_id": np.array([1, 1], dtype=np.int64),
        "bbox": np.array([list(map(float, b)) for b in _MASK_BOXES], dtype=np.float64),
        "area": np.array([64.0, 64.0], dtype=np.float64),
        "iscrowd": np.zeros(2, dtype=np.uint8),
        "segmentation": [_polygon(*b) for b in _MASK_BOXES],
    }
    doc = {
        **SEGM_GT,
        "annotations": [
            {**a, "segmentation": _polygon(*b)} for a, b in zip(SEGM_GT["annotations"], _MASK_BOXES)
        ],
    }
    gt_bytes = json.dumps(doc).encode()

    right = _from_arrays(
        {
            "id": np.array([1], dtype=np.int64),
            "width": np.array([32], dtype=np.int64),
            "height": np.array([32], dtype=np.int64),
        },
        columns,
        SEGM_GT["categories"],
    )
    assert right.dataset_hash == _core.CocoDataset.from_json(gt_bytes).dataset_hash
    assert _segm_stats(right) == _segm_stats(gt_bytes)

    # A wrong dimension is not merely visible, it is refused: core checks
    # the declared mask size against the image the array route carried.
    wrong = _from_arrays(
        {
            "id": np.array([1], dtype=np.int64),
            "width": np.array([16], dtype=np.int64),
            "height": np.array([32], dtype=np.int64),
        },
        columns,
        SEGM_GT["categories"],
    )
    with pytest.raises(ValueError, match=r"but image is \[32, 16\]"):
        _segm_stats(wrong)


def test_segmentation_column_refuses_a_stacked_array() -> None:
    """An ``(N, H, W)`` array is not the column shape, and is not iterated into one.

    NumPy arrays satisfy the sequence protocol, so a stacked bitmask
    array would otherwise be silently walked into N planes — a
    plausible-looking answer from a payload this column does not accept.
    """
    columns = {
        "id": np.array([1, 2], dtype=np.int64),
        "image_id": np.array([1, 1], dtype=np.int64),
        "category_id": np.array([1, 1], dtype=np.int64),
        "bbox": np.array([list(map(float, b)) for b in _MASK_BOXES], dtype=np.float64),
        "area": np.array([64.0, 64.0], dtype=np.float64),
        "iscrowd": np.zeros(2, dtype=np.uint8),
        "segmentation": np.stack([_bitmask(*b) for b in _MASK_BOXES]),
    }
    images = {
        "id": np.array([1], dtype=np.int64),
        "width": np.array([32], dtype=np.int64),
        "height": np.array([32], dtype=np.int64),
    }
    with pytest.raises(TypeError, match=r"annotations\.segmentation: expected a list"):
        _from_arrays(images, columns, SEGM_GT["categories"])


def test_object_dtype_columns_are_accepted_by_the_per_entry_columns() -> None:
    """A DataFrame column is an ``object`` array, and cannot be the mistake.

    The guard above refuses arrays because iterating a stacked bitmask
    would produce a plausible-looking wrong answer. An ``object`` array
    cannot be that: it holds one Python object per entry, which is
    precisely this column's shape. It is also the ordinary spelling out
    of a DataFrame (``df["segmentation"].to_numpy()``), and turning it
    away would send the caller back through ``.tolist()`` — a
    per-annotation Python cost on the route that exists to remove one.
    """
    polygons = [_polygon(*b) for b in _MASK_BOXES]
    boxed = np.empty(len(polygons), dtype=object)
    for i, polygon in enumerate(polygons):
        boxed[i] = polygon
    names = np.empty(1, dtype=object)
    names[0] = "one.jpg"

    def build(segmentation: Any, file_name: Any) -> Any:
        return _from_arrays(
            {
                "id": np.array([1], dtype=np.int64),
                "width": np.array([32], dtype=np.int64),
                "height": np.array([32], dtype=np.int64),
                "file_name": file_name,
            },
            {
                "id": np.array([1, 2], dtype=np.int64),
                "image_id": np.array([1, 1], dtype=np.int64),
                "category_id": np.array([1, 1], dtype=np.int64),
                "bbox": np.array([list(map(float, b)) for b in _MASK_BOXES], dtype=np.float64),
                "area": np.array([64.0, 64.0], dtype=np.float64),
                "iscrowd": np.zeros(2, dtype=np.uint8),
                "segmentation": segmentation,
            },
            SEGM_GT["categories"],
        )

    doc = {
        **SEGM_GT,
        "images": [{**SEGM_GT["images"][0], "file_name": "one.jpg"}],
        "annotations": [
            {**a, "segmentation": polygon} for a, polygon in zip(SEGM_GT["annotations"], polygons)
        ],
    }
    reference = _core.CocoDataset.from_json(json.dumps(doc).encode())
    assert build(boxed, names).dataset_hash == reference.dataset_hash
    assert build(polygons, ["one.jpg"]).dataset_hash == reference.dataset_hash


def test_a_bad_segmentation_names_the_gt_column_not_the_detection_one() -> None:
    """Rejections are rooted at the argument the caller actually passed."""
    columns = {
        "id": np.array([1, 2], dtype=np.int64),
        "image_id": np.array([1, 1], dtype=np.int64),
        "category_id": np.array([1, 1], dtype=np.int64),
        "bbox": np.array([list(map(float, b)) for b in _MASK_BOXES], dtype=np.float64),
        "area": np.array([64.0, 64.0], dtype=np.float64),
        "iscrowd": np.zeros(2, dtype=np.uint8),
        "segmentation": [_polygon(*_MASK_BOXES[0]), 17],
    }
    images = {
        "id": np.array([1], dtype=np.int64),
        "width": np.array([32], dtype=np.int64),
        "height": np.array([32], dtype=np.int64),
    }
    with pytest.raises(TypeError, match=r"annotations\[1\]\.segmentation"):
        _from_arrays(images, columns, SEGM_GT["categories"])


# --- payload-shape matrix: keypoints -----------------------------------------

KP_GT: dict[str, Any] = {
    "images": [{"id": 1, "width": 64, "height": 64}],
    "categories": [{"id": 1, "name": "person"}],
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [0, 0, 40, 40],
            "area": 1600.0,
            "iscrowd": 0,
            "keypoints": [10, 10, 2, 20, 20, 2, 30, 30, 1] + [0, 0, 0] * 14,
            "num_keypoints": 3,
        },
        {
            "id": 2,
            "image_id": 1,
            "category_id": 1,
            "bbox": [40, 40, 20, 20],
            "area": 400.0,
            "iscrowd": 0,
            "keypoints": [45, 45, 2, 50, 50, 1] + [0, 0, 0] * 15,
            "num_keypoints": 2,
        },
    ],
}


def _kp_dt() -> bytes:
    return json.dumps(
        [
            {
                "image_id": 1,
                "category_id": 1,
                "bbox": [0.0, 0.0, 40.0, 40.0],
                "score": 0.9,
                "keypoints": [10, 10, 2, 20, 20, 2, 30, 30, 1] + [0, 0, 0] * 14,
            },
            {
                "image_id": 1,
                "category_id": 1,
                "bbox": [40.0, 40.0, 20.0, 20.0],
                "score": 0.5,
                "keypoints": [46, 45, 2, 50, 51, 1] + [0, 0, 0] * 15,
            },
        ]
    ).encode()


#: OKS sigmas for the 17-joint COCO person skeleton, as the FFI takes
#: them (per-category). Pinned here so the keypoints comparisons below
#: are not sensitive to a default changing.
_KP_SIGMAS = {
    1: [
        0.026,
        0.025,
        0.025,
        0.035,
        0.035,
        0.079,
        0.079,
        0.072,
        0.072,
        0.062,
        0.062,
        0.107,
        0.107,
        0.087,
        0.087,
        0.089,
        0.089,
    ]
}


def _kp_stats(gt: Any) -> Any:
    if isinstance(gt, bytes):
        return _core.evaluate_keypoints_summary(
            gt, _kp_dt(), parity_mode="strict", max_dets=[20], use_cats=True, sigmas=_KP_SIGMAS
        ).stats
    return _core.evaluate_keypoints_summary(
        gt, _kp_dt(), parity_mode="strict", max_dets=[20], use_cats=True, sigmas=_KP_SIGMAS
    ).stats


def test_keypoints_and_num_keypoints_survive_the_array_route() -> None:
    """``(N, K, 3)`` keypoints plus the ``num_keypoints`` count (quirk **D2**).

    D2 makes a GT with zero visible keypoints an implicit ignore region,
    so ``num_keypoints`` is load-bearing rather than decorative and has to
    travel.
    """
    anns = KP_GT["annotations"]
    columns = {
        "id": np.array([a["id"] for a in anns], dtype=np.int64),
        "image_id": np.array([a["image_id"] for a in anns], dtype=np.int64),
        "category_id": np.array([a["category_id"] for a in anns], dtype=np.int64),
        "bbox": np.array([a["bbox"] for a in anns], dtype=np.float64),
        "area": np.array([a["area"] for a in anns], dtype=np.float64),
        "iscrowd": np.zeros(len(anns), dtype=np.uint8),
        "keypoints": np.array(
            [np.asarray(a["keypoints"], dtype=np.float64).reshape(17, 3) for a in anns],
            dtype=np.float64,
        ),
        "num_keypoints": np.array([a["num_keypoints"] for a in anns], dtype=np.int64),
    }
    images = {
        "id": np.array([1], dtype=np.int64),
        "width": np.array([64], dtype=np.int64),
        "height": np.array([64], dtype=np.int64),
    }
    candidate = _from_arrays(images, columns, KP_GT["categories"])
    gt_bytes = json.dumps(KP_GT).encode()
    assert _kp_stats(candidate) == _kp_stats(gt_bytes)
    assert candidate.dataset_hash == _core.CocoDataset.from_json(gt_bytes).dataset_hash


def test_num_keypoints_absent_is_distinguishable_from_zero() -> None:
    """``-1`` is absent; ``0`` is a real count that fires D2's implicit ignore."""
    anns = KP_GT["annotations"]
    base = {
        "id": np.array([a["id"] for a in anns], dtype=np.int64),
        "image_id": np.array([a["image_id"] for a in anns], dtype=np.int64),
        "category_id": np.array([a["category_id"] for a in anns], dtype=np.int64),
        "bbox": np.array([a["bbox"] for a in anns], dtype=np.float64),
        "area": np.array([a["area"] for a in anns], dtype=np.float64),
        "iscrowd": np.zeros(len(anns), dtype=np.uint8),
        "keypoints": np.array(
            [np.asarray(a["keypoints"], dtype=np.float64).reshape(17, 3) for a in anns],
            dtype=np.float64,
        ),
    }
    images = {
        "id": np.array([1], dtype=np.int64),
        "width": np.array([64], dtype=np.int64),
        "height": np.array([64], dtype=np.int64),
    }

    absent = _from_arrays(
        images, {**base, "num_keypoints": np.array([-1, -1], dtype=np.int64)}, KP_GT["categories"]
    )
    omitted = _from_arrays(images, dict(base), KP_GT["categories"])
    zeroed = _from_arrays(
        images, {**base, "num_keypoints": np.zeros(2, dtype=np.int64)}, KP_GT["categories"]
    )

    # A -1 column and no column at all are the same document ...
    assert absent.dataset_hash == omitted.dataset_hash
    # ... and both are the JSON document with no `num_keypoints` key.
    no_key = {
        **KP_GT,
        "annotations": [
            {k: v for k, v in a.items() if k != "num_keypoints"} for a in KP_GT["annotations"]
        ],
    }
    no_key_bytes = json.dumps(no_key).encode()
    assert absent.dataset_hash == _core.CocoDataset.from_json(no_key_bytes).dataset_hash
    assert _kp_stats(absent) == _kp_stats(no_key_bytes)

    # ... and a real 0 is a *different* document, which is what makes the
    # sentinel load-bearing rather than cosmetic: D2 turns a GT with zero
    # visible keypoints into an implicit ignore region.
    assert zeroed.dataset_hash != absent.dataset_hash
    zero_doc = {
        **KP_GT,
        "annotations": [{**a, "num_keypoints": 0} for a in KP_GT["annotations"]],
    }
    zero_bytes = json.dumps(zero_doc).encode()
    assert zeroed.dataset_hash == _core.CocoDataset.from_json(zero_bytes).dataset_hash
    assert _kp_stats(zeroed) == _kp_stats(zero_bytes)
    assert _kp_stats(zeroed) != _kp_stats(absent)


def test_non_finite_keypoints_are_refused() -> None:
    """A NaN keypoint scored a perfect OKS once; it does not get in here."""
    anns = KP_GT["annotations"]
    kp = np.array(
        [np.asarray(a["keypoints"], dtype=np.float64).reshape(17, 3) for a in anns],
        dtype=np.float64,
    )
    kp[1, 0, 0] = np.nan
    columns = {
        "id": np.array([a["id"] for a in anns], dtype=np.int64),
        "image_id": np.array([a["image_id"] for a in anns], dtype=np.int64),
        "category_id": np.array([a["category_id"] for a in anns], dtype=np.int64),
        "bbox": np.array([a["bbox"] for a in anns], dtype=np.float64),
        "area": np.array([a["area"] for a in anns], dtype=np.float64),
        "iscrowd": np.zeros(len(anns), dtype=np.uint8),
        "keypoints": kp,
    }
    images = {
        "id": np.array([1], dtype=np.int64),
        "width": np.array([64], dtype=np.int64),
        "height": np.array([64], dtype=np.int64),
    }
    with pytest.raises(
        ValueError, match=r"annotations\[1\]\.keypoints\[0\]: expected a finite float"
    ):
        _from_arrays(images, columns, KP_GT["categories"])


# --- categories --------------------------------------------------------------


def test_categories_define_the_k_axis_and_its_order() -> None:
    """Category order is the K axis; reversing it must reach the per-class output."""
    forward = _as_arrays()
    reversed_doc = {**GT, "categories": list(reversed(GT["categories"]))}
    backward = _from_arrays(_image_columns(GT), _ann_columns(GT), reversed_doc["categories"])
    assert forward.num_categories == backward.num_categories == 2
    # The file route agrees with each ordering separately.
    assert _normalize(_grid(backward).eval_imgs()) == _normalize(
        _grid(_gt_bytes(reversed_doc)).eval_imgs()
    )


def test_category_supercategory_is_optional_and_carried() -> None:
    ds = _from_arrays(
        _image_columns(GT),
        _ann_columns(GT),
        [{"id": 2, "name": "cat", "supercategory": "animal"}, {"id": 5, "name": "dog"}],
    )
    assert ds.num_categories == 2


def test_a_malformed_category_names_its_index() -> None:
    with pytest.raises(ValueError, match=r"categories\[1\]\.name: missing required field"):
        _from_arrays(_image_columns(GT), _ann_columns(GT), [{"id": 2, "name": "cat"}, {"id": 5}])
