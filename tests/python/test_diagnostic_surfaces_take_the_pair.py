"""ADR-0064: the `(CocoDataset, DetectionsInput)` pair on every instance surface.

Each case evaluates *one document* in both spellings — GT bytes vs. a
handle, results JSON vs. an ``(N, 7)`` matrix in the JSON's own order
(so quirk **J1**'s positional auto-ids match) — and asserts the same
result, not merely that the call runs.

The four standalone diagnostics refuse LVIS federated ground truth; the
``Evaluator.evaluate`` branches accept it and must agree with each other.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from vernier.instance import (
    Bbox,
    Boundary,
    CocoDataset,
    Evaluator,
    Segm,
    confusion_matrix,
    error_decomposition,
    fp_iou_histogram,
    optimal_lrp,
)

from .parity.conftest import loadres_to_detections

_TIDE_FIXTURES = Path(__file__).parent / "oracle" / "tide" / "fixtures"
_PARTITION_FIXTURE = Path(__file__).parent / "parity" / "fixtures" / "partition_tiny"
_LVIS_FIXTURE = Path(__file__).parent / "parity_lvis" / "fixtures" / "federated_min"

#: Fixtures whose bbox detections exercise a different bin mix — a
#: clean diagonal, a class swap, a localization miss and a duplicate.
_CASES = ["all_perfect", "all_cls", "all_loc", "all_dupe"]

#: The mask-kernel fixture: same-class pairs overlap partially, so every
#: diagnostic's matching pass calls the kernel (and so fills a cache).
_SEGM_FIXTURE = "segm_all_loc"


def _load(root: Path) -> tuple[bytes, bytes]:
    return (root / "gt.json").read_bytes(), (root / "dt.json").read_bytes()


def _case(root: Path) -> tuple[bytes, bytes, CocoDataset, NDArray[np.float64]]:
    """One fixture read once, in both spellings.

    The bytes and the `(handle, matrix)` pair describe the same
    document, which is the whole point of every assertion below.
    """
    gt_bytes, dt_bytes = _load(root)
    return gt_bytes, dt_bytes, CocoDataset.from_json(gt_bytes), _dt_matrix(dt_bytes)


def _dt_matrix(dt_bytes: bytes) -> NDArray[np.float64]:
    """The ADR-0057 ``(N, 7)`` matrix for a ``loadRes``-shaped payload.

    Rows stay in the JSON array's order, which is what makes the two
    routes the *same* detections: `CocoDetections::from_inputs` assigns
    auto-ids by position (quirk **J1**), so a reordering would be a
    different document even with identical boxes.
    """
    records: list[dict[str, Any]] = json.loads(dt_bytes)
    rows = [
        [
            float(r["image_id"]),
            *(float(v) for v in r["bbox"]),
            float(r["score"]),
            float(r["category_id"]),
        ]
        for r in records
    ]
    return np.asarray(rows, dtype=np.float64).reshape(len(rows), 7)


def _counts(df: object) -> dict[tuple[str, str], int]:
    rows = df.iter_rows(named=True)  # type: ignore[attr-defined]
    return {(r["gt_class"], r["dt_class"]): int(r["count"]) for r in rows}


def _nan_safe(value: object) -> object:
    """Normalize LRP's ``NaN`` "undefined" sentinel.

    A class with no TP at any tau has no deployable ``tau``, which the
    report spells ``NaN`` — and ``nan != nan``, so a plain dataclass
    comparison reports a difference where the two reports agree. Map it
    to a token that compares equal to itself, recursively, so the
    normalizer covers whatever fields the dataclass holds.
    """
    if isinstance(value, float) and np.isnan(value):
        return "undefined"
    if isinstance(value, dict):
        return {k: _nan_safe(v) for k, v in cast("dict[str, object]", value).items()}
    if isinstance(value, list):
        return [_nan_safe(v) for v in cast("list[object]", value)]
    return value


def _lrp_shape(report: Any) -> object:
    """An :class:`LrpReport` as a NaN-normalized plain structure.

    Via :func:`dataclasses.asdict` rather than a hand-written field
    list: a field added to ``LrpReport`` or ``LrpPerClass`` must not be
    able to drop silently out of this comparison, which is the one
    failure an equality test exists to catch.
    """
    return _nan_safe(dataclasses.asdict(report))


# ---------------------------------------------------------------------------
# The four standalone diagnostics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", _CASES)
def test_error_decomposition_reads_the_pair(fixture: str) -> None:
    gt_bytes, dt_bytes, handle, matrix = _case(_TIDE_FIXTURES / fixture)

    from_bytes = error_decomposition(gt_bytes, dt_bytes, iou=Bbox())
    from_pair = error_decomposition(handle, matrix, iou=Bbox())

    assert from_pair == from_bytes


@pytest.mark.parametrize("fixture", _CASES)
def test_fp_iou_histogram_reads_the_pair(fixture: str) -> None:
    gt_bytes, dt_bytes, handle, matrix = _case(_TIDE_FIXTURES / fixture)

    from_bytes = fp_iou_histogram(gt_bytes, dt_bytes, iou=Bbox())
    from_pair = fp_iou_histogram(handle, matrix, iou=Bbox())

    assert np.array_equal(from_pair.iou_same, from_bytes.iou_same)
    assert np.array_equal(from_pair.iou_cross, from_bytes.iou_cross)
    assert (from_pair.n_fps, from_pair.n_total_dts) == (
        from_bytes.n_fps,
        from_bytes.n_total_dts,
    )
    assert from_pair.kernel == from_bytes.kernel
    assert from_pair.t_f == from_bytes.t_f


@pytest.mark.parametrize("fixture", _CASES)
def test_optimal_lrp_reads_the_pair(fixture: str) -> None:
    gt_bytes, dt_bytes, handle, matrix = _case(_TIDE_FIXTURES / fixture)

    from_bytes = optimal_lrp(gt_bytes, dt_bytes, iou=Bbox())
    from_pair = optimal_lrp(handle, matrix, iou=Bbox())

    assert _lrp_shape(from_pair) == _lrp_shape(from_bytes)


@pytest.mark.parametrize("fixture", _CASES)
def test_confusion_matrix_reads_the_pair(fixture: str) -> None:
    pytest.importorskip("polars", reason="`vernier[tables]` extra not installed")
    gt_bytes, dt_bytes, handle, matrix = _case(_TIDE_FIXTURES / fixture)

    from_bytes = _counts(confusion_matrix(gt_bytes, dt_bytes, iou=Bbox()))
    from_pair = _counts(confusion_matrix(handle, matrix, iou=Bbox()))

    assert from_pair == from_bytes
    assert from_bytes


# ---------------------------------------------------------------------------
# The two Evaluator.evaluate paths
# ---------------------------------------------------------------------------


def test_evaluate_tables_reads_the_pair() -> None:
    pytest.importorskip("polars", reason="`vernier[tables]` extra not installed")
    gt_bytes, dt_bytes, handle, matrix = _case(_PARTITION_FIXTURE)
    ev = Evaluator(iou=Bbox())

    from_bytes = ev.evaluate(gt_bytes, dt_bytes, tables=("per_class", "per_image"))
    from_pair = ev.evaluate(handle, matrix, tables=("per_class", "per_image"))

    assert from_pair.summary is not None
    assert from_bytes.summary is not None
    assert from_pair.summary.stats == from_bytes.summary.stats
    assert from_pair.per_class.equals(from_bytes.per_class)
    assert from_pair.per_image.equals(from_bytes.per_image)


def test_evaluate_manifest_reads_the_pair() -> None:
    pytest.importorskip("polars", reason="`vernier[tables]` extra not installed")
    gt_bytes, dt_bytes, handle, matrix = _case(_PARTITION_FIXTURE)
    manifest = json.loads((_PARTITION_FIXTURE / "weather_x_tod.json").read_bytes())
    ev = Evaluator(iou=Bbox())

    from_bytes = ev.evaluate(gt_bytes, dt_bytes, manifest=manifest)
    from_pair = ev.evaluate(handle, matrix, manifest=manifest)

    assert from_pair.summary is not None
    assert from_bytes.summary is not None
    assert from_pair.summary.stats == from_bytes.summary.stats
    assert from_pair.slices.equals(from_bytes.slices)


def test_optimal_lrp_manifest_reads_the_pair() -> None:
    """The ``manifest=`` form of LRP has its own FFI family; widening
    ``optimal_lrp`` without it would leave half the surface behind."""
    pytest.importorskip("polars", reason="`vernier[tables]` extra not installed")
    gt_bytes, dt_bytes, handle, matrix = _case(_PARTITION_FIXTURE)
    manifest = json.loads((_PARTITION_FIXTURE / "weather_x_tod.json").read_bytes())

    from_bytes = optimal_lrp(gt_bytes, dt_bytes, iou=Bbox(), manifest=manifest)
    from_pair = optimal_lrp(handle, matrix, iou=Bbox(), manifest=manifest)

    assert _lrp_shape(from_pair.overall) == _lrp_shape(from_bytes.overall)
    assert from_pair.slices.equals(from_bytes.slices)


# ---------------------------------------------------------------------------
# The detections side is the whole union, not just the matrix
# ---------------------------------------------------------------------------


def test_columnar_detections_reach_the_diagnostics() -> None:
    """The ``(N, 7)`` matrix is one member of ``DetectionsInput``; the
    ADR-0030 columnar dicts are another, and a mask pipeline has only
    those. Both must land on the same report."""
    gt_bytes, dt_bytes = _load(_TIDE_FIXTURES / "all_cls")
    handle = CocoDataset.from_json(gt_bytes)
    columnar = loadres_to_detections(json.loads(gt_bytes), json.loads(dt_bytes), "bbox")

    from_bytes = error_decomposition(gt_bytes, dt_bytes, iou=Bbox())
    from_columns = error_decomposition(handle, columnar, iou=Bbox())

    assert from_columns == from_bytes


def test_cast_inputs_gates_a_float32_matrix() -> None:
    """``cast_inputs`` means here exactly what it means on ``Evaluator``:
    off, ADR-0004's f64 boundary refuses; on, the dtype is converted."""
    gt_bytes, dt_bytes = _load(_TIDE_FIXTURES / "all_cls")
    handle = CocoDataset.from_json(gt_bytes)
    # `DetectionsInput` pins the matrix to float64, so the narrow array
    # is a type error by design — which is the thing being tested, and
    # why it reaches the runtime check through `cast`.
    narrow = cast("Any", _dt_matrix(dt_bytes).astype(np.float32))

    with pytest.raises((TypeError, ValueError)):
        error_decomposition(handle, narrow, iou=Bbox())

    widened = error_decomposition(handle, narrow, iou=Bbox(), cast_inputs=True)
    assert widened == error_decomposition(gt_bytes, dt_bytes, iou=Bbox())


# ---------------------------------------------------------------------------
# The mask kernels, and the handle's GT caches
# ---------------------------------------------------------------------------


def _histogram_shape(h: Any) -> object:
    return (h.iou_same.tolist(), h.iou_cross.tolist(), h.n_fps, h.n_total_dts, h.kernel)


def _confusion_shape(df: object) -> object:
    pytest.importorskip("polars", reason="`vernier[tables]` extra not installed")
    return _counts(df)


_MASK_DIAGNOSTICS = [
    pytest.param(error_decomposition, lambda r: r, id="error_decomposition"),
    pytest.param(fp_iou_histogram, _histogram_shape, id="fp_iou_histogram"),
    pytest.param(optimal_lrp, _lrp_shape, id="optimal_lrp"),
    pytest.param(confusion_matrix, _confusion_shape, id="confusion_matrix"),
]


@pytest.mark.parametrize(
    ("iou", "cache_len"),
    [(Segm(), "segm_cache_len"), (Boundary(), "boundary_cache_len")],
    ids=["segm", "boundary"],
)
@pytest.mark.parametrize(("call", "shape"), _MASK_DIAGNOSTICS)
def test_mask_diagnostics_fill_and_reuse_the_handle_cache(
    call: Any, shape: Any, iou: object, cache_len: str
) -> None:
    """Each mask entry point takes the handle, fills its GT cache, and a
    warm cache reproduces the bytes result bit for bit."""
    gt_bytes, dt_bytes = _load(_TIDE_FIXTURES / _SEGM_FIXTURE)
    handle = CocoDataset.from_json(gt_bytes)
    expected = shape(call(gt_bytes, dt_bytes, iou=iou))

    assert shape(call(handle, dt_bytes, iou=iou)) == expected
    assert getattr(handle, cache_len) > 0
    assert shape(call(handle, dt_bytes, iou=iou)) == expected


def test_diagnostics_share_the_cache_the_evaluator_fills() -> None:
    gt_bytes, dt_bytes = _load(_TIDE_FIXTURES / _SEGM_FIXTURE)
    handle = CocoDataset.from_json(gt_bytes)
    Evaluator(iou=Segm()).evaluate(handle, dt_bytes)
    warmed = handle.segm_cache_len

    report = error_decomposition(handle, dt_bytes, iou=Segm())

    assert handle.segm_cache_len == warmed
    assert report == error_decomposition(gt_bytes, dt_bytes, iou=Segm())


def test_a_bad_manifest_fails_before_the_detections_are_read() -> None:
    """The manifest is canonicalized before the ``dt`` ingest (and before
    the grid pass), so a bad one never pays for either."""
    gt_bytes, _ = _load(_PARTITION_FIXTURE)
    refused_dt = cast("Any", np.zeros((3, 5)))
    missing = _PARTITION_FIXTURE / "absent.json"

    with pytest.raises(ValueError, match="manifest file read failed"):
        optimal_lrp(gt_bytes, refused_dt, iou=Bbox(), manifest=missing)
    with pytest.raises(ValueError, match="manifest file read failed"):
        Evaluator(iou=Bbox()).evaluate(gt_bytes, refused_dt, manifest=missing)


# ---------------------------------------------------------------------------
# LVIS federated ground truth
# ---------------------------------------------------------------------------


def _federated_handle() -> CocoDataset:
    return CocoDataset.from_lvis_json((_LVIS_FIXTURE / "gt.json").read_bytes())


def _partitioned_lrp(gt: Any, dt: Any, *, iou: Any) -> object:
    pytest.importorskip("polars", reason="`vernier[tables]` extra not installed")
    manifest = {
        "manifest_version": "1",
        "key_kind": "image_id",
        "rows": [{"key": 1, "split": "a"}, {"key": 2, "split": "b"}],
    }
    return optimal_lrp(gt, dt, iou=iou, manifest=manifest)


@pytest.mark.parametrize("iou", [Bbox(), Segm(), Boundary()], ids=["bbox", "segm", "boundary"])
@pytest.mark.parametrize(
    "call",
    [
        pytest.param(error_decomposition, id="error_decomposition"),
        pytest.param(fp_iou_histogram, id="fp_iou_histogram"),
        pytest.param(optimal_lrp, id="optimal_lrp"),
        pytest.param(_partitioned_lrp, id="optimal_lrp_manifest"),
        pytest.param(confusion_matrix, id="confusion_matrix"),
    ],
)
def test_federated_ground_truth_is_refused(call: Any, iou: object) -> None:
    """Every diagnostic entry point, per kernel: each is its own FFI call site."""
    dt_bytes = (_LVIS_FIXTURE / "dt.json").read_bytes()
    with pytest.raises(NotImplementedError, match="federated"):
        call(_federated_handle(), dt_bytes, iou=iou)


def test_a_flat_handle_over_the_same_annotations_is_not_refused() -> None:
    """The refusal is about the federated metadata, which
    ``CocoDataset.from_json`` drops."""
    gt_bytes, dt_bytes = _load(_LVIS_FIXTURE)
    flat = CocoDataset.from_json(gt_bytes)
    assert not flat.is_federated

    from_pair = error_decomposition(flat, _dt_matrix(dt_bytes), iou=Bbox())
    assert from_pair == error_decomposition(gt_bytes, dt_bytes, iou=Bbox())


def test_every_evaluate_branch_applies_the_federated_trim() -> None:
    """ADR-0026 AC2 caps detections per image, across categories, on every
    ``Evaluator.evaluate`` branch — plain, ``tables=`` and ``manifest=``.

    100 high-scoring category-1 false positives crowd the only category-2
    true positive out of the image's top 100, so an untrimmed path scores
    category 2 at AP 1 and a trimmed one at AP 0.
    """
    pytest.importorskip("polars", reason="`vernier[tables]` extra not installed")
    image = {"id": 1, "width": 100, "height": 100}
    gt = {
        "images": [{**image, "neg_category_ids": [], "not_exhaustive_category_ids": []}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "area": 100},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [50, 50, 10, 10], "area": 100},
        ],
        "categories": [
            {"id": 1, "name": "a", "frequency": "f"},
            {"id": 2, "name": "b", "frequency": "f"},
        ],
    }
    fps = [[1, 80, 80, 5, 5, 0.99 - i * 1e-3, 1] for i in range(100)]
    dt = np.asarray([*fps, [1, 50, 50, 10, 10, 0.01, 2]], dtype=np.float64)
    handle = CocoDataset.from_lvis_json(json.dumps(gt).encode())
    manifest = {"manifest_version": "1", "key_kind": "image_id", "rows": [{"key": 1, "s": "x"}]}
    ev = Evaluator(iou=Bbox())

    plain = ev.evaluate(handle, dt)
    tabled = ev.evaluate(handle, dt, tables=("per_class",)).summary
    sliced = ev.evaluate(handle, dt, manifest=manifest).summary

    assert tabled is not None
    assert sliced is not None
    assert plain.stats == tabled.stats == sliced.stats
    assert plain.stats == ev.evaluate(handle, dt[:100]).stats
