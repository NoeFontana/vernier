"""ADR-0064: the `(CocoDataset, DetectionsInput)` pair on every instance surface.

ADR-0063 returns inputs rather than a metric so that every vernier
surface composes with one conversion. Five surfaces did not hold up
their end — TIDE, LRP, the confusion matrix, the FP-IoU histogram and
the ``tables=`` / ``manifest=`` paths each refused a
:class:`CocoDataset` and took detections only as JSON bytes. This
module is the evidence that they now do.

**Same result, not merely "it runs".** Each case builds the two
spellings of *one document* — a handle parsed from the GT bytes, and
an ``(N, 7)`` detection matrix laid out in the JSON array's own order
so the auto-id assignment of quirk **J1** lands identically — and
asserts the surface returns the same thing from both. A test that only
checked for the absence of an exception would pass while the handle
path quietly evaluated something else.

The refusal that remains is LVIS federated ground truth: the matching
pass would apply the AA3/AA4 branches while the ADR-0026 AC2 detection
trim, which lives on the grid path, would not. Half the semantics with
no oracle for the other half is the plausible-but-wrong number ADR-0057
rules out, so it raises. ``tables=`` / ``manifest=`` route through the
grid and are correspondingly *not* restricted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from vernier.instance import (
    Bbox,
    CocoDataset,
    Evaluator,
    confusion_matrix,
    error_decomposition,
    fp_iou_histogram,
    optimal_lrp,
)

_TIDE_FIXTURES = Path(__file__).parent / "oracle" / "tide" / "fixtures"
_PARTITION_FIXTURE = Path(__file__).parent / "parity" / "fixtures" / "partition_tiny"
_LVIS_FIXTURE = Path(__file__).parent / "parity_lvis" / "fixtures" / "federated_min"

#: Fixtures whose bbox detections exercise a different bin mix — a
#: clean diagonal, a class swap, a localization miss and a duplicate.
_CASES = ["all_perfect", "all_cls", "all_loc", "all_dupe"]


def _load(root: Path) -> tuple[bytes, bytes]:
    return (root / "gt.json").read_bytes(), (root / "dt.json").read_bytes()


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


def _pair(root: Path) -> tuple[CocoDataset, NDArray[np.float64]]:
    gt_bytes, dt_bytes = _load(root)
    return CocoDataset.from_json(gt_bytes), _dt_matrix(dt_bytes)


def _counts(df: object) -> dict[tuple[str, str], int]:
    rows = df.iter_rows(named=True)  # type: ignore[attr-defined]
    return {(r["gt_class"], r["dt_class"]): int(r["count"]) for r in rows}


def _nan_safe(value: float) -> object:
    """``NaN`` is LRP's "undefined" sentinel (a class with no TP has no
    deployable ``tau``), and ``nan != nan``, so a plain dataclass
    comparison reports a difference where the two reports agree. Map it
    to a token that compares equal to itself."""
    return "undefined" if np.isnan(value) else value


def _lrp_shape(report: Any) -> tuple[object, ...]:
    """Every field of an :class:`LrpReport`, NaN-normalized."""
    return (
        _nan_safe(report.olrp),
        _nan_safe(report.loc),
        _nan_safe(report.fp),
        _nan_safe(report.fn),
        report.n_empty_classes,
        report.config,
        tuple(
            (
                row.category_id,
                _nan_safe(row.olrp),
                _nan_safe(row.olrp_loc),
                _nan_safe(row.olrp_fp),
                _nan_safe(row.olrp_fn),
                _nan_safe(row.tau),
            )
            for row in report.per_class
        ),
    )


# ---------------------------------------------------------------------------
# The four standalone diagnostics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", _CASES)
def test_error_decomposition_reads_the_pair(fixture: str) -> None:
    gt_bytes, dt_bytes = _load(_TIDE_FIXTURES / fixture)
    handle, matrix = _pair(_TIDE_FIXTURES / fixture)

    from_bytes = error_decomposition(gt_bytes, dt_bytes, iou=Bbox())
    from_pair = error_decomposition(handle, matrix, iou=Bbox())

    assert from_pair == from_bytes


@pytest.mark.parametrize("fixture", _CASES)
def test_fp_iou_histogram_reads_the_pair(fixture: str) -> None:
    gt_bytes, dt_bytes = _load(_TIDE_FIXTURES / fixture)
    handle, matrix = _pair(_TIDE_FIXTURES / fixture)

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
    gt_bytes, dt_bytes = _load(_TIDE_FIXTURES / fixture)
    handle, matrix = _pair(_TIDE_FIXTURES / fixture)

    from_bytes = optimal_lrp(gt_bytes, dt_bytes, iou=Bbox())
    from_pair = optimal_lrp(handle, matrix, iou=Bbox())

    assert _lrp_shape(from_pair) == _lrp_shape(from_bytes)


@pytest.mark.parametrize("fixture", _CASES)
def test_confusion_matrix_reads_the_pair(fixture: str) -> None:
    pytest.importorskip("polars", reason="`vernier[tables]` extra not installed")
    gt_bytes, dt_bytes = _load(_TIDE_FIXTURES / fixture)
    handle, matrix = _pair(_TIDE_FIXTURES / fixture)

    from_bytes = _counts(confusion_matrix(gt_bytes, dt_bytes, iou=Bbox()))
    from_pair = _counts(confusion_matrix(handle, matrix, iou=Bbox()))

    assert from_pair == from_bytes
    assert from_bytes


# ---------------------------------------------------------------------------
# The two Evaluator.evaluate paths
# ---------------------------------------------------------------------------


def test_evaluate_tables_reads_the_pair() -> None:
    pytest.importorskip("polars", reason="`vernier[tables]` extra not installed")
    gt_bytes, dt_bytes = _load(_PARTITION_FIXTURE)
    handle, matrix = _pair(_PARTITION_FIXTURE)
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
    gt_bytes, dt_bytes = _load(_PARTITION_FIXTURE)
    handle, matrix = _pair(_PARTITION_FIXTURE)
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
    gt_bytes, dt_bytes = _load(_PARTITION_FIXTURE)
    handle, matrix = _pair(_PARTITION_FIXTURE)
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

    records: list[dict[str, Any]] = json.loads(dt_bytes)
    by_image: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        by_image.setdefault(int(record["image_id"]), []).append(record)
    columnar = [
        {
            "image_id": image_id,
            "boxes": np.asarray([r["bbox"] for r in dets], dtype=np.float64),
            "scores": np.asarray([r["score"] for r in dets], dtype=np.float64),
            "labels": np.asarray([r["category_id"] for r in dets], dtype=np.int64),
        }
        for image_id, dets in sorted(by_image.items())
    ]

    from_bytes = error_decomposition(gt_bytes, dt_bytes, iou=Bbox())
    from_columns = error_decomposition(handle, columnar, iou=Bbox())  # type: ignore[arg-type]

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
# The refusal that stays
# ---------------------------------------------------------------------------


def _federated_handle() -> CocoDataset:
    return CocoDataset.from_lvis_json((_LVIS_FIXTURE / "gt.json").read_bytes())


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(error_decomposition, id="error_decomposition"),
        pytest.param(fp_iou_histogram, id="fp_iou_histogram"),
        pytest.param(optimal_lrp, id="optimal_lrp"),
        pytest.param(confusion_matrix, id="confusion_matrix"),
    ],
)
def test_federated_ground_truth_is_refused(call: Any) -> None:
    dt_bytes = (_LVIS_FIXTURE / "dt.json").read_bytes()
    with pytest.raises(NotImplementedError, match="federated"):
        call(_federated_handle(), dt_bytes, iou=Bbox())


def test_a_flat_handle_over_the_same_annotations_is_not_refused() -> None:
    """The refusal is about the metadata, not about LVIS-shaped data:
    loading the same file through the COCO parser drops the federated
    extras, and then the diagnostics have nothing to be half-right
    about."""
    gt_bytes, dt_bytes = _load(_LVIS_FIXTURE)
    flat = CocoDataset.from_json(gt_bytes)
    assert not flat.is_federated

    from_pair = error_decomposition(flat, _dt_matrix(dt_bytes), iou=Bbox())
    assert from_pair == error_decomposition(gt_bytes, dt_bytes, iou=Bbox())
