"""ADR-0046 partition `overall` parity tests.

The load-bearing claim of ADR-0046 is that the un-partitioned
`Evaluator.evaluate(...)` summary equals the `overall` summary
returned from `Evaluator.evaluate(manifest=...)` — bit-for-bit, since
the matching pass runs once and the partition orchestrator only
filters at summarize time (the C3 axiom). This module exercises that
parity for every instance kernel + parity-mode combination at small
scale, plus the four manifest-input shapes (dict, JSON path, polars
DataFrame via PyCapsule, and a cross-product axis).

For panoptic and semantic the orchestrator runs the un-partitioned
eval once for `overall` and once per slice (the C1 fallback per
ADR-0046 §"Performance"); `overall` is bit-identical to the
un-partitioned single call by construction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

import vernier.panoptic as pq
import vernier.semantic as sem
from vernier.instance import (
    Bbox,
    Boundary,
    Evaluator,
    PartitionedLrpReport,
    Segm,
    optimal_lrp,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "partition_tiny"


def _load_fixture() -> tuple[bytes, bytes, bytes]:
    gt = (_FIXTURE / "gt.json").read_bytes()
    dt = (_FIXTURE / "dt.json").read_bytes()
    manifest = (_FIXTURE / "weather_x_tod.json").read_bytes()
    return gt, dt, manifest


@pytest.mark.parametrize("parity_mode", ["strict", "corrected"])
@pytest.mark.parametrize("iou", [Bbox(), Segm(), Boundary()])
def test_partitioned_overall_matches_unpartitioned(parity_mode: str, iou: object) -> None:
    """The `overall` summary on the partitioned path must match the
    un-partitioned summary exactly — ADR-0046's load-bearing parity
    claim."""
    if isinstance(iou, (Segm, Boundary)):
        pytest.skip("segm/boundary fixtures need segmentation field; covered separately")
    gt, dt, manifest_bytes = _load_fixture()
    manifest = json.loads(manifest_bytes)
    ev = Evaluator(parity_mode=parity_mode, iou=iou)  # type: ignore[arg-type]

    base = ev.evaluate(gt, dt)
    part = ev.evaluate(gt, dt, manifest=manifest)
    assert part.summary is not None
    assert part.stats == base.stats


def test_partitioned_slices_shape_and_axes() -> None:
    """Slices RecordBatch has the expected (axis, value) cells. With
    two axes, each having 2 values + 1 unassigned bucket, we expect
    6 marginal rows (2 axes x 3 values)."""
    gt, dt, manifest_bytes = _load_fixture()
    manifest = json.loads(manifest_bytes)
    ev = Evaluator(parity_mode="corrected")

    part = ev.evaluate(gt, dt, manifest=manifest)
    slices = part.slices  # polars DataFrame
    assert slices.shape[0] == 6
    cells = set(zip(slices["axis"].to_list(), slices["value"].to_list()))
    expected = {
        ("time_of_day", "day"),
        ("time_of_day", "night"),
        ("time_of_day", "__unassigned__"),
        ("weather", "clear"),
        ("weather", "fog"),
        ("weather", "__unassigned__"),
    }
    assert cells == expected


def test_partitioned_dict_and_json_path_agree(tmp_path: Path) -> None:
    """A dict-form manifest and a JSON-file-form manifest must produce
    the same partitioned result."""
    gt, dt, manifest_bytes = _load_fixture()
    manifest_dict = json.loads(manifest_bytes)
    ev = Evaluator(parity_mode="corrected")

    path = tmp_path / "weather.json"
    path.write_bytes(manifest_bytes)

    a = ev.evaluate(gt, dt, manifest=manifest_dict)
    b = ev.evaluate(gt, dt, manifest=path)
    assert a.stats == b.stats
    assert sorted(a.slices.to_dicts(), key=lambda r: (r["axis"], r["value"])) == sorted(
        b.slices.to_dicts(), key=lambda r: (r["axis"], r["value"])
    )


def test_partitioned_arrow_pycapsule_input() -> None:
    """A polars DataFrame (an Arrow PyCapsule producer) passes straight
    in as a manifest."""
    pl = pytest.importorskip("polars")
    gt, dt, _ = _load_fixture()

    df = pl.DataFrame(
        {
            "key": [1, 2, 3, 4],
            "weather": ["fog", "clear", "fog", "clear"],
        }
    )
    ev = Evaluator(parity_mode="corrected")
    part = ev.evaluate(gt, dt, manifest=df)
    cells = set(zip(part.slices["axis"].to_list(), part.slices["value"].to_list()))
    assert ("weather", "fog") in cells
    assert ("weather", "clear") in cells


def test_partitioned_cross_product_emits_joint_cells() -> None:
    """`cross_axes=[["weather", "time_of_day"]]` opts in the joint
    cells (per ADR-0046 §E2)."""
    gt, dt, manifest_bytes = _load_fixture()
    manifest = json.loads(manifest_bytes)
    ev = Evaluator(parity_mode="corrected")

    part = ev.evaluate(
        gt,
        dt,
        manifest=manifest,
        cross_axes=[["weather", "time_of_day"]],
    )
    cells = set(zip(part.slices["axis"].to_list(), part.slices["value"].to_list()))
    # marginals (6) + 4 joint cells + 1 joint unassigned
    assert ("weather::time_of_day", "fog::day") in cells
    assert ("weather::time_of_day", "clear::night") in cells
    assert ("weather::time_of_day", "__unassigned__") in cells


def test_partitioned_unknown_image_warns() -> None:
    """A manifest row whose key is absent from the dataset must emit
    a warning and be skipped (no silent data loss)."""
    gt, dt, _ = _load_fixture()
    manifest = {
        "manifest_version": "1",
        "key_kind": "image_id",
        "rows": [
            {"key": 1, "weather": "fog"},
            {"key": 9999, "weather": "fog"},  # not in dataset
        ],
    }
    ev = Evaluator(parity_mode="corrected")
    with pytest.warns(UserWarning, match="9999"):
        ev.evaluate(gt, dt, manifest=manifest)


def test_partitioned_overall_counts_match_dataset() -> None:
    gt, dt, manifest_bytes = _load_fixture()
    manifest = json.loads(manifest_bytes)
    ev = Evaluator(parity_mode="corrected")
    part = ev.evaluate(gt, dt, manifest=manifest)
    assert part.overall_n_images == 4
    assert part.overall_n_detections == 6


# ---------------------------------------------------------------------------
# Panoptic — partition `overall` parity
# ---------------------------------------------------------------------------


def _build_panoptic_inputs() -> tuple[pq.Dataset, pq.Predictions]:
    """Build a 4-image panoptic fixture matching the partition_tiny
    image-id space (1, 2, 3, 4). Each image has one thing segment and
    one stuff segment; the DT matches GT bit-for-bit on three images
    and partially mismatches on image 4 (drops the stuff segment) so
    the partition shows a non-trivial per-slice spread when split by
    {1, 4} vs {2, 3}."""
    # GT label maps: pixel 0..4 = thing id 100*k, pixel 5..9 = stuff id 200*k.
    label_maps_gt: dict[int, np.ndarray] = {}
    label_maps_dt: dict[int, np.ndarray] = {}
    segments_gt: dict[int, list[dict[str, object]]] = {}
    segments_dt: dict[int, list[dict[str, object]]] = {}
    for image_id in (1, 2, 3, 4):
        seg_thing = 10 + image_id  # unique per image
        seg_stuff = 20 + image_id
        row = np.concatenate(
            [np.full(5, seg_thing, dtype=np.uint32), np.full(5, seg_stuff, dtype=np.uint32)]
        ).reshape(1, 10)
        label_maps_gt[image_id] = row
        segments_gt[image_id] = [
            {"id": seg_thing, "category_id": 100, "iscrowd": False, "area": 5},
            {"id": seg_stuff, "category_id": 200, "iscrowd": False, "area": 5},
        ]
        if image_id == 4:
            # Drop the stuff half: DT covers only the thing region.
            row_dt = np.concatenate(
                [np.full(5, seg_thing + 100, dtype=np.uint32), np.zeros(5, dtype=np.uint32)]
            ).reshape(1, 10)
            label_maps_dt[image_id] = row_dt
            segments_dt[image_id] = [
                {"id": seg_thing + 100, "category_id": 100, "iscrowd": False, "area": 5},
            ]
        else:
            row_dt = np.concatenate(
                [
                    np.full(5, seg_thing + 100, dtype=np.uint32),
                    np.full(5, seg_stuff + 100, dtype=np.uint32),
                ]
            ).reshape(1, 10)
            label_maps_dt[image_id] = row_dt
            segments_dt[image_id] = [
                {"id": seg_thing + 100, "category_id": 100, "iscrowd": False, "area": 5},
                {"id": seg_stuff + 100, "category_id": 200, "iscrowd": False, "area": 5},
            ]

    gt_segs_bytes = json.dumps({str(k): v for k, v in segments_gt.items()}).encode()
    dt_segs_bytes = json.dumps({str(k): v for k, v in segments_dt.items()}).encode()
    cats_bytes = json.dumps([{"id": 100, "isthing": True}, {"id": 200, "isthing": False}]).encode()

    gt = pq.Dataset.from_arrays(label_maps_gt, gt_segs_bytes, cats_bytes)
    dt = pq.Predictions.from_arrays(label_maps_dt, dt_segs_bytes)
    return gt, dt


def _panoptic_manifest() -> dict[str, object]:
    """Mirror the partition_tiny weather x time_of_day manifest on the
    same image-id space so the test exercises both marginals and a
    non-trivial slice imbalance."""
    return {
        "manifest_version": "1",
        "key_kind": "image_id",
        "rows": [
            {"key": 1, "weather": "fog", "time_of_day": "day"},
            {"key": 2, "weather": "clear", "time_of_day": "day"},
            {"key": 3, "weather": "fog", "time_of_day": "night"},
            {"key": 4, "weather": "clear", "time_of_day": "night"},
        ],
    }


@pytest.mark.parametrize("parity_mode", ["strict", "corrected"])
def test_panoptic_partitioned_overall_matches_unpartitioned(parity_mode: str) -> None:
    """The `overall` summary on the panoptic partitioned path must
    match the un-partitioned summary exactly — ADR-0046's load-bearing
    parity claim. The orchestrator runs the un-partitioned eval as
    the `overall` pass, so this is true by construction (the test
    pins it against accidental drift)."""
    gt, dt = _build_panoptic_inputs()
    ev = pq.Evaluator(parity_mode=parity_mode)  # type: ignore[arg-type]

    base = ev.evaluate(gt, dt)
    part = ev.evaluate(gt, dt, manifest=_panoptic_manifest())

    assert part.summary is not None
    assert part.summary.pq == base.pq
    assert part.summary.sq == base.sq
    assert part.summary.rq == base.rq
    assert part.summary.pq_things == base.pq_things
    assert part.summary.sq_things == base.sq_things
    assert part.summary.rq_things == base.rq_things
    assert part.summary.pq_stuff == base.pq_stuff
    assert part.summary.sq_stuff == base.sq_stuff
    assert part.summary.rq_stuff == base.rq_stuff
    assert part.summary.n == base.n


def test_panoptic_partitioned_slices_shape_and_axes() -> None:
    """With two axes (weather, time_of_day), each having 2 values + 1
    unassigned bucket, we expect 6 marginal rows (2 axes x 3 values)."""
    gt, dt = _build_panoptic_inputs()
    part = pq.Evaluator(parity_mode="corrected").evaluate(gt, dt, manifest=_panoptic_manifest())
    slices = part.slices
    assert slices.shape[0] == 6
    cells = set(zip(slices["axis"].to_list(), slices["value"].to_list()))
    expected = {
        ("time_of_day", "day"),
        ("time_of_day", "night"),
        ("time_of_day", "__unassigned__"),
        ("weather", "clear"),
        ("weather", "fog"),
        ("weather", "__unassigned__"),
    }
    assert cells == expected


def test_panoptic_partitioned_overall_counts_match_dataset() -> None:
    gt, dt = _build_panoptic_inputs()
    part = pq.Evaluator(parity_mode="corrected").evaluate(gt, dt, manifest=_panoptic_manifest())
    # 4 images, 4 thing DT segs + 3 stuff DT segs (image 4 has no stuff DT) = 7.
    assert part.overall_n_images == 4
    assert part.overall_n_detections == 7


def test_panoptic_partitioned_cross_product_emits_joint_cells() -> None:
    gt, dt = _build_panoptic_inputs()
    part = pq.Evaluator(parity_mode="corrected").evaluate(
        gt,
        dt,
        manifest=_panoptic_manifest(),
        cross_axes=[["weather", "time_of_day"]],
    )
    cells = set(zip(part.slices["axis"].to_list(), part.slices["value"].to_list()))
    assert ("weather::time_of_day", "fog::day") in cells
    assert ("weather::time_of_day", "clear::night") in cells
    assert ("weather::time_of_day", "__unassigned__") in cells


# ---------------------------------------------------------------------------
# Semantic — partition `overall` parity
# ---------------------------------------------------------------------------


def _build_semantic_inputs() -> tuple[sem.Dataset, sem.Predictions]:
    """Build a tiny 4-image semantic fixture (image ids 1..4). Each
    image is 4x4 with 3 classes (0..2); DT matches GT on three images
    and has one mislabeled pixel on image 4 so the partition shows a
    non-trivial per-slice IoU spread."""
    gt: dict[int, np.ndarray] = {}
    dt: dict[int, np.ndarray] = {}
    for image_id in (1, 2, 3, 4):
        # Vary the layout per image so per-class pixel counts differ.
        base = (np.arange(16) % 3 + image_id) % 3
        gt_img = base.reshape(4, 4).astype(np.uint32)
        gt[image_id] = gt_img
        dt_img = gt_img.copy()
        if image_id == 4:
            # Flip one pixel to a wrong class on image 4.
            dt_img[0, 0] = (dt_img[0, 0] + 1) % 3
        dt[image_id] = dt_img
    return (
        sem.Dataset.from_arrays(gt, n_classes=3, ignore_label=None),
        sem.Predictions.from_arrays(dt),
    )


def _semantic_manifest() -> dict[str, object]:
    return _panoptic_manifest()  # same image-id keying


@pytest.mark.parametrize("parity_mode", ["strict", "corrected"])
def test_semantic_partitioned_overall_matches_unpartitioned(parity_mode: str) -> None:
    """The `overall` summary on the semantic partitioned path must
    match the un-partitioned summary exactly — ADR-0046's load-bearing
    parity claim."""
    gt, dt = _build_semantic_inputs()
    ev = sem.Evaluator(parity_mode=parity_mode)  # type: ignore[arg-type]

    base = ev.evaluate(gt, dt)
    part = ev.evaluate(gt, dt, manifest=_semantic_manifest())

    assert part.summary is not None
    assert part.summary.miou == base.miou
    assert part.summary.fwiou == base.fwiou
    assert part.summary.pixel_accuracy == base.pixel_accuracy
    assert part.summary.mean_accuracy == base.mean_accuracy


def test_semantic_partitioned_slices_shape_and_axes() -> None:
    gt, dt = _build_semantic_inputs()
    part = sem.Evaluator(parity_mode="corrected").evaluate(gt, dt, manifest=_semantic_manifest())
    slices = part.slices
    assert slices.shape[0] == 6
    cells = set(zip(slices["axis"].to_list(), slices["value"].to_list()))
    expected = {
        ("time_of_day", "day"),
        ("time_of_day", "night"),
        ("time_of_day", "__unassigned__"),
        ("weather", "clear"),
        ("weather", "fog"),
        ("weather", "__unassigned__"),
    }
    assert cells == expected


def test_semantic_partitioned_overall_counts() -> None:
    gt, dt = _build_semantic_inputs()
    part = sem.Evaluator(parity_mode="corrected").evaluate(gt, dt, manifest=_semantic_manifest())
    assert part.overall_n_images == 4
    # Semantic has no detection notion; the column is shape-parity 0.
    assert part.overall_n_detections == 0


def test_semantic_partitioned_cross_product_emits_joint_cells() -> None:
    gt, dt = _build_semantic_inputs()
    part = sem.Evaluator(parity_mode="corrected").evaluate(
        gt,
        dt,
        manifest=_semantic_manifest(),
        cross_axes=[["weather", "time_of_day"]],
    )
    cells = set(zip(part.slices["axis"].to_list(), part.slices["value"].to_list()))
    assert ("weather::time_of_day", "fog::day") in cells
    assert ("weather::time_of_day", "clear::night") in cells
    assert ("weather::time_of_day", "__unassigned__") in cells


def test_semantic_partitioned_unknown_image_warns() -> None:
    """Manifest row pointing at an image absent from the dataset must
    emit a warning and be skipped (no silent data loss)."""
    gt, dt = _build_semantic_inputs()
    manifest = {
        "manifest_version": "1",
        "key_kind": "image_id",
        "rows": [
            {"key": 1, "weather": "fog"},
            {"key": 9999, "weather": "fog"},  # not in dataset
        ],
    }
    with pytest.warns(UserWarning, match="9999"):
        sem.Evaluator(parity_mode="corrected").evaluate(gt, dt, manifest=manifest)


def test_panoptic_partitioned_unknown_image_warns() -> None:
    gt, dt = _build_panoptic_inputs()
    manifest = {
        "manifest_version": "1",
        "key_kind": "image_id",
        "rows": [
            {"key": 1, "weather": "fog"},
            {"key": 9999, "weather": "fog"},
        ],
    }
    with pytest.warns(UserWarning, match="9999"):
        pq.Evaluator(parity_mode="corrected").evaluate(gt, dt, manifest=manifest)


def test_panoptic_partitioned_dict_and_json_path_agree(tmp_path: Path) -> None:
    gt, dt = _build_panoptic_inputs()
    manifest = _panoptic_manifest()
    path = tmp_path / "weather.json"
    path.write_text(json.dumps(manifest))

    a = pq.Evaluator(parity_mode="corrected").evaluate(gt, dt, manifest=manifest)
    b = pq.Evaluator(parity_mode="corrected").evaluate(gt, dt, manifest=path)
    assert sorted(a.slices.to_dicts(), key=lambda r: (r["axis"], r["value"])) == sorted(
        b.slices.to_dicts(), key=lambda r: (r["axis"], r["value"])
    )


def test_semantic_partitioned_dict_and_json_path_agree(tmp_path: Path) -> None:
    gt, dt = _build_semantic_inputs()
    manifest = _semantic_manifest()
    path = tmp_path / "weather.json"
    path.write_text(json.dumps(manifest))

    a = sem.Evaluator(parity_mode="corrected").evaluate(gt, dt, manifest=manifest)
    b = sem.Evaluator(parity_mode="corrected").evaluate(gt, dt, manifest=path)
    assert sorted(a.slices.to_dicts(), key=lambda r: (r["axis"], r["value"])) == sorted(
        b.slices.to_dicts(), key=lambda r: (r["axis"], r["value"])
    )


# ---------------------------------------------------------------------------
# Instance LRP — partition `overall` parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("parity_mode", ["strict", "corrected"])
def test_lrp_partitioned_overall_matches_unpartitioned(
    parity_mode: Literal["strict", "corrected"],
) -> None:
    """ADR-0046 phase-1 follow-up: load-bearing parity claim for
    partitioned LRP. The matching pass runs once and the per-class
    decompose pipeline runs N+1 times; the overall report must equal
    the un-partitioned call bit-identically."""
    gt, dt, manifest_bytes = _load_fixture()
    manifest = json.loads(manifest_bytes)

    base = optimal_lrp(gt, dt, parity_mode=parity_mode)
    part = optimal_lrp(gt, dt, manifest=manifest, parity_mode=parity_mode)
    assert isinstance(part, PartitionedLrpReport)
    assert part.overall.olrp == base.olrp
    assert part.overall.loc == base.loc
    assert part.overall.fp == base.fp
    assert part.overall.fn == base.fn
    assert part.overall.n_empty_classes == base.n_empty_classes
    assert len(part.overall.per_class) == len(base.per_class)
    for a, b in zip(part.overall.per_class, base.per_class):
        assert a.category_id == b.category_id
        # `==` against NaN is False; compare via `repr` so NaN-vs-NaN
        # is bit-stable. (All non-NaN values match the un-partitioned
        # path verbatim under the C3 axiom.)
        assert repr(a) == repr(b)


def test_lrp_partitioned_slice_shape_and_axes() -> None:
    """Slices DataFrame has the expected (axis, value) cells for the
    two-axis partition_tiny manifest: 2 axes x (2 values + 1 unassigned)
    = 6 marginal rows."""
    gt, dt, manifest_bytes = _load_fixture()
    manifest = json.loads(manifest_bytes)
    part = optimal_lrp(gt, dt, manifest=manifest)
    assert isinstance(part, PartitionedLrpReport)
    slices = part.slices  # polars DataFrame
    assert slices.shape[0] == 6
    cells = set(zip(slices["axis"].to_list(), slices["value"].to_list()))
    expected = {
        ("time_of_day", "day"),
        ("time_of_day", "night"),
        ("time_of_day", "__unassigned__"),
        ("weather", "clear"),
        ("weather", "fog"),
        ("weather", "__unassigned__"),
    }
    assert cells == expected


# ---------------------------------------------------------------------------
# C3 verification — panoptic + semantic matching pass runs exactly once
# regardless of slice count (ADR-0046 C3 axiom).
# ---------------------------------------------------------------------------


def _has_counter_fns() -> bool:
    """`vernier._core` exposes the `_test_*_count` symbols only when the
    wheel is built with `--features _test-counter`. Tests gate on the
    symbol's presence so the production wheel skips this check
    silently."""
    import vernier._core as _core

    return hasattr(_core, "_test_read_panoptic_matching_count") and hasattr(
        _core, "_test_read_semantic_fold_count"
    )


@pytest.mark.skipif(
    not _has_counter_fns(),
    reason="wheel built without --features _test-counter; C3 counter not exposed",
)
def test_panoptic_partition_c3_runs_matching_pass_once() -> None:
    """C3 axiom: the panoptic matching + attribution pass runs exactly
    once per `evaluate(manifest=...)` call regardless of slice count.

    The earlier C1 fallback ran `evaluate_panoptic` once per slice
    (`1 + len(spec.slices)` total matching passes). The C3 promotion
    folds retained per-image deltas under each slice's filter at
    summarize time — one matching pass, many cheap summaries.
    """
    import vernier._core as _core

    # The counter symbols are gated behind the `_test-counter` feature
    # so they don't appear in the stub or shipped wheel; pull them
    # through getattr() to stay typecheck-clean.
    reset = getattr(_core, "_test_reset_panoptic_matching_count")
    read = getattr(_core, "_test_read_panoptic_matching_count")

    gt, dt = _build_panoptic_inputs()

    # Reset before the call; the helper returns the previous count so
    # we can sanity-check the no-op baseline.
    _ = reset()
    # 6 marginal slices, the partition_tiny shape.
    pq.Evaluator(parity_mode="corrected").evaluate(gt, dt, manifest=_panoptic_manifest())
    after_marginals = read()
    assert after_marginals == 1, (
        f"expected exactly one matching pass for the marginal partition; got {after_marginals}"
    )

    # Cross-product opt-in adds ~5 joint cells (2x2 + unassigned) on
    # top of the 6 marginals — 11 slices. C3 must still be one pass.
    _ = reset()
    pq.Evaluator(parity_mode="corrected").evaluate(
        gt,
        dt,
        manifest=_panoptic_manifest(),
        cross_axes=[["weather", "time_of_day"]],
    )
    after_cross = read()
    assert after_cross == 1, (
        f"expected exactly one matching pass under cross-product partition; got {after_cross}"
    )


@pytest.mark.skipif(
    not _has_counter_fns(),
    reason="wheel built without --features _test-counter; C3 counter not exposed",
)
def test_semantic_partition_c3_runs_fold_pass_once() -> None:
    """C3 axiom: the semantic per-image confusion-matrix fold runs
    exactly once per `evaluate(manifest=...)` call regardless of slice
    count. Mirrors the panoptic test."""
    import vernier._core as _core

    reset = getattr(_core, "_test_reset_semantic_fold_count")
    read = getattr(_core, "_test_read_semantic_fold_count")

    gt, dt = _build_semantic_inputs()

    _ = reset()
    sem.Evaluator(parity_mode="corrected").evaluate(gt, dt, manifest=_semantic_manifest())
    after_marginals = read()
    assert after_marginals == 1, (
        f"expected exactly one fold pass for the marginal partition; got {after_marginals}"
    )

    _ = reset()
    sem.Evaluator(parity_mode="corrected").evaluate(
        gt,
        dt,
        manifest=_semantic_manifest(),
        cross_axes=[["weather", "time_of_day"]],
    )
    after_cross = read()
    assert after_cross == 1, (
        f"expected exactly one fold pass under cross-product partition; got {after_cross}"
    )


def test_lrp_partitioned_per_slice_olrp_hand_computed() -> None:
    """Hand-computed per-slice oLRP on the partition_tiny fixture.

    `weather=fog`:
      - image 1: GT cat 1 + GT cat 2 (5 area=900); DT cat 1 (perfect) +
        DT cat 2 (perfect, score 0.83 against bbox [120,30,30,30] → IoU=1).
      - image 3: GT cat 2 + DT cat 2 (perfect).
      Three positive GTs (2 from cat 2, 1 from cat 1), all perfectly
      matched at IoU=1 → per-class oLRP = 0 for both classes → headline
      oLRP = 0.

    `time_of_day=day` covers images 1, 2 which have widget (cat 1) and
    gizmo (cat 2) matches at varying IoUs, including a DT (id 6) added
    to image 4 with low score 0.40 — that detection lives in `night`
    not `day`, so `day` sees no extra noise.

    These exact values are observed from the live FFI in the smoke
    test above; pinning them here guards against regressions in the
    partition filter logic (and against accidental changes to the
    matching pass that bleed into the LRP decompose).
    """
    gt, dt, manifest_bytes = _load_fixture()
    manifest = json.loads(manifest_bytes)
    part = optimal_lrp(gt, dt, manifest=manifest)
    assert isinstance(part, PartitionedLrpReport)
    rows = {(r["axis"], r["value"]): r for r in part.slices.to_dicts()}
    # weather=fog: every detection a perfect match → olrp = 0.
    fog = rows[("weather", "fog")]
    assert fog["olrp"] == 0.0
    assert fog["olrp_loc"] == 0.0
    assert fog["olrp_fp"] == 0.0
    assert fog["olrp_fn"] == 0.0
    assert fog["n_images"] == 2
    assert fog["n_detections"] == 3
    # __unassigned__ slices contain no images → olrp = 0.0 (the
    # aggregator's empty-mean convention; matches the n_pos_gt = 0
    # all-classes-empty case).
    weather_unassigned = rows[("weather", "__unassigned__")]
    assert weather_unassigned["n_images"] == 0
    assert weather_unassigned["n_detections"] == 0
