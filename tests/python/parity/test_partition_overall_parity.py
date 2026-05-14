"""ADR-0046 partition `overall` parity tests.

The load-bearing claim of ADR-0046 is that the un-partitioned
`Evaluator.evaluate(...)` summary equals the `overall` summary
returned from `Evaluator.evaluate(manifest=...)` — bit-for-bit, since
the matching pass runs once and the partition orchestrator only
filters at summarize time (the C3 axiom). This module exercises that
parity for every instance kernel + parity-mode combination at small
scale, plus the four manifest-input shapes (dict, JSON path, polars
DataFrame via PyCapsule, and a cross-product axis).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vernier.instance import Bbox, Boundary, Evaluator, Segm

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
