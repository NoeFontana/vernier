"""Tests for the top-level :func:`vernier.aggregate` (ADR-0046 §G2).

Covers the cross-paradigm fan-in over partition manifests: Arrow inputs,
JSON-path inputs, JSON / CSV manifests, baseline rPC, metric filtering,
schema metadata, and the missing-label warning path.
"""

from __future__ import annotations

import json
import os
import subprocess
import warnings
from pathlib import Path

import pyarrow as pa
import pytest

import vernier
from vernier.aggregate import AggregateError, aggregate

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PARTITION_TINY_FIXTURE = (
    WORKSPACE_ROOT / "tests" / "python" / "parity" / "fixtures" / "partition_tiny"
)


def _make_slice_batch(label: str, ap: float, ap_50: float, ap_75: float) -> pa.RecordBatch:
    """Build a single-row 'overall' slice batch stamped with a vernier.label."""
    arrays = [
        pa.array(["overall"], type=pa.string()),
        pa.array(["overall"], type=pa.string()),
        pa.array([100], type=pa.uint64()),
        pa.array([200], type=pa.uint64()),
        pa.array([ap], type=pa.float64()),
        pa.array([ap_50], type=pa.float64()),
        pa.array([ap_75], type=pa.float64()),
    ]
    names = ["axis", "value", "n_images", "n_detections", "ap", "ap_50", "ap_75"]
    schema = pa.schema(
        [pa.field(n, a.type) for n, a in zip(names, arrays, strict=True)],
        metadata={b"vernier.label": label.encode("utf-8")},
    )
    return pa.RecordBatch.from_arrays(arrays, schema=schema)


@pytest.fixture
def three_runs() -> list[pa.RecordBatch]:
    return [
        _make_slice_batch("clean", ap=0.80, ap_50=0.90, ap_75=0.85),
        _make_slice_batch("fog", ap=0.40, ap_50=0.50, ap_75=0.45),
        _make_slice_batch("snow", ap=0.20, ap_50=0.25, ap_75=0.22),
    ]


@pytest.fixture
def manifest_dict() -> dict[str, object]:
    return {
        "manifest_version": "1",
        "key_kind": "result",
        "rows": [
            {"key": "clean", "weather": "clear"},
            {"key": "fog", "weather": "fog"},
            {"key": "snow", "weather": "snow"},
        ],
    }


def test_basic_aggregate_three_rows_no_rpc(
    three_runs: list[pa.RecordBatch],
    manifest_dict: dict[str, object],
) -> None:
    """Three single-axis runs → three rows, mean of one is the value itself, no rPC columns."""
    result = aggregate(three_runs, manifest_dict)
    assert isinstance(result, pa.RecordBatch)
    assert result.num_rows == 3
    # Deterministic order: axis asc, value asc → (weather, clear/fog/snow).
    assert result.column("axis").to_pylist() == ["weather", "weather", "weather"]
    assert result.column("value").to_pylist() == ["clear", "fog", "snow"]
    assert result.column("n_runs").to_pylist() == [1, 1, 1]
    assert result.column("ap").to_pylist() == pytest.approx([0.80, 0.40, 0.20])
    assert result.column("ap_50").to_pylist() == pytest.approx([0.90, 0.50, 0.25])
    assert result.column("ap_75").to_pylist() == pytest.approx([0.85, 0.45, 0.22])
    # No rPC columns when no baseline= is set.
    assert "ap__rpc" not in result.schema.names
    assert "ap_50__rpc" not in result.schema.names


def test_aggregate_with_baseline_emits_rpc_columns(
    three_runs: list[pa.RecordBatch],
    manifest_dict: dict[str, object],
) -> None:
    """``baseline='clear'`` adds ``<metric>__rpc`` columns with the right ratios."""
    result = aggregate(three_runs, manifest_dict, baseline="clear")
    # Original metric columns still present.
    assert "ap" in result.schema.names
    assert "ap_50" in result.schema.names
    # rPC columns appended, alphabetical w/ metric order.
    assert "ap__rpc" in result.schema.names
    assert "ap_50__rpc" in result.schema.names
    assert "ap_75__rpc" in result.schema.names
    # Ratios — clear row is 1.0; fog/snow are metric/baseline.
    rpc_ap = result.column("ap__rpc").to_pylist()
    rpc_ap_50 = result.column("ap_50__rpc").to_pylist()
    rpc_ap_75 = result.column("ap_75__rpc").to_pylist()
    assert rpc_ap == pytest.approx([1.0, 0.40 / 0.80, 0.20 / 0.80])
    assert rpc_ap_50 == pytest.approx([1.0, 0.50 / 0.90, 0.25 / 0.90])
    assert rpc_ap_75 == pytest.approx([1.0, 0.45 / 0.85, 0.22 / 0.85])


def test_manifest_as_json_path(
    three_runs: list[pa.RecordBatch],
    manifest_dict: dict[str, object],
    tmp_path: Path,
) -> None:
    """A path to a JSON manifest produces the same result as the dict."""
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest_dict), encoding="utf-8")
    via_dict = aggregate(three_runs, manifest_dict)
    via_path = aggregate(three_runs, str(path))
    # Compare column-by-column — pa.RecordBatch.equals doesn't compare
    # schema metadata by default but is fine for our purposes.
    assert via_dict.equals(via_path)


def test_manifest_as_csv_path(
    three_runs: list[pa.RecordBatch],
    manifest_dict: dict[str, object],
    tmp_path: Path,
) -> None:
    """A path to a CSV manifest with ``key,weather`` header produces the same result."""
    path = tmp_path / "manifest.csv"
    path.write_text(
        "key,weather\nclean,clear\nfog,fog\nsnow,snow\n",
        encoding="utf-8",
    )
    via_dict = aggregate(three_runs, manifest_dict)
    via_csv = aggregate(three_runs, str(path))
    assert via_dict.equals(via_csv)


def test_result_json_on_disk(
    manifest_dict: dict[str, object],
    tmp_path: Path,
) -> None:
    """A v2 result JSON on disk works as a results entry; label comes from
    the JSON ``label`` field."""
    # v2 result JSON with explicit label "fog" and a single overall row.
    doc_clean = {
        "version": "2",
        "label": "clean",
        "slices": [
            {
                "axis": "overall",
                "value": "overall",
                "n_images": 100,
                "n_detections": 200,
                "stats": {"ap": 0.80, "ap_50": 0.90, "ap_75": 0.85},
            }
        ],
    }
    doc_fog = {
        "version": "2",
        "label": "fog",
        "slices": [
            {
                "axis": "overall",
                "value": "overall",
                "n_images": 100,
                "n_detections": 200,
                "stats": {"ap": 0.40, "ap_50": 0.50, "ap_75": 0.45},
            }
        ],
    }
    doc_snow = {
        "version": "2",
        "label": "snow",
        "slices": [
            {
                "axis": "overall",
                "value": "overall",
                "n_images": 100,
                "n_detections": 200,
                "stats": {"ap": 0.20, "ap_50": 0.25, "ap_75": 0.22},
            }
        ],
    }
    p_clean = tmp_path / "clean.json"
    p_fog = tmp_path / "fog.json"
    p_snow = tmp_path / "snow.json"
    p_clean.write_text(json.dumps(doc_clean), encoding="utf-8")
    p_fog.write_text(json.dumps(doc_fog), encoding="utf-8")
    p_snow.write_text(json.dumps(doc_snow), encoding="utf-8")

    result = aggregate([str(p_clean), str(p_fog), str(p_snow)], manifest_dict)
    assert result.num_rows == 3
    assert result.column("value").to_pylist() == ["clear", "fog", "snow"]
    assert result.column("ap").to_pylist() == pytest.approx([0.80, 0.40, 0.20])
    assert result.column("ap_50").to_pylist() == pytest.approx([0.90, 0.50, 0.25])


def test_schema_metadata(
    three_runs: list[pa.RecordBatch],
    manifest_dict: dict[str, object],
) -> None:
    """The output schema carries the documented vernier.* metadata keys."""
    result = aggregate(three_runs, manifest_dict)
    meta = result.schema.metadata
    assert meta is not None
    assert meta.get(b"vernier.schema_version") == b"1"
    assert meta.get(b"vernier.table") == b"aggregate"


def test_missing_label_emits_warning_and_drops_run(
    three_runs: list[pa.RecordBatch],
) -> None:
    """A run whose label is absent from the manifest fires a warning and is dropped."""
    partial_manifest = {
        "manifest_version": "1",
        "key_kind": "result",
        "rows": [
            {"key": "clean", "weather": "clear"},
            {"key": "fog", "weather": "fog"},
            # No 'snow' row — the snow run should be dropped + warned.
        ],
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = aggregate(three_runs, partial_manifest)
    assert any("snow" in str(w.message) for w in caught), [str(w.message) for w in caught]
    # Two rows survive.
    assert result.num_rows == 2
    assert result.column("value").to_pylist() == ["clear", "fog"]


def test_metric_filter_selects_single_column(
    three_runs: list[pa.RecordBatch],
    manifest_dict: dict[str, object],
) -> None:
    """``metric='ap'`` keeps only the ``ap`` Float64 column."""
    result = aggregate(three_runs, manifest_dict, metric="ap")
    names = result.schema.names
    # axis, value, n_runs, ap — no ap_50 / ap_75.
    assert names == ["axis", "value", "n_runs", "ap"]


def test_aggregate_is_top_level_attribute() -> None:
    """``vernier.aggregate`` must be the function itself (re-export check)."""
    assert callable(vernier.aggregate)
    assert vernier.aggregate is aggregate


# ---------------------------------------------------------------------------
# Defensive-input coverage (beyond the spec's 9 tests, but cheap to assert)
# ---------------------------------------------------------------------------


def test_rejects_image_id_key_kind(
    three_runs: list[pa.RecordBatch],
) -> None:
    """A ``key_kind='image_id'`` manifest is rejected with a typed error."""
    bad = {
        "manifest_version": "1",
        "key_kind": "image_id",
        "rows": [{"key": 100, "weather": "fog"}],
    }
    with pytest.raises(AggregateError, match="key_kind"):
        aggregate(three_runs, bad)


def test_pre_resolved_manifest_dict(
    three_runs: list[pa.RecordBatch],
    manifest_dict: dict[str, object],
) -> None:
    """A bare ``{label: {axis: value}}`` dict is accepted as pre-resolved."""
    resolved = {
        "clean": {"weather": "clear"},
        "fog": {"weather": "fog"},
        "snow": {"weather": "snow"},
    }
    via_canonical = aggregate(three_runs, manifest_dict)
    via_resolved = aggregate(three_runs, resolved)
    assert via_canonical.equals(via_resolved)


def test_csv_manifest_with_bom(
    three_runs: list[pa.RecordBatch],
    tmp_path: Path,
) -> None:
    """A CSV manifest with a leading BOM (utf-8-sig) parses cleanly."""
    path = tmp_path / "manifest_bom.csv"
    # ﻿ is the UTF-8 BOM.
    path.write_text(
        "﻿key,weather\nclean,clear\nfog,fog\nsnow,snow\n",
        encoding="utf-8",
    )
    result = aggregate(three_runs, str(path))
    assert result.num_rows == 3
    assert result.column("value").to_pylist() == ["clear", "fog", "snow"]


def test_unknown_metric_raises() -> None:
    """An explicit ``metric=`` filter for an absent column raises AggregateError."""
    batch = _make_slice_batch("clean", ap=0.5, ap_50=0.6, ap_75=0.55)
    manifest = {"clean": {"weather": "clear"}}
    with pytest.raises(AggregateError, match="not present"):
        aggregate([batch], manifest, metric="does_not_exist")


def test_no_results_matched_manifest_raises() -> None:
    """All runs missing from the manifest → AggregateError, not silent empty output."""
    batch = _make_slice_batch("clean", ap=0.5, ap_50=0.6, ap_75=0.55)
    manifest = {"other_run": {"weather": "clear"}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(AggregateError, match="no results matched"):
            aggregate([batch], manifest)


def test_v1_result_json_rejected(tmp_path: Path) -> None:
    """A v1 result JSON path (no slices, version='1') is rejected with a clear error."""
    p = tmp_path / "v1.json"
    p.write_text(json.dumps({"version": "1", "stats": []}), encoding="utf-8")
    with pytest.raises(AggregateError, match="v2"):
        aggregate([str(p)], {"v1": {"weather": "clear"}})


def test_two_runs_same_axis_value_means_them(
    manifest_dict: dict[str, object],
) -> None:
    """Two runs assigned to the same axis value → mean across them, n_runs=2."""
    r1 = _make_slice_batch("fog", ap=0.40, ap_50=0.50, ap_75=0.45)
    r2 = _make_slice_batch("fog", ap=0.60, ap_50=0.70, ap_75=0.65)
    manifest = {
        "manifest_version": "1",
        "key_kind": "result",
        "rows": [{"key": "fog", "weather": "fog"}],
    }
    # Need to distinguish the two fog runs by label — give them
    # different labels and put both in the manifest.
    r1_clean = _relabel(r1, "fog_a")
    r2_clean = _relabel(r2, "fog_b")
    manifest_two = {
        "manifest_version": "1",
        "key_kind": "result",
        "rows": [
            {"key": "fog_a", "weather": "fog"},
            {"key": "fog_b", "weather": "fog"},
        ],
    }
    result = aggregate([r1_clean, r2_clean], manifest_two)
    assert result.num_rows == 1
    assert result.column("n_runs").to_pylist() == [2]
    assert result.column("ap").to_pylist() == pytest.approx([0.50])
    assert result.column("ap_50").to_pylist() == pytest.approx([0.60])
    # Unused fixture; here so the lint doesn't fire on `manifest`.
    _ = manifest_dict
    _ = manifest


def _relabel(batch: pa.RecordBatch, new_label: str) -> pa.RecordBatch:
    """Return a copy of ``batch`` with its ``vernier.label`` metadata replaced."""
    schema = batch.schema.with_metadata({b"vernier.label": new_label.encode("utf-8")})
    return pa.RecordBatch.from_arrays(
        [batch.column(i) for i in range(batch.num_columns)],
        schema=schema,
    )


# ---------------------------------------------------------------------------
# CLI v2 JSON shape coverage (the on-disk envelope produced by the Rust
# CLI's `vernier eval --manifest` lane).
# ---------------------------------------------------------------------------


def test_cli_v2_json_positional_stats_shape_unit() -> None:
    """A v2 doc with positional ``stats`` + ``lines`` deserialises to the
    CLI alias columns.

    Mirrors the CLI's `crates/vernier-cli/src/format/json.rs` envelope
    shape (and the canonical mapping in `commands/aggregate.rs`) so the
    test is bound to that contract rather than to a hand-built fixture.
    """

    # Build a single-slice v2 doc with the CLI's positional stats /
    # lines shape. Twelve lines mirror the canonical detection summary.
    def _line(
        metric: str,
        iou_label: str,
        area: str,
        max_dets: int,
        value: float,
    ) -> dict[str, object]:
        return {
            "metric": metric,
            "iou_threshold": None if iou_label == "0.50:0.95" else float(iou_label),
            "iou_threshold_label": iou_label,
            "area": area,
            "max_dets": max_dets,
            "value": value,
        }

    lines = [
        _line("AP", "0.50:0.95", "all", 100, 0.80),
        _line("AP", "0.50", "all", 100, 0.90),
        _line("AP", "0.75", "all", 100, 0.85),
        _line("AP", "0.50:0.95", "small", 100, 0.70),
        _line("AP", "0.50:0.95", "medium", 100, 0.75),
        _line("AP", "0.50:0.95", "large", 100, 0.95),
        _line("AR", "0.50:0.95", "all", 1, 0.60),
        _line("AR", "0.50:0.95", "all", 10, 0.65),
        _line("AR", "0.50:0.95", "all", 100, 0.68),
        _line("AR", "0.50:0.95", "small", 100, 0.55),
        _line("AR", "0.50:0.95", "medium", 100, 0.62),
        _line("AR", "0.50:0.95", "large", 100, 0.78),
    ]
    doc = {
        "version": "2",
        "label": "run_a",
        "iou_type": "bbox",
        "overall": {
            "lines": lines,
            "stats": [line["value"] for line in lines],
            "n_images": 4,
            "n_detections": 6,
        },
    }

    from vernier.aggregate import _v2_json_to_batch

    batch = _v2_json_to_batch(doc, source="<test>")
    schema_names = set(batch.schema.names)
    # The CLI aliases are present (pycocotools-style nicknames).
    assert {"ap", "ap50", "ap75", "ap_small", "ap_medium", "ap_large"} <= schema_names
    assert {"ar_1", "ar_10", "ar_100", "ar_small", "ar_medium", "ar_large"} <= schema_names
    # Canonical names are present too — both forms address the same cell.
    assert "AP_0.50:0.95_all_100" in schema_names
    assert "AR_0.50:0.95_all_1" in schema_names
    # Values resolve to the right line — `ap` is the AP[0.50:0.95]/all/100
    # tuple, value 0.80.
    assert batch.column("ap").to_pylist() == pytest.approx([0.80])
    assert batch.column("ap50").to_pylist() == pytest.approx([0.90])
    assert batch.column("ar_1").to_pylist() == pytest.approx([0.60])
    # Slice-level support counts surface at the row level.
    assert batch.column("n_images").to_pylist() == [4]
    assert batch.column("n_detections").to_pylist() == [6]


def test_aggregate_consumes_cli_v2_json_doc_directly(tmp_path: Path) -> None:
    """A v2 JSON doc with the CLI's positional ``stats`` shape round-trips
    through :func:`vernier.aggregate`.

    Skipping the subprocess: we write the JSON shape the CLI emits
    (verified by `crates/vernier-cli/tests/eval_manifest.rs`) directly,
    then aggregate over it. The end-to-end subprocess variant is
    :func:`test_aggregate_consumes_cli_v2_json_subprocess`.
    """

    def _line(
        metric: str,
        iou_label: str,
        area: str,
        max_dets: int,
        value: float,
    ) -> dict[str, object]:
        return {
            "metric": metric,
            "iou_threshold": None if iou_label == "0.50:0.95" else float(iou_label),
            "iou_threshold_label": iou_label,
            "area": area,
            "max_dets": max_dets,
            "value": value,
        }

    def _doc(label: str, ap: float, ap50: float, ap75: float) -> dict[str, object]:
        lines = [
            _line("AP", "0.50:0.95", "all", 100, ap),
            _line("AP", "0.50", "all", 100, ap50),
            _line("AP", "0.75", "all", 100, ap75),
        ]
        return {
            "version": "2",
            "label": label,
            "iou_type": "bbox",
            "overall": {
                "lines": lines,
                "stats": [line["value"] for line in lines],
                "n_images": 100,
                "n_detections": 200,
            },
        }

    p_clean = tmp_path / "clean.json"
    p_fog = tmp_path / "fog.json"
    p_clean.write_text(json.dumps(_doc("clean", 0.80, 0.90, 0.85)), encoding="utf-8")
    p_fog.write_text(json.dumps(_doc("fog", 0.40, 0.50, 0.45)), encoding="utf-8")

    rmanifest = {
        "manifest_version": "1",
        "key_kind": "result",
        "rows": [
            {"key": "clean", "weather": "clear"},
            {"key": "fog", "weather": "fog"},
        ],
    }
    result = aggregate([str(p_clean), str(p_fog)], rmanifest, baseline="clear")
    # Pycocotools-style aliases are first-class column names.
    assert "ap" in result.schema.names
    assert "ap50" in result.schema.names
    assert "ap75" in result.schema.names
    # Plus the canonical form (the CLI emits both).
    assert "AP_0.50:0.95_all_100" in result.schema.names
    # rPC ratios resolve.
    assert result.column("ap__rpc").to_pylist() == pytest.approx([1.0, 0.40 / 0.80])
    assert result.column("ap50__rpc").to_pylist() == pytest.approx([1.0, 0.50 / 0.90])
    assert result.column("value").to_pylist() == ["clear", "fog"]


@pytest.fixture(scope="session")
def vernier_bin() -> Path:
    """Locate (and on miss, build) the `vernier` CLI binary.

    Mirrors the bootstrap pattern in
    `tests/python/parity/test_cli.py::vernier_bin` so the two
    out-of-process suites agree on where the binary lives. Cargo build
    is incremental — repeated invocations on a clean tree are fast.
    """
    target_dir = Path(os.environ.get("CARGO_TARGET_DIR", str(WORKSPACE_ROOT / "target")))
    debug_binary = target_dir / "debug" / "vernier"
    release_binary = target_dir / "release" / "vernier"
    if release_binary.exists():
        return release_binary
    if debug_binary.exists():
        return debug_binary
    # Build once. Use --release for parity with how the wheel is
    # produced; the difference (debug vs release) does not affect the
    # JSON envelope shape.
    subprocess.run(
        ["cargo", "build", "--release", "-p", "vernier-cli"],
        cwd=WORKSPACE_ROOT,
        check=True,
    )
    assert release_binary.exists(), f"cargo build did not produce {release_binary}"
    return release_binary


@pytest.mark.slow
def test_aggregate_consumes_cli_v2_json_subprocess(
    vernier_bin: Path,
    tmp_path: Path,
) -> None:
    """End-to-end: ``vernier eval --manifest`` writes v2 JSON; ``vernier.aggregate``
    reads it.

    Pipes the real CLI output through the Python entry point — the
    test the CLI-shape compatibility gap was filed to close. Marked
    ``slow`` because the subprocess fork dominates the run time.
    """
    gt = PARTITION_TINY_FIXTURE / "gt.json"
    dt = PARTITION_TINY_FIXTURE / "dt.json"
    manifest = PARTITION_TINY_FIXTURE / "weather_x_tod.json"
    assert gt.exists()
    assert dt.exists()
    assert manifest.exists()

    out_a = tmp_path / "run_a.json"
    out_b = tmp_path / "run_b.json"
    for out, label in ((out_a, "run_a"), (out_b, "run_b")):
        result = subprocess.run(
            [
                str(vernier_bin),
                "eval",
                "--gt",
                str(gt),
                "--dt",
                str(dt),
                "--iou-type",
                "bbox",
                "--manifest",
                str(manifest),
                "--label",
                label,
                "--emit",
                f"json={out}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        # eval to stderr is fine; we just want the file written.
        assert out.exists(), f"CLI did not write {out}: stderr={result.stderr!r}"

    rmanifest = {
        "manifest_version": "1",
        "key_kind": "result",
        "rows": [
            {"key": "run_a", "weather": "clear"},
            {"key": "run_b", "weather": "fog"},
        ],
    }
    aggregated = aggregate([str(out_a), str(out_b)], rmanifest)
    # The AP column round-trips. We do not assert on numeric values
    # here — that is the parity suite's job; the CLI v2 → aggregate
    # shape compatibility is the contract this test pins.
    assert "ap" in aggregated.schema.names
    assert aggregated.num_rows == 2
    assert set(aggregated.column("value").to_pylist()) == {"clear", "fog"}
    # rPC requires an explicit baseline; without it, no __rpc columns.
    assert not any(n.endswith("__rpc") for n in aggregated.schema.names)


@pytest.mark.slow
def test_aggregate_consumes_cli_v2_json_subprocess_with_baseline(
    vernier_bin: Path,
    tmp_path: Path,
) -> None:
    """End-to-end with ``baseline=`` — verifies rPC columns get the right
    column names when the JSON shape is the CLI v2 envelope.
    """
    gt = PARTITION_TINY_FIXTURE / "gt.json"
    dt = PARTITION_TINY_FIXTURE / "dt.json"
    manifest = PARTITION_TINY_FIXTURE / "weather_x_tod.json"

    out_clean = tmp_path / "clean.json"
    out_fog = tmp_path / "fog.json"
    for out, label in ((out_clean, "clean"), (out_fog, "fog")):
        subprocess.run(
            [
                str(vernier_bin),
                "eval",
                "--gt",
                str(gt),
                "--dt",
                str(dt),
                "--iou-type",
                "bbox",
                "--manifest",
                str(manifest),
                "--label",
                label,
                "--emit",
                f"json={out}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    rmanifest = {
        "manifest_version": "1",
        "key_kind": "result",
        "rows": [
            {"key": "clean", "weather": "clear"},
            {"key": "fog", "weather": "fog"},
        ],
    }
    aggregated = aggregate(
        [str(out_clean), str(out_fog)],
        rmanifest,
        baseline="clear",
        metric="ap",
    )
    # `metric=ap` filtered output: axis, value, n_runs, ap, ap__rpc.
    assert aggregated.schema.names == ["axis", "value", "n_runs", "ap", "ap__rpc"]
    # The "clear" baseline row resolves to rPC=1.0 by construction.
    rpc_values = aggregated.column("ap__rpc").to_pylist()
    clear_idx = aggregated.column("value").to_pylist().index("clear")
    assert rpc_values[clear_idx] == pytest.approx(1.0)
