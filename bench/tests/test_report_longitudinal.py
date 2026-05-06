"""``bench report --since`` end-to-end: synthetic 30-day tree, assert the
SVG carries one polyline per series and the markdown lists every point."""

from __future__ import annotations

import itertools
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from bench.harness.orchestrate import result_dir
from bench.harness.schema import (
    Aggregation,
    BenchResult,
    IouType,
    MemoryAggregation,
    RepResult,
    StageAggregation,
    StageTimings,
)
from bench.reports.load import load_tree
from bench.reports.longitudinal import (
    build_series,
    filter_since,
    parse_since,
)
from bench.reports.render import (
    render_longitudinal_markdown,
    render_longitudinal_svg,
)

_FP = "fp9876543210"


def _write_point(
    root: Path,
    *,
    day: int,
    impl: str,
    total_ns: int,
    workload: str = "smoke",
    iou: IouType = "bbox",
    now: datetime,
) -> Path:
    git_sha = f"sha{day:09d}"
    out_dir = result_dir(
        results_root=root,
        git_sha=git_sha,
        machine_fp=_FP,
        workload_id=workload,
        iou_type=iou,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    result = BenchResult(
        paradigm="instance",
        impl=impl,
        impl_version="0.0.1",
        iou_type=iou,
        workload_id=workload,
        git_sha=git_sha,
        machine_fingerprint=_FP,
        harness_version="0.0.0",
        mode="release",
        run_seed=0,
        reps_count=1,
        warmup_discarded=0,
        reps=[
            RepResult(
                rep=0,
                warmup=False,
                stages={"total": StageTimings(wall_ns=total_ns)},
                summary_stats={},
                ru_maxrss_bytes=150 * 1024 * 1024,
                parent_wall_ns=total_ns,
            )
        ],
        aggregation=Aggregation(
            stages={
                "total": StageAggregation(
                    median_ns=total_ns,
                    iqr_ns=total_ns // 100,
                    min_ns=total_ns,
                    max_ns=total_ns,
                )
            },
            memory=MemoryAggregation(
                median_bytes=150 * 1024 * 1024,
                min_bytes=150 * 1024 * 1024,
                max_bytes=150 * 1024 * 1024,
            ),
        ),
        artifact_paths={"tensor": f"{impl}.npy"},
        artifact_sha256={"tensor": "0" * 64},
    )
    json_path = out_dir / f"{impl}.json"
    json_path.write_text(result.model_dump_json())
    # Stamp mtime so filter_since's window has something to bite on.
    ts = (now - timedelta(days=day)).timestamp()
    os.utime(json_path, (ts, ts))
    return json_path


@pytest.fixture
def thirty_day_tree(tmp_path: Path) -> tuple[Path, datetime]:
    now = datetime(2026, 5, 1, tzinfo=timezone.utc)
    for day in range(30):
        _write_point(
            tmp_path,
            day=day,
            impl="vernier",
            total_ns=900_000 + day * 1_000,
            now=now,
        )
        _write_point(
            tmp_path,
            day=day,
            impl="pycocotools",
            total_ns=5_000_000 - day * 5_000,
            now=now,
        )
    return tmp_path, now


def test_parse_since_units() -> None:
    assert parse_since("30d") == timedelta(days=30)
    assert parse_since("6h") == timedelta(hours=6)
    assert parse_since("2w") == timedelta(weeks=2)
    assert parse_since("90m") == timedelta(minutes=90)


def test_parse_since_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="--since"):
        parse_since("yesterday")


def test_series_have_thirty_points_per_impl(thirty_day_tree: tuple[Path, datetime]) -> None:
    root, now = thirty_day_tree
    df = load_tree(root)
    df = filter_since(df, since=timedelta(days=31), now=now)
    series = build_series(df)
    assert len(series) == 2
    for points in series.values():
        assert len(points) == 30
        # Time-ordered, monotonic non-decreasing.
        assert all(a.timestamp <= b.timestamp for a, b in itertools.pairwise(points))


def test_filter_since_drops_old_rows(thirty_day_tree: tuple[Path, datetime]) -> None:
    root, now = thirty_day_tree
    df = load_tree(root)
    # 7-day window keeps days 0..7 inclusive (8 points per impl, 16 total).
    df = filter_since(df, since=timedelta(days=7), now=now)
    series = build_series(df)
    total_points = sum(len(p) for p in series.values())
    assert total_points == 16


def test_longitudinal_markdown_lists_every_point(thirty_day_tree: tuple[Path, datetime]) -> None:
    root, now = thirty_day_tree
    df = filter_since(load_tree(root), since=timedelta(days=31), now=now)
    md = render_longitudinal_markdown(build_series(df))
    assert md.count("## smoke / bbox / vernier") == 1
    assert md.count("## smoke / bbox / pycocotools") == 1
    assert "RAM (max RSS)" in md
    assert "150.0 MiB" in md
    # 2 impls x 30 points = 60 data rows; match the leading-pipe-then-date pattern
    # so section headers and blank lines don't get counted.
    data_row_re = re.compile(r"^\| \d{4}-\d{2}-\d{2} ", re.MULTILINE)
    assert len(data_row_re.findall(md)) == 60


def test_longitudinal_svg_has_one_polyline_per_series(
    thirty_day_tree: tuple[Path, datetime],
) -> None:
    root, now = thirty_day_tree
    df = filter_since(load_tree(root), since=timedelta(days=31), now=now)
    svg = render_longitudinal_svg(build_series(df))
    assert svg.startswith("<svg")
    assert svg.count("<polyline") == 2
    assert "vernier-bench longitudinal" in svg
    assert "smoke/bbox/vernier" in svg
    assert "smoke/bbox/pycocotools" in svg


def test_longitudinal_svg_handles_empty_input() -> None:
    svg = render_longitudinal_svg({})
    assert svg.startswith("<svg")
    assert "no data" in svg
    assert "<polyline" not in svg
