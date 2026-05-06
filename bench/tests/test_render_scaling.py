"""Scaling renderer — table layout + log-log SVG structure."""

from __future__ import annotations

import polars as pl

from bench.reports.render import (
    render_scaling_svg,
    render_scaling_table,
)
from bench.reports.scaling import (
    ScalingPoint,
    group_by_synthetic_param,
    parse_synthetic_param,
)


def _make_curve(*, slope: int, x_values: list[int]) -> list[ScalingPoint]:
    """Linear curve so the test has a predictable median series."""
    return [
        ScalingPoint(
            x_value=float(x),
            median_ns=slope * x,
            iqr_ns=slope * x // 100,
            ru_maxrss_bytes=150 * 1024 * 1024,
        )
        for x in x_values
    ]


def test_scaling_table_renders_one_row_per_impl_x_value() -> None:
    points = {
        "vernier": _make_curve(slope=100, x_values=[1000, 10000, 50000]),
        "pycocotools": _make_curve(slope=1500, x_values=[1000, 10000, 50000]),
    }
    md = render_scaling_table(
        points,
        x_param="n_images",
        iou_type="bbox",
        workload_family="synthetic_c80_g10_d30_s0",
    )
    assert "synthetic_c80_g10_d30_s0" in md
    assert "| impl | n_images | median |" in md
    data_lines = [ln for ln in md.splitlines() if ln.startswith(("| vernier |", "| pycocotools |"))]
    assert len(data_lines) == 6
    assert "10k" in md
    assert "50k" in md


def test_scaling_table_vs_vernier_ratio_uses_per_x_denominator() -> None:
    points = {
        "vernier": [
            ScalingPoint(x_value=1000.0, median_ns=100, iqr_ns=1, ru_maxrss_bytes=None),
            ScalingPoint(x_value=10000.0, median_ns=1000, iqr_ns=10, ru_maxrss_bytes=None),
        ],
        "pycocotools": [
            ScalingPoint(x_value=1000.0, median_ns=1500, iqr_ns=15, ru_maxrss_bytes=None),
            ScalingPoint(x_value=10000.0, median_ns=20000, iqr_ns=200, ru_maxrss_bytes=None),
        ],
    }
    md = render_scaling_table(
        points,
        x_param="n_images",
        iou_type="bbox",
        workload_family="fam",
    )
    assert "15.00x" in md
    assert "20.00x" in md
    assert "1.00x" in md


def test_scaling_table_handles_empty_input() -> None:
    md = render_scaling_table(
        {},
        x_param="n_images",
        iou_type="bbox",
        workload_family="fam",
    )
    assert "Scaling — fam / bbox" in md
    assert "No matching cells" in md


def test_scaling_svg_emits_one_polyline_per_impl() -> None:
    points = {
        "vernier": _make_curve(slope=100, x_values=[1000, 10000, 100000]),
        "pycocotools": _make_curve(slope=1500, x_values=[1000, 10000, 100000]),
    }
    svg = render_scaling_svg(points, x_param="n_images")
    assert svg.startswith("<svg")
    assert svg.count("<polyline") == 2
    assert "n_images (log)" in svg
    assert ">vernier<" in svg
    assert ">pycocotools<" in svg


def test_scaling_svg_handles_empty_input() -> None:
    svg = render_scaling_svg({}, x_param="n_images")
    assert svg.startswith("<svg")
    assert "no data" in svg
    assert "<polyline" not in svg


def test_scaling_svg_log_axes_place_vernier_below_pycocotools() -> None:
    points = {
        "vernier": _make_curve(slope=100, x_values=[1000, 100000]),
        "pycocotools": _make_curve(slope=10000, x_values=[1000, 100000]),
    }
    svg = render_scaling_svg(points, x_param="n_images")
    import re

    polylines = re.findall(r'<polyline[^>]*points="([^"]+)"', svg)
    assert len(polylines) == 2

    def first_y(coords: str) -> float:
        first = coords.split(" ", 1)[0]
        return float(first.split(",", 1)[1])

    pyc_first_y = first_y(polylines[0])
    ver_first_y = first_y(polylines[1])
    # SVG y grows downward, so faster (smaller median) plots LOWER on screen.
    assert ver_first_y > pyc_first_y


def test_parse_synthetic_param_pulls_each_axis() -> None:
    wid = "synthetic_n10000_c80_g10_d30_s0"
    assert parse_synthetic_param(wid, "n_images") == 10000
    assert parse_synthetic_param(wid, "n_categories") == 80
    assert parse_synthetic_param(wid, "gt_per_image") == 10
    assert parse_synthetic_param(wid, "dt_per_image") == 30
    assert parse_synthetic_param(wid, "seed") == 0


def test_parse_synthetic_param_handles_iscrowd_suffix() -> None:
    wid = "synthetic_n2000_c80_g20_d30_x50_s0"
    assert parse_synthetic_param(wid, "n_images") == 2000
    assert parse_synthetic_param(wid, "dt_per_image") == 30
    assert parse_synthetic_param(wid, "seed") == 0


def test_parse_synthetic_param_returns_none_for_non_synthetic() -> None:
    assert parse_synthetic_param("smoke_perfect_match_segm", "n_images") is None


def _synth_row(*, wid: str, impl: str, median_ns: int, iou: str = "bbox") -> dict[str, object]:
    return {
        "git_sha": "deadbeef",
        "machine_fingerprint": "fp" * 8,
        "paradigm": "instance",
        "workload_id": wid,
        "iou_type": iou,
        "impl": impl,
        "impl_version": "0.0.1",
        "mode": "release",
        "reps_count": 10,
        "total_median_ns": median_ns,
        "total_iqr_ns": median_ns // 100,
        "ru_maxrss_median_bytes": 150 * 1024 * 1024,
        "ru_maxrss_max_bytes": 160 * 1024 * 1024,
        "tensor_sha256": "0" * 64,
        "mtime": 0.0,
    }


def test_group_by_synthetic_param_slices_matching_rows() -> None:
    df = pl.DataFrame(
        [
            _synth_row(wid="synthetic_n1000_c80_g10_d30_s0", impl="vernier", median_ns=1_000_000),
            _synth_row(wid="synthetic_n10000_c80_g10_d30_s0", impl="vernier", median_ns=10_000_000),
            _synth_row(wid="synthetic_n50000_c80_g10_d30_s0", impl="vernier", median_ns=50_000_000),
            _synth_row(
                wid="synthetic_n1000_c80_g10_d30_s0",
                impl="pycocotools",
                median_ns=15_000_000,
            ),
            _synth_row(
                wid="synthetic_n10000_c80_g20_d30_s0",
                impl="vernier",
                median_ns=99_000_000,
            ),
            _synth_row(
                wid="synthetic_n1000_c80_g10_d30_s0",
                impl="vernier",
                median_ns=99_000_000,
                iou="segm",
            ),
        ]
    )
    out = group_by_synthetic_param(
        df,
        vary="n_images",
        fix={"n_categories": 80, "gt_per_image": 10, "dt_per_image": 30, "seed": 0},
        iou_type="bbox",
    )
    assert set(out) == {"vernier", "pycocotools"}
    assert [p.x_value for p in out["vernier"]] == [1000.0, 10000.0, 50000.0]
    assert [p.median_ns for p in out["vernier"]] == [1_000_000, 10_000_000, 50_000_000]
    assert [p.x_value for p in out["pycocotools"]] == [1000.0]


def test_group_by_synthetic_param_empty_df_returns_empty() -> None:
    df = pl.DataFrame(schema={"workload_id": pl.Utf8})
    out = group_by_synthetic_param(
        df,
        vary="n_images",
        fix={"n_categories": 80},
        iou_type="bbox",
    )
    assert out == {}


def test_scaling_table_iqr_relative_to_median() -> None:
    points = {
        "vernier": [
            ScalingPoint(x_value=1000.0, median_ns=10000, iqr_ns=200, ru_maxrss_bytes=None),
        ],
    }
    md = render_scaling_table(points, x_param="n_images", iou_type="bbox", workload_family="fam")
    assert "2.00%" in md
