"""``bench compare`` end-to-end across a synthetic two-tree input.

Builds two ``results/<sha>/<fp>/<workload>/<iou>/<impl>.json`` trees,
runs the loader → compare → render pipeline, and asserts the rendered
markdown contains the expected per-cell rows.
"""

from __future__ import annotations

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
from bench.reports.compare import compare_shas
from bench.reports.load import load_tree
from bench.reports.render import render_compare_markdown

_FP = "fp0001234567"


def _stages(total_ns: int) -> dict[str, StageTimings]:
    return {"total": StageTimings(wall_ns=total_ns)}


def _aggregation(total_ns: int, *, rss_bytes: int) -> Aggregation:
    return Aggregation(
        stages={
            "total": StageAggregation(
                median_ns=total_ns, iqr_ns=0, min_ns=total_ns, max_ns=total_ns
            )
        },
        memory=MemoryAggregation(median_bytes=rss_bytes, min_bytes=rss_bytes, max_bytes=rss_bytes),
    )


def _write_result(
    root: Path,
    *,
    git_sha: str,
    workload: str,
    iou: IouType,
    impl: str,
    total_ns: int,
    rss_bytes: int = 100 * 1024 * 1024,
) -> Path:
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
                stages=_stages(total_ns),
                summary_stats={},
                ru_maxrss_bytes=rss_bytes,
                parent_wall_ns=total_ns,
            )
        ],
        aggregation=_aggregation(total_ns, rss_bytes=rss_bytes),
        artifact_paths={"tensor": f"{impl}.npy"},
        artifact_sha256={"tensor": "0" * 64},
    )
    json_path = out_dir / f"{impl}.json"
    json_path.write_text(result.model_dump_json())
    return json_path


@pytest.fixture
def synthetic_tree(tmp_path: Path) -> Path:
    base, head = "aaaaaaaa1111", "bbbbbbbb2222"
    # vernier 5% faster on head; pycocotools 10% slower.
    _write_result(
        tmp_path,
        git_sha=base,
        workload="smoke",
        iou="bbox",
        impl="vernier",
        total_ns=1_000_000,
        rss_bytes=200 * 1024 * 1024,
    )
    _write_result(
        tmp_path,
        git_sha=head,
        workload="smoke",
        iou="bbox",
        impl="vernier",
        total_ns=950_000,
        rss_bytes=210 * 1024 * 1024,
    )
    _write_result(
        tmp_path,
        git_sha=base,
        workload="smoke",
        iou="bbox",
        impl="pycocotools",
        total_ns=5_000_000,
    )
    _write_result(
        tmp_path,
        git_sha=head,
        workload="smoke",
        iou="bbox",
        impl="pycocotools",
        total_ns=5_500_000,
    )
    # Cell that exists only on head — must surface as head_only.
    _write_result(
        tmp_path,
        git_sha=head,
        workload="smoke",
        iou="bbox",
        impl="faster-coco-eval",
        total_ns=2_000_000,
    )
    return tmp_path


def test_compare_rows_have_expected_deltas(synthetic_tree: Path) -> None:
    df = load_tree(synthetic_tree)
    rows = compare_shas(df, base_sha="aaaaaaaa1111", head_sha="bbbbbbbb2222")
    by_impl = {r.key.impl: r for r in rows}

    assert by_impl["vernier"].status == "ok"
    assert by_impl["vernier"].delta_ns == -50_000
    assert by_impl["vernier"].delta_relative == pytest.approx(-0.05, abs=1e-9)

    assert by_impl["pycocotools"].status == "ok"
    assert by_impl["pycocotools"].delta_ns == 500_000
    assert by_impl["pycocotools"].delta_relative == pytest.approx(0.10, abs=1e-9)

    assert by_impl["faster-coco-eval"].status == "head_only"
    assert by_impl["faster-coco-eval"].delta_ns is None

    assert by_impl["vernier"].base_ru_maxrss_bytes == 200 * 1024 * 1024
    assert by_impl["vernier"].head_ru_maxrss_bytes == 210 * 1024 * 1024


def test_compare_markdown_renders_each_impl(synthetic_tree: Path) -> None:
    df = load_tree(synthetic_tree)
    rows = compare_shas(df, base_sha="aaaaaaaa1111", head_sha="bbbbbbbb2222")
    md = render_compare_markdown(rows, base_sha="aaaaaaaa1111", head_sha="bbbbbbbb2222")
    assert "aaaaaaaa1111" in md
    assert "bbbbbbbb2222" in md
    assert "| vernier |" in md
    assert "| pycocotools |" in md
    assert "| faster-coco-eval |" in md
    assert "▼" in md
    assert "▲" in md
    assert "-5.00%" in md
    assert "+10.00%" in md
    assert "RAM base" in md
    assert "RAM head" in md
    assert "200.0 MiB" in md
    assert "210.0 MiB" in md


def test_compare_empty_when_neither_sha_present(tmp_path: Path) -> None:
    df = load_tree(tmp_path)
    rows = compare_shas(df, base_sha="0" * 12, head_sha="1" * 12)
    assert rows == []
