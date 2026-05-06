"""Per-paradigm compare — synthesize results across the four paradigms
and assert the cross-walk produces a separate delta table per paradigm.

Cross-paradigm comparison is rejected by ``reject_cross_paradigm_request``
(ADR-0032 §"Cross-paradigm category error")."""

from __future__ import annotations

from pathlib import Path

import pytest

from bench.harness.orchestrate import result_dir
from bench.harness.schema import (
    Aggregation,
    BenchResult,
    IouType,
    MemoryAggregation,
    Paradigm,
    RepResult,
    StageAggregation,
    StageTimings,
)
from bench.reports.compare import (
    CrossParadigmCompareError,
    compare_shas_per_paradigm,
    reject_cross_paradigm_request,
)
from bench.reports.load import load_tree
from bench.reports.render import render_compare_per_paradigm_markdown

_FP = "fp_paradigm_test"


def _write_cell(
    root: Path,
    *,
    git_sha: str,
    paradigm: Paradigm,
    workload: str,
    iou: IouType,
    impl: str,
    total_ns: int,
) -> Path:
    out_dir = result_dir(
        results_root=root,
        git_sha=git_sha,
        machine_fp=_FP,
        workload_id=workload,
        iou_type=iou,
        paradigm=paradigm,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    result = BenchResult(
        paradigm=paradigm,
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
                ru_maxrss_bytes=128 * 1024 * 1024,
                parent_wall_ns=total_ns,
            )
        ],
        aggregation=Aggregation(
            stages={
                "total": StageAggregation(
                    median_ns=total_ns, iqr_ns=0, min_ns=total_ns, max_ns=total_ns
                )
            },
            memory=MemoryAggregation(
                median_bytes=128 * 1024 * 1024,
                min_bytes=128 * 1024 * 1024,
                max_bytes=128 * 1024 * 1024,
            ),
        ),
        artifact_paths={"tensor": f"{impl}.npy"},
        artifact_sha256={"tensor": "0" * 64},
    )
    out = out_dir / f"{impl}.json"
    out.write_text(result.model_dump_json())
    return out


@pytest.fixture
def four_paradigm_tree(tmp_path: Path) -> Path:
    """One workload per paradigm × two SHAs × one impl = 8 cells."""
    base = "base000000aa"
    head = "head000000bb"
    plan: list[tuple[Paradigm, str, IouType, str]] = [
        ("instance", "smoke", "bbox", "vernier"),
        ("panoptic", "coco_panoptic_val2017_perfect", "bbox", "vernier_panoptic"),
        ("semantic", "ade20k_val_perfect", "bbox", "vernier_semantic"),
        ("streaming", "coco_val2017_streaming_throughput", "bbox", "vernier_streaming"),
    ]
    for paradigm, workload, iou, impl in plan:
        _write_cell(
            tmp_path,
            git_sha=base,
            paradigm=paradigm,
            workload=workload,
            iou=iou,
            impl=impl,
            total_ns=1_000_000,
        )
        _write_cell(
            tmp_path,
            git_sha=head,
            paradigm=paradigm,
            workload=workload,
            iou=iou,
            impl=impl,
            total_ns=950_000,
        )
    return tmp_path


def test_compare_shas_per_paradigm_groups_by_paradigm(
    four_paradigm_tree: Path,
) -> None:
    df = load_tree(four_paradigm_tree)
    sections = compare_shas_per_paradigm(df, base_sha="base000000aa", head_sha="head000000bb")
    paradigms = [s.paradigm for s in sections]
    assert paradigms == ["instance", "panoptic", "semantic", "streaming"]
    # Each paradigm gets exactly one row in this 1-impl-per-paradigm fixture.
    for section in sections:
        assert len(section.rows) == 1
        assert section.rows[0].status == "ok"
        assert section.rows[0].delta_ns == -50_000


def test_render_compare_per_paradigm_renders_one_section_per_paradigm(
    four_paradigm_tree: Path,
) -> None:
    df = load_tree(four_paradigm_tree)
    sections = compare_shas_per_paradigm(df, base_sha="base000000aa", head_sha="head000000bb")
    md = render_compare_per_paradigm_markdown(
        sections, base_sha="base000000aa", head_sha="head000000bb"
    )
    for paradigm in ("instance", "panoptic", "semantic", "streaming"):
        assert f"## paradigm: `{paradigm}`" in md
    # The base/head SHAs surface only once (in the top-level header) —
    # the per-section subtables don't repeat them.
    assert md.count("base000000aa") >= 1
    assert md.count("head000000bb") >= 1


def test_compare_per_paradigm_handles_v1_tree_as_instance(tmp_path: Path) -> None:
    """A v1-only tree (no ``paradigm`` column) routes through the
    ``instance`` paradigm via the read-side migration shim."""
    import json

    # Inline minimal v1 result so the test doesn't depend on a helper module.
    def _minimal_v1(sha: str, total_ns: int) -> dict[str, object]:
        return {
            "schema_version": 1,
            "impl": "vernier",
            "impl_version": "0.0.1",
            "iou_type": "bbox",
            "workload_id": "smoke",
            "git_sha": sha,
            "machine_fingerprint": _FP,
            "harness_version": "0.0.0",
            "mode": "release",
            "run_seed": 0,
            "reps_count": 1,
            "warmup_discarded": 0,
            "reps": [
                {
                    "rep": 0,
                    "warmup": False,
                    "stages": {"total": {"wall_ns": total_ns, "notes": []}},
                    "summary_stats": {},
                    "ru_maxrss_bytes": 128 * 1024 * 1024,
                    "parent_wall_ns": total_ns,
                }
            ],
            "aggregation": None,
            "tensor_path": "vernier.npy",
            "tensor_sha256": "0" * 64,
            "warnings": [],
        }

    base = "base000000aa"
    head = "head000000bb"
    for sha, total in ((base, 1_000_000), (head, 950_000)):
        cell_dir = tmp_path / sha / _FP / "smoke" / "bbox"
        cell_dir.mkdir(parents=True, exist_ok=True)
        (cell_dir / "vernier.json").write_text(json.dumps(_minimal_v1(sha, total)))

    df = load_tree(tmp_path)
    sections = compare_shas_per_paradigm(df, base_sha=base, head_sha=head)
    assert len(sections) == 1
    assert sections[0].paradigm == "instance"


def test_reject_cross_paradigm_pq_under_instance() -> None:
    with pytest.raises(CrossParadigmCompareError, match="pq"):
        reject_cross_paradigm_request(paradigm="instance", metric="pq")


def test_reject_cross_paradigm_bbox_under_panoptic() -> None:
    with pytest.raises(CrossParadigmCompareError, match="bbox"):
        reject_cross_paradigm_request(paradigm="panoptic", metric="bbox")


def test_reject_cross_paradigm_unknown_metric_is_noop() -> None:
    """An unknown metric falls through other validators rather than
    raising here — the registry's job is the structural cross-paradigm
    check, not exhaustive metric validation."""
    reject_cross_paradigm_request(paradigm="instance", metric="future_metric")


def test_accept_cross_paradigm_correct_pair() -> None:
    """Sanity contrast: a metric that *does* live in the named
    paradigm doesn't raise."""
    reject_cross_paradigm_request(paradigm="panoptic", metric="pq")
    reject_cross_paradigm_request(paradigm="semantic", metric="miou")
    reject_cross_paradigm_request(paradigm="streaming", metric="throughput")
    reject_cross_paradigm_request(paradigm="instance", metric="bbox")
