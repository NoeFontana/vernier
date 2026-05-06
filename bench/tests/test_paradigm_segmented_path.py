"""ADR-0033 §"Paradigm-segmented result-store path".

Asserts that:
- ``result_dir`` emits the v2 path with the paradigm segment;
- the loader walks both v1 (no paradigm) and v2 (with paradigm) trees;
- a v1 result file's content reads back through the migration shim
  with paradigm="instance" populated.
"""

from __future__ import annotations

import json
from pathlib import Path

from bench.harness.orchestrate import result_dir
from bench.reports.load import load_tree


def test_result_dir_emits_paradigm_segment_for_instance() -> None:
    root = Path("/tmp/bench-test-results")
    out = result_dir(
        results_root=root,
        git_sha="abcdef123456",
        machine_fp="dev-unfp-m1",
        workload_id="smoke_perfect_match",
        iou_type="bbox",
    )
    # v2 layout: <root>/<sha>/<fp>/<paradigm>/<workload>/<metric>/
    expected = (
        root / "abcdef123456" / "dev-unfp-m1" / "instance" / "smoke_perfect_match" / "bbox"
    )
    assert out == expected


def test_result_dir_paradigm_kwarg_routes_panoptic_under_panoptic_segment() -> None:
    root = Path("/tmp/bench-test-results")
    out = result_dir(
        results_root=root,
        git_sha="abcdef123456",
        machine_fp="dev-unfp-m1",
        workload_id="coco_panoptic_val2017_perfect",
        iou_type="bbox",  # placeholder for the metric slot until B1 lands
        paradigm="panoptic",
    )
    assert out.parts[-3] == "panoptic"


def _v1_result_dict(*, workload: str, iou: str, impl: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "impl": impl,
        "impl_version": "0.0.1",
        "iou_type": iou,
        "workload_id": workload,
        "git_sha": "v1abcdef0001",
        "machine_fingerprint": "fp_v1",
        "harness_version": "0.0.0",
        "mode": "release",
        "run_seed": 0,
        "reps_count": 1,
        "warmup_discarded": 0,
        "reps": [
            {
                "rep": 0,
                "warmup": False,
                "stages": {"total": {"wall_ns": 1_000_000, "notes": []}},
                "summary_stats": {},
                "ru_maxrss_bytes": 50 * 1024 * 1024,
                "parent_wall_ns": 1_000_000,
            }
        ],
        "aggregation": None,
        "tensor_path": f"{impl}.npy",
        "tensor_sha256": "0" * 64,
        "warnings": [],
    }


def _v2_result_dict(*, workload: str, iou: str, impl: str, paradigm: str) -> dict[str, object]:
    return {
        "schema_version": 2,
        "paradigm": paradigm,
        "impl": impl,
        "impl_version": "0.0.1",
        "iou_type": iou,
        "workload_id": workload,
        "git_sha": "v2abcdef0002",
        "machine_fingerprint": "fp_v2",
        "harness_version": "0.0.0",
        "mode": "release",
        "run_seed": 0,
        "reps_count": 1,
        "warmup_discarded": 0,
        "reps": [
            {
                "rep": 0,
                "warmup": False,
                "stages": {"total": {"wall_ns": 2_000_000, "notes": []}},
                "summary_stats": {},
                "ru_maxrss_bytes": 50 * 1024 * 1024,
                "parent_wall_ns": 2_000_000,
            }
        ],
        "aggregation": None,
        "artifact_paths": {"tensor": f"{impl}.npy"},
        "artifact_sha256": {"tensor": "0" * 64},
        "warnings": [],
    }


def test_loader_walks_v1_tree_and_lifts_paradigm_to_instance(tmp_path: Path) -> None:
    """A v1-shaped result file (no paradigm segment) reads through the
    migration shim with ``paradigm="instance"`` populated."""
    v1_path = tmp_path / "v1abcdef0001" / "fp_v1" / "smoke" / "bbox" / "vernier.json"
    v1_path.parent.mkdir(parents=True, exist_ok=True)
    v1_path.write_text(json.dumps(_v1_result_dict(workload="smoke", iou="bbox", impl="vernier")))

    df = load_tree(tmp_path)
    assert not df.is_empty()
    assert df["paradigm"][0] == "instance"
    assert df["workload_id"][0] == "smoke"
    assert df["iou_type"][0] == "bbox"
    # v1's single tensor sha256 lifts under the canonical "tensor" slot
    # and surfaces back into the loader's flat column.
    assert df["tensor_sha256"][0] == "0" * 64


def test_loader_walks_v2_tree_under_paradigm_segment(tmp_path: Path) -> None:
    """A v2-shaped result file at the paradigm-segmented path reads
    back with the right paradigm column."""
    v2_path = (
        tmp_path
        / "v2abcdef0002"
        / "fp_v2"
        / "instance"
        / "smoke"
        / "bbox"
        / "vernier.json"
    )
    v2_path.parent.mkdir(parents=True, exist_ok=True)
    v2_path.write_text(
        json.dumps(
            _v2_result_dict(workload="smoke", iou="bbox", impl="vernier", paradigm="instance")
        )
    )

    df = load_tree(tmp_path)
    assert not df.is_empty()
    assert df["paradigm"][0] == "instance"
    assert df["workload_id"][0] == "smoke"


def test_loader_walks_mixed_v1_and_v2_trees(tmp_path: Path) -> None:
    """During the A-thick migration window, v1 cells still exist
    alongside freshly-emitted v2 cells. The loader walks both."""
    v1_path = tmp_path / "v1abcdef0001" / "fp_v1" / "smoke" / "bbox" / "vernier.json"
    v1_path.parent.mkdir(parents=True, exist_ok=True)
    v1_path.write_text(json.dumps(_v1_result_dict(workload="smoke", iou="bbox", impl="vernier")))

    v2_path = (
        tmp_path
        / "v2abcdef0002"
        / "fp_v2"
        / "instance"
        / "smoke"
        / "bbox"
        / "vernier.json"
    )
    v2_path.parent.mkdir(parents=True, exist_ok=True)
    v2_path.write_text(
        json.dumps(
            _v2_result_dict(workload="smoke", iou="bbox", impl="vernier", paradigm="instance")
        )
    )

    df = load_tree(tmp_path)
    assert df.height == 2, df
    assert set(df["paradigm"].to_list()) == {"instance"}
    # Both rows surface their (different) git_sha so the caller
    # can scope by sha downstream.
    assert set(df["git_sha"].to_list()) == {"v1abcdef0001", "v2abcdef0002"}
