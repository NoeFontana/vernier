"""``--no-keep-v1-paths`` removes v1 cells once the v2 mirror exists."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.harness.migrations.v1_to_v2_tree import migrate_tree


def _v1_result_dict(*, workload: str, iou: str, impl: str, sha: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "impl": impl,
        "impl_version": "0.0.1",
        "iou_type": iou,
        "workload_id": workload,
        "git_sha": sha,
        "machine_fingerprint": "fp_test",
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


@pytest.fixture
def v1_tree(tmp_path: Path) -> Path:
    """Two cells under one sha — enough to exercise both removal and
    pruning of empty intermediate dirs."""
    root = tmp_path / "results"
    sha = "sha000000000aa"
    for iou in ("bbox", "segm"):
        cell_dir = root / sha / "fp_test" / "smoke" / iou
        cell_dir.mkdir(parents=True, exist_ok=True)
        (cell_dir / "vernier.json").write_text(
            json.dumps(_v1_result_dict(workload="smoke", iou=iou, impl="vernier", sha=sha))
        )
        (cell_dir / "vernier.npy").write_bytes(b"\x00\x00\x00\x00")
    return root


def test_no_keep_v1_paths_removes_old_directories(v1_tree: Path) -> None:
    stats = migrate_tree(v1_tree, keep_v1_paths=False)
    assert stats.v1_paths_removed == 2

    sha = "sha000000000aa"
    # v1 paths must be gone.
    assert not (v1_tree / sha / "fp_test" / "smoke" / "bbox").exists()
    assert not (v1_tree / sha / "fp_test" / "smoke" / "segm").exists()
    # v2 paths must exist.
    assert (v1_tree / sha / "fp_test" / "instance" / "smoke" / "bbox" / "vernier.json").is_file()
    assert (v1_tree / sha / "fp_test" / "instance" / "smoke" / "segm" / "vernier.json").is_file()
    # The empty ``smoke/`` parent must have been pruned (it's a v1
    # intermediate directory with no remaining children).
    assert not (v1_tree / sha / "fp_test" / "smoke").exists()


def test_keep_v1_paths_does_not_remove(v1_tree: Path) -> None:
    """Sanity contrast — default ``keep_v1_paths=True`` leaves v1
    paths in place. Pairs with the test above."""
    stats = migrate_tree(v1_tree, keep_v1_paths=True)
    assert stats.v1_paths_removed == 0

    sha = "sha000000000aa"
    assert (v1_tree / sha / "fp_test" / "smoke" / "bbox" / "vernier.json").is_file()
    assert (v1_tree / sha / "fp_test" / "smoke" / "segm" / "vernier.json").is_file()


def test_no_keep_v1_paths_is_idempotent(v1_tree: Path) -> None:
    """Re-running with ``keep_v1_paths=False`` after the v1 paths are
    already gone is a no-op (no errors, no further removals)."""
    migrate_tree(v1_tree, keep_v1_paths=False)
    stats = migrate_tree(v1_tree, keep_v1_paths=False)
    assert stats.cells_migrated == 0
    assert stats.v1_paths_removed == 0
