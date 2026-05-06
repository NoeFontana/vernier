"""Forward migration tool — synthesize a fake v1 result tree, walk
``migrate_tree``, assert the v2 mirror exists with the right shape and
that re-running is a no-op (idempotency)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.harness.migrations.v1_to_v2_tree import migrate_tree


def _v1_result_dict(*, workload: str, iou: str, impl: str, sha: str) -> dict[str, object]:
    """Detection-shaped v1 ``BenchResult`` JSON dict."""
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


def _write_v1_cell(
    root: Path,
    *,
    sha: str,
    fp: str,
    workload: str,
    iou: str,
    impl: str,
) -> Path:
    """Lay down one v1 cell: result JSON + a placeholder tensor."""
    cell_dir = root / sha / fp / workload / iou
    cell_dir.mkdir(parents=True, exist_ok=True)
    json_path = cell_dir / f"{impl}.json"
    json_path.write_text(json.dumps(_v1_result_dict(workload=workload, iou=iou, impl=impl, sha=sha)))
    # 4 placeholder bytes — enough to round-trip a copy but cheap.
    (cell_dir / f"{impl}.npy").write_bytes(b"\x00\x00\x00\x00")
    return cell_dir


@pytest.fixture
def v1_tree(tmp_path: Path) -> Path:
    """3 sha × 1 fp × 2 workloads × 2 iou types — 12 cells total."""
    root = tmp_path / "results"
    for sha_idx in range(3):
        sha = f"sha{sha_idx:09d}aa"  # 12-char sha
        for workload in ("smoke", "synthetic_n1000_seed0"):
            for iou in ("bbox", "segm"):
                _write_v1_cell(
                    root,
                    sha=sha,
                    fp="fp_test",
                    workload=workload,
                    iou=iou,
                    impl="vernier",
                )
    return root


def test_migrate_tree_emits_v2_paradigm_segment(v1_tree: Path) -> None:
    stats = migrate_tree(v1_tree, keep_v1_paths=True)
    assert stats.cells_migrated == 12
    # Each v1 cell had 2 files (json + npy); 24 files copied at first run.
    assert stats.files_copied == 24

    # Every cell should now have a v2 mirror under ``instance/<workload>/<iou>``.
    for sha_idx in range(3):
        sha = f"sha{sha_idx:09d}aa"
        for workload in ("smoke", "synthetic_n1000_seed0"):
            for iou in ("bbox", "segm"):
                v2_dir = v1_tree / sha / "fp_test" / "instance" / workload / iou
                assert v2_dir.is_dir(), f"missing v2 mirror at {v2_dir}"
                assert (v2_dir / "vernier.json").is_file()
                assert (v2_dir / "vernier.npy").is_file()


def test_migrated_v2_json_carries_paradigm_field(v1_tree: Path) -> None:
    migrate_tree(v1_tree, keep_v1_paths=True)
    v2_json = (
        v1_tree / "sha000000000aa" / "fp_test" / "instance" / "smoke" / "bbox" / "vernier.json"
    )
    payload = json.loads(v2_json.read_text())
    assert payload["schema_version"] == 2
    assert payload["paradigm"] == "instance"
    assert payload["artifact_paths"] == {"tensor": "vernier.npy"}
    assert payload["artifact_sha256"] == {"tensor": "0" * 64}
    # The lifted v1 fields must not round-trip.
    assert "tensor_path" not in payload
    assert "tensor_sha256" not in payload


def test_migrate_tree_is_idempotent(v1_tree: Path) -> None:
    """A second migration run on an already-migrated tree must do
    nothing measurable: the cells stat as already-migrated and no
    new files are written."""
    migrate_tree(v1_tree, keep_v1_paths=True)
    stats = migrate_tree(v1_tree, keep_v1_paths=True)
    # Cells are still walked (the v1 directories still exist with
    # keep_v1_paths=True), but every per-file operation hits the
    # already-v2 path.
    assert stats.cells_migrated == 12
    assert stats.files_copied == 0
    assert stats.files_skipped_already_v2 == 24


def test_migrate_tree_handles_empty_root(tmp_path: Path) -> None:
    """An empty results root migrates to nothing without raising."""
    stats = migrate_tree(tmp_path / "does-not-exist", keep_v1_paths=True)
    assert stats.cells_migrated == 0
    assert stats.files_copied == 0


def test_migrate_tree_skips_intermediate_dir(tmp_path: Path) -> None:
    """``.intermediate/`` per-rep scratch is ignored by the migrator."""
    root = tmp_path / "results"
    cell_dir = _write_v1_cell(
        root,
        sha="sha000000000aa",
        fp="fp_test",
        workload="smoke",
        iou="bbox",
        impl="vernier",
    )
    intermediate = cell_dir / ".intermediate"
    intermediate.mkdir()
    (intermediate / "vernier-rep0.json").write_text("{}")
    (intermediate / "vernier-rep0.npy").write_bytes(b"\x00\x00\x00\x00")

    migrate_tree(root, keep_v1_paths=True)
    v2_dir = root / "sha000000000aa" / "fp_test" / "instance" / "smoke" / "bbox"
    assert v2_dir.is_dir()
    # ``.intermediate`` is not mirrored — it's per-rep scratch.
    assert not (v2_dir / ".intermediate").exists()


def test_migrate_tree_preserves_v1_paths_by_default(v1_tree: Path) -> None:
    """``keep_v1_paths=True`` (default) leaves v1 paths in place so the
    migration window is reversible."""
    migrate_tree(v1_tree, keep_v1_paths=True)
    # Pick one v1 cell at random — must still exist.
    v1_dir = v1_tree / "sha000000000aa" / "fp_test" / "smoke" / "bbox"
    assert v1_dir.is_dir()
    assert (v1_dir / "vernier.json").is_file()
