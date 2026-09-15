"""Per-cell CPU budget: selection, subprocess pinning, and the evidence
the runners record (``cpu_ns`` + the ``cpu_affinity`` note)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from bench.harness import cpu_affinity
from bench.harness.cpu_affinity import cpu_budget, select_cpus
from bench.harness.orchestrate import _spawn_subprocess
from bench.harness.paths import BENCH_ROOT
from bench.harness.timing import StageTable


def test_default_cell_budget_is_one_cpu() -> None:
    assert cpu_budget(None) == 1
    assert cpu_budget(4) == 4


@pytest.fixture
def smt_4c8t(monkeypatch: pytest.MonkeyPatch) -> set[int]:
    """4 physical cores, 2 hyperthreads each: cpu 2k and 2k+1 share core k."""
    monkeypatch.setattr(cpu_affinity, "_physical_core", lambda cpu: (0, cpu // 2))
    return set(range(8))


def test_select_spreads_across_physical_cores_before_smt(smt_4c8t: set[int]) -> None:
    four = select_cpus(4, smt_4c8t)
    assert len({cpu // 2 for cpu in four}) == 4, four
    eight = select_cpus(8, smt_4c8t)
    assert sorted(eight) == list(range(8))
    assert eight[:4] == four


def test_single_cpu_avoids_cpu0(smt_4c8t: set[int]) -> None:
    assert select_cpus(1, smt_4c8t) != (0,)


def test_select_rejects_budget_beyond_available(smt_4c8t: set[int]) -> None:
    with pytest.raises(ValueError, match="cannot pin 9"):
        select_cpus(9, smt_4c8t)
    with pytest.raises(ValueError, match="cannot pin 0"):
        select_cpus(0, smt_4c8t)


def test_spawned_child_inherits_pinned_cpus(tmp_path: Path) -> None:
    cpus = select_cpus(1)
    out = tmp_path / "affinity.txt"
    script = f"import os; open({str(out)!r}, 'w').write(repr(sorted(os.sched_getaffinity(0))))"
    status, _rusage, _wall = _spawn_subprocess(
        bench_root=BENCH_ROOT,
        impl="vernier",
        cmd=[sys.executable, "-c", script],
        cpus=cpus,
    )
    assert os.waitstatus_to_exitcode(status) == 0
    assert out.read_text() == repr(sorted(cpus))


def test_record_total_sums_cpu_and_notes_affinity() -> None:
    stages = StageTable()
    with stages.stage("load"):
        sum(range(10_000))
    with stages.stage("evaluate"):
        sum(range(10_000))
    stages.record_total()
    table = stages.to_dict()
    total = table["total"]
    assert total.wall_ns == table["load"].wall_ns + table["evaluate"].wall_ns
    assert total.cpu_ns == (table["load"].cpu_ns or 0) + (table["evaluate"].cpu_ns or 0)
    expected = ",".join(str(c) for c in sorted(os.sched_getaffinity(0)))
    assert total.notes == [f"cpu_affinity={expected}"]
