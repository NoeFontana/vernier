"""Per-cell CPU budget, enforced by pinning runner subprocesses.

Thread-count knobs alone cannot make cross-impl cells comparable:
faster-coco-eval ≥1.8 sizes its C++ worker pools from
``std::thread::hardware_concurrency()`` with no user-facing knob, and
hotcoco's rayon pool defaults to every visible CPU. The one constraint
every impl honours is the scheduler's CPU affinity mask, so the
orchestrator pins each instance / LVIS runner subprocess to
:func:`cpu_budget` logical CPUs before ``exec``. Runners additionally
forward the budget to whatever knob their library exposes, so a
library doesn't oversubscribe the pinned set with idle threads.

CPU selection is topology-aware: one logical CPU per physical core
first, SMT siblings only once every core is taken. Pinning ``nt=4`` to
two cores' worth of hyperthreads on a 4-core/8-thread host would
understate every impl's scaling.
"""

from __future__ import annotations

import os
from pathlib import Path

_SYSFS_CPU = Path("/sys/devices/system/cpu")


def cpu_budget(num_threads: int | None) -> int:
    """Logical CPUs a cell may use. The default (``None``) cell is the
    single-threaded headline, so it gets exactly one CPU."""
    return 1 if num_threads is None else num_threads


def _physical_core(cpu: int) -> tuple[int, int]:
    """``(package_id, core_id)`` for a logical CPU. Falls back to a
    unique pseudo-core when sysfs topology is unavailable, which
    degrades to "no SMT awareness" rather than failing."""
    topo = _SYSFS_CPU / f"cpu{cpu}" / "topology"
    try:
        package = int((topo / "physical_package_id").read_text())
        core = int((topo / "core_id").read_text())
    except (OSError, ValueError):
        return (-1, cpu)
    return (package, core)


def select_cpus(n: int, available: set[int] | None = None) -> tuple[int, ...]:
    """Pick ``n`` logical CPUs from ``available`` (default: this
    process's affinity mask), spreading across physical cores before
    doubling up on SMT siblings.

    Cores are taken highest-numbered first so the single-CPU cell stays
    off CPU 0, which typically services the bulk of device interrupts.
    """
    pool = sorted(available if available is not None else os.sched_getaffinity(0))
    if not 1 <= n <= len(pool):
        raise ValueError(f"cannot pin {n} CPU(s): {len(pool)} available ({pool})")
    by_core: dict[tuple[int, int], list[int]] = {}
    for cpu in pool:
        by_core.setdefault(_physical_core(cpu), []).append(cpu)
    cores = [by_core[k] for k in sorted(by_core, reverse=True)]
    max_smt = max(len(c) for c in cores)
    ordered = [c[rank] for rank in range(max_smt) for c in cores if rank < len(c)]
    return tuple(ordered[:n])
