"""ADR-0017 test plan §4 — same seed → same shuffle, different seeds → different shuffles.

The schedule is the only thing protecting release runs from thermal-drift
artefacts; if it weren't deterministic we couldn't replay a flaky run.
"""

from __future__ import annotations

from collections import Counter

from bench.harness.orchestrate import _build_schedule

_IMPLS = ["vernier", "pycocotools", "faster-coco-eval", "boundary-iou-api"]


def test_same_seed_yields_same_schedule() -> None:
    a = _build_schedule(_IMPLS, n_warmup=2, n_measurement=10, run_seed=42)
    b = _build_schedule(_IMPLS, n_warmup=2, n_measurement=10, run_seed=42)
    assert a == b


def test_different_seeds_yield_different_schedules() -> None:
    a = _build_schedule(_IMPLS, n_warmup=2, n_measurement=10, run_seed=1)
    b = _build_schedule(_IMPLS, n_warmup=2, n_measurement=10, run_seed=2)
    assert a != b


def test_each_rep_visits_every_impl_exactly_once() -> None:
    schedule = _build_schedule(_IMPLS, n_warmup=2, n_measurement=10, run_seed=0)
    by_rep: dict[int, list[str]] = {}
    for impl, rep_idx, _warmup in schedule:
        by_rep.setdefault(rep_idx, []).append(impl)
    assert sorted(by_rep) == list(range(12))
    for rep_idx, impls in by_rep.items():
        assert Counter(impls) == Counter(_IMPLS), (
            f"rep {rep_idx} did not visit every impl exactly once"
        )


def test_warmup_flag_aligned_with_warmup_count() -> None:
    schedule = _build_schedule(_IMPLS, n_warmup=2, n_measurement=10, run_seed=0)
    for impl, rep_idx, warmup in schedule:
        assert warmup == (rep_idx < 2), (impl, rep_idx, warmup)


def test_empty_impl_list_yields_empty_schedule() -> None:
    assert _build_schedule([], n_warmup=2, n_measurement=10, run_seed=0) == []


def test_at_least_one_rep_has_a_non_canonical_order() -> None:
    """Sanity: the permutation actually shuffles for non-trivial impl lists.

    Across 12 reps over 4 impls (24 possible orderings), the chance of
    every rep coming out as the canonical order is 1/24**12 — effectively
    zero for any reasonable PRNG.
    """
    schedule = _build_schedule(_IMPLS, n_warmup=2, n_measurement=10, run_seed=0)
    by_rep: dict[int, list[str]] = {}
    for impl, rep_idx, _warmup in schedule:
        by_rep.setdefault(rep_idx, []).append(impl)
    assert any(order != _IMPLS for order in by_rep.values())
