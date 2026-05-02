"""ADR-0017 test plan — every mode's documented defaults must match
``MODE_REPS``. The test exists so a future refactor that drifts the
table from the ADR fails loudly."""

from __future__ import annotations

import pytest

from bench.harness.orchestrate import MODE_REPS, mode_defaults


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        # dev: 1 rep, no warmup — under-30s inner loop on a laptop.
        ("dev", (0, 1)),
        # release: 2 warmup + 10 measurement, governor + IQR gate.
        ("release", (2, 10)),
        # profile: 1 rep — instrumentation perturbs measurement.
        ("profile", (0, 1)),
    ],
)
def test_mode_defaults(mode: str, expected: tuple[int, int]) -> None:
    assert MODE_REPS[mode] == expected  # type: ignore[index]
    assert mode_defaults(mode) == expected  # type: ignore[arg-type]


def test_mode_table_covers_every_mode_literal() -> None:
    """Adding a new ``Mode`` literal without a row in ``MODE_REPS`` is a bug."""
    from typing import get_args

    from bench.harness.schema import Mode

    assert set(MODE_REPS) == set(get_args(Mode))
