"""ADR-0017 test plan §6 — fingerprint must be deterministic across calls.

A nondeterministic input (timestamps, hostname-then-uname, anything that
varies per call) would re-bucket every run into a fresh result tree.
"""

from __future__ import annotations

from bench.harness.machine import MachineInputs, collect_inputs, fingerprint


def test_fingerprint_is_stable_across_calls() -> None:
    values = {fingerprint() for _ in range(100)}
    assert len(values) == 1


def test_fingerprint_is_12_hex_chars() -> None:
    fp = fingerprint()
    assert len(fp) == 12
    int(fp, 16)


def test_fingerprint_responds_to_input_drift() -> None:
    """Two inputs that differ in any field hash to different values.

    Catches a future refactor that drops a field from the hash input.
    """
    base = MachineInputs(
        cpu_model="cpu", n_cores=4, total_ram_kb=1024, os_release="r", glibc_version="g"
    )

    def fp_for(inputs: MachineInputs) -> str:
        import hashlib

        return hashlib.sha256(inputs.hash_input().encode()).hexdigest()[:12]

    base_fp = fp_for(base)
    for changed in (
        MachineInputs("cpu2", 4, 1024, "r", "g"),
        MachineInputs("cpu", 8, 1024, "r", "g"),
        MachineInputs("cpu", 4, 2048, "r", "g"),
        MachineInputs("cpu", 4, 1024, "r2", "g"),
        MachineInputs("cpu", 4, 1024, "r", "g2"),
    ):
        assert fp_for(changed) != base_fp


def test_collect_inputs_returns_populated_struct() -> None:
    """Linux test runner: every input is non-empty / non-zero."""
    inputs = collect_inputs()
    assert inputs.cpu_model
    assert inputs.n_cores > 0
    assert inputs.total_ram_kb > 0
    assert inputs.os_release
    assert inputs.glibc_version
