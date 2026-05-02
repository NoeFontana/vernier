"""ADR-0017 test plan §5 — corrupt the cached COCO val2017 GT and assert
that the workload module re-downloads to recover. Network-gated: routine
``bench-test`` does NOT hit the COCO CDN."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from bench.runners._protocol import file_sha256
from bench.workloads import coco_val2017

_GATE = "VERNIER_BENCH_DOWNLOAD_TESTS"

pytestmark = pytest.mark.skipif(
    os.environ.get(_GATE) != "1",
    reason=f"set {_GATE}=1 to run network-downloading tests",
)


def test_gt_downloads_verifies_and_recovers_from_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One download exercises both code paths: a clean download must
    sha256-verify, and a subsequent corruption must trigger re-download
    rather than return a bad cached file."""
    monkeypatch.setenv("VERNIER_BENCH_CACHE", str(tmp_path))
    monkeypatch.delenv("VERNIER_COCO_GT_PATH", raising=False)

    gt = coco_val2017.gt_path()
    assert gt.exists()
    assert file_sha256(gt) == coco_val2017.EXPECTED_SHA256

    gt.write_bytes(b"not the right bytes")
    assert file_sha256(gt) != coco_val2017.EXPECTED_SHA256
    refreshed = coco_val2017.gt_path()
    assert file_sha256(refreshed) == coco_val2017.EXPECTED_SHA256
