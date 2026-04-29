"""Out-of-process parity tests for the `vernier eval` binary (ADR-0015).

This module is the binary-boundary complement to the in-process parity
suite in `test_parity.py` and the Rust integration tests in
`crates/vernier-cli/tests/eval.rs`. It builds `vernier-cli` with `cargo`
and invokes the resulting binary via `subprocess`, then asserts:

- Strict-mode text output is byte-equal to pycocotools 2.0.11's
  `COCOeval.summarize()` stdout. This is the canonical parity guarantee
  the CLI ships (ADR-0002 §"strict tier"; ADR-0015 §"Formatter: text").
- The JSON document agrees with the in-process `vernier._core` summary
  (`stats[i] == lines[i]["value"]`, schema `version == "1"`,
  `iou_type` echoes the flag).
- Multi-emit invocations are output-equivalent to single-emit
  invocations. The CLI runs eval once per process and writes each
  formatter's output independently; the test cannot observe the count
  directly, but it pins the divergence-free invariant.
- JSON output is byte-deterministic across runs of the same input
  (ADR-0015 §"Output determinism").
- Kind-coupled flags (`--dilation-ratio`, `--sigmas`) and unknown IoU
  types are rejected with exit code 2 at parse/validation time.

Each test is marked `@pytest.mark.parity` so `just test-parity` and
`uv run pytest -m parity` pick it up. Test 2 cross-checks against the
in-process kernel; if `vernier._core` cannot be imported (e.g. the
maturin build failed in the local environment), it is skipped rather
than failing.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .harness import IouType

FIXTURES = Path(__file__).parent / "fixtures"
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]

# Fixture per IoU kind. The CLI surface covers four kinds (bbox / segm /
# boundary / keypoints); boundary needs an additional `--dilation-ratio`
# to be exercised meaningfully and has its own dedicated parity track,
# so the binary-boundary tests cover the three plain kinds.
_FIXTURE_BY_KIND: dict[IouType, str] = {
    "bbox": "perfect_match",
    "segm": "perfect_match_segm",
    "keypoints": "keypoints_perfect_match",
}

_HAS_VERNIER_CORE = importlib.util.find_spec("vernier._core") is not None


@pytest.fixture(scope="session")
def vernier_bin() -> Path:
    """Build `vernier-cli` once per session and return the binary path.

    Uses `cargo build --release -p vernier-cli` directly rather than
    `just build`, which would invoke `maturin develop` and pull in the
    Python extension build (libpython link). The CLI is a pure-Rust
    workspace member with no PyO3, so it links cleanly on its own.
    """
    cargo_toml = WORKSPACE_ROOT / "Cargo.toml"
    assert cargo_toml.exists(), f"workspace root sanity check failed: {cargo_toml}"

    target_dir = Path(os.environ.get("CARGO_TARGET_DIR", str(WORKSPACE_ROOT / "target")))
    subprocess.run(
        ["cargo", "build", "--release", "-p", "vernier-cli"],
        cwd=WORKSPACE_ROOT,
        check=True,
    )
    binary = target_dir / "release" / "vernier"
    assert binary.exists(), f"cargo build did not produce {binary}"
    return binary


def _fixture_paths(kind: IouType) -> tuple[Path, Path]:
    name = _FIXTURE_BY_KIND[kind]
    return FIXTURES / name / "gt.json", FIXTURES / name / "dt.json"


def _run_cli(
    vernier_bin: Path,
    gt: Path,
    dt: Path,
    iou_type: IouType,
    *extra: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            str(vernier_bin),
            "eval",
            "--gt",
            str(gt),
            "--dt",
            str(dt),
            "--iou-type",
            iou_type,
            *extra,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def _pycocotools_summarize_stdout(gt: Path, dt: Path, iou_type: IouType) -> str:
    """Capture only `COCOeval.summarize()`'s stdout.

    `evaluate()` and `accumulate()` also print progress lines we do not
    want compared (the CLI never emits them), so they are wrapped in a
    discarding redirect first.
    """
    coco_gt = COCO(str(gt))
    coco_dt = coco_gt.loadRes(str(dt))
    cocoeval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    with contextlib.redirect_stdout(io.StringIO()):
        cocoeval.evaluate()
        cocoeval.accumulate()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cocoeval.summarize()
    return buf.getvalue()


@pytest.mark.parity
@pytest.mark.parametrize("iou_type", ["bbox", "segm", "keypoints"])
def test_strict_text_matches_pycocotools_stdout(
    vernier_bin: Path,
    iou_type: IouType,
) -> None:
    gt, dt = _fixture_paths(iou_type)
    cli = _run_cli(vernier_bin, gt, dt, iou_type)
    assert cli.returncode == 0, f"stderr: {cli.stderr!r}"

    expected = _pycocotools_summarize_stdout(gt, dt, iou_type)
    assert cli.stdout == expected, (
        f"strict-mode text output diverged from pycocotools.summarize() stdout for "
        f"iou_type={iou_type}.\n--- CLI ---\n{cli.stdout}\n--- pycocotools ---\n{expected}"
    )


@pytest.mark.parity
@pytest.mark.parametrize("iou_type", ["bbox", "segm", "keypoints"])
def test_json_stats_match_in_process_summary(
    vernier_bin: Path,
    tmp_path: Path,
    iou_type: IouType,
) -> None:
    if not _HAS_VERNIER_CORE:
        pytest.skip("vernier._core is not importable in this environment")
    # Local import: harness imports `vernier._core` at module scope, so
    # we can only pull it in when the extension exists.
    from .harness import snapshot

    gt, dt = _fixture_paths(iou_type)
    out = tmp_path / "result.json"
    cli = _run_cli(vernier_bin, gt, dt, iou_type, "--emit", f"json={out}")
    assert cli.returncode == 0, f"stderr: {cli.stderr!r}"
    assert cli.stdout == "", f"file-only emit must leave stdout empty: {cli.stdout!r}"

    doc: dict[str, Any] = json.loads(out.read_bytes())
    assert doc["version"] == "1"
    assert doc["iou_type"] == iou_type
    lines = doc["lines"]
    stats = doc["stats"]
    assert len(lines) == len(stats)
    for i, (line, stat) in enumerate(zip(lines, stats, strict=True)):
        assert line["value"] == stat, f"lines[{i}].value != stats[{i}]: {line['value']} vs {stat}"

    in_process = snapshot("vernier", gt, dt, iou_type)
    np.testing.assert_array_equal(np.asarray(stats), in_process.stats)


@pytest.mark.parity
def test_multi_emit_runs_eval_once(vernier_bin: Path, tmp_path: Path) -> None:
    gt, dt = _fixture_paths("bbox")

    text_only = _run_cli(vernier_bin, gt, dt, "bbox", "--emit", "text")
    assert text_only.returncode == 0, f"stderr: {text_only.stderr!r}"

    a_json = tmp_path / "a.json"
    json_only = _run_cli(vernier_bin, gt, dt, "bbox", "--emit", f"json={a_json}")
    assert json_only.returncode == 0, f"stderr: {json_only.stderr!r}"

    b_json = tmp_path / "b.json"
    multi = _run_cli(vernier_bin, gt, dt, "bbox", "--emit", "text", "--emit", f"json={b_json}")
    assert multi.returncode == 0, f"stderr: {multi.stderr!r}"

    # JSON output is independent of whether text was also emitted.
    assert a_json.read_bytes() == b_json.read_bytes()
    # Text on stdout is independent of whether JSON was also emitted to a file.
    assert multi.stdout == text_only.stdout


@pytest.mark.parity
def test_json_byte_deterministic(vernier_bin: Path, tmp_path: Path) -> None:
    gt, dt = _fixture_paths("bbox")
    paths = [tmp_path / f"out_{i}.json" for i in range(3)]
    for path in paths:
        cli = _run_cli(vernier_bin, gt, dt, "bbox", "--emit", f"json={path}")
        assert cli.returncode == 0, f"stderr: {cli.stderr!r}"

    first = paths[0].read_bytes()
    for path in paths[1:]:
        assert path.read_bytes() == first, f"{path} diverged from {paths[0]}"


@pytest.mark.parity
@pytest.mark.parametrize(
    ("iou_type", "extra_flag", "extra_value"),
    [
        ("bbox", "--dilation-ratio", "0.02"),
        ("segm", "--sigmas", "dummy.json"),
    ],
)
def test_kind_coupled_flag_rejected(
    vernier_bin: Path,
    iou_type: IouType,
    extra_flag: str,
    extra_value: str,
) -> None:
    # Use a real GT/DT pair so validation, not file IO, is the rejection
    # path. The dummy sigmas file is intentionally never opened.
    gt, dt = _fixture_paths(iou_type)
    cli = _run_cli(vernier_bin, gt, dt, iou_type, extra_flag, extra_value)
    assert cli.returncode == 2, f"expected exit 2, got {cli.returncode}; stderr: {cli.stderr!r}"
    assert cli.stderr != "", "expected a validation error on stderr"


@pytest.mark.parity
def test_unknown_iou_type_rejected(vernier_bin: Path) -> None:
    # clap rejects unknown enum values at parse time. We need *some*
    # value for --gt / --dt; using existing files keeps the rejection
    # at the parse layer (before we'd hit IO).
    gt, dt = _fixture_paths("bbox")
    cli = subprocess.run(
        [
            str(vernier_bin),
            "eval",
            "--gt",
            str(gt),
            "--dt",
            str(dt),
            "--iou-type",
            "lvis",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert cli.returncode == 2, f"expected exit 2, got {cli.returncode}; stderr: {cli.stderr!r}"
    assert cli.stderr != "", "expected a clap parse error on stderr"


@pytest.mark.parity
def test_text_default_no_flags(vernier_bin: Path) -> None:
    gt, dt = _fixture_paths("bbox")
    cli = _run_cli(vernier_bin, gt, dt, "bbox")
    assert cli.returncode == 0
    assert cli.stderr == "", f"unexpected stderr: {cli.stderr!r}"
    assert cli.stdout != ""
    # Default formatter is text; the first line is the canonical
    # pycocotools-shaped AP @ IoU=0.50:0.95 / area=all / maxDets=100 row.
    assert cli.stdout.startswith(" Average Precision  (AP) @[ IoU=0.50:0.95 |"), (
        f"first line shape diverged: {cli.stdout.splitlines()[0]!r}"
    )
