#!/usr/bin/env python3
"""End-to-end measurement of the parsed-once ``Dataset`` cache benefit
on the COCO val2017 perfect-segm workload (ADR-0020).

Models the training-loop validation pattern: same GT, fresh DT each
pass. The bench harness in ``bench/`` runs single-shot evaluates per
impl and so doesn't surface what the cache buys; this script does two
passes back-to-back in one process and reports both for both the
bytes-path and the ``Dataset``-path.

Mirrors the Rust example at
``crates/vernier-core/examples/cache_speedup_val2017.rs`` but exercises
the Python surface (``Evaluator(...).evaluate(Dataset, dt)``) so
regressions in the FFI/PyO3 wiring are caught here too.

Inputs (same conventions as the bench harness):
- ``VERNIER_COCO_GT_PATH`` -> GT JSON (falls back to
  ``~/.cache/vernier-bench/coco_val2017/instances_val2017.json``)
- ``VERNIER_COCO_DT_SEGM_PATH`` -> DT JSON (falls back to
  ``<repo>/.cache/coco-val2017/perfect_dt_segm.json``)

Run:
    uv run python tools/cache_speedup_val2017.py
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

from vernier.instance import Boundary, Dataset, Evaluator


def _gt_path() -> Path:
    env = os.environ.get("VERNIER_COCO_GT_PATH")
    if env:
        return Path(env)
    return Path.home() / ".cache/vernier-bench/coco_val2017/instances_val2017.json"


def _dt_path() -> Path:
    env = os.environ.get("VERNIER_COCO_DT_SEGM_PATH")
    if env:
        return Path(env)
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / ".cache/coco-val2017/perfect_dt_segm.json"


def _timed(label: str, work: Callable[[], object]) -> float:
    t = time.perf_counter()
    work()
    ms = (time.perf_counter() - t) * 1000.0
    print(f"{label:<48} {ms:>9.0f} ms")
    return ms


def main() -> int:
    gt_p = _gt_path()
    dt_p = _dt_path()
    if not gt_p.exists():
        print(f"missing GT: {gt_p}", file=sys.stderr)
        return 1
    if not dt_p.exists():
        print(f"missing DT: {dt_p}", file=sys.stderr)
        return 1
    print(f"GT: {gt_p}")
    print(f"DT: {dt_p}")

    gt_bytes = gt_p.read_bytes()
    dt_bytes = dt_p.read_bytes()
    print(f"Loaded GT={len(gt_bytes) / 1_048_576:.1f} MiB, DT={len(dt_bytes) / 1_048_576:.1f} MiB")

    evaluator = Evaluator(iou=Boundary(), parity_mode="strict")

    print("\n=== bytes-path: two back-to-back evaluate calls (no cache) ===")
    _timed("bytes call 1", lambda: evaluator.evaluate(gt_bytes, dt_bytes))
    bytes2_ms = _timed("bytes call 2", lambda: evaluator.evaluate(gt_bytes, dt_bytes))

    print("\n=== Dataset-path: parse-once + warm-cache reuse ===")
    t = time.perf_counter()
    ds = Dataset.from_json(gt_bytes)
    print(f"{'Dataset.from_json':<48} {(time.perf_counter() - t) * 1000.0:>9.0f} ms")
    _timed("Dataset call 1 (cold cache, populates)", lambda: evaluator.evaluate(ds, dt_bytes))
    ds2_ms = _timed("Dataset call 2 (warm cache, hits)", lambda: evaluator.evaluate(ds, dt_bytes))

    speedup = bytes2_ms / ds2_ms if ds2_ms > 0 else float("inf")
    saved_ms = bytes2_ms - ds2_ms
    print("\n=== summary ===")
    print(f"bytes-path call 2  : {bytes2_ms:>9.0f} ms")
    print(f"Dataset call 2     : {ds2_ms:>9.0f} ms")
    print(f"saved per warm call: {saved_ms:>9.0f} ms ({speedup:.2f}x faster)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
