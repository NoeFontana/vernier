"""Smoke-test an installed vernier wheel: load, evaluate, and (on Linux) fork.

Run against a built wheel in a fresh environment (CI's ``smoke`` job) and
from ``tests/python/test_wheel_smoke.py``. Exits non-zero on failure.

The fork check covers the extension's allocator (ADR-0070): a worker
thread keeps evaluating with a per-call thread pool — so pool threads
start, allocate and exit throughout — while the main thread forks
children that each run an evaluation of their own. A child that inherits
an allocator lock held by a thread that no longer exists in it would
hang; each child must finish within the timeout.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import warnings

import vernier
import vernier._core as core
from vernier._types import PARITY_STRICT

FORKS = 16
CHILD_TIMEOUT_S = 60.0


def _payloads() -> tuple[bytes, bytes]:
    images = [{"id": i, "width": 640, "height": 480} for i in range(1, 41)]
    gts, dts = [], []
    for img in images:
        for j in range(5):
            x, y = 20.0 + 90 * j, 30.0 + 7 * j
            gts.append(
                {
                    "id": len(gts) + 1,
                    "image_id": img["id"],
                    "category_id": 1 + j % 3,
                    "bbox": [x, y, 60.0, 80.0],
                    "area": 4800.0,
                    "iscrowd": 0,
                    "segmentation": [[x, y, x + 60, y, x + 60, y + 80, x, y + 80]],
                }
            )
            dts.append(
                {
                    "image_id": img["id"],
                    "category_id": 1 + j % 3,
                    "bbox": [x + 3, y - 2, 58.0, 83.0],
                    "score": 0.9 - 0.1 * j,
                    "segmentation": [[x + 3, y - 2, x + 61, y - 2, x + 61, y + 81, x + 3, y + 81]],
                }
            )
    categories = [{"id": c, "name": f"c{c}"} for c in (1, 2, 3)]
    gt = {"images": images, "annotations": gts, "categories": categories}
    return json.dumps(gt).encode(), json.dumps(dts).encode()


def _evaluate(gt: bytes, dt: bytes, num_threads: int | None) -> float:
    stats = []
    for fn in (core.evaluate_bbox_grid, core.evaluate_segm_grid):
        grid = fn(
            gt,
            dt,
            parity_mode=PARITY_STRICT,
            max_dets_per_image=100,
            use_cats=True,
            num_threads=num_threads,
        )
        stats.append(grid.accumulate([1, 10, 100]).summarize([1, 10, 100]).stats[0])
    if not all(0.0 < s <= 1.0 for s in stats):
        raise AssertionError(f"unexpected AP {stats}")
    return stats[0]


def _fork_while_evaluating(gt: bytes, dt: bytes, expected: float) -> None:
    stop = threading.Event()
    errors: list[BaseException] = []

    def churn() -> None:
        try:
            while not stop.is_set():
                _evaluate(gt, dt, num_threads=2)
        except BaseException as e:
            errors.append(e)

    worker = threading.Thread(target=churn, daemon=True)
    worker.start()
    try:
        with warnings.catch_warnings():
            # Python warns that forking a multi-threaded process may
            # deadlock the child; whether it does is what this checks.
            warnings.simplefilter("ignore", DeprecationWarning)
            for _ in range(FORKS):
                pid = os.fork()
                if pid == 0:
                    code = 0 if _evaluate(gt, dt, num_threads=None) == expected else 3
                    os._exit(code)
                _wait(pid)
    finally:
        stop.set()
        worker.join()
    if errors:
        raise errors[0]


def _wait(pid: int) -> None:
    deadline = time.monotonic() + CHILD_TIMEOUT_S
    while time.monotonic() < deadline:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            code = os.waitstatus_to_exitcode(status)
            if code != 0:
                raise AssertionError(f"forked child exited with {code}")
            return
        time.sleep(0.05)
    os.kill(pid, 9)
    os.waitpid(pid, 0)
    raise AssertionError(f"forked child hung for {CHILD_TIMEOUT_S:.0f} s")


def main() -> int:
    gt, dt = _payloads()
    expected = _evaluate(gt, dt, num_threads=None)
    if _evaluate(gt, dt, num_threads=2) != expected:
        raise AssertionError("threaded evaluation disagrees with sequential")
    if sys.platform == "linux":
        _fork_while_evaluating(gt, dt, expected)
    print(f"vernier {vernier.version()} smoke OK (AP {expected:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
