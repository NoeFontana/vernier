"""One-time vendor-verification fixture (ADR-0036, refresh step 7).

Asserts that the vendored `IoUMetric` at the pinned SHA produces
bit-identical metrics to a real `pip install mmsegmentation==<version>`
on a known label-map fixture. This is a **manual, run-locally** check —
it is deliberately NOT wired into CI, because the whole point of the
vendor (per AP5 in `docs/engineering/sem-seg-quirks.md`) is to avoid
pulling the ~3 GB mmsegmentation transitive into the test environment.

## Usage

Run twice, in two separate virtualenvs, and diff the outputs:

1. **Vendored side** — from the vernier repo root, in the dev venv:
   ```
   uv run python tests/python/parity_semantic/oracle/mmsegmentation/_verify_against_pip.py vendored
   ```
   Writes `/tmp/vernier_mmseg_oracle_vendored.json`.

2. **Real pip-installed side** — in a fresh, *separate* virtualenv
   that has `mmsegmentation==1.2.2` (matching the SHA pin) and torch
   installed:
   ```
   python -m venv /tmp/mmseg-pip-verify
   source /tmp/mmseg-pip-verify/bin/activate
   pip install mmsegmentation==1.2.2 torch
   python tests/python/parity_semantic/oracle/mmsegmentation/_verify_against_pip.py pip
   deactivate
   ```
   Writes `/tmp/vernier_mmseg_oracle_pip.json`.

3. **Diff the two**:
   ```
   diff /tmp/vernier_mmseg_oracle_{vendored,pip}.json && echo OK
   ```
   No diff = the vendored bytes + our stubs reproduce the real
   `IoUMetric` exactly. Drift = either upstream's `mmseg/registry`
   does something the stub doesn't replicate, or the pin is on the
   wrong SHA.

## When to run

- **At vendor time**: confirm the initial SHA pin is correct.
- **At every SHA refresh** (ADR-0036's "How to refresh" step 7).
- **When the stub surface changes**: if `_mmengine_stub.py` adds or
  removes a symbol, re-run to confirm `IoUMetric`'s observable
  behavior is unchanged.

Document the run + diff result in the commit message that flips the
SHA. Do not check the `/tmp/*.json` files into the repo.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_OUT_VENDORED = Path("/tmp/vernier_mmseg_oracle_vendored.json")
_OUT_PIP = Path("/tmp/vernier_mmseg_oracle_pip.json")


def _build_fixture():
    # Mirrors the fixture in test_parity_semantic.py but bigger (32x32,
    # 5 classes, mixed values + ignore-region) so we exercise more of
    # torch.histc's bin boundaries than the smoke test does.
    rng = np.random.default_rng(20260509)
    gt = rng.integers(0, 5, size=(32, 32), dtype=np.int64)
    pred = rng.integers(0, 5, size=(32, 32), dtype=np.int64)
    # Inject an ignore region so we exercise the mask path too.
    gt[:8, :8] = 255
    return pred, gt


def _run(mode: str) -> dict[str, object]:
    if mode == "vendored":
        # Match the conftest stub-injection so this script can be run
        # directly (no pytest involvement).
        oracle_path = Path(__file__).parent
        sys.path.insert(0, str(oracle_path))
        import types

        import _mmengine_stub

        def _install(name: str, attrs: dict[str, object]) -> None:
            mod = types.ModuleType(name)
            for k, v in attrs.items():
                setattr(mod, k, v)
            sys.modules[name] = mod

        _install("mmengine", {})
        _install("mmengine.dist", {"is_main_process": _mmengine_stub.is_main_process})
        _install("mmengine.evaluator", {"BaseMetric": _mmengine_stub.BaseMetric})
        _install(
            "mmengine.logging",
            {"MMLogger": _mmengine_stub.MMLogger, "print_log": _mmengine_stub.print_log},
        )
        _install("mmengine.utils", {"mkdir_or_exist": _mmengine_stub.mkdir_or_exist})
        _install("mmseg.registry", {"METRICS": _mmengine_stub.METRICS})
        _install("prettytable", {"PrettyTable": _mmengine_stub.PrettyTable})
    elif mode == "pip":
        # Trust the active venv: `mmsegmentation==1.2.2` is pip-installed.
        # No sys.modules fiddling — the real packages resolve normally.
        pass
    else:
        raise SystemExit(f"unknown mode {mode!r}; use 'vendored' or 'pip'")

    import torch
    from mmseg.evaluation.metrics.iou_metric import IoUMetric

    pred, gt = _build_fixture()
    pred_t = torch.from_numpy(pred)  # pyright: ignore[reportPrivateImportUsage]
    gt_t = torch.from_numpy(gt)  # pyright: ignore[reportPrivateImportUsage]

    intersect, union, area_pred, area_label = IoUMetric.intersect_and_union(
        pred_t, gt_t, num_classes=5, ignore_index=255
    )
    # Upstream annotates `total_area_to_metrics` parameters as
    # `np.ndarray` but actually passes torch.Tensor in
    # `compute_metrics` (line 124-127); the function calls
    # `.numpy()` itself before returning. The annotation is
    # misleading; the runtime call path is correct.
    metrics = IoUMetric.total_area_to_metrics(
        intersect,  # pyright: ignore[reportArgumentType]
        union,  # pyright: ignore[reportArgumentType]
        area_pred,  # pyright: ignore[reportArgumentType]
        area_label,  # pyright: ignore[reportArgumentType]
        ["mIoU", "mDice", "mFscore"],
    )

    return {
        "intersect": intersect.tolist(),
        "union": union.tolist(),
        "area_pred": area_pred.tolist(),
        "area_label": area_label.tolist(),
        # mFscore is the float-edge-sensitive branch; including it
        # exercises torch.tensor() construction beyond histc.
        "metrics": {
            k: (v.tolist() if hasattr(v, "tolist") else float(v)) for k, v in metrics.items()
        },
    }


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in {"vendored", "pip"}:
        print(__doc__, file=sys.stderr)
        return 2
    mode = sys.argv[1]
    result = _run(mode)
    out = _OUT_VENDORED if mode == "vendored" else _OUT_PIP
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
