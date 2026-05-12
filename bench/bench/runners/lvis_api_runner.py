"""lvis-api runner — invoked as a subprocess in ``bench/envs/lvis-api``
(ADR-0026 + ADR-0033).

Wraps ``lvis.eval.LVISEval`` — the strict-mode reference oracle per
ADR-0026, vendored at ``ORACLE_LVIS_COMMIT_SHA`` (PyPI ``lvis==0.5.3``,
git ``031ac21f939bcb5f1ca8de2ab8704082e101ff9b``). Lifted from
``tests/python/parity_lvis/harness.py:_lvis_snapshot`` with stage
timers around each phase.

The 13-entry plan keys (``AP``, ``AP50``, ``AP75``, ``APs``, ``APm``,
``APl``, ``APr``, ``APc``, ``APf``, ``AR@300``, ``ARs@300``,
``ARm@300``, ``ARl@300``) come from ``LVISEval.results``; vernier_lvis
mirrors the same keys via ``lvis_stat_names``. The ``(T, R, K, A)``
precision tensor at ``LVISEval.eval["precision"]`` is the strict-tier
parity carrier — matches vernier's accum.precision shape (AF5: no
M-axis).
"""

from __future__ import annotations

import contextlib
import io
import sys

import numpy as np

from bench.harness.timing import StageTable
from bench.runners._protocol import (
    lvis_stat_names,
    parse_lvis_runner_args,
    write_lvis_outputs,
)

# Mirrors ``ORACLE_LVIS_COMMIT_SHA`` in
# ``crates/vernier-core/src/lvis_parity.rs``. The bench env's
# ``pyproject.toml`` pins this same SHA. Bumping is an ADR-level
# decision (every quirk vernier reproduces in strict mode is keyed to
# this commit). The full SHA is recorded in ``impl_version`` so the
# docs renderer can link it to the upstream commit (mirrors
# panopticapi_runner / boundary_iou_api_runner).
_ORACLE_SHA: str = "031ac21f939bcb5f1ca8de2ab8704082e101ff9b"


def main() -> int:
    args = parse_lvis_runner_args()
    iou = args.iou_type
    if iou != "bbox":
        raise ValueError(
            f"lvis-api runner: iou_type={iou!r} mirrors the vernier_lvis "
            f"matrix entry (bbox-only today). The oracle natively supports "
            f"both bbox + segm; the cell waits on `evaluate_segm_grid_with_dataset`."
        )
    max_dets: int = int(args.max_dets)
    stages = StageTable()

    # Lazy import: lvis is in the lvis-api env's deps. The pin
    # mirrors ORACLE_LVIS_COMMIT_SHA in lvis_parity.rs.
    #
    # Workaround mirror (parity_lvis/conftest.py:_restore_numpy_float_alias):
    # lvis 0.5.3 predates NumPy 1.20's removal of ``np.float`` and the
    # upstream fix (master HEAD, commit ``d5a663fb``) never made a PyPI
    # release. Per ADR-0026 §"Parity strategy" the oracle is not edited;
    # the alias is re-bound at runtime. This is a no-op once a future
    # ORACLE_LVIS_COMMIT_SHA bump picks up the fix.
    import numpy as _np

    if not hasattr(_np, "float"):
        _np.float = float  # type: ignore[attr-defined]
    from lvis import LVIS, LVISEval, LVISResults  # type: ignore[import-not-found]

    # The oracle prints progress from inside ``LVIS()`` and
    # ``LVISEval.summarize()``; one outer redirect is cleaner than
    # sprinkling them. Mirrors ``run_cocoeval_pipeline`` in
    # ``_protocol.py``.
    with contextlib.redirect_stdout(io.StringIO()):
        with stages.stage("load"):
            lvis_gt = LVIS(str(args.gt))
            lvis_dt = LVISResults(lvis_gt, str(args.dt), max_dets=max_dets)
            ev = LVISEval(lvis_gt, lvis_dt, iou_type="bbox")
            ev.params.max_dets = max_dets

        with stages.stage("evaluate"):
            ev.evaluate()

        with stages.stage("accumulate"):
            ev.accumulate()

        with stages.stage("summarize"):
            ev.summarize()

    precision = np.asarray(ev.eval["precision"], dtype=np.float64)

    keys = lvis_stat_names(max_dets)
    summary_stats: dict[str, float] = {}
    for k in keys:
        if k not in ev.results:
            raise AssertionError(
                f"lvis-api results is missing key {k!r}; "
                f"present keys: {sorted(ev.results.keys())}"
            )
        summary_stats[k] = float(ev.results[k])

    stages.record("total", stages.total_so_far_ns())

    write_lvis_outputs(
        args=args,
        impl="lvis-api",
        impl_version=_ORACLE_SHA,
        stages=stages.to_dict(),
        summary_stats=summary_stats,
        precision_tensor=precision,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
