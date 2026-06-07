"""End-to-end TIDE validation against real-model predictions on COCO val2017.

The numpy-oracle parity contract (ADR-0021, exercised in
``tests/python/oracle/tide/``) proves correctness on synthetic
fixtures; this module proves the same machinery doesn't fall over on
real data — 5000 images, 80 classes, real-model error distributions.

What the suite gates:

- **Coherence** — per-bin ΔmAP values are non-negative and bounded
  above by the all-FPs-removed sanity total. The TIDE paper's bins
  overlap meaningfully (correcting a Cls error can incidentally help
  Loc), so the only structural invariant is per-bin non-negativity +
  the upper bound; tighter assertions belong in the oracle parity
  suite, not here.
- **Determinism** — running TIDE twice on the same cached predictions
  produces byte-identical reports. This isolates vernier's
  determinism from the model's inference-time non-determinism (which
  is upstream of the cached predictions and therefore out of scope).
- **Numpy-oracle parity on real data (DETR-R50, bbox)** — closes the
  ADR-0022 follow-up that flagged ``t_b = 0.1`` as "tentative" pending
  empirical validation on a set-prediction transformer detector. The
  aligned-tier gate asserts agreement at ``rtol = 8 * eps`` on every
  float surface both sides expose (``baseline_map``, the six per-bin
  ``delta`` values, ``delta_all_fp_removed``) — the closest signal to
  "per-bucket detection counts agree" the current TIDE Python /
  Rust surfaces allow, and exactly what an empirical ``t_b``
  ratification consumes downstream. Both sides run in f64; reduction
  order across ~150k detections x 80 classes x 10 IoU thresholds
  caps drift at single-digit ULP, so 8 ULP keeps a wrong-``t_b``
  boundary or a wrong-IoU numerator well above the gate.

What the suite does *not* gate (deferred to ``run.py``):

- Wall-clock or memory budgets — hardware-dependent; the README pins
  the target.
- Specific bin-mass distributions — model-dependent and not a
  property of the implementation.

Skips cleanly when the relevant prediction cache is unpopulated
(rfdetr inference unrun for the rfdetr cells; DETR-R50 cache absent
for the DETR cell) or when ``VERNIER_COCO_CACHE`` doesn't point at a
populated val2017 layout (GT JSON + images directory).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import vernier
from vernier._tide import KernelName
from vernier.instance import Bbox, Boundary, Segm

from .conftest import TideModelName

pytestmark = [pytest.mark.real_models, pytest.mark.slow]


_KERNELS: dict[KernelName, object] = {
    "bbox": Bbox(),
    "segm": Segm(),
    "boundary": Boundary(dilation_ratio=0.02),
}

#: 8 ULP of float64 — used as BOTH ``rtol`` and ``atol`` on the
#: vernier ↔ numpy-oracle parity gate. Mirrors the band the SOTA
#: real-model panoptic test uses (see
#: ``sota/test_mask2former_panoptic_real_models.py``) for the same
#: reduction-order reason. ``assert_allclose`` evaluates
#: ``|a-b| ≤ atol + rtol * |b|``: a per-bin ``delta`` that collapses
#: to exact ``0.0`` on one side (a structurally empty bin: no Cls
#: errors on a class-agnostic model, no Dupe errors on a set-prediction
#: detector with one query per object) would fail an ``rtol``-only
#: gate at ``rtol * 0 = 0`` if the other side yielded a sub-ULP
#: non-zero. The ``atol`` band absorbs exactly that — both sides are
#: f64, the only legitimate drift source at the 150k-detection scale
#: is reduction order across 80 classes x 10 IoU thresholds, capped
#: at single-digit ULP.
_TIDE_ORACLE_TOL = 8.0 * float(np.finfo(np.float64).eps)


def _assert_report_coherent(report: vernier.instance.TideReport) -> None:
    """Structural sanity invariants for any TIDE report.

    Per ADR-0021's correctness model: each per-bin delta is the mAP gain
    if every detection in that bin were corrected; the all-FPs-removed
    delta is the upper bound for any single-bin correction. Bins overlap
    in the TIDE paper, so a sum-greater-than-bound is *not* a bug — only
    individual deltas exceeding the bound are.
    """
    assert 0.0 <= report.baseline_map <= 1.0, f"baseline_map outside [0, 1]: {report.baseline_map}"
    assert report.delta_all_fp_removed >= 0.0, (
        f"delta_all_fp_removed negative: {report.delta_all_fp_removed}"
    )
    assert report.baseline_map + report.delta_all_fp_removed <= 1.0 + 1e-9, (
        f"baseline + perfect-rejection exceeds 1.0: "
        f"{report.baseline_map} + {report.delta_all_fp_removed}"
    )

    expected_bins = {"cls", "loc", "both", "dupe", "bkg", "missed"}
    assert set(report.delta) == expected_bins, (
        f"unexpected bins in report: got {set(report.delta)}, expected {expected_bins}"
    )
    for bin_name, value in report.delta.items():
        assert value >= -1e-9, f"bin {bin_name!r} has negative ΔmAP: {value}"
        assert value <= report.delta_all_fp_removed + 1e-9, (
            f"bin {bin_name!r} ΔmAP {value} exceeds perfect-rejection upper "
            f"bound {report.delta_all_fp_removed}"
        )


@pytest.mark.parametrize(
    ("model_name", "kernel_name"),
    [
        ("nano", "bbox"),
        ("segnano", "segm"),
        ("segnano", "boundary"),
        # DETR-R50 is a set-prediction transformer detector; including it
        # in the coherence matrix exercises the t_b = 0.1 bbox default
        # (ADR-0022) on a fundamentally different prediction distribution
        # than rf-detr's NMS-free anchor-based output — the score
        # long-tail concentrates differently. No segm output, so bbox
        # only.
        ("detr-r50", "bbox"),
    ],
)
def test_tide_coherence_on_real_predictions(
    model_name: TideModelName,
    kernel_name: KernelName,
    coco_gt_bytes: bytes,
    predictions_for: Callable[[TideModelName], bytes],
) -> None:
    """Run TIDE on cached predictions; verify structural invariants hold.

    Parametrized to cover all three production kernels with the smallest
    relevant model that exercises each (RFDETRNano covers bbox;
    RFDETRSegNano covers segm + boundary, since boundary IoU is computed
    from the same masks segm already needs) plus DETR-R50 for a
    set-prediction transformer reading on the bbox kernel.
    """
    report = vernier.instance.error_decomposition(
        coco_gt_bytes,
        predictions_for(model_name),
        iou=_KERNELS[kernel_name],
    )
    _assert_report_coherent(report)


def test_tide_determinism_on_real_predictions(
    coco_gt_bytes: bytes,
    predictions_for: Callable[[TideModelName], bytes],
) -> None:
    """Two calls on the same inputs produce byte-equivalent reports.

    Asserts vernier-side determinism only — model inference is upstream
    of the cached prediction bytes, so torch / CUDA non-determinism (if
    any) is filtered out by construction. Exercised on the nano model
    + bbox kernel as the lightest-weight check; if vernier had a
    determinism bug, it would surface here as readily as on the heavier
    paths.
    """
    predictions = predictions_for("nano")
    a = vernier.instance.error_decomposition(coco_gt_bytes, predictions, iou=Bbox())
    b = vernier.instance.error_decomposition(coco_gt_bytes, predictions, iou=Bbox())

    assert a == b, (
        f"non-deterministic TIDE: report A and B differ.\n"
        f"  A.baseline_map = {a.baseline_map}\n"
        f"  B.baseline_map = {b.baseline_map}\n"
        f"  A.delta = {a.delta}\n"
        f"  B.delta = {b.delta}\n"
        f"  A.delta_all_fp_removed = {a.delta_all_fp_removed}\n"
        f"  B.delta_all_fp_removed = {b.delta_all_fp_removed}\n"
    )


def test_tide_parity_vs_numpy_oracle_detr_r50(
    coco_gt_bytes: bytes,
    coco_gt_dict: dict[str, Any],
    predictions_for: Callable[[TideModelName], bytes],
) -> None:
    """vernier ↔ numpy-oracle parity on DETR-R50 bbox predictions.

    Closes the ADR-0022 follow-up flagging ``t_b = 0.1`` as "tentative"
    pending empirical validation on a set-prediction transformer
    detector. The numpy oracle in ``tests/python/oracle/tide`` is the
    executable spec per ADR-0021; running it on the real DETR-R50 cache
    proves the Rust evaluator and the spec agree on the same ~150k
    detections across 80 categories that an empirical ratification of
    ``t_b`` would consume.

    Tolerance is ``rtol = 8 * eps`` on every float surface both sides
    expose (``baseline_map``, the six per-bin ``delta`` entries,
    ``delta_all_fp_removed``). Both sides run in f64 throughout; the
    legitimate drift source at the 150k-detection scale is reduction
    order, capped at single-digit ULP. 8 ULP keeps a wrong-``t_b``
    boundary or a wrong-IoU numerator well above the gate. Per-bucket
    detection-count surfaces are not yet exposed by either side; once
    they are, the per-bucket integer arrays drop into the same
    parametrize loop and assert bit-equality trivially.
    """
    from ....oracle.tide.oracle import error_decomposition as oracle_decomposition

    dt_bytes = predictions_for("detr-r50")
    dt_list = json.loads(dt_bytes)

    rust_report = vernier.instance.error_decomposition(
        coco_gt_bytes,
        dt_bytes,
        iou=Bbox(),
    )
    oracle_report = oracle_decomposition(coco_gt_dict, dt_list)

    np.testing.assert_allclose(
        rust_report.baseline_map,
        oracle_report["baseline_map"],
        rtol=_TIDE_ORACLE_TOL,
        atol=_TIDE_ORACLE_TOL,
        err_msg="baseline_map: rust vs numpy-oracle (DETR-R50 bbox)",
    )
    for bin_name in ("cls", "loc", "both", "dupe", "bkg", "missed"):
        np.testing.assert_allclose(
            rust_report.delta[bin_name],
            oracle_report["delta"][bin_name],
            rtol=_TIDE_ORACLE_TOL,
            atol=_TIDE_ORACLE_TOL,
            err_msg=f"delta[{bin_name}]: rust vs numpy-oracle (DETR-R50 bbox)",
        )
    np.testing.assert_allclose(
        rust_report.delta_all_fp_removed,
        oracle_report["delta_all_fp_removed"],
        rtol=_TIDE_ORACLE_TOL,
        atol=_TIDE_ORACLE_TOL,
        err_msg="delta_all_fp_removed: rust vs numpy-oracle (DETR-R50 bbox)",
    )
