"""End-to-end TIDE validation against real-model predictions on COCO val2017.

The numpy-oracle parity contract (ADR-0021, exercised in
``tests/python/oracle/tide/``) proves correctness on synthetic
fixtures; this module proves the same machinery doesn't fall over on
real data — 5000 images, 80 classes, real-model error distributions.

What the suite gates:

- **Coherence** — minimal structural invariants on the report shape
  (``baseline_map`` ∈ [0, 1], ``delta_all_fp_removed`` non-negative
  and ≤ ``1 - baseline_map``, the six expected bin keys present), plus
  per-bin sign guarantees on the *two* bins where they hold across
  every kernel + model: ``bkg`` (non-negative AND bounded above by
  ``delta_all_fp_removed`` — bkg is a strict subset of the all-FP
  rewrite) and ``missed`` (non-negative — the missed fix shrinks
  ``n_pos_gt`` without dropping any TP). The other four FP bins
  (``cls`` / ``loc`` / ``both`` / ``dupe``) carry NEITHER sign nor
  upper-bound invariants on real data; see :func:`_assert_report_coherent`
  for the math.
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

    Per ADR-0021's correctness model, the only properties that hold
    for *every* real-data model + kernel triple are:

    - ``baseline_map`` lives in ``[0, 1]``.
    - ``delta_all_fp_removed`` is non-negative and
      ``baseline_map + delta_all_fp_removed <= 1`` (the upper bound
      mAP can possibly reach once every false positive is dropped).
    - ``delta_all_fp_removed >= delta[bkg]``: removing every false
      positive (the all-FP pass) is by construction a superset of
      removing background-only FPs, and both deltas come from the
      same kind of correction (drop the DT, no TP reclassification).
      So the bkg delta is bounded above by the all-FP delta —
      transitively also non-negative.
    - ``delta[missed] >= 0``: the missed fix deletes unmatched non-
      ignore GTs from the dataset, which strictly shrinks the AP
      denominator without removing any TPs, so the AP curve can
      only move up (see ``crates/vernier-core/src/tide/rewrite.rs``).

    Crucially, **the other four FP bins (``cls`` / ``loc`` / ``both``
    / ``dupe``) carry NEITHER guarantee** on a real-data run:

    - They can exceed ``delta_all_fp_removed``: cls / loc / both
      reclassify FPs into TPs (relabel for cls, snap-bbox for loc),
      which lifts recall in addition to the precision gain the all-
      FP pass already counts. Empirically observed on DETR-R50 (the
      loc bin lands at ~0.076 vs an all-FP total of ~0.053).
    - They can go *negative*: the rewrite operates only at the
      single binning threshold ``t_f`` (per ADR-0021), but AP is
      averaged over the 10-IoU ladder. A DT that's a duplicate at
      ``t_f`` may be the only DT clearing a higher IoU threshold
      against the same GT — dropping it loses recall at that
      stricter threshold and pulls the per-cell AP down. Both
      vernier and the numpy oracle agree on this sign on DETR-R50
      (dupe delta around ``-0.007``); the negativity is a property
      of the mAP-across-IoUs measure, not an implementation bug.

    So the per-bin coherence we gate here is non-negativity for
    ``bkg`` and ``missed`` plus the bkg upper bound. Bin-by-bin
    numerics for all six live in
    :func:`test_tide_parity_vs_numpy_oracle_detr_r50` below.
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

    bkg_value = report.delta["bkg"]
    assert bkg_value >= -1e-9, f"bin 'bkg' has negative ΔmAP: {bkg_value}"
    assert bkg_value <= report.delta_all_fp_removed + 1e-9, (
        f"bin 'bkg' ΔmAP {bkg_value} exceeds perfect-rejection upper "
        f"bound {report.delta_all_fp_removed} — bkg is a strict subset "
        f"of the all-FP rewrite, so this bound MUST hold"
    )

    missed_value = report.delta["missed"]
    assert missed_value >= -1e-9, (
        f"bin 'missed' has negative ΔmAP: {missed_value} — the missed "
        f"fix shrinks the AP denominator without dropping any TP, so "
        f"the per-cell AP can only move up"
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


def _oracle_compatible_gt(coco_gt_dict: dict[str, Any]) -> dict[str, Any]:
    """Strip GT shapes the numpy oracle was never designed to model.

    The TIDE numpy oracle at ``tests/python/oracle/tide/oracle.py`` is
    the executable spec (ADR-0021), but its surface only supports the
    fixtures it was built around — polygon segmentation and non-crowd
    GTs. COCO val2017 carries both shapes the oracle rejects:

    - **RLE segmentation** — oracle ``_normalise`` (around line 975)
      raises ``ValueError`` on any GT whose ``segmentation`` field is a
      dict (the RLE shape pycocotools and vernier both consume on the
      ``segm`` kernel). Strip the field unconditionally — the bbox
      kernel never reads it anyway.
    - **iscrowd anns** — the oracle's ``bbox_iou`` (~line 95) computes
      symmetric ``intersection / union`` for *every* GT, with no
      crowd-aware asymmetric branch (the docstring around line 392
      states fixtures "avoid crowd GTs to keep the hand-math clean").
      vernier and pycocotools correctly switch crowd GTs to
      ``intersection / dt_area`` (quirk **E1** in
      ``crates/vernier-core/src/similarity/bbox.rs``). Running the
      oracle on crowd-bearing data therefore systematically
      under-counts matches against crowd regions and inflates the
      per-class missed/bkg accounting — a known limitation of the
      oracle, not a vernier bug. We filter ``iscrowd=1`` anns out
      symmetrically (both sides) so the comparison stays apples-to-
      apples on the geometry shapes the oracle *can* model.

    Both filters are pure GT-side; the detection list is unchanged
    on either side of the gate. The bbox parity gate is therefore
    "vernier ↔ oracle on the crowd-free, RLE-free GT subset of COCO
    val2017", which is exactly what ADR-0021's spec covers.
    """
    return {
        **coco_gt_dict,
        "annotations": [
            {k: v for k, v in ann.items() if k != "segmentation"}
            for ann in coco_gt_dict["annotations"]
            if not ann.get("iscrowd", 0)
        ],
    }


def test_tide_parity_vs_numpy_oracle_detr_r50(
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

    Both sides see the same GT slice — :func:`_oracle_compatible_gt`
    drops the GT shapes the oracle was never designed to model (RLE
    segmentation, iscrowd anns). The detection list is unchanged on
    either side; only the GT subset shifts. Vernier is called with
    ``parity_mode="strict"`` to bind the comparison to the disposition
    the oracle implements (corrected-mode honours user-supplied
    ``ignore_flag``; the oracle and pycocotools both treat
    ``effective_ignore = iscrowd`` per quirk D1's strict disposition).

    Tolerance is ``rtol = atol = 8 * eps`` on every float surface both
    sides expose (``baseline_map``, the six per-bin ``delta`` entries,
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

    parity_gt = _oracle_compatible_gt(coco_gt_dict)
    parity_gt_bytes = json.dumps(parity_gt).encode()

    rust_report = vernier.instance.error_decomposition(
        parity_gt_bytes,
        dt_bytes,
        iou=Bbox(),
        parity_mode="strict",
    )
    oracle_report = oracle_decomposition(parity_gt, dt_list)

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
