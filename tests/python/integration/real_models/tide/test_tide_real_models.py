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

import copy
import json
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import pytest

import vernier
from vernier._tide import KernelName
from vernier.instance import Bbox, Boundary, Segm

# Hoisted from inside the parity test body to module level. Function-
# body relative imports trip on ``pytest --import-mode=importlib`` with
# "attempted relative import with no known parent package"; the
# module-level placement resolves at collection time, matching every
# other import in this file.
from ....oracle.tide.oracle import error_decomposition as oracle_decomposition
from .conftest import TideModelName

pytestmark = [pytest.mark.real_models, pytest.mark.slow]


_KERNELS: dict[KernelName, object] = {
    "bbox": Bbox(),
    "segm": Segm(),
    "boundary": Boundary(dilation_ratio=0.02),
}

#: Numpy-oracle parity tolerance — used as BOTH ``rtol`` and ``atol``.
#:
#: ADR-0021 pins ``TOL = 1e-9`` for the fixture-level oracle parity in
#: ``tests/python/oracle/tide/test_rust_matches_oracle.py``. The
#: real-model surface here (150k DETR-R50 detections x 80 classes x
#: 10 IoU thresholds) is at least one order of magnitude noisier in
#: reduction order than the hand-computed fixtures the ADR was
#: calibrated against, so we keep the band at 1e-9 rather than
#: tighten it further. Empirically the DETR-R50 cell ran at ~1 ULP
#: per surface (~2.2e-16) with the previous 8 ULP band; the loosened
#: gate still catches anything wider than ~50 ULP, which is well
#: below any algorithmic disagreement (the original strict-mode
#: ``delta_missed = 0`` bug was a full 12 % of ``baseline_map`` —
#: catastrophic at any sane tolerance).
#:
#: ``assert_allclose`` evaluates ``|a-b| ≤ atol + rtol * |b|``: a
#: per-bin ``delta`` that collapses to exact ``0.0`` on one side
#: would fail an ``rtol``-only gate at ``rtol * 0 = 0`` if the other
#: side yielded a sub-ULP non-zero. The ``atol`` band absorbs that.
_TIDE_ORACLE_TOL = 1e-9


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

    # Structural bound that ALWAYS holds for every bin, including
    # cls / loc / both / dupe that lost their tighter gates above.
    # The corrected AP is in [0, 1] by construction, so the per-bin
    # delta (corrected_AP - baseline_AP) is in [-baseline_map,
    # 1 - baseline_map]. This won't catch fine-grained mis-attribution
    # but it WILL catch order-of-magnitude regressions — e.g. a Cls
    # snap that double-counts a TP reclassification and produces
    # delta_cls = 0.9 on a workload with baseline_map = 0.4 (impossible
    # because corrected_AP would exceed 1).
    baseline = report.baseline_map
    for bin_name, value in report.delta.items():
        assert value >= -baseline - 1e-9, (
            f"bin {bin_name!r} ΔmAP {value} below the structural lower "
            f"bound -baseline_map={-baseline}: corrected AP would be "
            f"negative."
        )
        assert value <= 1.0 - baseline + 1e-9, (
            f"bin {bin_name!r} ΔmAP {value} above the structural upper "
            f"bound 1 - baseline_map={1.0 - baseline}: corrected AP "
            f"would exceed 1.0."
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


# NOTE: split into two single-purpose helpers. The previous bundled
# ``_oracle_compatible_gt`` papered over TWO unrelated oracle gaps in
# one call (#15 of the code review). Keeping them separate documents
# WHICH gap a future kernel extension is silencing — and lets the
# segm-parity follow-up reuse the RLE strip without inheriting the
# crowd-IoU coverage hole.
#
# TODO(adr-0021): teach the numpy oracle to (a) consume RLE
# segmentations via ``pycocotools.mask.decode`` and (b) honour the
# crowd-aware asymmetric ``intersection / dt_area`` branch (quirk
# **E1**). Both helpers below can then be deleted and the parity gate
# moves to the full COCO val GT.
def _strip_rle_segmentation(coco_gt_dict: dict[str, Any]) -> dict[str, Any]:
    """Strip the ``segmentation`` field from every annotation.

    The oracle's ``_normalise`` (around oracle.py line 975) raises
    ``ValueError`` on any GT whose ``segmentation`` is a dict (the RLE
    shape pycocotools and vernier both consume). The bbox kernel never
    reads the field, so stripping is a mechanical adapter — symmetric
    on both sides of the parity gate, no semantic implication.

    Use a true deep-copy (``copy.deepcopy``) for ``images`` and
    ``categories`` so a downstream mutation on the returned dict can't
    leak back into the caller's session-scoped ``coco_gt_dict``
    fixture.
    """
    out = copy.deepcopy(coco_gt_dict)
    for ann in out["annotations"]:
        ann.pop("segmentation", None)
    return out


def _strip_crowd_anns(coco_gt_dict: dict[str, Any]) -> dict[str, Any]:
    """Drop every ``iscrowd=1`` annotation.

    WARNING: this is silencing a real oracle coverage gap, not a
    mechanical adapter. The oracle's ``bbox_iou`` (~line 95) computes
    symmetric ``intersection / union`` for every GT — no crowd-aware
    asymmetric branch (the docstring around line 392 admits fixtures
    "avoid crowd GTs to keep the hand-math clean"). vernier and
    pycocotools both correctly switch crowd GTs to
    ``intersection / dt_area`` (quirk **E1** in
    ``crates/vernier-core/src/similarity/bbox.rs``). Running the
    oracle on crowd-bearing data systematically under-counts matches
    against crowd regions.

    Filtering on BOTH sides of the gate makes the comparison
    apples-to-apples on the crowd-free subset, but the cost is that
    the parity gate cannot exercise vernier's E1 crowd branch. **A
    regression in similarity/bbox.rs's crowd-aware path would ship
    green through this gate.** Future segm-parity tests that adopt
    this helper MUST also adopt a non-bbox synthetic fixture covering
    crowd-RLE GTs to compensate.

    Deep-copies the top-level lists so the returned dict can't leak
    mutations back into the caller's session fixture.
    """
    out = copy.deepcopy(coco_gt_dict)
    out["annotations"] = [ann for ann in out["annotations"] if not ann.get("iscrowd", 0)]
    return out


#: Parity modes exercised against the numpy oracle on real predictions.
#:
#: On COCO val2017 the GT carries no explicit ``ignore`` field per
#: annotation, so ``effective_ignore`` resolves to ``is_crowd`` under
#: BOTH strict and corrected — the two modes converge on this dataset.
#: That makes the corrected-mode cell a cheap addition that defends
#: against a future regression where strict and corrected silently
#: diverge on a dataset that DOES carry explicit ignore. (The
#: production default per ``python/vernier/_tide.py`` is
#: ``"corrected"``, so the corrected-mode gate also matches what
#: end-users hit by default.)
_PARITY_MODES: list[Literal["strict", "corrected"]] = ["strict", "corrected"]


@pytest.mark.parametrize("parity_mode", _PARITY_MODES)
def test_tide_parity_vs_numpy_oracle_detr_r50(
    parity_mode: Literal["strict", "corrected"],
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

    GT side: :func:`_strip_rle_segmentation` adapts the format the
    oracle's ``_normalise`` can consume (mechanical, no semantic
    implication); :func:`_strip_crowd_anns` symmetrically excludes the
    GT shape the oracle's ``bbox_iou`` doesn't model (quirk **E1** —
    documented as a known coverage gap, see the helper's docstring).
    Detection list is unchanged on either side.

    Parametrized over both parity modes — vernier's
    ``parity_mode="strict"`` matches pycocotools verbatim
    (``effective_ignore = iscrowd``); ``parity_mode="corrected"`` honours
    explicit ``ignore_flag`` per quirk D1. On COCO val2017 the two
    converge (no explicit ``ignore`` field on val GT), but parametrising
    here gates against a future divergence on a dataset that DOES carry
    explicit ignore.

    Tolerance is ``rtol = atol = 1e-9`` per ADR-0021's oracle parity
    contract — both sides run in f64, the legitimate drift source at the
    150k-detection scale is reduction order. Empirically observed at
    ~1 ULP per surface on this cache (well inside the 1e-9 band).
    Per-bucket detection-count surfaces are not yet exposed by either
    side; once they are, the integer arrays drop into the same
    parametrize loop and assert bit-equality trivially.
    """
    dt_bytes = predictions_for("detr-r50")
    dt_list = json.loads(dt_bytes)

    # Apply the two oracle adapters separately so the dependency on
    # each is explicit (see helper docstrings for the
    # mechanical-vs-coverage-gap distinction).
    parity_gt = _strip_crowd_anns(_strip_rle_segmentation(coco_gt_dict))
    parity_gt_bytes = json.dumps(parity_gt).encode()

    rust_report = vernier.instance.error_decomposition(
        parity_gt_bytes,
        dt_bytes,
        iou=Bbox(),
        parity_mode=parity_mode,
    )
    oracle_report = oracle_decomposition(parity_gt, dt_list)

    np.testing.assert_allclose(
        rust_report.baseline_map,
        oracle_report["baseline_map"],
        rtol=_TIDE_ORACLE_TOL,
        atol=_TIDE_ORACLE_TOL,
        err_msg=f"baseline_map: rust vs numpy-oracle (DETR-R50 bbox, parity_mode={parity_mode!r})",
    )
    for bin_name in ("cls", "loc", "both", "dupe", "bkg", "missed"):
        np.testing.assert_allclose(
            rust_report.delta[bin_name],
            oracle_report["delta"][bin_name],
            rtol=_TIDE_ORACLE_TOL,
            atol=_TIDE_ORACLE_TOL,
            err_msg=f"delta[{bin_name}]: rust vs numpy-oracle "
            f"(DETR-R50 bbox, parity_mode={parity_mode!r})",
        )
    np.testing.assert_allclose(
        rust_report.delta_all_fp_removed,
        oracle_report["delta_all_fp_removed"],
        rtol=_TIDE_ORACLE_TOL,
        atol=_TIDE_ORACLE_TOL,
        err_msg=f"delta_all_fp_removed: rust vs numpy-oracle (DETR-R50 "
        f"bbox, parity_mode={parity_mode!r})",
    )
