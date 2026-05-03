"""End-to-end TIDE validation against rf-detr predictions on COCO val2017.

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

What the suite does *not* gate (deferred to ``run.py``):

- Wall-clock or memory budgets — hardware-dependent; the README pins
  the target.
- Specific bin-mass distributions — model-dependent and not a
  property of the implementation.

Skips cleanly when the ``real-models`` extra is missing or
``VERNIER_COCO_CACHE`` doesn't point at a populated val2017 layout
(GT JSON + images directory).
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

import vernier
from vernier import Bbox, Boundary, Segm
from vernier._tide import KernelName

from ._rfdetr_predict import ModelName

pytestmark = pytest.mark.real_models


_KERNELS: dict[KernelName, object] = {
    "bbox": Bbox(),
    "segm": Segm(),
    "boundary": Boundary(dilation_ratio=0.02),
}


def _assert_report_coherent(report: vernier.TideReport) -> None:
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
    ],
)
def test_tide_coherence_on_real_predictions(
    model_name: ModelName,
    kernel_name: KernelName,
    coco_gt_bytes: bytes,
    predictions_for: Callable[[ModelName], bytes],
) -> None:
    """Run TIDE on rf-detr predictions; verify structural invariants hold.

    Parametrized to cover all three production kernels with the smallest
    relevant model that exercises each (RFDETRNano covers bbox;
    RFDETRSegNano covers segm + boundary, since boundary IoU is computed
    from the same masks segm already needs).
    """
    report = vernier.error_decomposition(
        coco_gt_bytes,
        predictions_for(model_name),
        iou=_KERNELS[kernel_name],
    )
    _assert_report_coherent(report)


def test_tide_determinism_on_real_predictions(
    coco_gt_bytes: bytes,
    predictions_for: Callable[[ModelName], bytes],
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
    a = vernier.error_decomposition(coco_gt_bytes, predictions, iou=Bbox())
    b = vernier.error_decomposition(coco_gt_bytes, predictions, iou=Bbox())

    assert a == b, (
        f"non-deterministic TIDE: report A and B differ.\n"
        f"  A.baseline_map = {a.baseline_map}\n"
        f"  B.baseline_map = {b.baseline_map}\n"
        f"  A.delta = {a.delta}\n"
        f"  B.delta = {b.delta}\n"
        f"  A.delta_all_fp_removed = {a.delta_all_fp_removed}\n"
        f"  B.delta_all_fp_removed = {b.delta_all_fp_removed}\n"
    )
