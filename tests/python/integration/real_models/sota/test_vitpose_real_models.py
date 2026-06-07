"""End-to-end parity validation against ViTPose-base-simple predictions on COCO val2017.

Sibling cell to the DETR-R50 (bbox) / Mask2Former (panoptic + semantic)
SOTA parity tests; this module closes the keypoints loop. The fixture-
based parity suite proves bit-equality vs pycocotools on hand-curated
OKS quirks; the ``coco_val`` smokes prove the same on user-provided
keypoints JSONs. This suite produces those keypoints JSONs from a SOTA,
license-clean, CPU-runnable Hugging Face model — ViTPose-base-simple
— so the parity claim runs end-to-end against a realistic prediction
distribution (per-joint heatmap confidences, real visibility surface,
top-down crop-driven coordinate warp) instead of synthetic / GT-
derived fixtures.

What this suite gates:

- **Strict bit-equality on the OKS-driving aggregates** — the 10-stat
  keypoints summary (re-indexed area grid per quirk D5: no ``_S`` row),
  the dense precision / recall / counts aggregates. These are the
  numbers a user reads from `Evaluator.summarize()`; they must match
  pycocotools' ``iouType="keypoints"`` exactly.
- **Up-to-2-ULP-relative tolerance on per-image `dtScores`** in the
  eval_imgs snapshot, and on the COCOeval `scores` tensor (the
  per-recall-grid score-threshold projection of dtScores). Same
  ``serde_json`` vs ``strtod`` near-tie rounding drift as the DETR
  cell documents — the divergence is parser-level and does NOT
  propagate past the score-threshold projection (precision / recall /
  AP are bit-equal because OKS depends only on detection ORDER).
  Tolerance expressed as `rtol = 2*eps` (relative), matching the DETR
  cell's rationale verbatim.

Quirk F5 (the v∈{0,1,2} visibility surface) is exercised on the GT
side via the parity fixtures, not by these real-model DT outputs:
pycocotools' OKS evaluator ignores DT-side ``v`` entirely (only the
GT's ``v`` gates which keypoints contribute to the score), and the
predictor hard-pins DT-side ``v`` to {1, 2} (``v=0`` is a GT-side
"not labelled" concept). A DT-side ``seen_v == {1, 2}`` assertion
would be tautological on a working ViTPose run rather than a
meaningful F5 check.

Skips cleanly when the ``real-models`` extra is missing
(``transformers`` import fails in conftest), when the ViTPose
revision is the ``_UNPINNED_REVISION`` sentinel, or when
``VERNIER_COCO_CACHE`` doesn't point at a populated val2017 layout
(keypoints GT JSON + images directory). First-time inference takes
~2-3h on an 8-core CPU (~11k GT person boxes x ~1s/crop);
subsequent runs are seconds thanks to the predictions cache (see
``real_predictions_cache.vitpose_cache_path``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ....parity.harness import EvalSnapshot, assert_snapshots_equal, snapshot

pytestmark = [pytest.mark.real_models, pytest.mark.slow]

#: 2 ULP of float64 expressed as a RELATIVE tolerance — absorbs the
#: documented ``serde_json`` vs ``strtod`` rounding drift on near-tie
#: JSON-encoded scores. Same rationale as the DETR-R50 cell: ``rtol``
#: (not ``atol``) so the band tracks score magnitude across the full
#: [0.0, 1.0] keypoint-score range, keeping the "narrower than 4 ULP"
#: guarantee true even at the low-confidence tail.
_DTSCORES_RTOL = 2.0 * float(np.finfo(np.float64).eps)


def test_vitpose_keypoints_parity_vs_pycocotools(
    coco_kp_gt_path: Path,
    vitpose_predictions_path: Path,
) -> None:
    """Run vernier + pycocotools on ViTPose predictions; assert OKS parity.

    See the module docstring for the strict-vs-aligned split: the
    10-stat keypoints summary + dense precision/recall/counts are
    strict bit-equal; eval_imgs.dtScores + the COCOeval scores tensor
    are aligned to 2 ULP (relative) to absorb the parser-level
    rounding drift.
    """
    ref = snapshot("pycocotools", coco_kp_gt_path, vitpose_predictions_path, "keypoints")
    cand = snapshot("vernier", coco_kp_gt_path, vitpose_predictions_path, "keypoints")

    # Strict tier — these are the numbers a user reads. Identical
    # surface to what the fixture parity suite would assert (minus the
    # ``scores`` tensor, which the aligned-tier call below covers).
    _assert_summary_strict(ref, cand)

    # Aligned tier — eval_imgs.dtScores absorbs the parser drift.
    assert_snapshots_equal(ref, cand, rtol=_DTSCORES_RTOL)


def _assert_summary_strict(a: EvalSnapshot, b: EvalSnapshot) -> None:
    """Strict bit-equality on the OKS-driving aggregates.

    Mirrors the DETR cell's helper exactly: carves out the surface
    ``assert_snapshots_equal`` checks minus ``eval_imgs`` and the
    ``scores`` tensor (which the aligned-tier call below covers). Keeps
    the parity story bisectable: if this strict gate fires, the
    parser-level drift escaped from the dtScores path into precision /
    recall / AP, and a deeper investigation is warranted.

    The ``stats`` array is the 10-element keypoints summary (per
    ADR-0012, quirk D5 — re-indexed A-axis, no ``_S`` row), distinct
    from detection's 12-stat vector; ``np.testing.assert_array_equal``
    handles both shapes transparently.
    """
    assert a.counts == b.counts, f"counts differ: {a.counts} vs {b.counts}"
    np.testing.assert_array_equal(a.precision, b.precision, err_msg="precision")
    np.testing.assert_array_equal(a.recall, b.recall, err_msg="recall")
    np.testing.assert_array_equal(a.stats, b.stats, err_msg="stats")
