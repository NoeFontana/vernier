"""End-to-end parity validation against DETR-R50 predictions on COCO val2017.

The fixture-based parity suite (``tests/python/parity/test_parity.py``)
proves bit-equality vs pycocotools on hand-curated quirks; the
``coco_val`` smokes (``tests/python/parity/test_coco_val.py``) prove
the same on user-provided detection JSONs. This module closes the loop
by producing those detection JSONs from a SOTA, license-clean,
CPU-runnable Hugging Face model — DETR-R50 — so the parity claim runs
end-to-end against a realistic prediction distribution (long-tail
scores, low-confidence false positives, real class imbalance) instead
of synthetic / GT-derived fixtures.

What this suite gates:

- **Strict bit-equality on the AP-driving aggregates** — mAP and
  every entry of the 12-stat det summary, the dense precision and
  recall tensors, and per-cell `counts`. These are what a user reads
  from `Evaluator.summarize()`; they must match pycocotools exactly.
- **Up-to-2-ULP-relative tolerance on per-image `dtScores`** in the
  eval_imgs snapshot, and on the COCOeval `scores` tensor (the
  per-recall-grid score-threshold projection of dtScores; not to be
  confused with the per-detection model scores). On the first DETR-R50
  val2017 run, ~16% of dtScores diverge by exactly 1 ULP of float64
  and the `scores` tensor inherits the same drift via the recall-grid
  projection. The root cause is a known difference between Rust's
  `serde_json` and Python's `strtod`-based json parser on near-tie
  decimal-to-binary rounding for specific JSON-encoded scores (e.g.
  `0.9992794394493103` parses to the lower-adjacent double in CPython
  and to the upper-adjacent in serde_json). The divergence is
  parser-level and does NOT propagate past the score-threshold
  projection: precision / recall / mAP are bit-equal because AP
  depends only on detection ORDER, not the exact score value. The
  tolerance is expressed as `rtol = 2*eps` (relative), not `atol`, so
  the band tracks score magnitude — `atol = 2*eps` would be ~40 ULP
  at the 0.05 score floor and silently absorb genuine kernel drift on
  the low-confidence tail. Tracked as a follow-up parity item
  (consider tightening serde_json's f64 parser or normalizing scores
  at ingest time); not a blocker for shipping real-prediction
  benchmarks.

Skips cleanly when the ``real-models`` extra is missing
(``transformers`` import fails in conftest), or ``VERNIER_COCO_CACHE``
doesn't point at a populated val2017 layout (GT JSON + images
directory). First-time inference takes ~12-15h on an 8-core CPU
(5000 images x ~9s/image); subsequent runs are seconds thanks to the
predictions cache (see ``real_predictions_cache.detr_resnet50_cache_path``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ....parity.harness import EvalSnapshot, assert_snapshots_equal, snapshot

pytestmark = [pytest.mark.real_models, pytest.mark.slow]

#: 2 ULP of float64 expressed as a RELATIVE tolerance — absorbs the
#: documented ``serde_json`` vs ``strtod`` rounding drift on near-tie
#: JSON-encoded scores. Relative, not absolute: ``eps`` is ULP-at-1.0,
#: so ``atol = 2*eps`` would be ~40 ULP at the 0.05 score floor and
#: hide genuine kernel drift on low-confidence dtScores. ``rtol = 2*eps``
#: scales the band with score magnitude, keeping the "narrower than
#: 4 ULP" guarantee true across the full [0.05, 1.0] score range.
_DTSCORES_RTOL = 2.0 * float(np.finfo(np.float64).eps)


def test_detr_r50_bbox_parity_vs_pycocotools(
    coco_gt_path: Path,
    detr_predictions_path: Path,
) -> None:
    """Run vernier + pycocotools on DETR-R50 predictions; assert parity.

    See the module docstring for the strict-vs-aligned split:
    summary numbers are strict bit-equal; eval_imgs.dtScores is
    aligned to 2 ULP (relative) to absorb the parser-level rounding
    drift.
    """
    ref = snapshot("pycocotools", coco_gt_path, detr_predictions_path, "bbox")
    cand = snapshot("vernier", coco_gt_path, detr_predictions_path, "bbox")

    # Strict tier — these are the numbers a user reads. Identical to
    # what the fixture parity suite would assert.
    _assert_summary_strict(ref, cand)

    # Aligned tier — eval_imgs.dtScores absorbs the parser drift.
    assert_snapshots_equal(ref, cand, rtol=_DTSCORES_RTOL)


def _assert_summary_strict(a: EvalSnapshot, b: EvalSnapshot) -> None:
    """Strict bit-equality on the AP-driving aggregates.

    Carves out exactly the surface ``assert_snapshots_equal`` checks,
    minus ``eval_imgs`` and the ``scores`` tensor (which the
    aligned-tier call below covers). Keeps the parity story
    bisectable: if this strict gate fires, the parser-level drift
    escaped from the dtScores path into precision / recall / mAP,
    and a deeper investigation is warranted.
    """
    assert a.counts == b.counts, f"counts differ: {a.counts} vs {b.counts}"
    np.testing.assert_array_equal(a.precision, b.precision, err_msg="precision")
    np.testing.assert_array_equal(a.recall, b.recall, err_msg="recall")
    np.testing.assert_array_equal(a.stats, b.stats, err_msg="stats")
