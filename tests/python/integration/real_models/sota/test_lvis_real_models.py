"""Real-prediction parity smoke for Deformable-DETR LVIS vs ``lvis-api``.

Sibling to ``test_detr_real_models.py`` /
``test_mask2former_panoptic_real_models.py``: real DT, real GT,
strict + aligned two-tier parity claim. The model is
``facebook/deformable-detr-box-supervised`` (Box-Supervised
DeformDETR-R50-4x from the Detic release); predictions land at
``lvis_detector_cache_path()`` as one LVIS results JSON keyed on the
pinned hub commit SHA.

This is the only real-prediction cell exercising the federated-
evaluation paradigm — AA3 (``not_exhaustive_category_ids`` ->
``dt_ignore`` propagation) and AA4 (per-image cell skip when the
category is missing from both positive AND not-exhaustive id sets)
are gated on synthetic ``federated_min`` fixtures elsewhere; this
cell exercises both on a real model's output distribution.

What this suite gates:

- **Coverage** — every image in the (sub-sampled) GT prefix has at
  least the right to a DT entry (vacuous empty list is fine; missing
  image ids are not). A partial populator run cannot pass this gate.
- **Strict bit-equality on the per-cell integer surface** —
  per-(category, area, image) ``dt_matches``, ``dt_ignore``,
  ``gt_ignore`` boolean arrays. Integer / boolean reductions cannot
  drift on reduction order, so any divergence here is a real
  federated-evaluation accumulator bug. This is the load-bearing
  parity claim against ``lvis-api``'s ``LVISEval.eval_imgs``.
- **Aligned-tier float drift, 8 ULP relative + absolute** — the
  ``(T, R, K, A)`` precision tensor and the 13-entry LVIS summary
  plan (AP, AP50, AP75, APs/m/l, AP_r/c/f, AR@300, ARs/m/l@300). The
  ``rtol=atol=8*eps`` band is the same gate the Mask2Former panoptic
  cell uses; ``atol`` keeps a category whose oracle metric collapses
  to ``0.0`` from spuriously failing an ``rtol``-only gate.

## Sub-sampling (memory bound)

Same constraint as ``tests/python/parity_lvis/test_lvis_val.py``:
vernier's evaluation orchestrator stores cells in a dense
``Vec<Option<PerImageEval>>`` of length
``n_categories * n_area_ranges * n_images``. Full LVIS v1 val is
1203 * 4 * 19809 ~= 95M slots (~22 GB even when most cells are
``None``, before populated payloads). To keep this test runnable on
a 16 GB dev box AND materialise per-cell payloads for the strict
integer-surface diff, we default to a 1000-image prefix
(``VERNIER_LVIS_REAL_VAL_SAMPLE_IMAGES=1000``). Override to ``-1``
on a 32 GB+ host to run the full corpus.

Skips cleanly when:

- ``real-models`` extra is missing (conftest's ``pytest.importorskip``).
- LVIS detector revision is the ``_UNPINNED_REVISION`` sentinel.
- LVIS v1 val cache is not provisioned (``python -m lvis_v1_val_cache fetch``).
- Vendored ``lvis-api`` oracle is missing from ``sys.path``.

First-time inference takes ~48-72 h on an 8-core CPU
(19,809 images at ~10s/image for Deformable-DETR's 300-query
6-layer decoder). Subsequent runs read from disk in seconds.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import numpy as np
import pytest

from ....parity_lvis.harness import assert_snapshots_equal, snapshot, subsample_bytes

pytestmark = [pytest.mark.real_models, pytest.mark.slow]

#: 8 ULP of float64 — used as BOTH ``rtol`` and ``atol`` on the
#: ``precision`` tensor and 13-entry summary plan. Same gate the
#: Mask2Former panoptic cell uses (see
#: ``test_mask2former_panoptic_real_models._PANOPTIC_PARITY_RTOL``).
#: Reduction-order drift on the 1203-category federated mean stays
#: within 1-2 ULP per category on the live cache; 8 ULP keeps any
#: genuine kernel divergence (e.g., a wrong IoU numerator for one
#: image) well above the gate.
_LVIS_PARITY_RTOL = 8.0 * float(np.finfo(np.float64).eps)
_LVIS_PARITY_ATOL = _LVIS_PARITY_RTOL

#: Env-var contract for the sub-sample knob. Distinct from
#: ``VERNIER_LVIS_VAL_SAMPLE_IMAGES`` (which gates the
#: synthetic-DT parity_lvis smoke) so a host that's sized to run the
#: full synthetic smoke can keep a smaller real-data prefix without
#: a per-test override.
_SAMPLE_IMAGES_ENV = "VERNIER_LVIS_REAL_VAL_SAMPLE_IMAGES"
_DEFAULT_SAMPLE_IMAGES = 1000


def _sample_image_count() -> int:
    raw = os.environ.get(_SAMPLE_IMAGES_ENV)
    if raw is None or raw == "":
        return _DEFAULT_SAMPLE_IMAGES
    try:
        n = int(raw)
    except ValueError:
        return _DEFAULT_SAMPLE_IMAGES
    if n == 0:
        return _DEFAULT_SAMPLE_IMAGES
    return n


def _assert_coverage(gt_bytes: bytes, dt_bytes: bytes) -> None:
    """Every GT image has at least one DT record emitted for it.

    The populator runs forward on every image in ``gt['images']``
    and writes a per-image record at least once (the LVIS results
    JSON is a flat list, but the populator writes one entry per
    above-threshold detection — so a fully empty image, when its
    forward pass produced zero above-threshold scores, would still
    show up via a sentinel ``image_id``-only row from the populator's
    contract). This gate covers the case where the populator's
    per-image loop silently skipped an image entirely — e.g. an
    unhandled exception caught and continued — leaving the GT image
    with NO corresponding DT entry at all. Mirrors the Mask2Former
    panoptic coverage gate in
    ``test_mask2former_panoptic_real_models.py``.

    Run BEFORE sub-sampling: the sub-sample filter keeps only DT
    records whose ``image_id`` is in the prefix, so running this gate
    on post-subsample bytes is tautological (a populator that dropped
    half the GT would still pass). On the full bytes, a populator
    that skipped an image leaves a real ``missing = gt - dt`` gap we
    surface as a contract regression.
    """
    gt = json.loads(gt_bytes)
    dt = json.loads(dt_bytes)
    gt_image_ids = {int(im["id"]) for im in gt["images"]}
    dt_image_ids = {int(d["image_id"]) for d in dt}
    missing = gt_image_ids - dt_image_ids
    assert not missing, (
        f"populator produced predictions for {len(dt_image_ids)} images but "
        f"GT has {len(gt_image_ids)} — {len(missing)} GT images missing from "
        f"DT (first 5: {sorted(missing)[:5]}). Either the populator's "
        f"per-image loop silently skipped images, or the GT/DT caches were "
        f"pinned to different revisions."
    )


def test_lvis_detector_parity_vs_lvis_api(
    lvis_v1_val_paths: tuple[Path, Path],
    lvis_detector_predictions_path: Path,
) -> None:
    """Two-tier parity vs ``lvis-api`` on real Deformable-DETR predictions.

    Strict tier: per-cell ``dt_matches`` / ``dt_ignore`` / ``gt_ignore``
    bit-equal across all materialized cells (the integer surface that
    cannot drift on reduction order; this is where the federated-
    evaluation AA3 + AA4 semantics live on real data).

    Aligned tier: precision tensor + 13-entry LVIS summary plan
    (AP, AP50/75, APs/m/l, AP_r/c/f, AR@300, ARs/m/l@300) at
    8 ULP relative AND absolute. Same band as the Mask2Former
    panoptic cell — symmetric across the float-zero boundary so a
    rare category whose oracle AP_r collapses to 0.0 doesn't fail
    an rtol-only gate.

    Sub-sampled to ``VERNIER_LVIS_REAL_VAL_SAMPLE_IMAGES`` (default
    1000) — see the module docstring for the memory-bound rationale.
    """
    gt_path, _ = lvis_v1_val_paths
    gt_bytes = gt_path.read_bytes()
    dt_bytes = lvis_detector_predictions_path.read_bytes()

    # Coverage check runs on the FULL bytes — sub-sampling drops DT
    # records whose image_id isn't in the prefix, which would mask a
    # populator skip on any image outside the prefix.
    _assert_coverage(gt_bytes, dt_bytes)
    gt_sub, dt_sub = subsample_bytes(gt_bytes, dt_bytes, _sample_image_count())

    # Sequence the two snapshots — holding both peaks of the
    # cell-list materialisation simultaneously is the OOM path on
    # the sub-sampled 1000-image prefix. ``assert_snapshots_equal``
    # is the merged strict (eval_imgs) + aligned (precision / stats)
    # check; passing rtol/atol scopes the parametric band to the
    # float-surface entries only.
    ref = snapshot("lvis_api", gt_sub, dt_sub, max_dets=300, include_eval_imgs=True)
    gc.collect()
    cand = snapshot("vernier", gt_sub, dt_sub, max_dets=300, include_eval_imgs=True)

    assert_snapshots_equal(
        ref,
        cand,
        rtol=_LVIS_PARITY_RTOL,
        atol=_LVIS_PARITY_ATOL,
    )
