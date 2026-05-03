"""LVIS v1 val parity smoke (ADR-0026 PR-6).

Env-gated end-to-end check: vernier's federated bbox evaluation
against the vendored ``lvis-api`` oracle on the LVIS v1 val
dataset. Strict bit-equality on the 13-entry summary plan is the
headline contract; per-cell divergence is what the fixture-scale
``federated_min`` test pins, so this smoke focuses on the summary
shape and skips per-cell materialization to keep the resident set
bounded.

Skipped on a clean checkout — populate the cache with
``python -m lvis_val_cache`` first.

## Sub-sampling (memory bound)

vernier's evaluation orchestrator stores cells in a dense
``Vec<Option<PerImageEval>>`` of length
``n_categories * n_area_ranges * n_images``. On full LVIS v1 val
that is 1203 * 4 * 19809 ~= 95 M slots, ~232 bytes per slot —
about 22 GB resident even when most cells are ``None`` (federated
AA4 skip is a discriminant tag, not a slot omission). Out of reach
on a 16 GB dev box.

The smoke defaults to a 1000-image prefix
(``VERNIER_LVIS_VAL_SAMPLE_IMAGES=1000``) which keeps the peak
under ~5 GB and runs in well under a minute on each impl. This is
substantial enough to exercise every federated path (~85% of LVIS
val's 1203 categories appear at least once in the first 1000
images), the AC2 trim, the AF6 sentinel on rare classes that don't
hit the prefix, and the AB3 frequency-bucketed mean. The dense-grid
memory blow-up is a known orchestrator-side limit; sparse cell
storage is the follow-up perf push.

To run on the full corpus, set ``VERNIER_LVIS_VAL_SAMPLE_IMAGES=-1``
on a machine with ≥ 32 GB RAM.
"""

from __future__ import annotations

import gc
import json

import pytest

from .harness import assert_snapshots_equal, snapshot
from .lvis_val_paths import require_perfect_dt_artifacts, sample_image_count


def _subsample(gt_bytes: bytes, dt_bytes: bytes, n_images: int) -> tuple[bytes, bytes]:
    """Trim the GT + DT to the first ``n_images`` image ids
    (id-ascending). When ``n_images < 0`` returns the inputs verbatim.

    Uses the input image order rather than a sort because the upstream
    `lvis_v1_val.json` has been shuffled by frequency at publication
    time; taking the first N gives a frequency-balanced slice without
    having to bucket.
    """
    if n_images < 0:
        return gt_bytes, dt_bytes
    gt = json.loads(gt_bytes)
    dt = json.loads(dt_bytes)
    keep_imgs = gt["images"][:n_images]
    keep_ids = {im["id"] for im in keep_imgs}
    gt_sub = {
        **gt,
        "images": keep_imgs,
        "annotations": [a for a in gt["annotations"] if a["image_id"] in keep_ids],
    }
    dt_sub = [d for d in dt if d["image_id"] in keep_ids]
    return json.dumps(gt_sub).encode("utf-8"), json.dumps(dt_sub).encode("utf-8")


@pytest.mark.parity_lvis_val
@pytest.mark.slow
def test_lvis_v1_val_bbox_strict_parity_perfect_dt() -> None:
    gt_path, dt_path = require_perfect_dt_artifacts("perfect_dt.json")
    gt_bytes, dt_bytes = _subsample(
        gt_path.read_bytes(), dt_path.read_bytes(), sample_image_count()
    )
    # `include_eval_imgs=False` skips the per-cell payload list at
    # both impls. The dense grid still allocates the cell *slots*
    # — the boolean only short-circuits the harness's Python-side
    # materialization. The precision tensor + 13-stat summary diff
    # captures every aggregate divergence the per-cell payload
    # would; the fixture-scale `federated_min` test pins per-cell
    # behavior on a small corpus where the materialization fits.
    ref = snapshot("lvis_api", gt_bytes, dt_bytes, max_dets=300, include_eval_imgs=False)
    # Free the lvis-api oracle's intermediate storage before vernier
    # runs. Holding both peaks simultaneously is the OOM path;
    # sequencing them keeps the resident set bounded by the larger
    # of the two.
    gc.collect()
    cand = snapshot("vernier", gt_bytes, dt_bytes, max_dets=300, include_eval_imgs=False)
    assert_snapshots_equal(ref, cand)
