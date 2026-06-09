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

import pytest

from .harness import assert_snapshots_equal, snapshot, subsample_bytes
from .lvis_val_paths import require_perfect_dt_artifacts, sample_image_count


@pytest.mark.parity_lvis_val
@pytest.mark.slow
def test_lvis_v1_val_bbox_strict_parity_perfect_dt() -> None:
    gt_path, dt_path = require_perfect_dt_artifacts("perfect_dt.json")
    gt_bytes, dt_bytes = subsample_bytes(
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
