"""End-to-end LVIS parity test (PR-3 of ADR-0026).

Diffs vernier's federated bbox evaluation against the vendored
``lvis-api`` oracle on the ``federated_min`` fixture. PR-3 ships
**bbox-only, raw-precision-only**; PR-4 extends this file with the
13-entry summary plan once the orchestrator's federated cells are
proven correct.
"""

from __future__ import annotations

import pytest

from .harness import (
    LvisSnapshot,
    assert_snapshots_equal,
    fixture_bytes,
    snapshot,
)


@pytest.mark.parity_lvis
def test_federated_min_bbox_strict_parity() -> None:
    gt, dt = fixture_bytes("federated_min")
    ref: LvisSnapshot = snapshot("lvis_api", gt, dt, max_dets=300)
    cand: LvisSnapshot = snapshot("vernier", gt, dt, max_dets=300)
    assert_snapshots_equal(ref, cand)


@pytest.mark.parity_lvis
def test_federated_min_aa4_skips_image1_cat3() -> None:
    # The image 1 x cat 3 cell sits at flat index `k=2, a=0..3, i=0`.
    # vernier must emit `None` there; both the oracle and vernier
    # already agree on `eval_imgs` shape per `assert_snapshots_equal`,
    # but pinning the cell-level absence here protects against a
    # future refactor that "accidentally" populates it via a bug
    # downstream of the K-axis layout.
    gt, dt = fixture_bytes("federated_min")
    cand = snapshot("vernier", gt, dt, max_dets=300)
    # K=3 (alpha, beta, gamma sorted by id-ascending), A=4 (all/s/m/l),
    # I=2 (image 1, image 2 sorted by id-ascending). We know image 1
    # is i=0, cat 3 is k=2, the `all` area bucket is a=0.
    n_a, n_i = 4, 2
    flat_image1_cat3_all = 2 * n_a * n_i + 0 * n_i + 0
    assert cand.eval_imgs[flat_image1_cat3_all] is None, (
        "AA4: image 1 x cat 3 must be skipped (no GT, no neg listing)"
    )
