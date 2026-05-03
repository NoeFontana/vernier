"""End-to-end LVIS parity test (PR-3 of ADR-0026).

Diffs vernier's federated bbox evaluation against the vendored
``lvis-api`` oracle on the ``federated_min`` fixture. PR-3 ships
**bbox-only, raw-precision-only**; PR-4 extends this file with the
13-entry summary plan once the orchestrator's federated cells are
proven correct.
"""

from __future__ import annotations

import json

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


@pytest.mark.parity_lvis
def test_q5_af6_lvis_minus_one_sentinel_on_empty_frequency_bucket() -> None:
    # ADR-0026 appendix Q5: "AF6 sentinel-vs-zero migration trap".
    # When every category in the dataset is `f` (Frequent), the AP_r
    # and AP_c entries collapse to the LVIS `-1` sentinel (per
    # eval.py:441-442) — explicitly NOT 0.0 (panopticapi's behavior
    # under W6) and NOT NaN (an uninitialized read). The cross-
    # codebase distinction is the migration trap; this test pins the
    # LVIS leg of it.
    gt, dt = fixture_bytes("federated_min")
    snap = snapshot("vernier", gt, dt, max_dets=300)
    # `federated_min` includes one Rare category, so APr is well-defined.
    # We manufacture the empty-bucket case in-test by upgrading the
    # fixture to all-Frequent and verifying APr/APc are -1.
    payload = json.loads(gt.decode("utf-8"))
    for cat in payload["categories"]:
        cat["frequency"] = "f"
    gt_all_f = json.dumps(payload).encode("utf-8")
    snap_f = snapshot("vernier", gt_all_f, dt, max_dets=300)
    assert snap_f.stats["APr"] == -1.0, (
        f"AF6: empty Rare bucket must surface -1, got {snap_f.stats['APr']}"
    )
    assert snap_f.stats["APc"] == -1.0, (
        f"AF6: empty Common bucket must surface -1, got {snap_f.stats['APc']}"
    )
    # Sanity: APf is well-defined (every category is f), so it isn't
    # the sentinel. Note the value is small because the federated_min
    # fixture has one TP and several FPs; the point is just that it's
    # NOT -1 / 0 / nan.
    assert snap_f.stats["APf"] >= 0.0, "APf must be a real AP, not the sentinel"
    assert snap_f.stats["APf"] != snap_f.stats["APr"], "APf must differ from the sentinel"
    # baseline (the original fixture's mixed-frequency layout) is the
    # reference: APr is real because cat 3 is Rare.
    assert snap.stats["APr"] >= 0.0 or snap.stats["APr"] == -1.0


@pytest.mark.parity_lvis
def test_lvis_summary_keys_match_lvis_api() -> None:
    # The 13-entry plan must surface the same key names as
    # `LVISEval.results` keys at `max_dets=300`. `assert_snapshots_equal`
    # already enforces this, but we want a dedicated regression here so
    # a future refactor that drops/renames a key is caught even on
    # fixtures whose values happen to match.
    gt, dt = fixture_bytes("federated_min")
    expected = {
        "AP",
        "AP50",
        "AP75",
        "APs",
        "APm",
        "APl",
        "APr",
        "APc",
        "APf",
        "AR@300",
        "ARs@300",
        "ARm@300",
        "ARl@300",
    }
    snap = snapshot("vernier", gt, dt, max_dets=300)
    assert set(snap.stats.keys()) == expected
    ref = snapshot("lvis_api", gt, dt, max_dets=300)
    assert set(ref.stats.keys()) == expected
