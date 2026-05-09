"""Smoke tests for the vendored mmsegmentation `IoUMetric` oracle
(ADR-0036).

This file's job is to prove the vendored bytes + stub harness work —
that ``IoUMetric`` imports cleanly through the stubs, runs against a
hand-built fixture, and returns the metrics we expect from a
first-principles calculation. Per-quirk strict-mode parity claims
(every ``ms``-keyed row in ``docs/engineering/sem-seg-quirks.md``)
land in a follow-up PR that wires the vernier ``Evaluator`` as the
candidate side.
"""

from __future__ import annotations

import numpy as np
import pytest

from .harness import run_mmsegmentation


@pytest.mark.parity_semantic
def test_oracle_imports_and_runs_on_minimal_fixture() -> None:
    # 4x4 image, 3 classes + ignore_index=255. Construction:
    #   row 0: class 0 (4 pixels)  -- 3 pred 0, 1 pred 1
    #   row 1: class 1 (4 pixels)  -- 4 pred 1
    #   row 2: class 2 (4 pixels)  -- 1 pred 0, 3 pred 2
    #   row 3: ignore  (4 pixels)  -- pred values do not affect metrics
    gt = np.array(
        [
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [255, 255, 255, 255],
        ],
        dtype=np.int64,
    )
    pred = np.array(
        [
            [0, 0, 0, 1],
            [1, 1, 1, 1],
            [0, 2, 2, 2],
            [99, 99, 99, 99],  # under ignore mask, free to be anything
        ],
        dtype=np.int64,
    )

    snap = run_mmsegmentation(pred, gt, num_classes=3, ignore_index=255)

    np.testing.assert_array_equal(snap.intersect, [3.0, 4.0, 3.0])
    np.testing.assert_array_equal(snap.union, [5.0, 5.0, 4.0])
    np.testing.assert_array_equal(snap.pred, [4.0, 5.0, 3.0])
    np.testing.assert_array_equal(snap.label, [4.0, 4.0, 4.0])

    # Per-class IoU: 3/5, 4/5, 3/4
    np.testing.assert_allclose(snap.iou, [0.6, 0.8, 0.75])
    # Per-class Acc (recall): 3/4, 4/4, 3/4
    np.testing.assert_allclose(snap.acc, [0.75, 1.0, 0.75])
    # Overall accuracy: 10 correct / 12 evaluated (ignore excluded)
    np.testing.assert_allclose(snap.aacc, 10.0 / 12.0)


@pytest.mark.parity_semantic
def test_oracle_rejects_unsupported_metric() -> None:
    # Surface check: the oracle's metric-validation path is reachable
    # through our stubs (no mmengine logging machinery suppresses the
    # KeyError IoUMetric raises).
    import torch
    from mmseg.evaluation.metrics.iou_metric import (  # pyright: ignore[reportMissingImports]
        IoUMetric,
    )

    one = torch.tensor([1.0])  # pyright: ignore[reportPrivateImportUsage]
    with pytest.raises(KeyError, match="not supported"):
        IoUMetric.total_area_to_metrics(one, one, one, one, ["mUnknown"])
