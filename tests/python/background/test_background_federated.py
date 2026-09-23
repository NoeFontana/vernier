"""A federated ``CocoDataset`` streams to the batch evaluator's numbers (ADR-0065).

The ADR-0026 AC2 per-image cap is applied per ``submit()``, which is the
whole-dataset cap because an image's detections must arrive in one batch.
"""

from __future__ import annotations

import numpy as np
import pytest

from vernier.instance import Bbox, Evaluator

from ..federated_crowding import crowded, federated_handle

_IMAGES = (1, 2)


@pytest.mark.parametrize("num_threads", [None, 2], ids=["serial", "parallel"])
def test_streamed_federated_handle_matches_the_batch_evaluator(num_threads: int | None) -> None:
    handle = federated_handle(_IMAGES)
    ev = Evaluator(iou=Bbox())
    batch = ev.evaluate(handle, np.concatenate([crowded(i) for i in _IMAGES]))

    with ev.background(handle, num_threads=num_threads) as bg:
        for i in _IMAGES:
            bg.submit(crowded(i))
        streamed = bg.finalize()

    assert streamed.stats == batch.stats
    assert batch.stats[0] == 0.0
