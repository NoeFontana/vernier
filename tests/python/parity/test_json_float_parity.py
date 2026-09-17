"""Vernier's JSON float parsing must be bit-equal to CPython's.

``serde_json``'s default parser rounds some near-tie decimals to the
adjacent double. That surfaced on the DETR-R50 real-prediction gate as
~16 % of ``eval_imgs.dtScores`` drifting by exactly 1 ULP against
pycocotools, documented in ``docs/engineering/real-predictions-parity.md``
and held at aligned tier. AP itself stayed bit-equal because it depends
on detection *order*, and 1 ULP does not reorder.

ADR-0054 turns on ``serde_json``'s ``float_roundtrip`` feature, which
makes the parse correctly rounded. This test pins that against the
oracle that matters — CPython's own ``json`` module, i.e. ``strtod`` —
by feeding the same scores through both the JSON path (Rust parses) and
the array path (CPython parses, ADR-0030) and requiring the resulting
``dtScores`` to be identical bit for bit, not merely close.

Both thread counts run: the parallel loader (ADR-0054) must parse the
same doubles as the serial one.
"""

from __future__ import annotations

import json
import struct
from typing import Any

import pytest

from vernier import _core as _vernier_core

# Decimals that `serde_json`'s default (non-round-trip) parser rounds to
# a different double than `strtod`. The first is the value named in the
# real-predictions parity doc as the one that drifted on real DETR-R50
# output; the rest are classic near-ties and boundary cases.
NEAR_TIE_SCORES = [
    0.9992794394493103,
    0.9999999999999999,
    0.30000000000000004,
    0.000244140625,
    0.05,
    0.1,
    123456789.123456789 / 1e9,
    0.8999999999999999,
    0.7000000000000001,
    0.6999999999999998,
]


def _bits(value: float) -> int:
    """The exact f64 payload, so a 1-ULP drift cannot hide in a repr."""
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def _gt_bytes() -> bytes:
    return json.dumps(
        {
            "images": [{"id": 1, "width": 200, "height": 200, "file_name": "a.jpg"}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [10.0, 10.0, 50.0, 50.0],
                    "area": 2500.0,
                    "iscrowd": 0,
                }
            ],
            "categories": [{"id": 1, "name": "thing", "supercategory": "stuff"}],
        }
    ).encode()


def _dt_bytes() -> bytes:
    # `repr` round-trips a float to the shortest decimal that reads back
    # as the same double, so the JSON text names exactly these doubles —
    # any drift is the reader's, not the writer's.
    records = [
        {
            "image_id": 1,
            "category_id": 1,
            "bbox": [10.0, 10.0, 50.0, 50.0],
            "score": score,
        }
        for score in NEAR_TIE_SCORES
    ]
    return json.dumps(records).encode()


def _dt_scores_from_grid(num_threads: int | None) -> list[float]:
    grid = _vernier_core.evaluate_bbox_grid(
        _gt_bytes(),
        _dt_bytes(),
        "strict",
        100,
        use_cats=True,
        num_threads=num_threads,
        # `eval_imgs()` reads per-cell metadata, which is opt-in since
        # retention became a caller's choice (ADR-0055).
        retain_meta=True,
    )
    scores: list[float] = []
    cells: list[dict[str, Any] | None] = grid.eval_imgs()
    for cell in cells:
        if cell is not None:
            scores.extend(float(s) for s in cell["dtScores"])
    return scores


@pytest.mark.parity
@pytest.mark.parametrize("num_threads", [None, 1, 4])
def test_dt_scores_are_bit_equal_to_cpython_json(num_threads: int | None) -> None:
    """The doubles vernier reads must be the doubles CPython reads."""
    oracle = sorted(_bits(record["score"]) for record in json.loads(_dt_bytes().decode()))
    # Every cell of the grid repeats the same detections across the four
    # area ranges, so compare the distinct multiset per area range.
    got = sorted(_bits(s) for s in _dt_scores_from_grid(num_threads))
    assert len(got) % len(oracle) == 0
    repeats = len(got) // len(oracle)
    assert got == sorted(oracle * repeats), (
        "vernier's JSON float parse diverged from CPython's by at least 1 ULP"
    )


@pytest.mark.parity
def test_serial_and_parallel_loaders_read_the_same_doubles() -> None:
    """ADR-0054's split loader must not change a single bit."""
    serial = [_bits(s) for s in _dt_scores_from_grid(None)]
    for num_threads in (1, 2, 4, 8):
        parallel = [_bits(s) for s in _dt_scores_from_grid(num_threads)]
        assert parallel == serial, f"num_threads={num_threads} changed the parsed scores"
