"""Slices RecordBatch schema goldens (ADR-0046).

Each (paradigm, metric) variant of the slices table has a golden
JSON schema under ``tests/python/tables/schemas/``. The harness
materializes a live RecordBatch via the FFI, then compares its
schema (column names, types, nullability, and the
``vernier.schema_version`` / ``vernier.table`` / ``vernier.paradigm``
/ ``vernier.metric`` metadata) against the golden.

A regression that bumps a column type, drops a column, or forgets
to stamp the metadata fails the gold-pin — exactly the contract
ADR-0019 § "schema metadata" mandates and ADR-0046 § F2 inherits.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vernier._core import slices_batch_panoptic, slices_batch_semantic
from vernier.instance import Evaluator, optimal_lrp

from ._schema_assertions import assert_matches_golden

_FIXTURE = Path(__file__).parent.parent / "parity" / "fixtures" / "partition_tiny"


def _evaluate_partitioned_bbox() -> object:
    """Run a small partitioned bbox eval against the partition_tiny
    fixture and return the resulting EvalResult (carrying the slices
    RecordBatch)."""
    gt = (_FIXTURE / "gt.json").read_bytes()
    dt = (_FIXTURE / "dt.json").read_bytes()
    manifest = json.loads((_FIXTURE / "weather_x_tod.json").read_bytes())
    return Evaluator().evaluate(gt, dt, manifest=manifest)


def test_slices_instance_ap_schema() -> None:
    pa = pytest.importorskip("pyarrow")
    result = _evaluate_partitioned_bbox()
    # Pull the RecordBatch from the EvalResult's PyCapsule producer.
    batch = pa.record_batch(result._slices_batch)  # type: ignore[attr-defined]
    assert_matches_golden(batch.schema, "slices_instance_ap")


def test_slices_panoptic_schema() -> None:
    pa = pytest.importorskip("pyarrow")
    # Construct a tiny synthetic row-set for the panoptic builder; we
    # only need to exercise the schema producer.
    batch_capsule = slices_batch_panoptic(
        [("weather", "fog", 1, 1, 0.5, 0.6, 0.83), ("weather", "clear", 1, 1, 0.7, 0.8, 0.9)]
    )
    batch = pa.record_batch(batch_capsule)
    assert_matches_golden(batch.schema, "slices_panoptic")


def test_slices_semantic_schema() -> None:
    pa = pytest.importorskip("pyarrow")
    batch_capsule = slices_batch_semantic(
        [
            ("weather", "fog", 1, 1, 0.45, 0.55, 0.85, 0.65),
            ("weather", "clear", 1, 1, 0.5, 0.6, 0.9, 0.7),
        ]
    )
    batch = pa.record_batch(batch_capsule)
    assert_matches_golden(batch.schema, "slices_semantic")


def test_slices_instance_lrp_schema() -> None:
    pa = pytest.importorskip("pyarrow")
    gt = (_FIXTURE / "gt.json").read_bytes()
    dt = (_FIXTURE / "dt.json").read_bytes()
    manifest = json.loads((_FIXTURE / "weather_x_tod.json").read_bytes())
    report = optimal_lrp(gt, dt, manifest=manifest)
    # Pull the RecordBatch from the partitioned-LRP report's PyCapsule
    # producer. Same access pattern as the AP path.
    batch = pa.record_batch(report._slices_batch)
    assert_matches_golden(batch.schema, "slices_instance_lrp")
