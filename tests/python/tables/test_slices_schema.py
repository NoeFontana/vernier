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
from typing import TYPE_CHECKING, Any

import pytest

from vernier._core import slices_batch_panoptic, slices_batch_semantic
from vernier.instance import Evaluator, optimal_lrp

if TYPE_CHECKING:
    import pyarrow as pa

_FIXTURE = Path(__file__).parent.parent / "parity" / "fixtures" / "partition_tiny"
_SCHEMA_DIR = Path(__file__).parent / "schemas"


def _arrow_type_token(t: pa.DataType) -> str:
    """Map a pyarrow DataType to the lowercase token the golden uses."""
    s = str(t).lower()
    # Normalize a handful of common spellings.
    if s in {"utf8", "large_string", "string", "large_utf8"}:
        return "utf8"
    if s in {"uint64", "uint64()"}:
        return "uint64"
    if s in {"double", "float64"}:
        return "float64"
    if s in {"uint32"}:
        return "uint32"
    if s in {"int64"}:
        return "int64"
    return s


def _schema_to_records(schema: pa.Schema) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for field in schema:
        out.append(
            {
                "name": field.name,
                "type": _arrow_type_token(field.type),
                "nullable": bool(field.nullable),
            }
        )
    return out


def _schema_metadata(schema: pa.Schema) -> dict[str, str]:
    if schema.metadata is None:
        return {}
    return {k.decode("utf-8"): v.decode("utf-8") for k, v in schema.metadata.items()}


def _load_golden(name: str) -> dict[str, Any]:
    path = _SCHEMA_DIR / f"{name}.json"
    return json.loads(path.read_text())


def _assert_matches_golden(live_schema: pa.Schema, golden_name: str) -> None:
    golden = _load_golden(golden_name)
    live_fields = _schema_to_records(live_schema)
    assert live_fields == golden["fields"]
    live_meta = _schema_metadata(live_schema)
    for k, v in golden["metadata"].items():
        assert live_meta.get(k) == v, f"metadata key {k!r}: live={live_meta.get(k)!r}, golden={v!r}"


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
    _assert_matches_golden(batch.schema, "slices_instance_ap")


def test_slices_panoptic_schema() -> None:
    pa = pytest.importorskip("pyarrow")
    # Construct a tiny synthetic row-set for the panoptic builder; we
    # only need to exercise the schema producer.
    batch_capsule = slices_batch_panoptic(
        [("weather", "fog", 1, 1, 0.5, 0.6, 0.83), ("weather", "clear", 1, 1, 0.7, 0.8, 0.9)]
    )
    batch = pa.record_batch(batch_capsule)
    _assert_matches_golden(batch.schema, "slices_panoptic")


def test_slices_semantic_schema() -> None:
    pa = pytest.importorskip("pyarrow")
    batch_capsule = slices_batch_semantic(
        [
            ("weather", "fog", 1, 1, 0.45, 0.55, 0.85, 0.65),
            ("weather", "clear", 1, 1, 0.5, 0.6, 0.9, 0.7),
        ]
    )
    batch = pa.record_batch(batch_capsule)
    _assert_matches_golden(batch.schema, "slices_semantic")


def test_slices_instance_lrp_schema() -> None:
    pa = pytest.importorskip("pyarrow")
    gt = (_FIXTURE / "gt.json").read_bytes()
    dt = (_FIXTURE / "dt.json").read_bytes()
    manifest = json.loads((_FIXTURE / "weather_x_tod.json").read_bytes())
    report = optimal_lrp(gt, dt, manifest=manifest)
    # Pull the RecordBatch from the partitioned-LRP report's PyCapsule
    # producer. Same access pattern as the AP path.
    batch = pa.record_batch(report._slices_batch)
    _assert_matches_golden(batch.schema, "slices_instance_lrp")
