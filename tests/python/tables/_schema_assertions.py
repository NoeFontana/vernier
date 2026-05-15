"""Shared helpers for Arrow-schema golden-pinning tests.

Used by ``test_slices_schema.py`` (ADR-0046) and
``test_calibration_schema.py`` (ADR-0018). The shape is identical: a
live ``pyarrow.Schema`` extracted from a vernier-emitted
``RecordBatch`` is compared against a JSON-encoded golden under
``tests/python/tables/schemas/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pyarrow as pa

SCHEMA_DIR: Path = Path(__file__).parent / "schemas"


def arrow_type_token(t: pa.DataType) -> str:
    """Map a pyarrow DataType to the lowercase token the golden uses."""
    s = str(t).lower()
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


def schema_to_records(schema: pa.Schema) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for field in schema:
        out.append(
            {
                "name": field.name,
                "type": arrow_type_token(field.type),
                "nullable": bool(field.nullable),
            }
        )
    return out


def schema_metadata(schema: pa.Schema) -> dict[str, str]:
    if schema.metadata is None:
        return {}
    return {k.decode("utf-8"): v.decode("utf-8") for k, v in schema.metadata.items()}


def load_golden(name: str) -> dict[str, Any]:
    path = SCHEMA_DIR / f"{name}.json"
    return json.loads(path.read_text())


def assert_matches_golden(live_schema: pa.Schema, golden_name: str) -> None:
    golden = load_golden(golden_name)
    live_fields = schema_to_records(live_schema)
    assert live_fields == golden["fields"]
    live_meta = schema_metadata(live_schema)
    for k, v in golden["metadata"].items():
        assert live_meta.get(k) == v, f"metadata key {k!r}: live={live_meta.get(k)!r}, golden={v!r}"
