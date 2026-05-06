"""Round-trip the four ``Workload`` variants through the Pydantic
discriminated union (ADR-0033 §"Workload tagged union").

Asserts every variant parses back to the same concrete class and that
the discriminator narrows correctly for downstream callers (the
detection runners that ``isinstance`` against ``InstanceWorkload``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from bench.workloads import (
    InstanceWorkload,
    PanopticWorkload,
    SemanticWorkload,
    StreamingWorkload,
    WorkloadAdapter,
)


def test_instance_workload_roundtrip() -> None:
    wl = InstanceWorkload(
        workload_id="smoke_perfect_match_segm",
        gt_path=Path("/tmp/gt.json"),
        dt_path=Path("/tmp/dt.json"),
        supported_iou_types=frozenset({"bbox", "segm", "boundary"}),
    )
    payload = wl.model_dump(mode="json")
    assert payload["paradigm"] == "instance"

    rebuilt = WorkloadAdapter.validate_python(payload)
    assert isinstance(rebuilt, InstanceWorkload)
    assert rebuilt == wl


def test_panoptic_workload_roundtrip() -> None:
    wl = PanopticWorkload(
        workload_id="coco_panoptic_val2017_perfect",
        gt_png_dir=Path("/tmp/gt_pngs"),
        gt_json=Path("/tmp/gt.json"),
        dt_png_dir=Path("/tmp/dt_pngs"),
        dt_json=Path("/tmp/dt.json"),
        categories_json=Path("/tmp/categories.json"),
    )
    payload = wl.model_dump(mode="json")
    assert payload["paradigm"] == "panoptic"

    rebuilt = WorkloadAdapter.validate_python(payload)
    assert isinstance(rebuilt, PanopticWorkload)
    assert rebuilt == wl


def test_semantic_workload_roundtrip() -> None:
    wl = SemanticWorkload(
        workload_id="cityscapes_val_perfect",
        gt_label_maps=Path("/tmp/gt_labels"),
        dt_label_maps=Path("/tmp/dt_labels"),
        n_classes=19,
        ignore_label=255,
        label_remap={7: 0, 8: 1},
    )
    payload = wl.model_dump(mode="json")
    assert payload["paradigm"] == "semantic"

    rebuilt = WorkloadAdapter.validate_python(payload)
    assert isinstance(rebuilt, SemanticWorkload)
    assert rebuilt == wl


def test_streaming_workload_roundtrip() -> None:
    wl = StreamingWorkload(
        workload_id="coco_val2017_streaming_throughput",
        gt_path=Path("/tmp/gt.json"),
        dt_path=Path("/tmp/dt.json"),
        iou_type="bbox",
        chunk_schedule=(100, 100, 100),
    )
    payload = wl.model_dump(mode="json")
    assert payload["paradigm"] == "streaming"

    rebuilt = WorkloadAdapter.validate_python(payload)
    assert isinstance(rebuilt, StreamingWorkload)
    assert rebuilt == wl


def test_discriminator_routes_panoptic_payload_to_panoptic_variant() -> None:
    raw = {
        "paradigm": "panoptic",
        "workload_id": "coco_panoptic_val2017_perfect",
        "gt_png_dir": "/tmp/gt",
        "gt_json": "/tmp/gt.json",
        "dt_png_dir": "/tmp/dt",
        "dt_json": "/tmp/dt.json",
        "categories_json": "/tmp/cats.json",
    }
    wl = WorkloadAdapter.validate_python(raw)
    assert isinstance(wl, PanopticWorkload)
    # The discriminator narrowed away the instance variant — its
    # ``supported_iou_types`` field doesn't exist on PanopticWorkload.
    assert not hasattr(wl, "supported_iou_types")


def test_unknown_paradigm_is_rejected() -> None:
    raw = {"paradigm": "tracking", "workload_id": "anything"}
    with pytest.raises(ValidationError):
        WorkloadAdapter.validate_python(raw)


def test_panoptic_payload_missing_field_is_rejected() -> None:
    """A panoptic payload without the panoptic-specific fields must
    fail at parse time — that's the entire point of the discriminated
    union over the flat-dataclass shape."""
    raw = {"paradigm": "panoptic", "workload_id": "coco_panoptic_val2017_perfect"}
    with pytest.raises(ValidationError):
        WorkloadAdapter.validate_python(raw)


def test_json_roundtrip_uses_discriminator() -> None:
    """The JSON form carries the ``paradigm`` field; the round-trip
    re-validates through the discriminator."""
    wl = StreamingWorkload(
        workload_id="coco_val2017_streaming_throughput",
        gt_path=Path("/tmp/gt.json"),
        dt_path=Path("/tmp/dt.json"),
        iou_type="bbox",
        chunk_schedule=(50, 50),
    )
    raw = json.loads(wl.model_dump_json())
    rebuilt = WorkloadAdapter.validate_python(raw)
    assert isinstance(rebuilt, StreamingWorkload)
    assert rebuilt == wl
