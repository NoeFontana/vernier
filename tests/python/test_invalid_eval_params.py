"""Tests for the cross-paradigm ``InvalidEvalParams`` exception hierarchy.

ADR-0039 ratifies the shape: a single ``InvalidEvalParams`` base
(subclass of ``ValueError``) with three paradigm-specific subclasses
re-exported from every paradigm's namespace. Each instance carries
``field``, ``value``, and ``remediation`` attributes; the formatted
message embeds the field name and value verbatim so users debugging
a misconfigured ``Evaluator`` can grep for either.

Phase 1 ships the hierarchy alone; phases 2-4 wire the per-paradigm
``__post_init__`` validation against it.
"""

from __future__ import annotations

import pytest

from vernier import instance, panoptic, semantic


def test_base_class_object_is_shared_across_paradigms() -> None:
    assert instance.InvalidEvalParams is semantic.InvalidEvalParams
    assert instance.InvalidEvalParams is panoptic.InvalidEvalParams


def test_subclasses_inherit_from_shared_base() -> None:
    assert issubclass(instance.InvalidInstanceParams, instance.InvalidEvalParams)
    assert issubclass(semantic.InvalidSemanticParams, instance.InvalidEvalParams)
    assert issubclass(panoptic.InvalidPanopticParams, instance.InvalidEvalParams)


def test_base_inherits_from_value_error() -> None:
    assert issubclass(instance.InvalidEvalParams, ValueError)


@pytest.mark.parametrize(
    "exc",
    [
        instance.InvalidInstanceParams,
        semantic.InvalidSemanticParams,
        panoptic.InvalidPanopticParams,
    ],
)
def test_constructor_populates_attributes_and_message(
    exc: type[instance.InvalidEvalParams],
) -> None:
    e = exc(field="iou_thresholds", value=[1.5], remediation="must be in [0.0, 1.0]")
    assert e.field == "iou_thresholds"
    assert e.value == [1.5]
    assert e.remediation == "must be in [0.0, 1.0]"
    msg = str(e)
    assert "iou_thresholds" in msg
    assert "1.5" in msg
    assert "must be in [0.0, 1.0]" in msg


def test_caller_can_catch_paradigm_specific_or_base() -> None:
    """A user can catch the paradigm-specific subclass or the shared base."""
    with pytest.raises(instance.InvalidInstanceParams):
        raise instance.InvalidInstanceParams(field="x", value=1, remediation="bad")
    with pytest.raises(instance.InvalidEvalParams):
        raise instance.InvalidInstanceParams(field="x", value=1, remediation="bad")
    with pytest.raises(ValueError, match="invalid x"):
        raise instance.InvalidInstanceParams(field="x", value=1, remediation="bad")


def test_constructor_is_keyword_only() -> None:
    """The constructor refuses positional args — the field/value/remediation
    triplet is non-obvious in positional form, so we force keyword use."""
    with pytest.raises(TypeError):
        instance.InvalidInstanceParams("iou_thresholds", [1.5], "bad")  # type: ignore[misc]
