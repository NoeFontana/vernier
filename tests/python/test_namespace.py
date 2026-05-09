"""Tests for the ADR-0029 namespace restructure.

Asserts the public-API contract:

- ``vernier.instance.X`` resolves for every relocated AP-fold symbol.
- ``vernier.panoptic.X`` resolves for every relocated panoptic symbol.
- The flat root keeps only the cross-paradigm shared types and the
  pycocotools migration shim.
- Old flat-root names raise :class:`AttributeError` (B1 chosen — no
  re-exports for moved symbols).
- ``IouKind`` discriminator identity survives the move (a value
  constructed via :class:`vernier.instance.Bbox` matches the discriminated
  union the evaluator's ``case Bbox():`` arm pattern-matches against).
"""

from __future__ import annotations

import json

import pytest

import vernier
from vernier.instance import Bbox, Boundary, Evaluator, IouKind, Keypoints, Segm

ROOT_STAYS: tuple[str, ...] = (
    "COCOeval",
    "Frequency",
    "ParityMode",
    "__version__",
    "patch_pycocotools",
    "version",
    "instance",
    "panoptic",
    "semantic",
)

# Symbols that previously lived at ``vernier.X`` but have moved into the
# instance/panoptic/semantic submodules per ADR-0029. None of them should
# resolve at the root any more.
RELOCATED_NAMES: tuple[str, ...] = (
    "BackgroundEvaluator",
    "Bbox",
    "Boundary",
    "ClassPanopticStats",
    "ClassSemanticStats",
    "ConfusionMatrix",
    "Dataset",
    "EvalResult",
    "Evaluator",
    "FpIouHistogram",
    "IouKind",
    "Keypoints",
    "MemoryBudgetWarning",
    "OutOfBudgetError",
    "PanopticDataset",
    "PanopticEvaluator",
    "PanopticPredictions",
    "PanopticSummary",
    "QueueFullError",
    "Segm",
    "SemanticDataset",
    "SemanticEvaluator",
    "SemanticPredictions",
    "SemanticSummary",
    "StreamingEvaluator",
    "StreamingSemanticEvaluator",
    "Summary",
    "TableName",
    "TablesConfig",
    "TideConfig",
    "TideReport",
    "confusion_matrix",
    "error_decomposition",
    "fp_iou_histogram",
)

INSTANCE_NAMES: tuple[str, ...] = (
    "BackgroundEvaluator",
    "Bbox",
    "Boundary",
    "CocoDataset",
    "EvalResult",
    "Evaluator",
    "FpIouHistogram",
    "IouKind",
    "Keypoints",
    "MemoryBudgetWarning",
    "OutOfBudgetError",
    "QueueFullError",
    "Segm",
    "Summary",
    "TableName",
    "TablesConfig",
    "TideConfig",
    "TideReport",
    "confusion_matrix",
    "error_decomposition",
    "fp_iou_histogram",
)

PANOPTIC_NAMES: tuple[str, ...] = (
    "ClassPanopticStats",
    "Dataset",
    "Evaluator",
    "Predictions",
    "Summary",
)

SEMANTIC_NAMES: tuple[str, ...] = (
    "ClassSemanticStats",
    "ConfusionMatrix",
    "Dataset",
    "Evaluator",
    "Predictions",
    "Summary",
)


@pytest.mark.parametrize("name", ROOT_STAYS)
def test_root_keeps_only_shared_and_shim(name: str) -> None:
    assert hasattr(vernier, name), f"root must keep: {name}"


@pytest.mark.parametrize("name", RELOCATED_NAMES)
def test_old_flat_root_names_are_gone(name: str) -> None:
    """Pre-1.0, B1 chosen: no re-exports for moved symbols."""
    assert not hasattr(vernier, name), (
        f"{name!r} should not resolve at the root after ADR-0029; "
        "relocated to vernier.instance, vernier.panoptic, or vernier.semantic"
    )


@pytest.mark.parametrize("name", INSTANCE_NAMES)
def test_instance_submodule_exports(name: str) -> None:
    assert hasattr(vernier.instance, name), f"missing vernier.instance.{name}"


@pytest.mark.parametrize("name", PANOPTIC_NAMES)
def test_panoptic_submodule_exports(name: str) -> None:
    assert hasattr(vernier.panoptic, name), f"missing vernier.panoptic.{name}"


@pytest.mark.parametrize("name", SEMANTIC_NAMES)
def test_semantic_submodule_exports(name: str) -> None:
    assert hasattr(vernier.semantic, name), f"missing vernier.semantic.{name}"


def test_ffi_reexport_identities() -> None:
    """Re-exported FFI types must preserve identity (ADR-0029)."""
    # Instance re-exports
    from vernier._core import (
        BackgroundEvaluator as FfiBackground,
    )
    from vernier._core import (
        CocoDataset as FfiDataset,
    )
    from vernier._core import (
        Summary as FfiSummary,
    )

    assert vernier.instance.BackgroundEvaluator is FfiBackground
    assert vernier.instance.CocoDataset is FfiDataset
    assert vernier.instance.Summary is FfiSummary

    # Panoptic re-exports
    from vernier._core import (
        ClassPanopticStats as FfiPanopticStats,
    )
    from vernier._core import (
        PanopticDataset as FfiPanopticDataset,
    )
    from vernier._core import (
        PanopticPredictions as FfiPanopticPredictions,
    )
    from vernier._core import (
        PanopticSummary as FfiPanopticSummary,
    )

    assert vernier.panoptic.ClassPanopticStats is FfiPanopticStats
    assert vernier.panoptic.Dataset is FfiPanopticDataset
    assert vernier.panoptic.Predictions is FfiPanopticPredictions
    assert vernier.panoptic.Summary is FfiPanopticSummary

    # Semantic re-exports
    from vernier._core import (
        ClassSemanticStats as FfiSemanticStats,
    )
    from vernier._core import (
        ConfusionMatrix as FfiConfusion,
    )
    from vernier._core import (
        SemanticSummary as FfiSemanticSummary,
    )

    assert vernier.semantic.ClassSemanticStats is FfiSemanticStats
    assert vernier.semantic.ConfusionMatrix is FfiConfusion
    assert vernier.semantic.Summary is FfiSemanticSummary


def test_streaming_evaluator_is_not_exposed_anywhere() -> None:
    """ADR-0035: the streaming pyclasses are Rust-internal. They do not
    appear on any paradigm namespace, on the FFI module, or under a
    private ``vernier._impl`` shim.
    """
    import vernier._core as _core
    import vernier.instance as inst
    import vernier.panoptic as pq
    import vernier.semantic as sem

    assert not hasattr(inst, "StreamingEvaluator")
    assert not hasattr(pq, "StreamingEvaluator")
    assert not hasattr(sem, "StreamingEvaluator")

    assert not hasattr(_core, "StreamingEvaluator")
    assert not hasattr(_core, "StreamingPanopticEvaluator")
    assert not hasattr(_core, "StreamingSemanticEvaluator")

    with pytest.raises(ModuleNotFoundError):
        import vernier._impl  # type: ignore[import-not-found]  # noqa: F401


def test_iou_kind_discriminator_identity_preserved() -> None:
    """The Bbox / Segm / Boundary / Keypoints variants under the new
    namespace must still satisfy :data:`IouKind` and round-trip through
    :meth:`Evaluator.evaluate` without a ``TypeError`` from the
    ``_reject_unknown_iou`` defensive arm."""
    for variant in (Bbox(), Segm(), Boundary(), Keypoints()):
        assert isinstance(variant, IouKind)

    gt_bytes = json.dumps(
        {
            "images": [{"id": 1, "width": 100, "height": 100}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [10.0, 10.0, 20.0, 20.0],
                    "area": 400.0,
                    "iscrowd": 0,
                },
            ],
            "categories": [{"id": 1, "name": "thing"}],
        }
    ).encode()
    dt_bytes = json.dumps(
        [
            {
                "image_id": 1,
                "category_id": 1,
                "bbox": [10.0, 10.0, 20.0, 20.0],
                "score": 0.9,
            }
        ]
    ).encode()
    summary = Evaluator(iou=Bbox()).evaluate(gt_bytes, dt_bytes)
    # AP@perfect-match collapses to 1.0; the assertion only needs the
    # call to dispatch correctly through the Bbox arm.
    assert summary.stats[0] == pytest.approx(1.0)


def test_root_re_exports_pycocotools_shim() -> None:
    """ADR-0029 commits to keeping COCOeval and patch_pycocotools at the
    root for the pycocotools migration story (ADR-0007)."""
    assert callable(vernier.patch_pycocotools)
    assert isinstance(vernier.COCOeval, type)
