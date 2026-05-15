"""Semantic-segmentation evaluation surface (ADR-0028).

Per ADR-0029, the semantic-segmentation evaluation paradigm lives
under ``vernier.semantic``. Sibling to :mod:`vernier.instance` (the
AP fold) and :mod:`vernier.panoptic` (panoptic-quality). The Rust
kernel ships in the ``vernier-semantic`` crate; this module is a thin
Python wrapper.

Surface:

- :class:`Dataset` / :class:`Predictions` — frozen dataclasses
  carrying the per-image label maps + dataset-level config
  (``n_classes`` / ``ignore_label``).
- :class:`Evaluator` — frozen dataclass holding ``parity_mode`` and
  optional ``label_remap``. ``Evaluator.evaluate(gt, dt)`` returns a
  :class:`Summary` (the FFI pyclass).
- :class:`Summary` / :class:`ClassSemanticStats` /
  :class:`ConfusionMatrix` — re-exported FFI pyclasses (under their
  unprefixed names per ADR-0029).
- Per-dataset presets — :meth:`Dataset.cityscapes`,
  :meth:`Dataset.ade20k`, :meth:`Dataset.pascal_voc` — bake the
  canonical ignore-label and class-count conventions; the user only
  passes the PNG paths.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, overload

import numpy as np
from numpy.typing import NDArray

from vernier._core import (
    BackgroundSemanticEvaluator as BackgroundEvaluator,
)

# Re-export the five distributed-eval exception types under the
# semantic namespace so callers catching `vernier.semantic.PartialFormatMismatch`
# match the same class object as `vernier.instance.PartialFormatMismatch`
# (ADR-0032: shared paradigm-agnostic error classes).
from vernier._core import (
    Breakdown,
    ClassSemanticStats,
    ConfusionMatrix,
    PartialDatasetMismatch,
    PartialFormatMismatch,
    PartialParamsMismatch,
    PartialPartitionOverlap,
    PartialRankCollision,
    evaluate_semantic_from_arrays,
    evaluate_semantic_from_pngs,
    semantic_per_class_to_arrow_pycapsule,
)
from vernier._core import (
    SemanticSummary as Summary,
)
from vernier._core import (
    evaluate_semantic_to_partial as _evaluate_semantic_to_partial,
)
from vernier._core import (
    merge_semantic_partials as _merge_semantic_partials,
)
from vernier._tables import arrow_to_dataframe
from vernier._types import (
    CategoryFilter,
    CategoryFilterAll,
    CategoryFilterByGrouping,
    CategoryFilterByIds,
    CategoryFilterFrequency,
    InvalidEvalParams,
    InvalidSemanticParams,
    ParityMode,
    normalize_tables_arg,
)

if TYPE_CHECKING:  # pragma: no cover — type-checker only
    import polars as pl

#: Tables ``Evaluator.evaluate(tables=...)`` accepts on the semantic
#: paradigm. Per-detection and per-pair tables are instance-only (no
#: detections in semantic); per-image breakdown is deferred to avoid
#: conflicting with ADR-0037's fused decode+fold path.
TableName = Literal["per_class"]
SUPPORTED_TABLES: frozenset[TableName] = frozenset({"per_class"})

__all__ = [
    "ADE20K_IGNORE_LABEL",
    "ADE20K_N_CLASSES",
    "CITYSCAPES_IGNORE_LABEL",
    "CITYSCAPES_N_CLASSES",
    "PASCAL_VOC_IGNORE_LABEL",
    "PASCAL_VOC_N_CLASSES",
    "BackgroundEvaluator",
    "Breakdown",
    "CategoryFilter",
    "CategoryFilterAll",
    "CategoryFilterByGrouping",
    "CategoryFilterByIds",
    "CategoryFilterFrequency",
    "ClassSemanticStats",
    "ConfusionMatrix",
    "Dataset",
    "EvalResult",
    "Evaluator",
    "InvalidEvalParams",
    "InvalidSemanticParams",
    "ParityMode",
    "PartialDatasetMismatch",
    "PartialFormatMismatch",
    "PartialParamsMismatch",
    "PartialPartitionOverlap",
    "PartialRankCollision",
    "Predictions",
    "Summary",
    "TableName",
    "decode_label_map_png",
]


@dataclass(frozen=True)
class EvalResult:
    """Opt-in result of :meth:`Evaluator.evaluate` when ``tables=`` is
    passed. Carries :class:`Summary` plus a polars DataFrame view of
    the per-class semantic breakdown in the canonical mmseg / ADE20K
    column order. Tables on :meth:`Evaluator.evaluate_from_pngs`
    (ADR-0037 fused path) are deferred.

    Also returned when ``manifest=`` is passed (ADR-0046 partitioned
    eval): the ``.summary`` field carries the bit-identical-to-
    un-partitioned ``overall`` summary, and the :attr:`slices`
    DataFrame view exposes the per-``(axis, value)`` cell metrics."""

    summary: Summary
    _per_class_batch: object | None = field(default=None, repr=False)
    #: Slices RecordBatch (ADR-0046). ``None`` unless ``manifest=`` was
    #: passed to :meth:`Evaluator.evaluate`. The cached
    #: :attr:`slices` DataFrame view reads it.
    _slices_batch: object | None = field(default=None, repr=False)
    #: Image count behind ``summary`` on the partitioned path; ``None``
    #: on the un-partitioned path.
    overall_n_images: int | None = field(default=None)
    #: Always ``0`` for semantic on the partitioned path (semantic has
    #: no detection notion; the column is shape-parity with panoptic /
    #: instance). ``None`` on the un-partitioned path.
    overall_n_detections: int | None = field(default=None)

    @cached_property
    def per_class(self) -> pl.DataFrame:
        """One row per class. Columns: ``category_id``, ``iou``,
        ``accuracy``, ``precision``, ``n_gt_pixels``, ``n_dt_pixels``,
        ``tp_pixels``, ``fp_pixels``, ``fn_pixels``."""
        return arrow_to_dataframe(self._per_class_batch, "per_class")

    @cached_property
    def slices(self) -> pl.DataFrame:
        """One row per ``(axis, value)`` partition cell (ADR-0046).
        Columns: ``axis``, ``value``, ``n_images``, ``n_detections``,
        ``miou``, ``fwiou``, ``pixel_accuracy``, ``mean_accuracy``.
        ``n_detections`` is always 0 — semantic has no detection
        notion; the column is shape-parity with panoptic / instance.
        Available only when the originating ``evaluate(...)`` call
        carried a ``manifest=`` keyword; raises :class:`RuntimeError`
        otherwise."""
        return arrow_to_dataframe(self._slices_batch, "slices")


#: Cityscapes ignore-label convention (255). Mirrors
#: ``vernier_semantic::parity::CITYSCAPES_IGNORE_LABEL``; pinned here
#: too so the Python preset doesn't need an FFI round-trip to look it
#: up.
CITYSCAPES_IGNORE_LABEL: Final[int] = 255
#: Cityscapes 19-class evaluation count.
CITYSCAPES_N_CLASSES: Final[int] = 19
#: ADE20K (SceneParse150) ignore label. Class 0 is "other/unlabeled".
ADE20K_IGNORE_LABEL: Final[int] = 0
#: ADE20K (SceneParse150) class count.
ADE20K_N_CLASSES: Final[int] = 150
#: Pascal VOC ignore-label convention (255, void at object boundary).
PASCAL_VOC_IGNORE_LABEL: Final[int] = 255
#: Pascal VOC class count (20 objects + background indexed at 0).
PASCAL_VOC_N_CLASSES: Final[int] = 21


def decode_label_map_png(path: str | Path) -> NDArray[np.uint32]:
    """Load a single-channel PNG label map as a ``(H, W)`` ``uint32``
    array.

    Lazy-imports Pillow; raises a structured :class:`ImportError` if
    Pillow is not installed. Multi-channel (RGB-encoded) panoptic
    PNGs belong in :func:`vernier.panoptic.decode_label_map_png` —
    they are rejected here with :class:`ValueError`.
    """
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError(
            "Pillow is required for `vernier.semantic.decode_label_map_png` "
            "(and the per-dataset presets); install via `pip install Pillow` "
            "(or include it in your dev environment)."
        ) from e
    img = Image.open(path)
    arr = np.asarray(img)
    if arr.ndim != 2:
        raise ValueError(
            f"semantic label-map PNG {path!s} must be single-channel (2-D); "
            f"got shape {arr.shape!r}. RGB-encoded panoptic segment ids belong "
            f"in `vernier.panoptic.Dataset`, not the semantic surface."
        )
    return arr.astype(np.uint32, copy=False)


def _decode_png_paths(
    png_paths: Mapping[int, str | Path],
) -> dict[int, NDArray[np.uint32]]:
    """Decode a mapping of image_id → png path into the ``uint32``
    label-map dict the FFI consumes."""
    return {image_id: decode_label_map_png(p) for image_id, p in png_paths.items()}


def _merge_binary_masks(
    masks: NDArray[np.uint8],
    *,
    merge: Literal["argmax", "first", "highest_class_id"],
    unlabeled_class: int,
) -> NDArray[np.uint32]:
    """Merge a `(K, H, W)` stack of per-class binary masks into a
    single `(H, W)` ``uint32`` class-id label map. Per quirk **AN3**,
    the merge rule is explicit and load-bearing.

    - ``argmax``: standard argmax over the channel axis. Requires score
      channels (or, on binary inputs, ties go to the lowest class id —
      argmax tiebreak).
    - ``first``: the first channel with value 1 wins (order-dependent).
    - ``highest_class_id``: the highest class id with value 1 wins
      (convention-dependent).

    Pixels where every mask is zero are written as ``unlabeled_class``
    (per quirk **AN4**); typical use is to set this to the dataset's
    ``ignore_label``.
    """
    if masks.ndim != 3:
        raise ValueError(f"binary masks must be 3-D `(n_classes, H, W)`; got shape {masks.shape!r}")
    n_classes, h, w = masks.shape
    out = np.full((h, w), unlabeled_class, dtype=np.uint32)
    any_set = masks.any(axis=0)
    if merge in ("argmax", "first"):
        # On binary input these collapse to the same operation: numpy's
        # argmax tiebreak picks the lowest channel index, which is "the
        # first channel with value 1". On score input (argmax only),
        # the highest score wins.
        chosen = masks.argmax(axis=0).astype(np.uint32, copy=False)
    elif merge == "highest_class_id":
        # Reverse argmax: the highest channel index with value 1 wins.
        chosen = (n_classes - 1 - masks[::-1].argmax(axis=0)).astype(np.uint32, copy=False)
    else:
        raise ValueError(
            f"unknown merge rule {merge!r}; expected 'argmax', 'first', or 'highest_class_id'"
        )
    out = np.where(any_set, chosen, out)
    return out


@dataclass(frozen=True, slots=True)
class Dataset:
    """Semantic-segmentation ground truth.

    Carries per-image label maps, the evaluation class count, and the
    ignore label. Constructors:

    - :meth:`from_arrays` — pre-decoded ``uint8`` / ``uint16`` /
      ``uint32`` ``ndarray`` per image. Zero-copy on the FFI hot
      path; the kernel walks at native dtype (ADR-0037).
    - :meth:`from_files` — PNG paths; decoded via Pillow.
    - :meth:`cityscapes` / :meth:`ade20k` / :meth:`pascal_voc` —
      per-dataset presets that bake the canonical ``n_classes`` and
      ``ignore_label`` conventions.

    Predictions are constructed via the sibling :class:`Predictions`
    type — same shape, no class-count metadata (the dataset is the
    authoritative source).
    """

    label_maps: Mapping[int, np.ndarray]
    n_classes: int
    ignore_label: int | None = None

    def __post_init__(self) -> None:
        if self.n_classes < 1:
            raise ValueError(f"n_classes must be >= 1; got {self.n_classes!r}")
        if self.ignore_label is not None and self.ignore_label < 0:
            raise ValueError(
                f"ignore_label must be a non-negative integer; got {self.ignore_label!r}"
            )

    @classmethod
    def from_arrays(
        cls,
        label_maps: Mapping[int, np.ndarray],
        n_classes: int,
        ignore_label: int | None = None,
    ) -> Dataset:
        """Construct from pre-decoded label-map ``ndarray`` per image.

        Each ``label_maps[image_id]`` must be a 2-D integer array. The
        FFI boundary accepts ``uint8`` / ``uint16`` / ``uint32`` natively
        (ADR-0037); the kernel walks at native dtype, so passing
        ``uint8`` from a ``torch.Tensor`` avoids the 4x upcast that
        earlier wheel versions paid here.
        """
        return cls(label_maps=dict(label_maps), n_classes=n_classes, ignore_label=ignore_label)

    @classmethod
    def from_files(
        cls,
        png_paths: Mapping[int, str | Path],
        n_classes: int,
        ignore_label: int | None = None,
    ) -> Dataset:
        """Decode PNG paths via Pillow into a :class:`Dataset`.

        Each PNG must be a single-channel label map. Multi-channel
        (RGB-encoded) panoptic PNGs belong in
        :class:`vernier.panoptic.Dataset` — they are rejected here
        with a :class:`ValueError`.
        """
        return cls(
            label_maps=_decode_png_paths(png_paths),
            n_classes=n_classes,
            ignore_label=ignore_label,
        )

    @classmethod
    def cityscapes(cls, png_paths: Mapping[int, str | Path]) -> Dataset:
        """Cityscapes 19-class evaluation preset.

        ``n_classes=19``, ``ignore_label=255``. PNGs are expected to
        be ``*_labelTrainIds.png`` files (already in trainId space —
        the 30+ raw label space pre-mapped to 19 classes by the
        ``cityscapesscripts/preparation/createTrainIdLabelImgs.py``
        helper).

        Users with predictions in the raw 30+ label space should
        construct :class:`Evaluator` with an explicit ``label_remap``
        dict that maps raw ids to trainIds.
        """
        return cls.from_files(
            png_paths,
            n_classes=CITYSCAPES_N_CLASSES,
            ignore_label=CITYSCAPES_IGNORE_LABEL,
        )

    @classmethod
    def ade20k(cls, png_paths: Mapping[int, str | Path]) -> Dataset:
        """ADE20K (SceneParse150) preset.

        ``n_classes=150``, ``ignore_label=0``. Class 0 is the
        "other/unlabeled" sentinel; predictions are expected in
        ``[1, 150]``.
        """
        return cls.from_files(
            png_paths,
            n_classes=ADE20K_N_CLASSES,
            ignore_label=ADE20K_IGNORE_LABEL,
        )

    @classmethod
    def pascal_voc(cls, png_paths: Mapping[int, str | Path]) -> Dataset:
        """Pascal VOC preset.

        ``n_classes=21``, ``ignore_label=255``. 20 object classes plus
        a ``background`` class indexed at 0; the 255 sentinel marks
        ``void`` pixels at object boundaries.
        """
        return cls.from_files(
            png_paths,
            n_classes=PASCAL_VOC_N_CLASSES,
            ignore_label=PASCAL_VOC_IGNORE_LABEL,
        )


@dataclass(frozen=True, slots=True)
class Predictions:
    """Semantic-segmentation predictions.

    Carries per-image label maps only — class-count and ignore-label
    metadata live on the sibling :class:`Dataset` (the authoritative
    source). Constructors mirror :class:`Dataset`:

    - :meth:`from_arrays` — pre-decoded ``uint8`` / ``uint16`` /
      ``uint32`` ``ndarray`` per image (ADR-0037).
    - :meth:`from_files` — PNG paths; decoded via Pillow.
    - :meth:`from_binary_masks` — per-class binary masks merged into
      a single class-id label map per image (quirk **AN2**).
    """

    label_maps: Mapping[int, np.ndarray]

    @classmethod
    def from_arrays(cls, label_maps: Mapping[int, np.ndarray]) -> Predictions:
        """Construct from pre-decoded label-map ``ndarray`` per image.

        Accepts ``uint8`` / ``uint16`` / ``uint32`` natively (ADR-0037);
        the FFI walks at the input dtype.
        """
        return cls(label_maps=dict(label_maps))

    @classmethod
    def from_files(cls, png_paths: Mapping[int, str | Path]) -> Predictions:
        """Decode PNG paths via Pillow into a :class:`Predictions`.

        See :meth:`Dataset.from_files` for the PNG shape contract.
        """
        return cls(label_maps=_decode_png_paths(png_paths))

    @classmethod
    def from_binary_masks(
        cls,
        masks: Mapping[int, NDArray[np.uint8]],
        *,
        merge: Literal["argmax", "first", "highest_class_id"] = "argmax",
        unlabeled_class: int = 255,
    ) -> Predictions:
        """Merge per-image `(n_classes, H, W)` binary mask stacks into
        single class-id label maps (quirks **AN2**, **AN3**, **AN4**).

        ``merge`` selects the precedence rule for overlapping pixels:

        - ``"argmax"`` (default) — argmax over channels; on score
          input the highest score wins, on binary input ties go to
          the lowest class id.
        - ``"first"`` — the first channel with value 1 wins
          (order-dependent).
        - ``"highest_class_id"`` — the highest class id with value 1
          wins (convention-dependent).

        ``unlabeled_class`` is written into pixels where every mask
        is zero (default ``255`` matches the Cityscapes / Pascal VOC
        ignore convention).
        """
        merged: dict[int, NDArray[np.uint32]] = {
            image_id: _merge_binary_masks(arr, merge=merge, unlabeled_class=unlabeled_class)
            for image_id, arr in masks.items()
        }
        return cls(label_maps=merged)


def _validate_semantic_class_filter(cf: CategoryFilter, class_grouping: Breakdown | None) -> None:
    """Validate a ``class_filter`` for the semantic paradigm (ADR-0041).

    ``Frequency`` is rejected (LVIS-only — no semantic frequency tag).
    ``ByIds`` requires non-empty unique ids; bounds against ``n_classes``
    are checked at evaluate time once the Dataset is in scope.
    ``ByGrouping`` requires that ``class_grouping`` is configured and
    that the named label exists in it.
    """
    if isinstance(cf, CategoryFilterFrequency):
        raise InvalidSemanticParams(
            field="class_filter",
            value=cf,
            remediation=(
                "Frequency is LVIS-only - semantic has no per-class frequency "
                "tag analogous to r/c/f. Use ByIds or ByGrouping instead "
                "(ADR-0026 lines 178-182, ADR-0041)"
            ),
        )
    if isinstance(cf, CategoryFilterByIds) and len(cf.ids) == 0:
        raise InvalidSemanticParams(
            field="class_filter",
            value=cf,
            remediation="ByIds must contain at least one class id (ADR-0041)",
        )
    if isinstance(cf, CategoryFilterByGrouping):
        if class_grouping is None:
            raise InvalidSemanticParams(
                field="class_filter",
                value=cf,
                remediation=(
                    "ByGrouping requires class_grouping to also be set; "
                    "the label is resolved against the grouping's labels (ADR-0041)"
                ),
            )
        labels = {label for label, _ in class_grouping.class_groups}
        if cf.label not in labels:
            raise InvalidSemanticParams(
                field="class_filter",
                value=cf,
                remediation=(
                    f"ByGrouping label {cf.label!r} is not a label of class_grouping "
                    f"(known labels: {sorted(labels)!r}); ADR-0041"
                ),
            )


def _resolve_class_filter(
    cf: CategoryFilter | None,
    class_grouping: Breakdown | None,
) -> list[int] | None:
    """Resolve an ADR-0041 ``class_filter`` to the kernel's class-id form.

    ``None`` and :class:`CategoryFilterAll` produce ``None`` (no
    filtering at the kernel). :class:`CategoryFilterByIds` returns a
    sorted list of its ids. :class:`CategoryFilterByGrouping` resolves
    the label against the active ``class_grouping`` partition and
    returns the union of that group's class ids.

    The construction-time validator (:func:`_validate_semantic_class_filter`)
    has already rejected :class:`CategoryFilterFrequency` and
    cross-checked the grouping label, so this function trusts inputs.
    """
    if cf is None or isinstance(cf, CategoryFilterAll):
        return None
    if isinstance(cf, CategoryFilterByIds):
        return sorted(cf.ids)
    if isinstance(cf, CategoryFilterByGrouping):
        # __post_init__ guarantees class_grouping is set and the label
        # exists when the filter is ByGrouping.
        assert class_grouping is not None
        for label, ids in class_grouping.class_groups:
            if label == cf.label:
                return list(ids)
        # Should not reach here — validator runs at construction.
        raise InvalidSemanticParams(
            field="class_filter",
            value=cf,
            remediation=f"label {cf.label!r} disappeared from class_grouping",
        )
    # Frequency was rejected at construction; this branch is defensive.
    raise InvalidSemanticParams(
        field="class_filter",
        value=cf,
        remediation="unsupported CategoryFilter variant",
    )


def _resolve_class_grouping(
    bd: Breakdown | None,
) -> list[tuple[str, list[int]]] | None:
    """Resolve a ``class_grouping`` :class:`Breakdown` to the kernel's
    list-of-pairs form. ``None`` passes through; the kernel skips the
    per-group rollup when this is ``None``."""
    if bd is None:
        return None
    return [(label, list(ids)) for label, ids in bd.class_groups]


@dataclass(frozen=True, slots=True)
class Evaluator:
    """Semantic-segmentation evaluator (ADR-0028, ADR-0041).

    Sibling to :class:`vernier.instance.Evaluator` (AP fold) and
    :class:`vernier.panoptic.Evaluator` (panoptic-quality). Computes
    mIoU, FWIoU, pixel accuracy, mean accuracy, per-class IoU, and
    per-class accuracy from per-image confusion matrices accumulated
    into a global confusion matrix.

    The instance is immutable per ADR-0006: construct once, call
    :meth:`evaluate` per dataset/predictions pair.

    Defaults match net-new-user expectations: ``parity_mode="corrected"``
    (per ADR-0002 recommendation); migrating users wanting bit-exact
    mmsegmentation behavior should set ``parity_mode="strict"``.

    ``class_filter`` and ``class_grouping`` (ADR-0041) parameterize the
    evaluation scope and rollup. ``class_filter`` restricts the headline
    scalars (``miou`` etc.) to a subset of classes; ``class_grouping``
    adds an optional per-group rollup alongside ``per_class``. Both
    default to ``None`` (kernel-canonical: every class contributes to
    headline scalars; no per-group rollup).

    **PR scope cut:** the kernel-side plumbing for honoring custom
    ``class_filter`` / ``class_grouping`` is a follow-up to ADR-0041
    (`per_group` lives on `SemanticSummary` once the Rust struct gains
    it). Until that lands, ``evaluate()`` raises
    :class:`NotImplementedError` when either is set; the surface
    (fields, validation, ``with_options`` threading) is in place.
    """

    parity_mode: ParityMode = "corrected"
    label_remap: Mapping[int, int] | None = field(default=None)
    class_filter: CategoryFilter | None = None
    class_grouping: Breakdown | None = None

    def __post_init__(self) -> None:
        if self.class_grouping is not None and self.class_grouping.kind != "class_groups":
            raise InvalidSemanticParams(
                field="class_grouping",
                value=self.class_grouping,
                remediation=(
                    "must be a class-groups Breakdown "
                    "(Breakdown.from_class_groups(...)); range Breakdowns "
                    "belong on instance.Evaluator.area_ranges (ADR-0041)"
                ),
            )
        if self.class_filter is not None:
            _validate_semantic_class_filter(self.class_filter, self.class_grouping)

    def _has_custom_class_params(self) -> bool:
        """``True`` when any ADR-0041 field is set."""
        return self.class_filter is not None or self.class_grouping is not None

    @overload
    def evaluate(
        self,
        gt: Dataset,
        dt: Predictions,
        *,
        tables: None = None,
        manifest: None = None,
        cross_axes: None = None,
    ) -> Summary: ...

    @overload
    def evaluate(
        self,
        gt: Dataset,
        dt: Predictions,
        *,
        tables: Literal["all"] | tuple[TableName, ...],
        manifest: None = None,
        cross_axes: None = None,
    ) -> EvalResult: ...

    @overload
    def evaluate(
        self,
        gt: Dataset,
        dt: Predictions,
        *,
        tables: None = None,
        manifest: object,
        cross_axes: Sequence[Sequence[str]] | None = None,
    ) -> EvalResult: ...

    def evaluate(
        self,
        gt: Dataset,
        dt: Predictions,
        *,
        tables: Literal["all"] | tuple[TableName, ...] | None = None,
        manifest: object | None = None,
        cross_axes: Sequence[Sequence[str]] | None = None,
    ) -> Summary | EvalResult:
        """Run the semantic-segmentation evaluation.

        ``gt`` and ``dt`` may be constructed via any combination of
        the per-class constructors (:meth:`Dataset.from_arrays`,
        :meth:`Predictions.from_files`, etc.); the FFI consumes
        whatever ``uint32`` ndarrays the constructors produced.

        Returns a :class:`Summary` carrying the four headline scalars
        (mIoU / FWIoU / pixel_accuracy / mean_accuracy), the per-class
        breakdown, and the accumulated :class:`ConfusionMatrix`.

        ``tables=`` is the opt-in keyword for result tables (ADR-0038).
        Defaults to ``None``, returning :class:`Summary` (existing
        behavior). Pass ``"all"`` or a tuple of :data:`TableName`
        values to opt into the wider :class:`EvalResult` return type.

        ``manifest=`` opts into ADR-0046 partitioned eval. Accepts a
        dict (the canonical JSON-records shape), a file path
        (``.json``), or any object exposing the Arrow PyCapsule
        Interface (a polars / pandas / pyarrow DataFrame of per-image
        metadata). Returns an :class:`EvalResult` whose ``.summary``
        is bit-identical to the un-partitioned call and whose
        ``.slices`` property is a polars DataFrame with one row per
        ``(axis, value)`` cell. ``cross_axes=`` opts joint cells in
        (per ADR-0046 §E2; marginals are the default).

        Per ADR-0046 §"Performance", semantic partitioned eval runs as
        one :func:`evaluate_semantic_from_arrays` call per slice (the
        C1 path) — the semantic substrate accumulates per-image
        confusion matrices into a global matrix without an image-id
        filter at summarize time. The ``overall`` summary is the
        un-partitioned eval over the full mappings, preserving the
        bit-identical-overall parity contract by construction.
        """
        if manifest is not None:
            from vernier.semantic._partition import (
                evaluate_partitioned as _evaluate_partitioned,
            )

            if tables is not None:
                raise ValueError(
                    "tables= and manifest= cannot be combined on the semantic "
                    "paradigm yet; the partitioned evaluate returns EvalResult "
                    "carrying the slices RecordBatch but per-class partitioned "
                    "tables are a follow-up."
                )
            if self._has_custom_class_params():
                raise InvalidSemanticParams(
                    field="manifest",
                    value=manifest,
                    remediation=(
                        "semantic partitioned eval does not yet propagate the "
                        "ADR-0041 custom fields (class_filter / class_grouping). "
                        "Re-run with a default-config evaluator, or run "
                        "manifest= and custom params separately."
                    ),
                )
            resolved_filter = _resolve_class_filter(self.class_filter, self.class_grouping)
            resolved_groups = _resolve_class_grouping(self.class_grouping)
            overall, slices_batch, n_images, n_dets = _evaluate_partitioned(
                gt.label_maps,
                dt.label_maps,
                n_classes=gt.n_classes,
                parity_mode=self.parity_mode,
                ignore_label=gt.ignore_label,
                label_remap=dict(self.label_remap) if self.label_remap is not None else None,
                class_filter=resolved_filter,
                class_grouping=resolved_groups,
                manifest=manifest,
                cross_axes=cross_axes,
            )
            return EvalResult(
                summary=overall,
                _slices_batch=slices_batch,
                overall_n_images=n_images,
                overall_n_detections=n_dets,
            )
        if cross_axes is not None:
            raise ValueError(
                "cross_axes= is only meaningful alongside manifest= (it opts "
                "joint cells into the manifest partition spec; with no "
                "manifest there is nothing to cross)"
            )
        # ADR-0041 custom axes resolve Python-side. ByGrouping → ByIds
        # via the active class_grouping (the kernel's filter primitive
        # is class-id-keyed; group labels live in user-space).
        resolved_filter = _resolve_class_filter(self.class_filter, self.class_grouping)
        resolved_groups = _resolve_class_grouping(self.class_grouping)
        summary = evaluate_semantic_from_arrays(
            dict(gt.label_maps),
            dict(dt.label_maps),
            n_classes=gt.n_classes,
            parity_mode=self.parity_mode,
            ignore_label=gt.ignore_label,
            label_remap=dict(self.label_remap) if self.label_remap is not None else None,
            class_filter=resolved_filter,
            class_grouping=resolved_groups,
        )
        if tables is None:
            return summary
        requested = normalize_tables_arg(tables, SUPPORTED_TABLES)
        per_class_batch = (
            semantic_per_class_to_arrow_pycapsule(summary) if "per_class" in requested else None
        )
        return EvalResult(summary=summary, _per_class_batch=per_class_batch)

    def evaluate_from_pngs(
        self,
        gt_paths: Mapping[int, str | Path],
        dt_paths: Mapping[int, str | Path],
        *,
        n_classes: int,
        ignore_label: int | None = None,
    ) -> Summary:
        """Run the evaluation directly against 8-bit grayscale PNG label
        maps on disk (ADR-0037).

        Skips the Pillow → ndarray → ``astype(np.uint32)`` pipeline:
        the libpng decode and the per-image confusion-matrix fold run
        in Rust under ``py.detach``, so the GIL is released for the
        whole batch and only one decoded label-map per side is in
        flight at a time. Memory ceiling is bounded by image size,
        not dataset size.

        Format contract: 8-bit grayscale PNGs only (the natural width
        for class-id label maps up to 256 classes; covers every dataset
        in the per-paradigm presets — Cityscapes, ADE20K, Pascal VOC,
        plus panoptic-derived semantic). Wider class-id ranges should
        use :meth:`evaluate` with ``np.uint16`` / ``np.uint32`` ndarrays.

        ``label_remap`` does not propagate to the fused path; callers
        needing a remap should rewrite the DT PNGs upstream or fall
        back to :meth:`evaluate`.
        """
        if self.label_remap is not None:
            raise NotImplementedError(
                "Evaluator.evaluate_from_pngs does not yet propagate label_remap; "
                "rewrite the DT PNGs upstream or fall back to Evaluator.evaluate."
            )
        return evaluate_semantic_from_pngs(
            dict(gt_paths),
            dict(dt_paths),
            n_classes=n_classes,
            parity_mode=self.parity_mode,
            ignore_label=ignore_label,
        )

    def evaluate_to_partial(
        self,
        gt: Dataset,
        dt: Predictions,
        *,
        rank_id: int,
    ) -> bytes:
        """Run the evaluation as a per-rank streaming submit and return
        the serialized partial bytes (ADR-0032, ADR-0035).

        ``rank_id`` identifies this evaluator's rank in a multi-process
        eval. The partial bytes can be gathered across ranks and merged
        with :meth:`from_partials` to produce a global :class:`Summary`
        bit-equal to a batch :meth:`evaluate` over the union (semantic
        confusion-matrix sums are u64-additive, so strict-mode
        bit-equality is unconditional per ADR-0032).

        ``label_remap`` does not propagate to the streaming path —
        callers needing remap apply it on the DT arrays themselves
        before passing the dataset in.
        """
        if self.label_remap is not None:
            raise NotImplementedError(
                "Evaluator.evaluate_to_partial does not yet propagate label_remap; "
                "apply the remap on the DT arrays before constructing Predictions."
            )
        return _evaluate_semantic_to_partial(
            dict(gt.label_maps),
            dict(dt.label_maps),
            n_classes=gt.n_classes,
            parity_mode=self.parity_mode,
            rank_id=rank_id,
            ignore_label=gt.ignore_label,
        )

    @classmethod
    def from_partials(
        cls,
        n_classes: int,
        partials: Sequence[bytes],
        /,
        *,
        parity_mode: ParityMode = "corrected",
        ignore_label: int | None = None,
    ) -> Summary:
        """Merge ``partials`` (one per rank) into a global :class:`Summary`
        (ADR-0032, ADR-0035).

        ``n_classes``, ``parity_mode``, and ``ignore_label`` must match
        what each rank used to produce its partial. Mismatches raise
        the structured ``Partial*`` errors re-exported on this module.
        """
        return _merge_semantic_partials(
            n_classes, list(partials), parity_mode, ignore_label=ignore_label
        )

    def background(
        self,
        n_classes: int,
        ignore_label: int | None = None,
        *,
        rank_id: int | None = None,
        queue_capacity: int = 8,
        worker_affinity: int | None = None,
        worker_nice: int = 5,
        shutdown_timeout_seconds: float = 5.0,
    ) -> BackgroundEvaluator:
        """Build a :class:`BackgroundEvaluator` (ADR-0014 + ADR-0032)
        that shares this evaluator's ``parity_mode``.

        The returned wrapper owns a single dedicated worker thread
        running a :class:`StreamingEvaluator` of the same shape; calls
        to :meth:`BackgroundEvaluator.submit` enqueue per-image
        ``(gt, dt)`` pairs and return immediately. Use this when the
        confusion-matrix fold measurably stalls the training loop
        (most users won't notice — semantic is fast — but per-image
        update on dense Cityscapes pixels is the case where this
        starts to matter).

        ``rank_id``, when set, identifies this evaluator's rank in a
        multi-process eval (ADR-0032). The five queueing /
        scheduling knobs mirror :class:`vernier.instance.Evaluator
        .background`.

        ``label_remap`` does not propagate to the background path —
        callers needing remap on a streaming path apply it on the DT
        arrays themselves before each ``submit`` call.
        """
        if self.label_remap is not None:
            raise NotImplementedError(
                "Evaluator.background does not yet propagate label_remap; "
                "apply the remap on the DT arrays before each submit call."
            )
        return BackgroundEvaluator(
            n_classes,
            self.parity_mode,
            ignore_label=ignore_label,
            rank_id=rank_id,
            queue_capacity=queue_capacity,
            worker_affinity=worker_affinity,
            worker_nice=worker_nice,
            shutdown_timeout_seconds=shutdown_timeout_seconds,
        )
