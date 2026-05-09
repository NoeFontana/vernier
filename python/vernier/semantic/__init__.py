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
from pathlib import Path
from typing import Final, Literal

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
    ClassSemanticStats,
    ConfusionMatrix,
    PartialDatasetMismatch,
    PartialFormatMismatch,
    PartialParamsMismatch,
    PartialPartitionOverlap,
    PartialRankCollision,
    evaluate_semantic_from_arrays,
    evaluate_semantic_from_pngs,
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
from vernier._types import ParityMode

__all__ = [
    "ADE20K_IGNORE_LABEL",
    "ADE20K_N_CLASSES",
    "CITYSCAPES_IGNORE_LABEL",
    "CITYSCAPES_N_CLASSES",
    "PASCAL_VOC_IGNORE_LABEL",
    "PASCAL_VOC_N_CLASSES",
    "BackgroundEvaluator",
    "ClassSemanticStats",
    "ConfusionMatrix",
    "Dataset",
    "Evaluator",
    "ParityMode",
    "PartialDatasetMismatch",
    "PartialFormatMismatch",
    "PartialParamsMismatch",
    "PartialPartitionOverlap",
    "PartialRankCollision",
    "Predictions",
    "Summary",
    "decode_label_map_png",
]

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


@dataclass(frozen=True, slots=True)
class Evaluator:
    """Semantic-segmentation evaluator (ADR-0028).

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
    """

    parity_mode: ParityMode = "corrected"
    label_remap: Mapping[int, int] | None = field(default=None)

    def evaluate(self, gt: Dataset, dt: Predictions) -> Summary:
        """Run the semantic-segmentation evaluation.

        ``gt`` and ``dt`` may be constructed via any combination of
        the per-class constructors (:meth:`Dataset.from_arrays`,
        :meth:`Predictions.from_files`, etc.); the FFI consumes
        whatever ``uint32`` ndarrays the constructors produced.

        Returns a :class:`Summary` carrying the four headline scalars
        (mIoU / FWIoU / pixel_accuracy / mean_accuracy), the per-class
        breakdown, and the accumulated :class:`ConfusionMatrix`.
        """
        return evaluate_semantic_from_arrays(
            dict(gt.label_maps),
            dict(dt.label_maps),
            n_classes=gt.n_classes,
            parity_mode=self.parity_mode,
            ignore_label=gt.ignore_label,
            label_remap=dict(self.label_remap) if self.label_remap is not None else None,
        )

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
