"""Public surface for the confusion-matrix capability (ADR-0023).

Lives in its own module so the lazy ``polars`` import stays contained:
``import vernier`` does not pull in polars; calling
:func:`confusion_matrix` is what triggers it. Mirrors the lazy-import
pattern in :mod:`vernier._types`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, NoReturn

from vernier._core import (
    CocoDataset,
    confusion_matrix_bbox,
    confusion_matrix_boundary,
    confusion_matrix_segm,
)
from vernier._types import ParityMode

if TYPE_CHECKING:  # pragma: no cover - type-checker only
    import polars as pl

    from vernier.instance import IouKind


def confusion_matrix(
    gt: bytes | CocoDataset,
    dt: bytes,
    *,
    iou: IouKind | None = None,
    t_f: float = 0.5,
    max_dets_per_image: int = 100,
    use_cats: bool = True,
    parity_mode: ParityMode = "corrected",
) -> pl.DataFrame:
    """Confusion matrix counts in long format (ADR-0023).

    Counts ``(true_class, predicted_class)`` pairs across the dataset
    using the same cross-class IoU side pass that powers
    :func:`vernier.error_decomposition`. One per-image walk produces:

    - **Diagonal cells** (``gt_class == dt_class``) — true positives.
    - **Off-diagonal cells** (``gt_class != dt_class``) — classification
      confusion (a detection of class B was the best overlap of a GT
      of class A at IoU >= ``t_f``).
    - **``__none__`` row** (``gt_class == "__none__"``) — false
      positives: the detection had no overlapping GT at the threshold.
    - **``__none__`` column** (``dt_class == "__none__"``) — missed GTs:
      a non-ignore GT was not covered by any detection at the threshold.

    Output is a :class:`polars.DataFrame` with three columns:

    - ``gt_class: str`` — the true class id as a decimal string, or the
      sentinel ``"__none__"`` for false-positive rows.
    - ``dt_class: str`` — the predicted class id as a decimal string, or
      ``"__none__"`` for missed-GT columns.
    - ``count: i64`` — number of ``(gt_class, dt_class)`` pairs in the
      dataset.

    The class columns are typed ``str`` rather than mixed ``int|str``
    because polars does not have a clean dtype for the union (the
    ``__none__`` sentinel is fundamentally not an integer). Callers
    wanting numeric ids can ``df.with_columns(pl.col("gt_class").cast(pl.Int64,
    strict=False))`` — ``__none__`` rows surface as ``null``, the
    natural representation of "no class".

    Output is **long-format** (one row per non-zero cell) rather than
    wide-format (a square matrix) because long-format composes better
    with polars' filter / group / agg idioms. Pivot to wide via
    ``df.pivot(values="count", index="gt_class", on="dt_class")`` if
    needed for visualization.

    Args:
        gt: Ground-truth COCO JSON payload as bytes (the same shape
            ``pycocotools.COCO(...)`` consumes). The :class:`CocoDataset`
            handle from ADR-0020 is not yet wired through this path —
            passing one raises :class:`NotImplementedError`.
        dt: Detection COCO JSON payload as bytes.
        iou: Kernel selector. Pass :class:`Bbox()` (default),
            :class:`Segm()`, or :class:`Boundary(dilation_ratio=...)`.
            :class:`Keypoints` is rejected per ADR-0024 (OKS is
            single-class in COCO; cross-class confusion is undefined).
        t_f: Foreground IoU threshold for declaring a ``(gt, dt)`` pair
            matched. Default ``0.5`` matches the COCO convention.
        max_dets_per_image: Per-image detection cap (matches the
            matching path's cap). Default ``100``.
        use_cats: Reserved; must be ``True``. A category-collapsed
            evaluation has no meaningful confusion matrix (every cell
            collapses to a single virtual class).
        parity_mode: ``"strict"`` or ``"corrected"`` per ADR-0002.
            Defaults to ``"corrected"``.

    Returns:
        A :class:`polars.DataFrame` with columns ``gt_class``,
        ``dt_class``, ``count``.

    Raises:
        NotImplementedError: ``iou=Keypoints(...)`` (ADR-0024) or
            ``gt`` is a :class:`CocoDataset` handle (ADR-0020 forward-compat
            marker not yet wired through).
        ValueError: ``t_f`` outside ``[0, 1]``, ``max_dets_per_image
            < 1``, or ``use_cats=False``.
        ImportError: ``polars`` not installed (install via
            ``pip install 'vernier[tables]'``).

    Example:
        >>> import vernier
        >>> df = vernier.confusion_matrix(gt_bytes, dt_bytes, iou=vernier.Bbox())
        >>> df.filter(pl.col("gt_class") != pl.col("dt_class"))  # only mistakes
        >>> df.pivot(values="count", index="gt_class", on="dt_class")  # wide
    """
    # Lazy: a CocoDataset handle on the gt= path needs FFI threading we
    # haven't done yet. Match the same forward-compat marker the
    # tables= path emits.
    if isinstance(gt, CocoDataset):
        raise NotImplementedError(
            "confusion_matrix requires GT JSON bytes; CocoDataset handles are not "
            "yet wired through the cross-class side pass"
        )

    # Lazy import: `vernier/__init__.py` imports this module, so
    # reaching back up at module-import time would spin a cycle.
    # Calling :func:`confusion_matrix` is what triggers the lookup; by
    # then `vernier` is fully initialized.
    from vernier.instance import Bbox, Boundary, Keypoints, Segm

    if iou is None:
        iou = Bbox()

    match iou:
        case Bbox():
            result = confusion_matrix_bbox(gt, dt, parity_mode, t_f, max_dets_per_image, use_cats)
        case Segm():
            result = confusion_matrix_segm(gt, dt, parity_mode, t_f, max_dets_per_image, use_cats)
        case Boundary(dilation_ratio=r):
            result = confusion_matrix_boundary(
                gt, dt, parity_mode, t_f, max_dets_per_image, use_cats, r
            )
        case Keypoints():
            raise NotImplementedError(
                "confusion_matrix on keypoints (OKS) is deferred per ADR-0024 — "
                "OKS is single-class in COCO, so cross-class confusion is "
                "undefined for this kernel"
            )
        case _:
            _reject_unknown_iou(iou)

    return _to_dataframe(result)


def _to_dataframe(result: Mapping[str, object]) -> pl.DataFrame:
    """Lazy polars import + dict→DataFrame conversion. Raises a
    structured :class:`ImportError` when polars is absent (steering the
    user to the install command).

    The FFI returns parallel arrays in a dict; we project the three
    long-format columns onto a polars frame. The auxiliary
    ``iou_threshold`` / ``kernel`` keys are dropped — they're
    introspection metadata, not part of the table shape.
    """
    try:
        import polars as pl
    except ImportError as e:  # pragma: no cover - exercised in lazy-import test
        raise ImportError(
            "confusion_matrix returns a polars.DataFrame; install polars via "
            "`pip install 'vernier[tables]'`"
        ) from e
    # `result` is the FFI dict: gt_class/dt_class/count/iou_threshold/kernel.
    return pl.DataFrame(
        {
            "gt_class": result["gt_class"],
            "dt_class": result["dt_class"],
            "count": result["count"],
        }
    )


def _reject_unknown_iou(iou: object) -> NoReturn:
    raise TypeError(
        f"unsupported iou kernel {iou!r}; expected Bbox(), Segm(), "
        f"Boundary(...) — see vernier.IouKind"
    )
