"""Core public types and result-table surface.

Holds :data:`ParityMode` (the three-tier parity selector) and its
associated constants, as well as :class:`EvalResult` (cached polars
views over the locked-spine outputs) and :class:`TablesConfig` (knobs
for the expensive tables).

Lives in its own module so core types are available to all submodules
without circularity, and so the lazy polars import stays contained:
``import vernier`` does not pull in polars; first attribute access on
:class:`EvalResult` is what triggers it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Final, Literal, TypeVar

from vernier._core import Summary
from vernier._tables import arrow_to_dataframe

if TYPE_CHECKING:  # pragma: no cover — type-checker only
    import polars as pl

#: Three-tier parity mode (ADR-0002).
#:
#: - ``"strict"`` — bit-exact parity with pycocotools / panopticapi /
#:   mmsegmentation. Reproduces upstream quirks and floating-point
#:   accumulation order exactly.
#: - ``"corrected"`` — the recommended mode for net-new users. Fixes
#:   documented bugs and precision issues while staying
#:   output-compatible with the COCO metric definitions.
ParityMode = Literal["strict", "corrected"]

#: Constant for strict parity mode.
PARITY_STRICT: Final[ParityMode] = "strict"
#: Constant for corrected parity mode.
PARITY_CORRECTED: Final[ParityMode] = "corrected"

#: Default boundary-IoU dilation ratio (0.02). Mirrors
#: `BoundaryIou::Default` in vernier-core (Cheng et al. 2021); the
#: bowenc0221 oracle and pycocotools use the same value as their
#: default.
DEFAULT_DILATION_RATIO: Final[float] = 0.02

#: The set of result-table identifiers ``Evaluator.evaluate(tables=...)``
#: accepts. Used in the keyword's :class:`tuple` form
#: (``tables=("per_image", "per_class")``); the literal alias
#: ``"all"`` requests every supported table for the active phase.
TableName = Literal["per_image", "per_class", "per_detection", "per_pair"]


@dataclass(frozen=True, slots=True)
class TablesConfig:
    """Configuration knobs for the expensive result tables. Inert when
    the corresponding flag is not requested via ``tables=``."""

    #: IoU floor for ``per_pair``. Pairs with IoU below this are
    #: dropped from the table. Default ``0.1``.
    per_pair_iou_floor: float = 0.1
    #: Hard cap on ``per_pair`` row count. Exceeding it raises
    #: ``ValueError``.
    per_pair_max_rows: int = 10_000_000
    #: Whether ``per_detection`` rows include bbox geometry columns.
    #: Off by default — most callers don't need them.
    per_detection_with_geometry: bool = False


@dataclass(frozen=True)
class EvalResult:
    """Result of an opt-in result-tables evaluate(...) call.

    Returned only when ``tables=`` is non-``None`` on
    :meth:`vernier.Evaluator.evaluate`. The default ``tables=None`` path
    still returns the bit-identical :class:`vernier.Summary` it always
    has.

    Tables are exposed as cached :class:`polars.DataFrame` properties;
    polars is imported lazily on first attribute access (installed via
    the ``vernier[tables]`` extra). pandas / duckdb / pyarrow consumers
    can round-trip on the returned DataFrame, or call the underlying
    Arrow producer (``self._per_image_batch.__arrow_c_array__()``)
    directly — the leading-underscore name signals implementation detail.
    """

    summary: Summary
    # Underlying Arrow producers (PyCapsule-emitting). `None` for tables
    # the caller didn't request. Leading underscore: implementation
    # detail; the supported access path is the cached_property below.
    _per_image_batch: object | None = field(default=None, repr=False)
    _per_class_batch: object | None = field(default=None, repr=False)
    _per_detection_batch: object | None = field(default=None, repr=False)
    _per_pair_batch: object | None = field(default=None, repr=False)

    @property
    def stats(self) -> list[float]:
        """Pass-through to ``self.summary.stats`` for callers who keep
        the existing ``result.stats`` shape."""
        return self.summary.stats

    @cached_property
    def per_image(self) -> pl.DataFrame:
        """One row per image rollup. Raises ``RuntimeError`` if
        ``per_image`` was not in the ``tables=`` request."""
        return arrow_to_dataframe(self._per_image_batch, "per_image")

    @cached_property
    def per_class(self) -> pl.DataFrame:
        """One row per category. Raises ``RuntimeError`` if
        ``per_class`` was not in the ``tables=`` request."""
        return arrow_to_dataframe(self._per_class_batch, "per_class")

    @cached_property
    def per_detection(self) -> pl.DataFrame:
        """One row per detection. Raises ``RuntimeError`` if
        ``per_detection`` was not in the ``tables=`` request."""
        return arrow_to_dataframe(self._per_detection_batch, "per_detection")

    @cached_property
    def per_pair(self) -> pl.DataFrame:
        """One row per (DT, GT) pair. Raises ``RuntimeError`` if
        ``per_pair`` was not in the ``tables=`` request."""
        return arrow_to_dataframe(self._per_pair_batch, "per_pair")


#: Tables the active build of vernier knows how to produce on the
#: instance paradigm. ``"all"`` expands to exactly this set on
#: :class:`vernier.instance.Evaluator.evaluate`. Per-paradigm
#: equivalents live in ``vernier.panoptic`` / ``vernier.semantic``.
SUPPORTED_TABLES: frozenset[TableName] = frozenset(
    {"per_image", "per_class", "per_detection", "per_pair"}
)


_T = TypeVar("_T", bound=str)


def normalize_tables_arg(
    tables: tuple[_T, ...] | Literal["all"],
    supported: frozenset[_T],
) -> set[_T]:
    """Normalize the ``tables=`` keyword to a concrete set of names.

    ``"all"`` expands to ``supported``. Bare-string inputs (other than
    ``"all"``) raise — the "I forgot the comma in my one-tuple"
    footgun. ``supported`` is the per-paradigm allowlist; pyright's
    ``Literal`` narrows the input independently.
    """
    if tables == "all":
        return set(supported)
    if isinstance(tables, str):
        raise ValueError(
            f"tables= must be a tuple of names or the literal 'all'; "
            f"got bare string {tables!r}. Pass ({tables!r},) for a single table."
        )
    return set(tables)
