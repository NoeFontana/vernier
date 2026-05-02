"""Public types for the opt-in result-tables surface.

Holds :class:`EvalResult` (cached polars views over the locked-spine
outputs) and :class:`TablesConfig` (knobs for the expensive tables).
Lives in its own module so the lazy polars import stays contained:
``import vernier`` does not pull in polars; first attribute access on
:class:`EvalResult` is what triggers it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Literal

from vernier._core import Summary

if TYPE_CHECKING:  # pragma: no cover — type-checker only
    import polars as pl

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
        return _arrow_to_dataframe(self._per_image_batch, "per_image")

    @cached_property
    def per_class(self) -> pl.DataFrame:
        """One row per category. Raises ``RuntimeError`` if
        ``per_class`` was not in the ``tables=`` request."""
        return _arrow_to_dataframe(self._per_class_batch, "per_class")

    @cached_property
    def per_detection(self) -> pl.DataFrame:
        """One row per detection. Raises ``RuntimeError`` if
        ``per_detection`` was not in the ``tables=`` request."""
        return _arrow_to_dataframe(self._per_detection_batch, "per_detection")

    @cached_property
    def per_pair(self) -> pl.DataFrame:
        """One row per (DT, GT) pair. Raises ``RuntimeError`` if
        ``per_pair`` was not in the ``tables=`` request."""
        return _arrow_to_dataframe(self._per_pair_batch, "per_pair")


def _arrow_to_dataframe(batch: object | None, name: str) -> pl.DataFrame:
    """Lazy polars import + Arrow zero-copy conversion. Raises a
    structured ``ImportError`` when polars is absent (steering the user
    to the install command), and a structured ``RuntimeError`` when the
    requested table wasn't built."""
    if batch is None:
        raise RuntimeError(
            f"{name!r} was not in the tables= request — pass it explicitly via "
            f"Evaluator.evaluate(..., tables=({name!r},)) or tables='all'"
        )
    try:
        import polars as pl
    except ImportError as e:  # pragma: no cover — exercised in lazy-import test
        raise ImportError(
            "result tables expose polars.DataFrame; install polars via "
            "`pip install 'vernier[tables]'`"
        ) from e
    df = pl.from_arrow(batch)
    if isinstance(df, pl.Series):
        df = df.to_frame()
    return df


#: Tables the active build of vernier knows how to produce. ``"all"``
#: expands to exactly this set. Users who want a stable set across
#: versions should write the tuple explicitly instead of ``"all"``.
SUPPORTED_TABLES: frozenset[TableName] = frozenset(
    {"per_image", "per_class", "per_detection", "per_pair"}
)


def normalize_tables_arg(
    tables: tuple[TableName, ...] | Literal["all"],
) -> set[TableName]:
    """Normalize the ``tables=`` keyword to a concrete set of names.

    The literal ``"all"`` expands to every table name :data:`SUPPORTED_TABLES`
    contains. Bare-string inputs (other than ``"all"``) raise — that's
    the "I forgot the comma in my one-tuple" footgun.
    """
    if tables == "all":
        return set(SUPPORTED_TABLES)
    if isinstance(tables, str):
        raise ValueError(
            f"tables= must be a tuple of names or the literal 'all'; "
            f"got bare string {tables!r}. Pass ({tables!r},) for a single table."
        )
    return set(tables)
