"""Core public types and result-table surface.

Holds :data:`ParityMode` (the two-tier parity selector) and its
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
from typing import TYPE_CHECKING, Final, Literal, TypeAlias, TypeVar

from vernier._core import Summary
from vernier._tables import arrow_to_dataframe

if TYPE_CHECKING:  # pragma: no cover — type-checker only
    import polars as pl

#: Parity mode (ADR-0002, amended 2026-05-10 — `aligned` collapsed
#: into `strict`).
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


class InvalidEvalParams(ValueError):  # noqa: N818 — ADR-0039 ratifies this exact name
    """Base for paradigm-specific ``Evaluator`` construction errors.

    Raised at ``Evaluator.__post_init__`` time by every paradigm in
    response to invalid parameter values (out of range, wrong shape,
    duplicate, conflicting, etc.). Per ADR-0039, validation runs at
    construction so misconfiguration surfaces fast — ``evaluate()``
    cannot fail on misconfigured params, only on bad data.

    Each subclass carries the offending field name, the offending
    value, and a one-line remediation pointer (typically the relevant
    ADR or doc page).
    """

    def __init__(self, *, field: str, value: object, remediation: str) -> None:
        self.field = field
        self.value = value
        self.remediation = remediation
        super().__init__(f"invalid {field}={value!r}: {remediation}")


# Subclasses need an explicit ``__init__`` even though they only call
# super: pyright won't propagate ``InvalidEvalParams.__init__`` through
# ``ValueError`` in the MRO, so direct construction
# (``InvalidInstanceParams(field=..., value=..., remediation=...)``)
# would otherwise type-check as ``ValueError(*args)``.


class InvalidInstanceParams(InvalidEvalParams):
    """Invalid ``vernier.instance.Evaluator`` parameter (ADR-0040)."""

    def __init__(self, *, field: str, value: object, remediation: str) -> None:
        super().__init__(field=field, value=value, remediation=remediation)


class InvalidSemanticParams(InvalidEvalParams):
    """Invalid ``vernier.semantic.Evaluator`` parameter (ADR-0041)."""

    def __init__(self, *, field: str, value: object, remediation: str) -> None:
        super().__init__(field=field, value=value, remediation=remediation)


class InvalidPanopticParams(InvalidEvalParams):
    """Invalid ``vernier.panoptic.Evaluator`` parameter (ADR-0042)."""

    def __init__(self, *, field: str, value: object, remediation: str) -> None:
        super().__init__(field=field, value=value, remediation=remediation)


class IncompatibleSummaryPlan(ValueError):  # noqa: N818 — ADR-0040 ratifies this exact name
    """Raised when a custom evaluation grid is incompatible with the
    canonical fixed-shape summary plan.

    The COCO 12-stat / keypoints 10-stat / LVIS 13-stat summary plans
    address slots in the ``(T, R, K, A, M)`` accumulator by hardcoded
    indices — ``AP_S`` is "the second area-bucket entry of the all-IoU
    slice at maxDet=100", not "the small-area slot". A custom
    ``iou_thresholds`` ladder, ``recall_thresholds`` ladder, or
    ``area_ranges`` breakdown breaks this index assumption.

    Per ADR-0040, custom-grid users get the result-tables surface
    (``Evaluator.evaluate_tables(...)``, ADR-0019), which carries
    explicit labels per row and composes cleanly with arbitrary grid
    layouts. The ``remediation`` field on this exception names the
    method to call.
    """

    def __init__(self, *, field: str, value: object, plan: str, remediation: str) -> None:
        self.field = field
        self.value = value
        self.plan = plan
        self.remediation = remediation
        super().__init__(
            f"custom {field}={value!r} is incompatible with the {plan} summary plan: {remediation}"
        )


# --- CategoryFilter discriminated union (ADR-0026 + ADR-0041 extension) -------
#
# Mirrors the Rust ``CategoryFilter`` enum at
# ``crates/vernier-core/src/summarize.rs``. Exposed as a Python-side
# discriminated union of frozen dataclasses (the ``IouKind`` precedent
# in ``vernier.instance``), letting paradigm validators do
# ``isinstance`` discrimination cleanly.
#
# - ``CategoryFilterAll`` — every category contributes (the COCO
#   default; the Rust ``CategoryFilter::All`` variant).
# - ``CategoryFilterFrequency`` — LVIS-only; rejected on semantic /
#   panoptic per ADR-0026's frequency-vs-breakdown distinction (ADR-0041).
# - ``CategoryFilterByIds`` — explicit subset of class / category ids.
# - ``CategoryFilterByGrouping`` — name-of-a-group from the active
#   ``class_grouping`` breakdown (ADR-0041 extension consumed by
#   ADR-0042). Resolution maps the label against
#   ``class_grouping.class_groups`` at evaluator boundary; the kernel
#   never sees this variant directly.


@dataclass(frozen=True, slots=True)
class CategoryFilterAll:
    """Match every category. The COCO default."""


@dataclass(frozen=True, slots=True)
class CategoryFilterFrequency:
    """Match by LVIS frequency tag (``"r"``, ``"c"``, ``"f"``).

    Valid only on instance evaluation against an LVIS-shaped dataset
    (ADR-0026). Semantic and panoptic Evaluators reject this variant
    at construction time per ADR-0041 / ADR-0042 — frequency tags are
    a sum type that doesn't generalize to non-numeric axes; class
    groupings carry the user's per-group rollup intent on those
    paradigms.
    """

    tag: Literal["r", "c", "f"]


@dataclass(frozen=True, slots=True)
class CategoryFilterByIds:
    """Match an explicit set of class / category ids."""

    ids: frozenset[int]


@dataclass(frozen=True, slots=True)
class CategoryFilterByGrouping:
    """Match every class id in the named group of the active
    ``class_grouping`` breakdown.

    Only meaningful when the Evaluator's ``class_grouping`` is also
    set; the validator at ``__post_init__`` rejects ``ByGrouping``
    when no grouping is configured or when ``label`` is not a
    grouping label.
    """

    label: str


CategoryFilter: TypeAlias = (
    CategoryFilterAll | CategoryFilterFrequency | CategoryFilterByIds | CategoryFilterByGrouping
)


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

    #: Canonical pycocotools-shaped summary. ``None`` when the
    #: evaluator was configured with an ADR-0040 custom grid
    #: (``iou_thresholds`` / ``recall_thresholds`` / ``area_ranges``):
    #: the canonical 12-stat / 10-stat / 13-stat plans are keyed on
    #: hardcoded slot indices that don't generalize to user-defined
    #: grids. Custom-grid callers consume the per-axis result tables
    #: directly. The default-grid path always populates this field.
    summary: Summary | None
    # Underlying Arrow producers (PyCapsule-emitting). `None` for tables
    # the caller didn't request. Leading underscore: implementation
    # detail; the supported access path is the cached_property below.
    _per_image_batch: object | None = field(default=None, repr=False)
    _per_class_batch: object | None = field(default=None, repr=False)
    _per_detection_batch: object | None = field(default=None, repr=False)
    _per_pair_batch: object | None = field(default=None, repr=False)

    @property
    def stats(self) -> list[float]:
        """Pass-through to ``self.summary.stats``. Raises
        :class:`AttributeError` on ADR-0040 custom-grid results — the
        slot-indexed summary doesn't apply; read per-axis tables instead."""
        if self.summary is None:
            raise AttributeError(
                "EvalResult.stats is unavailable for custom-grid evaluations; "
                "read result.per_class / result.per_image instead."
            )
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
