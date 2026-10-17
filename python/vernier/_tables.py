"""Paradigm-agnostic Arrow PyCapsule → polars DataFrame bridge.

Lives in its own module so the per-paradigm result-type modules
(:mod:`vernier.instance`, :mod:`vernier.panoptic`,
:mod:`vernier.semantic`) can import the helper without dragging in
each other's types. The module itself is private (leading-underscore
name); ``arrow_to_dataframe`` is the supported intra-package entry
point. Importing this module does **not** import polars; polars is
imported lazily on the first call to :func:`arrow_to_dataframe`
(which is what each paradigm's ``EvalResult.per_*`` cached property
triggers).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — type-checker only
    import polars as pl


def arrow_to_dataframe(batch: object | None, name: str) -> pl.DataFrame:
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
    # Construct polars.DataFrame directly from the Arrow PyCapsule to avoid
    # the Polars 2.0 deprecation warning on pl.from_arrow. Polars' FrameInitTypes
    # does not type arbitrary __arrow_c_array__ PyCapsules under strict mode.
    return pl.DataFrame(batch)  # pyright: ignore[reportArgumentType]
