"""Top-level ``vernier.aggregate`` — cross-paradigm fan-in over slice tables.

The fan-in half of ADR-0046's slice-and-aggregate pipeline: read N
already-computed evaluation results, join them to a *partition manifest*
(the same artifact ``vernier eval --manifest`` consumes), group by
manifest axis value, mean the metric columns across runs in each group,
and optionally compute rPC against a baseline slice value.

This module lives at the top level (not under a paradigm submodule)
because the aggregation is paradigm-agnostic — it consumes Arrow
``RecordBatch`` slice tables of the canonical wide shape regardless of
whether they came from ``vernier.instance``, ``vernier.panoptic``, or
``vernier.semantic``. Cf. ADR-0029 (namespace logic) and ADR-0046 §G2.

Per ADR-0018, ``polars-rs`` is rejected in the FFI layer; this module
stays pure-Python specifically so that grouping / mean / rPC are done
where the workspace deny-list does not apply. The Arrow ``RecordBatch``
input/output flows via the Arrow PyCapsule Interface (ADR-0019), letting
polars / pandas / duckdb / pyarrow consumers round-trip zero-copy.
"""

from __future__ import annotations

import csv
import json
import os
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Union, cast

import pyarrow as pa

if TYPE_CHECKING:  # pragma: no cover — type-checker only
    from vernier._types import EvalResult


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

#: Type alias for entries in the ``results`` sequence. Each entry is one
#: of: an Arrow ``RecordBatch`` / ``Table`` (the canonical wide
#: slice-table shape ADR-0046 §F2 pins), an ``EvalResult`` (duck-typed:
#: anything with a ``.slices`` attribute and optionally ``.label``), or
#: a filesystem path to a v2 result JSON document.
ResultInput = Union[pa.RecordBatch, pa.Table, "EvalResult", str, "os.PathLike[str]"]

#: Type alias for the ``manifest`` parameter. A canonical
#: ``{manifest_version, key_kind, rows}`` dict, a pre-resolved
#: ``{label: {axis: value}}`` dict, a filesystem path to a ``.json`` or
#: ``.csv`` manifest, or any object exposing the Arrow PyCapsule
#: stream/array interface.
ManifestInput = Union[Mapping[str, object], str, "os.PathLike[str]", object]


class AggregateError(ValueError):
    """Raised when ``vernier.aggregate`` cannot proceed.

    Covers malformed manifests (missing required keys, wrong
    ``key_kind``), unparseable result documents (v1 schema, missing
    metric columns under an explicit ``metric=`` filter), and Arrow
    shape mismatches.
    """


# ---------------------------------------------------------------------------
# Result-input coercion
# ---------------------------------------------------------------------------


def _arrow_metadata_label(schema: pa.Schema) -> str | None:
    """Read the ``vernier.label`` metadata field if present.

    Arrow schema metadata is byte-keyed; ``schema.metadata`` returns
    ``None`` when no metadata is attached, otherwise a ``dict[bytes,
    bytes]``. We accept either UTF-8 or latin-1 fallbacks for the value.
    """
    meta: Mapping[bytes, bytes] | None = schema.metadata
    if meta is None:
        return None
    raw = meta.get(b"vernier.label")
    if raw is None:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _coerce_table_to_batch(table: pa.Table) -> pa.RecordBatch:
    """Combine an Arrow ``Table`` into a single ``RecordBatch``.

    ADR-0019 pins "one ``RecordBatch`` per table"; ``pa.Table`` is
    accepted for ergonomics, but the aggregation logic operates on
    a single contiguous batch.
    """
    combined = table.combine_chunks()
    batches = combined.to_batches()
    if not batches:
        schema = combined.schema
        empty_arrays: list[pa.Array] = [
            pa.array([], type=schema.field(i).type) for i in range(len(schema))
        ]
        return pa.RecordBatch.from_arrays(empty_arrays, schema=schema)
    if len(batches) == 1:
        return batches[0]
    # combine_chunks should yield a single batch; fall back to a
    # concat-then-reslice for the pathological multi-batch case.
    return pa.Table.from_batches(batches).combine_chunks().to_batches()[0]


def _as_float_or_none(value: object) -> float | None:
    """Narrow a JSON-loaded ``object`` to a ``float`` if it's numeric, else ``None``."""
    if isinstance(value, bool):
        # bool is a subclass of int in Python — exclude it explicitly
        # so metric tables don't end up with True coerced to 1.0.
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _as_int_or_zero(value: object) -> int:
    """Narrow a JSON-loaded ``object`` to an ``int`` for support-count columns."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def _v2_json_to_batch(doc: Mapping[str, object], *, source: str) -> pa.RecordBatch:
    """Convert a v2 result JSON document into the wide slice-table shape.

    Each v2 document carries a ``slices`` array (one entry per ``(axis,
    value)`` cell) plus an ``overall`` entry. Entries have ``axis``,
    ``value``, plus a ``stats`` map of metric -> float. We flatten the
    stats into top-level columns and emit a single ``RecordBatch``.
    """
    raw_slices = doc.get("slices")
    if raw_slices is None:
        # ADR-0046 §B2: un-partitioned v2 documents still carry an
        # "overall" entry — but for aggregate purposes we treat a
        # missing slices array as "use overall as the single row".
        overall = doc.get("overall")
        if not isinstance(overall, Mapping):
            raise AggregateError(
                f"result document {source!r} lacks both 'slices' and "
                f"'overall' entries (is this a v1 result? aggregate requires v2)",
            )
        slices: list[Mapping[str, object]] = [
            {"axis": "overall", "value": "overall", **overall},
        ]
    else:
        if not isinstance(raw_slices, list):
            raise AggregateError(
                f"result document {source!r} 'slices' must be a list "
                f"(got {type(raw_slices).__name__})",
            )
        slices = []
        for i, row in enumerate(cast("list[object]", raw_slices)):
            if not isinstance(row, Mapping):
                raise AggregateError(
                    f"result document {source!r} slices[{i}] must be an object "
                    f"(got {type(row).__name__})",
                )
            slices.append(cast("Mapping[str, object]", row))

    # Collect the union of metric column names across all rows,
    # alphabetically — value-determinism (ADR-0019).
    metric_cols: set[str] = set()
    for row in slices:
        stats_obj = row.get("stats", {})
        if isinstance(stats_obj, Mapping):
            stats = cast("Mapping[str, object]", stats_obj)
            for k, v in stats.items():
                if _as_float_or_none(v) is not None:
                    metric_cols.add(k)
    metric_order = sorted(metric_cols)

    axes: list[str] = []
    values: list[str] = []
    n_images_col: list[int | None] = []
    n_dets_col: list[int | None] = []
    metric_arrays: dict[str, list[float | None]] = {m: [] for m in metric_order}
    has_n_images = False
    has_n_dets = False
    for row in slices:
        axes.append(str(row.get("axis", "")))
        values.append(str(row.get("value", "")))
        stats_obj = row.get("stats", {})
        stats: Mapping[str, object] = (
            cast("Mapping[str, object]", stats_obj)
            if isinstance(stats_obj, Mapping)
            else {}
        )
        if "n_images" in row or "n_images" in stats:
            has_n_images = True
            n_images_col.append(_as_int_or_zero(row.get("n_images", stats.get("n_images"))))
        else:
            n_images_col.append(None)
        if "n_detections" in row or "n_detections" in stats:
            has_n_dets = True
            n_dets_col.append(
                _as_int_or_zero(row.get("n_detections", stats.get("n_detections"))),
            )
        else:
            n_dets_col.append(None)
        for m in metric_order:
            metric_arrays[m].append(_as_float_or_none(stats.get(m)))

    arrays: list[pa.Array] = [
        pa.array(axes, type=pa.string()),
        pa.array(values, type=pa.string()),
    ]
    names: list[str] = ["axis", "value"]
    if has_n_images:
        arrays.append(pa.array(n_images_col, type=pa.uint64()))
        names.append("n_images")
    if has_n_dets:
        arrays.append(pa.array(n_dets_col, type=pa.uint64()))
        names.append("n_detections")
    for m in metric_order:
        arrays.append(pa.array(metric_arrays[m], type=pa.float64()))
        names.append(m)
    return pa.RecordBatch.from_arrays(arrays, names=names)


def _coerce_result_entry(
    entry: ResultInput,
    index: int,
) -> tuple[str, pa.RecordBatch]:
    """Coerce one ``results`` entry into ``(label, slices_batch)``.

    See module docstring for accepted input shapes. The label is read
    from the appropriate carrier (Arrow metadata / ``.label`` attr /
    JSON ``label`` field / file basename) with a numeric ``result_{i}``
    fallback when no label is present.
    """
    # pyarrow's stubs export everything as Any (placeholder __getattr__);
    # isinstance(entry, pa.RecordBatch) doesn't narrow under pyright
    # because the second arg is Any. Branch on the runtime classes and
    # access .schema on the narrowed `cast` view.
    if isinstance(entry, pa.RecordBatch):
        batch: pa.RecordBatch = cast("pa.RecordBatch", entry)
        label = _arrow_metadata_label(batch.schema) or f"result_{index}"
        return label, batch
    if isinstance(entry, pa.Table):
        table: pa.Table = cast("pa.Table", entry)
        label = _arrow_metadata_label(table.schema) or f"result_{index}"
        return label, _coerce_table_to_batch(table)
    if isinstance(entry, (str, os.PathLike)):
        path = Path(os.fspath(entry))
        with path.open(encoding="utf-8") as f:
            doc_obj = json.load(f)
        if not isinstance(doc_obj, dict):
            raise AggregateError(
                f"result JSON at {path!s} must be a JSON object "
                f"(got {type(doc_obj).__name__})",
            )
        doc = cast("Mapping[str, object]", doc_obj)
        version = doc.get("version")
        if version is not None and str(version) not in {"2", "v2"}:
            raise AggregateError(
                f"result JSON at {path!s} has version={version!r}; "
                f"aggregate requires v2 (vernier eval --json-schema-version 2)",
            )
        raw_label = doc.get("label")
        label = str(raw_label) if raw_label is not None else path.stem
        return label, _v2_json_to_batch(doc, source=str(path))
    # Duck-typed EvalResult: has `.slices` attribute (Arrow batch or
    # PyCapsule-bearing object), optionally `.label`.
    entry_obj = cast("object", entry)
    slices_attr = getattr(entry_obj, "slices", None)
    if slices_attr is not None:
        label_attr = getattr(entry_obj, "label", None)
        label = str(label_attr) if label_attr is not None else f"result_{index}"
        batch = _coerce_arrow_input(slices_attr, context=f"results[{index}].slices")
        return label, batch
    # Last-ditch: an object with `.summary` and no slices — wrap the
    # scalar metrics as a single-row "overall" batch. Useful for the
    # un-partitioned EvalResult path (no .slices populated).
    summary = getattr(entry_obj, "summary", None)
    if summary is not None:
        label_attr = getattr(entry_obj, "label", None)
        label = str(label_attr) if label_attr is not None else f"result_{index}"
        return label, _summary_to_overall_batch(summary)
    raise AggregateError(
        f"results[{index}] is of type {type(entry).__name__}; expected "
        "pa.RecordBatch, pa.Table, EvalResult, or path to a v2 result JSON",
    )


def _coerce_arrow_input(obj: object, *, context: str) -> pa.RecordBatch:
    """Pull any Arrow-PyCapsule-bearing object into a ``RecordBatch``."""
    if isinstance(obj, pa.RecordBatch):
        return obj
    if isinstance(obj, pa.Table):
        return _coerce_table_to_batch(obj)
    # PyCapsule consumer path — try the array protocol first, fall back
    # to the stream protocol. pyarrow.record_batch handles both.
    if hasattr(obj, "__arrow_c_array__") or hasattr(obj, "__arrow_c_stream__"):
        try:
            batch = pa.record_batch(obj)
        except (TypeError, pa.ArrowInvalid) as e:
            # Some PyCapsule producers only export a Table-shaped stream;
            # try the table consumer too.
            try:
                table = pa.table(obj)
            except (TypeError, pa.ArrowInvalid) as e2:
                raise AggregateError(
                    f"could not pull {context} into an Arrow batch: {e}; {e2}",
                ) from e2
            return _coerce_table_to_batch(table)
        return batch
    raise AggregateError(
        f"{context} is of type {type(obj).__name__}; "
        "expected pyarrow.RecordBatch / pyarrow.Table or an Arrow PyCapsule object",
    )


def _summary_to_overall_batch(summary: object) -> pa.RecordBatch:
    """Wrap a paradigm ``Summary`` into a single-row overall batch.

    The fallback path when an ``EvalResult`` has no ``slices`` populated
    (un-partitioned evaluation). Reads ``summary.stats`` (a ``list[float]``
    per ADR-0040) and emits one metric column per stat as ``stat_{i}``.
    """
    stats_attr = getattr(summary, "stats", None)
    if stats_attr is None:
        raise AggregateError(
            f"summary of type {type(summary).__name__} has no .stats attribute",
        )
    if not isinstance(stats_attr, Sequence):
        raise AggregateError(
            f"summary.stats is of type {type(stats_attr).__name__}; expected a sequence",
        )
    stats = cast("Sequence[float]", stats_attr)
    arrays: list[pa.Array] = [
        pa.array(["overall"], type=pa.string()),
        pa.array(["overall"], type=pa.string()),
    ]
    names: list[str] = ["axis", "value"]
    for i, v in enumerate(stats):
        arrays.append(pa.array([float(v)], type=pa.float64()))
        names.append(f"stat_{i}")
    return pa.RecordBatch.from_arrays(arrays, names=names)


# ---------------------------------------------------------------------------
# Manifest coercion
# ---------------------------------------------------------------------------


def _coerce_manifest(manifest: ManifestInput) -> dict[str, dict[str, str]]:
    """Coerce a manifest to ``{label: {axis: value}}``.

    Accepted inputs (be lenient):

    - Canonical dict ``{manifest_version, key_kind, rows}`` with
      ``key_kind == "result"``. Rejects ``key_kind == "image_id"``
      (that's for ``vernier eval --manifest``, not ``aggregate``).
    - Pre-resolved dict ``{label: {axis: value}}`` (string keys both
      ways, dict values). Used as-is.
    - Filesystem path: ``.json`` (JSON-records shape) or ``.csv``
      (header row supplies axis names, first column ``key``).
    - Arrow-PyCapsule object: pulled into a pyarrow ``Table``,
      converted row-by-row. The ``key`` column is the join key.
    """
    if isinstance(manifest, Mapping):
        return _coerce_manifest_dict(cast("Mapping[str, object]", manifest))
    if isinstance(manifest, (str, os.PathLike)):
        path = Path(os.fspath(manifest))
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return _coerce_manifest_csv(path)
        if suffix in {".json", ""}:
            with path.open(encoding="utf-8") as f:
                doc_obj = json.load(f)
            if not isinstance(doc_obj, dict):
                raise AggregateError(
                    f"manifest at {path!s} must be a JSON object "
                    f"(got {type(doc_obj).__name__})",
                )
            return _coerce_manifest_dict(cast("Mapping[str, object]", doc_obj))
        raise AggregateError(
            f"manifest path {path!s} has unsupported extension {suffix!r}; "
            "expected .json or .csv",
        )
    if hasattr(manifest, "__arrow_c_array__") or hasattr(manifest, "__arrow_c_stream__"):
        return _coerce_manifest_arrow(manifest)
    raise AggregateError(
        f"manifest is of type {type(manifest).__name__}; expected "
        "dict, str/PathLike, or Arrow PyCapsule object",
    )


def _coerce_manifest_dict(doc: Mapping[str, object]) -> dict[str, dict[str, str]]:
    """Coerce a dict manifest into ``{label: {axis: value}}``."""
    # Canonical-shape detection: must have manifest_version + rows.
    if "manifest_version" in doc and "rows" in doc:
        version = str(doc.get("manifest_version", ""))
        if version != "1":
            raise AggregateError(
                f"manifest_version={version!r} is not supported "
                "(expected '1')",
            )
        key_kind = doc.get("key_kind")
        if key_kind != "result":
            raise AggregateError(
                f"aggregate requires key_kind='result' "
                f"(got {key_kind!r}); key_kind='image_id' is consumed by "
                "vernier eval --manifest, not vernier.aggregate",
            )
        rows = doc.get("rows")
        if not isinstance(rows, list):
            raise AggregateError(
                f"manifest 'rows' must be a list (got {type(rows).__name__})",
            )
        out: dict[str, dict[str, str]] = {}
        for i, row in enumerate(cast("list[object]", rows)):
            if not isinstance(row, Mapping):
                raise AggregateError(
                    f"manifest rows[{i}] must be an object "
                    f"(got {type(row).__name__})",
                )
            row_map = cast("Mapping[str, object]", row)
            key = row_map.get("key")
            if key is None:
                raise AggregateError(
                    f"manifest rows[{i}] missing required 'key' field",
                )
            label = str(key)
            axes: dict[str, str] = {
                str(axis): str(value) for axis, value in row_map.items() if axis != "key"
            }
            out[label] = axes
        return out
    # Pre-resolved-shape: every value is a Mapping. Use as-is.
    if all(isinstance(v, Mapping) for v in doc.values()):
        resolved: dict[str, dict[str, str]] = {}
        for label, axes_obj in doc.items():
            axes_map = cast("Mapping[str, object]", axes_obj)
            resolved[str(label)] = {str(a): str(v) for a, v in axes_map.items()}
        return resolved
    raise AggregateError(
        "manifest dict must be either canonical "
        "{manifest_version, key_kind, rows} or pre-resolved "
        "{label: {axis: value}}",
    )


def _coerce_manifest_csv(path: Path) -> dict[str, dict[str, str]]:
    """Read a CSV manifest. Header row supplies axis names; first column ``key``."""
    # Use utf-8-sig to transparently strip a BOM if present.
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise AggregateError(
                f"CSV manifest at {path!s} is empty (no header row)",
            )
        if "key" not in reader.fieldnames:
            raise AggregateError(
                f"CSV manifest at {path!s} missing 'key' column "
                f"(header: {reader.fieldnames!r})",
            )
        out: dict[str, dict[str, str]] = {}
        for i, row in enumerate(reader):
            key = row.get("key")
            if key is None or key == "":
                raise AggregateError(
                    f"CSV manifest at {path!s} row {i} missing 'key' value",
                )
            axes = {
                str(axis): str(value)
                for axis, value in row.items()
                if axis != "key" and value is not None
            }
            out[str(key)] = axes
    return out


def _coerce_manifest_arrow(obj: object) -> dict[str, dict[str, str]]:
    """Coerce an Arrow-PyCapsule manifest into ``{label: {axis: value}}``."""
    try:
        table = pa.table(obj)
    except (TypeError, pa.ArrowInvalid) as e:
        raise AggregateError(
            f"could not pull Arrow manifest into a pyarrow.Table: {e}",
        ) from e
    column_names: list[str] = list(table.column_names)
    if "key" not in column_names:
        raise AggregateError(
            f"Arrow manifest missing 'key' column (columns: {column_names!r})",
        )
    pylist: list[Mapping[str, object]] = list(table.to_pylist())
    out: dict[str, dict[str, str]] = {}
    for i, row in enumerate(pylist):
        key = row.get("key")
        if key is None:
            raise AggregateError(f"Arrow manifest row {i} has null 'key'")
        axes = {
            str(axis): str(value)
            for axis, value in row.items()
            if axis != "key" and value is not None
        }
        out[str(key)] = axes
    return out


# ---------------------------------------------------------------------------
# Aggregation core
# ---------------------------------------------------------------------------

#: Columns the slice-table schema reserves for slice identification /
#: support counts. These are never aggregated as metrics.
_RESERVED_SLICE_COLS: frozenset[str] = frozenset({"axis", "value", "n_images", "n_detections"})


def _select_metric_columns(
    batch: pa.RecordBatch,
    requested: str | Sequence[str] | None,
) -> list[str]:
    """Resolve the ``metric=`` filter against a batch's schema.

    ``None`` → every Float64 column not in :data:`_RESERVED_SLICE_COLS`.
    ``str`` / ``Sequence[str]`` → exactly those columns; error if any is
    absent or non-float in the batch.
    """
    schema = batch.schema
    schema_names: list[str] = list(schema.names)
    all_names = set(schema_names)
    if requested is None:
        return sorted(
            name
            for name in schema_names
            if name not in _RESERVED_SLICE_COLS
            and pa.types.is_floating(schema.field(name).type)
        )
    if isinstance(requested, str):
        requested_list: list[str] = [requested]
    else:
        requested_list = list(requested)
    out: list[str] = []
    for m in requested_list:
        if m not in all_names:
            raise AggregateError(
                f"metric={m!r} not present in slice table "
                f"(available: {sorted(all_names - _RESERVED_SLICE_COLS)!r})",
            )
        out.append(m)
    return sorted(out)


def aggregate(
    results: Sequence[ResultInput],
    manifest: ManifestInput,
    *,
    baseline: str | None = None,
    metric: str | Sequence[str] | None = None,
) -> pa.RecordBatch:
    """Aggregate N evaluation results over a partition manifest.

    The fan-in half of ADR-0046's slice-and-aggregate pipeline. Reads a
    sequence of ``EvalResult`` / Arrow ``RecordBatch`` / Arrow ``Table``
    / v2 result-JSON-path inputs, joins each to a row of the manifest
    by label, groups by manifest axis value, and emits a comparative
    per-slice ``RecordBatch`` with one Float64 column per metric. When
    ``baseline`` names an axis value, the relative Performance under
    Corruption (rPC) of each non-baseline slice — ``mean(metric) /
    mean(baseline-metric)`` — is appended as ``<metric>__rpc`` columns.

    Args:
        results: A sequence of evaluation result carriers. Accepted:

            - ``pyarrow.RecordBatch`` / ``pyarrow.Table`` slice tables
              of the canonical wide shape (``axis``, ``value``, plus
              metric columns). Label is read from the schema metadata
              key ``vernier.label`` (UTF-8) if present.
            - An :class:`vernier._types.EvalResult` (duck-typed: any
              object with a ``.slices`` attribute returning an Arrow
              ``RecordBatch`` / ``Table`` / PyCapsule object, and
              optionally ``.label``).
            - A ``str`` or ``os.PathLike`` pointing at a v2 result
              JSON document.

            When no label is available on an entry, a numeric stamp
            ``result_{i}`` is used (matching the entry's index).

        manifest: The partition manifest. Accepted:

            - The canonical JSON-records dict
              ``{manifest_version, key_kind, rows}`` with
              ``key_kind == "result"``.
            - A pre-resolved dict ``{label: {axis: value}}`` — useful
              when the caller already has the mapping in hand.
            - A ``str`` / ``os.PathLike`` pointing at a ``.json``
              manifest (canonical shape) or a ``.csv`` manifest (first
              column ``key``, header supplies axis names).
            - Any object exposing the Arrow PyCapsule stream/array
              interface (e.g. a polars / pandas DataFrame) — pulled
              into a pyarrow ``Table`` and converted row-by-row. The
              ``key`` column is the join key.

        baseline: Optional baseline axis value (e.g. ``"clean"``). When
            set, ``<metric>__rpc`` columns are appended with the ratio
            ``mean(metric_in_slice) / mean(metric_in_baseline_slice)``
            per manifest axis. Axes lacking a baseline-value row in
            the manifest are silently skipped for rPC.

        metric: Optional filter for which metric columns to aggregate.
            ``None`` keeps every Float64 column not in
            ``{axis, value, n_images, n_detections}``. A ``str`` or
            ``Sequence[str]`` selects exactly those columns; absent
            columns raise :class:`AggregateError`.

    Returns:
        A single ``pyarrow.RecordBatch`` with columns ``axis: utf8``,
        ``value: utf8``, ``n_runs: uint64``, one Float64 column per
        metric (alphabetical), and — when ``baseline`` is set — one
        Float64 column per ``<metric>__rpc``. Schema metadata carries
        ``vernier.schema_version = "1"`` and
        ``vernier.table = "aggregate"`` so downstream consumers can
        validate they got the right table shape.

    Raises:
        AggregateError: On malformed manifests, unparseable results,
            missing-metric filters, or schema-mismatched inputs.

    Notes:
        Runs missing from the manifest emit a ``UserWarning`` (via
        :mod:`warnings`) and are dropped from the aggregation. Per
        ADR-0046, slice output order is ``(axis ascending, value
        ascending)`` with deterministic column ordering for the
        value-determinism contract (ADR-0019).

    Example:
        >>> import polars as pl                                # doctest: +SKIP
        >>> r_clean = Evaluator().evaluate(gt, dt_clean)        # doctest: +SKIP
        >>> r_fog = Evaluator().evaluate(gt, dt_fog)            # doctest: +SKIP
        >>> summary = aggregate(                                # doctest: +SKIP
        ...     [r_clean, r_fog],
        ...     manifest={"manifest_version": "1", "key_kind": "result",
        ...               "rows": [{"key": "clean", "weather": "clear"},
        ...                        {"key": "fog",   "weather": "fog"}]},
        ...     baseline="clear",
        ... )
        >>> pl.from_arrow(summary).write_parquet("rpc.parquet")  # doctest: +SKIP
    """
    if not results:
        raise AggregateError("results must be a non-empty sequence")
    manifest_map = _coerce_manifest(manifest)

    # 1. Coerce every result entry; drop runs missing from the manifest
    #    with a warning.
    coerced: list[tuple[str, pa.RecordBatch, dict[str, str]]] = []
    for i, entry in enumerate(results):
        label, batch = _coerce_result_entry(entry, i)
        if label not in manifest_map:
            warnings.warn(
                f"results[{i}] label={label!r} is not in the manifest; "
                "the run is dropped from the aggregation.",
                stacklevel=2,
            )
            continue
        coerced.append((label, batch, manifest_map[label]))

    if not coerced:
        raise AggregateError(
            "no results matched the manifest; check the labels "
            "(use --label on vernier eval, or set the .label attribute / "
            "vernier.label Arrow metadata key on Arrow inputs)",
        )

    # 2. Resolve the metric column set from the first batch (sanity-
    #    checked against every other batch below). The selection is
    #    deterministic (alphabetical) per ADR-0019.
    metric_cols = _select_metric_columns(coerced[0][1], metric)
    if not metric_cols:
        raise AggregateError(
            "no metric columns to aggregate; the slice table has no "
            "Float64 columns outside {axis, value, n_images, n_detections}",
        )

    # 3. Group across runs. Key: (axis, value) — the *manifest* axis
    #    and value, not the slice-table's (axis, value). The slice
    #    table's (axis, value) is the slicing axis the run was
    #    partitioned on; the manifest tells us which corruption-axis
    #    value the *run as a whole* corresponds to. We mean each
    #    metric across runs sharing the same manifest assignment.
    groups: dict[tuple[str, str], dict[str, list[float]]] = {}
    n_runs_per_group: dict[tuple[str, str], int] = {}

    for _label, batch, axes_map in coerced:
        # Validate every batch has the metric columns we resolved.
        batch_cols: set[str] = set(batch.schema.names)
        missing = [m for m in metric_cols if m not in batch_cols]
        if missing:
            raise AggregateError(
                f"slice table missing metric columns {missing!r} "
                f"(available: {sorted(batch_cols - _RESERVED_SLICE_COLS)!r})",
            )

        # Compute per-batch per-metric scalar from the overall row.
        # ADR-0046 §B2 invariant: the "overall" row of a v2 result is
        # bit-identical to the un-partitioned eval — collapsing to it
        # gives one numeric per metric per run.
        row_idx = _pick_overall_row_idx(batch)
        per_metric: dict[str, float | None] = {}
        for m in metric_cols:
            col = batch.column(m)
            val = col[row_idx].as_py()
            per_metric[m] = float(val) if isinstance(val, (int, float)) else None

        # Fan-out into the manifest axes the run belongs to.
        for axis, value in axes_map.items():
            key = (axis, value)
            grp = groups.setdefault(key, {m: [] for m in metric_cols})
            for m in metric_cols:
                v = per_metric[m]
                if v is not None:
                    grp[m].append(v)
            n_runs_per_group[key] = n_runs_per_group.get(key, 0) + 1

    # 4. Build the output table. Deterministic order: (axis asc,
    #    value asc) with __unassigned__ floated to the end per
    #    ADR-0046 §"Determinism".
    ordered_keys = sorted(groups.keys(), key=_axis_value_sort_key)

    out_axes = [k[0] for k in ordered_keys]
    out_values = [k[1] for k in ordered_keys]
    out_n_runs = [n_runs_per_group[k] for k in ordered_keys]
    out_metrics: dict[str, list[float | None]] = {m: [] for m in metric_cols}
    for k in ordered_keys:
        for m in metric_cols:
            vals = groups[k][m]
            out_metrics[m].append(sum(vals) / len(vals) if vals else None)

    arrays: list[pa.Array] = [
        pa.array(out_axes, type=pa.string()),
        pa.array(out_values, type=pa.string()),
        pa.array(out_n_runs, type=pa.uint64()),
    ]
    names: list[str] = ["axis", "value", "n_runs"]
    for m in metric_cols:
        arrays.append(pa.array(out_metrics[m], type=pa.float64()))
        names.append(m)

    # 5. rPC: per axis, divide each non-baseline slice metric by the
    #    baseline-slice metric. Axes lacking the baseline value get
    #    null rPC cells (the column is still emitted so the schema
    #    stays consistent across axes).
    if baseline is not None:
        baseline_means: dict[str, dict[str, float | None]] = {}
        for (axis, value), per_metric_lists in groups.items():
            if value == baseline:
                baseline_means[axis] = {
                    m: (sum(vs) / len(vs) if vs else None)
                    for m, vs in per_metric_lists.items()
                }
        for m in metric_cols:
            rpc_col: list[float | None] = []
            for idx, (axis, _value) in enumerate(ordered_keys):
                bvals = baseline_means.get(axis, {})
                bmean = bvals.get(m)
                cell = out_metrics[m][idx]
                if bmean is None or bmean == 0.0 or cell is None:
                    rpc_col.append(None)
                else:
                    rpc_col.append(cell / bmean)
            arrays.append(pa.array(rpc_col, type=pa.float64()))
            names.append(f"{m}__rpc")

    metadata: dict[bytes, bytes] = {
        b"vernier.schema_version": b"1",
        b"vernier.table": b"aggregate",
    }
    schema = pa.schema(
        [pa.field(name, arr.type) for name, arr in zip(names, arrays, strict=True)],
        metadata=metadata,
    )
    return pa.RecordBatch.from_arrays(arrays, schema=schema)


def _pick_overall_row_idx(batch: pa.RecordBatch) -> int:
    """Choose the row of a slice table whose metrics describe the whole run.

    Prefer the explicit ``axis == "overall"`` row (ADR-0046 §B2);
    fall back to row 0 for tables that pre-date the overall convention.
    """
    schema_names: list[str] = list(batch.schema.names)
    if "axis" not in schema_names:
        return 0
    axis_col: list[object] = list(batch.column("axis").to_pylist())
    for i, v in enumerate(axis_col):
        if v == "overall":
            return i
    return 0


def _axis_value_sort_key(key: tuple[str, str]) -> tuple[str, int, str]:
    """Sort ``(axis, value)`` ascending, floating ``__unassigned__`` to the end.

    Per ADR-0046 §A2 / §"Determinism": slice output order is
    ``(axis ascending, value ascending)`` with ``__unassigned__``
    sorted last on each axis. The secondary key is 0 for "regular"
    values and 1 for ``__unassigned__``; the tertiary key is the
    value itself for tie-breaking within either bucket.
    """
    axis, value = key
    bucket = 1 if value == "__unassigned__" else 0
    return (axis, bucket, value)
