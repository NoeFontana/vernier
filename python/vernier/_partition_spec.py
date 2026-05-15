"""Python-side partition spec builder (ADR-0046 §"Panoptic / semantic
partitioning").

The instance paradigm reaches partitioned eval through the locked
spine: matching runs once, accumulate+summarize fans out over image
subsets (the C3 axiom). The panoptic and semantic paradigms compute
their summaries from per-image accumulations rather than from an
AP-shaped accumulator tensor; cleanly subsetting that pipeline would
require refactoring both summarize stages. As a pragmatic
phase-1 fallback (per ADR-0046 §"Performance"), the panoptic /
semantic Python wrappers loop the unchanged single-eval over each
slice's image subset and feed the resulting metrics back through the
shared :func:`slices_batch_panoptic` / :func:`slices_batch_semantic`
Arrow builders. This is closer to the C1 path than C3; LVIS-scale
panoptic / semantic users who need C3 performance for partitioned
eval should file an issue.

This module owns the manifest → slice list resolution shared by both
paradigm wrappers. Mirrors ``vernier_core::partition::PartitionSpec``
verbatim:

- canonical slice order is ``(axis ascending, value ascending,
  __unassigned__ last)``;
- every axis gets an ``__unassigned__`` slice for dataset images not
  covered by any of that axis's manifest values;
- ``cross_axes`` opts in joint cells via the ``::`` cross-separator
  (per ADR-0046 §E2);
- a hard cap of 256 total slices matches ``SLICES_CAP`` in the Rust
  spec builder.

The bit-identical-overall contract (ADR-0046's load-bearing parity
claim, since pycocotools has no slicing notion) is preserved by
construction: the ``overall`` summary is computed by running the
unchanged un-partitioned ``evaluate(...)`` over the full GT/DT pair,
i.e. the same code path the user would call without ``manifest=``.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from vernier._core import manifest_to_json_bytes

#: Sentinel value the partition spec assigns to dataset images absent
#: from the manifest, mirroring ``vernier_core::partition::UNASSIGNED``.
UNASSIGNED: Final[str] = "__unassigned__"

#: Cross-product cross-axis separator joining axis names and value
#: tuples, mirroring ``vernier_core::partition::CROSS_SEPARATOR``.
CROSS_SEPARATOR: Final[str] = "::"

#: Hard cap on the total number of slices a single spec may carry,
#: mirroring ``vernier_core::partition::SLICES_CAP``.
SLICES_CAP: Final[int] = 256

#: Canonical manifest schema version this builder understands. Mirrors
#: ``vernier_core::manifest::MANIFEST_VERSION``.
MANIFEST_VERSION: Final[str] = "1"


@dataclass(frozen=True, slots=True)
class PartitionSlice:
    """One slice in a Python-side panoptic / semantic partition spec.

    Mirrors ``vernier_core::partition::Slice`` minus the
    ``image_indices`` field (the Python loop operates on image ids
    directly — there is no flat I-axis to resolve against).
    """

    axis: str
    value: str
    image_ids: frozenset[int]


@dataclass(frozen=True, slots=True)
class PartitionSpec:
    """Resolved partition spec consumed by the panoptic / semantic
    per-slice loop. Slices are pre-sorted in the canonical order."""

    slices: tuple[PartitionSlice, ...]


def _parse_canonical_manifest(
    manifest: object,
    *,
    known_image_ids: frozenset[int],
) -> tuple[dict[str, dict[str, frozenset[int]]], list[str]]:
    """Resolve any of the four ``manifest=`` shapes (dict, JSON path,
    Arrow PyCapsule, canonical JSON dict) into a
    ``per_axis[axis][value] -> {image_ids}`` mapping plus a list of
    unknown-key warnings.

    Defers shape coercion to the FFI's ``manifest_to_json_bytes``
    helper so all four shapes land on the same canonical JSON wire
    form. Manifest rows whose ``key`` is absent from
    ``known_image_ids`` are skipped with a warning string (the caller
    surfaces them via :func:`warnings.warn`).
    """
    bytes_blob = manifest_to_json_bytes(manifest)
    # `json.loads` returns `Any` by design; assert the top-level shape
    # so the rest of the function can carry typed locals.
    doc_any: Any = json.loads(bytes_blob)
    if not isinstance(doc_any, dict):
        raise ValueError("manifest must be a JSON object")
    # pyright narrows isinstance(dict) only to `dict[Unknown, Unknown]`;
    # the explicit annotation re-establishes the value type.
    doc: dict[str, Any] = doc_any  # pyright: ignore[reportUnknownVariableType]

    version = doc.get("manifest_version")
    if version != MANIFEST_VERSION:
        raise ValueError(f"unsupported manifest_version {version!r}; expected {MANIFEST_VERSION!r}")
    key_kind = doc.get("key_kind")
    if key_kind != "image_id":
        raise ValueError(
            f'partitioned eval consumes key_kind="image_id" manifests; '
            f'got {key_kind!r}. A key_kind="result" manifest must be routed '
            f"through vernier.aggregate."
        )
    rows_obj: Any = doc.get("rows", [])
    if not isinstance(rows_obj, list):
        raise ValueError("manifest 'rows' must be a list")
    rows: list[Any] = rows_obj  # pyright: ignore[reportUnknownVariableType]

    axis_names: list[str] | None = None
    per_axis: dict[str, dict[str, set[int]]] = {}
    warnings_seen: list[str] = []

    for row_idx, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"row {row_idx} is not a JSON object: {row!r}")
        row_typed: Mapping[str, Any] = row  # pyright: ignore[reportUnknownVariableType]
        if "key" not in row_typed:
            raise ValueError(f"row {row_idx} is missing the 'key' column")
        # `key` is i64 for image_id manifests.
        try:
            image_id = int(row_typed["key"])
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"row {row_idx}: key {row_typed['key']!r} is not an integer image id"
            ) from e
        row_axes: list[str] = sorted(str(k) for k in row_typed if k != "key")
        if not row_axes:
            raise ValueError(
                f"row {row_idx} has no axis columns; the manifest must declare at least one axis"
            )
        for ax in row_axes:
            if CROSS_SEPARATOR in ax:
                raise ValueError(
                    f"manifest axis {ax!r} contains the reserved separator "
                    f"{CROSS_SEPARATOR!r}; rename the column"
                )
        if axis_names is None:
            axis_names = row_axes
        elif axis_names != row_axes:
            raise ValueError(
                f"row {row_idx} axes {row_axes!r} differ from first row "
                f"{axis_names!r}; vernier rejects ragged manifests"
            )
        if image_id not in known_image_ids:
            warnings_seen.append(str(image_id))
            continue
        for ax in row_axes:
            value: Any = row_typed[ax]
            if not isinstance(value, str):
                raise ValueError(f"row {row_idx} axis {ax!r}: value {value!r} is not a string")
            per_axis.setdefault(ax, {}).setdefault(value, set()).add(image_id)

    # Freeze the inner sets before returning.
    frozen: dict[str, dict[str, frozenset[int]]] = {
        ax: {v: frozenset(ids) for v, ids in values.items()} for ax, values in per_axis.items()
    }
    return frozen, warnings_seen


def _validate_cross_axes(
    per_axis: Mapping[str, Mapping[str, frozenset[int]]],
    cross_axes: Sequence[Sequence[str]],
) -> None:
    """Mirror ``vernier_core::partition::validate_cross_axes``: each
    tuple must have >=2 axes, every name must be a known axis, no
    repeated axes within a tuple."""
    for tup in cross_axes:
        if len(tup) < 2:
            raise ValueError(
                f"--cross tuple {list(tup)!r} must name at least two axes; "
                f"a single-axis cross is a marginal and is rejected"
            )
        seen: set[str] = set()
        for ax in tup:
            if ax not in per_axis:
                raise ValueError(
                    f"--cross tuple {list(tup)!r} names unknown axis {ax!r}; "
                    f"known axes: {sorted(per_axis.keys())!r}"
                )
            if ax in seen:
                raise ValueError(f"--cross tuple {list(tup)!r} repeats axis {ax!r}")
            seen.add(ax)


def _expand_cross_slices(
    axes: Sequence[str],
    per_axis: Mapping[str, Mapping[str, frozenset[int]]],
    all_image_ids: frozenset[int],
) -> list[PartitionSlice]:
    """Cartesian-product the per-axis value sets and intersect the
    image-id sets to produce one joint cell per combination. Adds a
    final ``__unassigned__`` joint bucket for dataset images covered by
    no joint cell of this tuple."""
    joined_axis = CROSS_SEPARATOR.join(axes)
    # Pre-sort each axis's values so the cartesian product order is
    # deterministic across runs.
    per_axis_sorted: list[list[tuple[str, frozenset[int]]]] = [
        sorted(per_axis[ax].items(), key=lambda kv: kv[0]) for ax in axes
    ]
    # Iterative cartesian expansion. The slice cap is enforced after the
    # full set is built so the error message can include the total.
    combos: list[list[tuple[str, frozenset[int]]]] = [[]]
    for axis_values in per_axis_sorted:
        next_combos: list[list[tuple[str, frozenset[int]]]] = []
        for combo in combos:
            for entry in axis_values:
                next_combos.append([*combo, entry])
        combos = next_combos

    out: list[PartitionSlice] = []
    covered: set[int] = set()
    for combo in combos:
        if not combo:
            continue
        # Intersection across this tuple's per-axis value sets.
        joint_ids: frozenset[int] = combo[0][1]
        for _, ids in combo[1:]:
            joint_ids = joint_ids & ids
        covered.update(joint_ids)
        joined_value = CROSS_SEPARATOR.join(v for v, _ in combo)
        out.append(PartitionSlice(axis=joined_axis, value=joined_value, image_ids=joint_ids))
    missing = all_image_ids - frozenset(covered)
    out.append(PartitionSlice(axis=joined_axis, value=UNASSIGNED, image_ids=missing))
    # Match `vernier_core::partition::expand_cross_axes`: plain lex
    # sort by (axis, value). `__unassigned__` lands alphabetically, not
    # forced last — mirroring the Rust spec builder so the panoptic /
    # semantic and instance lanes emit slices in matching order.
    out.sort(key=lambda s: (s.axis, s.value))
    return out


def build_spec(
    manifest: object,
    *,
    all_image_ids: Iterable[int],
    cross_axes: Sequence[Sequence[str]] | None = None,
) -> PartitionSpec:
    """Resolve a manifest input (any of the four shapes) plus the live
    dataset image-id set into a :class:`PartitionSpec`.

    Emits :class:`UserWarning` for manifest rows whose key is absent
    from ``all_image_ids`` (the "no silent data loss" discipline from
    ADR-0046). The warning text names the offending key so the user
    can locate the stale row in their manifest workbook.
    """
    known = frozenset(all_image_ids)
    per_axis, unknown_keys = _parse_canonical_manifest(manifest, known_image_ids=known)
    for k in unknown_keys:
        warnings.warn(
            f"manifest row key={k!r} is absent from the dataset; row skipped",
            UserWarning,
            stacklevel=3,
        )

    if cross_axes is None:
        cross_axes_norm: list[list[str]] = []
    else:
        cross_axes_norm = [list(t) for t in cross_axes]
    _validate_cross_axes(per_axis, cross_axes_norm)

    marginal: list[PartitionSlice] = []
    for axis in sorted(per_axis.keys()):
        values = per_axis[axis]
        covered: set[int] = set()
        for value in sorted(values.keys()):
            ids = values[value]
            covered.update(ids)
            marginal.append(PartitionSlice(axis=axis, value=value, image_ids=ids))
        missing = known - frozenset(covered)
        marginal.append(PartitionSlice(axis=axis, value=UNASSIGNED, image_ids=missing))

    joint: list[PartitionSlice] = []
    for tup in cross_axes_norm:
        joint.extend(_expand_cross_slices(tup, per_axis, known))

    total = len(marginal) + len(joint)
    if total > SLICES_CAP:
        raise ValueError(
            f"partition would produce {total} slices but the cap is {SLICES_CAP}; "
            f"reduce cross_axes or narrow the manifest"
        )

    slices = (*marginal, *joint)
    return PartitionSpec(slices=slices)
