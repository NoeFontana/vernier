"""LVIS parity harness — diff vernier's federated evaluation against
the vendored ``lvis-api`` oracle (ADR-0026, PR-3).

Mirrors ``tests/python/parity_boundary/harness.py`` and
``tests/python/parity/harness.py`` in shape: a ``snapshot`` function
that returns a comparable dataclass for each implementation, plus an
``assert_snapshots_equal`` helper for bit-equality on the strict-mode
fields and tolerated drift on aligned-mode ones.

This is the **PR-3 skeleton**: bbox-only, AP-only. We diff the
per-cell ``eval_imgs`` payload (the ``Option<PerImageEval>`` shape) and
the raw ``(T, R, K, A)`` precision tensor — the 13-entry summary plan
arrives in PR-4 and the harness gains a ``Summary`` slot then.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from vernier._core import evaluate_bbox_grid_with_dataset

# The oracle goes via the vendored tree (conftest patches `sys.path`);
# vernier ships an FFI grid helper used by the candidate path.
from vernier.instance import Dataset

ImplName = Literal["vernier", "lvis_api"]


@dataclass(frozen=True, slots=True)
class LvisSnapshot:
    """Output of one parity run.

    Diffs up to three layers, every one strict-mode bit-equal:

    - ``eval_imgs``: the per-(category, area, image) cell payload
      (`Option<PerImageEval>` shape). Optional — set
      ``include_eval_imgs=False`` on :func:`snapshot` to skip
      materializing the cell list for whole-dataset smokes that
      would otherwise OOM (LVIS v1 val has 1203 * 4 * 19809 = 95M
      cells; even at one Python ``None`` reference per skipped cell,
      that's ~760 MB per impl plus the populated-cell payloads).
    - ``precision``: the raw `(T, R, K, A)` tensor at
      `max_dets=300` — the input to every entry in
      ``stats``. AF5: no M-axis.
    - ``stats``: an ordered ``dict[str, float]`` of the 13 LVIS
      summary entries (AF1, AF4). The keys mirror lvis-api's
      ``LVISEval.results`` keys exactly so a single
      ``assert_array_equal`` pins the whole shape.

    Per-cell divergence on a federated dataset shows up in the
    `(T, R, K, A)` tensor as a category-row drift; the summary
    collapse can hide it under the unweighted-mean. Where memory
    allows (every fixture-level test), keep ``include_eval_imgs=True``;
    drop it only for the val smoke.
    """

    eval_imgs: list[dict[str, Any] | None]
    """Per-(category, area, image) cell. ``None`` matches lvis-api's
    ``eval_imgs[idx] = None``; otherwise a dict with ``dt_scores``,
    ``dt_matches``, ``dt_ignore``, ``gt_ignore`` arrays normalized to
    the same numeric layout the oracle emits. Empty list when the
    snapshot was produced with ``include_eval_imgs=False``."""

    precision: NDArray[np.float64]
    """Shape ``(T, R, K, A)`` precision tensor — no M-axis (AF5).
    Pinned here so a regression on the K-axis filter is visible
    *before* the summary collapse."""

    stats: dict[str, float]
    """13-entry LVIS plan output, ordered as
    ``[AP, AP50, AP75, APs, APm, APl, APr, APc, APf, AR@300,
    ARs@300, ARm@300, ARl@300]``. Keys are the lvis-api shape so the
    diff is direct."""


def _vernier_snapshot(
    gt_bytes: bytes,
    dt_bytes: bytes,
    *,
    max_dets: int,
    include_eval_imgs: bool,
) -> LvisSnapshot:
    # The JSON-bytes grid path strips federated metadata at GT load
    # (it goes through `from_json_bytes`); for ADR-0026 the harness
    # has to thread a parsed-once `Dataset.from_lvis_json` through
    # so the orchestrator's AA3/AA4 branches actually fire.
    gt_dataset = Dataset.from_lvis_json(gt_bytes)
    grid = evaluate_bbox_grid_with_dataset(
        gt_dataset,
        dt_bytes,
        "strict",
        max_dets,
        True,
    )
    accum = grid.accumulate([max_dets])
    norm: list[dict[str, Any] | None] = []
    if include_eval_imgs:
        cells = grid.eval_imgs()
        # vernier exposes per-cell payloads in pycocotools-shaped
        # keys (`dtScores`, `dtMatches`, `dtIgnore`, `gtIgnore`); the
        # oracle uses snake_case. We normalize to snake_case for the
        # snapshot dict so downstream diffs read evenly.
        for cell in cells:
            if cell is None:
                norm.append(None)
                continue
            # `dtMatches` is `(T, D)` int (GT id on a hit, 0 on a
            # miss); collapse to bool so vernier's id-space (J1)
            # doesn't have to align with the oracle's.
            # `dtIgnore`/`gtIgnore` ship as uint8 — cast to bool so
            # equality checks are exact.
            dt_matches_id = np.asarray(cell["dtMatches"], dtype=np.int64)
            norm.append(
                {
                    "dt_scores": np.asarray(cell["dtScores"], dtype=np.float64),
                    "dt_matches": dt_matches_id > 0,
                    "dt_ignore": np.asarray(cell["dtIgnore"], dtype=bool),
                    "gt_ignore": np.asarray(cell["gtIgnore"], dtype=bool),
                }
            )
        # Drop the FFI cell list before the summary path — the val
        # smoke holds two snapshots in memory simultaneously, and
        # the cells dominate the footprint.
        del cells
    # accum.precision is `(T, R, K, A, M)`; M is always 1 here, drop it.
    # Force a copy (`np.array(..., copy=True)`) so the snapshot does
    # not keep the FFI `accum` alive via NumPy's view-of-Rust-buffer.
    # On the LVIS v1 val grid, a kept `accum` would in turn keep
    # `grid` alive — and `grid.eval_imgs` is the 95M-cell allocation
    # the val smoke is trying to release.
    precision = np.array(accum.precision, dtype=np.float64, copy=True)
    if precision.ndim == 5:
        assert precision.shape[-1] == 1, (
            f"vernier precision M-axis must be 1 at max_dets={max_dets}; got {precision.shape}"
        )
        precision = precision[..., 0].copy()
    summary = accum.summarize_lvis(gt_dataset, [max_dets])
    stats = _vernier_summary_stats(summary, max_dets)
    return LvisSnapshot(eval_imgs=norm, precision=precision, stats=stats)


# Mirrors lvis-api `LVISEval.results` key order. The plan layout is
# pinned in `crates/vernier-core/src/summarize.rs::lvis_default`; if
# that order ever drifts, this list (and the parity diff) catches it.
_LVIS_STATS_KEYS_TEMPLATE: tuple[str, ...] = (
    "AP",
    "AP50",
    "AP75",
    "APs",
    "APm",
    "APl",
    "APr",
    "APc",
    "APf",
    "AR@{max_dets}",
    "ARs@{max_dets}",
    "ARm@{max_dets}",
    "ARl@{max_dets}",
)


def _stats_keys(max_dets: int) -> tuple[str, ...]:
    return tuple(k.format(max_dets=max_dets) for k in _LVIS_STATS_KEYS_TEMPLATE)


def _vernier_summary_stats(summary: Any, max_dets: int) -> dict[str, float]:
    keys = _stats_keys(max_dets)
    values = [float(line) for line in summary.stats]
    if len(values) != len(keys):
        raise AssertionError(
            f"vernier lvis_default returned {len(values)} stats; expected {len(keys)} (AF1)"
        )
    return dict(zip(keys, values, strict=True))


def _lvis_snapshot(
    gt_bytes: bytes,
    dt_bytes: bytes,
    *,
    max_dets: int,
    include_eval_imgs: bool,
) -> LvisSnapshot:
    # The oracle accepts file paths (AH1); write to a NamedTemporaryFile
    # so the harness stays pure-bytes-in/snapshot-out.
    from lvis import LVIS, LVISEval, LVISResults  # type: ignore[import-not-found]

    with NamedTemporaryFile("wb", suffix="_gt.json", delete=False) as fgt:
        fgt.write(gt_bytes)
        gt_path = fgt.name
    with NamedTemporaryFile("wb", suffix="_dt.json", delete=False) as fdt:
        fdt.write(dt_bytes)
        dt_path = fdt.name
    try:
        lvis_gt = LVIS(gt_path)
        lvis_dt = LVISResults(lvis_gt, dt_path, max_dets=max_dets)
        ev = LVISEval(lvis_gt, lvis_dt, iou_type="bbox")
        ev.params.max_dets = max_dets
        ev.evaluate()
        ev.accumulate()
    finally:
        Path(gt_path).unlink(missing_ok=True)
        Path(dt_path).unlink(missing_ok=True)

    # ev.eval_imgs is a flat (K, A, I) list; cells are dicts (or None
    # when the AA4 skip fired). lvis-api uses snake_case keys —
    # different from pycocotools' camelCase — so the projection below
    # uses snake_case throughout. `dt_matches` ships as a (T, D) i64
    # array of GT ids (0 for unmatched); we collapse to the bool
    # "matched at all" so the J1 id-space mismatch with vernier
    # doesn't fail the diff.
    norm: list[dict[str, Any] | None] = []
    if include_eval_imgs:
        for cell in ev.eval_imgs:
            if cell is None:
                norm.append(None)
                continue
            dt_matched = np.asarray(cell["dt_matches"], dtype=np.int64) > 0
            norm.append(
                {
                    "dt_scores": np.asarray(cell["dt_scores"], dtype=np.float64),
                    "dt_matches": dt_matched,
                    "dt_ignore": np.asarray(cell["dt_ignore"], dtype=bool),
                    "gt_ignore": np.asarray(cell["gt_ignore"], dtype=bool),
                }
            )
    # ev.eval["precision"] is `(T, R, K, A)` — no M-axis (AF5), so
    # the shape lines up with vernier's drop above.
    precision = np.asarray(ev.eval["precision"], dtype=np.float64)
    # `LVISEval.summarize` writes `self.results` as an OrderedDict.
    # Run it before harvesting; print suppression is the migration
    # guide's job (the oracle's `print_results` is the upstream
    # surface — we never call it here).
    ev.summarize()
    keys = _stats_keys(max_dets)
    stats: dict[str, float] = {}
    for k in keys:
        if k not in ev.results:
            raise AssertionError(
                f"lvis-api results is missing key {k!r}; present keys: {sorted(ev.results.keys())}"
            )
        stats[k] = float(ev.results[k])
    return LvisSnapshot(eval_imgs=norm, precision=precision, stats=stats)


def snapshot(
    impl: ImplName,
    gt_bytes: bytes,
    dt_bytes: bytes,
    *,
    max_dets: int = 300,
    include_eval_imgs: bool = True,
) -> LvisSnapshot:
    """Run one implementation and return a comparable snapshot.

    ``max_dets`` defaults to LVIS's canonical ``300`` (AC1) but is
    explicit so per-fixture tests can exercise the trim's edge cases.

    ``include_eval_imgs`` defaults to ``True`` for fixture-scale tests
    where the per-cell payload is the load-bearing diff. Set to
    ``False`` for whole-dataset smokes — the LVIS v1 val grid is
    1203 categories * 4 area buckets * 19809 images = ~95M cells,
    and materializing both impls' lists simultaneously costs ~1.5 GB
    of Python references on top of the actual eval payloads. Enough
    headroom on a 16 GB box to OOM the test even though the precision
    tensors and 13-stat summaries comfortably fit.
    """
    if impl == "vernier":
        return _vernier_snapshot(
            gt_bytes, dt_bytes, max_dets=max_dets, include_eval_imgs=include_eval_imgs
        )
    if impl == "lvis_api":
        return _lvis_snapshot(
            gt_bytes, dt_bytes, max_dets=max_dets, include_eval_imgs=include_eval_imgs
        )
    raise ValueError(f"unknown impl: {impl!r}")


def assert_snapshots_equal(
    a: LvisSnapshot,
    b: LvisSnapshot,
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> None:
    """Strict-mode bit-equality on `eval_imgs` (cell-by-cell `None`
    discrimination + per-array equality) and tolerated drift on
    `precision` (rtol/atol — defaults to bit-equal but parametric so
    the LVIS_PARITY_EPS tolerance can be threaded through).
    """
    if a.eval_imgs or b.eval_imgs:
        # Empty lists on both sides means both snapshots opted out of
        # eval_imgs materialization (whole-dataset smoke); skip the
        # cell-by-cell diff. The precision-tensor diff below is the
        # load-bearing check in that case.
        assert len(a.eval_imgs) == len(b.eval_imgs), (
            f"eval_imgs length mismatch: {len(a.eval_imgs)} vs {len(b.eval_imgs)}"
        )
        for idx, (ca, cb) in enumerate(zip(a.eval_imgs, b.eval_imgs, strict=True)):
            if ca is None and cb is None:
                continue
            if ca is None or cb is None:
                raise AssertionError(
                    f"eval_imgs[{idx}] None mismatch: vernier={ca is None}, lvis_api={cb is None}"
                )
            for key in ("dt_scores", "dt_matches", "dt_ignore", "gt_ignore"):
                np.testing.assert_array_equal(
                    ca[key], cb[key], err_msg=f"eval_imgs[{idx}].{key} differs"
                )
    np.testing.assert_allclose(
        a.precision, b.precision, rtol=rtol, atol=atol, err_msg="precision tensor differs"
    )
    if a.stats.keys() != b.stats.keys():
        raise AssertionError(
            f"stats keys differ: vernier={sorted(a.stats.keys())} lvis_api={sorted(b.stats.keys())}"
        )
    for key in a.stats:
        np.testing.assert_allclose(
            a.stats[key],
            b.stats[key],
            rtol=rtol,
            atol=atol,
            err_msg=f"stats[{key!r}] differs",
        )


def fixture_bytes(fixture_name: str) -> tuple[bytes, bytes]:
    """Read a fixture's `gt.json` + `dt.json` as the bytes the harness
    feeds to both impls.
    """
    base = Path(__file__).parent / "fixtures" / fixture_name
    gt = (base / "gt.json").read_bytes()
    dt = (base / "dt.json").read_bytes()
    return gt, dt


def fixture_payload(fixture_name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Same as :func:`fixture_bytes` but parsed — useful when a test
    needs to assert on individual fields of the fixture."""
    base = Path(__file__).parent / "fixtures" / fixture_name
    gt = json.loads((base / "gt.json").read_text())
    dt = json.loads((base / "dt.json").read_text())
    return gt, dt
