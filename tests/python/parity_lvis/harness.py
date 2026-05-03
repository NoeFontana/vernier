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

# The oracle goes via the vendored tree (conftest patches `sys.path`);
# vernier ships an FFI grid helper used by the candidate path.
from vernier import Dataset
from vernier._core import evaluate_bbox_grid_with_dataset

ImplName = Literal["vernier", "lvis_api"]


@dataclass(frozen=True, slots=True)
class LvisSnapshot:
    """Output of one parity run.

    The harness in PR-3 only diffs ``eval_imgs`` (per-cell match
    payload) and ``precision`` (the ``(T, R, K, A)`` tensor at
    `max_dets=300`). PR-4 will extend this with a ``stats`` field for
    the 13-entry summary plan.
    """

    eval_imgs: list[dict[str, Any] | None]
    """Per-(category, area, image) cell. ``None`` matches lvis-api's
    ``eval_imgs[idx] = None``; otherwise a dict with ``dt_scores``,
    ``dt_matches``, ``dt_ignore``, ``gt_ignore`` arrays normalized to
    the same numeric layout the oracle emits."""

    precision: NDArray[np.float64]
    """Shape ``(T, R, K, A)`` precision tensor — no M-axis (AF5).
    The accumulator at ``max_dets=300`` produces this directly; we
    pin it here so a regression on the K-axis filter (PR-4) is
    visible *before* the summary collapse."""


def _vernier_snapshot(
    gt_bytes: bytes,
    dt_bytes: bytes,
    *,
    max_dets: int,
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
    cells = grid.eval_imgs()
    # vernier exposes per-cell payloads in pycocotools-shaped keys
    # (`dtScores`, `dtMatches`, `dtIgnore`, `gtIgnore`); the oracle
    # uses the same names. We normalize to snake_case for the
    # snapshot dict so downstream diffs read evenly.
    norm: list[dict[str, Any] | None] = []
    for cell in cells:
        if cell is None:
            norm.append(None)
            continue
        # `dtMatches` is `(T, D)` int (GT id on a hit, 0 on a miss);
        # we collapse to bool so vernier's id-space (J1) doesn't have
        # to align with the oracle's. `dtIgnore`/`gtIgnore` ship as
        # uint8 — cast to bool so equality checks are exact.
        dt_matches_id = np.asarray(cell["dtMatches"], dtype=np.int64)
        norm.append(
            {
                "dt_scores": np.asarray(cell["dtScores"], dtype=np.float64),
                "dt_matches": dt_matches_id > 0,
                "dt_ignore": np.asarray(cell["dtIgnore"], dtype=bool),
                "gt_ignore": np.asarray(cell["gtIgnore"], dtype=bool),
            }
        )
    # accum.precision is `(T, R, K, A, M)`; M is always 1 here, drop it.
    precision = np.asarray(accum.precision, dtype=np.float64)
    if precision.ndim == 5:
        assert precision.shape[-1] == 1, (
            f"vernier precision M-axis must be 1 at max_dets={max_dets}; got {precision.shape}"
        )
        precision = precision[..., 0]
    return LvisSnapshot(eval_imgs=norm, precision=precision)


def _lvis_snapshot(
    gt_bytes: bytes,
    dt_bytes: bytes,
    *,
    max_dets: int,
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
    return LvisSnapshot(eval_imgs=norm, precision=precision)


def snapshot(
    impl: ImplName,
    gt_bytes: bytes,
    dt_bytes: bytes,
    *,
    max_dets: int = 300,
) -> LvisSnapshot:
    """Run one implementation and return a comparable snapshot.

    ``max_dets`` defaults to LVIS's canonical ``300`` (AC1) but is
    explicit so per-fixture tests can exercise the trim's edge cases.
    """
    if impl == "vernier":
        return _vernier_snapshot(gt_bytes, dt_bytes, max_dets=max_dets)
    if impl == "lvis_api":
        return _lvis_snapshot(gt_bytes, dt_bytes, max_dets=max_dets)
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
