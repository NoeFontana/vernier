"""Parity harness for the panoptic-quality evaluation surface (ADR-0025).

Mirrors `tests/python/parity_lvis/harness.py`: builds a
[`PanopticSnapshot`] from each implementation (vendored
`panopticapi` oracle vs vernier) and diffs them via
[`assert_snapshots_equal`].

The oracle path calls ``pq_compute_single_core`` **directly** with
``proc_id=0`` per ADR-0025 §"Strict-mode parity claim" — bypassing
``pq_compute``'s multiprocessing pool, which has no ``num_proc``
parameter and would otherwise pin the comparison to the harness
host's CPU count (X1, X2). Multi-process traces are out of scope
here; they get bounded-ULP equality under ``ParityMode::Aligned`` only.

Fixtures are constructed as Python ``np.uint32`` arrays. The oracle
path round-trips them through PNGs (via :func:`id2rgb` + Pillow),
written to a temp dir; the vernier path passes the arrays directly
to :meth:`PanopticDataset.from_arrays`.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image as PILImage

import vernier

ImplName = Literal["panopticapi", "vernier"]


@dataclass(frozen=True, slots=True)
class PanopticSnapshot:
    """Captured per-implementation result. The shared shape is the
    panopticapi ``pq_compute`` return: All / Things / Stuff buckets
    + per-class rows. ``per_class`` carries the strict W8 shape
    (``{pq, sq, rq}`` only) so both implementations produce
    identical dicts; vernier's count fields are accessed via
    `to_dict(strict=False)` separately when a test cares about them.
    """

    pq: float
    sq: float
    rq: float
    n: int
    pq_things: float
    sq_things: float
    rq_things: float
    n_things: int
    pq_stuff: float
    sq_stuff: float
    rq_stuff: float
    n_stuff: int
    per_class: dict[int, dict[str, float]] = field(default_factory=dict)


def id2rgb(id_map: NDArray[np.uint32]) -> NDArray[np.uint8]:
    """Inverse of ``panopticapi.utils.rgb2id``. Encodes a panoptic
    label map (``id = R + 256*G + 256²*B``) as a 3-channel uint8
    RGB image suitable for PNG round-tripping."""
    out = np.zeros((*id_map.shape, 3), dtype=np.uint8)
    work = id_map.astype(np.uint32, copy=True)
    for i in range(3):
        out[..., i] = (work % 256).astype(np.uint8)
        work //= 256
    return out


def _write_png(label_map: NDArray[np.uint32], path: Path) -> None:
    rgb = id2rgb(label_map)
    PILImage.fromarray(rgb, mode="RGB").save(path)


def _oracle_snapshot(
    label_maps_gt: Mapping[int, NDArray[np.uint32]],
    segments_gt: Mapping[int, Sequence[Mapping[str, Any]]],
    label_maps_dt: Mapping[int, NDArray[np.uint32]],
    segments_dt: Mapping[int, Sequence[Mapping[str, Any]]],
    categories: Sequence[Mapping[str, Any]],
) -> PanopticSnapshot:
    # Lazy import: panopticapi lives on `sys.path` via `conftest.py`.
    from panopticapi.evaluation import pq_compute_single_core  # type: ignore[import-not-found]

    cats_dict = {c["id"]: dict(c) for c in categories}

    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        gt_dir = tmp_p / "gt"
        dt_dir = tmp_p / "dt"
        gt_dir.mkdir()
        dt_dir.mkdir()

        gt_anns = []
        dt_anns = []
        for image_id, label_map in label_maps_gt.items():
            file_name = f"{image_id:012d}.png"
            _write_png(label_map, gt_dir / file_name)
            gt_anns.append(
                {
                    "image_id": image_id,
                    "file_name": file_name,
                    "segments_info": [dict(s) for s in segments_gt[image_id]],
                }
            )
        for image_id, label_map in label_maps_dt.items():
            file_name = f"{image_id:012d}.png"
            _write_png(label_map, dt_dir / file_name)
            dt_anns.append(
                {
                    "image_id": image_id,
                    "file_name": file_name,
                    "segments_info": [dict(s) for s in segments_dt[image_id]],
                }
            )

        annotation_set = list(zip(gt_anns, dt_anns, strict=True))
        pq_stat = pq_compute_single_core(0, annotation_set, str(gt_dir), str(dt_dir), cats_dict)

    return pq_stat_to_snapshot(pq_stat, cats_dict)


def pq_stat_to_snapshot(
    pq_stat: Any, cats_dict: Mapping[int, Mapping[str, Any]]
) -> PanopticSnapshot:
    """Project a panopticapi `PQStat` into a `PanopticSnapshot`. Shared
    by `_oracle_snapshot` (fixture-driven) and the val-smoke test
    (real GT/DT). The three `pq_average` calls are independent —
    the All bucket carries the full `per_class` row map."""
    all_d, per_class = pq_stat.pq_average(cats_dict, isthing=None)
    things_d, _ = pq_stat.pq_average(cats_dict, isthing=True)
    stuff_d, _ = pq_stat.pq_average(cats_dict, isthing=False)
    return PanopticSnapshot(
        pq=all_d["pq"],
        sq=all_d["sq"],
        rq=all_d["rq"],
        n=all_d["n"],
        pq_things=things_d["pq"],
        sq_things=things_d["sq"],
        rq_things=things_d["rq"],
        n_things=things_d["n"],
        pq_stuff=stuff_d["pq"],
        sq_stuff=stuff_d["sq"],
        rq_stuff=stuff_d["rq"],
        n_stuff=stuff_d["n"],
        per_class={int(k): dict(v) for k, v in per_class.items()},
    )


def summary_to_snapshot(summary: vernier.PanopticSummary) -> PanopticSnapshot:
    """Project a `vernier.PanopticSummary` into a `PanopticSnapshot`.
    Coerces the `Option<f64>` / `Option<usize>` bucket fields to
    `0.0` / `0` for the empty-bucket case — panopticapi's
    `pq_average` returns the same shape (its `n!=0` guard avoids
    `ZeroDivisionError` only because the test fixtures keep both
    buckets non-empty)."""
    return PanopticSnapshot(
        pq=summary.pq,
        sq=summary.sq,
        rq=summary.rq,
        n=summary.n,
        pq_things=summary.pq_things if summary.pq_things is not None else 0.0,
        sq_things=summary.sq_things if summary.sq_things is not None else 0.0,
        rq_things=summary.rq_things if summary.rq_things is not None else 0.0,
        n_things=summary.n_things if summary.n_things is not None else 0,
        pq_stuff=summary.pq_stuff if summary.pq_stuff is not None else 0.0,
        sq_stuff=summary.sq_stuff if summary.sq_stuff is not None else 0.0,
        rq_stuff=summary.rq_stuff if summary.rq_stuff is not None else 0.0,
        n_stuff=summary.n_stuff if summary.n_stuff is not None else 0,
        per_class={
            int(cat): {"pq": row.pq, "sq": row.sq, "rq": row.rq}
            for cat, row in summary.per_class().items()
        },
    )


def _vernier_snapshot(
    label_maps_gt: Mapping[int, NDArray[np.uint32]],
    segments_gt: Mapping[int, Sequence[Mapping[str, Any]]],
    label_maps_dt: Mapping[int, NDArray[np.uint32]],
    segments_dt: Mapping[int, Sequence[Mapping[str, Any]]],
    categories: Sequence[Mapping[str, Any]],
    parity_mode: vernier.ParityMode,
) -> PanopticSnapshot:
    gt_segs_bytes = json.dumps({str(k): list(v) for k, v in segments_gt.items()}).encode()
    dt_segs_bytes = json.dumps({str(k): list(v) for k, v in segments_dt.items()}).encode()
    cats_bytes = json.dumps([dict(c) for c in categories]).encode()

    gt = vernier.PanopticDataset.from_arrays(
        {int(k): v for k, v in label_maps_gt.items()},
        gt_segs_bytes,
        cats_bytes,
    )
    dt = vernier.PanopticPredictions.from_arrays(
        {int(k): v for k, v in label_maps_dt.items()},
        dt_segs_bytes,
    )
    summary = vernier.PanopticEvaluator(parity_mode=parity_mode, things_stuff_split=True).evaluate(
        gt, dt
    )
    return summary_to_snapshot(summary)


def snapshot(
    impl: ImplName,
    label_maps_gt: Mapping[int, NDArray[np.uint32]],
    segments_gt: Mapping[int, Sequence[Mapping[str, Any]]],
    label_maps_dt: Mapping[int, NDArray[np.uint32]],
    segments_dt: Mapping[int, Sequence[Mapping[str, Any]]],
    categories: Sequence[Mapping[str, Any]],
    *,
    parity_mode: vernier.ParityMode = "corrected",
) -> PanopticSnapshot:
    """Run one implementation against the fixture and return the
    [`PanopticSnapshot`]. ``parity_mode`` is honored on the vernier
    side only; the oracle is panopticapi's single canonical behavior."""
    if impl == "panopticapi":
        return _oracle_snapshot(label_maps_gt, segments_gt, label_maps_dt, segments_dt, categories)
    elif impl == "vernier":
        return _vernier_snapshot(
            label_maps_gt,
            segments_gt,
            label_maps_dt,
            segments_dt,
            categories,
            parity_mode,
        )
    else:
        raise ValueError(f"unknown impl {impl!r}")


def assert_snapshots_equal(
    a: PanopticSnapshot,
    b: PanopticSnapshot,
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> None:
    """Bit-equal by default. Pass ``rtol=BOUNDARY_PARITY_EPS`` (or
    ``PANOPTIC_PARITY_EPS``) to gate aligned-mode comparisons. The
    `n_*` count fields are always compared exactly."""
    np.testing.assert_allclose(a.pq, b.pq, rtol=rtol, atol=atol, err_msg="global PQ differs")
    np.testing.assert_allclose(a.sq, b.sq, rtol=rtol, atol=atol, err_msg="global SQ differs")
    np.testing.assert_allclose(a.rq, b.rq, rtol=rtol, atol=atol, err_msg="global RQ differs")
    assert a.n == b.n, f"global n differs: {a.n} vs {b.n}"

    np.testing.assert_allclose(
        a.pq_things, b.pq_things, rtol=rtol, atol=atol, err_msg="things PQ differs"
    )
    np.testing.assert_allclose(
        a.sq_things, b.sq_things, rtol=rtol, atol=atol, err_msg="things SQ differs"
    )
    np.testing.assert_allclose(
        a.rq_things, b.rq_things, rtol=rtol, atol=atol, err_msg="things RQ differs"
    )
    assert a.n_things == b.n_things, f"n_things differs: {a.n_things} vs {b.n_things}"

    np.testing.assert_allclose(
        a.pq_stuff, b.pq_stuff, rtol=rtol, atol=atol, err_msg="stuff PQ differs"
    )
    np.testing.assert_allclose(
        a.sq_stuff, b.sq_stuff, rtol=rtol, atol=atol, err_msg="stuff SQ differs"
    )
    np.testing.assert_allclose(
        a.rq_stuff, b.rq_stuff, rtol=rtol, atol=atol, err_msg="stuff RQ differs"
    )
    assert a.n_stuff == b.n_stuff, f"n_stuff differs: {a.n_stuff} vs {b.n_stuff}"

    # per_class: union of keys, then per-row pq/sq/rq.
    all_keys = sorted(set(a.per_class) | set(b.per_class))
    for k in all_keys:
        row_a = a.per_class.get(k, {"pq": 0.0, "sq": 0.0, "rq": 0.0})
        row_b = b.per_class.get(k, {"pq": 0.0, "sq": 0.0, "rq": 0.0})
        for metric in ("pq", "sq", "rq"):
            np.testing.assert_allclose(
                row_a[metric],
                row_b[metric],
                rtol=rtol,
                atol=atol,
                err_msg=f"per_class[{k}][{metric}] differs",
            )
