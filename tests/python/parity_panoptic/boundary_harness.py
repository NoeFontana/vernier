"""Parity harness for the **Boundary PQ** panoptic surface (ADR-0010
boundary-IoU extension applied to panoptic; see plan
``priority-3-boundary-glowing-hollerith``).

Mirrors :mod:`tests.python.parity_panoptic.harness` but the oracle
binding is ``bowenc0221/boundary-iou-api``'s
``coco_panoptic_api.evaluation.pq_compute_single_core`` invoked with
``iou_type="boundary"`` and ``dilation_ratio=0.02``. The upstream
implementation is the **same kernel as panopticapi's PQ orchestrator**,
modified by Bowen Cheng to compute the per-pair ``min(mask_iou,
boundary_iou)`` substitution and to mutate the JSON-order
``segments_info`` in place when computing boundary masks (see
``coco_panoptic_api/evaluation.py`` lines 124-148).

Snapshot shape: same :class:`PanopticSnapshot` as the non-boundary
harness — so :func:`tests.python.parity_panoptic.harness.assert_snapshots_equal`
applies verbatim. The dataclass is re-exported here for callers' import
convenience.

Layout / sys.path:
   The vendored ``boundary_iou_api`` lives at
   ``oracle/boundary_iou_api/`` with subpackages ``coco_panoptic_api``
   and ``utils`` (the latter is a single-line shim re-exporting
   ``mask_to_boundary`` from the parity_boundary vendor at the same
   pinned SHA — see ``oracle/VENDORING.md``). We insert two roots on
   ``sys.path`` at import time:

   1. The panopticapi vendor root (so the upstream
      ``from panopticapi.utils import get_traceback, rgb2id`` succeeds;
      the panoptic conftest already does this, but we do it again here
      so the harness is import-time self-sufficient when used outside
      pytest).
   2. The parity_boundary vendor root (so the utils shim's
      ``from boundary_iou.utils.boundary_utils import mask_to_boundary``
      resolves — the boundary parity tree maintains the canonical
      ``boundary_utils`` copy and this harness re-uses it as a single
      source of truth).
   3. The ``oracle/`` directory (so the new ``boundary_iou_api``
      package itself is importable by name).
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

import vernier

from .harness import (
    PanopticSnapshot,
    assert_snapshots_equal,  # re-export
    pq_stat_to_snapshot,
    prepare_pq_inputs,
    summary_to_snapshot,
)

__all__ = [
    "PanopticSnapshot",
    "assert_snapshots_equal",
    "boundary_snapshot",
    "oracle_boundary_snapshot",
    "vernier_boundary_snapshot",
]

ImplName = Literal["boundary_iou_api", "vernier"]

# Strict-tier bit-equality target. Held in sync with the Rust-side
# ``BOUNDARY_PARITY_EPS`` in ``crates/vernier-core/src/boundary_parity.rs``
# (the constant the Rust agent pins for the boundary surface). For the
# panoptic + boundary fold the floating-point work is the same shape as
# the non-boundary panoptic path — pure division of integer sums — so
# bit-equality at 0.0 is the strict target. We expose the constant so
# tests can opt into a documented tolerance when running under aligned
# / corrected modes.
BOUNDARY_PARITY_EPS: float = 1e-9


def _ensure_sys_path() -> None:
    """Insert the three sys.path roots needed for the vendored
    panoptic-boundary oracle to import cleanly. Idempotent.

    Order: parity_panoptic's panopticapi vendor first (so the upstream
    ``from panopticapi.utils import ...`` in ``evaluation.py`` line 22
    resolves), then the parity_boundary tree's ``boundary_iou_api/``
    (so ``from boundary_iou.utils.boundary_utils import
    mask_to_boundary`` resolves inside our utils shim), then this
    tree's ``oracle/`` directory (so ``import boundary_iou_api``
    resolves to the new vendored package).
    """
    here = Path(__file__).parent
    panoptic_oracle = here / "oracle" / "panopticapi"
    boundary_oracle = here.parent / "parity_boundary" / "oracle" / "boundary_iou_api"
    new_oracle_root = here / "oracle"
    for p in (panoptic_oracle, boundary_oracle, new_oracle_root):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)


_ensure_sys_path()


def oracle_boundary_snapshot(
    label_maps_gt: Mapping[int, NDArray[np.uint32]],
    segments_gt: Mapping[int, Sequence[Mapping[str, Any]]],
    label_maps_dt: Mapping[int, NDArray[np.uint32]],
    segments_dt: Mapping[int, Sequence[Mapping[str, Any]]],
    categories: Sequence[Mapping[str, Any]],
    *,
    dilation_ratio: float = 0.02,
) -> PanopticSnapshot:
    """Run the vendored ``bowenc0221/boundary-iou-api`` oracle's
    ``pq_compute_single_core(..., iou_type="boundary", dilation_ratio=...)``
    against the fixture and return a :class:`PanopticSnapshot`.

    Shares the temp-dir + PNG-write + annotation-list scaffolding with
    the non-boundary panoptic harness (:func:`prepare_pq_inputs`) — the
    only difference is the extra ``"boundary", dilation_ratio``
    positional pair on the call.
    """
    _ensure_sys_path()
    # Lazy import: vendored package is only on sys.path after
    # _ensure_sys_path runs.
    from boundary_iou_api.coco_panoptic_api.evaluation import (  # type: ignore[import-not-found]
        pq_compute_single_core,
    )

    with prepare_pq_inputs(label_maps_gt, segments_gt, label_maps_dt, segments_dt, categories) as (
        gt_dir,
        dt_dir,
        annotation_set,
        cats_dict,
    ):
        pq_stat = pq_compute_single_core(
            0,
            annotation_set,
            str(gt_dir),
            str(dt_dir),
            cats_dict,
            "boundary",
            dilation_ratio,
        )
        return pq_stat_to_snapshot(pq_stat, cats_dict)


def vernier_boundary_snapshot(
    label_maps_gt: Mapping[int, NDArray[np.uint32]],
    segments_gt: Mapping[int, Sequence[Mapping[str, Any]]],
    label_maps_dt: Mapping[int, NDArray[np.uint32]],
    segments_dt: Mapping[int, Sequence[Mapping[str, Any]]],
    categories: Sequence[Mapping[str, Any]],
    *,
    parity_mode: vernier.ParityMode = "strict",
    dilation_ratio: float = 0.02,
) -> PanopticSnapshot:
    """Run vernier's panoptic evaluator with ``boundary=True`` and
    return a :class:`PanopticSnapshot`. ``parity_mode`` selects strict
    vs corrected behavior; ``dilation_ratio`` is forwarded to the
    vernier-side boundary-band computation.

    The API surface is pinned by the parallel Rust+Python agent's
    work (see plan ``priority-3-boundary-glowing-hollerith``); the
    keyword arguments here are the contract:

    .. code-block:: python

       vernier.panoptic.Evaluator(
           parity_mode="strict",
           boundary=True,
           dilation_ratio=0.02,
       )
    """
    gt_segs_bytes = json.dumps({str(k): list(v) for k, v in segments_gt.items()}).encode()
    dt_segs_bytes = json.dumps({str(k): list(v) for k, v in segments_dt.items()}).encode()
    cats_bytes = json.dumps([dict(c) for c in categories]).encode()

    gt = vernier.panoptic.Dataset.from_arrays(
        {int(k): v for k, v in label_maps_gt.items()},
        gt_segs_bytes,
        cats_bytes,
    )
    dt = vernier.panoptic.Predictions.from_arrays(
        {int(k): v for k, v in label_maps_dt.items()},
        dt_segs_bytes,
    )
    # ``dilation_ratio=`` is part of the pinned contract with the
    # parallel Rust+Python agent landing the kernel side of boundary PQ.
    # The field is not yet on the public ``Evaluator`` dataclass — the
    # ``type: ignore`` removes once the integration PR adds the field.
    summary = vernier.panoptic.Evaluator(
        parity_mode=parity_mode,
        things_stuff_split=True,
        boundary=True,
        dilation_ratio=dilation_ratio,
    ).evaluate(gt, dt)
    return summary_to_snapshot(summary)


def boundary_snapshot(
    impl: ImplName,
    label_maps_gt: Mapping[int, NDArray[np.uint32]],
    segments_gt: Mapping[int, Sequence[Mapping[str, Any]]],
    label_maps_dt: Mapping[int, NDArray[np.uint32]],
    segments_dt: Mapping[int, Sequence[Mapping[str, Any]]],
    categories: Sequence[Mapping[str, Any]],
    *,
    parity_mode: vernier.ParityMode = "strict",
    dilation_ratio: float = 0.02,
) -> PanopticSnapshot:
    """Dispatch to one of the two boundary-PQ implementations and
    return a :class:`PanopticSnapshot`. ``parity_mode`` is honored on
    the vernier side only; the oracle is the boundary-iou-api fork's
    single canonical behavior."""
    if impl == "boundary_iou_api":
        return oracle_boundary_snapshot(
            label_maps_gt,
            segments_gt,
            label_maps_dt,
            segments_dt,
            categories,
            dilation_ratio=dilation_ratio,
        )
    elif impl == "vernier":
        return vernier_boundary_snapshot(
            label_maps_gt,
            segments_gt,
            label_maps_dt,
            segments_dt,
            categories,
            parity_mode=parity_mode,
            dilation_ratio=dilation_ratio,
        )
    else:
        raise ValueError(f"unknown impl {impl!r}")
