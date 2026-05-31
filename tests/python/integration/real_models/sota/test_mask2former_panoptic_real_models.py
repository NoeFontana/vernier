"""Real-prediction parity smoke for Mask2Former Swin-T panoptic vs panopticapi.

Sibling to ``test_detr_real_models.py``: real DT, real GT, strict+aligned
two-tier parity claim. The model is
``facebook/mask2former-swin-tiny-coco-panoptic``; predictions land
under ``mask2former_panoptic_cache_dir()`` as one rgb2id-encoded PNG
per image + per-image JSON sidecars + a single aggregated
``panoptic_dt.json``, all keyed on the pinned hub commit SHA.

What this suite gates:

- **Coverage** — every GT image has a matching DT entry. A partial
  populator run (SIGINT mid-sweep before the aggregate JSON write)
  cannot pass this gate; coupled with the populator's per-image
  atomic-resume contract, the test only runs on a fully-populated
  cache.
- **Strict bit-equality on the per-class integer surface** —
  ``PQStatCat.tp / fp / fn`` (panopticapi) ↔ ``n_tp / n_fp / n_fn``
  (vernier ``EvalResult.per_class``). The category-id sets must
  match exactly. A divergence here is a real accumulator bug — float
  reduction order can't shift integers.
- **Aligned-tier float drift, 8 ULP relative + absolute** — per-class
  ``iou_sum`` (the only reduction where reduction order matters at the
  integer-input layer), and the float averages emitted by
  ``PanopticSnapshot`` (global / Things / Stuff PQ/SQ/RQ + per-class
  PQ/SQ/RQ rows). The constant carries both ``rtol`` AND ``atol`` so
  a class whose oracle metric collapses to exact ``0.0`` while
  vernier's reduction yields a sub-ULP non-zero still passes the gate
  — the float-zero boundary is symmetric.

Skips cleanly when:

- ``real-models`` extra is missing (conftest's ``pytest.importorskip``).
- ``MASK2FORMER_PANOPTIC_REVISION`` is the ``_UNPINNED_REVISION``
  sentinel.
- COCO panoptic val2017 cache is not provisioned (i.e. user hasn't
  run ``python -m panoptic_val_cache``).
- COCO val2017 images are not present (``VERNIER_COCO_CACHE`` points
  at an incomplete layout).

First-time inference takes ~20-25h on an 8-core CPU (5000 images x
~14-18s/image; Mask2Former Swin-T runs 100 queries through 9
transformer decoder layers per image). Subsequent runs are seconds
once the prediction cache is on disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ....parity_panoptic.harness import (
    PanopticSnapshot,
    assert_snapshots_equal,
    pq_stat_to_snapshot,
    summary_to_snapshot,
)

pytestmark = pytest.mark.real_models

#: 8 ULP of float64 — used as BOTH ``rtol`` and ``atol`` on the
#: float-surface assertions. Absorbs the reduction-order drift on
#: per-category ``iou_sum``, per-class PQ/SQ/RQ, and the
#: (Things|Stuff) bucket averages.
#:
#: The integer surface (``PQStatCat.tp / fp / fn`` ↔
#: ``n_tp / n_fp / n_fn``) is asserted bit-equal separately and is
#: the load-bearing parity claim; the float ``sum(iou)`` per category
#: and the per-category ``avg(metric)`` per bucket reduce in different
#: orders between panopticapi (Python dict iteration over
#: ``pq_per_cat``) and vernier (its own per-category accumulator),
#: accumulating up to a few ULP on a 5000-image / 133-class corpus.
#:
#: Observed on the live cache (2026-05-31):
#: - bucket level (Things SQ): max abs diff ``1.11e-16`` (= 0.5 ULP)
#: - per-class level (cat 3 PQ): max abs diff ``5.55e-16`` (= 2.5 ULP)
#: All other 5 buckets and 132/133 per-class rows are bit-equal.
#:
#: Why ``atol`` matters: ``assert_allclose`` evaluates
#: ``|a-b| ≤ atol + rtol * |b|``. For a category whose oracle metric
#: rounds to exact ``0.0`` while vernier's reduction yields
#: ``1.11e-16``, an ``rtol``-only gate fails because ``rtol * 0 = 0``
#: — even though the drift is exactly the reduction-order kind the
#: gate exists to absorb. Setting ``atol = 8 * eps`` makes the band
#: symmetric across the float-zero boundary.
#:
#: 8 ULP keeps any genuine kernel divergence (e.g. a wrong IoU
#: numerator for one image) above the gate; a single bit-flip in the
#: numerator would shift the per-class PQ by orders of magnitude more
#: than this floor.
_PANOPTIC_PARITY_RTOL = 8.0 * float(np.finfo(np.float64).eps)
_PANOPTIC_PARITY_ATOL = _PANOPTIC_PARITY_RTOL


def _build_label_maps(
    annotations: list[dict[str, Any]],
    png_dir: Path,
) -> dict[int, Any]:
    """Decode a list of (image_id, file_name) → uint32 label-map dict.

    Mirrors ``test_panoptic_val._build_label_maps``: rgb2id decode via
    Pillow + numpy. Pulled inline rather than imported to avoid
    coupling the SOTA harness to the parity_panoptic module's internal
    helpers (which are private to the val-smoke test).
    """
    import numpy as np
    from PIL import Image as PILImage

    out: dict[int, Any] = {}
    for ann in annotations:
        path = png_dir / ann["file_name"]
        rgb = np.array(PILImage.open(path), dtype=np.uint32)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise ValueError(f"non-RGB panoptic PNG: {path}")
        out[int(ann["image_id"])] = rgb[:, :, 0] + 256 * rgb[:, :, 1] + 256 * 256 * rgb[:, :, 2]
    return out


def _oracle_snapshot(
    gt_dict: dict[str, Any],
    gt_png_dir: Path,
    dt_annotations: list[dict[str, Any]],
    dt_png_dir: Path,
) -> tuple[PanopticSnapshot, Any, list[dict[str, Any]], dict[int, dict[str, Any]]]:
    """Run pq_compute_single_core on the (intersect of GT, DT) image set.

    Returns ``(snapshot, raw_pq_stat, matched_gt_anns, cats_dict)`` —
    ``raw_pq_stat`` is the panopticapi ``PQStat`` whose
    ``pq_per_cat[label].(tp/fp/fn/iou)`` carries the integer surface
    asserted bit-equal by the test (the float snapshot drops these
    integer fields by design).
    """
    from panopticapi.evaluation import pq_compute_single_core  # type: ignore[import-not-found]

    dt_by_image = {a["image_id"]: a for a in dt_annotations}
    matched: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for gt_ann in gt_dict["annotations"]:
        if gt_ann["image_id"] in dt_by_image:
            matched.append((gt_ann, dt_by_image[gt_ann["image_id"]]))

    cats_dict: dict[int, dict[str, Any]] = {c["id"]: c for c in gt_dict["categories"]}
    pq_stat = pq_compute_single_core(0, matched, str(gt_png_dir), str(dt_png_dir), cats_dict)
    snap = pq_stat_to_snapshot(pq_stat, cats_dict)
    gt_anns = [m[0] for m in matched]
    return snap, pq_stat, gt_anns, cats_dict


def _np_default(o: object) -> int | float | list[Any]:
    """JSON encoder for numpy scalars / arrays — same shape as the
    sibling :func:`tests.python.parity_panoptic.test_panoptic_val._np_default`.

    ``pq_compute_single_core`` mutates ``segments_info[*]['area']`` with
    a numpy int64 value derived from the rasterized PNG; re-encoding
    through vernier needs the cast back to native Python.
    """
    import numpy as np

    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _vernier_snapshot(
    gt_anns: list[dict[str, Any]],
    gt_png_dir: Path,
    dt_annotations: list[dict[str, Any]],
    dt_png_dir: Path,
    cats_dict: dict[int, dict[str, Any]],
) -> tuple[PanopticSnapshot, Any]:
    """Run vernier-panoptic over the same matched image set.

    Returns ``(snapshot, per_class_df)`` — the per-class polars
    DataFrame carries the integer ``n_tp / n_fp / n_fn / iou_sum``
    surface the test asserts bit-equal against panopticapi's
    ``PQStatCat``. Requested via ``tables="per_class"`` (ADR-0038);
    the float snapshot drops these by design so a separate accessor
    is the only way to read them without leaving vernier's public
    API.
    """
    import vernier

    dt_by_image = {a["image_id"]: a for a in dt_annotations}
    matched_dt = [dt_by_image[ann["image_id"]] for ann in gt_anns]

    gt_label_maps = _build_label_maps(gt_anns, gt_png_dir)
    dt_label_maps = _build_label_maps(matched_dt, dt_png_dir)
    gt_segs = {ann["image_id"]: ann["segments_info"] for ann in gt_anns}
    dt_segs = {ann["image_id"]: ann["segments_info"] for ann in matched_dt}

    gt_segs_bytes = json.dumps(
        {str(k): v for k, v in gt_segs.items()}, default=_np_default
    ).encode()
    dt_segs_bytes = json.dumps(
        {str(k): v for k, v in dt_segs.items()}, default=_np_default
    ).encode()
    cats_bytes = json.dumps(list(cats_dict.values()), default=_np_default).encode()

    gt = vernier.panoptic.Dataset.from_arrays(gt_label_maps, gt_segs_bytes, cats_bytes)
    dt = vernier.panoptic.Predictions.from_arrays(dt_label_maps, dt_segs_bytes)
    result = vernier.panoptic.Evaluator(parity_mode="corrected").evaluate(
        gt, dt, tables=("per_class",)
    )
    return summary_to_snapshot(result.summary), result.per_class


def _assert_integer_surface_strict(
    pq_stat: Any,
    vernier_per_class: Any,
) -> None:
    """Bit-equal ``tp / fp / fn`` per category between panopticapi and vernier.

    panopticapi's ``PQStat.pq_per_cat`` is a ``defaultdict`` keyed by
    category id; vernier's ``EvalResult.per_class`` polars DataFrame has
    one row per category with ``n_tp / n_fp / n_fn / iou_sum`` columns.
    Category id sets must match exactly. ``iou_sum`` is a float
    reduction, so it falls through to the aligned-tier 8-ULP gate.
    """
    vernier_rows = {int(r["category_id"]): r for r in vernier_per_class.iter_rows(named=True)}
    oracle_cats = set(pq_stat.pq_per_cat.keys())
    vernier_cats = set(vernier_rows.keys())
    only_oracle = oracle_cats - vernier_cats
    assert not only_oracle, f"oracle has categories vernier dropped: {sorted(only_oracle)[:10]}"
    only_vernier = vernier_cats - oracle_cats
    assert not only_vernier, f"vernier has categories oracle dropped: {sorted(only_vernier)[:10]}"
    for cat_id in sorted(oracle_cats):
        oracle = pq_stat.pq_per_cat[cat_id]
        vernier_row = vernier_rows[cat_id]
        assert oracle.tp == vernier_row["n_tp"], (
            f"cat {cat_id} TP diverges: oracle={oracle.tp} vs vernier={vernier_row['n_tp']}"
        )
        assert oracle.fp == vernier_row["n_fp"], (
            f"cat {cat_id} FP diverges: oracle={oracle.fp} vs vernier={vernier_row['n_fp']}"
        )
        assert oracle.fn == vernier_row["n_fn"], (
            f"cat {cat_id} FN diverges: oracle={oracle.fn} vs vernier={vernier_row['n_fn']}"
        )
        np.testing.assert_allclose(
            oracle.iou,
            vernier_row["iou_sum"],
            rtol=_PANOPTIC_PARITY_RTOL,
            atol=_PANOPTIC_PARITY_ATOL,
            err_msg=f"cat {cat_id} iou_sum (aligned-tier 8 ULP)",
        )


def test_mask2former_panoptic_parity_vs_panopticapi(
    coco_panoptic_gt: tuple[Path, Path, dict[str, Any]],
    mask2former_panoptic_cache_paths: tuple[Path, Path],
) -> None:
    """Two-tier parity vs panopticapi on real Mask2Former predictions.

    Strict tier: per-class ``tp / fp / fn`` integers + coverage.
    Aligned tier: per-class ``iou_sum`` + PQ/SQ/RQ float averages, at
    8 ULP relative AND absolute. The DETR-R50 ``dtScores`` ULP drift
    does not surface here — panoptic doesn't carry a score-tensor
    pipeline.
    """
    _, gt_png_dir, gt_dict = coco_panoptic_gt
    dt_png_dir, dt_json_path = mask2former_panoptic_cache_paths
    dt_json = json.loads(dt_json_path.read_bytes())
    dt_annotations = dt_json["annotations"]

    # Coverage gate: the populator only writes the aggregate JSON at
    # the end of a full run, so its presence implies every GT image
    # was processed. Assert it explicitly so a future change to the
    # populator's atomic-write contract can't silently shrink the
    # evaluation surface.
    gt_image_ids = {int(a["image_id"]) for a in gt_dict["annotations"]}
    dt_image_ids = {int(a["image_id"]) for a in dt_annotations}
    missing = gt_image_ids - dt_image_ids
    assert not missing, (
        f"populator produced {len(dt_image_ids)} predictions but GT has "
        f"{len(gt_image_ids)} images — {len(missing)} GT images missing "
        f"from DT (first 5: {sorted(missing)[:5]}). Either the populator "
        f"died mid-write without leaving the aggregate atomic, or the "
        f"GT/DT caches were pinned to different revisions."
    )

    oracle, pq_stat, gt_anns, cats_dict = _oracle_snapshot(
        gt_dict, gt_png_dir, dt_annotations, dt_png_dir
    )
    candidate, vernier_per_class = _vernier_snapshot(
        gt_anns, gt_png_dir, dt_annotations, dt_png_dir, cats_dict
    )

    # Strict tier — per-category integer surface bit-equal. Drift here
    # is a real accumulator divergence (float reduction order cannot
    # shift integers).
    _assert_integer_surface_strict(pq_stat, vernier_per_class)

    # Aligned tier — float reductions at 8 ULP rtol AND atol. The atol
    # is what keeps a PQ-collapses-to-zero category from spuriously
    # failing the rtol-only gate (rtol * 0 = 0).
    assert_snapshots_equal(
        oracle,
        candidate,
        rtol=_PANOPTIC_PARITY_RTOL,
        atol=_PANOPTIC_PARITY_ATOL,
    )
