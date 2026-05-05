"""End-to-end boundary-IoU parity harness (ADR-0010 §"Oracle (E2 + E3)").

Mirrors ``tests/python/parity/harness.py`` for the bbox/segm tracks: dual-runs
the vendored ``bowenc0221/boundary-iou-api`` oracle and vernier's Rust
``evaluate_boundary_grid`` chain on the same fixture, captures every
intermediate of the COCOeval state machine, and exposes a comparator that
diffs them elementwise.

Vernier's ``min(mask_iou, boundary_iou)`` composition (quirk **O1**) and
crowd-fallback to mask-only IoU (quirk **O2**) are exercised via
``evaluate_boundary_grid``; the oracle exercises the equivalent path inside
``COCOeval.computeBoundaryIoU``. Anything that isn't bit-equal points at a
divergence in one of those compositions or in upstream segm parity (which
boundary inherits).
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

# ``conftest.py`` puts the vendored oracle on ``sys.path`` and installs the
# matplotlib / multiprocessing / ``np.float`` shims it needs to import on a
# modern interpreter. Importing here (rather than inside the function) lets
# ``ImportError`` surface at collection time if the oracle ever drifts.
from boundary_iou.coco_instance_api.coco import COCO  # type: ignore[import-not-found]
from boundary_iou.coco_instance_api.cocoeval import COCOeval  # type: ignore[import-not-found]

import vernier._core as _vernier_core
from vernier._types import DEFAULT_DILATION_RATIO, PARITY_STRICT

Impl = Literal["oracle", "vernier"]

# Detection defaults from `pycocotools.cocoeval.Params.setDetParams`; the
# oracle's `Params(iouType="boundary")` reuses them.
_DEFAULT_MAX_DETS: tuple[int, ...] = (1, 10, 100)


@dataclass(frozen=True)
class BoundaryEvalSnapshot:
    eval_imgs: list[dict[str, Any] | None]
    precision: np.ndarray
    recall: np.ndarray
    scores: np.ndarray
    counts: list[int]
    stats: np.ndarray


def snapshot(
    impl: Impl,
    gt_path: Path,
    dt_path: Path,
    *,
    dilation_ratio: float = DEFAULT_DILATION_RATIO,
) -> BoundaryEvalSnapshot:
    if impl == "oracle":
        return _run_oracle(gt_path, dt_path, dilation_ratio)
    if impl == "vernier":
        return _run_vernier(gt_path, dt_path, dilation_ratio)
    raise ValueError(f"unknown impl: {impl!r}")


def assert_snapshots_equal(
    a: BoundaryEvalSnapshot,
    b: BoundaryEvalSnapshot,
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> None:
    """Diff two snapshots elementwise.

    Defaults are zero — strict bit-equality. Loosen per fixture only when
    documenting a known numerical drift (e.g. cv2 erode vs the
    morphological reference rounding at sub-pixel dilations).
    """
    assert a.counts == b.counts, f"counts differ: {a.counts} vs {b.counts}"
    _assert_eval_imgs_equal(a.eval_imgs, b.eval_imgs, rtol=rtol, atol=atol)
    np.testing.assert_allclose(a.precision, b.precision, rtol=rtol, atol=atol, err_msg="precision")
    np.testing.assert_allclose(a.recall, b.recall, rtol=rtol, atol=atol, err_msg="recall")
    np.testing.assert_allclose(a.scores, b.scores, rtol=rtol, atol=atol, err_msg="scores")
    np.testing.assert_allclose(a.stats, b.stats, rtol=rtol, atol=atol, err_msg="stats")


def _run_oracle(gt_path: Path, dt_path: Path, dilation_ratio: float) -> BoundaryEvalSnapshot:
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(gt_path), get_boundary=True, dilation_ratio=dilation_ratio)
        dt = gt.loadRes(str(dt_path))
        cocoeval = COCOeval(gt, dt, iouType="boundary", dilation_ratio=dilation_ratio)
        cocoeval.evaluate()
        cocoeval.accumulate()
        cocoeval.summarize()

    return BoundaryEvalSnapshot(
        eval_imgs=[_normalize_eval_img(e) for e in cocoeval.evalImgs],
        precision=np.asarray(cocoeval.eval["precision"]).copy(),
        recall=np.asarray(cocoeval.eval["recall"]).copy(),
        scores=np.asarray(cocoeval.eval["scores"]).copy(),
        counts=list(cocoeval.eval["counts"]),
        stats=np.asarray(cocoeval.stats).copy(),
    )


def _run_vernier(gt_path: Path, dt_path: Path, dilation_ratio: float) -> BoundaryEvalSnapshot:
    gt_bytes = gt_path.read_bytes()
    dt_bytes = dt_path.read_bytes()
    max_dets = list(_DEFAULT_MAX_DETS)

    grid = _vernier_core.evaluate_boundary_grid(
        gt_bytes,
        dt_bytes,
        PARITY_STRICT,
        max(max_dets),
        use_cats=True,
        dilation_ratio=dilation_ratio,
    )
    acc = grid.accumulate(max_dets)
    summary = acc.summarize(max_dets)

    return BoundaryEvalSnapshot(
        eval_imgs=[_normalize_eval_img(e) for e in grid.eval_imgs()],
        precision=np.asarray(acc.precision).copy(),
        recall=np.asarray(acc.recall).copy(),
        scores=np.asarray(acc.scores).copy(),
        counts=list(acc.counts),
        stats=np.asarray(summary.stats, dtype=np.float64).copy(),
    )


def _normalize_eval_img(e: Any) -> dict[str, Any] | None:
    if e is None:
        return None
    return {
        "image_id": int(e["image_id"]),
        "category_id": int(e["category_id"]),
        "aRng": [float(x) for x in e["aRng"]],
        "maxDet": int(e["maxDet"]),
        "dtIds": [int(i) for i in e["dtIds"]],
        "gtIds": [int(i) for i in e["gtIds"]],
        "dtMatches": np.asarray(e["dtMatches"], dtype=np.float64),
        "gtMatches": np.asarray(e["gtMatches"], dtype=np.float64),
        "dtScores": [float(s) for s in e["dtScores"]],
        "gtIgnore": np.asarray(e["gtIgnore"], dtype=np.float64),
        "dtIgnore": np.asarray(e["dtIgnore"], dtype=np.float64),
    }


def _assert_eval_imgs_equal(
    a: list[dict[str, Any] | None],
    b: list[dict[str, Any] | None],
    *,
    rtol: float,
    atol: float,
) -> None:
    assert len(a) == len(b), f"eval_imgs length differs: {len(a)} vs {len(b)}"
    for idx, (ea, eb) in enumerate(zip(a, b, strict=True)):
        if ea is None and eb is None:
            continue
        assert ea is not None, f"eval_imgs[{idx}] is None in a but not b"
        assert eb is not None, f"eval_imgs[{idx}] is None in b but not a"
        for key in ("image_id", "category_id", "maxDet", "aRng", "dtIds", "gtIds"):
            assert ea[key] == eb[key], f"eval_imgs[{idx}].{key}: {ea[key]} vs {eb[key]}"
        for key in ("dtMatches", "gtMatches", "gtIgnore", "dtIgnore"):
            np.testing.assert_allclose(
                ea[key], eb[key], rtol=rtol, atol=atol, err_msg=f"eval_imgs[{idx}].{key}"
            )
        np.testing.assert_allclose(
            ea["dtScores"],
            eb["dtScores"],
            rtol=rtol,
            atol=atol,
            err_msg=f"eval_imgs[{idx}].dtScores",
        )
