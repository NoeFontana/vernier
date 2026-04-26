"""Parity harness: dual-run two implementations and diff every intermediate.

Public API:
    snapshot(impl, gt_path, dt_path, iou_type) -> EvalSnapshot
    assert_snapshots_equal(a, b, *, rtol=0, atol=0) -> None

The reference implementation is pycocotools 2.0.11, pinned in pyproject.toml.
The candidate ("vernier") implementation currently delegates to pycocotools
because the Rust evaluator hasn't landed yet. When real eval code ships,
``_run_vernier`` is the single function to update.

Snapshot contents capture the entire COCOeval state machine:
- ``eval_imgs``: the [K*A*I] flat list of per-(category, area-range, image)
  match dicts produced by ``COCOeval.evaluate``. ``None`` entries (cells with
  no GT and no DT) are preserved.
- ``precision`` (T,R,K,A,M), ``recall`` (T,K,A,M), ``scores`` (T,R,K,A,M):
  the dense arrays produced by ``COCOeval.accumulate``.
- ``stats``: the 12-element det / 10-element kp summary array.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

IouType = Literal["bbox", "segm", "keypoints"]
Impl = Literal["pycocotools", "vernier"]


@dataclass(frozen=True)
class EvalSnapshot:
    eval_imgs: list[dict[str, Any] | None]
    precision: np.ndarray
    recall: np.ndarray
    scores: np.ndarray
    counts: list[int]
    stats: np.ndarray


def snapshot(impl: Impl, gt_path: Path, dt_path: Path, iou_type: IouType) -> EvalSnapshot:
    if impl == "pycocotools":
        return _run_pycocotools(gt_path, dt_path, iou_type)
    if impl == "vernier":
        return _run_vernier(gt_path, dt_path, iou_type)
    raise ValueError(f"unknown impl: {impl!r}")


def assert_snapshots_equal(
    a: EvalSnapshot,
    b: EvalSnapshot,
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> None:
    """Assert two snapshots match elementwise.

    Default tolerances are zero — strict bit-equality. Loosen per fixture only
    when documenting a known numerical drift (e.g. accumulated float ordering
    inside a stable reduction).
    """
    assert a.counts == b.counts, f"counts differ: {a.counts} vs {b.counts}"
    _assert_eval_imgs_equal(a.eval_imgs, b.eval_imgs, rtol=rtol, atol=atol)
    np.testing.assert_allclose(a.precision, b.precision, rtol=rtol, atol=atol, err_msg="precision")
    np.testing.assert_allclose(a.recall, b.recall, rtol=rtol, atol=atol, err_msg="recall")
    np.testing.assert_allclose(a.scores, b.scores, rtol=rtol, atol=atol, err_msg="scores")
    np.testing.assert_allclose(a.stats, b.stats, rtol=rtol, atol=atol, err_msg="stats")


def _run_pycocotools(gt_path: Path, dt_path: Path, iou_type: IouType) -> EvalSnapshot:
    gt = COCO(str(gt_path))
    dt = gt.loadRes(str(dt_path))
    cocoeval = COCOeval(gt, dt, iouType=iou_type)

    with contextlib.redirect_stdout(io.StringIO()):
        cocoeval.evaluate()
        cocoeval.accumulate()
        cocoeval.summarize()

    return EvalSnapshot(
        eval_imgs=[_normalize_eval_img(e) for e in cocoeval.evalImgs],
        precision=np.asarray(cocoeval.eval["precision"]).copy(),
        recall=np.asarray(cocoeval.eval["recall"]).copy(),
        scores=np.asarray(cocoeval.eval["scores"]).copy(),
        counts=list(cocoeval.eval["counts"]),
        stats=np.asarray(cocoeval.stats).copy(),
    )


def _run_vernier(gt_path: Path, dt_path: Path, iou_type: IouType) -> EvalSnapshot:
    # Stub: until the Rust evaluator lands, vernier delegates to pycocotools.
    # When that changes, the parity suite stops being a tautology.
    return _run_pycocotools(gt_path, dt_path, iou_type)


def _normalize_eval_img(e: Any) -> dict[str, Any] | None:
    # Accepts pycocotools' _ImageEvaluationResult TypedDict or None. We type
    # the parameter as Any because the stubs export a TypedDict and a plain
    # dict-shaped param would reject it under strict pyright.
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
