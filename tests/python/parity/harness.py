"""Parity harness: dual-run two implementations and diff every intermediate.

Public API:
    snapshot(impl, gt_path, dt_path, iou_type, *, sigmas=None) -> EvalSnapshot
    assert_snapshots_equal(a, b, *, rtol=0, atol=0) -> None

The reference implementation is pycocotools 2.0.11, pinned in pyproject.toml.
The candidate ("vernier") routes through ``vernier._core``'s granular
``evaluate_bbox_grid`` / ``accumulate`` / ``summarize`` chain so each
intermediate stage can be diffed independently against pycocotools.

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
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import vernier._core as _vernier_core

IouType = Literal["bbox", "segm", "keypoints"]
Impl = Literal["pycocotools", "vernier"]

# Pycocotools' detection defaults (cocoeval.Params.setDetParams).
_DEFAULT_MAX_DETS: tuple[int, ...] = (1, 10, 100)
# Pycocotools' keypoints defaults (cocoeval.Params.setKpParams).
_DEFAULT_KP_MAX_DETS: tuple[int, ...] = (20,)
# ADR-0002's net-new default is "corrected"; the parity harness pins
# "strict" because its job is bit-equality vs pycocotools.
_PARITY_MODE: Literal["strict", "corrected"] = "strict"

#: Per-category sigma override; ``Mapping[category_id, sigmas_already_divided_by_10]``.
#: ``None`` means "use the kernel default" (COCO-person 17 sigmas on both
#: sides). Sigmas are passed verbatim to vernier and stored on
#: ``cocoeval.params.kpt_oks_sigmas`` for pycocotools — pycocotools' setKpParams
#: divides the raw table by 10 once at construction; users override post-divide.
SigmasMap = Mapping[int, tuple[float, ...]]


@dataclass(frozen=True)
class EvalSnapshot:
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
    iou_type: IouType,
    *,
    sigmas: SigmasMap | None = None,
) -> EvalSnapshot:
    if impl == "pycocotools":
        return _run_pycocotools(gt_path, dt_path, iou_type, sigmas=sigmas)
    if impl == "vernier":
        return _run_vernier(gt_path, dt_path, iou_type, sigmas=sigmas)
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


def _run_pycocotools(
    gt_path: Path,
    dt_path: Path,
    iou_type: IouType,
    *,
    sigmas: SigmasMap | None = None,
) -> EvalSnapshot:
    gt = COCO(str(gt_path))
    dt = gt.loadRes(str(dt_path))
    cocoeval = COCOeval(gt, dt, iouType=iou_type)
    if sigmas is not None:
        # Pycocotools holds a single global sigma vector on Params (quirk
        # F1 hardcodes COCO-person sigmas; downstream forks monkey-patch).
        # The fixture's per-category mapping is collapsed to that single
        # vector by reading the cat_id present on the GT — the kp parity
        # fixtures use one category, so this is unambiguous.
        if len(sigmas) != 1:
            raise ValueError(
                "pycocotools side accepts a single sigma vector; the fixture "
                f"supplied {len(sigmas)} per-category overrides. Split the "
                "fixture into one-cat fixtures or extend this helper.",
            )
        ((_cat, sig),) = sigmas.items()
        cocoeval.params.kpt_oks_sigmas = np.asarray(sig, dtype=np.float64)

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


def _run_vernier(
    gt_path: Path,
    dt_path: Path,
    iou_type: IouType,
    *,
    sigmas: SigmasMap | None = None,
) -> EvalSnapshot:
    gt_bytes = gt_path.read_bytes()
    # pycocotools' DT JSON is a bare list of detections; vernier accepts
    # the same shape via CocoDetections::from_json_bytes.
    dt_bytes = dt_path.read_bytes()

    if iou_type == "keypoints":
        max_dets = list(_DEFAULT_KP_MAX_DETS)
        sigmas_dict: dict[int, list[float]] = (
            {} if sigmas is None else {cat: list(sig) for cat, sig in sigmas.items()}
        )
        grid = _vernier_core.evaluate_keypoints_grid(
            gt_bytes, dt_bytes, _PARITY_MODE, max(max_dets), True, sigmas_dict
        )
        acc = grid.accumulate(max_dets)
        # The FFI Accumulated.summarize hardcodes the detection 12-stat
        # plan, which would index off the end of the kp accumulator's
        # 3-bucket A-axis. The unified `evaluate_keypoints_summary`
        # entrypoint dispatches to the kp summary plan; call it for the
        # stats vector and reuse the grid for eval_imgs / acc for the
        # tensors. Same input bytes → byte-identical eval_imgs / acc.
        summary = _vernier_core.evaluate_keypoints_summary(
            gt_bytes, dt_bytes, _PARITY_MODE, max_dets, True, sigmas_dict
        )
    else:
        max_dets = list(_DEFAULT_MAX_DETS)
        grid_fn = (
            _vernier_core.evaluate_segm_grid
            if iou_type == "segm"
            else _vernier_core.evaluate_bbox_grid
        )
        grid = grid_fn(gt_bytes, dt_bytes, _PARITY_MODE, max(max_dets), use_cats=True)
        acc = grid.accumulate(max_dets)
        summary = acc.summarize(max_dets)

    return EvalSnapshot(
        eval_imgs=[_normalize_eval_img(e) for e in grid.eval_imgs()],
        precision=np.asarray(acc.precision).copy(),
        recall=np.asarray(acc.recall).copy(),
        scores=np.asarray(acc.scores).copy(),
        counts=list(acc.counts),
        stats=np.asarray(summary.stats, dtype=np.float64).copy(),
    )


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
