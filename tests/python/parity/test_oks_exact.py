"""Bit-exactness of the OKS similarity matrix against real pycocotools.

The rest of the keypoints parity suite compares `eval_imgs` / `precision`
/ `stats`, which is what users see but not what this file is about: a
1-ULP move in an OKS cell usually washes out by the time it reaches AP,
and only shows up when it happens to flip a match at a threshold
boundary (quirk **B2**). These tests compare the `ious` matrices
themselves, cell by cell, by bit pattern -- so quirks **F8** (numpy's
pairwise summation) and **F9** (the sigma table's separate `/ 10.0`)
are held directly rather than through whatever survives the accumulator.

Coverage is a sweep rather than a fixture corpus, because the quirks
being pinned are *statistical*: any single cell has a decent chance of
rounding the same way under a left fold. Before quirks F8/F9 were fixed,
these same datasets ran 54.7 %-87.8 % bit-exact against pycocotools,
with individual cells off by up to 2688 ULP.

`exp` is in scope here only incidentally. vernier calls Rust's
`f64::exp` and pycocotools calls `np.exp`, and nothing makes those two
agree by construction -- numpy dispatches on SIMD width and Rust goes
to the platform libm. They happen to agree on every input this sweep
generates, which is a measurement of this machine, not a guarantee.
Should a platform turn up where they do not, the divergence belongs in
its own quirk row with its own disposition; it must not be absorbed by
loosening the comparison here to a tolerance.
"""

from __future__ import annotations

import contextlib
import io
import random
from typing import Any, cast

import numpy as np
import pytest
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval as PycocotoolsReferenceCOCOeval

from vernier import COCOeval

#: COCO-person. 17 < NPY_PW_BLOCKSIZE, so the reduction stays in
#: numpy's 8-accumulator block.
K_PERSON = 17

#: COCO-WholeBody. 133 > 128, so the reduction reaches the *recursive*
#: arm of `DOUBLE_pairwise_sum` and splits 64 + 69. Reachable only
#: because quirk **F1** ships per-category sigmas of arbitrary length.
K_WHOLEBODY = 133

#: A WholeBody-shaped sigma vector. The real table is not vendored here;
#: what matters for F8 is the length and that the values vary.
_WHOLEBODY_SIGMAS: tuple[float, ...] = tuple((26.0 + (i % 84)) / 1000.0 for i in range(K_WHOLEBODY))


def _coco(dataset: dict[str, Any]) -> COCO:
    coco = COCO()
    # pycocotools' stubs type `dataset` as its file-loaded TypedDict.
    cast(Any, coco).dataset = dataset
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco


def _build(
    k: int,
    *,
    seed: int,
    n_images: int = 8,
    gts_per_image: int = 6,
    dts_per_image: int = 9,
    visibility: str = "random",
    zero_area: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """A synthetic keypoints dataset with integer coordinates.

    Integer coordinates are the point, not an accident: that is what
    makes pycocotools' `np.array(ann['keypoints'])` an **int64** array,
    which is the case quirk **F10**'s bound exists to cover.

    ``visibility``:
      ``"random"``  -- a realistic mix, `k1` lands wherever it lands.
      ``"sweep"``   -- `k1` walks 0..k across annotations, so both the
                       `k1 == 0` surrogate branch (**F3**) and every
                       term count in between are exercised.
      ``"none"``    -- every flag 0, so every GT takes the **F3**
                       bbox-surrogate path.
    """
    rng = random.Random(seed)
    images = [{"id": i, "width": 640, "height": 480} for i in range(n_images)]
    categories = [{"id": 1, "name": "person", "supercategory": "person"}]
    gt_anns: list[dict[str, Any]] = []
    dt_anns: list[dict[str, Any]] = []

    for image_id in range(n_images):
        for _ in range(gts_per_image):
            xs = [rng.randrange(0, 600) for _ in range(k)]
            ys = [rng.randrange(0, 440) for _ in range(k)]
            if visibility == "none":
                vs = [0] * k
            elif visibility == "sweep":
                n_visible = (len(gt_anns) * 7) % (k + 1)
                vs = [2] * n_visible + [0] * (k - n_visible)
            else:
                vs = [rng.choice([0, 1, 2, 2, 2]) for _ in range(k)]
            keypoints: list[int] = []
            for x, y, v in zip(xs, ys, vs, strict=True):
                keypoints += [x, y, v]
            gt_anns.append(
                {
                    "id": len(gt_anns) + 1,
                    "image_id": image_id,
                    "category_id": 1,
                    "keypoints": keypoints,
                    "num_keypoints": sum(1 for v in vs if v > 0),
                    # Zero area exercises the `area + np.spacing(1)`
                    # guard (quirk **F2**) rather than tripping over it.
                    "area": 0 if zero_area else rng.randrange(400, 200_000),
                    "bbox": [float(min(xs)), float(min(ys)), 40.0, 60.0],
                    "iscrowd": 0,
                }
            )
        for _ in range(dts_per_image):
            xs = [rng.randrange(0, 600) for _ in range(k)]
            ys = [rng.randrange(0, 440) for _ in range(k)]
            keypoints = []
            for x, y in zip(xs, ys, strict=True):
                keypoints += [x, y, 1]
            dt_anns.append(
                {
                    "id": len(dt_anns) + 1,
                    "image_id": image_id,
                    "category_id": 1,
                    "keypoints": keypoints,
                    "score": rng.random(),
                    "area": 2400,
                    "bbox": [0.0, 0.0, 40.0, 60.0],
                    "iscrowd": 0,
                }
            )

    gt = {"images": images, "annotations": gt_anns, "categories": categories}
    dt = {"images": images, "annotations": dt_anns, "categories": categories}
    return gt, dt


def _ious(
    evaluator_type: type[Any],
    gt: dict[str, Any],
    dt: dict[str, Any],
    sigmas: tuple[float, ...] | None,
) -> dict[Any, Any]:
    evaluator = evaluator_type(_coco(gt), _coco(dt), iouType="keypoints")
    if sigmas is not None:
        evaluator.params.kpt_oks_sigmas = np.asarray(sigmas, dtype=np.float64)
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.evaluate()
    return dict(evaluator.ious)


def _assert_ious_bit_equal(
    gt: dict[str, Any],
    dt: dict[str, Any],
    sigmas: tuple[float, ...] | None = None,
    *,
    min_cells: int = 100,
) -> None:
    reference = _ious(PycocotoolsReferenceCOCOeval, gt, dt, sigmas)
    candidate = _ious(COCOeval, gt, dt, sigmas)
    assert set(candidate) == set(reference)

    compared = 0
    for key, expected in reference.items():
        want = np.asarray(expected, dtype=np.float64).ravel()
        got = np.asarray(candidate[key], dtype=np.float64).ravel()
        assert want.shape == got.shape, f"{key}: shape {got.shape} != {want.shape}"
        if want.size == 0:
            continue
        compared += want.size
        want_bits = want.view(np.uint64)
        got_bits = got.view(np.uint64)
        bad = np.flatnonzero(want_bits != got_bits)
        if bad.size:
            i = int(bad[0])
            ulps = np.abs(want_bits.astype(np.int64) - got_bits.astype(np.int64))
            raise AssertionError(
                f"{key}: {bad.size}/{want.size} OKS cells are not bit-equal "
                f"(max {int(ulps.max())} ULP). First at flat index {i}: "
                f"vernier {got[i]!r} (0x{int(got_bits[i]):016x}) vs pycocotools "
                f"{want[i]!r} (0x{int(want_bits[i]):016x})."
            )

    # A vacuous pass is the failure mode this guards: an all-empty `ious`
    # would satisfy every assertion above.
    assert compared >= min_cells, f"only {compared} cells compared"


@pytest.mark.parity
def test_oks_matrix_is_bit_equal_for_coco_person() -> None:
    """K=17 with a realistic visibility mix (quirks F8, F9)."""
    gt, dt = _build(K_PERSON, seed=1)
    _assert_ious_bit_equal(gt, dt, min_cells=400)


@pytest.mark.parity
def test_oks_matrix_is_bit_equal_across_the_whole_k1_range() -> None:
    """`k1` walks 0..17, so every term count the reduction can see is hit.

    Includes `k1 == 0`, which takes the **F3** bbox-surrogate branch and
    sums over all `k` terms instead of the visible subset -- a different
    reduction length for the same annotation.
    """
    gt, dt = _build(K_PERSON, seed=2, visibility="sweep")
    _assert_ious_bit_equal(gt, dt, min_cells=400)


@pytest.mark.parity
def test_oks_matrix_is_bit_equal_on_the_zero_visibility_surrogate() -> None:
    """Every GT takes the **F3** surrogate path (quirk F4's expansion)."""
    gt, dt = _build(K_PERSON, seed=3, visibility="none")
    _assert_ious_bit_equal(gt, dt, min_cells=400)


@pytest.mark.parity
def test_oks_matrix_is_bit_equal_with_zero_area_ground_truth() -> None:
    """`area == 0` leans the whole normaliser on `np.spacing(1)` (F2).

    The exponent then blows up and most terms underflow `exp(-e)` to
    exactly `0.0`, which is precisely the regime where a wrong summation
    order is *invisible* -- so this fixture is here for the `+ eps`
    guard, not for F8.
    """
    gt, dt = _build(K_PERSON, seed=4, zero_area=True)
    _assert_ious_bit_equal(gt, dt, min_cells=400)


@pytest.mark.parity
def test_oks_matrix_is_bit_equal_for_wholebody_133_keypoints() -> None:
    """K=133 reaches the recursive arm of numpy's pairwise sum.

    133 > `NPY_PW_BLOCKSIZE` (128), so `DOUBLE_pairwise_sum` splits into
    64 + 69 rather than running one block. This is the arm that a
    "keypoint vectors are always short" reading of the port would delete
    -- and COCO-WholeBody makes it a production path, not a hypothetical.
    """
    gt, dt = _build(K_WHOLEBODY, seed=5, visibility="sweep", n_images=6)
    _assert_ious_bit_equal(gt, dt, sigmas=_WHOLEBODY_SIGMAS, min_cells=200)
