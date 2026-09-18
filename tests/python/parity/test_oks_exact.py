"""What vernier's OKS matrix guarantees against real pycocotools, and where.

The rest of the keypoints parity suite compares `eval_imgs` / `precision`
/ `stats`, which is what users see but not what this file is about: a
1-ULP move in an OKS cell usually washes out by the time it reaches AP,
and only shows up when it happens to flip a match at a threshold
boundary (quirk **B2**). These tests compare the `ious` matrices
themselves, cell by cell, by bit pattern.

## The claim, in three parts

`computeOks` is `np.sum(np.exp(-e)) / e.shape[0]` (ce:232). Everything
except the `exp` is exactly reproducible; `exp` is not, because **NumPy
has no single f64 `exp` kernel**. It dispatches on CPU features (SVML
`__svml_exp8*` on AVX512, otherwise the system libm), and the libm it
falls through to dispatches again on its own ifunc (glibc's FMA and
non-FMA `exp` disagree at 706 ppm, always by exactly 1 ULP). vernier
calls Rust's `f64::exp`, which is a libcall into the system libm. So:

1. **Unconditionally bit-exact, on every machine.** Everything upstream
   of `exp` -- the construction of `e` (three successive left-associative
   divisions, quirk **F9**'s separate `/ 10.0` on the sigma table,
   `(2*sigma)**2` as one multiply, quirk **F2**'s `area + np.spacing(1)`
   guard, and the visible-keypoint count that is both the mask and the
   divisor) -- and everything downstream of it -- quirk **F8**'s numpy
   pairwise reduction tree and quirk **F7**'s final divide by the term
   count. Asserted by `test_reduction_*`, `test_e_terms_*`,
   `test_default_sigmas_*` and `test_threshold_ladders_*`, none of which
   can be satisfied by a coincidence in `exp`.
2. **Conditionally bit-exact.** The *final* OKS value is bit-equal to
   pycocotools' **iff the local NumPy dispatches `exp` to the same libm
   Rust's `f64::exp` calls**. That is a property of the machine, not of
   vernier, so the `test_oks_matrix_is_bit_equal_*` tests are gated on a
   behavioural probe (`_exp_kernel_probe`) and skip loudly, naming the
   detected kernel, where they do not hold.
3. **Unconditional at the decision level.** Where the kernels do differ,
   each term moves by at most 1 ULP, so the mean moves by at most
   `2**-52` relative -- and `delta = 2**-46` holds with 64x margin. More
   usefully, matching gates on `iou >= t` (quirk **B2**), and no
   comparison in this suite lands on opposite sides of a threshold.
   Asserted unconditionally by `test_divergence_*`.

The divergence in (2) must **not** be absorbed into a tolerance on the
cell value: the first two parts are exact claims and stay exact. What
changed is the scope of the third, which was previously asserted as if
it were part of the first.

## Why a sweep rather than a fixture corpus

The quirks being pinned are *statistical*: any single cell has a decent
chance of rounding the same way under a left fold. Before quirks F8/F9
were fixed, these same datasets ran 54.7 %-87.8 % bit-exact against
pycocotools, with individual cells off by up to 2688 ULP.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import io
import random
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pycocotools.cocoeval as reference_cocoeval
import pytest
from numpy.typing import NDArray
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

#: The certified bound on `|vernier_oks - pycocotools_oks|` when the two
#: `exp` kernels differ. Derivation: `exp(-e) in (0, 1]` and two faithful
#: kernels land on adjacent doubles, so `|d_i| <= ulp(y_i) <= 2**-52 *
#: y_i`; `OKS = (sum y_i) / n`, so `|d_OKS| <= 2**-52 * OKS <= 2**-52`.
#: Note `n` **cancels** -- the term count and the shape of the reduction
#: tree do no work in this bound, which is why it is stated per-value and
#: not per-term. (`ulp(y) <= 2**-53 * y` would be wrong: for
#: `y in [2**k, 2**(k+1))`, `ulp(y)/y` reaches `2**-52`.) Measured tight
#: at `2.220e-16 == 2**-52` over 30 000 cells recomputed with a genuinely
#: different kernel. `2**-46` is the published constant, and it carries
#: 64x margin over `2**-52`.
OKS_DELTA = 2.0**-46


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


def _bits(a: NDArray[np.float64]) -> NDArray[np.uint64]:
    return np.ascontiguousarray(a, dtype=np.float64).view(np.uint64)


def _ulps(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.int64]:
    """Distance in representable doubles. Both sides here are positive."""
    return np.abs(_bits(a).astype(np.int64) - _bits(b).astype(np.int64))


def _fingerprint(a: NDArray[np.float64]) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# The exp-kernel probe
# ---------------------------------------------------------------------------
#
# A behavioural probe, not introspection. `np.__config__` and
# `np.core._multiarray_umath.__cpu_features__` describe the *build*, and
# the shipped NumPy wheel contains both SVML and a libm import -- the
# choice is made at runtime, so the build tells you nothing.
#
# The trick that makes it behavioural: an OKS cell whose GT has exactly
# one visible keypoint reduces to `numpy_pairwise_sum([t]) / 1.0`, which
# is `t == exp(-e)` exactly (`+0.0 + t == t`, `t / 1.0 == t`). So
# vernier's *public* `ious` hands back single `f64::exp` results, and the
# probe compares the two kernels that actually matter -- the one vernier
# calls and the one pycocotools calls -- rather than a proxy for them.
#
# The arguments are constructed to be exactly representable and exactly
# predictable: `sigma = 0.5` makes `vars = (2*sigma)**2 == 1.0`, and
# `area = 2**42` (with `area + np.spacing(1) == area`) makes the
# remaining two divisions exact power-of-two shifts. So
# `e == (dx*dx + dy*dy) / 2**43` exactly, for integer `dx`, `dy` well
# inside quirk **F10**'s `2**25` bound.

_PROBE_N_GT = 1700
_PROBE_N_DT = 20  # `Params.maxDets` for keypoints is [20]; more would be truncated
_PROBE_AREA_EXP = 42
_PROBE_SIGMA = 0.5

#: Measured with `decimal` at 45 digits: of the 34 000 probe arguments,
#: this many are ones where glibc 2.39's `exp` is **not** correctly
#: rounded (882 ppm, in line with the 745-817 ppm measured over separate
#: million-sample sweeps). Those are the near-ties where two faithful
#: kernels with different polynomials round apart, so they are what gives
#: the probe teeth: a kernel differing from the local libm at the ~700 ppm
#: rate of glibc's own FMA/non-FMA ifunc split is missed with probability
#: ~exp(-24) ~ 4e-11. It is recorded rather than recomputed because
#: deriving it needs 34 000 exact-decimal exponentials.
_PROBE_GLIBC_MISROUNDS = 30

#: Kernels this probe has actually seen, by fingerprint, so an unfamiliar
#: one is recognisable as unfamiliar rather than merely unequal. Not used
#: for gating -- the gate is the direct comparison, which needs no table
#: and cannot go stale on a libc update.
#:
#: * ``5413677b16f038c8`` -- glibc 2.39 x86-64, FMA path. What vernier's
#:   ``f64::exp`` resolves to, and what NumPy resolves to on a CPU
#:   without AVX512 (AMD EPYC-Milan locally; AMD EPYC 7763 and an
#:   AVX512-masked EPYC 9V74 on GitHub's runners).
#: * ``f0c033a043e12cfe`` -- NumPy 2.4.4 on an AVX512 AMD EPYC 9V74
#:   (GitHub Actions, 2026-09-18). Disagrees with the libm fingerprint on
#:   **1647 of 34 000** arguments (4.8 %) and **every one of them by
#:   exactly 1 ULP**. That last number settles a question the dispatch
#:   investigation had to leave open for want of AVX512 hardware: NumPy
#:   is calling the *high-accuracy* SVML variant (``__svml_exp8_ha``,
#:   faithful to ~0.5 ULP), not the ~4-ULP one. A 4-ULP kernel would
#:   have broken the per-term premise `OKS_DELTA` rests on; it does not.
#:
#: Note what the 4.8 % rate is not: a bound on anything. It is 68x the
#: 706 ppm of glibc's own FMA/non-FMA split, which is what a different
#: polynomial rather than a different rounding mode looks like.

#: Whether the AVX512 path is taken is a property of the *runner*, not of
#: the NumPy version: the same `ubuntu-latest` pool served both an EPYC
#: 9V74 exposing the full AVX512 flag set and one exposing none. So the
#: pre-gate failure was a lottery, not a deterministic per-leg failure --
#: worth knowing before reading anything into which matrix leg went red.


@dataclass(frozen=True)
class _ExpKernelProbe:
    """Whether NumPy's `exp` and vernier's `f64::exp` are the same kernel."""

    n_args: int
    n_differ: int
    max_ulp: int
    numpy_chunked_differs: bool
    vernier_fingerprint: str
    numpy_fingerprint: str
    libm_verdict: str

    @property
    def agrees(self) -> bool:
        return self.n_differ == 0 and not self.numpy_chunked_differs

    def reason(self) -> str:
        return (
            "the local NumPy does not dispatch `exp` to the kernel vernier's "
            "`f64::exp` calls, so the *final* OKS value cannot be bit-equal to "
            "pycocotools' on this machine -- which is a property of the machine, "
            "not a vernier regression. "
            f"Behavioural probe: {self.n_differ}/{self.n_args} single-term OKS "
            f"cells differ (max {self.max_ulp} ULP; {_PROBE_GLIBC_MISROUNDS} of "
            "those arguments are near-ties glibc itself misrounds, which is what "
            "makes the probe able to tell two faithful kernels apart); "
            f"np.exp fingerprint {self.numpy_fingerprint}, "
            f"vernier f64::exp fingerprint {self.vernier_fingerprint}; "
            f"{self.libm_verdict}"
            + (
                "; np.exp also disagrees with itself between a 34 000-element "
                "array and 17-element chunks, i.e. it dispatches on array length"
                if self.numpy_chunked_differs
                else ""
            )
            + ". The bit-exact claim is gated, not loosened: the reduction tree, "
            "`e` and the threshold ladders are still asserted bit-exactly by the "
            "unconditional tests in this file, and the decision-level guarantee "
            "is asserted by test_divergence_stays_within_delta_and_flips_no_match."
        )


def _probe_dataset() -> tuple[dict[str, Any], dict[str, Any], NDArray[np.float64]]:
    """Cells whose `e` is known exactly, laid out (n_dt, n_gt) like `ious`."""
    rng = np.random.default_rng(20260918)
    g_off = np.arange(_PROBE_N_GT, dtype=np.int64) * 7919
    dx = rng.integers(1 << 21, 1 << 22, size=_PROBE_N_DT, dtype=np.int64)
    dy = rng.integers(1 << 20, 1 << 22, size=_PROBE_N_DT, dtype=np.int64)
    images = [{"id": 0, "width": 64, "height": 64}]
    categories = [{"id": 1, "name": "probe", "supercategory": "probe"}]
    gt = {
        "images": images,
        "categories": categories,
        "annotations": [
            {
                "id": g + 1,
                "image_id": 0,
                "category_id": 1,
                "keypoints": [int(off), 0, 2],
                "num_keypoints": 1,
                "area": float(2**_PROBE_AREA_EXP),
                "bbox": [0.0, 0.0, 1.0, 1.0],
                "iscrowd": 0,
            }
            for g, off in enumerate(g_off)
        ],
    }
    dt = {
        "images": images,
        "categories": categories,
        "annotations": [
            # Scores strictly decreasing in index order, so the
            # sort-by-score inside the evaluator is the identity and
            # row `d` of `ious` is detection `d`.
            {
                "id": d + 1,
                "image_id": 0,
                "category_id": 1,
                "keypoints": [int(x), int(y), 1],
                "score": 1.0 - d / (2 * _PROBE_N_DT),
                "area": 1.0,
                "bbox": [0.0, 0.0, 1.0, 1.0],
                "iscrowd": 0,
            }
            for d, (x, y) in enumerate(zip(dx, dy, strict=True))
        ],
    }
    m = (dx[:, None] - g_off[None, :]) ** 2 + dy[:, None] ** 2
    e = m.astype(np.float64) / float(2**_PROBE_AREA_EXP) / 2.0
    return gt, dt, e


def _libm_verdict(args: NDArray[np.float64], vernier: NDArray[np.float64]) -> str:
    """Name each side's kernel against the system libm, for the report.

    Computed on every probe, agreeing or not, so the happy path is the
    same code the skip message goes through. It is a *naming* step, not
    the gate: the gate is the direct vernier-vs-NumPy comparison.
    """
    import ctypes
    import ctypes.util

    try:
        lib = ctypes.CDLL(ctypes.util.find_library("m") or "libm.so.6")
        lib.exp.restype = ctypes.c_double
        lib.exp.argtypes = [ctypes.c_double]
        libm = np.array([lib.exp(float(x)) for x in args], dtype=np.float64)
    except OSError as exc:  # pragma: no cover - platform dependent
        return (
            f"the system libm could not be reached through ctypes ({exc}), so the kernel is unnamed"
        )
    numpy_is_libm = bool((_bits(np.exp(args)) == _bits(libm)).all())
    vernier_is_libm = bool((_bits(vernier) == _bits(libm)).all())
    if vernier_is_libm and not numpy_is_libm:
        return (
            "vernier matches the system libm and NumPy does not, so NumPy is on a "
            "vectorised kernel -- on x86-64 with AVX512 that is SVML (`__svml_exp8*`), "
            "which the NumPy wheel ships alongside its libm import"
        )
    if numpy_is_libm and not vernier_is_libm:
        return "NumPy matches the system libm and vernier's `f64::exp` does not"
    if numpy_is_libm and vernier_is_libm:
        return "both are the system libm on the sampled arguments"
    return "neither matches the system libm on the sampled arguments"


@functools.cache
def _exp_kernel_probe() -> _ExpKernelProbe:
    gt, dt, e = _probe_dataset()
    matrices = _ious(COCOeval, gt, dt, (_PROBE_SIGMA,))
    if len(matrices) != 1:
        raise AssertionError(f"probe expected one (image, category) cell, got {len(matrices)}")
    vernier = np.asarray(next(iter(matrices.values())), dtype=np.float64)
    if vernier.shape != e.shape:
        raise AssertionError(f"probe cell is {vernier.shape}, expected {e.shape}")
    args = np.ascontiguousarray(-e).ravel()
    if args.size < 30_000:
        raise AssertionError(f"probe degenerated to {args.size} arguments")

    # Two NumPy legs. The bulk one gives the vectorised inner loop a long
    # contiguous buffer; the chunked one calls `np.exp` the way
    # `computeOks` does, on one COCO-person-sized array at a time, in
    # case dispatch varies with length.
    bulk = np.exp(args)
    chunked = np.concatenate(
        [np.exp(args[i : i + K_PERSON]) for i in range(0, args.size, K_PERSON)]
    )
    flat_vernier = vernier.ravel()
    differ = _bits(flat_vernier) != _bits(bulk)
    n_differ = int(differ.sum())
    sample = slice(0, 4096)
    return _ExpKernelProbe(
        n_args=int(args.size),
        n_differ=n_differ,
        max_ulp=int(_ulps(flat_vernier, bulk).max()) if n_differ else 0,
        numpy_chunked_differs=bool((_bits(chunked) != _bits(bulk)).any()),
        vernier_fingerprint=_fingerprint(flat_vernier),
        numpy_fingerprint=_fingerprint(bulk),
        libm_verdict=_libm_verdict(args[sample], flat_vernier[sample]),
    )


def _require_matching_exp_kernel() -> None:
    """Skip loudly, and only on a genuine kernel difference.

    A failure *inside* the probe propagates as an error rather than a
    skip: an unreachable probe must not quietly disarm the assertion it
    guards.
    """
    probe = _exp_kernel_probe()
    if not probe.agrees:
        pytest.skip(probe.reason())


# ---------------------------------------------------------------------------
# Recovering vernier's own per-term `exp(-e)` values
# ---------------------------------------------------------------------------


def _single_keypoint_terms(
    gt: dict[str, Any],
    dt: dict[str, Any],
    sigmas: tuple[float, ...] | None,
    k: int,
) -> dict[tuple[int, int], NDArray[np.float64]]:
    """vernier's `exp(-e_i)` terms, per `(annotation id, detection row)`.

    Same one-visible-keypoint reduction as the probe, applied to the real
    sweep datasets: for each GT and each of its visible keypoints, a
    surrogate GT that keeps only that keypoint visible. The returned
    array is indexed `[detection_row, term]` with terms in keypoint
    order, matching the `e` vector `computeOks` builds.
    """
    surrogates: list[dict[str, Any]] = []
    # Ground-truth column index *within its own image*, which is the
    # layout `ious` uses; global list position is not it.
    columns: dict[int, list[int]] = {}
    per_image: dict[int, int] = {}
    for ann in gt["annotations"]:
        visible = [i for i, v in enumerate(ann["keypoints"][2::3]) if v > 0]
        if not visible:
            continue
        image_id = ann["image_id"]
        assigned: list[int] = []
        for i in visible:
            keypoints = list(ann["keypoints"])
            for t in range(k):
                keypoints[3 * t + 2] = 2 if t == i else 0
            assigned.append(per_image.get(image_id, 0))
            per_image[image_id] = per_image.get(image_id, 0) + 1
            surrogates.append(
                {**ann, "id": len(surrogates) + 1, "keypoints": keypoints, "num_keypoints": 1}
            )
        columns[ann["id"]] = assigned

    matrices = _ious(COCOeval, {**gt, "annotations": surrogates}, dt, sigmas)
    out: dict[tuple[int, int], NDArray[np.float64]] = {}
    for ann in gt["annotations"]:
        assigned = columns.get(ann["id"], [])
        if not assigned:
            continue
        key = next(k2 for k2 in matrices if int(k2[0]) == ann["image_id"])
        cell = np.asarray(matrices[key], dtype=np.float64)
        for d in range(cell.shape[0]):
            out[(ann["id"], d)] = np.array([cell[d, c] for c in assigned], dtype=np.float64)
    return out


def _capture_pycocotools_e(
    gt: dict[str, Any],
    dt: dict[str, Any],
    sigmas: tuple[float, ...] | None,
) -> tuple[dict[Any, Any], list[NDArray[np.float64]]]:
    """Run the oracle, keeping every array it hands to `np.exp`.

    `computeOks` calls `np.exp` exactly once per `(gt, dt)` pair, with
    `-e` as the argument, so the captures are the pre-`exp` quantity
    itself: the three divisions, the sigma table, the `spacing(1)` guard
    and the visible-keypoint mask, all in one vector whose length is the
    divisor.
    """
    captured: list[NDArray[np.float64]] = []

    class _Spy:
        def __getattr__(self, name: str) -> Any:
            return getattr(np, name)

        def exp(self, a: Any) -> Any:
            captured.append(np.array(a, dtype=np.float64, copy=True))
            return np.exp(a)

    # `cocoeval` does `import numpy as np` at module scope and reaches
    # `np.exp` through that binding, so swapping the module attribute is
    # what intercepts it. The stubs do not declare the binding, hence the
    # cast.
    module = cast(Any, reference_cocoeval)
    module.np = _Spy()
    try:
        matrices = _ious(PycocotoolsReferenceCOCOeval, gt, dt, sigmas)
    finally:
        module.np = np
    return matrices, captured


def _oracle_cells(
    gt: dict[str, Any],
    dt: dict[str, Any],
) -> list[tuple[Any, int, dict[str, Any], list[dict[str, Any]]]]:
    """`(ious key, gt column, gt ann, detections)` in `computeOks` order.

    `computeOks` iterates ground truth in dataset order and detections
    sorted by descending score, and writes `ious[detection, gt]`. The
    order is reproduced here rather than assumed: every consumer checks
    it by reconstructing the oracle's own cell value from the captures.
    """
    out = []
    for image in gt["images"]:
        image_id = image["id"]
        gts = [a for a in gt["annotations"] if a["image_id"] == image_id]
        dts = sorted(
            (a for a in dt["annotations"] if a["image_id"] == image_id),
            key=lambda a: -a["score"],
        )
        if not gts or not dts:
            continue
        for column, ann in enumerate(gts):
            out.append(((image_id, 1), column, ann, dts))
    return out


# ---------------------------------------------------------------------------
# 1. Unconditional: everything except `exp`
# ---------------------------------------------------------------------------


@pytest.mark.parity
@pytest.mark.parametrize(
    ("k", "seed", "sigmas", "min_cells", "min_left_fold_diffs"),
    [
        pytest.param(K_PERSON, 2, None, 300, 12, id="coco-person-17"),
        pytest.param(K_WHOLEBODY, 5, _WHOLEBODY_SIGMAS, 150, 80, id="wholebody-133"),
    ],
)
def test_reduction_and_divide_are_bit_exact_whatever_exp_does(
    k: int,
    seed: int,
    sigmas: tuple[float, ...] | None,
    min_cells: int,
    min_left_fold_diffs: int,
) -> None:
    """Quirks **F8** and **F7**, asserted with `exp` factored out entirely.

    The comparison is `vernier_cell == np.sum(vernier_terms) / n` where
    `vernier_terms` are vernier's **own** `exp(-e_i)` values, recovered
    through one-visible-keypoint surrogate cells. Both sides therefore
    carry whatever `exp` the local machine has, and what is left under
    test is exactly the numpy pairwise reduction tree and the final
    divide by the term count. This holds on every machine, and it is the
    part of the claim this PR actually moved.

    `visibility="sweep"` walks `k1` across its whole range, so the
    plain-left-fold arm (`n < 8`), the 8-accumulator block (`8..=128`)
    and -- at K=133 -- the recursive arm that splits 64 + 69 are all
    reached.
    """
    gt, dt = _build(k, seed=seed, visibility="sweep", n_images=6)
    vernier = _ious(COCOeval, gt, dt, sigmas)
    terms = _single_keypoint_terms(gt, dt, sigmas, k)

    compared = 0
    left_fold_would_differ = 0
    term_counts: set[int] = set()
    for key, column, ann, dts in _oracle_cells(gt, dt):
        cell = np.asarray(vernier[next(k2 for k2 in vernier if tuple(map(int, k2)) == key)])
        for d in range(len(dts)):
            recovered = terms.get((ann["id"], d))
            if recovered is None:  # k1 == 0: the F3 surrogate sums over all k
                continue
            n = recovered.size
            term_counts.add(n)
            got = np.float64(cell[d, column])
            want = np.sum(recovered) / float(n)
            assert _bits(np.array([got]))[0] == _bits(np.array([want]))[0], (
                f"{key} cell (dt={d}, gt={column}) with {n} terms: vernier "
                f"{got!r} (0x{int(_bits(np.array([got]))[0]):016x}) != "
                f"np.sum(vernier's own terms)/{n} = {want!r} "
                f"(0x{int(_bits(np.array([want]))[0]):016x}). The exp kernel "
                "cannot explain this -- both sides used it identically."
            )
            compared += 1
            accumulator = np.float64(0.0)
            for term in recovered:
                accumulator = accumulator + term
            if _bits(np.array([accumulator / float(n)]))[0] != _bits(np.array([want]))[0]:
                left_fold_would_differ += 1

    assert compared >= min_cells, f"only {compared} cells compared"
    assert max(term_counts) == k, f"term counts {sorted(term_counts)} never reached k={k}"
    if k > 128:
        assert max(term_counts) > 128, "the recursive arm of the pairwise sum was never reached"
    # Teeth, not decoration: the assertion above is only worth anything if
    # the wrong answer is actually a different double on this data. A left
    # fold is the wrong answer quirk **F8** exists to rule out. Cells with
    # fewer than 8 terms cannot contribute -- numpy's own reduction *is* a
    # left fold below `NPY_PW_BLOCKSIZE`'s block size -- so the floors are
    # counts, not rates. Measured here: 21 of 180 eligible cells at K=17
    # (the `k1` sweep spends a third of its cells below 8 terms) and 137 of
    # 297 at K=133.
    assert left_fold_would_differ >= min_left_fold_diffs, (
        f"a left fold of the same terms would differ on only "
        f"{left_fold_would_differ} of {compared} cells (floor "
        f"{min_left_fold_diffs}), so this test barely discriminates the "
        "reduction order it claims to pin"
    )


@pytest.mark.parity
@pytest.mark.parametrize(
    ("k", "seed", "sigmas", "zero_area"),
    [
        pytest.param(K_PERSON, 1, None, False, id="coco-person-17"),
        pytest.param(K_PERSON, 4, None, True, id="zero-area-spacing-guard"),
        pytest.param(K_WHOLEBODY, 5, _WHOLEBODY_SIGMAS, False, id="wholebody-133"),
    ],
)
def test_e_terms_agree_with_pycocotools_term_for_term(
    k: int, seed: int, sigmas: tuple[float, ...] | None, zero_area: bool
) -> None:
    """The pre-`exp` pipeline, compared against the oracle's own `e`.

    `e` is observable only through `exp`, so this is stated at the tightest
    scope that survives a kernel swap: **every term is within 1 ULP**, and
    **bit-equal** whenever the probe says the kernels match.

    The 1-ULP form still has teeth against a wrong `e`, and the amount is
    quantified rather than asserted: `d(exp(-e)) / exp(-e) = -de`, so a
    1-ULP error in `e` moves the term by about `e * 2**-53` relative --
    roughly `e / 2` ULP. For every term with `e >= 4` a 1-ULP error in `e`
    therefore shows up as `>= 2` ULP on the term and is caught here even
    on a machine whose kernels differ. The test asserts that a large share
    of the terms are in that regime, so the bound is not satisfied by
    terms too small to constrain anything.

    Covered by construction, since all of it is folded into `e`: the three
    successive left-associative divisions `/(vars) /(area + eps) /2`
    (fusing them into one `/(vars * area * 2)` is a different number),
    quirk **F9**'s separate `/ 10.0` on the sigma table, `(2*sigma)**2` as
    a single multiply, quirk **F2**'s `np.spacing(1)` guard (leaned on
    hardest by the `zero_area` case, where it *is* the normaliser), and
    the visible-keypoint count -- which is simultaneously the mask, the
    length of `e` and the divisor.
    """
    gt, dt = _build(k, seed=seed, visibility="random", n_images=6, zero_area=zero_area)
    oracle, captured = _capture_pycocotools_e(gt, dt, sigmas)
    terms = _single_keypoint_terms(gt, dt, sigmas, k)
    kernels_match = _exp_kernel_probe().agrees

    compared = n_differ = n_sensitive = 0
    worst = 0
    for cell_index, (key, column, ann, dts) in enumerate(_oracle_cells(gt, dt)):
        oracle_cell = np.asarray(oracle[next(k2 for k2 in oracle if tuple(map(int, k2)) == key)])
        for d in range(len(dts)):
            arg = captured[cell_index * len(dts) + d]
            # Independent proof that the capture is aligned with the cell:
            # the oracle's own value has to fall out of its own argument.
            rebuilt = np.sum(np.exp(arg)) / float(arg.shape[0])
            assert _bits(np.array([rebuilt]))[0] == _bits(np.array([oracle_cell[d, column]]))[0], (
                f"capture misaligned at {key} (dt={d}, gt={column})"
            )
            recovered = terms.get((ann["id"], d))
            if recovered is None:  # k1 == 0: the F3 surrogate path
                continue
            assert recovered.size == arg.shape[0], (
                f"{key} (dt={d}, gt={column}): vernier summed {recovered.size} terms, "
                f"pycocotools summed {arg.shape[0]}"
            )
            expected = np.exp(arg)
            ulps = _ulps(recovered, expected)
            n_sensitive += int((-arg >= 4.0).sum())
            compared += recovered.size
            n_differ += int((ulps != 0).sum())
            worst = max(worst, int(ulps.max()))
            assert ulps.max() <= 1, (
                f"{key} (dt={d}, gt={column}): a term is {int(ulps.max())} ULP from "
                "pycocotools'. Two faithful `exp` kernels are at most 1 ULP apart, so "
                "this is a divergence in `e` itself -- the sigma table, the divisions, "
                "the area guard or the visible-keypoint mask -- not in `exp`."
            )

    assert compared >= 2000, f"only {compared} terms compared"
    assert n_sensitive >= compared // 10, (
        f"only {n_sensitive}/{compared} terms have e >= 4, where the 1-ULP bound "
        "detects a 1-ULP error in `e`; the assertion would be too loose to mean much"
    )
    if kernels_match:
        assert n_differ == 0, (
            f"the exp-kernel probe reports NumPy and vernier on the same kernel, so "
            f"every term must be bit-equal, but {n_differ}/{compared} differ "
            f"(max {worst} ULP) -- that is a divergence in `e`, not in `exp`"
        )


@pytest.mark.parity
def test_default_sigmas_keep_the_divide_by_ten_as_a_separate_rounding() -> None:
    """Quirk **F9** at the surface, with no arithmetic in between.

    `kpt_oks_sigmas` is `np.array([...]) / 10.0`, and that divide is a
    *second* rounding of already-rounded decimal literals. Five of the
    seventeen values land one ULP above the folded literal.
    """
    vernier = COCOeval(
        _coco({"images": [], "annotations": [], "categories": []}), iouType="keypoints"
    )
    oracle = (
        np.array(
            [
                0.26,
                0.25,
                0.25,
                0.35,
                0.35,
                0.79,
                0.79,
                0.72,
                0.72,
                0.62,
                0.62,
                1.07,
                1.07,
                0.87,
                0.87,
                0.89,
                0.89,
            ]
        )
        / 10.0
    )
    got = np.asarray(vernier.params.kpt_oks_sigmas, dtype=np.float64)
    assert got.shape == oracle.shape
    assert (_bits(got) == _bits(oracle)).all(), "default sigma table is not bit-equal to ce:523"

    folded = np.array(
        [
            0.026,
            0.025,
            0.025,
            0.035,
            0.035,
            0.079,
            0.079,
            0.072,
            0.072,
            0.062,
            0.062,
            0.107,
            0.107,
            0.087,
            0.087,
            0.089,
            0.089,
        ]
    )
    drift = sorted(int(i) for i in np.flatnonzero(_bits(folded) != _bits(oracle)))
    assert drift == [0, 3, 4, 11, 12], (
        f"the pre-folded literals now drift at {drift}; if this changed, the "
        "assertion above lost its teeth rather than the quirk going away"
    )


@pytest.mark.parity
def test_threshold_ladders_are_bit_equal_to_pycocotools() -> None:
    """`linspace` for `iouThrs` / `recThrs`, which gate every match.

    `np.linspace` is not `start + i * step`: it computes `step` once and
    then overwrites the endpoint, and the recall ladder differs from
    `i / 100` at ten indices. Bit-equality here is what makes the
    decision-level guarantee below meaningful -- a threshold that moved
    would flip matches on its own.

    This pins the drop-in's ladder; the Rust port that the native engine
    uses is pinned separately and unconditionally by
    `vernier_core::parity::tests::recall_thresholds_differ_from_i_over_100_at_ten_indices`.
    """
    empty = {"images": [], "annotations": [], "categories": []}
    got = COCOeval(_coco(empty), iouType="keypoints").params
    want = PycocotoolsReferenceCOCOeval(_coco(empty), iouType="keypoints").params
    for name in ("iouThrs", "recThrs"):
        mine = np.asarray(getattr(got, name), dtype=np.float64)
        theirs = np.asarray(getattr(want, name), dtype=np.float64)
        assert mine.shape == theirs.shape, name
        assert (_bits(mine) == _bits(theirs)).all(), f"{name} is not bit-equal to pycocotools'"


# ---------------------------------------------------------------------------
# 2. Conditional: the final OKS value, gated on the exp kernel
# ---------------------------------------------------------------------------


def _assert_ious_bit_equal(
    gt: dict[str, Any],
    dt: dict[str, Any],
    sigmas: tuple[float, ...] | None = None,
    *,
    min_cells: int = 100,
) -> None:
    _require_matching_exp_kernel()
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
        want_bits = _bits(want)
        got_bits = _bits(got)
        bad = np.flatnonzero(want_bits != got_bits)
        if bad.size:
            i = int(bad[0])
            ulps = _ulps(want, got)
            raise AssertionError(
                f"{key}: {bad.size}/{want.size} OKS cells are not bit-equal "
                f"(max {int(ulps.max())} ULP). First at flat index {i}: "
                f"vernier {got[i]!r} (0x{int(got_bits[i]):016x}) vs pycocotools "
                f"{want[i]!r} (0x{int(want_bits[i]):016x}). The exp-kernel probe "
                "reported both sides on the same kernel, so this is a real "
                "divergence and not the dispatch split this file gates on."
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
    guard, not for F8. It is also the one case that survives an `exp`
    kernel swap unscathed, for the same reason: every kernel underflows
    to the same exact zero.
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


# ---------------------------------------------------------------------------
# 3. Unconditional: the bound, and the decision it protects
# ---------------------------------------------------------------------------


@pytest.mark.parity
def test_divergence_stays_within_delta_and_flips_no_match() -> None:
    """What holds on *every* machine, including one where `exp` differs.

    Two statements, and they are deliberately not derived from one
    another -- a tolerance on OKS is not a tolerance on AP:

    * **The value.** `|vernier - pycocotools| <= OKS_DELTA` wherever the
      two differ. The bound is `2**-52` by the derivation on
      `OKS_DELTA`; `2**-46` is what is published, with 64x margin.
    * **The decision.** Matching gates on `iou >= t` (quirk **B2**) over
      the 10-threshold ladder, and any `delta > 0` can in principle flip
      one. This counts every `(cell, threshold)` comparison where the two
      sides land on opposite sides and asserts zero.

    Zero flips is a measurement, not a proof. Across this suite and the
    30 000-cell kernel-swap experiment behind `OKS_DELTA`, 0 flips in
    ~1.3M comparisons bounds the rate at roughly **2e-6** per comparison
    by the rule of three -- not at 0. The cases that sit exactly on a
    ladder threshold are, helpfully, the ones where every kernel agrees
    exactly: `exp(0)` is `1.0` and large `e` underflows to `0.0` in any
    kernel, and sums of exact `1.0`s are exact in any order.
    """
    thresholds = np.asarray(
        COCOeval(
            _coco({"images": [], "annotations": [], "categories": []}), iouType="keypoints"
        ).params.iouThrs,
        dtype=np.float64,
    )
    datasets = [
        (_build(K_PERSON, seed=1), None),
        (_build(K_PERSON, seed=2, visibility="sweep"), None),
        (_build(K_PERSON, seed=3, visibility="none"), None),
        (_build(K_PERSON, seed=4, zero_area=True), None),
        (_build(K_WHOLEBODY, seed=5, visibility="sweep", n_images=6), _WHOLEBODY_SIGMAS),
    ]

    cells = differing = flips = 0
    worst = 0.0
    for (gt, dt), sigmas in datasets:
        reference = _ious(PycocotoolsReferenceCOCOeval, gt, dt, sigmas)
        candidate = _ious(COCOeval, gt, dt, sigmas)
        for key, expected in reference.items():
            want = np.asarray(expected, dtype=np.float64).ravel()
            got = np.asarray(candidate[key], dtype=np.float64).ravel()
            if want.size == 0:
                continue
            cells += want.size
            delta = np.abs(got - want)
            differing += int((_bits(want) != _bits(got)).sum())
            worst = max(worst, float(delta.max()))
            over = np.flatnonzero(delta > OKS_DELTA)
            assert over.size == 0, (
                f"{key}: {over.size} cells exceed delta = 2**-46; worst "
                f"{float(delta[over].max()):.3e} at flat index {int(over[0])} "
                f"(vernier {got[over[0]]!r}, pycocotools {want[over[0]]!r})"
            )
            flips += int(
                (
                    (got[:, None] >= thresholds[None, :]) != (want[:, None] >= thresholds[None, :])
                ).sum()
            )

    comparisons = cells * thresholds.size
    assert cells >= 2000, f"only {cells} cells compared"
    assert flips == 0, (
        f"{flips} of {comparisons} `iou >= t` comparisons land on opposite sides of a "
        f"threshold ({differing} cells differ, worst |delta| {worst:.3e}). A flipped "
        "match changes AP discontinuously and can cascade through the greedy "
        "assignment, so this is a finding, not a tolerance to widen."
    )
