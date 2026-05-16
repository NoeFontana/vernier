"""ADR-0047 Stage B cross-thread parity for the semantic paradigm.

Pins the load-bearing claim of ADR-0047 §"Semantic": the per-image
confusion-matrix fold is u64-additive and commutative, so
``Evaluator.evaluate(..., num_threads=N)`` produces a confusion matrix
bit-equal to ``num_threads=None`` for every ``N`` and every fixture.

The kernel walks at native dtype (ADR-0037) — parametrising over
``uint8`` / ``uint16`` / ``uint32`` also catches a per-dtype
monomorphisation regression that the thread axis alone would miss.

Mirrors :func:`tests.python.parity.test_parity.test_parity_across_thread_counts_strict_bit_equal`
on the instance side; the test asserts vernier-vs-vernier bit-equality
on the integer ``ConfusionMatrix.counts()`` totals (the strict-mode
parity surface — derived float scalars follow trivially).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import vernier.semantic as vsem

# Per ADR-0047 §"Test plan", every existing fixture runs at
# `num_threads ∈ {None, 1, 2, 4, 8}` and asserts bit-equality to the
# sequential (`num_threads=None`) baseline. `None` exercises today's
# default-path code; `1` pins the explicit-sequential shape; `2/4/8`
# fan out through the scoped rayon pool under different scheduling
# regimes.
ADR_0047_THREAD_COUNTS: tuple[int | None, ...] = (None, 1, 2, 4, 8)


def _make_fixture(
    rng: np.random.Generator,
    n_images: int,
    height: int,
    width: int,
    ignore_label: int,
) -> dict[int, np.ndarray]:
    """Generate a deterministic per-image label-map fixture in [0, 4)
    with ~5% ignore-label pixels."""
    out: dict[int, np.ndarray] = {}
    for iid in range(n_images):
        arr = rng.integers(0, 4, size=(height, width), dtype=np.uint8)
        mask = rng.random(size=(height, width)) < 0.05
        arr[mask] = ignore_label
        out[iid] = arr
    return out


@pytest.mark.parity_threads
@pytest.mark.parity_semantic
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32])
@pytest.mark.parametrize("num_threads", ADR_0047_THREAD_COUNTS)
def test_evaluate_bit_equal_across_thread_counts(dtype: type, num_threads: int | None) -> None:
    """``Evaluator.evaluate(..., num_threads=N)`` produces a confusion
    matrix bit-equal to ``num_threads=None`` for every ``N`` and every
    natural class-id dtype (ADR-0047 §"Semantic")."""
    rng = np.random.default_rng(0xADAB0475EA1)
    n_classes = 4
    ignore_label = 255

    gt_u8 = _make_fixture(rng, n_images=24, height=18, width=22, ignore_label=ignore_label)
    dt_u8 = _make_fixture(rng, n_images=24, height=18, width=22, ignore_label=ignore_label)
    gt = {iid: arr.astype(dtype, copy=False) for iid, arr in gt_u8.items()}
    dt = {iid: arr.astype(dtype, copy=False) for iid, arr in dt_u8.items()}

    evaluator = vsem.Evaluator(parity_mode="strict")
    baseline = evaluator.evaluate(
        vsem.Dataset.from_arrays(gt, n_classes=n_classes, ignore_label=ignore_label),
        vsem.Predictions.from_arrays(dt),
        num_threads=None,
    )
    candidate = evaluator.evaluate(
        vsem.Dataset.from_arrays(gt, n_classes=n_classes, ignore_label=ignore_label),
        vsem.Predictions.from_arrays(dt),
        num_threads=num_threads,
    )
    np.testing.assert_array_equal(
        candidate.confusion_matrix.counts(),
        baseline.confusion_matrix.counts(),
        err_msg=(
            f"semantic confusion matrix must be bit-equal across thread counts: "
            f"dtype={dtype.__name__}, num_threads={num_threads!r}"
        ),
    )
    # Headline scalars are derived from the same u64 totals via the
    # same float arithmetic; bit-equality follows by construction but
    # we pin it explicitly so a future regression that decouples the
    # derivation from the totals trips this assertion first.
    assert candidate.miou == baseline.miou
    assert candidate.fwiou == baseline.fwiou
    assert candidate.pixel_accuracy == baseline.pixel_accuracy
    assert candidate.mean_accuracy == baseline.mean_accuracy


@pytest.mark.parity_threads
@pytest.mark.parity_semantic
@pytest.mark.parametrize("num_threads", ADR_0047_THREAD_COUNTS)
def test_evaluate_from_pngs_bit_equal_across_thread_counts(
    tmp_path: Path, num_threads: int | None
) -> None:
    """The fused libpng decode + fold path (ADR-0037) carries the same
    cross-thread bit-equality property as the array path."""
    from PIL import Image

    rng = np.random.default_rng(0xADAB047)
    n_classes = 4
    ignore_label = 255

    arrays_gt = _make_fixture(rng, n_images=12, height=16, width=20, ignore_label=ignore_label)
    arrays_dt = _make_fixture(rng, n_images=12, height=16, width=20, ignore_label=ignore_label)

    gt_dir = tmp_path / "gt"
    dt_dir = tmp_path / "dt"
    gt_dir.mkdir()
    dt_dir.mkdir()
    gt_paths: dict[int, Path] = {}
    dt_paths: dict[int, Path] = {}
    for iid, arr in arrays_gt.items():
        p = gt_dir / f"{iid}.png"
        Image.fromarray(arr, mode="L").save(p, format="PNG")
        gt_paths[iid] = p
    for iid, arr in arrays_dt.items():
        p = dt_dir / f"{iid}.png"
        Image.fromarray(arr, mode="L").save(p, format="PNG")
        dt_paths[iid] = p

    evaluator = vsem.Evaluator(parity_mode="strict")
    baseline = evaluator.evaluate_from_pngs(
        gt_paths, dt_paths, n_classes=n_classes, ignore_label=ignore_label, num_threads=None
    )
    candidate = evaluator.evaluate_from_pngs(
        gt_paths,
        dt_paths,
        n_classes=n_classes,
        ignore_label=ignore_label,
        num_threads=num_threads,
    )
    np.testing.assert_array_equal(
        candidate.confusion_matrix.counts(),
        baseline.confusion_matrix.counts(),
        err_msg=(
            f"evaluate_from_pngs confusion matrix must be bit-equal across "
            f"thread counts: num_threads={num_threads!r}"
        ),
    )


@pytest.mark.parity_threads
@pytest.mark.parity_semantic
@pytest.mark.parametrize("num_threads", ADR_0047_THREAD_COUNTS)
def test_evaluate_to_partial_bit_equal_across_thread_counts(
    num_threads: int | None,
) -> None:
    """The streaming-to-partial path (ADR-0035) carries the same
    cross-thread bit-equality property as the batch path: a partial
    serialised with ``num_threads=N`` produces a summary bit-equal to
    one serialised sequentially."""
    rng = np.random.default_rng(0xADAB0475707)
    n_classes = 4
    ignore_label = 255

    gt = _make_fixture(rng, n_images=16, height=14, width=18, ignore_label=ignore_label)
    dt = _make_fixture(rng, n_images=16, height=14, width=18, ignore_label=ignore_label)
    dataset = vsem.Dataset.from_arrays(
        {iid: arr.astype(np.uint32, copy=False) for iid, arr in gt.items()},
        n_classes=n_classes,
        ignore_label=ignore_label,
    )
    predictions = vsem.Predictions.from_arrays(
        {iid: arr.astype(np.uint32, copy=False) for iid, arr in dt.items()}
    )

    evaluator = vsem.Evaluator(parity_mode="strict")
    baseline_bytes = evaluator.evaluate_to_partial(
        dataset, predictions, rank_id=0, num_threads=None
    )
    candidate_bytes = evaluator.evaluate_to_partial(
        dataset, predictions, rank_id=0, num_threads=num_threads
    )
    baseline_summary = vsem.Evaluator.from_partials(
        n_classes, [baseline_bytes], parity_mode="strict", ignore_label=ignore_label
    )
    candidate_summary = vsem.Evaluator.from_partials(
        n_classes, [candidate_bytes], parity_mode="strict", ignore_label=ignore_label
    )
    np.testing.assert_array_equal(
        candidate_summary.confusion_matrix.counts(),
        baseline_summary.confusion_matrix.counts(),
        err_msg=(
            f"evaluate_to_partial confusion matrix must be bit-equal across "
            f"thread counts: num_threads={num_threads!r}"
        ),
    )
