#!/usr/bin/env python3
"""Regenerate the golden fixture for `parity.rs::numpy_pairwise_sum`.

`np.sum` on a contiguous ``float64`` buffer is not a left fold: it is
``DOUBLE_pairwise_sum``. vernier's OKS kernel reduces over one term per
visible keypoint and has to match it bit-for-bit, so the Rust port is
pinned against values produced *here*, by the real NumPy.

Two artefacts are emitted, both consumed by the unit tests in
``crates/vernier-core/src/parity.rs``:

``GOLDEN_PREFIX_SUM_BITS``
    One fixed 133-element input (also committed, as bit patterns) and
    the expected ``np.sum`` of every prefix ``n in 1..=133``. Directly
    readable in review, and the shape a human can re-derive by hand.

``RANDOM_SWEEP_DIGEST``
    An exhaustive sweep: for every ``n in 1..=133``, ``REPS`` random
    vectors, each summed and folded into an FNV-1a digest of the result
    bit patterns. Committing the digest rather than 1.33M u64s keeps the
    fixture reviewable while still pinning every single sum. The Rust
    test regenerates the identical vectors from the identical PRNG, so
    a digest mismatch means a real divergence -- either in the port or
    in NumPy.

**If re-running this script changes the committed numbers, that is a
finding, not a rebase.** The pinned oracle (`pycocotools==2.0.11`) is
evaluated by whatever NumPy is installed alongside it; a moved digest
means NumPy changed its reduction tree, so the parity target moved and
the NumPy version needs stating in the parity contract before the new
numbers are accepted.

Run:  ``uv run python tools/gen_numpy_pairwise_sum_golden.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "crates/vernier-core/src/golden/numpy_pairwise_sum_golden.rs"

#: Longest reduction `computeOks` can perform: COCO-WholeBody's 133
#: keypoints. 133 > NPY_PW_BLOCKSIZE (128), so it reaches the recursion.
MAX_N = 133

#: Random vectors per `n` in the exhaustive sweep.
REPS = 10_000

SEED_BASE = 0x0123456789ABCDEF
INPUT_SEED = 0xFEDCBA9876543210

U64 = np.uint64
MASK = np.uint64(0xFFFFFFFFFFFFFFFF)
GAMMA = np.uint64(0x9E3779B97F4A7C15)
M1 = np.uint64(0xBF58476D1CE4E5B9)
M2 = np.uint64(0x94D049BB133111EB)

#: Biased exponents drawn from by `bits_to_f64`, one per 4 PRNG bits.
#:
#: Weighted 11/16 into a narrow band around 1.0 and 5/16 onto extremes
#: (subnormal, 2^-1022, 2^-323, 2^17, 2^277). The narrow band is what
#: makes the fixture *discriminate*: terms of comparable magnitude with
#: mixed signs are exactly where a reassociated sum lands on a different
#: double, whereas a vector whose largest term dwarfs the rest sums to
#: the same bits in any order. The extremes keep subnormals, near-
#: cancellation across binades and signed zeros in the sweep.
#:
#: The top of the range stops well short of 2046 on purpose: 133 terms
#: in the largest binade would overflow to inf and `inf + -inf` would
#: put a NaN in the fixture, whose payload is not something to pin
#: across architectures (CI runs the Rust suite on x86_64 *and*
#: aarch64). Every generated value, and every sum of them, is finite.
EXPONENTS = [
    0,
    1,
    700,
    1010,
    1018,
    1020,
    1021,
    1022,
    1023,
    1023,
    1024,
    1025,
    1026,
    1030,
    1040,
    1300,
]


def splitmix64(seed: int, count: int) -> np.ndarray:
    """`count` SplitMix64 outputs, as a uint64 array.

    SplitMix64's state is `seed + i * GAMMA`, so the whole stream is
    addressable in closed form and vectorises. The Rust side steps it
    one call at a time; both produce the same sequence.
    """
    i = np.arange(1, count + 1, dtype=U64)
    state = (U64(seed & 0xFFFFFFFFFFFFFFFF) + i * GAMMA) & MASK
    z = state
    z = ((z ^ (z >> U64(30))) * M1) & MASK
    z = ((z ^ (z >> U64(27))) * M2) & MASK
    return (z ^ (z >> U64(31))) & MASK


#: Exponents for the *fixed* golden input. Deliberately narrower than
#: `EXPONENTS`: one term of vastly larger magnitude swamps the rest and
#: makes the prefix sum order-independent, which would leave the
#: committed fixture unable to tell a pairwise sum from a left fold.
#: Comparable magnitudes plus mixed signs is the discriminating regime;
#: the exotic values in this vector are hand-placed instead (see
#: `main`), so the fixture covers them without losing its teeth.
GOLDEN_EXPONENTS = [
    1020,
    1021,
    1022,
    1022,
    1023,
    1023,
    1023,
    1023,
    1023,
    1023,
    1023,
    1024,
    1024,
    1024,
    1025,
    1026,
]


def bits_to_f64(u: np.ndarray, exponents: list[int] = EXPONENTS) -> np.ndarray:
    """Map raw PRNG words onto adversarially-spread finite float64s."""
    exps = np.array(exponents, dtype=U64)
    sign = (u >> U64(63)) & U64(1)
    idx = (u >> U64(59)) & U64(0xF)
    mant = u & U64(0x000FFFFFFFFFFFFF)
    biased = exps[idx.astype(np.intp)]
    bits = (sign << U64(63)) | (biased << U64(52)) | mant
    return bits.view(np.float64)


def fnv1a(values: np.ndarray) -> int:
    """FNV-1a over the little-endian bytes of each u64 in `values`."""
    h = 0xCBF29CE484222325
    for word in values.tolist():
        w = int(word)
        for shift in range(0, 64, 8):
            h ^= (w >> shift) & 0xFF
            h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def naive_left_fold(row: np.ndarray) -> float:
    acc = 0.0
    for x in row.tolist():
        acc += x
    return acc


def main() -> int:
    # Sanity: a row-wise reduction over the last (contiguous) axis must
    # be bit-identical to summing each row on its own, or the sweep
    # below would be pinning the wrong reduction tree.
    probe = bits_to_f64(splitmix64(1, 7 * 133)).reshape(7, 133)
    rowwise = probe.sum(axis=1)
    for r in range(7):
        assert rowwise[r].view(np.uint64) == np.sum(probe[r]).view(np.uint64), (
            "arr.sum(axis=1) is not bit-identical to np.sum(arr[r]); "
            "regenerate with an explicit per-row loop"
        )

    # --- fixed golden input -------------------------------------------------
    golden_input = bits_to_f64(splitmix64(INPUT_SEED, MAX_N), GOLDEN_EXPONENTS).copy()
    # Hand-placed specials, so the committed fixture provably covers the
    # cases a uniform draw would only reach by luck: both signed zeros
    # (numpy's reduction seed normalises -0.0 away, and only the seed
    # does), the smallest subnormal, and a value 2^-1000 below its
    # neighbours. Positions are spread across the block boundary at 8
    # and the recursion split at 64.
    golden_input[0] = 0.0
    golden_input[7] = -0.0
    golden_input[8] = -0.0
    golden_input[63] = 5e-324
    golden_input[64] = float.fromhex("0x1p-1000")
    golden_input[132] = -golden_input[131]

    prefix_sums = np.empty(MAX_N, dtype=np.float64)
    discriminating = 0
    for n in range(1, MAX_N + 1):
        row = np.ascontiguousarray(golden_input[:n])
        prefix_sums[n - 1] = np.sum(row)
        assert np.isfinite(prefix_sums[n - 1]), f"prefix n={n} is not finite"
        if prefix_sums[n - 1].view(np.uint64) != np.float64(naive_left_fold(row)).view(np.uint64):
            discriminating += 1

    # --- exhaustive random sweep -------------------------------------------
    digests: list[int] = []
    first_sums: list[int] = []
    sweep_discriminating = 0
    sweep_total = 0
    for n in range(1, MAX_N + 1):
        raw = splitmix64(SEED_BASE ^ n, n * REPS)
        rows = bits_to_f64(raw).reshape(REPS, n)
        sums = rows.sum(axis=1)
        assert np.isfinite(sums).all(), f"n={n}: sweep produced a non-finite sum"
        bits = sums.view(np.uint64)
        digests.append(fnv1a(bits))
        first_sums.append(int(bits[0]))
        # How often the sweep would catch a left fold, sampled cheaply.
        for r in range(0, REPS, 500):
            sweep_total += 1
            if bits[r] != np.float64(naive_left_fold(rows[r])).view(np.uint64):
                sweep_discriminating += 1

    lines: list[str] = []
    w = lines.append
    w("// @generated by tools/gen_numpy_pairwise_sum_golden.py -- do not edit by hand.")
    w("//")
    w("// Golden values for `parity::numpy_pairwise_sum`, produced by the real")
    w(f"// NumPy {np.__version__} on Python {'.'.join(map(str, sys.version_info[:3]))}.")
    w("//")
    w("// If regenerating this file changes a number, NumPy's reduction tree moved:")
    w("// the parity target changed and that is a finding to report, not a rebase")
    w("// to accept. See the module doc on the generator.")
    w("")
    w(f'pub(super) const GOLDEN_NUMPY_VERSION: &str = "{np.__version__}";')
    w(f"pub(super) const MAX_N: usize = {MAX_N};")
    w(f"pub(super) const REPS: usize = {REPS};")
    w(f"pub(super) const SEED_BASE: u64 = 0x{SEED_BASE:016X};")
    w("/// Provenance only: the Rust side reads `GOLDEN_INPUT_BITS` directly")
    w("/// rather than regenerating the vector, so this seed is recorded, not used.")
    w("#[allow(dead_code)]")
    w(f"pub(super) const INPUT_SEED: u64 = 0x{INPUT_SEED:016X};")
    w("/// How many of the 133 prefixes a plain left fold gets *wrong*. The")
    w("/// fixture is only worth committing if it discriminates; this pins that.")
    w(f"pub(super) const DISCRIMINATING_PREFIXES: usize = {discriminating};")
    w("")

    def emit(name: str, values: list[int], doc: str) -> None:
        for line in doc.splitlines():
            w(f"/// {line}".rstrip())
        w(f"pub(super) const {name}: [u64; {len(values)}] = [")
        for start in range(0, len(values), 4):
            chunk = values[start : start + 4]
            w("    " + " ".join(f"0x{v:016X}," for v in chunk))
        w("];")
        w("")

    emit(
        "SWEEP_EXPONENTS",
        EXPONENTS,
        "Biased exponents the sweep's value generator draws from, emitted\n"
        "here so the Rust and Python halves of the PRNG cannot drift apart.",
    )
    emit(
        "GOLDEN_INPUT_BITS",
        [int(v) for v in golden_input.view(np.uint64)],
        "The fixed 133-element input, as raw bit patterns so no decimal\n"
        "round-trip can smuggle in a 1-ULP change.",
    )
    emit(
        "GOLDEN_PREFIX_SUM_BITS",
        [int(v) for v in prefix_sums.view(np.uint64)],
        "`np.sum(GOLDEN_INPUT[..n])` for n = 1..=133, indexed by `n - 1`.",
    )
    emit(
        "RANDOM_SWEEP_DIGEST",
        digests,
        f"FNV-1a over the sum bit patterns of {REPS} random vectors per n.",
    )
    emit(
        "RANDOM_SWEEP_FIRST_SUM_BITS",
        first_sums,
        "The first vector's sum for each n -- a concrete value to print\n"
        "when the digest above fails, so the failure is debuggable.",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)}")
    print(f"  numpy {np.__version__}; {discriminating}/{MAX_N} prefixes discriminate a left fold")
    print(f"  sweep: {sweep_discriminating}/{sweep_total} sampled vectors discriminate a left fold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
