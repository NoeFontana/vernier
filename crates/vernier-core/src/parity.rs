//! Parity-mode flag and pinned numerical constants.
//!
//! This module is the single home for every numerical constant and
//! algorithmic primitive that vernier needs to reproduce bit-exactly to
//! match `pycocotools` 2.0.11. Each item is doc-tagged with the quirk ID
//! from `docs/engineering/pycocotools-quirks.md` it corresponds to, and
//! with the ADR that ratifies its choice.
//!
//! The whole file is load-bearing for the parity contract from ADR-0002.
//! Changes here ripple through every algorithm crate and require their
//! own ADR.

use std::cmp::Ordering;
use std::sync::OnceLock;

/// Parity mode (per ADR-0002, amended 2026-05-10).
///
/// Picks which disposition vernier honors for each row of the
/// pycocotools quirks survey. `Strict` is the canonical migration path
/// for users with downstream tooling calibrated to pycocotools' exact
/// numerical behavior; `Corrected` is the default for net-new users.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParityMode {
    /// Reproduce every pycocotools behavior bit-exactly, including known
    /// bugs (D1's overwritten ignore field, H2's silent merge, …). The
    /// default when migrating from pycocotools.
    Strict,
    /// Apply opinionated fixes for behaviors classified `corrected` in
    /// the disposition table. Default for net-new users; opt-out via
    /// `Strict`.
    #[default]
    Corrected,
}

/// Substitute for `numpy.spacing(1)` in pycocotools' precision/recall
/// arithmetic. (Quirk **C8** — strict.) On every supported platform
/// `f64::EPSILON == 2.220446049250313e-16`, identical to `np.spacing(1)`
/// to all bits.
pub const PARITY_EPS: f64 = f64::EPSILON;

/// Tolerance fudge in the matching loop's initial best-IoU seed. Used as
/// `min(threshold, 1.0 - IOU_BOUNDARY_EPS)` so a detection with IoU
/// exactly at the threshold still matches. (Quirk **B1** — strict.)
pub const IOU_BOUNDARY_EPS: f64 = 1e-10;

/// The 10 IoU thresholds at which COCO eval reports AP. Built via the
/// same `linspace(0.5, 0.95, 10)` formula pycocotools uses (not
/// `arange`, which accumulates float error). (Quirk **L1** — strict.)
pub fn iou_thresholds() -> &'static [f64] {
    static IOU_THRESHOLDS: OnceLock<Vec<f64>> = OnceLock::new();
    IOU_THRESHOLDS.get_or_init(|| linspace(0.5, 0.95, 10))
}

/// The 101 recall thresholds used for AP integration. Built via
/// `linspace(0.0, 1.0, 101)`. (Quirk **L2**, **C1** — strict.)
pub fn recall_thresholds() -> &'static [f64] {
    static RECALL_THRESHOLDS: OnceLock<Vec<f64>> = OnceLock::new();
    RECALL_THRESHOLDS.get_or_init(|| linspace(0.0, 1.0, 101))
}

/// Quirk **P1** — strict. numpy quantile method `'linear'` for
/// calibration bin-edges (ADR-0018; see
/// `docs/engineering/calibration-quirks.md`).
///
/// Pinned as a string for documentation parity with numpy's `method=`
/// kwarg; the actual computation is implemented in the crate-private
/// `quantile_linear`. Recorded here so any drift away from the pinned
/// method shows up as a constant-rename in code review.
pub const CALIBRATION_QUANTILE_METHOD: &str = "linear";

/// Linear-interpolation quantile, bit-equivalent to
/// `numpy.quantile(values, q, method='linear')` (Quirk **P1** — strict).
///
/// Used by the calibration summarizer (ADR-0018) to derive bin-edges
/// from the score distribution. The caller is responsible for sorting
/// `sorted_values` ascending; a debug-only assertion guards the
/// invariant.
///
/// # Algorithm
///
/// For each `q` in `qs`, compute `pos = q * (n - 1)`, then
/// `lo = floor(pos)`, `hi = ceil(pos)`, `frac = pos - lo`. The result
/// is `values[lo] + (values[hi] - values[lo]) * frac`. This matches
/// numpy's `'linear'` interpolation rule precisely.
///
/// # Edge cases
///
/// - Empty `sorted_values` returns an empty output (no quantiles to
///   interpolate). Callers must guard against this case explicitly if
///   it carries domain meaning.
/// - `qs` values outside `[0, 1]` are not validated; the caller is
///   expected to pass values produced by `linspace(0.0, 1.0, n+1)` or
///   similar.
pub(crate) fn quantile_linear(sorted_values: &[f64], qs: &[f64]) -> Vec<f64> {
    if sorted_values.is_empty() {
        return Vec::new();
    }
    debug_assert!(
        sorted_values.windows(2).all(|w| w[0] <= w[1]),
        "quantile_linear: sorted_values must be ascending"
    );
    let n = sorted_values.len();
    if n == 1 {
        let only = sorted_values[0];
        return qs.iter().map(|_| only).collect();
    }
    let last = n - 1;
    let mut out = Vec::with_capacity(qs.len());
    for &q in qs {
        let pos = q * (last as f64);
        let lo_idx = pos.floor() as usize;
        let hi_idx = pos.ceil() as usize;
        // Clamp defensively in case of floating drift; with q in [0,1]
        // the indices are already within bounds, but the clamp keeps
        // the function total without panicking on out-of-domain inputs.
        let lo_idx = lo_idx.min(last);
        let hi_idx = hi_idx.min(last);
        let frac = pos - (lo_idx as f64);
        let lo = sorted_values[lo_idx];
        let hi = sorted_values[hi_idx];
        out.push(lo + (hi - lo) * frac);
    }
    out
}

/// Stable score-descending argsort. (Quirk **A1** — strict.)
///
/// Mirrors `np.argsort(-scores, kind='mergesort')`: the returned
/// permutation indexes `scores` such that `scores[perm[0]] >=
/// scores[perm[1]] >= ...`, with ties resolved to the input order.
/// Used by the matching engine for per-image DT ordering and by the
/// accumulator for the merged-stream re-sort across images.
pub fn argsort_score_desc(scores: &[f64]) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..scores.len()).collect();
    // `slice::sort_by` is a *stable* sort (the unstable variant is
    // `sort_unstable_by`), so equal scores keep their input order —
    // this is exactly numpy's `kind='mergesort'` tie behavior (A1).
    perm.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(Ordering::Equal));
    perm
}

/// Block size below which numpy stops halving and switches to its
/// 8-accumulator unrolled loop. `NPY_PW_BLOCKSIZE` in
/// `numpy/_core/src/umath/loops_utils.h`; 128 since the pairwise
/// reduction was introduced and unchanged through numpy 2.4.
const PAIRWISE_BLOCKSIZE: usize = 128;

/// `numpy.sum` over a contiguous `float64` array, bit-exactly.
/// (Quirk **F8** — strict.)
///
/// `np.sum` is **not** a left fold. `np.add.reduce` on a stride-1
/// `float64` buffer dispatches to `DOUBLE_pairwise_sum`
/// (`numpy/_core/src/umath/loops_arithm_fp.dispatch.c.src`), which:
///
/// 1. left-folds from `+0.0` for fewer than 8 elements;
/// 2. for 8..=128 elements, runs **eight independent accumulators**
///    seeded with `a[0..8]`, strides them by 8, combines them as
///    `((r0+r1) + (r2+r3)) + ((r4+r5) + (r6+r7))`, then left-folds the
///    `n % 8` tail;
/// 3. above 128, splits at `n/2` rounded *down* to a multiple of 8 and
///    recurses on both halves.
///
/// Step 2's eight accumulators are not an optimisation we could drop:
/// numpy's own source comment says the unroll exists so that "the
/// autovectorizer" cannot change the summation order, which is what
/// makes this part of the reference **dispatch-invariant** — unlike
/// `exp`, every numpy build on every CPU sums in this exact order. That
/// is what makes it portable to pin, and worth pinning.
///
/// Step 3 is reachable in production, not theoretical: `computeOks`
/// reduces over one term per *visible* keypoint, so `n` is 17 for
/// COCO-person but **133** for COCO-WholeBody, which recurses into
/// `64 + 69`.
///
/// Rust will not reassociate these adds on its own — `llvm.fadd`
/// without the `reassoc` flag, no fast-math anywhere in the build — so
/// the port holds today by default. It holds *silently*, which is why
/// [`tests::numpy_pairwise_sum_golden_bits`] pins the result as bit
/// patterns rather than merely comparing against a second Rust
/// expression: a future `#[target_feature]` wrapper or a fast-math
/// crate would otherwise change the answer with nothing to catch it.
///
/// # One canonical reduction
///
/// This is the crate's only numpy-*order* `f64` reduction, which is
/// not the same claim as "no sequential `f64` fold is left in the
/// crate". Two folds remain, both in [`crate::tide`]: the per-`(t, k)`
/// precision mean (`s += v` over the 101 recall points) and the final
/// `ap_values.iter().sum()`. Both are left folds where the oracle does
/// a numpy mean, and both stay that way deliberately — ADR-0021
/// contracts TIDE to agree with its oracle within `1e-9`, not
/// bit-exactly, so the reassociation sits inside that contract. The
/// rule is therefore about paths, not about the crate as a whole:
/// anything on a **bit-exact** path reduces through this function.
///
/// It absorbed
/// `summarize::pairwise_sum`, which had the same three arms and served
/// [`crate::summarize`]'s `np.mean(s[s>-1])`, [`crate::tables`]' per-row
/// means and [`crate::calibration`]'s per-bin sums. That copy was
/// correct on the arms it documented but **missing the reduction's
/// identity seed**, so it returned `-0.0` where `np.sum` returns
/// `+0.0`. Its callers only ever sum non-negative values, so the gap
/// was unreachable for them — but it was a live bug waiting for the
/// next caller, and exactly the kind of drift a second copy invites.
/// Its numpy pin survives as
/// [`tests::numpy_pairwise_sum_matches_add_reduce_on_the_summary_reduction_pin`].
pub(crate) fn numpy_pairwise_sum(a: &[f64]) -> f64 {
    // `np.add.reduce` seeds its accumulator with `add`'s identity and
    // adds the pairwise total to it: `0.0 + DOUBLE_pairwise_sum(...)`.
    // For every value but one that seed is invisible; the exception is
    // `-0.0`, which numpy therefore reports as `+0.0`. `np.sum` of
    // eight `-0.0`s is `0.0`, while the block loop alone yields `-0.0`.
    0.0 + pairwise_sum_inner(a)
}

/// `DOUBLE_pairwise_sum` proper, without the reduction's identity seed.
fn pairwise_sum_inner(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        // numpy initialises to +0.0 and left-folds.
        let mut res = 0.0_f64;
        for &x in a {
            res += x;
        }
        res
    } else if n <= PAIRWISE_BLOCKSIZE {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let stop = n - (n % 8);
        let mut i = 8;
        while i < stop {
            r[0] += a[i];
            r[1] += a[i + 1];
            r[2] += a[i + 2];
            r[3] += a[i + 3];
            r[4] += a[i + 4];
            r[5] += a[i + 5];
            r[6] += a[i + 6];
            r[7] += a[i + 7];
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum_inner(&a[..n2]) + pairwise_sum_inner(&a[n2..])
    }
}

/// Reproduces `numpy.linspace(start, stop, num, endpoint=True)`.
///
/// Numpy's algorithm computes `step = (stop - start) / (num - 1)` once
/// (in f64) and emits `start + i * step` for `i in 0..num`, with the
/// final element snapped to `stop` exactly. This implementation follows
/// the same shape, so the resulting array is bit-equal to numpy's
/// across the platforms we target.
pub(crate) fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    if num == 0 {
        return Vec::new();
    }
    if num == 1 {
        return vec![start];
    }
    let last = num - 1;
    let step = (stop - start) / (last as f64);
    let mut out = Vec::with_capacity(num);
    for i in 0..last {
        out.push(start + (i as f64) * step);
    }
    out.push(stop);
    out
}

#[cfg(test)]
mod tests {

    /// JSON float parsing must be correctly rounded — bit-equal to
    /// CPython's `strtod`-based `json` module, not merely close.
    ///
    /// `serde_json`'s default parser is not: it rounds some near-tie
    /// decimals to the adjacent double. That surfaced as ~16 % of
    /// `eval_imgs.dtScores` drifting by exactly 1 ULP on the DETR-R50
    /// real-prediction gate (`docs/engineering/real-predictions-parity.md`),
    /// which held `dtScores` and the `scores` tensor under that page's
    /// float-tolerance gate while the summary stayed bit-equal — AP
    /// depends on detection *order*, and 1 ULP does not reorder.
    ///
    /// The `float_roundtrip` feature switches `serde_json` to a
    /// correctly-rounded path. Rust's own `str::parse::<f64>` is
    /// correctly rounded too, so it stands in for the oracle here;
    /// `tests/python/` pins the same values against CPython directly.
    #[test]
    fn json_float_parsing_is_correctly_rounded() {
        // The first entry is the value named in the parity doc as the
        // one that drifted on real DETR-R50 output.
        for literal in [
            "0.9992794394493103",
            "0.9999999999999999",
            "2.2250738585072011e-308",
            "0.05",
            "0.1",
            "0.30000000000000004",
            "1e-7",
            "8.98846567431158e307",
            "123456789.123456789",
            "0.000244140625",
        ] {
            let parsed: f64 = serde_json::from_str(literal).expect("serde_json parse");
            let oracle: f64 = literal.parse().expect("std parse");
            assert_eq!(
                parsed.to_bits(),
                oracle.to_bits(),
                "{literal}: serde_json {parsed:?} != correctly-rounded {oracle:?}"
            );
        }
    }
    use super::*;

    #[test]
    fn parity_eps_matches_numpy_spacing_1() {
        // np.spacing(1) == 2.220446049250313e-16 on every platform we
        // support; equal to f64::EPSILON to all bits.
        assert_eq!(PARITY_EPS, 2.220446049250313e-16);
    }

    #[test]
    fn iou_boundary_eps_is_1e_neg_10() {
        // pycocotools/cocoeval.py:276 — `min(t, 1 - 1e-10)` seeds the
        // initial best-IoU so detections at exactly the threshold match.
        assert_eq!(IOU_BOUNDARY_EPS, 1e-10);
    }

    #[test]
    fn iou_thresholds_match_numpy_linspace() {
        // Ground truth captured from `numpy.linspace(0.5, 0.95, 10)`
        // (NumPy 2.0). Index 8 is `0.8999999999999999` — one ulp below
        // 0.9 — and is what every pycocotools install on every supported
        // platform actually receives. Pinning the bit pattern is the
        // entire point of quirk **L1**.
        let expected_bits: [u64; 10] = [
            4602678819172646912, // 0.5
            4603129179135383962, // 0.55
            4603579539098121011, // 0.6
            4604029899060858061, // 0.65
            4604480259023595110, // 0.7
            4604930618986332160, // 0.75
            4605380978949069210, // 0.8
            4605831338911806259, // 0.85
            4606281698874543308, // 0.8999999999999999
            4606732058837280358, // 0.95 (snapped)
        ];
        let got = iou_thresholds();
        assert_eq!(got.len(), expected_bits.len());
        for (i, (g, e)) in got.iter().zip(expected_bits.iter()).enumerate() {
            assert_eq!(
                g.to_bits(),
                *e,
                "iouThr[{i}] differs: got bits {} ({:e})",
                g.to_bits(),
                g
            );
        }
    }

    #[test]
    fn recall_thresholds_have_101_points_endpoints_pinned() {
        let r = recall_thresholds();
        assert_eq!(r.len(), 101);
        assert_eq!(r[0], 0.0);
        assert_eq!(r[100], 1.0);
        // Midpoint check — bit-equal to numpy's linspace.
        assert_eq!(r[50].to_bits(), 0.5_f64.to_bits());
    }

    #[test]
    fn linspace_handles_degenerate_sizes() {
        assert!(linspace(0.0, 1.0, 0).is_empty());
        assert_eq!(linspace(0.5, 0.5, 1), vec![0.5]);
    }

    /// The 101 recall thresholds are `linspace(0, 1, 101)`, **not**
    /// `i / 100.0`. Ten of them differ: numpy emits `0 + i * (1/100)`
    /// and `1.0 / 100.0` is not exactly `0.01`, so the accumulated
    /// product lands one ULP high at those indices. The ladder is the
    /// x-axis the PR curve is integrated over (quirk **L2**/**C1**), so
    /// a "cleanup" to `i as f64 / 100.0` would silently move ten
    /// interpolation points.
    ///
    /// Captured from `numpy.linspace(0.0, 1.0, 101)` (NumPy 2.4.4).
    #[test]
    fn recall_thresholds_differ_from_i_over_100_at_ten_indices() {
        let r = recall_thresholds();
        let drifting: Vec<usize> = (0..101)
            .filter(|&i| r[i].to_bits() != ((i as f64) / 100.0).to_bits())
            .collect();
        assert_eq!(
            drifting,
            vec![35, 41, 47, 57, 69, 70, 82, 83, 94, 95],
            "recThrs no longer tracks numpy's linspace"
        );
        // Spot-pin two of them by value, so the list above cannot be
        // "fixed" by making both sides equally wrong.
        assert_eq!(r[35].to_bits(), 0.35000000000000003_f64.to_bits());
        assert_eq!(r[70].to_bits(), 0.7000000000000001_f64.to_bits());
    }

    // ----------------------------------------------------------------
    // numpy_pairwise_sum (quirk F8)
    // ----------------------------------------------------------------

    include!("golden/numpy_pairwise_sum_golden.rs");

    /// SplitMix64, the PRNG shared with
    /// `tools/gen_numpy_pairwise_sum_golden.py`. Its state is
    /// `seed + i * GAMMA`, which is why the generator can vectorise the
    /// same stream in numpy while this side steps it one call at a time.
    struct SplitMix64(u64);

    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        /// The generator's `bits_to_f64`: sign from the top bit, a
        /// biased exponent chosen by bits 59..63 from
        /// [`SWEEP_EXPONENTS`], mantissa from the low 52. Never inf or
        /// NaN; reaches subnormals and both signed zeros.
        fn next_f64(&mut self) -> f64 {
            let u = self.next_u64();
            let sign = (u >> 63) & 1;
            let biased = SWEEP_EXPONENTS[((u >> 59) & 0xF) as usize];
            let mant = u & 0x000F_FFFF_FFFF_FFFF;
            f64::from_bits((sign << 63) | (biased << 52) | mant)
        }
    }

    fn fnv1a(seed: u64, word: u64) -> u64 {
        let mut h = seed;
        for byte in word.to_le_bytes() {
            h ^= u64::from(byte);
            h = h.wrapping_mul(0x0000_0100_0000_01B3);
        }
        h
    }

    fn left_fold(a: &[f64]) -> f64 {
        let mut acc = 0.0_f64;
        for &x in a {
            acc += x;
        }
        acc
    }

    /// The committed golden fixture: one fixed 133-element vector and
    /// the `np.sum` of every prefix of it, as raw bit patterns.
    ///
    /// This is deliberately a *frozen* pin rather than a live
    /// comparison. Rust does not reassociate `f64` adds today (no
    /// fast-math, `llvm.fadd` without `reassoc`), so a live check
    /// against a second Rust expression would pass even if the port
    /// were rewritten into a form that a `#[target_feature]` wrapper or
    /// a fast-math crate could later vectorise. Bit patterns produced
    /// by the real NumPy cannot be satisfied by accident.
    ///
    /// **If this test starts failing after a NumPy upgrade and the Rust
    /// side is unchanged, NumPy's reduction tree moved.** That is a
    /// finding: the parity target changed, and the NumPy version the
    /// oracle runs under has to be stated in the parity contract before
    /// these numbers are regenerated. It is not a fixture to rebase.
    /// Regenerate only with `tools/gen_numpy_pairwise_sum_golden.py`.
    #[test]
    fn numpy_pairwise_sum_golden_bits() {
        let input: Vec<f64> = GOLDEN_INPUT_BITS
            .iter()
            .copied()
            .map(f64::from_bits)
            .collect();
        assert_eq!(input.len(), MAX_N);
        for n in 1..=MAX_N {
            let got = numpy_pairwise_sum(&input[..n]);
            assert_eq!(
                got.to_bits(),
                GOLDEN_PREFIX_SUM_BITS[n - 1],
                "n={n}: got {got:?} (0x{:016X}), numpy {} says 0x{:016X}",
                got.to_bits(),
                GOLDEN_NUMPY_VERSION,
                GOLDEN_PREFIX_SUM_BITS[n - 1],
            );
        }
    }

    /// A golden fixture that any summation order satisfies pins
    /// nothing. This asserts the committed vector actually has teeth:
    /// a plain left fold must get a large, specific number of the 133
    /// prefixes wrong.
    #[test]
    fn numpy_pairwise_sum_golden_fixture_discriminates_a_left_fold() {
        let input: Vec<f64> = GOLDEN_INPUT_BITS
            .iter()
            .copied()
            .map(f64::from_bits)
            .collect();
        let wrong = (1..=MAX_N)
            .filter(|&n| left_fold(&input[..n]).to_bits() != GOLDEN_PREFIX_SUM_BITS[n - 1])
            .count();
        assert_eq!(
            wrong, DISCRIMINATING_PREFIXES,
            "the golden vector no longer separates a pairwise sum from a left fold"
        );
        assert!(
            wrong > MAX_N / 2,
            "fixture is too weak to be worth committing"
        );
    }

    /// Exhaustive sweep: every `n` in `1..=133`, [`REPS`] random
    /// vectors each, against digests of what real NumPy returned.
    ///
    /// Magnitudes are adversarial by construction (see
    /// `SplitMix64::next_f64`): mixed exponents from subnormal to
    /// ~2^277, random signs so near-cancellation is common, and both
    /// signed zeros reachable. Uniform positive draws would barely
    /// exercise the tree — a vector whose largest term dwarfs the rest
    /// sums to the same double in any order.
    ///
    /// The digest is FNV-1a over the result bit patterns, so all
    /// 1.33M sums are pinned without committing 1.33M u64s. When it
    /// fails, [`RANDOM_SWEEP_FIRST_SUM_BITS`] gives a concrete first
    /// value to compare against.
    #[test]
    fn numpy_pairwise_sum_matches_numpy_over_exhaustive_random_sweep() {
        let mut buf: Vec<f64> = Vec::with_capacity(MAX_N);
        for n in 1..=MAX_N {
            let mut rng = SplitMix64(SEED_BASE ^ (n as u64));
            let mut digest = 0xCBF2_9CE4_8422_2325_u64;
            let mut first = 0_u64;
            for rep in 0..REPS {
                buf.clear();
                for _ in 0..n {
                    buf.push(rng.next_f64());
                }
                let bits = numpy_pairwise_sum(&buf).to_bits();
                if rep == 0 {
                    first = bits;
                }
                digest = fnv1a(digest, bits);
            }
            assert_eq!(
                first,
                RANDOM_SWEEP_FIRST_SUM_BITS[n - 1],
                "n={n}: first vector summed to 0x{first:016X}, numpy {} says 0x{:016X}",
                GOLDEN_NUMPY_VERSION,
                RANDOM_SWEEP_FIRST_SUM_BITS[n - 1],
            );
            assert_eq!(
                digest,
                RANDOM_SWEEP_DIGEST[n - 1],
                "n={n}: digest over {REPS} sums differs from numpy {}'s. \
                 Regenerate only with tools/gen_numpy_pairwise_sum_golden.py, \
                 and only after establishing *why* it moved.",
                GOLDEN_NUMPY_VERSION,
            );
        }
    }

    /// The `n > 128` recursion arm is reachable in production —
    /// COCO-WholeBody has 133 keypoints — so it must not be "simplified
    /// away". This pins the split point numpy uses: `n2 = n / 2` rounded
    /// *down* to a multiple of 8, i.e. 64 + 69 at n = 133, not 66 + 67.
    #[test]
    fn numpy_pairwise_sum_recursion_splits_at_a_multiple_of_eight() {
        let a: Vec<f64> = (0..133).map(|i| 1.0 + (i as f64) * 1e-16).collect();
        let naive_half = numpy_pairwise_sum(&a[..66]) + numpy_pairwise_sum(&a[66..]);
        let numpy_half = numpy_pairwise_sum(&a[..64]) + numpy_pairwise_sum(&a[64..]);
        assert_eq!(numpy_pairwise_sum(&a).to_bits(), numpy_half.to_bits());
        assert_ne!(
            numpy_pairwise_sum(&a).to_bits(),
            naive_half.to_bits(),
            "premise: a 66/67 split must be a different double, or this \
             test cannot see the rounding-down of n/2"
        );
    }

    /// A second, independently-derived numpy pin, carried over from the
    /// summary reduction this function absorbed
    /// (`summarize::mean_ignoring_sentinel`, which rides on
    /// `np.mean(s[s>-1])`): 1010 alternating elements, large enough to
    /// drive both the 8-lane block and the recursive split.
    ///
    /// The expected hex is `np.add.reduce(v).hex()` for the same
    /// sequence. Naive forward summation lands one ULP higher
    /// (`0x1.f900000002309p+8`).
    #[test]
    fn numpy_pairwise_sum_matches_add_reduce_on_the_summary_reduction_pin() {
        let v: Vec<f64> = (0..1010)
            .map(|i| if i % 2 == 0 { 1.0 } else { 1e-12 })
            .collect();
        let got = numpy_pairwise_sum(&v);
        let expected = f64::from_bits(0x407f_9000_0000_22b4);
        assert_eq!(
            got.to_bits(),
            expected.to_bits(),
            "drifts from numpy: got {got:e}, expected {expected:e}",
        );
    }

    #[test]
    fn numpy_pairwise_sum_handles_short_inputs_with_naive_fallback() {
        // n < 8 uses the simple loop; verify a hand-checked tiny case.
        let v = [1.0_f64, 2.0, 3.0, 4.0];
        assert_eq!(numpy_pairwise_sum(&v), 10.0);
        assert_eq!(numpy_pairwise_sum(&[]), 0.0);
        assert_eq!(numpy_pairwise_sum(&[42.0]), 42.0);
    }

    /// Below 8 elements numpy left-folds from `+0.0`; from 8 to 128 it
    /// runs eight accumulators seeded with `a[0..8]`. The boundary is
    /// observable.
    #[test]
    fn numpy_pairwise_sum_switches_strategy_at_eight_elements() {
        // Seven elements: plain left fold, so it equals one.
        let seven = [1.0_f64, 1e-16, 1e-16, 1e-16, 1e-16, 1e-16, 1e-16];
        assert_eq!(
            numpy_pairwise_sum(&seven).to_bits(),
            left_fold(&seven).to_bits()
        );
        // Eight: the unrolled block keeps the tiny terms in separate
        // accumulators, where they survive instead of being absorbed.
        let eight = [1.0_f64, 1e-16, 1e-16, 1e-16, 1e-16, 1e-16, 1e-16, 1e-16];
        assert_ne!(
            numpy_pairwise_sum(&eight).to_bits(),
            left_fold(&eight).to_bits(),
            "the 8-accumulator block must not degenerate to a left fold"
        );
    }

    /// numpy's reduction seeds its accumulator with `add`'s identity
    /// (`+0.0`) and adds the pairwise total to that. The seed is
    /// invisible everywhere except the sign of a zero result:
    /// `np.sum(np.full(8, -0.0))` is `+0.0`, while
    /// `DOUBLE_pairwise_sum` on its own returns `-0.0`. Dropping the
    /// seed would be a one-bit bug that no magnitude test can see.
    #[test]
    fn numpy_pairwise_sum_normalises_negative_zero_like_numpys_reduce_seed() {
        for n in [1_usize, 7, 8, 9, 128, 129, 133] {
            let a = vec![-0.0_f64; n];
            assert_eq!(
                numpy_pairwise_sum(&a).to_bits(),
                0.0_f64.to_bits(),
                "n={n}: np.sum of all -0.0 is +0.0"
            );
        }
        // The seed must not flip a genuinely negative result, nor an
        // ordinary zero-valued cancellation's sign convention.
        assert_eq!(numpy_pairwise_sum(&[-1.0, -2.0]), -3.0);
        assert_eq!(
            numpy_pairwise_sum(&[1.0, -1.0]).to_bits(),
            0.0_f64.to_bits()
        );
        assert_eq!(numpy_pairwise_sum(&[]).to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn parity_mode_default_is_corrected() {
        // ADR-0002: corrected is the default for net-new users; strict
        // is the migration mode.
        assert_eq!(ParityMode::default(), ParityMode::Corrected);
    }

    #[test]
    fn calibration_quantile_method_pinned_to_linear() {
        // ADR-0018 / Quirk P1: bin-edges use numpy method='linear'.
        assert_eq!(CALIBRATION_QUANTILE_METHOD, "linear");
    }

    #[test]
    fn quantile_linear_matches_numpy_on_arange() {
        // Reference values from numpy:
        //   import numpy as np
        //   v = np.arange(11, dtype=float)  # 0..10
        //   np.quantile(v, [0.0, 0.25, 0.5, 0.75, 1.0], method='linear')
        //   -> array([ 0. ,  2.5,  5. ,  7.5, 10. ])
        let v: Vec<f64> = (0..=10).map(|x| x as f64).collect();
        let got = quantile_linear(&v, &[0.0, 0.25, 0.5, 0.75, 1.0]);
        assert_eq!(got, vec![0.0, 2.5, 5.0, 7.5, 10.0]);
    }

    #[test]
    fn quantile_linear_interpolates_two_points() {
        // np.quantile([0.0, 1.0], [0.0, 0.5, 1.0], method='linear')
        //   -> array([0. , 0.5, 1. ])
        let got = quantile_linear(&[0.0, 1.0], &[0.0, 0.5, 1.0]);
        assert_eq!(got, vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn quantile_linear_single_element_replicates() {
        // Single-element input → that element for every q.
        let got = quantile_linear(&[0.42], &[0.0, 0.3, 1.0]);
        assert_eq!(got, vec![0.42, 0.42, 0.42]);
    }

    #[test]
    fn quantile_linear_empty_input_is_empty() {
        // Empty input is a no-op: the caller is expected to short-circuit
        // before constructing bin-edges.
        let got = quantile_linear(&[], &[0.0, 0.5, 1.0]);
        assert!(got.is_empty());
    }

    #[test]
    fn quantile_linear_endpoints_pinned() {
        // q=0 → first, q=1 → last, regardless of distribution shape.
        let v = [0.1, 0.2, 0.8, 0.9];
        let got = quantile_linear(&v, &[0.0, 1.0]);
        assert_eq!(got, vec![0.1, 0.9]);
    }
}
