//! Object Keypoint Similarity (`iouType="keypoints"`) — Phase 3, ADR-0012.
//!
//! Mirrors `pycocotools.cocoeval.COCOeval.computeOks` (cocoeval.py:215-235
//! at the pinned `pycocotools==2.0.11`). For each `(g, d)` pair, the
//! kernel sums Gaussian-weighted exponentials of squared keypoint
//! distances over the visible-GT subset (or, when GT carries no visible
//! keypoints, over an asymmetric expansion of the GT bbox), then divides
//! by the count of summed terms.
//!
//! The inner OKS expression operates on at most a handful of f64s per
//! cell (17 keypoints for COCO-person). At that grain a `pulp::Arch::dispatch`
//! wrapper buys nothing: the per-cell work is already fully scalar-vectorizable
//! by LLVM and the call shape (one row of `vars`, one row of (xg, yg, vg))
//! does not amortize a SIMD setup cost. We ship the scalar form. Revisit
//! if a benchmark on a real keypoint workload says otherwise.
//!
//! ## Quirk dispositions (ADR-0012)
//!
//! - **F1** (`corrected`): per-category sigmas live in
//!   [`OksSimilarity::sigmas`] as `HashMap<i64, Vec<f64>>`. An empty
//!   override map means "use [`COCO_PERSON_SIGMAS`] for every category".
//! - **F2** (`strict`): area normaliser uses `gt.area + f64::EPSILON`
//!   (numpy's `np.spacing(1)` on f64). Outputs match within ULP of the
//!   reference oracle.
//! - **F3** (`strict`): when GT has zero visible keypoints (`k1 == 0`),
//!   the per-keypoint distance falls back to the bbox-surrogate
//!   computation. The whole keypoint vector contributes (no `vg > 0`
//!   mask).
//! - **F4** (`strict`): bbox expansion is asymmetric on both axes —
//!   `[bb.x - bb.w, bb.x + 2 * bb.w]` and `[bb.y - bb.h, bb.y + 2 * bb.h]`.
//!   The lower bound subtracts one width while the upper bound adds two,
//!   matching pycocotools verbatim.
//! - **F5** (`strict`): empty `gts` or `dts` returns the zero-shape
//!   matrix unchanged. Mirrors the segm/bbox kernels.
//! - **F7** (`strict`): the mean over the summed terms is a *division*
//!   by the term count, matching `np.sum(np.exp(-e)) / e.shape[0]`
//!   (ce:232). Multiplying by a hoisted `1.0 / count` reciprocal is
//!   not bit-equal — the reciprocal rounds once and the product rounds
//!   again, versus one correctly-rounded divide — and the resulting
//!   1-ULP shift can flip a match at a threshold boundary. See the
//!   `f7_*` tests.
//! - **F8** (`strict`): `np.sum` is not a left fold. The reduction goes
//!   through numpy's `DOUBLE_pairwise_sum`, ported in
//!   [`crate::parity::numpy_pairwise_sum`]. Left-folding the `exp(-e)`
//!   terms moves 22.9 % of COCO-person cells and 74.6 % of
//!   COCO-WholeBody cells by at least one ULP.
//! - **F9** (`strict`): [`COCO_PERSON_SIGMAS`] is
//!   `np.array([.26, .25, …]) / 10.0` — two roundings, not one. Folding
//!   the divide into decimal literals (`0.026`, `0.035`, …) shifts five
//!   of the seventeen sigmas by one ULP, which moves 8.2 % of cells.
//! - **F10** (`strict`): `np.array(gt['keypoints'])` is **int64** when
//!   the JSON holds integers, so `dx**2 + dy**2` is exact integer
//!   arithmetic. vernier carries keypoints as `f64` and cannot see the
//!   JSON dtype, so it reproduces that exactness by *bounding* the
//!   input instead: see [`MAX_ABS_KEYPOINT_COORD`].
//!
//! Everything above is bit-exact. The one term that is **not** is
//! `exp` itself: `np.exp`'s result depends on the numpy build's SIMD
//! dispatch and the platform libm, so there is no single reference to
//! port. It is deliberately out of scope here.
//!
//! Quirk **D2** (DT keypoint visibility flags are unconstrained at the
//! dataset boundary) is a *dataset* concern enforced by `loadRes`-equivalent
//! code, not by this kernel. The OKS expression only reads DT
//! `(x_d, y_d)` pairs and never branches on `v_d`.

use std::collections::HashMap;

use ndarray::ArrayViewMut2;

use super::Similarity;
use crate::error::EvalError;
use crate::parity::numpy_pairwise_sum;

/// Default `kpt_oks_sigmas` for COCO-person (already scaled by `1/10`,
/// matching what pycocotools applies as `kpt_oks_sigmas`).
///
/// Source: `pycocotools.cocoeval.Params.setKpParams` divides the raw
/// table by 10 once at construction; users of the Rust kernel pass the
/// post-divide values directly so we do not double-divide.
///
/// Quirk **F9** — strict. The `/ 10.0` is written out rather than
/// folded into decimal literals *on purpose*, and the difference is
/// not cosmetic. `.26 / 10.0` is `0.026000000000000002`, one ULP above
/// the literal `0.026`; the same happens at `.35`, `.35`, `1.07` and
/// `1.07` (one ULP below, for the two `.35`s). Five of the seventeen
/// sigmas move, `vars` moves with them, and 8.2 % of random
/// COCO-person cells land on a different double. Writing
/// `0.026, 0.025, …` here is exactly the kind of "harmless" cleanup
/// that silently breaks parity — do not make it.
pub const COCO_PERSON_SIGMAS: [f64; 17] = [
    0.26 / 10.0,
    0.25 / 10.0,
    0.25 / 10.0,
    0.35 / 10.0,
    0.35 / 10.0,
    0.79 / 10.0,
    0.79 / 10.0,
    0.72 / 10.0,
    0.72 / 10.0,
    0.62 / 10.0,
    0.62 / 10.0,
    1.07 / 10.0,
    1.07 / 10.0,
    0.87 / 10.0,
    0.87 / 10.0,
    0.89 / 10.0,
    0.89 / 10.0,
];

/// Largest keypoint coordinate magnitude the kernel accepts, `2^25`.
/// (Quirk **F10** — strict.)
///
/// pycocotools reads keypoints with `np.array(ann['keypoints'])`. COCO
/// ground truth stores them as JSON integers, so that array is
/// **int64** and `dx**2 + dy**2` is computed in exact integer
/// arithmetic before the first division promotes it to `float64`.
/// vernier carries keypoints as `f64` from the parser onward and has no
/// way to recover the JSON dtype, so it reproduces the exactness by
/// bounding the input rather than by emulating int64.
///
/// The bound is derived, not guessed. With `|x| <= 2^25` for every
/// coordinate: `dx = x_d - x_g` is an integer of magnitude `<= 2^26`
/// and therefore exact; `dx * dx <= 2^52` is exact; and
/// `dx * dx + dy * dy <= 2^53` is exact. Under those conditions the
/// `f64` expression is bit-identical to numpy's int64 one. Above them
/// it silently is not — which is why the kernel rejects rather than
/// assumes.
///
/// Non-integral coordinates need no bound at all: a JSON float makes
/// numpy's array `float64` too, so both sides already do the same `f64`
/// arithmetic. The check is applied uniformly anyway, because the
/// kernel cannot tell the two cases apart and `2^25` = 33 554 432 is
/// some 300× beyond the largest pixel coordinate any real dataset
/// carries.
pub const MAX_ABS_KEYPOINT_COORD: f64 = 33_554_432.0;

/// Annotation shape consumed by [`OksSimilarity`]. The matching engine
/// constructs these from a [`crate::dataset::CocoAnnotation`] before
/// invoking [`Similarity::compute`].
///
/// `keypoints` is the COCO flat triplet layout
/// `[x_0, y_0, v_0, x_1, y_1, v_1, ...]`. Length must equal
/// `3 * sigmas_for(category_id).len()`; mismatch is a typed
/// [`EvalError::DimensionMismatch`].
#[derive(Debug, Clone, PartialEq)]
pub struct OksAnn {
    /// Category id used to look up per-category sigmas (quirk **F1**).
    pub category_id: i64,
    /// Flat keypoint triplets: `[x_0, y_0, v_0, x_1, y_1, v_1, ...]`.
    pub keypoints: Vec<f64>,
    /// COCO `num_keypoints` count of *visible* keypoints (`v > 0`).
    /// Read on the GT side to drive the **F3** bbox-surrogate branch;
    /// ignored on the DT side.
    pub num_keypoints: u32,
    /// Tight bbox `[x, y, w, h]`. Used on the GT side for the **F3**
    /// surrogate path and for the **F4** asymmetric expansion; ignored
    /// on the DT side.
    pub bbox: [f64; 4],
    /// GT object area (segmentation area, per pycocotools). Drives the
    /// **F2** OKS normaliser. Ignored on the DT side.
    pub area: f64,
}

/// OKS [`Similarity`] impl. Carries an optional per-category sigma
/// override map; the matching engine reads only the [`Similarity`]
/// trait so the knob lives here, not in matching (per ADR-0005).
#[derive(Debug, Clone, Default)]
pub struct OksSimilarity {
    /// Per-category sigma override. Empty = use [`COCO_PERSON_SIGMAS`]
    /// for every category. Sigmas must be passed already scaled (i.e.,
    /// post-divide-by-10 as pycocotools applies internally). Quirk **F1**
    /// disposition `corrected`.
    pub sigmas: HashMap<i64, Vec<f64>>,
}

impl OksSimilarity {
    /// Construct from a per-category sigma map. An empty map is a valid
    /// configuration meaning "default COCO-person sigmas everywhere".
    #[must_use]
    pub fn new(sigmas: HashMap<i64, Vec<f64>>) -> Self {
        Self { sigmas }
    }

    /// Sigmas for a given category id, falling back to
    /// [`COCO_PERSON_SIGMAS`] when no override is registered.
    #[inline]
    fn sigmas_for(&self, category_id: i64) -> &[f64] {
        self.sigmas
            .get(&category_id)
            .map(Vec::as_slice)
            .unwrap_or(&COCO_PERSON_SIGMAS)
    }
}

/// pycocotools' GT-visibility predicate, `vg > 0`, written once.
///
/// Both the term count (`k1 = np.count_nonzero(vg > 0)`) and the term
/// mask (`e = e[vg > 0]`) are the *same* comparison in cocoeval, so
/// they have to be the same comparison here. Spelling the mask as its
/// negation (`v <= 0.0`) is **not** equivalent: a NaN visibility flag
/// satisfies neither `>` nor `<=`, so the count would drop that
/// keypoint while the mask kept it, and the cell would be divided by
/// the wrong term count. numpy's mask drops it on both sides.
#[inline]
fn is_visible(v: f64) -> bool {
    v > 0.0
}

impl Similarity for OksSimilarity {
    type Annotation = OksAnn;

    fn compute(
        &self,
        gts: &[OksAnn],
        dts: &[OksAnn],
        out: &mut ArrayViewMut2<'_, f64>,
    ) -> Result<(), EvalError> {
        if out.nrows() != gts.len() || out.ncols() != dts.len() {
            return Err(EvalError::DimensionMismatch {
                detail: format!(
                    "OKS output is {}x{}, expected {}x{}",
                    out.nrows(),
                    out.ncols(),
                    gts.len(),
                    dts.len()
                ),
            });
        }
        // F5: empty inputs leave the zero-shape matrix as-is.
        if gts.is_empty() || dts.is_empty() {
            return Ok(());
        }

        // Validate keypoint vector shape against per-category sigmas
        // up-front. The kernel hot-loop assumes `keypoints.len() == 3 * k`
        // and `sigmas.len() == k`, so any mismatch is a typed error here
        // rather than a panic deeper in the loop.
        for (side, anns) in [("gt", gts), ("dt", dts)] {
            for (idx, ann) in anns.iter().enumerate() {
                let k = self.sigmas_for(ann.category_id).len();
                if ann.keypoints.len() != 3 * k {
                    return Err(EvalError::DimensionMismatch {
                        detail: format!(
                            "OKS {side}[{idx}] (cat {}): keypoints len {} != 3 * sigmas len {}",
                            ann.category_id,
                            ann.keypoints.len(),
                            k
                        ),
                    });
                }
                // F10: the (x, y) pairs must stay inside the range where
                // f64 reproduces numpy's int64 keypoint arithmetic
                // exactly. Visibility flags are excluded — they are only
                // ever compared against zero, and `NaN > 0` is false on
                // both sides alike.
                for (t, triplet) in ann.keypoints.chunks_exact(3).enumerate() {
                    for (axis, v) in [("x", triplet[0]), ("y", triplet[1])] {
                        if !v.is_finite() {
                            return Err(EvalError::NonFinite {
                                context: "OKS keypoint coordinate",
                            });
                        }
                        if v.abs() > MAX_ABS_KEYPOINT_COORD {
                            return Err(EvalError::InvalidAnnotation {
                                detail: format!(
                                    "OKS {side}[{idx}] (cat {cat}): keypoint {t} {axis} = {v} \
                                     exceeds the exact-arithmetic bound \
                                     {MAX_ABS_KEYPOINT_COORD}; beyond it vernier's f64 \
                                     `dx*dx + dy*dy` stops matching pycocotools' int64 one \
                                     (quirk F10)",
                                    cat = ann.category_id,
                                ),
                            });
                        }
                    }
                }
            }
        }

        // Scratch for the per-cell `exp(-e)` terms. numpy reduces them
        // with `DOUBLE_pairwise_sum`, which needs the whole vector in
        // hand rather than a running accumulator (**F8**). One
        // allocation for the whole (G, D) cell, reused per pair.
        let mut terms: Vec<f64> = Vec::new();

        for (g, gt) in gts.iter().enumerate() {
            let sigmas = self.sigmas_for(gt.category_id);
            let k = sigmas.len();
            terms.reserve(k);
            // vars[i] = (2 * sigma_i)^2; precomputed once per GT row.
            // `k` is tiny (17 typical) and the alloc is dwarfed by
            // the per-cell exp(); no need for a SmallVec.
            //
            // numpy's `DOUBLE_power` has an `in2 == 2.0` fast path that
            // returns `in1 * in1` — one multiply, one rounding — so
            // `(sigmas * 2)**2` is `t * t` and nothing more. Written out
            // rather than `powi(2)` so that stays legible.
            let vars: Vec<f64> = sigmas
                .iter()
                .map(|s| {
                    let t = 2.0 * s;
                    t * t
                })
                .collect();
            let area_norm = gt.area + f64::EPSILON;
            let k1 = gt
                .keypoints
                .chunks_exact(3)
                .filter(|t| is_visible(t[2]))
                .count();

            // F4: asymmetric bbox expansion. Lower bound subtracts one
            // width / height; upper bound adds two. Pycocotools verbatim.
            let [bx, by, bw, bh] = gt.bbox;
            let (x0, x1) = (bx - bw, bx + 2.0 * bw);
            let (y0, y1) = (by - bh, by + 2.0 * bh);

            // Denominator is fixed per GT row: `k1` visible terms on the
            // standard path, `k` total terms on the F3 surrogate path.
            // Hoisted out of the DT loop so cells that share the same
            // row don't re-derive it.
            let denom_count = if k1 > 0 { k1 } else { k };
            if denom_count == 0 {
                for d in 0..dts.len() {
                    out[[g, d]] = 0.0;
                }
                continue;
            }
            // F7: divide, never multiply by a precomputed reciprocal.
            let denom = denom_count as f64;

            for (d, dt) in dts.iter().enumerate() {
                terms.clear();

                if k1 > 0 {
                    // Standard path: only visible GT keypoints contribute.
                    for (i, (gt_t, dt_t)) in gt
                        .keypoints
                        .chunks_exact(3)
                        .zip(dt.keypoints.chunks_exact(3))
                        .enumerate()
                    {
                        if !is_visible(gt_t[2]) {
                            continue;
                        }
                        let dx = dt_t[0] - gt_t[0];
                        let dy = dt_t[1] - gt_t[1];
                        let e = (dx * dx + dy * dy) / vars[i] / area_norm / 2.0;
                        terms.push((-e).exp());
                    }
                } else {
                    // F3: bbox-surrogate path. Every keypoint contributes;
                    // the "distance" is how far the DT keypoint sits
                    // outside the F4-expanded GT bbox.
                    for (i, dt_t) in dt.keypoints.chunks_exact(3).enumerate() {
                        let xd = dt_t[0];
                        let yd = dt_t[1];
                        let dx = (x0 - xd).max(0.0) + (xd - x1).max(0.0);
                        let dy = (y0 - yd).max(0.0) + (yd - y1).max(0.0);
                        let e = (dx * dx + dy * dy) / vars[i] / area_norm / 2.0;
                        terms.push((-e).exp());
                    }
                }

                // F8 then F7: `np.sum(np.exp(-e)) / e.shape[0]`
                // verbatim — numpy's pairwise reduction, then one
                // correctly-rounded divide by the term count.
                debug_assert_eq!(terms.len(), denom_count);
                out[[g, d]] = numpy_pairwise_sum(&terms) / denom;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// Builds an [`OksAnn`] from `(x, y, v)` triplets. `bbox` is given as
    /// `[x, y, w, h]`. `num_keypoints` is derived from the visibilities.
    fn ann(cat: i64, kps: &[(f64, f64, u32)], bbox: [f64; 4], area: f64) -> OksAnn {
        let mut keypoints = Vec::with_capacity(kps.len() * 3);
        let mut visible = 0_u32;
        for (x, y, v) in kps {
            keypoints.push(*x);
            keypoints.push(*y);
            keypoints.push(f64::from(*v));
            if *v > 0 {
                visible += 1;
            }
        }
        OksAnn {
            category_id: cat,
            keypoints,
            num_keypoints: visible,
            bbox,
            area,
        }
    }

    /// 17 visible COCO-person keypoints all at `(x, y)`. Useful as a
    /// degenerate fixture when the test cares about the exponent shape,
    /// not the geometry.
    fn const_kps(x: f64, y: f64, v: u32) -> Vec<(f64, f64, u32)> {
        vec![(x, y, v); 17]
    }

    fn compute(sim: &OksSimilarity, gts: &[OksAnn], dts: &[OksAnn]) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((gts.len(), dts.len()));
        sim.compute(gts, dts, &mut out.view_mut()).unwrap();
        out
    }

    #[test]
    fn empty_gts_produces_zero_row_matrix() {
        let dts = vec![ann(1, &const_kps(0.0, 0.0, 2), [0.0, 0.0, 1.0, 1.0], 1.0); 4];
        let mut out = Array2::<f64>::from_elem((0, 4), 7.0);
        OksSimilarity::default()
            .compute(&[], &dts, &mut out.view_mut())
            .unwrap();
        assert_eq!(out.shape(), &[0, 4]);
    }

    #[test]
    fn empty_dts_produces_zero_col_matrix() {
        let gts = vec![ann(1, &const_kps(0.0, 0.0, 2), [0.0, 0.0, 1.0, 1.0], 1.0); 3];
        let mut out = Array2::<f64>::from_elem((3, 0), 7.0);
        OksSimilarity::default()
            .compute(&gts, &[], &mut out.view_mut())
            .unwrap();
        assert_eq!(out.shape(), &[3, 0]);
    }

    #[test]
    fn both_empty_produces_zero_zero_matrix() {
        let mut out = Array2::<f64>::zeros((0, 0));
        OksSimilarity::default()
            .compute(&[], &[], &mut out.view_mut())
            .unwrap();
        assert_eq!(out.shape(), &[0, 0]);
    }

    #[test]
    fn single_perfect_match_is_one() {
        // All 17 keypoints aligned exactly → every per-keypoint
        // exponent is 0, exp(0) = 1, sum / 17 = 1.0. The F2 epsilon
        // never matters because the exponent is zero anyway.
        let kps = const_kps(5.0, 7.0, 2);
        let g = ann(1, &kps, [0.0, 0.0, 10.0, 10.0], 100.0);
        let d = ann(1, &kps, [0.0, 0.0, 10.0, 10.0], 100.0);
        let m = compute(&OksSimilarity::default(), &[g], &[d]);
        assert!((m[[0, 0]] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn bbox_surrogate_path_when_no_visible_keypoints() {
        // F3: GT has all visibilities 0 → bbox-surrogate kicks in. DT
        // keypoints sit inside the F4-expanded bbox, so dx = dy = 0
        // for every keypoint and OKS = 1.0. This pins both that the
        // surrogate runs (no panic on k1=0) and that "inside the
        // expanded box" yields zero distance.
        let gt_kps: Vec<_> = (0..17).map(|_| (0.0, 0.0, 0)).collect();
        let dt_kps = const_kps(5.0, 5.0, 2);
        let g = ann(1, &gt_kps, [0.0, 0.0, 10.0, 10.0], 100.0);
        let d = ann(1, &dt_kps, [0.0, 0.0, 10.0, 10.0], 100.0);
        let m = compute(&OksSimilarity::default(), &[g], &[d]);
        assert!((m[[0, 0]] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn per_category_sigma_override_changes_output() {
        // Same fixture, different sigmas: must produce different OKS.
        // GT and DT differ by a 1-pixel x-offset on every keypoint
        // (all visible). With the larger override sigmas (0.5) the
        // exponent shrinks and OKS rises; with defaults it falls.
        let gt_kps = const_kps(5.0, 5.0, 2);
        let dt_kps = const_kps(6.0, 5.0, 2);
        let g = ann(1, &gt_kps, [0.0, 0.0, 10.0, 10.0], 100.0);
        let d = ann(1, &dt_kps, [0.0, 0.0, 10.0, 10.0], 100.0);

        let default = compute(
            &OksSimilarity::default(),
            std::slice::from_ref(&g),
            std::slice::from_ref(&d),
        );

        let mut override_map = HashMap::new();
        override_map.insert(1_i64, vec![0.5_f64; 17]);
        let custom = compute(&OksSimilarity::new(override_map), &[g], &[d]);

        // Hand check the override case: dx²+dy² = 1, vars = (2*0.5)² = 1,
        // e = 1 / 1 / (100 + EPS) / 2 ≈ 0.005.
        let area_norm = 100.0_f64 + f64::EPSILON;
        let e = 1.0_f64 / 1.0_f64 / area_norm / 2.0;
        let expected = (-e).exp();
        assert!((custom[[0, 0]] - expected).abs() < 1e-10);

        // And the override genuinely diverges from the default sigmas
        // (the override is wired through, not silently shadowed).
        assert!((custom[[0, 0]] - default[[0, 0]]).abs() > 1e-6);
    }

    #[test]
    fn f4_bbox_expansion_is_asymmetric_on_x() {
        // F3 path (k1 = 0) plus F4 expansion on x:
        //   x0 = bb.x - bb.w = 10 - 5 = 5
        //   x1 = bb.x + 2*bb.w = 10 + 10 = 20
        // A DT keypoint at x=20-1e-9 is inside (dx contribution 0); a DT
        // at x=20+1e-9 sits eps past x1 (dx contribution ~1e-9, e ≈ 0,
        // OKS ≈ 1). To distinguish meaningfully we push past x1 by a
        // visible margin and assert the cell drops below 1.0.
        let gt_kps: Vec<_> = (0..17).map(|_| (0.0, 0.0, 0)).collect();
        let g = ann(1, &gt_kps, [10.0, 0.0, 5.0, 1.0], 1.0);

        // y stays inside [-1, 2] = [bb.y-bb.h, bb.y+2*bb.h]; the only
        // distance source is x.
        let inside_kps = const_kps(19.999, 0.5, 2);
        let outside_kps = const_kps(25.0, 0.5, 2);
        let d_inside = ann(1, &inside_kps, [0.0, 0.0, 1.0, 1.0], 1.0);
        let d_outside = ann(1, &outside_kps, [0.0, 0.0, 1.0, 1.0], 1.0);

        let m = compute(&OksSimilarity::default(), &[g], &[d_inside, d_outside]);

        assert!((m[[0, 0]] - 1.0).abs() < 1e-6, "inside x1 should be ~1.0");
        assert!(m[[0, 1]] < 1.0 - 1e-6, "outside x1 should drop below 1.0");

        // And confirm the lower bound is at x0 = bb.x - bb.w (asymmetric
        // — not bb.x - bb.w/2). A DT point at x = bb.x - bb.w + eps is
        // inside; at x = bb.x - bb.w - 5 it is outside.
        let lower_in = const_kps(5.001, 0.5, 2);
        let lower_out = const_kps(0.0, 0.5, 2);
        let d_lower_in = ann(1, &lower_in, [0.0, 0.0, 1.0, 1.0], 1.0);
        let d_lower_out = ann(1, &lower_out, [0.0, 0.0, 1.0, 1.0], 1.0);
        let g2 = ann(
            1,
            &(0..17).map(|_| (0.0, 0.0, 0)).collect::<Vec<_>>(),
            [10.0, 0.0, 5.0, 1.0],
            1.0,
        );
        let m2 = compute(&OksSimilarity::default(), &[g2], &[d_lower_in, d_lower_out]);
        assert!((m2[[0, 0]] - 1.0).abs() < 1e-6, "inside x0 should be ~1.0");
        assert!(m2[[0, 1]] < 1.0 - 1e-6, "outside x0 should drop below 1.0");
    }

    #[test]
    fn sigma_length_mismatch_returns_typed_error() {
        // Override registers 16 sigmas for cat 1; annotation carries
        // 17 keypoints (51 floats). The kernel must surface this as
        // DimensionMismatch, not a panic.
        let g = ann(1, &const_kps(0.0, 0.0, 2), [0.0, 0.0, 10.0, 10.0], 100.0);
        let d = g.clone();

        let mut override_map = HashMap::new();
        override_map.insert(1_i64, vec![0.05_f64; 16]);
        let sim = OksSimilarity::new(override_map);

        let mut out = Array2::<f64>::zeros((1, 1));
        let err = sim.compute(&[g], &[d], &mut out.view_mut()).unwrap_err();
        match err {
            EvalError::DimensionMismatch { detail } => {
                assert!(
                    detail.contains("keypoints"),
                    "expected keypoints detail, got {detail}",
                );
            }
            other => panic!("expected DimensionMismatch, got {other:?}"),
        }
    }

    #[test]
    fn output_shape_mismatch_returns_typed_error() {
        let g = ann(1, &const_kps(0.0, 0.0, 2), [0.0, 0.0, 10.0, 10.0], 100.0);
        let d = g.clone();
        let mut out = Array2::<f64>::zeros((2, 3));
        let err = OksSimilarity::default()
            .compute(&[g], &[d], &mut out.view_mut())
            .unwrap_err();
        assert!(matches!(err, EvalError::DimensionMismatch { .. }));
    }

    #[test]
    fn f2_area_epsilon_handles_zero_area_gt_without_nan() {
        // F2: gt.area = 0 → divide by EPSILON, OKS finite.
        // With perfect alignment, exponent is 0 regardless and OKS = 1.
        // The test pins that we don't NaN out on zero area.
        let kps = const_kps(0.0, 0.0, 2);
        let g = ann(1, &kps, [0.0, 0.0, 0.0, 0.0], 0.0);
        let d = ann(1, &kps, [0.0, 0.0, 0.0, 0.0], 0.0);
        let m = compute(&OksSimilarity::default(), &[g], &[d]);
        assert!(m[[0, 0]].is_finite());
        assert!((m[[0, 0]] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn invisible_gt_keypoints_excluded_from_standard_path() {
        // 17 keypoints, only the first visible. DT matches the first
        // keypoint exactly but is wildly off elsewhere. Standard path
        // (k1 > 0) only sums over the visible subset, so the answer
        // is exp(0)/1 = 1.0 exactly. If the kernel forgot to mask by
        // vg > 0, the wildly-off keypoints would drag it well below 1.
        let mut gt_kps = vec![(0.0, 0.0, 0); 17];
        gt_kps[0] = (5.0, 5.0, 2);
        let mut dt_kps = vec![(1000.0, 1000.0, 2); 17];
        dt_kps[0] = (5.0, 5.0, 2);
        let g = ann(1, &gt_kps, [0.0, 0.0, 10.0, 10.0], 100.0);
        let d = ann(1, &dt_kps, [0.0, 0.0, 10.0, 10.0], 100.0);
        let m = compute(&OksSimilarity::default(), &[g], &[d]);
        assert!((m[[0, 0]] - 1.0).abs() < 1e-12);
    }

    /// F7: the OKS mean must be `sum / count`, not `sum * (1 / count)`.
    ///
    /// With 49 keypoints all aligned exactly, every term is `exp(0) == 1.0`
    /// and the sum is exactly `49.0`, so pycocotools' `np.sum(...) / 49`
    /// is exactly `1.0`. `49.0 * (1.0 / 49.0)` rounds to
    /// `0.9999999999999999` — one ULP low. `k = 49` is one of eight counts
    /// at or below 200 (49, 98, 103, 107, 161, 187, 196, 197) where a
    /// *perfect* match stops being bit-exactly 1.0 under the reciprocal.
    #[test]
    fn f7_perfect_match_is_bit_exactly_one_for_reciprocal_hostile_k() {
        for k in [49_usize, 98, 103, 107] {
            let kps: Vec<(f64, f64, u32)> = vec![(5.0, 7.0, 2); k];
            let g = ann(1, &kps, [0.0, 0.0, 10.0, 10.0], 100.0);
            let d = ann(1, &kps, [0.0, 0.0, 10.0, 10.0], 100.0);

            let mut override_map = HashMap::new();
            override_map.insert(1_i64, vec![0.05_f64; k]);
            let m = compute(&OksSimilarity::new(override_map), &[g], &[d]);

            // Bit-exact, not approximate: `assert_eq!` on f64 is the point.
            assert_eq!(
                m[[0, 0]],
                1.0,
                "k={k}: perfect match must be exactly 1.0 (got {:?})",
                m[[0, 0]]
            );
        }
    }

    /// F7: a reachable 1-ULP shift that flips a match on the *default*
    /// COCO threshold ladder.
    ///
    /// 35 keypoints, all GT-visible. 28 DT keypoints sit exactly on their
    /// GT (`exp(0) == 1.0`); the remaining 7 are 1000px away, whose
    /// exponent (`1e6 / 0.01 / 100 / 2 == 5e5`) underflows `exp(-e)` to
    /// exactly `0.0`. The sum is therefore exactly `28.0`, and
    /// `28.0 / 35.0` is exactly `0.8` — the COCO ladder's 7th threshold,
    /// which matches under `iou >= t`. `28.0 * (1.0 / 35.0)` is
    /// `0.7999999999999999`, which does not. Keypoint counts other than 17
    /// are reachable because quirk **F1** (`corrected`) ships per-category
    /// sigmas of arbitrary length.
    #[test]
    fn f7_reciprocal_would_flip_a_match_at_the_oks_0_80_threshold() {
        const K: usize = 35;
        let gt_kps: Vec<(f64, f64, u32)> = vec![(0.0, 0.0, 2); K];
        let mut dt_kps: Vec<(f64, f64, u32)> = vec![(0.0, 0.0, 2); K];
        for slot in dt_kps.iter_mut().skip(28) {
            *slot = (1000.0, 0.0, 2);
        }
        let g = ann(1, &gt_kps, [0.0, 0.0, 10.0, 10.0], 100.0);
        let d = ann(1, &dt_kps, [0.0, 0.0, 10.0, 10.0], 100.0);

        let mut override_map = HashMap::new();
        override_map.insert(1_i64, vec![0.05_f64; K]);
        let m = compute(&OksSimilarity::new(override_map), &[g], &[d]);

        assert_eq!(
            m[[0, 0]],
            0.8,
            "28 of 35 exact keypoints must land bit-exactly on 0.80 (got {:?})",
            m[[0, 0]]
        );
        // The matching ladder's gate is `iou >= t` (quirk B2). The
        // reciprocal form lands one ULP below and silently drops the match.
        assert!(m[[0, 0]] >= 0.8, "must match at the OKS=0.80 threshold");
        assert_ne!(28.0_f64 * (1.0 / K as f64), 0.8, "premise of this test");
    }

    /// Recomputes a cell's `exp(-e)` terms with the same expression the
    /// kernel uses, so the `f8_*` tests can pin the *reduction order*
    /// without re-deriving the exponent.
    fn cell_terms(gt: &OksAnn, dt: &OksAnn, sigmas: &[f64]) -> Vec<f64> {
        let vars: Vec<f64> = sigmas
            .iter()
            .map(|s| {
                let t = 2.0 * s;
                t * t
            })
            .collect();
        let area_norm = gt.area + f64::EPSILON;
        let mut out = Vec::new();
        for (i, (g, d)) in gt
            .keypoints
            .chunks_exact(3)
            .zip(dt.keypoints.chunks_exact(3))
            .enumerate()
        {
            if !is_visible(g[2]) {
                continue;
            }
            let dx = d[0] - g[0];
            let dy = d[1] - g[1];
            out.push((-((dx * dx + dy * dy) / vars[i] / area_norm / 2.0)).exp());
        }
        out
    }

    fn left_fold(a: &[f64]) -> f64 {
        let mut acc = 0.0_f64;
        for &x in a {
            acc += x;
        }
        acc
    }

    /// F9: the default sigma table is `np.array([.26, …]) / 10.0`, not
    /// the decimal literals that divide looks like it produces.
    ///
    /// Five of the seventeen entries disagree by exactly one ULP. This
    /// test names them, so folding the divide back into literals fails
    /// here with the indices in the message rather than surfacing as an
    /// unexplained AP wobble on a keypoints run.
    #[test]
    fn f9_default_sigmas_keep_the_divide_by_ten_as_a_separate_rounding() {
        const RAW: [f64; 17] = [
            0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87,
            0.87, 0.89, 0.89,
        ];
        const FOLDED: [f64; 17] = [
            0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107,
            0.107, 0.087, 0.087, 0.089, 0.089,
        ];

        for i in 0..17 {
            assert_eq!(
                COCO_PERSON_SIGMAS[i].to_bits(),
                (RAW[i] / 10.0).to_bits(),
                "sigma[{i}] is not `{} / 10.0`",
                RAW[i]
            );
        }

        let drifting: Vec<usize> = (0..17)
            .filter(|&i| COCO_PERSON_SIGMAS[i].to_bits() != FOLDED[i].to_bits())
            .collect();
        assert_eq!(
            drifting,
            vec![0, 3, 4, 11, 12],
            "the set of sigmas that the pre-folded literals get wrong has changed"
        );
        // Concretely, at index 0: `.26 / 10.0` is one ULP above `0.026`.
        assert_eq!(
            COCO_PERSON_SIGMAS[0].to_bits(),
            FOLDED[0].to_bits() + 1,
            "expected exactly one ULP of separation at index 0"
        );
    }

    /// F8: the OKS cell reduces its terms with numpy's pairwise sum,
    /// not a running accumulator.
    ///
    /// Terms are recomputed here with the kernel's own exponent
    /// expression — `exp` is explicitly *not* what this pins — and then
    /// reduced two ways. The kernel must agree with the pairwise one and
    /// disagree with the left fold; the second half is what makes the
    /// first half mean something.
    #[test]
    fn f8_cell_reduces_with_numpys_pairwise_sum_not_a_left_fold() {
        // 17 visible keypoints with offsets chosen so the 17 terms are
        // of comparable magnitude — the regime where summation order
        // changes the result.
        let gt_kps: Vec<(f64, f64, u32)> = (0..17).map(|i| (10.0 * i as f64, 7.0, 2)).collect();
        let dt_kps: Vec<(f64, f64, u32)> = (0..17)
            .map(|i| {
                (
                    10.0 * i as f64 + 1.0 + 0.25 * (i % 5) as f64,
                    7.0 + (i % 3) as f64,
                    2,
                )
            })
            .collect();
        let g = ann(1, &gt_kps, [0.0, 0.0, 10.0, 10.0], 137.0);
        let d = ann(1, &dt_kps, [0.0, 0.0, 10.0, 10.0], 137.0);

        let terms = cell_terms(&g, &d, &COCO_PERSON_SIGMAS);
        assert_eq!(terms.len(), 17);
        let pairwise = crate::parity::numpy_pairwise_sum(&terms) / 17.0;
        let folded = left_fold(&terms) / 17.0;
        assert_ne!(
            pairwise.to_bits(),
            folded.to_bits(),
            "premise: this fixture must separate the two reductions"
        );

        let m = compute(&OksSimilarity::default(), &[g], &[d]);
        assert_eq!(m[[0, 0]].to_bits(), pairwise.to_bits());
    }

    /// F8: COCO-WholeBody's 133 keypoints exceed numpy's
    /// `NPY_PW_BLOCKSIZE` of 128, so a real keypoint workload reaches
    /// the *recursive* arm of the pairwise sum — the one a "this is
    /// always short, simplify it" reading would delete. The split is
    /// 64 + 69, not 66 + 67.
    #[test]
    fn f8_wholebody_133_keypoints_reach_the_pairwise_recursion() {
        const K: usize = 133;
        let sigmas: Vec<f64> = (0..K).map(|i| (26.0 + (i % 84) as f64) / 1000.0).collect();
        let gt_kps: Vec<(f64, f64, u32)> = (0..K).map(|i| (3.0 * i as f64, 5.0, 2)).collect();

        // Whether a 66/67 halving happens to round to the same double as
        // numpy's 64/69 one is a coin flip per fixture, so sweep a few
        // detections: every one must equal the 64/69 split, and at least
        // one must separate it from the 66/67 alternative.
        let mut separates_the_split_point = 0_usize;
        let mut separates_a_left_fold = 0_usize;
        for phase in 0..8_usize {
            let dt_kps: Vec<(f64, f64, u32)> = (0..K)
                .map(|i| {
                    (
                        3.0 * i as f64 + 0.45 * ((i * 37 + phase * 5) % 23) as f64,
                        5.0 + 0.3 * ((i * 11 + phase) % 17) as f64,
                        2,
                    )
                })
                .collect();
            let g = ann(9, &gt_kps, [0.0, 0.0, 10.0, 10.0], 4096.0);
            let d = ann(9, &dt_kps, [0.0, 0.0, 10.0, 10.0], 4096.0);

            let mut map = HashMap::new();
            map.insert(9_i64, sigmas.clone());
            let m = compute(
                &OksSimilarity::new(map),
                std::slice::from_ref(&g),
                std::slice::from_ref(&d),
            );

            let terms = cell_terms(&g, &d, &sigmas);
            assert_eq!(terms.len(), K);

            // The value the kernel produced must be the 64 + 69 split.
            let split = crate::parity::numpy_pairwise_sum(&terms[..64])
                + crate::parity::numpy_pairwise_sum(&terms[64..]);
            assert_eq!(
                m[[0, 0]].to_bits(),
                (split / K as f64).to_bits(),
                "phase {phase}"
            );

            if m[[0, 0]].to_bits() != (left_fold(&terms) / K as f64).to_bits() {
                separates_a_left_fold += 1;
            }
            let halved = crate::parity::numpy_pairwise_sum(&terms[..66])
                + crate::parity::numpy_pairwise_sum(&terms[66..]);
            if m[[0, 0]].to_bits() != (halved / K as f64).to_bits() {
                separates_the_split_point += 1;
            }
        }
        assert!(
            separates_a_left_fold > 0,
            "no swept detection separates the pairwise sum from a left fold"
        );
        assert!(
            separates_the_split_point > 0,
            "no swept detection separates numpy's 64/69 split from a 66/67 one; \
             the recursion arm's split point is then untested here"
        );
    }

    /// F10: a keypoint coordinate past the exact-arithmetic bound is a
    /// typed error, not a silently-wrong cell.
    ///
    /// Past `2^25` the `f64` product `dx * dx` stops being exact, so
    /// vernier would quietly diverge from pycocotools' int64 keypoint
    /// arithmetic. Refusing is the honest answer: the bound is ~300x
    /// beyond any real pixel coordinate, so nothing legitimate trips it.
    #[test]
    fn f10_out_of_range_keypoint_coordinate_is_rejected() {
        for (label, bad) in [
            ("over the bound", MAX_ABS_KEYPOINT_COORD + 1.0),
            ("negative, over the bound", -(MAX_ABS_KEYPOINT_COORD + 1.0)),
        ] {
            let mut kps = const_kps(0.0, 0.0, 2);
            kps[3] = (bad, 0.0, 2);
            let g = ann(1, &kps, [0.0, 0.0, 10.0, 10.0], 100.0);
            let d = ann(1, &const_kps(0.0, 0.0, 2), [0.0, 0.0, 10.0, 10.0], 100.0);
            let mut out = Array2::<f64>::zeros((1, 1));
            let err = OksSimilarity::default()
                .compute(&[g], &[d], &mut out.view_mut())
                .unwrap_err();
            match err {
                EvalError::InvalidAnnotation { detail } => {
                    assert!(detail.contains("F10"), "{label}: got {detail}");
                }
                other => panic!("{label}: expected InvalidAnnotation, got {other:?}"),
            }
        }

        // Exactly on the bound is still exact, so it is accepted.
        let mut kps = const_kps(0.0, 0.0, 2);
        kps[3] = (MAX_ABS_KEYPOINT_COORD, 0.0, 2);
        let g = ann(1, &kps, [0.0, 0.0, 10.0, 10.0], 100.0);
        let d = ann(1, &const_kps(0.0, 0.0, 2), [0.0, 0.0, 10.0, 10.0], 100.0);
        let mut out = Array2::<f64>::zeros((1, 1));
        OksSimilarity::default()
            .compute(&[g], &[d], &mut out.view_mut())
            .unwrap();
    }

    /// F10: NaN / infinite coordinates are rejected too. They are not a
    /// precision question — `f64::max(NaN, 0.0)` returns `0.0` while
    /// numpy's `np.max` propagates the NaN, so the **F3** surrogate path
    /// would diverge structurally rather than by a ULP.
    #[test]
    fn f10_non_finite_keypoint_coordinate_is_rejected() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut kps = const_kps(0.0, 0.0, 2);
            kps[0] = (0.0, bad, 2);
            let g = ann(1, &const_kps(0.0, 0.0, 2), [0.0, 0.0, 10.0, 10.0], 100.0);
            let d = ann(1, &kps, [0.0, 0.0, 10.0, 10.0], 100.0);
            let mut out = Array2::<f64>::zeros((1, 1));
            let err = OksSimilarity::default()
                .compute(&[g], &[d], &mut out.view_mut())
                .unwrap_err();
            assert!(
                matches!(err, EvalError::NonFinite { context } if context.contains("keypoint")),
                "expected NonFinite for {bad:?}, got {err:?}"
            );
        }

        // Visibility flags are *not* coordinates and are not rejected:
        // numpy reduces them through the single mask `vg > 0`, which a
        // NaN fails, so the keypoint is dropped from both the count and
        // the terms. [`is_visible`] is that same comparison, so vernier
        // drops it too — and the result is the 16-visible-keypoint cell,
        // not a 17-term sum divided by 16.
        let mut g = ann(1, &const_kps(3.0, 4.0, 2), [0.0, 0.0, 10.0, 10.0], 100.0);
        g.keypoints[2] = f64::NAN;
        let d = ann(1, &const_kps(3.0, 4.0, 2), [0.0, 0.0, 10.0, 10.0], 100.0);
        let mut out = Array2::<f64>::zeros((1, 1));
        OksSimilarity::default()
            .compute(&[g], &[d], &mut out.view_mut())
            .unwrap();
        // 16 exact matches, 16 terms of exactly 1.0, divided by 16.
        assert_eq!(out[[0, 0]], 1.0);
    }

    #[test]
    fn impl_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<OksSimilarity>();
    }
}
