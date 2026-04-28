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
//! - **F2** (`aligned`): area normaliser uses `gt.area + f64::EPSILON`
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
//! - **F5** (`aligned`): empty `gts` or `dts` returns the zero-shape
//!   matrix unchanged. Mirrors the segm/bbox kernels.
//!
//! Quirk **D2** (DT keypoint visibility flags are unconstrained at the
//! dataset boundary) is a *dataset* concern enforced by `loadRes`-equivalent
//! code, not by this kernel. The OKS expression only reads DT
//! `(x_d, y_d)` pairs and never branches on `v_d`.

use std::collections::HashMap;

use ndarray::ArrayViewMut2;

use super::Similarity;
use crate::error::EvalError;

/// Default `kpt_oks_sigmas` for COCO-person (already scaled by `1/10`,
/// matching what pycocotools applies as `kpt_oks_sigmas`).
///
/// Source: `pycocotools.cocoeval.Params.setKpParams` divides the raw
/// table by 10 once at construction; users of the Rust kernel pass the
/// post-divide values directly so we do not double-divide.
pub const COCO_PERSON_SIGMAS: [f64; 17] = [
    0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107,
    0.087, 0.087, 0.089, 0.089,
];

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
            }
        }

        for (g, gt) in gts.iter().enumerate() {
            let sigmas = self.sigmas_for(gt.category_id);
            let k = sigmas.len();
            // vars[i] = (2 * sigma_i)^2; precomputed once per GT row.
            // `k` is tiny (17 typical) and the alloc is dwarfed by
            // the per-cell exp(); no need for a SmallVec.
            let vars: Vec<f64> = sigmas.iter().map(|s| (2.0 * s).powi(2)).collect();
            let area_norm = gt.area + f64::EPSILON;
            let k1 = gt.keypoints.chunks_exact(3).filter(|t| t[2] > 0.0).count();

            // F4: asymmetric bbox expansion. Lower bound subtracts one
            // width / height; upper bound adds two. Pycocotools verbatim.
            let [bx, by, bw, bh] = gt.bbox;
            let (x0, x1) = (bx - bw, bx + 2.0 * bw);
            let (y0, y1) = (by - bh, by + 2.0 * bh);

            // Denominator is fixed per GT row: `k1` visible terms on the
            // standard path, `k` total terms on the F3 surrogate path.
            // Hoisted out of the DT loop so cells that share the same
            // row don't re-derive it. Falls back to 1 only as a guard
            // for the (degenerate) `k == 0` config.
            let denom_count = if k1 > 0 { k1 } else { k };
            if denom_count == 0 {
                for d in 0..dts.len() {
                    out[[g, d]] = 0.0;
                }
                continue;
            }
            let inv_denom = 1.0 / (denom_count as f64);

            for (d, dt) in dts.iter().enumerate() {
                let mut e_sum = 0.0_f64;

                if k1 > 0 {
                    // Standard path: only visible GT keypoints contribute.
                    for (i, (gt_t, dt_t)) in gt
                        .keypoints
                        .chunks_exact(3)
                        .zip(dt.keypoints.chunks_exact(3))
                        .enumerate()
                    {
                        if gt_t[2] <= 0.0 {
                            continue;
                        }
                        let dx = dt_t[0] - gt_t[0];
                        let dy = dt_t[1] - gt_t[1];
                        let e = (dx * dx + dy * dy) / vars[i] / area_norm / 2.0;
                        e_sum += (-e).exp();
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
                        e_sum += (-e).exp();
                    }
                }

                out[[g, d]] = e_sum * inv_denom;
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

    #[test]
    fn impl_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<OksSimilarity>();
    }
}
