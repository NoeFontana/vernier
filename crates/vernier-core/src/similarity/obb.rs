//! Oriented-box similarity: `RotatedBox` and `Quad` (ADR-0063).
//!
//! Two kernels, three implementations. The kernel a call uses is fixed
//! at construction from the [`ParityMode`], so nothing downstream has to
//! branch on it:
//!
//! | Kernel | `strict` | `corrected` |
//! | --- | --- | --- |
//! | [`RotatedBoxIou`] | detectron2 replica, f32 | canonical f64 |
//! | [`QuadIou`] | DOTA_devkit replica, f64 | canonical f64 |
//!
//! ADR-0008 rejected exactly this kind of parity-mode branch for the
//! bbox kernel, and said why: the alternative there only bought
//! throughput. Here `corrected` has a correctness mandate. detectron2
//! runs its geometry in f32 and can return IoU outside `[0, 1]`
//! (detectron2#350, quirk **OB8**); DOTA_devkit's origin-hinged
//! triangle fan returns non-zero residues on polygon-disjoint pairs. A
//! single kernel would have to inherit one of those defects, so there
//! are two.
//!
//! # What lives here and what does not
//!
//! Geometry lives in `vernier-geom`. This module is the bridge: it
//! turns `(CocoAnnotation, CocoDetection)` into prepared geometry,
//! applies the crowd asymmetry (**E1**), and writes the `[G, D]` f64
//! matrix the ADR-0005 spine expects. No trigonometry, no clipping and
//! no tolerance appears below this line.

use ndarray::ArrayViewMut2;
use vernier_geom::broad::{aabb_overlap_mask, DtEnvelopes, SLACK, SMALL_CELL_THRESHOLD_OBB};
use vernier_geom::kernel::{quad_iou, rbox_iou, Denominator};
use vernier_geom::{replica, Aabb, Convention, GeomError, PreparedQuad, PreparedRBox, RotatedBox};

use super::Similarity;
use crate::error::EvalError;
use crate::parity::ParityMode;

/// Which implementation a parity mode selects.
///
/// Fixed at kernel construction. Carrying it as a field rather than
/// re-deriving it per call is the ADR-0062 lesson applied early: a
/// mapping re-derived at several sites is a mapping that eventually
/// disagrees with itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ObbFlavor {
    /// The canonical f64 kernel — `corrected` for both geometries.
    Canonical,
    /// detectron2's `single_box_iou_rotated<float>` — `strict` for
    /// `RotatedBox`.
    D2Replica,
    /// DOTA_devkit's `iou_poly` composed with its horizontal-box gate —
    /// `strict` for `Quad`.
    DkReplica,
}

impl ObbFlavor {
    /// Flavor for the `RotatedBox` kernel under `mode`.
    #[must_use]
    pub const fn for_rotated_box(mode: ParityMode) -> Self {
        match mode {
            ParityMode::Strict => Self::D2Replica,
            ParityMode::Corrected => Self::Canonical,
        }
    }

    /// Flavor for the `Quad` kernel under `mode`.
    #[must_use]
    pub const fn for_quad(mode: ParityMode) -> Self {
        match mode {
            ParityMode::Strict => Self::DkReplica,
            ParityMode::Corrected => Self::Canonical,
        }
    }

    /// Whether this flavor produces an f32-valued matrix, and therefore
    /// needs the ADR-0063 T1 threshold projection.
    #[must_use]
    pub const fn needs_f32_threshold_ladder(self) -> bool {
        matches!(self, Self::D2Replica)
    }
}

/// Express a rotated box in detectron2's own convention: degrees,
/// counter-clockwise on screen.
///
/// The replica is bit-equal to an oracle that speaks exactly one
/// dialect, so input in another has to be translated before it gets
/// there. Two cases:
///
/// - **Rotation.** `sigma` flips the sign of `theta` and nothing else,
///   and negation is exact, so `screen_cw` input reaches the oracle
///   with no loss.
/// - **Unit.** Radians must be scaled by `180/pi`, which rounds. The
///   pinned `strict` claim is therefore keyed to `unit = "deg"`; with
///   `unit = "rad"` the kernel still runs the oracle's code, but the
///   value it is handed is one rounding away from the user's (quirk
///   **OB20**). Declare degrees if bit-equality is the point.
#[must_use]
pub fn to_d2_params(b: RotatedBox, conv: Convention) -> [f64; 5] {
    let deg = match conv.unit {
        vernier_geom::AngleUnit::Deg => b.theta,
        vernier_geom::AngleUnit::Rad => b.theta.to_degrees(),
    };
    let theta = match conv.rotation {
        vernier_geom::Rotation::ScreenCcw => deg,
        vernier_geom::Rotation::ScreenCw => -deg,
    };
    [b.cx, b.cy, b.w, b.h, theta]
}

/// A ground truth or detection prepared for the rotated-box kernel.
#[derive(Debug, Clone, Copy)]
pub struct RotatedBoxAnn {
    /// Prepared geometry under the caller's convention.
    pub prepared: PreparedRBox,
    /// The same box in detectron2's dialect, for the `strict` replica.
    pub d2_params: [f64; 5],
    /// COCO `iscrowd`. Always `false` on the DT side (quirks **E2** /
    /// **J4**).
    pub is_crowd: bool,
}

impl RotatedBoxAnn {
    /// Build from a raw `[cx, cy, w, h, theta]` payload.
    ///
    /// # Errors
    /// [`GeomError::NonFinite`] when any slot is `NaN` or infinite.
    pub fn new(raw: &[f64; 5], conv: Convention, is_crowd: bool) -> Result<Self, GeomError> {
        let b = RotatedBox::from_slice(raw)?;
        Ok(Self {
            prepared: PreparedRBox::new(b, conv),
            d2_params: to_d2_params(b, conv),
            is_crowd,
        })
    }
}

/// A ground truth or detection prepared for the quad kernel.
#[derive(Debug, Clone, Copy)]
pub struct QuadAnn {
    /// Validated, positively-wound, split into convex pieces. Carries
    /// the submitted payload verbatim for the `strict` replica.
    pub prepared: PreparedQuad,
    /// COCO `iscrowd`.
    pub is_crowd: bool,
}

impl QuadAnn {
    /// Build from a raw `[x0, y0, ..., x3, y3]` payload.
    ///
    /// `strict` tolerates a self-intersecting quad because DK does;
    /// `corrected` rejects it (quirk **OB16**). Zero area is a typed
    /// error in both modes — DK evaluates `0/0` there and the matrix
    /// contract forbids `NaN` (quirk **OB7**).
    ///
    /// # Errors
    /// [`GeomError`] for non-finite coordinates, zero area, or — in
    /// `corrected` — self-intersection.
    pub fn new(raw: &[f64; 8], flavor: ObbFlavor, is_crowd: bool) -> Result<Self, GeomError> {
        let allow_self_intersection = matches!(flavor, ObbFlavor::DkReplica);
        Ok(Self {
            prepared: PreparedQuad::new(raw, allow_self_intersection)?,
            is_crowd,
        })
    }
}

/// Rotated-box IoU. `unit` and `rotation` are required at construction
/// and enter the params fingerprint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RotatedBoxIou {
    /// Angle unit and rotation sign.
    pub conv: Convention,
    /// Which implementation to run.
    pub flavor: ObbFlavor,
}

impl RotatedBoxIou {
    /// Construct from a convention and a parity mode.
    #[must_use]
    pub const fn new(conv: Convention, mode: ParityMode) -> Self {
        Self {
            conv,
            flavor: ObbFlavor::for_rotated_box(mode),
        }
    }
}

/// Quad IoU over four-vertex polygons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QuadIou {
    /// Which implementation to run.
    pub flavor: ObbFlavor,
}

impl QuadIou {
    /// Construct from a parity mode.
    #[must_use]
    pub const fn new(mode: ParityMode) -> Self {
        Self {
            flavor: ObbFlavor::for_quad(mode),
        }
    }
}

/// Shape guard shared by both kernels.
fn check_shape(
    what: &str,
    g: usize,
    d: usize,
    out: &ArrayViewMut2<'_, f64>,
) -> Result<(), EvalError> {
    if out.nrows() == g && out.ncols() == d {
        Ok(())
    } else {
        Err(EvalError::DimensionMismatch {
            detail: format!(
                "{what} output is {}x{}, expected {g}x{d}",
                out.nrows(),
                out.ncols()
            ),
        })
    }
}

/// Crowd ground truth has no oracle behavior to reproduce under
/// `strict`: detectron2 asserts it away (`assert all(c == 0 for c in
/// is_crowd)`) and DOTA has no crowd concept at all. Rather than invent
/// one, `strict` refuses (quirk **OB9**); `corrected` applies the
/// ordinary **E1** asymmetry.
fn crowd_under_strict_err(what: &str) -> EvalError {
    EvalError::InvalidAnnotation {
        detail: format!(
            "{what}: a ground truth is marked iscrowd=1, which no oriented-box \
             oracle defines. detectron2's RotatedCOCOeval asserts crowds are \
             absent and DOTA_devkit has no crowd concept, so parity_mode=\"strict\" \
             has nothing to reproduce. Use parity_mode=\"corrected\" to apply the \
             usual crowd asymmetry (quirk E1), or drop the crowd flag."
        ),
    }
}

/// Denominator for one ground truth: the **E1** crowd asymmetry, read
/// once per row rather than once per pair.
#[inline]
fn denominator(is_crowd: bool) -> Denominator {
    if is_crowd {
        Denominator::CrowdGt
    } else {
        Denominator::Union
    }
}

/// Cell-level upper bound on `vernier_geom::broad::pair_slack`.
///
/// `pair_slack` maxes over one pair's extents and origin magnitudes, so
/// the same max taken over every box in the cell dominates it. Hoisting
/// it turns a per-pair six-way `max` chain into one pass over `G + D`
/// boxes, and a larger pad can only keep pairs — the direction the
/// admissibility contract allows.
fn cell_slack(gts: &[RotatedBoxAnn], dts: &[RotatedBoxAnn]) -> f64 {
    let mut scale = 0.0_f64;
    for ann in gts.iter().chain(dts) {
        let a = &ann.prepared.aabb;
        scale = scale
            .max(a.max_x - a.min_x)
            .max(a.max_y - a.min_y)
            .max(a.min_x.abs())
            .max(a.min_y.abs());
    }
    if scale.is_finite() {
        scale * SLACK
    } else {
        f64::INFINITY
    }
}

/// Broad phase, then narrow phase on the survivors.
///
/// The `G x D` envelope test is the only genuinely quadratic term in
/// the kernel, so above [`SMALL_CELL_THRESHOLD_OBB`] it runs as one
/// branchless, `pulp`-dispatched sweep over the detection envelopes in
/// structure-of-arrays form, and the narrow phase — clipping, or the
/// op-exact replica — runs only where the mask says a pair can overlap.
/// Below that threshold the two allocations cost more than the branch
/// they remove, so the same predicate runs inline.
///
/// `pad` is a *cell-level* margin, computed once from quantities that
/// every per-pair margin is monotone in. `narrow` is called only for
/// surviving pairs; every other entry is written as exactly `+0.0`,
/// which is what the prefilter's admissibility contract promises the
/// kernel would have returned.
fn broad_then_narrow<A>(
    gts: &[A],
    dts: &[A],
    pad: f64,
    out: &mut ArrayViewMut2<'_, f64>,
    aabb: impl Fn(&A) -> Aabb,
    narrow: impl Fn(&A, &A) -> f64,
) {
    let (n_g, n_d) = (gts.len(), dts.len());
    if n_g * n_d < SMALL_CELL_THRESHOLD_OBB {
        for (gi, gt) in gts.iter().enumerate() {
            let ga = aabb(gt);
            let mut row = out.row_mut(gi);
            for (di, dt) in dts.iter().enumerate() {
                row[di] = if ga.overlaps(&aabb(dt), pad) {
                    narrow(gt, dt)
                } else {
                    0.0
                };
            }
        }
        return;
    }

    let gt_aabbs: Vec<Aabb> = gts.iter().map(&aabb).collect();
    let dt_aabbs: Vec<Aabb> = dts.iter().map(&aabb).collect();
    let envelopes = DtEnvelopes::new(&dt_aabbs);
    let mut mask = vec![0_u8; n_g * n_d];
    // The only failure mode is a shape disagreement, which the caller
    // has already ruled out. Keeping every pair rather than emitting a
    // matrix of zeros is the safe reading if that ever stops holding.
    if !aabb_overlap_mask(&gt_aabbs, &envelopes, pad, &mut mask) {
        mask.fill(1);
    }
    for (gi, gt) in gts.iter().enumerate() {
        let keep = &mask[gi * n_d..(gi + 1) * n_d];
        let mut row = out.row_mut(gi);
        for (di, dt) in dts.iter().enumerate() {
            row[di] = if keep[di] == 1 { narrow(gt, dt) } else { 0.0 };
        }
    }
}

impl Similarity for RotatedBoxIou {
    type Annotation = RotatedBoxAnn;

    fn compute(
        &self,
        gts: &[Self::Annotation],
        dts: &[Self::Annotation],
        out: &mut ArrayViewMut2<'_, f64>,
    ) -> Result<(), EvalError> {
        check_shape("rotated-box IoU", gts.len(), dts.len(), out)?;
        if gts.is_empty() || dts.is_empty() {
            return Ok(());
        }
        // Both checks are properties of the kernel instance and the GT
        // slice, never of a pair, so neither belongs in the `G x D`
        // loop. Dispatching on the flavor once also lets each body be
        // straight-line: the previous shape re-tested a
        // construction-time constant per pair, two of whose three arms
        // could not fire.
        if self.flavor != ObbFlavor::Canonical && gts.iter().any(|g| g.is_crowd) {
            return Err(crowd_under_strict_err("rotated-box IoU"));
        }
        match self.flavor {
            ObbFlavor::Canonical => {
                let conv = self.conv;
                broad_then_narrow(
                    gts,
                    dts,
                    cell_slack(gts, dts),
                    out,
                    |a| a.prepared.aabb,
                    |g, d| rbox_iou(&g.prepared, &d.prepared, conv, denominator(g.is_crowd)),
                );
                Ok(())
            }
            ObbFlavor::D2Replica => {
                // Admissible prefilter: outside the oracle's own
                // epsilon reach it returns exactly +0.0, so skipping is
                // not an approximation (ADR-0063 open question 10,
                // resolved as padded-AABB). The margin is bounded once
                // for the cell rather than recomputed per pair; see
                // `replica::d2::ReachBound`.
                let mut bound = replica::d2::ReachBound::new();
                for ann in gts.iter().chain(dts) {
                    bound.add(&ann.d2_params);
                }
                broad_then_narrow(
                    gts,
                    dts,
                    bound.pad(),
                    out,
                    |a| a.prepared.aabb,
                    // Argument order is load-bearing: detectron2 calls
                    // `pairwise_iou_rotated(dt, gt)`.
                    |g, d| replica::d2::iou(&d.d2_params, &g.d2_params),
                );
                Ok(())
            }
            // A rotated box has no quad payload to hand DK.
            ObbFlavor::DkReplica => Err(EvalError::InvalidConfig {
                detail: "the DOTA_devkit replica is the Quad kernel's \
                         strict oracle and cannot serve RotatedBox"
                    .into(),
            }),
        }
    }
}

impl Similarity for QuadIou {
    type Annotation = QuadAnn;

    fn compute(
        &self,
        gts: &[Self::Annotation],
        dts: &[Self::Annotation],
        out: &mut ArrayViewMut2<'_, f64>,
    ) -> Result<(), EvalError> {
        check_shape("quad IoU", gts.len(), dts.len(), out)?;
        if gts.is_empty() || dts.is_empty() {
            return Ok(());
        }

        if self.flavor != ObbFlavor::Canonical && gts.iter().any(|g| g.is_crowd) {
            return Err(crowd_under_strict_err("quad IoU"));
        }
        // Dispatched once, for the same reason the rotated-box kernel
        // dispatches once. There is no envelope prefilter over the
        // pair loop here: the canonical path already carries an exact
        // one inside `quad_intersection`, and the DK path's gate is
        // DOTA_devkit's own Pascal-VOC `+1` test, which is *part of the
        // composed oracle* rather than an optimization (quirk
        // **OB17**) and therefore has to stay inside `replica::dk::iou`
        // where the oracle puts it.
        match self.flavor {
            ObbFlavor::Canonical => {
                for (g, gt) in gts.iter().enumerate() {
                    let denom = denominator(gt.is_crowd);
                    let mut row = out.row_mut(g);
                    for (d, dt) in dts.iter().enumerate() {
                        row[d] = quad_iou(&gt.prepared, &dt.prepared, denom);
                    }
                }
                Ok(())
            }
            ObbFlavor::DkReplica => {
                for (g, gt) in gts.iter().enumerate() {
                    let mut row = out.row_mut(g);
                    for (d, dt) in dts.iter().enumerate() {
                        row[d] = replica::dk::iou(&gt.prepared.raw, &dt.prepared.raw);
                    }
                }
                Ok(())
            }
            ObbFlavor::D2Replica => Err(EvalError::InvalidConfig {
                detail: "the detectron2 replica is the RotatedBox kernel's \
                         strict oracle and cannot serve Quad"
                    .into(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The broad phase is an *optimization*, and the only thing that
    /// makes that true is admissibility: a pair it rejects must be one
    /// the kernel would have scored `+0.0` anyway. Asserted over random
    /// cells that straddle the small/large dispatch boundary, so both
    /// the scalar path and the `pulp`-dispatched mask are exercised.
    ///
    /// The reference side is genuinely unfiltered for the D2 replica —
    /// `replica::d2::iou` tests nothing before it clips. For the
    /// canonical kernel it is `rbox_iou`, which carries its own tighter
    /// envelope and SAT tests inside; what the comparison pins there is
    /// that the *outer* cell-level mask rejects nothing the inner
    /// per-pair tests would have kept.
    ///
    /// This is also the regression test for the shape of `compute`:
    /// hoisting the flavor branch and replacing the per-pair
    /// `replica::d2::reach` with a cell-level bound measured 1.3x-11x
    /// faster on sparse cells and neutral when every pair overlaps, and
    /// it is only a speedup if the numbers do not move.
    #[test]
    fn the_broad_phase_never_changes_a_number() {
        use ndarray::Array2;

        let mut seed = 0x0123_4567_89ab_cdef_u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            ((seed >> 11) as f64) * (1.0 / 9_007_199_254_740_992.0)
        };
        let mut make = |n: usize, spread: f64| -> Vec<RotatedBoxAnn> {
            (0..n)
                .map(|_| {
                    let raw = [
                        next() * spread,
                        next() * spread,
                        20.0 + next() * 80.0,
                        8.0 + next() * 20.0,
                        next() * 360.0 - 180.0,
                    ];
                    RotatedBoxAnn::new(&raw, Convention::D2, false).expect("finite")
                })
                .collect()
        };

        // Below and above `SMALL_CELL_THRESHOLD_OBB`, and at two
        // densities: `spread` large means most pairs are disjoint (the
        // mask earns its keep), small means almost none are (the mask
        // rejects nothing and must still not lie).
        for (n_g, n_d, spread) in [
            (3, 4, 400.0),
            (3, 4, 40.0),
            (17, 23, 2000.0),
            (17, 23, 60.0),
        ] {
            let gts = make(n_g, spread);
            let dts = make(n_d, spread);
            let mut got = Array2::<f64>::zeros((n_g, n_d));

            for flavor in [ObbFlavor::Canonical, ObbFlavor::D2Replica] {
                let kernel = RotatedBoxIou {
                    conv: Convention::D2,
                    flavor,
                };
                kernel
                    .compute(&gts, &dts, &mut got.view_mut())
                    .expect("compute");

                for (g, gt) in gts.iter().enumerate() {
                    for (d, dt) in dts.iter().enumerate() {
                        // No prefilter of any kind on this side.
                        let want = match flavor {
                            ObbFlavor::D2Replica => replica::d2::iou(&dt.d2_params, &gt.d2_params),
                            _ => rbox_iou(
                                &gt.prepared,
                                &dt.prepared,
                                Convention::D2,
                                Denominator::Union,
                            ),
                        };
                        assert_eq!(
                            got[(g, d)].to_bits(),
                            want.to_bits(),
                            "{flavor:?} cell {n_g}x{n_d} spread {spread}: pair ({g}, {d})"
                        );
                    }
                }
            }
        }
    }
    use ndarray::Array2;
    use vernier_geom::{AngleUnit, Rotation};

    const D2: Convention = Convention::D2;

    fn ann(raw: [f64; 5], crowd: bool) -> RotatedBoxAnn {
        RotatedBoxAnn::new(&raw, D2, crowd).unwrap()
    }

    fn qann(raw: [f64; 8], flavor: ObbFlavor, crowd: bool) -> QuadAnn {
        QuadAnn::new(&raw, flavor, crowd).unwrap()
    }

    fn matrix<S: Similarity>(k: &S, gts: &[S::Annotation], dts: &[S::Annotation]) -> Array2<f64> {
        let mut m = Array2::zeros((gts.len(), dts.len()));
        k.compute(gts, dts, &mut m.view_mut()).unwrap();
        m
    }

    #[test]
    fn canonical_rotated_box_matrix() {
        let k = RotatedBoxIou::new(D2, ParityMode::Corrected);
        let gts = [ann([0.0, 0.0, 2.0, 2.0, 0.0], false)];
        let dts = [
            ann([0.0, 0.0, 2.0, 2.0, 0.0], false),
            ann([1.0, 0.0, 2.0, 2.0, 0.0], false),
            ann([50.0, 50.0, 2.0, 2.0, 0.0], false),
        ];
        let m = matrix(&k, &gts, &dts);
        assert_eq!(m[[0, 0]], 1.0);
        assert_eq!(m[[0, 1]], 2.0 / 6.0);
        assert_eq!(m[[0, 2]], 0.0);
    }

    #[test]
    fn strict_rotated_box_matches_the_replica_directly() {
        let k = RotatedBoxIou::new(D2, ParityMode::Strict);
        let g = [0.0, 0.0, 6.0, 3.0, 20.0];
        let d = [1.0, 0.5, 5.0, 4.0, -10.0];
        let m = matrix(&k, &[ann(g, false)], &[ann(d, false)]);
        assert_eq!(m[[0, 0]], replica::d2::iou(&d, &g));
    }

    #[test]
    fn strict_and_corrected_disagree_but_only_slightly() {
        let g = [10.0, 10.0, 8.0, 3.0, 30.0];
        let d = [11.0, 9.0, 6.0, 4.0, -20.0];
        let strict = matrix(
            &RotatedBoxIou::new(D2, ParityMode::Strict),
            &[ann(g, false)],
            &[ann(d, false)],
        );
        let corrected = matrix(
            &RotatedBoxIou::new(D2, ParityMode::Corrected),
            &[ann(g, false)],
            &[ann(d, false)],
        );
        let (a, b) = (strict[[0, 0]], corrected[[0, 0]]);
        assert_ne!(a, b, "f32 and f64 geometry should not agree to the bit");
        assert!((a - b).abs() < 1e-6, "{a} vs {b}");
    }

    #[test]
    fn crowd_is_refused_under_strict_and_applied_under_corrected() {
        let gts = [ann([0.0, 0.0, 100.0, 100.0, 0.0], true)];
        let dts = [ann([1.0, 1.0, 2.0, 2.0, 0.0], false)];
        let strict = RotatedBoxIou::new(D2, ParityMode::Strict);
        let mut m = Array2::zeros((1, 1));
        assert!(matches!(
            strict.compute(&gts, &dts, &mut m.view_mut()),
            Err(EvalError::InvalidAnnotation { .. })
        ));
        let corrected = RotatedBoxIou::new(D2, ParityMode::Corrected);
        assert_eq!(matrix(&corrected, &gts, &dts)[[0, 0]], 1.0);
    }

    #[test]
    fn screen_cw_is_an_exact_negation_for_the_replica() {
        let cw = Convention::new(AngleUnit::Deg, Rotation::ScreenCw);
        let b = RotatedBox {
            cx: 3.0,
            cy: 4.0,
            w: 6.0,
            h: 2.0,
            theta: 35.0,
        };
        assert_eq!(to_d2_params(b, cw), [3.0, 4.0, 6.0, 2.0, -35.0]);
        assert_eq!(to_d2_params(b, D2), [3.0, 4.0, 6.0, 2.0, 35.0]);
    }

    #[test]
    fn both_conventions_describe_the_same_geometry() {
        // A box at +35 under screen_cw is the same shape as one at -35
        // under screen_ccw, so the IoU against a shared reference must
        // agree bit for bit.
        let cw = Convention::new(AngleUnit::Deg, Rotation::ScreenCw);
        let g_cw = RotatedBoxAnn::new(&[0.0, 0.0, 6.0, 2.0, 35.0], cw, false).unwrap();
        let d_cw = RotatedBoxAnn::new(&[1.0, 1.0, 5.0, 3.0, 10.0], cw, false).unwrap();
        let g_ccw = RotatedBoxAnn::new(&[0.0, 0.0, 6.0, 2.0, -35.0], D2, false).unwrap();
        let d_ccw = RotatedBoxAnn::new(&[1.0, 1.0, 5.0, 3.0, -10.0], D2, false).unwrap();
        let a = matrix(
            &RotatedBoxIou::new(cw, ParityMode::Corrected),
            &[g_cw],
            &[d_cw],
        );
        let b = matrix(
            &RotatedBoxIou::new(D2, ParityMode::Corrected),
            &[g_ccw],
            &[d_ccw],
        );
        assert_eq!(a[[0, 0]], b[[0, 0]]);
    }

    #[test]
    fn quad_kernels_agree_on_a_simple_pair() {
        let g = [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0];
        let d = [1.0, 0.0, 3.0, 0.0, 3.0, 2.0, 1.0, 2.0];
        for mode in [ParityMode::Strict, ParityMode::Corrected] {
            let k = QuadIou::new(mode);
            let m = matrix(&k, &[qann(g, k.flavor, false)], &[qann(d, k.flavor, false)]);
            assert!(
                (m[[0, 0]] - 2.0 / 6.0).abs() < 1e-12,
                "{mode:?}: {}",
                m[[0, 0]]
            );
        }
    }

    #[test]
    fn quad_strict_applies_the_hbb_gate() {
        // Far apart: the composed oracle short-circuits to exactly zero
        // rather than letting the fan produce a residue.
        let g = [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0];
        let d = [500.0, 500.0, 502.0, 500.0, 502.0, 502.0, 500.0, 502.0];
        let k = QuadIou::new(ParityMode::Strict);
        let m = matrix(&k, &[qann(g, k.flavor, false)], &[qann(d, k.flavor, false)]);
        assert_eq!(m[[0, 0]].to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn self_intersecting_quads_are_mode_dependent() {
        let bowtie = [0.0, 0.0, 4.0, 4.0, 4.0, 0.0, 0.0, 1.0];
        assert!(QuadAnn::new(&bowtie, ObbFlavor::Canonical, false).is_err());
        assert!(QuadAnn::new(&bowtie, ObbFlavor::DkReplica, false).is_ok());
    }

    #[test]
    fn zero_area_quads_are_refused_in_both_modes() {
        let flat = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
        for flavor in [ObbFlavor::Canonical, ObbFlavor::DkReplica] {
            assert!(QuadAnn::new(&flat, flavor, false).is_err(), "{flavor:?}");
        }
    }

    #[test]
    fn shape_mismatch_is_reported() {
        let k = RotatedBoxIou::new(D2, ParityMode::Corrected);
        let gts = [ann([0.0, 0.0, 2.0, 2.0, 0.0], false)];
        let dts = [ann([0.0, 0.0, 2.0, 2.0, 0.0], false)];
        let mut m = Array2::zeros((2, 2));
        assert!(matches!(
            k.compute(&gts, &dts, &mut m.view_mut()),
            Err(EvalError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn empty_cells_are_no_ops() {
        let k = RotatedBoxIou::new(D2, ParityMode::Corrected);
        let mut m = Array2::zeros((0, 0));
        assert!(k.compute(&[], &[], &mut m.view_mut()).is_ok());
    }

    #[test]
    fn only_the_d2_replica_wants_the_f32_ladder() {
        assert!(ObbFlavor::D2Replica.needs_f32_threshold_ladder());
        assert!(!ObbFlavor::DkReplica.needs_f32_threshold_ladder());
        assert!(!ObbFlavor::Canonical.needs_f32_threshold_ladder());
    }
}
