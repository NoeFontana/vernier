//! Axis-aligned bbox IoU.
//!
//! Mirrors `pycocotools.cocoeval.COCOeval.computeIoU` for `iouType="bbox"`.
//! Per ADR-0008, intermediates are `f64` end-to-end so the kernel matches
//! pycocotools' `maskUtils.iou` (also f64) bit-for-bit. Per ADR-0003, the
//! inner loop is wrapped in [`pulp::Arch::dispatch`] so it compiles to
//! AVX2 / AVX-512 / NEON variants picked at process start.
//!
//! ## Quirk dispositions
//!
//! - **E1** (`strict`): when GT is a crowd region, IoU is asymmetric —
//!   `intersect / dt_area`, *not* `intersect / union`. A small DT inside
//!   a large crowd scores 1.0. The asymmetry lives here so that
//!   matching code stays IoU-type-agnostic (per ADR-0005).
//! - **I3** (`aligned`): pycocotools uses two different zero guards
//!   (`u==0` for RLE, `w<=0 || h<=0` for bbox). Both yield IoU=0; we
//!   express it as a single `denom > 0` guard at the last step.
//! - **I4** (`strict`): edge-sharing boxes (e.g. `[0,0,1,1]` and
//!   `[1,0,1,1]`) yield zero IoU. Falls out of the `(min - max).max(0)`
//!   intersection formula automatically.
//!
//! Quirks **E2** and **J4** (DT `iscrowd` is always 0) are enforced at
//! the future `loadRes`-equivalent on the dataset side, not here. The
//! `dts` slice this kernel receives may carry an `is_crowd` field for
//! storage symmetry, but it is ignored — only `gts[g].is_crowd` drives
//! the asymmetric branch.

use std::sync::OnceLock;

use ndarray::ArrayViewMut2;

use super::Similarity;
use crate::dataset::Bbox;
use crate::error::EvalError;

/// Process-wide [`pulp::Arch`] cache. `Arch::new()` runs `cpuid`-gated
/// feature detection; per ADR-0003 we construct it once and reuse it on
/// every kernel call rather than paying the detect cost per `(image,
/// category)` cell.
fn arch() -> &'static pulp::Arch {
    static ARCH: OnceLock<pulp::Arch> = OnceLock::new();
    ARCH.get_or_init(pulp::Arch::new)
}

/// Skip the [`pulp::Arch::dispatch`] closure when `gts.len() *
/// dts.len()` is below this. The dispatch boundary's fixed cost
/// dominates per-call wall time on cells with tiny inner loops, and
/// extracting the loop body as a free `#[inline(always)]` function
/// also changes LLVM's unroll heuristics inside the closure for cells
/// near the threshold — both effects are quantified in the divan
/// `production_sparse` arm. A threshold of 32 keeps the fast path
/// active across the whole regime where it materially helps
/// (G·D ≤ 25) and falls through to the dispatched path well before
/// AVX-512 vectorization headroom (on CPUs that have it) starts to
/// matter.
const SMALL_CELL_THRESHOLD: usize = 32;

/// Stage-0 instrumentation: process-global histogram of `(kernel,
/// g_count, d_count, wall_ns)` per call. Off by default; enabled at
/// compile time via the `bench-histogram` feature for the bbox-IoU
/// optimization plan's measurement runs (see
/// `docs/engineering/benchmarking/`).
///
/// Never compiled into shipped wheels — when the feature is off this
/// module is absent and the kernels carry zero overhead. Crate-private
/// because the public surface lives at
/// [`crate::dump_bbox_iou_histogram_csv`]; callers shouldn't depend on
/// the internal module path.
#[cfg(feature = "bench-histogram")]
pub(crate) mod histogram {
    use std::path::Path;
    use std::sync::Mutex;
    use std::time::Instant;

    /// Kernel discriminator for histogram records. `FullIou` is the
    /// standalone bbox-IoU path; `OverlapMask` is the segm/boundary
    /// survivor-bit prefilter.
    #[derive(Clone, Copy)]
    pub(crate) enum KernelKind {
        FullIou,
        OverlapMask,
    }

    impl KernelKind {
        /// Self-describing label written into the `kind` CSV column.
        fn label(self) -> &'static str {
            match self {
                Self::FullIou => "FullIou",
                Self::OverlapMask => "OverlapMask",
            }
        }
    }

    #[derive(Clone, Copy)]
    struct Record {
        kind: KernelKind,
        g: u32,
        d: u32,
        wall_ns: u64,
    }

    static RECORDS: Mutex<Vec<Record>> = Mutex::new(Vec::new());

    /// Drop-on-end timer. Constructed at the top of each kernel call;
    /// records `(kind, g, d, wall_ns)` into the global buffer when it
    /// drops at end-of-scope.
    pub(super) struct CallTimer {
        kind: KernelKind,
        g: u32,
        d: u32,
        start: Instant,
    }

    impl CallTimer {
        pub(super) fn new(kind: KernelKind, g: usize, d: usize) -> Self {
            Self {
                kind,
                g: u32::try_from(g).unwrap_or(u32::MAX),
                d: u32::try_from(d).unwrap_or(u32::MAX),
                start: Instant::now(),
            }
        }
    }

    impl Drop for CallTimer {
        fn drop(&mut self) {
            let elapsed = self.start.elapsed().as_nanos();
            let wall_ns = u64::try_from(elapsed).unwrap_or(u64::MAX);
            let record = Record {
                kind: self.kind,
                g: self.g,
                d: self.d,
                wall_ns,
            };
            if let Ok(mut records) = RECORDS.lock() {
                records.push(record);
            }
        }
    }

    /// Write all recorded calls to `path` as CSV (header
    /// `kind,g,d,wall_ns`; `kind` is the variant name `FullIou` or
    /// `OverlapMask`), then clear the buffer so subsequent runs start
    /// fresh. Returns the number of records written.
    pub(crate) fn dump_csv(path: &Path) -> std::io::Result<usize> {
        use std::io::Write;
        let mut records = RECORDS.lock().unwrap_or_else(|p| p.into_inner());
        let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
        writeln!(file, "kind,g,d,wall_ns")?;
        for r in records.iter() {
            writeln!(file, "{},{},{},{}", r.kind.label(), r.g, r.d, r.wall_ns)?;
        }
        let n = records.len();
        records.clear();
        file.flush()?;
        Ok(n)
    }

    /// Number of records currently buffered. Test-only hook for the
    /// smoke test.
    #[cfg(test)]
    pub(super) fn len() -> usize {
        let records = RECORDS.lock().unwrap_or_else(|p| p.into_inner());
        records.len()
    }
}

/// Annotation shape consumed by [`BboxIou`]. The matching engine
/// constructs these from a concrete [`crate::dataset::CocoAnnotation`]
/// (or any future [`crate::dataset::EvalDataset`] impl) before invoking
/// [`Similarity::compute`].
///
/// Kept deliberately minimal: only the fields the kernel actually reads.
/// Other metadata (image_id, category_id, area, score) flows through
/// the matching engine's parallel arrays, not through here.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BboxAnn {
    /// Axis-aligned bounding box (COCO `(x, y, w, h)` convention).
    pub bbox: Bbox,
    /// Crowd flag. Drives the E1 asymmetry on the GT side; ignored on
    /// the DT side.
    pub is_crowd: bool,
}

/// Bbox IoU [`Similarity`] impl. Stateless.
#[derive(Debug, Default, Clone, Copy)]
pub struct BboxIou;

impl Similarity for BboxIou {
    type Annotation = BboxAnn;

    fn compute(
        &self,
        gts: &[BboxAnn],
        dts: &[BboxAnn],
        out: &mut ArrayViewMut2<'_, f64>,
    ) -> Result<(), EvalError> {
        if out.nrows() != gts.len() || out.ncols() != dts.len() {
            return Err(EvalError::DimensionMismatch {
                detail: format!(
                    "bbox IoU output is {}x{}, expected {}x{}",
                    out.nrows(),
                    out.ncols(),
                    gts.len(),
                    dts.len()
                ),
            });
        }
        if gts.is_empty() || dts.is_empty() {
            return Ok(());
        }

        #[cfg(feature = "bench-histogram")]
        let _guard =
            histogram::CallTimer::new(histogram::KernelKind::FullIou, gts.len(), dts.len());

        // `dispatch` runs the closure with the best-available SIMD
        // target features enabled, so LLVM auto-vectorizes the inner
        // loop across AVX2 / AVX-512 / NEON without per-arch source
        // duplication. The crowd flag (E1) is hoisted to the outer
        // loop so each inner pass is branch-free FMA-chain math.
        //
        // For cells smaller than `SMALL_CELL_THRESHOLD` the dispatch
        // boundary itself dominates, so call the inner body directly
        // — same algorithm, bit-identical output (no FMA introduced),
        // no parity contract change.
        if gts.len().saturating_mul(dts.len()) < SMALL_CELL_THRESHOLD {
            full_iou_inner(gts, dts, out);
        } else {
            arch().dispatch(|| full_iou_inner(gts, dts, out));
        }

        Ok(())
    }
}

#[inline(always)]
fn full_iou_inner(gts: &[BboxAnn], dts: &[BboxAnn], out: &mut ArrayViewMut2<'_, f64>) {
    for (g, gt) in gts.iter().enumerate() {
        let gxa = gt.bbox.x;
        let gya = gt.bbox.y;
        let gw = gt.bbox.w;
        let gh = gt.bbox.h;
        let gxb = gxa + gw;
        let gyb = gya + gh;
        let g_area = gw * gh;

        let mut row = out.row_mut(g);
        if gt.is_crowd {
            for (d, dt) in dts.iter().enumerate() {
                row[d] = iou_pair(gxa, gya, gxb, gyb, dt.bbox, CrowdDenom);
            }
        } else {
            for (d, dt) in dts.iter().enumerate() {
                row[d] = iou_pair(gxa, gya, gxb, gyb, dt.bbox, UnionDenom(g_area));
            }
        }
    }
}

impl BboxIou {
    /// Survivor-bit prefilter for the segm and boundary IoU kernels.
    ///
    /// Writes `1.0` to `out[[g, d]]` iff `gts[g]` and `dts[d]` have a
    /// strictly positive bbox intersection, `0.0` otherwise. Drops the
    /// area multiplication and `vdivpd` from the inner loop — only
    /// `vminpd` / `vmaxpd` / `vsubpd` plus the comparison remain.
    ///
    /// **Crowd-agnostic by design.** For the segm/boundary prefilter,
    /// the only bit the consumer reads is "did the pair survive the
    /// zero-gate?" That bit is independent of the E1 asymmetric
    /// denominator:
    ///
    /// - Non-crowd: `inter > 0 ⇒ both areas > 0 ⇒ denom > 0`, so
    ///   `iou_pair > 0 ⇔ inter > 0`.
    /// - Crowd: `denom = d_area`, and `inter > 0 ⇒ inter ≤ d_area ⇒
    ///   d_area > 0`, so again `iou_pair > 0 ⇔ inter > 0`.
    ///
    /// Quirk **I4** (edge-sharing → zero) still falls out automatically:
    /// when `gxb == dxa`, `iw = 0` and the `> 0.0` test is false, so
    /// the cell receives the `0.0` sentinel.
    ///
    /// **Not a [`Similarity`] trait method.** The operation is
    /// bbox-specific (the sentinel output shape doesn't generalize to
    /// other IoU types) and intentionally not exposed: the standalone
    /// `iouType="bbox"` eval keeps [`Similarity::compute`] for its real
    /// f64 IoU values.
    ///
    /// Same shape contract as [`Similarity::compute`]: `out` must be
    /// `gts.len() x dts.len()`.
    pub(super) fn compute_overlap_mask(
        &self,
        gts: &[BboxAnn],
        dts: &[BboxAnn],
        out: &mut ArrayViewMut2<'_, f64>,
    ) -> Result<(), EvalError> {
        if out.nrows() != gts.len() || out.ncols() != dts.len() {
            return Err(EvalError::DimensionMismatch {
                detail: format!(
                    "bbox overlap-mask output is {}x{}, expected {}x{}",
                    out.nrows(),
                    out.ncols(),
                    gts.len(),
                    dts.len()
                ),
            });
        }
        if gts.is_empty() || dts.is_empty() {
            return Ok(());
        }

        #[cfg(feature = "bench-histogram")]
        let _guard =
            histogram::CallTimer::new(histogram::KernelKind::OverlapMask, gts.len(), dts.len());

        // Same small-cell fast path as `compute`. The val2017 segm
        // prefilter median is `G·D = 1`, so the bulk of calls land
        // here.
        if gts.len().saturating_mul(dts.len()) < SMALL_CELL_THRESHOLD {
            overlap_mask_inner(gts, dts, out);
        } else {
            arch().dispatch(|| overlap_mask_inner(gts, dts, out));
        }

        Ok(())
    }
}

#[inline(always)]
fn overlap_mask_inner(gts: &[BboxAnn], dts: &[BboxAnn], out: &mut ArrayViewMut2<'_, f64>) {
    for (g, gt) in gts.iter().enumerate() {
        let gxa = gt.bbox.x;
        let gya = gt.bbox.y;
        let gxb = gxa + gt.bbox.w;
        let gyb = gya + gt.bbox.h;

        let mut row = out.row_mut(g);
        for (d, dt) in dts.iter().enumerate() {
            let dxa = dt.bbox.x;
            let dya = dt.bbox.y;
            let dxb = dxa + dt.bbox.w;
            let dyb = dya + dt.bbox.h;

            let iw = gxb.min(dxb) - gxa.max(dxa);
            let ih = gyb.min(dyb) - gya.max(dya);
            row[d] = if iw > 0.0 && ih > 0.0 { 1.0 } else { 0.0 };
        }
    }
}

/// Marker trait for the E1 crowd branch hoisted out of the inner loop.
///
/// Crowd GT uses the asymmetric `intersect / dt_area`; non-crowd GT uses
/// the symmetric `intersect / (g_area + d_area - intersect)`. Choosing
/// once per GT row keeps each inner loop branch-free.
trait Denom: Copy {
    fn denom(self, d_area: f64, inter: f64) -> f64;
}

#[derive(Clone, Copy)]
struct CrowdDenom;
impl Denom for CrowdDenom {
    #[inline(always)]
    fn denom(self, d_area: f64, _inter: f64) -> f64 {
        d_area
    }
}

#[derive(Clone, Copy)]
struct UnionDenom(f64);
impl Denom for UnionDenom {
    #[inline(always)]
    fn denom(self, d_area: f64, inter: f64) -> f64 {
        self.0 + d_area - inter
    }
}

#[inline(always)]
fn iou_pair<D: Denom>(gxa: f64, gya: f64, gxb: f64, gyb: f64, dt: Bbox, denom: D) -> f64 {
    let dxa = dt.x;
    let dya = dt.y;
    let dw = dt.w;
    let dh = dt.h;
    let dxb = dxa + dw;
    let dyb = dya + dh;
    let d_area = dw * dh;

    // Quirk I4: edge-sharing → zero. `(min - max).max(0)` gives 0 when
    // the boxes touch on a side rather than overlap.
    let iw = (gxb.min(dxb) - gxa.max(dxa)).max(0.0);
    let ih = (gyb.min(dyb) - gya.max(dya)).max(0.0);
    let inter = iw * ih;

    let denom = denom.denom(d_area, inter);
    // Quirk I3: single zero-denominator guard.
    if denom > 0.0 {
        inter / denom
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn make_ann(x: f64, y: f64, w: f64, h: f64, is_crowd: bool) -> BboxAnn {
        BboxAnn {
            bbox: Bbox { x, y, w, h },
            is_crowd,
        }
    }

    fn compute(gts: &[BboxAnn], dts: &[BboxAnn]) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((gts.len(), dts.len()));
        BboxIou.compute(gts, dts, &mut out.view_mut()).unwrap();
        out
    }

    #[test]
    fn perfect_overlap_is_one() {
        let gts = [make_ann(0.0, 0.0, 10.0, 10.0, false)];
        let dts = [make_ann(0.0, 0.0, 10.0, 10.0, false)];
        let m = compute(&gts, &dts);
        assert_eq!(m[[0, 0]].to_bits(), 1.0_f64.to_bits());
    }

    #[test]
    fn no_overlap_is_zero() {
        let gts = [make_ann(0.0, 0.0, 1.0, 1.0, false)];
        let dts = [make_ann(10.0, 10.0, 1.0, 1.0, false)];
        let m = compute(&gts, &dts);
        assert_eq!(m[[0, 0]].to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn i4_edge_sharing_is_zero() {
        // Quirk I4: boxes that share an edge but do not overlap have
        // zero IoU. `[0,0,1,1]` and `[1,0,1,1]` touch at x=1.
        let gts = [make_ann(0.0, 0.0, 1.0, 1.0, false)];
        let dts = [make_ann(1.0, 0.0, 1.0, 1.0, false)];
        let m = compute(&gts, &dts);
        assert_eq!(m[[0, 0]].to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn quarter_overlap_matches_hand_traced_value() {
        // GT [0,0,2,2] (area 4); DT [1,1,2,2] (area 4); intersect 1×1=1.
        // IoU = 1 / (4 + 4 - 1) = 1/7, bit-equal to f64 1/7 (ADR-0008).
        let gts = [make_ann(0.0, 0.0, 2.0, 2.0, false)];
        let dts = [make_ann(1.0, 1.0, 2.0, 2.0, false)];
        let m = compute(&gts, &dts);
        let expected = 1.0_f64 / 7.0_f64;
        assert_eq!(m[[0, 0]].to_bits(), expected.to_bits());
    }

    #[test]
    fn e1_crowd_gt_uses_dt_area_denominator() {
        // GT covers the whole image as a crowd; DT is a 1×1 inside it.
        // Symmetric IoU = 1/100 = 0.01. Crowd IoU = inter/dt_area = 1/1
        // = 1.0. The asymmetry is the test.
        let gts_crowd = [make_ann(0.0, 0.0, 10.0, 10.0, true)];
        let gts_normal = [make_ann(0.0, 0.0, 10.0, 10.0, false)];
        let dts = [make_ann(2.0, 2.0, 1.0, 1.0, false)];
        let crowd_m = compute(&gts_crowd, &dts);
        let normal_m = compute(&gts_normal, &dts);
        assert_eq!(crowd_m[[0, 0]].to_bits(), 1.0_f64.to_bits());
        let expected_normal = 1.0_f64 / 100.0_f64;
        assert_eq!(normal_m[[0, 0]].to_bits(), expected_normal.to_bits());
    }

    #[test]
    fn dt_iscrowd_flag_is_ignored() {
        // Quirks E2/J4: DT iscrowd is enforced 0 at load. Even if the
        // caller smuggles `is_crowd: true` into a DT, the kernel must
        // not branch on it — only GT.is_crowd drives the E1 asymmetry.
        //
        // E3 cross-ref: there is no DT-side iscrowd vector by type
        // construction — `Similarity::compute` takes a single GT slice
        // plus a single DT slice, with no parallel `dt_iscrowd` array.
        // The asymmetry of pycocotools' `iou()` API is enforced
        // structurally, so this runtime test covers the observable
        // behavior; no separate type-signature assertion is needed.
        let gts = [make_ann(0.0, 0.0, 2.0, 2.0, false)];
        let dts_marked = [make_ann(1.0, 1.0, 2.0, 2.0, true)];
        let dts_clean = [make_ann(1.0, 1.0, 2.0, 2.0, false)];
        let with_flag = compute(&gts, &dts_marked);
        let without = compute(&gts, &dts_clean);
        assert_eq!(with_flag[[0, 0]].to_bits(), without[[0, 0]].to_bits());
    }

    #[test]
    fn zero_area_gt_with_zero_inter_yields_zero_not_nan() {
        // Degenerate GT (w=0). g_area = 0, inter = 0, union = 0 + d_area
        // - 0 = d_area > 0. Returns 0.0, never NaN. Quirk I3.
        let gts = [make_ann(5.0, 5.0, 0.0, 5.0, false)];
        let dts = [make_ann(0.0, 0.0, 10.0, 10.0, false)];
        let m = compute(&gts, &dts);
        assert!(m[[0, 0]].is_finite());
        assert_eq!(m[[0, 0]].to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn zero_area_gt_and_dt_both_zero_yields_zero_via_denom_guard() {
        // Degenerate on both sides: g_area = d_area = inter = 0 so
        // denom = 0. The I3 single-guard returns 0, not NaN.
        let gts = [make_ann(5.0, 5.0, 0.0, 0.0, false)];
        let dts = [make_ann(5.0, 5.0, 0.0, 0.0, false)];
        let m = compute(&gts, &dts);
        assert_eq!(m[[0, 0]].to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn dimension_mismatch_returns_typed_error() {
        let gts = [make_ann(0.0, 0.0, 1.0, 1.0, false); 2];
        let dts = [make_ann(0.0, 0.0, 1.0, 1.0, false); 3];
        let mut out = Array2::<f64>::zeros((1, 1));
        let err = BboxIou
            .compute(&gts, &dts, &mut out.view_mut())
            .unwrap_err();
        match err {
            EvalError::DimensionMismatch { detail } => {
                assert!(detail.contains("2"));
                assert!(detail.contains("3"));
            }
            other => panic!("expected DimensionMismatch, got {other:?}"),
        }
    }

    #[test]
    fn empty_inputs_return_unchanged_matrix() {
        // 0 × 3 and 3 × 0 are valid: nothing to compute. The matrix
        // shape just needs to match.
        let dts = [make_ann(0.0, 0.0, 1.0, 1.0, false); 3];
        let mut out = Array2::<f64>::from_elem((0, 3), 7.0);
        BboxIou.compute(&[], &dts, &mut out.view_mut()).unwrap();
        assert_eq!(out.shape(), &[0, 3]);
    }

    #[test]
    fn three_by_three_matrix_all_pairs_evaluated() {
        let gts = [
            make_ann(0.0, 0.0, 2.0, 2.0, false),
            make_ann(5.0, 5.0, 2.0, 2.0, false),
            make_ann(0.0, 0.0, 10.0, 10.0, true),
        ];
        let dts = [
            make_ann(0.0, 0.0, 2.0, 2.0, false),
            make_ann(1.0, 1.0, 2.0, 2.0, false),
            make_ann(20.0, 20.0, 1.0, 1.0, false),
        ];
        let m = compute(&gts, &dts);

        assert_eq!(m[[0, 0]].to_bits(), 1.0_f64.to_bits());
        assert_eq!(m[[0, 1]].to_bits(), (1.0_f64 / 7.0_f64).to_bits());
        assert_eq!(m[[0, 2]].to_bits(), 0.0_f64.to_bits());

        assert_eq!(m[[1, 0]].to_bits(), 0.0_f64.to_bits());
        assert_eq!(m[[1, 1]].to_bits(), 0.0_f64.to_bits());
        assert_eq!(m[[1, 2]].to_bits(), 0.0_f64.to_bits());

        assert_eq!(m[[2, 0]].to_bits(), 1.0_f64.to_bits());
        assert_eq!(m[[2, 1]].to_bits(), 1.0_f64.to_bits());
        assert_eq!(m[[2, 2]].to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn impl_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<BboxIou>();
    }

    fn overlap_mask(gts: &[BboxAnn], dts: &[BboxAnn]) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((gts.len(), dts.len()));
        BboxIou
            .compute_overlap_mask(gts, dts, &mut out.view_mut())
            .unwrap();
        out
    }

    #[test]
    fn overlap_mask_writes_only_zero_or_one_sentinels() {
        // Exhaust the variants the segm/boundary prefilter sees: perfect,
        // partial, no overlap, edge-sharing (I4), zero-area GT (I3) and
        // crowd GT (E1). Every cell must end up bit-equal to 0.0 or 1.0.
        let gts = [
            make_ann(0.0, 0.0, 2.0, 2.0, false),
            make_ann(5.0, 5.0, 2.0, 2.0, false),
            make_ann(0.0, 0.0, 10.0, 10.0, true),
            make_ann(5.0, 5.0, 0.0, 5.0, false), // zero-width GT
        ];
        let dts = [
            make_ann(0.0, 0.0, 2.0, 2.0, false),
            make_ann(1.0, 1.0, 2.0, 2.0, false),
            make_ann(2.0, 0.0, 1.0, 1.0, false), // edge-sharing with gt0
            make_ann(20.0, 20.0, 1.0, 1.0, false),
        ];
        let m = overlap_mask(&gts, &dts);
        let zero = 0.0_f64.to_bits();
        let one = 1.0_f64.to_bits();
        for g in 0..gts.len() {
            for d in 0..dts.len() {
                let bits = m[[g, d]].to_bits();
                assert!(
                    bits == zero || bits == one,
                    "overlap_mask[{g},{d}] = {} ({:#x}); expected 0.0 or 1.0",
                    m[[g, d]],
                    bits,
                );
            }
        }
    }

    #[test]
    fn overlap_mask_survivor_bit_matches_full_iou() {
        // Pins the algebraic claim that `compute_overlap_mask` and
        // `Similarity::compute` agree on the survivor set: for every
        // cell, `mask > 0` iff `iou > 0`. The IoU value is allowed to
        // diverge (the mask writes a 1.0 sentinel; iou writes the real
        // number) — only the bit matters for the segm/boundary gate.
        let gts = [
            make_ann(0.0, 0.0, 2.0, 2.0, false),
            make_ann(5.0, 5.0, 2.0, 2.0, false),
            make_ann(0.0, 0.0, 10.0, 10.0, true),
            make_ann(5.0, 5.0, 0.0, 5.0, false),
            make_ann(5.0, 5.0, 0.0, 0.0, false), // fully degenerate GT
        ];
        let dts = [
            make_ann(0.0, 0.0, 2.0, 2.0, false),
            make_ann(1.0, 1.0, 2.0, 2.0, false),
            make_ann(2.0, 0.0, 1.0, 1.0, false),
            make_ann(20.0, 20.0, 1.0, 1.0, false),
            make_ann(5.0, 5.0, 0.0, 0.0, false),
        ];
        let iou = compute(&gts, &dts);
        let mask = overlap_mask(&gts, &dts);
        for g in 0..gts.len() {
            for d in 0..dts.len() {
                let iou_pos = iou[[g, d]] > 0.0;
                let mask_pos = mask[[g, d]] > 0.0;
                assert_eq!(
                    iou_pos,
                    mask_pos,
                    "survivor-bit mismatch at ({g},{d}): iou={}, mask={}",
                    iou[[g, d]],
                    mask[[g, d]],
                );
            }
        }
    }

    #[test]
    fn overlap_mask_dimension_mismatch_returns_typed_error() {
        let gts = [make_ann(0.0, 0.0, 1.0, 1.0, false); 2];
        let dts = [make_ann(0.0, 0.0, 1.0, 1.0, false); 3];
        let mut out = Array2::<f64>::zeros((1, 1));
        let err = BboxIou
            .compute_overlap_mask(&gts, &dts, &mut out.view_mut())
            .unwrap_err();
        match err {
            EvalError::DimensionMismatch { detail } => {
                assert!(detail.contains("2"));
                assert!(detail.contains("3"));
            }
            other => panic!("expected DimensionMismatch, got {other:?}"),
        }
    }

    #[test]
    fn overlap_mask_empty_inputs_return_unchanged_matrix() {
        let dts = [make_ann(0.0, 0.0, 1.0, 1.0, false); 3];
        let mut out = Array2::<f64>::from_elem((0, 3), 7.0);
        BboxIou
            .compute_overlap_mask(&[], &dts, &mut out.view_mut())
            .unwrap();
        assert_eq!(out.shape(), &[0, 3]);
    }

    #[cfg(feature = "bench-histogram")]
    #[test]
    fn histogram_records_kernel_calls_when_feature_on() {
        // Smoke test for the Stage-0 instrumentation: verifies the
        // CallTimer guard fires on both kernel paths and the CSV dump
        // round-trips with the documented schema. Other bbox tests run
        // in parallel and also push records into the same global
        // buffer, so we only assert monotonic growth — not exact
        // counts. The kind labels (`FullIou`, `OverlapMask`) are part
        // of the dump-CSV contract; pin them here so a future rename
        // breaks the test instead of silently breaking downstream
        // post-processors.
        use super::histogram;

        let gts = [make_ann(0.0, 0.0, 2.0, 2.0, false)];
        let dts = [make_ann(1.0, 1.0, 2.0, 2.0, false)];

        let _ = compute(&gts, &dts);
        let _ = overlap_mask(&gts, &dts);
        assert!(histogram::len() >= 2);

        let tmp = std::env::temp_dir().join("vernier-bench-histogram-smoke.csv");
        let n = histogram::dump_csv(&tmp).expect("dump_csv should succeed");
        assert!(n >= 2);
        let csv = std::fs::read_to_string(&tmp).expect("dumped file should be readable");
        assert!(csv.starts_with("kind,g,d,wall_ns\n"));
        assert!(csv.lines().count() > n);
        assert!(
            csv.contains("FullIou,"),
            "expected FullIou rows in CSV: {csv}"
        );
        assert!(
            csv.contains("OverlapMask,"),
            "expected OverlapMask rows in CSV: {csv}"
        );
        std::fs::remove_file(&tmp).ok();
    }
}
