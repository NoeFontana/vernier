//! Generalized **Breakdown** axis — value-typed slice descriptor that
//! replaces the hardcoded COCO area ranges (ADR-0016, *proposed*).
//!
//! ## What this module provides
//!
//! - [`Bucket`] — one slot on a [`Breakdown`]: a `[lo, hi]` closed
//!   range over an annotation-derived `f64` key, paired with a
//!   human-readable label and the `A`-axis position the summarizer
//!   indexes by. (Closed-on-both-ends, per quirks **D6/D7**: see
//!   [`Bucket::contains`].)
//! - [`Breakdown`] — an axis name plus a list of [`Bucket`]s; the
//!   value-typed configuration the orchestrator and summarizer share.
//!
//! ## Why this exists
//!
//! Today the per-image orchestrator binds annotations into A-axis
//! buckets via [`crate::AreaRange`], and the summarizer reads them back
//! out via [`crate::AreaRng`]. The two shapes are coupled (`index` on
//! one matches `index` on the other) but live in different modules and
//! describe the same conceptual axis with different field sets.
//!
//! `Breakdown` unifies them: one value owns the bucket layout, can hand
//! out [`AreaRange`]s for the evaluator and [`AreaRng`]s for the
//! summarizer, and carries an *axis name* — the seed CrowdPose's
//! `crowdIndex` and any future depth/occlusion axes plug into. The
//! constructors [`Breakdown::coco_area_det`] and
//! [`Breakdown::coco_area_keypoints`] reproduce the canonical
//! pycocotools layouts bit-for-bit; everything that runs through them
//! yields the same numbers as the prior hardcoded path.
//!
//! ## Scope
//!
//! Internal-only for now. The FFI/Python boundary keeps the existing
//! `AreaRange` / `AreaRng` shape unchanged so byte-identical parity is
//! preserved. Custom user-defined Breakdowns are usable from Rust today
//! (the `coco_area_det` / `coco_area_keypoints` constructors are
//! examples), but no Python entry point accepts one yet — that is a
//! follow-up ADR (CrowdPose-driven).
//!
//! [`AreaRange`]: crate::AreaRange
//! [`AreaRng`]: crate::AreaRng

use std::borrow::Cow;

use crate::evaluate::{AreaRange, AREA_UNBOUNDED};
use crate::summarize::{AreaRng, MaxDetSelector, Metric, StatRequest};

/// One bucket on a [`Breakdown`].
///
/// A bucket carries everything the per-image orchestrator and the
/// summarizer need to slice a single A-axis cell: the inclusive `[lo,
/// hi]` range used to test annotation membership, the label rendered in
/// the summary table, and the `index` position on the
/// [`crate::Accumulated`] A-axis where the cell's outputs land.
///
/// **Inclusivity.** `[lo, hi]` is closed on *both* ends, mirroring
/// pycocotools' `cocoeval.py:251` predicate `not (area < lo or area >
/// hi)` (quirk **D6** strict). An annotation whose key sits exactly on
/// a boundary lands in *both* adjacent buckets. This is *not* the
/// half-open convention `Range<f64>` would imply — buckets are not a
/// partition, they are an overlap-tolerant cover.
#[derive(Debug, Clone, PartialEq)]
pub struct Bucket {
    /// `A`-axis position. `0` is conventionally the `all` bucket on
    /// area-keyed Breakdowns.
    pub index: usize,
    /// Label rendered by the summarizer (e.g. `"all"`, `"small"`,
    /// `"medium"`, `"large"` for area; `"easy"`, `"hard"` for a future
    /// CrowdPose breakdown).
    pub label: Cow<'static, str>,
    /// Lower bound (inclusive — quirks D6/D7).
    pub lo: f64,
    /// Upper bound (inclusive — quirks D6/D7). Use [`AREA_UNBOUNDED`]
    /// (`1e10`) for "no upper bound"; pycocotools uses exactly that
    /// value for the `all` and `large` buckets.
    pub hi: f64,
}

impl Bucket {
    /// `const`-friendly constructor for compile-time string labels.
    pub const fn from_static(index: usize, label: &'static str, lo: f64, hi: f64) -> Self {
        Self {
            index,
            label: Cow::Borrowed(label),
            lo,
            hi,
        }
    }

    /// Constructor for owned-string labels (e.g., labels built at
    /// runtime from a config file).
    pub fn new(index: usize, label: impl Into<Cow<'static, str>>, lo: f64, hi: f64) -> Self {
        Self {
            index,
            label: label.into(),
            lo,
            hi,
        }
    }

    /// `true` when `key` falls inside `[lo, hi]` (closed on both ends —
    /// quirks D6/D7).
    pub fn contains(&self, key: f64) -> bool {
        key >= self.lo && key <= self.hi
    }

    /// Lift this bucket into the [`AreaRange`] shape the per-image
    /// orchestrator consumes.
    pub fn to_area_range(&self) -> AreaRange {
        AreaRange {
            index: self.index,
            lo: self.lo,
            hi: self.hi,
        }
    }

    /// Lift this bucket into the [`AreaRng`] shape the summarizer
    /// consumes.
    pub fn to_area_rng(&self) -> AreaRng {
        AreaRng::new(self.index, self.label.clone())
    }
}

/// A value-typed slice axis: a name plus a list of [`Bucket`]s.
///
/// `Breakdown` is the kernel summarizer's per-bucket slice descriptor.
/// The two pre-defined constructors [`Breakdown::coco_area_det`] and
/// [`Breakdown::coco_area_keypoints`] reproduce pycocotools' default
/// area-grid layouts; user-defined Breakdowns are constructed via
/// [`Breakdown::new`].
///
/// The `axis` name has no effect on numeric output today — labels on
/// individual buckets are what reach the summary table — but is kept
/// alongside the bucket list so a future schema-bump can surface it
/// (e.g., `"breakdown_axis": "area"` in the CLI's JSON output, ADR-0015).
///
/// ## Invariants (debug-checked at construction)
///
/// - `buckets` is non-empty.
/// - Every `Bucket::index` is unique and lies in `0..buckets.len()`.
///
/// These invariants let downstream consumers ([`Breakdown::area_ranges`],
/// [`Breakdown::summary_areas`], the orchestrator) treat the layout as a
/// dense, gap-free A-axis without re-validating.
#[derive(Debug, Clone, PartialEq)]
pub struct Breakdown {
    axis: Cow<'static, str>,
    buckets: Vec<Bucket>,
}

impl Breakdown {
    /// Construct a breakdown from an axis name and a list of buckets.
    ///
    /// # Panics
    ///
    /// In debug builds, panics if `buckets` is empty or has duplicate
    /// or out-of-range indices. Release builds silently accept the
    /// degenerate input — downstream slicing will produce empty / `-1`
    /// sentinel cells but cannot violate memory safety
    /// (`#![forbid(unsafe_code)]`).
    pub fn new(axis: impl Into<Cow<'static, str>>, buckets: Vec<Bucket>) -> Self {
        let out = Self {
            axis: axis.into(),
            buckets,
        };
        debug_assert!(!out.buckets.is_empty(), "Breakdown must have >= 1 bucket");
        let n = out.buckets.len();
        debug_assert!(
            out.buckets.iter().all(|b| b.index < n),
            "Breakdown bucket index out of range",
        );
        let mut seen = vec![false; n];
        for b in &out.buckets {
            if b.index < n {
                debug_assert!(!seen[b.index], "Breakdown has duplicate bucket index");
                seen[b.index] = true;
            }
        }
        out
    }

    /// The four-bucket COCO area grid for det-family kernels (bbox /
    /// segm / boundary): `all`, `small`, `medium`, `large` with
    /// inclusive `[lo, hi]` ranges `[0, 1e10]`, `[0, 32^2]`, `[32^2,
    /// 96^2]`, `[96^2, 1e10]`.
    ///
    /// Indices line up with the legacy [`AreaRng`] constants: `0 =
    /// all`, `1 = small`, `2 = medium`, `3 = large`. The literal numbers
    /// are pinned by quirk **D4** (strict) and reproduce pycocotools'
    /// `Params.areaRng` bit-for-bit.
    pub fn coco_area_det() -> Self {
        // The literal bound values mirror evaluate.rs::AreaRange::coco_default.
        Self::new(
            "area",
            vec![
                Bucket::from_static(0, "all", 0.0, AREA_UNBOUNDED),
                Bucket::from_static(1, "small", 0.0, 32.0 * 32.0),
                Bucket::from_static(2, "medium", 32.0 * 32.0, 96.0 * 96.0),
                Bucket::from_static(3, "large", 96.0 * 96.0, AREA_UNBOUNDED),
            ],
        )
    }

    /// The three-bucket keypoints area grid: `all`, `medium`, `large`.
    ///
    /// Quirk **D5** (strict, ratified by ADR-0012): pycocotools omits
    /// the `small` bucket for `iouType="keypoints"`. The A-axis is
    /// compressed to three entries — `0 = all`, `1 = medium`, `2 =
    /// large` — matching what the kp summarizer's
    /// [`StatRequest::coco_keypoints_default`] expects.
    pub fn coco_area_keypoints() -> Self {
        Self::new(
            "area",
            vec![
                Bucket::from_static(0, "all", 0.0, AREA_UNBOUNDED),
                Bucket::from_static(1, "medium", 32.0 * 32.0, 96.0 * 96.0),
                Bucket::from_static(2, "large", 96.0 * 96.0, AREA_UNBOUNDED),
            ],
        )
    }

    /// Axis name (e.g., `"area"`).
    pub fn axis(&self) -> &str {
        &self.axis
    }

    /// All buckets, in construction order.
    pub fn buckets(&self) -> &[Bucket] {
        &self.buckets
    }

    /// Number of buckets — the size of the `A`-axis the orchestrator
    /// emits and the accumulator slices on.
    pub fn len(&self) -> usize {
        self.buckets.len()
    }

    /// `true` when [`Self::len`] is `0` (degenerate; only reachable in
    /// release builds with a malformed [`Self::new`] call).
    pub fn is_empty(&self) -> bool {
        self.buckets.is_empty()
    }

    /// Bucket at `A`-axis position `index`, or `None` if absent.
    pub fn bucket_at(&self, index: usize) -> Option<&Bucket> {
        self.buckets.iter().find(|b| b.index == index)
    }

    /// Materialize the [`AreaRange`] slice the per-image orchestrator
    /// consumes via [`crate::EvaluateParams::area_ranges`].
    ///
    /// Cheap (`buckets.len()` allocations) and idempotent — call as
    /// often as needed; cache the result in a local if profiling shows
    /// it on a hot path.
    pub fn area_ranges(&self) -> Vec<AreaRange> {
        self.buckets.iter().map(Bucket::to_area_range).collect()
    }

    /// Materialize the [`AreaRng`] slice the summarizer consumes when
    /// rendering each per-bucket [`crate::StatLine`]. Useful for callers
    /// that build custom `StatRequest` plans against this breakdown's
    /// A-axis layout.
    pub fn summary_areas(&self) -> Vec<AreaRng> {
        self.buckets.iter().map(Bucket::to_area_rng).collect()
    }

    /// Build the canonical 12-row pycocotools detection plan over this
    /// breakdown.
    ///
    /// The breakdown must have the four-bucket layout `(0, 1, 2, 3)`
    /// matching `(all, small, medium, large)`; otherwise this method
    /// returns `None` and the caller falls back to
    /// [`StatRequest::coco_detection_default`] (which assumes the
    /// canonical layout). For breakdowns with non-canonical bucket
    /// counts (e.g., a 5-bucket fine-grained area split), callers
    /// should compose their own plan via
    /// [`StatRequest::new`] + [`Self::summary_areas`].
    ///
    /// Returning `Option` rather than `Result` keeps the surface lean:
    /// the only failure mode is "this breakdown isn't the canonical
    /// detection shape", which the caller resolves by composing a
    /// custom plan, not by surfacing an error.
    pub fn detection_plan(&self) -> Option<[StatRequest; 12]> {
        if self.len() != 4 {
            return None;
        }
        let all = self.bucket_at(0)?.to_area_rng();
        let small = self.bucket_at(1)?.to_area_rng();
        let medium = self.bucket_at(2)?.to_area_rng();
        let large = self.bucket_at(3)?.to_area_rng();
        use MaxDetSelector::{Largest, Value};
        use Metric::{AveragePrecision, AverageRecall};
        Some([
            StatRequest::new(AveragePrecision, None, all.clone(), Largest),
            StatRequest::new(AveragePrecision, Some(0.5), all.clone(), Largest),
            StatRequest::new(AveragePrecision, Some(0.75), all.clone(), Largest),
            StatRequest::new(AveragePrecision, None, small.clone(), Largest),
            StatRequest::new(AveragePrecision, None, medium.clone(), Largest),
            StatRequest::new(AveragePrecision, None, large.clone(), Largest),
            StatRequest::new(AverageRecall, None, all.clone(), Value(1)),
            StatRequest::new(AverageRecall, None, all.clone(), Value(10)),
            StatRequest::new(AverageRecall, None, all, Value(100)),
            StatRequest::new(AverageRecall, None, small, Largest),
            StatRequest::new(AverageRecall, None, medium, Largest),
            StatRequest::new(AverageRecall, None, large, Largest),
        ])
    }

    /// Build the canonical 10-row pycocotools keypoints plan over this
    /// breakdown.
    ///
    /// Mirrors [`StatRequest::coco_keypoints_default`] but pulls the
    /// `(all, medium, large)` labels from this breakdown's buckets at
    /// indices 0/1/2 — useful when a future caller wants the kp plan
    /// shape but with custom labels (e.g., dataset-specific
    /// "small-medium / medium / large" wording).
    ///
    /// Returns `None` unless the breakdown has exactly three buckets
    /// at indices 0/1/2.
    pub fn keypoints_plan(&self) -> Option<[StatRequest; 10]> {
        if self.len() != 3 {
            return None;
        }
        let all = self.bucket_at(0)?.to_area_rng();
        let medium = self.bucket_at(1)?.to_area_rng();
        let large = self.bucket_at(2)?.to_area_rng();
        use MaxDetSelector::Largest;
        use Metric::{AveragePrecision, AverageRecall};
        Some([
            StatRequest::new(AveragePrecision, None, all.clone(), Largest),
            StatRequest::new(AveragePrecision, Some(0.5), all.clone(), Largest),
            StatRequest::new(AveragePrecision, Some(0.75), all.clone(), Largest),
            StatRequest::new(AveragePrecision, None, medium.clone(), Largest),
            StatRequest::new(AveragePrecision, None, large.clone(), Largest),
            StatRequest::new(AverageRecall, None, all.clone(), Largest),
            StatRequest::new(AverageRecall, Some(0.5), all.clone(), Largest),
            StatRequest::new(AverageRecall, Some(0.75), all, Largest),
            StatRequest::new(AverageRecall, None, medium, Largest),
            StatRequest::new(AverageRecall, None, large, Largest),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accumulate::{accumulate, AccumulateParams};
    use crate::evaluate::AreaRange;
    use crate::parity::{iou_thresholds, recall_thresholds, ParityMode};
    use crate::summarize::{summarize_with, Metric};
    use ndarray::{Array4, Array5};

    #[test]
    fn coco_area_det_matches_legacy_area_range_coco_default_bitwise() {
        // Strict-parity contract: the new Breakdown surface must produce
        // identical bound bytes to the legacy `AreaRange::coco_default`.
        // Anything looser silently re-bins annotations near a bucket
        // boundary and breaks quirk D6.
        let bd = Breakdown::coco_area_det();
        let legacy = AreaRange::coco_default();
        let ranges = bd.area_ranges();
        assert_eq!(ranges.len(), legacy.len());
        for (got, want) in ranges.iter().zip(legacy.iter()) {
            assert_eq!(got.index, want.index, "index drift on bucket {}", got.index);
            assert_eq!(
                got.lo.to_bits(),
                want.lo.to_bits(),
                "lo bound drift on bucket {}",
                got.index,
            );
            assert_eq!(
                got.hi.to_bits(),
                want.hi.to_bits(),
                "hi bound drift on bucket {}",
                got.index,
            );
        }
    }

    #[test]
    fn coco_area_keypoints_matches_legacy_keypoints_default_bitwise() {
        let bd = Breakdown::coco_area_keypoints();
        let legacy = AreaRange::keypoints_default();
        let ranges = bd.area_ranges();
        assert_eq!(ranges.len(), legacy.len());
        for (got, want) in ranges.iter().zip(legacy.iter()) {
            assert_eq!(got.index, want.index);
            assert_eq!(got.lo.to_bits(), want.lo.to_bits());
            assert_eq!(got.hi.to_bits(), want.hi.to_bits());
        }
    }

    #[test]
    fn coco_area_det_summary_labels_pin_canonical_strings() {
        let bd = Breakdown::coco_area_det();
        let labels: Vec<&str> = bd.buckets().iter().map(|b| b.label.as_ref()).collect();
        assert_eq!(labels, ["all", "small", "medium", "large"]);
        assert_eq!(bd.axis(), "area");
    }

    #[test]
    fn coco_area_keypoints_drops_small_bucket_per_d5() {
        // D5 / ADR-0012: kp summary has no "small" row. The breakdown's
        // labels and indices encode that — three entries (all, medium,
        // large) at indices (0, 1, 2). A future regression that
        // re-introduces "small" trips this assertion.
        let bd = Breakdown::coco_area_keypoints();
        assert_eq!(bd.len(), 3);
        let labels: Vec<&str> = bd.buckets().iter().map(|b| b.label.as_ref()).collect();
        assert_eq!(labels, ["all", "medium", "large"]);
        assert!(!labels.contains(&"small"));
        // Indices 0..3 dense, no gap — accumulator A-axis is dense.
        let indices: Vec<usize> = bd.buckets().iter().map(|b| b.index).collect();
        assert_eq!(indices, [0, 1, 2]);
    }

    #[test]
    fn bucket_contains_is_inclusive_on_both_ends() {
        // D6 strict: a key equal to either bound lands inside the bucket.
        // Pycocotools' `not (area < lo or area > hi)` is bit-equivalent.
        let small = Bucket::from_static(1, "small", 0.0, 32.0 * 32.0);
        let medium = Bucket::from_static(2, "medium", 32.0 * 32.0, 96.0 * 96.0);
        // Boundary value 1024 belongs to BOTH buckets (overlap-tolerant).
        assert!(small.contains(1024.0));
        assert!(medium.contains(1024.0));
        // Strictly out-of-range values are excluded.
        assert!(!small.contains(-1.0));
        assert!(!medium.contains(1023.999));
        assert!(small.contains(0.0));
        assert!(medium.contains(96.0 * 96.0));
    }

    #[test]
    fn detection_plan_matches_canonical_default_bitwise() {
        // The Breakdown-built detection plan and the static
        // `coco_detection_default` must produce stat-by-stat equal results
        // when summarized over the same Accumulated. This pins the
        // "default Breakdown produces byte-identical output to the prior
        // hardcoded path" invariant.
        let iou = iou_thresholds();
        let max_dets = [1usize, 10, 100];
        let accum = crate::Accumulated {
            precision: Array5::<f64>::from_elem((iou.len(), 101, 1, 4, 3), 0.5),
            recall: Array4::<f64>::from_elem((iou.len(), 1, 4, 3), 0.7),
            scores: Array5::<f64>::from_elem((iou.len(), 101, 1, 4, 3), 1.0),
        };

        let static_plan = StatRequest::coco_detection_default();
        let bd = Breakdown::coco_area_det();
        let bd_plan = bd.detection_plan().expect("4-bucket layout");

        let from_static = summarize_with(&accum, &static_plan, iou, &max_dets).unwrap();
        let from_bd = summarize_with(&accum, &bd_plan, iou, &max_dets).unwrap();

        assert_eq!(from_static.stats(), from_bd.stats());
        // Stat-by-stat: metric, threshold, label, max_dets, value.
        for (s, b) in from_static.lines.iter().zip(from_bd.lines.iter()) {
            assert_eq!(s.metric, b.metric);
            assert_eq!(s.iou_threshold, b.iou_threshold);
            assert_eq!(s.area.label, b.area.label);
            assert_eq!(s.area.index, b.area.index);
            assert_eq!(s.max_dets, b.max_dets);
            assert_eq!(s.value.to_bits(), b.value.to_bits());
        }
    }

    #[test]
    fn keypoints_plan_matches_canonical_default_bitwise() {
        let iou = iou_thresholds();
        let max_dets = [20usize];
        let accum = crate::Accumulated {
            precision: Array5::<f64>::from_elem((iou.len(), 101, 1, 3, 1), 0.5),
            recall: Array4::<f64>::from_elem((iou.len(), 1, 3, 1), 0.7),
            scores: Array5::<f64>::from_elem((iou.len(), 101, 1, 3, 1), 1.0),
        };

        let static_plan = StatRequest::coco_keypoints_default();
        let bd = Breakdown::coco_area_keypoints();
        let bd_plan = bd.keypoints_plan().expect("3-bucket kp layout");

        let from_static = summarize_with(&accum, &static_plan, iou, &max_dets).unwrap();
        let from_bd = summarize_with(&accum, &bd_plan, iou, &max_dets).unwrap();

        assert_eq!(from_static.stats(), from_bd.stats());
        for (s, b) in from_static.lines.iter().zip(from_bd.lines.iter()) {
            assert_eq!(s.area.index, b.area.index);
            assert_eq!(s.area.label, b.area.label);
            assert_eq!(s.value.to_bits(), b.value.to_bits());
        }
    }

    #[test]
    fn detection_plan_returns_none_for_non_canonical_size() {
        // Non-4-bucket breakdowns can't be cast into the canonical
        // 12-stat plan; the helper returns None so callers fall back to
        // composing their own plan rather than getting a silently-wrong
        // shape.
        let bd = Breakdown::coco_area_keypoints(); // 3 buckets
        assert!(bd.detection_plan().is_none());
    }

    #[test]
    fn keypoints_plan_returns_none_for_non_canonical_size() {
        let bd = Breakdown::coco_area_det(); // 4 buckets
        assert!(bd.keypoints_plan().is_none());
    }

    #[test]
    fn fine_grained_five_bucket_breakdown_extends_a_axis() {
        // The headline test: a non-default breakdown plugs in to the
        // pipeline as a leaf change. We add a "tiny" bucket below
        // "small" and a "huge" bucket above "large", run an Accumulated
        // shaped to the 5-bucket A-axis, and assert that the resulting
        // Summary has one line per requested bucket and that each value
        // is what hand-tracing the Accumulated cells predicts.
        let bd = Breakdown::new(
            "area",
            vec![
                Bucket::from_static(0, "tiny", 0.0, 16.0 * 16.0),
                Bucket::from_static(1, "small", 16.0 * 16.0, 32.0 * 32.0),
                Bucket::from_static(2, "medium", 32.0 * 32.0, 96.0 * 96.0),
                Bucket::from_static(3, "large", 96.0 * 96.0, 192.0 * 192.0),
                Bucket::from_static(4, "huge", 192.0 * 192.0, AREA_UNBOUNDED),
            ],
        );
        assert_eq!(bd.len(), 5);

        // Hand-craft an Accumulated whose A-axis matches the 5-bucket
        // layout. Each bucket's precision tensor is set to a distinct
        // value (0.10..0.50) so the per-bucket Summary value pins which
        // slice the summarizer read.
        let iou = iou_thresholds();
        let max_dets = [100usize];
        let mut precision = Array5::<f64>::from_elem((iou.len(), 101, 1, 5, 1), -1.0);
        let mut recall = Array4::<f64>::from_elem((iou.len(), 1, 5, 1), -1.0);
        let scores = Array5::<f64>::from_elem((iou.len(), 101, 1, 5, 1), 1.0);
        for a in 0..5 {
            let pr_val = 0.1 * (a as f64 + 1.0);
            let rc_val = 0.2 * (a as f64 + 1.0);
            for t in 0..iou.len() {
                for r in 0..101 {
                    precision[(t, r, 0, a, 0)] = pr_val;
                }
                recall[(t, 0, a, 0)] = rc_val;
            }
        }
        let accum = crate::Accumulated {
            precision,
            recall,
            scores,
        };

        // Compose a custom plan: AP-wide, no IoU pin, one line per bucket.
        let plan: Vec<StatRequest> = bd
            .buckets()
            .iter()
            .map(|b| {
                StatRequest::new(
                    Metric::AveragePrecision,
                    None,
                    b.to_area_rng(),
                    MaxDetSelector::Largest,
                )
            })
            .collect();

        let summary = summarize_with(&accum, &plan, iou, &max_dets).unwrap();
        assert_eq!(summary.lines.len(), 5, "one line per bucket");

        // Hand-traced expectations: AP for bucket a is the constant
        // we packed into the precision tensor at A-axis position a
        // (the mean over a constant slice is the constant).
        let expected_labels = ["tiny", "small", "medium", "large", "huge"];
        for (a, line) in summary.lines.iter().enumerate() {
            let expected = 0.1 * (a as f64 + 1.0);
            assert_eq!(
                line.area.label.as_ref(),
                expected_labels[a],
                "bucket {a} label drift",
            );
            assert!(
                (line.value - expected).abs() < 1e-12,
                "bucket {a} value: got {got}, expected {expected}",
                got = line.value,
            );
        }

        // The breakdown can also be lifted into the AreaRange slice the
        // orchestrator consumes — the `accumulate` round trip below pins
        // that the A-axis size derived from the breakdown matches the
        // Accumulated tensor's A-axis size, so the orchestrator → accum
        // → summarize pipeline lines up.
        let area_ranges = bd.area_ranges();
        assert_eq!(area_ranges.len(), 5);
        let acc_params = AccumulateParams {
            iou_thresholds: iou,
            recall_thresholds: recall_thresholds(),
            max_dets: &max_dets,
            n_categories: 1,
            n_area_ranges: area_ranges.len(),
            n_images: 0,
        };
        // Empty grid; we just verify the shape contract end-to-end.
        let empty = accumulate(&[], acc_params, ParityMode::Strict).unwrap();
        assert_eq!(empty.precision.shape()[3], 5);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "Breakdown must have >= 1 bucket")]
    fn empty_breakdown_panics_in_debug() {
        let _ = Breakdown::new("axis", vec![]);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "Breakdown bucket index out of range")]
    fn out_of_range_index_panics_in_debug() {
        let _ = Breakdown::new("axis", vec![Bucket::from_static(5, "x", 0.0, 1.0)]);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "Breakdown has duplicate bucket index")]
    fn duplicate_index_panics_in_debug() {
        let _ = Breakdown::new(
            "axis",
            vec![
                Bucket::from_static(0, "a", 0.0, 1.0),
                Bucket::from_static(0, "b", 1.0, 2.0),
            ],
        );
    }
}
