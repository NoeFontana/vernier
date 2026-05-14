//! Slice-and-aggregate partition orchestrator (ADR-0046).
//!
//! Implements option C3 of the ADR: the locked spine's matching engine
//! runs once and produces dense `eval_imgs` cells; this module filters
//! them into per-slice subsets and invokes [`accumulate`] +
//! [`summarize_with`] / [`summarize_detection`] for each slice. The
//! ADR-0005-locked spine is untouched — partition is the image-axis
//! analogue of ADR-0026's subset-at-summarize-time pattern for the
//! K-axis (LVIS).
//!
//! ## Performance: filtered-flatten, not zero-out
//!
//! Per slice we allocate a fresh `Vec<Option<Box<PerImageEval>>>` and
//! `clone` only the cells whose I-axis index belongs to the slice's
//! image set. Cells outside the slice land at `None` — `accumulate`
//! treats them the same as cells the orchestrator never emitted, which
//! is the same coupling pycocotools' `evalImgs[None]` slots use.
//!
//! The alternative — mutating one shared dense vec in place to "zero
//! out" non-slice cells per pass — was rejected: at LVIS scale the
//! grid is ~95M `Option<Box>` slots and the per-pass walk would be
//! `O(K * A * I)` independent of slice size. Filtered-flatten pays
//! `O(slice_image_count * K * A)` clones, which dominates only when
//! slices cover most of the dataset (in which case the user has
//! defeated the point of slicing anyway).
//!
//! ## What this module does not decide
//!
//! - **Manifest parsing.** That lives in [`crate::manifest`]; this
//!   module consumes a fully resolved [`PartitionSpec`].
//! - **The summary plan.** AP / keypoints plans come from
//!   [`crate::summarize`]; this module only orchestrates loops over
//!   subsets of `eval_imgs`.
//! - **LRP partitioning.** A sibling [`evaluate_partitioned_lrp`]
//!   function lives in the LRP module — its decomposition path is
//!   different enough that sharing the loop here would be a false
//!   economy.

use std::collections::{HashMap, HashSet};

use crate::accumulate::{accumulate, AccumulateParams, PerImageEval};
use crate::dataset::ImageId;
use crate::error::EvalError;
use crate::parity::{recall_thresholds, ParityMode};
use crate::summarize::{summarize_detection, summarize_with, StatRequest, Summary};

/// Reserved slice value for any dataset key not present in the
/// manifest. Materialized as one row per axis in every partitioned
/// result. Per the ADR-0046 "no silent data loss" discipline, dataset
/// images that the manifest omits land here rather than vanishing.
pub const UNASSIGNED: &str = "__unassigned__";

/// Separator joining axis names / values inside a cross-product slice
/// label. Chosen so it does not appear in axis names produced by the
/// JSON / CSV manifest parser (which restricts both axis names and
/// values to non-empty strings without colon runs).
pub const CROSS_SEPARATOR: &str = "::";

/// Hard cap on the total number of slices a single [`PartitionSpec`]
/// may carry — summed across per-axis marginals, `__unassigned__`
/// buckets, and joint cells.
///
/// Cheap insurance against a manifest typo that produces an
/// exponential cross-product. Callers that need more should split the
/// partition into multiple runs.
pub const SLICES_CAP: usize = 256;

/// Discriminator on the manifest's primary key.
///
/// `Image` manifests drive [`evaluate_partitioned`] — each row maps a
/// dataset image id to axis values. `Result` manifests drive the
/// `vernier aggregate` fan-in (CLI + Python `vernier.aggregate`) —
/// each row maps a result document's label to axis values, and this
/// module is not its consumer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyKind {
    /// Manifest keys are dataset image ids.
    Image,
    /// Manifest keys are result-document labels.
    Result,
}

/// One slice in a [`PartitionSpec`].
///
/// `axis` is the manifest column name for a marginal (`"weather"`) or
/// the [`CROSS_SEPARATOR`]-joined tuple for a joint cell
/// (`"weather::time_of_day"`). `value` is the categorical level for a
/// marginal (`"fog"`) or the joined tuple (`"fog::night"`).
/// `image_ids` is the canonical set of dataset image ids the manifest
/// assigned to this slice (retained for diagnostics and Arrow output).
/// `image_indices` is the resolution into the live grid's I-axis;
/// populated at [`PartitionSpec::build`] time and consumed by
/// [`evaluate_partitioned`].
#[derive(Debug, Clone)]
pub struct Slice {
    /// Manifest column name, or cross-tuple join.
    pub axis: String,
    /// Categorical level, or cross-tuple join.
    pub value: String,
    /// Image ids assigned to this slice. Includes the dataset images
    /// the manifest could resolve; missing ids are warned about during
    /// manifest parsing.
    pub image_ids: HashSet<ImageId>,
    /// Resolved I-axis positions for the grid the spec was built
    /// against. Empty slices are legal: their summary lines all
    /// resolve to the `-1.0` sentinel (no detections, no non-ignore
    /// GTs in this subset).
    pub image_indices: HashSet<usize>,
}

/// Partition specification consumed by [`evaluate_partitioned`].
///
/// Slice order is deterministic: marginals first in
/// `(axis ascending, value ascending, __unassigned__ last)` order;
/// cross-product slices follow in the same canonical order applied
/// to the joined `axis` / `value` strings.
#[derive(Debug, Clone)]
pub struct PartitionSpec {
    /// What the manifest's primary key references.
    pub key_kind: KeyKind,
    /// All slices, marginals before joint cells, in the canonical
    /// ordering described on [`PartitionSpec`].
    pub slices: Vec<Slice>,
}

impl PartitionSpec {
    /// Build from per-axis marginal maps plus optional cross-product
    /// axis tuples, resolving image ids against the live grid's image
    /// set in one pass.
    ///
    /// `per_axis[axis][value]` is the set of image ids the manifest
    /// assigned to that `(axis, value)` cell. The function adds an
    /// `__unassigned__` slice on every axis whose union does not cover
    /// `all_image_ids` and orders the final [`Self::slices`] vector.
    /// `cross_axes` names tuples of axis names whose joint cells should
    /// be expanded (E2 opt-in); pass an empty slice for marginals
    /// only.
    ///
    /// # Errors
    ///
    /// - [`EvalError::InvalidConfig`] when a cross-axis tuple names an
    ///   axis missing from `per_axis`, when an axis tuple is shorter
    ///   than two axes (a "cross" of a single axis is a marginal —
    ///   reject so the caller catches a malformed `--cross` flag),
    ///   when the resolved slice count exceeds [`SLICES_CAP`], or when
    ///   any axis name / value contains the [`CROSS_SEPARATOR`].
    pub fn build(
        key_kind: KeyKind,
        per_axis: &HashMap<String, HashMap<String, HashSet<ImageId>>>,
        all_image_ids: &HashSet<ImageId>,
        image_id_to_idx: &HashMap<ImageId, usize>,
        cross_axes: &[Vec<String>],
    ) -> Result<Self, EvalError> {
        validate_cross_axes(per_axis, cross_axes)?;

        let mut marginal_slices: Vec<Slice> = Vec::new();
        let mut axes_sorted: Vec<&String> = per_axis.keys().collect();
        axes_sorted.sort();

        for axis in axes_sorted {
            // `validate_cross_axes` above has already walked
            // `per_axis.keys()` so every key resolves; the lookup here
            // is infallible in practice but stays defensive.
            let values = match per_axis.get(axis) {
                Some(v) => v,
                None => continue,
            };
            let mut value_keys: Vec<&String> = values.keys().collect();
            value_keys.sort();
            let mut covered: HashSet<ImageId> = HashSet::new();
            for value in &value_keys {
                let ids = values.get(*value).cloned().unwrap_or_default();
                covered.extend(ids.iter().copied());
                marginal_slices.push(make_slice(axis, value, ids, image_id_to_idx));
            }
            let missing: HashSet<ImageId> = all_image_ids
                .iter()
                .copied()
                .filter(|id| !covered.contains(id))
                .collect();
            marginal_slices.push(make_slice(axis, UNASSIGNED, missing, image_id_to_idx));
        }

        let mut joint_slices: Vec<Slice> = Vec::new();
        for axes in cross_axes {
            joint_slices.extend(expand_cross_axes(
                axes,
                per_axis,
                all_image_ids,
                image_id_to_idx,
            )?);
        }
        // Joint slices have predictable axis names already (CROSS_SEPARATOR
        // join); sort them by (axis, value) so cross-product output order
        // is stable across runs.
        joint_slices.sort_by(|a, b| a.axis.cmp(&b.axis).then_with(|| a.value.cmp(&b.value)));

        let total = marginal_slices.len() + joint_slices.len();
        if total > SLICES_CAP {
            return Err(EvalError::InvalidConfig {
                detail: format!(
                    "partition would produce {total} slices but the cap is {SLICES_CAP}; \
                     reduce --cross axes or narrow the manifest"
                ),
            });
        }

        let mut slices = marginal_slices;
        slices.extend(joint_slices);
        // Marginals are already canonically ordered by the per-axis
        // walk; the joint sort above preserves their relative order.
        // No further sort needed.

        Ok(Self { key_kind, slices })
    }

    /// Number of slices, including `__unassigned__` buckets and joint
    /// cells.
    pub fn len(&self) -> usize {
        self.slices.len()
    }

    /// `true` when no axes were declared (i.e., the partition would
    /// emit only `overall`).
    pub fn is_empty(&self) -> bool {
        self.slices.is_empty()
    }
}

/// Summary attached to one slice of a [`PartitionedSummary`].
#[derive(Debug, Clone)]
pub struct SliceResult {
    /// The slice this summary describes. `image_ids` is preserved
    /// alongside `image_indices` so the Arrow / JSON projections can
    /// surface the manifest-assigned set.
    pub slice: Slice,
    /// Summary lines for this slice, in the plan order.
    pub summary: Summary,
    /// Number of dataset images assigned to this slice (= len of
    /// `slice.image_ids`). This matches the manifest's reported
    /// assignment, not the in-grid resolution — out-of-dataset ids in
    /// the manifest are warned about during parsing and do not count.
    pub n_images: u64,
    /// Total detections across every `(category, area)` cell whose
    /// image index falls in `slice.image_indices`. Counted once per
    /// detection (only the `a=0` "all" bucket contributes), matching
    /// the per-image count semantics the existing per_image table
    /// uses.
    pub n_detections: u64,
}

/// Result of [`evaluate_partitioned`].
///
/// `overall` is bit-identical to a non-partitioned [`summarize_detection`]
/// (or [`summarize_with`] under a custom plan) over the same grid —
/// this is the load-bearing parity contract of ADR-0046, since
/// pycocotools has no slicing notion to compare against.
#[derive(Debug, Clone)]
pub struct PartitionedSummary {
    /// Un-partitioned summary; bit-identical to today's path.
    pub overall: Summary,
    /// Total dataset image count behind `overall`.
    pub overall_n_images: u64,
    /// Total detection count behind `overall`.
    pub overall_n_detections: u64,
    /// One entry per `spec.slices` element, in the spec's order.
    pub slices: Vec<SliceResult>,
}

/// How [`evaluate_partitioned`] summarizes each accumulator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SummaryKind {
    /// Standard COCO 12-stat detection plan at `max_dets=[1, 10, 100]`.
    DetectionDefault,
    /// Keypoints 10-stat plan at `max_dets=[20]` (ADR-0012).
    KeypointsDefault,
}

impl SummaryKind {
    fn max_dets(self) -> &'static [usize] {
        match self {
            Self::DetectionDefault => &DETECTION_MAX_DETS,
            Self::KeypointsDefault => &KEYPOINTS_MAX_DETS,
        }
    }
}

const DETECTION_MAX_DETS: [usize; 3] = [1, 10, 100];
const KEYPOINTS_MAX_DETS: [usize; 1] = [20];

/// Grid dimensions consumed by [`evaluate_partitioned`].
///
/// Matches the axis lengths on [`crate::evaluate::EvalGrid`]; the
/// caller copies them across rather than passing the full grid so the
/// orchestrator stays decoupled from `EvalGrid`'s extra retention
/// state.
#[derive(Debug, Clone, Copy)]
pub struct GridDims {
    /// `K`: number of categories on the K-axis.
    pub n_categories: usize,
    /// `A`: number of area ranges on the A-axis.
    pub n_area_ranges: usize,
    /// `I`: number of images on the I-axis.
    pub n_images: usize,
}

/// Run the partitioned accumulation + summarize loop over an already-
/// matched evaluation grid.
///
/// The caller is responsible for producing `eval_imgs` from
/// [`crate::evaluate::evaluate_with`] (or a paradigm-specific entry
/// point). This function does not touch matching — option C3 ratifies
/// that matching runs once before this loop.
///
/// # Errors
///
/// Propagates [`EvalError`] from the underlying [`accumulate`] and
/// summarize calls. Returns [`EvalError::DimensionMismatch`] when
/// `eval_imgs.len()` does not equal `grid.n_categories *
/// grid.n_area_ranges * grid.n_images`.
pub fn evaluate_partitioned(
    eval_imgs: &[Option<Box<PerImageEval>>],
    grid: GridDims,
    spec: &PartitionSpec,
    iou_thresholds: &[f64],
    parity_mode: ParityMode,
    summary_kind: SummaryKind,
) -> Result<PartitionedSummary, EvalError> {
    let expected = grid.n_categories * grid.n_area_ranges * grid.n_images;
    if eval_imgs.len() != expected {
        return Err(EvalError::DimensionMismatch {
            detail: format!(
                "eval_imgs len {} != n_categories({}) * n_area_ranges({}) * n_images({}) = {}",
                eval_imgs.len(),
                grid.n_categories,
                grid.n_area_ranges,
                grid.n_images,
                expected,
            ),
        });
    }

    let max_dets = summary_kind.max_dets();
    let accum_params = AccumulateParams {
        iou_thresholds,
        recall_thresholds: recall_thresholds(),
        max_dets,
        n_categories: grid.n_categories,
        n_area_ranges: grid.n_area_ranges,
        n_images: grid.n_images,
    };

    // Overall: bit-identical to today's path — we hand the same
    // `eval_imgs` slice straight to `accumulate`.
    let accum_overall = accumulate(eval_imgs, accum_params, parity_mode)?;
    let overall = summarize_with_kind(&accum_overall, iou_thresholds, summary_kind)?;
    let overall_n_detections = count_detections(eval_imgs, grid, None);
    let overall_n_images = grid.n_images as u64;

    let mut slices_out: Vec<SliceResult> = Vec::with_capacity(spec.slices.len());
    for slice in &spec.slices {
        let filtered = filtered_flatten(eval_imgs, grid, &slice.image_indices);
        let accum = accumulate(&filtered, accum_params, parity_mode)?;
        let summary = summarize_with_kind(&accum, iou_thresholds, summary_kind)?;
        let n_detections = count_detections(eval_imgs, grid, Some(&slice.image_indices));
        slices_out.push(SliceResult {
            n_images: slice.image_ids.len() as u64,
            n_detections,
            slice: slice.clone(),
            summary,
        });
    }

    Ok(PartitionedSummary {
        overall,
        overall_n_images,
        overall_n_detections,
        slices: slices_out,
    })
}

/// Run a custom summary plan over a partitioned grid.
///
/// Same orchestration as [`evaluate_partitioned`] but routes through
/// [`summarize_with`] with a caller-supplied plan + `max_dets` ladder.
/// Useful for LVIS-shape partitioned evals or any user-defined plan
/// that doesn't fit [`SummaryKind`].
///
/// # Errors
///
/// As [`evaluate_partitioned`].
pub fn evaluate_partitioned_with(
    eval_imgs: &[Option<Box<PerImageEval>>],
    grid: GridDims,
    spec: &PartitionSpec,
    iou_thresholds: &[f64],
    max_dets: &[usize],
    parity_mode: ParityMode,
    plan: &[StatRequest],
) -> Result<PartitionedSummary, EvalError> {
    let expected = grid.n_categories * grid.n_area_ranges * grid.n_images;
    if eval_imgs.len() != expected {
        return Err(EvalError::DimensionMismatch {
            detail: format!(
                "eval_imgs len {} != n_categories({}) * n_area_ranges({}) * n_images({}) = {}",
                eval_imgs.len(),
                grid.n_categories,
                grid.n_area_ranges,
                grid.n_images,
                expected,
            ),
        });
    }

    let accum_params = AccumulateParams {
        iou_thresholds,
        recall_thresholds: recall_thresholds(),
        max_dets,
        n_categories: grid.n_categories,
        n_area_ranges: grid.n_area_ranges,
        n_images: grid.n_images,
    };

    let accum_overall = accumulate(eval_imgs, accum_params, parity_mode)?;
    let overall = summarize_with(&accum_overall, plan, iou_thresholds, max_dets)?;
    let overall_n_detections = count_detections(eval_imgs, grid, None);
    let overall_n_images = grid.n_images as u64;

    let mut slices_out: Vec<SliceResult> = Vec::with_capacity(spec.slices.len());
    for slice in &spec.slices {
        let filtered = filtered_flatten(eval_imgs, grid, &slice.image_indices);
        let accum = accumulate(&filtered, accum_params, parity_mode)?;
        let summary = summarize_with(&accum, plan, iou_thresholds, max_dets)?;
        let n_detections = count_detections(eval_imgs, grid, Some(&slice.image_indices));
        slices_out.push(SliceResult {
            n_images: slice.image_ids.len() as u64,
            n_detections,
            slice: slice.clone(),
            summary,
        });
    }

    Ok(PartitionedSummary {
        overall,
        overall_n_images,
        overall_n_detections,
        slices: slices_out,
    })
}

/// Build a fresh dense `eval_imgs` vec that retains only cells whose
/// I-axis index belongs to `slice_indices`; cells outside the slice
/// are `None`. The Boxes for in-slice cells are deep-cloned because
/// `accumulate` borrows the dense slice and cannot share ownership
/// with the source grid.
fn filtered_flatten(
    eval_imgs: &[Option<Box<PerImageEval>>],
    grid: GridDims,
    slice_indices: &HashSet<usize>,
) -> Vec<Option<Box<PerImageEval>>> {
    let total = grid.n_categories * grid.n_area_ranges * grid.n_images;
    let mut out: Vec<Option<Box<PerImageEval>>> = Vec::with_capacity(total);
    for k in 0..grid.n_categories {
        for a in 0..grid.n_area_ranges {
            for i in 0..grid.n_images {
                let flat = k * grid.n_area_ranges * grid.n_images + a * grid.n_images + i;
                let cell = if slice_indices.contains(&i) {
                    eval_imgs.get(flat).and_then(|c| c.clone())
                } else {
                    None
                };
                out.push(cell);
            }
        }
    }
    out
}

/// Sum `dt_scores.len()` over every cell at A-axis 0 (the "all"
/// bucket — area filtering would double-count) whose I-axis index
/// matches `slice_indices`, or the full grid when
/// `slice_indices = None`. This is the canonical per-detection count
/// used by the per_image table.
fn count_detections(
    eval_imgs: &[Option<Box<PerImageEval>>],
    grid: GridDims,
    slice_indices: Option<&HashSet<usize>>,
) -> u64 {
    let mut total: u64 = 0;
    for k in 0..grid.n_categories {
        for i in 0..grid.n_images {
            if let Some(set) = slice_indices {
                if !set.contains(&i) {
                    continue;
                }
            }
            // a=0 is the "all" area bucket; small/medium/large would
            // double-count any DT whose area sits on a bucket boundary
            // (quirk D6).
            let flat = k * grid.n_area_ranges * grid.n_images + i;
            if let Some(cell) = eval_imgs.get(flat).and_then(|c| c.as_deref()) {
                total = total.saturating_add(cell.dt_scores.len() as u64);
            }
        }
    }
    total
}

fn summarize_with_kind(
    accum: &crate::accumulate::Accumulated,
    iou_thresholds: &[f64],
    kind: SummaryKind,
) -> Result<Summary, EvalError> {
    match kind {
        SummaryKind::DetectionDefault => {
            summarize_detection(accum, iou_thresholds, &DETECTION_MAX_DETS)
        }
        SummaryKind::KeypointsDefault => {
            let plan = StatRequest::coco_keypoints_default();
            summarize_with(accum, &plan, iou_thresholds, &KEYPOINTS_MAX_DETS)
        }
    }
}

fn make_slice(
    axis: &str,
    value: &str,
    image_ids: HashSet<ImageId>,
    image_id_to_idx: &HashMap<ImageId, usize>,
) -> Slice {
    let image_indices: HashSet<usize> = image_ids
        .iter()
        .filter_map(|id| image_id_to_idx.get(id).copied())
        .collect();
    Slice {
        axis: axis.to_owned(),
        value: value.to_owned(),
        image_ids,
        image_indices,
    }
}

fn validate_cross_axes(
    per_axis: &HashMap<String, HashMap<String, HashSet<ImageId>>>,
    cross_axes: &[Vec<String>],
) -> Result<(), EvalError> {
    for axis in per_axis.keys() {
        if axis.contains(CROSS_SEPARATOR) {
            return Err(EvalError::InvalidConfig {
                detail: format!(
                    "manifest axis name {axis:?} contains the reserved separator \
                     {CROSS_SEPARATOR:?}; rename the axis"
                ),
            });
        }
    }
    for axes in cross_axes {
        if axes.len() < 2 {
            return Err(EvalError::InvalidConfig {
                detail: format!(
                    "--cross requires at least two axes per tuple; got {} ({:?})",
                    axes.len(),
                    axes
                ),
            });
        }
        let mut seen: HashSet<&String> = HashSet::with_capacity(axes.len());
        for ax in axes {
            if !per_axis.contains_key(ax) {
                return Err(EvalError::InvalidConfig {
                    detail: format!(
                        "--cross references axis {ax:?} which is not present in the manifest"
                    ),
                });
            }
            if !seen.insert(ax) {
                return Err(EvalError::InvalidConfig {
                    detail: format!("--cross tuple {axes:?} repeats axis {ax:?}"),
                });
            }
        }
    }
    Ok(())
}

// Per-axis-value entry borrowed during cross-product expansion:
// `(axis_value, image_ids)`.
type AxisValueEntry<'a> = (&'a str, &'a HashSet<ImageId>);

fn expand_cross_axes(
    axes: &[String],
    per_axis: &HashMap<String, HashMap<String, HashSet<ImageId>>>,
    all_image_ids: &HashSet<ImageId>,
    image_id_to_idx: &HashMap<ImageId, usize>,
) -> Result<Vec<Slice>, EvalError> {
    // Resolve value sets per axis in input order so the joint cell
    // tuples carry the user's intended ordering on both label and
    // image-id intersection.
    let mut value_sets: Vec<(&str, Vec<AxisValueEntry<'_>>)> = Vec::with_capacity(axes.len());
    for axis in axes {
        let by_value = per_axis.get(axis).ok_or_else(|| EvalError::InvalidConfig {
            detail: format!("--cross axis {axis:?} missing during expansion"),
        })?;
        let mut entries: Vec<(&str, &HashSet<ImageId>)> =
            by_value.iter().map(|(v, ids)| (v.as_str(), ids)).collect();
        entries.sort_by_key(|(v, _)| *v);
        value_sets.push((axis.as_str(), entries));
    }

    // Cartesian product. For the small cell counts a partition admits
    // (cap-gated upstream), an iterative expansion is plenty.
    let mut combos: Vec<Vec<(&str, &str, &HashSet<ImageId>)>> = vec![Vec::new()];
    for (axis_name, values) in &value_sets {
        let mut next: Vec<Vec<(&str, &str, &HashSet<ImageId>)>> = Vec::new();
        for combo in &combos {
            for (value, ids) in values {
                let mut extended = combo.clone();
                extended.push((axis_name, value, ids));
                next.push(extended);
            }
        }
        combos = next;
    }

    let joined_axis: String = axes.join(CROSS_SEPARATOR);
    let mut out: Vec<Slice> = Vec::with_capacity(combos.len() + 1);
    let mut covered: HashSet<ImageId> = HashSet::new();
    for combo in combos {
        // Joint image set is the intersection across this tuple's
        // per-axis value sets.
        let mut iter = combo.iter().map(|(_, _, ids)| *ids);
        let mut joint: HashSet<ImageId> = match iter.next() {
            Some(first) => first.clone(),
            None => HashSet::new(),
        };
        for ids in iter {
            joint = joint.intersection(ids).copied().collect();
        }
        covered.extend(joint.iter().copied());
        let joined_value: String = combo
            .iter()
            .map(|(_, v, _)| (*v).to_owned())
            .collect::<Vec<_>>()
            .join(CROSS_SEPARATOR);
        out.push(make_slice(
            &joined_axis,
            &joined_value,
            joint,
            image_id_to_idx,
        ));
    }
    // __unassigned__ joint bucket: dataset images covered by no joint
    // cell of this tuple. Materialized so the joint partition is
    // exhaustive for the same reason marginals are.
    let missing: HashSet<ImageId> = all_image_ids
        .iter()
        .copied()
        .filter(|id| !covered.contains(id))
        .collect();
    out.push(make_slice(
        &joined_axis,
        UNASSIGNED,
        missing,
        image_id_to_idx,
    ));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accumulate::PerImageEval;
    use ndarray::Array2;

    fn id(n: i64) -> ImageId {
        ImageId(n)
    }

    fn build_image_grid(n: i64) -> (HashSet<ImageId>, HashMap<ImageId, usize>) {
        let all: HashSet<ImageId> = (1..=n).map(id).collect();
        let map: HashMap<ImageId, usize> = (1..=n).map(|i| (id(i), (i - 1) as usize)).collect();
        (all, map)
    }

    #[test]
    fn marginal_order_is_axis_then_value_then_unassigned_last() {
        let mut per_axis: HashMap<String, HashMap<String, HashSet<ImageId>>> = HashMap::new();
        per_axis.insert(
            "weather".into(),
            HashMap::from([
                ("fog".into(), HashSet::from([id(1)])),
                ("clear".into(), HashSet::from([id(2)])),
            ]),
        );
        per_axis.insert(
            "time".into(),
            HashMap::from([("day".into(), HashSet::from([id(1), id(2)]))]),
        );
        let (all, map) = build_image_grid(3);

        let spec = PartitionSpec::build(KeyKind::Image, &per_axis, &all, &map, &[]).unwrap();
        let labels: Vec<(&str, &str)> = spec
            .slices
            .iter()
            .map(|s| (s.axis.as_str(), s.value.as_str()))
            .collect();
        assert_eq!(
            labels,
            vec![
                ("time", "day"),
                ("time", UNASSIGNED),
                ("weather", "clear"),
                ("weather", "fog"),
                ("weather", UNASSIGNED),
            ]
        );
    }

    #[test]
    fn unassigned_collects_dataset_images_not_in_any_value() {
        // Manifest covers only image 1 for axis `weather`; images 2..3
        // must land in __unassigned__.
        let mut per_axis: HashMap<String, HashMap<String, HashSet<ImageId>>> = HashMap::new();
        per_axis.insert(
            "weather".into(),
            HashMap::from([("fog".into(), HashSet::from([id(1)]))]),
        );
        let (all, map) = build_image_grid(3);

        let spec = PartitionSpec::build(KeyKind::Image, &per_axis, &all, &map, &[]).unwrap();
        let unassigned = spec
            .slices
            .iter()
            .find(|s| s.axis == "weather" && s.value == UNASSIGNED)
            .expect("expected an unassigned slice on `weather`");
        let mut ids: Vec<i64> = unassigned.image_ids.iter().map(|i| i.0).collect();
        ids.sort();
        assert_eq!(ids, vec![2, 3]);
    }

    #[test]
    fn cross_axes_emits_intersection_joint_cells() {
        let mut per_axis: HashMap<String, HashMap<String, HashSet<ImageId>>> = HashMap::new();
        per_axis.insert(
            "weather".into(),
            HashMap::from([
                ("fog".into(), HashSet::from([id(1), id(2)])),
                ("clear".into(), HashSet::from([id(3), id(4)])),
            ]),
        );
        per_axis.insert(
            "time".into(),
            HashMap::from([
                ("day".into(), HashSet::from([id(1), id(3)])),
                ("night".into(), HashSet::from([id(2), id(4)])),
            ]),
        );
        let (all, map) = build_image_grid(4);

        let cross = vec![vec!["weather".to_string(), "time".to_string()]];
        let spec = PartitionSpec::build(KeyKind::Image, &per_axis, &all, &map, &cross).unwrap();
        // Find the joint cells.
        let joint: Vec<&Slice> = spec
            .slices
            .iter()
            .filter(|s| s.axis.contains(CROSS_SEPARATOR))
            .collect();
        // 2*2 = 4 combos plus an unassigned bucket.
        assert_eq!(joint.len(), 5);
        let fog_day = joint
            .iter()
            .find(|s| s.value == "fog::day")
            .expect("fog::day must exist");
        let mut ids: Vec<i64> = fog_day.image_ids.iter().map(|i| i.0).collect();
        ids.sort();
        assert_eq!(ids, vec![1]);
    }

    #[test]
    fn cross_axes_with_unknown_axis_is_rejected() {
        let mut per_axis: HashMap<String, HashMap<String, HashSet<ImageId>>> = HashMap::new();
        per_axis.insert(
            "weather".into(),
            HashMap::from([("fog".into(), HashSet::from([id(1)]))]),
        );
        let (all, map) = build_image_grid(2);
        let cross = vec![vec!["weather".into(), "missing".into()]];
        let err = PartitionSpec::build(KeyKind::Image, &per_axis, &all, &map, &cross).unwrap_err();
        assert!(matches!(err, EvalError::InvalidConfig { .. }));
    }

    #[test]
    fn cross_axes_singleton_tuple_is_rejected() {
        let mut per_axis: HashMap<String, HashMap<String, HashSet<ImageId>>> = HashMap::new();
        per_axis.insert(
            "weather".into(),
            HashMap::from([("fog".into(), HashSet::from([id(1)]))]),
        );
        let (all, map) = build_image_grid(2);
        let cross = vec![vec!["weather".into()]];
        let err = PartitionSpec::build(KeyKind::Image, &per_axis, &all, &map, &cross).unwrap_err();
        assert!(matches!(err, EvalError::InvalidConfig { .. }));
    }

    #[test]
    fn slice_cap_is_enforced() {
        // Manufacture a single axis with SLICES_CAP+1 values; each
        // value gets one image. The __unassigned__ bucket adds one
        // more slice, well over the cap.
        let mut by_value: HashMap<String, HashSet<ImageId>> = HashMap::new();
        let n = SLICES_CAP + 1;
        for i in 1..=n as i64 {
            by_value.insert(format!("v{i}"), HashSet::from([id(i)]));
        }
        let mut per_axis: HashMap<String, HashMap<String, HashSet<ImageId>>> = HashMap::new();
        per_axis.insert("axis".into(), by_value);
        let (all, map) = build_image_grid(n as i64);
        let err = PartitionSpec::build(KeyKind::Image, &per_axis, &all, &map, &[]).unwrap_err();
        assert!(matches!(err, EvalError::InvalidConfig { .. }));
    }

    #[test]
    fn axis_name_with_cross_separator_is_rejected() {
        let mut per_axis: HashMap<String, HashMap<String, HashSet<ImageId>>> = HashMap::new();
        per_axis.insert(
            "weather::extra".into(),
            HashMap::from([("fog".into(), HashSet::from([id(1)]))]),
        );
        let (all, map) = build_image_grid(2);
        let err = PartitionSpec::build(KeyKind::Image, &per_axis, &all, &map, &[]).unwrap_err();
        assert!(matches!(err, EvalError::InvalidConfig { .. }));
    }

    fn fake_cell(n_dts: usize) -> Box<PerImageEval> {
        Box::new(PerImageEval {
            dt_scores: vec![0.5; n_dts],
            dt_matched: Array2::default((1, n_dts)),
            dt_ignore: Array2::default((1, n_dts)),
            gt_ignore: vec![false],
        })
    }

    #[test]
    fn filtered_flatten_keeps_only_in_slice_cells() {
        // Grid: K=1, A=1, I=3. Each (k=0, a=0, i) has a cell.
        let grid = GridDims {
            n_categories: 1,
            n_area_ranges: 1,
            n_images: 3,
        };
        let eval_imgs: Vec<Option<Box<PerImageEval>>> =
            vec![Some(fake_cell(2)), Some(fake_cell(3)), Some(fake_cell(4))];

        let slice_indices: HashSet<usize> = HashSet::from([0, 2]);
        let filtered = filtered_flatten(&eval_imgs, grid, &slice_indices);
        assert!(filtered[0].is_some());
        assert!(filtered[1].is_none());
        assert!(filtered[2].is_some());
    }

    #[test]
    fn count_detections_skips_out_of_slice_images() {
        // Two images, two categories. Use a=0 only (single area).
        let grid = GridDims {
            n_categories: 2,
            n_area_ranges: 1,
            n_images: 2,
        };
        // Layout: [(k=0,i=0), (k=0,i=1), (k=1,i=0), (k=1,i=1)].
        let eval_imgs: Vec<Option<Box<PerImageEval>>> = vec![
            Some(fake_cell(1)),
            Some(fake_cell(2)),
            Some(fake_cell(3)),
            Some(fake_cell(4)),
        ];
        let total = count_detections(&eval_imgs, grid, None);
        assert_eq!(total, 1 + 2 + 3 + 4);

        let only_first = count_detections(&eval_imgs, grid, Some(&HashSet::from([0])));
        // Sums across k for i=0: 1 + 3 = 4.
        assert_eq!(only_first, 4);
    }
}
