//! Confusion-matrix FFI: wraps [`vernier_core::compute_confusion_matrix`]
//! and exposes a per-kernel entry point pair (`bbox` / `segm` /
//! `boundary`) to Python. By policy, this module contains only data
//! conversion — the algorithm and the cross-class side pass live in
//! [`vernier_core::tide`].
//!
//! ## Output shape
//!
//! Each entry point returns a `dict` shaped as parallel arrays so the
//! Python wrapper can pass it straight to `polars.DataFrame.from_dict`:
//!
//! ```text
//! {
//!     "gt_class": list[str],   # length N, "__none__" for FP rows
//!     "dt_class": list[str],   # length N, "__none__" for missed cols
//!     "count": list[int],      # length N
//!     "iou_threshold": float,  # echoed back for caller introspection
//!     "kernel": str,           # "bbox" | "segm" | "boundary"
//! }
//! ```
//!
//! Mixed-type class columns (int category id vs string `"__none__"`
//! sentinel) are awkward to model in polars; the FFI casts every class
//! id to its decimal-string repr so the column has uniform `pl.Utf8`
//! dtype. Users wanting numeric ids can `df.with_columns(pl.col("gt_class").cast(pl.Int64,
//! strict=False))` themselves; the `__none__` rows become `null` in
//! that cast, which is the natural representation of "no class".
//!
//! Long-format (one row per `(gt_class, dt_class)` cell) rather than
//! the wide-format matrix; pivoting is `df.pivot(...)` on the user
//! side. Long-format composes better with polars' filter / group / agg
//! idioms.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use vernier_core::similarity::{BboxIou, BoundaryIou, SegmIou};
use vernier_core::tide::{compute_confusion_matrix, ConfusionMatrixCounts};
use vernier_core::{CocoDataset, CocoDetections, EvalError, ParityMode};

use crate::{parse_dt, parse_gt, parse_parity_mode, validate_dilation_ratio};

/// Sentinel string surfaced in the `gt_class` / `dt_class` columns
/// when the row is a false positive (`gt_class == "__none__"`) or a
/// missed GT (`dt_class == "__none__"`). Pinned here so the Rust
/// surface and the Python wrapper agree on the literal.
const NONE_SENTINEL: &str = "__none__";

/// One row of the long-format confusion matrix as the FFI carries it
/// internally before the `(gt, dt) -> count` map is flattened into
/// the parallel-arrays dict. Aliased to dodge clippy's type-complexity
/// lint on the sort step that wants a `Vec<...>` of this exact shape.
type CountRow = ((Option<usize>, Option<usize>), u64);

/// Common per-call plumbing for the three confusion-matrix kernel
/// entry points: parse parity, copy JSON bytes off the GIL, run the
/// kernel-specific orchestrator inside `py.detach`, and materialize
/// the dict. `kernel_call` carries the kernel-specific dispatch
/// (and any extra knobs like `dilation_ratio`) closed over by the
/// per-kernel wrappers below.
#[allow(clippy::too_many_arguments)]
fn run_confusion_pass<'py, F>(
    py: Python<'py>,
    gt_bytes: &Bound<'py, PyBytes>,
    dt_bytes: &Bound<'py, PyBytes>,
    parity_mode: &str,
    iou_threshold: f64,
    max_dets_per_image: usize,
    use_cats: bool,
    kernel_name: &'static str,
    kernel_call: F,
) -> PyResult<Bound<'py, PyDict>>
where
    F: FnOnce(
            &CocoDataset,
            &CocoDetections,
            ParityMode,
        ) -> Result<ConfusionMatrixCounts, EvalError>
        + Send,
{
    let parity = parse_parity_mode(parity_mode)?;
    if !iou_threshold.is_finite() || !(0.0..=1.0).contains(&iou_threshold) {
        return Err(PyValueError::new_err(format!(
            "iou_threshold must be a finite float in [0, 1], got {iou_threshold}"
        )));
    }
    if max_dets_per_image == 0 {
        return Err(PyValueError::new_err(
            "max_dets_per_image must be >= 1 (matches the matching path's per-image cap)",
        ));
    }
    // `use_cats=False` would invalidate the per-image argmax (a
    // category-collapsed evaluation has no meaningful confusion
    // matrix; every cell would land on a single virtual class). The
    // matching path supports the flag for mAP-style summaries; the
    // confusion matrix doesn't have a coherent shape under collapse.
    if !use_cats {
        return Err(PyValueError::new_err(
            "use_cats=False is not supported for confusion_matrix — \
             a category-collapsed evaluation collapses every cell to a \
             single virtual class and so has no meaningful confusion matrix",
        ));
    }

    // Copy the JSON bytes off the GIL-tied PyBytes borrow so the parse
    // and the side pass can run inside `py.detach`.
    let gt_bytes = gt_bytes.as_bytes().to_vec();
    let dt_bytes = dt_bytes.as_bytes().to_vec();

    let cm = py.detach(move || -> PyResult<ConfusionMatrixCounts> {
        let gt = parse_gt(&gt_bytes)?;
        let dt = parse_dt(&dt_bytes)?;
        kernel_call(&gt, &dt, parity).map_err(|e| PyValueError::new_err(format!("{e}")))
    })?;

    counts_to_dict(py, &cm, iou_threshold, kernel_name)
}

/// Confusion matrix for the bbox kernel (per ADR-0023, sibling
/// capability of TIDE error decomposition).
///
/// `gt_bytes` and `dt_bytes` are the COCO ground-truth and detection
/// JSON payloads as bytes (the same shapes pycocotools' `COCO(...)` /
/// `loadRes(...)` consume). `parity_mode` is `"strict"` or
/// `"corrected"` per ADR-0002. `iou_threshold` is the foreground
/// threshold for declaring a `(gt, dt)` pair matched (0.5 is the COCO
/// canonical default). `max_dets_per_image` matches the matching
/// path's per-image cap so the side-pass rows line up. `use_cats` is
/// reserved (must be `True`); see the module doc for the rationale.
///
/// Returns the long-format dict described in the module docstring.
#[pyfunction]
#[pyo3(signature = (gt_bytes, dt_bytes, parity_mode, iou_threshold, max_dets_per_image, use_cats))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn confusion_matrix_bbox<'py>(
    py: Python<'py>,
    gt_bytes: &Bound<'py, PyBytes>,
    dt_bytes: &Bound<'py, PyBytes>,
    parity_mode: &str,
    iou_threshold: f64,
    max_dets_per_image: usize,
    use_cats: bool,
) -> PyResult<Bound<'py, PyDict>> {
    run_confusion_pass(
        py,
        gt_bytes,
        dt_bytes,
        parity_mode,
        iou_threshold,
        max_dets_per_image,
        use_cats,
        "bbox",
        move |gt, dt, parity| {
            compute_confusion_matrix(gt, dt, &BboxIou, iou_threshold, max_dets_per_image, parity)
        },
    )
}

/// Confusion matrix for the segm kernel.
///
/// Same signature as [`confusion_matrix_bbox`]; `gt_bytes` /
/// `dt_bytes` must carry COCO `segmentation` fields (polygon or RLE).
#[pyfunction]
#[pyo3(signature = (gt_bytes, dt_bytes, parity_mode, iou_threshold, max_dets_per_image, use_cats))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn confusion_matrix_segm<'py>(
    py: Python<'py>,
    gt_bytes: &Bound<'py, PyBytes>,
    dt_bytes: &Bound<'py, PyBytes>,
    parity_mode: &str,
    iou_threshold: f64,
    max_dets_per_image: usize,
    use_cats: bool,
) -> PyResult<Bound<'py, PyDict>> {
    run_confusion_pass(
        py,
        gt_bytes,
        dt_bytes,
        parity_mode,
        iou_threshold,
        max_dets_per_image,
        use_cats,
        "segm",
        move |gt, dt, parity| {
            compute_confusion_matrix(gt, dt, &SegmIou, iou_threshold, max_dets_per_image, parity)
        },
    )
}

/// Confusion matrix for the boundary-segm kernel (ADR-0010 +
/// ADR-0023). Same shape as [`confusion_matrix_bbox`] except
/// `dilation_ratio` pins the boundary band thickness (ADR-0010
/// default `0.02` for COCO, `0.008` for LVIS).
#[pyfunction]
#[pyo3(signature = (gt_bytes, dt_bytes, parity_mode, iou_threshold, max_dets_per_image, use_cats, dilation_ratio))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn confusion_matrix_boundary<'py>(
    py: Python<'py>,
    gt_bytes: &Bound<'py, PyBytes>,
    dt_bytes: &Bound<'py, PyBytes>,
    parity_mode: &str,
    iou_threshold: f64,
    max_dets_per_image: usize,
    use_cats: bool,
    dilation_ratio: f64,
) -> PyResult<Bound<'py, PyDict>> {
    validate_dilation_ratio(dilation_ratio)?;
    let kernel = BoundaryIou { dilation_ratio };
    run_confusion_pass(
        py,
        gt_bytes,
        dt_bytes,
        parity_mode,
        iou_threshold,
        max_dets_per_image,
        use_cats,
        "boundary",
        move |gt, dt, parity| {
            compute_confusion_matrix(gt, dt, &kernel, iou_threshold, max_dets_per_image, parity)
        },
    )
}

/// Materialize a [`ConfusionMatrixCounts`] into the parallel-arrays
/// dict shape pinned in the module docstring. Class indices are
/// resolved to COCO ids via `category_ids`; the `None` sentinel
/// becomes the literal `"__none__"`.
fn counts_to_dict<'py>(
    py: Python<'py>,
    cm: &ConfusionMatrixCounts,
    iou_threshold: f64,
    kernel_name: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let n = cm.counts.len();
    let mut gt_class: Vec<String> = Vec::with_capacity(n);
    let mut dt_class: Vec<String> = Vec::with_capacity(n);
    let mut counts: Vec<u64> = Vec::with_capacity(n);

    // Sort the entries for deterministic output: GT (None first, then
    // ascending by COCO id), then DT (same convention). Determinism
    // matters because Python tests / golden files round-trip the
    // dict; HashMap iteration order is not stable.
    let mut entries: Vec<CountRow> = cm.counts.iter().map(|(k, v)| (*k, *v)).collect();
    entries.sort_by_key(|((g, d), _)| (idx_sort_key(*g), idx_sort_key(*d)));

    for ((g_idx, d_idx), c) in entries {
        gt_class.push(class_label(g_idx, &cm.category_ids));
        dt_class.push(class_label(d_idx, &cm.category_ids));
        counts.push(c);
    }

    let out = PyDict::new(py);
    out.set_item("gt_class", gt_class)?;
    out.set_item("dt_class", dt_class)?;
    out.set_item("count", counts)?;
    out.set_item("iou_threshold", iou_threshold)?;
    out.set_item("kernel", kernel_name)?;
    Ok(out)
}

/// Stable sort key that puts the `None` sentinel after every real
/// class index. Sentinel rows trail the diagonal / off-diagonal
/// entries in the output, which reads nicely top-to-bottom.
fn idx_sort_key(idx: Option<usize>) -> (u8, usize) {
    match idx {
        Some(i) => (0, i),
        None => (1, usize::MAX),
    }
}

/// Map a `Some(idx)` to the decimal-string repr of the COCO id; map
/// `None` to the `"__none__"` sentinel.
fn class_label(idx: Option<usize>, category_ids: &[i64]) -> String {
    match idx {
        Some(i) => category_ids
            .get(i)
            .map_or_else(|| NONE_SENTINEL.to_owned(), |id| id.to_string()),
        None => NONE_SENTINEL.to_owned(),
    }
}
