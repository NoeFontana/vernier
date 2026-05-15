//! FFI surface for the detection-family calibration summarizer
//! (ADR-0018), exposing the [`vernier_core::calibration`] kernel
//! through PyO3 plus the ADR-0019 Arrow PyCapsule mechanism.
//!
//! This module is a pure data-conversion layer: it wraps the kernel's
//! `Vec<Option<Box<PerImageEval>>>` cell store inside an opaque
//! [`EvalCells`] `#[pyclass]`, runs [`summarize_calibration`] under
//! `py.detach`, and packs the resulting reliability /
//! per-class tables into Arrow `RecordBatch`es via
//! [`crate::arrow_helpers::wrap_batch`]. No business logic lives here
//! (per the FFI crate's policy in `lib.rs`); the parity-pinned numerics
//! and quirk dispositions live in `crates/vernier-core/src/calibration.rs`
//! and `docs/engineering/calibration-quirks.md`.
//!
//! Implementation plan: see
//! `~/.claude/plans/adr-0018-calibration-metrics-zany-wall.md`,
//! "Unit 2 — Arrow output via FFI".
//!
//! No unsafe code. The DLPack escape hatch documented at the crate
//! root does not apply here — calibration is built entirely on safe
//! Arrow / PyO3 abstractions.

use std::sync::Arc;

use arrow_array::{ArrayRef, Float64Array, RecordBatch, UInt32Array, UInt64Array};
use arrow_schema::{ArrowError, DataType, Field, Schema};
use numpy::ndarray::Array2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};

use vernier_core::accumulate::PerImageEval;
use vernier_core::calibration::{
    summarize_calibration, Aggregation, Binning, CalibrationParams, CalibrationSummary,
    ConfidenceKind, PerClassTable, ReliabilityTable,
};
use vernier_core::parity::PARITY_EPS;
use vernier_core::{EvalError, ParityMode};

use crate::arrow_helpers::{arrow_err, wrap_batch, ArrowRecordBatchPy};
use crate::tables::table_metadata;

/// Return shape of [`EvalCells::calibrate`]: scalars plus the two
/// Arrow PyCapsule wrappers (the per-class capsule is `None` when the
/// caller asked for the marginal-only summary).
type CalibrateOutput = (
    f64,
    f64,
    u64,
    usize,
    Py<ArrowRecordBatchPy>,
    Option<Py<ArrowRecordBatchPy>>,
);

// ---------------------------------------------------------------------------
// EvalCells — opaque PyO3 handle over the per-image cell store.
// ---------------------------------------------------------------------------

/// Opaque handle over the cell store needed by the calibration
/// summarizer (ADR-0018). Built from an
/// [`crate::PyEvalGrid`] via [`cells_from_grid`] and consumed by
/// [`EvalCells::calibrate`] to produce a [`CalibrationSummary`] folded
/// into Arrow `RecordBatch`es.
#[pyclass(module = "vernier._core", name = "EvalCells", frozen)]
pub struct EvalCells {
    cells: Vec<Option<Box<PerImageEval>>>,
    n_categories: usize,
    n_area_ranges: usize,
    iou_thresholds: Vec<f64>,
    parity_mode: ParityMode,
}

impl EvalCells {
    /// Build a new handle. Crate-internal; the public entry point is
    /// [`cells_from_grid`].
    pub(crate) fn new(
        cells: Vec<Option<Box<PerImageEval>>>,
        n_categories: usize,
        n_area_ranges: usize,
        iou_thresholds: Vec<f64>,
        parity_mode: ParityMode,
    ) -> Self {
        Self {
            cells,
            n_categories,
            n_area_ranges,
            iou_thresholds,
            parity_mode,
        }
    }
}

#[pymethods]
impl EvalCells {
    /// Resolve a user-supplied IoU float to the kernel's T-axis index.
    ///
    /// The match tolerance is [`PARITY_EPS`]
    /// (`f64::EPSILON`); a value that doesn't land within that of any
    /// pinned threshold raises `ValueError`. This is the dual of the
    /// matcher's bit-exact T-axis lookup: callers think in `iou=0.5`
    /// but the kernel addresses cells by integer index.
    fn iou_to_index(&self, iou: f64) -> PyResult<usize> {
        if !iou.is_finite() {
            return Err(PyValueError::new_err(format!(
                "iou must be finite, got {iou}"
            )));
        }
        for (idx, &thr) in self.iou_thresholds.iter().enumerate() {
            if (thr - iou).abs() <= PARITY_EPS {
                return Ok(idx);
            }
        }
        Err(PyValueError::new_err(format!(
            "iou {iou} does not match any pinned threshold within PARITY_EPS \
             ({PARITY_EPS:e}); available: {:?}",
            self.iou_thresholds
        )))
    }

    /// Fold the cell store into a calibration summary and return a
    /// 6-tuple:
    ///
    /// `(ece, mce, n_detections, effective_n_bins, reliability_capsule,
    ///   per_class_capsule_or_none)`.
    ///
    /// String-typed enum arguments (`binning`, `confidence`,
    /// `per_class_aggregation`) are converted to the kernel's Rust
    /// enums here; unknown values raise `ValueError`. The kernel runs
    /// under `py.detach` per ADR-0006.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        iou_index, n_bins, binning, min_score, confidence,
        per_class, per_class_aggregation,
    ))]
    fn calibrate(
        &self,
        py: Python<'_>,
        iou_index: usize,
        n_bins: usize,
        binning: &str,
        min_score: f64,
        confidence: &str,
        per_class: bool,
        per_class_aggregation: &str,
    ) -> PyResult<CalibrateOutput> {
        let binning = parse_binning(binning)?;
        let confidence = parse_confidence(confidence)?;
        let aggregation = parse_aggregation(per_class_aggregation)?;

        let params = CalibrationParams {
            iou_index,
            n_bins,
            binning,
            min_score,
            confidence,
            per_class,
            per_class_aggregation: aggregation,
        };

        let cells = &self.cells;
        let n_categories = self.n_categories;
        let n_area_ranges = self.n_area_ranges;
        let parity_mode = self.parity_mode;
        let summary = py
            .detach(move || -> Result<CalibrationSummary, EvalError> {
                summarize_calibration(cells, n_categories, n_area_ranges, &params, parity_mode)
            })
            .map_err(|e| crate::eval_error_to_pyerr(py, e))?;

        let reliability_batch = reliability_to_arrow(&summary).map_err(|e| arrow_err(&e))?;
        let reliability_py = Py::new(py, wrap_batch(reliability_batch))?;

        let per_class_py = match &summary.per_class {
            Some(table) => {
                let batch = per_class_calibration_to_arrow(table).map_err(|e| arrow_err(&e))?;
                Some(Py::new(py, wrap_batch(batch))?)
            }
            None => None,
        };

        Ok((
            summary.ece,
            summary.mce,
            summary.n_detections,
            summary.effective_n_bins,
            reliability_py,
            per_class_py,
        ))
    }

    fn __repr__(&self) -> String {
        let populated = self.cells.iter().filter(|c| c.is_some()).count();
        format!(
            "EvalCells(n_categories={}, n_area_ranges={}, n_cells={}, populated={})",
            self.n_categories,
            self.n_area_ranges,
            self.cells.len(),
            populated
        )
    }

    /// Test-only constructor used by the ADR-0018 parity harness
    /// (`tests/python/parity_calibration/harness.py`).
    ///
    /// Builds an `EvalCells` directly from a Python dict carrying the
    /// dense `(k, a, i)` cell grid. Lets unit-level parity tests feed
    /// the kernel synthetic cells produced by the parity-oracle
    /// fixtures (see
    /// `tests/python/parity_calibration/fixtures/seed.py`) without
    /// driving the full Evaluator → grid → cells pipeline. The same
    /// `EvalCells::calibrate` codepath the production grid uses is
    /// exercised end-to-end.
    ///
    /// Expected layout of `cells_json`:
    ///
    /// ```python
    /// {
    ///     "n_categories": int,
    ///     "n_area_ranges": int,
    ///     "n_images": int,
    ///     "iou_thresholds": [float, ...],   # length T
    ///     "parity_mode": "strict" | "corrected",
    ///     "cells": [
    ///         # length n_categories * n_area_ranges * n_images, in
    ///         # the same `k * A * I + a * I + i` row-major order as
    ///         # `Vec<Option<Box<PerImageEval>>>`.
    ///         None,
    ///         {
    ///             "dt_scores":  [f64, ...],          # length D
    ///             "dt_matched": [[bool, ...], ...],  # shape [T][D]
    ///             "dt_ignore":  [[bool, ...], ...],  # shape [T][D]
    ///         },
    ///         ...
    ///     ],
    /// }
    /// ```
    ///
    /// Not on the public Python stability surface. Lives outside
    /// `cfg(test)` so the parity harness can wire it without a special
    /// build profile — the production `calibrate` method is the unit
    /// under test, and this constructor is the test fixture loader.
    #[staticmethod]
    fn from_python_cells(cells_json: &Bound<'_, PyAny>) -> PyResult<Self> {
        decode_cells_dict(cells_json)
    }
}

/// Decode the `cells_json` PyDict described in
/// [`EvalCells::from_python_cells`].
fn decode_cells_dict(obj: &Bound<'_, PyAny>) -> PyResult<EvalCells> {
    let d = obj
        .cast::<PyDict>()
        .map_err(|_| PyValueError::new_err("from_python_cells: expected a dict argument"))?;

    let n_categories: usize = get_required(d, "n_categories")?.extract()?;
    let n_area_ranges: usize = get_required(d, "n_area_ranges")?.extract()?;
    let n_images: usize = get_required(d, "n_images")?.extract()?;
    let iou_thresholds: Vec<f64> = get_required(d, "iou_thresholds")?.extract()?;
    let parity_mode_str: String = get_required(d, "parity_mode")?.extract()?;
    let parity_mode = match parity_mode_str.as_str() {
        "strict" => ParityMode::Strict,
        "corrected" => ParityMode::Corrected,
        other => {
            return Err(PyValueError::new_err(format!(
                "parity_mode must be 'strict' or 'corrected', got {other:?}"
            )));
        }
    };

    let cells_any = get_required(d, "cells")?;
    let cells_list = cells_any
        .cast::<PyList>()
        .map_err(|_| PyValueError::new_err("from_python_cells: 'cells' must be a list"))?;

    let expected_len = n_categories
        .saturating_mul(n_area_ranges)
        .saturating_mul(n_images);
    if cells_list.len() != expected_len {
        return Err(PyValueError::new_err(format!(
            "from_python_cells: cells len {} != n_categories({}) * n_area_ranges({}) * n_images({}) = {}",
            cells_list.len(),
            n_categories,
            n_area_ranges,
            n_images,
            expected_len,
        )));
    }

    let t = iou_thresholds.len();
    let mut cells: Vec<Option<Box<PerImageEval>>> = Vec::with_capacity(cells_list.len());
    for (idx, item) in cells_list.iter().enumerate() {
        if item.is_none() {
            cells.push(None);
            continue;
        }
        let cell = decode_one_cell(&item, t, idx)?;
        cells.push(Some(Box::new(cell)));
    }

    Ok(EvalCells::new(
        cells,
        n_categories,
        n_area_ranges,
        iou_thresholds,
        parity_mode,
    ))
}

fn get_required<'py>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    d.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("from_python_cells: missing key {key:?}")))
}

fn decode_one_cell(item: &Bound<'_, PyAny>, t: usize, idx: usize) -> PyResult<PerImageEval> {
    let cd = item.cast::<PyDict>().map_err(|_| {
        PyValueError::new_err(format!(
            "from_python_cells: cells[{idx}] must be None or a dict"
        ))
    })?;

    let dt_scores: Vec<f64> = get_required(cd, "dt_scores")?.extract()?;
    let dt_matched_rows: Vec<Vec<bool>> = get_required(cd, "dt_matched")?.extract()?;
    let dt_ignore_rows: Vec<Vec<bool>> = get_required(cd, "dt_ignore")?.extract()?;

    let d = dt_scores.len();
    if dt_matched_rows.len() != t {
        return Err(PyValueError::new_err(format!(
            "from_python_cells: cells[{idx}].dt_matched outer len {} != n_iou_thresholds {}",
            dt_matched_rows.len(),
            t,
        )));
    }
    if dt_ignore_rows.len() != t {
        return Err(PyValueError::new_err(format!(
            "from_python_cells: cells[{idx}].dt_ignore outer len {} != n_iou_thresholds {}",
            dt_ignore_rows.len(),
            t,
        )));
    }

    let mut dt_matched: Array2<bool> = Array2::from_elem((t, d), false);
    let mut dt_ignore: Array2<bool> = Array2::from_elem((t, d), false);
    for (ti, row) in dt_matched_rows.iter().enumerate() {
        if row.len() != d {
            return Err(PyValueError::new_err(format!(
                "from_python_cells: cells[{idx}].dt_matched[{ti}] len {} != D {}",
                row.len(),
                d,
            )));
        }
        for (di, &v) in row.iter().enumerate() {
            dt_matched[(ti, di)] = v;
        }
    }
    for (ti, row) in dt_ignore_rows.iter().enumerate() {
        if row.len() != d {
            return Err(PyValueError::new_err(format!(
                "from_python_cells: cells[{idx}].dt_ignore[{ti}] len {} != D {}",
                row.len(),
                d,
            )));
        }
        for (di, &v) in row.iter().enumerate() {
            dt_ignore[(ti, di)] = v;
        }
    }

    Ok(PerImageEval {
        dt_scores,
        dt_matched,
        dt_ignore,
        gt_ignore: Vec::new(),
    })
}

// ---------------------------------------------------------------------------
// Enum string parsers.
// ---------------------------------------------------------------------------

fn parse_binning(s: &str) -> PyResult<Binning> {
    match s {
        "quantile" => Ok(Binning::Quantile),
        "equal_width" => Ok(Binning::EqualWidth),
        other => Err(PyValueError::new_err(format!(
            "binning must be 'quantile' or 'equal_width', got {other:?}"
        ))),
    }
}

fn parse_confidence(s: &str) -> PyResult<ConfidenceKind> {
    match s {
        "wilson" => Ok(ConfidenceKind::Wilson),
        "clopper_pearson" => Ok(ConfidenceKind::ClopperPearson),
        other => Err(PyValueError::new_err(format!(
            "confidence must be 'wilson' or 'clopper_pearson', got {other:?}"
        ))),
    }
}

fn parse_aggregation(s: &str) -> PyResult<Aggregation> {
    match s {
        "macro" => Ok(Aggregation::Macro),
        "micro" => Ok(Aggregation::Micro),
        other => Err(PyValueError::new_err(format!(
            "per_class_aggregation must be 'macro' or 'micro', got {other:?}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Arrow RecordBatch builders.
// ---------------------------------------------------------------------------

/// Build the reliability table's Arrow `RecordBatch`. One row per
/// effective bin; per-bin floats are `NaN` on zero-count bins (the
/// kernel's R2-mitigation convention).
pub(crate) fn reliability_to_arrow(cal: &CalibrationSummary) -> Result<RecordBatch, ArrowError> {
    let r: &ReliabilityTable = &cal.reliability;
    let schema = Arc::new(reliability_schema());

    let bin_id: ArrayRef = Arc::new(UInt32Array::from_iter_values(r.bin_id.iter().copied()));
    let score_lo: ArrayRef = Arc::new(Float64Array::from_iter_values(r.score_lo.iter().copied()));
    let score_hi: ArrayRef = Arc::new(Float64Array::from_iter_values(r.score_hi.iter().copied()));
    let mean_score: ArrayRef =
        Arc::new(Float64Array::from_iter_values(r.mean_score.iter().copied()));
    let accuracy: ArrayRef = Arc::new(Float64Array::from_iter_values(r.accuracy.iter().copied()));
    let count: ArrayRef = Arc::new(UInt64Array::from_iter_values(r.count.iter().copied()));
    let gap: ArrayRef = Arc::new(Float64Array::from_iter_values(r.gap.iter().copied()));
    let ci_lo: ArrayRef = Arc::new(Float64Array::from_iter_values(r.ci_lo.iter().copied()));
    let ci_hi: ArrayRef = Arc::new(Float64Array::from_iter_values(r.ci_hi.iter().copied()));

    RecordBatch::try_new(
        schema,
        vec![
            bin_id, score_lo, score_hi, mean_score, accuracy, count, gap, ci_lo, ci_hi,
        ],
    )
}

/// Build the per-class calibration table's Arrow `RecordBatch`.
///
/// Column-major construction (R6 mitigation): each `class_id` /
/// `ece` / `mce` / `n` column is materialized in one pass over the
/// kernel's columnar vectors, never a per-class loop.
pub(crate) fn per_class_calibration_to_arrow(
    table: &PerClassTable,
) -> Result<RecordBatch, ArrowError> {
    let schema = Arc::new(per_class_calibration_schema());

    let class_id: ArrayRef = Arc::new(UInt32Array::from_iter_values(
        table.class_id.iter().copied(),
    ));
    let ece: ArrayRef = Arc::new(Float64Array::from_iter_values(table.ece.iter().copied()));
    let mce: ArrayRef = Arc::new(Float64Array::from_iter_values(table.mce.iter().copied()));
    let n: ArrayRef = Arc::new(UInt64Array::from_iter_values(table.n.iter().copied()));

    RecordBatch::try_new(schema, vec![class_id, ece, mce, n])
}

fn reliability_schema() -> Schema {
    Schema::new(vec![
        Field::new("bin_id", DataType::UInt32, false),
        Field::new("score_lo", DataType::Float64, false),
        Field::new("score_hi", DataType::Float64, false),
        Field::new("mean_score", DataType::Float64, false),
        Field::new("accuracy", DataType::Float64, false),
        Field::new("count", DataType::UInt64, false),
        Field::new("gap", DataType::Float64, false),
        Field::new("ci_lo", DataType::Float64, false),
        Field::new("ci_hi", DataType::Float64, false),
    ])
    .with_metadata(table_metadata("calibration_reliability"))
}

fn per_class_calibration_schema() -> Schema {
    Schema::new(vec![
        Field::new("class_id", DataType::UInt32, false),
        Field::new("ece", DataType::Float64, false),
        Field::new("mce", DataType::Float64, false),
        Field::new("n", DataType::UInt64, false),
    ])
    .with_metadata(table_metadata("calibration_per_class"))
}

// ---------------------------------------------------------------------------
// `cells_from_grid` — free-function constructor used by the Python
// `Evaluator.evaluate(..., calibration=True)` wrapper to bridge an
// existing `PyEvalGrid` into an `EvalCells` handle.
// ---------------------------------------------------------------------------

/// Build an [`EvalCells`] handle from a fully-populated
/// [`crate::PyEvalGrid`].
///
/// Cell retention is opt-in at the caller level — the Python wrapper
/// passes `calibration=True` on
/// `instance.Evaluator.evaluate(...)` to keep the grid alive past the
/// summarize step, then calls this function to mint the handle the
/// `EvalResult.calibration(...)` accessor consumes.
///
/// Each populated cell is **deep-cloned** (the grid keeps its
/// `eval_imgs` intact for subsequent table builders / LRP / TIDE folds
/// off the same grid). `None` slots are skipped, so the work is
/// proportional to detection volume — not the dense `K * A * I` grid —
/// but each populated `PerImageEval` carries `dt_scores: Vec<f64>` plus
/// two `Array2<bool>`, all of which are copied. Acceptable today
/// because cells are opt-in via `calibration=True`; revisit with `Arc`
/// or ownership-transfer if a streaming-calibration path lands.
#[pyfunction]
pub(crate) fn cells_from_grid(grid: &crate::PyEvalGrid) -> PyResult<EvalCells> {
    let inner = grid.eval_grid_ref();
    let cells: Vec<Option<Box<PerImageEval>>> = inner
        .eval_imgs
        .iter()
        .map(|slot| slot.as_ref().map(|cell| Box::new(cell.as_ref().clone())))
        .collect();
    Ok(EvalCells::new(
        cells,
        inner.n_categories,
        inner.n_area_ranges,
        grid.iou_thresholds().to_vec(),
        grid.parity_mode(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use vernier_core::accumulate::PerImageEval;
    use vernier_core::calibration::{CalibrationParams, PerClassTable, ReliabilityTable};
    use vernier_core::ParityMode;

    fn cell(scores: &[f64], matched: &[bool], ignore: &[bool]) -> PerImageEval {
        let d = scores.len();
        let mut dt_matched: Array2<bool> = Array2::from_elem((1, d), false);
        let mut dt_ignore: Array2<bool> = Array2::from_elem((1, d), false);
        for (j, &m) in matched.iter().enumerate() {
            dt_matched[(0, j)] = m;
        }
        for (j, &ig) in ignore.iter().enumerate() {
            dt_ignore[(0, j)] = ig;
        }
        PerImageEval {
            dt_scores: scores.to_vec(),
            dt_matched,
            dt_ignore,
            gt_ignore: Vec::new(),
        }
    }

    type AnyError = Box<dyn std::error::Error>;

    #[test]
    fn reliability_record_batch_columns_and_metadata() -> Result<(), AnyError> {
        // Sanity: build a known-good summary by running the kernel and
        // verify the Arrow batch comes out with the contracted columns
        // and metadata stamp.
        let grid: Vec<Option<Box<PerImageEval>>> = vec![Some(Box::new(cell(
            &[0.1, 0.5, 0.9],
            &[false, true, true],
            &[false, false, false],
        )))];
        let params = CalibrationParams {
            n_bins: 3,
            min_score: 0.0,
            ..CalibrationParams::default()
        };
        let summary = summarize_calibration(&grid, 1, 1, &params, ParityMode::Strict)?;
        let batch = reliability_to_arrow(&summary)?;

        let schema = batch.schema();
        let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(
            names,
            vec![
                "bin_id",
                "score_lo",
                "score_hi",
                "mean_score",
                "accuracy",
                "count",
                "gap",
                "ci_lo",
                "ci_hi",
            ]
        );
        assert_eq!(
            schema
                .metadata()
                .get("vernier.schema_version")
                .map(String::as_str),
            Some("1")
        );
        assert_eq!(
            schema.metadata().get("vernier.table").map(String::as_str),
            Some("calibration_reliability")
        );
        assert_eq!(batch.num_rows(), summary.effective_n_bins);
        Ok(())
    }

    #[test]
    fn per_class_calibration_record_batch_columns_and_metadata() -> Result<(), AnyError> {
        let table = PerClassTable {
            class_id: vec![0, 3, 7],
            ece: vec![0.1, 0.2, 0.05],
            mce: vec![0.3, 0.4, 0.07],
            n: vec![10, 20, 5],
        };
        let batch = per_class_calibration_to_arrow(&table)?;
        let schema = batch.schema();
        let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(names, vec!["class_id", "ece", "mce", "n"]);
        assert_eq!(
            schema.metadata().get("vernier.table").map(String::as_str),
            Some("calibration_per_class")
        );
        assert_eq!(batch.num_rows(), 3);
        Ok(())
    }

    #[test]
    fn reliability_empty_summary_yields_empty_batch() -> Result<(), AnyError> {
        // No populated cells → empty summary → zero-row batch.
        let grid: Vec<Option<Box<PerImageEval>>> = vec![None];
        let params = CalibrationParams {
            min_score: 0.0,
            ..CalibrationParams::default()
        };
        let summary = summarize_calibration(&grid, 1, 1, &params, ParityMode::Strict)?;
        let batch = reliability_to_arrow(&summary)?;
        assert_eq!(batch.num_rows(), 0);
        // Schema is still well-formed.
        assert_eq!(batch.schema().fields().len(), 9);
        Ok(())
    }

    // Construct a dummy ReliabilityTable to exercise the column-major
    // builder independent of the kernel.
    #[test]
    fn reliability_columns_handle_nan_count_zero() -> Result<(), AnyError> {
        let summary = CalibrationSummary {
            ece: f64::NAN,
            mce: f64::NAN,
            n_detections: 0,
            effective_n_bins: 2,
            reliability: ReliabilityTable {
                bin_id: vec![0, 1],
                score_lo: vec![0.0, 0.5],
                score_hi: vec![0.5, 1.0],
                mean_score: vec![f64::NAN, 0.75],
                accuracy: vec![f64::NAN, 1.0],
                count: vec![0, 2],
                gap: vec![f64::NAN, 0.25],
                ci_lo: vec![f64::NAN, 0.5],
                ci_hi: vec![f64::NAN, 0.95],
            },
            per_class: None,
        };
        let batch = reliability_to_arrow(&summary)?;
        assert_eq!(batch.num_rows(), 2);
        Ok(())
    }
}
