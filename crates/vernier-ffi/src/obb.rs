//! Oriented-box diagnostics exposed to Python (ADR-0063).
//!
//! These are *not* evaluation. They answer questions a user asks before
//! or beside an evaluation — "is my angle convention right?", "what does
//! my label format cost me?", "was this detection misplaced or just
//! misoriented?" — and they route through `vernier-geom` so there is
//! exactly one implementation of the geometry in the project.

use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::parse_convention;

/// Convert `[cx, cy, w, h, theta]` to `[x0, y0, ..., x3, y3]`.
///
/// These are the **corrected** kernel's corner bits, which is worth
/// stating because it is a limit on what the result can be used for: a
/// DK-strict claim must be made against the quads you would actually
/// submit to DOTA_devkit, not against quads vernier synthesized. For
/// everything else — visualizing, converting a rotated-box dataset to a
/// quad one, comparing formats — this is the conversion.
#[pyfunction]
pub(crate) fn obb_rbox_to_quad(rbox: [f64; 5], unit: &str, rotation: &str) -> PyResult<[f64; 8]> {
    let conv = parse_convention(unit, rotation)?;
    let b = vernier_geom::RotatedBox::from_slice(&rbox)
        .map_err(|e| PyValueError::new_err(format!("rbox: {e}")))?;
    Ok(b.corners(conv))
}

/// Minimum-area enclosing rectangle of a quad, as `[cx, cy, w, h,
/// theta]` under `(unit, rotation)`.
///
/// Returns `None` for a degenerate quad — fewer than three distinct,
/// non-collinear vertices — rather than a zero-area rectangle that would
/// read as a real answer.
///
/// Explicitly **not** bit-equal to `cv2.minAreaRect`, and not a parity
/// surface. See `vernier_geom::calipers`.
#[pyfunction]
pub(crate) fn obb_min_area_rect(
    quad: [f64; 8],
    unit: &str,
    rotation: &str,
) -> PyResult<Option<[f64; 5]>> {
    let conv = parse_convention(unit, rotation)?;
    // Validate first so a non-finite coordinate is a typed error rather
    // than the `None` that also means "degenerate". `min_area_rect`
    // refuses non-finite input too, but it cannot tell the caller which
    // of the two happened.
    vernier_geom::Quad::from_slice(&quad)
        .map_err(|e| PyValueError::new_err(format!("quad: {e}")))?;
    Ok(vernier_geom::min_area_rect(&quad, conv).map(vernier_geom::RotatedBox::to_slice))
}

/// Per-class label ceiling: what a rectangle-predicting model gives up
/// on quad ground truth before it makes a single mistake.
///
/// For each annotation, `IoU(quad, minAreaRect(quad))` under the
/// canonical f64 kernel; the return is `category_id -> (mean, count)`.
/// A class at `0.93` is telling you that a *perfect* rotated-box
/// detector caps out near `0.93` IoU on it, which is below the `0.95`
/// rung of the COCO ladder — a headline-AP ceiling that has nothing to
/// do with the detector.
///
/// Degenerate quads are skipped rather than scored as zero: they are a
/// data problem, and averaging them in would hide the signal this
/// function exists to show. The per-class `count` is the number of
/// annotations that *were* scored, so a large gap against the class's
/// annotation count is itself a finding.
#[pyfunction]
// PyO3 extracts the sequences by value; the body only reads them, so
// clippy is right that they could be slices — but the extraction is
// what owns them, and there is no caller to borrow from.
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn obb_label_ceiling(
    quads: Vec<[f64; 8]>,
    category_ids: Vec<i64>,
    unit: &str,
    rotation: &str,
) -> PyResult<HashMap<i64, (f64, usize)>> {
    if quads.len() != category_ids.len() {
        return Err(PyValueError::new_err(format!(
            "quads and category_ids disagree on length: {} vs {}",
            quads.len(),
            category_ids.len()
        )));
    }
    let conv = parse_convention(unit, rotation)?;
    let mut sums: HashMap<i64, (f64, usize)> = HashMap::new();
    for (i, quad) in quads.iter().enumerate() {
        // A degenerate quad is skipped below, by design; a non-finite
        // one is not the same thing and gets said out loud, with the
        // index, because the answer would otherwise be a silently
        // shorter `n_scored`.
        vernier_geom::Quad::from_slice(quad)
            .map_err(|e| PyValueError::new_err(format!("quads[{i}]: {e}")))?;
    }
    for (quad, &cat) in quads.iter().zip(&category_ids) {
        let Some(rect) = vernier_geom::min_area_rect(quad, conv) else {
            continue;
        };
        let Ok(q) = vernier_geom::PreparedQuad::new(quad, false) else {
            continue;
        };
        let Ok(r) = vernier_geom::PreparedQuad::new(&rect.corners(conv), false) else {
            continue;
        };
        let iou = vernier_geom::quad_iou(&q, &r, vernier_geom::Denominator::Union);
        let slot = sums.entry(cat).or_insert((0.0, 0));
        slot.0 += iou;
        slot.1 += 1;
    }
    Ok(sums
        .into_iter()
        .map(|(cat, (sum, n))| {
            let mean = if n > 0 { sum / n as f64 } else { 0.0 };
            (cat, (mean, n))
        })
        .collect())
}

/// Orientation error between two rotated boxes, in degrees, in
/// `[0, 90]`.
///
/// Invariant to the angle parameterization — `theta + 180` and the
/// `w`/`h` swap describe the same box, so they must report the same
/// error — and independent of the IoU. `tau` is the near-square
/// tolerance: below it the box has no distinguishable long axis and the
/// error is reduced modulo 90 degrees rather than 180.
#[pyfunction]
#[pyo3(signature = (gt, dt, unit, rotation, tau = vernier_geom::NEAR_SQUARE_TAU))]
pub(crate) fn obb_angle_error_deg(
    gt: [f64; 5],
    dt: [f64; 5],
    unit: &str,
    rotation: &str,
    tau: f64,
) -> PyResult<f64> {
    let conv = parse_convention(unit, rotation)?;
    let prep = |v: &[f64; 5], what: &str| {
        vernier_geom::RotatedBox::from_slice(v)
            .map(|b| vernier_geom::PreparedRBox::new(b, conv))
            .map_err(|e| PyValueError::new_err(format!("{what}: {e}")))
    };
    Ok(vernier_geom::angle_error_deg(
        &prep(&gt, "gt")?,
        &prep(&dt, "dt")?,
        conv,
        tau,
    ))
}
