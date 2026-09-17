//! COCO *result*-shaped detection ingest: the two routes that carry
//! `loadRes`-shaped detections without going through JSON.
//!
//! Both routes are pure data conversion. They terminate in
//! `Vec<DetectionInput>` — the same column layout
//! [`CocoDetections::from_json_bytes`] hands to
//! [`CocoDetections::from_inputs`] — so the `loadRes` semantics
//! (quirks **J1**, **J3**, **J4**/**E2**) are *inherited* from core
//! rather than restated here. Nothing downstream can tell which route
//! produced a payload, because after this module there is no route:
//! there is one `Vec<DetectionInput>`.
//!
//! - [`ann_dicts_to_inputs`] — a Python list of per-annotation result
//!   dicts, the shape `pycocotools.COCO.loadRes` consumes and the one
//!   TorchMetrics-style callers build. Keys are fetched with
//!   module-lifetime interned strings (`intern!`), so `PyDict_GetItem`
//!   takes the pointer-equality fast path instead of hashing and
//!   comparing a freshly-built `str` per annotation per field.
//! - [`matrix_to_inputs`] — an `(N, 7)` C-contiguous float64 array laid
//!   out as `image_id, x, y, w, h, score, category_id`. No per-element
//!   Python object is touched at all.
//!
//! Neither routine can release the GIL: both read Python objects. They
//! are written as one tight pass so that the serial floor they impose
//! is as short as it can be, and the caller detaches immediately after.

use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PySequence};

use vernier_core::dataset::{Bbox, CategoryId, DetectionInput, ImageId};

use crate::array_ingest::{extract_one_rle, ArrayIouType, CastCtx};
use crate::dlpack;

/// Largest magnitude at which every integer is exactly representable in
/// a float64. Beyond it, consecutive integers share a double, so an
/// `image_id` or `category_id` that arrived as f64 can no longer be
/// trusted to be the one the caller meant.
const MAX_EXACT_INT_F64: f64 = 9_007_199_254_740_992.0; // 2^53

// ---------------------------------------------------------------------------
// Route 1: list of per-annotation result dicts
// ---------------------------------------------------------------------------

/// Does this dict look like a COCO *result annotation* rather than a
/// columnar `Detections` dict (ADR-0030)?
///
/// The two shapes are disjoint in practice: a `Detections` dict carries
/// the plural, columnar `boxes`, while a result annotation carries the
/// singular `bbox` and a per-annotation `category_id`. `boxes` is
/// checked first and wins, so an existing ADR-0030 payload can never be
/// re-routed by this predicate.
pub(crate) fn dict_is_result_annotation(dict: &Bound<'_, PyDict>) -> PyResult<bool> {
    let py = dict.py();
    if dict.contains(intern!(py, "boxes"))? {
        return Ok(false);
    }
    Ok(dict.contains(intern!(py, "bbox"))? || dict.contains(intern!(py, "category_id"))?)
}

/// Convert a list of per-annotation result dicts into the shared column
/// layout.
///
/// One pass, no intermediate allocation per annotation beyond the
/// `DetectionInput` itself. Every key is an interned, module-lifetime
/// `PyString`: `intern!` caches it in the interpreter's static string
/// table on first use, so the `PyDict_GetItem` probe compares pointers
/// on the fast path.
pub(crate) fn ann_dicts_to_inputs<'py>(
    py: Python<'py>,
    dicts: &[Bound<'py, PyDict>],
    iou_type: ArrayIouType,
    ctx: &CastCtx<'py, '_>,
) -> PyResult<Vec<DetectionInput>> {
    // Resolve the eight keys once for the whole batch rather than once
    // per annotation. `intern!` is already a cached lookup, but hoisting
    // it also hoists the `Bound` clone out of the inner loop.
    let k_image_id = intern!(py, "image_id");
    let k_category_id = intern!(py, "category_id");
    let k_bbox = intern!(py, "bbox");
    let k_score = intern!(py, "score");
    let k_id = intern!(py, "id");
    let k_segmentation = intern!(py, "segmentation");
    let k_keypoints = intern!(py, "keypoints");
    let k_num_keypoints = intern!(py, "num_keypoints");

    let mut inputs = Vec::with_capacity(dicts.len());
    for (i, dict) in dicts.iter().enumerate() {
        let image_id: i64 = required(dict, k_image_id, i, "image_id")?
            .extract()
            .map_err(|e| field_err(i, "image_id", &format!("expected int: {e}")))?;
        let category_id: i64 = required(dict, k_category_id, i, "category_id")?
            .extract()
            .map_err(|e| field_err(i, "category_id", &format!("expected int: {e}")))?;
        let score: f64 = required(dict, k_score, i, "score")?
            .extract()
            .map_err(|e| field_err(i, "score", &format!("expected float: {e}")))?;
        let bbox = extract_bbox(&required(dict, k_bbox, i, "bbox")?, i)?;

        // Quirk J1 (**strict**): an absent `id` is auto-assigned
        // 1..N by position, in `from_inputs`. A supplied `id` is
        // preserved rather than overwritten — vernier's long-standing
        // documented position on J1, unchanged by this route, and the
        // reason this function reads `id` at all instead of dropping
        // it the way `loadRes` effectively does.
        let id = match dict.get_item(k_id)? {
            Some(v) if !v.is_none() => {
                Some(vernier_core::dataset::AnnId(v.extract().map_err(|e| {
                    field_err(i, "id", &format!("expected int: {e}"))
                })?))
            }
            _ => None,
        };

        let segmentation = match iou_type {
            ArrayIouType::Segm | ArrayIouType::Boundary => {
                let obj = required(dict, k_segmentation, i, "segmentation")?;
                Some(extract_one_rle(&obj, i, ctx)?)
            }
            ArrayIouType::Bbox | ArrayIouType::Keypoints => dict
                .get_item(k_segmentation)?
                .filter(|v| !v.is_none())
                .map(|obj| extract_one_rle(&obj, i, ctx))
                .transpose()?,
        };

        let keypoints = match iou_type {
            ArrayIouType::Keypoints => Some(extract_keypoints(
                &required(dict, k_keypoints, i, "keypoints")?,
                i,
            )?),
            _ => None,
        };
        let num_keypoints = match dict.get_item(k_num_keypoints)? {
            Some(v) if !v.is_none() => Some(
                v.extract()
                    .map_err(|e| field_err(i, "num_keypoints", &format!("expected int: {e}")))?,
            ),
            _ => None,
        };

        inputs.push(DetectionInput {
            id,
            image_id: ImageId(image_id),
            category_id: CategoryId(category_id),
            score,
            bbox,
            segmentation,
            keypoints,
            num_keypoints,
        });
    }
    Ok(inputs)
}

fn required<'py>(
    dict: &Bound<'py, PyDict>,
    key: &Bound<'py, pyo3::types::PyString>,
    i: usize,
    name: &str,
) -> PyResult<Bound<'py, PyAny>> {
    dict.get_item(key)?
        .ok_or_else(|| field_err(i, name, "missing required field"))
}

fn field_err(i: usize, name: &str, detail: &str) -> PyErr {
    PyValueError::new_err(format!("detections[{i}].{name}: {detail}"))
}

/// `[x, y, w, h]`, accepted from any 4-element sequence so lists,
/// tuples and 1-D numpy arrays all work.
fn extract_bbox(obj: &Bound<'_, PyAny>, i: usize) -> PyResult<Bbox> {
    let seq = obj
        .cast::<PySequence>()
        .map_err(|e| field_err(i, "bbox", &format!("expected a 4-element sequence: {e}")))?;
    let len = seq.len()?;
    if len != 4 {
        return Err(field_err(
            i,
            "bbox",
            &format!("expected length-4 [x, y, w, h], got length {len}"),
        ));
    }
    let mut v = [0.0f64; 4];
    for (j, slot) in v.iter_mut().enumerate() {
        *slot = seq
            .get_item(j)?
            .extract()
            .map_err(|e| field_err(i, "bbox", &format!("element {j} is not a float: {e}")))?;
    }
    Ok(Bbox {
        x: v[0],
        y: v[1],
        w: v[2],
        h: v[3],
    })
}

/// Flat `[x, y, v, ...]` triplets, per ADR-0012.
fn extract_keypoints(obj: &Bound<'_, PyAny>, i: usize) -> PyResult<Vec<f64>> {
    let seq = obj.cast::<PySequence>().map_err(|e| {
        field_err(
            i,
            "keypoints",
            &format!("expected a flat [x, y, v, ...] sequence: {e}"),
        )
    })?;
    let len = seq.len()?;
    if len % 3 != 0 {
        return Err(field_err(
            i,
            "keypoints",
            &format!("length {len} is not a multiple of 3 (x, y, v triplets)"),
        ));
    }
    let mut out = Vec::with_capacity(len);
    for j in 0..len {
        out.push(
            seq.get_item(j)?.extract().map_err(|e| {
                field_err(i, "keypoints", &format!("element {j} is not a float: {e}"))
            })?,
        );
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Route 2: (N, 7) float64 matrix
// ---------------------------------------------------------------------------

/// Column order of the `(N, 7)` matrix route.
const MATRIX_COLS: usize = 7;
const COL_NAMES: [&str; MATRIX_COLS] = ["image_id", "x", "y", "w", "h", "score", "category_id"];

/// Convert an `(N, 7)` C-contiguous float64 array into the shared
/// column layout.
///
/// Layout: `image_id, x, y, w, h, score, category_id`.
///
/// Validation is explicit and refuses rather than repairs:
///
/// - **dtype** must be float64 and **C-contiguity** is required; both
///   are enforced by [`dlpack::extract_f64_2d`], which raises a
///   `TypeError` naming `np.ascontiguousarray` as the fix. We do not
///   silently copy: a hidden copy of an `(N, 7)` array is exactly the
///   cost this route exists to avoid, so the caller is told instead.
/// - **shape** must be 2-D with exactly 7 columns.
/// - `image_id` and `category_id` are **round-trip checked**: a value
///   that is not finite, has a fractional part, or exceeds 2^53 is
///   rejected. Truncating silently would turn `category_id=3.5` into
///   category 3 and a 2^60 image id into a neighbouring image's
///   detections — both of which change the evaluated dataset without
///   any diagnostic.
///
/// `score` is taken verbatim; `from_inputs` rejects a non-finite score.
pub(crate) fn matrix_to_inputs(obj: &Bound<'_, PyAny>) -> PyResult<Vec<DetectionInput>> {
    let view = dlpack::extract_f64_2d(obj, "detections", MATRIX_COLS)?;
    let flat = view.as_slice();
    let n = flat.len() / MATRIX_COLS;
    let mut inputs = Vec::with_capacity(n);
    for i in 0..n {
        let row = &flat[i * MATRIX_COLS..(i + 1) * MATRIX_COLS];
        inputs.push(DetectionInput {
            id: None,
            image_id: ImageId(exact_int(row[0], 0, i)?),
            category_id: CategoryId(exact_int(row[6], 6, i)?),
            score: row[5],
            bbox: Bbox {
                x: row[1],
                y: row[2],
                w: row[3],
                h: row[4],
            },
            segmentation: None,
            keypoints: None,
            num_keypoints: None,
        });
    }
    Ok(inputs)
}

/// Exact-integer round-trip check for a float64 id column.
fn exact_int(v: f64, col: usize, row: usize) -> PyResult<i64> {
    let name = COL_NAMES[col];
    if !v.is_finite() {
        return Err(PyValueError::new_err(format!(
            "detections[{row}].{name}: expected an integer, got {v}"
        )));
    }
    if v.fract() != 0.0 {
        return Err(PyValueError::new_err(format!(
            "detections[{row}].{name}: {v} is not an integer; \
             the (N, 7) float64 route will not truncate an id"
        )));
    }
    if v.abs() > MAX_EXACT_INT_F64 {
        return Err(PyValueError::new_err(format!(
            "detections[{row}].{name}: {v} exceeds 2^53, beyond which float64 \
             cannot represent every integer exactly; pass this dataset as a \
             list of result dicts or as JSON"
        )));
    }
    Ok(v as i64)
}

/// Is this object plausibly the `(N, 7)` matrix rather than a sequence
/// of dicts? NumPy arrays and torch tensors satisfy the `Sequence`
/// protocol, so the dispatch has to probe for the buffer protocol
/// *before* the sequence branch or an array would be iterated row by
/// row and rejected as "not a dict".
pub(crate) fn looks_like_matrix(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    obj.hasattr("__dlpack_device__")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_int_accepts_integral_values() -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(exact_int(0.0, 0, 0)?, 0);
        assert_eq!(exact_int(42.0, 6, 0)?, 42);
        assert_eq!(exact_int(-7.0, 0, 0)?, -7);
        assert_eq!(exact_int(MAX_EXACT_INT_F64, 0, 0)?, 9_007_199_254_740_992);
        Ok(())
    }

    // Rendering a `PyErr` needs a live interpreter, which these unit
    // tests do not have (`auto-initialize` is off). The *text* of each
    // rejection is asserted from Python instead, in
    // `tests/python/test_ingest_route_equivalence.py`; here we pin the
    // decision itself.
    #[test]
    fn exact_int_rejects_fractional_values() {
        assert!(exact_int(3.5, 6, 11).is_err());
        assert!(exact_int(-0.5, 0, 0).is_err());
        assert!(exact_int(1e-9, 0, 0).is_err());
    }

    #[test]
    fn exact_int_rejects_beyond_two_pow_53() {
        assert!(exact_int(MAX_EXACT_INT_F64 * 4.0, 0, 0).is_err());
        assert!(exact_int(-MAX_EXACT_INT_F64 * 4.0, 0, 0).is_err());
    }

    #[test]
    fn exact_int_rejects_non_finite() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(exact_int(bad, 0, 0).is_err(), "{bad} must be rejected");
        }
    }

    #[test]
    fn column_names_cover_the_matrix_width() {
        assert_eq!(COL_NAMES.len(), MATRIX_COLS);
        assert_eq!(COL_NAMES[0], "image_id");
        assert_eq!(COL_NAMES[6], "category_id");
    }
}
