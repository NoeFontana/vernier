//! Array-based detection ingest for `StreamingEvaluator.update` and
//! `BackgroundEvaluator.submit` (ADR-0030).
//!
//! Routes Python `Detections` dicts through the same
//! `CocoDetections::from_inputs` constructor the JSON path uses; parity
//! is structural, not a separate invariant.

use std::sync::atomic::{AtomicBool, Ordering};

use pyo3::exceptions::{PyTypeError, PyUserWarning, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict, PySequence};

use vernier_core::{
    Bbox, CategoryId, CocoDetections, DetectionInput, ImageId, Segmentation, SegmentationRle,
    SegmentationRleCounts,
};

use crate::dlpack;
use crate::emit_warning;

// ---------------------------------------------------------------------------
// Top-level dispatch types
// ---------------------------------------------------------------------------

/// One incoming `update`/`submit` argument. The two variants without
/// payload data (`Bytes` and the dict-shape branches) are distinguished
/// at the call site so the array path doesn't carry a never-used `Bytes`
/// arm into its kernel-typed dispatch.
pub(crate) enum DetectionsArg<'py> {
    /// Legacy `loadRes`-shaped JSON bytes; routed straight to the
    /// existing JSON entry on the streaming/background state.
    Bytes(Vec<u8>),
    /// One or more per-image `Detections` dicts. Single-image inputs
    /// land here as a one-element vec.
    Dicts(Vec<Bound<'py, PyDict>>),
}

impl<'py> DetectionsArg<'py> {
    /// Classify the input. `bytes` is checked first because `bytes`
    /// instances also satisfy the `Sequence` protocol.
    pub(crate) fn extract(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(b) = obj.cast::<PyBytes>() {
            return Ok(Self::Bytes(b.as_bytes().to_vec()));
        }
        if let Ok(d) = obj.cast::<PyDict>() {
            return Ok(Self::Dicts(vec![d.clone()]));
        }
        if let Ok(seq) = obj.cast::<PySequence>() {
            let len = seq.len()?;
            let mut dicts: Vec<Bound<'py, PyDict>> = Vec::with_capacity(len);
            for i in 0..len {
                let item = seq.get_item(i)?;
                let dict = item.cast_into::<PyDict>().map_err(|e| {
                    PyTypeError::new_err(format!(
                        "detections[{i}]: expected a Detections dict: {e}"
                    ))
                })?;
                dicts.push(dict);
            }
            return Ok(Self::Dicts(dicts));
        }
        let type_name = obj
            .get_type()
            .name()
            .map_or_else(|_| "<unknown>".to_string(), |n| n.to_string());
        Err(PyTypeError::new_err(format!(
            "detections must be bytes, a Detections dict, or a sequence of Detections dicts; \
             got {type_name}"
        )))
    }
}

// ---------------------------------------------------------------------------
// IoU type discriminator
// ---------------------------------------------------------------------------

/// Which fields are required on each `Detections` dict, picked at
/// evaluator construction time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ArrayIouType {
    Bbox,
    Segm,
    Boundary,
    Keypoints,
}

impl ArrayIouType {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Bbox => "bbox",
            Self::Segm => "segm",
            Self::Boundary => "boundary",
            Self::Keypoints => "keypoints",
        }
    }
}

// ---------------------------------------------------------------------------
// cast_inputs state — Some(latch) when enabled, None when off. The latch
// is a one-shot AtomicBool that fires the UserWarning at most once over
// the evaluator's lifetime.
// ---------------------------------------------------------------------------

pub(crate) type CastState = Option<AtomicBool>;

pub(crate) fn new_cast_state(enabled: bool) -> CastState {
    if enabled {
        Some(AtomicBool::new(false))
    } else {
        None
    }
}

fn emit_cast_warning_once(py: Python<'_>, latch: &AtomicBool) -> PyResult<()> {
    if latch.swap(true, Ordering::Relaxed) {
        return Ok(());
    }
    emit_warning::<PyUserWarning>(
        py,
        "vernier-0030: cast_inputs=True silently promotes input dtypes (f32→f64, i32→i64); \
         disable to enforce the strict ADR-0004 boundary",
    )
}

// ---------------------------------------------------------------------------
// Per-image extraction
// ---------------------------------------------------------------------------

/// Per-call state threaded through validation helpers. Bundles the
/// Python token with the optional cast plan; `cast.is_some()` gates the
/// f32→f64 / i32→i64 promotion path. `numpy.ascontiguousarray` is
/// resolved once per [`dicts_to_detections`] call so the inner loops
/// don't re-walk `sys.modules` per (dict × field).
struct CastCtx<'py, 'a> {
    py: Python<'py>,
    cast: Option<(&'a AtomicBool, &'a Bound<'py, PyAny>)>,
}

impl<'py, 'a> CastCtx<'py, 'a> {
    fn maybe_cast(
        &self,
        obj: Bound<'py, PyAny>,
        field: &str,
        dtype: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        match self.cast {
            None => Ok(obj),
            Some((latch, ascontig)) => cast_via_numpy(self.py, &obj, field, dtype, latch, ascontig),
        }
    }
}

/// Extract one `Detections` dict into a flat `Vec<DetectionInput>` ready
/// to feed `CocoDetections::from_inputs`. `iou_type` controls which
/// fields are required and which are silently ignored.
fn extract_inputs_one(
    dict: &Bound<'_, PyDict>,
    iou_type: ArrayIouType,
    ctx: &CastCtx<'_, '_>,
) -> PyResult<Vec<DetectionInput>> {
    let image_id_obj = dict.get_item("image_id")?.ok_or_else(|| {
        PyValueError::new_err(
            "detections: missing required field 'image_id' \
             (each Detections dict must carry an integer image id)",
        )
    })?;
    let image_id_raw: i64 = image_id_obj.extract().map_err(|e| {
        PyValueError::new_err(format!("detections.image_id: expected int, got {e}"))
    })?;
    let image_id = ImageId(image_id_raw);

    let boxes_obj = dict.get_item("boxes")?.ok_or_else(|| {
        PyValueError::new_err("detections: missing required field 'boxes' (N×4 float64 xywh array)")
    })?;
    let boxes_obj = ctx.maybe_cast(boxes_obj, "boxes", "float64")?;
    let boxes_view = dlpack::extract_f64_2d(&boxes_obj, "boxes", 4)?;
    let boxes = boxes_view.as_slice();
    let n = boxes.len() / 4;

    let scores_obj = dict.get_item("scores")?.ok_or_else(|| {
        PyValueError::new_err("detections: missing required field 'scores' (length-N float64)")
    })?;
    let scores_obj = ctx.maybe_cast(scores_obj, "scores", "float64")?;
    let scores_view = dlpack::extract_f64_1d(&scores_obj, "scores")?;
    if scores_view.len() != n {
        return Err(PyValueError::new_err(format!(
            "detections.scores: length {} disagrees with boxes (N={n})",
            scores_view.len()
        )));
    }
    let scores = scores_view.as_slice();

    let labels_obj = dict.get_item("labels")?.ok_or_else(|| {
        PyValueError::new_err("detections: missing required field 'labels' (length-N int64)")
    })?;
    let labels_obj = ctx.maybe_cast(labels_obj, "labels", "int64")?;
    let labels_view = dlpack::extract_i64_1d(&labels_obj, "labels")?;
    if labels_view.len() != n {
        return Err(PyValueError::new_err(format!(
            "detections.labels: length {} disagrees with boxes (N={n})",
            labels_view.len()
        )));
    }
    let labels = labels_view.as_slice();

    let mut rles: Vec<Option<Segmentation>> = match iou_type {
        ArrayIouType::Segm | ArrayIouType::Boundary => {
            let rles_obj = dict.get_item("rles")?.ok_or_else(|| {
                PyValueError::new_err(format!(
                    "detections: iou_type={} requires a 'rles' field \
                     (sequence of {{counts: uint32, size: (h, w)}} dicts)",
                    iou_type.as_str()
                ))
            })?;
            extract_rles(&rles_obj, n)?.into_iter().map(Some).collect()
        }
        ArrayIouType::Bbox | ArrayIouType::Keypoints => Vec::new(),
    };

    let kp_data = match iou_type {
        ArrayIouType::Keypoints => {
            let kp_obj = dict.get_item("keypoints")?.ok_or_else(|| {
                PyValueError::new_err(
                    "detections: iou_type='keypoints' requires a 'keypoints' field \
                     ((N, K, 3) float64 array of [x, y, v] triplets)",
                )
            })?;
            let kp_obj = ctx.maybe_cast(kp_obj, "keypoints", "float64")?;
            let (view, k) = dlpack::extract_f64_3d_kp(&kp_obj, "keypoints")?;
            let stride = k * 3;
            let expected = n.checked_mul(stride).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "detections.keypoints: shape product (N={n}, K={k}, 3) overflows usize"
                ))
            })?;
            if view.len() != expected {
                return Err(PyValueError::new_err(format!(
                    "detections.keypoints: flat length {} disagrees with shape \
                     (N={n}, K={k}, 3)",
                    view.len()
                )));
            }
            Some((view, stride))
        }
        _ => None,
    };

    let mut inputs: Vec<DetectionInput> = Vec::with_capacity(n);
    for i in 0..n {
        let bbox = Bbox {
            x: boxes[4 * i],
            y: boxes[4 * i + 1],
            w: boxes[4 * i + 2],
            h: boxes[4 * i + 3],
        };
        let segmentation = rles.get_mut(i).and_then(Option::take);
        let keypoints = kp_data.as_ref().map(|(view, stride)| {
            let start = i * *stride;
            view.as_slice()[start..start + *stride].to_vec()
        });
        inputs.push(DetectionInput {
            id: None,
            image_id,
            category_id: CategoryId(labels[i]),
            score: scores[i],
            bbox,
            segmentation,
            keypoints,
            num_keypoints: None,
        });
    }
    Ok(inputs)
}

/// Pull a sequence of `RLE` dicts out of a Python value and decode each
/// into a `Segmentation::Rle` ready to land on `DetectionInput`.
fn extract_rles(obj: &Bound<'_, PyAny>, n: usize) -> PyResult<Vec<Segmentation>> {
    let seq = obj.cast::<PySequence>().map_err(|e| {
        PyTypeError::new_err(format!(
            "detections.rles: expected a sequence of RLE dicts: {e}"
        ))
    })?;
    let len = seq.len()?;
    if len != n {
        return Err(PyValueError::new_err(format!(
            "detections.rles: length {len} disagrees with boxes (N={n})"
        )));
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let item = seq.get_item(i)?;
        let dict = item.cast_into::<PyDict>().map_err(|e| {
            PyTypeError::new_err(format!(
                "detections.rles[{i}]: expected RLE dict {{counts, size}}: {e}"
            ))
        })?;
        let counts_obj = dict.get_item("counts")?.ok_or_else(|| {
            PyValueError::new_err(format!(
                "detections.rles[{i}]: missing 'counts' (uint32 1-D array)"
            ))
        })?;
        let size_obj = dict.get_item("size")?.ok_or_else(|| {
            PyValueError::new_err(format!("detections.rles[{i}]: missing 'size' (h, w) tuple"))
        })?;
        let (h, w) = extract_size_tuple(&size_obj, i)?;
        let counts_field = format!("rles[{i}].counts");
        let counts_view = dlpack::extract_u32_1d(&counts_obj, &counts_field)?;
        out.push(Segmentation::Rle(SegmentationRle {
            size: [h, w],
            counts: SegmentationRleCounts::Uncompressed(counts_view.as_slice().to_vec()),
        }));
    }
    Ok(out)
}

fn extract_size_tuple(obj: &Bound<'_, PyAny>, i: usize) -> PyResult<(u32, u32)> {
    // Accept any 2-element sequence so users can pass tuples, lists, or numpy arrays.
    let seq = obj.cast::<PySequence>().map_err(|e| {
        PyTypeError::new_err(format!(
            "detections.rles[{i}].size: expected (h, w) sequence: {e}"
        ))
    })?;
    if seq.len()? != 2 {
        return Err(PyValueError::new_err(format!(
            "detections.rles[{i}].size: expected length-2 (h, w), got length {}",
            seq.len()?
        )));
    }
    let h: u32 = seq.get_item(0)?.extract().map_err(|e| {
        PyValueError::new_err(format!(
            "detections.rles[{i}].size[0] (height): expected non-negative int: {e}"
        ))
    })?;
    let w: u32 = seq.get_item(1)?.extract().map_err(|e| {
        PyValueError::new_err(format!(
            "detections.rles[{i}].size[1] (width): expected non-negative int: {e}"
        ))
    })?;
    Ok((h, w))
}

// ---------------------------------------------------------------------------
// cast_inputs helper — promotes dtypes via np.ascontiguousarray.
// ---------------------------------------------------------------------------

fn cast_via_numpy<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
    field: &str,
    dtype: &str,
    latch: &AtomicBool,
    ascontiguousarray: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", dtype)?;
    let cast_result = ascontiguousarray.call((obj,), Some(&kwargs)).map_err(|e| {
        PyTypeError::new_err(format!(
            "detections.{field}: cast_inputs=True failed to coerce to {dtype}: {e}"
        ))
    })?;
    emit_cast_warning_once(py, latch)?;
    Ok(cast_result)
}

/// Resolve `numpy.ascontiguousarray` once per batch when `cast_inputs=True`.
fn resolve_ascontiguousarray(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    let np = py
        .import("numpy")
        .map_err(|e| PyTypeError::new_err(format!("cast_inputs=True requires numpy: {e}")))?;
    np.getattr("ascontiguousarray")
}

// ---------------------------------------------------------------------------
// Multi-image dispatch
// ---------------------------------------------------------------------------

/// Convert per-image dict payloads into a single `CocoDetections`,
/// matching what `from_json_bytes` produces for the same logical input.
pub(crate) fn dicts_to_detections(
    py: Python<'_>,
    dicts: &[Bound<'_, PyDict>],
    iou_type: ArrayIouType,
    cast_state: &CastState,
) -> PyResult<CocoDetections> {
    let ascontig = match cast_state {
        Some(_) => Some(resolve_ascontiguousarray(py)?),
        None => None,
    };
    let ctx = CastCtx {
        py,
        cast: cast_state.as_ref().zip(ascontig.as_ref()),
    };
    let mut all_inputs: Vec<DetectionInput> = Vec::new();
    for dict in dicts {
        all_inputs.extend(extract_inputs_one(dict, iou_type, &ctx)?);
    }
    CocoDetections::from_inputs(all_inputs)
        .map_err(|e| PyValueError::new_err(format!("detections array ingest: {e}")))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// `from_inputs` of array-derived records yields the same indices as
    /// `from_json_bytes` for the same logical detections. The new path is
    /// just another way to construct the same `Vec<DetectionInput>`.
    #[test]
    fn array_inputs_match_json_for_bbox_only() -> Result<(), Box<dyn std::error::Error>> {
        let inputs = vec![
            DetectionInput {
                id: None,
                image_id: ImageId(1),
                category_id: CategoryId(7),
                score: 0.9,
                bbox: Bbox {
                    x: 1.0,
                    y: 2.0,
                    w: 3.0,
                    h: 4.0,
                },
                segmentation: None,
                keypoints: None,
                num_keypoints: None,
            },
            DetectionInput {
                id: None,
                image_id: ImageId(1),
                category_id: CategoryId(7),
                score: 0.8,
                bbox: Bbox {
                    x: 5.0,
                    y: 6.0,
                    w: 7.0,
                    h: 8.0,
                },
                segmentation: None,
                keypoints: None,
                num_keypoints: None,
            },
        ];

        let json = serde_json::to_vec(&inputs)?;
        let from_json = CocoDetections::from_json_bytes(&json)?;
        let from_arr = CocoDetections::from_inputs(inputs)?;

        assert_eq!(from_json.detections().len(), from_arr.detections().len());
        for (a, b) in from_json
            .detections()
            .iter()
            .zip(from_arr.detections().iter())
        {
            assert_eq!(a.id, b.id);
            assert_eq!(a.image_id, b.image_id);
            assert_eq!(a.category_id, b.category_id);
            assert_eq!(a.score, b.score);
            assert_eq!(a.bbox, b.bbox);
            assert_eq!(a.area, b.area);
            assert_eq!(a.segmentation, b.segmentation);
        }
        assert_eq!(
            from_json.indices_for(ImageId(1), CategoryId(7)).len(),
            from_arr.indices_for(ImageId(1), CategoryId(7)).len()
        );
        Ok(())
    }

    #[test]
    fn iou_type_as_str_pins_user_facing_strings() {
        // The strings flow through error messages users grep for; pin them.
        assert_eq!(ArrayIouType::Bbox.as_str(), "bbox");
        assert_eq!(ArrayIouType::Segm.as_str(), "segm");
        assert_eq!(ArrayIouType::Boundary.as_str(), "boundary");
        assert_eq!(ArrayIouType::Keypoints.as_str(), "keypoints");
    }
}
