//! Shared numpy / PyDict parsing helpers used by the panoptic and
//! semantic FFI modules.
//!
//! Both surfaces accept a Python dict of `image_id -> uint32 (H, W)
//! ndarray` as the canonical input shape (panoptic encodes segment
//! ids; semantic encodes class ids). The parse logic — extract the
//! key, extract the 2-D uint32 array, validate the shape fits in
//! `u32`, materialize to a flat `Vec<u32>` — is identical between
//! them. This module is the single home.

use std::collections::HashMap;

use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};

/// Image id type; `i64` matches the existing surface in
/// [`vernier_core::dataset::ImageId`] and the panoptic / semantic
/// crates. Aliased here to keep the helper signature readable.
pub(crate) type ImageId = i64;

/// Per-image label-map shape `(height, width, flat_pixels)`. The
/// kernel walks the flattened pixel buffer linearly; row-major
/// flatten is a single allocation per image.
pub(crate) type LabelMap = (u32, u32, Vec<u32>);

/// Parse a Python dict of `image_id -> 2-D uint32 ndarray` into a
/// `HashMap<ImageId, LabelMap>`.
///
/// `context` is the error-message prefix (e.g., `"panoptic"`,
/// `"semantic gt"`, `"semantic dt"`); it is interpolated into every
/// `PyValueError` this function can raise. Callers pass a static
/// string identifying both the subsystem and the side of the input
/// pair so the user can attribute the failure without reading
/// stack-trace context.
///
/// Surfaces typed `PyValueError`s for: non-integer image ids, non-2-D
/// uint32 arrays, shapes exceeding `u32::MAX` (defensive), and
/// duplicate image ids in the dict (which `PyDict` shouldn't allow
/// but a non-canonical `Mapping` shim could).
pub(crate) fn parse_uint32_label_maps<'py>(
    label_maps: &Bound<'py, PyDict>,
    context: &str,
) -> PyResult<HashMap<ImageId, LabelMap>> {
    let mut out: HashMap<ImageId, LabelMap> = HashMap::with_capacity(label_maps.len());
    for (key, value) in label_maps.iter() {
        let image_id: ImageId = key.extract().map_err(|e| {
            PyValueError::new_err(format!(
                "{context} label_maps dict key must be an integer image id: {e}"
            ))
        })?;
        let arr: PyReadonlyArray2<u32> = value.extract().map_err(|e| {
            PyValueError::new_err(format!(
                "{context} label_maps[{image_id}] must be a 2-D uint32 ndarray: {e}"
            ))
        })?;
        let view = arr.as_array();
        let (h, w) = (view.shape()[0], view.shape()[1]);
        if h > u32::MAX as usize || w > u32::MAX as usize {
            return Err(PyValueError::new_err(format!(
                "{context} label_maps[{image_id}] shape ({h}, {w}) exceeds u32 bounds"
            )));
        }
        let buf: Vec<u32> = view.iter().copied().collect();
        if out.insert(image_id, (h as u32, w as u32, buf)).is_some() {
            return Err(PyValueError::new_err(format!(
                "{context} label_maps has duplicate image_id={image_id}"
            )));
        }
    }
    Ok(out)
}
