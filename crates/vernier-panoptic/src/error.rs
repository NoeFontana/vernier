//! Typed errors raised by the panoptic evaluation surface.
//!
//! Mirrors the structured-variant pattern of [`vernier_core::error::EvalError`]:
//! every error carries enough context for the FFI layer to surface a
//! Python exception with attributes the user can lift programmatically
//! (image id, segment id, gt/dt shapes, etc.). Catch-all `String`
//! detail fields are reserved for cases where the structured fields
//! would be more code than they are worth.
//!
//! Each variant is doc-tagged with the quirk ID from the ADR-0025
//! appendix it corresponds to — these are the load-bearing
//! `corrected`-disposition rows: each replaces a panopticapi raise
//! site that surfaces an unattributed `KeyError` / `Exception` /
//! `ValueError` mid-eval.

use thiserror::Error;

/// Errors raised by the panoptic evaluation kernel and the
/// [`from_files`] / [`from_arrays`] dataset constructors.
///
/// [`from_files`]: crate
/// [`from_arrays`]: crate
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PanopticError {
    /// GT and DT label maps for the same image disagree on shape.
    /// panopticapi produces a NumPy broadcast error inside the OFFSET
    /// multiplication with no failing-image attribution (quirk **R4**);
    /// vernier rejects with this typed variant at the FFI boundary so
    /// the offending image is surfaced. Disposition: **corrected**.
    #[error("panoptic shape mismatch on image_id={image_id}: gt={gt_shape:?}, dt={dt_shape:?}")]
    ShapeMismatch {
        /// COCO image id of the offending image.
        image_id: i64,
        /// Ground-truth label-map shape `(height, width)`.
        gt_shape: (u32, u32),
        /// Detection label-map shape `(height, width)`.
        dt_shape: (u32, u32),
    },

    /// A non-VOID id encountered in the prediction PNG has no entry in
    /// `pred['segments_info']`. panopticapi raises a bare `KeyError`
    /// (quirk **S1**, `evaluation.py:96-101`); vernier surfaces the
    /// id and image so the user can patch their `segments_info` JSON.
    /// Disposition: **strict** (matches upstream; structured field is
    /// the only deviation).
    #[error(
        "unknown prediction segment id {segment_id} in image_id={image_id}: not present in segments_info"
    )]
    UnknownPredSegmentId {
        /// COCO image id where the unknown segment appears.
        image_id: i64,
        /// Segment id present in the PNG but absent from `segments_info`.
        segment_id: u32,
    },

    /// A segment id declared in `pred['segments_info']` is absent from
    /// the prediction PNG. panopticapi catches this via
    /// `pred_labels_set` non-emptiness after the PNG walk
    /// (`evaluation.py:106-107`); vernier raises this typed variant
    /// — a missed pred entry would hide an FP or a TP. Disposition:
    /// **strict** (sibling to S8 GT-side; quirk **S11**).
    #[error(
        "prediction segment id {segment_id} declared in segments_info is missing from PNG for image_id={image_id}"
    )]
    MissingPredSegmentInPng {
        /// COCO image id.
        image_id: i64,
        /// Segment id declared in JSON but absent from PNG.
        segment_id: u32,
    },

    /// Two `segments_info` entries on the same image carry the same
    /// `id`. panopticapi silently keeps the **last** via dict
    /// comprehension `{el['id']: el for el in segments_info}`
    /// (`evaluation.py:91-92`, quirk **S7**); vernier rejects in the
    /// `Corrected` default. `Strict` mode reproduces the last-wins
    /// behavior. Disposition: **corrected**.
    #[error(
        "duplicate segment id {segment_id} in segments_info for image_id={image_id} (side={side})"
    )]
    DuplicateSegmentId {
        /// COCO image id.
        image_id: i64,
        /// The duplicated segment id.
        segment_id: u32,
        /// `"gt"` or `"dt"` — which side carries the duplicate.
        side: &'static str,
    },

    /// A panoptic PNG file referenced by the dataset is absent.
    /// panopticapi raises bare `FileNotFoundError` from `Image.open`
    /// (quirk **R6**, `evaluation.py:86`) with no image-id context;
    /// vernier surfaces the image id and the path. Disposition:
    /// **strict** (predictions-must-cover-GT, Y4).
    #[error("panoptic PNG missing for image_id={image_id}: {path}")]
    MissingPanopticImage {
        /// COCO image id whose PNG is absent.
        image_id: i64,
        /// Path that was opened.
        path: String,
    },

    /// A GT image has no corresponding prediction. panopticapi raises
    /// bare `Exception` (`evaluation.py:215-216`, quirk **Y4**); vernier
    /// surfaces the image id. Predictions-must-cover-GT is metric-defining
    /// (no "missing pred → all FN" branch in upstream). Disposition:
    /// **strict** (with structured field).
    #[error("missing prediction for GT image_id={image_id}")]
    MissingPredictionsForImage {
        /// COCO image id.
        image_id: i64,
    },

    /// A pixel in the prediction PNG has a panoptic mode (RGBA / P /
    /// L) that vernier rejects at the FFI boundary. panopticapi
    /// silently drops alpha on RGBA (`utils.py:77` `rgb2id` indexes
    /// `[:,:,0..2]` only) and crashes mid-eval on P / L. Disposition:
    /// **corrected** (quirk **R2** non-RGB path).
    #[error("non-RGB panoptic PNG for image_id={image_id}: mode={mode}")]
    NonRgbPng {
        /// COCO image id.
        image_id: i64,
        /// PNG color mode as decoded by the `png` crate.
        mode: &'static str,
    },

    /// Malformed segments_info or category JSON.
    #[error(transparent)]
    Json(#[from] serde_json::Error),

    /// PNG decode failure surfaced from the `png` crate.
    #[error("PNG decode failed: {0}")]
    Png(String),

    /// Generic invalid-input case for situations that don't warrant a
    /// dedicated variant (e.g. negative shape, non-finite IoU). Uses a
    /// `String` detail for the same reason `EvalError::InvalidConfig`
    /// does in `vernier-core`.
    #[error("invalid panoptic input: {detail}")]
    InvalidInput {
        /// Human-readable description of the failure.
        detail: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_mismatch_carries_structured_fields() {
        let e = PanopticError::ShapeMismatch {
            image_id: 42,
            gt_shape: (480, 640),
            dt_shape: (640, 480),
        };
        let msg = format!("{e}");
        assert!(msg.contains("image_id=42"));
        assert!(msg.contains("(480, 640)"));
        assert!(msg.contains("(640, 480)"));
    }

    #[test]
    fn duplicate_segment_id_distinguishes_side() {
        let gt = PanopticError::DuplicateSegmentId {
            image_id: 1,
            segment_id: 99,
            side: "gt",
        };
        let dt = PanopticError::DuplicateSegmentId {
            image_id: 1,
            segment_id: 99,
            side: "dt",
        };
        assert!(format!("{gt}").contains("side=gt"));
        assert!(format!("{dt}").contains("side=dt"));
    }

    #[test]
    fn json_error_round_trips_through_from() {
        let parse_err = serde_json::from_str::<i32>("not json").unwrap_err();
        let panoptic: PanopticError = parse_err.into();
        assert!(matches!(panoptic, PanopticError::Json(_)));
    }
}
