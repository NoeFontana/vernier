//! Typed errors raised by the semantic-segmentation evaluation surface.
//!
//! Mirrors the structured-variant pattern of
//! [`vernier_panoptic::error::PanopticError`] and
//! [`vernier_core::error::EvalError`]: every error carries enough
//! context for the FFI layer to surface a Python exception with
//! attributes the user can lift programmatically (image id, class id,
//! gt/dt shapes, etc.). Catch-all `String` detail fields are reserved
//! for cases where the structured fields would be more code than they
//! are worth.
//!
//! Each variant is doc-tagged with the quirk ID from the
//! [sem-seg quirks survey](../../../docs/engineering/sem-seg-quirks.md)
//! it corresponds to. The `(quirk_id, oracle) → mode` cells in that
//! survey determine which oracles a given variant is `strict` /
//! `corrected` against.

use thiserror::Error;

/// COCO image id. Mirrors [`vernier_panoptic::dataset::ImageId`] in
/// width so the cross-crate FFI surface uses the same integer type.
pub type ImageId = i64;

/// Semantic-segmentation class id. `u32` matches the input pixel
/// width: per ADR-0028 §"Numerical layout", semantic GT / DT pixels
/// fit comfortably in `u32` (Cityscapes uses `u8`, ADE20K `u16`; the
/// Rust side normalizes to `u32` at the FFI boundary). The error
/// variants surface invalid class ids in this width to match what the
/// kernel sees.
pub type ClassId = u32;

/// Errors raised by the semantic evaluation kernel and the
/// [`from_arrays`] / [`from_files`] / [`from_binary_masks`] dataset
/// constructors that PR-B3 onward implements.
///
/// [`from_arrays`]: crate
/// [`from_files`]: crate
/// [`from_binary_masks`]: crate
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum SemanticError {
    /// GT and DT label maps for the same image disagree on shape.
    /// mmsegmentation's `IoUMetric` raises a `ValueError` from NumPy
    /// on shape mismatch with no failing-image attribution; vernier
    /// rejects with this typed variant at the dataset-constructor
    /// boundary so the offending image is surfaced. Disposition:
    /// **corrected** against MS / CS / PA (quirk **AI6**).
    #[error("semantic shape mismatch on image_id={image_id}: gt={gt_shape:?}, dt={dt_shape:?}")]
    ShapeMismatch {
        /// Image id of the offending image.
        image_id: ImageId,
        /// Ground-truth label-map shape `(height, width)`.
        gt_shape: (u32, u32),
        /// Detection label-map shape `(height, width)`.
        dt_shape: (u32, u32),
    },

    /// A prediction pixel value is outside the valid `[0, n_classes)`
    /// class-id range. mmsegmentation silently truncates the
    /// over-class entries by reshaping the bincount output;
    /// cityscapesScripts validates eagerly and errors. vernier rejects
    /// with this typed variant at `SemanticPredictions::from_arrays`
    /// (quirk **AI4** — strict against CS, corrected by default
    /// against MS).
    #[error(
        "out-of-range prediction class_id={class_id} on image_id={image_id}: \
         valid range is [0, {n_classes})"
    )]
    OutOfRangePrediction {
        /// Image id where the offending pixel was found.
        image_id: ImageId,
        /// The over-range class id (the actual pixel value).
        class_id: ClassId,
        /// Configured class count for this evaluator. The error means
        /// `class_id >= n_classes`.
        n_classes: u32,
    },

    /// A prediction contains the configured `ignore_label`. Strict
    /// against cityscapesScripts (which errors out on this case);
    /// silently dropped under MS-strict. Disposition:
    /// **strict** against CS, **strict** against MS (quirk **AI3**).
    /// vernier surfaces the image id under
    /// `SemanticEvaluator(parity_mode="strict")` against `cityscapes()`
    /// for the strict-against-CS path.
    #[error(
        "prediction on image_id={image_id} contains ignore_label={ignore_label}; \
         cityscapesScripts strict-mode rejects this case (quirk AI3)"
    )]
    PredictionContainsIgnore {
        /// Image id where the ignore-label pixel was found.
        image_id: ImageId,
        /// The configured ignore label value.
        ignore_label: ClassId,
    },

    /// An image is in the GT set but missing from the prediction set.
    /// All three oracles error on this case, with different shapes
    /// (mmsegmentation per-image API: dataloader `KeyError`;
    /// cityscapesScripts: filesystem `FileNotFoundError`). vernier
    /// surfaces a typed variant with the missing image id.
    /// Disposition: **strict** against MS / CS / PA (quirk **AM1**).
    #[error("missing prediction for image_id={image_id}")]
    MissingPrediction {
        /// Image id present in the GT set with no matching prediction.
        image_id: ImageId,
    },

    /// Duplicate image id in either the GT or DT input. Both oracles
    /// rely on file-system uniqueness; vernier accepts a `Mapping`
    /// at the FFI boundary which can have duplicates if constructed
    /// programmatically. Disposition: **corrected** by construction.
    #[error("duplicate image_id={image_id} in {side}")]
    DuplicateImageId {
        /// The repeated image id.
        image_id: ImageId,
        /// `"gt"` or `"dt"` to disambiguate which side of the input
        /// pair the duplicate appeared on.
        side: &'static str,
    },

    /// The dataset contains no images. cityscapesScripts errors with
    /// a `ZeroDivisionError` on the per-class IoU computation; vernier
    /// surfaces a typed variant up-front so the user gets a clear
    /// error rather than a downstream NaN.
    #[error("semantic dataset is empty: no images to evaluate")]
    EmptyDataset,

    /// Configured `ignore_label` collides with a real evaluation
    /// class. mmsegmentation handles `ignore_label = 0` correctly on
    /// ADE20K (where class 0 is "other/unlabeled") by treating the
    /// label as a sentinel before the bincount; vernier permits the
    /// same via the `ade20k()` preset. This variant fires only when
    /// the value is invalid for a different reason — e.g., not
    /// representable in the input pixel width. Disposition:
    /// **strict** against MS for ADE20K (quirk **AJ5**).
    #[error("invalid ignore_label={ignore_label}: not representable in the input pixel width")]
    InvalidIgnoreLabel {
        /// The offending ignore label value.
        ignore_label: u64,
    },

    /// PNG decode failed on the `from_files` path. Wrapped from
    /// [`png::DecodingError`] with the offending path attached so the
    /// FFI layer can surface a structured Python exception.
    #[error("PNG decode failed for {path}: {source}")]
    PngDecode {
        /// Path of the file that failed to decode.
        path: String,
        /// Underlying `png` crate error.
        #[source]
        source: png::DecodingError,
    },

    /// I/O error reading a label-map file from disk on the `from_files`
    /// path. Wrapped from [`std::io::Error`] with the offending path
    /// attached.
    #[error("I/O error reading {path}: {source}")]
    Io {
        /// Path that failed to read.
        path: String,
        /// Underlying `std::io::Error`.
        #[source]
        source: std::io::Error,
    },
}
