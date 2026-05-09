//! Native PNG decode + confusion-matrix fold in one walk per image.
//!
//! Replaces the Pillow → `np.asarray` → `astype(np.uint32)` →
//! cross-thread `submit` pipeline with a direct libpng → kernel walk.
//! For the val2017-scale parity workload (5000 images, ~270k px each)
//! this saves the per-pixel Python-level cast and the GIL-held submit
//! copy — same pattern panopticapi shipped via
//! [`vernier_panoptic::decode`] (commit 7e5ba96, 2.75× over the
//! array-input path).
//!
//! Single-threaded by design (per the parent-PR no-rayon scope). The
//! per-image loop runs inside `py.detach`, so the GIL is released for
//! the whole batch.
//!
//! Format contract: 8-bit grayscale PNGs only. RGB / paletted / 16-bit
//! grayscale are rejected with [`SemanticError::UnsupportedPngFormat`].
//! Callers with wider class-id ranges should fall back to the
//! array-input path with `np.uint16` / `np.uint32` ndarrays.

use std::collections::HashMap;
use std::fs;
use std::io::Cursor;
use std::path::PathBuf;

use crate::error::{ImageId, SemanticError};
use crate::kernel::{accumulate_confusion, ConfusionMatrix};
use crate::parity::ParityMode;
use crate::summarize::{summarize, SemanticSummary};

/// Decode an 8-bit grayscale PNG byte buffer into a row-major `(H, W)`
/// `u8` label map. Errors carry `image_id` so the FFI layer can
/// surface the offending file's id even though this primitive doesn't
/// know its filesystem path.
fn decode_grayscale8(
    image_id: ImageId,
    bytes: &[u8],
) -> Result<(Vec<u8>, (u32, u32)), SemanticError> {
    let decoder = png::Decoder::new(Cursor::new(bytes));
    let mut reader = decoder.read_info().map_err(|e| SemanticError::PngDecode {
        path: format!("image_id={image_id}"),
        source: e,
    })?;
    let info = reader.info();
    if info.color_type != png::ColorType::Grayscale || info.bit_depth != png::BitDepth::Eight {
        return Err(SemanticError::UnsupportedPngFormat {
            image_id,
            mode: format!("{:?}/{:?}", info.color_type, info.bit_depth),
        });
    }
    let height = info.height;
    let width = info.width;
    let n_pixels = (height as usize) * (width as usize);
    let mut buf = vec![0u8; n_pixels];
    let frame = reader
        .next_frame(&mut buf)
        .map_err(|e| SemanticError::PngDecode {
            path: format!("image_id={image_id}"),
            source: e,
        })?;
    debug_assert_eq!(frame.buffer_size(), n_pixels);
    Ok((buf, (height, width)))
}

/// Fold one `(gt_bytes, dt_bytes)` PNG pair into a running confusion
/// matrix. The kernel walks both buffers at native `u8` width per
/// ADR-0037; shape mismatch surfaces as
/// [`SemanticError::ShapeMismatch`] with `image_id` attribution.
///
/// This is the streaming primitive shared by [`evaluate_from_pngs`]
/// (batch path) and the upcoming `submit_png` FFI entry point.
pub fn fold_pair_bytes(
    image_id: ImageId,
    gt_bytes: &[u8],
    dt_bytes: &[u8],
    ignore_label: Option<u32>,
    confusion: &mut ConfusionMatrix,
) -> Result<(), SemanticError> {
    let (gt_buf, gt_dims) = decode_grayscale8(image_id, gt_bytes)?;
    let (dt_buf, dt_dims) = decode_grayscale8(image_id, dt_bytes)?;
    if gt_dims != dt_dims {
        return Err(SemanticError::ShapeMismatch {
            image_id,
            gt_shape: gt_dims,
            dt_shape: dt_dims,
        });
    }
    accumulate_confusion(&gt_buf, &dt_buf, ignore_label, confusion);
    Ok(())
}

/// Run the semantic-segmentation evaluation directly against PNG files.
///
/// Iteration order is sorted by `image_id` (mirrors
/// [`crate::stream::StreamingSemanticEvaluator::update`]'s contract,
/// quirk **AM5**). Each image is decoded + folded in turn; only one
/// pair of decoded buffers is in flight at a time.
///
/// `gt_paths` is a sorted `(image_id, path)` slice; `dt_paths` is a
/// dict from image_id to path. A missing prediction surfaces as
/// [`SemanticError::MissingPrediction`].
pub fn evaluate_from_pngs(
    gt_paths: &[(ImageId, PathBuf)],
    dt_paths: &HashMap<ImageId, PathBuf>,
    n_classes: u32,
    ignore_label: Option<u32>,
    mode: ParityMode,
) -> Result<SemanticSummary, SemanticError> {
    if gt_paths.is_empty() {
        return Err(SemanticError::EmptyDataset);
    }
    let mut confusion = ConfusionMatrix::zeros(n_classes);
    for (image_id, gt_path) in gt_paths {
        let dt_path = dt_paths
            .get(image_id)
            .ok_or(SemanticError::MissingPrediction {
                image_id: *image_id,
            })?;
        let gt_bytes = fs::read(gt_path).map_err(|e| SemanticError::Io {
            path: gt_path.display().to_string(),
            source: e,
        })?;
        let dt_bytes = fs::read(dt_path).map_err(|e| SemanticError::Io {
            path: dt_path.display().to_string(),
            source: e,
        })?;
        fold_pair_bytes(
            *image_id,
            &gt_bytes,
            &dt_bytes,
            ignore_label,
            &mut confusion,
        )?;
    }
    Ok(summarize(confusion, mode))
}

#[cfg(test)]
mod tests {
    use png::Encoder;

    use super::*;

    fn encode_grayscale8(data: &[u8], height: u32, width: u32) -> Vec<u8> {
        let mut buf = Vec::new();
        {
            let mut encoder = Encoder::new(&mut buf, width, height);
            encoder.set_color(png::ColorType::Grayscale);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().expect("write header");
            writer.write_image_data(data).expect("write image data");
            writer.finish().expect("finish encoder");
        }
        buf
    }

    fn encode_rgb8(data: &[u8], height: u32, width: u32) -> Vec<u8> {
        let mut buf = Vec::new();
        {
            let mut encoder = Encoder::new(&mut buf, width, height);
            encoder.set_color(png::ColorType::Rgb);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().expect("write header");
            writer.write_image_data(data).expect("write image data");
            writer.finish().expect("finish encoder");
        }
        buf
    }

    #[test]
    fn fused_path_matches_array_path_perfect_dt() {
        // Three images, 4×4 each, 3 classes + ignore=255. The fused
        // PNG path must produce a confusion matrix bit-equal to the
        // existing accumulate_confusion(&[u8], ...) call on the same
        // pixel data.
        let pixels: [Vec<u8>; 3] = [
            vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
            vec![1, 1, 1, 1, 0, 0, 2, 2, 0, 1, 2, 0, 255, 255, 0, 1],
            vec![2, 0, 1, 2, 0, 1, 2, 0, 255, 0, 0, 0, 1, 1, 1, 1],
        ];

        let mut fused_cm = ConfusionMatrix::zeros(3);
        for (i, pix) in pixels.iter().enumerate() {
            let bytes = encode_grayscale8(pix, 4, 4);
            fold_pair_bytes(i as ImageId, &bytes, &bytes, Some(255), &mut fused_cm)
                .expect("fused fold");
        }

        let mut reference_cm = ConfusionMatrix::zeros(3);
        for pix in &pixels {
            accumulate_confusion(pix.as_slice(), pix.as_slice(), Some(255), &mut reference_cm);
        }

        assert_eq!(
            fused_cm, reference_cm,
            "fused PNG path must produce a bit-equal confusion matrix"
        );
    }

    #[test]
    fn fused_path_rejects_rgb_png() {
        let bytes = encode_rgb8(&[0u8; 4 * 4 * 3], 4, 4);
        let mut cm = ConfusionMatrix::zeros(3);
        let err = fold_pair_bytes(7, &bytes, &bytes, None, &mut cm).unwrap_err();
        match err {
            SemanticError::UnsupportedPngFormat { image_id, mode } => {
                assert_eq!(image_id, 7);
                assert!(
                    mode.contains("Rgb"),
                    "mode must surface the color type, got {mode}"
                );
            }
            other => panic!("expected UnsupportedPngFormat, got {other:?}"),
        }
    }

    #[test]
    fn fused_path_rejects_shape_mismatch() {
        let gt = encode_grayscale8(&[0u8; 4 * 4], 4, 4);
        let dt = encode_grayscale8(&[0u8; 2 * 2], 2, 2);
        let mut cm = ConfusionMatrix::zeros(3);
        let err = fold_pair_bytes(11, &gt, &dt, None, &mut cm).unwrap_err();
        assert!(matches!(
            err,
            SemanticError::ShapeMismatch { image_id: 11, .. }
        ));
    }

    #[test]
    fn fused_path_rejects_empty_dataset() {
        let dt: HashMap<ImageId, PathBuf> = HashMap::new();
        let err = evaluate_from_pngs(&[], &dt, 3, None, ParityMode::Corrected).unwrap_err();
        assert!(matches!(err, SemanticError::EmptyDataset));
    }
}
