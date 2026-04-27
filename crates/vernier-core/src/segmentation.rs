//! COCO `segmentation` field deserialization and normalization.
//!
//! pycocotools accepts three on-disk shapes for the COCO `segmentation`
//! field on both the GT and DT side:
//!
//! 1. `[[x0, y0, x1, y1, …], …]` — list of polygons. One inner list
//!    per polygon. Multi-polygon GT (e.g., a person split across an
//!    occluder) is merged into a single mask via union (quirk **K2**,
//!    `strict`).
//! 2. `{"size": [h, w], "counts": [u32, …]}` — uncompressed RLE,
//!    counts as a JSON array of integers.
//! 3. `{"size": [h, w], "counts": "…"}` — compressed RLE, counts as
//!    the COCO 6-bit char string. Quirk **K3** (`aligned`): we accept
//!    `str` only because JSON has no bytes type; the decoder treats
//!    every wire byte as ASCII regardless.
//!
//! The matching engine consumes a normalized [`vernier_mask::Rle`].
//! [`Segmentation::to_rle`] performs that normalization eagerly when
//! requested; the dataset stores the field verbatim (lazy). This
//! matches pycocotools' `annToRLE` — RLE materialization happens at
//! eval time, not at load time.
//!
//! ## Quirk dispositions
//!
//! - **K1** (`corrected`): degenerate polygon inputs (<3 vertices,
//!   odd coordinate counts, non-finite values) are rejected by
//!   [`vernier_mask::Rle::from_polygon`]; the error surfaces here as
//!   [`EvalError::Mask`]. pycocotools accepts these and silently
//!   produces malformed RLE.
//! - **K2** (`strict`): polygon-side normalization unions every
//!   sub-polygon into a single RLE, matching `mask.merge`.
//! - **K3** (`aligned`): bytes-vs-str distinction collapses on the
//!   JSON wire — counts are always a `String` here.
//! - **H2** (`corrected`): an RLE whose declared `size` disagrees
//!   with the requested `(h, w)` raises
//!   [`EvalError::DimensionMismatch`] instead of silently emitting an
//!   empty `0x0` RLE the way pycocotools does.

use serde::{Deserialize, Serialize};
use vernier_mask::Rle;

use crate::error::EvalError;

/// One COCO `segmentation` field, in any of the three shapes
/// pycocotools accepts. The dataset stores this verbatim;
/// [`Self::to_rle`] normalizes to a single [`Rle`] at eval time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Segmentation {
    /// Multi-polygon shape (`[[x0, y0, x1, y1, …], …]`). Each inner
    /// vector is a flat `(x, y)` pair sequence. Merged into a single
    /// RLE via union per **K2**.
    Polygons(Vec<Vec<f64>>),
    /// COCO RLE shape (`{"size": [h, w], "counts": …}`). Counts
    /// payload may be either the compressed 6-bit char string
    /// (typical) or an uncompressed JSON array of integers.
    Rle(SegmentationRle),
}

/// COCO RLE wrapper carrying the declared `(h, w)` shape alongside
/// either the compressed string or the uncompressed counts array.
///
/// `size` is the COCO `[h, w]` order — this is **not** `(w, h)`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SegmentationRle {
    /// `[h, w]` per the COCO spec.
    pub size: [u32; 2],
    /// Run lengths in either compressed or uncompressed form.
    pub counts: SegmentationRleCounts,
}

/// Counts payload of [`SegmentationRle`]. JSON shapes the variants
/// disambiguate untagged: a string parses as [`Self::Compressed`], a
/// number array as [`Self::Uncompressed`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SegmentationRleCounts {
    /// Compressed 6-bit char string per ADR-0002 / quirk **G1–G3**.
    Compressed(String),
    /// Raw run-length array. Used for GT round-tripping where the
    /// dataset author stored uncompressed counts (rare on COCO,
    /// common on synthetic fixtures).
    Uncompressed(Vec<u32>),
}

impl Segmentation {
    /// Normalizes this segmentation into a single [`Rle`] of shape
    /// `(h, w)`.
    ///
    /// - Polygons are rasterized via `Rle::from_polygon` and unioned
    ///   into one RLE (**K2**).
    /// - An RLE variant is checked against the requested `(h, w)`;
    ///   mismatch raises [`EvalError::DimensionMismatch`] (**H2**
    ///   `corrected`).
    pub fn to_rle(&self, h: u32, w: u32) -> Result<Rle, EvalError> {
        match self {
            Self::Polygons(polys) => Ok(Rle::from_polygons(polys, h, w)?),
            Self::Rle(rle) => {
                let [rh, rw] = rle.size;
                if rh != h || rw != w {
                    return Err(EvalError::DimensionMismatch {
                        detail: format!(
                            "segmentation declares size [{rh}, {rw}] but image is [{h}, {w}]"
                        ),
                    });
                }
                match &rle.counts {
                    SegmentationRleCounts::Compressed(s) => {
                        Ok(Rle::from_string_bytes(s.as_bytes(), h, w)?)
                    }
                    SegmentationRleCounts::Uncompressed(counts) => Ok(Rle {
                        h,
                        w,
                        counts: counts.clone(),
                    }),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(json: &str) -> Segmentation {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn parses_polygon_shape() {
        let s = parse("[[0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0]]");
        match s {
            Segmentation::Polygons(p) => {
                assert_eq!(p.len(), 1);
                assert_eq!(p[0].len(), 8);
            }
            other => panic!("expected Polygons, got {other:?}"),
        }
    }

    #[test]
    fn parses_compressed_rle_shape() {
        let s = parse(r#"{"size": [10, 10], "counts": "PPYo`0"}"#);
        match s {
            Segmentation::Rle(rle) => {
                assert_eq!(rle.size, [10, 10]);
                assert!(matches!(rle.counts, SegmentationRleCounts::Compressed(_)));
            }
            other => panic!("expected Rle, got {other:?}"),
        }
    }

    #[test]
    fn parses_uncompressed_rle_shape() {
        let s = parse(r#"{"size": [2, 2], "counts": [0, 4]}"#);
        match s {
            Segmentation::Rle(rle) => {
                assert_eq!(rle.size, [2, 2]);
                match rle.counts {
                    SegmentationRleCounts::Uncompressed(c) => assert_eq!(c, vec![0, 4]),
                    other => panic!("expected Uncompressed, got {other:?}"),
                }
            }
            other => panic!("expected Rle, got {other:?}"),
        }
    }

    #[test]
    fn polygon_to_rle_rasterizes_and_unions_k2() {
        // Two unit squares side by side as separate polygons. K2: the
        // result is the unioned mask, not two distinct objects.
        let json = r#"[
            [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0],
            [3.0, 0.0, 5.0, 0.0, 5.0, 2.0, 3.0, 2.0]
        ]"#;
        let s: Segmentation = serde_json::from_str(json).unwrap();
        let rle = s.to_rle(8, 8).unwrap();
        // Two 2×2 foreground regions → 4 + 4 = 8 foreground pixels.
        assert_eq!(rle.area(), 8);
    }

    #[test]
    fn compressed_rle_to_rle_round_trips() {
        // Produce a compressed counts string from a known RLE, then
        // round-trip it through Segmentation::to_rle.
        let original = Rle {
            h: 4,
            w: 4,
            counts: vec![0, 4, 4, 4, 4],
        };
        let counts = String::from_utf8(original.to_string_bytes()).unwrap();
        let json = format!(r#"{{"size": [4, 4], "counts": "{counts}"}}"#);
        let s: Segmentation = serde_json::from_str(&json).unwrap();
        let rle = s.to_rle(4, 4).unwrap();
        assert_eq!(rle, original);
    }

    #[test]
    fn uncompressed_rle_to_rle_uses_counts_verbatim() {
        let s = parse(r#"{"size": [2, 2], "counts": [0, 4]}"#);
        let rle = s.to_rle(2, 2).unwrap();
        assert_eq!(rle.h, 2);
        assert_eq!(rle.w, 2);
        assert_eq!(rle.counts, vec![0, 4]);
        assert_eq!(rle.area(), 4);
    }

    #[test]
    fn rle_size_mismatch_errors_h2_corrected() {
        let s = parse(r#"{"size": [10, 10], "counts": [0, 100]}"#);
        let err = s.to_rle(20, 20).unwrap_err();
        match err {
            EvalError::DimensionMismatch { detail } => {
                assert!(detail.contains("[10, 10]"));
                assert!(detail.contains("[20, 20]"));
            }
            other => panic!("expected DimensionMismatch, got {other:?}"),
        }
    }

    #[test]
    fn empty_polygon_list_yields_all_background_at_requested_shape() {
        // `[]` is degenerate but legal COCO; pycocotools silently
        // emits a 0×0 RLE here. K2/H2 disposition: produce a
        // well-formed all-bg RLE at the caller's (h, w).
        let s = parse("[]");
        let rle = s.to_rle(4, 4).unwrap();
        assert_eq!(rle.h, 4);
        assert_eq!(rle.w, 4);
        assert_eq!(rle.area(), 0);
    }

    #[test]
    fn polygon_with_too_few_vertices_propagates_k1_error() {
        // Two-point polygon: 4 floats. K1 corrected: reject.
        let s = parse("[[0.0, 0.0, 1.0, 1.0]]");
        let err = s.to_rle(8, 8).unwrap_err();
        assert!(matches!(err, EvalError::Mask(_)));
    }
}
