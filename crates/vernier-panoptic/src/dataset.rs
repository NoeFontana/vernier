//! Panoptic dataset types and validation.
//!
//! The panoptic data model is **per-image label maps + segments_info
//! JSON**, not COCO bboxes/RLEs (per ADR-0025). Each image carries a
//! flat `Vec<u32>` of size `height * width` whose pixel values are
//! segment ids (RGB-encoded as `id = R + 256·G + 256²·B`, decoded
//! upstream via `rgb2id`), and a `HashMap<u32, SegmentInfo>` mapping
//! each id to its category and crowd flag.
//!
//! This module owns the pure-Rust data shape and the structural
//! validations that don't require I/O. The PNG-decode and JSON-parse
//! glue lives in `vernier-ffi::panoptic`.

use std::collections::HashMap;

use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};

use crate::error::PanopticError;

/// COCO image id. `i64` matches the existing surface in `vernier-core`
/// (`vernier_core::dataset::ImageId`) so the cross-crate FFI surface
/// uses the same integer width.
pub type ImageId = i64;

/// COCO category id. `i64` for the same reason as [`ImageId`].
pub type CategoryId = i64;

/// Per-segment metadata bundled into an image entry.
///
/// `id` is the panoptic-encoded segment id (as decoded from the PNG by
/// `rgb2id`); it indexes into [`ImageEntry::label_map`]. `area` is the
/// number of foreground pixels — read from the JSON for GT (quirk
/// **S4**) and computed from the PNG for DT (quirk **S3**, load-bearing
/// for the IoU denominator). `iscrowd` is a bool here; the FFI accepts
/// `int | bool` per quirk **S5** and normalizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentInfo {
    /// Segment id (panoptic-encoded; same value the PNG carries).
    pub id: u32,
    /// COCO category id this segment belongs to.
    pub category_id: CategoryId,
    /// Whether this segment is a crowd region (GT side; ignored on DT
    /// per quirk **S6**).
    pub iscrowd: bool,
    /// Foreground pixel count. GT-side: read from JSON. DT-side:
    /// computed from PNG marginals.
    pub area: u64,
}

/// Per-category metadata. Dataset-level (GT-only per quirk **S9**;
/// `pred_json['categories']` is ignored).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CategoryMeta {
    /// COCO category id.
    pub id: CategoryId,
    /// Whether this category is a thing (instance) or stuff
    /// (amorphous region). Consulted only at aggregation time, not
    /// during matching, per quirk **S10**.
    pub isthing: bool,
}

/// One image's worth of panoptic ground truth or prediction.
///
/// `label_map` is the flat `height * width` `Vec<u32>` of segment
/// ids; `segments` maps each id encountered (or expected) in the
/// label map to its [`SegmentInfo`]. The two are constructed together
/// and validated by [`ImageEntry::from_components`].
#[derive(Debug, Clone)]
pub struct ImageEntry {
    /// Image height in pixels.
    pub height: u32,
    /// Image width in pixels.
    pub width: u32,
    /// Flat pixel buffer of length `height * width`. Pixel values are
    /// panoptic-encoded segment ids; pixel `0` is `VOID` (quirk
    /// **R3**).
    pub label_map: Vec<u32>,
    /// Maps each segment id present (or declared) in this image to
    /// its metadata. Uses FxHash because the keys are internal `u32`
    /// segment ids; SipHash's DoS resistance is wasted work on the
    /// per-pixel validation pass that walks `O(H*W)` pixels.
    pub segments: FxHashMap<u32, SegmentInfo>,
}

impl ImageEntry {
    /// Build an [`ImageEntry`] from already-decoded components,
    /// running structural validation. Used by the FFI's
    /// `from_arrays` constructor.
    ///
    /// Validates:
    /// - `label_map.len() == height * width` (otherwise
    ///   [`PanopticError::InvalidInput`])
    /// - On the prediction side, every non-VOID id present in the PNG
    ///   has an entry in `segments`
    ///   ([`PanopticError::UnknownPredSegmentId`], quirk **S1**).
    /// - Every id in `segments` is present in the PNG
    ///   ([`PanopticError::MissingPredSegmentInPng`], quirk **S11**).
    /// - `segments` carries no duplicate ids
    ///   ([`PanopticError::DuplicateSegmentId`], quirk **S7**).
    ///
    /// `side` is `"gt"` or `"dt"`. The S1/S11 checks are pred-only
    /// (S8 documents that GT JSON-extras are silently kept; matching
    /// iterates the histogram, not the dict).
    pub fn from_components(
        image_id: ImageId,
        height: u32,
        width: u32,
        label_map: Vec<u32>,
        segments_list: Vec<SegmentInfo>,
        side: &'static str,
    ) -> Result<Self, PanopticError> {
        let expected = (height as usize)
            .checked_mul(width as usize)
            .ok_or_else(|| PanopticError::InvalidInput {
                detail: format!(
                    "image_id={image_id}: height={height} * width={width} overflows usize"
                ),
            })?;
        if label_map.len() != expected {
            return Err(PanopticError::InvalidInput {
                detail: format!(
                    "image_id={image_id}: label_map.len()={} but height*width={expected}",
                    label_map.len()
                ),
            });
        }

        let mut segments: FxHashMap<u32, SegmentInfo> =
            FxHashMap::with_capacity_and_hasher(segments_list.len(), Default::default());
        for seg in segments_list {
            if segments.insert(seg.id, seg).is_some() {
                return Err(PanopticError::DuplicateSegmentId {
                    image_id,
                    segment_id: seg.id,
                    side,
                });
            }
        }

        if side == "dt" {
            let mut declared: FxHashSet<u32> = segments.keys().copied().collect();
            for &px in &label_map {
                if px == crate::parity::PANOPTIC_VOID {
                    continue;
                }
                if !segments.contains_key(&px) {
                    return Err(PanopticError::UnknownPredSegmentId {
                        image_id,
                        segment_id: px,
                    });
                }
                declared.remove(&px);
            }
            if let Some(&missing) = declared.iter().next() {
                return Err(PanopticError::MissingPredSegmentInPng {
                    image_id,
                    segment_id: missing,
                });
            }
        }

        Ok(Self {
            height,
            width,
            label_map,
            segments,
        })
    }

    /// Recompute pixel-area marginals from the PNG, populating
    /// `segments[id].area`. Mirrors panopticapi's prediction-side
    /// recompute (quirk **S3**, `evaluation.py:102`): pred areas are
    /// always overwritten from the PNG, ignoring any JSON `area`
    /// field. GT areas are *not* overwritten by this method per
    /// quirk **S4** (asymmetric with S3); `Corrected` mode can
    /// invoke this on the GT side too and emit a `GtAreaMismatch`
    /// warning if the JSON `area` disagrees — that's a follow-up
    /// hook, not a v1 must-have.
    pub fn recompute_areas_from_png(&mut self) {
        for seg in self.segments.values_mut() {
            seg.area = 0;
        }
        for &px in &self.label_map {
            if px == crate::parity::PANOPTIC_VOID {
                continue;
            }
            if let Some(seg) = self.segments.get_mut(&px) {
                seg.area += 1;
            }
        }
    }
}

/// Panoptic ground truth — a collection of [`ImageEntry`] keyed by
/// image id, plus the dataset-level [`CategoryMeta`] table.
#[derive(Debug, Clone)]
pub struct PanopticDataset {
    /// Per-image GT entries.
    pub images: HashMap<ImageId, ImageEntry>,
    /// Category taxonomy. Quirk **S9**: GT-only; predictions never
    /// carry a taxonomy.
    pub categories: HashMap<CategoryId, CategoryMeta>,
}

impl PanopticDataset {
    /// Build a dataset from already-decoded image entries and
    /// categories. Used by the FFI's `from_arrays` constructor.
    pub fn from_components(
        images: HashMap<ImageId, ImageEntry>,
        categories: HashMap<CategoryId, CategoryMeta>,
    ) -> Self {
        Self { images, categories }
    }

    /// Iterate `(image_id, isthing_categories)` pairs for the
    /// things/stuff bucket aggregation in the summarize layer.
    pub fn category_isthing(&self, cat: CategoryId) -> Option<bool> {
        self.categories.get(&cat).map(|m| m.isthing)
    }
}

/// Panoptic predictions — sibling shape to [`PanopticDataset`] but
/// without categories (quirk **S9**). Constructors do **not** accept
/// a `categories` argument.
#[derive(Debug, Clone)]
pub struct PanopticPredictions {
    /// Per-image prediction entries.
    pub images: HashMap<ImageId, ImageEntry>,
}

impl PanopticPredictions {
    /// Build predictions from already-decoded image entries.
    pub fn from_components(images: HashMap<ImageId, ImageEntry>) -> Self {
        Self { images }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segs(items: &[(u32, i64, bool)]) -> Vec<SegmentInfo> {
        items
            .iter()
            .map(|&(id, cat, iscrowd)| SegmentInfo {
                id,
                category_id: cat,
                iscrowd,
                area: 0,
            })
            .collect()
    }

    #[test]
    fn from_components_validates_label_map_length() {
        let err = ImageEntry::from_components(1, 2, 3, vec![1, 1], segs(&[(1, 100, false)]), "gt")
            .unwrap_err();
        assert!(matches!(err, PanopticError::InvalidInput { .. }));
    }

    #[test]
    fn from_components_rejects_duplicate_ids() {
        let err = ImageEntry::from_components(
            7,
            2,
            2,
            vec![1, 1, 1, 1],
            segs(&[(1, 100, false), (1, 200, false)]),
            "gt",
        )
        .unwrap_err();
        match err {
            PanopticError::DuplicateSegmentId {
                image_id,
                segment_id,
                side,
            } => {
                assert_eq!(image_id, 7);
                assert_eq!(segment_id, 1);
                assert_eq!(side, "gt");
            }
            other => panic!("expected DuplicateSegmentId, got {other:?}"),
        }
    }

    #[test]
    fn dt_side_rejects_unknown_segment_in_png() {
        let err = ImageEntry::from_components(
            42,
            2,
            2,
            vec![1, 99, 1, 1],
            segs(&[(1, 100, false)]),
            "dt",
        )
        .unwrap_err();
        match err {
            PanopticError::UnknownPredSegmentId {
                image_id,
                segment_id,
            } => {
                assert_eq!(image_id, 42);
                assert_eq!(segment_id, 99);
            }
            other => panic!("expected UnknownPredSegmentId, got {other:?}"),
        }
    }

    #[test]
    fn dt_side_rejects_missing_segment_in_png() {
        let err = ImageEntry::from_components(
            42,
            2,
            2,
            vec![1, 1, 1, 1],
            segs(&[(1, 100, false), (2, 200, false)]),
            "dt",
        )
        .unwrap_err();
        match err {
            PanopticError::MissingPredSegmentInPng {
                image_id,
                segment_id,
            } => {
                assert_eq!(image_id, 42);
                assert_eq!(segment_id, 2);
            }
            other => panic!("expected MissingPredSegmentInPng, got {other:?}"),
        }
    }

    #[test]
    fn gt_side_does_not_validate_png_membership() {
        // Quirk S8: GT JSON-extras are silently kept; matching
        // iterates the histogram, not the dict. So a GT segment
        // declared in `segments_info` but absent from the PNG is
        // *not* an error on the GT side — it just contributes
        // nothing to TP/FN counts.
        let entry = ImageEntry::from_components(
            42,
            2,
            2,
            vec![1, 1, 1, 1],
            segs(&[(1, 100, false), (99, 200, false)]),
            "gt",
        )
        .unwrap();
        assert_eq!(entry.segments.len(), 2);
    }

    #[test]
    fn recompute_areas_from_png_counts_marginals() {
        let mut entry = ImageEntry::from_components(
            1,
            2,
            3,
            vec![1, 1, 2, 2, 0, 1],
            segs(&[(1, 100, false), (2, 200, false)]),
            "gt",
        )
        .unwrap();
        entry.recompute_areas_from_png();
        assert_eq!(entry.segments[&1].area, 3);
        assert_eq!(entry.segments[&2].area, 2);
    }
}
