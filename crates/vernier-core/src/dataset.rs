//! Dataset abstraction and the COCO ground-truth implementation.
//!
//! Per ADR-0005, the matching engine and accumulator are written once
//! and never edited; they are generic over a dataset trait, never over
//! a concrete dataset type. Phase 2 (LVIS) and a future phase 4+
//! (custom datasets) will add new `EvalDataset` impls without touching
//! anything in `matching.rs` or `accumulate.rs`.
//!
//! The trait is shaped around two access patterns the matching loop
//! drives:
//!
//! - "Give me the GTs for image `i`." — driven by the per-image
//!   evaluation outer loop.
//! - "Give me the GTs for category `k` across all images." — driven
//!   by the per-category accumulation that happens after matching.
//!
//! Both go through index slices (`&[usize]`) into a single flat
//! storage. The convenience method `ann_iter_for_image` builds an
//! iterator on top of the slice; callers that want raw indices (e.g.,
//! to interleave bbox / segm / keypoint lookups) use the slice form.
//!
//! ## Quirk dispositions
//!
//! The COCO loader honors the dataset-level dispositions ratified in
//! ADR-0002:
//!
//! - **D1** (`corrected`): we store both the JSON `iscrowd` flag and
//!   the optional `ignore` flag verbatim. The eval-time
//!   [`CocoAnnotation::effective_ignore`] computes the flag per
//!   parity mode, instead of overwriting one with the other at load
//!   time the way pycocotools does.
//! - **D3** (`aligned`): annotations are not mutated mid-evaluation;
//!   the per-call `_ignore` (which combines the dataset flag with the
//!   current area range) is computed at eval time.
//! - **J3** (`strict`): detection-side area derivation lives in the
//!   future `loadRes`-equivalent path, not here.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::error::EvalError;
use crate::parity::ParityMode;

/// Newtype for image ids. Sourced from the JSON `id` field; preserved
/// verbatim. Crowd_region's image with `id = 1` becomes
/// `ImageId(1)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ImageId(pub i64);

/// Newtype for category ids.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CategoryId(pub i64);

/// Newtype for annotation ids.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AnnId(pub i64);

/// Per-image metadata. We keep only what the eval algorithm reads;
/// fields like `coco_url`, `flickr_url`, `date_captured` are dropped on
/// load (round-trip is via the typed COCO data, not raw JSON).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ImageMeta {
    /// Image id.
    pub id: ImageId,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// File name as recorded in the dataset JSON; useful for tracing
    /// fixtures back to source images.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_name: Option<String>,
}

/// Per-category metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CategoryMeta {
    /// Category id.
    pub id: CategoryId,
    /// Human-readable category name (e.g., `"person"`).
    pub name: String,
    /// Optional supercategory grouping (e.g., `"animal"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supercategory: Option<String>,
}

/// Axis-aligned bounding box in COCO format `(x, y, w, h)`, where
/// `(x, y)` is the top-left corner in pixels (typically with sub-pixel
/// floats) and `(w, h)` are the width and height.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(from = "[f64; 4]", into = "[f64; 4]")]
pub struct Bbox {
    /// Top-left x (pixels).
    pub x: f64,
    /// Top-left y (pixels).
    pub y: f64,
    /// Width (pixels).
    pub w: f64,
    /// Height (pixels).
    pub h: f64,
}

impl From<[f64; 4]> for Bbox {
    fn from([x, y, w, h]: [f64; 4]) -> Self {
        Self { x, y, w, h }
    }
}

impl From<Bbox> for [f64; 4] {
    fn from(b: Bbox) -> Self {
        [b.x, b.y, b.w, b.h]
    }
}

/// A COCO annotation as stored on the dataset side (ground truth).
///
/// Detection annotations follow a separate path — see the future
/// `loadRes`-equivalent — because their `iscrowd` is always 0 (quirk
/// **E2**) and their `area` is auto-derived (quirk **J3**). Conflating
/// the two would let a DT bug silently corrupt GT semantics.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CocoAnnotation {
    /// Annotation id (preserved verbatim from JSON).
    pub id: AnnId,
    /// Image this annotation belongs to.
    pub image_id: ImageId,
    /// Category this annotation belongs to.
    pub category_id: CategoryId,
    /// Pixel area as recorded in JSON. For GT, COCO stores this
    /// directly; we trust the field.
    pub area: f64,
    /// Crowd flag (the COCO `iscrowd` field). pycocotools coerces this
    /// to bool via truthiness, so 0/1 ints round-trip identically.
    #[serde(rename = "iscrowd", default, deserialize_with = "deserialize_bool_int")]
    pub is_crowd: bool,
    /// Optional explicit `ignore` flag.
    ///
    /// `None` means the JSON had no `ignore` field. pycocotools (quirk
    /// **D1**) silently overwrites whatever was here with `is_crowd`;
    /// vernier preserves it and lets [`Self::effective_ignore`] resolve
    /// the strict vs corrected disposition at eval time.
    #[serde(
        rename = "ignore",
        default,
        deserialize_with = "deserialize_opt_bool_int"
    )]
    pub ignore_flag: Option<bool>,
    /// Bounding box. Required for every COCO ground-truth annotation
    /// (even keypoint-only annotations carry a bbox; the bbox is what
    /// `J3` derives DT-area from). Phase 2 adds `segmentation`,
    /// Phase 3 adds `keypoints` as additional optional fields.
    pub bbox: Bbox,
}

impl CocoAnnotation {
    /// Resolves the effective ignore flag for this annotation under a
    /// given parity mode (per ADR-0002 / quirk **D1**).
    ///
    /// - `Strict` reproduces pycocotools: the user's `ignore` field is
    ///   discarded, and `ignore` is set to `is_crowd`.
    /// - `Corrected` honors the user's explicit `ignore` field when
    ///   present; falls back to `is_crowd` when absent.
    pub fn effective_ignore(&self, mode: ParityMode) -> bool {
        match mode {
            ParityMode::Strict => self.is_crowd,
            ParityMode::Corrected => self.ignore_flag.unwrap_or(self.is_crowd),
        }
    }
}

/// Common interface every annotation type on every dataset implements.
///
/// The matching engine (per ADR-0005) reads only this trait — it does
/// not see [`CocoAnnotation`] or any future `LvisAnnotation` directly.
pub trait Annotation {
    /// Image this annotation belongs to.
    fn image_id(&self) -> ImageId;
    /// Category this annotation belongs to.
    fn category_id(&self) -> CategoryId;
    /// Pixel area.
    fn area(&self) -> f64;
    /// Crowd flag (raw, before parity resolution).
    fn is_crowd(&self) -> bool;
    /// Effective ignore flag under the given parity mode.
    fn effective_ignore(&self, mode: ParityMode) -> bool;
}

impl Annotation for CocoAnnotation {
    fn image_id(&self) -> ImageId {
        self.image_id
    }
    fn category_id(&self) -> CategoryId {
        self.category_id
    }
    fn area(&self) -> f64 {
        self.area
    }
    fn is_crowd(&self) -> bool {
        self.is_crowd
    }
    fn effective_ignore(&self, mode: ParityMode) -> bool {
        Self::effective_ignore(self, mode)
    }
}

/// Trait every dataset (COCO, LVIS, CrowdPose, custom) implements.
///
/// `Send + Sync` is required by the future `BackgroundEvaluator`
/// (separate ADR) so the dataset can be shared across worker threads
/// without copying.
pub trait EvalDataset: Send + Sync {
    /// Concrete annotation type. For [`CocoDataset`] this is
    /// [`CocoAnnotation`]; future datasets may use their own type with
    /// extra metadata (LVIS, for example, will need
    /// `not_exhaustive_categories`).
    type Annotation: Annotation;

    /// All images in the dataset, in input order.
    fn images(&self) -> &[ImageMeta];

    /// All categories in the dataset, in input order.
    fn categories(&self) -> &[CategoryMeta];

    /// Flat slice of every annotation in the dataset, in input order.
    fn annotations(&self) -> &[Self::Annotation];

    /// Indices into [`Self::annotations`] for a given image.
    /// Returns an empty slice when the image is unknown.
    fn ann_indices_for_image(&self, image_id: ImageId) -> &[usize];

    /// Indices into [`Self::annotations`] for a given category.
    /// Returns an empty slice when the category is unknown.
    fn ann_indices_for_category(&self, cat_id: CategoryId) -> &[usize];

    /// Convenience iterator over annotations for a given image.
    fn ann_iter_for_image(&self, image_id: ImageId) -> AnnotationIter<'_, Self::Annotation> {
        AnnotationIter {
            anns: self.annotations(),
            indices: self.ann_indices_for_image(image_id).iter(),
        }
    }

    /// Convenience iterator over annotations for a given category.
    fn ann_iter_for_category(&self, cat_id: CategoryId) -> AnnotationIter<'_, Self::Annotation> {
        AnnotationIter {
            anns: self.annotations(),
            indices: self.ann_indices_for_category(cat_id).iter(),
        }
    }
}

/// Iterator that walks a slice of annotation indices and yields
/// references into the flat annotation storage. Returned by the
/// `*_iter_for_*` methods on [`EvalDataset`].
pub struct AnnotationIter<'a, A> {
    anns: &'a [A],
    indices: std::slice::Iter<'a, usize>,
}

impl<'a, A> Iterator for AnnotationIter<'a, A> {
    type Item = &'a A;

    fn next(&mut self) -> Option<Self::Item> {
        let idx = *self.indices.next()?;
        self.anns.get(idx)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.indices.size_hint()
    }
}

impl<'a, A> ExactSizeIterator for AnnotationIter<'a, A> {}

/// On-disk shape of a COCO ground-truth JSON file.
///
/// Only the fields vernier reads are typed; unknown top-level fields
/// (`info`, `licenses`, …) are dropped on load. Round-tripping in tests
/// uses the same struct; user JSON that round-trips through vernier
/// will lose those fields. We document this loudly because pycocotools
/// 2.0.11 added a single line preserving the `info` field on `loadRes`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CocoJson {
    /// All images.
    pub images: Vec<ImageMeta>,
    /// All annotations.
    pub annotations: Vec<CocoAnnotation>,
    /// All categories.
    pub categories: Vec<CategoryMeta>,
}

/// COCO ground-truth dataset.
///
/// Storage is a single `Arc<Vec<CocoAnnotation>>` plus per-image and
/// per-category index vectors. The `Arc` makes the dataset cheaply
/// shareable across worker threads (the `BackgroundEvaluator` from a
/// future ADR depends on this); the index vectors are owned by the
/// `CocoDataset` because they're cheap to rebuild and rebuild needs
/// to happen exactly when the annotation set changes.
#[derive(Debug, Clone)]
pub struct CocoDataset {
    images: Arc<Vec<ImageMeta>>,
    categories: Arc<Vec<CategoryMeta>>,
    annotations: Arc<Vec<CocoAnnotation>>,
    by_image: HashMap<ImageId, Vec<usize>>,
    by_category: HashMap<CategoryId, Vec<usize>>,
}

impl CocoDataset {
    /// Loads a dataset from a JSON byte slice.
    ///
    /// Validates that every annotation references a known image and a
    /// known category; missing references raise [`EvalError::InvalidAnnotation`]
    /// rather than producing a silently-empty dataset.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, EvalError> {
        let raw: CocoJson = serde_json::from_slice(bytes)?;
        Self::from_parts(raw.images, raw.annotations, raw.categories)
    }

    /// Loads a dataset from already-typed parts.
    pub fn from_parts(
        images: Vec<ImageMeta>,
        annotations: Vec<CocoAnnotation>,
        categories: Vec<CategoryMeta>,
    ) -> Result<Self, EvalError> {
        let known_images: HashMap<ImageId, ()> = images.iter().map(|i| (i.id, ())).collect();
        let known_categories: HashMap<CategoryId, ()> =
            categories.iter().map(|c| (c.id, ())).collect();

        let mut by_image: HashMap<ImageId, Vec<usize>> = HashMap::new();
        let mut by_category: HashMap<CategoryId, Vec<usize>> = HashMap::new();

        for (idx, ann) in annotations.iter().enumerate() {
            if !known_images.contains_key(&ann.image_id) {
                return Err(EvalError::InvalidAnnotation {
                    detail: format!(
                        "annotation id={} references unknown image_id={}",
                        ann.id.0, ann.image_id.0
                    ),
                });
            }
            if !known_categories.contains_key(&ann.category_id) {
                return Err(EvalError::InvalidAnnotation {
                    detail: format!(
                        "annotation id={} references unknown category_id={}",
                        ann.id.0, ann.category_id.0
                    ),
                });
            }
            by_image.entry(ann.image_id).or_default().push(idx);
            by_category.entry(ann.category_id).or_default().push(idx);
        }

        // Pre-seed empty entries for images and categories that exist
        // but have no annotations. The matching loop relies on
        // `ann_indices_for_image` returning an empty slice (not a
        // panic) for these cases.
        for img in &images {
            by_image.entry(img.id).or_default();
        }
        for cat in &categories {
            by_category.entry(cat.id).or_default();
        }

        Ok(Self {
            images: Arc::new(images),
            categories: Arc::new(categories),
            annotations: Arc::new(annotations),
            by_image,
            by_category,
        })
    }

    /// Round-trips the dataset to the on-disk JSON shape, preserving
    /// every field vernier carries. Useful for fixture authoring and
    /// for debugging serde mismatches.
    pub fn to_json_value(&self) -> CocoJson {
        CocoJson {
            images: (*self.images).clone(),
            annotations: (*self.annotations).clone(),
            categories: (*self.categories).clone(),
        }
    }
}

impl EvalDataset for CocoDataset {
    type Annotation = CocoAnnotation;

    fn images(&self) -> &[ImageMeta] {
        &self.images
    }

    fn categories(&self) -> &[CategoryMeta] {
        &self.categories
    }

    fn annotations(&self) -> &[CocoAnnotation] {
        &self.annotations
    }

    fn ann_indices_for_image(&self, image_id: ImageId) -> &[usize] {
        self.by_image.get(&image_id).map_or(&[][..], Vec::as_slice)
    }

    fn ann_indices_for_category(&self, cat_id: CategoryId) -> &[usize] {
        self.by_category.get(&cat_id).map_or(&[][..], Vec::as_slice)
    }
}

// ---------------------------------------------------------------------------
// serde glue
// ---------------------------------------------------------------------------

/// Deserialize a JSON int (`0` / `1`) or bool into a `bool`. COCO JSON
/// uses `0`/`1` ints for `iscrowd`; a permissive reader accepts bool.
fn deserialize_bool_int<'de, D>(de: D) -> Result<bool, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error as DeError;

    #[derive(Deserialize)]
    #[serde(untagged)]
    enum BoolOrInt {
        Bool(bool),
        Int(i64),
    }

    match BoolOrInt::deserialize(de)? {
        BoolOrInt::Bool(b) => Ok(b),
        BoolOrInt::Int(0) => Ok(false),
        BoolOrInt::Int(1) => Ok(true),
        BoolOrInt::Int(other) => Err(DeError::custom(format!(
            "expected 0 or 1 for COCO bool field, got {other}"
        ))),
    }
}

fn deserialize_opt_bool_int<'de, D>(de: D) -> Result<Option<bool>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error as DeError;

    #[derive(Deserialize)]
    #[serde(untagged)]
    enum OptBoolOrInt {
        Bool(bool),
        Int(i64),
        Null,
    }

    match Option::<OptBoolOrInt>::deserialize(de)? {
        None | Some(OptBoolOrInt::Null) => Ok(None),
        Some(OptBoolOrInt::Bool(b)) => Ok(Some(b)),
        Some(OptBoolOrInt::Int(0)) => Ok(Some(false)),
        Some(OptBoolOrInt::Int(1)) => Ok(Some(true)),
        Some(OptBoolOrInt::Int(other)) => Err(DeError::custom(format!(
            "expected 0 or 1 for optional COCO bool field, got {other}"
        ))),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    const CROWD_REGION_GT: &str = r#"{
        "images": [
            {"id": 1, "width": 200, "height": 200, "file_name": "img1.png"}
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1,
             "bbox": [100, 100, 50, 50], "area": 2500, "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 1,
             "bbox": [0, 0, 200, 200], "area": 40000, "iscrowd": 1}
        ],
        "categories": [
            {"id": 1, "name": "widget", "supercategory": "thing"}
        ]
    }"#;

    fn load_crowd_region() -> CocoDataset {
        CocoDataset::from_json_bytes(CROWD_REGION_GT.as_bytes()).unwrap()
    }

    #[test]
    fn loads_crowd_region_fixture() {
        let ds = load_crowd_region();
        assert_eq!(ds.images().len(), 1);
        assert_eq!(ds.categories().len(), 1);
        assert_eq!(ds.annotations().len(), 2);
        assert_eq!(ds.images()[0].file_name.as_deref(), Some("img1.png"));
        assert_eq!(ds.categories()[0].name, "widget");
    }

    #[test]
    fn by_image_index_returns_both_anns() {
        let ds = load_crowd_region();
        let idxs = ds.ann_indices_for_image(ImageId(1));
        assert_eq!(idxs.len(), 2);
        let anns: Vec<_> = ds.ann_iter_for_image(ImageId(1)).collect();
        assert_eq!(anns.len(), 2);
        assert_eq!(anns[0].id, AnnId(1));
        assert_eq!(anns[1].id, AnnId(2));
    }

    #[test]
    fn by_category_index_returns_both_anns() {
        let ds = load_crowd_region();
        let idxs = ds.ann_indices_for_category(CategoryId(1));
        assert_eq!(idxs.len(), 2);
    }

    #[test]
    fn unknown_image_returns_empty_slice() {
        let ds = load_crowd_region();
        assert!(ds.ann_indices_for_image(ImageId(999)).is_empty());
        assert!(ds.ann_indices_for_category(CategoryId(999)).is_empty());
    }

    #[test]
    fn empty_image_or_category_returns_empty_slice_not_missing() {
        // A dataset with an image that has no annotations: the index
        // must be present (empty), so the matching loop can ask
        // without special-casing.
        const ONLY_EMPTY_IMG: &str = r#"{
            "images": [{"id": 7, "width": 1, "height": 1}],
            "annotations": [],
            "categories": [{"id": 3, "name": "thing"}]
        }"#;
        let ds = CocoDataset::from_json_bytes(ONLY_EMPTY_IMG.as_bytes()).unwrap();
        assert!(ds.ann_indices_for_image(ImageId(7)).is_empty());
        assert!(ds.ann_indices_for_category(CategoryId(3)).is_empty());
    }

    #[test]
    fn rejects_annotation_referencing_unknown_image() {
        const BAD: &str = r#"{
            "images": [{"id": 1, "width": 10, "height": 10}],
            "annotations": [
                {"id": 1, "image_id": 99, "category_id": 1,
                 "bbox": [0, 0, 1, 1], "area": 1, "iscrowd": 0}
            ],
            "categories": [{"id": 1, "name": "thing"}]
        }"#;
        let err = CocoDataset::from_json_bytes(BAD.as_bytes()).unwrap_err();
        match err {
            EvalError::InvalidAnnotation { detail } => {
                assert!(detail.contains("image_id=99"), "msg: {detail}");
            }
            other => panic!("expected InvalidAnnotation, got {other:?}"),
        }
    }

    #[test]
    fn rejects_annotation_referencing_unknown_category() {
        const BAD: &str = r#"{
            "images": [{"id": 1, "width": 10, "height": 10}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 42,
                 "bbox": [0, 0, 1, 1], "area": 1, "iscrowd": 0}
            ],
            "categories": [{"id": 1, "name": "thing"}]
        }"#;
        let err = CocoDataset::from_json_bytes(BAD.as_bytes()).unwrap_err();
        match err {
            EvalError::InvalidAnnotation { detail } => {
                assert!(detail.contains("category_id=42"), "msg: {detail}");
            }
            other => panic!("expected InvalidAnnotation, got {other:?}"),
        }
    }

    #[test]
    fn round_trips_through_json() {
        let ds = load_crowd_region();
        let json = serde_json::to_string(&ds.to_json_value()).unwrap();
        let again = CocoDataset::from_json_bytes(json.as_bytes()).unwrap();
        assert_eq!(ds.images(), again.images());
        assert_eq!(ds.categories(), again.categories());
        assert_eq!(ds.annotations(), again.annotations());
    }

    // -- Quirk D1: effective_ignore differs by parity mode ----------------

    #[test]
    fn d1_strict_mode_drops_explicit_ignore_field() {
        // Annotation with iscrowd=0 and explicit ignore=1.
        // Strict (pycocotools): ignore := iscrowd → false.
        // Corrected: respects user's ignore=1 → true.
        const ANN_JSON: &str = r#"{
            "images": [{"id": 1, "width": 10, "height": 10}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1,
                 "bbox": [0, 0, 1, 1], "area": 1,
                 "iscrowd": 0, "ignore": 1}
            ],
            "categories": [{"id": 1, "name": "thing"}]
        }"#;
        let ds = CocoDataset::from_json_bytes(ANN_JSON.as_bytes()).unwrap();
        let ann = &ds.annotations()[0];
        assert!(!ann.effective_ignore(ParityMode::Strict));
        assert!(ann.effective_ignore(ParityMode::Corrected));
    }

    #[test]
    fn d1_strict_mode_uses_iscrowd_when_ignore_absent() {
        // Annotation with iscrowd=1 and no ignore field.
        // Both modes: ignore = is_crowd = true.
        const ANN_JSON: &str = r#"{
            "images": [{"id": 1, "width": 10, "height": 10}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1,
                 "bbox": [0, 0, 1, 1], "area": 1, "iscrowd": 1}
            ],
            "categories": [{"id": 1, "name": "thing"}]
        }"#;
        let ds = CocoDataset::from_json_bytes(ANN_JSON.as_bytes()).unwrap();
        let ann = &ds.annotations()[0];
        assert!(ann.effective_ignore(ParityMode::Strict));
        assert!(ann.effective_ignore(ParityMode::Corrected));
    }

    // -- Property: index invariants hold across arbitrary datasets --------

    fn arb_image() -> impl Strategy<Value = ImageMeta> {
        (1i64..1000, 1u32..2048, 1u32..2048).prop_map(|(id, w, h)| ImageMeta {
            id: ImageId(id),
            width: w,
            height: h,
            file_name: None,
        })
    }

    fn arb_category() -> impl Strategy<Value = CategoryMeta> {
        (1i64..100, "[a-z]{1,8}").prop_map(|(id, name)| CategoryMeta {
            id: CategoryId(id),
            name,
            supercategory: None,
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn index_invariants_hold(
            // Generate a small image set, a small category set, and a
            // bag of annotations whose (image_id, category_id) pick
            // from those sets uniformly. The invariant we check: every
            // annotation appears in exactly one by_image bucket and
            // exactly one by_category bucket, and no bucket contains a
            // stray index.
            images in proptest::collection::vec(arb_image(), 1..6),
            categories in proptest::collection::vec(arb_category(), 1..6),
            n_anns in 0usize..40,
            ann_seed in any::<u64>(),
        ) {
            // De-duplicate ids; HashMaps in `from_parts` collapse them
            // anyway and tests should not depend on prop generators
            // accidentally minting collisions.
            let mut images = images;
            images.sort_by_key(|i| i.id);
            images.dedup_by_key(|i| i.id);
            let mut categories = categories;
            categories.sort_by_key(|c| c.id);
            categories.dedup_by_key(|c| c.id);

            // Cheap deterministic PRNG from ann_seed; avoids pulling
            // in `rand` for a single proptest helper.
            let mut state = ann_seed.wrapping_add(1);
            let mut next = || {
                state = state.wrapping_mul(6364136223846793005)
                             .wrapping_add(1442695040888963407);
                state
            };

            let mut annotations = Vec::with_capacity(n_anns);
            for ann_idx in 0..n_anns {
                let img = &images[(next() as usize) % images.len()];
                let cat = &categories[(next() as usize) % categories.len()];
                annotations.push(CocoAnnotation {
                    id: AnnId(ann_idx as i64 + 1),
                    image_id: img.id,
                    category_id: cat.id,
                    area: 1.0,
                    is_crowd: false,
                    ignore_flag: None,
                    bbox: Bbox { x: 0.0, y: 0.0, w: 1.0, h: 1.0 },
                });
            }

            let ds = CocoDataset::from_parts(
                images.clone(), annotations.clone(), categories.clone()
            ).unwrap();

            // Every annotation index appears exactly once across all
            // by_image buckets and exactly once across all by_category
            // buckets.
            let mut seen_img: Vec<usize> = images.iter()
                .flat_map(|i| ds.ann_indices_for_image(i.id).iter().copied())
                .collect();
            seen_img.sort_unstable();
            let expected: Vec<usize> = (0..annotations.len()).collect();
            prop_assert_eq!(&seen_img, &expected);

            let mut seen_cat: Vec<usize> = categories.iter()
                .flat_map(|c| ds.ann_indices_for_category(c.id).iter().copied())
                .collect();
            seen_cat.sort_unstable();
            prop_assert_eq!(&seen_cat, &expected);

            // Cross-check: every index in by_image[i] has image_id == i.
            for img in &images {
                for &idx in ds.ann_indices_for_image(img.id) {
                    prop_assert_eq!(ds.annotations()[idx].image_id, img.id);
                }
            }
            for cat in &categories {
                for &idx in ds.ann_indices_for_category(cat.id) {
                    prop_assert_eq!(ds.annotations()[idx].category_id, cat.id);
                }
            }
        }
    }
}
