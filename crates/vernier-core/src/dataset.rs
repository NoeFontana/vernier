//! Dataset abstraction and the COCO ground-truth implementation.
//!
//! Per ADR-0005, the matching engine and accumulator are written once
//! and never edited; they are generic over a dataset trait, never over
//! a concrete dataset type. Future datasets (custom corpora, Phase 3
//! keypoint datasets such as CrowdPose) add new `EvalDataset` impls
//! without touching anything in `matching.rs` or `accumulate.rs`.
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
//! - **J3** (`strict`): detection-side area is derived at construction
//!   from the bbox (`bbox.w * bbox.h`) and never read from JSON.
//! - **J1** (`aligned`): user-supplied DT ids are preserved verbatim;
//!   absent ids are auto-assigned sequentially during construction.
//! - **E2 / J4** (`strict`): detections never carry an `iscrowd` flag
//!   — the type does not have the field. JSON inputs that include
//!   `iscrowd=1` are silently dropped, matching pycocotools' overwrite.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::error::EvalError;
use crate::parity::ParityMode;
use crate::segmentation::Segmentation;

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
    /// `J3` derives DT-area from). Phase 3 adds `keypoints` as an
    /// additional optional field.
    pub bbox: Bbox,
    /// COCO `segmentation` field, in any of the three shapes
    /// pycocotools accepts (multi-polygon, uncompressed RLE,
    /// compressed RLE). `None` for keypoint-only annotations or
    /// fixtures that omit it. The matching engine normalizes via
    /// [`Segmentation::to_rle`] at eval time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub segmentation: Option<Segmentation>,
    /// Flat keypoint triplets `[x_1, y_1, v_1, x_2, y_2, v_2, ...]`
    /// (per ADR-0012). `None` for non-keypoint annotations; the eval
    /// pipeline raises [`EvalError::InvalidAnnotation`] when a GT is
    /// missing keypoints under `iouType="keypoints"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keypoints: Option<Vec<f64>>,
    /// COCO `num_keypoints` count of *visible* keypoints (`v > 0`),
    /// per ADR-0012. pycocotools precomputes this on GT (driving the
    /// quirk **D2** implicit-ignore branch); on DT it is not required
    /// and is derived from `keypoints` when needed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_keypoints: Option<u32>,
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
/// not see [`CocoAnnotation`] or any future per-dataset annotation type
/// directly.
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

/// Trait every dataset (COCO, CrowdPose, custom) implements.
///
/// `Send + Sync` is required by the future `BackgroundEvaluator`
/// (separate ADR) so the dataset can be shared across worker threads
/// without copying.
pub trait EvalDataset: Send + Sync {
    /// Concrete annotation type. For [`CocoDataset`] this is
    /// [`CocoAnnotation`]; future datasets may use their own type with
    /// extra metadata.
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

/// LVIS category-frequency tier (quirk **AB1** of ADR-0026).
///
/// Each LVIS category is tagged at dataset publication with one of
/// three buckets, keyed by how many *training* images contain at least
/// one annotation of that category:
///
/// - [`Frequency::Rare`]: `< 10` train images
/// - [`Frequency::Common`]: `[10, 100)` train images
/// - [`Frequency::Frequent`]: `≥ 100` train images
///
/// The boundaries are pinned by the upstream eval code at
/// `lvis/eval.py:537-541`; the LVIS paper's prose ("1-10 / 11-100 /
/// `>100`") is loose — a 10-image category is `Common`, not `Rare`.
/// The `frequency` field is precomputed at dataset publication;
/// vernier reads it as-is and never derives it from `image_count`
/// (quirk **AB2**).
///
/// Serializes to/from the single-letter form (`"r"` / `"c"` / `"f"`)
/// the LVIS JSON schema uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Frequency {
    /// `< 10` train images.
    #[serde(rename = "r")]
    Rare,
    /// `[10, 100)` train images.
    #[serde(rename = "c")]
    Common,
    /// `≥ 100` train images.
    #[serde(rename = "f")]
    Frequent,
}

/// On-disk LVIS image record. Carries the COCO image fields plus the
/// LVIS-specific federated lists. The `pos_category_ids` set is
/// **derived** from GT annotations at load (quirk **AA1**) and is not
/// a JSON field — only `neg` and `not_exhaustive` are explicit.
#[derive(Debug, Clone, Deserialize)]
struct LvisImageRaw {
    id: ImageId,
    width: u32,
    height: u32,
    #[serde(default)]
    file_name: Option<String>,
    /// LVIS-only: categories verified absent from this image. `None`
    /// in the wild means a malformed LVIS JSON; v1 spec requires the
    /// field on every image (possibly empty).
    #[serde(default)]
    neg_category_ids: Option<Vec<CategoryId>>,
    /// LVIS-only: categories whose annotations on this image are not
    /// guaranteed exhaustive. Subset of `pos` by spec; consumed by
    /// quirk **AA3** to extend `dt_ignore` on unmatched DTs in the
    /// cell.
    #[serde(default)]
    not_exhaustive_category_ids: Option<Vec<CategoryId>>,
}

/// On-disk LVIS category record. Carries the COCO category fields
/// plus the `frequency` tag (quirk **AB1**). `image_count` and
/// `instance_count` are stored on the upstream JSON but **not read**
/// by the eval code (quirk **AB2**); we drop them on load.
#[derive(Debug, Clone, Deserialize)]
struct LvisCategoryRaw {
    id: CategoryId,
    name: String,
    #[serde(default)]
    supercategory: Option<String>,
    /// Required field on every LVIS v1 category. `None` here means the
    /// JSON entry omitted it; collected and surfaced via
    /// [`EvalError::MissingFrequency`] (quirk **AB6** corrected).
    #[serde(default)]
    frequency: Option<Frequency>,
}

/// On-disk shape of an LVIS v1 ground-truth JSON file. Structurally
/// COCO JSON (quirk **AG1**) plus the federated extras on per-image
/// and per-category records. Annotations are byte-identical between
/// COCO and LVIS schemas, so [`CocoAnnotation`] is reused.
#[derive(Debug, Clone, Deserialize)]
struct LvisJson {
    images: Vec<LvisImageRaw>,
    annotations: Vec<CocoAnnotation>,
    categories: Vec<LvisCategoryRaw>,
}

/// COCO ground-truth dataset.
///
/// Storage is a single `Arc<Vec<CocoAnnotation>>` plus per-image and
/// per-category index vectors. The `Arc` makes the dataset cheaply
/// shareable across worker threads (the `BackgroundEvaluator` from a
/// future ADR depends on this); the index vectors are owned by the
/// `CocoDataset` because they're cheap to rebuild and rebuild needs
/// to happen exactly when the annotation set changes.
///
/// ## LVIS federated metadata (ADR-0026)
///
/// Optional per-image positive / negative / not-exhaustive category
/// sets and per-category frequency tags. Populated only by
/// [`CocoDataset::from_lvis_json_bytes`]; the COCO loader leaves them
/// all `None`. Their absence is the COCO default — the orchestrator
/// reading these fields treats `None` as "not provided" and falls
/// back to COCO semantics on those cells (quirks **AA1**–**AA7**,
/// **AB1**, **AG1**).
///
/// The semantics are inert at the dataset layer: storing the maps
/// neither matches nor accumulates differently. PR-3 of the ADR-0026
/// rollout adds the orchestrator branches that read them.
#[derive(Debug, Clone)]
pub struct CocoDataset {
    images: Arc<Vec<ImageMeta>>,
    categories: Arc<Vec<CategoryMeta>>,
    annotations: Arc<Vec<CocoAnnotation>>,
    by_image: HashMap<ImageId, Vec<usize>>,
    by_category: HashMap<CategoryId, Vec<usize>>,
    by_image_cat: HashMap<(ImageId, CategoryId), Vec<usize>>,
    pos_category_ids: Option<HashMap<ImageId, HashSet<CategoryId>>>,
    neg_category_ids: Option<HashMap<ImageId, HashSet<CategoryId>>>,
    not_exhaustive_category_ids: Option<HashMap<ImageId, HashSet<CategoryId>>>,
    category_frequency: Option<HashMap<CategoryId, Frequency>>,
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
        let known_images: HashSet<ImageId> = images.iter().map(|i| i.id).collect();
        let known_categories: HashSet<CategoryId> = categories.iter().map(|c| c.id).collect();

        let mut by_image: HashMap<ImageId, Vec<usize>> = HashMap::with_capacity(images.len());
        let mut by_category: HashMap<CategoryId, Vec<usize>> =
            HashMap::with_capacity(categories.len());
        let mut by_image_cat: HashMap<(ImageId, CategoryId), Vec<usize>> = HashMap::new();

        for (idx, ann) in annotations.iter().enumerate() {
            if !known_images.contains(&ann.image_id) {
                return Err(EvalError::InvalidAnnotation {
                    detail: format!(
                        "annotation id={} references unknown image_id={}",
                        ann.id.0, ann.image_id.0
                    ),
                });
            }
            if !known_categories.contains(&ann.category_id) {
                return Err(EvalError::InvalidAnnotation {
                    detail: format!(
                        "annotation id={} references unknown category_id={}",
                        ann.id.0, ann.category_id.0
                    ),
                });
            }
            by_image.entry(ann.image_id).or_default().push(idx);
            by_category.entry(ann.category_id).or_default().push(idx);
            by_image_cat
                .entry((ann.image_id, ann.category_id))
                .or_default()
                .push(idx);
        }

        Ok(Self {
            images: Arc::new(images),
            categories: Arc::new(categories),
            annotations: Arc::new(annotations),
            by_image,
            by_category,
            by_image_cat,
            pos_category_ids: None,
            neg_category_ids: None,
            not_exhaustive_category_ids: None,
            category_frequency: None,
        })
    }

    /// Loads an LVIS v1 ground-truth dataset from a JSON byte slice.
    ///
    /// LVIS JSON is structurally COCO JSON plus per-image
    /// `neg_category_ids` / `not_exhaustive_category_ids` and
    /// per-category `frequency` (quirk **AG1**). This loader reads the
    /// extras into the federated metadata fields on the returned
    /// dataset; the underlying `images` / `annotations` / `categories`
    /// projections match what [`Self::from_json_bytes`] would produce
    /// on the same JSON.
    ///
    /// ## Validation
    ///
    /// - **AA1.** `pos_category_ids[I]` is **derived** from GT
    ///   annotations: `pos[I] = {ann.category_id for ann in
    ///   annotations[I]}`. Not a JSON field. A category with zero
    ///   annotations on `I` is *not* in `pos[I]`.
    /// - **AA7 (corrected).** Disjointness invariants are enforced at
    ///   load:
    ///     - `pos[I] ∩ neg[I] = ∅` — a category with GT on an image
    ///       cannot also be in `neg[I]`.
    ///     - `not_exhaustive[I] ⊆ pos[I]` — by spec, not_exhaustive
    ///       is a subset of pos.
    ///     - `not_exhaustive[I] ∩ neg[I] = ∅` — equivalent restatement
    ///       given the prior two.
    ///
    ///   The first violation surfaces as
    ///   [`EvalError::LvisFederatedConflict`] with the offending
    ///   `(image_id, category_id)`.
    /// - **AB6 (corrected).** Every category must carry a `frequency`
    ///   tag. Missing tags are collected across the full categories
    ///   list and surfaced once via [`EvalError::MissingFrequency`]
    ///   with a sorted id list — more debuggable than lvis-api's
    ///   mid-eval `KeyError` on the first miss.
    ///
    /// Per-image `neg_category_ids` and `not_exhaustive_category_ids`
    /// are optional in the JSON: an absent field is treated as an
    /// empty set, which matches the LVIS v1 semantic ("no negatives /
    /// nothing flagged non-exhaustive on this image").
    pub fn from_lvis_json_bytes(bytes: &[u8]) -> Result<Self, EvalError> {
        let raw: LvisJson = serde_json::from_slice(bytes)?;

        let images: Vec<ImageMeta> = raw
            .images
            .iter()
            .map(|im| ImageMeta {
                id: im.id,
                width: im.width,
                height: im.height,
                file_name: im.file_name.clone(),
            })
            .collect();
        let categories: Vec<CategoryMeta> = raw
            .categories
            .iter()
            .map(|c| CategoryMeta {
                id: c.id,
                name: c.name.clone(),
                supercategory: c.supercategory.clone(),
            })
            .collect();

        // AB6 (corrected): collect all categories missing `frequency`
        // and raise once with the full list. Sorted ascending for
        // stable error messages.
        let mut missing_freq: Vec<i64> = raw
            .categories
            .iter()
            .filter(|c| c.frequency.is_none())
            .map(|c| c.id.0)
            .collect();
        if !missing_freq.is_empty() {
            missing_freq.sort_unstable();
            return Err(EvalError::MissingFrequency {
                category_ids: missing_freq,
            });
        }
        let category_frequency: HashMap<CategoryId, Frequency> = raw
            .categories
            .iter()
            .filter_map(|c| c.frequency.map(|f| (c.id, f)))
            .collect();

        // Build the dataset spine via the existing constructor — that
        // gives us the ref-integrity validation (J5 / AG1) for free.
        let mut dataset = Self::from_parts(images, raw.annotations, categories)?;

        // AA1: derive pos[I] from GTs. Defaults each image to an empty
        // set so callers can ask without special-casing.
        let mut pos: HashMap<ImageId, HashSet<CategoryId>> =
            HashMap::with_capacity(raw.images.len());
        for im in &raw.images {
            pos.entry(im.id).or_default();
        }
        for ann in dataset.annotations.iter() {
            pos.entry(ann.image_id).or_default().insert(ann.category_id);
        }

        // Project explicit `neg` / `not_exhaustive` fields onto sets;
        // treat absent / empty as the empty set.
        let mut neg: HashMap<ImageId, HashSet<CategoryId>> =
            HashMap::with_capacity(raw.images.len());
        let mut nel: HashMap<ImageId, HashSet<CategoryId>> =
            HashMap::with_capacity(raw.images.len());
        for im in &raw.images {
            let neg_set: HashSet<CategoryId> = im
                .neg_category_ids
                .as_deref()
                .unwrap_or(&[])
                .iter()
                .copied()
                .collect();
            let nel_set: HashSet<CategoryId> = im
                .not_exhaustive_category_ids
                .as_deref()
                .unwrap_or(&[])
                .iter()
                .copied()
                .collect();
            neg.insert(im.id, neg_set);
            nel.insert(im.id, nel_set);
        }

        // AA7 (corrected): disjointness validation.
        for im in &raw.images {
            let image_id = im.id;
            let pos_i = pos.get(&image_id).map_or_else(HashSet::new, Clone::clone);
            let neg_i = &neg[&image_id];
            let nel_i = &nel[&image_id];

            // pos ∩ neg: a category with GT on this image cannot also
            // be in neg.
            if let Some(c) = pos_i.intersection(neg_i).next().copied() {
                return Err(EvalError::LvisFederatedConflict {
                    image_id: image_id.0,
                    category_id: c.0,
                    detail: "category has GT on image but is also in neg_category_ids",
                });
            }
            // not_exhaustive ⊆ pos: by spec.
            if let Some(c) = nel_i.difference(&pos_i).next().copied() {
                return Err(EvalError::LvisFederatedConflict {
                    image_id: image_id.0,
                    category_id: c.0,
                    detail:
                        "category in not_exhaustive_category_ids but not in pos (no GT on image)",
                });
            }
            // not_exhaustive ∩ neg: implied by the first two but
            // checked explicitly so a malformed JSON gets the most
            // direct error.
            if let Some(c) = nel_i.intersection(neg_i).next().copied() {
                return Err(EvalError::LvisFederatedConflict {
                    image_id: image_id.0,
                    category_id: c.0,
                    detail: "category in both not_exhaustive_category_ids and neg_category_ids",
                });
            }
        }

        dataset.pos_category_ids = Some(pos);
        dataset.neg_category_ids = Some(neg);
        dataset.not_exhaustive_category_ids = Some(nel);
        dataset.category_frequency = Some(category_frequency);
        Ok(dataset)
    }

    /// Per-image positive-category set, derived from GTs at load time
    /// (quirk **AA1**). `Some` only when the dataset was built by
    /// [`Self::from_lvis_json_bytes`].
    pub fn pos_category_ids(&self) -> Option<&HashMap<ImageId, HashSet<CategoryId>>> {
        self.pos_category_ids.as_ref()
    }

    /// Per-image negative-category set, read verbatim from the LVIS
    /// JSON (quirk **AA2**). `Some` only when the dataset was built
    /// by [`Self::from_lvis_json_bytes`].
    pub fn neg_category_ids(&self) -> Option<&HashMap<ImageId, HashSet<CategoryId>>> {
        self.neg_category_ids.as_ref()
    }

    /// Per-image not-exhaustive-category set, read verbatim from the
    /// LVIS JSON (quirk **AA3**). `Some` only when the dataset was
    /// built by [`Self::from_lvis_json_bytes`].
    pub fn not_exhaustive_category_ids(&self) -> Option<&HashMap<ImageId, HashSet<CategoryId>>> {
        self.not_exhaustive_category_ids.as_ref()
    }

    /// Per-category frequency tag, read verbatim from the LVIS JSON
    /// (quirk **AB1**). `Some` only when the dataset was built by
    /// [`Self::from_lvis_json_bytes`]; missing-on-some-categories
    /// inputs are rejected at load (quirk **AB6**).
    pub fn category_frequency(&self) -> Option<&HashMap<CategoryId, Frequency>> {
        self.category_frequency.as_ref()
    }

    /// `true` when the dataset carries LVIS federated metadata.
    /// Equivalent to `self.pos_category_ids().is_some()`; used by the
    /// orchestrator (PR-3) to gate the cell-skip and `dt_ignore`
    /// branches.
    pub fn is_federated(&self) -> bool {
        self.pos_category_ids.is_some()
    }

    /// Round-trips the dataset to the on-disk JSON shape, preserving
    /// every field vernier carries. Useful for fixture authoring and
    /// for debugging serde mismatches.
    ///
    /// LVIS federated metadata is **not** included in the output —
    /// the round trip targets the COCO schema only. Callers needing
    /// to round-trip LVIS JSON must use the source bytes directly.
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

impl CocoDataset {
    /// Indices into [`Self::annotations`] for a given `(image, category)`
    /// cell. Empty when no GT of that category exists on that image.
    pub fn ann_indices_for(&self, image: ImageId, cat: CategoryId) -> &[usize] {
        self.by_image_cat
            .get(&(image, cat))
            .map_or(&[][..], Vec::as_slice)
    }
}

// ---------------------------------------------------------------------------
// detections (DT side)
// ---------------------------------------------------------------------------

/// One COCO detection record (the DT side, what `loadRes` consumes).
///
/// Per the dispositions in this module's header:
///
/// - `is_crowd` does not exist as a field — quirks **E2 / J4**.
/// - `area` is derived from `bbox` at construction (`bbox.w * bbox.h`) —
///   quirk **J3**.
/// - `id` is honored when the user supplies one and auto-assigned
///   otherwise — quirk **J1** (`aligned`, an opinionated improvement
///   over pycocotools' silent overwrite).
#[derive(Debug, Clone, PartialEq)]
pub struct CocoDetection {
    /// Detection id. Either user-supplied (J1) or auto-assigned by
    /// [`CocoDetections::from_inputs`].
    pub id: AnnId,
    /// Image this detection is on.
    pub image_id: ImageId,
    /// Category this detection predicts.
    pub category_id: CategoryId,
    /// Confidence score. Sort key for the matching engine.
    pub score: f64,
    /// Bounding box (`(x, y, w, h)`).
    pub bbox: Bbox,
    /// Pixel area, derived from `bbox` per quirk **J3**.
    pub area: f64,
    /// Segmentation prediction, when the detector emits one. `None`
    /// for bbox-only detectors. Parity dispositions match
    /// [`CocoAnnotation::segmentation`].
    pub segmentation: Option<Segmentation>,
    /// Flat keypoint triplets `[x_1, y_1, v_1, x_2, y_2, v_2, ...]`
    /// (per ADR-0012). `None` for bbox-/segm-only detectors; the eval
    /// pipeline raises [`EvalError::InvalidAnnotation`] when a DT is
    /// missing keypoints under `iouType="keypoints"`.
    pub keypoints: Option<Vec<f64>>,
    /// COCO `num_keypoints` count of *visible* keypoints. On DT this
    /// field is not required (pycocotools never reads it); the OKS
    /// pipeline derives it from `keypoints` when needed. Tracked here
    /// for shape-parity with [`CocoAnnotation::num_keypoints`].
    pub num_keypoints: Option<u32>,
}

impl Annotation for CocoDetection {
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
        false
    }
    fn effective_ignore(&self, _: ParityMode) -> bool {
        false
    }
}

/// Caller-side input for one detection. Mirrors the shape of a single
/// entry of a COCO results JSON array but uses typed ids.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DetectionInput {
    /// Optional user-supplied id (quirk **J1**). Absent → auto-assigned.
    #[serde(default)]
    pub id: Option<AnnId>,
    /// Image id.
    pub image_id: ImageId,
    /// Category id.
    pub category_id: CategoryId,
    /// Confidence score.
    pub score: f64,
    /// Bounding box.
    pub bbox: Bbox,
    /// Optional segmentation prediction. `None` for bbox-only
    /// detectors. Stored verbatim and normalized via
    /// [`Segmentation::to_rle`] at eval time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub segmentation: Option<Segmentation>,
    /// Optional keypoint prediction (flat `[x, y, v, ...]` triplets,
    /// per ADR-0012). `None` for non-keypoint detectors.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keypoints: Option<Vec<f64>>,
    /// Optional `num_keypoints` count. The OKS path derives this from
    /// `keypoints` when absent (DT side does not require it).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_keypoints: Option<u32>,
}

/// COCO detections collection — flat storage plus `(image, category)`-
/// and per-image indices for the per-cell gather.
#[derive(Debug, Clone)]
pub struct CocoDetections {
    detections: Arc<Vec<CocoDetection>>,
    by_image_cat: HashMap<(ImageId, CategoryId), Vec<usize>>,
    by_image: HashMap<ImageId, Vec<usize>>,
}

impl CocoDetections {
    /// Loads detections from the JSON array shape pycocotools'
    /// `loadRes` consumes (a list of objects with `image_id`,
    /// `category_id`, `bbox`, `score`, optional `id`).
    ///
    /// `iscrowd` and `area` fields, if present, are silently dropped:
    /// quirks **E2/J4** force `is_crowd=0` and quirk **J3** derives
    /// `area` from `bbox`.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, EvalError> {
        let raw: Vec<DetectionInput> = serde_json::from_slice(bytes)?;
        Self::from_inputs(raw)
    }

    /// Builds a [`CocoDetections`] from typed inputs. Auto-assigns ids
    /// (quirk **J1**) for inputs that did not supply one, validates
    /// finite scores, and derives areas (quirk **J3**).
    pub fn from_inputs(inputs: Vec<DetectionInput>) -> Result<Self, EvalError> {
        let mut detections = Vec::with_capacity(inputs.len());
        let mut next_auto = 1i64;
        for input in inputs {
            if !input.score.is_finite() {
                return Err(EvalError::NonFinite {
                    context: "detection score",
                });
            }
            let id = match input.id {
                Some(id) => id,
                None => {
                    let id = AnnId(next_auto);
                    next_auto += 1;
                    id
                }
            };
            detections.push(CocoDetection {
                id,
                image_id: input.image_id,
                category_id: input.category_id,
                score: input.score,
                bbox: input.bbox,
                area: input.bbox.w * input.bbox.h,
                segmentation: input.segmentation,
                keypoints: input.keypoints,
                num_keypoints: input.num_keypoints,
            });
        }

        let mut by_image_cat: HashMap<(ImageId, CategoryId), Vec<usize>> = HashMap::new();
        let mut by_image: HashMap<ImageId, Vec<usize>> = HashMap::new();
        for (idx, dt) in detections.iter().enumerate() {
            by_image_cat
                .entry((dt.image_id, dt.category_id))
                .or_default()
                .push(idx);
            by_image.entry(dt.image_id).or_default().push(idx);
        }

        Ok(Self {
            detections: Arc::new(detections),
            by_image_cat,
            by_image,
        })
    }

    /// Build from already-resolved records, preserving their ids and
    /// fields verbatim. Used by the streaming evaluator to assemble a
    /// `CocoDetections` view across batches at finalize/snapshot time
    /// without re-running the auto-id and area-derivation logic in
    /// [`Self::from_inputs`].
    pub fn from_records(records: Vec<CocoDetection>) -> Self {
        let mut by_image_cat: HashMap<(ImageId, CategoryId), Vec<usize>> = HashMap::new();
        let mut by_image: HashMap<ImageId, Vec<usize>> = HashMap::new();
        for (idx, dt) in records.iter().enumerate() {
            by_image_cat
                .entry((dt.image_id, dt.category_id))
                .or_default()
                .push(idx);
            by_image.entry(dt.image_id).or_default().push(idx);
        }
        Self {
            detections: Arc::new(records),
            by_image_cat,
            by_image,
        }
    }

    /// Flat slice of every detection.
    pub fn detections(&self) -> &[CocoDetection] {
        &self.detections
    }

    /// Indices into [`Self::detections`] for one `(image, category)`
    /// cell. Empty slice when the cell is empty (no detections of that
    /// category on that image).
    pub fn indices_for(&self, image: ImageId, cat: CategoryId) -> &[usize] {
        self.by_image_cat
            .get(&(image, cat))
            .map_or(&[][..], Vec::as_slice)
    }

    /// Indices into [`Self::detections`] for every detection on an
    /// image, regardless of category. Path used when `useCats=false`
    /// (quirk **L4**).
    pub fn indices_for_image(&self, image: ImageId) -> &[usize] {
        self.by_image.get(&image).map_or(&[][..], Vec::as_slice)
    }
}

// ---------------------------------------------------------------------------
// serde glue
// ---------------------------------------------------------------------------

/// COCO JSON uses `0`/`1` ints for `iscrowd` / `ignore`, but a
/// permissive reader also accepts bool literals. Shared between the
/// required and optional flag deserializers below.
#[derive(Deserialize)]
#[serde(untagged)]
enum BoolOrInt {
    Bool(bool),
    Int(i64),
}

impl BoolOrInt {
    fn into_bool<E: serde::de::Error>(self) -> Result<bool, E> {
        match self {
            Self::Bool(b) => Ok(b),
            Self::Int(0) => Ok(false),
            Self::Int(1) => Ok(true),
            Self::Int(other) => Err(E::custom(format!(
                "expected 0 or 1 for COCO bool field, got {other}"
            ))),
        }
    }
}

fn deserialize_bool_int<'de, D>(de: D) -> Result<bool, D::Error>
where
    D: serde::Deserializer<'de>,
{
    BoolOrInt::deserialize(de)?.into_bool()
}

fn deserialize_opt_bool_int<'de, D>(de: D) -> Result<Option<bool>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Option::<BoolOrInt>::deserialize(de)?
        .map(BoolOrInt::into_bool)
        .transpose()
}

#[cfg(test)]
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

    // -- Per-cell index ((image, category)) -------------------------------

    #[test]
    fn ann_indices_for_image_cat_returns_correct_subset() {
        const TWO_CATS: &str = r#"{
            "images": [{"id": 1, "width": 10, "height": 10}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1,
                 "bbox": [0, 0, 1, 1], "area": 1, "iscrowd": 0},
                {"id": 2, "image_id": 1, "category_id": 2,
                 "bbox": [0, 0, 1, 1], "area": 1, "iscrowd": 0},
                {"id": 3, "image_id": 1, "category_id": 1,
                 "bbox": [0, 0, 1, 1], "area": 1, "iscrowd": 0}
            ],
            "categories": [
                {"id": 1, "name": "a"}, {"id": 2, "name": "b"}
            ]
        }"#;
        let ds = CocoDataset::from_json_bytes(TWO_CATS.as_bytes()).unwrap();
        let cat1: Vec<AnnId> = ds
            .ann_indices_for(ImageId(1), CategoryId(1))
            .iter()
            .map(|&i| ds.annotations()[i].id)
            .collect();
        assert_eq!(cat1, vec![AnnId(1), AnnId(3)]);
        let cat2: Vec<AnnId> = ds
            .ann_indices_for(ImageId(1), CategoryId(2))
            .iter()
            .map(|&i| ds.annotations()[i].id)
            .collect();
        assert_eq!(cat2, vec![AnnId(2)]);
        assert!(ds.ann_indices_for(ImageId(1), CategoryId(99)).is_empty());
        assert!(ds.ann_indices_for(ImageId(99), CategoryId(1)).is_empty());
    }

    // -- CocoDetections: J1 (auto-id), J3 (area from bbox), validation ----

    fn dt_input(image: i64, cat: i64, score: f64, bbox: (f64, f64, f64, f64)) -> DetectionInput {
        DetectionInput {
            id: None,
            image_id: ImageId(image),
            category_id: CategoryId(cat),
            score,
            bbox: Bbox {
                x: bbox.0,
                y: bbox.1,
                w: bbox.2,
                h: bbox.3,
            },
            segmentation: None,
            keypoints: None,
            num_keypoints: None,
        }
    }

    #[test]
    fn j1_auto_assigns_ids_when_absent() {
        let dts = CocoDetections::from_inputs(vec![
            dt_input(1, 1, 0.9, (0.0, 0.0, 1.0, 1.0)),
            dt_input(1, 1, 0.8, (0.0, 0.0, 1.0, 1.0)),
        ])
        .unwrap();
        let ids: Vec<AnnId> = dts.detections().iter().map(|d| d.id).collect();
        assert_eq!(ids, vec![AnnId(1), AnnId(2)]);
    }

    #[test]
    fn j1_preserves_user_supplied_ids() {
        let mut a = dt_input(1, 1, 0.9, (0.0, 0.0, 1.0, 1.0));
        a.id = Some(AnnId(42));
        let mut b = dt_input(1, 1, 0.8, (0.0, 0.0, 1.0, 1.0));
        b.id = Some(AnnId(7));
        let dts = CocoDetections::from_inputs(vec![a, b]).unwrap();
        let ids: Vec<AnnId> = dts.detections().iter().map(|d| d.id).collect();
        assert_eq!(ids, vec![AnnId(42), AnnId(7)]);
    }

    #[test]
    fn j3_derives_area_from_bbox() {
        let dts =
            CocoDetections::from_inputs(vec![dt_input(1, 1, 0.5, (10.0, 10.0, 4.0, 5.0))]).unwrap();
        assert_eq!(dts.detections()[0].area, 20.0);
    }

    #[test]
    fn rejects_non_finite_score() {
        let err = CocoDetections::from_inputs(vec![dt_input(1, 1, f64::NAN, (0.0, 0.0, 1.0, 1.0))])
            .unwrap_err();
        assert!(matches!(
            err,
            EvalError::NonFinite {
                context: "detection score"
            }
        ));
    }

    #[test]
    fn detections_indices_per_image_cat() {
        let dts = CocoDetections::from_inputs(vec![
            dt_input(1, 1, 0.9, (0.0, 0.0, 1.0, 1.0)),
            dt_input(1, 2, 0.8, (0.0, 0.0, 1.0, 1.0)),
            dt_input(2, 1, 0.7, (0.0, 0.0, 1.0, 1.0)),
        ])
        .unwrap();
        assert_eq!(dts.indices_for(ImageId(1), CategoryId(1)), &[0]);
        assert_eq!(dts.indices_for(ImageId(1), CategoryId(2)), &[1]);
        assert_eq!(dts.indices_for(ImageId(2), CategoryId(1)), &[2]);
        assert!(dts.indices_for(ImageId(99), CategoryId(1)).is_empty());
        // Quirk L4 path: indices_for_image returns every category.
        let img1: Vec<usize> = dts.indices_for_image(ImageId(1)).to_vec();
        assert_eq!(img1, vec![0, 1]);
    }

    #[test]
    fn loads_detections_from_json_array() {
        const JSON: &str = r#"[
            {"image_id": 1, "category_id": 1, "score": 0.9,
             "bbox": [0, 0, 2, 3]},
            {"id": 7, "image_id": 1, "category_id": 1, "score": 0.5,
             "bbox": [1, 1, 1, 1]}
        ]"#;
        let dts = CocoDetections::from_json_bytes(JSON.as_bytes()).unwrap();
        let ds = dts.detections();
        assert_eq!(ds[0].id, AnnId(1)); // auto-assigned
        assert_eq!(ds[0].area, 6.0); // J3
        assert_eq!(ds[1].id, AnnId(7)); // user-supplied (J1)
        assert!(!ds[0].is_crowd()); // E2/J4
        assert!(ds[0].segmentation.is_none());
    }

    // -- Phase 2: segmentation field on GT and DT -----------------------------

    #[test]
    fn gt_loads_polygon_segmentation() {
        const JSON: &str = r#"{
            "images": [{"id": 1, "width": 10, "height": 10}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1,
                 "bbox": [0, 0, 4, 4], "area": 16, "iscrowd": 0,
                 "segmentation": [[0, 0, 4, 0, 4, 4, 0, 4]]}
            ],
            "categories": [{"id": 1, "name": "thing"}]
        }"#;
        let ds = CocoDataset::from_json_bytes(JSON.as_bytes()).unwrap();
        let seg = ds.annotations()[0].segmentation.as_ref().unwrap();
        let rle = seg.to_rle(10, 10).unwrap();
        assert_eq!(rle.area(), 16);
    }

    #[test]
    fn gt_loads_compressed_rle_segmentation() {
        let counts_str = String::from_utf8(vernier_mask::encode_counts(&[0, 16])).unwrap();
        let json = format!(
            r#"{{
            "images": [{{"id": 1, "width": 4, "height": 4}}],
            "annotations": [
                {{"id": 1, "image_id": 1, "category_id": 1,
                 "bbox": [0, 0, 4, 4], "area": 16, "iscrowd": 1,
                 "segmentation": {{"size": [4, 4], "counts": "{counts_str}"}}}}
            ],
            "categories": [{{"id": 1, "name": "thing"}}]
        }}"#
        );
        let ds = CocoDataset::from_json_bytes(json.as_bytes()).unwrap();
        let seg = ds.annotations()[0].segmentation.as_ref().unwrap();
        let rle = seg.to_rle(4, 4).unwrap();
        assert_eq!((rle.h, rle.w), (4, 4));
        assert_eq!(rle.area(), 16);
    }

    #[test]
    fn gt_segmentation_round_trips_through_to_json_value() {
        const JSON: &str = r#"{
            "images": [{"id": 1, "width": 10, "height": 10}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1,
                 "bbox": [0, 0, 4, 4], "area": 16, "iscrowd": 0,
                 "segmentation": [[0, 0, 4, 0, 4, 4, 0, 4]]}
            ],
            "categories": [{"id": 1, "name": "thing"}]
        }"#;
        let ds = CocoDataset::from_json_bytes(JSON.as_bytes()).unwrap();
        let serialized = serde_json::to_string(&ds.to_json_value()).unwrap();
        let again = CocoDataset::from_json_bytes(serialized.as_bytes()).unwrap();
        assert_eq!(ds.annotations(), again.annotations());
    }

    #[test]
    fn gt_without_segmentation_field_loads_as_none() {
        let ds = load_crowd_region();
        assert!(ds.annotations().iter().all(|a| a.segmentation.is_none()));
    }

    #[test]
    fn dt_loads_compressed_rle_segmentation() {
        const JSON: &str = r#"[
            {"image_id": 1, "category_id": 1, "score": 0.9,
             "bbox": [0, 0, 4, 4],
             "segmentation": {"size": [4, 4], "counts": "04L4"}}
        ]"#;
        let dts = CocoDetections::from_json_bytes(JSON.as_bytes()).unwrap();
        assert!(dts.detections()[0].segmentation.is_some());
    }

    #[test]
    fn dt_without_segmentation_loads_as_none() {
        const JSON: &str = r#"[
            {"image_id": 1, "category_id": 1, "score": 0.9, "bbox": [0, 0, 1, 1]}
        ]"#;
        let dts = CocoDetections::from_json_bytes(JSON.as_bytes()).unwrap();
        assert!(dts.detections()[0].segmentation.is_none());
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
                    segmentation: None,
                    keypoints: None,
                    num_keypoints: None,
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

    // -- ADR-0026: LVIS federated metadata loader -----------------------------

    /// Minimal valid LVIS GT: 2 images, 2 categories with frequencies,
    /// 1 GT on image 1 (cat 1) and 1 GT on image 2 (cat 2). Image 1
    /// has cat 2 in `neg`, image 2 has cat 1 flagged not-exhaustive.
    /// Used as the base fixture for the AA1 / AA7 / AB6 tests; the
    /// negative tests mutate it to violate one constraint at a time.
    const LVIS_MIN_VALID: &str = r#"{
        "images": [
            {"id": 1, "width": 100, "height": 100,
             "neg_category_ids": [2], "not_exhaustive_category_ids": []},
            {"id": 2, "width": 100, "height": 100,
             "neg_category_ids": [], "not_exhaustive_category_ids": [2]}
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1,
             "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0},
            {"id": 2, "image_id": 2, "category_id": 2,
             "bbox": [0, 0, 20, 20], "area": 400, "iscrowd": 0}
        ],
        "categories": [
            {"id": 1, "name": "a", "frequency": "f"},
            {"id": 2, "name": "b", "frequency": "r"}
        ]
    }"#;

    #[test]
    fn lvis_loads_minimal_valid_dataset() {
        let ds = CocoDataset::from_lvis_json_bytes(LVIS_MIN_VALID.as_bytes()).unwrap();
        // Spine identical to a COCO load.
        assert_eq!(ds.images().len(), 2);
        assert_eq!(ds.categories().len(), 2);
        assert_eq!(ds.annotations().len(), 2);
        // Federated metadata populated.
        assert!(ds.is_federated());
        let pos = ds.pos_category_ids().unwrap();
        let neg = ds.neg_category_ids().unwrap();
        let nel = ds.not_exhaustive_category_ids().unwrap();
        let freq = ds.category_frequency().unwrap();
        // AA1: pos derived from GTs.
        assert_eq!(pos[&ImageId(1)], HashSet::from([CategoryId(1)]));
        assert_eq!(pos[&ImageId(2)], HashSet::from([CategoryId(2)]));
        // AA2: neg read verbatim.
        assert_eq!(neg[&ImageId(1)], HashSet::from([CategoryId(2)]));
        assert_eq!(neg[&ImageId(2)], HashSet::new());
        // AA3: not_exhaustive read verbatim.
        assert_eq!(nel[&ImageId(1)], HashSet::new());
        assert_eq!(nel[&ImageId(2)], HashSet::from([CategoryId(2)]));
        // AB1: frequency tags.
        assert_eq!(freq[&CategoryId(1)], Frequency::Frequent);
        assert_eq!(freq[&CategoryId(2)], Frequency::Rare);
    }

    #[test]
    fn aa1_pos_derived_from_gts_does_not_include_zero_ann_categories() {
        // Cat 2 has a GT only on image 2; pos[image 1] must NOT
        // contain cat 2 (it's only in neg there).
        let ds = CocoDataset::from_lvis_json_bytes(LVIS_MIN_VALID.as_bytes()).unwrap();
        let pos = ds.pos_category_ids().unwrap();
        assert!(!pos[&ImageId(1)].contains(&CategoryId(2)));
        assert!(!pos[&ImageId(2)].contains(&CategoryId(1)));
    }

    #[test]
    fn from_json_bytes_leaves_federated_metadata_none() {
        // The COCO loader on the same JSON shape ignores the LVIS
        // extras and leaves federated metadata empty (the orchestrator
        // then runs COCO semantics on the cells).
        let ds = CocoDataset::from_json_bytes(LVIS_MIN_VALID.as_bytes()).unwrap();
        assert!(!ds.is_federated());
        assert!(ds.pos_category_ids().is_none());
        assert!(ds.neg_category_ids().is_none());
        assert!(ds.not_exhaustive_category_ids().is_none());
        assert!(ds.category_frequency().is_none());
    }

    #[test]
    fn aa7_pos_intersect_neg_rejected() {
        // Cat 1 has a GT on image 1 → it's in pos[1]; the JSON also
        // lists cat 1 in image 1's neg → conflict.
        const BAD: &str = r#"{
            "images": [
                {"id": 1, "width": 10, "height": 10,
                 "neg_category_ids": [1], "not_exhaustive_category_ids": []}
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1,
                 "bbox": [0, 0, 5, 5], "area": 25, "iscrowd": 0}
            ],
            "categories": [{"id": 1, "name": "a", "frequency": "f"}]
        }"#;
        let err = CocoDataset::from_lvis_json_bytes(BAD.as_bytes()).unwrap_err();
        match err {
            EvalError::LvisFederatedConflict {
                image_id,
                category_id,
                detail,
            } => {
                assert_eq!(image_id, 1);
                assert_eq!(category_id, 1);
                assert!(detail.contains("GT"));
            }
            other => panic!("expected LvisFederatedConflict, got {other:?}"),
        }
    }

    #[test]
    fn aa7_not_exhaustive_outside_pos_rejected() {
        // Image 1 lists cat 2 in not_exhaustive but has no GT of cat 2
        // → not_exhaustive ⊄ pos.
        const BAD: &str = r#"{
            "images": [
                {"id": 1, "width": 10, "height": 10,
                 "neg_category_ids": [], "not_exhaustive_category_ids": [2]}
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1,
                 "bbox": [0, 0, 5, 5], "area": 25, "iscrowd": 0}
            ],
            "categories": [
                {"id": 1, "name": "a", "frequency": "f"},
                {"id": 2, "name": "b", "frequency": "r"}
            ]
        }"#;
        let err = CocoDataset::from_lvis_json_bytes(BAD.as_bytes()).unwrap_err();
        match err {
            EvalError::LvisFederatedConflict {
                image_id,
                category_id,
                detail,
            } => {
                assert_eq!(image_id, 1);
                assert_eq!(category_id, 2);
                assert!(detail.contains("not_exhaustive"));
            }
            other => panic!("expected LvisFederatedConflict, got {other:?}"),
        }
    }

    #[test]
    fn ab6_missing_frequency_collects_all_offenders() {
        // Two categories, neither has a frequency. The error must
        // surface both ids in sorted order, not just the first miss.
        const BAD: &str = r#"{
            "images": [
                {"id": 1, "width": 10, "height": 10,
                 "neg_category_ids": [], "not_exhaustive_category_ids": []}
            ],
            "annotations": [],
            "categories": [
                {"id": 7, "name": "g"},
                {"id": 3, "name": "c"}
            ]
        }"#;
        let err = CocoDataset::from_lvis_json_bytes(BAD.as_bytes()).unwrap_err();
        match err {
            EvalError::MissingFrequency { category_ids } => {
                assert_eq!(category_ids, vec![3, 7]);
            }
            other => panic!("expected MissingFrequency, got {other:?}"),
        }
    }

    #[test]
    fn lvis_loader_treats_absent_neg_field_as_empty() {
        // LVIS schema requires neg/not_exhaustive on every image, but a
        // tolerant loader treats absence as empty (matches the LVIS v1
        // semantic where a missing field → no negatives).
        const TOLERANT: &str = r#"{
            "images": [{"id": 1, "width": 10, "height": 10}],
            "annotations": [],
            "categories": [{"id": 1, "name": "a", "frequency": "c"}]
        }"#;
        let ds = CocoDataset::from_lvis_json_bytes(TOLERANT.as_bytes()).unwrap();
        let neg = ds.neg_category_ids().unwrap();
        let nel = ds.not_exhaustive_category_ids().unwrap();
        assert!(neg[&ImageId(1)].is_empty());
        assert!(nel[&ImageId(1)].is_empty());
    }

    #[test]
    fn frequency_round_trips_serde() {
        for f in [Frequency::Rare, Frequency::Common, Frequency::Frequent] {
            let s = serde_json::to_string(&f).unwrap();
            let back: Frequency = serde_json::from_str(&s).unwrap();
            assert_eq!(f, back);
        }
        // Confirm the serde rename targets the LVIS single-letter form.
        assert_eq!(serde_json::to_string(&Frequency::Rare).unwrap(), "\"r\"");
        assert_eq!(serde_json::to_string(&Frequency::Common).unwrap(), "\"c\"");
        assert_eq!(
            serde_json::to_string(&Frequency::Frequent).unwrap(),
            "\"f\""
        );
    }

    #[test]
    fn lvis_loader_inherits_invalid_annotation_validation() {
        // Annotation references unknown image — the spine validation
        // (J5 / AG1) must fire before AA7.
        const BAD: &str = r#"{
            "images": [
                {"id": 1, "width": 10, "height": 10,
                 "neg_category_ids": [], "not_exhaustive_category_ids": []}
            ],
            "annotations": [
                {"id": 1, "image_id": 99, "category_id": 1,
                 "bbox": [0, 0, 1, 1], "area": 1, "iscrowd": 0}
            ],
            "categories": [{"id": 1, "name": "a", "frequency": "f"}]
        }"#;
        let err = CocoDataset::from_lvis_json_bytes(BAD.as_bytes()).unwrap_err();
        assert!(matches!(err, EvalError::InvalidAnnotation { .. }));
    }
}
