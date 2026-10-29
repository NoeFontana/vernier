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
//! - **D3** (`strict`): annotations are not mutated mid-evaluation;
//!   the per-call `_ignore` (which combines the dataset flag with the
//!   current area range) is computed at eval time.
//! - **J3** (`strict`): detection-side area is derived at construction
//!   from the bbox (`bbox.w * bbox.h`) and never read from JSON — the
//!   `loadRes` surface. [`DetectionArea::Supplied`] opts into reading a
//!   supplied `area` instead, the `COCOeval` surface, which takes
//!   `d['area']` off an already-built `cocoDt`.
//! - **J1** (`strict`): user-supplied DT ids are preserved verbatim;
//!   absent ids are auto-assigned sequentially during construction.
//! - **E2 / J4** (`strict`): detections never carry an `iscrowd` flag
//!   — the type does not have the field. JSON inputs that include
//!   `iscrowd=1` are silently dropped, matching pycocotools' overwrite.

use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::sync::{Arc, OnceLock};

use serde::{Deserialize, Serialize};

use crate::error::EvalError;
use crate::parity::ParityMode;
use crate::segmentation::{Segmentation, SegmentationRleCounts};

/// Compressed-sparse-row (CSR) index mapping keys `K` to contiguous
/// `&[usize]` row-index slices, in one flat allocation (`indices`) plus a
/// key → row-range map, instead of one heap `Vec<usize>` per distinct key.
///
/// Rows within a group are ascending (the push order of the per-key `Vec`s
/// it replaces). A lookup is one hash probe and one slice. Keys are
/// user-controlled ids, so the map is seeded per instance
/// (`foldhash::fast::RandomState`): an unkeyed hash would let a crafted
/// results file collide every key and make the build quadratic.
///
/// Wrapped in an `Arc` so cloning `CocoDataset` or `CocoDetections` is
/// `O(1)`. Built in two linear passes; pass 1 keeps a one-entry
/// `(last_key, last_slot)` cache so contiguous per-image runs skip the
/// hash, pass 2 does no hashing.
#[derive(Debug)]
struct CsrIndexInner<K> {
    /// Key → `start..end` into `indices`.
    ranges: HashMap<K, (usize, usize), foldhash::fast::RandomState>,
    indices: Vec<usize>,
    max_group_len: usize,
}

#[derive(Debug)]
pub(crate) struct CsrIndex<K> {
    inner: Arc<CsrIndexInner<K>>,
}

impl<K> Clone for CsrIndex<K> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<K: Copy + Eq + std::hash::Hash> CsrIndex<K> {
    /// Index rows `0..len` by `key_at(row)`. `group_capacity_hint` sizes
    /// the key map up front; an over-estimate is trimmed after the build.
    pub(crate) fn build(
        len: usize,
        group_capacity_hint: usize,
        mut key_at: impl FnMut(usize) -> K,
    ) -> Self {
        // Pass 1: assign each distinct key a dense slot (stored in the
        // range map's `.0` until the prefix sum turns it into a range)
        // and count its rows.
        let mut ranges: HashMap<K, (usize, usize), foldhash::fast::RandomState> =
            HashMap::with_capacity_and_hasher(group_capacity_hint, Default::default());
        let mut counts: Vec<usize> = Vec::with_capacity(group_capacity_hint);
        let mut row_slots: Vec<usize> = Vec::with_capacity(len);
        let mut last: Option<(K, usize)> = None;
        for row in 0..len {
            let key = key_at(row);
            let slot = match last {
                Some((prev_key, prev_slot)) if prev_key == key => prev_slot,
                _ => {
                    let next_slot = counts.len();
                    let slot = ranges.entry(key).or_insert((next_slot, 0)).0;
                    if slot == next_slot {
                        counts.push(0);
                    }
                    last = Some((key, slot));
                    slot
                }
            };
            counts[slot] += 1;
            row_slots.push(slot);
        }

        // Prefix sum into per-slot start offsets; ranges follow from them.
        let mut heads: Vec<usize> = Vec::with_capacity(counts.len());
        let mut start = 0;
        for &n in &counts {
            heads.push(start);
            start += n;
        }
        let max_group_len = counts.iter().copied().max().unwrap_or(0);
        for range in ranges.values_mut() {
            let slot = range.0;
            *range = (heads[slot], heads[slot] + counts[slot]);
        }

        // Pass 2: scatter rows in ascending order behind each slot's head.
        let mut indices = vec![0usize; len];
        for (row, &slot) in row_slots.iter().enumerate() {
            indices[heads[slot]] = row;
            heads[slot] += 1;
        }
        // The hint is sized for the common shape; trim a gross
        // over-estimate (one image, one category) rather than hold it for
        // the life of the dataset.
        if ranges.capacity() > 2 * ranges.len() + 16 {
            ranges.shrink_to_fit();
        }

        Self {
            inner: Arc::new(CsrIndexInner {
                ranges,
                indices,
                max_group_len,
            }),
        }
    }

    #[inline]
    pub(crate) fn get(&self, key: &K) -> &[usize] {
        match self.inner.ranges.get(key) {
            Some(&(start, end)) => &self.inner.indices[start..end],
            None => &[],
        }
    }

    #[inline]
    pub(crate) fn max_group_len(&self) -> usize {
        self.inner.max_group_len
    }

    /// Distinct keys, in unspecified (per-instance) order.
    #[inline]
    pub(crate) fn keys(&self) -> impl ExactSizeIterator<Item = &K> + '_ {
        self.inner.ranges.keys()
    }
}

/// Parse vs `from_parts` split for COCO GT / DT loaders, gated on `bench-timings`.
#[cfg(feature = "bench-timings")]
pub(crate) mod dataset_timings {
    use crate::bench_counters::BenchCounterSet;

    pub(super) const GT_PARSE_NS: usize = 0;
    pub(super) const GT_FROM_PARTS_NS: usize = 1;
    pub(super) const DT_PARSE_NS: usize = 2;
    pub(super) const DT_FROM_INPUTS_NS: usize = 3;

    pub(super) static COUNTERS: BenchCounterSet<4> = BenchCounterSet::new();

    pub(crate) fn read_and_reset() -> (u64, u64, u64, u64) {
        let [a, b, c, d] = COUNTERS.read_and_reset();
        (a, b, c, d)
    }
}

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

impl Frequency {
    /// LVIS single-letter form (`"r"` / `"c"` / `"f"`). Mirrors the
    /// `serde(rename = ...)` tags on the variants — same canonical
    /// form the JSON schema uses, available without going through
    /// serde for places (FFI, log lines) that just need the string.
    pub const fn as_letter(self) -> &'static str {
        match self {
            Self::Rare => "r",
            Self::Common => "c",
            Self::Frequent => "f",
        }
    }
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

/// LVIS federated metadata bundle (ADR-0026). Carried as a single
/// `Option` on [`CocoDataset`] because the four fields are all
/// populated together by [`CocoDataset::from_lvis_json_bytes`] and
/// all `None` after the COCO loader path. Storing one optional
/// struct (rather than four separate `Option<...>` fields) reflects
/// the all-or-none semantics and lets the orchestrator gate
/// federated branches on a single `is_some()` check.
#[derive(Debug, Clone)]
pub struct FederatedMetadata {
    /// Per-image positive-category set, derived from GT annotations
    /// at load (quirk **AA1**, not a JSON field).
    pub pos_category_ids: HashMap<ImageId, HashSet<CategoryId>>,
    /// Per-image negative-category set, read verbatim from the JSON
    /// (quirk **AA2**).
    pub neg_category_ids: HashMap<ImageId, HashSet<CategoryId>>,
    /// Per-image not-exhaustive-category set, read verbatim from the
    /// JSON (quirk **AA3**).
    pub not_exhaustive_category_ids: HashMap<ImageId, HashSet<CategoryId>>,
    /// Per-category frequency tag (quirk **AB1**). Required on every
    /// category by `from_lvis_json_bytes`; missing entries raise
    /// [`EvalError::MissingFrequency`] at load (quirk **AB6**
    /// corrected).
    pub category_frequency: HashMap<CategoryId, Frequency>,
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
/// `federated` is `Some` exactly when the dataset was loaded via
/// [`CocoDataset::from_lvis_json_bytes`]. The orchestrator's
/// federated branches gate on `federated.is_some()`; absence is the
/// COCO default, where the matching engine runs unchanged.
#[derive(Debug, Clone)]
pub struct CocoDataset {
    images: Arc<Vec<ImageMeta>>,
    categories: Arc<Vec<CategoryMeta>>,
    annotations: Arc<Vec<CocoAnnotation>>,
    by_image: CsrIndex<ImageId>,
    by_category: CsrIndex<CategoryId>,
    by_image_cat: CsrIndex<(ImageId, CategoryId)>,
    federated: Option<FederatedMetadata>,
    /// 32-byte BLAKE3 fingerprint of the dataset's canonical form.
    /// Cached lazily on first call to [`Self::dataset_hash`]; carried
    /// in distributed-eval partial headers (ADR-0031). Wrapped in
    /// `Arc<OnceLock>` so cheap clones share the same cache, matching
    /// the existing Arc-shared layout for `images` / `categories` /
    /// `annotations`.
    cached_hash: Arc<OnceLock<[u8; 32]>>,
    /// Positions into `images` in id-ascending order: the `I` axis of
    /// every evaluation grid. Built on first use and shared by clones, so
    /// repeated evaluations (and every [`crate::stream::StreamingEvaluator`]
    /// `update`) do not re-sort the whole image list.
    cached_image_order: Arc<OnceLock<Box<[usize]>>>,
}

/// The images of a [`CocoDataset`] in grid order (id-ascending) — the
/// one definition of the `I` axis every evaluator, table builder and
/// diagnostic indexes. A borrowed view: no per-call gather or sort.
#[derive(Clone, Copy)]
pub(crate) struct GridImages<'a> {
    metas: &'a [ImageMeta],
    order: &'a [usize],
}

impl<'a> GridImages<'a> {
    /// Number of images, `I`.
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.order.len()
    }

    /// Image at grid index `i`.
    #[inline]
    pub(crate) fn get(&self, i: usize) -> &'a ImageMeta {
        &self.metas[self.order[i]]
    }

    /// Images in grid order.
    pub(crate) fn iter(&self) -> impl ExactSizeIterator<Item = &'a ImageMeta> + Clone + 'a {
        let metas = self.metas;
        self.order.iter().map(move |&p| &metas[p])
    }
}

impl CocoDataset {
    /// Loads a dataset from a JSON byte slice.
    ///
    /// Validates that every annotation references a known image and a
    /// known category; missing references raise [`EvalError::InvalidAnnotation`]
    /// rather than producing a silently-empty dataset.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, EvalError> {
        #[cfg(feature = "bench-timings")]
        let t0 = std::time::Instant::now();
        let raw: CocoJson = serde_json::from_slice(bytes)?;
        #[cfg(feature = "bench-timings")]
        let parse_ns = u64::try_from(t0.elapsed().as_nanos()).unwrap_or(u64::MAX);
        #[cfg(feature = "bench-timings")]
        let t1 = std::time::Instant::now();
        let result = Self::from_parts(raw.images, raw.annotations, raw.categories);
        #[cfg(feature = "bench-timings")]
        {
            let from_parts_ns = u64::try_from(t1.elapsed().as_nanos()).unwrap_or(u64::MAX);
            dataset_timings::COUNTERS.add(dataset_timings::GT_PARSE_NS, parse_ns);
            dataset_timings::COUNTERS.add(dataset_timings::GT_FROM_PARTS_NS, from_parts_ns);
        }
        result
    }

    /// Parallel sibling of [`Self::from_json_bytes`] (ADR-0054).
    ///
    /// Splits the three top-level arrays into byte ranges and hands
    /// each range to the same `serde_json` element deserializers the
    /// serial loader uses, so the parsed values are bit-identical and
    /// input order is preserved. Anything the splitter does not plainly
    /// recognize — a different document shape, a duplicate key, a parse
    /// error — falls back to [`Self::from_json_bytes`], which is also
    /// what reports the canonical error for malformed input.
    ///
    /// `threads` is the caller's budget. Per ADR-0047 the
    /// `num_threads=None` path never calls this; it stays on
    /// [`Self::from_json_bytes`] and never enters rayon. A budget below
    /// `MIN_THREADS_TO_SPLIT` also stays serial — the structural scan is
    /// only worth its cost once there are threads to spend it on.
    ///
    /// # Errors
    ///
    /// Same as [`Self::from_json_bytes`].
    pub fn from_json_bytes_parallel(bytes: &[u8], threads: usize) -> Result<Self, EvalError> {
        if threads < crate::json_split::MIN_THREADS_TO_SPLIT {
            return Self::from_json_bytes(bytes);
        }
        #[cfg(feature = "bench-timings")]
        let t0 = std::time::Instant::now();
        let parts = Self::split_parse(bytes, threads);
        #[cfg(feature = "bench-timings")]
        let parse_ns = u64::try_from(t0.elapsed().as_nanos()).unwrap_or(u64::MAX);
        match parts {
            Some((images, annotations, categories)) => {
                #[cfg(feature = "bench-timings")]
                let t1 = std::time::Instant::now();
                let result = Self::from_parts(images, annotations, categories);
                #[cfg(feature = "bench-timings")]
                {
                    let from_parts_ns = u64::try_from(t1.elapsed().as_nanos()).unwrap_or(u64::MAX);
                    dataset_timings::COUNTERS.add(dataset_timings::GT_PARSE_NS, parse_ns);
                    dataset_timings::COUNTERS.add(dataset_timings::GT_FROM_PARTS_NS, from_parts_ns);
                }
                result
            }
            // The fallback records nothing of its own: `from_json_bytes`
            // owns both counters, and adding the abandoned scan to
            // `GT_PARSE_NS` here would count the same phase twice.
            None => Self::from_json_bytes(bytes),
        }
    }

    /// The parallel parse itself. `None` means "fall back".
    fn split_parse(
        bytes: &[u8],
        threads: usize,
    ) -> Option<(Vec<ImageMeta>, Vec<CocoAnnotation>, Vec<CategoryMeta>)> {
        let arrays = crate::json_split::top_level_object_arrays(
            bytes,
            &["images", "annotations", "categories"],
        )?;
        let [images, annotations, categories] = <[Vec<Range<usize>>; 3]>::try_from(arrays).ok()?;
        let n = crate::json_split::chunks_for(threads);
        Some((
            crate::json_split::parse_elements_parallel(bytes, &images, n).ok()?,
            crate::json_split::parse_elements_parallel(bytes, &annotations, n).ok()?,
            crate::json_split::parse_elements_parallel(bytes, &categories, n).ok()?,
        ))
    }

    /// Loads a dataset from already-typed parts.
    pub fn from_parts(
        images: Vec<ImageMeta>,
        annotations: Vec<CocoAnnotation>,
        categories: Vec<CategoryMeta>,
    ) -> Result<Self, EvalError> {
        let n_anns = annotations.len();
        let by_image = CsrIndex::build(n_anns, images.len(), |idx| annotations[idx].image_id);
        let by_category =
            CsrIndex::build(n_anns, categories.len(), |idx| annotations[idx].category_id);

        // Every annotation must name a known image and category. Checked
        // over the indexes' distinct keys — `O(images + categories)`
        // probes rather than two per annotation — and reported for the
        // first offending annotation in input order, image before
        // category, exactly as a per-annotation scan would.
        let known_images: HashSet<ImageId> = images.iter().map(|i| i.id).collect();
        let known_categories: HashSet<CategoryId> = categories.iter().map(|c| c.id).collect();
        let first_row = |rows: &[usize]| rows.first().copied().unwrap_or(usize::MAX);
        let bad_image = by_image
            .keys()
            .filter(|id| !known_images.contains(id))
            .map(|id| first_row(by_image.get(id)))
            .min();
        let bad_category = by_category
            .keys()
            .filter(|id| !known_categories.contains(id))
            .map(|id| first_row(by_category.get(id)))
            .min();
        if let Some(row) = bad_image.into_iter().chain(bad_category).min() {
            let ann = &annotations[row];
            let (field, value) = if known_images.contains(&ann.image_id) {
                ("category_id", ann.category_id.0)
            } else {
                ("image_id", ann.image_id.0)
            };
            return Err(EvalError::InvalidAnnotation {
                detail: format!(
                    "annotation id={} references unknown {field}={value}",
                    ann.id.0
                ),
            });
        }

        let by_image_cat = CsrIndex::build(n_anns, n_anns / 2, |idx| {
            (annotations[idx].image_id, annotations[idx].category_id)
        });

        Ok(Self {
            images: Arc::new(images),
            categories: Arc::new(categories),
            annotations: Arc::new(annotations),
            by_image,
            by_category,
            by_image_cat,
            federated: None,
            cached_hash: Arc::new(OnceLock::new()),
            cached_image_order: Arc::new(OnceLock::new()),
        })
    }

    /// The images in grid order (id-ascending), computed once per dataset.
    ///
    /// Stable by id, so a (malformed) repeated image id keeps declaration
    /// order; every grid consumer reads this one order, so their `I` axes
    /// agree whatever the input.
    pub(crate) fn grid_images(&self) -> GridImages<'_> {
        let order = self.cached_image_order.get_or_init(|| {
            let mut order: Vec<usize> = (0..self.images.len()).collect();
            order.sort_by_key(|&p| self.images[p].id.0);
            order.into_boxed_slice()
        });
        GridImages {
            metas: &self.images,
            order,
        }
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
        Self::from_lvis_parts(serde_json::from_slice(bytes)?)
    }

    /// Parallel sibling of [`Self::from_lvis_json_bytes`] (ADR-0054).
    ///
    /// Same split-and-deserialize path as
    /// [`Self::from_json_bytes_parallel`]; LVIS JSON is structurally
    /// COCO JSON with the same three top-level arrays, only richer
    /// element types. Falls back to the serial loader on anything the
    /// splitter does not recognize.
    ///
    /// # Errors
    ///
    /// Same as [`Self::from_lvis_json_bytes`].
    pub fn from_lvis_json_bytes_parallel(bytes: &[u8], threads: usize) -> Result<Self, EvalError> {
        if threads < crate::json_split::MIN_THREADS_TO_SPLIT {
            return Self::from_lvis_json_bytes(bytes);
        }
        match Self::split_parse_lvis(bytes, threads) {
            Some(raw) => Self::from_lvis_parts(raw),
            None => Self::from_lvis_json_bytes(bytes),
        }
    }

    /// The parallel LVIS parse. `None` means "fall back".
    fn split_parse_lvis(bytes: &[u8], threads: usize) -> Option<LvisJson> {
        let arrays = crate::json_split::top_level_object_arrays(
            bytes,
            &["images", "annotations", "categories"],
        )?;
        let [images, annotations, categories] = <[Vec<Range<usize>>; 3]>::try_from(arrays).ok()?;
        let n = crate::json_split::chunks_for(threads);
        Some(LvisJson {
            images: crate::json_split::parse_elements_parallel(bytes, &images, n).ok()?,
            annotations: crate::json_split::parse_elements_parallel(bytes, &annotations, n).ok()?,
            categories: crate::json_split::parse_elements_parallel(bytes, &categories, n).ok()?,
        })
    }

    /// Build an LVIS dataset from already-parsed JSON parts. Shared by
    /// the serial and parallel loaders so the federated-metadata
    /// derivation has exactly one implementation.
    fn from_lvis_parts(raw: LvisJson) -> Result<Self, EvalError> {
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

        dataset.federated = Some(FederatedMetadata {
            pos_category_ids: pos,
            neg_category_ids: neg,
            not_exhaustive_category_ids: nel,
            category_frequency,
        });
        Ok(dataset)
    }

    /// LVIS federated metadata bundle. `Some` only when the dataset
    /// was built by [`Self::from_lvis_json_bytes`]; the orchestrator's
    /// AA3/AA4 branches gate on this.
    pub fn federated(&self) -> Option<&FederatedMetadata> {
        self.federated.as_ref()
    }

    /// Per-image positive-category set, derived from GTs at load time
    /// (quirk **AA1**). `Some` only when the dataset is federated.
    pub fn pos_category_ids(&self) -> Option<&HashMap<ImageId, HashSet<CategoryId>>> {
        self.federated.as_ref().map(|f| &f.pos_category_ids)
    }

    /// Per-image negative-category set, read verbatim from the LVIS
    /// JSON (quirk **AA2**). `Some` only when the dataset is federated.
    pub fn neg_category_ids(&self) -> Option<&HashMap<ImageId, HashSet<CategoryId>>> {
        self.federated.as_ref().map(|f| &f.neg_category_ids)
    }

    /// Per-image not-exhaustive-category set, read verbatim from the
    /// LVIS JSON (quirk **AA3**). `Some` only when the dataset is
    /// federated.
    pub fn not_exhaustive_category_ids(&self) -> Option<&HashMap<ImageId, HashSet<CategoryId>>> {
        self.federated
            .as_ref()
            .map(|f| &f.not_exhaustive_category_ids)
    }

    /// Per-category frequency tag, read verbatim from the LVIS JSON
    /// (quirk **AB1**). `Some` only when the dataset is federated;
    /// missing-on-some-categories inputs are rejected at load
    /// (quirk **AB6**).
    pub fn category_frequency(&self) -> Option<&HashMap<CategoryId, Frequency>> {
        self.federated.as_ref().map(|f| &f.category_frequency)
    }

    /// `true` when the dataset carries LVIS federated metadata.
    /// Cheap shortcut for orchestration code that gates behaviour on
    /// the federated flag.
    pub fn is_federated(&self) -> bool {
        self.federated.is_some()
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
        self.by_image.get(&image_id)
    }

    fn ann_indices_for_category(&self, cat_id: CategoryId) -> &[usize] {
        self.by_category.get(&cat_id)
    }
}

impl CocoDataset {
    /// Indices into [`Self::annotations`] for a given `(image, category)`
    /// cell. Empty when no GT of that category exists on that image.
    pub fn ann_indices_for(&self, image: ImageId, cat: CategoryId) -> &[usize] {
        self.by_image_cat.get(&(image, cat))
    }
}

// ---------------------------------------------------------------------------
// dataset_hash — canonical-form fingerprint for ADR-0031 partials.
//
// The hash is the BLAKE3 digest of a deterministic byte stream built
// from the dataset's images + categories + annotations + federated
// metadata. Independent of input order: each section is sorted by id
// before hashing. The canonical form is the load-bearing wire-format
// invariant that makes "this partial was computed against the same GT
// I have" a strict, refusable check; format_version bumps when the
// canonical form changes (per ADR-0031 §"Wire format" backward-compat
// rules).
//
// Domain separators (4-byte ASCII tags) precede each section so a
// rearranged stream cannot collide with the canonical one.
// ---------------------------------------------------------------------------

/// Domain-separated section tag for the canonical-form stream.
const HASH_TAG_DATASET: &[u8; 4] = b"DSET";
const HASH_TAG_IMAGES: &[u8; 4] = b"IMGS";
const HASH_TAG_CATEGORIES: &[u8; 4] = b"CATS";
const HASH_TAG_ANNOTATIONS: &[u8; 4] = b"ANNS";
const HASH_TAG_FEDERATED: &[u8; 4] = b"FEDM";

/// Bumped when the canonical-form layout changes. Read into the
/// stream once, before any section, so a v1 hash can never collide
/// with a v2 hash even on identical underlying data.
const HASH_CANONICAL_VERSION: u8 = 1;

#[inline]
fn hash_u8(h: &mut blake3::Hasher, v: u8) {
    h.update(&[v]);
}
#[inline]
fn hash_u32(h: &mut blake3::Hasher, v: u32) {
    h.update(&v.to_le_bytes());
}
#[inline]
fn hash_i64(h: &mut blake3::Hasher, v: i64) {
    h.update(&v.to_le_bytes());
}
#[inline]
fn hash_u64(h: &mut blake3::Hasher, v: u64) {
    h.update(&v.to_le_bytes());
}
#[inline]
fn hash_f64(h: &mut blake3::Hasher, v: f64) {
    // Bit-exact representation; canonical for finite values. NaN
    // payloads matter (two NaNs with different bits hash differently);
    // the dataset loader rejects non-finite area / bbox / keypoints
    // upstream so the surface here is f64s the user actually trusts.
    h.update(&v.to_bits().to_le_bytes());
}
#[inline]
fn hash_bool(h: &mut blake3::Hasher, v: bool) {
    hash_u8(h, u8::from(v));
}
#[inline]
fn hash_bytes(h: &mut blake3::Hasher, bytes: &[u8]) {
    hash_u64(h, bytes.len() as u64);
    h.update(bytes);
}
#[inline]
fn hash_string(h: &mut blake3::Hasher, s: &str) {
    hash_bytes(h, s.as_bytes());
}
#[inline]
fn hash_option<T>(
    h: &mut blake3::Hasher,
    opt: Option<T>,
    write: impl FnOnce(&mut blake3::Hasher, T),
) {
    match opt {
        None => hash_u8(h, 0),
        Some(v) => {
            hash_u8(h, 1);
            write(h, v);
        }
    }
}

fn hash_bbox(h: &mut blake3::Hasher, b: &Bbox) {
    hash_f64(h, b.x);
    hash_f64(h, b.y);
    hash_f64(h, b.w);
    hash_f64(h, b.h);
}

fn hash_segmentation(h: &mut blake3::Hasher, seg: Option<&Segmentation>) {
    match seg {
        None => hash_u8(h, 0),
        Some(Segmentation::Polygons(polys)) => {
            hash_u8(h, 1);
            hash_u64(h, polys.len() as u64);
            for poly in polys {
                hash_u64(h, poly.len() as u64);
                for &v in poly {
                    hash_f64(h, v);
                }
            }
        }
        Some(Segmentation::Rle(rle)) => {
            let [rh, rw] = rle.size;
            match &rle.counts {
                SegmentationRleCounts::Compressed(s) => {
                    hash_u8(h, 2);
                    hash_u32(h, rh);
                    hash_u32(h, rw);
                    hash_string(h, s);
                }
                SegmentationRleCounts::Uncompressed(counts) => {
                    hash_u8(h, 3);
                    hash_u32(h, rh);
                    hash_u32(h, rw);
                    hash_u64(h, counts.len() as u64);
                    for &c in counts.iter() {
                        hash_u32(h, c);
                    }
                }
            }
        }
    }
}

/// Walk a slice in id-sorted order, prefixed by a domain tag and the
/// element count, hashing each element via `write`. The id projection
/// (`key`) returns the i64 id; sort is by that key, unstable (ids are
/// unique by construction). Avoids materializing a fresh sorted Vec
/// of items by sorting an index permutation instead.
fn hash_id_sorted<T>(
    h: &mut blake3::Hasher,
    tag: &[u8; 4],
    items: &[T],
    key: impl Fn(&T) -> i64,
    write: impl Fn(&mut blake3::Hasher, &T),
) {
    h.update(tag);
    let mut order: Vec<usize> = (0..items.len()).collect();
    order.sort_unstable_by_key(|&i| key(&items[i]));
    hash_u64(h, order.len() as u64);
    for &i in &order {
        write(h, &items[i]);
    }
}

fn hash_image_meta(h: &mut blake3::Hasher, im: &ImageMeta) {
    let ImageMeta {
        id,
        width,
        height,
        file_name,
    } = im;
    hash_i64(h, id.0);
    hash_u32(h, *width);
    hash_u32(h, *height);
    hash_option(h, file_name.as_deref(), hash_string);
}

fn hash_category_meta(h: &mut blake3::Hasher, c: &CategoryMeta) {
    let CategoryMeta {
        id,
        name,
        supercategory,
    } = c;
    hash_i64(h, id.0);
    hash_string(h, name);
    hash_option(h, supercategory.as_deref(), hash_string);
}

fn hash_coco_annotation(h: &mut blake3::Hasher, a: &CocoAnnotation) {
    // Exhaustive destructure: adding a field to CocoAnnotation is a
    // compile error here, forcing the canonical form to stay in sync.
    let CocoAnnotation {
        id,
        image_id,
        category_id,
        area,
        is_crowd,
        ignore_flag,
        bbox,
        segmentation,
        keypoints,
        num_keypoints,
    } = a;
    hash_i64(h, id.0);
    hash_i64(h, image_id.0);
    hash_i64(h, category_id.0);
    hash_f64(h, *area);
    hash_bool(h, *is_crowd);
    hash_option(h, *ignore_flag, hash_bool);
    hash_bbox(h, bbox);
    hash_segmentation(h, segmentation.as_ref());
    hash_option(h, keypoints.as_deref(), |h, kps| {
        hash_u64(h, kps.len() as u64);
        for &v in kps {
            hash_f64(h, v);
        }
    });
    hash_option(h, *num_keypoints, hash_u32);
}

fn hash_federated(h: &mut blake3::Hasher, fed: &FederatedMetadata) {
    h.update(HASH_TAG_FEDERATED);

    // category_frequency: sort by category id, write (id, letter byte).
    let mut freq_pairs: Vec<(i64, &Frequency)> = fed
        .category_frequency
        .iter()
        .map(|(k, v)| (k.0, v))
        .collect();
    freq_pairs.sort_unstable_by_key(|(k, _)| *k);
    hash_u64(h, freq_pairs.len() as u64);
    for (cid, freq) in freq_pairs {
        hash_i64(h, cid);
        // `as_letter` is a single ASCII char; one byte is enough.
        hash_u8(h, freq.as_letter().as_bytes()[0]);
    }

    // pos / neg / not_exhaustive: each is HashMap<ImageId, HashSet<CategoryId>>.
    // Hash all three sections via the same canonical form: sort by image id,
    // then for each image sort the category ids ascending and write count + ids.
    type FedSection<'a> = (&'a [u8; 3], &'a HashMap<ImageId, HashSet<CategoryId>>);
    let sections: [FedSection<'_>; 3] = [
        (b"POS", &fed.pos_category_ids),
        (b"NEG", &fed.neg_category_ids),
        (b"NEX", &fed.not_exhaustive_category_ids),
    ];
    for (tag, map) in sections {
        h.update(tag);
        let mut entries: Vec<(i64, Vec<i64>)> = map
            .iter()
            .map(|(image_id, cats)| {
                let mut cat_ids: Vec<i64> = cats.iter().map(|c| c.0).collect();
                cat_ids.sort_unstable();
                (image_id.0, cat_ids)
            })
            .collect();
        entries.sort_unstable_by_key(|(image_id, _)| *image_id);
        hash_u64(h, entries.len() as u64);
        for (image_id, cat_ids) in entries {
            hash_i64(h, image_id);
            hash_u64(h, cat_ids.len() as u64);
            for cid in cat_ids {
                hash_i64(h, cid);
            }
        }
    }
}

impl CocoDataset {
    /// 32-byte BLAKE3 fingerprint of this dataset's canonical form.
    /// Stable across input orderings: images, categories, annotations
    /// are sorted by id before hashing. Lazily cached on first call;
    /// shared across [`Clone`]s via the underlying `Arc<OnceLock>`.
    ///
    /// Carried in distributed-eval partial headers (ADR-0031); a
    /// receiving rank refuses to merge partials whose `dataset_hash`
    /// disagrees with its live dataset's.
    pub fn dataset_hash(&self) -> [u8; 32] {
        *self.cached_hash.get_or_init(|| self.compute_dataset_hash())
    }

    fn compute_dataset_hash(&self) -> [u8; 32] {
        let mut h = blake3::Hasher::new();
        h.update(HASH_TAG_DATASET);
        hash_u8(&mut h, HASH_CANONICAL_VERSION);

        hash_id_sorted(
            &mut h,
            HASH_TAG_IMAGES,
            &self.images,
            |im| im.id.0,
            hash_image_meta,
        );
        hash_id_sorted(
            &mut h,
            HASH_TAG_CATEGORIES,
            &self.categories,
            |c| c.id.0,
            hash_category_meta,
        );
        hash_id_sorted(
            &mut h,
            HASH_TAG_ANNOTATIONS,
            &self.annotations,
            |a| a.id.0,
            hash_coco_annotation,
        );

        // Federated metadata, when present (LVIS path).
        match self.federated.as_ref() {
            None => hash_u8(&mut h, 0),
            Some(fed) => {
                hash_u8(&mut h, 1);
                hash_federated(&mut h, fed);
            }
        }

        *h.finalize().as_bytes()
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
///   quirk **J3** — unless built with [`DetectionArea::Supplied`].
/// - `id` is honored when the user supplies one and auto-assigned
///   otherwise — quirk **J1** (`strict`: pycocotools-shaped outputs are
///   bit-equal; preserving user ids is a structural difference, not a
///   behavioral divergence).
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
    /// Pixel area, derived from `bbox` per quirk **J3** unless built with
    /// [`DetectionArea::Supplied`].
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
    /// Optional supplied area. Ignored unless the detections are built
    /// with [`DetectionArea::Supplied`] (quirk **J3**).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub area: Option<f64>,
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

/// Where [`CocoDetection::area`] comes from (quirk **J3**).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DetectionArea {
    /// `bbox.w * bbox.h`, ignoring any supplied area — pycocotools'
    /// `loadRes`, which overwrites `area` on every result.
    #[default]
    FromBbox,
    /// [`DetectionInput::area`] verbatim, falling back to
    /// `bbox.w * bbox.h` when absent — pycocotools' `COCOeval`, which
    /// reads `d['area']` off whatever `cocoDt` it is handed.
    Supplied,
    /// The foreground pixel count of the detection's RLE segmentation —
    /// pycocotools' `maskUtils.area`, which `loadRes` derives for segm
    /// results. A detection without an RLE segmentation is an error.
    Mask,
}

/// Columnar caller-side input for a batch of detections (ADR-0066).
///
/// Columnar sibling of `Vec<DetectionInput>`: required columns
/// (`image_ids`, `category_ids`, `scores`, `bboxes`) are dense arrays
/// of equal length `N`, while optional columns (`ids`, `areas`,
/// `segmentations`, `keypoints`, `num_keypoints`) are `None` when
/// omitted for the whole batch (zero allocation for bbox-only / auto-id
/// / bbox-area callers) or `Some(vec)` of length `N` when supplied.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DetectionColumnsInput {
    /// Optional per-row user-supplied annotation ids (quirk **J1**).
    /// `None` auto-assigns `1..=N` in `O(1)` memory.
    pub ids: Option<Vec<Option<AnnId>>>,
    /// Per-detection image id (`len == N`).
    pub image_ids: Vec<ImageId>,
    /// Per-detection category id (`len == N`).
    pub category_ids: Vec<CategoryId>,
    /// Per-detection confidence score (`len == N`).
    pub scores: Vec<f64>,
    /// Per-detection `[x, y, w, h]` bounding box (`len == N`).
    pub bboxes: Vec<Bbox>,
    /// Optional per-row supplied areas (`len == N` when `Some`; quirk **J3**).
    pub areas: Option<Vec<Option<f64>>>,
    /// Optional per-row segmentation predictions (`len == N` when `Some`).
    pub segmentations: Option<Vec<Option<Segmentation>>>,
    /// Optional per-row keypoint triplets (`len == N` when `Some`).
    pub keypoints: Option<Vec<Option<Vec<f64>>>>,
    /// Optional per-row visible-keypoint counts (`len == N` when `Some`).
    pub num_keypoints: Option<Vec<Option<u32>>>,
}

impl DetectionColumnsInput {
    /// Empty columns with room for `n` rows.
    pub fn with_capacity(n: usize) -> Self {
        Self {
            image_ids: Vec::with_capacity(n),
            category_ids: Vec::with_capacity(n),
            scores: Vec::with_capacity(n),
            bboxes: Vec::with_capacity(n),
            ..Self::default()
        }
    }

    /// Append one row.
    ///
    /// The optional fields stay sparse: an optional column is not
    /// allocated until a row first supplies a value, and earlier rows are
    /// then back-filled with `None` — so a bbox-only, auto-id batch never
    /// allocates them at all. The one owner of that rule for every row-wise
    /// producer (typed inputs here, result dicts at the FFI).
    pub fn push(&mut self, row: DetectionInput) {
        let idx = self.image_ids.len();
        let cap = self.image_ids.capacity();
        push_sparse_opt(&mut self.ids, idx, cap, row.id);
        push_sparse_opt(&mut self.areas, idx, cap, row.area);
        push_sparse_opt(&mut self.segmentations, idx, cap, row.segmentation);
        push_sparse_opt(&mut self.keypoints, idx, cap, row.keypoints);
        push_sparse_opt(&mut self.num_keypoints, idx, cap, row.num_keypoints);
        self.image_ids.push(row.image_id);
        self.category_ids.push(row.category_id);
        self.scores.push(row.score);
        self.bboxes.push(row.bbox);
    }

    /// `parts` joined in order, keeping [`Self::push`]'s sparse rule: an
    /// optional column absent from every part stays unallocated.
    fn concat(parts: Vec<Self>) -> Self {
        let total = parts.iter().map(|part| part.image_ids.len()).sum();
        let mut out = Self::with_capacity(total);
        for part in parts {
            let idx = out.image_ids.len();
            let rows = part.image_ids.len();
            append_sparse(&mut out.ids, idx, rows, total, part.ids);
            append_sparse(&mut out.areas, idx, rows, total, part.areas);
            append_sparse(&mut out.segmentations, idx, rows, total, part.segmentations);
            append_sparse(&mut out.keypoints, idx, rows, total, part.keypoints);
            append_sparse(&mut out.num_keypoints, idx, rows, total, part.num_keypoints);
            out.image_ids.extend(part.image_ids);
            out.category_ids.extend(part.category_ids);
            out.scores.extend(part.scores);
            out.bboxes.extend(part.bboxes);
        }
        out
    }
}

/// Deserializes a JSON array of [`DetectionInput`] straight into
/// [`DetectionColumnsInput`], with no row-major intermediate. A supplied
/// `area` is dropped unless `keep_area`: only a [`DetectionArea::Supplied`]
/// build reads it (quirk **J3**).
#[derive(Clone, Copy)]
struct DetectionColumnsSeed {
    keep_area: bool,
}

impl DetectionColumnsSeed {
    fn push(self, cols: &mut DetectionColumnsInput, mut row: DetectionInput) {
        if !self.keep_area {
            row.area = None;
        }
        cols.push(row);
    }
}

impl<'de> serde::de::DeserializeSeed<'de> for DetectionColumnsSeed {
    type Value = DetectionColumnsInput;

    fn deserialize<D: serde::Deserializer<'de>>(self, de: D) -> Result<Self::Value, D::Error> {
        de.deserialize_seq(self)
    }
}

impl<'de> serde::de::Visitor<'de> for DetectionColumnsSeed {
    type Value = DetectionColumnsInput;

    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `Vec<T>`'s own wording, so error text is unchanged by the switch.
        f.write_str("a sequence")
    }

    fn visit_seq<S: serde::de::SeqAccess<'de>>(self, mut seq: S) -> Result<Self::Value, S::Error> {
        let mut cols = DetectionColumnsInput::with_capacity(seq.size_hint().unwrap_or(0));
        while let Some(row) = seq.next_element()? {
            self.push(&mut cols, row);
        }
        Ok(cols)
    }
}

#[derive(Debug, Clone, PartialEq)]
enum IdColumn {
    /// Auto-assigned sequential ids `start + idx as i64` (`start = 1`
    /// for `from_columns` / `from_inputs`). Zero heap bytes.
    Sequential { start: i64 },
    /// Resolved per-row ids when user-supplied ids are present or after
    /// index subsetting (`lvis_trim`, `from_records`).
    Explicit(Vec<AnnId>),
}

impl IdColumn {
    #[inline]
    fn id_at(&self, idx: usize) -> AnnId {
        match self {
            Self::Sequential { start } => {
                AnnId(start.saturating_add(i64::try_from(idx).unwrap_or(i64::MAX)))
            }
            Self::Explicit(ids) => ids[idx],
        }
    }

    /// Raw ids at `kept` into `out`, replacing its contents. The storage
    /// kind is matched once, so each arm is a branch-free loop.
    fn gather_into(&self, kept: &[usize], out: &mut Vec<i64>) {
        out.clear();
        match self {
            Self::Sequential { start } => out.extend(
                kept.iter()
                    .map(|&idx| start.saturating_add(i64::try_from(idx).unwrap_or(i64::MAX))),
            ),
            Self::Explicit(ids) => out.extend(kept.iter().map(|&idx| ids[idx].0)),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum AreaColumn {
    /// Derived on the fly as `bbox.w * bbox.h` (quirk **J3** default).
    /// Zero heap bytes and bit-identical IEEE-754 product.
    FromBbox,
    /// Pre-resolved per-row areas (`DetectionArea::Supplied` with
    /// explicit areas, `DetectionArea::Mask` RLE pixel counts, or
    /// `from_records`).
    Explicit(Vec<f64>),
}

impl AreaColumn {
    #[inline]
    fn area_at(&self, idx: usize, bboxes: &[Bbox]) -> f64 {
        match self {
            Self::FromBbox => {
                let b = bboxes[idx];
                b.w * b.h
            }
            Self::Explicit(areas) => areas[idx],
        }
    }

    /// [`IdColumn::gather_into`] for areas.
    fn gather_into(&self, kept: &[usize], bboxes: &[Bbox], out: &mut Vec<f64>) {
        out.clear();
        match self {
            Self::FromBbox => out.extend(kept.iter().map(|&idx| {
                let b = bboxes[idx];
                b.w * b.h
            })),
            Self::Explicit(areas) => out.extend(kept.iter().map(|&idx| areas[idx])),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct DetectionColumns {
    ids: IdColumn,
    image_ids: Vec<ImageId>,
    category_ids: Vec<CategoryId>,
    scores: Vec<f64>,
    bboxes: Vec<Bbox>,
    areas: AreaColumn,
    segmentations: Option<Vec<Option<Segmentation>>>,
    keypoints: Option<Vec<Option<Vec<f64>>>>,
    num_keypoints: Option<Vec<Option<u32>>>,
}

impl DetectionColumns {
    fn gather(&self, kept: &[usize]) -> Self {
        Self {
            ids: IdColumn::Explicit(kept.iter().map(|&idx| self.ids.id_at(idx)).collect()),
            image_ids: gather(&self.image_ids, kept),
            category_ids: gather(&self.category_ids, kept),
            scores: gather(&self.scores, kept),
            bboxes: gather(&self.bboxes, kept),
            areas: match &self.areas {
                AreaColumn::FromBbox => AreaColumn::FromBbox,
                AreaColumn::Explicit(src) => AreaColumn::Explicit(gather(src, kept)),
            },
            segmentations: self.segmentations.as_ref().map(|src| gather(src, kept)),
            keypoints: self.keypoints.as_ref().map(|src| gather(src, kept)),
            num_keypoints: self.num_keypoints.as_ref().map(|src| gather(src, kept)),
        }
    }
}

/// `src[kept[0]], src[kept[1]], ...` as a new column.
fn gather<T: Clone>(src: &[T], kept: &[usize]) -> Vec<T> {
    kept.iter().map(|&idx| src[idx].clone()).collect()
}

/// [`push_sparse_opt`] for a run of `rows` rows starting at `idx`.
fn append_sparse<T>(
    slot: &mut Option<Vec<Option<T>>>,
    idx: usize,
    rows: usize,
    total: usize,
    part: Option<Vec<Option<T>>>,
) {
    match (slot, part) {
        (Some(vec), Some(part)) => vec.extend(part),
        (Some(vec), None) => vec.resize_with(idx + rows, || None),
        (slot @ None, Some(part)) => {
            let mut vec = Vec::with_capacity(total);
            vec.resize_with(idx, || None);
            vec.extend(part);
            *slot = Some(vec);
        }
        (None, None) => {}
    }
}

#[inline]
fn push_sparse_opt<T>(slot: &mut Option<Vec<Option<T>>>, idx: usize, total: usize, val: Option<T>) {
    match (slot, val) {
        (Some(vec), v) => vec.push(v),
        (slot @ None, Some(v)) => {
            let mut vec = Vec::with_capacity(total);
            vec.resize_with(idx, || None);
            vec.push(Some(v));
            *slot = Some(vec);
        }
        (None, None) => {}
    }
}

/// [`DetectionArea::Mask`]: the RLE foreground area of a detection.
fn mask_area_from_seg(
    segmentation: Option<&Segmentation>,
    image_id: ImageId,
    id: AnnId,
) -> Result<f64, EvalError> {
    let area = match segmentation {
        Some(seg) => seg.rle_area()?,
        None => None,
    };
    // Exact: a pixel count stays far below 2^53.
    #[allow(clippy::cast_precision_loss)]
    area.map(|pixels| pixels as f64)
        .ok_or_else(|| EvalError::InvalidAnnotation {
            detail: format!(
                "DT id={} on image {}: dt_area=\"mask\" requires an RLE `segmentation` \
                 (maskUtils.area does not take {})",
                id.0,
                image_id.0,
                if segmentation.is_some() {
                    "polygons"
                } else {
                    "a missing segmentation"
                }
            ),
        })
}

fn check_col_len(name: &str, actual: usize, expected: usize) -> Result<(), EvalError> {
    if actual != expected {
        return Err(EvalError::InvalidAnnotation {
            detail: format!(
                "detection column `{name}` length {actual} does not match `scores` length {expected}"
            ),
        });
    }
    Ok(())
}

/// COCO detections collection — pure Struct-of-Arrays (SoA) columnar
/// storage plus CSR `(image, category)`- and per-image indices for the
/// per-cell gather (ADR-0066).
#[derive(Debug, Clone)]
pub struct CocoDetections {
    columns: Arc<DetectionColumns>,
    by_image_cat: CsrIndex<(ImageId, CategoryId)>,
    by_image: CsrIndex<ImageId>,
}

impl CocoDetections {
    fn from_resolved_columns(columns: DetectionColumns) -> Self {
        let len = columns.scores.len();
        let by_image_cat = CsrIndex::build(len, len / 4, |idx| {
            (columns.image_ids[idx], columns.category_ids[idx])
        });
        // ~64 detections per image: COCO-shaped sets cap at 100, so this
        // seeds above the distinct-image count and the key map never grows;
        // a gross over-estimate is trimmed by the index's `shrink_to_fit`.
        let by_image = CsrIndex::build(len, len / 64, |idx| columns.image_ids[idx]);
        Self {
            columns: Arc::new(columns),
            by_image_cat,
            by_image,
        }
    }

    /// Loads detections from the JSON array shape pycocotools'
    /// `loadRes` consumes (a list of objects with `image_id`,
    /// `category_id`, `bbox`, `score`, optional `id`).
    ///
    /// `iscrowd` and `area` fields, if present, are silently dropped:
    /// quirks **E2/J4** force `is_crowd=0` and quirk **J3** derives
    /// `area` from `bbox`.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, EvalError> {
        Self::from_json_bytes_with_area(bytes, DetectionArea::FromBbox)
    }

    /// [`Self::from_json_bytes`] with an explicit [`DetectionArea`]
    /// source.
    pub fn from_json_bytes_with_area(bytes: &[u8], area: DetectionArea) -> Result<Self, EvalError> {
        #[cfg(feature = "bench-timings")]
        let t0 = std::time::Instant::now();
        let seed = DetectionColumnsSeed {
            keep_area: matches!(area, DetectionArea::Supplied),
        };
        let mut de = serde_json::Deserializer::from_slice(bytes);
        let columns = serde::de::DeserializeSeed::deserialize(seed, &mut de)?;
        de.end()?;
        #[cfg(feature = "bench-timings")]
        let parse_ns = u64::try_from(t0.elapsed().as_nanos()).unwrap_or(u64::MAX);
        #[cfg(feature = "bench-timings")]
        let t1 = std::time::Instant::now();
        let result = Self::from_columns_with_area(columns, area);
        #[cfg(feature = "bench-timings")]
        {
            let from_inputs_ns = u64::try_from(t1.elapsed().as_nanos()).unwrap_or(u64::MAX);
            dataset_timings::COUNTERS.add(dataset_timings::DT_PARSE_NS, parse_ns);
            dataset_timings::COUNTERS.add(dataset_timings::DT_FROM_INPUTS_NS, from_inputs_ns);
        }
        result
    }

    /// Parallel sibling of [`Self::from_json_bytes`] (ADR-0054).
    ///
    /// The detection payload is a top-level array, so there is no key
    /// to find — only element boundaries. Elements go through the same
    /// `serde_json` deserializer as the serial path, in input order,
    /// which matters beyond parity: [`Self::from_inputs`] assigns
    /// auto-ids by position (quirk **J1**).
    ///
    /// A budget below `MIN_THREADS_TO_SPLIT` stays on the serial loader;
    /// see that constant for the crossover measurement.
    ///
    /// # Errors
    ///
    /// Same as [`Self::from_json_bytes`].
    pub fn from_json_bytes_parallel(bytes: &[u8], threads: usize) -> Result<Self, EvalError> {
        Self::from_json_bytes_parallel_with_area(bytes, threads, DetectionArea::FromBbox)
    }

    /// [`Self::from_json_bytes_parallel`] with an explicit
    /// [`DetectionArea`] source.
    ///
    /// The area source is applied by [`Self::from_inputs_with_area`]
    /// after the split, so it is independent of how the payload was
    /// parsed. Both serial fallbacks below carry it too, which is what
    /// keeps the result independent of the thread budget.
    ///
    /// # Errors
    ///
    /// Same as [`Self::from_json_bytes`].
    pub fn from_json_bytes_parallel_with_area(
        bytes: &[u8],
        threads: usize,
        area: DetectionArea,
    ) -> Result<Self, EvalError> {
        if threads < crate::json_split::MIN_THREADS_TO_SPLIT {
            return Self::from_json_bytes_with_area(bytes, area);
        }
        #[cfg(feature = "bench-timings")]
        let t0 = std::time::Instant::now();
        let seed = DetectionColumnsSeed {
            keep_area: matches!(area, DetectionArea::Supplied),
        };
        let parsed = crate::json_split::document_object_array(bytes).and_then(|elements| {
            crate::json_split::parse_elements_parallel_into(
                bytes,
                &elements,
                crate::json_split::chunks_for(threads),
                DetectionColumnsInput::with_capacity,
                |cols, row| seed.push(cols, row),
            )
            .ok()
        });
        #[cfg(feature = "bench-timings")]
        let parse_ns = u64::try_from(t0.elapsed().as_nanos()).unwrap_or(u64::MAX);
        match parsed {
            Some(parts) => {
                #[cfg(feature = "bench-timings")]
                let t1 = std::time::Instant::now();
                let result =
                    Self::from_columns_with_area(DetectionColumnsInput::concat(parts), area);
                #[cfg(feature = "bench-timings")]
                {
                    let from_inputs_ns = u64::try_from(t1.elapsed().as_nanos()).unwrap_or(u64::MAX);
                    dataset_timings::COUNTERS.add(dataset_timings::DT_PARSE_NS, parse_ns);
                    dataset_timings::COUNTERS
                        .add(dataset_timings::DT_FROM_INPUTS_NS, from_inputs_ns);
                }
                result
            }
            // As on the GT side: the fallback's own instrumentation is
            // the only one that fires, so the abandoned scan is not
            // added to `DT_PARSE_NS` a second time.
            None => Self::from_json_bytes_with_area(bytes, area),
        }
    }

    /// Builds a [`CocoDetections`] directly from columnar inputs
    /// (ADR-0066), using [`DetectionArea::FromBbox`].
    pub fn from_columns(cols: DetectionColumnsInput) -> Result<Self, EvalError> {
        Self::from_columns_with_area(cols, DetectionArea::FromBbox)
    }

    /// [`Self::from_columns`] with an explicit [`DetectionArea`] source.
    pub fn from_columns_with_area(
        cols: DetectionColumnsInput,
        area: DetectionArea,
    ) -> Result<Self, EvalError> {
        let n = cols.scores.len();
        check_col_len("image_ids", cols.image_ids.len(), n)?;
        check_col_len("category_ids", cols.category_ids.len(), n)?;
        check_col_len("bboxes", cols.bboxes.len(), n)?;
        if let Some(ids) = &cols.ids {
            check_col_len("ids", ids.len(), n)?;
        }
        if let Some(areas) = &cols.areas {
            check_col_len("areas", areas.len(), n)?;
        }
        if let Some(segs) = &cols.segmentations {
            check_col_len("segmentations", segs.len(), n)?;
        }
        if let Some(kps) = &cols.keypoints {
            check_col_len("keypoints", kps.len(), n)?;
        }
        if let Some(nkps) = &cols.num_keypoints {
            check_col_len("num_keypoints", nkps.len(), n)?;
        }

        // Resolve ids (quirk J1) in input order.
        let ids = match cols.ids {
            None => IdColumn::Sequential { start: 1 },
            Some(raw_ids) if raw_ids.iter().all(Option::is_none) => {
                IdColumn::Sequential { start: 1 }
            }
            Some(raw_ids) => {
                let mut next_auto = 1i64;
                let mut resolved = Vec::with_capacity(n);
                for opt_id in raw_ids {
                    let id = match opt_id {
                        Some(id) => id,
                        None => {
                            let id = AnnId(next_auto);
                            next_auto += 1;
                            id
                        }
                    };
                    resolved.push(id);
                }
                IdColumn::Explicit(resolved)
            }
        };

        // Validate finite scores and resolve areas (quirk J3), preserving
        // exact row-by-row error precedence when `DetectionArea::Mask`
        // validates segmentation RLEs alongside scores.
        // `Mask` checks scores row by row inside its loop instead, so a
        // non-finite score and a bad mask report in row order.
        if !matches!(area, DetectionArea::Mask) && cols.scores.iter().any(|s| !s.is_finite()) {
            return Err(EvalError::NonFinite {
                context: "detection score",
            });
        }
        let areas = match area {
            DetectionArea::FromBbox => AreaColumn::FromBbox,
            DetectionArea::Supplied => match cols.areas {
                None => AreaColumn::FromBbox,
                Some(raw_areas) if raw_areas.iter().all(Option::is_none) => AreaColumn::FromBbox,
                Some(raw_areas) => {
                    let mut resolved = Vec::with_capacity(n);
                    for (idx, opt_area) in raw_areas.into_iter().enumerate() {
                        let b = cols.bboxes[idx];
                        resolved.push(opt_area.unwrap_or(b.w * b.h));
                    }
                    AreaColumn::Explicit(resolved)
                }
            },
            DetectionArea::Mask => {
                let mut resolved = Vec::with_capacity(n);
                for idx in 0..n {
                    if !cols.scores[idx].is_finite() {
                        return Err(EvalError::NonFinite {
                            context: "detection score",
                        });
                    }
                    let seg = cols
                        .segmentations
                        .as_ref()
                        .and_then(|segs| segs[idx].as_ref());
                    resolved.push(mask_area_from_seg(
                        seg,
                        cols.image_ids[idx],
                        ids.id_at(idx),
                    )?);
                }
                AreaColumn::Explicit(resolved)
            }
        };

        let segmentations = cols.segmentations.filter(|v| v.iter().any(Option::is_some));
        let keypoints = cols.keypoints.filter(|v| v.iter().any(Option::is_some));
        let num_keypoints = cols.num_keypoints.filter(|v| v.iter().any(Option::is_some));

        Ok(Self::from_resolved_columns(DetectionColumns {
            ids,
            image_ids: cols.image_ids,
            category_ids: cols.category_ids,
            scores: cols.scores,
            bboxes: cols.bboxes,
            areas,
            segmentations,
            keypoints,
            num_keypoints,
        }))
    }

    /// Builds a [`CocoDetections`] from typed inputs. Auto-assigns ids
    /// (quirk **J1**) for inputs that did not supply one, validates
    /// finite scores, and derives areas (quirk **J3**).
    pub fn from_inputs(inputs: Vec<DetectionInput>) -> Result<Self, EvalError> {
        Self::from_inputs_with_area(inputs, DetectionArea::FromBbox)
    }

    /// [`Self::from_inputs`] with an explicit [`DetectionArea`] source.
    pub fn from_inputs_with_area(
        inputs: Vec<DetectionInput>,
        area: DetectionArea,
    ) -> Result<Self, EvalError> {
        let seed = DetectionColumnsSeed {
            keep_area: matches!(area, DetectionArea::Supplied),
        };
        let mut columns = DetectionColumnsInput::with_capacity(inputs.len());
        for input in inputs {
            seed.push(&mut columns, input);
        }
        Self::from_columns_with_area(columns, area)
    }

    /// Build from already-resolved records, preserving their ids and
    /// fields verbatim. Used by the streaming evaluator and tests to
    /// assemble a `CocoDetections` view without re-running the auto-id
    /// and area-derivation logic in [`Self::from_inputs`].
    pub fn from_records(records: Vec<CocoDetection>) -> Self {
        let n = records.len();
        let mut ids = Vec::with_capacity(n);
        let mut areas = Vec::with_capacity(n);
        // Ids and areas are already resolved, so they bypass the input's
        // sparse `Option` columns; everything else goes through `push`.
        let mut cols = DetectionColumnsInput::with_capacity(n);
        for rec in records {
            ids.push(rec.id);
            areas.push(rec.area);
            cols.push(DetectionInput {
                id: None,
                image_id: rec.image_id,
                category_id: rec.category_id,
                score: rec.score,
                bbox: rec.bbox,
                area: None,
                segmentation: rec.segmentation,
                keypoints: rec.keypoints,
                num_keypoints: rec.num_keypoints,
            });
        }

        Self::from_resolved_columns(DetectionColumns {
            ids: IdColumn::Explicit(ids),
            image_ids: cols.image_ids,
            category_ids: cols.category_ids,
            scores: cols.scores,
            bboxes: cols.bboxes,
            areas: AreaColumn::Explicit(areas),
            segmentations: cols.segmentations,
            keypoints: cols.keypoints,
            num_keypoints: cols.num_keypoints,
        })
    }

    /// Total number of detections in the collection.
    #[inline]
    pub fn len(&self) -> usize {
        self.columns.scores.len()
    }

    /// `true` when the collection contains zero detections.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.columns.scores.is_empty()
    }

    /// Resolved annotation id at flat row `idx` (`O(1)`).
    #[inline]
    pub fn id_at(&self, idx: usize) -> AnnId {
        self.columns.ids.id_at(idx)
    }

    /// Image id at flat row `idx`.
    #[inline]
    pub fn image_id_at(&self, idx: usize) -> ImageId {
        self.columns.image_ids[idx]
    }

    /// Dense slice of per-detection image ids (`len == self.len()`).
    #[inline]
    pub fn image_ids(&self) -> &[ImageId] {
        &self.columns.image_ids
    }

    /// Distinct image ids that carry at least one detection.
    pub fn distinct_image_ids(&self) -> impl ExactSizeIterator<Item = ImageId> + '_ {
        self.by_image.keys().copied()
    }

    /// Category id at flat row `idx`.
    #[inline]
    pub fn category_id_at(&self, idx: usize) -> CategoryId {
        self.columns.category_ids[idx]
    }

    /// Dense slice of per-detection category ids (`len == self.len()`).
    #[inline]
    pub fn category_ids(&self) -> &[CategoryId] {
        &self.columns.category_ids
    }

    /// Confidence score at flat row `idx`.
    #[inline]
    pub fn score_at(&self, idx: usize) -> f64 {
        self.columns.scores[idx]
    }

    /// Dense slice of per-detection confidence scores (`len == self.len()`).
    #[inline]
    pub fn scores(&self) -> &[f64] {
        &self.columns.scores
    }

    /// Bounding box at flat row `idx`.
    #[inline]
    pub fn bbox_at(&self, idx: usize) -> Bbox {
        self.columns.bboxes[idx]
    }

    /// Dense slice of per-detection bounding boxes (`len == self.len()`).
    #[inline]
    pub fn bboxes(&self) -> &[Bbox] {
        &self.columns.bboxes
    }

    /// Pixel area at flat row `idx` (`O(1)`, derived from bbox when
    /// built with [`DetectionArea::FromBbox`]).
    #[inline]
    pub fn area_at(&self, idx: usize) -> f64 {
        self.columns.areas.area_at(idx, &self.columns.bboxes)
    }

    /// Raw annotation ids of rows `kept` into `out`, replacing its
    /// contents — the per-cell bulk form of [`Self::id_at`].
    #[inline]
    pub fn gather_ids_into(&self, kept: &[usize], out: &mut Vec<i64>) {
        self.columns.ids.gather_into(kept, out);
    }

    /// Areas of rows `kept` into `out`, replacing its contents — the
    /// per-cell bulk form of [`Self::area_at`].
    #[inline]
    pub fn gather_areas_into(&self, kept: &[usize], out: &mut Vec<f64>) {
        self.columns
            .areas
            .gather_into(kept, &self.columns.bboxes, out);
    }

    /// Optional segmentation prediction at flat row `idx`.
    #[inline]
    pub fn segmentation_at(&self, idx: usize) -> Option<&Segmentation> {
        self.columns
            .segmentations
            .as_ref()
            .and_then(|segs| segs[idx].as_ref())
    }

    /// Optional keypoint triplets at flat row `idx`.
    #[inline]
    pub fn keypoints_at(&self, idx: usize) -> Option<&[f64]> {
        self.columns
            .keypoints
            .as_ref()
            .and_then(|kps| kps[idx].as_deref())
    }

    /// Optional `num_keypoints` count at flat row `idx`.
    #[inline]
    pub fn num_keypoints_at(&self, idx: usize) -> Option<u32> {
        self.columns.num_keypoints.as_ref().and_then(|nk| nk[idx])
    }

    /// Reconstructs one [`CocoDetection`] record by value from the
    /// underlying SoA columns. Intended for cold diagnostic, table, and
    /// test surfaces; hot evaluation paths read the columnar accessors.
    pub fn record_at(&self, idx: usize) -> CocoDetection {
        CocoDetection {
            id: self.id_at(idx),
            image_id: self.image_id_at(idx),
            category_id: self.category_id_at(idx),
            score: self.score_at(idx),
            bbox: self.bbox_at(idx),
            area: self.area_at(idx),
            segmentation: self.segmentation_at(idx).cloned(),
            keypoints: self.keypoints_at(idx).map(<[f64]>::to_vec),
            num_keypoints: self.num_keypoints_at(idx),
        }
    }

    /// Cold iterator yielding reconstructed [`CocoDetection`] records
    /// in flat row order (`0..self.len()`).
    pub fn records(&self) -> impl ExactSizeIterator<Item = CocoDetection> + '_ {
        (0..self.len()).map(|idx| self.record_at(idx))
    }

    /// Flat row indices for one `(image, category)` cell. Empty slice
    /// when the cell is empty (no detections of that category on that
    /// image).
    pub fn indices_for(&self, image: ImageId, cat: CategoryId) -> &[usize] {
        self.by_image_cat.get(&(image, cat))
    }

    /// Flat row indices for every detection on an image, regardless of
    /// category. Path used when `useCats=false` (quirk **L4**).
    pub fn indices_for_image(&self, image: ImageId) -> &[usize] {
        self.by_image.get(&image)
    }

    /// [`Self::lvis_trim`] to `max_dets` when `gt` is federated and some
    /// image exceeds the cap; `self` unchanged otherwise (ADR-0026 AC2,
    /// ADR-0065). Applied by whoever assembles the detection set, as
    /// `LVISResults` does at construction.
    #[must_use]
    pub fn trim_for(self, gt: &CocoDataset, max_dets: usize) -> Self {
        // Nothing to drop leaves the set as is: only the flat order would
        // differ, and consumers read detections by id and per-cell score.
        if !gt.is_federated() || self.by_image.max_group_len() <= max_dets {
            return self;
        }
        // Saturate: a wrapped cap would land in AC5's negative "disabled" range.
        self.lvis_trim(i64::try_from(max_dets).unwrap_or(i64::MAX))
    }

    /// LVIS per-image top-`max_dets` trim (quirk **AC2** of ADR-0026).
    ///
    /// Mirrors `LVISResults.limit_dets_per_image` at
    /// `lvis/results.py:73-84`: groups detections by `image_id`,
    /// sorts each group by score descending (stable — quirk
    /// **AC4**), and keeps the top `max_dets` across **all
    /// categories combined**. The cross-class consequence (quirk
    /// **AC3**): 250 cat-A + 350 cat-B detections on one image trim
    /// to **300 total**, not 250 + min(350, 300).
    ///
    /// `max_dets < 0` (or `i64::MIN`) disables the trim entirely
    /// (quirk **AC5**, mirroring the `if max_dets >= 0` guard at
    /// `results.py:39-40`). `max_dets == 0` keeps zero detections —
    /// edge case the upstream allows but isn't useful in practice.
    ///
    /// The output preserves DT ids and per-detection fields verbatim;
    /// only the membership of the flat detections columns and the
    /// per-cell CSR indices change.
    ///
    /// Within each image's group, ties on `score` resolve in input
    /// order: Rust's `slice::sort_by` is stable, matching Python's
    /// `sorted(_, reverse=True)` Timsort behavior.
    pub fn lvis_trim(&self, max_dets: i64) -> CocoDetections {
        if max_dets < 0 {
            // AC5: negative cap disables the trim. `Arc::clone` on both
            // `columns` and CSR indices makes this O(1).
            return self.clone();
        }
        let cap = usize::try_from(max_dets).unwrap_or(usize::MAX);
        let mut image_ids: Vec<ImageId> = self.by_image.keys().copied().collect();
        image_ids.sort_unstable_by_key(|i| i.0);

        let upper_bound = self.len().min(cap.saturating_mul(image_ids.len()));
        let mut kept: Vec<usize> = Vec::with_capacity(upper_bound);
        let mut group: Vec<usize> = Vec::with_capacity(self.by_image.max_group_len());
        let scores = self.scores();
        for image_id in image_ids {
            group.clear();
            group.extend_from_slice(self.indices_for_image(image_id));
            // Stable sort by score descending. `partial_cmp` returns
            // `None` only on NaN; `from_columns` rejects NaN scores
            // upstream (quirk **AD3** corrected).
            group.sort_by(|&a, &b| {
                scores[b]
                    .partial_cmp(&scores[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            kept.extend(group.iter().copied().take(cap));
        }
        Self::from_resolved_columns(self.columns.gather(&kept))
    }
}

// ---------------------------------------------------------------------------
// serde glue
// ---------------------------------------------------------------------------

/// COCO JSON uses `0`/`1` ints for `iscrowd` / `ignore`, but a
/// permissive reader also accepts bool literals. Shared between the
/// required and optional flag deserializers below. Dispatched by JSON
/// type, not `#[serde(untagged)]`, which buffers each value first.
enum BoolOrInt {
    Bool(bool),
    Int(i64),
}

impl<'de> Deserialize<'de> for BoolOrInt {
    fn deserialize<D: serde::Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        struct FlagVisitor;

        impl serde::de::Visitor<'_> for FlagVisitor {
            type Value = BoolOrInt;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a bool or an integer")
            }

            fn visit_bool<E: serde::de::Error>(self, v: bool) -> Result<BoolOrInt, E> {
                Ok(BoolOrInt::Bool(v))
            }

            fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<BoolOrInt, E> {
                Ok(BoolOrInt::Int(v))
            }

            fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<BoolOrInt, E> {
                i64::try_from(v)
                    .map(BoolOrInt::Int)
                    .map_err(|_| E::invalid_value(serde::de::Unexpected::Unsigned(v), &self))
            }
        }

        de.deserialize_any(FlagVisitor)
    }
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
            area: None,
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
        let ids: Vec<AnnId> = dts.records().map(|d| d.id).collect();
        assert_eq!(ids, vec![AnnId(1), AnnId(2)]);
    }

    #[test]
    fn j1_preserves_user_supplied_ids() {
        let mut a = dt_input(1, 1, 0.9, (0.0, 0.0, 1.0, 1.0));
        a.id = Some(AnnId(42));
        let mut b = dt_input(1, 1, 0.8, (0.0, 0.0, 1.0, 1.0));
        b.id = Some(AnnId(7));
        let dts = CocoDetections::from_inputs(vec![a, b]).unwrap();
        let ids: Vec<AnnId> = dts.records().map(|d| d.id).collect();
        assert_eq!(ids, vec![AnnId(42), AnnId(7)]);
    }

    #[test]
    fn j3_derives_area_from_bbox() {
        let dts =
            CocoDetections::from_inputs(vec![dt_input(1, 1, 0.5, (10.0, 10.0, 4.0, 5.0))]).unwrap();
        assert_eq!(dts.area_at(0), 20.0);
    }

    #[test]
    fn j3_ignores_supplied_area_by_default() {
        let json = br#"[{"image_id": 1, "category_id": 1, "score": 0.5, "bbox": [0, 0, 4, 5], "area": 99.0}]"#;
        let dts = CocoDetections::from_json_bytes(json).unwrap();
        assert_eq!(dts.area_at(0), 20.0);
    }

    #[test]
    fn j3_supplied_area_is_read_verbatim_with_bbox_fallback() {
        let json = br#"[
            {"image_id": 1, "category_id": 1, "score": 0.5, "bbox": [0, 0, 4, 5], "area": 99.0},
            {"image_id": 1, "category_id": 1, "score": 0.4, "bbox": [0, 0, 4, 5]}
        ]"#;
        let dts = CocoDetections::from_json_bytes_with_area(json, DetectionArea::Supplied).unwrap();
        assert_eq!(dts.area_at(0), 99.0);
        assert_eq!(dts.area_at(1), 20.0);
    }

    fn rle_input(counts: SegmentationRleCounts) -> DetectionInput {
        DetectionInput {
            segmentation: Some(Segmentation::Rle(crate::segmentation::SegmentationRle {
                size: [4, 4],
                counts,
            })),
            ..dt_input(1, 1, 0.5, (0.0, 0.0, 4.0, 4.0))
        }
    }

    #[test]
    fn j3_mask_area_is_the_rle_foreground_for_both_count_forms() {
        // Runs: 3 background, 5 foreground, 2 background, 6 foreground.
        let runs = vec![3u32, 5, 2, 6];
        let compressed = vernier_mask::Rle::from_counts(4, 4, runs.clone()).to_string_bytes();
        let dts = CocoDetections::from_inputs_with_area(
            vec![
                rle_input(SegmentationRleCounts::Compressed(
                    String::from_utf8(compressed).unwrap(),
                )),
                rle_input(SegmentationRleCounts::Uncompressed(runs.into())),
            ],
            DetectionArea::Mask,
        )
        .unwrap();
        assert_eq!(dts.area_at(0), 11.0);
        assert_eq!(dts.area_at(1), 11.0);
    }

    #[test]
    fn j3_mask_area_rejects_detections_without_an_rle() {
        let polygon = DetectionInput {
            segmentation: Some(Segmentation::Polygons(vec![vec![
                0.0, 0.0, 4.0, 0.0, 4.0, 4.0,
            ]])),
            ..dt_input(1, 1, 0.5, (0.0, 0.0, 4.0, 4.0))
        };
        for input in [dt_input(1, 1, 0.5, (0.0, 0.0, 4.0, 4.0)), polygon] {
            let err = CocoDetections::from_inputs_with_area(vec![input], DetectionArea::Mask)
                .unwrap_err();
            assert!(matches!(err, EvalError::InvalidAnnotation { .. }));
        }
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

    // ---- ADR-0054: split-and-parallel loaders -------------------------

    /// Ground truth exercising every shape the splitter has to survive:
    /// RLE `counts` holding structural bytes, polygon segmentation,
    /// ragged optional fields and non-compact whitespace.
    const SPLITTER_GT: &str = r#"{
        "info": {"description": "unknown top-level members are skipped"},
        "images": [
            {"id": 1, "width": 100, "height": 80, "file_name": "a}.jpg"},
            {"id": 2, "width": 60, "height": 60, "file_name": "b,{}.jpg"}
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 7, "bbox": [0, 0, 10, 10],
             "area": 100.0, "iscrowd": 0, "segmentation": [[0,0, 10,0, 10,10]]},
            {"id": 2, "image_id": 1, "category_id": 7, "bbox": [5, 5, 20, 20],
             "area": 400.0, "iscrowd": 1,
             "segmentation": {"counts": "a},{b", "size": [80, 100]}},
            {"id": 3, "image_id": 2, "category_id": 9, "bbox": [1, 1, 2, 2],
             "area": 4.0, "iscrowd": 0}
        ],
        "categories": [
            {"id": 7, "name": "thing", "supercategory": "stuff"},
            {"id": 9, "name": "other", "supercategory": "stuff"}
        ]
    }"#;

    #[test]
    fn parallel_gt_loader_matches_the_serial_loader() {
        let serial = CocoDataset::from_json_bytes(SPLITTER_GT.as_bytes()).expect("serial");
        for threads in [1usize, 2, 4, 8] {
            let parallel = CocoDataset::from_json_bytes_parallel(SPLITTER_GT.as_bytes(), threads)
                .expect("parallel");
            assert_eq!(
                format!("{:?}", serial.annotations()),
                format!("{:?}", parallel.annotations()),
                "threads={threads}"
            );
            assert_eq!(
                format!("{:?}", serial.images()),
                format!("{:?}", parallel.images())
            );
            assert_eq!(
                format!("{:?}", serial.categories()),
                format!("{:?}", parallel.categories())
            );
        }
    }

    /// A shape the splitter does not recognize must still load, via the
    /// serial fallback — and a malformed one must still fail the same
    /// way, since the fallback is what reports the error.
    #[test]
    fn parallel_gt_loader_falls_back_instead_of_failing() {
        // `images` is an object, not an array of objects.
        let odd = r#"{"images": {"id": 1}, "annotations": [], "categories": []}"#;
        assert_eq!(
            CocoDataset::from_json_bytes(odd.as_bytes()).is_err(),
            CocoDataset::from_json_bytes_parallel(odd.as_bytes(), 4).is_err()
        );

        let malformed = r#"{"images": [{"id": 1}], "annotations": [ , ], "categories": []}"#;
        let serial = CocoDataset::from_json_bytes(malformed.as_bytes()).unwrap_err();
        let parallel = CocoDataset::from_json_bytes_parallel(malformed.as_bytes(), 4).unwrap_err();
        assert_eq!(serial.to_string(), parallel.to_string());

        // An annotation referencing an unknown image is a *semantic*
        // error from `from_parts`, identical on both paths.
        let dangling = r#"{"images": [], "annotations": [{"id": 1, "image_id": 4,
            "category_id": 1, "bbox": [0,0,1,1], "area": 1.0, "iscrowd": 0}],
            "categories": [{"id": 1, "name": "x", "supercategory": "y"}]}"#;
        let serial = CocoDataset::from_json_bytes(dangling.as_bytes()).unwrap_err();
        let parallel = CocoDataset::from_json_bytes_parallel(dangling.as_bytes(), 4).unwrap_err();
        assert_eq!(serial.to_string(), parallel.to_string());
    }

    /// Detection ids are assigned by position (quirk **J1**), so a
    /// splitter that reordered or dropped an element would show up here
    /// even when the values themselves survived.
    #[test]
    fn parallel_dt_loader_preserves_input_order_and_auto_ids() {
        let mut json = String::from("[");
        for i in 0..200 {
            if i > 0 {
                json.push(',');
            }
            // No `id` field: ids come from position.
            json.push_str(&format!(
                r#"{{"image_id": {}, "category_id": 3, "bbox": [0,0,{},2], "score": 0.5}}"#,
                i % 7,
                i + 1
            ));
        }
        json.push(']');

        let serial = CocoDetections::from_json_bytes(json.as_bytes()).expect("serial");
        let serial_recs: Vec<CocoDetection> = serial.records().collect();
        for threads in [1usize, 3, 8] {
            let parallel =
                CocoDetections::from_json_bytes_parallel(json.as_bytes(), threads).expect("par");
            assert_eq!(
                format!("{:?}", serial_recs),
                format!("{:?}", parallel.records().collect::<Vec<_>>()),
                "threads={threads}"
            );
        }
    }

    /// The parallel loader parses into one column set per chunk; an
    /// optional field first seen in a late chunk must still land on its
    /// own row once the chunks are joined.
    #[test]
    fn parallel_dt_loader_aligns_sparse_columns_across_chunks() {
        let mut json = String::from("[");
        for i in 0..200 {
            if i > 0 {
                json.push(',');
            }
            let id = if i % 50 == 49 {
                format!(r#""id": {}, "#, 1000 + i)
            } else {
                String::new()
            };
            let area = if i == 150 { r#", "area": 7.5"# } else { "" };
            let seg = if i >= 180 && i % 3 == 0 {
                r#", "segmentation": {"size": [4, 4], "counts": [0, 16]}"#
            } else {
                ""
            };
            json.push_str(&format!(
                r#"{{{id}"image_id": {}, "category_id": 3, "bbox": [0,0,{},2], "score": 0.5{area}{seg}}}"#,
                i % 7,
                i + 1
            ));
        }
        json.push(']');

        for area in [DetectionArea::FromBbox, DetectionArea::Supplied] {
            let serial =
                CocoDetections::from_json_bytes_with_area(json.as_bytes(), area).expect("serial");
            assert_eq!(serial.id_at(49), AnnId(1049));
            assert!(serial.segmentation_at(183).is_some());
            assert!(serial.segmentation_at(184).is_none());
            let expected_area = if area == DetectionArea::Supplied {
                7.5
            } else {
                302.0
            };
            assert_eq!(serial.area_at(150), expected_area);
            let serial_recs = format!("{:?}", serial.records().collect::<Vec<_>>());
            for threads in [3usize, 8] {
                let parallel = CocoDetections::from_json_bytes_parallel_with_area(
                    json.as_bytes(),
                    threads,
                    area,
                )
                .expect("par");
                assert_eq!(
                    serial_recs,
                    format!("{:?}", parallel.records().collect::<Vec<_>>()),
                    "threads={threads} area={area:?}"
                );
            }
        }
    }

    /// LVIS GT with the federated extras the COCO fixture cannot
    /// reach: `neg_category_ids`, `not_exhaustive_category_ids` and
    /// per-category `frequency`, plus a skipped top-level member.
    const SPLITTER_LVIS_GT: &str = r#"{
        "info": {"version": "1.0"},
        "licenses": [{"id": 1, "name": "cc"}],
        "images": [
            {"id": 1, "width": 100, "height": 80, "file_name": "a,{}.jpg",
             "neg_category_ids": [9], "not_exhaustive_category_ids": [7]},
            {"id": 2, "width": 60, "height": 60, "file_name": "b.jpg",
             "neg_category_ids": []}
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 7, "bbox": [0, 0, 10, 10],
             "area": 100.0, "iscrowd": 0,
             "segmentation": {"counts": "a},{b", "size": [80, 100]}},
            {"id": 2, "image_id": 2, "category_id": 9, "bbox": [1, 1, 2, 2],
             "area": 4.0, "iscrowd": 0}
        ],
        "categories": [
            {"id": 7, "name": "thing", "supercategory": "stuff", "frequency": "c"},
            {"id": 9, "name": "other", "supercategory": "stuff", "frequency": "r"}
        ]
    }"#;

    /// `LvisImageRaw` / `LvisCategoryRaw` reach `serde_json` through
    /// the splitter's element ranges rather than through one
    /// `from_slice`, so the federated fields need their own equivalence
    /// gate — the COCO fixture never deserializes them.
    #[test]
    fn parallel_lvis_loader_matches_the_serial_loader() {
        let serial =
            CocoDataset::from_lvis_json_bytes(SPLITTER_LVIS_GT.as_bytes()).expect("serial");
        let serial_fed = serial.federated().expect("federated");
        for threads in [1usize, 2, 4, 8] {
            let parallel =
                CocoDataset::from_lvis_json_bytes_parallel(SPLITTER_LVIS_GT.as_bytes(), threads)
                    .expect("parallel");
            assert_eq!(
                format!("{:?}", serial.annotations()),
                format!("{:?}", parallel.annotations()),
                "threads={threads}"
            );
            assert_eq!(
                format!("{:?}", serial.images()),
                format!("{:?}", parallel.images()),
                "threads={threads}"
            );
            assert_eq!(
                format!("{:?}", serial.categories()),
                format!("{:?}", parallel.categories()),
                "threads={threads}"
            );
            let fed = parallel.federated().expect("federated");
            for image in [ImageId(1), ImageId(2)] {
                assert_eq!(
                    serial_fed.pos_category_ids.get(&image),
                    fed.pos_category_ids.get(&image),
                    "pos threads={threads}"
                );
                assert_eq!(
                    serial_fed.neg_category_ids.get(&image),
                    fed.neg_category_ids.get(&image),
                    "neg threads={threads}"
                );
                assert_eq!(
                    serial_fed.not_exhaustive_category_ids.get(&image),
                    fed.not_exhaustive_category_ids.get(&image),
                    "not_exhaustive threads={threads}"
                );
            }
            let sorted = |m: &HashMap<CategoryId, Frequency>| {
                let mut v: Vec<(i64, Frequency)> = m.iter().map(|(k, f)| (k.0, *f)).collect();
                v.sort_unstable_by_key(|(id, _)| *id);
                v
            };
            assert_eq!(
                sorted(&serial_fed.category_frequency),
                sorted(&fed.category_frequency),
                "frequency threads={threads}"
            );
        }

        // A malformed LVIS document must fail identically on both.
        let malformed = SPLITTER_LVIS_GT.replacen(r#""version": "1.0""#, r#""version": tru"#, 1);
        let serial = CocoDataset::from_lvis_json_bytes(malformed.as_bytes()).unwrap_err();
        let parallel =
            CocoDataset::from_lvis_json_bytes_parallel(malformed.as_bytes(), 4).unwrap_err();
        assert_eq!(serial.to_string(), parallel.to_string());
    }

    /// Detections shaped like a real DT payload, for the equivalence
    /// gates below.
    const SPLITTER_DT: &str = r#"[
        {"image_id": 1, "category_id": 7, "bbox": [0, 0, 10, 10], "score": 0.5,
         "segmentation": {"counts": "a},{b", "size": [80, 100]}},
        {"id": 12, "image_id": 2, "category_id": 9, "bbox": [1, 1, 2, 2],
         "score": 0.9992794394493103},
        {"image_id": 2, "category_id": 9, "bbox": [1.5, 1.5, 2.5, 2.5], "score": 1e-3}
    ]"#;

    /// Acceptance must not be a function of `num_threads` (ADR-0054:
    /// "errors on malformed input must keep their current text and
    /// byte offsets"). The splitter walks past the parts of the
    /// document it does not parse, so every such part gets a case
    /// here: bytes after the document, and a member nobody asked for.
    #[test]
    fn parallel_loaders_accept_exactly_what_the_serial_loaders_accept() {
        let gt_cases = [
            // Trailing bytes after the top-level object.
            format!("{SPLITTER_GT}}}}}}}garbage"),
            format!("{SPLITTER_GT} ["),
            format!("{SPLITTER_GT}{SPLITTER_GT}"),
            // A skipped member that is not valid JSON.
            SPLITTER_GT.replacen(r#""description""#, r#""description": tru, "x""#, 1),
            SPLITTER_GT.replacen(r#"{"description""#, r#"{"a": NaN, "description""#, 1),
            SPLITTER_GT.replacen(r#"{"description""#, r#"{"a": 01+2x, "description""#, 1),
            SPLITTER_GT.replacen(r#""info""#, r#""in\qfo""#, 1),
            // And the well-formed original, which must still split.
            SPLITTER_GT.to_owned(),
        ];
        for case in &gt_cases {
            let serial = CocoDataset::from_json_bytes(case.as_bytes());
            for threads in [1usize, 2, 3, 8] {
                let parallel = CocoDataset::from_json_bytes_parallel(case.as_bytes(), threads);
                assert_outcomes_match(
                    serial.as_ref().map(|d| format!("{:?}", d.annotations())),
                    parallel.as_ref().map(|d| format!("{:?}", d.annotations())),
                    &format!("threads={threads} case={case}"),
                );
            }
        }

        let dt_cases = [
            format!("{SPLITTER_DT}{SPLITTER_DT}"),
            format!("{SPLITTER_DT} xyz"),
            format!("{SPLITTER_DT}]"),
            SPLITTER_DT.to_owned(),
        ];
        for case in &dt_cases {
            let serial = CocoDetections::from_json_bytes(case.as_bytes());
            for threads in [1usize, 2, 3, 8] {
                let parallel = CocoDetections::from_json_bytes_parallel(case.as_bytes(), threads);
                assert_outcomes_match(
                    serial
                        .as_ref()
                        .map(|d| format!("{:?}", d.records().collect::<Vec<_>>())),
                    parallel
                        .as_ref()
                        .map(|d| format!("{:?}", d.records().collect::<Vec<_>>())),
                    &format!("threads={threads} case={case}"),
                );
            }
        }
    }

    /// Serial and parallel must agree on acceptance, on the loaded
    /// values, and — when they reject — on the exact message, which is
    /// where the byte offset users debug against lives.
    fn assert_outcomes_match(
        serial: Result<String, &EvalError>,
        parallel: Result<String, &EvalError>,
        context: &str,
    ) {
        match (serial, parallel) {
            (Ok(a), Ok(b)) => assert_eq!(a, b, "values diverged: {context}"),
            (Err(a), Err(b)) => assert_eq!(
                a.to_string(),
                b.to_string(),
                "error text diverged: {context}"
            ),
            (Ok(_), Err(e)) => panic!("parallel rejected what serial accepted ({e}): {context}"),
            (Err(e), Ok(_)) => panic!("parallel accepted what serial rejected ({e}): {context}"),
        }
    }

    // Seeded byte mutations over the two fixtures, asserting the same
    // equivalence the hand-written cases do. Single-byte edits are the
    // shape that breaks a structural scanner: they turn a quote into a
    // brace, a digit into a letter, or a closing bracket into
    // whitespace, anywhere in the document — including the regions the
    // splitter only walks.
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(512))]

        #[test]
        fn mutated_gt_loads_identically_on_both_paths(
            offset in any::<prop::sample::Index>(),
            byte in any::<u8>(),
            threads in 1usize..=8,
        ) {
            let mut bytes = SPLITTER_GT.as_bytes().to_vec();
            let at = offset.index(bytes.len());
            bytes[at] = byte;
            let serial = CocoDataset::from_json_bytes(&bytes);
            let parallel = CocoDataset::from_json_bytes_parallel(&bytes, threads);
            assert_outcomes_match(
                serial.as_ref().map(|d| format!("{:?}", d.annotations())),
                parallel.as_ref().map(|d| format!("{:?}", d.annotations())),
                &format!("threads={threads} at={at} byte={byte}"),
            );
        }

        #[test]
        fn mutated_dt_loads_identically_on_both_paths(
            offset in any::<prop::sample::Index>(),
            byte in any::<u8>(),
            threads in 1usize..=8,
        ) {
            let mut bytes = SPLITTER_DT.as_bytes().to_vec();
            let at = offset.index(bytes.len());
            bytes[at] = byte;
            let serial = CocoDetections::from_json_bytes(&bytes);
            let parallel = CocoDetections::from_json_bytes_parallel(&bytes, threads);
            assert_outcomes_match(
                serial
                    .as_ref()
                    .map(|d| format!("{:?}", d.records().collect::<Vec<_>>())),
                parallel
                    .as_ref()
                    .map(|d| format!("{:?}", d.records().collect::<Vec<_>>())),
                &format!("threads={threads} at={at} byte={byte}"),
            );
        }

        // Truncation and insertion, which a single-byte overwrite
        // cannot reach: both move every later offset, so a scanner
        // that survived them by luck would not survive twice.
        #[test]
        fn truncated_or_extended_gt_loads_identically_on_both_paths(
            offset in any::<prop::sample::Index>(),
            byte in any::<u8>(),
            truncate in any::<bool>(),
            threads in 1usize..=8,
        ) {
            let mut bytes = SPLITTER_GT.as_bytes().to_vec();
            let at = offset.index(bytes.len());
            if truncate {
                bytes.truncate(at);
            } else {
                bytes.insert(at, byte);
            }
            let serial = CocoDataset::from_json_bytes(&bytes);
            let parallel = CocoDataset::from_json_bytes_parallel(&bytes, threads);
            assert_outcomes_match(
                serial.as_ref().map(|d| format!("{:?}", d.annotations())),
                parallel.as_ref().map(|d| format!("{:?}", d.annotations())),
                &format!("threads={threads} at={at} truncate={truncate} byte={byte}"),
            );
        }
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
        let ds: Vec<CocoDetection> = dts.records().collect();
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
        assert!(dts.segmentation_at(0).is_some());
    }

    #[test]
    fn dt_without_segmentation_loads_as_none() {
        const JSON: &str = r#"[
            {"image_id": 1, "category_id": 1, "score": 0.9, "bbox": [0, 0, 1, 1]}
        ]"#;
        let dts = CocoDetections::from_json_bytes(JSON.as_bytes()).unwrap();
        assert!(dts.segmentation_at(0).is_none());
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

    /// Minimal `CocoAnnotation` with the required ids set and every
    /// optional field defaulted. Tests that only care about identity /
    /// canonical-form invariance use this to skip the 10-field literal.
    fn make_min_annotation(
        id: AnnId,
        image_id: ImageId,
        category_id: CategoryId,
    ) -> CocoAnnotation {
        CocoAnnotation {
            id,
            image_id,
            category_id,
            area: 25.0,
            is_crowd: false,
            ignore_flag: None,
            bbox: Bbox {
                x: 0.0,
                y: 0.0,
                w: 5.0,
                h: 5.0,
            },
            segmentation: None,
            keypoints: None,
            num_keypoints: None,
        }
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

    // -- AC2/AC3/AC4/AC5: lvis_trim per-image top-K ---------------------------

    #[test]
    fn ac2_q1_trims_500_single_category_to_300() {
        // ADR-0026 appendix Q1: 500 single-category detections on one
        // image must trim to exactly 300, dropping the lowest-score
        // 200. Score-descending order is preserved.
        let dts = CocoDetections::from_inputs(
            (0..500)
                .map(|i| {
                    let score = 1.0 - (i as f64) / 1000.0; // 1.0, 0.999, …, 0.501
                    dt_input(1, 1, score, (0.0, 0.0, 1.0, 1.0))
                })
                .collect(),
        )
        .unwrap();
        let trimmed = dts.lvis_trim(300);
        assert_eq!(trimmed.len(), 300);
        // Scores must be descending and start at 1.0.
        let scores: Vec<f64> = trimmed.scores().to_vec();
        for w in scores.windows(2) {
            assert!(
                w[0] >= w[1],
                "lvis_trim must preserve score-descending order"
            );
        }
        assert!((scores[0] - 1.0).abs() < 1e-12);
        // The lowest score in the trimmed set is the 300th input
        // (1.0 - 299/1000 = 0.701).
        assert!((scores[299] - 0.701).abs() < 1e-12);
    }

    #[test]
    fn ac3_q2_cross_class_crowding_keeps_300_total_across_classes() {
        // ADR-0026 appendix Q2: 250 cat-A + 350 cat-B detections on
        // one image trim to **300 total** (top-300 across both
        // classes by score combined), not 250 + min(350, 300) = 550.
        // Score layouts are interleaved so the trim has to actually
        // sort across classes — a per-class trim would leave cat-A
        // intact and only trim cat-B.
        let mut inputs = Vec::with_capacity(600);
        for i in 0..250 {
            // cat 1 scores: 0.5, 0.498, …, 0.002 (250 values)
            let score = 0.5 - (i as f64) * 0.002;
            inputs.push(dt_input(1, 1, score, (0.0, 0.0, 1.0, 1.0)));
        }
        for i in 0..350 {
            // cat 2 scores: 1.0, 0.998, …, 0.302 (350 values).
            // The top 300 across both classes are all cat-2 (every
            // cat-2 score >= 0.302 > every cat-1 score 0.5 only at
            // its top, so cross-class trim keeps cat-2 dominant).
            // Actually score 0.302 < 0.5 so cat-1 top entries
            // survive — see assertion below.
            let score = 1.0 - (i as f64) * 0.002;
            inputs.push(dt_input(1, 2, score, (0.0, 0.0, 1.0, 1.0)));
        }
        let dts = CocoDetections::from_inputs(inputs).unwrap();
        let trimmed = dts.lvis_trim(300);
        // AC3: top-300 total — not per-class.
        assert_eq!(trimmed.len(), 300);
        // Counts per category in the trim — cat-2 has higher overall
        // scores so most of the trim is cat-2; cat-1's top entries
        // (score 0.5 ≥ 0.302) also make the cut.
        let n_cat1 = trimmed
            .category_ids()
            .iter()
            .filter(|&&c| c == CategoryId(1))
            .count();
        let n_cat2 = trimmed
            .category_ids()
            .iter()
            .filter(|&&c| c == CategoryId(2))
            .count();
        // cat-2 scores >= 0.5 are i in 0..=250; cat-1 scores >= 0.302
        // are i in 0..=99. The exact mix is determined by the sort
        // of all 600 scores; what we assert is the cross-class total.
        assert_eq!(n_cat1 + n_cat2, 300);
        // Sanity: neither class is fully empty (otherwise the trim
        // would have collapsed to per-class).
        assert!(n_cat1 > 0, "cat 1 must keep at least its top-score entries");
        assert!(n_cat2 > 0, "cat 2 must keep its high-score entries");
    }

    #[test]
    fn ac5_negative_max_dets_disables_trim() {
        // `max_dets < 0` is the upstream `if max_dets >= 0` guard
        // disabled. `lvis_trim(-1)` must return every input
        // detection unchanged.
        let dts = CocoDetections::from_inputs(
            (0..50)
                .map(|i| dt_input(1, 1, i as f64 / 100.0, (0.0, 0.0, 1.0, 1.0)))
                .collect(),
        )
        .unwrap();
        let trimmed = dts.lvis_trim(-1);
        assert_eq!(trimmed.len(), 50);
        // No reordering — the AC5 path doesn't even sort.
        for (i, &score) in trimmed.scores().iter().enumerate() {
            assert!((score - (i as f64 / 100.0)).abs() < 1e-12);
        }
    }

    #[test]
    fn ac5_max_dets_at_capacity_is_no_op() {
        // `max_dets >= n_dts` keeps every detection — but resorts
        // them score-descending. (We don't assert order preservation
        // because the trim is allowed to reorder; the contract is
        // count + membership.)
        let dts = CocoDetections::from_inputs(
            (0..10)
                .map(|i| dt_input(1, 1, i as f64 / 10.0, (0.0, 0.0, 1.0, 1.0)))
                .collect(),
        )
        .unwrap();
        let trimmed = dts.lvis_trim(100);
        assert_eq!(trimmed.len(), 10);
    }

    #[test]
    fn ac4_stable_sort_preserves_input_order_for_score_ties() {
        // Two detections with the exact same score — the trim must
        // keep them in input order. Python's `sorted(_,
        // reverse=True)` uses Timsort (stable); Rust's `slice::sort_by`
        // is also stable. This test pins the cross-language
        // invariant.
        let mut a = dt_input(1, 1, 0.5, (0.0, 0.0, 1.0, 1.0));
        a.id = Some(AnnId(100));
        let mut b = dt_input(1, 1, 0.5, (1.0, 0.0, 1.0, 1.0));
        b.id = Some(AnnId(200));
        let dts = CocoDetections::from_inputs(vec![a, b]).unwrap();
        let trimmed = dts.lvis_trim(2);
        let ids: Vec<AnnId> = trimmed.records().map(|d| d.id).collect();
        assert_eq!(
            ids,
            vec![AnnId(100), AnnId(200)],
            "AC4: stable sort must preserve input order on score ties"
        );
    }

    #[test]
    fn lvis_trim_groups_by_image_id() {
        // 3 images, each with 5 detections; trim to 2 per image.
        // Verify the group boundaries are honored: image 1 gets its
        // top-2 cat-1 entries, image 2 gets its top-2 cat-2 entries,
        // etc.
        let mut inputs = Vec::with_capacity(15);
        for img in 1..=3i64 {
            for i in 0..5 {
                let score = 1.0 - (img as f64) * 0.01 - (i as f64) * 0.001;
                inputs.push(dt_input(img, img, score, (0.0, 0.0, 1.0, 1.0)));
            }
        }
        let dts = CocoDetections::from_inputs(inputs).unwrap();
        let trimmed = dts.lvis_trim(2);
        assert_eq!(trimmed.len(), 6);
        // 2 per image:
        for img in 1..=3i64 {
            let n = trimmed
                .image_ids()
                .iter()
                .filter(|&&im| im == ImageId(img))
                .count();
            assert_eq!(n, 2, "image {img} must trim to 2");
        }
    }

    #[test]
    fn lvis_trim_zero_max_dets_keeps_nothing() {
        let dts = CocoDetections::from_inputs(vec![
            dt_input(1, 1, 0.9, (0.0, 0.0, 1.0, 1.0)),
            dt_input(1, 1, 0.5, (0.0, 0.0, 1.0, 1.0)),
        ])
        .unwrap();
        let trimmed = dts.lvis_trim(0);
        assert!(trimmed.is_empty());
    }

    #[test]
    fn from_columns_matches_from_inputs_and_rejects_mismatched_lengths() {
        let cols = DetectionColumnsInput {
            ids: None,
            image_ids: vec![ImageId(1), ImageId(1), ImageId(2)],
            category_ids: vec![CategoryId(3), CategoryId(3), CategoryId(5)],
            scores: vec![0.9, 0.4, 0.7],
            bboxes: vec![
                Bbox {
                    x: 1.0,
                    y: 2.0,
                    w: 3.0,
                    h: 4.0,
                },
                Bbox {
                    x: 0.0,
                    y: 0.0,
                    w: 5.0,
                    h: 6.0,
                },
                Bbox {
                    x: 2.0,
                    y: 3.0,
                    w: 1.0,
                    h: 2.0,
                },
            ],
            areas: None,
            segmentations: None,
            keypoints: None,
            num_keypoints: None,
        };
        let dts = CocoDetections::from_columns(cols.clone()).unwrap();
        assert_eq!(dts.len(), 3);
        assert_eq!(dts.id_at(0), AnnId(1));
        assert_eq!(dts.id_at(2), AnnId(3));
        assert_eq!(dts.area_at(0), 12.0);
        assert_eq!(dts.area_at(1), 30.0);
        assert_eq!(dts.indices_for(ImageId(1), CategoryId(3)), &[0, 1]);
        assert_eq!(dts.indices_for(ImageId(2), CategoryId(5)), &[2]);

        let mut bad = cols;
        bad.category_ids.pop();
        let err = CocoDetections::from_columns(bad).unwrap_err();
        assert!(matches!(err, EvalError::InvalidAnnotation { .. }));
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

    // -----------------------------------------------------------------
    // dataset_hash stability tests (ADR-0031)
    // -----------------------------------------------------------------

    #[test]
    fn dataset_hash_is_stable_for_equal_inputs() {
        let a = load_crowd_region();
        let b = load_crowd_region();
        assert_eq!(a.dataset_hash(), b.dataset_hash());
    }

    #[test]
    fn dataset_hash_caches_via_arc_clone() {
        // The cache is `Arc<OnceLock>` so a clone shares the slot. The
        // first call on either side populates it; the second call on
        // the clone should observe the cached value (i.e., equal).
        let a = load_crowd_region();
        let b = a.clone();
        let h1 = a.dataset_hash();
        let h2 = b.dataset_hash();
        assert_eq!(h1, h2);
    }

    #[test]
    fn dataset_hash_invariant_to_image_order() {
        // Two datasets that differ only in image declaration order
        // must hash identically.
        let order_a = r#"{
            "images": [
                {"id": 1, "width": 10, "height": 10},
                {"id": 2, "width": 20, "height": 20}
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1,
                 "bbox": [0, 0, 5, 5], "area": 25, "iscrowd": 0}
            ],
            "categories": [{"id": 1, "name": "x"}]
        }"#;
        let order_b = r#"{
            "images": [
                {"id": 2, "width": 20, "height": 20},
                {"id": 1, "width": 10, "height": 10}
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1,
                 "bbox": [0, 0, 5, 5], "area": 25, "iscrowd": 0}
            ],
            "categories": [{"id": 1, "name": "x"}]
        }"#;
        let a = CocoDataset::from_json_bytes(order_a.as_bytes()).unwrap();
        let b = CocoDataset::from_json_bytes(order_b.as_bytes()).unwrap();
        assert_eq!(a.dataset_hash(), b.dataset_hash());
    }

    #[test]
    fn dataset_hash_invariant_to_annotation_order() {
        let order_a = r#"{
            "images": [{"id": 1, "width": 200, "height": 200}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1,
                 "bbox": [0, 0, 5, 5], "area": 25, "iscrowd": 0},
                {"id": 2, "image_id": 1, "category_id": 1,
                 "bbox": [10, 10, 5, 5], "area": 25, "iscrowd": 0}
            ],
            "categories": [{"id": 1, "name": "x"}]
        }"#;
        let order_b = r#"{
            "images": [{"id": 1, "width": 200, "height": 200}],
            "annotations": [
                {"id": 2, "image_id": 1, "category_id": 1,
                 "bbox": [10, 10, 5, 5], "area": 25, "iscrowd": 0},
                {"id": 1, "image_id": 1, "category_id": 1,
                 "bbox": [0, 0, 5, 5], "area": 25, "iscrowd": 0}
            ],
            "categories": [{"id": 1, "name": "x"}]
        }"#;
        let a = CocoDataset::from_json_bytes(order_a.as_bytes()).unwrap();
        let b = CocoDataset::from_json_bytes(order_b.as_bytes()).unwrap();
        assert_eq!(a.dataset_hash(), b.dataset_hash());
    }

    #[test]
    fn dataset_hash_changes_when_bbox_changes_by_one_pixel() {
        let base = r#"{
            "images": [{"id": 1, "width": 200, "height": 200}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1,
                 "bbox": [10, 10, 5, 5], "area": 25, "iscrowd": 0}
            ],
            "categories": [{"id": 1, "name": "x"}]
        }"#;
        let shifted = r#"{
            "images": [{"id": 1, "width": 200, "height": 200}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1,
                 "bbox": [11, 10, 5, 5], "area": 25, "iscrowd": 0}
            ],
            "categories": [{"id": 1, "name": "x"}]
        }"#;
        let a = CocoDataset::from_json_bytes(base.as_bytes()).unwrap();
        let b = CocoDataset::from_json_bytes(shifted.as_bytes()).unwrap();
        assert_ne!(a.dataset_hash(), b.dataset_hash());
    }

    proptest! {
        #[test]
        fn dataset_hash_invariant_under_id_shuffle(
            mut images in proptest::collection::vec(arb_image(), 1..16),
            categories in proptest::collection::vec(arb_category(), 1..4),
        ) {
            // Dedup images / categories by id — `from_parts` doesn't
            // reject duplicates, but the canonical-form hash is only
            // well-defined over a unique set.
            images.sort_by_key(|im| im.id.0);
            images.dedup_by_key(|im| im.id.0);
            let mut unique_categories = categories;
            unique_categories.sort_by_key(|c| c.id.0);
            unique_categories.dedup_by_key(|c| c.id.0);
            prop_assume!(!images.is_empty());
            prop_assume!(!unique_categories.is_empty());

            // One annotation per image, all on the first category — the
            // shape doesn't matter, only that two datasets that differ
            // solely in declaration order should hash identically.
            let cat_id = unique_categories[0].id;
            let annotations: Vec<CocoAnnotation> = images
                .iter()
                .enumerate()
                .map(|(i, im)| make_min_annotation(AnnId((i as i64) + 1), im.id, cat_id))
                .collect();
            let mut shuffled = images.clone();
            shuffled.reverse();

            let a = CocoDataset::from_parts(
                images,
                annotations.clone(),
                unique_categories.clone(),
            ).unwrap();
            let b = CocoDataset::from_parts(
                shuffled,
                annotations,
                unique_categories,
            ).unwrap();
            prop_assert_eq!(a.dataset_hash(), b.dataset_hash());
        }
    }

    // -----------------------------------------------------------------
    // params_hash stability tests (ADR-0031)
    // -----------------------------------------------------------------

    #[test]
    fn params_hash_is_stable_for_equal_inputs() {
        use crate::evaluate::OwnedEvaluateParams;
        let a = OwnedEvaluateParams {
            iou_thresholds: vec![0.5, 0.55, 0.6],
            area_ranges: vec![],
            max_dets_per_image: 100,
            use_cats: true,
            retain_iou: false,
        };
        let b = a.clone();
        assert_eq!(a.params_hash().unwrap(), b.params_hash().unwrap());
    }

    #[test]
    fn params_hash_changes_when_thresholds_change() {
        use crate::evaluate::OwnedEvaluateParams;
        let a = OwnedEvaluateParams {
            iou_thresholds: vec![0.5, 0.55, 0.6],
            area_ranges: vec![],
            max_dets_per_image: 100,
            use_cats: true,
            retain_iou: false,
        };
        let mut b = a.clone();
        b.iou_thresholds.push(0.65);
        assert_ne!(a.params_hash().unwrap(), b.params_hash().unwrap());
    }

    #[test]
    fn params_hash_changes_when_use_cats_toggles() {
        use crate::evaluate::OwnedEvaluateParams;
        let a = OwnedEvaluateParams {
            iou_thresholds: vec![0.5],
            area_ranges: vec![],
            max_dets_per_image: 100,
            use_cats: true,
            retain_iou: false,
        };
        let mut b = a.clone();
        b.use_cats = false;
        assert_ne!(a.params_hash().unwrap(), b.params_hash().unwrap());
    }

    /// Each group lists its rows in ascending order — the push order the
    /// per-key `Vec<usize>` maps had, which matching's stable tie-breaks
    /// rely on — including when a key's runs are interleaved with others
    /// (the run-length cache's miss path).
    #[test]
    fn csr_index_groups_rows_in_ascending_order() {
        let keys = [3_i64, 3, 1, 3, 2, 2, 1, 3];
        let index = CsrIndex::build(keys.len(), 0, |i| keys[i]);
        assert_eq!(index.get(&3), &[0, 1, 3, 7]);
        assert_eq!(index.get(&1), &[2, 6]);
        assert_eq!(index.get(&2), &[4, 5]);
        assert!(index.get(&9).is_empty());
        assert_eq!(index.max_group_len(), 4);
        let mut distinct: Vec<i64> = index.keys().copied().collect();
        distinct.sort_unstable();
        assert_eq!(distinct, [1, 2, 3]);

        let empty = CsrIndex::<i64>::build(0, 0, |_| 0);
        assert!(empty.get(&0).is_empty());
        assert_eq!(empty.max_group_len(), 0);
    }

    /// `distinct_image_ids` is exactly the set of detection image ids, and
    /// `trim_for` trims a federated set only past the per-image cap.
    #[test]
    fn distinct_image_ids_and_the_trim_boundary() {
        let det = |image: i64, score: f64| DetectionInput {
            id: None,
            image_id: ImageId(image),
            category_id: CategoryId(1),
            score,
            bbox: Bbox {
                x: 0.0,
                y: 0.0,
                w: 1.0,
                h: 1.0,
            },
            area: None,
            segmentation: None,
            keypoints: None,
            num_keypoints: None,
        };
        // Image 1 carries 3 detections, image 2 carries 1.
        let inputs = vec![det(1, 0.9), det(2, 0.8), det(1, 0.7), det(1, 0.6)];
        let dt = CocoDetections::from_inputs(inputs).unwrap();
        let mut ids: Vec<i64> = dt.distinct_image_ids().map(|i| i.0).collect();
        ids.sort_unstable();
        assert_eq!(ids, [1, 2]);

        let lvis = CocoDataset::from_lvis_json_bytes(LVIS_MIN_VALID.as_bytes()).unwrap();
        assert_eq!(dt.clone().trim_for(&lvis, 3).len(), 4, "at the cap");
        let trimmed = dt.clone().trim_for(&lvis, 2);
        assert_eq!(trimmed.len(), 3, "one past the cap");
        assert_eq!(trimmed.indices_for_image(ImageId(1)).len(), 2);
        // Off-LVIS, nothing is trimmed whatever the cap.
        let coco = load_crowd_region();
        assert_eq!(dt.trim_for(&coco, 1).len(), 4);
    }

    /// The grid `I` axis is id-ascending whatever the declaration order,
    /// keeps declaration order among repeated ids (stable), and is
    /// computed once and shared by clones.
    #[test]
    fn grid_images_are_id_ascending_stable_and_shared() {
        let img = |id: i64, width: u32| ImageMeta {
            id: ImageId(id),
            file_name: None,
            width,
            height: 1,
        };
        let ds = CocoDataset::from_parts(
            vec![img(7, 1), img(2, 2), img(7, 3), img(-1, 4)],
            Vec::new(),
            vec![CategoryMeta {
                id: CategoryId(1),
                name: "x".into(),
                supercategory: None,
            }],
        )
        .unwrap();
        let order: Vec<(i64, u32)> = ds.grid_images().iter().map(|m| (m.id.0, m.width)).collect();
        assert_eq!(order, [(-1, 4), (2, 2), (7, 1), (7, 3)]);
        let clone = ds.clone();
        assert!(std::ptr::eq(
            ds.grid_images().iter().next().unwrap(),
            clone.grid_images().iter().next().unwrap()
        ));
        assert!(Arc::ptr_eq(
            &ds.cached_image_order,
            &clone.cached_image_order
        ));
    }

    /// Validation reports the first offending annotation in input order —
    /// image checked before category — whatever the index's key order.
    #[test]
    fn rejects_the_first_offending_annotation_in_input_order() {
        const BAD: &str = r#"{
            "images": [{"id": 1, "width": 10, "height": 10}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1,
                 "bbox": [0, 0, 1, 1], "area": 1, "iscrowd": 0},
                {"id": 2, "image_id": 1, "category_id": 7,
                 "bbox": [0, 0, 1, 1], "area": 1, "iscrowd": 0},
                {"id": 3, "image_id": 99, "category_id": 8,
                 "bbox": [0, 0, 1, 1], "area": 1, "iscrowd": 0},
                {"id": 4, "image_id": 98, "category_id": 1,
                 "bbox": [0, 0, 1, 1], "area": 1, "iscrowd": 0}
            ],
            "categories": [{"id": 1, "name": "thing"}]
        }"#;
        let EvalError::InvalidAnnotation { detail } =
            CocoDataset::from_json_bytes(BAD.as_bytes()).unwrap_err()
        else {
            panic!("expected InvalidAnnotation");
        };
        assert_eq!(detail, "annotation id=2 references unknown category_id=7");
        // An annotation wrong on both counts reports its image.
        let both = BAD.replacen(
            r#""image_id": 1, "category_id": 7"#,
            r#""image_id": 5, "category_id": 7"#,
            1,
        );
        let EvalError::InvalidAnnotation { detail } =
            CocoDataset::from_json_bytes(both.as_bytes()).unwrap_err()
        else {
            panic!("expected InvalidAnnotation");
        };
        assert_eq!(detail, "annotation id=2 references unknown image_id=5");
    }
}
