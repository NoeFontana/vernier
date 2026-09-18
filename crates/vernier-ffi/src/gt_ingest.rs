//! Columnar ground-truth ingest: the array route that carries a COCO GT
//! *document* without going through JSON (ADR-0060).
//!
//! This is the GT-side sibling of [`crate::array_ingest`] (ADR-0030) and
//! [`crate::result_ingest`] (ADR-0057), and it is built on the same
//! load-bearing principle: **the route produces the parts and stops**.
//! [`gt_parts_from_arrays`] terminates in the exact triple
//! `(Vec<ImageMeta>, Vec<CocoAnnotation>, Vec<CategoryMeta>)` that
//! `serde_json` produces from a GT file, and hands it to
//! [`CocoDataset::from_parts`] — the one constructor the JSON route also
//! ends at. Nothing downstream can tell which route produced a dataset,
//! because after this module there is no route: there is one
//! `CocoDataset`.
//!
//! That is what keeps the GT parity surface from widening. Every
//! GT-specific rule stays where it already lives and is not restated
//! here:
//!
//! - **D1** — the `ignore` / `iscrowd` overwrite — is resolved by
//!   `CocoAnnotation::effective_ignore` at eval time, from the
//!   `is_crowd` and `ignore_flag` this module fills in verbatim.
//! - **A4** — GTs sorted ascending by `_ignore` — happens in the
//!   matching engine, on whatever order this module produced.
//! - **E1** — the crowd IoA denominator — reads `is_crowd`.
//! - **D2** — the zero-visible-keypoints implicit ignore — reads
//!   `num_keypoints`.
//! - Reference integrity (an annotation naming an unknown image or
//!   category) is `from_parts`' check, not ours.
//!
//! GT `area` is carried **verbatim**. Unlike a detection's (quirk
//! **J3**, derived by default), a ground-truth annotation's `area` is
//! the number COCO recorded and the number the small/medium/large
//! bucketing reads, so `area` is a *required* column here rather than
//! something this route may compute.
//!
//! GT annotation ids are likewise **supplied, never assigned** — the
//! mirror image of quirk **J1** on the detection side. `id` is required,
//! and it is observable downstream through `evalImgs['gtIds']` and
//! through the `dtMatches` entries that carry a matched GT's id.
//!
//! # Absent versus zero
//!
//! A columnar array has no null, but two COCO GT fields are genuinely
//! *optional per annotation* and mean something different when absent
//! than when present-and-zero: `ignore` (see **D1** — absent lets
//! `corrected` fall back to `iscrowd`, present-and-0 pins it false) and
//! `num_keypoints`. One rule covers both:
//!
//! > An optional integer column may be passed as a **signed** array, in
//! > which a **negative entry means the field was absent on that
//! > annotation**. Omitting the column entirely means absent on every
//! > annotation.
//!
//! Neither field has a meaningful negative value otherwise — `ignore` is
//! a flag and `num_keypoints` is a count — so the encoding is
//! unambiguous, and it lets the array route express every GT document
//! the JSON route can, including one that carries `ignore` on some
//! annotations and not others.
//!
//! Extraction reads Python objects and therefore cannot release the GIL.
//! It is written as one pass per column so the serial floor it imposes
//! is as short as it can be; the caller detaches immediately after and
//! runs `from_parts` — the O(N) index build — without the GIL.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PySequence, PyString};

use vernier_core::dataset::{
    AnnId, Bbox, CategoryId, CategoryMeta, CocoAnnotation, ImageId, ImageMeta,
};
use vernier_core::segmentation::Segmentation;

use crate::array_ingest::{type_name_of, CastCtx, CountsShapes, FieldPath};
use crate::dlpack;
use crate::result_ingest::extract_segmentation_with_root;

/// The parts a GT document decomposes into — exactly what
/// `CocoDataset::from_parts` consumes, and exactly what `serde_json`
/// produces from a GT file.
pub(crate) type GtParts = (Vec<ImageMeta>, Vec<CocoAnnotation>, Vec<CategoryMeta>);

const IMAGES: &str = "images";
const ANNOTATIONS: &str = "annotations";
const CATEGORIES: &str = "categories";

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Convert the three columnar GT sections into [`GtParts`].
///
/// `images` and `annotations` are dicts of equal-length columns;
/// `categories` is a sequence of small per-category dicts (see the
/// module docs and ADR-0060 for why that one section is not columnar).
pub(crate) fn gt_parts_from_arrays<'py>(
    images: &Bound<'py, PyAny>,
    annotations: &Bound<'py, PyAny>,
    categories: &Bound<'py, PyAny>,
    ctx: &CastCtx<'py, '_>,
) -> PyResult<GtParts> {
    let images = extract_images(images, ctx)?;
    let categories = extract_categories(categories)?;
    let annotations = extract_annotations(annotations, ctx)?;
    Ok((images, annotations, categories))
}

// ---------------------------------------------------------------------------
// images
// ---------------------------------------------------------------------------

fn extract_images<'py>(
    obj: &Bound<'py, PyAny>,
    ctx: &CastCtx<'py, '_>,
) -> PyResult<Vec<ImageMeta>> {
    let dict = as_column_dict(obj, IMAGES)?;
    let py = dict.py();

    let id_obj = required_column(&dict, intern!(py, "id"), IMAGES, "id")?;
    let ids = i64_column(&id_obj, IMAGES, ".id", ctx)?;
    let n = ids.len();

    let w_obj = required_column(&dict, intern!(py, "width"), IMAGES, "width")?;
    let widths = i64_column(&w_obj, IMAGES, ".width", ctx)?;
    check_len(widths.len(), n, IMAGES, ".width", "id")?;

    let h_obj = required_column(&dict, intern!(py, "height"), IMAGES, "height")?;
    let heights = i64_column(&h_obj, IMAGES, ".height", ctx)?;
    check_len(heights.len(), n, IMAGES, ".height", "id")?;

    // `file_name` is the one image field with no array form; it is a
    // sequence of `str` when supplied and absent on every image when
    // not. Nothing in the eval reads it, but it participates in the
    // ADR-0031 `dataset_hash`, so a caller reproducing a JSON dataset
    // through this route needs to be able to carry it.
    let file_names = match optional_column(&dict, intern!(py, "file_name"))? {
        None => None,
        Some(o) => {
            let seq = as_object_sequence(&o, IMAGES, ".file_name")?;
            let len = seq.len()?;
            check_len(len, n, IMAGES, ".file_name", "id")?;
            let mut names = Vec::with_capacity(len);
            for i in 0..len {
                let item = seq.get_item(i)?;
                if item.is_none() {
                    names.push(None);
                    continue;
                }
                names.push(Some(item.extract::<String>().map_err(|e| {
                    PyTypeError::new_err(format!(
                        "{}: expected a str or None: {e}",
                        FieldPath::rooted(IMAGES, Some(i), ".file_name")
                    ))
                })?));
            }
            Some(names)
        }
    };

    let ids = ids.as_slice();
    let widths = widths.as_slice();
    let heights = heights.as_slice();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        // Never truncate: `ImageMeta` stores u32, the column arrives as
        // i64, and a negative or oversized value is the caller's bug,
        // not something to wrap around silently.
        let width = exact_u32(widths[i], IMAGES, i, ".width")?;
        let height = exact_u32(heights[i], IMAGES, i, ".height")?;
        out.push(ImageMeta {
            id: ImageId(ids[i]),
            width,
            height,
            file_name: file_names.as_ref().and_then(|v| v[i].clone()),
        });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// categories
// ---------------------------------------------------------------------------

/// The `categories` section, as a sequence of per-category dicts.
///
/// This section is deliberately *not* columnar. `name` and
/// `supercategory` are strings, which have no array form, so a columnar
/// spelling would be an `id` array beside two Python lists — three
/// objects to keep in step for a section that is 80 entries on COCO and
/// 1203 on LVIS. The per-annotation dict cost this ADR exists to remove
/// is an O(N)-in-annotations cost; categories are O(K) and K is small.
fn extract_categories(obj: &Bound<'_, PyAny>) -> PyResult<Vec<CategoryMeta>> {
    let seq = as_object_sequence(obj, CATEGORIES, "")?;
    let py = obj.py();
    let k_id = intern!(py, "id");
    let k_name = intern!(py, "name");
    let k_super = intern!(py, "supercategory");

    let len = seq.len()?;
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let item = seq.get_item(i)?;
        let dict = item.cast::<PyDict>().map_err(|_| {
            PyTypeError::new_err(format!(
                "{}: expected a dict with 'id' and 'name', got {}",
                FieldPath::rooted(CATEGORIES, Some(i), ""),
                type_name_of(&item)
            ))
        })?;
        let id: i64 = dict
            .get_item(k_id)?
            .ok_or_else(|| cat_err(i, ".id", "missing required field"))?
            .extract()
            .map_err(|e| cat_err(i, ".id", &format!("expected int: {e}")))?;
        let name: String = dict
            .get_item(k_name)?
            .ok_or_else(|| cat_err(i, ".name", "missing required field"))?
            .extract()
            .map_err(|e| cat_err(i, ".name", &format!("expected str: {e}")))?;
        let supercategory = match dict.get_item(k_super)? {
            Some(v) if !v.is_none() => Some(
                v.extract::<String>()
                    .map_err(|e| cat_err(i, ".supercategory", &format!("expected str: {e}")))?,
            ),
            _ => None,
        };
        out.push(CategoryMeta {
            id: CategoryId(id),
            name,
            supercategory,
        });
    }
    Ok(out)
}

fn cat_err(i: usize, name: &str, detail: &str) -> PyErr {
    PyValueError::new_err(format!(
        "{}: {detail}",
        FieldPath::rooted(CATEGORIES, Some(i), name)
    ))
}

// ---------------------------------------------------------------------------
// annotations
// ---------------------------------------------------------------------------

fn extract_annotations<'py>(
    obj: &Bound<'py, PyAny>,
    ctx: &CastCtx<'py, '_>,
) -> PyResult<Vec<CocoAnnotation>> {
    let dict = as_column_dict(obj, ANNOTATIONS)?;
    let py = dict.py();

    // `id` is required: GT ids are supplied, never assigned. This is the
    // mirror image of quirk J1, and it is observable through
    // `evalImgs['gtIds']`.
    let id_obj = required_column(&dict, intern!(py, "id"), ANNOTATIONS, "id")?;
    let ids = i64_column(&id_obj, ANNOTATIONS, ".id", ctx)?;
    let n = ids.len();

    let img_obj = required_column(&dict, intern!(py, "image_id"), ANNOTATIONS, "image_id")?;
    let image_ids = i64_column(&img_obj, ANNOTATIONS, ".image_id", ctx)?;
    check_len(image_ids.len(), n, ANNOTATIONS, ".image_id", "id")?;

    let cat_obj = required_column(
        &dict,
        intern!(py, "category_id"),
        ANNOTATIONS,
        "category_id",
    )?;
    let cat_ids = i64_column(&cat_obj, ANNOTATIONS, ".category_id", ctx)?;
    check_len(cat_ids.len(), n, ANNOTATIONS, ".category_id", "id")?;

    let bbox_obj = required_column(&dict, intern!(py, "bbox"), ANNOTATIONS, "bbox")?;
    let bbox_path = FieldPath::rooted(ANNOTATIONS, None, ".bbox").to_string();
    let bbox_obj = ctx.maybe_cast(bbox_obj, &bbox_path, "float64")?;
    let bboxes = dlpack::extract_f64_2d(&bbox_obj, &bbox_path, 4)?;
    check_len(bboxes.len() / 4, n, ANNOTATIONS, ".bbox", "id")?;

    // GT `area` is read verbatim — it is *not* derived from the bbox the
    // way a detection's is (quirk J3). Required, because there is no
    // honest default: silently substituting `w * h` would re-bucket
    // every polygon GT between AP-small / medium / large.
    let area_obj = required_column(&dict, intern!(py, "area"), ANNOTATIONS, "area")?;
    let area_path = FieldPath::rooted(ANNOTATIONS, None, ".area").to_string();
    let area_obj = ctx.maybe_cast(area_obj, &area_path, "float64")?;
    let areas = dlpack::extract_f64_1d(&area_obj, &area_path)?;
    check_len(areas.len(), n, ANNOTATIONS, ".area", "id")?;

    // `iscrowd` drives `gt_ignore` (D1) and the IoA denominator (E1). It
    // is not optional: absent-and-false is expressible as an all-zero
    // column, and defaulting it would make a crowd dataset silently
    // score as a non-crowd one.
    let crowd_obj = required_column(&dict, intern!(py, "iscrowd"), ANNOTATIONS, "iscrowd")?;
    let crowds = FlagColumn::extract(&crowd_obj, ANNOTATIONS, ".iscrowd")?;
    check_len(crowds.len(), n, ANNOTATIONS, ".iscrowd", "id")?;

    // `ignore` is optional *per annotation* (see the module docs on
    // absent-versus-zero). Absent is what lets `parity_mode="corrected"`
    // fall back to `iscrowd` under D1; present-and-0 pins it false.
    let ignores = match optional_column(&dict, intern!(py, "ignore"))? {
        None => None,
        Some(o) => {
            let col = FlagColumn::extract(&o, ANNOTATIONS, ".ignore")?;
            check_len(col.len(), n, ANNOTATIONS, ".ignore", "id")?;
            Some(col)
        }
    };

    let segmentations = match optional_column(&dict, intern!(py, "segmentation"))? {
        None => None,
        Some(o) => {
            let segs = extract_segmentation_column(&o, n, ctx)?;
            Some(segs)
        }
    };

    let (keypoints, kp_k) = match optional_column(&dict, intern!(py, "keypoints"))? {
        None => (None, 0),
        Some(o) => {
            let path = FieldPath::rooted(ANNOTATIONS, None, ".keypoints").to_string();
            let o = ctx.maybe_cast(o, &path, "float64")?;
            let (view, k) = dlpack::extract_f64_3d_kp(&o, &path)?;
            check_len(
                view.len() / (k * 3).max(1),
                n,
                ANNOTATIONS,
                ".keypoints",
                "id",
            )?;
            (Some(view), k)
        }
    };

    let num_keypoints = match optional_column(&dict, intern!(py, "num_keypoints"))? {
        None => None,
        Some(o) => {
            let col = i64_column(&o, ANNOTATIONS, ".num_keypoints", ctx)?;
            check_len(col.len(), n, ANNOTATIONS, ".num_keypoints", "id")?;
            Some(col)
        }
    };

    let ids = ids.as_slice();
    let image_ids = image_ids.as_slice();
    let cat_ids = cat_ids.as_slice();
    let bboxes = bboxes.as_slice();
    let areas = areas.as_slice();
    let kp = keypoints.as_ref().map(dlpack::DLPackView::as_slice);

    let mut out = Vec::with_capacity(n);
    let mut segmentations = segmentations;
    for i in 0..n {
        let b = &bboxes[i * 4..i * 4 + 4];
        // JSON has no NaN or Infinity literal, so a GT *file* can never
        // carry one: refusing them here is not a route divergence, it is
        // refusing input the JSON route could not have handed downstream
        // in the first place. Letting one through is how a NaN box ends
        // up scoring a perfect IoU against everything.
        for (j, &v) in b.iter().enumerate() {
            finite(v, ANNOTATIONS, i, ".bbox", Some(j))?;
        }
        finite(areas[i], ANNOTATIONS, i, ".area", None)?;

        let keypoints = match kp {
            None => None,
            Some(flat) => {
                let row = &flat[i * k3(kp_k)..(i + 1) * k3(kp_k)];
                for (j, &v) in row.iter().enumerate() {
                    finite(v, ANNOTATIONS, i, ".keypoints", Some(j))?;
                }
                Some(row.to_vec())
            }
        };

        out.push(CocoAnnotation {
            id: AnnId(ids[i]),
            image_id: ImageId(image_ids[i]),
            category_id: CategoryId(cat_ids[i]),
            area: areas[i],
            is_crowd: crowds.get(i),
            ignore_flag: match &ignores {
                None => None,
                Some(col) => col.get_optional(i),
            },
            bbox: Bbox {
                x: b[0],
                y: b[1],
                w: b[2],
                h: b[3],
            },
            segmentation: segmentations.as_mut().and_then(|s| s[i].take()),
            keypoints,
            num_keypoints: match &num_keypoints {
                None => None,
                Some(col) => {
                    let v = col.as_slice()[i];
                    if v < 0 {
                        None
                    } else {
                        Some(u32::try_from(v).map_err(|_| {
                            ann_err(
                                i,
                                ".num_keypoints",
                                &format!("{v} does not fit in a u32 count"),
                            )
                        })?)
                    }
                }
            },
        });
    }
    Ok(out)
}

#[inline]
fn k3(k: usize) -> usize {
    k * 3
}

/// The `segmentation` column: one entry per annotation, each entry
/// `None` or any shape a GT *file* carries.
///
/// A segmentation has no flat numeric form — a polygon ring is
/// variable-length and a compressed RLE is a string — so this column is
/// a sequence of per-annotation objects rather than an array. That is
/// not the cost this ADR removes: the caller already *holds* these
/// objects (they came out of their own GT structure); what it no longer
/// does is serialize them to text and parse them back.
fn extract_segmentation_column<'py>(
    obj: &Bound<'py, PyAny>,
    n: usize,
    ctx: &CastCtx<'py, '_>,
) -> PyResult<Vec<Option<Segmentation>>> {
    let seq = as_object_sequence(obj, ANNOTATIONS, ".segmentation")?;
    let len = seq.len()?;
    check_len(len, n, ANNOTATIONS, ".segmentation", "id")?;
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let item = seq.get_item(i)?;
        if item.is_none() {
            out.push(None);
            continue;
        }
        out.push(Some(extract_segmentation_with_root(
            &item,
            ANNOTATIONS,
            i,
            ".segmentation",
            ctx,
            CountsShapes::AlsoJsonWire,
        )?));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Column helpers
// ---------------------------------------------------------------------------

/// A `bool` / `uint8` / signed-int flag column, kept in whichever form
/// it arrived so the "negative means absent" rule can be applied only
/// where it is expressible.
enum FlagColumn<'py> {
    /// One byte per entry. Unsigned, so every entry is *present*;
    /// truthiness is `!= 0`, matching pycocotools' coercion and the
    /// JSON route's `deserialize_bool_int`.
    Bytes(dlpack::DLPackView<'py, u8>),
    /// Signed. A negative entry means the field was absent on that
    /// annotation; otherwise truthiness is `!= 0`.
    Signed(dlpack::DLPackView<'py, i64>),
}

impl<'py> FlagColumn<'py> {
    fn extract(obj: &Bound<'py, PyAny>, root: &str, name: &str) -> PyResult<Self> {
        let path = FieldPath::rooted(root, None, name).to_string();
        // bool / uint8 first: it is the natural spelling and the common
        // case. int64 is the spelling that can also say "absent".
        if let Ok(v) = dlpack::extract_u8_or_bool_1d(obj, &path) {
            return Ok(Self::Bytes(v));
        }
        match dlpack::extract_i64_1d(obj, &path) {
            Ok(v) => Ok(Self::Signed(v)),
            Err(_) => Err(PyTypeError::new_err(format!(
                "{path}: expected a 1-D bool, uint8 or int64 array, got {}; \
                 int64 is the spelling that can also carry \"absent\" \
                 (a negative entry)",
                type_name_of(obj)
            ))),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Bytes(v) => v.len(),
            Self::Signed(v) => v.len(),
        }
    }

    /// Truthiness, treating an absent entry as false. Used for
    /// `iscrowd`, where absent and false are the same thing (a GT
    /// without an `iscrowd` key deserializes to `false`).
    fn get(&self, i: usize) -> bool {
        match self {
            Self::Bytes(v) => v.as_slice()[i] != 0,
            Self::Signed(v) => v.as_slice()[i] > 0,
        }
    }

    /// Truthiness with absence preserved. Used for `ignore`, where
    /// absent and false are *not* the same thing under D1.
    fn get_optional(&self, i: usize) -> Option<bool> {
        match self {
            Self::Bytes(v) => Some(v.as_slice()[i] != 0),
            Self::Signed(v) => {
                let x = v.as_slice()[i];
                if x < 0 {
                    None
                } else {
                    Some(x != 0)
                }
            }
        }
    }
}

fn i64_column<'py>(
    obj: &Bound<'py, PyAny>,
    root: &str,
    name: &str,
    ctx: &CastCtx<'py, '_>,
) -> PyResult<dlpack::DLPackView<'py, i64>> {
    let path = FieldPath::rooted(root, None, name).to_string();
    let obj = ctx.maybe_cast(obj.clone(), &path, "int64")?;
    dlpack::extract_i64_1d(&obj, &path)
}

/// The `images=` / `annotations=` argument must be a mapping of columns.
fn as_column_dict<'py>(obj: &Bound<'py, PyAny>, root: &str) -> PyResult<Bound<'py, PyDict>> {
    obj.cast::<PyDict>().cloned().map_err(|_| {
        PyTypeError::new_err(format!(
            "{root}: expected a dict of columns, got {}",
            type_name_of(obj)
        ))
    })
}

/// A sequence of arbitrary Python objects (`segmentation`, `file_name`,
/// `categories`).
///
/// An array is **refused** even though it satisfies the sequence
/// protocol. NumPy arrays and torch tensors are `Sequence`s, so a
/// caller who passed a stacked `(N, H, W)` bitmask array here would have
/// it silently iterated into N 2-D planes — a plausible-looking result
/// from a payload this column does not accept. `str` is refused for the
/// same reason: it is a sequence of one-character strings.
fn as_object_sequence<'py>(
    obj: &Bound<'py, PyAny>,
    root: &str,
    name: &str,
) -> PyResult<Bound<'py, PySequence>> {
    let path = FieldPath::rooted(root, None, name);
    if obj.is_instance_of::<PyString>() || obj.hasattr("__dlpack_device__")? {
        return Err(PyTypeError::new_err(format!(
            "{path}: expected a list, got {}; this column holds one \
             Python object per entry and is not an array column",
            type_name_of(obj)
        )));
    }
    obj.cast::<PySequence>().cloned().map_err(|_| {
        PyTypeError::new_err(format!(
            "{path}: expected a sequence, got {}",
            type_name_of(obj)
        ))
    })
}

fn required_column<'py>(
    dict: &Bound<'py, PyDict>,
    key: &Bound<'py, PyString>,
    root: &str,
    name: &str,
) -> PyResult<Bound<'py, PyAny>> {
    match dict.get_item(key)? {
        Some(v) if !v.is_none() => Ok(v),
        _ => Err(PyValueError::new_err(format!(
            "{root}: missing required column '{name}'"
        ))),
    }
}

fn optional_column<'py>(
    dict: &Bound<'py, PyDict>,
    key: &Bound<'py, PyString>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    Ok(dict.get_item(key)?.filter(|v| !v.is_none()))
}

fn check_len(got: usize, want: usize, root: &str, name: &str, against: &str) -> PyResult<()> {
    if got == want {
        return Ok(());
    }
    Err(PyValueError::new_err(format!(
        "{}: length {got} disagrees with {root}['{against}'] (N={want})",
        FieldPath::rooted(root, None, name)
    )))
}

fn ann_err(i: usize, name: &str, detail: &str) -> PyErr {
    PyValueError::new_err(format!(
        "{}: {detail}",
        FieldPath::rooted(ANNOTATIONS, Some(i), name)
    ))
}

/// Reject a non-finite float, naming the offending element.
fn finite(v: f64, root: &str, i: usize, name: &str, elem: Option<usize>) -> PyResult<()> {
    if v.is_finite() {
        return Ok(());
    }
    let path = FieldPath::rooted(root, Some(i), name);
    let path = match elem {
        Some(j) => path.item(j),
        None => path,
    };
    Err(PyValueError::new_err(format!(
        "{path}: expected a finite float, got {v}; a GT JSON file cannot \
         carry NaN or Infinity, so neither does this route"
    )))
}

/// `i64` → `u32` without truncation. Image `width` / `height` arrive as
/// an int64 column because that is the dtype a caller's index arrays
/// already are; `ImageMeta` stores them as `u32`.
fn exact_u32(v: i64, root: &str, i: usize, name: &str) -> PyResult<u32> {
    u32::try_from(v).map_err(|_| {
        PyValueError::new_err(format!(
            "{}: {v} is out of range for a pixel dimension (0..=4294967295)",
            FieldPath::rooted(root, Some(i), name)
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Rendering a `PyErr` needs a live interpreter, which these unit
    // tests do not have (`auto-initialize` is off), and every other
    // helper here takes a `Bound`. The decisions that are pure
    // arithmetic are pinned here; the *text* of each rejection and the
    // route-equivalence property are asserted from Python, in
    // `tests/python/test_gt_ingest_route_equivalence.py`.

    #[test]
    fn exact_u32_accepts_pixel_dimensions() -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(exact_u32(0, IMAGES, 0, ".width")?, 0);
        assert_eq!(exact_u32(640, IMAGES, 0, ".width")?, 640);
        assert_eq!(
            exact_u32(i64::from(u32::MAX), IMAGES, 0, ".width")?,
            u32::MAX
        );
        Ok(())
    }

    #[test]
    fn exact_u32_refuses_to_truncate() {
        assert!(exact_u32(-1, IMAGES, 0, ".width").is_err());
        assert!(exact_u32(i64::from(u32::MAX) + 1, IMAGES, 0, ".height").is_err());
        assert!(exact_u32(i64::MIN, IMAGES, 0, ".width").is_err());
    }

    #[test]
    fn finite_rejects_nan_and_infinities() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                finite(bad, ANNOTATIONS, 0, ".bbox", Some(2)).is_err(),
                "{bad} must be rejected"
            );
        }
        assert!(finite(0.0, ANNOTATIONS, 0, ".area", None).is_ok());
        assert!(finite(-1.5, ANNOTATIONS, 0, ".bbox", Some(0)).is_ok());
    }

    #[test]
    fn keypoint_row_width_is_three_per_joint() {
        assert_eq!(k3(17), 51);
        assert_eq!(k3(0), 0);
    }
}
