//! Array-based detection ingest for `StreamingEvaluator.update` and
//! `BackgroundEvaluator.submit` (ADR-0030, ADR-0066).
//!
//! Routes Python `Detections` and `DetectionsBatch` dicts through the same
//! `CocoDetections::from_columns` constructor the JSON path uses; parity
//! is structural, not a separate invariant.

use std::cell::OnceCell;
use std::sync::atomic::{AtomicBool, Ordering};

use pyo3::exceptions::{PyTypeError, PyUserWarning, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::{
    PyAny, PyByteArray, PyBytes, PyDict, PyList, PyMemoryView, PySequence, PyString, PyTuple,
};

use vernier_core::dataset::{Bbox, CategoryId, DetectionColumnsInput, ImageId};
use vernier_core::segmentation::{Segmentation, SegmentationRle, SegmentationRleCounts};
use vernier_core::EvalError;
use vernier_mask::Rle;

use crate::dlpack;
use crate::emit_warning;
use crate::eval_error_to_pyerr;

// ---------------------------------------------------------------------------
// Top-level dispatch types
// ---------------------------------------------------------------------------

/// One incoming `update`/`submit` argument. The two variants without
/// payload data (`Bytes` and the dict-shape branches) are distinguished
/// at the call site so the array path doesn't carry a never-used `Bytes`
/// arm into its kernel-typed dispatch.
pub(crate) enum DetectionsArg<'py> {
    /// Legacy `loadRes`-shaped JSON bytes; routed straight to the
    /// existing JSON entry on the streaming/background state.
    /// Held as a [`PyBackedBytes`] so the underlying `bytes` buffer
    /// crosses `py.detach` without a copy — saves ~10 ms per val2017
    /// DT submission (17 MB jittered output).
    Bytes(PyBackedBytes),
    /// Multi-image / whole-dataset columnar `DetectionsBatch` dict (ADR-0066).
    Batch(Bound<'py, PyDict>),
    /// One or more per-image `Detections` dicts. Single-image inputs
    /// land here as a one-element vec.
    Dicts {
        dicts: Vec<Bound<'py, PyDict>>,
        /// `true` when the argument was a *sequence* of dicts, so a
        /// rejection can name the offending element (`detections[3]`).
        /// `false` for a single bare dict, where an index would be noise.
        indexed: bool,
    },
    /// A list of per-annotation COCO *result* dicts — the shape
    /// `loadRes` consumes and TorchMetrics-style callers build. Handled
    /// by `result_ingest::ann_dicts_to_columns`.
    AnnList(Vec<Bound<'py, PyDict>>),
    /// An `(N, 7)` C-contiguous float64 array laid out as
    /// `image_id, x, y, w, h, score, category_id`.
    Matrix(Bound<'py, PyAny>),
}

impl<'py> DetectionsArg<'py> {
    /// Classify the input. `bytes` is checked first because `bytes`
    /// instances also satisfy the `Sequence` protocol.
    pub(crate) fn extract(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(b) = obj.cast::<PyBytes>() {
            return Ok(Self::Bytes(PyBackedBytes::from(b.clone())));
        }
        if let Ok(d) = obj.cast::<PyDict>() {
            let d = d.clone();
            let py = d.py();
            let has_image_ids = d.contains(intern!(py, "image_ids"))?;
            let has_image_id = d.contains(intern!(py, "image_id"))?;
            if has_image_ids && has_image_id {
                return Err(PyValueError::new_err(
                    "detections: cannot specify both 'image_id' (single-image Detections) \
                     and 'image_ids' (multi-image DetectionsBatch)",
                ));
            }
            if has_image_ids {
                return Ok(Self::Batch(d));
            }
            return Ok(if crate::result_ingest::dict_is_result_annotation(&d)? {
                Self::AnnList(vec![d])
            } else {
                Self::Dicts {
                    dicts: vec![d],
                    indexed: false,
                }
            });
        }
        // The `(N, 7)` matrix must be probed *before* the sequence
        // branch: numpy arrays and torch tensors satisfy the `Sequence`
        // protocol, so an array reaching the loop below would be
        // iterated row by row and rejected as "not a dict".
        if crate::result_ingest::looks_like_matrix(obj)? {
            return Ok(Self::Matrix(obj.clone()));
        }
        if let Ok(seq) = obj.cast::<PySequence>() {
            let len = seq.len()?;
            let mut dicts: Vec<Bound<'py, PyDict>> = Vec::with_capacity(len);
            for i in 0..len {
                let item = seq.get_item(i)?;
                let dict = item.cast_into::<PyDict>().map_err(|e| {
                    PyTypeError::new_err(format!(
                        "detections[{i}]: expected a Detections dict: {e}"
                    ))
                })?;
                dicts.push(dict);
            }
            // The list is homogeneous by construction in every caller we
            // support, so the first element decides the route. `boxes`
            // (columnar) wins over `bbox` (per-annotation), so an
            // ADR-0030 payload can never be re-routed here.
            if let Some(first) = dicts.first() {
                if first.contains(intern!(first.py(), "image_ids"))? {
                    return Err(PyValueError::new_err(
                        "detections[0]: got 'image_ids' inside a sequence; pass a \
                         DetectionsBatch dict directly rather than wrapping it in a list",
                    ));
                }
            }
            let is_ann = match dicts.first() {
                Some(first) => crate::result_ingest::dict_is_result_annotation(first)?,
                // An empty list is the same empty payload either way.
                None => false,
            };
            return Ok(if is_ann {
                Self::AnnList(dicts)
            } else {
                Self::Dicts {
                    dicts,
                    indexed: true,
                }
            });
        }
        let type_name = type_name_of(obj);
        Err(PyTypeError::new_err(format!(
            "detections must be bytes, a Detections or DetectionsBatch dict, \
             a sequence of Detections dicts, a list of COCO result dicts, \
             or an (N, 7) float64 array; got {type_name}"
        )))
    }
}

// ---------------------------------------------------------------------------
// Field paths -- rendered only when an error needs them
// ---------------------------------------------------------------------------

/// Where in the `detections=` argument a helper is reading, e.g.
/// `detections.boxes`, `detections[3].segmentation`,
/// `detections[3].rles[7]`.
///
/// Every ingest helper takes one of these purely to *name* the offending
/// field in a rejection. Rendering the path eagerly costs one `String`
/// per annotation on the **success** path -- on a 500k-detection
/// submission that is 500k allocations nothing ever reads.
/// [`std::fmt::Display`] defers the work to the error branch that
/// actually formats it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FieldPath<'a> {
    /// The argument this path is rooted at, so a rejection names the
    /// parameter the caller actually passed. `"detections"` for the
    /// ADR-0030 / ADR-0057 detection routes; the ADR-0060 GT route
    /// roots at `"annotations"` / `"images"` / `"categories"`.
    root: &'a str,
    /// Position within a *sequence* `detections=` argument. `None` when
    /// the argument was a single bare dict or an array, where an index
    /// would be noise rather than a locator.
    ann: Option<usize>,
    /// The field, written with its leading dot (`".boxes"`), or `""` for
    /// the element itself.
    name: &'a str,
    /// Position within a sequence-valued field (`rles`, polygons).
    item: Option<usize>,
}

impl<'a> FieldPath<'a> {
    pub(crate) fn new(ann: Option<usize>, name: &'a str) -> Self {
        Self::rooted("detections", ann, name)
    }

    /// [`Self::new`] with an explicit root argument name.
    pub(crate) fn rooted(root: &'a str, ann: Option<usize>, name: &'a str) -> Self {
        Self {
            root,
            ann,
            name,
            item: None,
        }
    }

    /// The same path, narrowed to one item of a sequence-valued field.
    pub(crate) fn item(self, item: usize) -> Self {
        Self {
            item: Some(item),
            ..self
        }
    }
}

impl std::fmt::Display for FieldPath<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.root)?;
        if let Some(i) = self.ann {
            write!(f, "[{i}]")?;
        }
        f.write_str(self.name)?;
        if let Some(j) = self.item {
            write!(f, "[{j}]")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// IoU type discriminator
// ---------------------------------------------------------------------------

/// Which fields are required on each `Detections` dict, picked at
/// evaluator construction time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ArrayIouType {
    Bbox,
    Segm,
    Boundary,
    Keypoints,
}

impl ArrayIouType {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Bbox => "bbox",
            Self::Segm => "segm",
            Self::Boundary => "boundary",
            Self::Keypoints => "keypoints",
        }
    }
}

// ---------------------------------------------------------------------------
// cast_inputs state — Some(latch) when enabled, None when off. The latch
// is a one-shot AtomicBool that fires the UserWarning at most once over
// the evaluator's lifetime.
// ---------------------------------------------------------------------------

pub(crate) type CastState = Option<AtomicBool>;

pub(crate) fn new_cast_state(enabled: bool) -> CastState {
    if enabled {
        Some(AtomicBool::new(false))
    } else {
        None
    }
}

fn emit_cast_warning_once(py: Python<'_>, latch: &AtomicBool) -> PyResult<()> {
    if latch.swap(true, Ordering::Relaxed) {
        return Ok(());
    }
    emit_warning::<PyUserWarning>(
        py,
        "vernier-0030: cast_inputs=True silently promotes input dtypes (f32→f64, i32→i64); \
         disable to enforce the strict ADR-0004 boundary",
    )
}

// ---------------------------------------------------------------------------
// Per-image extraction
// ---------------------------------------------------------------------------

/// Per-call state threaded through validation helpers. `cast.is_some()`
/// gates the f32→f64 / i32→i64 promotion path. The numpy resolvers are
/// cached so inner loops don't re-walk `sys.modules` per (dict × field).
pub(crate) struct CastCtx<'py, 'a> {
    py: Python<'py>,
    cast: Option<(&'a AtomicBool, &'a Bound<'py, PyAny>)>,
    asfortranarray: OnceCell<Bound<'py, PyAny>>,
}

impl<'py, 'a> CastCtx<'py, 'a> {
    /// Wire a context from the per-call cast state and the resolver
    /// [`resolve_ascontiguousarray`] produced for it (`None` when
    /// `cast_inputs=False`, which is the strict ADR-0004 default).
    fn new(
        py: Python<'py>,
        cast_state: &'a CastState,
        ascontig: &'a Option<Bound<'py, PyAny>>,
    ) -> Self {
        Self {
            py,
            cast: cast_state.as_ref().zip(ascontig.as_ref()),
            asfortranarray: OnceCell::new(),
        }
    }

    pub(crate) fn maybe_cast(
        &self,
        obj: Bound<'py, PyAny>,
        field: &str,
        dtype: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        match self.cast {
            None => Ok(obj),
            Some((latch, ascontig)) => cast_via_numpy(self.py, &obj, field, dtype, latch, ascontig),
        }
    }

    fn asfortranarray(&self) -> PyResult<Bound<'py, PyAny>> {
        if let Some(f) = self.asfortranarray.get() {
            return Ok(f.clone());
        }
        let resolved = resolve_asfortranarray(self.py)?;
        let _ = self.asfortranarray.set(resolved.clone());
        Ok(resolved)
    }
}

struct DictKeys<'py> {
    image_id: &'py Bound<'py, PyString>,
    boxes: &'py Bound<'py, PyString>,
    scores: &'py Bound<'py, PyString>,
    labels: &'py Bound<'py, PyString>,
    rles: &'py Bound<'py, PyString>,
    keypoints: &'py Bound<'py, PyString>,
}

impl<'py> DictKeys<'py> {
    fn intern(py: Python<'py>) -> Self {
        Self {
            image_id: intern!(py, "image_id"),
            boxes: intern!(py, "boxes"),
            scores: intern!(py, "scores"),
            labels: intern!(py, "labels"),
            rles: intern!(py, "rles"),
            keypoints: intern!(py, "keypoints"),
        }
    }
}

/// Append one `Detections` dict directly into `DetectionColumnsInput` ready
/// to feed `CocoDetections::from_columns`. `iou_type` controls which
/// fields are required and which are silently ignored.
///
/// `ann` is the dict's position in a sequence `detections=` argument, so
/// a rejection can point at the offending element rather than leaving
/// the caller to find it in a 5000-entry list (quirk **J6**). It is
/// `None` when a single bare dict was passed.
fn append_dict_columns<'py>(
    dict: &Bound<'py, PyDict>,
    keys: &DictKeys<'py>,
    ann: Option<usize>,
    iou_type: ArrayIouType,
    ctx: &CastCtx<'py, '_>,
    out: &mut DetectionColumnsInput,
) -> PyResult<()> {
    let here = FieldPath::new(ann, "");
    let image_id_obj = dict.get_item(keys.image_id)?.ok_or_else(|| {
        PyValueError::new_err(format!(
            "{here}: missing required field 'image_id' \
             (each Detections dict must carry an integer image id)"
        ))
    })?;
    let image_id_raw: i64 = image_id_obj.extract().map_err(|e| {
        let path = FieldPath::new(ann, ".image_id");
        PyValueError::new_err(format!("{path}: expected int, got {e}"))
    })?;
    let image_id = ImageId(image_id_raw);

    let boxes_path = FieldPath::new(ann, ".boxes").to_string();
    let boxes_obj = dict.get_item(keys.boxes)?.ok_or_else(|| {
        PyValueError::new_err(format!(
            "{here}: missing required field 'boxes' (N×4 float64 xywh array)"
        ))
    })?;
    let boxes_obj = ctx.maybe_cast(boxes_obj, &boxes_path, "float64")?;
    let boxes_view = dlpack::extract_f64_2d(&boxes_obj, &boxes_path, 4)?;
    let boxes = boxes_view.as_slice();
    let n = boxes.len() / 4;

    let scores_path = FieldPath::new(ann, ".scores").to_string();
    let scores_obj = dict.get_item(keys.scores)?.ok_or_else(|| {
        PyValueError::new_err(format!(
            "{here}: missing required field 'scores' (length-N float64)"
        ))
    })?;
    let scores_obj = ctx.maybe_cast(scores_obj, &scores_path, "float64")?;
    let scores_view = dlpack::extract_f64_1d(&scores_obj, &scores_path)?;
    if scores_view.len() != n {
        return Err(PyValueError::new_err(format!(
            "{scores_path}: length {} disagrees with boxes (N={n})",
            scores_view.len()
        )));
    }
    let scores = scores_view.as_slice();

    let labels_path = FieldPath::new(ann, ".labels").to_string();
    let labels_obj = dict.get_item(keys.labels)?.ok_or_else(|| {
        PyValueError::new_err(format!(
            "{here}: missing required field 'labels' (length-N int64)"
        ))
    })?;
    let labels_obj = ctx.maybe_cast(labels_obj, &labels_path, "int64")?;
    let labels_view = dlpack::extract_i64_1d(&labels_obj, &labels_path)?;
    if labels_view.len() != n {
        return Err(PyValueError::new_err(format!(
            "{labels_path}: length {} disagrees with boxes (N={n})",
            labels_view.len()
        )));
    }
    let labels = labels_view.as_slice();

    if matches!(iou_type, ArrayIouType::Segm | ArrayIouType::Boundary) {
        let rles_obj = dict.get_item(keys.rles)?.ok_or_else(|| {
            PyValueError::new_err(format!(
                "{here}: iou_type={} requires a 'rles' field \
                 (sequence of RLE dicts or 2-D bool/uint8 bitmasks)",
                iou_type.as_str()
            ))
        })?;
        let extracted = extract_rles(&rles_obj, FieldPath::new(ann, ".rles"), n, ctx)?;
        let segs = out
            .segmentations
            .get_or_insert_with(|| Vec::with_capacity(n));
        segs.extend(extracted.into_iter().map(Some));
    }

    if matches!(iou_type, ArrayIouType::Keypoints) {
        let kp_path = FieldPath::new(ann, ".keypoints").to_string();
        let kp_obj = dict.get_item(keys.keypoints)?.ok_or_else(|| {
            PyValueError::new_err(format!(
                "{here}: iou_type='keypoints' requires a 'keypoints' field \
                 ((N, K, 3) float64 array of [x, y, v] triplets)"
            ))
        })?;
        let kp_obj = ctx.maybe_cast(kp_obj, &kp_path, "float64")?;
        let (view, k) = dlpack::extract_f64_3d_kp(&kp_obj, &kp_path)?;
        let stride = k * 3;
        let expected = n.checked_mul(stride).ok_or_else(|| {
            PyValueError::new_err(format!(
                "{kp_path}: shape product (N={n}, K={k}, 3) overflows usize"
            ))
        })?;
        if view.len() != expected {
            return Err(PyValueError::new_err(format!(
                "{kp_path}: flat length {} disagrees with shape (N={n}, K={k}, 3)",
                view.len()
            )));
        }
        let flat = view.as_slice();
        let kp_col = out.keypoints.get_or_insert_with(|| Vec::with_capacity(n));
        kp_col.reserve(n);
        for i in 0..n {
            let start = i * stride;
            kp_col.push(Some(flat[start..start + stride].to_vec()));
        }
    }

    out.image_ids.extend(std::iter::repeat_n(image_id, n));
    out.category_ids
        .extend(labels.iter().map(|&c| CategoryId(c)));
    out.scores.extend_from_slice(scores);
    out.bboxes.extend(boxes.chunks_exact(4).map(|b| Bbox {
        x: b[0],
        y: b[1],
        w: b[2],
        h: b[3],
    }));
    Ok(())
}

/// Per-item dispatcher: uncompressed dict, compressed (bytes) dict, or
/// 2-D bitmask. See `_array_types.RLEInput` for the public typing.
fn extract_rles<'py>(
    obj: &Bound<'py, PyAny>,
    path: FieldPath<'_>,
    n: usize,
    ctx: &CastCtx<'py, '_>,
) -> PyResult<Vec<Segmentation>> {
    let seq = obj.cast::<PySequence>().map_err(|e| {
        PyTypeError::new_err(format!(
            "{path}: expected a sequence of RLE dicts \
             or 2-D bool/uint8 bitmasks: {e}"
        ))
    })?;
    let len = seq.len()?;
    if len != n {
        return Err(PyValueError::new_err(format!(
            "{path}: length {len} disagrees with boxes (N={n})"
        )));
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(extract_one_rle(&seq.get_item(i)?, path.item(i), ctx)?);
    }
    Ok(out)
}

/// Which `counts` payloads a dict-shaped RLE accepts.
///
/// The two ingest surfaces take different sets by design. ADR-0030's
/// `rles` is an *in-memory* array surface: `counts` is either the bytes
/// `pycocotools.mask.encode` returns or a uint32 array. ADR-0057's
/// per-annotation `segmentation` is the *result-file* surface expressed
/// as Python objects, so it must also take what `json.load` of a results
/// file produces — a `str` counts (quirk **K3**) and a plain list of
/// ints — or the list route would reject payloads the file route
/// accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CountsShapes {
    /// `bytes` or a uint32 DLPack array. ADR-0030 `Detections.rles[i]`.
    InMemory,
    /// The above plus the JSON wire shapes: `str` counts and a sequence
    /// of ints. ADR-0057 `detections[i].segmentation`.
    AlsoJsonWire,
}

/// One segmentation in RLE or bitmask form, under the fully-qualified
/// `field` path (`detections.rles[3]` or `detections[3].segmentation`).
///
/// Polygons are **not** accepted here: this is ADR-0030's array surface,
/// where a polygon is not a shape a detector emits. The result-dict and
/// JSON routes take them — see `result_ingest::extract_segmentation`.
pub(crate) fn extract_one_rle<'py>(
    item: &Bound<'py, PyAny>,
    field: FieldPath<'_>,
    ctx: &CastCtx<'py, '_>,
) -> PyResult<Segmentation> {
    extract_one_rle_with(item, field, ctx, CountsShapes::InMemory)
}

/// [`extract_one_rle`] with the accepted `counts` shapes chosen by the
/// caller. Shared with the result-annotation route (`result_ingest`).
pub(crate) fn extract_one_rle_with<'py>(
    item: &Bound<'py, PyAny>,
    field: FieldPath<'_>,
    ctx: &CastCtx<'py, '_>,
    counts_shapes: CountsShapes,
) -> PyResult<Segmentation> {
    if let Ok(dict) = item.cast::<PyDict>() {
        extract_rle_dict(dict, field, counts_shapes)
    } else if item.hasattr("__dlpack_device__")? {
        // Cheap protocol probe: lets us name all three accepted forms
        // in the dispatch error below instead of falling through into
        // a DLPack-specific message that hides forms 1 + 2.
        extract_rle_bitmask(item, field, ctx)
    } else {
        Err(PyTypeError::new_err(format!(
            "{field}: expected RLE dict {{counts, size}} \
             or 2-D bool/uint8 array, got {} \
             (polygons are accepted on the result-dict and JSON routes, \
             not on the columnar `rles` field)",
            type_name_of(item)
        )))
    }
}

/// Best-effort type name for an error message.
pub(crate) fn type_name_of(obj: &Bound<'_, PyAny>) -> String {
    obj.get_type()
        .name()
        .map_or_else(|_| "<unknown>".to_string(), |n| n.to_string())
}

/// Forms 1 + 2: dict-shaped RLE. `counts` may be `bytes` (compressed
/// COCO 6-bit string) or a uint32 1-D DLPack array (uncompressed run
/// lengths); under [`CountsShapes::AlsoJsonWire`] it may additionally be
/// a `str` or a sequence of ints, which is what a results *file* carries.
fn extract_rle_dict(
    dict: &Bound<'_, PyDict>,
    field: FieldPath<'_>,
    counts_shapes: CountsShapes,
) -> PyResult<Segmentation> {
    // Interned, module-lifetime keys: a bare `&str` probe builds and
    // hashes a fresh `PyString` per call, and this runs once per
    // annotation on the segm list route. The other eight result-dict
    // keys are interned in `result_ingest::ann_dicts_to_inputs`; these
    // two were the stragglers.
    let py = dict.py();
    let counts_obj = dict.get_item(intern!(py, "counts"))?.ok_or_else(|| {
        PyValueError::new_err(format!(
            "{field}: missing 'counts' \
             (uint32 1-D array or bytes)"
        ))
    })?;
    let size_obj = dict
        .get_item(intern!(py, "size"))?
        .ok_or_else(|| PyValueError::new_err(format!("{field}: missing 'size' (h, w) tuple")))?;
    let (h, w) = extract_size_tuple(&size_obj, field)?;

    let json_wire = counts_shapes == CountsShapes::AlsoJsonWire;
    let counts = if let Ok(b) = counts_obj.cast::<PyBytes>() {
        let bytes = b.as_bytes();
        let s = std::str::from_utf8(bytes).map_err(|_| {
            PyValueError::new_err(format!(
                "{field}.counts: compressed RLE bytes must be \
                 valid UTF-8 ASCII (COCO 6-bit string)"
            ))
        })?;
        SegmentationRleCounts::Compressed(s.to_owned())
    } else if json_wire && counts_obj.is_instance_of::<PyString>() {
        // K3 (`aligned`): a results file has no bytes type, so its
        // compressed counts arrive as `str`. Same payload, same decoder.
        SegmentationRleCounts::Compressed(counts_obj.extract::<String>()?)
    } else if counts_obj.is_instance_of::<PyByteArray>()
        || counts_obj.is_instance_of::<PyMemoryView>()
    {
        // A `bytearray`/`memoryview` over the *compressed* 6-bit string
        // also satisfies the sequence protocol, and indexing it yields
        // one `int` per byte — so the sequence branch below would read
        // each character as a run length and decode a silently wrong
        // mask. The object cannot say which reading it meant, so neither
        // do we: refuse and name both fixes.
        return Err(PyTypeError::new_err(format!(
            "{field}.counts: got {}, which is ambiguous — its elements are \
             the bytes of a compressed COCO string on one reading and \
             uncompressed run lengths on the other. Pass `bytes(...)` (or \
             `str`) for compressed counts, or a uint32 array / list of ints \
             for uncompressed ones",
            type_name_of(&counts_obj)
        )));
    } else if json_wire
        && (counts_obj.is_instance_of::<PyList>() || counts_obj.is_instance_of::<PyTuple>())
    {
        // Uncompressed counts as a plain list of ints — the
        // `{"counts": [...], "size": [...]}` shape `serde_json` accepts.
        // Deliberately a `list`/`tuple` allow-list rather than "any
        // non-DLPack sequence": every other sequence that reaches here
        // is a buffer whose element reading is ambiguous (above), and
        // the arrays belong on the DLPack branch below.
        let seq = counts_obj.cast::<PySequence>().map_err(|e| {
            PyTypeError::new_err(format!(
                "{field}.counts: expected bytes, str, a uint32 array, \
                 or a sequence of ints: {e}"
            ))
        })?;
        let len = seq.len()?;
        let mut runs = Vec::with_capacity(len);
        for j in 0..len {
            runs.push(seq.get_item(j)?.extract::<u32>().map_err(|e| {
                PyValueError::new_err(format!(
                    "{field}.counts[{j}]: expected a non-negative int: {e}"
                ))
            })?);
        }
        SegmentationRleCounts::Uncompressed(runs.into())
    } else {
        let counts_field = format!("{field}.counts");
        let counts_view = dlpack::extract_u32_1d(&counts_obj, &counts_field)?;
        SegmentationRleCounts::Uncompressed(counts_view.as_slice().into())
    };

    Ok(Segmentation::Rle(SegmentationRle {
        size: [h, w],
        counts,
    }))
}

/// Form 3: 2-D bool/uint8 bitmask. C-contiguous input is copied once
/// via `numpy.asfortranarray` inside the dlpack helper before reaching
/// [`Rle::from_raster_bytes`].
fn extract_rle_bitmask<'py>(
    item: &Bound<'py, PyAny>,
    field: FieldPath<'_>,
    ctx: &CastCtx<'py, '_>,
) -> PyResult<Segmentation> {
    let asfortran = ctx.asfortranarray()?;
    // A bitmask ingest copies the whole raster, so rendering the path
    // here is free relative to the work it labels.
    let field = field.to_string();
    let (view, h, w) = dlpack::extract_u8_or_bool_2d_fortran(item, &field, &asfortran)?;
    let rle = Rle::from_raster_bytes(view.as_slice(), h, w)
        .map_err(|e| PyValueError::new_err(format!("{field}: {e}")))?;
    Ok(Segmentation::Rle(SegmentationRle {
        size: [rle.h, rle.w],
        counts: SegmentationRleCounts::Uncompressed(rle.counts),
    }))
}

fn extract_size_tuple(obj: &Bound<'_, PyAny>, field: FieldPath<'_>) -> PyResult<(u32, u32)> {
    // Accept any 2-element sequence, so `(h, w)` tuples and `[h, w]`
    // lists both work. A NumPy array does not: PyO3's `PySequence` cast
    // is an `isinstance(_, collections.abc.Sequence)` test, which
    // `ndarray` fails — as it does on the JSON route, where
    // `json.dumps` refuses it.
    let seq = obj.cast::<PySequence>().map_err(|e| {
        PyTypeError::new_err(format!("{field}.size: expected (h, w) sequence: {e}"))
    })?;
    if seq.len()? != 2 {
        return Err(PyValueError::new_err(format!(
            "{field}.size: expected length-2 (h, w), got length {}",
            seq.len()?
        )));
    }
    let h: u32 = seq.get_item(0)?.extract().map_err(|e| {
        PyValueError::new_err(format!(
            "{field}.size[0] (height): expected non-negative int: {e}"
        ))
    })?;
    let w: u32 = seq.get_item(1)?.extract().map_err(|e| {
        PyValueError::new_err(format!(
            "{field}.size[1] (width): expected non-negative int: {e}"
        ))
    })?;
    Ok((h, w))
}

// ---------------------------------------------------------------------------
// cast_inputs helper — promotes dtypes via np.ascontiguousarray.
// ---------------------------------------------------------------------------

fn cast_via_numpy<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
    field: &str,
    dtype: &str,
    latch: &AtomicBool,
    ascontiguousarray: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", dtype)?;
    let cast_result = ascontiguousarray.call((obj,), Some(&kwargs)).map_err(|e| {
        PyTypeError::new_err(format!(
            "{field}: cast_inputs=True failed to coerce to {dtype}: {e}"
        ))
    })?;
    emit_cast_warning_once(py, latch)?;
    Ok(cast_result)
}

/// Resolve `numpy.ascontiguousarray` once per batch when `cast_inputs=True`.
fn resolve_ascontiguousarray(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    let np = py
        .import("numpy")
        .map_err(|e| PyTypeError::new_err(format!("cast_inputs=True requires numpy: {e}")))?;
    np.getattr("ascontiguousarray")
}

/// Resolve `numpy.asfortranarray` lazily for the form-3 bitmask path
/// (ADR-0030 amendment). Mirrors [`resolve_ascontiguousarray`] but is
/// only called when a C-order bitmask is observed; F-order ingest is
/// zero-copy.
fn resolve_asfortranarray(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    let np = py.import("numpy").map_err(|e| {
        PyTypeError::new_err(format!(
            "C-order bitmask ingest requires numpy for an asfortranarray copy: {e}"
        ))
    })?;
    np.getattr("asfortranarray")
}

// ---------------------------------------------------------------------------
// Multi-image dispatch
// ---------------------------------------------------------------------------

/// Resolve `numpy.ascontiguousarray` once per call when
/// `cast_inputs=True`, so [`CastCtx::new`] has something to hold.
fn ascontig_for<'py>(
    py: Python<'py>,
    cast_state: &CastState,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    match cast_state {
        Some(_) => Ok(Some(resolve_ascontiguousarray(py)?)),
        None => Ok(None),
    }
}

/// Extract a multi-image `DetectionsBatch` dict (ADR-0066) directly into
/// [`DetectionColumnsInput`].
///
/// Everything here runs under the GIL, the numeric column copies included:
/// the `DLPackView` slices alias the caller's live, mutable buffers, and
/// per ADR-0006 conversion finishes before [`Python::detach`] — a
/// detached copy would race any Python thread still writing them. The
/// copies are streaming `memcpy`-shaped passes; the CSR index build in
/// `CocoDetections::from_columns` is what the consumer runs detached.
pub(crate) fn batch_to_columns<'py>(
    py: Python<'py>,
    dict: &Bound<'py, PyDict>,
    iou_type: ArrayIouType,
    cast_state: &CastState,
) -> PyResult<DetectionColumnsInput> {
    let ascontig = ascontig_for(py, cast_state)?;
    let ctx = CastCtx::new(py, cast_state, &ascontig);

    let k_boxes = intern!(py, "boxes");
    let k_scores = intern!(py, "scores");
    let k_labels = intern!(py, "labels");
    let k_image_ids = intern!(py, "image_ids");
    let k_counts = intern!(py, "counts");
    let k_rles = intern!(py, "rles");
    let k_keypoints = intern!(py, "keypoints");

    let boxes_obj = dict.get_item(k_boxes)?.ok_or_else(|| {
        PyValueError::new_err("detections: missing required field 'boxes' (N×4 float64 xywh array)")
    })?;
    let boxes_obj = ctx.maybe_cast(boxes_obj, "detections.boxes", "float64")?;
    let boxes_view = dlpack::extract_f64_2d(&boxes_obj, "detections.boxes", 4)?;
    let n = boxes_view.len() / 4;

    let scores_obj = dict.get_item(k_scores)?.ok_or_else(|| {
        PyValueError::new_err("detections: missing required field 'scores' (length-N float64)")
    })?;
    let scores_obj = ctx.maybe_cast(scores_obj, "detections.scores", "float64")?;
    let scores_view = dlpack::extract_f64_1d(&scores_obj, "detections.scores")?;
    if scores_view.len() != n {
        return Err(PyValueError::new_err(format!(
            "detections.scores: length {} disagrees with boxes (N={n})",
            scores_view.len()
        )));
    }

    let labels_obj = dict.get_item(k_labels)?.ok_or_else(|| {
        PyValueError::new_err("detections: missing required field 'labels' (length-N int64)")
    })?;
    let labels_obj = ctx.maybe_cast(labels_obj, "detections.labels", "int64")?;
    let labels_view = dlpack::extract_i64_1d(&labels_obj, "detections.labels")?;
    if labels_view.len() != n {
        return Err(PyValueError::new_err(format!(
            "detections.labels: length {} disagrees with boxes (N={n})",
            labels_view.len()
        )));
    }

    let image_ids_obj = dict.get_item(k_image_ids)?.ok_or_else(|| {
        PyValueError::new_err("detections: missing required field 'image_ids' (int64 array)")
    })?;
    let image_ids_obj = ctx.maybe_cast(image_ids_obj, "detections.image_ids", "int64")?;
    let image_ids_view = dlpack::extract_i64_1d(&image_ids_obj, "detections.image_ids")?;

    let counts_view = match dict.get_item(k_counts)? {
        Some(v) if !v.is_none() => {
            let v = ctx.maybe_cast(v, "detections.counts", "int64")?;
            let view = dlpack::extract_i64_1d(&v, "detections.counts")?;
            if view.len() != image_ids_view.len() {
                return Err(PyValueError::new_err(format!(
                    "detections.counts: length {} disagrees with detections.image_ids (M={})",
                    view.len(),
                    image_ids_view.len()
                )));
            }
            let mut sum: usize = 0;
            for (m, &c) in view.as_slice().iter().enumerate() {
                if c < 0 {
                    return Err(PyValueError::new_err(format!(
                        "detections.counts[{m}]: count must be non-negative, got {c}"
                    )));
                }
                let cu = usize::try_from(c).map_err(|_| {
                    PyValueError::new_err(format!(
                        "detections.counts[{m}]: count {c} overflows usize"
                    ))
                })?;
                sum = sum.checked_add(cu).ok_or_else(|| {
                    PyValueError::new_err("detections.counts: sum overflows usize")
                })?;
            }
            if sum != n {
                return Err(PyValueError::new_err(format!(
                    "detections.counts: sum ({sum}) disagrees with boxes (N={n})"
                )));
            }
            Some(view)
        }
        _ => {
            if image_ids_view.len() != n {
                return Err(PyValueError::new_err(format!(
                    "detections.image_ids: length {} disagrees with boxes (N={n})",
                    image_ids_view.len()
                )));
            }
            None
        }
    };

    // `DetectionsBatch` is the multi-image `Detections`, so it takes the
    // array surface's rule, not the file's: a mask pass needs `rles`.
    // Quirk J2 (a bbox rectangle standing in for a missing mask) belongs
    // to the routes that mirror a `loadRes` file — bytes, result dicts,
    // the `(N, 7)` matrix. An array caller who left the masks out has a
    // bug, and J2 would turn it into a plausible, wrong segm AP.
    let segmentations: Option<Vec<Option<Segmentation>>> = match iou_type {
        ArrayIouType::Segm | ArrayIouType::Boundary => {
            let rles_obj = dict
                .get_item(k_rles)?
                .filter(|v| !v.is_none())
                .ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "detections: iou_type={} requires a 'rles' field \
                         (sequence of RLE dicts or 2-D bool/uint8 bitmasks)",
                        iou_type.as_str()
                    ))
                })?;
            Some(
                extract_rles(&rles_obj, FieldPath::new(None, ".rles"), n, &ctx)?
                    .into_iter()
                    .map(Some)
                    .collect(),
            )
        }
        ArrayIouType::Bbox | ArrayIouType::Keypoints => None,
    };

    let kp_data = match iou_type {
        ArrayIouType::Keypoints => {
            let kp_obj = dict.get_item(k_keypoints)?.ok_or_else(|| {
                PyValueError::new_err(
                    "detections: iou_type='keypoints' requires a 'keypoints' field \
                     ((N, K, 3) float64 array of [x, y, v] triplets)",
                )
            })?;
            let kp_obj = ctx.maybe_cast(kp_obj, "detections.keypoints", "float64")?;
            let (view, k) = dlpack::extract_f64_3d_kp(&kp_obj, "detections.keypoints")?;
            let stride = k * 3;
            let expected = n.checked_mul(stride).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "detections.keypoints: shape product (N={n}, K={k}, 3) overflows usize"
                ))
            })?;
            if view.len() != expected {
                return Err(PyValueError::new_err(format!(
                    "detections.keypoints: flat length {} disagrees with shape (N={n}, K={k}, 3)",
                    view.len()
                )));
            }
            Some((view, stride))
        }
        _ => None,
    };

    let image_ids: Vec<ImageId> = match &counts_view {
        None => image_ids_view
            .as_slice()
            .iter()
            .map(|&id| ImageId(id))
            .collect(),
        Some(counts) => {
            let mut out = Vec::with_capacity(n);
            for (&id, &c) in image_ids_view.as_slice().iter().zip(counts.as_slice()) {
                // `c >= 0` and `sum(counts) == n` were validated above.
                #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
                out.extend(std::iter::repeat_n(ImageId(id), c as usize));
            }
            out
        }
    };
    let category_ids: Vec<CategoryId> = labels_view
        .as_slice()
        .iter()
        .map(|&c| CategoryId(c))
        .collect();
    let scores: Vec<f64> = scores_view.as_slice().to_vec();
    let bboxes: Vec<Bbox> = boxes_view
        .as_slice()
        .chunks_exact(4)
        .map(|b| Bbox {
            x: b[0],
            y: b[1],
            w: b[2],
            h: b[3],
        })
        .collect();
    // Indexed rather than `chunks_exact`: `K = 0` gives a zero stride.
    let keypoints: Option<Vec<Option<Vec<f64>>>> = kp_data.as_ref().map(|(view, stride)| {
        let flat = view.as_slice();
        (0..n)
            .map(|i| Some(flat[i * stride..(i + 1) * stride].to_vec()))
            .collect()
    });

    Ok(DetectionColumnsInput {
        ids: None,
        image_ids,
        category_ids,
        scores,
        bboxes,
        areas: None,
        segmentations,
        keypoints,
        num_keypoints: None,
    })
}

/// Convert per-image dict payloads directly into [`DetectionColumnsInput`].
/// `CocoDetections::from_columns` is intentionally **not** called here:
/// the consumer (foreground evaluator, streaming/background dispatch)
/// runs it inside its own `py.detach` block so the CSR index build runs
/// without holding the GIL.
///
/// `indexed` says whether the caller passed a *sequence* of dicts; when
/// it did, every rejection names the offending element.
pub(crate) fn dicts_to_columns(
    py: Python<'_>,
    dicts: &[Bound<'_, PyDict>],
    indexed: bool,
    iou_type: ArrayIouType,
    cast_state: &CastState,
) -> PyResult<DetectionColumnsInput> {
    let ascontig = ascontig_for(py, cast_state)?;
    let ctx = CastCtx::new(py, cast_state, &ascontig);
    let keys = DictKeys::intern(py);
    let mut out = DetectionColumnsInput {
        ids: None,
        image_ids: Vec::new(),
        category_ids: Vec::new(),
        scores: Vec::new(),
        bboxes: Vec::new(),
        areas: None,
        segmentations: matches!(iou_type, ArrayIouType::Segm | ArrayIouType::Boundary)
            .then(Vec::new),
        keypoints: matches!(iou_type, ArrayIouType::Keypoints).then(Vec::new),
        num_keypoints: None,
    };
    for (i, dict) in dicts.iter().enumerate() {
        let ann = indexed.then_some(i);
        append_dict_columns(dict, &keys, ann, iou_type, &ctx, &mut out)?;
    }
    Ok(out)
}

/// Sibling of [`dicts_to_columns`] for the per-annotation result-dict
/// route. Builds the same [`CastCtx`] (the `segmentation` field accepts
/// the same bitmask forms as `rles`) and delegates the pass itself to
/// `result_ingest`.
pub(crate) fn ann_dicts_to_columns(
    py: Python<'_>,
    dicts: &[Bound<'_, PyDict>],
    iou_type: ArrayIouType,
    cast_state: &CastState,
) -> PyResult<DetectionColumnsInput> {
    let ascontig = ascontig_for(py, cast_state)?;
    let ctx = CastCtx::new(py, cast_state, &ascontig);
    crate::result_ingest::ann_dicts_to_columns(py, dicts, iou_type, &ctx)
}

/// Sibling of [`dicts_to_columns`] for the ADR-0060 ground-truth route.
/// Builds the same [`CastCtx`] (the GT `segmentation` column accepts the
/// same shapes as `rles` and `detections[i].segmentation`) and delegates
/// the pass itself to `gt_ingest`.
///
/// Returns the three parts rather than a `CocoDataset`: the caller runs
/// `CocoDataset::from_parts` inside its own `py.detach` block, so the
/// reference-integrity scan and the index build happen without the GIL.
pub(crate) fn gt_parts_from_arrays(
    py: Python<'_>,
    images: &Bound<'_, PyAny>,
    annotations: &Bound<'_, PyAny>,
    categories: &Bound<'_, PyAny>,
    cast_state: &CastState,
) -> PyResult<crate::gt_ingest::GtParts> {
    let ascontig = ascontig_for(py, cast_state)?;
    let ctx = CastCtx::new(py, cast_state, &ascontig);
    crate::gt_ingest::gt_parts_from_arrays(images, annotations, categories, &ctx)
}

/// Sibling of [`dicts_to_columns`] for the `(N, 7)` matrix route. The
/// matrix is a single array, so the only thing the context carries that
/// it can use is the `cast_inputs=True` promotion — which it *does*
/// honour, on the same terms as `Detections.boxes`.
pub(crate) fn matrix_to_columns(
    py: Python<'_>,
    matrix: &Bound<'_, PyAny>,
    cast_state: &CastState,
) -> PyResult<DetectionColumnsInput> {
    let ascontig = ascontig_for(py, cast_state)?;
    let ctx = CastCtx::new(py, cast_state, &ascontig);
    crate::result_ingest::matrix_to_columns(matrix, &ctx)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use vernier_core::dataset::DetectionInput;
    use vernier_core::CocoDetections;

    /// `from_columns` of array-derived records yields the same indices as
    /// `from_json_bytes` for the same logical detections. The SoA path
    /// converges on the identical `CocoDetections` representation.
    #[test]
    fn array_inputs_match_json_for_bbox_only() -> Result<(), Box<dyn std::error::Error>> {
        let inputs = vec![
            DetectionInput {
                id: None,
                image_id: ImageId(1),
                category_id: CategoryId(7),
                score: 0.9,
                bbox: Bbox {
                    x: 1.0,
                    y: 2.0,
                    w: 3.0,
                    h: 4.0,
                },
                area: None,
                segmentation: None,
                keypoints: None,
                num_keypoints: None,
            },
            DetectionInput {
                id: None,
                image_id: ImageId(1),
                category_id: CategoryId(7),
                score: 0.8,
                bbox: Bbox {
                    x: 5.0,
                    y: 6.0,
                    w: 7.0,
                    h: 8.0,
                },
                area: None,
                segmentation: None,
                keypoints: None,
                num_keypoints: None,
            },
        ];

        let json = serde_json::to_vec(&inputs)?;
        let from_json = CocoDetections::from_json_bytes(&json)?;
        let from_arr = CocoDetections::from_columns(DetectionColumnsInput {
            ids: None,
            image_ids: vec![ImageId(1), ImageId(1)],
            category_ids: vec![CategoryId(7), CategoryId(7)],
            scores: vec![0.9, 0.8],
            bboxes: vec![
                Bbox {
                    x: 1.0,
                    y: 2.0,
                    w: 3.0,
                    h: 4.0,
                },
                Bbox {
                    x: 5.0,
                    y: 6.0,
                    w: 7.0,
                    h: 8.0,
                },
            ],
            areas: None,
            segmentations: None,
            keypoints: None,
            num_keypoints: None,
        })?;

        let json_recs = from_json.records();
        let arr_recs = from_arr.records();
        assert_eq!(json_recs.len(), arr_recs.len());
        for (a, b) in json_recs.zip(arr_recs) {
            assert_eq!(a.id, b.id);
            assert_eq!(a.image_id, b.image_id);
            assert_eq!(a.category_id, b.category_id);
            assert_eq!(a.score, b.score);
            assert_eq!(a.bbox, b.bbox);
            assert_eq!(a.area, b.area);
            assert_eq!(a.segmentation, b.segmentation);
        }
        assert_eq!(
            from_json.indices_for(ImageId(1), CategoryId(7)).len(),
            from_arr.indices_for(ImageId(1), CategoryId(7)).len()
        );
        Ok(())
    }

    #[test]
    fn iou_type_as_str_pins_user_facing_strings() {
        // The strings flow through error messages users grep for; pin them.
        assert_eq!(ArrayIouType::Bbox.as_str(), "bbox");
        assert_eq!(ArrayIouType::Segm.as_str(), "segm");
        assert_eq!(ArrayIouType::Boundary.as_str(), "boundary");
        assert_eq!(ArrayIouType::Keypoints.as_str(), "keypoints");
    }
}

// ---------------------------------------------------------------------------
// Mask area (ADR-0063)
// ---------------------------------------------------------------------------

/// Foreground pixel count for each segmentation in `rles`.
///
/// ADR-0060 makes a ground truth's `area` **required and read verbatim**
/// — "silently substituting `w * h` would re-bucket every polygon GT
/// between AP-small / medium / large" (`gt_ingest.rs`). A caller
/// assembling columnar ground truth from a training loop therefore needs
/// the mask's area, and under ADR-0063's `area="auto"` that is the
/// number to fall back to when the framework supplied none.
///
/// This is `pycocotools.mask.area` (quirk **G5**: the sum of the
/// odd-indexed runs), reached through
/// [`Segmentation::rle_area`][vernier_core::segmentation::Segmentation::rle_area]
/// so this route and the evaluator agree by construction rather than by
/// inspection. It is deliberately *internal*: the public `vernier.mask`
/// surface is a separate decision, and this is the subset ADR-0063 needs.
///
/// Accepts everything [`extract_one_rle`] accepts. Polygons are rejected
/// here exactly as they are on the columnar `rles` field, and for the
/// same reason `maskUtils.area` rejects them.
///
/// Extraction and summation are split so the decode — which for a
/// compressed RLE is the whole cost and touches no Python object — runs
/// with the GIL released, per ADR-0006. A caller with *bitmasks* should
/// not reach here at all: summing them in NumPy is ~19x faster than a
/// round trip through the RLE codec and bit-identical, so the Python
/// side reserves this for pre-encoded RLEs.
///
/// Returns the areas as native-endian `f64` bytes, which the caller
/// wraps with `numpy.frombuffer`. A `Vec<f64>` would cross the boundary
/// as one `PyFloat` per mask — at validation scale 10^5–10^6 transient
/// objects — for a value NumPy re-packs into a buffer immediately.
#[pyfunction]
pub(crate) fn rle_area<'py>(
    py: Python<'py>,
    rles: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyBytes>> {
    // No `cast_inputs`: neither accepted form has a dtype this could
    // widen — a dict's `counts` is `bytes` or `uint32` and a bitmask is
    // `bool` or `uint8`, all exact. A parameter here would promise
    // tolerance the extractor cannot deliver.
    let cast_state = new_cast_state(false);
    let ascontig = ascontig_for(py, &cast_state)?;
    let ctx = CastCtx::new(py, &cast_state, &ascontig);

    // A `PyIterator`'s `size_hint` is `(0, None)`, so reserve off the
    // sequence's own length; `len()` fails only for a true generator.
    let hint = rles.len().unwrap_or(0);
    let items = rles.try_iter()?;
    let mut segmentations: Vec<Segmentation> = Vec::with_capacity(hint);
    for (i, item) in items.enumerate() {
        let item = item?;
        segmentations.push(extract_one_rle(
            &item,
            FieldPath::rooted("rles", Some(i), ""),
            &ctx,
        )?);
    }

    let bytes = py
        .detach(|| {
            let mut areas: Vec<u8> = Vec::with_capacity(segmentations.len() * 8);
            for (i, segmentation) in segmentations.iter().enumerate() {
                // `rle_area` is `None` only for polygons, which
                // `extract_one_rle` has already refused, so the `None` arm is
                // unreachable through this entry point.
                let area = segmentation
                    .rle_area()
                    .map_err(|e| (i, e))?
                    .ok_or_else(|| {
                        (
                            i,
                            EvalError::InvalidConfig {
                                detail: "polygons have no RLE area".to_string(),
                            },
                        )
                    })?;
                areas.extend_from_slice(&(area as f64).to_ne_bytes());
            }
            Ok(areas)
        })
        .map_err(|(i, e): (usize, EvalError)| {
            let inner = eval_error_to_pyerr(py, e);
            PyValueError::new_err(format!("rles[{i}]: {inner}"))
        })?;
    Ok(PyBytes::new(py, &bytes))
}
