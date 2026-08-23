//! One dependency for the whole vernier evaluation toolkit.
//!
//! `vernier` is a **facade**: it contains no code of its own and
//! re-exports the published paradigm crates under one dependency and
//! one module map (ADR-0048). Depending on it is exactly equivalent to
//! depending on the leaf crates directly — the public API of this
//! crate *is* the union of theirs, by construction, so it cannot drift
//! from them.
//!
//! ```toml
//! [dependencies]
//! vernier = "0.2"
//! ```
//!
//! # Module map
//!
//! | Module | Crate | Paradigm |
//! |---|---|---|
//! | [`instance`] | [`vernier_core`] | Detection / instance-segmentation AP — bbox, segm, boundary, OKS keypoints, LVIS federated, LRP, TIDE, calibration |
//! | [`mask`] | [`vernier_mask`] | COCO RLE codec, polygon rasterizer, mask ops |
//! | `panoptic` | `vernier_panoptic` | Panoptic quality (PQ / SQ / RQ) — needs the `panoptic` feature |
//! | `semantic` | `vernier_semantic` | Semantic segmentation (mIoU / FWIoU / pixel accuracy) — needs the `semantic` feature |
//! | `partial` | `vernier_partial` | Distributed-eval wire envelope shared by the three paradigms — needs the `partial` feature |
//!
//! The module-per-paradigm shape is forced rather than chosen:
//! `vernier_core`, `vernier_panoptic`, and `vernier_semantic` each
//! export a `ParityMode` and a `VERSION`, so a flat glob re-export
//! would not compile. That the type system lands on the same shape
//! ADR-0029 chose for the Python surface
//! (`vernier.{instance, panoptic, semantic}`) is the same fact seen
//! twice: the names collide because the paradigms are genuinely
//! distinct. A reader moving between the Python and Rust surfaces
//! carries one mental model.
//!
//! There is no prelude. Every unambiguous name in vernier is a module
//! name, so `use vernier::prelude::*;` could only ever be a synonym
//! for `use vernier::{instance, panoptic};` — a second way to say the
//! same thing (ADR-0035).
//!
//! # Two-line orientation
//!
//! ```
//! // The facade's own version, lockstep with every crate it re-exports.
//! assert_eq!(vernier::VERSION, vernier::instance::VERSION);
//! assert_eq!(vernier::VERSION, vernier::mask::VERSION);
//! ```
//!
//! Each module below carries a worked example. For anything deeper —
//! the parity contract, the quirk disposition tables, the streaming
//! and distributed surfaces — follow the module link to the leaf
//! crate's own documentation, which is where the real reference
//! material lives.
//!
//! # Depending on a leaf crate instead
//!
//! The facade is a convenience, never a gate. A consumer who wants a
//! narrower dependency tree can depend on any leaf crate directly and
//! always could:
//!
//! ```toml
//! [dependencies]
//! vernier-core = "0.2"   # bbox / segm / boundary / keypoints AP only
//! ```
//!
//! Within the facade, the three optional paradigms can be trimmed
//! instead of abandoning the crate:
//!
//! ```toml
//! [dependencies]
//! vernier = { version = "0.2", default-features = false }   # instance + mask
//! ```
//!
//! Features here are additive and monotone: enabling one only adds
//! names, and `vernier` is non-empty in every one of the eight
//! reachable feature combinations. A feature of this crate changes
//! what is *nameable*, never what is *computed* — it is never
//! forwarded to a paradigm crate. ADR-0047's "one wheel, one
//! behavior" holds unchanged.
//!
//! # What is not here
//!
//! - **The CLI.** `cargo install vernier-cli` installs the `vernier`
//!   binary; this crate is the library. Depending on the binary's
//!   package from a library would drag `clap` into every consumer's
//!   tree, which ADR-0015 rejected and this crate inherits rather
//!   than relitigates.
//! - **The Python bindings.** `vernier-ffi` ships as the
//!   `vernier._core` extension module inside the `vernier` wheel
//!   (`pip install vernier`) and is not published to crates.io.
//! - **Logic of any kind.** This crate is a directory, not a layer.
//!   `src/lib.rs` is the only source file and there never will be
//!   another: if a module appears here, the firewall behind ADR-0009 /
//!   ADR-0025 / ADR-0028 has been breached and the code belongs in a
//!   paradigm crate. Whole-crate aliasing (`pub use vernier_core as
//!   instance;`) rather than a curated re-export list is what makes
//!   that structural instead of aspirational — the facade has no
//!   opportunity to develop an opinion. If a leaf crate exports
//!   something that should not be public, the fix belongs in the leaf
//!   crate.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

/// Detection and instance-segmentation evaluation — the AP fold.
///
/// Whole-crate alias for [`vernier_core`]: bbox / segm / boundary /
/// OKS-keypoints AP, LVIS federated evaluation, LRP-oLRP, TIDE, and
/// detection calibration, plus the streaming and distributed-partial
/// surfaces.
///
/// ```
/// use vernier::instance::{
///     evaluate::AREA_UNBOUNDED, evaluate_bbox, AreaRange, CocoDataset, CocoDetections,
///     EvaluateParams, ParityMode,
/// };
///
/// let gt = CocoDataset::from_json_bytes(
///     br#"{
///       "images": [{"id": 1, "width": 64, "height": 64}],
///       "annotations": [
///         {"id": 1, "image_id": 1, "category_id": 1,
///          "bbox": [8.0, 8.0, 16.0, 16.0], "area": 256.0, "iscrowd": 0}
///       ],
///       "categories": [{"id": 1, "name": "widget"}]
///     }"#,
/// )?;
/// let dt = CocoDetections::from_json_bytes(
///     br#"[{"image_id": 1, "category_id": 1,
///           "bbox": [8.0, 8.0, 16.0, 16.0], "score": 0.9}]"#,
/// )?;
///
/// let area_ranges = [AreaRange { index: 0, lo: 0.0, hi: AREA_UNBOUNDED }];
/// let grid = evaluate_bbox(
///     &gt,
///     &dt,
///     EvaluateParams {
///         iou_thresholds: &[0.5],
///         area_ranges: &area_ranges,
///         max_dets_per_image: 100,
///         use_cats: true,
///         retain_iou: false,
///     },
///     ParityMode::Strict,
/// )?;
///
/// // One (category, area-range, image) cell; the detection is a perfect
/// // overlap, so it matches at the single 0.5 IoU threshold.
/// let cell = grid.eval_imgs[0].as_ref().ok_or("cell was not evaluated")?;
/// assert!(cell.dt_matched[(0, 0)]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub use vernier_core as instance;

/// COCO mask primitives — RLE codec, polygon rasterizer, mask ops.
///
/// Whole-crate alias for [`vernier_mask`]. Useful on its own for
/// annotation tools and data loaders that read or write COCO RLE
/// without evaluating anything; [`instance`] consumes the same
/// primitives for segmentation and boundary IoU.
///
/// ```
/// use vernier::mask::Rle;
///
/// // A 4x4 mask: 4 background pixels, then a 6-pixel foreground run,
/// // then 6 background. Foreground runs sit at odd indices (quirk G5).
/// let rle = Rle::from_counts(4, 4, vec![4, 6, 6]);
/// assert_eq!(rle.area(), 6);
///
/// // COCO's 6-bit char-string codec round-trips.
/// let encoded = rle.to_string_bytes();
/// assert_eq!(Rle::from_string_bytes(&encoded, 4, 4)?, rle);
/// # Ok::<(), vernier::mask::MaskError>(())
/// ```
pub use vernier_mask as mask;

/// Panoptic-quality evaluation — PQ, SQ, RQ, things / stuff split.
///
/// Whole-crate alias for `vernier_panoptic`. Requires the `panoptic`
/// feature (on by default).
///
/// ```
/// use std::collections::HashMap;
///
/// use vernier::panoptic::{
///     dataset::{CategoryMeta, ImageEntry, SegmentInfo},
///     evaluate, PanopticDataset, PanopticPredictions, ParityMode,
/// };
///
/// // A 2x2 image with a single segment covering every pixel, predicted
/// // exactly: one true positive at IoU 1.0, so PQ = SQ = RQ = 1.
/// let segments = vec![SegmentInfo { id: 7, category_id: 1, iscrowd: false, area: 4 }];
/// let label_map = vec![7_u32; 4];
/// let gt_entry = ImageEntry::from_components(1, 2, 2, label_map.clone(), segments.clone(), "gt")?;
/// let dt_entry = ImageEntry::from_components(1, 2, 2, label_map, segments, "dt")?;
///
/// let gt = PanopticDataset::from_components(
///     HashMap::from([(1, gt_entry)]),
///     HashMap::from([(1, CategoryMeta { id: 1, isthing: true })]),
/// );
/// let dt = PanopticPredictions::from_components(HashMap::from([(1, dt_entry)]));
///
/// let summary = evaluate(&gt, &dt, ParityMode::Strict, false)?;
/// assert_eq!(summary.pq, 1.0);
///
/// assert_eq!(vernier::VERSION, vernier::panoptic::VERSION);
/// # Ok::<(), vernier::panoptic::PanopticError>(())
/// ```
#[cfg(feature = "panoptic")]
pub use vernier_panoptic as panoptic;

/// Semantic-segmentation evaluation — mIoU, FWIoU, pixel accuracy.
///
/// Whole-crate alias for `vernier_semantic`. Requires the `semantic`
/// feature (on by default).
///
/// ```
/// use vernier::semantic::{kernel::accumulate_confusion, summarize, ConfusionMatrix, ParityMode};
///
/// // Two classes over four pixels, predicted exactly.
/// let gt: [u8; 4] = [0, 0, 1, 1];
/// let dt: [u8; 4] = [0, 0, 1, 1];
///
/// let mut confusion = ConfusionMatrix::zeros(2);
/// accumulate_confusion(&gt, &dt, None, &mut confusion);
///
/// let summary = summarize(confusion, ParityMode::Strict);
/// assert_eq!(summary.miou, 1.0);
/// assert_eq!(summary.pixel_accuracy, 1.0);
///
/// assert_eq!(vernier::VERSION, vernier::semantic::VERSION);
/// ```
#[cfg(feature = "semantic")]
pub use vernier_semantic as semantic;

/// The distributed-evaluation wire envelope shared by all three
/// paradigms.
///
/// Whole-crate alias for `vernier_partial`. Requires the `partial`
/// feature (on by default). Most consumers reach this transitively
/// through [`instance`] / `panoptic` / `semantic`; the direct path is
/// for tooling that produces or consumes partials outside the paradigm
/// crates.
///
/// ```
/// use vernier::partial::{encode, traits::ParadigmKind, WireEnvelopeHeader, FORMAT_VERSION, MAGIC};
///
/// let header = WireEnvelopeHeader {
///     paradigm_kind: ParadigmKind::Instance.as_u8(),
///     discriminator: 0,
///     parity_mode: 0,
///     rank_id: Some(0),
///     dataset_hash: [0_u8; 32],
///     params_hash: [0_u8; 32],
///     shape_fingerprint: [1, 1, 1, 0],
/// };
///
/// // Magic, then format version, then the framed header + body + CRC.
/// let blob = encode(&header, &[])?;
/// assert_eq!(blob[..4], MAGIC);
/// assert_eq!(blob[4], FORMAT_VERSION);
/// # Ok::<(), vernier::partial::PartialError>(())
/// ```
#[cfg(feature = "partial")]
pub use vernier_partial as partial;

/// Facade version. Lockstep with every re-exported crate: `vernier`
/// inherits `workspace.package.version` and pins each path dependency
/// to it, so a release moves all seven crates together (ADR-0048
/// §"Versioning and publish order").
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_lockstep_with_unconditional_reexports() {
        assert!(!VERSION.is_empty());
        assert_eq!(VERSION, instance::VERSION);
        assert_eq!(VERSION, mask::VERSION);
    }

    /// Mechanical guard for the ADR-0048 hard invariant: a
    /// `#[cfg(feature = ...)]` in this crate may gate a `pub use` line
    /// and nothing else. Anything else — a gated function, a gated
    /// module, a gated dependency forwarding — is the door through
    /// which behavior fragmentation would re-enter.
    ///
    /// The needle is assembled at runtime so this test does not match
    /// its own source text.
    #[test]
    fn cfg_feature_gates_only_reexports() {
        let needle = concat!("#[cfg(", "feature");
        let src = include_str!("lib.rs");
        let lines: Vec<&str> = src.lines().collect();
        let mut gates = 0_usize;
        for (i, line) in lines.iter().enumerate() {
            if !line.trim_start().starts_with(needle) {
                continue;
            }
            gates += 1;
            let next = lines.get(i + 1).map(|l| l.trim_start()).unwrap_or("");
            assert!(
                next.starts_with("pub use "),
                "line {}: a cfg(feature) gate must be followed by a `pub use`, found `{next}`",
                i + 1,
            );
        }
        assert_eq!(gates, 3, "expected exactly three feature-gated re-exports");
    }
}
