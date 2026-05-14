//! Partition manifest schema + JSON parser (ADR-0046).
//!
//! Canonical wire format is the JSON-records shape documented in
//! ADR-0046 §"Manifest schema":
//!
//! ```json
//! {
//!   "manifest_version": "1",
//!   "key_kind": "image_id",
//!   "rows": [
//!     {"key": 100, "weather": "fog",   "time_of_day": "night"},
//!     {"key": 101, "weather": "clear", "time_of_day": "day"}
//!   ]
//! }
//! ```
//!
//! - `key_kind` is `"image_id"` (consumed by `evaluate_partitioned`)
//!   or `"result"` (consumed by `vernier aggregate`).
//! - Every column other than `key` is treated as an **axis**; cells
//!   are categorical string values.
//! - Unknown keys (manifest rows whose key is absent from the
//!   live dataset / result set) produce a [`ManifestWarning`] and are
//!   skipped — they never raise. The caller renders the warnings to
//!   stderr; core never writes there.
//!
//! CSV input is a sibling concern (`manifest_csv.rs` in phase 2)
//! that converts to this canonical shape at parse time.

use std::collections::{HashMap, HashSet};

use serde::Deserialize;

use crate::dataset::ImageId;
use crate::error::EvalError;
use crate::partition::{KeyKind, PartitionSpec, CROSS_SEPARATOR};

/// Canonical wire-format version this parser understands.
pub const MANIFEST_VERSION: &str = "1";

/// JSON wire shape — directly deserialized from the manifest bytes.
///
/// Validation happens on the parsed form (see [`parse_manifest`]);
/// keeping this type close to the wire keeps `serde` errors readable
/// when a field is missing.
#[derive(Debug, Deserialize)]
struct ManifestDoc {
    manifest_version: String,
    key_kind: String,
    rows: Vec<ManifestRow>,
}

#[derive(Debug, Deserialize)]
struct ManifestRow {
    /// `image_id` for an image-keyed manifest (i64 in the wire), the
    /// run `label` string for a result-keyed manifest. Both are
    /// represented as raw JSON values here so the discriminator
    /// validation can give targeted error messages.
    key: serde_json::Value,
    /// Every column other than `key` is treated as an axis. `flatten`
    /// is essential — without it serde would reject the extra fields.
    #[serde(flatten)]
    axes: serde_json::Map<String, serde_json::Value>,
}

/// Non-fatal observation collected during parsing.
///
/// Manifests can drift from the live dataset / run set without being
/// invalid — a manifest row for an image id the GT JSON no longer
/// contains is an artifact of a stale workbook, not a hard error.
/// These warnings let the caller surface the drift on stderr (CLI) or
/// as a Python `warnings.warn` (Python lane).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ManifestWarning {
    /// Manifest row references a key absent from the live dataset /
    /// run set. Carries the offending key for stable error messages.
    UnknownKey {
        /// The unrecognized key (image id as a stringified integer
        /// for `image_id` manifests, or the run label for `result`
        /// manifests).
        key: String,
    },
}

/// Parsed manifest plus the warnings emitted while resolving it.
#[derive(Debug, Clone)]
pub struct ParsedManifest {
    /// What the manifest's `key` column references.
    pub key_kind: KeyKind,
    /// `per_axis[axis][value]` → image ids the manifest assigned.
    /// Populated only for `key_kind == Image` manifests. For
    /// `key_kind == Result` the caller routes through a different
    /// data structure (label → axis values) — that path is owned by
    /// `vernier.aggregate` / the CLI aggregate verb and not built
    /// here.
    pub per_axis_image: HashMap<String, HashMap<String, HashSet<ImageId>>>,
    /// `per_label[label] -> {axis -> value}` — populated only for
    /// `key_kind == Result` manifests.
    pub per_label: HashMap<String, HashMap<String, String>>,
    /// Non-fatal observations from parsing. Empty when the manifest
    /// resolved cleanly against the caller-supplied key set.
    pub warnings: Vec<ManifestWarning>,
}

/// Parse a JSON-records manifest from raw bytes against a set of
/// known keys.
///
/// The caller supplies the live key set so unknown-key warnings can
/// be emitted up front rather than at evaluate time:
///
/// - For `key_kind == "image_id"` manifests, pass the GT image ids.
/// - For `key_kind == "result"` manifests, pass the result-document
///   labels.
///
/// # Errors
///
/// - [`EvalError::Json`] on malformed JSON bytes.
/// - [`EvalError::InvalidConfig`] when `manifest_version` is not
///   [`MANIFEST_VERSION`], `key_kind` is unknown, axis names contain
///   the cross-product separator, a row is missing the `key` column,
///   axis values are not strings, or rows disagree on the axis-name
///   set (every row must carry the same axes).
pub fn parse_manifest(
    bytes: &[u8],
    known_image_ids: &HashSet<ImageId>,
    known_labels: &HashSet<String>,
) -> Result<ParsedManifest, EvalError> {
    let doc: ManifestDoc = serde_json::from_slice(bytes)?;

    if doc.manifest_version != MANIFEST_VERSION {
        return Err(EvalError::InvalidConfig {
            detail: format!(
                "unsupported manifest_version {:?}; expected {:?}",
                doc.manifest_version, MANIFEST_VERSION
            ),
        });
    }

    let key_kind = match doc.key_kind.as_str() {
        "image_id" => KeyKind::Image,
        "result" => KeyKind::Result,
        other => {
            return Err(EvalError::InvalidConfig {
                detail: format!(
                    "unknown key_kind {other:?}; expected \"image_id\" or \"result\""
                ),
            });
        }
    };

    // Determine the axis-name set from the first row; subsequent rows
    // must agree exactly. Manifest rows are heterogeneous-record
    // tables in the canonical shape, but vernier rejects ragged
    // tables because a missing axis cell is ambiguous (is the image
    // unassigned to that value, or did the user forget the column?).
    let mut axis_names: Option<Vec<String>> = None;
    let mut per_axis_image: HashMap<String, HashMap<String, HashSet<ImageId>>> = HashMap::new();
    let mut per_label: HashMap<String, HashMap<String, String>> = HashMap::new();
    let mut warnings: Vec<ManifestWarning> = Vec::new();

    for (row_idx, row) in doc.rows.iter().enumerate() {
        let row_axes = collect_row_axes(row).map_err(|detail| EvalError::InvalidConfig {
            detail: format!("row {row_idx}: {detail}"),
        })?;
        match &axis_names {
            None => {
                for name in &row_axes {
                    if name.contains(CROSS_SEPARATOR) {
                        return Err(EvalError::InvalidConfig {
                            detail: format!(
                                "manifest axis {name:?} contains the reserved separator \
                                 {CROSS_SEPARATOR:?}; rename the column"
                            ),
                        });
                    }
                }
                axis_names = Some(row_axes.clone());
            }
            Some(prev) => {
                if prev != &row_axes {
                    return Err(EvalError::InvalidConfig {
                        detail: format!(
                            "row {row_idx} axes {row_axes:?} differ from first row {prev:?}; \
                             vernier rejects ragged manifests"
                        ),
                    });
                }
            }
        }

        match key_kind {
            KeyKind::Image => {
                let id = parse_image_id_key(&row.key).map_err(|detail| {
                    EvalError::InvalidConfig {
                        detail: format!("row {row_idx}: {detail}"),
                    }
                })?;
                if !known_image_ids.contains(&id) {
                    warnings.push(ManifestWarning::UnknownKey {
                        key: id.0.to_string(),
                    });
                    continue;
                }
                for axis in &row_axes {
                    let value = parse_axis_value(&row.axes[axis]).map_err(|detail| {
                        EvalError::InvalidConfig {
                            detail: format!("row {row_idx} axis {axis:?}: {detail}"),
                        }
                    })?;
                    per_axis_image
                        .entry(axis.clone())
                        .or_default()
                        .entry(value)
                        .or_default()
                        .insert(id);
                }
            }
            KeyKind::Result => {
                let label =
                    parse_result_label_key(&row.key).map_err(|detail| EvalError::InvalidConfig {
                        detail: format!("row {row_idx}: {detail}"),
                    })?;
                if !known_labels.contains(&label) {
                    warnings.push(ManifestWarning::UnknownKey { key: label });
                    continue;
                }
                let mut row_axis_values: HashMap<String, String> = HashMap::new();
                for axis in &row_axes {
                    let value = parse_axis_value(&row.axes[axis]).map_err(|detail| {
                        EvalError::InvalidConfig {
                            detail: format!("row {row_idx} axis {axis:?}: {detail}"),
                        }
                    })?;
                    row_axis_values.insert(axis.clone(), value);
                }
                per_label.insert(label, row_axis_values);
            }
        }
    }

    Ok(ParsedManifest {
        key_kind,
        per_axis_image,
        per_label,
        warnings,
    })
}

/// Convenience: build the partition spec straight from manifest bytes
/// plus the live grid information.
///
/// Combines [`parse_manifest`] and [`PartitionSpec::build`]. Errors on
/// `key_kind == "result"` because that manifest shape is consumed by
/// `vernier.aggregate`, not by `evaluate_partitioned`.
///
/// # Errors
///
/// As [`parse_manifest`] and [`PartitionSpec::build`], plus
/// [`EvalError::InvalidConfig`] when the manifest is `key_kind ==
/// "result"`.
pub fn partition_spec_from_manifest(
    bytes: &[u8],
    image_id_to_idx: &HashMap<ImageId, usize>,
    cross_axes: &[Vec<String>],
) -> Result<(PartitionSpec, Vec<ManifestWarning>), EvalError> {
    let known_image_ids: HashSet<ImageId> = image_id_to_idx.keys().copied().collect();
    let parsed = parse_manifest(bytes, &known_image_ids, &HashSet::new())?;
    if !matches!(parsed.key_kind, KeyKind::Image) {
        return Err(EvalError::InvalidConfig {
            detail: "evaluate_partitioned consumes key_kind=\"image_id\" manifests; \
                     a key_kind=\"result\" manifest must be routed through \
                     vernier.aggregate / `vernier aggregate`"
                .into(),
        });
    }
    let spec = PartitionSpec::build(
        parsed.key_kind,
        &parsed.per_axis_image,
        &known_image_ids,
        image_id_to_idx,
        cross_axes,
    )?;
    Ok((spec, parsed.warnings))
}

fn collect_row_axes(row: &ManifestRow) -> Result<Vec<String>, String> {
    if row.key == serde_json::Value::Null {
        return Err("missing `key` column".into());
    }
    let mut axes: Vec<String> = row.axes.keys().cloned().collect();
    axes.sort();
    if axes.is_empty() {
        return Err("row has no axis columns beyond `key`".into());
    }
    Ok(axes)
}

fn parse_image_id_key(value: &serde_json::Value) -> Result<ImageId, String> {
    match value {
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(ImageId(i))
            } else {
                Err(format!(
                    "image_id key must be an integer; got {n:?}"
                ))
            }
        }
        serde_json::Value::String(s) => s
            .parse::<i64>()
            .map(ImageId)
            .map_err(|_| format!("image_id key {s:?} is not an integer")),
        other => Err(format!("image_id key must be an integer; got {other:?}")),
    }
}

fn parse_result_label_key(value: &serde_json::Value) -> Result<String, String> {
    match value {
        serde_json::Value::String(s) => Ok(s.clone()),
        other => Err(format!(
            "result-keyed manifest needs a string key; got {other:?}"
        )),
    }
}

fn parse_axis_value(value: &serde_json::Value) -> Result<String, String> {
    match value {
        serde_json::Value::String(s) => Ok(s.clone()),
        // Numeric axis values are aspirational (a future Breakdown-style
        // extension) but we reject them today: ADR-0046 explicitly
        // names the manifest as "categorical string values".
        serde_json::Value::Number(_) => {
            Err("axis values must be strings; numeric slicing is the Breakdown axis".into())
        }
        serde_json::Value::Bool(_) => {
            Err("axis values must be strings; got a JSON boolean".into())
        }
        other => Err(format!("axis values must be strings; got {other:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn known_ids(n: i64) -> HashSet<ImageId> {
        (1..=n).map(ImageId).collect()
    }

    #[test]
    fn parses_minimum_image_manifest() {
        let bytes = br#"{
            "manifest_version": "1",
            "key_kind": "image_id",
            "rows": [
                {"key": 1, "weather": "fog"},
                {"key": 2, "weather": "clear"}
            ]
        }"#;
        let parsed = parse_manifest(bytes, &known_ids(2), &HashSet::new()).unwrap();
        assert_eq!(parsed.key_kind, KeyKind::Image);
        assert!(parsed.warnings.is_empty());
        let weather = parsed.per_axis_image.get("weather").unwrap();
        assert_eq!(weather["fog"].len(), 1);
        assert_eq!(weather["clear"].len(), 1);
    }

    #[test]
    fn unknown_image_id_emits_warning_and_is_skipped() {
        let bytes = br#"{
            "manifest_version": "1",
            "key_kind": "image_id",
            "rows": [
                {"key": 1, "weather": "fog"},
                {"key": 99, "weather": "fog"}
            ]
        }"#;
        let parsed = parse_manifest(bytes, &known_ids(2), &HashSet::new()).unwrap();
        assert_eq!(parsed.warnings.len(), 1);
        assert!(matches!(
            parsed.warnings[0],
            ManifestWarning::UnknownKey { ref key } if key == "99"
        ));
        // Only image 1 should be assigned.
        let weather = parsed.per_axis_image.get("weather").unwrap();
        assert_eq!(weather["fog"].len(), 1);
    }

    #[test]
    fn ragged_axes_are_rejected() {
        let bytes = br#"{
            "manifest_version": "1",
            "key_kind": "image_id",
            "rows": [
                {"key": 1, "weather": "fog", "time": "day"},
                {"key": 2, "weather": "clear"}
            ]
        }"#;
        let err = parse_manifest(bytes, &known_ids(2), &HashSet::new()).unwrap_err();
        assert!(matches!(err, EvalError::InvalidConfig { .. }));
    }

    #[test]
    fn version_mismatch_is_rejected() {
        let bytes = br#"{
            "manifest_version": "2",
            "key_kind": "image_id",
            "rows": []
        }"#;
        let err = parse_manifest(bytes, &known_ids(0), &HashSet::new()).unwrap_err();
        assert!(matches!(err, EvalError::InvalidConfig { .. }));
    }

    #[test]
    fn unknown_key_kind_is_rejected() {
        let bytes = br#"{
            "manifest_version": "1",
            "key_kind": "frame_id",
            "rows": []
        }"#;
        let err = parse_manifest(bytes, &known_ids(0), &HashSet::new()).unwrap_err();
        assert!(matches!(err, EvalError::InvalidConfig { .. }));
    }

    #[test]
    fn axis_value_must_be_string() {
        let bytes = br#"{
            "manifest_version": "1",
            "key_kind": "image_id",
            "rows": [{"key": 1, "weather": 5}]
        }"#;
        let err = parse_manifest(bytes, &known_ids(1), &HashSet::new()).unwrap_err();
        assert!(matches!(err, EvalError::InvalidConfig { .. }));
    }

    #[test]
    fn result_keyed_manifest_collects_per_label_axis_values() {
        let bytes = br#"{
            "manifest_version": "1",
            "key_kind": "result",
            "rows": [
                {"key": "clean", "weather": "clear"},
                {"key": "fog_run", "weather": "fog"}
            ]
        }"#;
        let labels: HashSet<String> =
            ["clean", "fog_run"].iter().map(|s| s.to_string()).collect();
        let parsed = parse_manifest(bytes, &HashSet::new(), &labels).unwrap();
        assert_eq!(parsed.key_kind, KeyKind::Result);
        assert_eq!(parsed.per_label["clean"]["weather"], "clear");
        assert_eq!(parsed.per_label["fog_run"]["weather"], "fog");
    }

    #[test]
    fn partition_spec_helper_rejects_result_keyed_manifest() {
        let bytes = br#"{
            "manifest_version": "1",
            "key_kind": "result",
            "rows": []
        }"#;
        let image_id_to_idx: HashMap<ImageId, usize> = HashMap::new();
        let err = partition_spec_from_manifest(bytes, &image_id_to_idx, &[]).unwrap_err();
        assert!(matches!(err, EvalError::InvalidConfig { .. }));
    }

    #[test]
    fn partition_spec_helper_emits_unassigned_for_unmentioned_images() {
        let bytes = br#"{
            "manifest_version": "1",
            "key_kind": "image_id",
            "rows": [
                {"key": 1, "weather": "fog"}
            ]
        }"#;
        // Three images in the grid; only id=1 mentioned in the manifest.
        let image_id_to_idx: HashMap<ImageId, usize> = (1..=3)
            .map(|i| (ImageId(i), (i - 1) as usize))
            .collect();
        let (spec, _warnings) = partition_spec_from_manifest(bytes, &image_id_to_idx, &[]).unwrap();
        let unassigned = spec
            .slices
            .iter()
            .find(|s| s.axis == "weather" && s.value == crate::partition::UNASSIGNED)
            .expect("unassigned bucket missing");
        let mut ids: Vec<i64> = unassigned.image_ids.iter().map(|i| i.0).collect();
        ids.sort();
        assert_eq!(ids, vec![2, 3]);
    }
}
