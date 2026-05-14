//! CSV manifest adapter (ADR-0046 §D2).
//!
//! Robotics attribute tables are spreadsheet-native; this module
//! accepts the CSV form documented in ADR-0046 §"Manifest schema" and
//! converts it to the canonical JSON-records shape that
//! [`crate::manifest::parse_manifest`] consumes. By design there is no
//! second validation surface: the CSV path normalizes to canonical
//! JSON and the JSON parser owns every semantic check.
//!
//! Wire shape:
//!
//! - First column header **must** be `key`. The cell underneath each
//!   row's `key` is the manifest's primary key — image id (parsed as
//!   an integer) for `KeyKind::Image`, run label (string) for
//!   `KeyKind::Result`. The CSV file itself carries no `key_kind`
//!   discriminator — the caller declares it via [`parse_csv_manifest`]
//!   / [`csv_to_canonical_json`].
//! - Every other column header is an axis name; every other cell is
//!   the categorical string value for that `(row, axis)`. Numeric
//!   slicing is the [`crate::breakdown`] axis, not this one.
//! - Empty cells in an axis column are rejected — same discipline the
//!   JSON path applies to ragged manifests (every row must carry a
//!   value for every axis).
//! - UTF-8 BOM (`EF BB BF`) and Windows line endings (`\r\n`) are
//!   tolerated. Quoted strings have their surrounding quotes stripped.
//!
//! The `Vec<u8>` JSON the [`csv_to_canonical_json`] helper returns is
//! the same bytes [`crate::manifest::parse_manifest`] would consume
//! from disk — round-tripping a CSV through this module produces a
//! byte slice the rest of the pipeline cannot distinguish from a
//! handwritten JSON manifest.

use std::collections::HashSet;
use std::io::Cursor;

use serde_json::{Map, Value};

use crate::dataset::ImageId;
use crate::error::EvalError;
use crate::manifest::{parse_manifest, ParsedManifest, MANIFEST_VERSION};
use crate::partition::KeyKind;

/// First-column header expected on every CSV manifest. Carrying the
/// name in a constant lets the helpers below give targeted error
/// messages without re-typing the literal.
const KEY_COLUMN: &str = "key";

/// Convert raw CSV bytes to the canonical JSON-records shape the
/// rest of the pipeline consumes.
///
/// Useful for diagnostic flows that want to inspect what the CSV path
/// is feeding the JSON validator. The returned bytes are valid input
/// for [`crate::manifest::parse_manifest`].
///
/// # Errors
///
/// - [`EvalError::InvalidConfig`] when the CSV is malformed (missing
///   header row, missing `key` column, ragged rows, BOM in a non-first
///   header field, empty cells in an axis column, image-keyed manifest
///   with a non-integer key cell).
/// - [`EvalError::InvalidConfig`] wrapping the underlying [`csv::Error`]
///   when the CSV parser refuses the bytes outright.
pub fn csv_to_canonical_json(bytes: &[u8], key_kind: KeyKind) -> Result<Vec<u8>, EvalError> {
    let stripped = strip_utf8_bom(bytes);
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(false)
        .from_reader(Cursor::new(stripped));

    let headers = reader
        .headers()
        .map_err(|e| EvalError::InvalidConfig {
            detail: format!("manifest CSV header read failed: {e}"),
        })?
        .clone();

    if headers.is_empty() {
        return Err(EvalError::InvalidConfig {
            detail: "manifest CSV has no header row".into(),
        });
    }
    let first_header = headers
        .get(0)
        .ok_or_else(|| EvalError::InvalidConfig {
            detail: "manifest CSV header is empty".into(),
        })?
        .trim();
    if first_header != KEY_COLUMN {
        return Err(EvalError::InvalidConfig {
            detail: format!(
                "manifest CSV: first column must be {KEY_COLUMN:?}, got {first_header:?}"
            ),
        });
    }
    if headers.len() < 2 {
        return Err(EvalError::InvalidConfig {
            detail: "manifest CSV must carry at least one axis column beyond `key`".into(),
        });
    }

    let axis_names: Vec<String> = headers
        .iter()
        .skip(1)
        .map(|h| h.trim().to_string())
        .collect();
    let mut seen_axes: HashSet<&str> = HashSet::with_capacity(axis_names.len());
    for axis in &axis_names {
        if axis.is_empty() {
            return Err(EvalError::InvalidConfig {
                detail: "manifest CSV axis column header is empty".into(),
            });
        }
        if !seen_axes.insert(axis.as_str()) {
            return Err(EvalError::InvalidConfig {
                detail: format!("manifest CSV duplicates axis column {axis:?}"),
            });
        }
    }

    let mut rows_out: Vec<Value> = Vec::new();
    for (row_idx, record) in reader.records().enumerate() {
        let record = record.map_err(|e| EvalError::InvalidConfig {
            detail: format!("manifest CSV row {row_idx}: {e}"),
        })?;
        if record.len() != headers.len() {
            // `flexible(false)` on the reader makes the underlying
            // parser already raise on ragged rows; this is a defensive
            // belt-and-suspenders check in case a future builder
            // change drops the strictness flag.
            return Err(EvalError::InvalidConfig {
                detail: format!(
                    "manifest CSV row {row_idx} has {} cells but header has {}",
                    record.len(),
                    headers.len()
                ),
            });
        }

        let raw_key = record
            .get(0)
            .ok_or_else(|| EvalError::InvalidConfig {
                detail: format!("manifest CSV row {row_idx} is missing its `key` cell"),
            })?
            .trim();
        if raw_key.is_empty() {
            return Err(EvalError::InvalidConfig {
                detail: format!("manifest CSV row {row_idx} has an empty `key` cell"),
            });
        }
        let key_value: Value = match key_kind {
            KeyKind::Image => {
                let id: i64 = raw_key.parse().map_err(|_| EvalError::InvalidConfig {
                    detail: format!(
                        "manifest CSV row {row_idx} key {raw_key:?} is not an integer image_id"
                    ),
                })?;
                Value::Number(serde_json::Number::from(id))
            }
            KeyKind::Result => Value::String(raw_key.to_string()),
        };

        let mut row_obj: Map<String, Value> = Map::new();
        row_obj.insert(KEY_COLUMN.to_string(), key_value);
        for (axis_idx, axis_name) in axis_names.iter().enumerate() {
            // +1 to skip the `key` column.
            let cell = record
                .get(axis_idx + 1)
                .ok_or_else(|| EvalError::InvalidConfig {
                    detail: format!(
                        "manifest CSV row {row_idx} missing cell for axis {axis_name:?}"
                    ),
                })?
                .trim();
            if cell.is_empty() {
                return Err(EvalError::InvalidConfig {
                    detail: format!(
                        "manifest CSV row {row_idx} has an empty cell for axis {axis_name:?}; \
                         every row must carry every axis"
                    ),
                });
            }
            row_obj.insert(axis_name.clone(), Value::String(cell.to_string()));
        }
        rows_out.push(Value::Object(row_obj));
    }

    let mut top: Map<String, Value> = Map::new();
    top.insert(
        "manifest_version".to_string(),
        Value::String(MANIFEST_VERSION.to_string()),
    );
    top.insert(
        "key_kind".to_string(),
        Value::String(key_kind_str(key_kind).to_string()),
    );
    top.insert("rows".to_string(), Value::Array(rows_out));
    let doc = Value::Object(top);

    serde_json::to_vec(&doc).map_err(EvalError::from)
}

/// Parse a CSV manifest from raw bytes against a set of known keys.
///
/// Equivalent to running [`csv_to_canonical_json`] then handing the
/// produced bytes to [`crate::manifest::parse_manifest`]; this entry
/// point is the one CLI / Python lanes call so the canonical JSON
/// intermediate stays an implementation detail.
///
/// # Errors
///
/// As [`csv_to_canonical_json`], plus every error
/// [`crate::manifest::parse_manifest`] can raise on the synthesized
/// canonical-JSON form.
pub fn parse_csv_manifest(
    bytes: &[u8],
    key_kind: KeyKind,
    known_image_ids: &HashSet<ImageId>,
    known_labels: &HashSet<String>,
) -> Result<ParsedManifest, EvalError> {
    let json = csv_to_canonical_json(bytes, key_kind)?;
    parse_manifest(&json, known_image_ids, known_labels)
}

fn key_kind_str(kind: KeyKind) -> &'static str {
    match kind {
        KeyKind::Image => "image_id",
        KeyKind::Result => "result",
    }
}

/// Drop a leading UTF-8 BOM (`EF BB BF`) if present.
///
/// `csv::Reader` does not strip BOMs on its own, and a BOM in the
/// `key` header is the most common reason a freshly-exported
/// spreadsheet refuses to parse here. Strip once at the top of the
/// pipeline so the rest of the code sees clean header text.
fn strip_utf8_bom(bytes: &[u8]) -> &[u8] {
    const BOM: &[u8; 3] = b"\xEF\xBB\xBF";
    if bytes.starts_with(BOM) {
        &bytes[BOM.len()..]
    } else {
        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn known_ids(n: i64) -> HashSet<ImageId> {
        (1..=n).map(ImageId).collect()
    }

    #[test]
    fn parses_minimum_image_manifest() {
        let bytes = b"key,weather\n1,fog\n2,clear\n";
        let parsed =
            parse_csv_manifest(bytes, KeyKind::Image, &known_ids(2), &HashSet::new()).unwrap();
        assert_eq!(parsed.key_kind, KeyKind::Image);
        assert!(parsed.warnings.is_empty());
        let weather = parsed.per_axis_image.get("weather").unwrap();
        assert_eq!(weather["fog"].len(), 1);
        assert_eq!(weather["clear"].len(), 1);
    }

    #[test]
    fn ragged_csv_is_rejected() {
        // Row 2 is missing the `time` cell.
        let bytes = b"key,weather,time\n1,fog,night\n2,clear\n";
        let err =
            parse_csv_manifest(bytes, KeyKind::Image, &known_ids(2), &HashSet::new()).unwrap_err();
        assert!(matches!(err, EvalError::InvalidConfig { .. }));
    }

    #[test]
    fn first_column_must_be_key() {
        let bytes = b"image_id,weather\n1,fog\n";
        let err =
            parse_csv_manifest(bytes, KeyKind::Image, &known_ids(2), &HashSet::new()).unwrap_err();
        assert!(matches!(err, EvalError::InvalidConfig { .. }));
    }

    #[test]
    fn key_column_in_position_one_parses_axis_cells_bare() {
        // The literal cell `"fog"` (no surrounding quotes in source)
        // must parse to the categorical value "fog".
        let bytes = b"key,weather\n1,fog\n";
        let parsed =
            parse_csv_manifest(bytes, KeyKind::Image, &known_ids(1), &HashSet::new()).unwrap();
        let weather = parsed.per_axis_image.get("weather").unwrap();
        assert_eq!(weather["fog"].len(), 1);
    }

    #[test]
    fn utf8_bom_is_tolerated() {
        // Same bytes as `parses_minimum_image_manifest`, prefixed with
        // a UTF-8 BOM (the byte sequence Excel slaps on every CSV
        // export from a Windows machine).
        let mut bytes: Vec<u8> = vec![0xEF, 0xBB, 0xBF];
        bytes.extend_from_slice(b"key,weather\n1,fog\n2,clear\n");
        let parsed =
            parse_csv_manifest(&bytes, KeyKind::Image, &known_ids(2), &HashSet::new()).unwrap();
        let weather = parsed.per_axis_image.get("weather").unwrap();
        assert_eq!(weather["fog"].len(), 1);
        assert_eq!(weather["clear"].len(), 1);
    }

    #[test]
    fn windows_line_endings_are_tolerated() {
        let bytes = b"key,weather\r\n1,fog\r\n2,clear\r\n";
        let parsed =
            parse_csv_manifest(bytes, KeyKind::Image, &known_ids(2), &HashSet::new()).unwrap();
        let weather = parsed.per_axis_image.get("weather").unwrap();
        assert_eq!(weather["fog"].len(), 1);
        assert_eq!(weather["clear"].len(), 1);
    }

    #[test]
    fn quoted_strings_unwrap_to_bare_values() {
        // RFC-4180 quoted fields. `csv::Reader` strips the surrounding
        // quotes; the manifest cell ends up as the unquoted string.
        let bytes = b"key,weather\n1,\"fog\"\n2,\"partly clear\"\n";
        let parsed =
            parse_csv_manifest(bytes, KeyKind::Image, &known_ids(2), &HashSet::new()).unwrap();
        let weather = parsed.per_axis_image.get("weather").unwrap();
        assert_eq!(weather["fog"].len(), 1);
        assert_eq!(weather["partly clear"].len(), 1);
    }

    #[test]
    fn result_keyed_csv_round_trips_to_per_label() {
        let bytes = b"key,weather\nrun_clean,clear\nrun_fog,fog\n";
        let mut labels: HashSet<String> = HashSet::new();
        labels.insert("run_clean".into());
        labels.insert("run_fog".into());
        let parsed = parse_csv_manifest(bytes, KeyKind::Result, &HashSet::new(), &labels).unwrap();
        assert_eq!(parsed.key_kind, KeyKind::Result);
        assert_eq!(parsed.per_label["run_clean"]["weather"], "clear");
        assert_eq!(parsed.per_label["run_fog"]["weather"], "fog");
    }

    #[test]
    fn non_integer_image_key_is_rejected() {
        let bytes = b"key,weather\nfoo,fog\n";
        let err =
            parse_csv_manifest(bytes, KeyKind::Image, &known_ids(2), &HashSet::new()).unwrap_err();
        assert!(matches!(err, EvalError::InvalidConfig { .. }));
    }

    #[test]
    fn empty_axis_cell_is_rejected() {
        let bytes = b"key,weather\n1,\n";
        let err =
            parse_csv_manifest(bytes, KeyKind::Image, &known_ids(1), &HashSet::new()).unwrap_err();
        assert!(matches!(err, EvalError::InvalidConfig { .. }));
    }

    #[test]
    fn csv_to_canonical_json_round_trips_through_parse_manifest() {
        // The canonical-JSON intermediate must itself parse cleanly,
        // proving the converter does not produce a shape the JSON
        // validator would reject.
        let bytes = b"key,weather,time\n1,fog,night\n2,clear,day\n";
        let json = csv_to_canonical_json(bytes, KeyKind::Image).unwrap();
        let parsed = parse_manifest(&json, &known_ids(2), &HashSet::new()).unwrap();
        let weather = parsed.per_axis_image.get("weather").unwrap();
        let time = parsed.per_axis_image.get("time").unwrap();
        assert_eq!(weather["fog"].len(), 1);
        assert_eq!(weather["clear"].len(), 1);
        assert_eq!(time["night"].len(), 1);
        assert_eq!(time["day"].len(), 1);
    }
}
