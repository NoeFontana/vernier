//! `vernier aggregate` — slice-and-aggregate fan-in (ADR-0046).
//!
//! Reads N result JSON documents (v1 or v2), joins them to a
//! `key_kind=result` manifest by `--label` (falling back to the file
//! path when a result has no label), groups by `(axis, value)`, and
//! emits an `aggregate_version: "1"` envelope with one row per cell
//! and the chosen metric columns meaned across each group's runs.
//!
//! When `--baseline VALUE` is set, every metric column gets a sibling
//! `<metric>__rpc` column carrying `mean(non-baseline) /
//! mean(baseline)` per the mPC / rPC convention (Michaelis et al.
//! NeurIPS-W 2019; ADR-0046 §"`vernier aggregate`").

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process;

use serde_json::{Map, Value};

use vernier_core::manifest::{self, ManifestWarning};
use vernier_core::manifest_csv;
use vernier_core::partition::{KeyKind, UNASSIGNED};

use crate::cli::{AggregateArgs, EmitDestination};
use crate::error::CliError;
use crate::format::aggregate_json::{render, AggregateRow, AggregateV1, AGGREGATE_VERSION};
use crate::format::FormatName;

use super::eval::write_atomic;

/// One result document loaded from disk plus its bookkeeping.
struct LoadedResult {
    /// Path the document was read from. Used as the join key when the
    /// document has no `label` field.
    path: PathBuf,
    /// `--label` value embedded in the document (v2 only — v1 has no
    /// label slot).
    label: Option<String>,
    /// Available metric columns and their numeric values. Built once
    /// at load time so the join loop never re-parses.
    metrics: BTreeMap<String, f64>,
}

/// End-to-end runner for `vernier aggregate`.
pub(crate) fn run(args: &AggregateArgs) -> Result<(), CliError> {
    let emits = args.validate()?;

    // Resolve the glob. Globs returning zero matches are almost always
    // a script bug; raise it loudly.
    let paths = expand_glob(&args.results)?;
    if paths.is_empty() {
        return Err(CliError::Validation(format!(
            "--results glob {:?} matched zero files",
            args.results
        )));
    }

    let results: Vec<LoadedResult> = paths
        .iter()
        .map(|p| load_result(p))
        .collect::<Result<Vec<_>, _>>()?;

    // Parse the manifest. The `key_kind` must be `result` here — we
    // don't try to be clever about misclassified manifests.
    let manifest_bytes = fs::read(&args.manifest).map_err(|source| CliError::InputRead {
        path: args.manifest.clone(),
        source,
    })?;
    let known_labels: HashSet<String> = results.iter().filter_map(|r| r.label.clone()).collect();
    let parsed = parse_manifest_any(&args.manifest, &manifest_bytes, &known_labels)?;
    if !matches!(parsed.key_kind, KeyKind::Result) {
        return Err(CliError::Validation(
            "vernier aggregate consumes key_kind=\"result\" manifests; \
             a key_kind=\"image_id\" manifest belongs on `vernier eval --manifest`"
                .into(),
        ));
    }
    if !args.quiet {
        report_manifest_warnings(&parsed.warnings);
    }

    // Join: for each result, look up its label in `per_label`. When
    // the result has no label, try the file path (without extension,
    // basename) as a fallback. Unjoinable results emit a warning and
    // are skipped — the same "no silent data loss" discipline used
    // throughout this codebase.
    //
    // We don't pre-filter the results; we walk and accumulate so the
    // warning path lives in one place.
    let mut joined: Vec<(HashMap<String, String>, &LoadedResult)> = Vec::new();
    for r in &results {
        let join_key = r
            .label
            .clone()
            .unwrap_or_else(|| basename_without_ext(&r.path));
        match parsed.per_label.get(&join_key) {
            Some(axes) => joined.push((axes.clone(), r)),
            None => {
                if !args.quiet {
                    let stderr = io::stderr();
                    let mut handle = stderr.lock();
                    let _ = writeln!(
                        handle,
                        "warning: result {} has no manifest row (join key {:?}); skipping",
                        r.path.display(),
                        join_key
                    );
                }
            }
        }
    }

    // Decide the metric column set. Defaults: every metric that
    // appears in at least one joined result, sorted for stability.
    let metric_names: Vec<String> = resolve_metric_columns(&args.metric, &joined)?;

    // Group joined runs by (axis, value) and mean each metric.
    let groups = group_runs(&joined);

    // Materialize rows in canonical order (axis ascending, value
    // ascending, __unassigned__ last per axis).
    let mut rows_intermediate: Vec<RowAccum> = groups.into_iter().collect();
    rows_intermediate.sort_by(|a, b| {
        a.axis
            .cmp(&b.axis)
            .then_with(|| canonical_value_cmp(&a.value, &b.value))
    });

    // Compute baseline means per (axis, metric) so the rPC pass below
    // can divide without an N^2 walk.
    let baseline_means: HashMap<(String, String), f64> = match &args.baseline {
        Some(b) => baseline_table(&rows_intermediate, b, &metric_names),
        None => HashMap::new(),
    };

    let final_metrics_order: Vec<String> = if args.baseline.is_some() {
        // Append <metric>__rpc after every metric, preserving the
        // base-metric ordering. The columns line up so a downstream
        // reader can pair them by name.
        let mut out: Vec<String> = Vec::with_capacity(metric_names.len() * 2);
        for m in &metric_names {
            out.push(m.clone());
            out.push(format!("{m}__rpc"));
        }
        out
    } else {
        metric_names.clone()
    };

    let mut rows_out: Vec<RenderRow> = Vec::with_capacity(rows_intermediate.len());
    for row in &rows_intermediate {
        let mut map: Map<String, Value> = Map::new();
        for m in &metric_names {
            let mean = row.metric_mean(m);
            insert_number(&mut map, m, mean);
            if args.baseline.is_some() {
                let baseline_mean = baseline_means.get(&(row.axis.clone(), m.clone())).copied();
                let rpc = compute_rpc(mean, baseline_mean);
                insert_number(&mut map, &format!("{m}__rpc"), rpc);
            }
        }
        rows_out.push(RenderRow {
            axis: row.axis.clone(),
            value: row.value.clone(),
            n_runs: row.n_runs() as u64,
            metrics: map,
        });
    }

    let metric_strs: Vec<&str> = final_metrics_order.iter().map(String::as_str).collect();
    let rows_borrowed: Vec<AggregateRow<'_>> = rows_out
        .iter()
        .map(|r| AggregateRow {
            axis: r.axis.as_str(),
            value: r.value.as_str(),
            n_runs: r.n_runs,
            metrics: r.metrics.clone(),
        })
        .collect();
    let doc = AggregateV1 {
        aggregate_version: AGGREGATE_VERSION,
        baseline: args.baseline.as_deref(),
        metrics: metric_strs,
        rows: rows_borrowed,
    };

    for spec in &emits {
        match (spec.format, &spec.destination) {
            (FormatName::Json, EmitDestination::Stdout) => {
                let stdout = io::stdout();
                let mut handle = stdout.lock();
                render(&doc, &mut handle)?;
            }
            (FormatName::Json, EmitDestination::File(path)) => {
                write_atomic(path, |w| render(&doc, w))?;
            }
            (FormatName::Text, dest) => {
                let render_text = |w: &mut dyn io::Write| render_text(&doc, w);
                match dest {
                    EmitDestination::Stdout => {
                        let stdout = io::stdout();
                        let mut handle = stdout.lock();
                        render_text(&mut handle)?;
                    }
                    EmitDestination::File(path) => {
                        write_atomic(path, |w| render_text(w))?;
                    }
                }
            }
        }
    }
    Ok(())
}

/// Process-exit shim for `main.rs`. Mirrors `eval::run_or_exit`.
pub(crate) fn run_or_exit(args: &AggregateArgs) -> ! {
    let quiet = args.quiet;
    match run(args) {
        Ok(()) => process::exit(0),
        Err(err) => {
            if !quiet {
                let mut stderr = io::stderr().lock();
                let _ = writeln!(stderr, "error: {err}");
            }
            process::exit(err.exit_code());
        }
    }
}

/// Per-run, per-metric accumulator. One entry per joined result.
struct RowAccum {
    axis: String,
    value: String,
    runs: Vec<BTreeMap<String, f64>>,
}

impl RowAccum {
    fn n_runs(&self) -> usize {
        self.runs.len()
    }

    /// Mean of `metric` across runs that carry it. NaN when no run
    /// emitted the column (encoded as JSON null at serialize time).
    fn metric_mean(&self, metric: &str) -> Option<f64> {
        let mut sum = 0.0_f64;
        let mut n = 0_u64;
        for r in &self.runs {
            if let Some(v) = r.get(metric) {
                if v.is_finite() {
                    sum += *v;
                    n += 1;
                }
            }
        }
        if n == 0 {
            None
        } else {
            Some(sum / (n as f64))
        }
    }
}

/// Group `joined` rows by `(axis, value)`. Returns the rows in
/// arbitrary order (the caller sorts).
fn group_runs(joined: &[(HashMap<String, String>, &LoadedResult)]) -> Vec<RowAccum> {
    // (axis, value) -> index into `acc`
    let mut index: HashMap<(String, String), usize> = HashMap::new();
    let mut acc: Vec<RowAccum> = Vec::new();
    for (axes_map, result) in joined {
        for (axis, value) in axes_map {
            let key = (axis.clone(), value.clone());
            let idx = match index.get(&key) {
                Some(i) => *i,
                None => {
                    let i = acc.len();
                    acc.push(RowAccum {
                        axis: axis.clone(),
                        value: value.clone(),
                        runs: Vec::new(),
                    });
                    index.insert(key, i);
                    i
                }
            };
            acc[idx].runs.push(result.metrics.clone());
        }
    }
    acc
}

/// Pre-compute the per-(axis, metric) baseline mean so the rPC pass
/// is O(rows × metrics) instead of O(rows × metrics × n_runs).
fn baseline_table(
    rows: &[RowAccum],
    baseline_value: &str,
    metric_names: &[String],
) -> HashMap<(String, String), f64> {
    let mut out: HashMap<(String, String), f64> = HashMap::new();
    for row in rows {
        if row.value != baseline_value {
            continue;
        }
        for m in metric_names {
            if let Some(mean) = row.metric_mean(m) {
                out.insert((row.axis.clone(), m.clone()), mean);
            }
        }
    }
    out
}

fn compute_rpc(value: Option<f64>, baseline: Option<f64>) -> Option<f64> {
    match (value, baseline) {
        (Some(v), Some(b)) if b != 0.0 && b.is_finite() => Some(v / b),
        _ => None,
    }
}

fn insert_number(map: &mut Map<String, Value>, key: &str, value: Option<f64>) {
    let v = match value {
        Some(x) if x.is_finite() => Value::from(x),
        _ => Value::Null,
    };
    map.insert(key.to_string(), v);
}

/// Canonical-order comparator: `__unassigned__` sorts last; everything
/// else uses lexical order. Mirrors the per-axis-value ordering
/// `PartitionSpec::build` produces.
fn canonical_value_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match (a == UNASSIGNED, b == UNASSIGNED) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => a.cmp(b),
    }
}

/// Borrowed row carrier — separate from `RowAccum` so the
/// `AggregateRow<'a>` lifetime can be tied to the rendered
/// document's lifetime cleanly.
struct RenderRow {
    axis: String,
    value: String,
    n_runs: u64,
    metrics: Map<String, Value>,
}

/// Minimal text rendering: a header, then one row per `(axis, value)`
/// with `metric=value` columns. Not a stability surface — the JSON
/// envelope is the contract.
fn render_text(doc: &AggregateV1<'_>, out: &mut dyn io::Write) -> Result<(), CliError> {
    writeln!(out, "aggregate_version = {}", doc.aggregate_version)?;
    if let Some(b) = doc.baseline {
        writeln!(out, "baseline = {b}")?;
    }
    writeln!(out, "metrics = {}", doc.metrics.join(", "))?;
    for row in &doc.rows {
        write!(
            out,
            "  axis={} value={} n_runs={}",
            row.axis, row.value, row.n_runs
        )?;
        for m in &doc.metrics {
            let cell = row.metrics.get(*m).cloned().unwrap_or(Value::Null);
            let rendered = match cell {
                Value::Null => "NaN".to_string(),
                Value::Number(n) => n.to_string(),
                other => other.to_string(),
            };
            write!(out, "  {m}={rendered}")?;
        }
        writeln!(out)?;
    }
    Ok(())
}

fn expand_glob(pattern: &str) -> Result<Vec<PathBuf>, CliError> {
    let entries = glob::glob(pattern).map_err(|e| {
        CliError::Validation(format!("--results glob {pattern:?} is malformed: {e}"))
    })?;
    let mut out: Vec<PathBuf> = Vec::new();
    for entry in entries {
        let path = entry.map_err(|e| {
            CliError::Validation(format!(
                "--results glob {pattern:?} produced an unreadable entry: {e}"
            ))
        })?;
        out.push(path);
    }
    // Stable order for downstream determinism.
    out.sort();
    out.dedup();
    Ok(out)
}

fn load_result(path: &Path) -> Result<LoadedResult, CliError> {
    let bytes = fs::read(path).map_err(|source| CliError::InputRead {
        path: path.to_path_buf(),
        source,
    })?;
    let json: Value = serde_json::from_slice(&bytes)?;
    let version = json.get("version").and_then(Value::as_str).ok_or_else(|| {
        CliError::Validation(format!(
            "result {} has no `version` field; not a vernier eval document",
            path.display()
        ))
    })?;
    let label = json
        .get("label")
        .and_then(Value::as_str)
        .map(str::to_string);
    let metrics = match version {
        "1" => extract_metrics_v1(&json, path)?,
        "2" => extract_metrics_v2(&json, path)?,
        other => {
            return Err(CliError::Validation(format!(
                "result {} has unrecognized version {:?}; expected \"1\" or \"2\"",
                path.display(),
                other
            )));
        }
    };
    Ok(LoadedResult {
        path: path.to_path_buf(),
        label,
        metrics,
    })
}

/// Read the `lines` array of a v1 / v2 `overall` block and turn each
/// line into a `(canonical_name, value)` pair. Aliases like `ap` are
/// added on top so users can refer to the most common stats by their
/// pycocotools-style nickname.
fn extract_metrics_v1(json: &Value, path: &Path) -> Result<BTreeMap<String, f64>, CliError> {
    let lines = json.get("lines").and_then(Value::as_array).ok_or_else(|| {
        CliError::Validation(format!("result {} v1 has no `lines` array", path.display()))
    })?;
    Ok(lines_to_metrics(lines))
}

fn extract_metrics_v2(json: &Value, path: &Path) -> Result<BTreeMap<String, f64>, CliError> {
    let lines = json
        .get("overall")
        .and_then(|o| o.get("lines"))
        .and_then(Value::as_array)
        .ok_or_else(|| {
            CliError::Validation(format!(
                "result {} v2 has no `overall.lines` array",
                path.display()
            ))
        })?;
    Ok(lines_to_metrics(lines))
}

/// Build a `{name -> value}` map from a `lines` array.
///
/// Each line gets two names: a canonical
/// `<metric>_<iou_label>_<area>_<max_dets>` (always unique within a
/// summary), and — when the line matches a pycocotools-table position
/// — a short nickname like `ap`, `ap50`, `ar_100`. Both are exposed so
/// `--metric ap` and `--metric AP_0.50:0.95_all_100` both resolve.
fn lines_to_metrics(lines: &[Value]) -> BTreeMap<String, f64> {
    let mut out: BTreeMap<String, f64> = BTreeMap::new();
    for (idx, line) in lines.iter().enumerate() {
        let metric = line.get("metric").and_then(Value::as_str).unwrap_or("?");
        let iou_label = line
            .get("iou_threshold_label")
            .and_then(Value::as_str)
            .unwrap_or("?");
        let area = line.get("area").and_then(Value::as_str).unwrap_or("?");
        let max_dets = line.get("max_dets").and_then(Value::as_u64).unwrap_or(0);
        let value = line.get("value").and_then(Value::as_f64);
        if let Some(v) = value {
            let canonical = format!("{metric}_{iou_label}_{area}_{max_dets}");
            out.insert(canonical, v);
            if let Some(alias) = position_alias(metric, iou_label, area, max_dets, idx) {
                out.entry(alias).or_insert(v);
            }
        }
    }
    out
}

/// Pycocotools' 12-stat (detection) and 10-stat (keypoints) tables
/// have well-known nicknames. We exploit only the unambiguous ones —
/// `ap` / `ap50` / `ap75` / `ap_small` / etc. The rest of the table
/// goes through the canonical naming.
fn position_alias(
    metric: &str,
    iou_label: &str,
    area: &str,
    max_dets: u64,
    _idx: usize,
) -> Option<String> {
    match (metric, iou_label, area, max_dets) {
        ("AP", "0.50:0.95", "all", _) => Some("ap".into()),
        ("AP", "0.50", "all", _) => Some("ap50".into()),
        ("AP", "0.75", "all", _) => Some("ap75".into()),
        ("AP", "0.50:0.95", "small", _) => Some("ap_small".into()),
        ("AP", "0.50:0.95", "medium", _) => Some("ap_medium".into()),
        ("AP", "0.50:0.95", "large", _) => Some("ap_large".into()),
        ("AR", "0.50:0.95", "all", 1) => Some("ar_1".into()),
        ("AR", "0.50:0.95", "all", 10) => Some("ar_10".into()),
        ("AR", "0.50:0.95", "all", 100) => Some("ar_100".into()),
        ("AR", "0.50:0.95", "small", _) => Some("ar_small".into()),
        ("AR", "0.50:0.95", "medium", _) => Some("ar_medium".into()),
        ("AR", "0.50:0.95", "large", _) => Some("ar_large".into()),
        _ => None,
    }
}

/// Resolve the metric-column set the user asked for, or the default
/// (every alias that appears in at least one loaded result, in stable
/// order).
fn resolve_metric_columns(
    user_metrics: &[String],
    joined: &[(HashMap<String, String>, &LoadedResult)],
) -> Result<Vec<String>, CliError> {
    if !user_metrics.is_empty() {
        // Validate: every requested column must exist on at least one
        // loaded result. Without this guard a typo silently produces a
        // table of nulls.
        let mut all_names: HashSet<&str> = HashSet::new();
        for (_, r) in joined {
            for k in r.metrics.keys() {
                all_names.insert(k.as_str());
            }
        }
        for m in user_metrics {
            if !all_names.contains(m.as_str()) {
                return Err(CliError::Validation(format!(
                    "--metric {m:?} does not appear on any joined result; \
                     available metrics include aliases ap / ap50 / ap75 / ar_100 plus the \
                     canonical <metric>_<iou_label>_<area>_<max_dets> form"
                )));
            }
        }
        return Ok(user_metrics.to_vec());
    }
    let mut all: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for (_, r) in joined {
        for k in r.metrics.keys() {
            if seen.insert(k.clone()) {
                all.push(k.clone());
            }
        }
    }
    all.sort();
    Ok(all)
}

fn parse_manifest_any(
    path: &Path,
    bytes: &[u8],
    known_labels: &HashSet<String>,
) -> Result<vernier_core::manifest::ParsedManifest, CliError> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase);
    let parsed = match ext.as_deref() {
        Some("json") | None => manifest::parse_manifest(bytes, &HashSet::new(), known_labels)?,
        Some("csv") => {
            manifest_csv::parse_csv_manifest(bytes, KeyKind::Result, &HashSet::new(), known_labels)?
        }
        Some(other) => {
            return Err(CliError::Validation(format!(
                "manifest extension {other:?} is not recognized; use .json or .csv"
            )));
        }
    };
    Ok(parsed)
}

fn report_manifest_warnings(warnings: &[ManifestWarning]) {
    if warnings.is_empty() {
        return;
    }
    let stderr = io::stderr();
    let mut handle = stderr.lock();
    for w in warnings {
        match w {
            ManifestWarning::UnknownKey { key } => {
                let _ = writeln!(
                    handle,
                    "warning: manifest references unknown label {key:?}; row skipped"
                );
            }
        }
    }
}

fn basename_without_ext(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}
