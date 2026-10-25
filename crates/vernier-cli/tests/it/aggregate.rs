//! Integration tests for `vernier aggregate` (ADR-0046 Phase 3).
//!
//! The aggregate verb is a pure data-shaping pass over already-produced
//! result documents; these tests synthesize the result JSONs inline
//! (the full `eval --manifest` -> file -> `aggregate` round-trip is
//! covered by the eval tests in [`eval_manifest`]).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};

use assert_cmd::Command;

/// Synthesize a minimal v1 result JSON with the canonical 12 detection
/// lines + a `label` field. Only `metric`, `iou_threshold_label`,
/// `area`, `max_dets`, and `value` are read by aggregate; the rest is
/// padding to keep the doc parseable as a v1 eval result.
fn make_result_v1(label: &str, ap: f64) -> String {
    // Lay out the 12 canonical (metric, iou_label, area, max_dets)
    // tuples. Values follow the AP scale so the aliases resolve.
    let lines = serde_json::json!([
        {"metric": "AP", "iou_threshold": null, "iou_threshold_label": "0.50:0.95", "area": "all",    "max_dets": 100, "value": ap},
        {"metric": "AP", "iou_threshold": 0.5,  "iou_threshold_label": "0.50",      "area": "all",    "max_dets": 100, "value": ap},
        {"metric": "AP", "iou_threshold": 0.75, "iou_threshold_label": "0.75",      "area": "all",    "max_dets": 100, "value": ap},
        {"metric": "AP", "iou_threshold": null, "iou_threshold_label": "0.50:0.95", "area": "small",  "max_dets": 100, "value": ap},
        {"metric": "AP", "iou_threshold": null, "iou_threshold_label": "0.50:0.95", "area": "medium", "max_dets": 100, "value": ap},
        {"metric": "AP", "iou_threshold": null, "iou_threshold_label": "0.50:0.95", "area": "large",  "max_dets": 100, "value": ap},
        {"metric": "AR", "iou_threshold": null, "iou_threshold_label": "0.50:0.95", "area": "all",    "max_dets": 1,   "value": ap},
        {"metric": "AR", "iou_threshold": null, "iou_threshold_label": "0.50:0.95", "area": "all",    "max_dets": 10,  "value": ap},
        {"metric": "AR", "iou_threshold": null, "iou_threshold_label": "0.50:0.95", "area": "all",    "max_dets": 100, "value": ap},
        {"metric": "AR", "iou_threshold": null, "iou_threshold_label": "0.50:0.95", "area": "small",  "max_dets": 100, "value": ap},
        {"metric": "AR", "iou_threshold": null, "iou_threshold_label": "0.50:0.95", "area": "medium", "max_dets": 100, "value": ap},
        {"metric": "AR", "iou_threshold": null, "iou_threshold_label": "0.50:0.95", "area": "large",  "max_dets": 100, "value": ap}
    ]);
    let stats: Vec<f64> = lines
        .as_array()
        .unwrap()
        .iter()
        .map(|l| l["value"].as_f64().unwrap())
        .collect();
    let doc = serde_json::json!({
        "version": "1",
        "label": label,
        "iou_type": "bbox",
        "parity_mode": "strict",
        "max_dets": [1, 10, 100],
        "use_cats": true,
        "lines": lines,
        "stats": stats,
    });
    serde_json::to_string(&doc).unwrap()
}

fn manifest_json() -> &'static str {
    r#"{
        "manifest_version": "1",
        "key_kind": "result",
        "rows": [
            {"key": "run_clean", "weather": "clean"},
            {"key": "run_fog",   "weather": "fog"},
            {"key": "run_noise", "weather": "noise"}
        ]
    }"#
}

fn write_three_runs(tmp: &Path) -> (PathBuf, PathBuf, PathBuf, PathBuf) {
    // High AP for clean, lower for fog, lowest for noise.
    let run_a = tmp.join("run_clean.json");
    let run_b = tmp.join("run_fog.json");
    let run_c = tmp.join("run_noise.json");
    std::fs::write(&run_a, make_result_v1("run_clean", 0.80)).unwrap();
    std::fs::write(&run_b, make_result_v1("run_fog", 0.40)).unwrap();
    std::fs::write(&run_c, make_result_v1("run_noise", 0.20)).unwrap();
    let manifest = tmp.join("corruptions.json");
    std::fs::write(&manifest, manifest_json()).unwrap();
    (run_a, run_b, run_c, manifest)
}

fn run_aggregate(manifest: &Path, glob: &str, out: &Path, extra: &[&str]) -> std::process::Output {
    let mut cmd = Command::cargo_bin("vernier").unwrap();
    let mut args: Vec<String> = vec![
        "aggregate".into(),
        "--manifest".into(),
        manifest.to_string_lossy().into_owned(),
        "--results".into(),
        glob.to_string(),
        "--emit".into(),
        format!("json={}", out.display()),
    ];
    args.extend(extra.iter().map(|s| (*s).to_string()));
    cmd.args(args).output().unwrap()
}

#[test]
fn baseline_appends_rpc_columns() {
    let tmp = tempdir();
    let (_a, _b, _c, manifest) = write_three_runs(tmp.path());
    let glob = tmp.path().join("run_*.json").to_string_lossy().into_owned();
    let out = tmp.path().join("summary.json");

    let output = run_aggregate(
        &manifest,
        &glob,
        &out,
        &["--baseline", "clean", "--metric", "ap"],
    );
    assert_eq!(
        output.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let parsed: serde_json::Value = serde_json::from_slice(&std::fs::read(&out).unwrap()).unwrap();
    assert_eq!(parsed["aggregate_version"], "1");
    assert_eq!(parsed["baseline"], "clean");
    let metrics: Vec<String> = parsed["metrics"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    assert!(metrics.contains(&"ap".to_string()));
    assert!(metrics.contains(&"ap__rpc".to_string()));

    let rows = parsed["rows"].as_array().unwrap();
    // 3 slices: clean, fog, noise (plus a possible __unassigned__ if
    // the join missed anything; in this fixture every run joins).
    assert_eq!(rows.len(), 3);

    // baseline row: rpc must be 1.0 (mean / mean)
    let clean = rows
        .iter()
        .find(|r| r["value"] == "clean")
        .expect("clean row missing");
    assert_eq!(clean["metrics"]["ap"], 0.80);
    assert!((clean["metrics"]["ap__rpc"].as_f64().unwrap() - 1.0).abs() < 1e-9);

    let fog = rows
        .iter()
        .find(|r| r["value"] == "fog")
        .expect("fog row missing");
    assert!((fog["metrics"]["ap"].as_f64().unwrap() - 0.40).abs() < 1e-9);
    assert!((fog["metrics"]["ap__rpc"].as_f64().unwrap() - 0.5).abs() < 1e-9);
}

#[test]
fn aggregate_without_baseline_has_no_rpc_columns() {
    let tmp = tempdir();
    let (_a, _b, _c, manifest) = write_three_runs(tmp.path());
    let glob = tmp.path().join("run_*.json").to_string_lossy().into_owned();
    let out = tmp.path().join("summary.json");

    let output = run_aggregate(&manifest, &glob, &out, &["--metric", "ap"]);
    assert_eq!(output.status.code(), Some(0));

    let parsed: serde_json::Value = serde_json::from_slice(&std::fs::read(&out).unwrap()).unwrap();
    assert!(parsed["baseline"].is_null());
    let metrics: Vec<String> = parsed["metrics"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    assert!(metrics.contains(&"ap".to_string()));
    assert!(!metrics.iter().any(|m| m.ends_with("__rpc")));
}

#[test]
fn glob_no_match_is_typed_error() {
    let tmp = tempdir();
    let (_a, _b, _c, manifest) = write_three_runs(tmp.path());
    // Use a pattern that matches nothing.
    let bogus = tmp
        .path()
        .join("nope_*.json")
        .to_string_lossy()
        .into_owned();
    let out = tmp.path().join("summary.json");

    let output = run_aggregate(&manifest, &bogus, &out, &[]);
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("matched zero files"), "stderr: {stderr}");
}

#[test]
fn unjoinable_result_warns_and_is_skipped() {
    let tmp = tempdir();
    let (_a, _b, _c, manifest) = write_three_runs(tmp.path());

    // Add a fourth result whose label is not in the manifest.
    let extra = tmp.path().join("run_orphan.json");
    std::fs::write(&extra, make_result_v1("run_orphan", 0.10)).unwrap();

    let glob = tmp.path().join("run_*.json").to_string_lossy().into_owned();
    let out = tmp.path().join("summary.json");

    let output = run_aggregate(&manifest, &glob, &out, &["--metric", "ap"]);
    assert_eq!(output.status.code(), Some(0));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("has no manifest row"),
        "expected warning, got: {stderr}"
    );

    // The output table must still cover the three known slices, not four.
    let parsed: serde_json::Value = serde_json::from_slice(&std::fs::read(&out).unwrap()).unwrap();
    let rows = parsed["rows"].as_array().unwrap();
    let values: Vec<&str> = rows.iter().map(|r| r["value"].as_str().unwrap()).collect();
    assert!(values.contains(&"clean"));
    assert!(values.contains(&"fog"));
    assert!(values.contains(&"noise"));
}

#[test]
fn missing_required_flags_exits_two() {
    let mut cmd = Command::cargo_bin("vernier").unwrap();
    let output = cmd.args(["aggregate", "--results", "x"]).output().unwrap();
    assert_eq!(output.status.code(), Some(2));
}

// --- tempdir helper ---

struct Tempdir {
    path: PathBuf,
}

impl Tempdir {
    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for Tempdir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn tempdir() -> Tempdir {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let base = std::env::var_os("CARGO_TARGET_TMPDIR")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let path = base.join(format!("vernier-cli-aggregate-test-{pid}-{n}"));
    std::fs::create_dir_all(&path).unwrap();
    Tempdir { path }
}
