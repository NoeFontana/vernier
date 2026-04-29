//! Integration tests for the `vernier eval` binary.
//!
//! Per ADR-0015 §"Parity harness", numerical parity is asserted from
//! the Python side; this file covers the argument layer at the
//! process boundary: missing/duplicate/conflicting flags, exit-code
//! mapping, file-emit semantics, and `--quiet` stderr suppression.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

use std::path::{Path, PathBuf};

use assert_cmd::Command;

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

fn bbox_args() -> [String; 6] {
    let dir = fixtures_dir().join("bbox");
    [
        "--gt".into(),
        dir.join("gt.json").to_string_lossy().into_owned(),
        "--dt".into(),
        dir.join("dt.json").to_string_lossy().into_owned(),
        "--iou-type".into(),
        "bbox".into(),
    ]
}

fn keypoints_args() -> [String; 6] {
    let dir = fixtures_dir().join("keypoints");
    [
        "--gt".into(),
        dir.join("gt.json").to_string_lossy().into_owned(),
        "--dt".into(),
        dir.join("dt.json").to_string_lossy().into_owned(),
        "--iou-type".into(),
        "keypoints".into(),
    ]
}

#[test]
fn missing_required_flags_exits_two() {
    let output = Command::cargo_bin("vernier")
        .unwrap()
        .args(["eval", "--iou-type", "bbox"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
}

#[test]
fn unknown_iou_type_exits_two() {
    let output = Command::cargo_bin("vernier")
        .unwrap()
        .args([
            "eval",
            "--gt",
            "/dev/null",
            "--dt",
            "/dev/null",
            "--iou-type",
            "detection",
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
}

#[test]
fn dilation_ratio_rejected_for_non_boundary() {
    let mut cmd = Command::cargo_bin("vernier").unwrap();
    let mut args: Vec<String> = vec!["eval".into()];
    args.extend(bbox_args());
    args.push("--dilation-ratio".into());
    args.push("0.02".into());
    let output = cmd.args(args).output().unwrap();
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("dilation-ratio"),
        "stderr did not name the offending flag: {stderr}"
    );
}

#[test]
fn sigmas_rejected_for_non_keypoints() {
    let mut cmd = Command::cargo_bin("vernier").unwrap();
    let mut args: Vec<String> = vec!["eval".into()];
    args.extend(bbox_args());
    args.push("--sigmas".into());
    args.push("/tmp/nonexistent-sigmas.json".into());
    let output = cmd.args(args).output().unwrap();
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("sigmas"),
        "stderr did not name the offending flag: {stderr}"
    );
}

#[test]
fn double_stdout_emit_rejected() {
    let mut cmd = Command::cargo_bin("vernier").unwrap();
    let mut args: Vec<String> = vec!["eval".into()];
    args.extend(bbox_args());
    args.push("--emit".into());
    args.push("text".into());
    args.push("--emit".into());
    args.push("json".into());
    let output = cmd.args(args).output().unwrap();
    assert_eq!(output.status.code(), Some(2));
}

#[test]
fn duplicate_path_emit_rejected() {
    let tmp = tempdir();
    let out = tmp.path().join("out.txt");
    let mut cmd = Command::cargo_bin("vernier").unwrap();
    let mut args: Vec<String> = vec!["eval".into()];
    args.extend(bbox_args());
    args.push("--emit".into());
    args.push(format!("text={}", out.display()));
    args.push("--emit".into());
    args.push(format!("json={}", out.display()));
    let output = cmd.args(args).output().unwrap();
    assert_eq!(output.status.code(), Some(2));
}

#[test]
fn emit_to_file_writes_to_path() {
    let tmp = tempdir();
    let out = tmp.path().join("result.json");
    let mut cmd = Command::cargo_bin("vernier").unwrap();
    let mut args: Vec<String> = vec!["eval".into()];
    args.extend(bbox_args());
    args.push("--emit".into());
    args.push(format!("json={}", out.display()));
    let output = cmd.args(args).output().unwrap();
    assert_eq!(output.status.code(), Some(0), "{:?}", output);
    assert!(out.exists(), "emit destination was not written");
    let bytes = std::fs::read(&out).unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(parsed["version"], "1");
    assert_eq!(parsed["iou_type"], "bbox");
    // No data on stdout in file-only mode.
    assert!(output.stdout.is_empty(), "stdout: {:?}", output.stdout);
}

#[test]
fn quiet_suppresses_stderr_on_error() {
    // Trigger a typed CliError by pointing --gt at a missing file.
    let mut cmd = Command::cargo_bin("vernier").unwrap();
    let output = cmd
        .args([
            "eval",
            "--gt",
            "/this/path/does/not/exist.json",
            "--dt",
            "/this/path/does/not/exist.json",
            "--iou-type",
            "bbox",
            "--quiet",
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(
        output.stderr.is_empty(),
        "--quiet did not suppress stderr: {:?}",
        output.stderr
    );
}

#[test]
fn happy_path_text_output_contains_pycocotools_lines() {
    let mut cmd = Command::cargo_bin("vernier").unwrap();
    let mut args: Vec<String> = vec!["eval".into()];
    args.extend(bbox_args());
    let output = cmd.args(args).output().unwrap();
    assert_eq!(output.status.code(), Some(0), "{:?}", output);
    let stdout = String::from_utf8(output.stdout).unwrap();
    // 12 detection lines, terminated with a newline.
    let lines: Vec<&str> = stdout.split_inclusive('\n').collect();
    assert_eq!(lines.len(), 12, "stdout was: {stdout}");
    assert!(stdout.contains("Average Precision"));
    assert!(stdout.contains("Average Recall"));
    assert!(stdout.contains("0.50:0.95"));
    assert!(stdout.contains("maxDets=100"));
    assert!(stdout.ends_with('\n'));
}

#[test]
fn happy_path_json_output_is_parseable_v1_doc() {
    let mut cmd = Command::cargo_bin("vernier").unwrap();
    let mut args: Vec<String> = vec!["eval".into()];
    args.extend(bbox_args());
    args.push("--emit".into());
    args.push("json".into());
    let output = cmd.args(args).output().unwrap();
    assert_eq!(output.status.code(), Some(0), "{:?}", output);
    let parsed: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(parsed["version"], "1");
    assert_eq!(parsed["iou_type"], "bbox");
    assert_eq!(parsed["parity_mode"], "strict");
    assert_eq!(parsed["use_cats"], true);
    let stats = parsed["stats"].as_array().unwrap();
    assert_eq!(stats.len(), 12);
    let lines = parsed["lines"].as_array().unwrap();
    assert_eq!(lines.len(), 12);
    // The bytes must end with a single newline (ADR-0015 §"Output
    // determinism").
    assert_eq!(*output.stdout.last().unwrap(), b'\n');
}

#[test]
fn keypoints_default_max_dets_resolves_to_twenty() {
    // ADR-0012 regression guard: kp without --max-dets must resolve to
    // [20] via the kernel-canonical default path, not via a CLI-side
    // hardcode.
    let mut cmd = Command::cargo_bin("vernier").unwrap();
    let mut args: Vec<String> = vec!["eval".into()];
    args.extend(keypoints_args());
    args.push("--emit".into());
    args.push("json".into());
    let output = cmd.args(args).output().unwrap();
    assert_eq!(output.status.code(), Some(0), "{:?}", output);
    let parsed: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let max_dets: Vec<u64> = parsed["max_dets"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect();
    assert_eq!(max_dets, vec![20]);
    // Also pin the kp summary plan length (10 lines per ADR-0012 / D5).
    assert_eq!(parsed["lines"].as_array().unwrap().len(), 10);
}

#[test]
fn detection_default_max_dets_resolves_to_one_ten_hundred() {
    let mut cmd = Command::cargo_bin("vernier").unwrap();
    let mut args: Vec<String> = vec!["eval".into()];
    args.extend(bbox_args());
    args.push("--emit".into());
    args.push("json".into());
    let output = cmd.args(args).output().unwrap();
    assert_eq!(output.status.code(), Some(0), "{:?}", output);
    let parsed: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let max_dets: Vec<u64> = parsed["max_dets"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect();
    assert_eq!(max_dets, vec![1, 10, 100]);
}

/// Tiny tempdir helper. Avoids pulling `tempfile` into the dep tree
/// (ADR-0015 caps top-level deps); this constructs a per-test
/// directory under `CARGO_TARGET_TMPDIR` (provided by Cargo at test
/// runtime) and removes it on Drop.
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
    let path = base.join(format!("vernier-cli-test-{pid}-{n}"));
    std::fs::create_dir_all(&path).unwrap();
    Tempdir { path }
}
