//! Integration tests for `vernier eval --manifest` (ADR-0046 Phase 2).
//!
//! Each test synthesizes a tiny COCO fixture (4 images / 2 categories /
//! 4 GTs / 4 DTs) plus a 2-axis JSON or CSV manifest, then runs the
//! binary and asserts on the v2 envelope shape.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};

use assert_cmd::Command;

/// Tiny GT shaped like a COCO `instances_*.json`. Four images, two
/// categories, one perfect-overlap GT per image.
fn gt_json() -> &'static str {
    r#"{
        "images": [
            {"id": 1, "width": 64, "height": 64, "file_name": "1.jpg"},
            {"id": 2, "width": 64, "height": 64, "file_name": "2.jpg"},
            {"id": 3, "width": 64, "height": 64, "file_name": "3.jpg"},
            {"id": 4, "width": 64, "height": 64, "file_name": "4.jpg"}
        ],
        "categories": [
            {"id": 1, "name": "cat"},
            {"id": 2, "name": "dog"}
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10.0, 10.0, 20.0, 20.0], "area": 400.0, "iscrowd": 0},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [10.0, 10.0, 20.0, 20.0], "area": 400.0, "iscrowd": 0},
            {"id": 3, "image_id": 3, "category_id": 2, "bbox": [10.0, 10.0, 20.0, 20.0], "area": 400.0, "iscrowd": 0},
            {"id": 4, "image_id": 4, "category_id": 2, "bbox": [10.0, 10.0, 20.0, 20.0], "area": 400.0, "iscrowd": 0}
        ]
    }"#
}

/// Matching detections — perfect overlap, score=1.0.
fn dt_json() -> &'static str {
    r#"[
        {"image_id": 1, "category_id": 1, "bbox": [10.0, 10.0, 20.0, 20.0], "score": 1.0},
        {"image_id": 2, "category_id": 1, "bbox": [10.0, 10.0, 20.0, 20.0], "score": 1.0},
        {"image_id": 3, "category_id": 2, "bbox": [10.0, 10.0, 20.0, 20.0], "score": 1.0},
        {"image_id": 4, "category_id": 2, "bbox": [10.0, 10.0, 20.0, 20.0], "score": 1.0}
    ]"#
}

/// 2-axis JSON manifest. `weather` splits clean / fog; `time_of_day`
/// splits day / night. Images 1 & 3 are clean/day, 2 & 4 fog/night.
fn manifest_json() -> &'static str {
    r#"{
        "manifest_version": "1",
        "key_kind": "image_id",
        "rows": [
            {"key": 1, "weather": "clean", "time_of_day": "day"},
            {"key": 2, "weather": "fog",   "time_of_day": "night"},
            {"key": 3, "weather": "clean", "time_of_day": "day"},
            {"key": 4, "weather": "fog",   "time_of_day": "night"}
        ]
    }"#
}

/// CSV form of the same manifest. The CLI dispatches on the extension.
fn manifest_csv() -> &'static str {
    "key,weather,time_of_day\n1,clean,day\n2,fog,night\n3,clean,day\n4,fog,night\n"
}

fn write_fixture(tmp: &Path) -> (PathBuf, PathBuf, PathBuf, PathBuf) {
    let gt = tmp.join("gt.json");
    let dt = tmp.join("dt.json");
    let mjson = tmp.join("manifest.json");
    let mcsv = tmp.join("manifest.csv");
    std::fs::write(&gt, gt_json()).unwrap();
    std::fs::write(&dt, dt_json()).unwrap();
    std::fs::write(&mjson, manifest_json()).unwrap();
    std::fs::write(&mcsv, manifest_csv()).unwrap();
    (gt, dt, mjson, mcsv)
}

fn run_eval_partitioned(
    gt: &Path,
    dt: &Path,
    manifest: &Path,
    out: &Path,
    extra: &[&str],
) -> std::process::Output {
    let mut cmd = Command::cargo_bin("vernier").unwrap();
    let mut args: Vec<String> = vec![
        "eval".into(),
        "--gt".into(),
        gt.to_string_lossy().into_owned(),
        "--dt".into(),
        dt.to_string_lossy().into_owned(),
        "--iou-type".into(),
        "bbox".into(),
        "--manifest".into(),
        manifest.to_string_lossy().into_owned(),
        "--emit".into(),
        format!("json={}", out.display()),
    ];
    args.extend(extra.iter().map(|s| (*s).to_string()));
    cmd.args(args).output().unwrap()
}

#[test]
fn json_manifest_produces_v2_with_marginal_slices() {
    let tmp = tempdir();
    let (gt, dt, mjson, _) = write_fixture(tmp.path());
    let out = tmp.path().join("result.json");

    let output = run_eval_partitioned(&gt, &dt, &mjson, &out, &[]);
    assert_eq!(
        output.status.code(),
        Some(0),
        "stderr: {:?}",
        String::from_utf8_lossy(&output.stderr)
    );

    let parsed: serde_json::Value = serde_json::from_slice(&std::fs::read(&out).unwrap()).unwrap();
    assert_eq!(parsed["version"], "2");
    assert_eq!(parsed["iou_type"], "bbox");

    // overall block: 12 stats, n_images=4, n_detections=4
    let overall = &parsed["overall"];
    assert_eq!(overall["stats"].as_array().unwrap().len(), 12);
    assert_eq!(overall["n_images"], 4);
    assert_eq!(overall["n_detections"], 4);

    // Per-axis marginals + __unassigned__ buckets:
    //   time_of_day: day, night, __unassigned__  (3)
    //   weather:     clean, fog, __unassigned__  (3)
    // = 6 marginals (no joint cells without --cross)
    let slices = parsed["slices"].as_array().unwrap();
    assert_eq!(slices.len(), 6, "marginals only without --cross");

    // Spot-check one slice's shape.
    let day = slices
        .iter()
        .find(|s| s["axis"] == "time_of_day" && s["value"] == "day")
        .expect("time_of_day=day slice missing");
    assert_eq!(day["n_images"], 2);
    assert_eq!(day["stats"].as_array().unwrap().len(), 12);
}

#[test]
fn csv_manifest_round_trips_to_same_partition_as_json() {
    let tmp = tempdir();
    let (gt, dt, mjson, mcsv) = write_fixture(tmp.path());
    let out_json = tmp.path().join("from_json.json");
    let out_csv = tmp.path().join("from_csv.json");

    let oj = run_eval_partitioned(&gt, &dt, &mjson, &out_json, &[]);
    assert_eq!(oj.status.code(), Some(0));
    let oc = run_eval_partitioned(&gt, &dt, &mcsv, &out_csv, &[]);
    assert_eq!(oc.status.code(), Some(0));

    // Compare stat columns slice-by-slice. Byte-identity is not the
    // contract here (axis-name comparisons go through HashMaps), but
    // the slice list and stat values must coincide.
    let a: serde_json::Value = serde_json::from_slice(&std::fs::read(&out_json).unwrap()).unwrap();
    let b: serde_json::Value = serde_json::from_slice(&std::fs::read(&out_csv).unwrap()).unwrap();
    assert_eq!(a["overall"]["stats"], b["overall"]["stats"]);

    let slices_a = a["slices"].as_array().unwrap();
    let slices_b = b["slices"].as_array().unwrap();
    assert_eq!(slices_a.len(), slices_b.len());
    for (sa, sb) in slices_a.iter().zip(slices_b.iter()) {
        assert_eq!(sa["axis"], sb["axis"]);
        assert_eq!(sa["value"], sb["value"]);
        assert_eq!(sa["stats"], sb["stats"]);
        assert_eq!(sa["n_images"], sb["n_images"]);
    }
}

#[test]
fn cross_flag_adds_joint_cells() {
    let tmp = tempdir();
    let (gt, dt, mjson, _) = write_fixture(tmp.path());
    let out = tmp.path().join("crossed.json");

    let output = run_eval_partitioned(&gt, &dt, &mjson, &out, &["--cross", "weather,time_of_day"]);
    assert_eq!(
        output.status.code(),
        Some(0),
        "stderr: {:?}",
        String::from_utf8_lossy(&output.stderr)
    );

    let parsed: serde_json::Value = serde_json::from_slice(&std::fs::read(&out).unwrap()).unwrap();
    let slices = parsed["slices"].as_array().unwrap();

    // 6 marginals + 4 joint combos + 1 joint __unassigned__ = 11
    let joints: Vec<&serde_json::Value> = slices
        .iter()
        .filter(|s| s["axis"].as_str().unwrap_or("").contains("::"))
        .collect();
    assert_eq!(
        joints.len(),
        5,
        "4 cells + 1 unassigned bucket on the joint"
    );
    let clean_day = joints
        .iter()
        .find(|s| s["value"] == "clean::day")
        .expect("clean::day joint cell missing");
    assert_eq!(clean_day["n_images"], 2);
}

#[test]
fn label_stamps_into_document() {
    let tmp = tempdir();
    let (gt, dt, mjson, _) = write_fixture(tmp.path());
    let out = tmp.path().join("labelled.json");

    let output = run_eval_partitioned(&gt, &dt, &mjson, &out, &["--label", "run_2026_05_14"]);
    assert_eq!(output.status.code(), Some(0));
    let parsed: serde_json::Value = serde_json::from_slice(&std::fs::read(&out).unwrap()).unwrap();
    assert_eq!(parsed["label"], "run_2026_05_14");
}

#[test]
fn text_output_renders_overall_and_per_slice_block() {
    let tmp = tempdir();
    let (gt, dt, mjson, _) = write_fixture(tmp.path());

    let mut cmd = Command::cargo_bin("vernier").unwrap();
    let output = cmd
        .args([
            "eval",
            "--gt",
            gt.to_string_lossy().as_ref(),
            "--dt",
            dt.to_string_lossy().as_ref(),
            "--iou-type",
            "bbox",
            "--manifest",
            mjson.to_string_lossy().as_ref(),
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(0));
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("overall"),
        "overall header missing: {stdout}"
    );
    assert!(stdout.contains("==>"), "per-slice header missing: {stdout}");
    assert!(stdout.contains("weather"));
    assert!(stdout.contains("time_of_day"));
}

// --- tempdir helper (kept private; copy-paste of eval.rs's pattern) ---

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
    let path = base.join(format!("vernier-cli-manifest-test-{pid}-{n}"));
    std::fs::create_dir_all(&path).unwrap();
    Tempdir { path }
}
