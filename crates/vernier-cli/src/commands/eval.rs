//! `vernier eval` — the only verb at v0.2.
//!
//! This module is the orchestration layer described in ADR-0015
//! §"Crate layout": no business logic lives here, only argument-to-
//! kernel-config translation, a single eval call, and a flat dispatch
//! loop over the parsed `--emit` list.
//!
//! Per ADR-0015 §"Formatter abstraction", eval runs **exactly once**
//! and the resulting `Summary` is borrowed by each formatter. Adding a
//! second `--emit` does not re-build the dataset, re-run matching, or
//! re-call summarize.

use std::collections::HashMap;
use std::fs;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process;

use vernier_core::accumulate::{accumulate, sort_max_dets, AccumulateParams};
use vernier_core::boundary_parity::BOUNDARY_DILATION_RATIO_DEFAULT;
use vernier_core::parity::{iou_thresholds, recall_thresholds};
use vernier_core::summarize::{summarize_detection, summarize_with, StatRequest};
use vernier_core::{
    evaluate_bbox, evaluate_boundary, evaluate_keypoints, evaluate_segm, AreaRange, CocoDataset,
    CocoDetections, EvalError, EvaluateParams, ParityMode, Summary,
};

use crate::cli::{EmitDestination, EmitSpec, EvalArgs, IouTypeArg};
use crate::error::CliError;
use crate::format::{registry, FormatContext, Formatter};

/// Detection-canonical max_dets ladder used when `--max-dets` is
/// absent and `--iou-type` is anything other than `keypoints`. Mirrors
/// pycocotools' default and the in-process FFI default.
const DETECTION_MAX_DETS_DEFAULT: [usize; 3] = [1, 10, 100];
/// Keypoints-canonical max_dets default per ADR-0012. Single-rung
/// ladder; all 10 kp summary lines resolve to it.
const KEYPOINTS_MAX_DETS_DEFAULT: [usize; 1] = [20];

/// End-to-end runner. Reads the GT/DT JSON, resolves the kernel
/// config, runs the pipeline, and writes every `--emit` destination.
///
/// Stderr is reserved for diagnostics; stdout carries the summary
/// bytes (or nothing, if every `--emit` writes to a file). Per
/// ADR-0015 §"Stdout / stderr split", the binary does not deviate
/// from this discipline.
pub(crate) fn run(args: &EvalArgs) -> Result<(), CliError> {
    let emits = args.validate()?;

    let parity_mode: ParityMode = args.parity_mode.into();
    let use_cats = args.effective_use_cats();

    // Per ADR-0012, the kernel-canonical default lives in this
    // module's two `*_DEFAULT` constants — not in `cli.rs` — so the
    // argument layer never has to know which kind picks `[20]` vs
    // `[1, 10, 100]`.
    let parsed_max_dets = args.parsed_max_dets()?;
    let mut max_dets: Vec<usize> = match (parsed_max_dets, args.iou_type) {
        (Some(d), _) => d,
        (None, IouTypeArg::Keypoints) => KEYPOINTS_MAX_DETS_DEFAULT.to_vec(),
        (None, _) => DETECTION_MAX_DETS_DEFAULT.to_vec(),
    };
    // Quirk A2 (aligned): pycocotools sorts max_dets ascending, and
    // the accumulator's M-axis depends on that order. Mirror it here
    // before any kernel call.
    sort_max_dets(&mut max_dets);

    let gt_bytes = read_input(&args.gt)?;
    let dt_bytes = read_input(&args.dt)?;
    let gt = CocoDataset::from_json_bytes(&gt_bytes)?;
    let dt = CocoDetections::from_json_bytes(&dt_bytes)?;

    let sigmas = match (&args.sigmas, args.iou_type) {
        (Some(path), IouTypeArg::Keypoints) => Some(load_sigmas(path)?),
        (Some(_), _) => {
            // Already rejected by EvalArgs::validate, but the explicit
            // match arm avoids a fall-through that would silently
            // ignore the file under the wrong kind.
            return Err(CliError::Validation(
                "--sigmas is only valid with --iou-type keypoints".into(),
            ));
        }
        (None, _) => None,
    };

    let dilation_ratio = match (args.dilation_ratio, args.iou_type) {
        (Some(d), IouTypeArg::Boundary) => d,
        (None, IouTypeArg::Boundary) => BOUNDARY_DILATION_RATIO_DEFAULT,
        // Already rejected for non-boundary kinds in `validate`.
        _ => 0.0,
    };

    let summary = run_pipeline(
        args.iou_type,
        &gt,
        &dt,
        parity_mode,
        &max_dets,
        use_cats,
        dilation_ratio,
        sigmas,
    )?;

    // ADR-0015 §"Formatter abstraction": eval runs exactly once
    // (above); the dispatch below is a flat for-loop, no re-eval.
    let ctx = FormatContext {
        iou_type: args.iou_type,
        parity_mode,
        max_dets: &max_dets,
        use_cats,
    };
    dispatch_emits(&emits, &summary, &ctx)
}

/// Convenience entry point for `main.rs`: maps a [`CliError`] to the
/// process exit code per ADR-0015 §"Exit codes" and prints the typed
/// message on stderr (unless `--quiet` is set).
pub(crate) fn run_or_exit(args: &EvalArgs) -> ! {
    let quiet = args.quiet;
    match run(args) {
        Ok(()) => process::exit(0),
        Err(err) => {
            if !quiet {
                let mut stderr = io::stderr().lock();
                // A failed write here is itself unrecoverable; ignore
                // the result so the exit-code path stays clean.
                let _ = writeln!(stderr, "error: {err}");
            }
            process::exit(err.exit_code());
        }
    }
}

// Eight args is one over the lint's threshold; the alternative is a
// throwaway config struct that exists only to thread these through, so
// allow the lint here rather than invent a type with no other use.
#[allow(clippy::too_many_arguments)]
fn run_pipeline(
    iou_type: IouTypeArg,
    gt: &CocoDataset,
    dt: &CocoDetections,
    parity: ParityMode,
    max_dets: &[usize],
    use_cats: bool,
    dilation_ratio: f64,
    sigmas: Option<HashMap<i64, Vec<f64>>>,
) -> Result<Summary, EvalError> {
    let iou_thr = iou_thresholds();
    let area: Vec<AreaRange> = match iou_type {
        IouTypeArg::Keypoints => AreaRange::keypoints_default().to_vec(),
        _ => AreaRange::coco_default().to_vec(),
    };
    let max_det_top = max_dets.iter().copied().max().unwrap_or(100);
    let eval_params = EvaluateParams {
        iou_thresholds: iou_thr,
        area_ranges: &area,
        max_dets_per_image: max_det_top,
        use_cats,
        retain_iou: false,
    };

    let grid = match iou_type {
        IouTypeArg::Bbox => evaluate_bbox(gt, dt, eval_params, parity)?,
        IouTypeArg::Segm => evaluate_segm(gt, dt, eval_params, parity)?,
        IouTypeArg::Boundary => evaluate_boundary(gt, dt, eval_params, parity, dilation_ratio)?,
        IouTypeArg::Keypoints => {
            evaluate_keypoints(gt, dt, eval_params, parity, sigmas.unwrap_or_default())?
        }
    };

    let acc_params = AccumulateParams {
        iou_thresholds: iou_thr,
        recall_thresholds: recall_thresholds(),
        max_dets,
        n_categories: grid.n_categories,
        n_area_ranges: grid.n_area_ranges,
        n_images: grid.n_images,
    };
    let acc = accumulate(&grid.eval_imgs, acc_params, parity)?;
    if matches!(iou_type, IouTypeArg::Keypoints) {
        // ADR-0012 / D5: kp summary uses the 10-stat plan over the
        // 3-bucket area grid.
        summarize_with(
            &acc,
            &StatRequest::coco_keypoints_default(),
            iou_thr,
            max_dets,
        )
    } else {
        summarize_detection(&acc, iou_thr, max_dets)
    }
}

fn dispatch_emits(
    emits: &[EmitSpec],
    summary: &Summary,
    ctx: &FormatContext<'_>,
) -> Result<(), CliError> {
    for spec in emits {
        let formatter = lookup_formatter(spec.format).ok_or_else(|| {
            // Unreachable in practice (validate already proved the name
            // is in the registry), but we surface a typed error rather
            // than panicking.
            CliError::Validation(format!(
                "internal: format {:?} disappeared from registry",
                spec.format
            ))
        })?;
        match &spec.destination {
            EmitDestination::Stdout => {
                let stdout = io::stdout();
                let mut handle = stdout.lock();
                formatter.render(summary, ctx, &mut handle)?;
            }
            EmitDestination::File(path) => {
                write_atomic(path, |w| formatter.render(summary, ctx, w))?;
            }
        }
    }
    Ok(())
}

fn lookup_formatter(name: crate::format::FormatName) -> Option<&'static dyn Formatter> {
    registry().iter().copied().find(|f| f.id() == name)
}

fn read_input(path: &Path) -> Result<Vec<u8>, CliError> {
    fs::read(path).map_err(|source| CliError::InputRead {
        path: path.to_path_buf(),
        source,
    })
}

/// Atomic write per ADR-0015 §"Output determinism": render into
/// `<PATH>.tmp.<pid>`, fsync, rename. Concurrent readers see either
/// the pre-existing contents or the new contents in full, never a
/// half-written file.
///
/// Parent-directory creation is *not* automatic: a non-existent parent
/// surfaces as a typed [`CliError::OutputWrite`] with exit code 1.
/// `--mkdir` is explicitly out of scope for v0.2 (per ADR-0015 §"What
/// this ADR explicitly does *not* decide").
fn write_atomic<F>(final_path: &Path, render: F) -> Result<(), CliError>
where
    F: FnOnce(&mut dyn io::Write) -> Result<(), CliError>,
{
    let parent = final_path.parent().filter(|p| !p.as_os_str().is_empty());
    if let Some(parent) = parent {
        if !parent.exists() {
            return Err(CliError::OutputWrite {
                path: final_path.to_path_buf(),
                source: io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("parent directory {} does not exist", parent.display()),
                ),
            });
        }
    }
    let tmp_path = sibling_tmp_path(final_path);
    let file = fs::File::create(&tmp_path).map_err(|source| CliError::OutputWrite {
        path: tmp_path.clone(),
        source,
    })?;
    let mut writer = BufWriter::new(file);
    let render_result = render(&mut writer);
    let render_err = render_result.err();
    let flush_err = writer.flush().err();
    let inner = writer.into_inner().ok();
    let sync_err = inner.as_ref().and_then(|f| f.sync_all().err());

    if let Some(err) = render_err {
        let _ = fs::remove_file(&tmp_path);
        return Err(err);
    }
    if let Some(err) = flush_err {
        let _ = fs::remove_file(&tmp_path);
        return Err(CliError::OutputWrite {
            path: tmp_path,
            source: err,
        });
    }
    if let Some(err) = sync_err {
        let _ = fs::remove_file(&tmp_path);
        return Err(CliError::OutputWrite {
            path: tmp_path,
            source: err,
        });
    }

    fs::rename(&tmp_path, final_path).map_err(|source| {
        // Best-effort cleanup; ignore the result of remove_file
        // because the rename failure is the load-bearing error.
        let _ = fs::remove_file(&tmp_path);
        CliError::OutputWrite {
            path: final_path.to_path_buf(),
            source,
        }
    })?;
    Ok(())
}

fn sibling_tmp_path(final_path: &Path) -> PathBuf {
    let pid = process::id();
    let mut name = final_path
        .file_name()
        .map(|s| s.to_os_string())
        .unwrap_or_default();
    name.push(format!(".tmp.{pid}"));
    match final_path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent.join(name),
        _ => PathBuf::from(name),
    }
}

fn load_sigmas(path: &Path) -> Result<HashMap<i64, Vec<f64>>, CliError> {
    let bytes = read_input(path)?;
    // The sigmas file is a `{category_id: [sigma, ...]}` JSON object.
    // Keep parsing tolerant of both `{"1": [...]}` (string keys, the
    // JSON-standard shape) and `{1: [...]}` (raw integer keys, which
    // serde_json rejects on parse — same as pycocotools).
    let parsed: HashMap<String, Vec<f64>> = serde_json::from_slice(&bytes)
        .map_err(|e| CliError::InvalidSigmas(format!("could not parse {}: {e}", path.display())))?;
    let mut out: HashMap<i64, Vec<f64>> = HashMap::with_capacity(parsed.len());
    for (k, v) in parsed {
        let key: i64 = k.parse().map_err(|_| {
            CliError::InvalidSigmas(format!(
                "sigmas key {k:?} is not a valid integer category_id"
            ))
        })?;
        out.insert(key, v);
    }
    Ok(out)
}
