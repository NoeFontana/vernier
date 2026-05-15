//! `vernier eval --manifest` — partitioned dispatch path (ADR-0046).
//!
//! Branched out of [`super::eval::run`] so the un-partitioned lane
//! stays byte-for-byte identical to the v0.2 release. The two lanes
//! share `EvaluateParams` construction and the same kernel entry
//! points; only matching-output consumption differs (this lane runs
//! `partition::evaluate_partitioned` instead of `accumulate +
//! summarize_detection`).
//!
//! This module deliberately does not link `vernier-ffi`. Per ADR-0015
//! the CLI is a pure-Rust binary; it calls vernier-core entry points
//! directly and skips PyO3 entirely.

use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::path::Path;

use vernier_core::accumulate::sort_max_dets;
use vernier_core::boundary_parity::BOUNDARY_DILATION_RATIO_DEFAULT;
use vernier_core::dataset::ImageId;
use vernier_core::lrp::{self, LrpKernelMarker, LrpParams};
use vernier_core::manifest;
use vernier_core::manifest_csv;
use vernier_core::parity::iou_thresholds;
use vernier_core::partition::{
    self, evaluate_partitioned_lrp, image_id_to_idx as build_image_id_to_idx, GridDims, KeyKind,
    PartitionSpec, PartitionedLrpReport, SummaryPlan,
};
use vernier_core::similarity::{BboxIou, BoundaryIou, OksSimilarity, SegmIou};
use vernier_core::{
    evaluate_bbox, evaluate_boundary, evaluate_keypoints, evaluate_segm, AreaRange, CocoDataset,
    CocoDetections, EvalError, EvalGrid, EvaluateParams, ParityMode,
};

use crate::cli::{EmitDestination, EvalArgs, IouTypeArg, MetricArg};
use crate::error::CliError;
use crate::format::{registry, EvalArtifact, FormatContext, FormatName, Formatter};

// Re-exposed so eval.rs / aggregate.rs can reuse the same atomic-write
// machinery without duplicating it. Kept in eval.rs as the canonical
// implementation per ADR-0015.
use super::eval::{write_atomic, DETECTION_MAX_DETS_DEFAULT, KEYPOINTS_MAX_DETS_DEFAULT};

// Re-import the warning-printing path so the partitioned lane mirrors
// the existing CLI stderr discipline.
use vernier_core::manifest::ManifestWarning as CoreWarning;

/// Run the partitioned-eval lane. Caller has already validated
/// `args.manifest.is_some()` and produced the parsed emit list.
pub(crate) fn run(args: &EvalArgs) -> Result<(), CliError> {
    let emits = args.validate()?;
    let parity_mode: ParityMode = args.parity_mode.into();
    let use_cats = args.effective_use_cats();

    if matches!(args.metric, MetricArg::Olrp) {
        return run_lrp(args, &emits, parity_mode, use_cats);
    }

    let inputs = load_partitioned_inputs(args)?;

    let grid = run_kernel(
        args.iou_type,
        &inputs.gt,
        &inputs.dt,
        parity_mode,
        &inputs.max_dets,
        use_cats,
        inputs.dilation_ratio,
        inputs.sigmas,
    )?;

    let summary_kind = match args.iou_type {
        IouTypeArg::Keypoints => SummaryPlan::KeypointsDefault,
        _ => SummaryPlan::DetectionDefault,
    };

    let dims = GridDims {
        n_categories: grid.n_categories,
        n_area_ranges: grid.n_area_ranges,
        n_images: grid.n_images,
    };
    let partitioned = partition::evaluate_partitioned(
        &grid.eval_imgs,
        dims,
        &inputs.spec,
        iou_thresholds(),
        parity_mode,
        summary_kind,
    )?;

    let ctx = FormatContext {
        iou_type: args.iou_type,
        parity_mode,
        max_dets: &inputs.max_dets,
        use_cats,
    };
    let artifact = EvalArtifact::Partitioned {
        summary: &partitioned,
        label: args.label.as_deref(),
    };
    emit_artifact(&emits, &ctx, &artifact)
}

/// Shared preamble across `run` and `run_lrp`: read GT/DT, resolve
/// kernel-specific defaults (max_dets, sigmas, dilation), parse the
/// manifest, build the `PartitionSpec`, surface warnings.
struct PartitionedInputs {
    gt: CocoDataset,
    dt: CocoDetections,
    max_dets: Vec<usize>,
    sigmas: Option<HashMap<i64, Vec<f64>>>,
    dilation_ratio: f64,
    spec: PartitionSpec,
}

fn load_partitioned_inputs(args: &EvalArgs) -> Result<PartitionedInputs, CliError> {
    let parsed_max_dets = args.parsed_max_dets()?;
    let mut max_dets: Vec<usize> = match (parsed_max_dets, args.iou_type) {
        (Some(d), _) => d,
        (None, IouTypeArg::Keypoints) => KEYPOINTS_MAX_DETS_DEFAULT.to_vec(),
        (None, _) => DETECTION_MAX_DETS_DEFAULT.to_vec(),
    };
    sort_max_dets(&mut max_dets);

    let gt_bytes = fs::read(&args.gt).map_err(|source| CliError::InputRead {
        path: args.gt.clone(),
        source,
    })?;
    let dt_bytes = fs::read(&args.dt).map_err(|source| CliError::InputRead {
        path: args.dt.clone(),
        source,
    })?;
    let gt = CocoDataset::from_json_bytes(&gt_bytes)?;
    let dt = CocoDetections::from_json_bytes(&dt_bytes)?;

    let sigmas = match (&args.sigmas, args.iou_type) {
        (Some(path), IouTypeArg::Keypoints) => Some(load_sigmas(path)?),
        (Some(_), _) => {
            return Err(CliError::Validation(
                "--sigmas is only valid with --iou-type keypoints".into(),
            ));
        }
        (None, _) => None,
    };

    let dilation_ratio = match (args.dilation_ratio, args.iou_type) {
        (Some(d), IouTypeArg::Boundary) => d,
        (None, IouTypeArg::Boundary) => BOUNDARY_DILATION_RATIO_DEFAULT,
        _ => 0.0,
    };

    let image_id_to_idx = build_image_id_to_idx(&gt);

    let manifest_path = args.manifest.as_ref().ok_or_else(|| {
        // Caller (eval::run) is supposed to have routed only manifest
        // dispatch here; defensive guard prevents a silent panic if
        // that ever drifts.
        CliError::Validation("internal: partitioned dispatch invoked without --manifest".into())
    })?;
    let manifest_bytes = fs::read(manifest_path).map_err(|source| CliError::InputRead {
        path: manifest_path.clone(),
        source,
    })?;
    let cross_axes = args.parsed_cross_axes()?;
    let (spec, warnings) = build_spec(
        manifest_path,
        &manifest_bytes,
        &image_id_to_idx,
        &cross_axes,
    )?;

    if !args.quiet {
        report_warnings(&warnings);
    }

    Ok(PartitionedInputs {
        gt,
        dt,
        max_dets,
        sigmas,
        dilation_ratio,
        spec,
    })
}

/// Walk the parsed `--emit` list, render the artifact to each
/// destination. Shared by the AP and LRP partition arms.
fn emit_artifact(
    emits: &[crate::cli::EmitSpec],
    ctx: &FormatContext<'_>,
    artifact: &EvalArtifact<'_>,
) -> Result<(), CliError> {
    for spec in emits {
        let formatter = lookup_formatter(spec.format).ok_or_else(|| {
            CliError::Validation(format!(
                "internal: format {:?} disappeared from registry",
                spec.format
            ))
        })?;
        match &spec.destination {
            EmitDestination::Stdout => {
                let stdout = io::stdout();
                let mut handle = stdout.lock();
                formatter.render(artifact, ctx, &mut handle)?;
            }
            EmitDestination::File(path) => {
                write_atomic(path, |w| formatter.render(artifact, ctx, w))?;
            }
        }
    }
    Ok(())
}

// Allow the same lint exception eval.rs takes; threading 8 args is the
// price of keeping the per-kernel branch in one place.
#[allow(clippy::too_many_arguments)]
fn run_kernel(
    iou_type: IouTypeArg,
    gt: &CocoDataset,
    dt: &CocoDetections,
    parity: ParityMode,
    max_dets: &[usize],
    use_cats: bool,
    dilation_ratio: f64,
    sigmas: Option<HashMap<i64, Vec<f64>>>,
) -> Result<EvalGrid, EvalError> {
    let iou_thr = iou_thresholds();
    let area: Vec<AreaRange> = iou_type.default_area_ranges();
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
    Ok(grid)
}

fn build_spec(
    manifest_path: &Path,
    bytes: &[u8],
    image_id_to_idx: &HashMap<ImageId, usize>,
    cross_axes: &[Vec<String>],
) -> Result<(PartitionSpec, Vec<CoreWarning>), CliError> {
    let ext = manifest_path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase);
    let known_ids: std::collections::HashSet<ImageId> = image_id_to_idx.keys().copied().collect();
    let parsed = match ext.as_deref() {
        Some("json") | None => manifest::parse_manifest(bytes, &known_ids, &Default::default())?,
        Some("csv") => manifest_csv::parse_csv_manifest(
            bytes,
            KeyKind::Image,
            &known_ids,
            &Default::default(),
        )?,
        Some(other) => {
            return Err(CliError::Validation(format!(
                "manifest extension {other:?} is not recognized; use .json or .csv"
            )));
        }
    };
    if !matches!(parsed.key_kind, KeyKind::Image) {
        return Err(CliError::Validation(
            "vernier eval --manifest consumes key_kind=\"image_id\" manifests; \
             a key_kind=\"result\" manifest must be routed through `vernier aggregate`"
                .into(),
        ));
    }

    let spec = PartitionSpec::build(
        parsed.key_kind,
        &parsed.per_axis_image,
        &known_ids,
        image_id_to_idx,
        cross_axes,
    )?;

    Ok((spec, parsed.warnings))
}

fn report_warnings(warnings: &[CoreWarning]) {
    if warnings.is_empty() {
        return;
    }
    let stderr = io::stderr();
    let mut handle = stderr.lock();
    for w in warnings {
        match w {
            CoreWarning::UnknownKey { key } => {
                // Best-effort write: stderr failure here is itself
                // unrecoverable; ignore so the eval continues to its
                // exit-code path.
                let _ = writeln!(
                    handle,
                    "warning: manifest references unknown key {key:?}; row skipped"
                );
            }
        }
    }
}

fn lookup_formatter(name: FormatName) -> Option<&'static dyn Formatter> {
    registry().iter().copied().find(|f| f.id() == name)
}

/// Partitioned LRP dispatch (ADR-0046 + ADR-0043). Mirrors the AP
/// arm's shape but routes through `evaluate_partitioned_lrp` — the LRP
/// kernel runs its own internal matching pass, so we never build an
/// `EvalGrid` here.
fn run_lrp(
    args: &EvalArgs,
    emits: &[crate::cli::EmitSpec],
    parity_mode: ParityMode,
    use_cats: bool,
) -> Result<(), CliError> {
    let inputs = load_partitioned_inputs(args)?;
    // LRP runs at a single `max_dets_per_image` rung — the top of the
    // ladder (mirrors the un-partitioned LRP path in `eval.rs`).
    let max_dets_per_image = inputs.max_dets.iter().copied().max().unwrap_or(100);

    // Resolve LRP params per ADR-0044 (single threshold, canonical
    // tau grid). Same shape as `run_lrp_pipeline` in `eval.rs` — the
    // partitioned and un-partitioned paths share the same kernel
    // defaults.
    let tp_threshold = lrp::tp_threshold_for(args.iou_type.kernel_kind());
    let iou_thr = [tp_threshold];
    let tau_grid = lrp::default_tau_grid();
    let area: Vec<AreaRange> = args.iou_type.default_area_ranges();
    let params = LrpParams {
        tp_threshold,
        tau_grid,
        max_dets_per_image,
        use_cats,
        iou_thresholds: &iou_thr,
        area_ranges: &area,
    };

    let partitioned = run_lrp_kernel(
        args.iou_type,
        &inputs.gt,
        &inputs.dt,
        params,
        parity_mode,
        &inputs.spec,
        inputs.dilation_ratio,
        inputs.sigmas,
    )?;

    let ctx = FormatContext {
        iou_type: args.iou_type,
        parity_mode,
        max_dets: &inputs.max_dets,
        use_cats,
    };
    let artifact = EvalArtifact::PartitionedLrp {
        summary: &partitioned,
        label: args.label.as_deref(),
    };
    emit_artifact(emits, &ctx, &artifact)
}

#[allow(clippy::too_many_arguments)]
fn run_lrp_kernel(
    iou_type: IouTypeArg,
    gt: &CocoDataset,
    dt: &CocoDetections,
    params: LrpParams<'_>,
    parity: ParityMode,
    spec: &PartitionSpec,
    dilation_ratio: f64,
    sigmas: Option<HashMap<i64, Vec<f64>>>,
) -> Result<PartitionedLrpReport, CliError> {
    let report = match iou_type {
        IouTypeArg::Bbox => evaluate_partitioned_lrp(
            gt,
            dt,
            &BboxIou,
            LrpKernelMarker::Bbox,
            params,
            parity,
            spec,
        ),
        IouTypeArg::Segm => evaluate_partitioned_lrp(
            gt,
            dt,
            &SegmIou,
            LrpKernelMarker::Segm,
            params,
            parity,
            spec,
        ),
        IouTypeArg::Boundary => {
            let kernel = BoundaryIou { dilation_ratio };
            evaluate_partitioned_lrp(
                gt,
                dt,
                &kernel,
                LrpKernelMarker::Boundary,
                params,
                parity,
                spec,
            )
        }
        IouTypeArg::Keypoints => {
            let kernel = OksSimilarity::new(sigmas.unwrap_or_default());
            evaluate_partitioned_lrp(
                gt,
                dt,
                &kernel,
                LrpKernelMarker::Keypoints,
                params,
                parity,
                spec,
            )
        }
    }
    .map_err(CliError::from)?;
    Ok(report)
}

fn load_sigmas(path: &Path) -> Result<HashMap<i64, Vec<f64>>, CliError> {
    let bytes = fs::read(path).map_err(|source| CliError::InputRead {
        path: path.to_path_buf(),
        source,
    })?;
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
