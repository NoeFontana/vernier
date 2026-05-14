//! Clap-derive argument layer plus cross-flag validation.
//!
//! Per ADR-0015 §"Surface" and §"Crate layout", the CLI is structured
//! as `vernier <verb>` with `eval` the only verb at v0.2. The argument
//! struct lives here so it can be unit-tested at the parse boundary
//! (clap's own `try_parse_from`) without spinning up a subprocess.
//!
//! Validation that requires looking at multiple flags at once (e.g.
//! `--dilation-ratio` is only valid with `--iou-type boundary`) lives
//! on [`EvalArgs::validate`]. `--max-dets` defaults are deliberately
//! *not* materialized here — per ADR-0012 the kernel-canonical default
//! is resolved at the eval call site so kp picks `[20]` and
//! det/segm/boundary pick `[1, 10, 100]` without the CLI carrying
//! per-kind defaults of its own.

use std::path::PathBuf;

use clap::{ArgAction, Parser, Subcommand, ValueEnum};
use vernier_core::ParityMode;

use crate::error::CliError;
use crate::format::FormatName;

/// Top-level `vernier` binary entry point. The verb structure leaves
/// room for future subcommands (`vernier check`, `vernier diff`, …)
/// without breaking flag layouts.
#[derive(Debug, Parser)]
#[command(
    name = "vernier",
    version,
    about = "COCO-style evaluation CLI (vernier)",
    propagate_version = true
)]
pub(crate) struct Cli {
    /// Subcommand verb. `eval` is the only verb at v0.2.
    #[command(subcommand)]
    pub(crate) command: Command,
}

/// Verb-level dispatch. Per ADR-0015 §"CLI structure", `eval` is the
/// v0.2 verb; ADR-0046 adds `aggregate` for cross-run fan-in.
#[derive(Debug, Subcommand)]
pub(crate) enum Command {
    /// Run a COCO-style eval and emit a summary.
    Eval(EvalArgs),
    /// Fan-in over many `vernier eval` result documents (ADR-0046).
    /// Joins each result to a `key_kind=result` manifest by `--label`
    /// (falling back to the result file path) and emits a comparative
    /// per-slice table. When `--baseline` names a slice value, the
    /// table gains rPC columns (`<metric>__rpc = mean_other /
    /// mean_baseline`).
    Aggregate(AggregateArgs),
}

/// Args for `vernier aggregate` (ADR-0046).
#[derive(Debug, Parser)]
pub(crate) struct AggregateArgs {
    /// Partition manifest. Must declare `key_kind: "result"` (or the
    /// CSV form, which is `result`-only on this verb).
    #[arg(long, value_name = "PATH")]
    pub(crate) manifest: PathBuf,

    /// Shell glob over result JSON files (e.g. `runs/*.json`). Each
    /// match is parsed as a `vernier eval` document (v1 or v2). A
    /// glob that matches zero files is a typed error.
    #[arg(long, value_name = "GLOB")]
    pub(crate) results: String,

    /// Baseline slice value (e.g. `clean`). When set, rPC columns
    /// (`<metric>__rpc = mean(non-baseline) / mean(baseline)`) are
    /// appended. Omit for the comparative table without relative
    /// reductions.
    #[arg(long, value_name = "VALUE")]
    pub(crate) baseline: Option<String>,

    /// Metric names to aggregate. Repeatable. Defaults to every
    /// numeric stat the slice tables expose (one column per
    /// `Summary::stats()` position, named after the corresponding
    /// pretty-line).
    #[arg(long, value_name = "NAME", action = ArgAction::Append)]
    pub(crate) metric: Vec<String>,

    /// Repeatable emit selector — same shape as `vernier eval`'s
    /// `--emit` flag.
    #[arg(long = "emit", value_name = "FMT[=PATH]", num_args = 1, action = ArgAction::Append)]
    pub(crate) emit: Vec<String>,

    /// Suppress diagnostic output on stderr.
    #[arg(long, action = ArgAction::SetTrue)]
    pub(crate) quiet: bool,
}

impl AggregateArgs {
    /// Resolve the emit list, enforcing the same single-stdout /
    /// unique-file-paths rules `EvalArgs::validate` does.
    pub(crate) fn validate(&self) -> Result<Vec<EmitSpec>, CliError> {
        let raw_emits: Vec<String> = if self.emit.is_empty() {
            vec!["text".to_string()]
        } else {
            self.emit.clone()
        };
        let mut parsed: Vec<EmitSpec> = Vec::with_capacity(raw_emits.len());
        let mut stdout_seen = false;
        for raw in &raw_emits {
            let spec = parse_emit(raw)?;
            match &spec.destination {
                EmitDestination::Stdout => {
                    if stdout_seen {
                        return Err(CliError::Validation(
                            "more than one --emit targets stdout; outputs would interleave".into(),
                        ));
                    }
                    stdout_seen = true;
                }
                EmitDestination::File(path) => {
                    let collides = parsed.iter().any(|e| match &e.destination {
                        EmitDestination::File(p) => p == path,
                        EmitDestination::Stdout => false,
                    });
                    if collides {
                        return Err(CliError::Validation(format!(
                            "--emit path {} appears more than once",
                            path.display()
                        )));
                    }
                }
            }
            parsed.push(spec);
        }
        Ok(parsed)
    }
}

/// Args for `vernier eval`. Field-level docs become clap's `--help`
/// text; keep them user-facing.
#[derive(Debug, Parser)]
pub(crate) struct EvalArgs {
    /// Path to the ground-truth JSON file (COCO `instances_*.json`
    /// shape).
    #[arg(long, value_name = "PATH")]
    pub(crate) gt: PathBuf,

    /// Path to the detection/result JSON file (the shape pycocotools'
    /// `loadRes` consumes).
    #[arg(long, value_name = "PATH")]
    pub(crate) dt: PathBuf,

    /// IoU kind to evaluate. Required, no default — the four kinds
    /// produce numerically distinct stats and silently defaulting one
    /// would hide a missing flag in CI scripts.
    #[arg(long = "iou-type", value_enum)]
    pub(crate) iou_type: IouTypeArg,

    /// Parity mode (per ADR-0002). Defaults to `strict` (the CLI's
    /// role as a parity oracle ranks above its role as an opinionated
    /// fixer).
    #[arg(long = "parity-mode", value_enum, default_value_t = ParityModeArg::Strict)]
    pub(crate) parity_mode: ParityModeArg,

    /// Comma-separated `max_dets` ladder (e.g. `1,10,100`). Omit to
    /// use the kernel-canonical default: `[20]` for keypoints,
    /// `[1, 10, 100]` for everything else (ADR-0012).
    ///
    /// Stored as a raw string at parse time; comma-splitting and
    /// integer-parsing happen in [`EvalArgs::validate`] so that clap's
    /// derive layer (which would otherwise treat `Vec<T>` as a
    /// repeatable arg) does not fight the single-flag-comma-list shape.
    #[arg(long = "max-dets", value_name = "a,b,c")]
    pub(crate) max_dets: Option<String>,

    /// Use the dataset's `category_id` field for matching. The default;
    /// pass `--no-use-cats` to collapse every category onto a single
    /// virtual bucket (quirk **L4**).
    #[arg(
        long = "use-cats",
        default_value_t = true,
        action = ArgAction::Set,
        value_name = "BOOL",
        num_args = 0..=1,
        default_missing_value = "true",
        overrides_with = "no_use_cats"
    )]
    pub(crate) use_cats: bool,

    /// Inverse of `--use-cats`; provided for shell-script ergonomics.
    /// Equivalent to `--use-cats false`.
    #[arg(long = "no-use-cats", action = ArgAction::SetTrue, hide = true)]
    pub(crate) no_use_cats: bool,

    /// Boundary band width as a fraction of the image diagonal
    /// (ADR-0010). Only valid with `--iou-type boundary`.
    #[arg(long = "dilation-ratio", value_name = "FLOAT")]
    pub(crate) dilation_ratio: Option<f64>,

    /// Path to a JSON file mapping `category_id` → per-keypoint
    /// sigmas (ADR-0012). Only valid with `--iou-type keypoints`.
    #[arg(long, value_name = "FILE")]
    pub(crate) sigmas: Option<PathBuf>,

    /// Repeatable emit selector. Each value is `FMT` (writes to
    /// stdout) or `FMT=PATH` (writes to a file). Default if absent:
    /// `text` on stdout. At most one emit may target stdout, and
    /// duplicate file paths are rejected.
    #[arg(long = "emit", value_name = "FMT[=PATH]", num_args = 1, action = ArgAction::Append)]
    pub(crate) emit: Vec<String>,

    /// Suppress diagnostic output on stderr. Stdout (the summary
    /// data) is unaffected.
    #[arg(long, action = ArgAction::SetTrue)]
    pub(crate) quiet: bool,

    /// Headline metric. Defaults to `ap` (the COCO-style AP/AR
    /// summary, identical to the pre-LRP CLI shape). `olrp` swaps in
    /// the LRP / oLRP error decomposition per ADR-0043 and emits
    /// per-class oLRP plus the deployable confidence threshold
    /// (`tau`).
    #[arg(long = "metric", value_enum, default_value_t = MetricArg::Ap)]
    pub(crate) metric: MetricArg,

    /// Partition manifest (ADR-0046). When set, eval fans out per
    /// `(axis, value)` cell described in the manifest and the output
    /// emits the schema-v2 `slices` document instead of v1. File
    /// extension drives the parser: `.json` is the canonical
    /// JSON-records shape; `.csv` is the spreadsheet form (CSV is
    /// `key_kind=image_id` only on this verb — `key_kind=result` is
    /// `vernier aggregate`'s input, not eval's).
    #[arg(long, value_name = "PATH")]
    pub(crate) manifest: Option<PathBuf>,

    /// Joint-cell axis tuples (ADR-0046 §E2). Per-axis marginals are
    /// always emitted; `--cross` opts into the joint cells of the
    /// named axes. Each `--cross` flag is a comma-separated tuple of
    /// at least two axis names; the flag is repeatable to declare
    /// multiple cross-product tuples. Only meaningful with
    /// `--manifest`; passing it alone is a typed-error.
    ///
    /// Stored as raw comma-joined strings at parse time; the
    /// tuple-per-flag boundary is what callers want to preserve
    /// (`value_delimiter = ','` would flatten every flag invocation
    /// into one big list, losing the tuple structure). Splitting
    /// happens in [`EvalArgs::parsed_cross_axes`].
    #[arg(long, value_name = "AXES", action = ArgAction::Append)]
    pub(crate) cross: Vec<String>,

    /// Run label (ADR-0046). Stamped into the result document so a
    /// subsequent `vernier aggregate` can join a `key_kind=result`
    /// manifest by label rather than by file path. Optional; absent
    /// when the user does not plan to aggregate.
    #[arg(long, value_name = "NAME")]
    pub(crate) label: Option<String>,
}

/// Headline-metric selector. Per ADR-0043 LRP / oLRP is opt-in (the
/// existing AP-table contract is unchanged for callers who do not
/// pass `--metric olrp`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "lower")]
pub(crate) enum MetricArg {
    /// Default — COCO-style AP / AR summary.
    Ap,
    /// Localization Recall Precision (Oksuz TPAMI 2021).
    Olrp,
}

/// IoU kind selector. Maps onto `vernier_core` kernels; see
/// `crate::commands::eval` for the dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "lower")]
pub(crate) enum IouTypeArg {
    /// Bounding-box IoU.
    Bbox,
    /// Segmentation-mask IoU.
    Segm,
    /// Boundary IoU (ADR-0010).
    Boundary,
    /// Object Keypoint Similarity (ADR-0012).
    Keypoints,
}

impl IouTypeArg {
    /// Lowercase user-facing name (matches `--iou-type` values and the
    /// `iou_type` field in the JSON schema).
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Bbox => "bbox",
            Self::Segm => "segm",
            Self::Boundary => "boundary",
            Self::Keypoints => "keypoints",
        }
    }

    /// Project to the `vernier-core` kernel discriminant. Used by the
    /// metric pipelines that need to look up per-kernel defaults.
    pub(crate) fn kernel_kind(self) -> vernier_core::evaluate::KernelKind {
        use vernier_core::evaluate::KernelKind;
        match self {
            Self::Bbox => KernelKind::Bbox,
            Self::Segm => KernelKind::Segm,
            Self::Boundary => KernelKind::Boundary,
            Self::Keypoints => KernelKind::Keypoints,
        }
    }

    /// Default area-range bucket set for this kernel. Keypoints use a
    /// dedicated table (no `small` bucket and a single `large` bucket);
    /// every other kernel uses the four-bucket COCO default.
    pub(crate) fn default_area_ranges(self) -> Vec<vernier_core::evaluate::AreaRange> {
        match self {
            Self::Keypoints => vernier_core::evaluate::AreaRange::keypoints_default().to_vec(),
            _ => vernier_core::evaluate::AreaRange::coco_default().to_vec(),
        }
    }
}

/// Parity-mode selector. Per ADR-0015 §"Surface", the CLI accepts
/// three values; `aligned` is output-equivalent to `strict` (per the
/// `ParityMode` doc in `vernier-core`) and is mapped to
/// [`ParityMode::Strict`] downstream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "lower")]
pub(crate) enum ParityModeArg {
    /// Reproduce pycocotools bit-exactly.
    Strict,
    /// Output-equivalent to `strict`; documented separately because the
    /// quirk-disposition table treats `aligned` and `strict` as distinct
    /// at the disposition level (per ADR-0002).
    Aligned,
    /// Apply the opinionated `corrected` fixes (per ADR-0002).
    Corrected,
}

impl From<ParityModeArg> for ParityMode {
    fn from(value: ParityModeArg) -> Self {
        match value {
            // `Aligned` collapses to `Strict` because aligned-tier
            // changes are output-equivalent to strict; the CLI exposes
            // the third value purely for documentation symmetry with
            // ADR-0002.
            ParityModeArg::Strict | ParityModeArg::Aligned => Self::Strict,
            ParityModeArg::Corrected => Self::Corrected,
        }
    }
}

/// Parsed `--emit FMT[=PATH]` entry.
#[derive(Debug, Clone)]
pub(crate) struct EmitSpec {
    /// Formatter name as registered in [`crate::format::registry`].
    pub(crate) format: FormatName,
    /// Destination: `None` writes to stdout.
    pub(crate) destination: EmitDestination,
}

/// Where an emit goes. `Stdout` and `File` are mutually exclusive on a
/// per-emit basis; cross-emit collisions (two stdout, two same-path
/// files) are rejected by [`EvalArgs::validate`].
#[derive(Debug, Clone)]
pub(crate) enum EmitDestination {
    /// Stdout. At most one emit may target this.
    Stdout,
    /// File path. Each path may appear at most once.
    File(PathBuf),
}

impl EvalArgs {
    /// Parse the raw `--max-dets` string into a typed
    /// `Option<Vec<usize>>`. Returns `Ok(None)` when the flag was
    /// absent. Per ADR-0012 the default ladder is resolved at the
    /// eval call site, not here.
    pub(crate) fn parsed_max_dets(&self) -> Result<Option<Vec<usize>>, CliError> {
        match &self.max_dets {
            None => Ok(None),
            Some(raw) => parse_max_dets(raw).map(Some).map_err(CliError::Validation),
        }
    }

    /// Split each raw `--cross` invocation on `,` to recover the
    /// `Vec<Vec<String>>` shape `PartitionSpec::build` consumes.
    ///
    /// Each flag invocation is one tuple; `--cross a,b --cross c,d`
    /// yields `[["a","b"], ["c","d"]]`. Tuples shorter than two axes
    /// or containing an empty axis name are rejected here so the
    /// downstream `vernier-core` error is a typed-CLI error instead
    /// of an `InvalidConfig` from the spec builder.
    pub(crate) fn parsed_cross_axes(&self) -> Result<Vec<Vec<String>>, CliError> {
        let mut out: Vec<Vec<String>> = Vec::with_capacity(self.cross.len());
        for (idx, raw) in self.cross.iter().enumerate() {
            let parts: Vec<String> = raw.split(',').map(|s| s.trim().to_string()).collect();
            if parts.iter().any(String::is_empty) {
                return Err(CliError::Validation(format!(
                    "--cross tuple #{idx} ({raw:?}) contains an empty axis name"
                )));
            }
            if parts.len() < 2 {
                return Err(CliError::Validation(format!(
                    "--cross tuple #{idx} ({raw:?}) must list at least two axes \
                     (a single-axis cross is a marginal — already emitted by default)"
                )));
            }
            out.push(parts);
        }
        Ok(out)
    }

    /// Effective `use_cats` after combining the `--use-cats` /
    /// `--no-use-cats` pair.
    pub(crate) fn effective_use_cats(&self) -> bool {
        if self.no_use_cats {
            false
        } else {
            self.use_cats
        }
    }

    /// Cross-flag validation per ADR-0015 §"Surface". Returns
    /// [`CliError::Validation`] on failure; clap's own parse errors
    /// are surfaced before this is ever called.
    ///
    /// Validation checks (each maps to ADR-0015's exit-code-2 cases):
    /// - `--dilation-ratio` requires `--iou-type boundary`.
    /// - `--sigmas` requires `--iou-type keypoints`.
    /// - At most one `--emit` may target stdout (the byte streams
    ///   would interleave).
    /// - No two `--emit` entries may target the same file path.
    /// - Each `--emit` must name a registered formatter.
    pub(crate) fn validate(&self) -> Result<Vec<EmitSpec>, CliError> {
        // Kind-coupling for boundary / keypoints flags.
        if self.dilation_ratio.is_some() && self.iou_type != IouTypeArg::Boundary {
            return Err(CliError::Validation(format!(
                "--dilation-ratio is only valid with --iou-type boundary; got --iou-type {}",
                self.iou_type.as_str(),
            )));
        }
        if self.sigmas.is_some() && self.iou_type != IouTypeArg::Keypoints {
            return Err(CliError::Validation(format!(
                "--sigmas is only valid with --iou-type keypoints; got --iou-type {}",
                self.iou_type.as_str(),
            )));
        }
        // Validate dilation_ratio finiteness up-front; vernier-core
        // would also reject this, but a typed CLI message is friendlier.
        if let Some(d) = self.dilation_ratio {
            if !d.is_finite() || d <= 0.0 {
                return Err(CliError::Validation(format!(
                    "--dilation-ratio must be a positive finite float; got {d}"
                )));
            }
        }
        // Empty max-dets is a CLI-level error so we don't have to walk
        // back through vernier-core's typed error for an obvious typo.
        if let Some(d) = self.parsed_max_dets()? {
            if d.is_empty() {
                return Err(CliError::Validation(
                    "--max-dets must contain at least one entry".into(),
                ));
            }
        }
        // ADR-0046: `--cross` is only meaningful with `--manifest`. A
        // bare `--cross` is almost always a script bug; raise it
        // explicitly rather than silently dropping the flag.
        if !self.cross.is_empty() && self.manifest.is_none() {
            return Err(CliError::Validation(
                "--cross requires --manifest; cross-product cells are a partition-only concept"
                    .into(),
            ));
        }
        // Surface the tuple-shape errors at validate time too, even
        // though the partition dispatch would catch them later. This
        // keeps the "validate the args before doing any work" promise
        // intact.
        let _ = self.parsed_cross_axes()?;
        // Default to a single text-on-stdout emit when no `--emit`
        // flag was supplied. This is the canonical no-flag invocation
        // shape ADR-0015 §"Surface" pins.
        let raw_emits: Vec<String> = if self.emit.is_empty() {
            vec!["text".to_string()]
        } else {
            self.emit.clone()
        };

        let mut parsed: Vec<EmitSpec> = Vec::with_capacity(raw_emits.len());
        let mut stdout_seen = false;
        for raw in &raw_emits {
            let spec = parse_emit(raw)?;
            match &spec.destination {
                EmitDestination::Stdout => {
                    if stdout_seen {
                        return Err(CliError::Validation(
                            "more than one --emit targets stdout; outputs would interleave".into(),
                        ));
                    }
                    stdout_seen = true;
                }
                EmitDestination::File(path) => {
                    let collides = parsed.iter().any(|e| match &e.destination {
                        EmitDestination::File(p) => p == path,
                        EmitDestination::Stdout => false,
                    });
                    if collides {
                        return Err(CliError::Validation(format!(
                            "--emit path {} appears more than once",
                            path.display()
                        )));
                    }
                }
            }
            parsed.push(spec);
        }
        Ok(parsed)
    }
}

fn parse_emit(raw: &str) -> Result<EmitSpec, CliError> {
    let (name, dest) = match raw.split_once('=') {
        Some((name, path)) => {
            if path.is_empty() {
                return Err(CliError::Validation(format!(
                    "--emit value {raw:?} has an empty path after '='"
                )));
            }
            (name, EmitDestination::File(PathBuf::from(path)))
        }
        None => (raw, EmitDestination::Stdout),
    };
    let format = FormatName::lookup(name).ok_or_else(|| {
        CliError::Validation(format!(
            "unknown --emit format {name:?}; known formats: {}",
            FormatName::known_names_joined()
        ))
    })?;
    Ok(EmitSpec {
        format,
        destination: dest,
    })
}

fn parse_max_dets(raw: &str) -> Result<Vec<usize>, String> {
    if raw.trim().is_empty() {
        return Err("--max-dets must not be empty".into());
    }
    let mut out = Vec::new();
    for part in raw.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            return Err(format!("--max-dets contains an empty entry; got {raw:?}"));
        }
        let v: usize = trimmed.parse().map_err(|e| {
            format!("--max-dets entry {trimmed:?} is not a non-negative integer: {e}")
        })?;
        out.push(v);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn parse(args: &[&str]) -> Cli {
        let mut full: Vec<&str> = vec!["vernier"];
        full.extend_from_slice(args);
        Cli::parse_from(full)
    }

    #[test]
    fn happy_path_parses() {
        let cli = parse(&[
            "eval",
            "--gt",
            "gt.json",
            "--dt",
            "dt.json",
            "--iou-type",
            "bbox",
        ]);
        let Command::Eval(args) = cli.command else {
            panic!("expected Eval")
        };
        assert_eq!(args.iou_type, IouTypeArg::Bbox);
        assert_eq!(args.parity_mode, ParityModeArg::Strict);
        assert!(args.max_dets.is_none());
        assert!(args.effective_use_cats());
    }

    #[test]
    fn dilation_ratio_rejected_for_non_boundary() {
        let cli = parse(&[
            "eval",
            "--gt",
            "gt.json",
            "--dt",
            "dt.json",
            "--iou-type",
            "bbox",
            "--dilation-ratio",
            "0.02",
        ]);
        let Command::Eval(args) = cli.command else {
            panic!("expected Eval")
        };
        let err = args.validate().unwrap_err();
        assert!(matches!(err, CliError::Validation(_)));
    }

    #[test]
    fn sigmas_rejected_for_non_keypoints() {
        let cli = parse(&[
            "eval",
            "--gt",
            "gt.json",
            "--dt",
            "dt.json",
            "--iou-type",
            "segm",
            "--sigmas",
            "sigmas.json",
        ]);
        let Command::Eval(args) = cli.command else {
            panic!("expected Eval")
        };
        let err = args.validate().unwrap_err();
        assert!(matches!(err, CliError::Validation(_)));
    }

    #[test]
    fn double_stdout_emit_rejected() {
        let cli = parse(&[
            "eval",
            "--gt",
            "gt.json",
            "--dt",
            "dt.json",
            "--iou-type",
            "bbox",
            "--emit",
            "text",
            "--emit",
            "json",
        ]);
        let Command::Eval(args) = cli.command else {
            panic!("expected Eval")
        };
        let err = args.validate().unwrap_err();
        assert!(matches!(err, CliError::Validation(_)));
    }

    #[test]
    fn duplicate_path_emit_rejected() {
        let cli = parse(&[
            "eval",
            "--gt",
            "gt.json",
            "--dt",
            "dt.json",
            "--iou-type",
            "bbox",
            "--emit",
            "text=out.txt",
            "--emit",
            "json=out.txt",
        ]);
        let Command::Eval(args) = cli.command else {
            panic!("expected Eval")
        };
        let err = args.validate().unwrap_err();
        assert!(matches!(err, CliError::Validation(_)));
    }

    #[test]
    fn no_use_cats_overrides_use_cats() {
        let cli = parse(&[
            "eval",
            "--gt",
            "gt.json",
            "--dt",
            "dt.json",
            "--iou-type",
            "bbox",
            "--no-use-cats",
        ]);
        let Command::Eval(args) = cli.command else {
            panic!("expected Eval")
        };
        assert!(!args.effective_use_cats());
    }

    #[test]
    fn max_dets_parses_comma_list() {
        let cli = parse(&[
            "eval",
            "--gt",
            "gt.json",
            "--dt",
            "dt.json",
            "--iou-type",
            "bbox",
            "--max-dets",
            "1,10,100",
        ]);
        let Command::Eval(args) = cli.command else {
            panic!("expected Eval")
        };
        assert_eq!(args.max_dets.as_deref(), Some("1,10,100"));
        assert_eq!(args.parsed_max_dets().unwrap(), Some(vec![1usize, 10, 100]));
    }

    #[test]
    fn validate_default_emits_text_to_stdout() {
        let cli = parse(&[
            "eval",
            "--gt",
            "gt.json",
            "--dt",
            "dt.json",
            "--iou-type",
            "bbox",
        ]);
        let Command::Eval(args) = cli.command else {
            panic!("expected Eval")
        };
        let emits = args.validate().unwrap();
        assert_eq!(emits.len(), 1);
        assert!(matches!(emits[0].destination, EmitDestination::Stdout));
        assert_eq!(emits[0].format, FormatName::Text);
    }

    #[test]
    fn unknown_emit_format_rejected() {
        let cli = parse(&[
            "eval",
            "--gt",
            "gt.json",
            "--dt",
            "dt.json",
            "--iou-type",
            "bbox",
            "--emit",
            "yaml",
        ]);
        let Command::Eval(args) = cli.command else {
            panic!("expected Eval")
        };
        let err = args.validate().unwrap_err();
        assert!(matches!(err, CliError::Validation(_)));
    }
}
