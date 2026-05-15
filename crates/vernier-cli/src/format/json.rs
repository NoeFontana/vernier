//! JSON formatter — schema-versioned structured output.
//!
//! Per ADR-0015 §"Formatter: JSON" and §"Output determinism", the JSON
//! formatter ships a stable schema versioned independently of the
//! vernier crate version. v0.2 emits `"version": "1"`. The shape is
//! deliberately *not* `serde_json::to_string(&summary)` — `Summary`'s
//! field layout is internal and reshaping it for users is the
//! formatter's job.
//!
//! Determinism requirements ADR-0015 §"Output determinism" pins:
//!
//! - Stable key order (the explicit order the [`SchemaV1`] struct
//!   declares its fields in).
//! - `lines` array in plan order — same order as
//!   `vernier_core::Summary::lines` / `Summary::pretty_lines`.
//! - No timestamps, host, user, cwd, or build-metadata fields.
//! - Trailing newline after the closing brace so shell pipelines that
//!   `cat` two outputs together get clean line boundaries.

use std::io;

use serde::Serialize;
use vernier_core::lrp::LrpReport;
use vernier_core::partition::{PartitionedLrpReport, PartitionedSummary};
use vernier_core::summarize::{Metric, StatLine};
use vernier_core::{ParityMode, Summary};

use crate::error::CliError;
use crate::format::{EvalArtifact, FormatContext, FormatName, Formatter};

/// Schema version pinned at v0.2.0. Bumping requires the same bar as
/// reshaping the on-disk format: a new ADR plus a major-version
/// release. See ADR-0015 §"Versioning and stability commitments".
pub(crate) const SCHEMA_VERSION: &str = "1";

/// Partitioned-output schema version (ADR-0046 / ADR-0039). The v1
/// shape stays in force for un-partitioned eval; v2 is only emitted
/// when `--manifest` is supplied. Un-partitioned eval is byte-stable
/// at v1 — that is the load-bearing contract this constant guards.
pub(crate) const SCHEMA_VERSION_V2: &str = "2";

/// Top-level JSON document. Field order is the wire-stable serialized
/// order, *not* serde's insertion order, because the struct's source
/// declaration *is* the schema. Tests assert on the produced bytes.
#[derive(Debug, Serialize)]
struct SchemaV1<'a> {
    /// Schema version pin (`"1"` at v0.2). Surfaces first so a
    /// downstream tool can sniff compatibility without parsing the
    /// rest.
    version: &'a str,
    /// IoU kind name as user-facing string (`bbox` / `segm` /
    /// `boundary` / `keypoints`).
    iou_type: &'a str,
    /// Parity mode after the CLI's `aligned`→`strict` collapse.
    parity_mode: &'a str,
    /// Resolved `max_dets` ladder.
    max_dets: &'a [usize],
    /// Effective `use_cats`.
    use_cats: bool,
    /// One entry per [`vernier_core::summarize::StatLine`] in plan order.
    lines: Vec<LineV1<'a>>,
    /// Numeric values in plan order — duplicated alongside `lines` so
    /// pycocotools-trained tooling gets a one-line port (per ADR-0015
    /// §"Formatter: JSON").
    stats: Vec<f64>,
}

/// Per-line shape. Field order is wire-stable and tested.
#[derive(Debug, Serialize)]
struct LineV1<'a> {
    /// `AP` or `AR`.
    metric: &'static str,
    /// `Some(t)` for an exact-threshold line, `None` for the averaged
    /// `0.50:0.95` line.
    iou_threshold: Option<f64>,
    /// Always populated: pretty-printable label for the IoU axis
    /// (either the threshold formatted to two decimals, or
    /// `"0.50:0.95"`).
    iou_threshold_label: String,
    /// Area-bucket label (`all`, `small`, `medium`, `large`, or a
    /// custom label for a non-canonical bucket).
    area: &'a str,
    /// `max_dets` cap selected for this line.
    max_dets: usize,
    /// Numeric value (`-1.0` for the cocoeval `-1` sentinel — quirk
    /// **C5**).
    value: f64,
}

/// Zero-sized JSON formatter.
pub(crate) struct Json;

impl Formatter for Json {
    fn name(&self) -> &'static str {
        "json"
    }

    fn id(&self) -> FormatName {
        FormatName::Json
    }

    fn render(
        &self,
        artifact: &EvalArtifact<'_>,
        ctx: &FormatContext<'_>,
        out: &mut dyn io::Write,
    ) -> Result<(), CliError> {
        match artifact {
            EvalArtifact::Ap(summary) => render_ap(summary, ctx, out),
            EvalArtifact::Lrp(report) => render_lrp(report, ctx, out),
            EvalArtifact::Partitioned { summary, label } => {
                render_partitioned(summary, *label, ctx, out)
            }
            EvalArtifact::PartitionedLrp { summary, label } => {
                render_partitioned_lrp(summary, *label, ctx, out)
            }
        }
    }
}

fn render_ap(
    summary: &Summary,
    ctx: &FormatContext<'_>,
    out: &mut dyn io::Write,
) -> Result<(), CliError> {
    let lines: Vec<LineV1<'_>> = summary.lines.iter().map(line_to_v1).collect();
    let stats = summary.stats();
    let doc = SchemaV1 {
        version: SCHEMA_VERSION,
        iou_type: ctx.iou_type.as_str(),
        parity_mode: parity_mode_str(ctx.parity_mode),
        max_dets: ctx.max_dets,
        use_cats: ctx.use_cats,
        lines,
        stats,
    };
    serde_json::to_writer(&mut *out, &doc)?;
    writeln!(out)?;
    Ok(())
}

fn render_lrp(
    report: &LrpReport,
    ctx: &FormatContext<'_>,
    out: &mut dyn io::Write,
) -> Result<(), CliError> {
    let per_class: Vec<LrpClassV1> = report
        .per_class
        .iter()
        .map(|c| LrpClassV1 {
            category_id: c.category_id,
            olrp: c.olrp,
            olrp_loc: c.olrp_loc,
            olrp_fp: c.olrp_fp,
            olrp_fn: c.olrp_fn,
            tau: c.tau,
        })
        .collect();
    let doc = LrpSchemaV1 {
        version: SCHEMA_VERSION,
        metric: "olrp",
        iou_type: ctx.iou_type.as_str(),
        parity_mode: parity_mode_str(ctx.parity_mode),
        max_dets: ctx.max_dets,
        use_cats: ctx.use_cats,
        olrp: report.olrp,
        olrp_loc: report.olrp_loc,
        olrp_fp: report.olrp_fp,
        olrp_fn: report.olrp_fn,
        per_class,
        n_empty_classes: report.n_empty_classes,
        kernel: report.config.kernel.as_str(),
        tp_threshold: report.config.tp_threshold,
        tau_grid_len: report.config.tau_grid_len,
    };
    serde_json::to_writer(&mut *out, &doc)?;
    writeln!(out)?;
    Ok(())
}

/// LRP/oLRP JSON shape. Field order is wire-stable.
#[derive(Debug, Serialize)]
struct LrpSchemaV1<'a> {
    /// Schema version (shared with the AP variant).
    version: &'a str,
    /// Always `"olrp"` for this variant — discriminator so a single
    /// JSON parser handles both metric shapes.
    metric: &'static str,
    iou_type: &'a str,
    parity_mode: &'a str,
    max_dets: &'a [usize],
    use_cats: bool,
    olrp: f64,
    olrp_loc: f64,
    olrp_fp: f64,
    olrp_fn: f64,
    per_class: Vec<LrpClassV1>,
    n_empty_classes: u32,
    kernel: &'a str,
    tp_threshold: f64,
    tau_grid_len: usize,
}

/// Per-class oLRP entry. `None` fields serialize as JSON `null`.
#[derive(Debug, Serialize)]
struct LrpClassV1 {
    category_id: i64,
    olrp: Option<f64>,
    olrp_loc: Option<f64>,
    olrp_fp: Option<f64>,
    olrp_fn: Option<f64>,
    tau: Option<f64>,
}

fn line_to_v1(line: &StatLine) -> LineV1<'_> {
    let iou_threshold_label = match line.iou_threshold {
        Some(t) => format!("{t:0.2}"),
        None => "0.50:0.95".to_string(),
    };
    LineV1 {
        metric: metric_str(line.metric),
        iou_threshold: line.iou_threshold,
        iou_threshold_label,
        area: line.area.label.as_ref(),
        max_dets: line.max_dets,
        value: line.value,
    }
}

fn metric_str(m: Metric) -> &'static str {
    match m {
        Metric::AveragePrecision => "AP",
        Metric::AverageRecall => "AR",
    }
}

fn parity_mode_str(m: ParityMode) -> &'static str {
    match m {
        ParityMode::Strict => "strict",
        ParityMode::Corrected => "corrected",
    }
}

// ---------------------------------------------------------------------------
// Schema v2 — partitioned (ADR-0046).
//
// Un-partitioned eval keeps emitting v1 verbatim; v2 is only ever
// emitted from the `--manifest` dispatch lane. Field order is the wire
// order (serde follows struct declaration order — the existing
// determinism contract carries straight through).
// ---------------------------------------------------------------------------

/// Top-level v2 document. Field order is wire-stable.
#[derive(Debug, Serialize)]
struct SchemaV2<'a> {
    /// Schema version pin (`"2"` for partitioned output).
    version: &'a str,
    /// `--label` value stamped on this run, or `null` when omitted.
    /// `vernier aggregate` joins by this field when present.
    label: Option<&'a str>,
    iou_type: &'a str,
    parity_mode: &'a str,
    max_dets: &'a [usize],
    use_cats: bool,
    /// Un-partitioned summary (bit-identical to today's v1 output for
    /// the same `(GT, DT)` pair — load-bearing parity contract per
    /// ADR-0046).
    overall: OverallV2<'a>,
    /// One entry per slice in the spec's canonical order (axis
    /// ascending, value ascending, `__unassigned__` last; joint cells
    /// follow the marginals).
    slices: Vec<SliceV2<'a>>,
}

/// `overall` sub-object on the v2 document.
#[derive(Debug, Serialize)]
struct OverallV2<'a> {
    lines: Vec<LineV1<'a>>,
    stats: Vec<f64>,
    n_images: u64,
    n_detections: u64,
}

/// Per-slice entry under `slices` on the v2 document.
#[derive(Debug, Serialize)]
struct SliceV2<'a> {
    axis: &'a str,
    value: &'a str,
    n_images: u64,
    n_detections: u64,
    lines: Vec<LineV1<'a>>,
    stats: Vec<f64>,
}

fn render_partitioned(
    summary: &PartitionedSummary,
    label: Option<&str>,
    ctx: &FormatContext<'_>,
    out: &mut dyn io::Write,
) -> Result<(), CliError> {
    let overall_lines: Vec<LineV1<'_>> = summary.overall.lines.iter().map(line_to_v1).collect();
    let overall_stats = summary.overall.stats();
    let overall = OverallV2 {
        lines: overall_lines,
        stats: overall_stats,
        n_images: summary.overall_n_images,
        n_detections: summary.overall_n_detections,
    };

    let mut slices: Vec<SliceV2<'_>> = Vec::with_capacity(summary.slices.len());
    for sr in &summary.slices {
        let lines: Vec<LineV1<'_>> = sr.summary.lines.iter().map(line_to_v1).collect();
        let stats = sr.summary.stats();
        slices.push(SliceV2 {
            axis: sr.slice.axis.as_str(),
            value: sr.slice.value.as_str(),
            n_images: sr.n_images,
            n_detections: sr.n_detections,
            lines,
            stats,
        });
    }

    let doc = SchemaV2 {
        version: SCHEMA_VERSION_V2,
        label,
        iou_type: ctx.iou_type.as_str(),
        parity_mode: parity_mode_str(ctx.parity_mode),
        max_dets: ctx.max_dets,
        use_cats: ctx.use_cats,
        overall,
        slices,
    };
    serde_json::to_writer(&mut *out, &doc)?;
    writeln!(out)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// LRP schema v2 — partitioned LRP / oLRP (ADR-0046 + ADR-0043).
//
// LRP has a different column shape from AP (`olrp` / `olrp_loc` /
// `olrp_fp` / `olrp_fn` headline numbers rather than the 12/10-stat
// plan) so it gets its own v2 envelope. The `metric` field is the
// discriminator: parsers handle both v2 shapes by switching on
// `(version, metric)` rather than guessing from key presence.
//
// `per_class` is intentionally omitted from the JSON envelope by
// default. At LVIS scale (1203 categories) each slice would carry a
// 1203-row breakdown, and a partition with 8 slices would balloon the
// document past 10k rows of per-class data — most of which the
// downstream `vernier aggregate` flow does not consume. The headline
// numbers per slice are what the slice document is for; per-class
// detail belongs to the un-partitioned `--metric olrp` run that the
// user can spawn alongside. A `--per-class` opt-in is anticipated as a
// follow-up if a workload ever needs it.
// ---------------------------------------------------------------------------

/// Top-level v2 document for partitioned LRP. Field order is the wire
/// order — serde follows struct declaration order — same determinism
/// contract as the AP v2 envelope.
#[derive(Debug, Serialize)]
struct LrpSchemaV2<'a> {
    /// Schema version pin (`"2"` for partitioned output).
    version: &'a str,
    /// Always `"olrp"` for this variant — discriminator so a single
    /// JSON parser handles both v2 metric shapes.
    metric: &'static str,
    /// `--label` value stamped on this run, or `null` when omitted.
    label: Option<&'a str>,
    iou_type: &'a str,
    parity_mode: &'a str,
    use_cats: bool,
    /// Un-partitioned LRP block (bit-identical to a single
    /// `optimal_lrp_*` call over the same `(GT, DT)` — load-bearing
    /// parity contract per ADR-0046).
    overall: LrpOverallV2<'a>,
    /// One entry per slice in the spec's canonical order (axis
    /// ascending, value ascending, `__unassigned__` last; joint cells
    /// follow the marginals).
    slices: Vec<LrpSliceV2<'a>>,
}

/// `overall` sub-object on the LRP v2 document.
#[derive(Debug, Serialize)]
struct LrpOverallV2<'a> {
    olrp: f64,
    olrp_loc: f64,
    olrp_fp: f64,
    olrp_fn: f64,
    n_empty_classes: u32,
    n_images: u64,
    n_detections: u64,
    config: LrpConfigV2<'a>,
}

/// Per-slice entry under `slices` on the LRP v2 document.
#[derive(Debug, Serialize)]
struct LrpSliceV2<'a> {
    axis: &'a str,
    value: &'a str,
    n_images: u64,
    n_detections: u64,
    olrp: f64,
    olrp_loc: f64,
    olrp_fp: f64,
    olrp_fn: f64,
    n_empty_classes: u32,
}

/// Resolved LRP configuration, mirroring [`vernier_core::lrp::LrpConfig`].
#[derive(Debug, Serialize)]
struct LrpConfigV2<'a> {
    tp_threshold: f64,
    tau_grid_len: usize,
    kernel: &'a str,
}

fn render_partitioned_lrp(
    summary: &PartitionedLrpReport,
    label: Option<&str>,
    ctx: &FormatContext<'_>,
    out: &mut dyn io::Write,
) -> Result<(), CliError> {
    let overall = LrpOverallV2 {
        olrp: summary.overall.olrp,
        olrp_loc: summary.overall.olrp_loc,
        olrp_fp: summary.overall.olrp_fp,
        olrp_fn: summary.overall.olrp_fn,
        n_empty_classes: summary.overall.n_empty_classes,
        n_images: summary.overall_n_images,
        n_detections: summary.overall_n_detections,
        config: LrpConfigV2 {
            tp_threshold: summary.overall.config.tp_threshold,
            tau_grid_len: summary.overall.config.tau_grid_len,
            kernel: summary.overall.config.kernel.as_str(),
        },
    };

    let mut slices: Vec<LrpSliceV2<'_>> = Vec::with_capacity(summary.slices.len());
    for sr in &summary.slices {
        slices.push(LrpSliceV2 {
            axis: sr.slice.axis.as_str(),
            value: sr.slice.value.as_str(),
            n_images: sr.n_images,
            n_detections: sr.n_detections,
            olrp: sr.report.olrp,
            olrp_loc: sr.report.olrp_loc,
            olrp_fp: sr.report.olrp_fp,
            olrp_fn: sr.report.olrp_fn,
            n_empty_classes: sr.report.n_empty_classes,
        });
    }

    let doc = LrpSchemaV2 {
        version: SCHEMA_VERSION_V2,
        metric: "olrp",
        label,
        iou_type: ctx.iou_type.as_str(),
        parity_mode: parity_mode_str(ctx.parity_mode),
        use_cats: ctx.use_cats,
        overall,
        slices,
    };
    serde_json::to_writer(&mut *out, &doc)?;
    writeln!(out)?;
    Ok(())
}
