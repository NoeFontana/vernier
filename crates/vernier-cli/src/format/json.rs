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
//!   [`vernier_core::Summary::lines`] / [`Summary::pretty_lines`].
//! - No timestamps, host, user, cwd, or build-metadata fields.
//! - Trailing newline after the closing brace so shell pipelines that
//!   `cat` two outputs together get clean line boundaries.

use std::io;

use serde::Serialize;
use vernier_core::{Metric, ParityMode, StatLine, Summary};

use crate::error::CliError;
use crate::format::{FormatContext, FormatName, Formatter};

/// Schema version pinned at v0.2.0. Bumping requires the same bar as
/// reshaping the on-disk format: a new ADR plus a major-version
/// release. See ADR-0015 §"Versioning and stability commitments".
pub(crate) const SCHEMA_VERSION: &str = "1";

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
    /// One entry per [`vernier_core::StatLine`] in plan order.
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
        // Pretty-print disabled: pretty bytes inflate the archive
        // size and add no signal `jq` can't reformat. The default
        // `to_writer` is the canonical compact form and is what
        // ADR-0015's byte-determinism contract is built around.
        serde_json::to_writer(&mut *out, &doc)?;
        // Trailing newline so concatenating two outputs (`cat a.json
        // b.json`) does not produce `}{` on a single line. ADR-0015
        // §"Output determinism" requires this.
        writeln!(out)?;
        Ok(())
    }
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
