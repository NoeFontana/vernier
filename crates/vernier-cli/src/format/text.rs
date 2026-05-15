//! Text formatter — the no-flag default.
//!
//! Per ADR-0015 §"Formatter: text (default)", the AP text formatter
//! is byte-identical to `Summary::pretty_lines` joined by `'\n'`
//! and terminated with `'\n'`. In strict mode, the bytes match
//! pycocotools' `summarize()` stdout output (modulo the trailing
//! newline, which pycocotools' `print()` also emits).
//!
//! The LRP variant ships a per-class table plus the four aggregated
//! numbers (per ADR-0043). Per ADR-0015's formatter-abstraction
//! contract the rendering is deterministic and timestamp-free.

use std::io;

use crate::error::CliError;
use crate::format::{EvalArtifact, FormatContext, FormatName, Formatter};

/// Zero-sized formatter that delegates to
/// [`vernier_core::Summary::pretty_lines`] for AP / AR, and emits a
/// fixed-width per-class table for LRP / oLRP.
pub(crate) struct Text;

impl Formatter for Text {
    fn name(&self) -> &'static str {
        "text"
    }

    fn id(&self) -> FormatName {
        FormatName::Text
    }

    fn render(
        &self,
        artifact: &EvalArtifact<'_>,
        _ctx: &FormatContext<'_>,
        out: &mut dyn io::Write,
    ) -> Result<(), CliError> {
        match artifact {
            EvalArtifact::Ap(summary) => {
                for line in summary.pretty_lines() {
                    writeln!(out, "{line}")?;
                }
                Ok(())
            }
            EvalArtifact::Lrp(report) => render_lrp(report, out),
            EvalArtifact::Partitioned { summary, label } => {
                render_partitioned(summary, *label, out)
            }
            EvalArtifact::PartitionedLrp { summary, label } => {
                render_partitioned_lrp(summary, *label, out)
            }
        }
    }
}

fn render_lrp(
    report: &vernier_core::lrp::LrpReport,
    out: &mut dyn io::Write,
) -> Result<(), CliError> {
    writeln!(out, "oLRP     = {olrp:.4}", olrp = report.olrp,)?;
    writeln!(out, "oLRP_Loc = {v:.4}", v = report.olrp_loc)?;
    writeln!(out, "oLRP_FP  = {v:.4}", v = report.olrp_fp)?;
    writeln!(out, "oLRP_FN  = {v:.4}", v = report.olrp_fn)?;
    writeln!(out, "n_empty_classes = {n}", n = report.n_empty_classes,)?;
    writeln!(
        out,
        "config: kernel={kernel} tp_threshold={tp:.3} tau_grid_len={g}",
        kernel = report.config.kernel.as_str(),
        tp = report.config.tp_threshold,
        g = report.config.tau_grid_len,
    )?;
    if !report.per_class.is_empty() {
        writeln!(
            out,
            "{:>10} {:>8} {:>8} {:>8} {:>8} {:>6}",
            "cat_id", "oLRP", "Loc", "FP", "FN", "tau"
        )?;
        for entry in &report.per_class {
            writeln!(
                out,
                "{:>10} {:>8} {:>8} {:>8} {:>8} {:>6}",
                entry.category_id,
                fmt_opt(entry.olrp),
                fmt_opt(entry.olrp_loc),
                fmt_opt(entry.olrp_fp),
                fmt_opt(entry.olrp_fn),
                fmt_opt(entry.tau),
            )?;
        }
    }
    Ok(())
}

fn fmt_opt(v: Option<f64>) -> String {
    match v {
        Some(x) => format!("{x:.4}"),
        None => "NaN".to_string(),
    }
}

/// Render a partitioned (ADR-0046 / v2) eval: the overall pretty-lines
/// table first, then one labelled block per slice with a header line
/// of the form `==> axis = value  (n_images=N, n_detections=M)`.
fn render_partitioned(
    summary: &vernier_core::partition::PartitionedSummary,
    label: Option<&str>,
    out: &mut dyn io::Write,
) -> Result<(), CliError> {
    write_partitioned_preamble(
        out,
        label,
        summary.overall_n_images,
        summary.overall_n_detections,
    )?;
    for line in summary.overall.pretty_lines() {
        writeln!(out, "{line}")?;
    }
    for sr in &summary.slices {
        writeln!(out)?;
        write_slice_header(
            out,
            &sr.slice.axis,
            &sr.slice.value,
            sr.n_images,
            sr.n_detections,
        )?;
        for line in sr.summary.pretty_lines() {
            writeln!(out, "{line}")?;
        }
    }
    Ok(())
}

/// Shared label + overall header, written by both `render_partitioned`
/// (AP) and `render_partitioned_lrp` (LRP). Centralised so the two
/// output formats can't drift on the canonical preamble shape.
fn write_partitioned_preamble(
    out: &mut dyn io::Write,
    label: Option<&str>,
    overall_n_images: u64,
    overall_n_detections: u64,
) -> Result<(), CliError> {
    if let Some(label) = label {
        writeln!(out, "label = {label}")?;
    }
    writeln!(
        out,
        "overall  (n_images={overall_n_images}, n_detections={overall_n_detections})"
    )?;
    Ok(())
}

/// Shared per-slice header. Same canonical shape across AP and LRP
/// partitioned text output.
fn write_slice_header(
    out: &mut dyn io::Write,
    axis: &str,
    value: &str,
    n_images: u64,
    n_detections: u64,
) -> Result<(), CliError> {
    writeln!(
        out,
        "==>  {axis} = {value}  (n_images={n_images}, n_detections={n_detections})"
    )?;
    Ok(())
}

/// Render a partitioned LRP (ADR-0046 / v2) result: the overall block
/// (mirrors the un-partitioned `render_lrp` headline shape, minus the
/// per-class table — that is omitted at LVIS scale by design, same as
/// the JSON envelope) then one labelled block per slice with the four
/// headline numbers (`oLRP` / `oLRP_Loc` / `oLRP_FP` / `oLRP_FN`).
fn render_partitioned_lrp(
    summary: &vernier_core::partition::PartitionedLrpReport,
    label: Option<&str>,
    out: &mut dyn io::Write,
) -> Result<(), CliError> {
    write_partitioned_preamble(
        out,
        label,
        summary.overall_n_images,
        summary.overall_n_detections,
    )?;
    writeln!(out, "oLRP     = {olrp:.4}", olrp = summary.overall.olrp)?;
    writeln!(out, "oLRP_Loc = {v:.4}", v = summary.overall.olrp_loc)?;
    writeln!(out, "oLRP_FP  = {v:.4}", v = summary.overall.olrp_fp)?;
    writeln!(out, "oLRP_FN  = {v:.4}", v = summary.overall.olrp_fn)?;
    writeln!(
        out,
        "n_empty_classes = {n}",
        n = summary.overall.n_empty_classes
    )?;
    writeln!(
        out,
        "config: kernel={kernel} tp_threshold={tp:.3} tau_grid_len={g}",
        kernel = summary.overall.config.kernel.as_str(),
        tp = summary.overall.config.tp_threshold,
        g = summary.overall.config.tau_grid_len,
    )?;
    for sr in &summary.slices {
        writeln!(out)?;
        write_slice_header(
            out,
            &sr.slice.axis,
            &sr.slice.value,
            sr.n_images,
            sr.n_detections,
        )?;
        writeln!(out, "oLRP     = {v:.4}", v = sr.report.olrp)?;
        writeln!(out, "oLRP_Loc = {v:.4}", v = sr.report.olrp_loc)?;
        writeln!(out, "oLRP_FP  = {v:.4}", v = sr.report.olrp_fp)?;
        writeln!(out, "oLRP_FN  = {v:.4}", v = sr.report.olrp_fn)?;
        writeln!(out, "n_empty_classes = {n}", n = sr.report.n_empty_classes)?;
    }
    Ok(())
}
