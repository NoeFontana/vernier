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
