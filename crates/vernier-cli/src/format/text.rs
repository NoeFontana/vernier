//! Text formatter — the no-flag default.
//!
//! Per ADR-0015 §"Formatter: text (default)", the text formatter is
//! byte-identical to [`Summary::pretty_lines`] joined by `'\n'` and
//! terminated with `'\n'`. In strict mode, the bytes match
//! pycocotools' `summarize()` stdout output (modulo the trailing
//! newline, which pycocotools' `print()` also emits).
//!
//! All semantic content lives upstream in `vernier_core::summarize`;
//! this module is intentionally a thin shell.

use std::io;

use vernier_core::Summary;

use crate::error::CliError;
use crate::format::{FormatContext, FormatName, Formatter};

/// Zero-sized formatter that delegates to [`Summary::pretty_lines`].
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
        summary: &Summary,
        _ctx: &FormatContext<'_>,
        out: &mut dyn io::Write,
    ) -> Result<(), CliError> {
        for line in summary.pretty_lines() {
            writeln!(out, "{line}")?;
        }
        Ok(())
    }
}
