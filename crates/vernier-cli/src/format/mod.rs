//! Formatter trait, registry, and shared context.
//!
//! Per ADR-0015 §"Formatter abstraction", a formatter consumes an
//! [`EvalArtifact`] by reference and writes its rendering to a
//! caller-provided writer. Borrowing (not consuming) the artifact
//! is what makes multi-emit free at the kernel level — eval pays
//! once, per-formatter render pays per `--emit`.
//!
//! Adding a new formatter is a one-file change: drop a new module here,
//! implement [`Formatter`] for its zero-sized struct, add it to the
//! [`registry`] slice, and add its name to the [`FormatName`] enum.
//! Nothing else moves.

pub(crate) mod json;
pub(crate) mod text;

use std::io;

use vernier_core::lrp::LrpReport;
use vernier_core::{ParityMode, Summary};

use crate::cli::IouTypeArg;
use crate::error::CliError;

/// Discriminated output of one eval pass.
///
/// Per ADR-0043 the `olrp` metric ships an `LrpReport` instead of
/// the existing `Summary` shape; both flow through the formatter
/// trait so adding a new metric is a one-arm change here plus the
/// per-format render branch.
pub(crate) enum EvalArtifact<'a> {
    /// Standard COCO-style AP / AR summary.
    Ap(&'a Summary),
    /// LRP / oLRP report per ADR-0043.
    Lrp(&'a LrpReport),
}

/// Per-formatter rendering context. Carries the eval-time configuration
/// the formatter may want to surface in its output (notably the JSON
/// formatter's top-level `iou_type` / `parity_mode` / `max_dets` /
/// `use_cats` fields). Format-specific knobs live on the formatter
/// struct itself, not here.
pub(crate) struct FormatContext<'a> {
    /// IoU kind that produced the summary.
    pub(crate) iou_type: IouTypeArg,
    /// Parity mode that produced the summary (after the
    /// `aligned`→`strict` collapse the CLI applies).
    pub(crate) parity_mode: ParityMode,
    /// Resolved `max_dets` ladder (kernel-canonical default applied).
    pub(crate) max_dets: &'a [usize],
    /// Effective `use_cats` (after combining `--use-cats` /
    /// `--no-use-cats`).
    pub(crate) use_cats: bool,
}

/// Stable formatter identifier. Each variant is the lowercase name a
/// user passes to `--emit FMT[=PATH]`. Names never change once shipped
/// (per ADR-0015 §"Formatter abstraction").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FormatName {
    /// `--emit text` — pycocotools-shaped 12-line table.
    Text,
    /// `--emit json` — schema-versioned structured document.
    Json,
}

impl FormatName {
    /// Look up a formatter by its CLI-visible name. Returns `None` for
    /// unknown names; callers emit a [`CliError::Validation`] in that
    /// case.
    pub(crate) fn lookup(name: &str) -> Option<Self> {
        registry().iter().find(|f| f.name() == name).map(|f| f.id())
    }

    /// Comma-separated list of known formatter names, for error
    /// messages.
    pub(crate) fn known_names_joined() -> String {
        let names: Vec<&str> = registry().iter().map(|f| f.name()).collect();
        names.join(", ")
    }
}

/// One render of a [`Summary`] into a writer.
///
/// Implementors are zero-sized, statically registered in [`registry`],
/// and selected by name via [`FormatName::lookup`]. Renders are
/// idempotent — calling `render` twice with the same arguments produces
/// the same bytes (per ADR-0015 §"Output determinism").
pub(crate) trait Formatter: Send + Sync {
    /// Stable identifier exposed on `--emit` and in tests.
    fn name(&self) -> &'static str;

    /// Enum-tagged identifier for [`FormatName::lookup`].
    fn id(&self) -> FormatName;

    /// Write a rendering of `artifact` (with eval-time `ctx`) into
    /// `out`. Errors propagate as [`CliError`].
    fn render(
        &self,
        artifact: &EvalArtifact<'_>,
        ctx: &FormatContext<'_>,
        out: &mut dyn io::Write,
    ) -> Result<(), CliError>;
}

/// Static registry of every formatter shipped at v0.2. Order is the
/// order users see in `--help` / error messages; do not reorder
/// without justifying it in a follow-up ADR.
pub(crate) fn registry() -> &'static [&'static dyn Formatter] {
    // The references are static singletons; the slice holds trait
    // objects so adding a new formatter is a one-line append.
    static TEXT: text::Text = text::Text;
    static JSON: json::Json = json::Json;
    static REGISTRY: &[&dyn Formatter] = &[&TEXT, &JSON];
    REGISTRY
}
