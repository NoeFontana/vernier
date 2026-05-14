//! `vernier aggregate` JSON envelope (ADR-0046 §"`vernier aggregate`").
//!
//! Lives in its own version namespace — `aggregate_version: "1"` —
//! independent of the eval JSON's `version` axis. The two outputs are
//! emitted by different verbs and have different consumers (eval is
//! one-result, aggregate is one-comparison-table); coupling their
//! version axes would force lockstep bumps that the lifecycles do not
//! warrant.
//!
//! Field order is the wire-stable order (serde follows struct
//! declaration order — the existing CLI determinism contract carries
//! through verbatim). Atomic file writes use [`super::super::commands::eval::write_atomic`].

use std::io;

use serde::Serialize;
use serde_json::{Map, Value};

use crate::error::CliError;

/// Pinned at `"1"`. Bumping requires the same ADR-bar as `version`
/// bumps on the eval JSON.
pub(crate) const AGGREGATE_VERSION: &str = "1";

/// Top-level `vernier aggregate` document. Field order is wire-stable.
#[derive(Debug, Serialize)]
pub(crate) struct AggregateV1<'a> {
    /// Schema version pin (`"1"`). Surfaces first so a downstream
    /// tool can sniff compatibility without parsing the rest.
    pub(crate) aggregate_version: &'a str,
    /// `--baseline` value passed on the command line, or `null` when
    /// rPC columns are not emitted.
    pub(crate) baseline: Option<&'a str>,
    /// Metric column names included in this document, in the order
    /// they appear inside each row's `metrics` map.
    pub(crate) metrics: Vec<&'a str>,
    /// One row per `(axis, value)` cell, in the canonical (axis
    /// ascending, value ascending, `__unassigned__` last) order.
    pub(crate) rows: Vec<AggregateRow<'a>>,
}

/// Per-`(axis, value)` aggregated row.
#[derive(Debug, Serialize)]
pub(crate) struct AggregateRow<'a> {
    pub(crate) axis: &'a str,
    pub(crate) value: &'a str,
    /// Number of results that joined to this `(axis, value)` cell.
    pub(crate) n_runs: u64,
    /// `{metric_name -> mean across runs}`, with `<metric>__rpc`
    /// columns appended when a baseline is set. Backed by a
    /// `serde_json::Map` so insertion order is preserved on the wire
    /// (the order declared by the producer is the order the consumer
    /// sees).
    pub(crate) metrics: Map<String, Value>,
}

/// Render an [`AggregateV1`] to a writer. The document is followed by
/// a single newline (mirrors the eval JSON's trailing-newline rule).
pub(crate) fn render(doc: &AggregateV1<'_>, out: &mut dyn io::Write) -> Result<(), CliError> {
    serde_json::to_writer(&mut *out, doc)?;
    writeln!(out)?;
    Ok(())
}
