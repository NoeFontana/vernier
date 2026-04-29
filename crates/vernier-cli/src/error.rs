//! Typed CLI errors.
//!
//! Per ADR-0015 §"Exit codes", the CLI uses two non-zero exit codes:
//! `1` for runtime/eval/IO failures, `2` for argument parse / validation
//! failures. The split lets shell pipelines distinguish "the binary was
//! misused" from "the eval refused".
//!
//! [`CliError`] is the single error type that flows through every code
//! path in `vernier-cli`. The `Validation` variant exits with code `2`;
//! every other variant exits with code `1`. Clap's own parse errors are
//! handled before we ever construct a [`CliError`] (clap exits the
//! process with code `2` on parse failure by default), so the
//! `Validation` variant is reserved for our own post-parse cross-flag
//! checks (e.g. `--dilation-ratio` rejected on `--iou-type bbox`).

use std::io;
use std::path::PathBuf;

use thiserror::Error;
use vernier_core::EvalError;

/// Exit code for argument parse / validation failures.
pub(crate) const EXIT_VALIDATION: i32 = 2;
/// Exit code for runtime / eval / IO failures.
pub(crate) const EXIT_RUNTIME: i32 = 1;

/// Unified error type for the CLI.
#[derive(Debug, Error)]
pub(crate) enum CliError {
    /// Cross-flag validation failed at the CLI layer (e.g. a flag was
    /// passed in combination with an `--iou-type` it does not apply
    /// to). Surfaces with [`EXIT_VALIDATION`] so shell scripts can tell
    /// it apart from runtime failures.
    #[error("invalid arguments: {0}")]
    Validation(String),

    /// A required input file (`--gt`, `--dt`, `--sigmas`) could not be
    /// read.
    #[error("failed to read {path}: {source}")]
    InputRead {
        /// Path the read was attempted at.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: io::Error,
    },

    /// An emit destination (`--emit FMT=PATH`) could not be written.
    #[error("failed to write {path}: {source}")]
    OutputWrite {
        /// Path the write was attempted at.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: io::Error,
    },

    /// Generic stdio write failure (e.g. broken pipe on stdout).
    #[error("io: {0}")]
    Io(#[from] io::Error),

    /// Sigmas-file JSON could not be parsed into the expected
    /// `{category_id: [sigma, ...]}` shape.
    #[error("invalid sigmas file: {0}")]
    InvalidSigmas(String),

    /// JSON serialization of an emit document failed.
    #[error("json serialization: {0}")]
    JsonSerialize(#[from] serde_json::Error),

    /// `vernier-core` returned a typed evaluation error.
    #[error("eval: {0}")]
    Eval(#[from] EvalError),
}

impl CliError {
    /// Map a [`CliError`] to its exit code per ADR-0015 §"Exit codes".
    pub(crate) fn exit_code(&self) -> i32 {
        match self {
            Self::Validation(_) => EXIT_VALIDATION,
            _ => EXIT_RUNTIME,
        }
    }
}
