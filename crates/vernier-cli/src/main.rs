//! `vernier` binary entry point.
//!
//! Per ADR-0015 §"Crate layout", the binary's job is parse → dispatch
//! → exit-code-mapping. All eval logic lives in `vernier-core`; all
//! output formatting lives under `crate::format`. This file is
//! intentionally short.
//!
//! Per ADR-0015 §"Stdout / stderr split": stdout carries the summary
//! bytes exclusively; stderr carries diagnostics. The `--quiet` flag
//! suppresses stderr only.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

use clap::Parser;

mod cli;
mod commands;
mod error;
mod format;

use crate::cli::{Cli, Command};

fn main() {
    // Clap exits the process with code 2 on parse errors by default,
    // which is exactly the contract ADR-0015 §"Exit codes" pins.
    let cli = Cli::parse();
    match &cli.command {
        Command::Eval(args) => commands::eval::run_or_exit(args),
    }
}
