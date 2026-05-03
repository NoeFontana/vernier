//! TIDE error decomposition (Bolya et al., 2020).
//!
//! Decomposes the headline ΔmAP into six bins — Cls / Loc / Both / Dupe /
//! Bkg / Missed — by running corrected accumulations per bin and
//! subtracting from baseline. The bin definitions and the algorithmic
//! contract live in the ADRs that govern this module:
//!
//! - **ADR-0021** — TIDE numpy oracle as the correctness model. The Rust
//!   implementation's correctness contract is `|delta_rust − delta_oracle|
//!   < 1e-9` per bin per fixture; this module is correct iff it agrees
//!   with the oracle.
//! - **ADR-0022** — Per-kernel `(t_f, t_b)` thresholds. Defaults live
//!   alongside the algorithm rather than as Python-side constants; the
//!   resolved thresholds are recorded on every [`report::TideReport`].
//! - **ADR-0023** — Cross-class IoU as an orchestrator-level side pass.
//!   The matching engine (ADR-0005) is left untouched; cross-class
//!   overlaps are gathered by a separate pass through the same
//!   [`crate::similarity::Similarity`] kernel and stored in
//!   [`crate::tables::CrossClassIous`].
//! - **ADR-0024** — TIDE on keypoints (OKS) is deferred to a later
//!   release; this module does not ship a keypoints branch.
//!
//! ## Module layout
//!
//! - [`bins`] — the [`bins::TideErrorBin`] enum naming the six bins.
//! - [`cross_class`] — the side-pass driver
//!   [`cross_class::compute_cross_class_ious`] that populates the
//!   [`crate::tables::CrossClassIous`] storage from a dataset and
//!   detection list. The storage type itself lives next to
//!   [`crate::tables::RetainedIous`] in [`crate::tables`].
//! - [`report`] — [`report::TideReport`] (the per-bin ΔmAP output) and
//!   [`report::TideConfig`] (the resolved thresholds + kernel marker
//!   recorded alongside, per ADR-0022).
//! - [`rewrite`] — placeholder for the Week-2 cell-rewrite layer; module
//!   doc only at this stage.
//!
//! ## Status
//!
//! Week 1 scaffolding only. Bin assignment, the eight-pass orchestration,
//! and the Python surface land in subsequent PRs.

pub mod bins;
pub mod cross_class;
pub mod report;
pub mod rewrite;

pub use bins::TideErrorBin;
pub use cross_class::compute_cross_class_ious;
pub use report::{TideConfig, TideReport};
