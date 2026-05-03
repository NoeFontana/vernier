//! TIDE error bins.
//!
//! The six bins the TIDE paper (Bolya et al., 2020) decomposes ΔmAP
//! into. Bin assignment lives in the Week-2 [`super::rewrite`] layer;
//! this module names the bins only.
//!
//! Bin semantics, summarized for reference (the ADR-0021 oracle is the
//! authoritative definition):
//!
//! - **Cls** — the detection's best-overlapping GT (across *any* class,
//!   per ADR-0023) lies in another class with IoU `>= t_f`. A
//!   classification mistake on an otherwise well-localized box.
//! - **Loc** — the detection's best same-class GT has IoU in
//!   `[t_b, t_f)`. Localization is off but the class is right.
//! - **Both** — same shape as Cls but with cross-class IoU in
//!   `[t_b, t_f)`: the detection misses on both class and box.
//! - **Dupe** — the detection would be a TP except a higher-scoring,
//!   same-class detection already claimed the GT.
//! - **Bkg** — the detection's best IoU against any GT is `< t_b`. Pure
//!   background false positive.
//! - **Missed** — a GT that no detection claims at IoU `>= t_f` and that
//!   none of the FP bins above explain. Counted on the GT side; the
//!   four FP bins above are counted on the DT side.

/// One of the six TIDE error bins.
///
/// Carries no payload: bin assignment is the rewrite layer's job, and
/// the rewrite output is per-cell mutations rather than a per-detection
/// label. The bin set is closed by the paper; new bins (if any) require
/// an ADR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TideErrorBin {
    /// Classification error — wrong-class GT overlaps at IoU `>= t_f`.
    Cls,
    /// Localization error — same-class GT overlaps at IoU
    /// `∈ [t_b, t_f)`.
    Loc,
    /// Combined classification + localization error — wrong-class GT
    /// overlaps at IoU `∈ [t_b, t_f)`.
    Both,
    /// Duplicate detection — a higher-scoring same-class DT already
    /// matched the GT.
    Dupe,
    /// Background false positive — best IoU against any GT is `< t_b`.
    Bkg,
    /// Missed ground truth — no detection claims this GT at IoU
    /// `>= t_f`. Counted on the GT side, in contrast to the four
    /// detection-side FP bins.
    Missed,
}
