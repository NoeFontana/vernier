//! Cell-rewrite layer for per-bin corrections — Week 2.
//!
//! Each TIDE bin's correction rewrites the per-image cells the matching
//! engine produced (moving DTs across `(category, image)` cells,
//! flipping `gt_ignore`, synthesizing matches) and re-runs
//! [`mod@crate::accumulate`] + [`mod@crate::summarize`]. Eight
//! accumulate passes per call, per ADR-0021.
//!
//! Intentionally empty in this PR: the implementation lands in the
//! Week-2 follow-up that introduces the bin-assignment logic and the
//! orchestrator that drives it. Splitting it out gives reviewers a
//! single PR to read for the most semantically delicate code in the
//! release.
