//! Verb-level dispatch. `eval` is at v0.2 (ADR-0015); `aggregate` is
//! the slice-and-aggregate fan-in verb (ADR-0046).

pub(crate) mod aggregate;
pub(crate) mod eval;
pub(crate) mod eval_partitioned;
