//! Op-exact ports of the two strict oracles.
//!
//! These modules exist to be *wrong* in exactly the ways their oracles
//! are wrong. They are not a second implementation of oriented-box IoU —
//! [`crate::kernel`] is that — and no cleanup, tolerance tuning or
//! numerical improvement belongs here. When a replica looks like it has
//! a bug, the first question is whether the oracle has it too, and the
//! answer is usually yes.
//!
//! Each module names its upstream file, commit and function per step.
//! `tests/python/parity_obb/oracle/VENDORING.md` records provenance and
//! the licensing position for each.
//!
//! # No FMA
//!
//! Neither replica may use `f32::mul_add` / `f64::mul_add`. ADR-0063
//! specifies `clippy::disallowed_methods` for this; a `clippy.toml`
//! would apply workspace-wide and the bbox golden tests legitimately use
//! `mul_add`, so the invariant is enforced by `tests::replicas_are_fma_free`
//! instead — a source scan, which is both narrower and impossible to
//! silence with an `allow` attribute.

pub mod d2;
pub mod dk;

#[cfg(test)]
mod tests {
    /// The replica sources, read at compile time.
    const D2_SRC: &str = include_str!("d2.rs");
    const DK_SRC: &str = include_str!("dk.rs");

    #[test]
    fn replicas_are_fma_free() {
        for (name, src) in [("d2.rs", D2_SRC), ("dk.rs", DK_SRC)] {
            for (lineno, line) in src.lines().enumerate() {
                // Skip the prose that explains the rule.
                let code = line.trim_start();
                if code.starts_with("//") {
                    continue;
                }
                assert!(
                    !code.contains("mul_add"),
                    "{name}:{} uses mul_add; replicas must stay unfused",
                    lineno + 1
                );
            }
        }
    }

    #[test]
    fn replicas_cite_their_upstream() {
        assert!(D2_SRC.contains("box_iou_rotated_utils.h"));
        assert!(DK_SRC.contains("polyiou.cpp"));
        assert!(DK_SRC.contains("dota_evaluation_task1.py"));
    }
}
