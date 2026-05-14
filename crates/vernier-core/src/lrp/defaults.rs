//! Per-kernel LRP defaults (per ADR-0044).
//!
//! The values live with the algorithm, not on the Python side, so a
//! future change moves through one file. See ADR-0044 for the canonical
//! defense.

use std::sync::OnceLock;

use crate::evaluate::KernelKind;

/// Resolve the per-kernel TP IoU/OKS threshold (per ADR-0044).
///
/// | Kernel    | threshold |
/// |-----------|-----------|
/// | bbox      | `0.5`     |
/// | segm      | `0.5`     |
/// | boundary  | `0.5`     |
/// | keypoints | `0.5`     |
///
/// The single value across kernels is intentional and defended on the
/// ADR's grounds: `0.5` is the operating point Oksuz TPAMI 2021 anchors
/// LRP on, and the reading transfers across every kernel's similarity
/// scale at the operating-point level. Tighter thresholds (`0.7` etc.)
/// do not transfer — users override per call when they need them.
#[must_use]
pub fn tp_threshold_for(kernel: KernelKind) -> f64 {
    match kernel {
        KernelKind::Bbox | KernelKind::Segm | KernelKind::Boundary | KernelKind::Keypoints => 0.5,
    }
}

/// Canonical 101-point tau grid (`0.00, 0.01, ..., 1.00`).
///
/// The same slice is reused across calls — the caller borrows it.
/// Initialised once via [`OnceLock`]; every consumer sees the same
/// allocation.
#[must_use]
pub fn default_tau_grid() -> &'static [f64] {
    static GRID: OnceLock<Vec<f64>> = OnceLock::new();
    GRID.get_or_init(|| (0..=100).map(|i| f64::from(i) / 100.0).collect())
        .as_slice()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_tau_grid_is_101_points() {
        let grid = default_tau_grid();
        assert_eq!(grid.len(), 101);
        assert!((grid[0]).abs() < 1e-12);
        assert!((grid[100] - 1.0).abs() < 1e-12);
        assert!((grid[50] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn tp_threshold_for_every_kernel_is_half() {
        for kind in [
            KernelKind::Bbox,
            KernelKind::Segm,
            KernelKind::Boundary,
            KernelKind::Keypoints,
        ] {
            assert!(
                (tp_threshold_for(kind) - 0.5).abs() < 1e-12,
                "kernel {kind:?}"
            );
        }
    }
}
