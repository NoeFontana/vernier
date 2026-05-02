"""Three-tier parity coupling for the bench harness (ADR-0017 §D2).

After every ``(workload, iou)`` cell finishes, every runner has written
its precision tensor to disk. This module loads those tensors and
asserts the cross-impl invariants documented in the parity ADRs:

- **strict** — vernier reproduces pycocotools bit-exactly
  (``np.array_equal``).
- **aligned** — vernier matches faster-coco-eval within a small absolute
  tolerance; faster-coco-eval has documented float-order quirks.
- **boundary** — vernier matches the boundary-iou-api oracle within
  ``BOUNDARY_PARITY_EPS`` (the magnitude pycocotools' own ``np.testing``
  uses for IoU comparisons; mirrored from
  ``crates/vernier-core/src/boundary_parity.rs``).

A cell that fails any tier writes ``divergence_report.json`` next to
its impl results. Dev mode keeps going (the orchestrator only logs);
release mode (M5) will turn this into a hard fail.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from bench.harness.schema import IouType

# Mirror of ``crates/vernier-core/src/parity.rs::PARITY_EPS`` —
# ``np.spacing(1)`` for f64. A unit test in that crate checks the
# numeric value; mirroring here avoids a vernier import in the harness.
PARITY_EPS: float = 2.220446049250313e-16

# Mirror of ``crates/vernier-core/src/boundary_parity.rs::BOUNDARY_PARITY_EPS``.
BOUNDARY_PARITY_EPS: float = 1e-9

# Aligned tier: 4 ULP. Faster-coco-eval reorders some accumulations so
# the tensor differs by a handful of ULP from pycocotools, even on
# fixtures where the strict tier is bit-equal.
ALIGNED_ATOL: float = 4.0 * PARITY_EPS

Tier = Literal["strict", "aligned", "boundary"]


class Divergence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: tuple[int, ...]
    value_a: float
    value_b: float
    abs_diff: float


class TierResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tier: Tier
    impl_a: str
    impl_b: str
    atol: float
    passed: bool
    divergent_count: int
    first_divergence: Divergence | None
    tensor_sha256_a: str
    tensor_sha256_b: str


class CellParityReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    workload_id: str
    iou_type: IouType
    tiers: list[TierResult]

    @property
    def passed(self) -> bool:
        return all(t.passed for t in self.tiers)


# Each iou cell defines the tier comparisons that apply when the named
# impls have produced tensors. Read as: "if both impls ran this cell,
# compare them at this tier with this tolerance."
_TIER_PAIRS: dict[IouType, tuple[tuple[Tier, str, str, float], ...]] = {
    "bbox": (
        ("strict", "vernier", "pycocotools", 0.0),
        ("aligned", "vernier", "faster-coco-eval", ALIGNED_ATOL),
    ),
    "segm": (
        ("strict", "vernier", "pycocotools", 0.0),
        ("aligned", "vernier", "faster-coco-eval", ALIGNED_ATOL),
    ),
    "keypoints": (
        ("strict", "vernier", "pycocotools", 0.0),
        ("aligned", "vernier", "faster-coco-eval", ALIGNED_ATOL),
    ),
    "boundary": (("boundary", "vernier", "boundary-iou-api", BOUNDARY_PARITY_EPS),),
}


def _compare_pair(
    *,
    tier: Tier,
    impl_a: str,
    impl_b: str,
    tensor_a: np.ndarray,
    tensor_b: np.ndarray,
    sha_a: str,
    sha_b: str,
    atol: float,
) -> TierResult:
    if tensor_a.shape != tensor_b.shape:
        raise ValueError(
            f"shape mismatch comparing {impl_a} vs {impl_b}: {tensor_a.shape} vs {tensor_b.shape}"
        )

    diff = np.abs(tensor_a - tensor_b)
    # Strict tier: bit-equality, not "diff <= 0" (NaNs in either tensor
    # would make a finite-diff check pass spuriously).
    if tier == "strict":
        divergent_mask = ~(tensor_a == tensor_b)
    else:
        divergent_mask = diff > atol

    divergent_count = int(divergent_mask.sum())
    first_divergence: Divergence | None = None
    if divergent_count > 0:
        first_idx = tuple(int(i) for i in np.argwhere(divergent_mask)[0])
        first_divergence = Divergence(
            index=first_idx,
            value_a=float(tensor_a[first_idx]),
            value_b=float(tensor_b[first_idx]),
            abs_diff=float(diff[first_idx]),
        )

    return TierResult(
        tier=tier,
        impl_a=impl_a,
        impl_b=impl_b,
        atol=atol,
        passed=divergent_count == 0,
        divergent_count=divergent_count,
        first_divergence=first_divergence,
        tensor_sha256_a=sha_a[:12],
        tensor_sha256_b=sha_b[:12],
    )


def compare_cell(
    *,
    workload_id: str,
    iou_type: IouType,
    impl_tensors: dict[str, np.ndarray],
    impl_sha256: dict[str, str],
) -> CellParityReport:
    """Run every applicable tier for ``iou_type`` over the impls present
    in ``impl_tensors``. Pairs whose impls aren't both present are
    silently skipped (e.g., bbox cell with only vernier + pycocotools
    skips the aligned tier)."""
    tiers: list[TierResult] = []
    for tier, impl_a, impl_b, atol in _TIER_PAIRS[iou_type]:
        if impl_a not in impl_tensors or impl_b not in impl_tensors:
            continue
        tiers.append(
            _compare_pair(
                tier=tier,
                impl_a=impl_a,
                impl_b=impl_b,
                tensor_a=impl_tensors[impl_a],
                tensor_b=impl_tensors[impl_b],
                sha_a=impl_sha256[impl_a],
                sha_b=impl_sha256[impl_b],
                atol=atol,
            )
        )
    return CellParityReport(workload_id=workload_id, iou_type=iou_type, tiers=tiers)


def write_report(report: CellParityReport, out_dir: Path) -> Path:
    """Persist ``divergence_report.json`` next to the cell's impl results."""
    path = out_dir / "divergence_report.json"
    path.write_text(report.model_dump_json(indent=2))
    return path
