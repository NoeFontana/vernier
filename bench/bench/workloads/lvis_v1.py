"""LVIS v1 val workloads — perfect-DT smoke + bbox/segm-jittered cell.

Per ADR-0026 LVIS shares the AP-fold core with COCO but layers on
federated semantics (``pos/neg/not_exhaustive`` per-image category
filters, plus the ``dt_ignore`` extension on rare classes). The
``vernier-core`` evaluator already supports this through
``CategoryFilter``; the bench-side adapter is just a path resolver
against the canonical :mod:`lvis_val_cache` package.

The cache is sha256-pinned and never committed (LVIS terms of use,
mirroring the COCO val2017 no-bytes principle). ``ensure_gt`` and
``ensure_perfect_dts`` populate ``$VERNIER_LVIS_CACHE`` (or the default
``<repo>/.cache/lvis-val/``) on first call; subsequent calls are
hash-only verifications. Honors ``VERNIER_LVIS_GT_PATH`` /
``VERNIER_LVIS_DT_PATH`` / ``VERNIER_LVIS_DT_SEGM_PATH`` so users who
already populated the parity cache (``just test-parity-lvis-val``)
don't pay the download twice.

Two workloads:

- ``lvis_v1_val_perfect`` — GT-as-DT smoke. Strict-tier oracle is
  ``lvis-api`` (vendored at ``tests/python/parity_lvis/oracle/``);
  the bench harness today does not ship a dedicated lvis-api runner,
  so the comparator surface remains the existing strict + aligned
  tier pairs over (vernier, pycocotools, faster-coco-eval). Divergence
  is expected and informational on this cell because pycocotools is
  not LVIS-aware; vernier-vs-vernier across reps remains bit-equal.
- ``lvis_v1_val_jittered_seed<N>`` — bbox + segm jitter through the
  shared :mod:`jittered_predictions` generator. The jitter parameters
  are unchanged from the COCO workload, so seed identity is a property
  of (workload, seed), not (paradigm, seed).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from lvis_val_cache import ensure_gt, ensure_perfect_dts

PERFECT_WORKLOAD_ID = "lvis_v1_val_perfect"

_LVIS_JITTERED_RE = re.compile(r"^lvis_v1_val_jittered_seed(\d+)$")


def gt_path() -> Path:
    """Return a verified path to the LVIS v1 val GT JSON.

    Honors ``VERNIER_LVIS_GT_PATH`` first (the parity-cache convention)
    and falls back to populating the canonical cache via
    :func:`lvis_val_cache.ensure_gt`. Sha256 verification happens inside
    :func:`ensure_gt`.
    """
    env_override = os.environ.get("VERNIER_LVIS_GT_PATH")
    if env_override:
        candidate = Path(env_override)
        if candidate.exists():
            return candidate
    return ensure_gt()


def perfect_dt_segm_path() -> Path:
    """Return the perfect-segm DT for the LVIS GT.

    Honors ``VERNIER_LVIS_DT_SEGM_PATH``; otherwise synthesizes via
    :func:`lvis_val_cache.ensure_perfect_dts`. The synth tool is
    ``tools/make-perfect-dt.py`` (subprocess from the cache module),
    same as the COCO path.
    """
    env_override = os.environ.get("VERNIER_LVIS_DT_SEGM_PATH")
    if env_override:
        candidate = Path(env_override)
        if candidate.exists():
            return candidate
    _, segm_dt = ensure_perfect_dts()
    return segm_dt


def jittered_workload_id(seed: int) -> str:
    return f"lvis_v1_val_jittered_seed{seed}"


def parse_jittered_seed(workload_name: str) -> int | None:
    """Parse seed from ``lvis_v1_val_jittered_seed<N>``; return ``None``
    if the name doesn't match. Lets the registry stay one large
    if-tree without spreading the regex across modules.
    """
    m = _LVIS_JITTERED_RE.match(workload_name)
    return int(m.group(1)) if m else None


__all__ = [
    "PERFECT_WORKLOAD_ID",
    "gt_path",
    "jittered_workload_id",
    "parse_jittered_seed",
    "perfect_dt_segm_path",
]
