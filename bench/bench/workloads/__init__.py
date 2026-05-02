"""Workload registry. Each workload resolves to (gt_path, dt_path) plus
the set of IoU types it can serve.

Workload identifiers:

- ``smoke`` — local fixture, all iou types.
- ``coco_val2017_jittered_seed<N>`` — COCO val2017 GT (sha256-pinned)
  with deterministic Gaussian-jittered DT for seed ``<N>``. Bbox-only
  for now.
- ``synthetic:k=v,k=v[,...]`` — parametric stress-test. Required keys:
  ``n_images``, ``seed``. Optional: ``n_categories`` (default 80),
  ``dt_per_image`` (default 30), ``gt_per_image`` (default 10) — chosen
  to match the ADR's release-mode ladder defaults.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from bench.harness.schema import IouType
from bench.workloads import coco_val2017, jittered_predictions, smoke, synthetic


@dataclass(frozen=True)
class Workload:
    workload_id: str
    gt_path: Path
    dt_path: Path
    supported_iou_types: frozenset[IouType]


_COCO_JITTERED_RE = re.compile(r"^coco_val2017_jittered_seed(\d+)$")
_SYNTHETIC_PREFIX = "synthetic:"
_SYNTHETIC_DEFAULTS: dict[str, int] = {
    "n_categories": 80,
    "dt_per_image": 30,
    "gt_per_image": 10,
}
_SYNTHETIC_REQUIRED: frozenset[str] = frozenset({"n_images", "seed"})
_SYNTHETIC_ALLOWED: frozenset[str] = _SYNTHETIC_REQUIRED | frozenset(_SYNTHETIC_DEFAULTS)


def _parse_synthetic_args(spec: str) -> dict[str, int]:
    if not spec:
        raise ValueError("synthetic: requires at least n_images=... and seed=...")
    out: dict[str, int] = {}
    for token in spec.split(","):
        if "=" not in token:
            raise ValueError(f"synthetic: param {token!r} is not k=v")
        key, value = token.split("=", 1)
        key = key.strip()
        if key not in _SYNTHETIC_ALLOWED:
            raise ValueError(
                f"synthetic: unknown param {key!r}; allowed: {sorted(_SYNTHETIC_ALLOWED)}"
            )
        try:
            out[key] = int(value)
        except ValueError as e:
            raise ValueError(f"synthetic: param {key}={value!r} must be int") from e
    missing = _SYNTHETIC_REQUIRED - out.keys()
    if missing:
        raise ValueError(f"synthetic: missing required param(s): {sorted(missing)}")
    return {**_SYNTHETIC_DEFAULTS, **out}


def resolve(workload_name: str, repo_root: Path) -> Workload:
    if workload_name == "smoke":
        gt, dt = smoke.paths(repo_root)
        return Workload(
            workload_id="smoke_perfect_match_segm",
            gt_path=gt,
            dt_path=dt,
            supported_iou_types=frozenset({"bbox", "segm", "boundary"}),
        )

    if m := _COCO_JITTERED_RE.match(workload_name):
        seed = int(m.group(1))
        gt = coco_val2017.gt_path()
        dt = jittered_predictions.dt_path(gt_path=gt, seed=seed)
        return Workload(
            workload_id=jittered_predictions.workload_id(seed),
            gt_path=gt,
            dt_path=dt,
            supported_iou_types=frozenset({"bbox"}),
        )

    if workload_name.startswith(_SYNTHETIC_PREFIX):
        params = _parse_synthetic_args(workload_name.removeprefix(_SYNTHETIC_PREFIX))
        gt, dt = synthetic.make_workload(**params)
        return Workload(
            workload_id=synthetic.workload_id(**params),
            gt_path=gt,
            dt_path=dt,
            supported_iou_types=frozenset({"bbox"}),
        )

    raise ValueError(
        f"unknown workload {workload_name!r}; "
        f"known: 'smoke', 'coco_val2017_jittered_seed<N>', 'synthetic:n_images=...,seed=...'"
    )
