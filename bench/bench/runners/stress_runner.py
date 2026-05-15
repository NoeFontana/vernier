"""Stress-test runner — materializes a regime (or a one-axis sweep)
and reports per-regime wall time + summary. Pair with the
`bench-histogram` Cargo feature to also capture the per-call
`(g, d, wall_ns)` distribution from `match_image`."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import vernier
from vernier.instance import Bbox, Evaluator, Segm

from bench.workloads.stress_matrix import REGIMES, SWEEPS, StressRegime, materialize


def _kernel(iou_type: str) -> Bbox | Segm:
    if iou_type == "bbox":
        return Bbox()
    if iou_type == "segm":
        return Segm()
    raise ValueError(f"unknown iou_type {iou_type!r}")


def run_regime(regime: StressRegime, work_dir: Path) -> dict[str, object]:
    gt_path, dt_path = materialize(regime, work_dir)
    gt_bytes = gt_path.read_bytes()
    dt_bytes = dt_path.read_bytes()

    evaluator = Evaluator(iou=_kernel(regime.iou_type))
    t0 = time.perf_counter()
    summary = evaluator.evaluate(gt_bytes, dt_bytes)
    wall_s = time.perf_counter() - t0

    return {
        "regime": asdict(regime),
        "wall_s": wall_s,
        "ap": float(summary.stats[0]),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group()
    g.add_argument("--regime", choices=[r.name for r in REGIMES])
    g.add_argument("--axis", choices=sorted(SWEEPS.keys()))
    g.add_argument("--all", action="store_true", help="Run every named regime.")
    p.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "results" / "stress" / "results.json",
    )
    p.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Directory for materialized GT/DT JSON. Default: a fresh tmpdir.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.regime:
        targets = [r for r in REGIMES if r.name == args.regime]
    elif args.axis:
        targets = list(SWEEPS[args.axis])
    elif args.all:
        targets = list(REGIMES)
    else:
        targets = [next(r for r in REGIMES if r.name == "coco-baseline")]

    work_dir = args.work_dir or Path(tempfile.mkdtemp(prefix="vernier-stress-"))
    results: list[dict[str, object]] = []
    for r in targets:
        print(f"[stress] {r.name} (n_images={r.n_images} cats={r.n_categories} "
              f"dt={r.dt_per_image} gt={r.gt_per_image} dims={r.image_w}x{r.image_h} "
              f"iou={r.iou_type})")
        res = run_regime(r, work_dir)
        print(f"[stress]   wall_s={res['wall_s']:.3f} ap={res['ap']:.4f}")
        results.append(res)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump({"vernier_version": vernier.__version__, "results": results}, f, indent=2)
    print(f"[stress] wrote {args.output}")


if __name__ == "__main__":
    main()
