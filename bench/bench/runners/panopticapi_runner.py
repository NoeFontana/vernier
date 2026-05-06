"""panopticapi runner — invoked as a subprocess in
``bench/envs/panopticapi`` (ADR-0033 §B1).

Wraps ``panopticapi.evaluation.pq_compute_single_core(proc_id=0, ...)``,
the strict-mode reference oracle per ADR-0025 §"Strict-mode parity
claim". Critically does **not** call ``pq_compute`` — the multi-core
variant has no ``num_proc`` parameter and would otherwise pin the
benchmark to the host's CPU count, defeating the cross-host
reproducibility contract.

Lifted from :func:`tests/python/parity_panoptic/harness.py:_oracle_snapshot`
and :func:`pq_stat_to_snapshot` (the harness's projection helper).
The bench runner adds stage-timer brackets around each phase and
emits a :class:`bench.harness.parity.PanopticSnapshot` JSON instead
of the harness's frozen-dataclass return.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage

from bench.harness.parity import PanopticSnapshot
from bench.harness.timing import StageTable
from bench.runners._protocol import parse_panoptic_runner_args, write_panoptic_outputs

# Mirrors ``ORACLE_COMMIT_SHA`` in
# ``crates/vernier-panoptic/src/parity.rs``. The bench env's
# ``pyproject.toml`` pins this same SHA; the tripwire test in
# ``bench/tests/test_panopticapi_env_pin.py`` parses the parity.rs
# constant and asserts both files agree.
_ORACLE_SHA_PREFIX = "7bb4655548f9"


def _decode_panoptic_png_to_uint32(path: Path) -> np.ndarray:
    """Decode a panoptic PNG to a uint32 label map via PIL + rgb2id.

    Mirrors panopticapi's eval-side decode (evaluation.py:86-89).
    Pillow is the decoder of record (the env pins
    ``Pillow==12.2.0``); RGB → ``r + 256*g + 256²*b``.
    """
    rgb = np.array(PILImage.open(path), dtype=np.uint32)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"non-RGB panoptic PNG: {path}")
    return rgb[:, :, 0] + 256 * rgb[:, :, 1] + 256 * 256 * rgb[:, :, 2]


def _pq_stat_to_snapshot(pq_stat: Any, cats_dict: dict[int, dict[str, Any]]) -> PanopticSnapshot:
    """Project a panopticapi ``PQStat`` into a ``PanopticSnapshot``.

    Same projection as
    :func:`tests/python/parity_panoptic/harness.py:pq_stat_to_snapshot`,
    written against ``bench.harness.parity.PanopticSnapshot`` (the
    bench-side Pydantic model, not the harness frozen dataclass).
    The three ``pq_average`` calls are independent — the All bucket
    carries the full ``per_class`` row map.
    """
    all_d, per_class = pq_stat.pq_average(cats_dict, isthing=None)
    things_d, _ = pq_stat.pq_average(cats_dict, isthing=True)
    stuff_d, _ = pq_stat.pq_average(cats_dict, isthing=False)
    return PanopticSnapshot(
        pq=float(all_d["pq"]),
        sq=float(all_d["sq"]),
        rq=float(all_d["rq"]),
        n=int(all_d["n"]),
        pq_things=float(things_d["pq"]),
        sq_things=float(things_d["sq"]),
        rq_things=float(things_d["rq"]),
        n_things=int(things_d["n"]),
        pq_stuff=float(stuff_d["pq"]),
        sq_stuff=float(stuff_d["sq"]),
        rq_stuff=float(stuff_d["rq"]),
        n_stuff=int(stuff_d["n"]),
        per_class={str(int(k)): {"pq": float(v["pq"]), "sq": float(v["sq"]), "rq": float(v["rq"])} for k, v in per_class.items()},
    )


def _per_class_table(snap: PanopticSnapshot) -> np.ndarray:
    """Build the per-class N×3 ``[pq, sq, rq]`` table.

    Float64 cast as uint64 for the on-disk artifact (the
    ``PanopticSnapshot`` JSON carries the labelled metric for human
    readers; the npy is the bit-stable carrier — uint64 view of f64
    avoids float-text rounding under ``np.save``).
    """
    rows: list[tuple[int, float, float, float]] = sorted(
        ((int(k), v["pq"], v["sq"], v["rq"]) for k, v in snap.per_class.items()),
        key=lambda r: r[0],
    )
    if not rows:
        return np.zeros((0, 3), dtype=np.uint64)
    arr = np.array([[r[1], r[2], r[3]] for r in rows], dtype=np.float64)
    return arr.view(np.uint64).reshape(arr.shape)


def main() -> int:
    args = parse_panoptic_runner_args()
    stages = StageTable()

    # Lazy import: panopticapi is in the panopticapi env's deps. The
    # tripwire test asserts the env pin matches ORACLE_COMMIT_SHA.
    from panopticapi.evaluation import pq_compute_single_core  # type: ignore[import-not-found]

    with stages.stage("load"):
        with args.gt_json.open() as f:
            gt = json.load(f)
        with args.dt_json.open() as f:
            dt = json.load(f)
        with args.categories_json.open() as f:
            categories = json.load(f)
        cats_dict = {int(c["id"]): dict(c) for c in categories}

    with stages.stage("decode_pngs"):
        pred_by_image = {a["image_id"]: a for a in dt["annotations"]}
        annotation_set: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for gt_ann in gt["annotations"]:
            if gt_ann["image_id"] in pred_by_image:
                annotation_set.append((gt_ann, pred_by_image[gt_ann["image_id"]]))

    with stages.stage("pq_compute"):
        # proc_id=0 per ADR-0025: the single-core path is the strict-
        # parity oracle. The multi-core variant has no num_proc knob
        # and pins the benchmark to the host CPU count.
        pq_stat = pq_compute_single_core(
            0,
            annotation_set,
            str(args.gt_png_dir),
            str(args.dt_png_dir),
            cats_dict,
        )

    with stages.stage("aggregate"):
        snap = _pq_stat_to_snapshot(pq_stat, cats_dict)
        per_class_array = _per_class_table(snap)
        snap_json = snap.model_dump_json().encode()

    stages.record("total", stages.total_so_far_ns())

    write_panoptic_outputs(
        args=args,
        impl="panopticapi",
        impl_version=_ORACLE_SHA_PREFIX,
        stages=stages.to_dict(),
        snapshot_json_bytes=snap_json,
        per_class_array=per_class_array,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
