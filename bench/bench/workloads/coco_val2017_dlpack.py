"""DLPack-vs-JSON ingest workload (ADR-0030).

``coco_val2017_dlpack_vs_json`` measures the wall-time delta between
the JSON-bytes ingest path and the array-form ``Detections`` path
(numpy / DLPack buffers, TypedDict per ``vernier._array_types``).

The two configs run through the *same* evaluator (``StreamingEvaluator``
instantiated with identical kwargs); the runner enumerates DT records
into either ``bytes`` or ``Detections`` rows and dispatches each path.
The comparator asserts byte-equal ``Summary.stats`` between them per
the ADR-0030 parity contract — the JSON path is the legacy oracle that
the array path mirrors. Wall-time delta + RSS curves are informational
artifacts.
"""

from __future__ import annotations

from bench.workloads import StreamingWorkload, coco_val2017


def dlpack_vs_json() -> StreamingWorkload:
    """DLPack-vs-JSON cell over COCO val2017 jittered seed-0 DT.

    Same fixture as the streaming throughput cell so a single ``uv sync``
    populates both. The runner reuses
    ``loadres_to_detections`` (lifted from
    ``tests/python/parity/conftest.py``) to translate JSON DT records
    into the per-image array form.
    """
    from bench.workloads import jittered_predictions

    gt = coco_val2017.gt_path()
    dt = jittered_predictions.dt_path(gt_path=gt, seed=0)
    return StreamingWorkload(
        workload_id="coco_val2017_dlpack_vs_json",
        gt_path=gt,
        dt_path=dt,
        iou_type="bbox",
        # One-shot ingest (a single per-image array per update); the
        # runner does not itself shard, since the cell is about ingest
        # path comparison rather than chunking.
        chunk_schedule=(),
    )


__all__ = ["dlpack_vs_json"]
