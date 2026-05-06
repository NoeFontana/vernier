"""``coco_val2017_bg_saturation`` workload — BackgroundEvaluator p99
latency cell. Reuses the val2017 jittered seed-0 DT (same fixture as
the streaming cells) so the saturation feeder cycles through real
detections without a new GT fetch.

The cell's parity tier is *informational* — there is no oracle for
tail latency. Comparator records ``parity_tier="informational"`` and
the optional regression check (warn when ``current p99 > 1.20 * prior
p99``) lives in the report layer over the latency_cdf artifact.
"""

from __future__ import annotations

from bench.workloads import StreamingWorkload, coco_val2017

WORKLOAD_ID = "coco_val2017_bg_saturation"
_DEFAULT_SEED = 0


def bg_saturation() -> StreamingWorkload:
    from bench.workloads import jittered_predictions

    gt = coco_val2017.gt_path()
    dt = jittered_predictions.dt_path(gt_path=gt, seed=_DEFAULT_SEED)
    return StreamingWorkload(
        workload_id=WORKLOAD_ID,
        gt_path=gt,
        dt_path=dt,
        iou_type="bbox",
        chunk_schedule=(),
    )


__all__ = ["WORKLOAD_ID", "bg_saturation"]
