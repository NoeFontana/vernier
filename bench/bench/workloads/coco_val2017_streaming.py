"""Streaming workloads against the COCO val2017 GT/DT pair.

Two registered cells share the same GT/DT inputs as the detection
``coco_val2017_jittered_seed0`` cell — same bytes on disk, same
fixture provenance — so streaming and batch results are directly
comparable.

- ``coco_val2017_streaming_throughput`` — exercises
  ``StreamingEvaluator.update(...)+finalize()`` per-image and measures
  throughput. The comparator's parity gate is bit-equal
  ``Summary.stats`` against the batch ``Evaluator.evaluate(...)`` path
  (per ``tests/python/parity/streaming/test_streaming_finalize_equals_batch.py``).
- ``coco_val2017_streaming_vs_naive`` — same shape but pairs vernier's
  streaming runner with the ``predictions.append(...); cocoeval.evaluate()``
  baseline. Throughput delta + RSS curves are informational artifacts;
  the parity gate is the same Summary.stats bit-equality (vernier
  batch matches pycocotools per the long-standing parity contract).

``chunk_schedule`` is a tuple of per-update batch sizes. The runner
sums to the total image count; an empty tuple means "all images in one
``update()`` call". The current default is per-image
(``chunk_schedule=()`` deferred to the runner) because the throughput
claim hinges on per-image dispatch — change here would silently shift
the measurement target.
"""

from __future__ import annotations

from bench.workloads import StreamingWorkload, coco_val2017

# Default chunk schedule: empty tuple means "one chunk per image" (the
# runner enumerates GT image_ids and shards DT accordingly). The
# alternative — pinning a fixed list of shard sizes — would couple the
# workload to the COCO val2017 image count (4952 with crowd, 5000
# nominally) which would silently break if the GT pin moved.
_PER_IMAGE_SCHEDULE: tuple[int, ...] = ()


def streaming_throughput() -> StreamingWorkload:
    """Streaming throughput cell over COCO val2017 jittered seed-0 DT.

    Reuses the same DT as ``coco_val2017_jittered_seed0`` (mask-space-
    jittered v2, per recent commit ``c4b583d``) so the streaming-vs-batch
    parity claim shares fixture state with the detection cell.
    """
    from bench.workloads import jittered_predictions

    gt = coco_val2017.gt_path()
    dt = jittered_predictions.dt_path(gt_path=gt, seed=0)
    return StreamingWorkload(
        workload_id="coco_val2017_streaming_throughput",
        gt_path=gt,
        dt_path=dt,
        iou_type="bbox",
        chunk_schedule=_PER_IMAGE_SCHEDULE,
    )


def streaming_vs_naive() -> StreamingWorkload:
    """Streaming-vs-naive-Python cell.

    Pairs the vernier streaming runner with the
    ``predictions.append(per_image); cocoeval.evaluate()`` user pattern.
    The cell's load-bearing claim is *RSS shape*: streaming holds
    constant in N, naive grows linearly. The bit-equality gate is the
    standard parity contract (vernier batch == pycocotools batch),
    inherited via the shared comparator.
    """
    from bench.workloads import jittered_predictions

    gt = coco_val2017.gt_path()
    dt = jittered_predictions.dt_path(gt_path=gt, seed=0)
    return StreamingWorkload(
        workload_id="coco_val2017_streaming_vs_naive",
        gt_path=gt,
        dt_path=dt,
        iou_type="bbox",
        chunk_schedule=_PER_IMAGE_SCHEDULE,
    )


__all__ = ["streaming_throughput", "streaming_vs_naive"]
