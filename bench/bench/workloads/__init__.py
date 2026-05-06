"""Workload registry — tagged union of paradigm-discriminated workload
shapes per ADR-0033.

The discriminator is ``paradigm: Literal["instance", "panoptic",
"semantic", "streaming"]``. Today, every concrete workload is an
``InstanceWorkload`` (the detection-only shape from ADR-0017); the
other three variants exist as Pydantic models that B1/B2/B3 will
register concrete workloads against.

Workload identifiers (instance — registered today):

- ``smoke`` — local fixture, all iou types.
- ``coco_val2017_jittered_seed<N>`` — COCO val2017 GT (sha256-pinned)
  with deterministic Gaussian-jittered bbox + mask-space-jittered segm
  DT for seed ``<N>``. Serves bbox, segm, and boundary cells.
- ``coco_val2017_perfect_segm`` — COCO val2017 GT paired with the
  ``perfect_dt_segm.json`` "GT-as-DT" predictions used by the parity
  suite. Exercises segm + boundary at val2017 scale (5000 imgs / ~36k
  anns); under-stresses matching because every DT lines up with a GT,
  which keeps it useful as a perf smoke but not as a realistic detector
  benchmark.
- ``synthetic:k=v,k=v[,...]`` — parametric stress-test. Required keys:
  ``n_images``, ``seed``. Optional: ``n_categories`` (default 80),
  ``dt_per_image`` (default 30), ``gt_per_image`` (default 10) — chosen
  to match the ADR's release-mode ladder defaults.
- ``coco_val2017_maskrcnn_r50fpn_d2_v1`` — Mask R-CNN R50-FPN
  predictions on val2017 (Detectron2 model zoo, 3x schedule). Sourced
  from a pinned Hugging Face dump; populate via
  ``./tools/fetch-real-predictions.sh --maskrcnn``. Serves bbox + segm
  + boundary.
- ``coco_val2017_rfdetr_nano_v<rfdetr-pin>`` — bbox-only rf-detr Nano
  predictions on val2017. Inferred locally by the TIDE harness;
  populate via ``pytest -m real_models``.
- ``coco_val2017_rfdetr_segnano_v<rfdetr-pin>`` — instance-seg rf-detr
  SegNano predictions on val2017. Same provenance as Nano. Serves bbox
  + segm + boundary.

Panoptic / semantic / streaming workload IDs are reserved by their
respective B-streams (B1 panoptic, B2 semantic, B3 streaming) and
registered through ``resolve()`` when they land. ``resolve()`` raises
``NotImplementedError("registered by Bx stream")`` for IDs in their
namespace until the cells exist.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from bench.harness.schema import IouType
from bench.workloads import (
    coco_panoptic_val2017,
    coco_val2017,
    jittered_predictions,
    lvis_v1,
    real_predictions,
    smoke,
    synthetic,
)

# Streaming workload modules import this module's StreamingWorkload —
# import them only inside ``resolve()`` to break the cycle. The modules
# are public; use ``from bench.workloads.coco_val2017_streaming import ...``
# directly when consuming outside the resolver.


class _WorkloadBase(BaseModel):
    """Common base — every variant is frozen and rejects unknown
    fields. Carries the ``workload_id`` (the human-facing handle the
    CLI prints) and the discriminator field on subclasses.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    workload_id: str


class InstanceWorkload(_WorkloadBase):
    """Detection paradigm — bbox / segm / keypoints / boundary cells.

    Carries the same ``(gt_path, dt_path, supported_iou_types)`` fields
    the v1 flat ``Workload`` dataclass exposed; the runner subprocess
    contract is unchanged.
    """

    paradigm: Literal["instance"] = "instance"
    gt_path: Path
    dt_path: Path
    supported_iou_types: frozenset[IouType]


class PanopticWorkload(_WorkloadBase):
    """Panoptic paradigm — PQ cell on a (PNG dir + JSON segments_info)
    pair against a categories descriptor.

    B1 will register concrete workloads (``coco_panoptic_val2017_perfect``
    and Stage 3 real-prediction workloads). The shape is fixed by
    ADR-0025 + the B1 plan; these fields match
    ``tests/python/parity_panoptic/harness.py``'s fixture pattern.
    """

    paradigm: Literal["panoptic"] = "panoptic"
    gt_png_dir: Path
    gt_json: Path
    dt_png_dir: Path
    dt_json: Path
    categories_json: Path


class SemanticWorkload(_WorkloadBase):
    """Semantic paradigm — mIoU cell over per-image label-map PNG dirs.

    B2 will register concrete workloads (``cityscapes_val_perfect``
    and Stage 3 real-prediction workloads). ``label_remap`` allows a
    dataset-id → train-id remap for Cityscapes-style evaluation;
    callers that don't need a remap set it empty.
    """

    paradigm: Literal["semantic"] = "semantic"
    gt_label_maps: Path
    dt_label_maps: Path
    n_classes: int
    ignore_label: int
    label_remap: dict[int, int] = Field(default_factory=dict)


class StreamingWorkload(_WorkloadBase):
    """Streaming paradigm — batch-vs-stream parity cell over a COCO-shaped
    GT/DT pair plus a chunk schedule.

    B3 will register concrete workloads
    (``coco_val2017_streaming_throughput`` etc.). ``chunk_schedule`` is
    a list of per-update batch sizes; the runner sums to ``len(GT)``.
    """

    paradigm: Literal["streaming"] = "streaming"
    gt_path: Path
    dt_path: Path
    iou_type: IouType
    chunk_schedule: tuple[int, ...]


# Discriminated union over the four variants. Pydantic validates the
# discriminator at parse time; a malformed JSON fails with the
# correct variant's error message rather than as a runtime
# AttributeError deep in a runner.
Workload = Annotated[
    InstanceWorkload | PanopticWorkload | SemanticWorkload | StreamingWorkload,
    Field(discriminator="paradigm"),
]
WorkloadAdapter: TypeAdapter[Workload] = TypeAdapter(Workload)


_COCO_JITTERED_RE = re.compile(r"^coco_val2017_jittered_seed(\d+)$")
_COCO_KP_JITTERED_RE = re.compile(r"^coco_val2017_keypoints_jittered_seed(\d+)$")
_SYNTHETIC_PREFIX = "synthetic:"
_SYNTHETIC_DEFAULTS: dict[str, int] = {
    "n_categories": 80,
    "dt_per_image": 30,
    "gt_per_image": 10,
}
_SYNTHETIC_REQUIRED: frozenset[str] = frozenset({"n_images", "seed"})
_SYNTHETIC_ALLOWED: frozenset[str] = _SYNTHETIC_REQUIRED | frozenset(_SYNTHETIC_DEFAULTS)

# Prefixes reserved for B1/B2/B3 workloads. ``resolve()`` recognizes
# these so the user-facing error message points at the right
# B-stream rather than collapsing into "unknown workload".
_PANOPTIC_PREFIXES: tuple[str, ...] = ("coco_panoptic_val2017",)
_SEMANTIC_PREFIXES: tuple[str, ...] = ("cityscapes_val", "ade20k_val")
_STREAMING_PREFIXES: tuple[str, ...] = (
    "coco_val2017_streaming",
    "coco_val2017_dlpack",
    "coco_val2017_bg_saturation",
)


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
        return InstanceWorkload(
            workload_id="smoke_perfect_match_segm",
            gt_path=gt,
            dt_path=dt,
            supported_iou_types=frozenset({"bbox", "segm", "boundary"}),
        )

    if m := _COCO_JITTERED_RE.match(workload_name):
        seed = int(m.group(1))
        gt = coco_val2017.gt_path()
        dt = jittered_predictions.dt_path(gt_path=gt, seed=seed)
        return InstanceWorkload(
            workload_id=jittered_predictions.workload_id(seed),
            gt_path=gt,
            dt_path=dt,
            supported_iou_types=frozenset({"bbox", "segm", "boundary"}),
        )

    if m := _COCO_KP_JITTERED_RE.match(workload_name):
        seed = int(m.group(1))
        gt = coco_val2017.gt_path()
        dt = jittered_predictions.keypoints_dt_path(gt_path=gt, seed=seed)
        return InstanceWorkload(
            workload_id=jittered_predictions.keypoints_workload_id(seed),
            gt_path=gt,
            dt_path=dt,
            supported_iou_types=frozenset({"keypoints"}),
        )

    if workload_name == lvis_v1.PERFECT_WORKLOAD_ID:
        gt = lvis_v1.gt_path()
        dt = lvis_v1.perfect_dt_segm_path()
        return InstanceWorkload(
            workload_id=lvis_v1.PERFECT_WORKLOAD_ID,
            gt_path=gt,
            dt_path=dt,
            supported_iou_types=frozenset({"bbox", "segm"}),
        )

    if (lvis_seed := lvis_v1.parse_jittered_seed(workload_name)) is not None:
        gt = lvis_v1.gt_path()
        dt = jittered_predictions.lvis_dt_path(gt_path=gt, seed=lvis_seed)
        return InstanceWorkload(
            workload_id=lvis_v1.jittered_workload_id(lvis_seed),
            gt_path=gt,
            dt_path=dt,
            supported_iou_types=frozenset({"bbox", "segm"}),
        )

    if workload_name == "coco_val2017_perfect_segm":
        gt = coco_val2017.gt_path()
        dt = coco_val2017.perfect_dt_segm_path()
        return InstanceWorkload(
            workload_id="coco_val2017_perfect_segm",
            gt_path=gt,
            dt_path=dt,
            supported_iou_types=frozenset({"segm", "boundary"}),
        )

    if workload_name.startswith(_SYNTHETIC_PREFIX):
        params = _parse_synthetic_args(workload_name.removeprefix(_SYNTHETIC_PREFIX))
        gt, dt = synthetic.make_workload(**params)
        return InstanceWorkload(
            workload_id=synthetic.workload_id(**params),
            gt_path=gt,
            dt_path=dt,
            supported_iou_types=frozenset({"bbox"}),
        )

    if workload_name == real_predictions.MASKRCNN_R50FPN_WORKLOAD_ID:
        return InstanceWorkload(
            workload_id=workload_name,
            gt_path=coco_val2017.gt_path(),
            dt_path=real_predictions.maskrcnn_dt_path(),
            supported_iou_types=frozenset({"bbox", "segm", "boundary"}),
        )

    if workload_name == real_predictions.RFDETR_NANO_WORKLOAD_ID:
        return InstanceWorkload(
            workload_id=workload_name,
            gt_path=coco_val2017.gt_path(),
            dt_path=real_predictions.rfdetr_dt_path("nano"),
            supported_iou_types=frozenset({"bbox"}),
        )

    if workload_name == real_predictions.RFDETR_SEGNANO_WORKLOAD_ID:
        return InstanceWorkload(
            workload_id=workload_name,
            gt_path=coco_val2017.gt_path(),
            dt_path=real_predictions.rfdetr_dt_path("segnano"),
            supported_iou_types=frozenset({"bbox", "segm", "boundary"}),
        )

    # B1 (ADR-0033 + ADR-0025): panoptic-quality cells.
    if workload_name == coco_panoptic_val2017.PERFECT_WORKLOAD_ID:
        gt_png_dir, gt_json, dt_png_dir, dt_json, cats_json = (
            coco_panoptic_val2017.perfect_workload_paths()
        )
        return PanopticWorkload(
            workload_id=workload_name,
            gt_png_dir=gt_png_dir,
            gt_json=gt_json,
            dt_png_dir=dt_png_dir,
            dt_json=dt_json,
            categories_json=cats_json,
        )

    # Reserved B-stream namespaces — the prefix tells the user which
    # follow-up PR registers the concrete workload.
    if any(workload_name.startswith(p) for p in _PANOPTIC_PREFIXES):
        raise NotImplementedError(
            f"workload {workload_name!r} is in the panoptic namespace; "
            f"registered by the B1 stream (ADR-0033 + ADR-0025)."
        )
    if any(workload_name.startswith(p) for p in _SEMANTIC_PREFIXES):
        raise NotImplementedError(
            f"workload {workload_name!r} is in the semantic namespace; "
            f"registered by the B2 stream (ADR-0033 + ADR-0028)."
        )
    # B3 streaming workloads — three concrete cells share the
    # ``StreamingWorkload`` shape. Lazy import keeps the cycle clean.
    if workload_name == "coco_val2017_streaming_throughput":
        from bench.workloads.coco_val2017_streaming import streaming_throughput

        return streaming_throughput()
    if workload_name == "coco_val2017_streaming_vs_naive":
        from bench.workloads.coco_val2017_streaming import streaming_vs_naive

        return streaming_vs_naive()
    if workload_name == "coco_val2017_dlpack_vs_json":
        from bench.workloads.coco_val2017_dlpack import dlpack_vs_json

        return dlpack_vs_json()
    if workload_name == "coco_val2017_bg_saturation":
        from bench.workloads.coco_val2017_bg_saturation import bg_saturation

        return bg_saturation()
    if any(workload_name.startswith(p) for p in _STREAMING_PREFIXES):
        raise NotImplementedError(
            f"workload {workload_name!r} is in the streaming namespace but "
            f"not registered. Known streaming workloads: "
            f"coco_val2017_streaming_throughput, "
            f"coco_val2017_streaming_vs_naive, "
            f"coco_val2017_dlpack_vs_json, "
            f"coco_val2017_bg_saturation."
        )

    raise ValueError(
        f"unknown workload {workload_name!r}; "
        f"known: 'smoke', 'coco_val2017_jittered_seed<N>', "
        f"'coco_val2017_keypoints_jittered_seed<N>', "
        f"'coco_val2017_perfect_segm', 'synthetic:n_images=...,seed=...', "
        f"'{real_predictions.MASKRCNN_R50FPN_WORKLOAD_ID}', "
        f"'{real_predictions.RFDETR_NANO_WORKLOAD_ID}', "
        f"'{real_predictions.RFDETR_SEGNANO_WORKLOAD_ID}', "
        f"'{lvis_v1.PERFECT_WORKLOAD_ID}', 'lvis_v1_val_jittered_seed<N>'"
    )
