"""Workload registry — tagged union of paradigm-discriminated workload
shapes per ADR-0033.

The discriminator is ``paradigm: Literal["instance", "panoptic",
"semantic", "streaming"]``. Today, every concrete workload is an
``InstanceWorkload`` (the detection-only shape from ADR-0017); the
other three variants are Pydantic models for the panoptic, semantic,
and streaming workloads.

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
- ``coco_val2017_detr_r50_v<short-sha>`` — bbox-only ``facebook/detr-resnet-50``
  predictions on val2017, inferred by the Hugging Face SOTA harness
  (``tests/python/integration/real_models/sota/``). Same ``real-models``
  extra as rf-detr (torch + transformers + huggingface_hub). The pin
  is the first 7 chars of the model's hub commit SHA. Serves bbox only.
- ``coco_panoptic_val2017_mask2former_swin_t_v<short-sha>`` — panoptic
  PQ cell, ``facebook/mask2former-swin-tiny-coco-panoptic`` predictions
  on COCO panoptic val2017. Same SOTA-harness extra; bench reads the
  rgb2id PNG dir + panoptic_dt.json sidecar without an inference dep.
- ``ade20k_val_mask2former_swin_t_v<short-sha>`` — semantic mIoU cell,
  ``facebook/mask2former-swin-tiny-ade-semantic`` predictions on
  ADE20K val. Reuses the SOTA-harness extra; ADE20K GT lives in its
  own ``ade20k_val_cache`` module parallel to ``panoptic_val_cache``.

Streaming workload IDs are reserved by B3 and registered through
``resolve()`` when they land. ``resolve()`` raises
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
    coco_val2017_semantic,
    jittered_predictions,
    lvis_v1,
    real_predictions,
    smoke,
    synthetic,
    synthetic_semantic,
)

# Streaming workload modules import this module's StreamingWorkload —
# import them only inside ``resolve()`` to break the cycle. The modules
# are public; use ``from bench.workloads.coco_val2017_streaming import ...``
# directly when consuming outside the resolver.


class _WorkloadBase(BaseModel):
    """Common base — every variant is frozen and rejects unknown
    fields. Carries the ``workload_id`` (the human-facing handle the
    CLI prints) and the discriminator field on subclasses.

    ``num_threads`` is the ADR-0047 fan-out axis: a tuple of thread
    counts (``None`` = library default, i.e. single-threaded) the CLI
    expands into one :class:`CellSpec` per entry. The default
    ``(None,)`` preserves the pre-ADR-0047 single-cell behavior — every
    existing workload keeps emitting exactly one cell.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    workload_id: str
    num_threads: tuple[int | None, ...] = (None,)


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

    First concrete cell lands at S3-B (ADE20K + mmseg). ``label_remap``
    allows a dataset-id → train-id remap (e.g. ADE20K's 0-indexed
    background); callers that don't need a remap set it empty.
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


class LvisWorkload(_WorkloadBase):
    """LVIS paradigm — federated AP cell over an LVIS-shaped JSON GT/DT
    pair (ADR-0026).

    Same on-disk shape as :class:`InstanceWorkload`, but the federated
    semantics (``not_exhaustive_category_ids`` / ``neg_category_ids``
    per image, ``frequency`` letters per category, the 13-entry
    ``lvis_default()`` summary plan) require a separate runner protocol
    and a separate oracle (``lvis-api``). Splitting into its own
    paradigm prevents COCO-side runners (pycocotools / faster-coco-eval)
    from being silently dispatched against LVIS data.
    """

    paradigm: Literal["lvis"] = "lvis"
    gt_path: Path
    dt_path: Path
    supported_iou_types: frozenset[IouType]


# Discriminated union over the five variants. Pydantic validates the
# discriminator at parse time; a malformed JSON fails with the
# correct variant's error message rather than as a runtime
# AttributeError deep in a runner.
Workload = Annotated[
    InstanceWorkload | PanopticWorkload | SemanticWorkload | StreamingWorkload | LvisWorkload,
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
_SYNTHETIC_FLOAT_DEFAULTS: dict[str, float] = {
    "iscrowd_fraction": 0.0,
}
_SYNTHETIC_REQUIRED: frozenset[str] = frozenset({"n_images", "seed"})
_SYNTHETIC_ALLOWED: frozenset[str] = (
    _SYNTHETIC_REQUIRED | frozenset(_SYNTHETIC_DEFAULTS) | frozenset(_SYNTHETIC_FLOAT_DEFAULTS)
)

# Per-paradigm workload-id prefixes; ``resolve()`` recognizes these
# so an unknown workload in a registered namespace surfaces a
# paradigm-specific error message.
_PANOPTIC_PREFIXES: tuple[str, ...] = ("coco_panoptic_val2017",)
_SEMANTIC_PREFIXES: tuple[str, ...] = ("ade20k_val", "coco_val2017_semantic")
_SYNTHETIC_SEMANTIC_PREFIX = "synthetic_semantic:"
_SYNTHETIC_SEMANTIC_REQUIRED: frozenset[str] = frozenset({"n_images", "n_classes", "seed"})
_SYNTHETIC_SEMANTIC_INT_OPTIONAL: frozenset[str] = frozenset({"ignore_label"})
_SYNTHETIC_SEMANTIC_FLOAT_OPTIONAL: frozenset[str] = frozenset({"jitter_rate"})
_STREAMING_PREFIXES: tuple[str, ...] = (
    "coco_val2017_streaming",
    "coco_val2017_dlpack",
    "coco_val2017_bg_saturation",
)


def _parse_synthetic_semantic_args(spec: str) -> tuple[dict[str, int], dict[str, float]]:
    """Parse ``synthetic_semantic:k=v,...`` into (int-params, float-params).

    Required: ``n_images``, ``n_classes``, ``seed``. Optional int:
    ``ignore_label``. Optional float: ``jitter_rate``.
    """
    if not spec:
        raise ValueError("synthetic_semantic: requires at least n_images=, n_classes=, seed=")
    int_out: dict[str, int] = {}
    float_out: dict[str, float] = {}
    allowed = (
        _SYNTHETIC_SEMANTIC_REQUIRED
        | _SYNTHETIC_SEMANTIC_INT_OPTIONAL
        | _SYNTHETIC_SEMANTIC_FLOAT_OPTIONAL
    )
    for token in spec.split(","):
        if "=" not in token:
            raise ValueError(f"synthetic_semantic: param {token!r} is not k=v")
        key, value = token.split("=", 1)
        key = key.strip()
        if key not in allowed:
            raise ValueError(
                f"synthetic_semantic: unknown param {key!r}; allowed: {sorted(allowed)}"
            )
        if key in _SYNTHETIC_SEMANTIC_FLOAT_OPTIONAL:
            try:
                fval = float(value)
            except ValueError as e:
                raise ValueError(f"synthetic_semantic: param {key}={value!r} must be float") from e
            if not 0.0 <= fval <= 1.0:
                raise ValueError(f"synthetic_semantic: param {key}={value!r} must be in [0.0, 1.0]")
            float_out[key] = fval
        else:
            try:
                int_out[key] = int(value)
            except ValueError as e:
                raise ValueError(f"synthetic_semantic: param {key}={value!r} must be int") from e
    missing = _SYNTHETIC_SEMANTIC_REQUIRED - int_out.keys()
    if missing:
        raise ValueError(f"synthetic_semantic: missing required param(s): {sorted(missing)}")
    return int_out, float_out


def _parse_synthetic_args(spec: str) -> tuple[dict[str, int], dict[str, float]]:
    """Parse ``synthetic:k=v,...`` into (int-params, float-params)."""
    if not spec:
        raise ValueError("synthetic: requires at least n_images=... and seed=...")
    int_out: dict[str, int] = {}
    float_out: dict[str, float] = {}
    for token in spec.split(","):
        if "=" not in token:
            raise ValueError(f"synthetic: param {token!r} is not k=v")
        key, value = token.split("=", 1)
        key = key.strip()
        if key not in _SYNTHETIC_ALLOWED:
            raise ValueError(
                f"synthetic: unknown param {key!r}; allowed: {sorted(_SYNTHETIC_ALLOWED)}"
            )
        if key in _SYNTHETIC_FLOAT_DEFAULTS:
            try:
                fval = float(value)
            except ValueError as e:
                raise ValueError(f"synthetic: param {key}={value!r} must be float") from e
            if not 0.0 <= fval <= 1.0:
                raise ValueError(f"synthetic: param {key}={value!r} must be in [0.0, 1.0]")
            float_out[key] = fval
        else:
            try:
                int_out[key] = int(value)
            except ValueError as e:
                raise ValueError(f"synthetic: param {key}={value!r} must be int") from e
    missing = _SYNTHETIC_REQUIRED - int_out.keys()
    if missing:
        raise ValueError(f"synthetic: missing required param(s): {sorted(missing)}")
    return (
        {**_SYNTHETIC_DEFAULTS, **int_out},
        {**_SYNTHETIC_FLOAT_DEFAULTS, **float_out},
    )


def resolve(workload_name: str, repo_root: Path) -> Workload:
    if workload_name == "smoke":
        gt, dt = smoke.paths(repo_root)
        return InstanceWorkload(
            workload_id="smoke_perfect_match_segm",
            gt_path=gt,
            dt_path=dt,
            supported_iou_types=frozenset({"bbox", "segm", "boundary"}),
        )

    # ADR-0047 threading-scaling smoke fixture. Reuses the synthetic
    # GT/DT factory but pins ``num_threads`` so the CLI fans out one
    # cell per thread count. Bbox-only — that's the kernel ADR-0047
    # Stage A parallelized first.
    if workload_name == synthetic.THREADS_SMOKE_WORKLOAD_ID:
        gt, dt = synthetic.threads_smoke_paths()
        return InstanceWorkload(
            workload_id=synthetic.THREADS_SMOKE_WORKLOAD_ID,
            gt_path=gt,
            dt_path=dt,
            supported_iou_types=frozenset({"bbox"}),
            num_threads=synthetic.THREADS_SMOKE_NUM_THREADS,
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
        gt = coco_val2017.kp_gt_path()
        dt = jittered_predictions.keypoints_dt_path(gt_path=gt, seed=seed)
        return InstanceWorkload(
            workload_id=jittered_predictions.keypoints_workload_id(seed),
            gt_path=gt,
            dt_path=dt,
            supported_iou_types=frozenset({"keypoints"}),
        )

    if workload_name == lvis_v1.PERFECT_WORKLOAD_ID:
        gt = lvis_v1.gt_path()
        # The bench is bbox-only on the LVIS paradigm today (the
        # parsed-once-dataset FFI only ships ``evaluate_bbox_grid_with_dataset``);
        # take the bbox-shape perfect-DT so the strict-tier parity gate
        # against lvis-api stays bit-equal. Once the segm FFI lands
        # this resolver flips to ``perfect_dt_segm_path`` for segm
        # cells (per-iou DT picked at CellSpec time).
        dt = lvis_v1.perfect_dt_bbox_path()
        return LvisWorkload(
            workload_id=lvis_v1.PERFECT_WORKLOAD_ID,
            gt_path=gt,
            dt_path=dt,
            supported_iou_types=frozenset({"bbox"}),
        )

    if (lvis_seed := lvis_v1.parse_jittered_seed(workload_name)) is not None:
        gt = lvis_v1.gt_path()
        dt = jittered_predictions.lvis_dt_path(gt_path=gt, seed=lvis_seed)
        return LvisWorkload(
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
        int_params, float_params = _parse_synthetic_args(
            workload_name.removeprefix(_SYNTHETIC_PREFIX)
        )
        iscrowd_fraction = float_params["iscrowd_fraction"]
        gt, dt = synthetic.make_workload(**int_params, iscrowd_fraction=iscrowd_fraction)
        return InstanceWorkload(
            workload_id=synthetic.workload_id(**int_params, iscrowd_fraction=iscrowd_fraction),
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

    if workload_name == real_predictions.DETR_R50_WORKLOAD_ID:
        return InstanceWorkload(
            workload_id=workload_name,
            gt_path=coco_val2017.gt_path(),
            dt_path=real_predictions.detr_r50_dt_path(),
            supported_iou_types=frozenset({"bbox"}),
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

    if workload_name == real_predictions.MASK2FORMER_PANOPTIC_WORKLOAD_ID:
        # Reuses the panoptic_val_cache GT side (already provisioned by
        # coco_panoptic_val2017_perfect for the smoke cell) and the
        # Mask2Former DT side from the SOTA harness's prediction cache.
        gt_png_dir, gt_json, _, _, cats_json = coco_panoptic_val2017.perfect_workload_paths()
        dt_png_dir, dt_json = real_predictions.mask2former_panoptic_dt_paths()
        return PanopticWorkload(
            workload_id=workload_name,
            gt_png_dir=gt_png_dir,
            gt_json=gt_json,
            dt_png_dir=dt_png_dir,
            dt_json=dt_json,
            categories_json=cats_json,
        )

    if workload_name == real_predictions.MASK2FORMER_ADE_WORKLOAD_ID:
        # ADE20K val GT lives in its own cache module (parallel to
        # panoptic_val_cache). The semantic cell consumes (gt_dir,
        # dt_dir, n_classes, ignore_label) — no label_remap because
        # both sides already use the mmseg reduce_zero_label
        # train-id space.
        from ade20k_val_cache import (
            ADE20K_IGNORE_LABEL,
            ADE20K_NUM_CLASSES,
            ensure_gt,
        )

        gt_dir, _, _ = ensure_gt()
        dt_dir = real_predictions.mask2former_ade_dt_path()
        return SemanticWorkload(
            workload_id=workload_name,
            gt_label_maps=gt_dir,
            dt_label_maps=dt_dir,
            n_classes=ADE20K_NUM_CLASSES,
            ignore_label=ADE20K_IGNORE_LABEL,
        )

    # Reserved B-stream namespaces — the prefix tells the user which
    # follow-up PR registers the concrete workload.
    if any(workload_name.startswith(p) for p in _PANOPTIC_PREFIXES):
        raise NotImplementedError(
            f"workload {workload_name!r} is in the panoptic namespace; "
            f"registered by the B1 stream (ADR-0033 + ADR-0025)."
        )
    # B2 (ADR-0033 + ADR-0028): semantic-segmentation cells. The
    # synthetic-semantic family is the vernier-only baseline; ADE20K
    # waits on the S3-B oracle vendoring + license-cleared cache.
    if workload_name.startswith(_SYNTHETIC_SEMANTIC_PREFIX):
        int_params, float_params = _parse_synthetic_semantic_args(
            workload_name.removeprefix(_SYNTHETIC_SEMANTIC_PREFIX)
        )
        ignore_label = int_params.pop("ignore_label", synthetic_semantic.DEFAULT_IGNORE_LABEL)
        jitter_rate = float_params.get("jitter_rate", synthetic_semantic.DEFAULT_JITTER_RATE)
        gt_dir, dt_dir = synthetic_semantic.make_workload(
            **int_params,
            jitter_rate=jitter_rate,
            ignore_label=ignore_label,
        )
        return SemanticWorkload(
            workload_id=synthetic_semantic.workload_id(**int_params, jitter_rate=jitter_rate),
            gt_label_maps=gt_dir,
            dt_label_maps=dt_dir,
            n_classes=int_params["n_classes"],
            ignore_label=ignore_label,
        )
    if workload_name == coco_val2017_semantic.PERFECT_WORKLOAD_ID:
        gt_dir, dt_dir, n_classes, _ = coco_val2017_semantic.perfect_workload_paths()
        return SemanticWorkload(
            workload_id=workload_name,
            gt_label_maps=gt_dir,
            dt_label_maps=dt_dir,
            n_classes=n_classes,
            ignore_label=coco_val2017_semantic.IGNORE_LABEL,
        )
    if any(workload_name.startswith(p) for p in _SEMANTIC_PREFIXES):
        raise NotImplementedError(
            f"workload {workload_name!r} is in the semantic namespace; "
            f"known cells: '{coco_val2017_semantic.PERFECT_WORKLOAD_ID}', "
            f"'{real_predictions.MASK2FORMER_ADE_WORKLOAD_ID}'. For a "
            f"vernier-only baseline, use "
            f"'synthetic_semantic:n_images=...,n_classes=...,seed=...'."
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
        f"'{real_predictions.DETR_R50_WORKLOAD_ID}', "
        f"'{real_predictions.MASK2FORMER_PANOPTIC_WORKLOAD_ID}', "
        f"'{real_predictions.MASK2FORMER_ADE_WORKLOAD_ID}', "
        f"'{lvis_v1.PERFECT_WORKLOAD_ID}', 'lvis_v1_val_jittered_seed<N>'"
    )
