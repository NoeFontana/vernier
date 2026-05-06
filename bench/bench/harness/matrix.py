"""(impl, paradigm, metric) compatibility matrix and per-impl path
helpers (ADR-0017 + ADR-0033).

Single source of truth for which runner serves which cell, plus the
env-dir / module-name conversions every component needs. ``IMPL_TO_ENV_NAME``
generalizes env discovery from impl-name to env-name (panoptic uses one
``panopticapi`` env shared between vernier_panoptic + panopticapi
runners, etc.).
"""

from __future__ import annotations

import os
from pathlib import Path

from bench.harness.schema import IouType, Metric, Paradigm

ALL_IMPLS: tuple[str, ...] = (
    "vernier",
    "pycocotools",
    "faster-coco-eval",
    "boundary-iou-api",
)

# Detection-only legacy mapping. Kept for callers that still index by
# IouType (see ``impls_for_iou`` below). Superseded for new code by
# ``IMPL_PARADIGM_SUPPORT[paradigm][impl]``.
IMPL_IOU_SUPPORT: dict[str, frozenset[IouType]] = {
    "vernier": frozenset({"bbox", "segm", "keypoints", "boundary"}),
    "pycocotools": frozenset({"bbox", "segm", "keypoints"}),
    "faster-coco-eval": frozenset({"bbox", "segm", "keypoints"}),
    "boundary-iou-api": frozenset({"boundary"}),
}

# Per-paradigm impl/metric matrix (ADR-0033). The instance entry is
# populated; panoptic / semantic / streaming entries are skeletons that
# B1/B2/B3 fill in when their cells land.
IMPL_PARADIGM_SUPPORT: dict[Paradigm, dict[str, frozenset[Metric]]] = {
    "instance": {
        "vernier": frozenset({"bbox", "segm", "keypoints", "boundary"}),
        "pycocotools": frozenset({"bbox", "segm", "keypoints"}),
        "faster-coco-eval": frozenset({"bbox", "segm", "keypoints"}),
        "boundary-iou-api": frozenset({"boundary"}),
    },
    # B1: vernier_panoptic + panopticapi, both producing ``pq`` per
    # ADR-0033 §"B1 — Panoptic MVB". The two impls share a single env
    # (``bench/envs/panopticapi/``) — see ``IMPL_TO_ENV_NAME`` below.
    "panoptic": {
        "vernier_panoptic": frozenset({"pq"}),
        "panopticapi": frozenset({"pq"}),
    },
    # Semantic Cityscapes MVB (ADR-0033 §B2). Both impls produce a
    # 19x19 uint64 confusion matrix indexed by Cityscapes trainId; the
    # comparator's strict tier asserts bit-equality on the integer
    # array, and the four float headline metrics (mIoU / FWIoU /
    # pixel_accuracy / mean_accuracy) inherit that bit-equality.
    # S3-B will add an `"mmseg": frozenset({"miou"})` entry alongside
    # the ADE20K env (deferred — separate ~5–8min `uv sync`).
    "semantic": {
        "vernier_semantic": frozenset({"miou"}),
        "cityscapesscripts": frozenset({"miou"}),
    },
    # B3 streaming impls — three coupled cells share these entries:
    # * ``vernier_streaming`` runs all three (throughput, vs_naive, dlpack)
    #   from one runner module gated by a ``--mode-flag`` argument.
    # * ``naive_python`` is the ``predictions.append(...); cocoeval.evaluate()``
    #   baseline that only participates in the vs_naive cell.
    "streaming": {
        "vernier_streaming": frozenset({"throughput", "vs_naive", "dlpack"}),
        "naive_python": frozenset({"vs_naive"}),
    },
}

# Env discovery: by default, ``bench/envs/<impl>/`` is the runner's
# uv-managed env. B1/B2/B3 will register multi-runner envs (e.g.
# ``vernier_panoptic`` and ``panopticapi`` runners both run in
# ``bench/envs/panopticapi/``); they extend this map at import time.
# Hyphenated impl names map to themselves (the env dir keeps the
# hyphen — the module-name substitution lives in ``runner_module``).
IMPL_TO_ENV_NAME: dict[str, str] = {
    "vernier": "vernier",
    "pycocotools": "pycocotools",
    "faster-coco-eval": "faster-coco-eval",
    "boundary-iou-api": "boundary-iou-api",
    # B1 (ADR-0033 §"B1 — Panoptic MVB"): both runners share the
    # ``panopticapi`` env so the parity comparator can run them in
    # identical Python state. The orchestrator still spawns each as
    # its own subprocess (one tensor-output per impl).
    "vernier_panoptic": "panopticapi",
    "panopticapi": "panopticapi",
    # Semantic Cityscapes MVB (ADR-0033 §B2): both runners share the
    # `bench/envs/cityscapes/` env (one `uv sync` covers both impls;
    # the env carries `vernier` for the trainId fold + `cityscapesScripts`
    # for the bincount-based oracle).
    "vernier_semantic": "cityscapes",
    "cityscapesscripts": "cityscapes",
    # B3 streaming impls share the existing detection envs (per
    # ADR-0033 §"reuse existing envs" — no ``bench/envs/streaming/``).
    # ``vernier_streaming`` runs in the vernier env so it can import
    # ``vernier.instance.StreamingEvaluator``; ``naive_python`` runs
    # in the pycocotools env so it can call cocoeval directly.
    "vernier_streaming": "vernier",
    "naive_python": "pycocotools",
}


def impls_for_iou(iou: IouType, impl_filter: tuple[str, ...] | None = None) -> list[str]:
    candidates = impl_filter if impl_filter is not None else ALL_IMPLS
    return [i for i in candidates if iou in IMPL_IOU_SUPPORT[i]]


def impls_for_metric(
    paradigm: Paradigm,
    metric: Metric,
    *,
    impl_filter: tuple[str, ...] | None = None,
) -> list[str]:
    """Per-paradigm impl filter — preferred over ``impls_for_iou`` for
    new (multi-paradigm) code paths.

    For unset paradigms (B1/B2/B3 have not yet populated their entries)
    returns ``[]``; the CLI raises a clear error rather than silently
    skipping.
    """
    table = IMPL_PARADIGM_SUPPORT[paradigm]
    candidates = impl_filter if impl_filter is not None else tuple(table.keys())
    return [i for i in candidates if i in table and metric in table[i]]


def env_name(impl: str) -> str:
    """The env-name (dir under ``bench/envs/``) that hosts ``impl``'s
    runner subprocess. Defaults to the impl name when an explicit
    mapping isn't registered."""
    return IMPL_TO_ENV_NAME.get(impl, impl)


def env_dir(bench_root: Path, impl: str) -> Path:
    return bench_root / "envs" / env_name(impl)


def runner_module(impl: str) -> str:
    # Impl names use hyphens (PyPI distribution names); Python module
    # names cannot. The runner filename and env dir both apply this
    # substitution so adding an impl is a name choice, not a path choice.
    return f"bench.runners.{impl.replace('-', '_')}_runner"


def uv_run_argv(bench_root: Path, impl: str, *trailing: str) -> list[str]:
    """``uv run --directory <env> python <trailing...>`` argv prefix
    every subprocess shares (orchestrator, isolation/contract tests)."""
    return ["uv", "run", "--directory", str(env_dir(bench_root, impl)), "python", *trailing]


def uv_run_env(bench_root: Path, impl: str) -> dict[str, str]:
    """OS env for spawning ``impl``'s runner subprocess.

    Carries the parent env plus any ``PYTHONPATH`` entries the runner
    needs at process start. Currently only ``boundary-iou-api`` does:
    its oracle is a vendored verbatim checkout with no ``pyproject.toml``
    (modifying it would violate ``oracle/VENDORING.md``), so uv can't
    install it. Injecting via ``PYTHONPATH`` here keeps the path
    plumbing out of the runner module itself.
    """
    env = dict(os.environ)
    if impl == "boundary-iou-api":
        oracle = str(env_dir(bench_root, impl) / "oracle")
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{oracle}{os.pathsep}{existing}" if existing else oracle
    return env
