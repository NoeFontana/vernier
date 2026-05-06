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
    # B1 populates with vernier_panoptic + panopticapi (each producing
    # ``pq``). Map shape: {impl_name: frozenset({"pq"})}.
    "panoptic": {},
    # B2 populates with vernier_semantic + cityscapesscripts (and
    # eventually mmseg in S3-B). Map shape: {impl_name: frozenset({"miou"})}.
    "semantic": {},
    # B3 populates with vernier_streaming + naive_python. Map shape:
    # {impl_name: frozenset({"throughput", "p99", "rss"})} or whatever
    # subset of the metric literal each impl produces.
    "streaming": {},
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
    # B1 will add (when the panopticapi env lands):
    #   "vernier_panoptic": "panopticapi",
    #   "panopticapi": "panopticapi",
    # B2 will add (when the cityscapes env lands):
    #   "vernier_semantic": "cityscapes",
    #   "cityscapesscripts": "cityscapes",
    # B3 will add the streaming impls (sharing the existing
    # ``vernier`` and ``pycocotools`` envs):
    #   "vernier_streaming": "vernier",
    #   "naive_python": "pycocotools",
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
