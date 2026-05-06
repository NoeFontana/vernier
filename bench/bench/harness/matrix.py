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

# Per-paradigm impl/metric matrix (ADR-0033). One source of truth for
# which impl serves which (paradigm, metric) cell; ``IMPL_IOU_SUPPORT``
# below derives from the instance entry for callers that still index
# by IouType.
IMPL_PARADIGM_SUPPORT: dict[Paradigm, dict[str, frozenset[Metric]]] = {
    "instance": {
        "vernier": frozenset({"bbox", "segm", "keypoints", "boundary"}),
        "pycocotools": frozenset({"bbox", "segm", "keypoints"}),
        "faster-coco-eval": frozenset({"bbox", "segm", "keypoints"}),
        "boundary-iou-api": frozenset({"boundary"}),
    },
    "panoptic": {
        "vernier_panoptic": frozenset({"pq"}),
        "panopticapi": frozenset({"pq"}),
    },
    # Cityscapes-vs-cityscapesScripts is strict on integer counts.
    # ADE20K + mmseg lands in S3-B as an additional `aligned`-tier
    # entry pending PR-B6/7/8 vendoring.
    "semantic": {
        "vernier_semantic": frozenset({"miou"}),
        "cityscapesscripts": frozenset({"miou"}),
    },
    "streaming": {
        "vernier_streaming": frozenset({"throughput", "vs_naive", "dlpack"}),
        "naive_python": frozenset({"vs_naive"}),
        # B5: BackgroundEvaluator p99 cell. Single impl; no oracle —
        # the latency_cdf artifact is informational (no parity gate).
        "vernier_bg": frozenset({"p99"}),
    },
}

# Detection-only IouType view, derived from the instance entry. Used
# by callers that still index by ``IouType`` (e.g. ``impls_for_iou``).
IMPL_IOU_SUPPORT: dict[str, frozenset[IouType]] = {
    impl: frozenset(metrics)  # type: ignore[arg-type]
    for impl, metrics in IMPL_PARADIGM_SUPPORT["instance"].items()
}

# Maps impl-name → env-dir under ``bench/envs/``. Multiple impls can
# share one env (panoptic + semantic group their two runners; streaming
# reuses the detection envs).
IMPL_TO_ENV_NAME: dict[str, str] = {
    "vernier": "vernier",
    "pycocotools": "pycocotools",
    "faster-coco-eval": "faster-coco-eval",
    "boundary-iou-api": "boundary-iou-api",
    "vernier_panoptic": "panopticapi",
    "panopticapi": "panopticapi",
    "vernier_semantic": "cityscapes",
    "cityscapesscripts": "cityscapes",
    # ``vernier_streaming`` runs in the vernier env so it can import
    # ``vernier.instance.StreamingEvaluator``; ``naive_python`` runs
    # in the pycocotools env so it can call cocoeval directly.
    "vernier_streaming": "vernier",
    "naive_python": "pycocotools",
    "vernier_bg": "vernier",
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

    For unset paradigms (no impls registered for the (paradigm, metric))
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
