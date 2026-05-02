"""(impl, iou) compatibility matrix and per-impl path helpers.

Single source of truth for which runner serves which IoU cell, plus the
env-dir / module-name conversions every component needs.
"""

from __future__ import annotations

import os
from pathlib import Path

from bench.harness.schema import IouType

ALL_IMPLS: tuple[str, ...] = (
    "vernier",
    "pycocotools",
    "faster-coco-eval",
    "boundary-iou-api",
)

IMPL_IOU_SUPPORT: dict[str, frozenset[IouType]] = {
    "vernier": frozenset({"bbox", "segm", "keypoints", "boundary"}),
    "pycocotools": frozenset({"bbox", "segm", "keypoints"}),
    "faster-coco-eval": frozenset({"bbox", "segm", "keypoints"}),
    "boundary-iou-api": frozenset({"boundary"}),
}


def impls_for_iou(iou: IouType, impl_filter: tuple[str, ...] | None = None) -> list[str]:
    candidates = impl_filter if impl_filter is not None else ALL_IMPLS
    return [i for i in candidates if iou in IMPL_IOU_SUPPORT[i]]


def env_dir(bench_root: Path, impl: str) -> Path:
    return bench_root / "envs" / impl


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
