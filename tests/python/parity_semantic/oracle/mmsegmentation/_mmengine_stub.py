"""Minimal stubs for the `mmengine` and `mmseg.registry` surface that
mmsegmentation's `IoUMetric` reaches at import time.

`iou_metric.py` is vendored verbatim at the SHA pinned in `VENDORING.md`;
its imports include:

    from mmengine.dist import is_main_process
    from mmengine.evaluator import BaseMetric
    from mmengine.logging import MMLogger, print_log
    from mmengine.utils import mkdir_or_exist
    from mmseg.registry import METRICS

mmengine pulls in mmcv, the mmsegmentation package pulls mmengine and
PyTorch — together ~3 GB. AP5 in `docs/engineering/sem-seg-quirks.md`
flagged this as the reason to vendor `IoUMetric` standalone. This stub
covers the entire mmengine + mmseg.registry surface the parity oracle
needs; the bulky transitives never enter the test environment.

`torch` is a real dependency — `IoUMetric.intersect_and_union` calls
`torch.histc` for label binning, which numpy does not replicate
bit-exactly. The torch pin lives in `pyproject.toml` and mirrors into
`crates/vernier-semantic/src/parity.rs::ORACLE_TORCH_PIN`.

Conftest registers each class under its upstream import path in
`sys.modules` before `iou_metric` is imported. The stubs are
deliberately minimal — only the symbols `iou_metric.py` reads at the
pinned SHA. If a refresh introduces new symbols, the import will fail
loudly and the stub is updated alongside the SHA bump (ADR-0001
refresh procedure).
"""

from __future__ import annotations

import os
from typing import Any


def is_main_process() -> bool:
    """Stub for ``mmengine.dist.is_main_process``.

    The parity harness runs single-process; always main.
    """
    return True


def mkdir_or_exist(dir_name: str | os.PathLike[str]) -> None:
    """Stub for ``mmengine.utils.mkdir_or_exist``.

    Reachable only from ``IoUMetric.__init__`` when ``output_dir`` is set.
    The parity harness never sets ``output_dir``; this is here to keep
    the import resolvable.
    """
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)


def print_log(msg: str, logger: Any = None, level: Any = None) -> None:
    """Stub for ``mmengine.logging.print_log``.

    Reachable only from ``IoUMetric.compute_metrics``; the parity
    harness asserts on the returned metric dict, not on log output.
    """
    del msg, logger, level


class _NoopLogger:
    def info(self, msg: str) -> None:
        del msg

    def warning(self, msg: str) -> None:
        del msg

    def error(self, msg: str) -> None:
        del msg


class MMLogger:
    """Stub for ``mmengine.logging.MMLogger``.

    Only ``MMLogger.get_current_instance()`` is called from
    ``IoUMetric.compute_metrics`` (line 114); the returned object only
    has ``.info`` invoked on it (line 116).
    """

    @classmethod
    def get_current_instance(cls) -> _NoopLogger:
        return _NoopLogger()


class BaseMetric:
    """Stub for ``mmengine.evaluator.BaseMetric``.

    `IoUMetric` reaches:

    - ``super().__init__(collect_device=..., prefix=...)`` (line 56)
    - ``self.results`` — list, populated in ``process`` (line 84),
      consumed in ``compute_metrics`` (line 121)
    - ``self.dataset_meta`` — dict the orchestrator sets before
      ``process``; read at line 77, 132

    Real ``BaseMetric`` does distributed-rank result collation via
    ``collect_device``; the parity harness is single-process so we
    just store results in a list.
    """

    def __init__(self, collect_device: str = "cpu", prefix: str | None = None) -> None:
        self.collect_device = collect_device
        self.prefix = prefix
        self.results: list[Any] = []
        self.dataset_meta: dict[str, Any] = {}


class PrettyTable:
    """Stub for ``prettytable.PrettyTable``.

    `IoUMetric.compute_metrics` builds a table for log output (lines
    154-159). The parity harness asserts on the returned metric dict,
    not on log content; the stub captures column data so
    ``get_string`` returns a deterministic string but the contents are
    never inspected.
    """

    def __init__(self) -> None:
        self._columns: list[tuple[str, Any]] = []

    def add_column(self, fieldname: str, column: Any) -> None:
        self._columns.append((fieldname, column))

    def get_string(self) -> str:
        return ""


class _MetricsRegistry:
    """Stub for ``mmseg.registry.METRICS``.

    `IoUMetric` is decorated with ``@METRICS.register_module()`` (line 18).
    The real registry maps name → class for runtime lookup; the parity
    harness imports `IoUMetric` directly so the registry is never read.
    The decorator is a no-op identity.
    """

    def register_module(
        self,
        name: str | None = None,
        force: bool = False,
        module: type | None = None,
    ):
        del name, force
        if module is not None:
            return module

        def _decorator(cls: type) -> type:
            return cls

        return _decorator


METRICS = _MetricsRegistry()
