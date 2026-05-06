"""The real-predictions adapter resolves cache paths from disk only —
never invokes inference, never downloads. The cache root override
(``VERNIER_REAL_PREDICTIONS_CACHE``) is the single hook used by these
tests so we don't touch the real per-user cache."""

from __future__ import annotations

from pathlib import Path

import pytest

from bench.harness.paths import REPO_ROOT
from bench.workloads import coco_val2017, real_predictions, resolve


@pytest.fixture
def fake_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("VERNIER_REAL_PREDICTIONS_CACHE", str(tmp_path))
    return tmp_path


def test_maskrcnn_dt_path_returns_cached_file(fake_cache: Path) -> None:
    blob = fake_cache / "maskrcnn-r50fpn-d2-v1-coco-val2017.json"
    blob.write_bytes(b"[]")
    path = real_predictions.maskrcnn_dt_path()
    assert path == blob


def test_maskrcnn_dt_path_missing_points_at_fetch_script(fake_cache: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"fetch-real-predictions\.sh --maskrcnn"):
        real_predictions.maskrcnn_dt_path()


def test_rfdetr_segnano_dt_path_returns_cached_file(fake_cache: Path) -> None:
    blob = fake_cache / f"rfdetr-segnano-{real_predictions.RFDETR_VERSION}-coco-val2017.json"
    blob.write_bytes(b"[]")
    assert real_predictions.rfdetr_dt_path("segnano") == blob


def test_rfdetr_dt_path_missing_points_at_real_models_extra(fake_cache: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"pytest -m real_models"):
        real_predictions.rfdetr_dt_path("nano")


@pytest.mark.parametrize(
    ("workload_id", "blob_filename", "expected_iou_types"),
    [
        (
            real_predictions.MASKRCNN_R50FPN_WORKLOAD_ID,
            "maskrcnn-r50fpn-d2-v1-coco-val2017.json",
            frozenset({"bbox", "segm", "boundary"}),
        ),
        (
            real_predictions.RFDETR_SEGNANO_WORKLOAD_ID,
            f"rfdetr-segnano-{real_predictions.RFDETR_VERSION}-coco-val2017.json",
            frozenset({"bbox", "segm", "boundary"}),
        ),
        (
            real_predictions.RFDETR_NANO_WORKLOAD_ID,
            f"rfdetr-nano-{real_predictions.RFDETR_VERSION}-coco-val2017.json",
            frozenset({"bbox"}),
        ),
    ],
)
def test_registry_resolves_real_predictions_workload(
    fake_cache: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    workload_id: str,
    blob_filename: str,
    expected_iou_types: frozenset[str],
) -> None:
    blob = fake_cache / blob_filename
    blob.write_bytes(b"[]")
    gt = tmp_path / "instances_val2017.json"
    gt.write_bytes(b"{}")
    monkeypatch.setattr(coco_val2017, "gt_path", lambda: gt)

    w = resolve(workload_id, REPO_ROOT)
    assert w.workload_id == workload_id
    assert w.dt_path == blob
    assert w.gt_path == gt
    assert w.supported_iou_types == expected_iou_types


def test_unknown_workload_error_lists_real_predictions_ids() -> None:
    with pytest.raises(ValueError, match=real_predictions.MASKRCNN_R50FPN_WORKLOAD_ID):
        resolve("nope", REPO_ROOT)


def test_populate_rfdetr_shells_into_real_models_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``--rfdetr`` flag must invoke ``uv run --extra real-models`` so
    the heavy inference deps stay out of the bench harness env. We check
    the full command vector, not just a substring, because a wrong
    invocation (missing --extra, wrong cwd) would silently fall back to
    the harness env and ImportError on rfdetr."""
    import real_predictions_cache

    captured: dict[str, object] = {}

    def fake_run(cmd: list[str], **kwargs: object) -> object:
        captured["cmd"] = cmd
        captured["cwd"] = kwargs.get("cwd")
        captured["check"] = kwargs.get("check")
        return None

    # String target keeps pyright from flagging an indirect-export read
    # of the subprocess attribute on the cache module.
    monkeypatch.setattr("real_predictions_cache.subprocess.run", fake_run)
    real_predictions_cache.populate_rfdetr("segnano")

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert cmd[:5] == ["uv", "run", "--extra", "real-models", "python"]
    assert "-m" in cmd
    module_idx = cmd.index("-m") + 1
    assert cmd[module_idx] == "tests.python.integration.real_models.tide._populate_cache"
    assert "--model" in cmd
    assert cmd[cmd.index("--model") + 1] == "segnano"
    assert captured["check"] is True
    # cwd must be the repo root so `uv run` finds the workspace pyproject
    # (which declares the [real-models] extra).
    assert captured["cwd"] == real_predictions_cache.REPO_ROOT


def test_populate_rfdetr_rejects_unknown_model() -> None:
    """Runtime check guards CLI args that bypass the Literal type at the
    boundary (e.g., via argparse parsing into a plain str)."""
    import real_predictions_cache
    from typing import cast

    from real_predictions_cache import RfdetrModelName

    with pytest.raises(ValueError, match="unknown rf-detr model"):
        real_predictions_cache.populate_rfdetr(cast(RfdetrModelName, "nope"))
