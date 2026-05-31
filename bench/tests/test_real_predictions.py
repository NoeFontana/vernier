"""The real-predictions adapter resolves cache paths from disk only —
never invokes inference, never downloads. The cache root override
(``VERNIER_REAL_PREDICTIONS_CACHE``) is the single hook used by these
tests so we don't touch the real per-user cache."""

from __future__ import annotations

from pathlib import Path

import pytest

from bench.harness.paths import REPO_ROOT
from bench.workloads import (
    InstanceWorkload,
    PanopticWorkload,
    SemanticWorkload,
    coco_panoptic_val2017,
    coco_val2017,
    real_predictions,
    resolve,
)


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


def test_detr_r50_dt_path_returns_cached_file(fake_cache: Path) -> None:
    from real_predictions_cache import DETR_RESNET50_REVISION

    blob = fake_cache / f"detr-r50-{DETR_RESNET50_REVISION}-coco-val2017.json"
    blob.write_bytes(b"[]")
    assert real_predictions.detr_r50_dt_path() == blob


def test_detr_r50_dt_path_missing_points_at_real_models_extra(fake_cache: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"pytest -m real_models"):
        real_predictions.detr_r50_dt_path()


def test_mask2former_panoptic_dt_paths_returns_cached_files(fake_cache: Path) -> None:
    from real_predictions_cache import MASK2FORMER_PANOPTIC_REVISION

    cache_dir = fake_cache / f"mask2former-pan-swin-t-{MASK2FORMER_PANOPTIC_REVISION}-coco-val2017"
    cache_dir.mkdir()
    dt_json = cache_dir / "panoptic_dt.json"
    dt_json.write_bytes(b'{"annotations": []}')

    out_dir, out_json = real_predictions.mask2former_panoptic_dt_paths()
    assert out_dir == cache_dir
    assert out_json == dt_json


def test_mask2former_panoptic_dt_paths_missing_points_at_real_models_extra(
    fake_cache: Path,
) -> None:
    with pytest.raises(
        FileNotFoundError, match=r"fetch-real-predictions\.sh --mask2former-panoptic"
    ):
        real_predictions.mask2former_panoptic_dt_paths()


def test_mask2former_ade_dt_path_returns_cached_dir(fake_cache: Path) -> None:
    from real_predictions_cache import MASK2FORMER_ADE_REVISION

    cache_dir = fake_cache / f"mask2former-ade-swin-t-{MASK2FORMER_ADE_REVISION}-ade20k-val"
    cache_dir.mkdir()
    (cache_dir / "1.png").write_bytes(b"x")
    assert real_predictions.mask2former_ade_dt_path() == cache_dir


def test_mask2former_ade_dt_path_missing_points_at_real_models_extra(fake_cache: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"fetch-real-predictions\.sh --mask2former-ade"):
        real_predictions.mask2former_ade_dt_path()


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
        (
            real_predictions.DETR_R50_WORKLOAD_ID,
            f"detr-r50-{real_predictions.DETR_RESNET50_REVISION}-coco-val2017.json",
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
    assert isinstance(w, InstanceWorkload)
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
    from typing import cast

    import real_predictions_cache
    from real_predictions_cache import RfdetrModelName

    with pytest.raises(ValueError, match="unknown rf-detr model"):
        real_predictions_cache.populate_rfdetr(cast(RfdetrModelName, "nope"))


def test_populate_detr_resnet50_shells_into_real_models_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same shape gate as the rfdetr test: a wrong invocation (missing
    --extra, wrong module path, wrong cwd) would silently fall back to
    the harness env and ImportError on transformers."""
    import real_predictions_cache

    captured: dict[str, object] = {}

    def fake_run(cmd: list[str], **kwargs: object) -> object:
        captured["cmd"] = cmd
        captured["cwd"] = kwargs.get("cwd")
        captured["check"] = kwargs.get("check")
        return None

    monkeypatch.setattr("real_predictions_cache.subprocess.run", fake_run)
    real_predictions_cache.populate_detr_resnet50()

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert cmd[:5] == ["uv", "run", "--extra", "real-models", "python"]
    assert "-m" in cmd
    module_idx = cmd.index("-m") + 1
    assert cmd[module_idx] == "tests.python.integration.real_models.sota._populate_cache"
    assert "--detr" in cmd
    assert captured["check"] is True
    assert captured["cwd"] == real_predictions_cache.REPO_ROOT


def _pin_mask2former_revisions(monkeypatch: pytest.MonkeyPatch) -> tuple[str, str]:
    """Swap the unpinned sentinels for valid 40-hex placeholders.

    The populator's preflight raises on the ``_UNPINNED_REVISION``
    sentinel; tests for the subprocess shape must pin first or the
    shell-out never happens. Returns the (panoptic, ade) test SHAs so
    tests can assert against them.
    """
    import real_predictions_cache

    panoptic_sha = "a" * 40
    ade_sha = "b" * 40
    monkeypatch.setattr(real_predictions_cache, "MASK2FORMER_PANOPTIC_REVISION", panoptic_sha)
    monkeypatch.setattr(real_predictions_cache, "MASK2FORMER_ADE_REVISION", ade_sha)
    return panoptic_sha, ade_sha


def test_populate_mask2former_panoptic_shells_into_real_models_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same shape gate as the DETR test, plus the pin-check preflight:
    an unpinned revision must fail loudly before the subprocess runs."""
    import real_predictions_cache

    _pin_mask2former_revisions(monkeypatch)
    captured: dict[str, object] = {}

    def fake_run(cmd: list[str], **kwargs: object) -> object:
        captured["cmd"] = cmd
        captured["cwd"] = kwargs.get("cwd")
        captured["check"] = kwargs.get("check")
        return None

    monkeypatch.setattr("real_predictions_cache.subprocess.run", fake_run)
    real_predictions_cache.populate_mask2former_panoptic()

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert cmd[:5] == ["uv", "run", "--extra", "real-models", "python"]
    module_idx = cmd.index("-m") + 1
    assert cmd[module_idx] == "tests.python.integration.real_models.sota._populate_cache"
    assert "--mask2former-panoptic" in cmd
    assert captured["check"] is True
    assert captured["cwd"] == real_predictions_cache.REPO_ROOT


def test_populate_mask2former_ade_shells_into_real_models_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same shape gate as the panoptic test."""
    import real_predictions_cache

    _pin_mask2former_revisions(monkeypatch)
    captured: dict[str, object] = {}

    def fake_run(cmd: list[str], **kwargs: object) -> object:
        captured["cmd"] = cmd
        captured["cwd"] = kwargs.get("cwd")
        captured["check"] = kwargs.get("check")
        return None

    monkeypatch.setattr("real_predictions_cache.subprocess.run", fake_run)
    real_predictions_cache.populate_mask2former_ade()

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert cmd[:5] == ["uv", "run", "--extra", "real-models", "python"]
    module_idx = cmd.index("-m") + 1
    assert cmd[module_idx] == "tests.python.integration.real_models.sota._populate_cache"
    assert "--mask2former-ade" in cmd
    assert captured["check"] is True
    assert captured["cwd"] == real_predictions_cache.REPO_ROOT


def test_populate_mask2former_panoptic_rejects_unpinned_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reverting ``MASK2FORMER_PANOPTIC_REVISION`` to the sentinel must
    raise — the cache filename embeds the revision, so populating
    against an unpinned sentinel would produce a cache file with
    "TODO_PIN..." in the name.

    Source-of-truth defaults to the pinned SHA; we monkeypatch back to
    the sentinel here so the regression catches a future "drop the pin
    guard" change. The bench-side adapter never invokes inference, so
    a working pin guard is the only thing standing between an unpinned
    constant and a stale-prediction cache.
    """
    import real_predictions_cache

    monkeypatch.setattr(
        real_predictions_cache,
        "MASK2FORMER_PANOPTIC_REVISION",
        real_predictions_cache._UNPINNED_REVISION,
    )
    with pytest.raises(RuntimeError, match="unpinned"):
        real_predictions_cache.populate_mask2former_panoptic()


def test_populate_mask2former_ade_rejects_unpinned_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sibling of the panoptic-side rejection test."""
    import real_predictions_cache

    monkeypatch.setattr(
        real_predictions_cache,
        "MASK2FORMER_ADE_REVISION",
        real_predictions_cache._UNPINNED_REVISION,
    )
    with pytest.raises(RuntimeError, match="unpinned"):
        real_predictions_cache.populate_mask2former_ade()


def test_registry_resolves_mask2former_panoptic_workload(
    fake_cache: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The panoptic workload's dt-side comes from the SOTA harness's
    cache directory; the gt-side is shared with the perfect-DT cell.
    We mock both halves to avoid hitting the real panoptic_val_cache."""
    from real_predictions_cache import MASK2FORMER_PANOPTIC_REVISION

    dt_dir = fake_cache / f"mask2former-pan-swin-t-{MASK2FORMER_PANOPTIC_REVISION}-coco-val2017"
    dt_dir.mkdir()
    (dt_dir / "panoptic_dt.json").write_bytes(b'{"annotations": []}')

    # Stub the panoptic GT resolution so the test doesn't depend on
    # the panoptic_val_cache being provisioned on disk.
    gt_png_dir = tmp_path / "gt_pngs"
    gt_png_dir.mkdir()
    gt_json = tmp_path / "panoptic_val2017.json"
    gt_json.write_bytes(b"{}")
    dt_png_dir_stub = tmp_path / "perfect_dt_pngs"  # ignored on this path
    dt_png_dir_stub.mkdir()
    dt_json_stub = tmp_path / "perfect_dt.json"
    dt_json_stub.write_bytes(b"{}")
    cats_json = tmp_path / "categories.json"
    cats_json.write_bytes(b"[]")
    monkeypatch.setattr(
        coco_panoptic_val2017,
        "perfect_workload_paths",
        lambda: (gt_png_dir, gt_json, dt_png_dir_stub, dt_json_stub, cats_json),
    )

    w = resolve(real_predictions.MASK2FORMER_PANOPTIC_WORKLOAD_ID, REPO_ROOT)
    assert isinstance(w, PanopticWorkload)
    assert w.dt_png_dir == dt_dir
    assert w.dt_json == dt_dir / "panoptic_dt.json"
    assert w.gt_png_dir == gt_png_dir
    assert w.gt_json == gt_json
    assert w.categories_json == cats_json


def test_registry_resolves_mask2former_ade_workload(
    fake_cache: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ADE semantic workload pulls its GT side from
    ``ade20k_val_cache.ensure_gt()``. We stub that to return
    in-tmp_path directories so the test doesn't hit the real cache."""
    import ade20k_val_cache
    from real_predictions_cache import MASK2FORMER_ADE_REVISION

    dt_dir = fake_cache / f"mask2former-ade-swin-t-{MASK2FORMER_ADE_REVISION}-ade20k-val"
    dt_dir.mkdir()
    (dt_dir / "1.png").write_bytes(b"x")

    gt_dir = tmp_path / "gt_train_ids"
    gt_dir.mkdir()
    images_dir = tmp_path / "val_images"
    images_dir.mkdir()
    monkeypatch.setattr(
        ade20k_val_cache,
        "ensure_gt",
        lambda **_: (gt_dir, images_dir, ade20k_val_cache.ADE20K_NUM_CLASSES),
    )

    w = resolve(real_predictions.MASK2FORMER_ADE_WORKLOAD_ID, REPO_ROOT)
    assert isinstance(w, SemanticWorkload)
    assert w.gt_label_maps == gt_dir
    assert w.dt_label_maps == dt_dir
    assert w.n_classes == ade20k_val_cache.ADE20K_NUM_CLASSES
    assert w.ignore_label == ade20k_val_cache.ADE20K_IGNORE_LABEL
