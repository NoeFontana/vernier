"""Stage-3 prediction-cache stub contract.

URL/SHA constants are ``None`` until hosting lands; the modules raise
informative ``RuntimeError`` rather than silently 404. These tests
pin that contract so a Stage-3 vendoring PR that forgets to set both
constants atomically gets caught.

Lives in the bench test suite because the tools/ packages don't carry
their own pytest config; the bench env has the cache packages as path
deps so the imports resolve here.
"""

from __future__ import annotations

import pytest

from real_predictions_cache import panoptic, semantic


def test_ensure_mask2former_raises_when_not_configured() -> None:
    assert panoptic.MASK2FORMER_URL is None
    assert panoptic.MASK2FORMER_SHA256 is None
    with pytest.raises(RuntimeError, match="Mask2Former"):
        panoptic.ensure_mask2former()


def test_ensure_hrnet_cityscapes_raises_when_not_configured() -> None:
    assert semantic.HRNET_CITYSCAPES_URL is None
    assert semantic.HRNET_CITYSCAPES_SHA256 is None
    with pytest.raises(RuntimeError, match="HRNet"):
        semantic.ensure_hrnet_cityscapes()


def test_ensure_ocrnet_ade20k_raises_when_not_configured() -> None:
    assert semantic.OCRNET_ADE20K_URL is None
    assert semantic.OCRNET_ADE20K_SHA256 is None
    with pytest.raises(RuntimeError, match="OCRNet"):
        semantic.ensure_ocrnet_ade20k()


def test_filenames_carry_version_and_dataset_id() -> None:
    assert "v1" in panoptic.mask2former_cache_filename()
    assert "coco-panoptic-val2017" in panoptic.mask2former_cache_filename()
    assert "v1" in semantic.hrnet_cityscapes_cache_filename()
    assert "cityscapes-val" in semantic.hrnet_cityscapes_cache_filename()
    assert "v1" in semantic.ocrnet_ade20k_cache_filename()
    assert "ade20k-val" in semantic.ocrnet_ade20k_cache_filename()
