"""vernier: high-performance, parity-preserving COCO-style evaluation.

Per ADR-0029, the public Python surface is split across per-paradigm
submodules:

- :mod:`vernier.instance` — bbox / segm / boundary / keypoints (AP fold).
- :mod:`vernier.panoptic` — panoptic-quality (PQ).
- :mod:`vernier.semantic` — semantic-segmentation mIoU (ADR-0028, planned).

The top-level :mod:`vernier` namespace keeps only the cross-paradigm
shared types and the pycocotools migration shim. Anything imported from
:mod:`vernier._core` directly is implementation detail and may change
without a deprecation cycle.
"""

from __future__ import annotations

from enum import Enum

from vernier import instance, panoptic
from vernier._compat import ParityMode
from vernier._compat import PycocotoolsCOCOeval as COCOeval
from vernier._core import version
from vernier.adapters import patch_pycocotools

__all__ = [
    "COCOeval",
    "Frequency",
    "ParityMode",
    "__version__",
    "instance",
    "panoptic",
    "patch_pycocotools",
    "version",
]


class Frequency(str, Enum):
    """LVIS category-frequency tier (ADR-0026, quirk **AB1**).

    Boundaries (matching ``lvis-api/lvis/eval.py:537-541``):

    - :attr:`Frequency.RARE` (``"r"``): ``< 10`` train images.
    - :attr:`Frequency.COMMON` (``"c"``): ``[10, 100)`` train images.
    - :attr:`Frequency.FREQUENT` (``"f"``): ``≥ 100`` train images.

    The values are the single-letter strings the LVIS JSON schema
    uses, so the enum round-trips through JSON without a custom
    converter and equates with the raw strings
    :attr:`vernier.instance.Dataset.category_frequency` returns. The
    ``(str, Enum)`` MRO is the Python 3.10-compatible spelling of
    ``StrEnum`` (added in 3.11).
    """

    RARE = "r"
    COMMON = "c"
    FREQUENT = "f"


__version__: str = version()
